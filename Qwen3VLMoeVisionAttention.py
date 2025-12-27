import torch
from torch import nn
from typing import Optional, Callable

from config import Qwen3VLMoeVisionConfig
from utils import create_tensor


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    # **kwargs: Unpack[TransformersKwargs],
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed


class Qwen3VLMoeVisionAttention(nn.Module):
    def __init__(self, config: Qwen3VLMoeVisionConfig) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.num_key_value_groups = 1  # needed for eager attention
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)
        self.scaling = self.head_dim**-0.5
        self.config = config
        self.attention_dropout = 0.0
        self.is_causal = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        attention_interface: Callable = eager_attention_forward
        # if self.config._attn_implementation != "eager":
        #     attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        # if self.config._attn_implementation == "flash_attention_2":
        #     # Flash Attention 2: Use cu_seqlens for variable length attention
        #     max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
        #     attn_output, _ = attention_interface(
        #         self,
        #         query_states,
        #         key_states,
        #         value_states,
        #         attention_mask=None,
        #         scaling=self.scaling,
        #         dropout=0.0 if not self.training else self.attention_dropout,
        #         cu_seq_lens_q=cu_seqlens,
        #         cu_seq_lens_k=cu_seqlens,
        #         max_length_q=max_seqlen,
        #         max_length_k=max_seqlen,
        #         is_causal=False,
        #         **kwargs,
        #     )
        # else:

        # Other implementations: Process each chunk separately
        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        splits = [torch.split(tensor, lengths.tolist(), dim=2) for tensor in (query_states, key_states, value_states)]

        attn_outputs = [
            attention_interface(
                self,
                q,
                k,
                v,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                is_causal=False,
                **kwargs,
            )[0]
            for q, k, v in zip(*splits)
        ]
        attn_output = torch.cat(attn_outputs, dim=1)

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output


def test_qwen3_vl_moe_vision_attention():
    config = Qwen3VLMoeVisionConfig()
    model = Qwen3VLMoeVisionAttention(config).to(device="cuda", dtype=torch.bfloat16)
    model.eval()
    hidden_states = create_tensor((11008, 1152), ndim=2, device="cuda", dtype=torch.bfloat16)
    cu_seqlens = torch.tensor([0, 11008], device="cuda:0", dtype=torch.int32)
    rotary_pos_emb = None
    position_embeddings = create_tensor((11008, 72), ndim=2, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        attn_output = model(
            hidden_states=hidden_states,
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=(position_embeddings, position_embeddings),
        )
    print(f"attn_output.shape: {attn_output.shape}")  # [11008,1152]


if __name__ == "__main__":
    test_qwen3_vl_moe_vision_attention()
