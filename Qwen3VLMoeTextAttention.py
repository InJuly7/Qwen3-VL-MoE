import torch
import torch.nn as nn
from typing import Optional, Callable
from config import Qwen3VLMoeTextConfig
from Qwen3VLMoeTextRMSNorm import Qwen3VLMoeTextRMSNorm
from Qwen3VLMoeTextRotaryEmbedding import Qwen3VLMoeTextRotaryEmbedding
from utils import DynamicCache
from utils import _prepare_decoder_attention_mask, create_tensor, build_position_ids


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


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


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen3VLMoeTextAttention(nn.Module):
    def __init__(self, config: Qwen3VLMoeTextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        self.q_norm = Qwen3VLMoeTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3VLMoeTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)  # thus post q_norm does not need reshape

    # @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        # past_key_values: Optional[Cache] = None,
        past_key_values: Optional[DynamicCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        # **kwargs: Unpack[FlashAttentionKwargs],
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        # if self.config._attn_implementation != "eager":
        #     attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


def test_qwen3_vl_moe_text_attention():
    config = Qwen3VLMoeTextConfig()
    model = Qwen3VLMoeTextAttention(config, layer_idx=0).to(device="cuda", dtype=torch.bfloat16)
    model.eval()
    B = 1
    S = 2768
    N1 = config.num_attention_heads
    N2 = config.num_key_value_heads
    G = N1 // N2
    H = config.hidden_size
    D = H // N1
    past_key_values_length = 0
    past_key_values = DynamicCache()

    hidden_states = create_tensor((B, S, H), dtype=torch.bfloat16, ndim=3, device="cuda")
    position_ids = build_position_ids(seq_len=S, text_len=128, grid_h=55, grid_w=48, device="cuda")  # [3, B, S]
    rope = Qwen3VLMoeTextRotaryEmbedding(config=config).to(device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        sin_cache, cos_cache = rope(hidden_states, position_ids)  # [B, S, D]
    position_embeddings = (cos_cache, sin_cache)
    attention_mask = torch.ones(B, S).to(dtype=torch.bfloat16, device="cuda")
    # prefill 阶段
    attention_mask = _prepare_decoder_attention_mask(
        attention_mask=attention_mask,
        input_shape=hidden_states.shape[:2],
        past_key_values_length=past_key_values_length,
        dtype=torch.bfloat16,
        device="cuda",
    )
    print(f"attention_mask: {attention_mask}")
    print(f"attention shape: {attention_mask.shape}")
    with torch.no_grad():
        attn_output, attn_weights = model(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )
    print(f"Layer 0 output shape: {attn_output.shape}, attn_weights shape: {attn_weights.shape}")
    print(f"kv cache length {past_key_values.get_seq_length()}")


if __name__ == "__main__":
    test_qwen3_vl_moe_text_attention()

"""
attention_mask: tensor([[[[ 0.0000e+00, -3.3895e+38, -3.3895e+38,  ..., -3.3895e+38,
           -3.3895e+38, -3.3895e+38],
          [ 0.0000e+00,  0.0000e+00, -3.3895e+38,  ..., -3.3895e+38,
           -3.3895e+38, -3.3895e+38],
          [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  ..., -3.3895e+38,
           -3.3895e+38, -3.3895e+38],
          ...,
          [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  ...,  0.0000e+00,
           -3.3895e+38, -3.3895e+38],
          [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  ...,  0.0000e+00,
            0.0000e+00, -3.3895e+38],
          [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  ...,  0.0000e+00,
            0.0000e+00,  0.0000e+00]]]], device='cuda:0', dtype=torch.bfloat16)
attention shape: torch.Size([1, 1, 2768, 2768])
Layer 0 output shape: torch.Size([1, 2768, 2048]), attn_weights shape: torch.Size([1, 32, 2768, 2768])
kv cache length 2768
"""
