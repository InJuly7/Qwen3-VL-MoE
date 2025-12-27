import torch
import torch.nn as nn
from utils import create_tensor, compare_tensor
from typing import Optional, Callable
from config import Qwen3VLMoeTextConfig
from utils import build_position_ids


def compute_default_rope_parameters(
    config: Optional[Qwen3VLMoeTextConfig] = None,
    device: Optional["torch.device"] = None,
    S: Optional[int] = None,
) -> tuple["torch.Tensor", float]:

    base = config.rope_theta
    D = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

    attention_factor = 1.0  # Unused in this type of RoPE

    # Compute the inverse frequencies
    # 偶数索引, 对应RoPE中的维度分组 (每对维度共享一个频率)
    # \theta = (50000)^{-2i/d} i = 0,1,2,...,d/2-1
    inv_freq = 1.0 / (base ** (torch.arange(0, D, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / D))  # [D // 2]
    return inv_freq, attention_factor


class Qwen3VLMoeTextRotaryEmbedding(nn.Module):
    # multidimensional RoPE 第 0/1/2 维分别给 RoPE 的三个“轴”提供位置坐标(序列/行/列)
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: Qwen3VLMoeTextConfig, device=None):
        super().__init__()
        # if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
        #     self.rope_type = config.rope_scaling.get("rope_type", "default")
        # else:
        #     self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        # self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        self.rope_init_fn: Callable = compute_default_rope_parameters

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)  # [D // 2] , float
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

        self.mrope_section = config.rope_scaling.get("mrope_section", [24, 20, 20])  # [T, H, W]

    def apply_interleaved_mrope(self, freqs, mrope_section):
        """Apply interleaved MRoPE to 3D rotary embeddings.
        Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
        interleaved [THTHWHTHW...TT], preserving frequency continuity.
        args:
            x: (3, bs, seq_len, head_dim // 2)
            mrope_section: (3,)
        returns:
            x_t: (bs, seq_len, head_dim // 2)
        """
        freqs_t = freqs[0]  # just overwrite the first dimension T # [B, S, D//2]
        # compare_tensor(freqs[0], freqs[1], torch.bfloat16)
        for dim, offset in enumerate((1, 2), start=1):  # H, W
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        # compare_tensor(freqs_t, freqs[0], torch.bfloat16)
        return freqs_t

    # @torch.no_grad()
    # @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        # In contrast to other models, Qwen3VLMoe has different position ids for the grids
        # So we expand the inv_freq to shape (3, ...)
        # 输入时 pisition_ids[0] != position_ids[1] != position_ids[2] [3,B,S]
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)  # [B,S] -> [3,B,S]
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)  # [3, B, D//2, 1]
        position_ids_expanded = position_ids[:, :, None, :].float()  # [3, B, 1, S]

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)  # [3, B, S, D//2]
            freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
            emb = torch.cat((freqs, freqs), dim=-1)  # [B, S, D]
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def test_qwen3_vl_moe_text_rope():
    config = Qwen3VLMoeTextConfig()
    mrope = Qwen3VLMoeTextRotaryEmbedding(config=config).to(device="cuda", dtype=torch.bfloat16)
    mrope.eval()
    seq_len = 2768
    position_ids = build_position_ids(seq_len=2768, text_len=128, grid_h=55, grid_w=48)
    position_ids = position_ids.to(dtype=torch.long, device="cuda")
    print(position_ids.shape)  # torch.Size([3, 1, 2768])
    # 验证一下：视觉段通常会出现 pos[0] != pos[1]
    diff01 = (position_ids[0, 0] != position_ids[1, 0]).sum().item()
    print("count(pos0 != pos1):", diff01)
    q = create_tensor((1, seq_len, config.num_attention_heads, config.head_dim), dtype=torch.bfloat16, ndim=4, device="cuda")  # [B,S,N,D]
    q = q.transpose(1, 2)  # [B,N,S,D]
    cos, sin = mrope.forward(q, position_ids)
    print(f"Cos cache shape: {cos.shape}")
    print(f"Sin cache shape: {sin.shape}")


if __name__ == "__main__":
    test_qwen3_vl_moe_text_rope()

"""
Output Log:
torch.Size([3, 1, 2768])
count(pos0 != pos1): 2640
Cos cache shape: torch.Size([1, 2768, 128])
Sin cache shape: torch.Size([1, 2768, 128])
"""
