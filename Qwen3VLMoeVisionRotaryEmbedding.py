import torch
import torch.nn as nn

from config import Qwen3VLMoeVisionConfig


class Qwen3VLMoeVisionRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


def test_qwen3_vl_moe_vision_rotary_embedding():
    seqlen = 128
    dim = 36
    rope = Qwen3VLMoeVisionRotaryEmbedding(dim).to(device="cuda", dtype=torch.bfloat16)
    rope.eval()
    with torch.no_grad():
        freqs = rope(seqlen)
    print(f"freqs.shape: {freqs.shape}")  # [128,18]


if __name__ == "__main__":
    test_qwen3_vl_moe_vision_rotary_embedding()
