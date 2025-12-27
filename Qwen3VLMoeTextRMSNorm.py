import torch
import torch.nn as nn
from config import Qwen3VLMoeTextConfig
from utils import create_tensor


class Qwen3VLMoeTextRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


def test_qwen3_vl_moe_text_rmsnorm():
    config = Qwen3VLMoeTextConfig()
    B = 2
    S = 2768
    eps = config.rms_norm_eps
    H = config.hidden_size
    model = Qwen3VLMoeTextRMSNorm(hidden_size=H, eps=eps).to(device="cuda", dtype=torch.bfloat16)
    hidden_states = create_tensor((B, S, H), ndim=3, device="cuda", dtype=torch.bfloat16)
    output = model(hidden_states)
    print("Output shape:", output.shape)


if __name__ == "__main__":
    test_qwen3_vl_moe_text_rmsnorm()

"""
Output Log:
Output shape: torch.Size([2, 2768, 2048])
"""
