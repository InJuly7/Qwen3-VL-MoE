import torch
import torch.nn as nn
from config import Qwen3VLMoeTextConfig
from utils import create_tensor
from utils import SiLUActivation


class Qwen3VLMoeTextMLP(nn.Module):
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size if intermediate_size is not None else config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = SiLUActivation()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def test_qwen3_vl_moe_text_mlp():
    config = Qwen3VLMoeTextConfig()
    model = Qwen3VLMoeTextMLP(config).to(device="cuda", dtype=torch.bfloat16)
    input_tensor = create_tensor((2, 4, config.hidden_size), dtype=torch.bfloat16, ndim=3, device="cuda")
    print("input shape:", input_tensor.shape)
    output = model(input_tensor)
    assert output.shape == (2, 4, config.hidden_size)
    print("output shape:", output.shape)


if __name__ == "__main__":
    test_qwen3_vl_moe_text_mlp()

"""
Output Log:
input shape: torch.Size([2, 4, 2048])
output shape: torch.Size([2, 4, 2048])
"""
