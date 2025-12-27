import torch
import torch.nn as nn
from config import Qwen3VLMoeVisionConfig
from utils import GELUTanh
from utils import create_tensor


class Qwen3VLMoeVisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.linear_fc1 = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)
        self.act_fn = GELUTanh()

    def forward(self, hidden_state):
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))


def test_qwen3_vl_moe_vision_mlp():
    config = Qwen3VLMoeVisionConfig()
    model = Qwen3VLMoeVisionMLP(config).to(device="cuda", dtype=torch.bfloat16)
    hidden_state = create_tensor((11008, 1152), ndim=2, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        hidden_state = model(hidden_state)
    print("hidden_state shape:", hidden_state.shape)


if __name__ == "__main__":
    test_qwen3_vl_moe_vision_mlp()
