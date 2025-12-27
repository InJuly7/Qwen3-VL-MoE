import torch
import torch.nn as nn
from config import Qwen3VLMoeTextConfig
from utils import create_tensor
from utils import SiLUActivation


class Qwen3VLMoeTextExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.intermediate_size = config.moe_intermediate_size
        self.hidden_size = config.hidden_size
        self.expert_dim = self.intermediate_size
        # [num_experts, H, 2 * expert_dim]
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, 2 * self.expert_dim))
        # [num_experts, expert_dim, H]
        self.down_proj = nn.Parameter(torch.empty((self.num_experts, self.expert_dim, self.hidden_size)))
        self.act_fn = SiLUActivation()

    def forward(self, hidden_states: torch.Tensor, routing_weights: torch.Tensor, router_indices: torch.Tensor) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)  # (B * S, H)
        # Inference
        # dim0 方向复制 num_experts 次, dim1 方向不变
        hidden_states = hidden_states.repeat(self.num_experts, 1)  # (num_experts * B * S, H)
        hidden_states = hidden_states.view(self.num_experts, -1, self.hidden_size)  # (num_experts, B * S, H)
        gate_up = torch.bmm(hidden_states, self.gate_up_proj)  # (num_experts, B * S, 2 * expert_dim)
        # (num_experts, B * S, expert_dim), (num_experts, B * S, expert_dim)
        gate, up = gate_up.chunk(2, dim=-1)  # not supported for DTensors
        next_states = torch.bmm((up * self.act_fn(gate)), self.down_proj)  # (num_experts, B * S, H)
        next_states = next_states.reshape(self.num_experts, batch_size, -1, self.hidden_size)  # (num_experts, B, S, H)
        # [num_experts, B, S, 1] 乘法时广播到 H 维度
        next_states = next_states * routing_weights.transpose(0, 1).view(self.num_experts, batch_size, -1)[..., None]
        next_states = next_states.sum(dim=0)  # (B, S, H)
        return next_states


def test_qwen3_vl_moe_text_experts():
    config = Qwen3VLMoeTextConfig()
    model = Qwen3VLMoeTextExperts(config).to(device="cuda", dtype=torch.bfloat16)
    B = 1
    S = 2768
    H = config.hidden_size
    num_experts = config.num_experts
    num_experts_per_tok = config.num_experts_per_tok

    hidden_states = create_tensor((B, S, H), dtype=torch.bfloat16, ndim=3, device="cuda")  # [B, S, H]
    routing_weights = create_tensor((B * S, num_experts), dtype=torch.bfloat16, ndim=2, device="cuda")  # [B * S, num_experts]
    _, router_indices = torch.topk(routing_weights, num_experts_per_tok, dim=-1)
    output = model(hidden_states, routing_weights, router_indices)
    print("output.shape:", output.shape)  # [B, S, H]


if __name__ == "__main__":
    test_qwen3_vl_moe_text_experts()
