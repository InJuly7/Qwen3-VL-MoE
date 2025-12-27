import torch
import torch.nn as nn
from config import Qwen3VLMoeTextConfig
from utils import create_tensor
from Qwen3VLMoeTextExperts import Qwen3VLMoeTextExperts


class Qwen3VLMoeTextSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = Qwen3VLMoeTextExperts(config)

        # since all the models use norm_topk_prob, we don't need to have a extra check for it
        # self.norm_topk_prob = config.norm_topk_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (B,S,H)
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)  # (B*S,H)
        router_logits = self.gate(hidden_states)  # (B*S, num_experts)
        # 转换到float进行softmax计算
        routing_weights = torch.nn.functional.softmax(router_logits, dim=-1, dtype=torch.float)  # (B*S, num_experts)
        routing_weights, router_indices = torch.topk(routing_weights, self.top_k, dim=-1)  # (B*S, top_k) , (B*S, top_k)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)  # normalize
        # convert to hidden_states dtype
        routing_weights = routing_weights.to(hidden_states.dtype)
        router_weights = torch.zeros_like(router_logits).scatter_(1, router_indices, routing_weights)  # (B*S, num_experts)
        hidden_states = hidden_states.reshape(batch_size, -1, self.hidden_size)  # (B,S,H)
        routed_out = self.experts(hidden_states, router_weights, router_indices)  # (B,S,H)
        return routed_out, router_logits


def test_qwen3_vl_moe_text_sparse_moe_block():
    config = Qwen3VLMoeTextConfig()
    model = Qwen3VLMoeTextSparseMoeBlock(config).to(device="cuda", dtype=torch.bfloat16)
    hidden_states = create_tensor((1, 2768, 2048), dtype=torch.bfloat16, ndim=3, device="cuda")
    print("hidden_states.shape:", hidden_states.shape)
    routed_out, router_logits = model(hidden_states)
    print("routed_out.shape:", routed_out.shape)
    print("router_logits.shape:", router_logits.shape)


if __name__ == "__main__":
    test_qwen3_vl_moe_text_sparse_moe_block()

"""
Output Log:
hidden_states.shape: torch.Size([1, 2768, 2048])
routed_out.shape: torch.Size([1, 2768, 2048])
router_logits.shape: torch.Size([2768, 128])
"""
