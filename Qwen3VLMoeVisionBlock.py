import torch
import torch.nn as nn
from typing import Optional
from config import Qwen3VLMoeVisionConfig
from Qwen3VLMoeVisionAttention import Qwen3VLMoeVisionAttention
from Qwen3VLMoeVisionMLP import Qwen3VLMoeVisionMLP
from utils import create_tensor


class Qwen3VLMoeVisionBlock(nn.Module):
    def __init__(self, config, attn_implementation: str = "sdpa") -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen3VLMoeVisionAttention(config=config)
        self.mlp = Qwen3VLMoeVisionMLP(config=config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


def test_qwen3_vl_moe_vision_block():
    config = Qwen3VLMoeVisionConfig()
    model = Qwen3VLMoeVisionBlock(config).to(device="cuda", dtype=torch.bfloat16)
    model.eval()
    hidden_states = create_tensor((11008, 1152), ndim=2, device="cuda", dtype=torch.bfloat16)
    cu_seqlens = torch.tensor([0, 11008], device="cuda:0", dtype=torch.int32)
    rotary_pos_emb = None
    position_embeddings = create_tensor((11008, 72), ndim=2, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        hidden_states = model(
            hidden_states=hidden_states,
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=(position_embeddings, position_embeddings),
        )
    print(f"hidden_states.shape: {hidden_states.shape}")  # [11008,1152]


if __name__ == "__main__":
    test_qwen3_vl_moe_vision_block()
