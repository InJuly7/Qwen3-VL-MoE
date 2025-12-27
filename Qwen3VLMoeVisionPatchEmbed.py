import torch
import torch.nn as nn
from config import Qwen3VLMoeVisionConfig
from utils import create_tensor


class Qwen3VLMoeVisionPatchEmbed(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size

        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(self.in_channels, self.embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )  # [11008, 3, 2, 16, 16]
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states


def test_qwen3_vl_moe_vision_patch_embed():
    config = Qwen3VLMoeVisionConfig()
    model = Qwen3VLMoeVisionPatchEmbed(config).to(device="cuda", dtype=torch.bfloat16)
    model.eval()
    hidden_states = create_tensor((11008, 1536), ndim=2, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        hidden_states = model(hidden_states=hidden_states)
    print(f"hidden_states.shape: {hidden_states.shape}")  # [11008,1152]


if __name__ == "__main__":
    test_qwen3_vl_moe_vision_patch_embed()
