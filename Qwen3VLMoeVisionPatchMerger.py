import torch
from torch import nn
from config import Qwen3VLMoeVisionConfig
from utils import create_tensor


class Qwen3VLMoeVisionPatchMerger(nn.Module):
    def __init__(self, config: Qwen3VLMoeVisionConfig, use_postshuffle_norm=False) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm
        self.norm = nn.LayerNorm(self.hidden_size if use_postshuffle_norm else config.hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_size, config.out_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x.view(-1, self.hidden_size) if self.use_postshuffle_norm else x).view(-1, self.hidden_size)
        x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return x


def test_qwen3_vl_moe_vision_patch_merger():
    config = Qwen3VLMoeVisionConfig()
    model = Qwen3VLMoeVisionPatchMerger(config).to(device="cuda", dtype=torch.bfloat16)
    model.eval()
    x = create_tensor((11008, 1152), ndim=2, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        x = model(x)
    print(f"x.shape: {x.shape}")  # [2752, 2048]


if __name__ == "__main__":
    test_qwen3_vl_moe_vision_patch_merger()
