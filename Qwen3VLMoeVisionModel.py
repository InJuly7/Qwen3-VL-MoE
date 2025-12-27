import torch
from torch import nn
from torch.nn import functional as F

from config import Qwen3VLMoeVisionConfig
from Qwen3VLMoeVisionPatchEmbed import Qwen3VLMoeVisionPatchEmbed
from Qwen3VLMoeVisionRotaryEmbedding import Qwen3VLMoeVisionRotaryEmbedding
from Qwen3VLMoeVisionBlock import Qwen3VLMoeVisionBlock
from Qwen3VLMoeVisionPatchMerger import Qwen3VLMoeVisionPatchMerger
from utils import create_tensor


class Qwen3VLMoeVisionModel(nn.Module):
    config: Qwen3VLMoeVisionConfig
    _no_split_modules = ["Qwen3VLMoeVisionBlock"]

    def __init__(self, config, *inputs, **kwargs) -> None:
        # super().__init__(config, *inputs, **kwargs)
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.patch_embed = Qwen3VLMoeVisionPatchEmbed(
            config=config,
        )

        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen3VLMoeVisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList([Qwen3VLMoeVisionBlock(config) for _ in range(config.depth)])
        self.merger = Qwen3VLMoeVisionPatchMerger(
            config=config,
            use_postshuffle_norm=False,
        )

        self.deepstack_visual_indexes = config.deepstack_visual_indexes
        self.deepstack_merger_list = nn.ModuleList(
            [
                Qwen3VLMoeVisionPatchMerger(
                    config=config,
                    use_postshuffle_norm=True,
                )
                for _ in range(len(config.deepstack_visual_indexes))
            ]
        )

        self.gradient_checkpointing = False

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        merge_size = self.spatial_merge_size

        max_hw = int(grid_thw[:, 1:].max().item())
        freq_table = self.rotary_pos_emb(max_hw)  # (max_hw, dim // 2)
        device = freq_table.device

        total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

        offset = 0
        for num_frames, height, width in grid_thw:
            merged_h, merged_w = height // merge_size, width // merge_size

            block_rows = torch.arange(merged_h, device=device)  # block row indices
            block_cols = torch.arange(merged_w, device=device)  # block col indices
            intra_row = torch.arange(merge_size, device=device)  # intra-block row offsets
            intra_col = torch.arange(merge_size, device=device)  # intra-block col offsets

            # Compute full-resolution positions
            row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
            col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]

            row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)

            coords = torch.stack((row_idx, col_idx), dim=-1)

            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)

            num_tokens = coords.shape[0]
            pos_ids[offset : offset + num_tokens] = coords
            offset += num_tokens

        embeddings = freq_table[pos_ids]  # lookup rotary embeddings
        embeddings = embeddings.flatten(1)
        return embeddings

    def fast_pos_embed_interpolate(self, grid_thw):
        grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for t, h, w in zip(grid_ts, grid_hs, grid_ws):
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)

            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)

            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor

            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side

            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]

            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]

            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=self.pos_embed.weight.device)
        weight_tensor = torch.tensor(weight_list, dtype=self.pos_embed.weight.dtype, device=self.pos_embed.weight.device)
        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws)])

        patch_pos_embeds_permute = []
        merge_size = self.config.spatial_merge_size
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1).permute(0, 1, 3, 2, 4, 5).flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)
        patch_pos_embeds = torch.cat(patch_pos_embeds_permute)
        return patch_pos_embeds

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `torch.Tensor`: hidden_states.
        """
        hidden_states = self.patch_embed(hidden_states)

        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            if layer_num in self.deepstack_visual_indexes:
                deepstack_feature = self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_num)](hidden_states)
                deepstack_feature_lists.append(deepstack_feature)

        hidden_states = self.merger(hidden_states)

        return hidden_states, deepstack_feature_lists


def test_qwen3_vl_moe_vision_model():
    config = Qwen3VLMoeVisionConfig()
    model = Qwen3VLMoeVisionModel(config=config, input=tuple()).to(device="cuda", dtype=torch.bfloat16)
    model.eval()
    # hidden_states shape: torch.Size([11008, 1536]), grid_thw: tensor([[  1,  86, 128]], device='cuda:0')
    hidden_states = create_tensor((11008, 1536), ndim=2, device="cuda", dtype=torch.bfloat16)
    grid_thw = torch.tensor([1, 86, 128], device="cuda").unsqueeze(0)
    with torch.no_grad():
        hidden_states, deepstack_feature_lists = model(hidden_states=hidden_states, grid_thw=grid_thw)
    print(f"output hidden_states shape: {hidden_states.shape}")
    print(f"deepstack_feature_lists lengths: {len(deepstack_feature_lists)}")
    print(f"deepstack_feature_lists: {deepstack_feature_lists}")
    for i, deepstack_feature in enumerate(deepstack_feature_lists):
        print(f"deepstack_feature {i} shape: {deepstack_feature.shape}")


if __name__ == "__main__":
    test_qwen3_vl_moe_vision_model()

"""
output hidden_states shape: torch.Size([2752, 2048])
deepstack_feature_lists lengths: 3
deepstack_feature_lists: [tensor([[ 0.2480,  0.1602,  0.0801,  ...,  0.1709, -0.3945,  0.0723],
        [ 0.1445,  0.5117, -0.3047,  ..., -0.3379, -0.2471,  0.1572],
        [-0.0364,  0.1787, -0.1865,  ..., -0.2441,  0.0219,  0.1846],
        ...,
        [-0.0649,  0.0693, -0.0664,  ..., -0.1011, -0.1128, -0.3496],
        [ 0.0310, -0.1562, -0.1572,  ..., -0.1045,  0.3164,  0.0354],
        [-0.1191, -0.1465, -0.1230,  ...,  0.2910, -0.0684,  0.4277]],
       device='cuda:0', dtype=torch.bfloat16), tensor([[-0.3184,  0.1328, -0.0566,  ..., -0.2383,  0.1631, -0.0771],
        [-0.2930,  0.1182,  0.2168,  ..., -0.0586,  0.0115, -0.1182],
        [-0.2480, -0.0649,  0.2715,  ..., -0.2314, -0.0264, -0.0366],
        ...,
        [-0.1748, -0.0933, -0.0564,  ...,  0.1348, -0.2178,  0.2100],
        [-0.6211,  0.1426, -0.0479,  ...,  0.3047,  0.0420,  0.1641],
        [-0.1133,  0.0649,  0.0122,  ...,  0.1816,  0.0693,  0.1118]],
       device='cuda:0', dtype=torch.bfloat16), tensor([[ 0.1260,  0.0825, -0.0977,  ...,  0.0046,  0.0762, -0.0767],
        [-0.2637,  0.0300,  0.2480,  ..., -0.1050,  0.1348,  0.3320],
        [-0.1582, -0.0972,  0.3281,  ..., -0.2373, -0.1484,  0.3164],
        ...,
        [ 0.1992,  0.1807, -0.0737,  ..., -0.1377,  0.0762,  0.4219],
        [-0.0305, -0.0654,  0.0131,  ..., -0.0454,  0.1040,  0.2490],
        [-0.2207,  0.0918,  0.0479,  ..., -0.0265,  0.1660, -0.1367]],
       device='cuda:0', dtype=torch.bfloat16)]
deepstack_feature 0 shape: torch.Size([2752, 2048])
deepstack_feature 1 shape: torch.Size([2752, 2048])
deepstack_feature 2 shape: torch.Size([2752, 2048])
"""