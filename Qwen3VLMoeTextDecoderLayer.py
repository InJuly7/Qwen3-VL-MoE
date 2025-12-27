import torch
import torch.nn as nn
from typing import Optional

from Qwen3VLMoeTextMLP import Qwen3VLMoeTextMLP
from Qwen3VLMoeTextAttention import Qwen3VLMoeTextAttention
from Qwen3VLMoeTextRMSNorm import Qwen3VLMoeTextRMSNorm
from Qwen3VLMoeTextSparseMoeBlock import Qwen3VLMoeTextSparseMoeBlock
from Qwen3VLMoeTextRotaryEmbedding import Qwen3VLMoeTextRotaryEmbedding

from config import Qwen3VLMoeTextConfig
from utils import DynamicCache
from utils import create_tensor, _prepare_decoder_attention_mask, build_position_ids


class Qwen3VLMoeTextDecoderLayer(nn.Module):
    def __init__(self, config: Qwen3VLMoeTextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Qwen3VLMoeTextAttention(config, layer_idx)

        # 当前层不在"仅使用普通MLP"的层列表中, 配置中专家数量大于0（启用了MoE）,当前层索引符合稀疏间隔规则
        # if (layer_idx not in config.mlp_only_layers) and (config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0):
        #     self.mlp = Qwen3VLMoeTextSparseMoeBlock(config)
        # else:
        #     self.mlp = Qwen3VLMoeTextMLP(config, intermediate_size=config.intermediate_size)
        self.mlp = Qwen3VLMoeTextSparseMoeBlock(config)
        self.input_layernorm = Qwen3VLMoeTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3VLMoeTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    # @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        # past_key_values: Optional[Cache] = None,
        past_key_values: Optional[DynamicCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        # **kwargs: Unpack[FlashAttentionKwargs],
        **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        # For the MoE layers, we need to unpack
        if isinstance(hidden_states, tuple):
            hidden_states, _ = hidden_states
        hidden_states = residual + hidden_states

        return hidden_states


def test_qwen3_vl_moe_text_decoder_layer():
    config = Qwen3VLMoeTextConfig()
    B = 1
    S = 2768
    H = config.hidden_size
    layer_idx = 0
    past_key_values_length = 0
    model = Qwen3VLMoeTextDecoderLayer(config, layer_idx).to(device="cuda", dtype=torch.bfloat16)
    model.eval()
    hidden_states = create_tensor((B, S, H), ndim=3, dtype=torch.bfloat16, device="cuda")
    print(f"Input hidden_states shape: {hidden_states.shape}")
    position_ids = build_position_ids(seq_len=S, text_len=128, grid_h=55, grid_w=48, device="cuda")  # [3, B, S]
    cache_position = position_ids[0]
    print(f"Position ids shape: {position_ids.shape}")
    print(f"Cache position shape: {cache_position.shape}")
    rope = Qwen3VLMoeTextRotaryEmbedding(config=config).to(device="cuda", dtype=torch.bfloat16)
    rope.eval()
    with torch.no_grad():
        sin_cache, cos_cache = rope(hidden_states, position_ids)  # [B, S, D]
    position_embeddings = (cos_cache, sin_cache)
    print(f"Position embeddings shapes: cos {cos_cache.shape}, sin {sin_cache.shape}")
    attention_mask = torch.ones(B, S).to(dtype=torch.bfloat16, device="cuda")
    # prefill 阶段
    attention_mask = _prepare_decoder_attention_mask(
        attention_mask=attention_mask,
        input_shape=hidden_states.shape[:2],
        past_key_values_length=past_key_values_length,
        dtype=torch.bfloat16,
        device="cuda",
    )
    print(f"attention_mask: {attention_mask}")
    print(f"attention shape: {attention_mask.shape}")
    past_key_values = DynamicCache()
    print(f"Initial kv cache length {past_key_values.get_seq_length()}")
    with torch.no_grad():
        hidden_states = model(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
        )
    print(f"Layer output shape: {hidden_states.shape}")


if __name__ == "__main__":
    test_qwen3_vl_moe_text_decoder_layer()

"""
Output Log:
Input hidden_states shape: torch.Size([1, 2768, 2048])
Position ids shape: torch.Size([3, 1, 2768])
Cache position shape: torch.Size([1, 2768])
Position embeddings shapes: cos torch.Size([1, 2768, 128]), sin torch.Size([1, 2768, 128])
attention_mask: tensor([[[[ 0.0000e+00, -3.3895e+38, -3.3895e+38,  ..., -3.3895e+38,
           -3.3895e+38, -3.3895e+38],
          [ 0.0000e+00,  0.0000e+00, -3.3895e+38,  ..., -3.3895e+38,
           -3.3895e+38, -3.3895e+38],
          [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  ..., -3.3895e+38,
           -3.3895e+38, -3.3895e+38],
          ...,
          [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  ...,  0.0000e+00,
           -3.3895e+38, -3.3895e+38],
          [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  ...,  0.0000e+00,
            0.0000e+00, -3.3895e+38],
          [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  ...,  0.0000e+00,
            0.0000e+00,  0.0000e+00]]]], device='cuda:0', dtype=torch.bfloat16)
attention shape: torch.Size([1, 1, 2768, 2768])
Initial kv cache length 0
Layer output shape: torch.Size([1, 2768, 2048])
"""
