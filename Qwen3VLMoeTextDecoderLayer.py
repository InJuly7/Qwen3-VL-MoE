import torch
import torch.nn as nn
from typing import Optional

from Qwen3VLMoeTextMLP import Qwen3VLMoeTextMLP
from Qwen3VLMoeTextAttention import Qwen3VLMoeTextAttention
from Qwen3VLMoeTextRMSNorm import Qwen3VLMoeTextRMSNorm
from Qwen3VLMoeTextSparseMoeBlock import Qwen3VLMoeTextSparseMoeBlock
from config import Qwen3VLMoeTextConfig
from utils import DynamicCache


class Qwen3VLMoeTextDecoderLayer(nn.Module):
    def __init__(self, config: Qwen3VLMoeTextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Qwen3VLMoeTextAttention(config, layer_idx)

        # 当前层不在"仅使用普通MLP"的层列表中, 配置中专家数量大于0（启用了MoE）,当前层索引符合稀疏间隔规则
        if (layer_idx not in config.mlp_only_layers) and (config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0):
            self.mlp = Qwen3VLMoeTextSparseMoeBlock(config)
        else:
            self.mlp = Qwen3VLMoeTextMLP(config, intermediate_size=config.intermediate_size)

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
