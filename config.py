from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict


@dataclass
class Qwen3_VL_MOE_Text_Config:
    """Text configuration for Qwen3 VL MOE model"""

    model_type: str = "qwen3_vl_moe_text"
    attention_bias: bool = False
    attention_dropout: float = 0.0
    bos_token_id: int = 151643
    eos_token_id: int = 151645
    decoder_sparse_step: int = 1
    dtype: str = "bfloat16"
    head_dim: int = 128
    hidden_act: str = "silu"
    hidden_size: int = 2048
    initializer_range: float = 0.02
    intermediate_size: int = 6144
    max_position_embeddings: int = 262144
    mlp_only_layers: List[int] = field(default_factory=list)
    moe_intermediate_size: int = 768
    norm_topk_prob: bool = True
    num_attention_heads: int = 32
    num_experts: int = 128
    num_experts_per_tok: int = 8
    num_hidden_layers: int = 2
    num_key_value_heads: int = 4
    rms_norm_eps: float = 1e-06
    rope_scaling: Dict = field(default_factory=lambda: {"mrope_interleaved": True, "mrope_section": [24, 20, 20], "rope_type": "default"})
    rope_theta: int = 5000000
    router_aux_loss_coef: float = 0.001
    use_cache: bool = True
    vocab_size: int = 151936


@dataclass
class Qwen3_VL_MOE_Vision_Config:
    """Vision configuration for Qwen3 VL MOE model"""

    model_type: str = "qwen3_vl_moe"
    deepstack_visual_indexes: List[int] = field(default_factory=lambda: [1, 3, 5])
    depth: int = 6
    hidden_act: str = "gelu_pytorch_tanh"
    hidden_size: int = 1152
    in_channels: int = 3
    initializer_range: float = 0.02
    intermediate_size: int = 4304
    num_heads: int = 16
    num_position_embeddings: int = 2304
    out_hidden_size: int = 2048
    patch_size: int = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2


@dataclass
class Qwen3_VL_MOE_Config:
    """Main configuration for Qwen3 VL MOE model"""

    architectures: str = "Qwen3VLMoeForConditionalGeneration"
    dtype: str = "bfloat16"
    image_token_id: int = 151655
    model_type: str = "qwen3_vl_moe"
    text_config: Qwen3_VL_MOE_Text_Config = field(default_factory=Qwen3_VL_MOE_Text_Config)
    vision_config: Qwen3_VL_MOE_Vision_Config = field(default_factory=Qwen3_VL_MOE_Vision_Config)
    tie_word_embeddings: bool = False
    transformers_version: str = "4.57.3"
    video_token_id: int = 151656
    vision_end_token_id: int = 151653
    vision_start_token_id: int = 151652
    device: str = "cuda"
    torch_dtype: str = "bfloat16"
    do_sample: bool = False
