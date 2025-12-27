import torch
from torch import Tensor, nn
import math

from typing import Optional, List, Union
from dataclasses import dataclass
import os
import functools


class DynamicCache:
    """动态缓存 - 每次都拼接, RoPE 在外部应用"""

    def __init__(self):
        self.key_cache = []
        self.value_cache = []

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if layer_idx >= len(self.key_cache):
            # 第一次，直接存储
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            # 拼接
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx=0):
        """获取当前缓存的序列长度"""
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[2]


class SiLUActivation(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return nn.functional.silu(input)


class GELUTanh(nn.Module):
    def __init__(self, use_gelu_tanh_python: bool = False):
        super().__init__()
        # gelu_pytorch_tanh
        self.act = functools.partial(nn.functional.gelu, approximate="tanh")

    def forward(self, input: Tensor) -> Tensor:
        return self.act(input)


def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0):
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)  # [tgt_len, tgt_len], -inf
    mask_cond = torch.arange(mask.size(-1), device=device)  # [tgt_len]
    # 左上顶点下三角矩阵置0
    # 比较时候广播: [tgt_len] , [tgt_len, 1] ==> [tgt_len, tgt_len]
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)  # (通过 view 实现) (tgt_len, 1)
    mask = mask.to(dtype)

    # Prefill 阶段的续写
    # chunk processing, 长文本分块处理
    if past_key_values_length > 0:
        # 构造右下顶点下三角矩阵置0 [tgt_len, past_key_values_length + tgt_len]
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)  # [B, 1, S1, S2]


# 多batch推理时序列长度对齐
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len
    # 只屏蔽 key padding
    # 如果将padding token 那一行全部置-inf, 那么softmax 的结果会变成 NaN值
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)  # [B,1,tgt_len,src_len]
    inverted_mask = 1.0 - expanded_mask  # [B,1,tgt_len,src_len]
    # 将padding位置(现在是1)填充为-inf
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


def _prepare_decoder_attention_mask(
    attention_mask, input_shape, inputs_embeds=None, past_key_values_length=0, dtype=torch.float16, device="cuda"
):

    # Decode 阶段
    if past_key_values_length != 0:
        attention_mask = None
        return attention_mask

    if inputs_embeds is not None:
        dtype = inputs_embeds.dtype
        device = inputs_embeds.device

    # Prefill 阶段 create causal mask
    combined_attention_mask = None
    if input_shape[-1] > 1:  # seq_len > 1 时才创建
        combined_attention_mask = _make_causal_mask(input_shape, dtype, device, past_key_values_length=past_key_values_length)

    if attention_mask is not None:
        expanded_attn_mask = _expand_mask(attention_mask, dtype, tgt_len=input_shape[-1]).to(device)
        combined_attention_mask = expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask

    return combined_attention_mask


def build_position_ids(seq_len=2768, text_len=128, grid_h=55, grid_w=48, device="cuda"):
    """
    返回 position_ids: [3, 1, L]
    约定：
      - 轴0: 全局序列位置(类似 1D position)
      - 轴1: 高度 h(视觉段为网格行号;文本段用全局位置占位)
      - 轴2: 宽度 w(视觉段为网格列号;文本段用全局位置占位)
    """
    assert text_len <= seq_len
    vision_len = seq_len - text_len
    assert grid_h * grid_w == vision_len, f"grid_h*grid_w 必须等于 vision_len={vision_len}"

    # axis 0：全局递增位置（让 KV cache / rope 连续更常见）
    pos0 = torch.arange(seq_len, dtype=torch.long, device=device)  # [L]

    # 文本段：为了简单，轴1/2也用全局位置（你也可以改成全 0）
    text_pos = torch.arange(text_len, dtype=torch.long, device=device)  # [T]
    pos1_text = text_pos.clone()
    pos2_text = text_pos.clone()

    # 视觉段：构造 h/w 网格坐标
    # h: 0..H-1，每个重复 W 次；w: 0..W-1，重复 H 次
    h = torch.arange(grid_h, dtype=torch.long, device=device).repeat_interleave(grid_w)  # [V]
    w = torch.arange(grid_w, dtype=torch.long, device=device).repeat(grid_h)  # [V]

    pos1 = torch.cat([pos1_text, h], dim=0)  # [L]
    pos2 = torch.cat([pos2_text, w], dim=0)  # [L]

    position_ids = torch.stack([pos0, pos1, pos2], dim=0).unsqueeze(1)  # [3, 1, L]
    return position_ids


# class Qwen3VLMoeCausalLMOutputWithPast(ModelOutput):
#     loss: Optional[torch.FloatTensor] = None
#     logits: Optional[torch.FloatTensor] = None
#     past_key_values: Optional[Cache] = None
#     hidden_states: Optional[tuple[torch.FloatTensor]] = None
#     attentions: Optional[tuple[torch.FloatTensor]] = None
#     rope_deltas: Optional[torch.LongTensor] = None
#     aux_loss: Optional[torch.FloatTensor] = None


# @dataclass
# @auto_docstring(
#     custom_intro="""
#     Base class for Llava outputs, with hidden states and attentions.
#     """
# )
# class Qwen3VLMoeModelOutputWithPast(ModelOutput):
#     last_hidden_state: Optional[torch.FloatTensor] = None
#     past_key_values: Optional[Cache] = None
#     hidden_states: Optional[tuple[torch.FloatTensor]] = None
#     attentions: Optional[tuple[torch.FloatTensor]] = None
#     rope_deltas: Optional[torch.LongTensor] = None


@dataclass
class BaseModelOutputWithPast:
    last_hidden_state: Optional[torch.FloatTensor] = None
    # past_key_values: Optional[Cache] = None
    past_key_values: Optional[DynamicCache] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None


@dataclass
class CausalLMOutputWithPast:
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    # past_key_values: Optional[Cache] = None
    past_key_values: Optional[DynamicCache] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None


class StoppingCriteria:
    def __init__(self, max_length: int, eos_token_id: Union[int, List[int]]):
        self.max_length = max_length
        self.eos_token_id = eos_token_id if isinstance(eos_token_id, list) else [eos_token_id]

    def __call__(self, input_ids: torch.LongTensor, **kwargs) -> torch.BoolTensor:
        batch_size = input_ids.shape[0]
        is_done = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

        # 检查长度
        length_done = input_ids.shape[1] >= self.max_length

        # 检查 EOS token (最后一个生成的 token)
        last_tokens = input_ids[:, -1]
        eos_done = torch.isin(last_tokens, torch.tensor(self.eos_token_id, device=input_ids.device))

        # 任一条件满足即停止
        is_done = length_done | eos_done

        return is_done


"""
@func: 创建一个 torch Tensor
@param: shape: 形状
@param: dtype: 数据类型
@param: ndim: 维度数量
@param: device: 设备名称
@return: torch.Tensor
"""


def create_tensor(shape, dtype=torch.float32, ndim=2, device="cpu"):
    if ndim == 1:
        return torch.randn((shape[0]), device=device).to(dtype).contiguous()
    elif ndim == 2:
        return torch.randn((shape[0], shape[1]), device=device).to(dtype).contiguous()
    elif ndim == 3:
        return torch.randn((shape[0], shape[1], shape[2]), device=device).to(dtype).contiguous()
    elif ndim == 4:
        return torch.randn((shape[0], shape[1], shape[2], shape[3]), device=device).to(dtype).contiguous()
    else:
        raise ValueError("Unsupported ndim")


"""
@func: 比较两个tensor是否相等
@param: A : 第一个tensor
@param: B : 第二个tensor
@return: bool
"""


# ∣A − B∣ ≤ atol + (rtol × ∣B∣)
def compare_tensor(A: torch.Tensor, B: torch.Tensor, dtype: torch.dtype, rtol: float = 1e-3, atol: float = 1e-3) -> bool:

    # 检查形状
    if A.shape != B.shape:
        print(f"Shape mismatch: {A.shape} vs {B.shape}")
        return False

    # 使用 torch.isclose 生成逐元素的对比掩码, is_close 是一个全是 True/False 的 Tensor
    if A.dtype == torch.float16:
        is_close = torch.isclose(A, B, rtol=rtol, atol=atol)
    elif A.dtype == torch.float32:
        is_close = torch.isclose(A, B, rtol=rtol, atol=atol)
    elif A.dtype == torch.bfloat16:
        is_close = torch.isclose(A, B, rtol=rtol, atol=atol)
    elif A.dtype == torch.long:
        is_close = A == B

    # 如果全部都 Close，则通过
    if torch.all(is_close):
        print("Tensor values are close enough.")
        return True

    # 定位错误
    print("Tensor values are not close enough.")

    # 找到不一致的位置 (False 的位置)
    mismatch_indices = torch.nonzero(~is_close, as_tuple=False)
    num_mismatches = mismatch_indices.shape[0]
    total_elements = A.numel()

    print(f"   Mismatched elements: {num_mismatches} / {total_elements} ({(num_mismatches/total_elements)*100:.2f}%)")

    # 计算最大绝对误差
    diff = torch.abs(A - B)
    max_diff = torch.max(diff)
    print(f"   Max absolute difference: {max_diff.item()}")

    # 4. 打印前 N 个具体的错误位置供调试
    print("\n   --- First 5 Mismatches ---")
    for i in range(min(5, num_mismatches)):
        idx = mismatch_indices[i]  # 获取由维度组成的索引，如 [0, 2, 1]

        # 将 tensor 索引转为 tuple 以便用于访问
        idx_tuple = tuple(idx.tolist())

        val_a = A[idx_tuple].item()
        val_b = B[idx_tuple].item()
        abs_err = abs(val_a - val_b)

        print(f"   Index {idx_tuple}:")
        print(f"     A: {val_a}")
        print(f"     B: {val_b}")
        print(f"     Diff: {abs_err}")

    return False


def save_tensor(A: torch.Tensor, rel_path: str):
    """
    保存 hidden_states tensor 到文件（相对路径）。

    参数:
        hidden_states: 要保存的 torch.Tensor(如 outputs.last_hidden_state)
        rel_path: 文件相对路径，如 "cache/hs.pt"

    返回:
        实际写入的文件绝对路径
    """
    if not isinstance(A, torch.Tensor):
        raise TypeError(f"A must be a torch.Tensor, got {type(A)}")

    abs_path = os.path.abspath(rel_path)
    os.makedirs(os.path.dirname(abs_path) or ".", exist_ok=True)

    # 建议保存到 CPU，避免某些环境下加载时没有 GPU/设备不匹配
    to_save = A.detach().cpu()
    torch.save(to_save, abs_path)
    print(f"Saved tensor of shape {A.shape} and dtype {A.dtype} to {abs_path}")
    exit(0)


def load_tensor_pt(rel_path: str, map_location: str | torch.device = "cpu") -> torch.Tensor:
    """
    读取 .pt 文件并返回 torch.Tensor

    参数:
        rel_path: 文件相对路径，如 "cache/last_hidden_state.pt"
        map_location: 设备映射，默认 "cpu"

    返回:
        读取到的 torch.Tensor
    """
    obj = torch.load(rel_path, map_location=map_location)

    if isinstance(obj, torch.Tensor):
        return obj
    raise TypeError(f"Loaded object is not a torch.Tensor, got {type(obj)}")
