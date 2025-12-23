import json
from pathlib import Path

import accelerate
import torch
from huggingface_hub import file_exists, hf_hub_download


"""
函数功能:
这段脚本用于从Hugging Face下载预训练的Qwen3-VL-30B-A3B-Thinking模型配置,
修改其架构参数(缩小模型规模),创建一个新的随机初始化的小型模型,并保存到本地。
主要用于模型架构测试、实验或创建一个可快速加载的小模型框架。

@param: 无直接参数(脚本式代码)
@return: 无返回值,但会在指定路径保存处理器和模型文件
"""

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    GenerationConfig,
    Qwen3VLMoeForConditionalGeneration,
    set_seed,
)

source_model_id = "Qwen/Qwen3-VL-30B-A3B-Thinking"
save_folder = "/root/autodl-tmp/model/qwen3-vl-moe"

processor = AutoProcessor.from_pretrained(source_model_id, trust_remote_code=True)
processor.save_pretrained(save_folder)

with open(hf_hub_download(source_model_id, filename="config.json", repo_type="model"), "r", encoding="utf-8") as f:
    config_json = json.load(f)

# 更新文本配置(text_config)部分的参数,这些参数定义了语言模型的架构
config_json["text_config"].update(
    {
        "head_dim": 128,  # 每个注意力头的维度
        "hidden_size": 2048,  # 隐藏层大小(从原30B模型缩小)
        "intermediate_size": 6144,  # FFN中间层大小
        "moe_intermediate_size": 768,  # MoE(Mixture of Experts)专家网络的中间层大小
        "num_hidden_layers": 2,  # 隐藏层数量(大幅减少,原模型可能有数十层)
        "num_attention_heads": 32,  # 注意力头的数量
        "num_key_value_heads": 4,  # KV缓存的头数量(用于GQA - Grouped Query Attention)
        "num_experts": 128,  # MoE架构中专家的数量
    }
)
# 更新RoPE(Rotary Position Embedding)缩放配置中的mrope_section参数
# 这定义了多分辨率位置编码的分段方式
config_json["text_config"]["rope_scaling"]["mrope_section"] = [24, 20, 20]

# 更新视觉配置(vision_config)部分的参数,定义视觉编码器的架构
config_json["vision_config"].update(
    {
        "hidden_size": 1152,  # 视觉编码器隐藏层大小
        "intermediate_size": 4304,  # 视觉FFN中间层大小
        "num_heads": 16,  # 视觉注意力头数量
        "out_hidden_size": 2048,  # 输出隐藏层大小(需匹配文本模型)
        "depth": 6,  # 视觉Transformer的深度(层数)
        "deepstack_visual_indexes": [1, 3, 5],  # DeepStack架构中用于深度融合的层索引
    }
)

# 将修改后的配置写入本地保存文件夹的 config.json
with open(f"{save_folder}/config.json", "w", encoding="utf-8") as f:
    # indent=2使JSON格式化输出,便于阅读
    json.dump(config_json, f, indent=2)

# 保存的本地配置文件加载AutoConfig对象
config = AutoConfig.from_pretrained(
    save_folder,
    trust_remote_code=True,  # 允许执行自定义代码
)
print(config)
torch.set_default_dtype(torch.bfloat16)
# 使用修改后的配置创建一个新的Qwen3VLMoE模型实例(随机初始化,非预训练权重)
model = Qwen3VLMoeForConditionalGeneration(config)


# torch.set_default_dtype(torch.float32)
# 检查源模型仓库中是否存在generation_config.json文件
if file_exists(filename="generation_config.json", repo_id=source_model_id, repo_type="model"):
    # 如果存在,从源模型加载生成配置
    model.generation_config = GenerationConfig.from_pretrained(
        source_model_id,
        trust_remote_code=True,
    )
    # 设置生成配置为采样模式(而非贪婪解码)
    model.generation_config.do_sample = True
    # 打印生成配置
    print(model.generation_config)

model = model.cpu()
# 使用torch.no_grad()上下文管理器禁用梯度计算(节省内存)
with torch.no_grad():
    # 遍历模型所有命名参数,sorted确保按名称排序
    for name, p in sorted(model.named_parameters()):
        # 使用正态分布(均值0,标准差0.1)随机初始化每个参数
        torch.nn.init.normal_(p, 0, 0.1)
        # 打印参数名称和形状
        print(name, p.shape)

# 将随机初始化的模型保存到本地文件夹
model.save_pretrained(save_folder)
