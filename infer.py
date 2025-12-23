import torch
import transformers
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    Qwen3VLMoeForConditionalGeneration,
)

model_dir = "/root/autodl-tmp/model/qwen3-vl-moe"
tokenizer = AutoTokenizer.from_pretrained(model_dir, fix_mistral_regex=True)  # 修复正则表达式问题
print("bos_token:", tokenizer.bos_token, "bos_token_id:", tokenizer.bos_token_id)
print("pad_token:", tokenizer.pad_token, "pad_token_id:", tokenizer.pad_token_id)
print("eos_token:", tokenizer.eos_token, "eos_token_id:", tokenizer.eos_token_id)

model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
    model_dir,
    dtype=torch.bfloat16,
    device_map="cuda",
    # attn_implementation="flash_attention_2",
)
cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)

processor = AutoProcessor.from_pretrained(model_dir)
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "./demo.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# Preparation for inference
inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(
    model.device
)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=32)
generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(output_text)
