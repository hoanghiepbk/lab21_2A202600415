---
base_model: unsloth/Qwen2.5-3B-bnb-4bit
library_name: peft
tags:
  - lora
  - qlora
  - peft
  - vietnamese
  - alpaca
  - aicb
language:
  - vi
license: apache-2.0
datasets:
  - 5CD-AI/Vietnamese-alpaca-gpt4-gg-translated
---

# Lab21 — Qwen2.5-3B Vietnamese Alpaca LoRA (r=16)

LoRA adapter fine-tuned trên `unsloth/Qwen2.5-3B-bnb-4bit` (4-bit NF4) bằng QLoRA + Unsloth + TRL SFTTrainer.

> **Lab 21 · AICB-P2T3 Day 21 — Fine-tuning LLMs**
> Học viên: **Phạm Hữu Hoàng Hiệp** (MSSV: 2A202600415)

## 📊 Kết quả

| Metric | Value |
|---|---|
| **Eval perplexity (FT, r=16)** | **4.55** |
| Eval perplexity (base) | 5.49 |
| **Improvement vs base** | **−17.1 %** |
| Train loss → final | 1.61 → 1.39 |
| Trainable params | 3,686,400 (0.12 % of base) |
| Training time | 4.28 phút (T4) |
| Peak VRAM | 6.62 GB |

## ⚙️ LoRA config

```python
r = 16
lora_alpha = 32
target_modules = ["q_proj", "v_proj"]
lora_dropout = 0
bias = "none"
random_state = 42
```

## 📚 Training config

| Setting | Value |
|---|---|
| Base model | `unsloth/Qwen2.5-3B-bnb-4bit` (NF4 4-bit) |
| Dataset | `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated` (200 samples) |
| Train / Eval split | 180 / 20 (90/10, seed=42) |
| max_seq_length | 512 (p95 round-up) |
| Epochs | 3 |
| Learning rate | 2e-4, cosine schedule |
| Warmup ratio | 0.10 |
| Effective batch | 8 (per_device=1 × grad_accum=8) |
| Optimizer | `adamw_8bit` (paged AdamW) |
| Gradient checkpointing | unsloth (−60 % VRAM) |
| GPU | NVIDIA Tesla T4 (Free Colab) |

## 🚀 Cách dùng

```python
from peft import PeftModel
from unsloth import FastLanguageModel

# Load base
base_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-3B-bnb-4bit",
    max_seq_length=512,
    load_in_4bit=True,
)

# Attach this LoRA adapter
model = PeftModel.from_pretrained(base_model, "hiepphambk/lap21_2A202600415")
FastLanguageModel.for_inference(model)

# Generate
prompt = "### Instruction:\nGiải thích khái niệm machine learning cho người mới bắt đầu.\n\n### Response:\n"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
out = model.generate(**inputs, max_new_tokens=200, temperature=0.7, top_p=0.9, do_sample=True)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

## 📂 Repo gốc

- **GitHub**: https://github.com/hoanghiepbk/Day21-Track3-Finetuning-LLMs-LoRA-QLoRA
- **Lab rubric**: `Lab21_Rubric_and_Format.md` (trong repo gốc)
- **Full evaluation report**: `REPORT.md` (trong repo gốc)

## 📜 References

- LoRA — Hu et al. 2021 — https://arxiv.org/abs/2106.09685
- QLoRA — Dettmers et al. 2023 — https://arxiv.org/abs/2305.14314
- Unsloth — https://github.com/unslothai/unsloth
