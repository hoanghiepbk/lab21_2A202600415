# Lab 21 — External Links

> **Học viên**: Phạm Hữu Hoàng Hiệp — MSSV: 2A202600415
> **Submission option**: B (GitHub + HuggingFace Hub) — bonus +5 pts

---

## 🔗 Public verifiable links

### 1. GitHub Repository (source code + report)

**https://github.com/hoanghiepbk/Day21-Track3-Finetuning-LLMs-LoRA-QLoRA**

Chứa:
- `REPORT.md` — Full evaluation report (6 sections)
- `notebooks/Lab21_LoRA_Finetuning_T4.ipynb` — Notebook đã chạy (stripped outputs)
- `results/` — Metrics CSVs + loss curve PNG
- `STUDENT_NOTES.md` — Submission summary

### 2. HuggingFace Hub Adapter (model weights)

**https://huggingface.co/hiepphambk/lap21_2A202600415**

Chứa:
- `adapter_model.safetensors` (14.7 MB) — LoRA weights cho Qwen2.5-3B 4-bit
- `adapter_config.json` — PEFT config (r=16, α=32, q+v target)
- `README.md` — Model card với metrics + usage example

---

## 📊 Quick reference

| Metric | Value |
|---|---|
| Base model | `unsloth/Qwen2.5-3B-bnb-4bit` |
| Best adapter | r=16 (α=32) |
| Eval perplexity (FT) | **4.55** |
| Eval perplexity (base) | 5.49 |
| Improvement vs base | **−17.1 %** |
| Trainable params | 3,686,400 (0.12 % of base) |
| Training time | 4.28 phút (T4) |
| Peak VRAM | 6.62 GB |
| Random seed | 42 (reproducible) |

---

## 🚀 Reproducibility — 1 cell to verify

```python
# Cell A: Reload từ HF Hub và compute perplexity
from peft import PeftModel
from unsloth import FastLanguageModel
import torch

base, tok = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-3B-bnb-4bit",
    max_seq_length=512,
    load_in_4bit=True,
)
model = PeftModel.from_pretrained(base, "hiepphambk/lap21_2A202600415")
FastLanguageModel.for_inference(model)

# Optional: compute perplexity on user-provided eval set → expect ~4.55
```

---

## ✅ Submission checklist

- [x] Adapter r=16 pushed to HF Hub (public, no auth required)
- [x] GitHub repo public với REPORT + results + notebook
- [x] LINKS.md (this file) trong submission ZIP
- [x] REPORT.md có HF Hub link ở header + Appendix
- [x] Submission ZIP nhỏ (~1 MB, không chứa adapter weights)
