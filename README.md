# Lab 21 — LoRA / QLoRA Fine-tuning · Submission

> **AICB-P2T3 · Ngày 21 · Chương 5 — Fine-tuning & An Toàn**
> Học viên: **Phạm Hữu Hoàng Hiệp** — MSSV: **2A202600415**

---

## 📋 Tổng quan

Repo này là kết quả thực hành **Lab 21**: fine-tune LLM **Qwen2.5-3B** (4-bit NF4) bằng kỹ thuật **LoRA + QLoRA** trên Free Colab T4, sau đó chạy thí nghiệm so sánh **3 rank khác nhau (r=8, r=16, r=64)** để hiểu trade-off giữa training time, VRAM, và quality.

**Dataset**: `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated` — 200 samples (180 train / 20 eval, seed=42).

---

## 📂 Cấu trúc repo

```
.
├── REPORT.md                            ← Evaluation report (6 sections)
├── notebooks/
│   └── Lab21_LoRA_Finetuning_T4.ipynb   ← Notebook đã chạy (stripped outputs)
├── adapters/r16/                        ← Best LoRA adapter
│   ├── adapter_model.safetensors        (14 MB)
│   └── adapter_config.json
├── results/
│   ├── rank_experiment_summary.csv      ← Bảng metrics 4 rows (base + r=8/16/64)
│   ├── qualitative_comparison.csv       ← 5 prompts side-by-side base vs FT
│   └── loss_curve.png                   ← Training loss curve r=16
├── requirements.txt                     ← Pinned versions
├── lab21_2A202600415_PhamHuuHoangHiep.zip   ← Submission bundle cho LMS
└── README.md (this file)
```

---

## 🎯 Highlight kết quả

| Rank | α   | Trainable | Train Time | Peak VRAM | Eval PPL | vs Base   |
|------|-----|-----------|------------|-----------|----------|-----------|
| Base | —   | 0         | —          | —         | **5.49** | —         |
| 8    | 16  | 1.84M     | 4.33 phút  | 7.22 GB   | **4.75** | −13.6 %   |
| **16** | 32  | **3.69M** | **4.28 phút** | **6.62 GB** | **4.55** | **−17.1 %** ← best ROI |
| 64   | 128 | 14.7M     | 4.04 phút  | 8.00 GB   | **4.38** | −20.2 %   |

**Conclusion**: r=16 là sweet spot — gain 17.1 % perplexity vs base với chỉ 0.12 % trainable params và VRAM thấp nhất trong 3 ranks. Đọc chi tiết phân tích trong [REPORT.md](REPORT.md).

---

## 🔁 Reproducibility

Để re-run từ đầu:

1. Mở [notebooks/Lab21_LoRA_Finetuning_T4.ipynb](notebooks/Lab21_LoRA_Finetuning_T4.ipynb) trên Google Colab
2. **Runtime → Change runtime type → T4 GPU**
3. **Runtime → Run all** (~12–15 phút trên T4)
4. Random seed cố định **42** ⇒ kết quả deterministic

**Dependencies**: xem [requirements.txt](requirements.txt). Colab tự install qua cell 4.

---

## 🛠️ Stack

| Tool | Version | Vai trò |
|------|---------|---------|
| Unsloth | 2026.5.2 | Custom CUDA kernels, 2× faster training |
| TRL | 0.15.2 | `SFTTrainer` |
| PEFT | latest | LoRA wrapper |
| bitsandbytes | latest | 4-bit NF4 quantization + paged AdamW |
| Transformers | 5.5.0 | Base model loading |
| PyTorch | 2.10.0+cu128 | Backend |

---

## 📤 Submission

- **ZIP cho LMS**: `lab21_2A202600415_PhamHuuHoangHiep.zip` (13 MB) — Option A Lightweight format theo rubric
- **Git link**: repo này (public on GitHub)

---

## 📜 References

- LoRA paper — Hu et al. 2021 — https://arxiv.org/abs/2106.09685
- QLoRA paper — Dettmers et al. 2023 — https://arxiv.org/abs/2305.14314
- Unsloth — https://github.com/unslothai/unsloth
