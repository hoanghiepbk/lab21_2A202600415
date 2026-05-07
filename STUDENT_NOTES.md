# Lab 21 — Submission Notes

> **AICB-P2T3 · Ngày 21 · Chương 5 — Fine-tuning & An Toàn**
> Học viên: **Phạm Hữu Hoàng Hiệp** — MSSV: **2A202600415**
> Submission option: **A (Lightweight ZIP)**

> 📌 README.md trong repo là tài liệu gốc của giảng viên. File này (`STUDENT_NOTES.md`) chỉ chứa thông tin nộp bài của học viên.

---

## 📋 Tổng quan

Repo này là kết quả thực hành **Lab 21**: fine-tune **Qwen2.5-3B** (4-bit NF4) bằng **LoRA + QLoRA** trên Free Colab T4, sau đó chạy thí nghiệm so sánh **3 rank (r=8, r=16, r=64)**.

**Dataset**: `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated` — 200 samples (180 train / 20 eval, seed=42).

---

## 📂 Deliverables (theo Option A của rubric)

```
.
├── REPORT.md                            ← Evaluation report 6 sections
├── notebooks/
│   └── Lab21_LoRA_Finetuning_T4.ipynb   ← Notebook đã chạy (stripped)
├── adapters/r16/                        ← Best LoRA adapter
│   ├── adapter_model.safetensors        (14 MB)
│   └── adapter_config.json
├── results/
│   ├── rank_experiment_summary.csv      ← 4 rows: base + r=8/16/64
│   ├── qualitative_comparison.csv       ← 5 prompts side-by-side
│   └── loss_curve.png                   ← r=16 training loss
├── requirements.txt                     ← Pinned versions
└── lab21_2A202600415_PhamHuuHoangHiep.zip   ← Bundle nộp LMS
```

---

## 🎯 Highlight kết quả

| Rank | α   | Trainable | Train Time | Peak VRAM | Eval PPL | vs Base   |
|------|-----|-----------|------------|-----------|----------|-----------|
| Base | —   | 0         | —          | —         | **5.49** | —         |
| 8    | 16  | 1.84M     | 4.33 phút  | 7.22 GB   | **4.75** | −13.6 %   |
| **16** | 32  | **3.69M** | **4.28 phút** | **6.62 GB** | **4.55** | **−17.1 %** ← best ROI |
| 64   | 128 | 14.7M     | 4.04 phút  | 8.00 GB   | **4.38** | −20.2 %   |

**Conclusion**: r=16 là sweet spot — gain 17.1 % perplexity vs base với chỉ 0.12 % trainable params và VRAM thấp nhất. Phân tích chi tiết trong [REPORT.md](REPORT.md).

---

## 🔁 Reproducibility

1. Mở [notebooks/Lab21_LoRA_Finetuning_T4.ipynb](notebooks/Lab21_LoRA_Finetuning_T4.ipynb) trên Google Colab
2. **Runtime → Change runtime type → T4 GPU**
3. **Runtime → Run all** (~12–15 phút)
4. Random seed = **42** ⇒ deterministic outputs

Dependencies pin trong [requirements.txt](requirements.txt). Colab tự install qua cell 4.

---

## 📤 Submission

| Phương thức | File / URL |
|---|---|
| **LMS (ZIP)** | [`lab21_2A202600415_PhamHuuHoangHiep.zip`](lab21_2A202600415_PhamHuuHoangHiep.zip) (13 MB) |
| **Git link** | https://github.com/hoanghiepbk/Day21-Track3-Finetuning-LLMs-LoRA-QLoRA |

---

## ⚖️ Honor code

Theo rubric line 276: *"được dùng AI assistant (Claude, ChatGPT) để debug và viết report — nhưng phải tự chạy training trên máy của bạn"*.

- Training: **tự chạy** trên Google Colab Free T4 (account của học viên)
- Adapter weights: **tự sinh** từ session Colab nói trên (seed=42 verifiable)
- Notebook patches + REPORT writing: hỗ trợ bởi Claude (theo rule cho phép)
