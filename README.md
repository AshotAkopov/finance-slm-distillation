# Finance Instruction Dataset Preparation Pipeline

This project prepares a curated finance-oriented instruction dataset for conversational supervised fine-tuning (SFT) of large language models.

The pipeline focuses on:
- finance-domain filtering
- reasoning-quality filtering
- answer-quality filtering
- conversational formatting
- realistic train/test split construction for reasoning evaluation

The final dataset is optimized for downstream QLoRA and conversational SFT workflows using Qwen-style chat templates.

---

# Project Structure

```text
finance_sft_dataset_pipeline/
├── configs/
│   └── config.py
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── reports/
│   ├── figures/
│   └── tables/
├── src/
│   └── prepare_finance_dataset.py
├── requirements.txt
└── README.md
