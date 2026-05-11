# Finance Instruction Dataset Preparation Pipeline

This project prepares a curated finance-oriented instruction dataset for conversational supervised fine-tuning.

## Project Structure

```text
finance_sft_dataset_pipeline/
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── reports/
│   ├── figures/
│   └── tables/
├── src/
│   └── prepare_dataset.py
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python src/prepare_dataset.py --config configs/config.yaml
```

## Outputs

The pipeline exports:

- `data/processed/train_sft.jsonl`
- `data/processed/test_sft.jsonl`
- `data/processed/df_filtered.parquet`
- `data/processed/df_clean.parquet`
- `data/processed/df_train.parquet`
- `data/processed/df_test.parquet`
- `reports/tables/preprocessing_report.csv`
- `reports/tables/finance_filtering_table.csv`
- `reports/tables/reasoning_quality_table.csv`
- `reports/figures/*.png`

## Notes

The finance filtering step is intentionally interpretable and auditable. It relies on keyword-based filtering and may produce false positives or false negatives.