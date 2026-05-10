# Finance SLM Distillation

Reasoning distillation pipeline for Small Language Models (SLMs) specialized in quantitative finance, actuarial science and mathematical reasoning.

The project focuses on transferring reasoning capabilities from DeepSeek-style Chain-of-Thought (CoT) traces toward compact instruction-tuned models such as Qwen2.5-7B using supervised fine-tuning (SFT) and QLoRA.

---

# Objectives

The main objectives of this project are:

- improve financial reasoning capabilities of SLMs;
- improve multi-step quantitative reasoning;
- reduce numerical hallucinations;
- produce concise and professional final answers;
- study reasoning transfer through Chain-of-Thought distillation.

The training strategy follows a two-stage pipeline:

1. reasoning transfer using explicit `<think>` traces;
2. response alignment using concise final answers only.

---

# Project Structure

```text
finance-slm-distillation/
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_preparation.py
│   ├── train.py
│   ├── evaluation.py
│   └── inference.py
│
├── configs/
│   ├── training.yaml
│   └── inference.yaml
│
├── notebooks/
│   ├── 01_dataset_preparation.ipynb
│   ├── 02_training_lora.ipynb
│   ├── 03_evaluation.ipynb
│   └── 04_error_analysis.ipynb
│
├── data/
│   ├── processed/
│   │   ├── reasoning_train.jsonl
│   │   ├── response_train.jsonl
│   │   ├── test.jsonl
│   │   ├── reasoning_train.parquet
│   │   ├── response_train.parquet
│   │   └── test.parquet
│   │
│   └── evaluation/
│       ├── predictions.csv
│       ├── benchmark_results.csv
│       └── hallucination_analysis.csv
│
├── adapters/
│   └── qwen2.5-7b-finance-lora/
│
├── reports/
│   ├── figures/
│   ├── tables/
│   └── final_report.pdf
│
├── requirements.txt
└── README.md
```

---

# Dataset

The processed datasets are hosted on Hugging Face:

https://huggingface.co/datasets/ash0t/finance-slm-distillation-data

The repository contains three datasets:

| Dataset | Description |
|---|---|
| `reasoning_train` | examples containing reasoning traces (`<think>`) |
| `response_train` | concise final-answer-only examples |
| `test` | evaluation dataset without reasoning traces |

Parquet versions are also provided for analysis and reproducibility purposes.

---

# Dataset Preparation Pipeline

The dataset preparation pipeline includes:

- response filtering;
- language filtering;
- finance-specific heuristic filtering;
- reasoning trace extraction;
- reasoning quality heuristics;
- train/test stratified split;
- JSONL export for SFT training.

The preparation script follows industrial Python practices:
- PEP8;
- docstrings;
- centralized configuration;
- logging;
- modular exports.

---

# Training Pipeline

The fine-tuning strategy follows two sequential phases.

## Phase 1 — Reasoning Transfer

The model is first trained on `reasoning_train`.

This phase teaches:
- Chain-of-Thought structures;
- financial reasoning patterns;
- multi-step calculations;
- quantitative logic.

## Phase 2 — Response Alignment

The same LoRA adapters are then continuously trained on `response_train`.

This phase improves:
- conciseness;
- response clarity;
- professional formatting;
- suppression of verbose reasoning traces.

---

# Models

Base model:

```text
unsloth/Qwen2.5-7B-Instruct-bnb-4bit
```

Training framework:
- Hugging Face Transformers
- TRL
- Unsloth
- QLoRA
- PEFT / LoRA

---

# Example Usage

## Dataset Preparation

```bash
python -m src.data_preparation
```

## Training

```bash
python -m src.train
```

## Evaluation

```bash
python -m src.evaluation
```

---

# Data Format

Training samples follow a Qwen-compatible chat template:

```text
<|im_start|>system
...
<|im_end|>

<|im_start|>user
...
<|im_end|>

<|im_start|>assistant
...
<|im_end|>
```

JSONL schema:

```json
{
  "text": "<formatted conversation>"
}
```

---

# Intended Use

This project is intended for:
- research on reasoning distillation;
- financial LLM experimentation;
- SLM fine-tuning;
- QLoRA experimentation;
- Chain-of-Thought transfer research.

---

# Limitations

The finance filtering strategy is heuristic and keyword-based.

As a result:
- some false positives may remain;
- some relevant examples may be excluded;
- reasoning quality evaluation remains partially heuristic.

This project is primarily intended for research and experimentation purposes.

---

# Acknowledgements

Original dataset:

```text
heladell/Finance_DeepSeek-R1-Distill-dataset
```

Base model:

```text
unsloth/Qwen2.5-7B-Instruct-bnb-4bit
```
