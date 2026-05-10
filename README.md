# Finance SLM Distillation

This project focuses on reasoning distillation for small language models (SLMs)
using curated Chain-of-Thought (CoT) datasets.

## Project Structure

- `src/` : preprocessing and training scripts
- `notebooks/` : exploratory analysis and experiments
- `configs/` : training configurations

## Dataset

The processed datasets are hosted on Hugging Face:

https://huggingface.co/datasets/ash0t/finance-slm-distillation-data

Datasets include:
- filtered reasoning samples
- train/test splits
- SFT-ready JSONL files

## Training

Example:

```bash
python train.py
