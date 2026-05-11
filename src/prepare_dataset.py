"""Finance SFT dataset preparation pipeline.

This script converts a raw Hugging Face finance instruction dataset into
curated conversational SFT datasets.

Pipeline steps:
    1. Load source dataset.
    2. Remove duplicates and missing responses.
    3. Filter irrelevant task categories and non-English samples.
    4. Apply finance-domain filtering.
    5. Apply reasoning-quality filtering.
    6. Build train/test splits and SFT-formatted JSONL files.
    7. Export datasets, reports, and visual diagnostics.
"""

from __future__ import annotations

import argparse
import logging
import random
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from datasets import load_dataset
from sklearn.model_selection import train_test_split


CONFIG_PATH = Path("configs/config.yaml")
THINK_PATTERN = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
THINK_EXTRACT_PATTERN = re.compile(r"<think>(.*?)</think>", flags=re.DOTALL)

LOGGER = logging.getLogger(__name__)


def setup_logging() -> None:
    """Configure pipeline logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_config(config_path: Path) -> dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        config_path: Path to the configuration file.

    Returns:
        Parsed configuration dictionary.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def set_random_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def ensure_directories(config: dict[str, Any]) -> None:
    """Create output directories if they do not already exist."""
    for key in ("processed_dir", "figures_dir", "tables_dir"):
        Path(config["paths"][key]).mkdir(parents=True, exist_ok=True)


def log_pipeline_step(
    pipeline_stats: list[dict[str, Any]],
    step: str,
    dataframe: pd.DataFrame,
) -> None:
    """Log and store the number of samples after a preprocessing step."""
    samples = len(dataframe)
    pipeline_stats.append({"step": step, "samples": samples})
    LOGGER.info("%s: %d samples", step, samples)


def load_source_dataset(config: dict[str, Any]) -> pd.DataFrame:
    """Load the Hugging Face dataset as a Pandas dataframe."""
    dataset_config = config["dataset"]
    LOGGER.info(
        "Loading dataset %s [%s]",
        dataset_config["name"],
        dataset_config["split"],
    )
    dataset = load_dataset(
        dataset_config["name"],
        split=dataset_config["split"],
    )
    return dataset.to_pandas()


def is_finance_related(row: pd.Series, keywords: list[str]) -> bool:
    """Return whether a sample contains finance-related keywords."""
    text = (
        f"{row['instruction']} "
        f"{row['response']} "
        f"{row.get('intent', '')} "
        f"{row.get('knowledge', '')}"
    ).lower()
    return any(keyword.lower() in text for keyword in keywords)


def has_think_block(response: Any) -> bool:
    """Return whether a response contains a `<think>` reasoning block."""
    if response is None:
        return False
    return bool(THINK_PATTERN.search(str(response)))


def extract_think(response: Any) -> str:
    """Extract reasoning content enclosed in `<think>` tags."""
    match = THINK_EXTRACT_PATTERN.search(str(response))
    return match.group(1).strip() if match else ""


def remove_think(response: Any) -> str:
    """Remove `<think>` blocks and keep the visible final answer."""
    return THINK_PATTERN.sub("", str(response)).strip()


def is_good_reasoning(row: pd.Series, config: dict[str, Any]) -> bool:
    """Evaluate whether a reasoning trace is suitable for training."""
    instruction = str(row["instruction"])
    response = str(row["response"])

    think = extract_think(response)
    final_answer = remove_think(response)

    instruction_words = len(instruction.split())
    think_words = len(think.split())
    final_answer_words = len(final_answer.split())

    think_instruction_ratio = think_words / max(instruction_words, 1)

    noise_score = sum(
        think.lower().count(word)
        for word in config["noise_words"]
    )

    has_structure = any(
        marker in think.lower()
        for marker in config["structure_markers"]
    )

    has_formula = any(
        symbol in think
        for symbol in config["formula_symbols"]
    )

    if think_words == 0:
        return False
    if think_words < config["min_reasoning_words"]:
        return False
    if final_answer_words <= config["min_final_answer_words"]:
        return False
    if noise_score > config["max_noise_score_global"]:
        return False
    if think_instruction_ratio > config["max_reasoning_instruction_ratio"]:
        return False
    if think_instruction_ratio <= 4:
        return noise_score <= 8
    if 4 < think_instruction_ratio <= 10:
        return noise_score <= 5 and (has_structure or has_formula)
    if 10 < think_instruction_ratio <= 20:
        return noise_score <= 2 and has_structure and has_formula

    return False


def clean_difficulty(value: str) -> str:
    """Merge rare difficulty levels to stabilize stratified splitting."""
    if value == "very easy":
        return "easy"
    if value == "very hard":
        return "hard"
    return value


def build_preprocessing_report(
    pipeline_stats: list[dict[str, Any]],
) -> pd.DataFrame:
    """Build preprocessing statistics with retention metrics."""
    stats_df = pd.DataFrame(pipeline_stats)

    stats_df["retention_rate"] = (
        100 * stats_df["samples"] / stats_df["samples"].iloc[0]
    ).round(2)

    stats_df["removed_samples"] = (
        stats_df["samples"].shift(1) - stats_df["samples"]
    ).fillna(0).astype(int)

    stats_df["step_retention_rate"] = (
        100 * stats_df["samples"] / stats_df["samples"].shift(1)
    ).fillna(100).round(2)

    return stats_df


def format_prompt(row: pd.Series, system_message: str) -> str:
    """Format a sample using a Qwen-compatible chat template."""
    return f"""<|im_start|>system
{system_message}
<|im_end|>

<|im_start|>user
{row["instruction"]}
<|im_end|>

<|im_start|>assistant
{row["response"]}
<|im_end|>
"""


def plot_dataset_size(stats_df: pd.DataFrame, output_path: Path) -> None:
    """Save a bar chart of dataset size across preprocessing steps."""
    ax = stats_df.plot(
        x="step",
        y="samples",
        kind="bar",
        legend=False,
        figsize=(10, 5),
    )
    ax.set_title("Dataset Size Across Preprocessing Steps")
    ax.set_xlabel("Preprocessing Step")
    ax.set_ylabel("Number of Samples")
    plt.tight_layout()
    ax.get_figure().savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_retention_rate(stats_df: pd.DataFrame, output_path: Path) -> None:
    """Save a line chart of retention rate across preprocessing steps."""
    ax = stats_df.plot(
        x="step",
        y="retention_rate",
        kind="line",
        marker="o",
        legend=False,
        figsize=(10, 5),
    )
    ax.set_title("Dataset Retention Rate Across Preprocessing Steps")
    ax.set_xlabel("Preprocessing Step")
    ax.set_ylabel("Retention Rate (%)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    ax.get_figure().savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_finance_heatmap(
    finance_filtering_table: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save a heatmap of finance relevance across task categories."""
    matrix = finance_filtering_table.iloc[:-1, :-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    image = ax.imshow(matrix)

    ax.set_title("Finance Relevance Across Task Categories")
    ax.set_xlabel("Finance-related")
    ax.set_ylabel("Task Category")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)

    fig.colorbar(image, ax=ax)
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_reasoning_quality(
    reasoning_quality_table: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save a stacked bar chart of reasoning-quality distribution."""
    matrix = reasoning_quality_table.iloc[:-1, :-1]
    ax = matrix.plot(kind="bar", stacked=True, figsize=(8, 5))

    ax.set_title("Reasoning Quality by Reasoning Trace Availability")
    ax.set_xlabel("Contains <think> Trace")
    ax.set_ylabel("Number of Samples")
    plt.tight_layout()
    ax.get_figure().savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_reasoning_proportion(
    dataframe: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save a pie chart of final reasoning trace proportion."""
    ax = dataframe["has_think"].value_counts().plot(
        kind="pie",
        autopct="%1.1f%%",
        figsize=(6, 6),
    )
    ax.set_title("Final Dataset Reasoning Trace Proportion")
    ax.set_ylabel("")
    plt.tight_layout()
    ax.get_figure().savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_difficulty_distribution(
    dataframe: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save a bar chart of final difficulty distribution."""
    ax = dataframe["difficulty_clean"].value_counts().plot(
        kind="bar",
        figsize=(7, 4),
    )
    ax.set_title("Final Difficulty Distribution")
    ax.set_xlabel("Difficulty Level")
    ax.set_ylabel("Number of Samples")
    plt.tight_layout()
    ax.get_figure().savefig(output_path, bbox_inches="tight")
    plt.close()


def run_pipeline(config: dict[str, Any]) -> None:
    """Run the full dataset preparation pipeline."""
    pipeline_stats: list[dict[str, Any]] = []

    processed_dir = Path(config["paths"]["processed_dir"])
    figures_dir = Path(config["paths"]["figures_dir"])
    tables_dir = Path(config["paths"]["tables_dir"])
    preprocessing_config = config["preprocessing"]

    dataframe = load_source_dataset(config)
    log_pipeline_step(pipeline_stats, "Initial dataset", dataframe)

    dataframe = dataframe.drop_duplicates(
        subset=preprocessing_config["duplicate_subset"],
    ).copy()
    log_pipeline_step(pipeline_stats, "Deduplication", dataframe)

    dataframe = dataframe[dataframe["response"].notna()].copy()
    log_pipeline_step(pipeline_stats, "Remove missing responses", dataframe)

    dataframe = dataframe[
        ~dataframe["task_category"].isin(
            preprocessing_config["excluded_categories"],
        )
    ].copy()
    log_pipeline_step(pipeline_stats, "Relevant category filtering", dataframe)

    dataframe = dataframe[
        dataframe["language"] == preprocessing_config["language"]
    ].copy()
    log_pipeline_step(pipeline_stats, "English-only filtering", dataframe)

    dataframe["finance_related"] = dataframe.apply(
        is_finance_related,
        axis=1,
        keywords=config["finance_filter"]["keywords"],
    )

    finance_filtering_table = pd.crosstab(
        dataframe["task_category"],
        dataframe["finance_related"],
        margins=True,
    )

    if config["finance_filter"]["preserve_math_category"]:
        dataframe = dataframe[
            (dataframe["task_category"] == "Math")
            | (dataframe["finance_related"])
        ].copy()
    else:
        dataframe = dataframe[dataframe["finance_related"]].copy()

    log_pipeline_step(pipeline_stats, "Finance filtering", dataframe)

    dataframe["has_think"] = dataframe["response"].apply(has_think_block)
    dataframe["good_reasoning"] = dataframe.apply(
        is_good_reasoning,
        axis=1,
        config=config["reasoning_quality"],
    )

    reasoning_quality_table = pd.crosstab(
        dataframe["has_think"],
        dataframe["good_reasoning"],
        margins=True,
    )

    dataframe = dataframe[
        (~dataframe["has_think"])
        | (dataframe["good_reasoning"])
    ].copy()

    log_pipeline_step(pipeline_stats, "Reasoning quality filtering", dataframe)

    df_filtered = dataframe.copy()

    clean_columns = [
        "instruction",
        "response",
        "task_category",
        "difficulty",
        "input_quality",
        "has_think",
        "finance_related",
        "good_reasoning",
    ]

    df_clean = df_filtered[clean_columns].copy()
    df_clean["difficulty_clean"] = df_clean["difficulty"].apply(
        clean_difficulty,
    )
    df_clean["stratify_col"] = (
        df_clean["has_think"].astype(str)
        + "_"
        + df_clean["difficulty_clean"].astype(str)
    )

    df_train, df_test = train_test_split(
        df_clean,
        test_size=preprocessing_config["test_size"],
        random_state=preprocessing_config["random_seed"],
        stratify=df_clean["stratify_col"],
    )

    system_message = config["prompt"]["system_message"].strip()

    train_sft = pd.DataFrame({
        "text": df_train.apply(
            format_prompt,
            axis=1,
            system_message=system_message,
        ),
    })

    test_sft = pd.DataFrame({
        "text": df_test.apply(
            format_prompt,
            axis=1,
            system_message=system_message,
        ),
    })

    stats_df = build_preprocessing_report(pipeline_stats)

    train_sft.to_json(
        processed_dir / "train_sft.jsonl",
        orient="records",
        lines=True,
        force_ascii=False,
    )
    test_sft.to_json(
        processed_dir / "test_sft.jsonl",
        orient="records",
        lines=True,
        force_ascii=False,
    )

    df_filtered.to_parquet(processed_dir / "df_filtered.parquet", index=False)
    df_clean.to_parquet(processed_dir / "df_clean.parquet", index=False)
    df_train.to_parquet(processed_dir / "df_train.parquet", index=False)
    df_test.to_parquet(processed_dir / "df_test.parquet", index=False)

    stats_df.to_csv(tables_dir / "preprocessing_report.csv", index=False)
    finance_filtering_table.to_csv(tables_dir / "finance_filtering_table.csv")
    reasoning_quality_table.to_csv(tables_dir / "reasoning_quality_table.csv")

    plot_dataset_size(stats_df, figures_dir / "dataset_filtering_pipeline.png")
    plot_retention_rate(stats_df, figures_dir / "dataset_retention_rate.png")
    plot_finance_heatmap(
        finance_filtering_table,
        figures_dir / "finance_filtering_heatmap.png",
    )
    plot_reasoning_quality(
        reasoning_quality_table,
        figures_dir / "reasoning_quality_distribution.png",
    )
    plot_reasoning_proportion(
        df_clean,
        figures_dir / "final_reasoning_proportion.png",
    )
    plot_difficulty_distribution(
        df_clean,
        figures_dir / "difficulty_distribution.png",
    )

    LOGGER.info("Dataset preparation pipeline completed successfully.")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare a finance instruction dataset for SFT.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the command-line entry point."""
    setup_logging()
    args = parse_args()
    config = load_config(args.config)

    ensure_directories(config)
    set_random_seed(config["preprocessing"]["random_seed"])

    run_pipeline(config)


if __name__ == "__main__":
    main()