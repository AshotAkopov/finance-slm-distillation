"""Prepare a curated finance instruction dataset for Qwen SFT.

This module converts the raw Finance DeepSeek-R1 distilled dataset into a
cleaned conversational supervised fine-tuning corpus. It keeps validated
reasoning traces in the training set, builds a test set without CoT leakage,
and exports JSONL, Parquet, CSV, and figure artifacts.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

try:
    from src.config import (
        DATASET_NAME,
        DATASET_SPLIT,
        DF_CLEAN_FILENAME,
        DF_FILTERED_FILENAME,
        DF_TEST_FILENAME,
        DF_TRAIN_FILENAME,
        EXCLUDED_TASK_CATEGORIES,
        FIGURES_DIR,
        FINANCE_FILTERING_TABLE_FILENAME,
        FINANCE_KEYWORDS,
        FORMULA_SYMBOLS,
        MAX_ANSWER_WORDS,
        MAX_REASONING_INSTRUCTION_RATIO,
        MIN_ANSWER_WORDS,
        MIN_REASONING_WORDS,
        MIN_UNIQUE_WORD_RATIO,
        NOISY_ANSWER_PATTERNS,
        NOISY_REASONING_PATTERNS,
        PREPROCESSING_REPORT_FILENAME,
        PROCESSED_DIR,
        RANDOM_STATE,
        REASONING_QUALITY_TABLE_FILENAME,
        REASONING_STRUCTURE_MARKERS,
        TARGET_LANGUAGE,
        TEST_SIZE,
        TEST_SFT_FILENAME,
        TRAIN_SFT_FILENAME,
    )
except ModuleNotFoundError:
    from config import (  # type: ignore[no-redef]
        DATASET_NAME,
        DATASET_SPLIT,
        DF_CLEAN_FILENAME,
        DF_FILTERED_FILENAME,
        DF_TEST_FILENAME,
        DF_TRAIN_FILENAME,
        EXCLUDED_TASK_CATEGORIES,
        FIGURES_DIR,
        FINANCE_FILTERING_TABLE_FILENAME,
        FINANCE_KEYWORDS,
        FORMULA_SYMBOLS,
        MAX_ANSWER_WORDS,
        MAX_REASONING_INSTRUCTION_RATIO,
        MIN_ANSWER_WORDS,
        MIN_REASONING_WORDS,
        MIN_UNIQUE_WORD_RATIO,
        NOISY_ANSWER_PATTERNS,
        NOISY_REASONING_PATTERNS,
        PREPROCESSING_REPORT_FILENAME,
        PROCESSED_DIR,
        RANDOM_STATE,
        REASONING_QUALITY_TABLE_FILENAME,
        REASONING_STRUCTURE_MARKERS,
        TARGET_LANGUAGE,
        TEST_SIZE,
        TEST_SFT_FILENAME,
        TRAIN_SFT_FILENAME,
    )

LOGGER = logging.getLogger(__name__)

THINK_PATTERN = re.compile(r"<think>(.*?)</think>", flags=re.DOTALL)

SYSTEM_PROMPT = """You are an expert assistant in quantitative finance, \
actuarial science, and financial risk analysis.

Provide accurate, concise, and well-structured answers.

When solving quantitative problems, clearly state assumptions, formulas, \
and calculations when relevant."""

CLEAN_COLUMNS = [
    "instruction",
    "response",
    "final_answer",
    "think",
    "task_category",
    "difficulty",
    "difficulty_clean",
    "input_quality",
    "has_think",
    "finance_related",
    "good_reasoning",
    "good_answer",
    "sample_type",
]

SFT_EXPORT_COLUMNS = [
    "text",
    "instruction",
    "final_answer",
    "think",
    "sample_type",
    "difficulty_clean",
]


def configure_logging() -> None:
    """Configure runtime logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def ensure_output_directories() -> None:
    """Create output directories if they do not already exist."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_source_dataset() -> pd.DataFrame:
    """Load the Hugging Face dataset and convert it to a DataFrame.

    Returns:
        Raw dataset as a Pandas DataFrame.
    """
    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    return dataset.to_pandas()


def add_pipeline_step(
    pipeline_stats: list[dict[str, Any]],
    step: str,
    df: pd.DataFrame,
) -> None:
    """Append a preprocessing step summary to the pipeline report.

    Args:
        pipeline_stats: Mutable list storing step-level statistics.
        step: Human-readable preprocessing step name.
        df: Current DataFrame after the preprocessing step.
    """
    pipeline_stats.append({"step": step, "samples": len(df)})
    LOGGER.info("%s: %s samples", step, len(df))


def normalize_text(value: Any) -> str:
    """Convert a value to a safe lowercase text string.

    Args:
        value: Any scalar value.

    Returns:
        Normalized text.
    """
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def has_think(response: Any) -> bool:
    """Check whether a response contains a reasoning trace.

    Args:
        response: Raw model response.

    Returns:
        True if a `<think>...</think>` block is present.
    """
    if response is None or pd.isna(response):
        return False
    return bool(THINK_PATTERN.search(str(response)))


def extract_think(response: Any) -> str:
    """Extract the first `<think>` block from a response.

    Args:
        response: Raw model response.

    Returns:
        Extracted reasoning trace, or an empty string if absent.
    """
    match = THINK_PATTERN.search(str(response))
    if match is None:
        return ""
    return match.group(1).strip()


def remove_think(response: Any) -> str:
    """Remove all `<think>` blocks from a response.

    Args:
        response: Raw model response.

    Returns:
        Final answer without explicit reasoning tags.
    """
    return THINK_PATTERN.sub("", str(response)).strip()


def clean_difficulty(value: Any) -> str:
    """Normalize difficulty labels.

    Args:
        value: Original difficulty value.

    Returns:
        Normalized difficulty label.
    """
    if pd.isna(value):
        return "unknown"

    difficulty = str(value).strip().lower()

    if difficulty == "very easy":
        return "easy"
    if difficulty == "very hard":
        return "hard"
    if difficulty in {"easy", "medium", "hard"}:
        return difficulty

    return "unknown"


def is_finance_related(row: pd.Series) -> bool:
    """Detect whether a sample is finance-related using keywords.

    Args:
        row: Dataset row.

    Returns:
        True if the row contains at least one finance keyword.
    """
    text = " ".join(
        [
            str(row.get("instruction", "")),
            str(row.get("response", "")),
            str(row.get("intent", "")),
            str(row.get("knowledge", "")),
        ]
    ).lower()

    return any(keyword in text for keyword in FINANCE_KEYWORDS)


def is_good_reasoning(row: pd.Series) -> bool:
    """Evaluate whether a reasoning trace is suitable for training.

    Args:
        row: Dataset row.

    Returns:
        True if the reasoning trace passes quality heuristics.
    """
    instruction = str(row.get("instruction", ""))
    think = extract_think(row.get("response", ""))
    final_answer = remove_think(row.get("response", ""))

    instruction_words = len(instruction.split())
    think_words = len(think.split())
    final_answer_words = len(final_answer.split())
    think_instruction_ratio = think_words / max(instruction_words, 1)

    noise_score = sum(
        think.lower().count(pattern)
        for pattern in NOISY_REASONING_PATTERNS
    )
    has_structure = any(
        marker in think.lower()
        for marker in REASONING_STRUCTURE_MARKERS
    )
    has_formula = any(symbol in think for symbol in FORMULA_SYMBOLS)
    has_final_answer = final_answer_words > MIN_ANSWER_WORDS

    if think_words == 0:
        return False
    if think_words < MIN_REASONING_WORDS:
        return False
    if not has_final_answer:
        return False
    if noise_score > 12:
        return False
    if think_instruction_ratio > MAX_REASONING_INSTRUCTION_RATIO:
        return False
    if think_instruction_ratio <= 4:
        return noise_score <= 8
    if 4 < think_instruction_ratio <= 10:
        return noise_score <= 5 and (has_structure or has_formula)
    if 10 < think_instruction_ratio <= 20:
        return noise_score <= 2 and has_structure and has_formula

    return False


def is_good_answer(row: pd.Series) -> bool:
    """Estimate whether the final answer is usable for training.

    Args:
        row: Dataset row.

    Returns:
        True if the answer passes quality heuristics.
    """
    instruction = str(row.get("instruction", ""))
    answer = str(row.get("final_answer", ""))
    answer_words = answer.split()
    answer_len = len(answer_words)

    if answer_len < MIN_ANSWER_WORDS:
        return False
    if answer_len > MAX_ANSWER_WORDS:
        return False

    unique_ratio = len({word.lower() for word in answer_words}) / max(
        answer_len,
        1,
    )
    if unique_ratio < MIN_UNIQUE_WORD_RATIO:
        return False

    noisy_count = sum(
        pattern in answer.lower()
        for pattern in NOISY_ANSWER_PATTERNS
    )
    if noisy_count >= 2:
        return False

    instruction_words = {
        word.lower()
        for word in instruction.split()
        if len(word) > 4
    }
    answer_words_set = {
        word.lower()
        for word in answer.split()
        if len(word) > 4
    }
    overlap = len(instruction_words & answer_words_set)

    return not (overlap == 0 and len(instruction_words) > 0)


def decide_sample_type(row: pd.Series) -> str:
    """Decide how a sample should be used.

    Args:
        row: Dataset row with answer and reasoning quality flags.

    Returns:
        One of `drop`, `cot`, or `answer_only`.
    """
    if not bool(row.get("good_answer", False)):
        return "drop"
    if bool(row.get("has_think", False)) and bool(
        row.get("good_reasoning", False),
    ):
        return "cot"
    return "answer_only"


def preprocess_dataset(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run all filtering and labeling steps.

    Args:
        df: Raw dataset DataFrame.

    Returns:
        Tuple containing cleaned DataFrame, preprocessing report,
        finance filtering table, and reasoning quality table.
    """
    pipeline_stats: list[dict[str, Any]] = []
    add_pipeline_step(pipeline_stats, "Initial dataset", df)

    df = df.drop_duplicates(subset=["instruction", "response"]).copy()
    add_pipeline_step(pipeline_stats, "Deduplication", df)

    df = df[df["response"].notna()].copy()
    add_pipeline_step(pipeline_stats, "Remove missing responses", df)

    df = df[~df["task_category"].isin(EXCLUDED_TASK_CATEGORIES)].copy()
    add_pipeline_step(pipeline_stats, "Relevant category filtering", df)

    df = df[df["language"] == TARGET_LANGUAGE].copy()
    add_pipeline_step(pipeline_stats, "English-only filtering", df)

    df["finance_related"] = df.apply(is_finance_related, axis=1)
    finance_filtering_table = pd.crosstab(
        df["task_category"],
        df["finance_related"],
        margins=True,
    )

    df = df[
        (df["task_category"] == "Math") | df["finance_related"]
    ].copy()
    add_pipeline_step(pipeline_stats, "Finance filtering", df)

    df["has_think"] = df["response"].apply(has_think)
    df["good_reasoning"] = df.apply(is_good_reasoning, axis=1)

    reasoning_quality_table = pd.crosstab(
        df["has_think"],
        df["good_reasoning"],
        margins=True,
    )

    df["final_answer"] = df["response"].apply(remove_think)
    df["think"] = df["response"].apply(extract_think)
    df["good_answer"] = df.apply(is_good_answer, axis=1)
    df["sample_type"] = df.apply(decide_sample_type, axis=1)

    df = df[df["sample_type"] != "drop"].copy()
    df["difficulty_clean"] = df["difficulty"].apply(clean_difficulty)

    add_pipeline_step(
        pipeline_stats,
        "Answer and reasoning quality filtering",
        df,
    )

    stats_df = build_preprocessing_report(pipeline_stats)
    df_clean = df[CLEAN_COLUMNS].copy()

    return df_clean, stats_df, finance_filtering_table, reasoning_quality_table


def build_preprocessing_report(
    pipeline_stats: list[dict[str, Any]],
) -> pd.DataFrame:
    """Build a preprocessing report from pipeline statistics.

    Args:
        pipeline_stats: Step-level sample counts.

    Returns:
        DataFrame with retention and removal metrics.
    """
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


def split_train_test(df_clean: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the dataset into train and test sets.

    All validated CoT samples are placed in the training set. The test set
    contains only answer-only samples and preserves difficulty distribution
    when stratification is feasible.

    Args:
        df_clean: Cleaned dataset.

    Returns:
        Train and test DataFrames.

    Raises:
        ValueError: If the dataset is too small or lacks enough answer-only
            samples for the requested test size.
    """
    df_cot = df_clean[df_clean["sample_type"] == "cot"].copy()
    df_answer_only = df_clean[
        df_clean["sample_type"] == "answer_only"
    ].copy()

    df_cot["difficulty_clean"] = (
        df_cot["difficulty_clean"].fillna("unknown").astype(str)
    )
    df_answer_only["difficulty_clean"] = (
        df_answer_only["difficulty_clean"].fillna("unknown").astype(str)
    )

    total_samples = len(df_clean)
    target_test_size = round(total_samples * TEST_SIZE)

    if target_test_size <= 0:
        raise ValueError("Dataset is too small to create a train/test split.")

    if len(df_answer_only) < target_test_size:
        raise ValueError(
            "Not enough answer-only samples to build a test set without CoT. "
            f"Required: {target_test_size}, "
            f"available: {len(df_answer_only)}."
        )

    difficulty_counts = df_answer_only["difficulty_clean"].value_counts()
    can_stratify = (
        difficulty_counts.min() >= 2
        and target_test_size >= df_answer_only["difficulty_clean"].nunique()
    )

    if can_stratify:
        _, df_test = train_test_split(
            df_answer_only,
            test_size=target_test_size,
            random_state=RANDOM_STATE,
            stratify=df_answer_only["difficulty_clean"],
        )
    else:
        df_test = df_answer_only.sample(
            n=target_test_size,
            random_state=RANDOM_STATE,
        )

    remaining_answer_only = df_answer_only.drop(df_test.index)
    df_train = pd.concat([df_cot, remaining_answer_only], axis=0)

    df_train = df_train.sample(
        frac=1,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)
    df_test = df_test.sample(
        frac=1,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    log_split_diagnostics(df_clean, df_train, df_test)
    return df_train, df_test


def log_split_diagnostics(
    df_clean: pd.DataFrame,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
) -> None:
    """Log train/test composition diagnostics.

    Args:
        df_clean: Full cleaned dataset.
        df_train: Training split.
        df_test: Test split.
    """
    total_samples = len(df_clean)
    LOGGER.info("Total samples: %s", total_samples)
    LOGGER.info(
        "Train samples: %s (%.1f%%)",
        len(df_train),
        100 * len(df_train) / total_samples,
    )
    LOGGER.info(
        "Test samples: %s (%.1f%%)",
        len(df_test),
        100 * len(df_test) / total_samples,
    )
    LOGGER.info("Train sample types:\n%s", df_train["sample_type"].value_counts())
    LOGGER.info("Test sample types:\n%s", df_test["sample_type"].value_counts())
    LOGGER.info(
        "Train difficulty distribution:\n%s",
        df_train["difficulty_clean"].value_counts(normalize=True).round(3),
    )
    LOGGER.info(
        "Test difficulty distribution:\n%s",
        df_test["difficulty_clean"].value_counts(normalize=True).round(3),
    )


def build_completion(row: pd.Series) -> str:
    """Build the assistant completion for SFT.

    Args:
        row: Dataset row.

    Returns:
        Assistant completion, with CoT only for validated reasoning samples.
    """
    answer = str(row.get("final_answer", "")).strip()

    if row.get("sample_type") == "cot":
        think = str(row.get("think", "")).strip()
        return f"""<think>
{think}
</think>

{answer}"""

    return answer


def format_qwen_chat(row: pd.Series) -> str:
    """Format one row using Qwen chat tokens.

    Args:
        row: Dataset row.

    Returns:
        Qwen-compatible SFT text.
    """
    completion = build_completion(row)

    return f"""<|im_start|>system
{SYSTEM_PROMPT}
<|im_end|>

<|im_start|>user
{row["instruction"]}
<|im_end|>

<|im_start|>assistant
{completion}
<|im_end|>
"""


def build_sft_exports(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create JSONL-ready SFT DataFrames.

    Args:
        df_train: Training split.
        df_test: Test split.

    Returns:
        Train and test SFT DataFrames.
    """
    df_train = df_train.copy()
    df_test = df_test.copy()

    df_train["text"] = df_train.apply(format_qwen_chat, axis=1)
    df_test["text"] = df_test.apply(format_qwen_chat, axis=1)

    return df_train[SFT_EXPORT_COLUMNS].copy(), df_test[SFT_EXPORT_COLUMNS].copy()


def save_barh_chart(
    series: pd.Series,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
) -> None:
    """Save a horizontal bar chart.

    Args:
        series: Series to plot.
        title: Chart title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        output_path: Destination path.
    """
    ax = series.sort_values(ascending=True).plot(
        kind="barh",
        figsize=(8, 5),
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.bar_label(ax.containers[0])

    plt.tight_layout()
    fig = ax.get_figure()
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def save_diagnostics(
    df_clean: pd.DataFrame,
    stats_df: pd.DataFrame,
    finance_filtering_table: pd.DataFrame,
    reasoning_quality_table: pd.DataFrame,
) -> None:
    """Save CSV reports and visual diagnostics.

    Args:
        df_clean: Cleaned dataset.
        stats_df: Preprocessing report.
        finance_filtering_table: Finance filtering crosstab.
        reasoning_quality_table: Reasoning quality crosstab.
    """
    stats_df.to_csv(PROCESSED_DIR / PREPROCESSING_REPORT_FILENAME, index=False)
    finance_filtering_table.to_csv(
        PROCESSED_DIR / FINANCE_FILTERING_TABLE_FILENAME,
    )
    reasoning_quality_table.to_csv(
        PROCESSED_DIR / REASONING_QUALITY_TABLE_FILENAME,
    )

    save_finance_distribution_chart(finance_filtering_table)
    save_reasoning_distribution_chart(df_clean)
    save_retention_chart(stats_df)
    save_barh_chart(
        series=df_clean["difficulty_clean"].value_counts(),
        title="Final Difficulty Distribution",
        xlabel="Number of Samples",
        ylabel="Difficulty Level",
        output_path=FIGURES_DIR / "difficulty_distribution.png",
    )


def save_finance_distribution_chart(finance_filtering_table: pd.DataFrame) -> None:
    """Save a finance relevance distribution chart.

    Args:
        finance_filtering_table: Crosstab including margins.
    """
    finance_counts = finance_filtering_table.iloc[:-1, :-1]
    if True in finance_counts.columns:
        finance_counts = finance_counts.sort_values(by=True, ascending=True)

    ax = finance_counts.plot(kind="barh", stacked=True, figsize=(10, 6))
    ax.set_title(
        "Finance Relevance by Task Category",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("Task Category")
    ax.legend(title="Finance Related")

    plt.tight_layout()
    fig = ax.get_figure()
    fig.savefig(
        FIGURES_DIR / "finance_filtering_distribution.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig)


def save_reasoning_distribution_chart(df_clean: pd.DataFrame) -> None:
    """Save the final reasoning sample composition chart.

    Args:
        df_clean: Cleaned dataset.
    """
    reasoning_counts = df_clean["sample_type"].value_counts().rename(
        {
            "cot": "Reasoning Samples",
            "answer_only": "Answer-only Samples",
        },
    )

    ax = reasoning_counts.plot(kind="bar", figsize=(7, 5))
    ax.set_title(
        "Final Training Sample Composition",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("")
    ax.set_ylabel("Number of Samples")
    ax.bar_label(ax.containers[0])
    plt.xticks(rotation=0)

    plt.tight_layout()
    fig = ax.get_figure()
    fig.savefig(
        FIGURES_DIR / "final_reasoning_distribution.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig)


def save_retention_chart(stats_df: pd.DataFrame) -> None:
    """Save the dataset retention rate chart.

    Args:
        stats_df: Preprocessing report.
    """
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
    fig = ax.get_figure()
    fig.savefig(
        FIGURES_DIR / "dataset_retention_rate.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig)


def export_datasets(
    df_filtered: pd.DataFrame,
    df_clean: pd.DataFrame,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    train_sft: pd.DataFrame,
    test_sft: pd.DataFrame,
) -> None:
    """Export processed datasets to disk.

    Args:
        df_filtered: Dataset after final filtering.
        df_clean: Cleaned dataset before split.
        df_train: Training split.
        df_test: Test split.
        train_sft: JSONL-ready training dataset.
        test_sft: JSONL-ready test dataset.
    """
    train_sft.to_json(
        PROCESSED_DIR / TRAIN_SFT_FILENAME,
        orient="records",
        lines=True,
        force_ascii=False,
    )
    test_sft.to_json(
        PROCESSED_DIR / TEST_SFT_FILENAME,
        orient="records",
        lines=True,
        force_ascii=False,
    )

    df_filtered.to_parquet(PROCESSED_DIR / DF_FILTERED_FILENAME, index=False)
    df_clean.to_parquet(PROCESSED_DIR / DF_CLEAN_FILENAME, index=False)
    df_train.to_parquet(PROCESSED_DIR / DF_TRAIN_FILENAME, index=False)
    df_test.to_parquet(PROCESSED_DIR / DF_TEST_FILENAME, index=False)


def run_pipeline() -> None:
    """Run the full dataset preparation pipeline."""
    configure_logging()
    ensure_output_directories()

    LOGGER.info("Loading source dataset: %s", DATASET_NAME)
    raw_df = load_source_dataset()

    df_clean, stats_df, finance_table, reasoning_table = preprocess_dataset(raw_df)
    df_train, df_test = split_train_test(df_clean)
    train_sft, test_sft = build_sft_exports(df_train, df_test)

    export_datasets(
        df_filtered=df_clean,
        df_clean=df_clean,
        df_train=df_train,
        df_test=df_test,
        train_sft=train_sft,
        test_sft=test_sft,
    )
    save_diagnostics(df_clean, stats_df, finance_table, reasoning_table)

    LOGGER.info("Dataset preparation completed successfully.")
    LOGGER.info("Processed files saved in: %s", PROCESSED_DIR)
    LOGGER.info("Figures saved in: %s", FIGURES_DIR)


if __name__ == "__main__":
    run_pipeline()
