"""Prepare finance distillation datasets for SFT training.

This script converts the source Hugging Face dataset into three training
artifacts used by the project:

- reasoning_train: examples with reasoning traces, used for reasoning transfer;
- response_train: concise final answers, used for output-style alignment;
- test: final-answer-only evaluation split.

The script is intentionally self-contained because the project may include
other preparation scripts later. All exports are written under the ``data/``
directory.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from src.config import (
    DATASET_NAME,
    DATASET_SPLIT,
    TEST_SIZE,
    RANDOM_STATE,
    RAW_DIR,
    PROCESSED_DIR,
)


# =============================================================================
# Constants
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXCLUDED_TASK_CATEGORIES = (
    "Coding & Debugging",
    "Advice seeking",
    "Editing",
    "Brainstorming",
    "Role playing",
    "Creative writing",
)

TARGET_LANGUAGE = "EN"

FINANCE_KEYWORDS = (
    # Corporate finance
    "ebitda",
    "cash flow",
    "free cash flow",
    "fcf",
    "capex",
    "valuation",
    "revenue",
    "profit",
    "margin",
    "dividend",
    # Financial markets
    "bond",
    "equity",
    "stock",
    "portfolio",
    "asset",
    "liability",
    "yield",
    "volatility",
    "option",
    "derivative",
    "credit",
    "interest rate",
    "inflation",
    "market risk",
    # Banking / insurance
    "loan",
    "mortgage",
    "insurance",
    "solvency",
    "scr",
    "actuarial",
    "taeg",
    # Accounting / financial analysis
    "balance sheet",
    "income statement",
    "financial statement",
    "return on equity",
    "roe",
    "roa",
    # Macroeconomics
    "gdp",
    "federal reserve",
    "fed",
    "central bank",
    "monetary policy",
)

NOISE_WORDS = (
    "wait",
    "maybe",
    "hmm",
    "perhaps",
    "let me think",
    "actually",
)

STRUCTURE_MARKERS = (
    "first",
    "then",
    "finally",
    "therefore",
    "step",
    "1.",
    "2.",
    "so,",
)

FORMULA_SYMBOLS = ("=", "+", "-", "/", "*", "%")

CLEAN_COLUMNS = (
    "instruction",
    "response",
    "reasoning",
    "final_answer",
    "task_category",
    "difficulty",
    "input_quality",
    "has_think",
    "finance_related",
    "good_reasoning",
)

SYSTEM_PROMPT_BASE = """You are an expert quantitative finance, actuarial science, and mathematical reasoning assistant.

Provide accurate, rigorous, and concise answers.

Use structured step-by-step reasoning only when necessary for solving complex problems.
For quantitative tasks, clearly explain formulas, assumptions, and calculations.

Avoid unnecessary verbosity, repetition, or low-value reasoning.
Focus on clarity, precision, and efficiency."""

REASONING_SYSTEM_INSTRUCTION = (
    "This example may include reasoning traces to help the model learn how to "
    "reason, but the final answer must remain concise."
)

RESPONSE_SYSTEM_INSTRUCTION = (
    "Do not expose internal reasoning. Output only the final concise answer."
)


# =============================================================================
# Logging
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# =============================================================================
# Text processing helpers
# =============================================================================

def extract_think(response: Any) -> str:
    """Extract the content between ``<think>`` and ``</think>``.

    Args:
        response: Raw assistant response.

    Returns:
        Extracted reasoning text. Returns an empty string when no reasoning
        block is found.
    """
    match = re.search(
        r"<think>(.*?)</think>",
        str(response),
        flags=re.DOTALL,
    )
    return match.group(1).strip() if match else ""


def remove_think(response: Any) -> str:
    """Remove ``<think>...</think>`` blocks from an assistant response.

    Args:
        response: Raw assistant response.

    Returns:
        Final answer with reasoning traces removed.
    """
    return re.sub(
        r"<think>.*?</think>",
        "",
        str(response),
        flags=re.DOTALL,
    ).strip()


def contains_think(response: Any) -> bool:
    """Check whether a response contains a reasoning block.

    Args:
        response: Raw assistant response.

    Returns:
        True if a ``<think>...</think>`` block is present.
    """
    return bool(
        re.search(
            r"<think>.*?</think>",
            str(response),
            flags=re.DOTALL,
        )
    )


# =============================================================================
# Filtering and quality checks
# =============================================================================

def is_finance_related(row: pd.Series) -> bool:
    """Detect whether an example is related to finance.

    This rule-based filter searches for finance-related keywords across the
    instruction, response and available metadata fields.

    Args:
        row: Dataset row.

    Returns:
        True if at least one finance keyword is found.
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


def clean_difficulty(value: Any) -> str:
    """Reduce sparse difficulty levels to more stable classes.

    Args:
        value: Original difficulty label.

    Returns:
        Cleaned difficulty label.
    """
    if value == "very easy":
        return "easy"
    if value == "very hard":
        return "hard"
    return str(value)


def is_good_reasoning(row: pd.Series) -> bool:
    """Evaluate whether a reasoning trace is suitable for training.

    The score is heuristic and designed for dataset curation. It rejects
    reasoning traces that are missing, too short, too noisy, disproportionate
    to the instruction, or not followed by a meaningful final answer.

    Args:
        row: Dataset row containing ``instruction`` and ``response``.

    Returns:
        True if the reasoning trace is considered exploitable.
    """
    instruction = str(row["instruction"])
    response = str(row["response"])

    think = extract_think(response)
    final_answer = remove_think(response)

    instruction_words = len(instruction.split())
    think_words = len(think.split())
    final_answer_words = len(final_answer.split())

    think_instruction_ratio = think_words / max(instruction_words, 1)
    noise_score = sum(think.lower().count(word) for word in NOISE_WORDS)

    has_structure = any(
        marker in think.lower()
        for marker in STRUCTURE_MARKERS
    )
    has_formula = any(symbol in think for symbol in FORMULA_SYMBOLS)
    has_final_answer = final_answer_words > 20

    if think_words == 0:
        return False

    if think_words < 60:
        return False

    if not has_final_answer:
        return False

    if noise_score > 12:
        return False

    if think_instruction_ratio > 20:
        return False

    if think_instruction_ratio <= 4:
        return noise_score <= 8

    if 4 < think_instruction_ratio <= 10:
        return noise_score <= 5 and (has_structure or has_formula)

    if 10 < think_instruction_ratio <= 20:
        return noise_score <= 2 and has_structure and has_formula

    return False


# =============================================================================
# Prompt formatting
# =============================================================================

def format_reasoning_example(row: pd.Series) -> str:
    """Format an example for the reasoning-transfer training phase.

    Args:
        row: Dataset row.

    Returns:
        Chat-formatted text containing the full response, including possible
        reasoning traces.
    """
    return f"""<|im_start|>system
{SYSTEM_PROMPT_BASE}

{REASONING_SYSTEM_INSTRUCTION}
<|im_end|>

<|im_start|>user
{row["instruction"]}
<|im_end|>

<|im_start|>assistant
{row["response"]}
<|im_end|>
"""


def format_response_example(row: pd.Series) -> str:
    """Format an example for final-answer-only SFT.

    Args:
        row: Dataset row.

    Returns:
        Chat-formatted text containing only the concise final answer.
    """
    return f"""<|im_start|>system
{SYSTEM_PROMPT_BASE}

{RESPONSE_SYSTEM_INSTRUCTION}
<|im_end|>

<|im_start|>user
{row["instruction"]}
<|im_end|>

<|im_start|>assistant
{row["final_answer"]}
<|im_end|>
"""


# =============================================================================
# Dataset preparation pipeline
# =============================================================================

def ensure_directories() -> None:
    """Create output directories if they do not already exist."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_source_dataset(dataset_name: str, split: str) -> pd.DataFrame:
    """Load the source Hugging Face dataset.

    Args:
        dataset_name: Hugging Face dataset identifier.
        split: Split name to load.

    Returns:
        Dataset as a Pandas DataFrame.
    """
    logger.info("Loading dataset '%s' split '%s'.", dataset_name, split)
    dataset = load_dataset(dataset_name, split=split)
    df = dataset.to_pandas()

    logger.info("Loaded source dataset with shape %s.", df.shape)
    return df


def clean_and_filter_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the core cleaning and filtering logic.

    Args:
        df: Raw source DataFrame.

    Returns:
        Filtered DataFrame with reasoning and quality columns.
    """
    logger.info("Starting cleaning and filtering.")

    df = df.copy()
    initial_rows = len(df)

    df = df[df["response"].notna()].copy()
    logger.info("Removed missing responses: %d -> %d.", initial_rows, len(df))

    df = df[~df["task_category"].isin(EXCLUDED_TASK_CATEGORIES)].copy()
    logger.info("Removed excluded task categories. Remaining rows: %d.", len(df))

    df = df[df["language"] == TARGET_LANGUAGE].copy()
    logger.info("Filtered language '%s'. Remaining rows: %d.", TARGET_LANGUAGE, len(df))

    df["finance_related"] = df.apply(is_finance_related, axis=1)
    df = df[
        (df["task_category"] == "Math")
        | (df["finance_related"])
    ].copy()
    logger.info("Filtered finance/math examples. Remaining rows: %d.", len(df))

    df["has_think"] = df["response"].apply(contains_think)
    df["reasoning"] = df["response"].apply(extract_think)
    df["final_answer"] = df["response"].apply(remove_think)
    df["good_reasoning"] = df.apply(is_good_reasoning, axis=1)

    logger.info(
        "Reasoning traces detected: %d / %d.",
        int(df["has_think"].sum()),
        len(df),
    )
    logger.info(
        "Good reasoning examples: %d / %d.",
        int(df["good_reasoning"].sum()),
        len(df),
    )

    return df


def build_clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only columns required for analysis and dataset construction.

    Args:
        df: Filtered source DataFrame.

    Returns:
        Clean DataFrame containing selected columns.
    """
    missing_columns = set(CLEAN_COLUMNS) - set(df.columns)
    if missing_columns:
        raise ValueError(
            "Cannot build clean dataset. Missing columns: "
            f"{sorted(missing_columns)}"
        )

    return df.loc[:, CLEAN_COLUMNS].copy()


def split_dataset(df_clean: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the clean dataset into train and test sets.

    The stratification column combines reasoning availability and cleaned
    difficulty level to preserve a stable distribution across splits.

    Args:
        df_clean: Clean DataFrame.

    Returns:
        Tuple containing the train and test DataFrames.
    """
    df_clean = df_clean.copy()
    df_clean["difficulty_clean"] = df_clean["difficulty"].apply(clean_difficulty)

    df_clean["stratify_col"] = (
        df_clean["has_think"].astype(str)
        + "_"
        + df_clean["difficulty_clean"].astype(str)
    )

    rare_classes = df_clean["stratify_col"].value_counts()
    rare_classes = rare_classes[rare_classes < 2]

    if not rare_classes.empty:
        logger.warning(
            "Some stratification classes contain fewer than two examples. "
            "Falling back to non-stratified split. Rare classes: %s",
            rare_classes.to_dict(),
        )
        stratify = None
    else:
        stratify = df_clean["stratify_col"]

    df_train, df_test = train_test_split(
        df_clean,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )

    logger.info("Train shape: %s.", df_train.shape)
    logger.info("Test shape: %s.", df_test.shape)

    return df_train.copy(), df_test.copy()


def build_training_artifacts(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create reasoning, response and test datasets.

    Args:
        df_train: Training split.
        df_test: Test split.

    Returns:
        Tuple containing reasoning_train, response_train and test DataFrames.
    """
    reasoning_train = df_train.copy()
    reasoning_train["text"] = reasoning_train.apply(
        format_reasoning_example,
        axis=1,
    )

    response_train = df_train.copy()
    response_train["text"] = response_train.apply(
        format_response_example,
        axis=1,
    )

    test = df_test.copy()
    test["text"] = test.apply(format_response_example, axis=1)

    logger.info("Built reasoning_train with shape %s.", reasoning_train.shape)
    logger.info("Built response_train with shape %s.", response_train.shape)
    logger.info("Built test with shape %s.", test.shape)

    return reasoning_train, response_train, test


def export_jsonl(df: pd.DataFrame, output_path: Path) -> None:
    """Export the ``text`` column to JSONL.

    Args:
        df: DataFrame containing a ``text`` column.
        output_path: Destination path.
    """
    df[["text"]].to_json(
        output_path,
        orient="records",
        lines=True,
        force_ascii=False,
    )
    logger.info("Exported JSONL: %s.", output_path)


def export_parquet(df: pd.DataFrame, output_path: Path) -> None:
    """Export a DataFrame to Parquet.

    Args:
        df: DataFrame to export.
        output_path: Destination path.
    """
    df.to_parquet(output_path, index=False)
    logger.info("Exported Parquet: %s.", output_path)


def export_artifacts(
    df_raw: pd.DataFrame,
    df_clean: pd.DataFrame,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    reasoning_train: pd.DataFrame,
    response_train: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Export all raw, clean, training and test artifacts.

    Args:
        df_raw: Filtered raw DataFrame with all computed columns.
        df_clean: Clean dataset.
        df_train: Training split.
        df_test: Test split.
        reasoning_train: Reasoning-transfer dataset.
        response_train: Final-answer training dataset.
        test: Final evaluation dataset.
    """
    export_parquet(df_raw, RAW_DIR / "df_raw.parquet")
    export_parquet(df_clean, PROCESSED_DIR / "df_clean.parquet")
    export_parquet(df_train, PROCESSED_DIR / "df_train.parquet")
    export_parquet(df_test, PROCESSED_DIR / "df_test.parquet")
    export_parquet(reasoning_train, PROCESSED_DIR / "reasoning_train.parquet")
    export_parquet(response_train, PROCESSED_DIR / "response_train.parquet")
    export_parquet(test, PROCESSED_DIR / "test.parquet")

    export_jsonl(reasoning_train, PROCESSED_DIR / "reasoning_train.jsonl")
    export_jsonl(response_train, PROCESSED_DIR / "response_train.jsonl")
    export_jsonl(test, PROCESSED_DIR / "test.jsonl")


def run_pipeline(dataset_name: str = DATASET_NAME, split: str = DATASET_SPLIT) -> None:
    """Run the full dataset preparation pipeline.

    Args:
        dataset_name: Hugging Face dataset identifier.
        split: Dataset split to load.
    """
    ensure_directories()

    df_source = load_source_dataset(dataset_name=dataset_name, split=split)
    df_raw = clean_and_filter_dataset(df_source)
    df_clean = build_clean_dataset(df_raw)

    df_train, df_test = split_dataset(df_clean)
    reasoning_train, response_train, test = build_training_artifacts(
        df_train=df_train,
        df_test=df_test,
    )

    export_artifacts(
        df_raw=df_raw,
        df_clean=df_clean,
        df_train=df_train,
        df_test=df_test,
        reasoning_train=reasoning_train,
        response_train=response_train,
        test=test,
    )

    logger.info("Dataset preparation completed successfully.")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Prepare finance distillation datasets for SFT training.",
    )
    parser.add_argument(
        "--dataset-name",
        default=DATASET_NAME,
        help="Hugging Face dataset identifier.",
    )
    parser.add_argument(
        "--split",
        default=DATASET_SPLIT,
        help="Dataset split to load.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    run_pipeline(dataset_name=args.dataset_name, split=args.split)


if __name__ == "__main__":
    main()
