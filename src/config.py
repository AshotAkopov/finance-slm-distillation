"""Project configuration for finance dataset preparation."""

from pathlib import Path


# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


# =============================================================================
# Dataset
# =============================================================================

DATASET_NAME = "heladell/Finance_DeepSeek-R1-Distill-dataset"
DATASET_SPLIT = "train"


# =============================================================================
# Train/Test Split
# =============================================================================

TEST_SIZE = 0.10
RANDOM_STATE = 42


# =============================================================================
# Filtering
# =============================================================================

TARGET_LANGUAGE = "EN"

EXCLUDED_TASK_CATEGORIES = (
    "Coding & Debugging",
    "Advice seeking",
    "Editing",
    "Brainstorming",
    "Role playing",
    "Creative writing",
)

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
    # Banking, insurance, actuarial science
    "loan",
    "mortgage",
    "insurance",
    "solvency",
    "scr",
    "actuarial",
    "taeg",
    # Accounting and financial analysis
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

NOISY_REASONING_PATTERNS = (
    "wait",
    "maybe",
    "hmm",
    "perhaps",
    "let me think",
    "actually",
)

NOISY_ANSWER_PATTERNS = (
    "i cannot",
    "i'm unable",
    "as an ai",
    "i do not have access",
    "i don't have access",
    "let me think",
    "wait",
    "actually",
    "maybe",
    "perhaps",
)

REASONING_STRUCTURE_MARKERS = (
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

MIN_ANSWER_WORDS = 20
MAX_ANSWER_WORDS = 900
MIN_UNIQUE_WORD_RATIO = 0.35
MIN_REASONING_WORDS = 60
MAX_REASONING_INSTRUCTION_RATIO = 20


# =============================================================================
# Output filenames
# =============================================================================

TRAIN_SFT_FILENAME = "train_sft.jsonl"
TEST_SFT_FILENAME = "test_sft.jsonl"
DF_FILTERED_FILENAME = "df_filtered.parquet"
DF_CLEAN_FILENAME = "df_clean.parquet"
DF_TRAIN_FILENAME = "df_train.parquet"
DF_TEST_FILENAME = "df_test.parquet"
PREPROCESSING_REPORT_FILENAME = "preprocessing_report.csv"
FINANCE_FILTERING_TABLE_FILENAME = "finance_filtering_table.csv"
REASONING_QUALITY_TABLE_FILENAME = "reasoning_quality_table.csv"
