from pathlib import Path


# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"

PROCESSED_DIR = DATA_DIR / "processed"


# =============================================================================
# Dataset
# =============================================================================

DATASET_NAME = "heladell/Finance_DeepSeek-R1-Distill-dataset"

DATASET_SPLIT = "train"


# =============================================================================
# Train/Test
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
