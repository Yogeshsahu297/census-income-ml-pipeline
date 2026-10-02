"""Shared constants for the training pipeline and the API."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "census.csv"
MODEL_DIR = ROOT_DIR / "model"
MODEL_PATH = MODEL_DIR / "model.pkl"
ENCODER_PATH = MODEL_DIR / "encoder.pkl"
LABEL_BINARIZER_PATH = MODEL_DIR / "lb.pkl"
SLICE_OUTPUT_PATH = ROOT_DIR / "slice_output.txt"

LABEL = "salary"
RANDOM_STATE = 42
TEST_SIZE = 0.20

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Continuous columns, in the order they appear in census.csv. process_data
# concatenates them (in this order) in front of the one-hot columns, so the
# API must build its input frame with exactly this ordering.
CONTINUOUS_FEATURES = [
    "age",
    "fnlgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]

FEATURE_ORDER = [
    "age",
    "workclass",
    "fnlgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
]
