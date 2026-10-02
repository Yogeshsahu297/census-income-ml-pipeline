"""Shared fixtures: a small, fast model trained on a sample of the data."""
import pandas as pd
import pytest

from starter import config
from starter.ml.data import process_data
from starter.ml.model import train_model


@pytest.fixture(scope="session")
def raw_data():
    df = pd.read_csv(config.DATA_PATH, skipinitialspace=True)
    df.columns = [column.strip() for column in df.columns]
    return df.sample(n=1500, random_state=0).reset_index(drop=True)


@pytest.fixture(scope="session")
def processed(raw_data):
    X, y, encoder, lb = process_data(
        raw_data,
        categorical_features=config.CAT_FEATURES,
        label=config.LABEL,
        training=True,
    )
    return X, y, encoder, lb


@pytest.fixture(scope="session")
def small_model(processed):
    X, y, _, _ = processed
    return train_model(X, y)
