"""Unit tests for data processing."""
import numpy as np

from starter import config
from starter.ml.data import process_data


def test_process_data_training_shapes(raw_data, processed):
    X, y, encoder, lb = processed
    assert isinstance(X, np.ndarray)
    assert X.shape[0] == len(raw_data) == y.shape[0]
    n_onehot = sum(len(c) for c in encoder.categories_)
    assert X.shape[1] == len(config.CONTINUOUS_FEATURES) + n_onehot
    assert list(lb.classes_) == ["<=50K", ">50K"]


def test_process_data_inference_without_label(raw_data, processed):
    _, _, encoder, lb = processed
    features = raw_data.drop(columns=[config.LABEL]).head(5)
    X, y, _, _ = process_data(
        features,
        categorical_features=config.CAT_FEATURES,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb,
    )
    assert X.shape[0] == 5
    assert y.size == 0
