"""Unit tests for the ML helpers (type/shape/value-range checks)."""
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from starter import config
from starter.ml.model import (
    compute_model_metrics,
    compute_slice_metrics,
    inference,
    load_artifact,
    save_artifact,
)


def test_train_model_returns_random_forest(small_model):
    assert isinstance(small_model, RandomForestClassifier)


def test_inference_returns_binary_ndarray(processed, small_model):
    X, _, _, _ = processed
    preds = inference(small_model, X)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (X.shape[0],)
    assert set(np.unique(preds)).issubset({0, 1})


def test_compute_model_metrics_known_values():
    y = np.array([1, 1, 0, 0])
    preds = np.array([1, 0, 0, 0])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert isinstance(precision, float)
    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(0.5)
    assert fbeta == pytest.approx(2 / 3)


def test_compute_model_metrics_in_unit_interval(processed, small_model):
    X, y, _, _ = processed
    metrics = compute_model_metrics(y, inference(small_model, X))
    assert len(metrics) == 3
    assert all(0.0 <= metric <= 1.0 for metric in metrics)


def test_save_and_load_roundtrip(tmp_path, processed, small_model):
    X, _, encoder, lb = processed
    for name, artifact in [
        ("model.pkl", small_model),
        ("encoder.pkl", encoder),
        ("lb.pkl", lb),
    ]:
        path = save_artifact(artifact, tmp_path / name)
        assert path.exists()
        assert type(load_artifact(path)) is type(artifact)
    reloaded = load_artifact(tmp_path / "model.pkl")
    assert np.array_equal(
        inference(reloaded, X[:50]), inference(small_model, X[:50])
    )


def test_slice_metrics_one_row_per_unique_value(raw_data, small_model, processed):
    _, _, encoder, lb = processed
    result = compute_slice_metrics(
        raw_data, "education", small_model, encoder, lb, config.CAT_FEATURES
    )
    assert len(result) == raw_data["education"].nunique()
    assert result["count"].sum() == len(raw_data)
    assert {"value", "count", "precision", "recall", "fbeta"}.issubset(
        result.columns
    )


def test_slice_metrics_unknown_feature_raises(raw_data, small_model, processed):
    _, _, encoder, lb = processed
    with pytest.raises(KeyError):
        compute_slice_metrics(
            raw_data, "nope", small_model, encoder, lb, config.CAT_FEATURES
        )
