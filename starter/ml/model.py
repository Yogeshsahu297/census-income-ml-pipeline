"""Model training, persistence, inference and evaluation helpers."""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score

from starter.ml.data import process_data


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.ndarray
        Training data.
    y_train : np.ndarray
        Labels.
    Returns
    -------
    model : RandomForestClassifier
        Trained machine learning model.
    """
    model = RandomForestClassifier(
        n_estimators=100,
        min_samples_leaf=5,
        max_depth=20,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.ndarray
        Known labels, binarized.
    preds : np.ndarray
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.ndarray
        Data used for prediction.
    Returns
    -------
    preds : np.ndarray
        Predictions from the model.
    """
    return model.predict(X)


def save_artifact(artifact, path):
    """Persist a model or a fitted encoder/binarizer to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
    return path


def load_artifact(path):
    """Load a model or a fitted encoder/binarizer from ``path``."""
    return joblib.load(Path(path))


# Aliases with more descriptive names for the model itself.
save_model = save_artifact
load_model = load_artifact


def compute_slice_metrics(
    df, feature, model, encoder, lb, categorical_features, label="salary"
):
    """Compute metrics with the value of ``feature`` held fixed.

    For every distinct value of ``feature`` in ``df`` the rows having that
    value are scored with the already trained ``model`` and the precision,
    recall and F1 are returned.

    Inputs
    ------
    df : pd.DataFrame
        Raw (unprocessed) data including the label column.
    feature : str
        Column whose value is held fixed for each slice.
    model, encoder, lb :
        Trained model, fitted OneHotEncoder and fitted LabelBinarizer.
    categorical_features : list[str]
        Categorical columns used by ``process_data``.
    label : str
        Name of the label column.
    Returns
    -------
    pd.DataFrame
        One row per distinct value with columns feature, value, count,
        precision, recall and fbeta.
    """
    if feature not in df.columns:
        raise KeyError(f"Unknown feature: {feature}")

    rows = []
    for value in sorted(df[feature].unique(), key=str):
        subset = df[df[feature] == value]
        X_slice, y_slice, _, _ = process_data(
            subset,
            categorical_features=categorical_features,
            label=label,
            training=False,
            encoder=encoder,
            lb=lb,
        )
        preds = inference(model, X_slice)
        precision, recall, fbeta = compute_model_metrics(y_slice, preds)
        rows.append(
            {
                "feature": feature,
                "value": value,
                "count": len(subset),
                "precision": precision,
                "recall": recall,
                "fbeta": fbeta,
            }
        )
    return pd.DataFrame(rows)
