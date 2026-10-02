"""Script to train the census income model.

Run from the repository root with either of::

    python starter/train_model.py
    python -m starter.train_model

It loads the data, creates a train/test split, fits the one-hot encoder and
label binarizer, trains the model, saves model + encoder + binarizer to
``model/``, prints overall test metrics and writes metrics for every slice of
every categorical feature to ``slice_output.txt``.
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from starter import config  # noqa: E402
from starter.ml.data import process_data  # noqa: E402
from starter.ml.model import (  # noqa: E402
    compute_model_metrics,
    compute_slice_metrics,
    inference,
    save_artifact,
    train_model,
)


def load_data(path=None):
    """Read the census csv (values have a leading space after each comma)."""
    df = pd.read_csv(path or config.DATA_PATH, skipinitialspace=True)
    df.columns = [column.strip() for column in df.columns]
    return df


def format_slice_report(slice_frames, overall):
    """Render slice metrics as plain text."""
    lines = [
        "Model performance on slices of the test data",
        "=" * 60,
        "Overall test set: precision={:.4f} recall={:.4f} fbeta={:.4f}".format(
            *overall
        ),
        "",
    ]
    for frame in slice_frames:
        feature = frame["feature"].iloc[0]
        lines.append(f"Feature: {feature}")
        lines.append("-" * 60)
        for row in frame.itertuples():
            lines.append(
                "{}={} | count={} | precision={:.4f} | recall={:.4f} "
                "| fbeta={:.4f}".format(
                    feature,
                    row.value,
                    row.count,
                    row.precision,
                    row.recall,
                    row.fbeta,
                )
            )
        lines.append("")
    return "\n".join(lines)


def main():
    data = load_data()
    train, test = train_test_split(
        data,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=data[config.LABEL],
    )

    X_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=config.CAT_FEATURES,
        label=config.LABEL,
        training=True,
    )
    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=config.CAT_FEATURES,
        label=config.LABEL,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    model = train_model(X_train, y_train)

    save_artifact(model, config.MODEL_PATH)
    save_artifact(encoder, config.ENCODER_PATH)
    save_artifact(lb, config.LABEL_BINARIZER_PATH)

    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    print(
        f"Test metrics: precision={precision:.4f} recall={recall:.4f} "
        f"fbeta={fbeta:.4f}"
    )

    slice_frames = [
        compute_slice_metrics(
            test, feature, model, encoder, lb, config.CAT_FEATURES,
            label=config.LABEL,
        )
        for feature in config.CAT_FEATURES
    ]
    report = format_slice_report(slice_frames, (precision, recall, fbeta))
    config.SLICE_OUTPUT_PATH.write_text(report + "\n", encoding="utf-8")
    print(f"Wrote slice metrics to {config.SLICE_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
