"""Tests for the training script (run end-to-end on a small sample)."""
from starter import config, train_model


def test_load_data_strips_whitespace():
    df = train_model.load_data()
    assert "marital-status" in df.columns
    assert df["workclass"].str.startswith(" ").sum() == 0
    assert set(df[config.LABEL].unique()) == {"<=50K", ">50K"}


def test_main_saves_artifacts_and_slice_output(tmp_path, monkeypatch, raw_data):
    sample_csv = tmp_path / "sample.csv"
    raw_data.to_csv(sample_csv, index=False)
    monkeypatch.setattr(config, "DATA_PATH", sample_csv)
    monkeypatch.setattr(config, "MODEL_PATH", tmp_path / "m" / "model.pkl")
    monkeypatch.setattr(config, "ENCODER_PATH", tmp_path / "m" / "enc.pkl")
    monkeypatch.setattr(config, "LABEL_BINARIZER_PATH", tmp_path / "m" / "lb.pkl")
    monkeypatch.setattr(config, "SLICE_OUTPUT_PATH", tmp_path / "slice_output.txt")

    train_model.main()

    for name in ("model.pkl", "enc.pkl", "lb.pkl"):
        assert (tmp_path / "m" / name).exists()
    report = (tmp_path / "slice_output.txt").read_text(encoding="utf-8")
    for feature in config.CAT_FEATURES:
        assert f"Feature: {feature}" in report
    assert "education=Bachelors" in report
