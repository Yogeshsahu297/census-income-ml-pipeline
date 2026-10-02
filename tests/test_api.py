"""Tests for the FastAPI app (GET and both possible POST outcomes)."""
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

HIGH_INCOME = {
    "age": 45,
    "workclass": "Private",
    "fnlgt": 160000,
    "education": "Masters",
    "education-num": 14,
    "marital-status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 15024,
    "capital-loss": 0,
    "hours-per-week": 50,
    "native-country": "United-States",
}

LOW_INCOME = {
    "age": 22,
    "workclass": "Private",
    "fnlgt": 201490,
    "education": "HS-grad",
    "education-num": 9,
    "marital-status": "Never-married",
    "occupation": "Handlers-cleaners",
    "relationship": "Own-child",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 20,
    "native-country": "United-States",
}


def test_get_root_status_and_body():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {
        "greeting": "Welcome to the Census Income Prediction API!"
    }


def test_post_predicts_greater_than_50k():
    r = client.post("/predict", json=HIGH_INCOME)
    assert r.status_code == 200
    assert r.json() == {"prediction": ">50K"}


def test_post_predicts_less_than_or_equal_50k():
    r = client.post("/predict", json=LOW_INCOME)
    assert r.status_code == 200
    assert r.json() == {"prediction": "<=50K"}


def test_post_incomplete_body_is_rejected():
    r = client.post("/predict", json={"age": 30})
    assert r.status_code == 422
    assert "detail" in r.json()


def test_openapi_schema_has_request_example():
    schema = client.get("/openapi.json").json()
    record = schema["components"]["schemas"]["CensusRecord"]
    assert record["example"]["education"] == "Bachelors"
    for name, prop in record["properties"].items():
        assert prop["examples"], f"{name} has no example value"
