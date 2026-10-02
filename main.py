"""FastAPI application serving the census income model."""
from functools import lru_cache
from typing import Literal

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from starter import config
from starter.ml.data import process_data
from starter.ml.model import inference, load_artifact

EXAMPLE = {
    "age": 39,
    "workclass": "State-gov",
    "fnlgt": 77516,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital-gain": 2174,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}


class CensusRecord(BaseModel):
    """One person's census attributes (hyphenated names are the aliases)."""

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={"example": EXAMPLE},
    )

    age: int = Field(..., examples=[39], ge=0, le=120)
    workclass: str = Field(..., examples=["State-gov"])
    fnlgt: int = Field(..., examples=[77516])
    education: str = Field(..., examples=["Bachelors"])
    education_num: int = Field(..., alias="education-num", examples=[13])
    marital_status: str = Field(
        ..., alias="marital-status", examples=["Never-married"]
    )
    occupation: str = Field(..., examples=["Adm-clerical"])
    relationship: str = Field(..., examples=["Not-in-family"])
    race: str = Field(..., examples=["White"])
    sex: str = Field(..., examples=["Male"])
    capital_gain: int = Field(..., alias="capital-gain", examples=[2174])
    capital_loss: int = Field(..., alias="capital-loss", examples=[0])
    hours_per_week: int = Field(
        ..., alias="hours-per-week", examples=[40], ge=0, le=168
    )
    native_country: str = Field(
        ..., alias="native-country", examples=["United-States"]
    )


class Prediction(BaseModel):
    """Response body of the inference endpoint."""

    model_config = ConfigDict(
        json_schema_extra={"example": {"prediction": "<=50K"}}
    )

    prediction: Literal["<=50K", ">50K"] = Field(..., examples=["<=50K"])


app = FastAPI(
    title="Census Income Prediction API",
    description="Predicts whether a person earns more than $50K a year.",
    version="1.0.0",
)


@lru_cache(maxsize=1)
def get_artifacts():
    """Load model, encoder and label binarizer once and cache them."""
    model = load_artifact(config.MODEL_PATH)
    encoder = load_artifact(config.ENCODER_PATH)
    lb = load_artifact(config.LABEL_BINARIZER_PATH)
    return model, encoder, lb


@app.get("/")
async def say_hello() -> dict:
    """Greet the caller."""
    return {"greeting": "Welcome to the Census Income Prediction API!"}


@app.post("/predict", response_model=Prediction)
async def predict(record: CensusRecord) -> Prediction:
    """Run model inference for one census record."""
    model, encoder, lb = get_artifacts()
    row = record.model_dump(by_alias=True)
    frame = pd.DataFrame([row], columns=config.FEATURE_ORDER)
    X, _, _, _ = process_data(
        frame,
        categorical_features=config.CAT_FEATURES,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb,
    )
    pred = inference(model, X)
    label = lb.inverse_transform(pred)[0]
    return Prediction(prediction=str(label))
