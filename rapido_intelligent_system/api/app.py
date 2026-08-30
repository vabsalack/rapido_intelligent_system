"""FastAPI application serving the four saved modelling pipelines.

Layout of this module
---------------------
* ``MODELS_DIR``        — the ``models/`` folder (derived, not imported, so the
                          server owns its own path logic).
* ``ModelSpec``         — dataclass describing one deployable model.
* ``REGISTRY``          — ``{model_id: ModelSpec}``, built at import time by
                          loading each pipeline + its ``_meta.json``.
* Request schemas       — one Pydantic model per endpoint, generated from the
                          feature catalog (numeric ``float`` / categorical ``str``).
* ``predict_one``       — the shared inference helper (polars row -> prediction).
* Routes                — ``/``, ``/health``, ``/models``, ``POST /predict/{id}``.

Every pipeline already contains its full preprocessing chain (feature
engineering -> impute -> scale / one-hot -> learner), so requests carry only the
**raw** feature values; the pipeline re-derives everything at inference time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import polars as pl
from pydantic import BaseModel, create_model

from rapido_intelligent_system.modeling_mnb import (
    BOOKING_CAT_FEATURES,
    BOOKING_CATEGORIES,
    BOOKING_FEATURES,
    BOOKING_NUM_FEATURES,
    BOOKING_VALUE_CAT_FEATURES,
    BOOKING_VALUE_CATEGORIES,
    BOOKING_VALUE_FEATURES,
    BOOKING_VALUE_NUM_FEATURES,
    CUSTOMER_BINARY_CLASSES,
    CUSTOMER_CAT_FEATURES,
    CUSTOMER_CATEGORIES,
    CUSTOMER_FEATURES,
    CUSTOMER_NUM_FEATURES,
    DRIVER_BINARY_CLASSES,
    DRIVER_CAT_FEATURES,
    DRIVER_CATEGORIES,
    DRIVER_FEATURES,
    DRIVER_NUM_FEATURES,
    TARGET_CLASSES,
    load_model,
)

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


# ----------------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------------
@dataclass
class ModelSpec:
    id: str
    title: str
    task: Literal["multiclass", "binary", "regression"]
    target: str
    pipeline: Any
    features: list[str]
    num_features: list[str]
    cat_features: list[str]
    categories: dict[str, list[str]]
    class_labels: list[str] | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def _load_spec(
    model_id: str,
    title: str,
    task: Literal["multiclass", "binary", "regression"],
    target: str,
    filename: str,
    features: list[str],
    num_features: list[str],
    cat_features: list[str],
    categories: dict[str, list[str]],
    class_labels: list[str] | None = None,
) -> ModelSpec:
    artifact = MODELS_DIR / filename
    meta_path = MODELS_DIR / f"{artifact.stem}_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return ModelSpec(
        id=model_id,
        title=title,
        task=task,
        target=target,
        pipeline=load_model(artifact),
        features=features,
        num_features=num_features,
        cat_features=cat_features,
        categories=categories,
        class_labels=class_labels,
        meta=meta,
    )


REGISTRY: dict[str, ModelSpec] = {
    spec.id: spec
    for spec in [
        _load_spec(
            "booking_status",
            "Booking status",
            "multiclass",
            "booking_status",
            "booking_status_model.joblib",
            BOOKING_FEATURES,
            BOOKING_NUM_FEATURES,
            BOOKING_CAT_FEATURES,
            BOOKING_CATEGORIES,
            TARGET_CLASSES,
        ),
        _load_spec(
            "booking_value",
            "Booking value (₹)",
            "regression",
            "booking_value",
            "booking_value_model.joblib",
            BOOKING_VALUE_FEATURES,
            BOOKING_VALUE_NUM_FEATURES,
            BOOKING_VALUE_CAT_FEATURES,
            BOOKING_VALUE_CATEGORIES,
        ),
        _load_spec(
            "customer_cancel_flag",
            "Customer cancel flag",
            "binary",
            "customer_cancel_flag",
            "customer_cancel_flag_model.joblib",
            CUSTOMER_FEATURES,
            CUSTOMER_NUM_FEATURES,
            CUSTOMER_CAT_FEATURES,
            CUSTOMER_CATEGORIES,
            CUSTOMER_BINARY_CLASSES,
        ),
        _load_spec(
            "driver_delay_flag",
            "Driver delay flag",
            "binary",
            "driver_delay_flag",
            "driver_delay_flag_model.joblib",
            DRIVER_FEATURES,
            DRIVER_NUM_FEATURES,
            DRIVER_CAT_FEATURES,
            DRIVER_CATEGORIES,
            DRIVER_BINARY_CLASSES,
        ),
    ]
}


# ----------------------------------------------------------------------------
# In-memory request schemas (one per model, generated from the feature catalog)
# ----------------------------------------------------------------------------
def _request_model(spec: ModelSpec) -> type[BaseModel]:
    fields: dict[str, type] = {}
    for feature in spec.num_features:
        fields[feature] = float
    for feature in spec.cat_features:
        fields[feature] = str
    return create_model(f"{spec.id}_request", **fields)


REQUEST_MODELS: dict[str, type[BaseModel]] = {
    spec.id: _request_model(spec) for spec in REGISTRY.values()
}

BookingStatusRequest = REQUEST_MODELS["booking_status"]
BookingValueRequest = REQUEST_MODELS["booking_value"]
CustomerCancelFlagRequest = REQUEST_MODELS["customer_cancel_flag"]
DriverDelayFlagRequest = REQUEST_MODELS["driver_delay_flag"]


# ----------------------------------------------------------------------------
# Inference helper
# ----------------------------------------------------------------------------
def predict_one(spec: ModelSpec, payload_data: dict[str, Any]) -> dict[str, Any]:
    """Run one raw row through the pipeline and return a JSON-safe result."""
    frame = pl.DataFrame([payload_data]).select(spec.features)
    raw = spec.pipeline.predict(frame)[0]

    if spec.task == "regression":
        return {"model": spec.id, "task": spec.task, "prediction": float(raw)}

    predicted = int(raw) if raw is not None and not isinstance(raw, str) else str(raw)
    result: dict[str, Any] = {"model": spec.id, "task": spec.task, "prediction": predicted}
    if spec.task == "binary" and spec.class_labels:
        classes = [str(c) for c in spec.pipeline.classes_]
        result["label"] = spec.class_labels[classes.index(str(predicted))]
    else:
        result["label"] = predicted

    if hasattr(spec.pipeline, "predict_proba"):
        classes = [str(c) for c in spec.pipeline.classes_]
        probs = spec.pipeline.predict_proba(frame)[0]
        result["probabilities"] = [{c: float(p)} for c, p in zip(classes, probs, strict=False)]
    return result


# ----------------------------------------------------------------------------
# App + middleware
# ----------------------------------------------------------------------------
app = FastAPI(
    title="Rapido Intelligent System — model API",
    description=(
        "Loads the four joblib pipelines (booking_status, booking_value, "
        "customer_cancel_flag, driver_delay_flag) and predicts from raw inputs."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev setting; the marimo frontend runs on another port
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------------------------------
# Response / info models
# ----------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str
    models: list[str]


class FeatureInfo(BaseModel):
    name: str
    type: Literal["number", "category"]
    options: list[str] | None = None


class ModelInfo(BaseModel):
    id: str
    title: str
    task: str
    target: str
    learner: str | None = None
    headline: str | None = None
    features: list[FeatureInfo]
    class_labels: list[str] | None = None


class PredictionResponse(BaseModel):
    model: str
    task: str
    prediction: int | float | str
    label: str | None = None
    probabilities: list[dict[str, float]] | None = None


def _model_info(spec: ModelSpec) -> ModelInfo:
    meta = spec.meta
    learner = meta.get("model")
    headline = (
        meta.get("val_r2")
        or meta.get("r2")
        or meta.get("val_f1_weighted") or meta.get("val_f1") or meta.get("val_accuracy")
    )
    features = [
        FeatureInfo(
            name=feature,
            type="number" if feature in spec.num_features else "category",
            options=spec.categories.get(feature),
        )
        for feature in spec.features
    ]
    return ModelInfo(
        id=spec.id,
        title=spec.title,
        task=spec.task,
        target=spec.target,
        learner=learner,
        headline=str(headline) if headline is not None else None,
        features=features,
        class_labels=spec.class_labels,
    )


# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------
@app.get("/")
def index() -> dict[str, Any]:
    return {
        "service": "rapido_intelligent_system model API",
        "docs": "/docs",
        "openapi": "/openapi.json",
        "models": sorted(REGISTRY),
        "predict_endpoints": [f"/predict/{model_id}" for model_id in sorted(REGISTRY)],
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    return HealthResponse(status="ok", models=sorted(REGISTRY))


@app.get("/models", response_model=list[ModelInfo], tags=["meta"])
def list_models() -> list[ModelInfo]:
    return [_model_info(spec) for spec in REGISTRY.values()]


@app.post(
    "/predict/booking_status",
    response_model=PredictionResponse,
    response_model_exclude_none=True,
    tags=["predict"],
)
def predict_booking_status(
    payload: BookingStatusRequest,
) -> PredictionResponse:
    return PredictionResponse(**predict_one(REGISTRY["booking_status"], payload.model_dump()))


@app.post(
    "/predict/booking_value",
    response_model=PredictionResponse,
    response_model_exclude_none=True,
    tags=["predict"],
)
def predict_booking_value(
    payload: BookingValueRequest,
) -> PredictionResponse:
    return PredictionResponse(**predict_one(REGISTRY["booking_value"], payload.model_dump()))


@app.post(
    "/predict/customer_cancel_flag",
    response_model=PredictionResponse,
    response_model_exclude_none=True,
    tags=["predict"],
)
def predict_customer_cancel_flag(
    payload: CustomerCancelFlagRequest,
) -> PredictionResponse:
    return PredictionResponse(
        **predict_one(REGISTRY["customer_cancel_flag"], payload.model_dump())
    )


@app.post(
    "/predict/driver_delay_flag",
    response_model=PredictionResponse,
    response_model_exclude_none=True,
    tags=["predict"],
)
def predict_driver_delay_flag(
    payload: DriverDelayFlagRequest,
) -> PredictionResponse:
    return PredictionResponse(
        **predict_one(REGISTRY["driver_delay_flag"], payload.model_dump())
    )


@app.get("/predict/{model_id}/example", response_model=dict, tags=["predict"])
def example_payload(model_id: str) -> dict[str, Any]:
    spec = REGISTRY.get(model_id)
    if spec is None:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_id}")
    return {
        feature: (spec.categories.get(feature, ["—"])[0] if feature in spec.cat_features else 0.0)
        for feature in spec.features
    }