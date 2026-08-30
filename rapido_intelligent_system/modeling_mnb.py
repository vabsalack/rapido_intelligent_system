import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import json
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import polars as pl
    import altair as alt
    import joblib
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.base import BaseEstimator, ClassifierMixin, clone
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import (
        accuracy_score, classification_report,
        precision_score, recall_score, f1_score,
        mean_absolute_error, root_mean_squared_error, r2_score,
        mean_absolute_percentage_error,
    )
    from loguru import logger
    from rapido_intelligent_system.config_mnb import MODELS_DIR

    # --------------------------------------------------------------------------
    # booking_status feature catalog (as saved by 0.02)
    # --------------------------------------------------------------------------
    BOOKING_FEATURES = [
        "ride_distance_km", "estimated_ride_time_min", "base_fare", "surge_multiplier",
        "traffic_level", "weather_condition", "avg_wait_time_min", "avg_surge_multiplier",
        "demand_level", "season", "vehicle_type",
    ]
    BOOKING_TARGET = "booking_status"

    BOOKING_NUM_FEATURES = [
        "ride_distance_km", "estimated_ride_time_min", "base_fare", "surge_multiplier",
        "avg_wait_time_min", "avg_surge_multiplier",
    ]
    BOOKING_CAT_FEATURES = [
        "traffic_level", "weather_condition",
        "demand_level", "season", "vehicle_type",
    ]
    BOOKING_ENG_FEATURES = [
        "fare_per_km", "surge_impact", "mins_per_km", "wait_share_of_ride",
    ]

    BOOKING_CATEGORIES = {
        "traffic_level": ["Low", "Medium", "High"],
        "weather_condition": ["Clear", "Rain", "Heavy Rain"],
        "demand_level": ["Low", "Medium"],
        "season": ["Monsoon", "Summer", "Winter"],
        "vehicle_type": ["Auto", "Bike", "Cab"],
    }
    TARGET_CLASSES = ["Cancelled", "Completed", "Incomplete"]

    # ------------------------------------------------------------------------
    # booking_value feature catalog (as saved by 0.02) — a REGRESSION task
    # ------------------------------------------------------------------------
    BOOKING_VALUE_FEATURES = [
        "ride_distance_km", "estimated_ride_time_min", "base_fare",
        "surge_multiplier", "pickup_location", "traffic_level",
        "avg_wait_time_min", "avg_surge_multiplier", "vehicle_type",
        "weather_condition", "hour_of_day",
    ]
    BOOKING_VALUE_TARGET = "booking_value"

    BOOKING_VALUE_NUM_FEATURES = [
        "ride_distance_km", "estimated_ride_time_min", "base_fare",
        "surge_multiplier", "avg_wait_time_min", "avg_surge_multiplier",
        "hour_of_day",
    ]
    BOOKING_VALUE_CAT_FEATURES = [
        "pickup_location", "traffic_level", "vehicle_type", "weather_condition",
    ]
    # NOTE: deliberate omission of `surge_impact`
    # booking_value ≈ base_fare × surge_multiplier (r ≈ 0.9985), so feeding the
    # product back as a feature would hand the model the answer — near-leakage.
    # The pipeline only adds interaction-unrelated efficiency features and lets
    # the learners (re)learn the base_fare × surge interaction on their own.
    BOOKING_VALUE_ENG_FEATURES = [
        "fare_per_km", "mins_per_km", "wait_share_of_ride",
    ]

    BOOKING_VALUE_CATEGORIES = {
        "traffic_level": ["Low", "Medium", "High"],
        "weather_condition": ["Clear", "Rain", "Heavy Rain"],
        "vehicle_type": ["Auto", "Bike", "Cab"],
    }

    # ------------------------------------------------------------------------
    # customer_cancel_flag feature catalog (as saved by 0.02) — BINARY task
    # ------------------------------------------------------------------------
    CUSTOMER_FEATURES = [
        "customer_gender", "customer_age", "customer_city",
        "customer_signup_days_ago", "total_bookings", "completed_rides",
        "incomplete_rides", "avg_customer_rating", "preferred_vehicle_type",
    ]
    CUSTOMER_TARGET = "customer_cancel_flag"

    CUSTOMER_NUM_FEATURES = [
        "customer_age", "customer_signup_days_ago", "total_bookings",
        "completed_rides", "incomplete_rides", "avg_customer_rating",
    ]
    CUSTOMER_CAT_FEATURES = [
        "customer_gender", "customer_city", "preferred_vehicle_type",
    ]
    CUSTOMER_ENG_FEATURES = [
        "completion_rate", "incomplete_rate", "bookings_per_year",
    ]

    CUSTOMER_CATEGORIES = {
        "customer_gender": ["Female", "Male", "Non-Binary"],
        "customer_city": ["Bangalore", "Chennai", "Delhi", "Hyderabad", "Mumbai"],
        "preferred_vehicle_type": ["Auto", "Bike", "Cab"],
    }
    CUSTOMER_BINARY_CLASSES = ["Not cancelled", "Cancelled"]

    # ------------------------------------------------------------------------
    # driver_delay_flag feature catalog (as saved by 0.02) — BINARY task
    # ------------------------------------------------------------------------
    DRIVER_FEATURES = [
        "driver_age", "driver_city", "vehicle_type", "driver_experience_years",
        "total_assigned_rides", "accepted_rides", "incomplete_rides",
        "acceptance_rate", "avg_driver_rating", "avg_pickup_delay_min",
    ]
    DRIVER_TARGET = "driver_delay_flag"

    DRIVER_NUM_FEATURES = [
        "driver_age", "driver_experience_years", "total_assigned_rides",
        "accepted_rides", "incomplete_rides", "acceptance_rate",
        "avg_driver_rating", "avg_pickup_delay_min",
    ]
    DRIVER_CAT_FEATURES = [
        "driver_city", "vehicle_type",
    ]
    DRIVER_ENG_FEATURES = [
        "incomplete_share", "delay_x_incomplete", "experience_share",
    ]

    DRIVER_CATEGORIES = {
        "driver_city": ["Bangalore", "Chennai", "Delhi", "Hyderabad", "Mumbai"],
        "vehicle_type": ["Auto", "Bike", "Cab"],
    }
    DRIVER_BINARY_CLASSES = ["No delay", "Delayed"]


# ----------------------------------------------------------------------------
# Feature engineering — runs INSIDE the pipeline at fit and predict time
# ----------------------------------------------------------------------------
@app.function
def engineer_booking_features(df: pl.DataFrame) -> pl.DataFrame:
    """Derive ride-efficiency features from the raw booking context columns.

    All divisions are guarded (`clip`) so the transformer stays finite even on nulls or
    zero denominators; the imputer downstream mops up any remaining NaN.
    """
    return df.with_columns([
        (pl.col("base_fare") / pl.col("ride_distance_km").clip(0.1)).alias("fare_per_km"),
        (pl.col("base_fare") * pl.col("surge_multiplier")).alias("surge_impact"),
        (pl.col("estimated_ride_time_min") / pl.col("ride_distance_km").clip(0.1)).alias("mins_per_km"),
        (pl.col("avg_wait_time_min") / pl.col("estimated_ride_time_min").clip(1.0)).alias("wait_share_of_ride"),
    ])


# ----------------------------------------------------------------------------
# Column-aware preprocessing (impute -> scale / one-hot)
# ----------------------------------------------------------------------------
@app.function
def booking_preprocessing() -> ColumnTransformer:
    """Impute + scale numerics (originals + engineered) and one-hot categories."""
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]),
                BOOKING_NUM_FEATURES + BOOKING_ENG_FEATURES,
            ),
            (
                "cat",
                Pipeline([
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]),
                BOOKING_CAT_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


@app.function
def build_booking_pipeline(classifier) -> Pipeline:
    """End-to-end pipeline: engineer -> preprocess -> classifier.

    Feeds on any row that has the 11 raw booking features and returns a label
    (or probability vector) — everything happens inside the pipeline.
    """
    return Pipeline([
        ("engineer", FunctionTransformer(engineer_booking_features, validate=False)),
        ("prep", booking_preprocessing()),
        ("clf", classifier),
    ])


# ----------------------------------------------------------------------------
# booking_value regression pipeline (all preprocessing INSIDE the pipeline)
# ----------------------------------------------------------------------------
@app.function
def engineer_booking_value_features(df: pl.DataFrame) -> pl.DataFrame:
    """Derive efficiency features for the regression input frame.

    Guarded (`clip`) divisions like the classification engineer; deliberately does NOT
    add `surge_impact` (= base_fare × surge_multiplier) because the target
    `booking_value` is generated from that product — adding it would leak the
    answer. The models must learn the interaction themselves.
    """
    return df.with_columns([
        (pl.col("base_fare") / pl.col("ride_distance_km").clip(0.1)).alias("fare_per_km"),
        (pl.col("estimated_ride_time_min") / pl.col("ride_distance_km").clip(0.1)).alias("mins_per_km"),
        (pl.col("avg_wait_time_min") / pl.col("estimated_ride_time_min").clip(1.0)).alias("wait_share_of_ride"),
    ])


@app.function
def booking_value_preprocessing() -> ColumnTransformer:
    """Impute + scale numerics (originals + engineered) and one-hot categories."""
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]),
                BOOKING_VALUE_NUM_FEATURES + BOOKING_VALUE_ENG_FEATURES,
            ),
            (
                "cat",
                Pipeline([
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]),
                BOOKING_VALUE_CAT_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


@app.function
def build_booking_value_pipeline(regressor) -> Pipeline:
    """End-to-end regression pipeline: engineer -> preprocess -> regressor."""
    return Pipeline([
        ("engineer", FunctionTransformer(engineer_booking_value_features, validate=False)),
        ("prep", booking_value_preprocessing()),
        ("clf", regressor),
    ])


@app.function
def metrics_row_reg(model_name: str, y_true, y_pred) -> dict:
    """One row of regression metrics: RMSE, MAE, R², MAPE.

    R² (headline, scale-free) alongside RMSE/MAE in the target's own units (₹).
    """
    return {
        "model": model_name,
        "rmse": round(float(root_mean_squared_error(y_true, y_pred)), 3),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 3),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "mape": round(float(mean_absolute_percentage_error(y_true, y_pred)), 4),
    }


@app.function
def metrics_table_reg(rows: list[dict]) -> pl.DataFrame:
    """Assemble and sort the baseline comparison, best R² first."""
    return pl.DataFrame(rows).sort("r2", descending=True)


# ----------------------------------------------------------------------------
# customer_cancel_flag binary classification pipeline (all INSIDE the pipeline)
# ----------------------------------------------------------------------------
@app.function
def engineer_customer_features(df: pl.DataFrame) -> pl.DataFrame:
    """Derive customer-behaviour ratio features from raw history columns.

    Ratios are guarded with `clip` so division never blows up on zero history;
    the imputer downstream handles any residual NaN.
    """
    return df.with_columns([
        (pl.col("completed_rides") / pl.col("total_bookings").clip(1)).alias("completion_rate"),
        (pl.col("incomplete_rides") / pl.col("total_bookings").clip(1)).alias("incomplete_rate"),
        (pl.col("total_bookings") / (pl.col("customer_signup_days_ago") / 365.0).clip(0.1)).alias("bookings_per_year"),
    ])


@app.function
def customer_preprocessing() -> ColumnTransformer:
    """Impute + scale numerics (originals + engineered) and one-hot categories."""
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]),
                CUSTOMER_NUM_FEATURES + CUSTOMER_ENG_FEATURES,
            ),
            (
                "cat",
                Pipeline([
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]),
                CUSTOMER_CAT_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


@app.function
def build_customer_pipeline(classifier) -> Pipeline:
    """End-to-end binary pipeline: engineer -> preprocess -> classifier."""
    return Pipeline([
        ("engineer", FunctionTransformer(engineer_customer_features, validate=False)),
        ("prep", customer_preprocessing()),
        ("clf", classifier),
    ])


@app.function
def metrics_row_bin(model_name: str, y_true, y_pred, zero_division: int = 0) -> dict:
    """One row of binary metrics: accuracy, precision, recall, F1.

    Precision/recall/F1 are for the positive class (the cancellation flag = 1);
    F1 is the headline metric for the (near-balanced) binary target.
    """
    return {
        "model": model_name,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=zero_division)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=zero_division)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=zero_division)), 4),
    }


@app.function
def metrics_table_bin(rows: list[dict]) -> pl.DataFrame:
    """Assemble and sort the baseline comparison, best F1 first."""
    return pl.DataFrame(rows).sort("f1", descending=True)


# ----------------------------------------------------------------------------
# driver_delay_flag binary classification pipeline (all INSIDE the pipeline)
# ----------------------------------------------------------------------------
@app.function
def engineer_driver_features(df: pl.DataFrame) -> pl.DataFrame:
    """Derive driver-behaviour ratio and interaction features.

    The strongest signals are `incomplete_rides` and `avg_pickup_delay_min`, so the
    engineered set pairs a share-of-assignments ratio with a delay × incompleteness
    interaction, plus an experience-share feature. Divisions are `clip`-guarded.
    """
    return df.with_columns([
        (pl.col("incomplete_rides") / pl.col("total_assigned_rides").clip(1)).alias("incomplete_share"),
        (pl.col("avg_pickup_delay_min") * pl.col("incomplete_rides").clip(0)).alias("delay_x_incomplete"),
        (pl.col("driver_experience_years") / pl.col("driver_age").clip(1)).alias("experience_share"),
    ])


@app.function
def driver_preprocessing() -> ColumnTransformer:
    """Impute + scale numerics (originals + engineered) and one-hot categories."""
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]),
                DRIVER_NUM_FEATURES + DRIVER_ENG_FEATURES,
            ),
            (
                "cat",
                Pipeline([
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]),
                DRIVER_CAT_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


@app.function
def build_driver_pipeline(classifier) -> Pipeline:
    """End-to-end binary pipeline: engineer -> preprocess -> classifier."""
    return Pipeline([
        ("engineer", FunctionTransformer(engineer_driver_features, validate=False)),
        ("prep", driver_preprocessing()),
        ("clf", classifier),
    ])


# ----------------------------------------------------------------------------
# Label-safe wrapper for learners that require numeric multi-class targets
# ----------------------------------------------------------------------------
@app.function
def LabelsafeClassifier(estimator) -> "LabelsafeClassifier":
    """Build a classifier wrapper that encodes string targets to 0..n-1 internally.

    Some learners (e.g. XGBoost) reject non-integer multi-class labels. The wrapper
    keeps the rest of the pipeline untouched: it fits on raw string labels, stores the
    label mapping, and maps predictions (and probabilities) back on inference.
    """
    class _Wrapper(BaseEstimator, ClassifierMixin):
        def __init__(self, estimator=None):
            self.estimator = estimator

        def fit(self, X, y, **fit_params):
            self.le_ = LabelEncoder().fit(y)
            self.classes_ = self.le_.classes_
            self.estimator_ = clone(self.estimator)
            self.estimator_.fit(X, self.le_.transform(y), **fit_params)
            return self

        def predict(self, X):
            return self.le_.inverse_transform(self.estimator_.predict(X))

        def predict_proba(self, X):
            return self.estimator_.predict_proba(X)

        def get_params(self, deep=True):
            return {"estimator": self.estimator}

        def set_params(self, **params):
            if "estimator" in params:
                self.estimator = params.pop("estimator")
            if params:
                raise ValueError(f"Unsupported params: {sorted(params)}")
            return self

    return _Wrapper(estimator=estimator)


# ----------------------------------------------------------------------------
# Metric helpers
# ----------------------------------------------------------------------------
@app.function
def metrics_row(model_name: str, y_true, y_pred, zero_division: int = 0) -> dict:
    """One row of accuracy + micro/macro/weighted precision-recall-F1.

    Micro-P/R/F1 collapse to accuracy for a single-label multiclass problem
    (all false positives balance all false negatives), so micro is filled from
    accuracy; macro/weighted are read from the report's aggregates.
    """
    acc = round(float(accuracy_score(y_true, y_pred)), 4)
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=zero_division)
    row = {"model": model_name, "accuracy": acc}
    _micro = {"precision": acc, "recall": acc, "f1-score": acc}
    for agg, prefix in [("micro avg", "micro"), ("macro avg", "macro"), ("weighted avg", "weighted")]:
        _src = _micro if agg == "micro avg" else rep[agg]
        for col, key in [("precision", "precision"), ("recall", "recall"), ("f1", "f1-score")]:
            row[f"{col}_{prefix}"] = round(float(_src[key]), 4)
    return row


@app.function
def metrics_table(rows: list[dict]) -> pl.DataFrame:
    """Assemble and sort the baseline comparison, best weighted-F1 first."""
    return pl.DataFrame(rows).sort("f1_weighted", descending=True)


@app.function
def metrics_bar_chart(df: pl.DataFrame, score_col: str = "f1_weighted") -> alt.Chart:
    """Horizontal bar chart of one score column across models."""
    plot_df = df.select(["model", score_col]).rename({score_col: "score"})
    return (
        alt.Chart(plot_df, title=f"{score_col} by model")
        .mark_bar()
        .encode(
            x=alt.X("score:Q", title=score_col),
            y=alt.Y("model:N", sort="-x", title=""),
            tooltip=["model", "score"],
        )
        .properties(width=480, height=340)
    )


# ----------------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------------
@app.function
def save_model(pipeline, model_dir: Path, filename: str, metadata: dict) -> tuple[Path, Path]:
    """joblib-dump the pipeline and write its metadata JSON."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / filename
    meta_path = model_dir / f"{Path(filename).stem}_meta.json"
    joblib.dump(pipeline, model_path)
    meta_path.write_text(json.dumps(metadata, indent=2))
    logger.success(f"Saved pipeline -> {model_path}")
    return model_path, meta_path


@app.function
def load_model(model_path) -> Pipeline:
    """Load a joblib-saved pipeline."""
    model_path = Path(model_path)
    pipe = joblib.load(model_path)
    logger.success(f"Loaded pipeline <- {model_path}")
    return pipe


if __name__ == "__main__":
    app.run()