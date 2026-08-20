import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import pandas as pd
    import numpy as np
    import warnings
    import joblib
    from pathlib import Path
    from loguru import logger
    from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures
    from sklearn.impute import SimpleImputer
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.svm import SVR
    from sklearn.ensemble import (
        RandomForestRegressor, GradientBoostingRegressor,
        AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor,
    )
    from rapido_intelligent_system.config_mnb import PROCESSED_DATA_DIR, MODELS_DIR
    warnings.filterwarnings("ignore")

    try:
        from xgboost import XGBRegressor
    except ImportError:
        XGBRegressor = None

    try:
        from lightgbm import LGBMRegressor
    except ImportError:
        LGBMRegressor = None

    try:
        from catboost import CatBoostRegressor
    except ImportError:
        CatBoostRegressor = None

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        optuna = None

    class RideEfficiencyFeatures(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            return self
        def transform(self, X):
            X = X.copy()
            dist = X["ride_distance_km"].replace(0, np.nan)
            est_time = X["estimated_ride_time_min"].replace(0, np.nan)
            base = X["base_fare"]
            surge = X["surge_multiplier"]
            X["fare_per_km"] = base / dist
            X["surge_impact"] = base * surge
            X["speed_kmh"] = dist / (est_time / 60)
            X["fare_per_min"] = base / est_time
            X["dist_x_surge"] = dist * surge
            return X


@app.function
def build_pipeline(regressor):
    return Pipeline([
        ("features", RideEfficiencyFeatures()),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("regressor", regressor),
    ])


@app.function
def build_models():
    from sklearn.metrics import make_scorer
    neg_rmse = make_scorer(lambda y, p: -np.sqrt(mean_squared_error(y, p)))
    neg_mae = make_scorer(lambda y, p: -mean_absolute_error(y, p))
    scoring = {"rmse": neg_rmse, "mae": neg_mae, "r2": make_scorer(r2_score)}
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "ElasticNet": ElasticNet(),
        # "PolyDeg2": Pipeline([
        #     ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        #     ("lr", LinearRegression()),
        # ]),
        "DecisionTree": DecisionTreeRegressor(random_state=42),
        # "SVR_RBF": SVR(),
    }
    # if XGBRegressor is not None:
    #     models["XGBoost"] = XGBRegressor(
    #         n_estimators=100, random_state=42, n_jobs=-1, verbosity=0, eval_metric="rmse",
    #     )
    # if LGBMRegressor is not None:
    #     models["LightGBM"] = LGBMRegressor(
    #         n_estimators=100, random_state=42, n_jobs=-1, verbose=-1,
    #     )
    # if CatBoostRegressor is not None:
    #     models["CatBoost"] = CatBoostRegressor(
    #         iterations=100, random_state=42, verbose=0,
    #     )
    return models, scoring


@app.function
def run_cv(models, scoring, X_train, y_train, cv=5):
    results = []
    for name, model in models.items():
        logger.info(f"Training {name}...")
        pipe = build_pipeline(model)
        cv_results = cross_validate(
            pipe, X_train, y_train, cv=cv, scoring=scoring,
            return_train_score=True, n_jobs=-1,
        )
        results.append({
            "Model": name,
            "Train RMSE": -cv_results["train_rmse"].mean(),
            "Val RMSE": -cv_results["test_rmse"].mean(),
            "Train MAE": -cv_results["train_mae"].mean(),
            "Val MAE": -cv_results["test_mae"].mean(),
            "Train R2": cv_results["train_r2"].mean(),
            "Val R2": cv_results["test_r2"].mean(),
        })
    return pd.DataFrame(results).sort_values("Val RMSE")


@app.function
def evaluate_on_test(model, X_train, y_train, X_test, y_test):
    pipe = build_pipeline(model)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    pct_error = np.abs(y_test.values - y_pred) / y_test.values * 100
    within_10 = (pct_error <= 10).mean() * 100
    return pipe, {"rmse": rmse, "mae": mae, "r2": r2, "within_10pct": within_10}


@app.cell
def _():
    mo.md("""
    # Booking Value Prediction Model

    This notebook builds a regression model to predict **booking_value** (fare amount) for ride-hailing trips.

    **Pipeline:** Custom feature engineering -> Imputation -> Scaling -> Regression model

    **Workflow:**
    1. Load pre-engineered features
    2. Train/test split
    3. Cross-validation across multiple algorithms
    4. Evaluate top model on test set
    5. Save final pipeline
    6. Interactive prediction UI
    """)
    return


@app.cell
def _():
    df_raw = pd.read_csv(PROCESSED_DATA_DIR / "features_booking_value.csv")
    logger.info(f"Loaded features: {df_raw.shape}")
    return (df_raw,)


@app.cell
def _():
    mo.md("""
    ## 1. Data Preparation
    """)
    return


@app.cell
def _(df_raw):
    target = "booking_value"
    feature_cols = [c for c in df_raw.columns if c != target]
    X = df_raw[feature_cols]
    y = df_raw[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
    )
    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
    return X_test, X_train, feature_cols, y_test, y_train


@app.cell
def _():
    mo.md("## 2. Model Training & Cross-Validation")
    mo.md("Training multiple algorithms with 5-fold CV and comparing performance...")
    return


@app.cell
def _(X_train, y_train):
    models_dict, scoring = build_models()
    results_df = run_cv(models_dict, scoring, X_train, y_train)
    return models_dict, results_df


@app.cell
def _(results_df):
    results_df
    return


@app.cell
def _(X_test, X_train, models_dict, results_df, y_test, y_train):
    mo.md("## 3. Test Set Evaluation")
    mo.md("Evaluating the best model from CV on the held-out test set...")
    top_name = results_df.iloc[0]["Model"]
    top_model = models_dict[top_name]
    tuned_pipe, metrics = evaluate_on_test(top_model, X_train, y_train, X_test, y_test)
    table_md = (
        f"### Test Set Evaluation - {top_name}\n\n"
        f"| Metric | Value |\n|--------|-------|\n"
        f"| RMSE | {metrics['rmse']:.2f} |\n"
        f"| MAE | {metrics['mae']:.2f} |\n"
        f"| R2 | {metrics['r2']:.4f} |\n"
        f"| Within 10% | {metrics['within_10pct']:.1f}% |"
    )
    return table_md, tuned_pipe


@app.cell
def _(table_md):
    mo.md(table_md)
    return


@app.cell
def _():
    final_pipe = None
    mo.md("**Note:** Optuna tuning skipped for speed. Using CV best model directly.")
    return (final_pipe,)


@app.cell
def _(final_pipe, tuned_pipe):
    mo.md("## 4. Save Model Pipeline")
    save_dir = MODELS_DIR / "booking_value"
    save_dir.mkdir(parents=True, exist_ok=True)
    to_save = final_pipe if final_pipe is not None else tuned_pipe
    joblib.dump(to_save, save_dir / "pipeline.joblib")
    logger.info(f"Saved pipeline to {save_dir / 'pipeline.joblib'}")
    return


@app.cell
def _():
    mo.md("## 5. Interactive Prediction UI")
    mo.md("Configure the inputs below and click **Predict Fare** to get a prediction.")
    return


@app.cell
def _():
    input_widgets = {
        "ride_distance_km": mo.ui.number(value=10.0, label="Ride Distance (km)"),
        "estimated_ride_time_min": mo.ui.number(value=30.0, label="Est. Ride Time (min)"),
        "base_fare": mo.ui.number(value=200.0, label="Base Fare"),
        "surge_multiplier": mo.ui.slider(1.0, 3.0, value=1.0, step=0.1, label="Surge"),
        "traffic_level": mo.ui.dropdown(options=["Low", "Medium", "High"], value="Medium", label="Traffic"),
        "weather_condition": mo.ui.dropdown(options=["Clear", "Rain", "Heavy Rain"], value="Clear", label="Weather"),
        "vehicle_type": mo.ui.dropdown(options=["Bike", "Auto", "Cab"], value="Cab", label="Vehicle"),
        "hour_of_day": mo.ui.slider(0, 23, value=12, label="Hour"),
        "day_of_week": mo.ui.dropdown(
            options=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            value="Monday", label="Day",
        ),
        "city": mo.ui.dropdown(options=["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad"], value="Mumbai", label="City"),
        "pickup_location": mo.ui.dropdown(options=[f"Loc_{i}" for i in range(1, 51)], value="Loc_1", label="Pickup"),
        "drop_location": mo.ui.dropdown(options=[f"Loc_{i}" for i in range(1, 51)], value="Loc_10", label="Drop"),
        "customer_age": mo.ui.number(value=30, label="Customer Age"),
        "driver_age": mo.ui.number(value=35, label="Driver Age"),
        "driver_experience_years": mo.ui.number(value=5, label="Driver Exp (yrs)"),
        "avg_driver_rating": mo.ui.slider(3.0, 5.0, value=4.5, step=0.1, label="Driver Rating"),
        "avg_customer_rating": mo.ui.slider(3.0, 5.0, value=4.0, step=0.1, label="Cust Rating"),
    }
    predict_btn = mo.ui.run_button(label="Predict Fare")
    return input_widgets, predict_btn


@app.cell
def _(feature_cols, input_widgets, predict_btn, tuned_pipe):
    raw = {k: w.value for k, w in input_widgets.items()}

    _cat_map = {
        "traffic_level": {"Low": 0, "Medium": 1, "High": 2},
        "weather_condition": {"Clear": 0, "Rain": 1, "Heavy Rain": 2},
        "vehicle_type": {"Bike": 0, "Auto": 1, "Cab": 2},
        "day_of_week": {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6},
        "city": {"Mumbai": 4, "Delhi": 1, "Bangalore": 2, "Chennai": 3, "Hyderabad": 0},
        "pickup_location": {f"Loc_{i}": i - 1 for i in range(1, 51)},
        "drop_location": {f"Loc_{i}": i - 1 for i in range(1, 51)},
    }

    encoded = {}
    for k, v in raw.items():
        if k in _cat_map:
            encoded[k] = _cat_map[k].get(v, 0)
        else:
            encoded[k] = v
    row = {col: encoded.get(col, 0) for col in feature_cols}
    X_input = pd.DataFrame([row])
    if predict_btn.value and tuned_pipe is not None:
        pred = tuned_pipe.predict(X_input)[0]
        pred_md = (
            f"### Predicted Booking Value: **Rs. {pred:.2f}**\n\n"
            f"- Distance: {raw['ride_distance_km']} km\n"
            f"- Est. Time: {raw['estimated_ride_time_min']} min\n"
            f"- Base Fare: Rs. {raw['base_fare']}\n"
            f"- Surge: {raw['surge_multiplier']}x\n"
            f"- Vehicle: {raw['vehicle_type']}\n"
            f"- City: {raw['city']}"
        )
    else:
        pred_md = "Configure inputs and click **Predict Fare**."
    return (pred_md,)


@app.cell
def _(pred_md):
    mo.md(pred_md)
    return


@app.cell
def _(input_widgets, predict_btn):
    inputs_layout = mo.vstack([
        mo.hstack([input_widgets["ride_distance_km"], input_widgets["estimated_ride_time_min"], input_widgets["base_fare"]]),
        mo.hstack([input_widgets["surge_multiplier"], input_widgets["traffic_level"], input_widgets["weather_condition"]]),
        mo.hstack([input_widgets["vehicle_type"], input_widgets["hour_of_day"], input_widgets["day_of_week"]]),
        mo.hstack([input_widgets["city"], input_widgets["pickup_location"], input_widgets["drop_location"]]),
        mo.hstack([input_widgets["customer_age"], input_widgets["driver_age"], input_widgets["driver_experience_years"]]),
        mo.hstack([input_widgets["avg_driver_rating"], input_widgets["avg_customer_rating"]]),
        predict_btn,
    ])
    return (inputs_layout,)


@app.cell
def _(inputs_layout):
    inputs_layout
    return


if __name__ == "__main__":
    app.run()
