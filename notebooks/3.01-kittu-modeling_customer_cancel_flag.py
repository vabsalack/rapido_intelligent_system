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
    from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.metrics import (
        accuracy_score, f1_score, roc_auc_score, confusion_matrix,
        classification_report, make_scorer,
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    from sklearn.ensemble import (
        RandomForestClassifier, GradientBoostingClassifier,
        AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier,
    )
    from rapido_intelligent_system.config_mnb import PROCESSED_DATA_DIR, MODELS_DIR
    warnings.filterwarnings("ignore")

    try:
        from xgboost import XGBClassifier
    except ImportError:
        XGBClassifier = None

    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        LGBMClassifier = None

    try:
        from catboost import CatBoostClassifier
    except ImportError:
        CatBoostClassifier = None

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


@app.cell
def _(results_df):
    results_df
    return


@app.function
def build_classifiers():
    scoring = {
        "accuracy": make_scorer(accuracy_score),
        "f1_weighted": make_scorer(f1_score, average="weighted"),
    }
    classifiers = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        # "SVM_RBF": SVC(kernel="rbf", probability=True, random_state=42),
        # "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        # "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        # "ExtraTrees": ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        # "AdaBoost": AdaBoostClassifier(random_state=42),
        # "Bagging": BaggingClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    }
    # if XGBClassifier is not None:
    #     classifiers["XGBoost"] = XGBClassifier(
    #         n_estimators=100, random_state=42, n_jobs=-1, verbosity=0, use_label_encoder=False, eval_metric="logloss",
    #     )
    # if LGBMClassifier is not None:
    #     classifiers["LightGBM"] = LGBMClassifier(
    #         n_estimators=100, random_state=42, n_jobs=-1, verbose=-1,
    #     )
    # if CatBoostClassifier is not None:
    #     classifiers["CatBoost"] = CatBoostClassifier(
    #         iterations=100, random_state=42, verbose=0,
    #     )
    return classifiers, scoring


@app.function
def run_cv(classifiers, scoring, X_train, y_train, cv=5):
    results = []
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    for name, clf in classifiers.items():
        logger.info(f"Training {name}...")
        pipe = build_pipeline(clf)
        cv_results = cross_validate(
            pipe, X_train, y_train, cv=skf, scoring=scoring,
            return_train_score=True, n_jobs=-1,
        )
        results.append({
            "Model": name,
            "Train Acc": cv_results["train_accuracy"].mean(),
            "Val Acc": cv_results["test_accuracy"].mean(),
            "Train F1": cv_results["train_f1_weighted"].mean(),
            "Val F1": cv_results["test_f1_weighted"].mean(),
        })
    return pd.DataFrame(results).sort_values("Val F1", ascending=False)


@app.function
def evaluate_on_test(clf, X_train, y_train, X_test, y_test):
    pipe = build_pipeline(clf)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test) if hasattr(pipe, "predict_proba") else None
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    auc = roc_auc_score(y_test, y_proba[:, 1]) if y_proba is not None else None
    cm = confusion_matrix(y_test, y_pred)
    return pipe, {"accuracy": acc, "f1": f1, "auc": auc, "confusion_matrix": cm}


@app.cell
def _():
    mo.md("""
    # Customer Cancel Flag Prediction Model

    This notebook builds a **binary classification** model to predict whether a customer will cancel a ride.

    **Target:** `customer_cancel_flag` (0 = Not Cancelled, 1 = Cancelled)

    **Metrics:** Accuracy, F1 (weighted), AUC, Confusion Matrix

    **Pipeline:** Custom feature engineering -> Imputation -> Scaling -> Classifier

    **Workflow:**
    1. Load pre-engineered features
    2. Train/test split (stratified)
    3. Cross-validation across multiple algorithms
    4. Evaluate top model on test set
    5. Save final pipeline
    6. Interactive prediction UI
    """)
    return


@app.cell
def _():
    df_raw = pd.read_csv(PROCESSED_DATA_DIR / "features_customer_cancel_flag.csv")
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
    target = "customer_cancel_flag"
    feature_cols = [c for c in df_raw.columns if c != target]
    X = df_raw[feature_cols]
    y = df_raw[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
    logger.info(f"Class dist (train): {y_train.value_counts().to_dict()}")
    return X_test, X_train, feature_cols, y_test, y_train


@app.cell
def _():
    mo.md("## 2. Model Training & Cross-Validation")
    mo.md("Training classifiers with 5-fold stratified CV...")
    return


@app.cell
def _(X_train, y_train):
    classifiers_dict, scoring = build_classifiers()
    results_df = run_cv(classifiers_dict, scoring, X_train, y_train)
    return classifiers_dict, results_df


@app.cell
def _(inputs_layout):
    inputs_layout
    return


@app.cell
def _(X_test, X_train, classifiers_dict, results_df, y_test, y_train):
    mo.md("## 3. Test Set Evaluation")
    mo.md("Evaluating the best model from CV on the held-out test set...")
    top_name = results_df.iloc[0]["Model"]
    top_clf = classifiers_dict[top_name]
    tuned_pipe, metrics = evaluate_on_test(top_clf, X_train, y_train, X_test, y_test)
    table_md = (
        f"### Test Set Evaluation - {top_name}\n\n"
        f"| Metric | Value |\n|--------|-------|\n"
        f"| Accuracy | {metrics['accuracy']:.4f} |\n"
        f"| F1 (weighted) | {metrics['f1']:.4f} |\n"
        f"| AUC | {metrics['auc']:.4f} |\n"
    )
    return metrics, table_md, tuned_pipe


@app.cell
def _(table_md):
    mo.md(table_md)
    return


@app.cell
def _(metrics):
    mo.md("### Confusion Matrix")
    cm = metrics["confusion_matrix"]
    cm_df = pd.DataFrame(cm, index=["Pred 0", "Pred 1"], columns=["True 0", "True 1"])
    cm_df
    return


@app.cell
def _():
    final_pipe = None
    mo.md("**Note:** Optuna tuning skipped for speed. Using CV best model directly.")
    return (final_pipe,)


@app.cell
def _(final_pipe, tuned_pipe):
    mo.md("## 4. Save Model Pipeline")
    save_dir = MODELS_DIR / "customer_cancel_flag"
    save_dir.mkdir(parents=True, exist_ok=True)
    to_save = final_pipe if final_pipe is not None else tuned_pipe
    joblib.dump(to_save, save_dir / "pipeline.joblib")
    logger.info(f"Saved pipeline to {save_dir / 'pipeline.joblib'}")
    return


@app.cell
def _():
    mo.md("## 5. Interactive Prediction UI")
    mo.md("Configure the inputs below and click **Predict Cancel** to get a prediction.")
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
        "pickup_location": mo.ui.dropdown(options=[f"Loc_{i}" for i in range(1, 9)], value="Loc_1", label="Pickup"),
        "drop_location": mo.ui.dropdown(options=[f"Loc_{i}" for i in range(1, 9)], value="Loc_5", label="Drop"),
        "customer_age": mo.ui.number(value=30, label="Customer Age"),
        "driver_age": mo.ui.number(value=35, label="Driver Age"),
        "driver_experience_years": mo.ui.number(value=5, label="Driver Exp (yrs)"),
        "avg_driver_rating": mo.ui.slider(3.0, 5.0, value=4.5, step=0.1, label="Driver Rating"),
        "avg_customer_rating": mo.ui.slider(3.0, 5.0, value=4.0, step=0.1, label="Cust Rating"),
    }
    predict_btn = mo.ui.run_button(label="Predict Cancel")
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
        "pickup_location": {f"Loc_{i}": i - 1 for i in range(1, 9)},
        "drop_location": {f"Loc_{i}": i - 1 for i in range(1, 9)},
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
        proba = tuned_pipe.predict_proba(X_input)[0]
        status_map = {0: "Not Cancelled", 1: "Cancelled"}
        pred_label = status_map.get(pred, str(pred))
        pred_md = (
            f"### Prediction: **{pred_label}** (class {pred})\n\n"
            f"**Cancel Probability:** {proba[1]:.1%}\n\n"
            f"| Class | Probability |\n|-------|-------------|\n"
            f"| Not Cancelled (0) | {proba[0]:.1%} |\n"
            f"| Cancelled (1) | {proba[1]:.1%} |\n\n"
            f"**Inputs:**\n"
            f"- Distance: {raw['ride_distance_km']} km\n"
            f"- Vehicle: {raw['vehicle_type']}\n"
            f"- City: {raw['city']}"
        )
    else:
        pred_md = "Configure inputs and click **Predict Cancel**."
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


@app.function
def build_pipeline(classifier):
    return Pipeline([
        ("features", RideEfficiencyFeatures()),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", classifier),
    ])


if __name__ == "__main__":
    app.run()
