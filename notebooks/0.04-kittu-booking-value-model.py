import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    from pathlib import Path
    import polars as pl
    import altair as alt
    from scipy.stats import loguniform, uniform
    from sklearn.base import clone
    from sklearn.model_selection import train_test_split, RandomizedSearchCV
    from sklearn.linear_model import SGDRegressor, ElasticNet
    from sklearn.svm import SVR
    from sklearn.ensemble import (
        RandomForestRegressor, HistGradientBoostingRegressor,
    )
    from xgboost import XGBRegressor
    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor
    from loguru import logger
    from rapido_intelligent_system.config_mnb import (
        PROCESSED_DATA_DIR, MODELS_DIR,
    )
    from rapido_intelligent_system.modeling_mnb import (
        BOOKING_VALUE_FEATURES, BOOKING_VALUE_TARGET, BOOKING_VALUE_ENG_FEATURES,
        BOOKING_VALUE_CATEGORIES,
        engineer_booking_value_features, booking_value_preprocessing,
        build_booking_value_pipeline, metrics_row_reg, metrics_table_reg,
        metrics_bar_chart, save_model, load_model,
    )


@app.cell
def _():
    mo.md(r"""
    # 0.04 — booking_value Model (Regression)

    **Project:** Rapido Intelligent System · **Author:** kittu
    **Goal:** predict the **fare amount** (`booking_value` in ₹) from the raw booking-context
    features selected in notebook 0.02.

    ### Workflow in this notebook
    1. **Holdout** — a validation set is carved from the training file; the untouched
       `data/processed/booking_value_test.csv` is kept for a **final generalization check only**.
    2. **Pipelines** — every experiment is one scikit-learn `Pipeline` that does **feature
       engineering → impute → scale / one-hot → regressor**, so the same object serves
       training *and* inference on raw inputs.
    3. **Baselines** — SGD, ElasticNet, SVR, Random Forest, HistGradientBoosting, XGBoost,
       CatBoost and LightGBM. Slow learners get a **sample** of the data so cells pass the
       smoke test fast.
    4. **Select** the best baseline, **fine-tune** it with randomized-search CV, **save** it,
       reload it, and estimate generalisation on the test split.
    5. **Interactive UI** — input ride context and see a predicted fare from the saved pipeline.
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ### Why these metrics?

    `booking_value` is a **regression** target (₹ amounts, no class imbalance), so the right
    metrics live in both "goodness of fit" and "error in currency" flavours:

    - **RMSE (₹)** — headline error in the target's units; penalises big misses quadratically
      (the business cares about expensive under/over-charges),
    - **MAE (₹)** — the typical error, robust to a few extreme fares,
    - **R²** — fraction of variance explained; scale-free, the fair *selection* metric across
      models on the same target,
    - **MAPE (%)** — relative error, handy for talking to non-technical stakeholders.

    → **Selection metric: R²** (scale-free), reported together with RMSE / MAE / MAPE.
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ### Pipeline architecture (everything inside the pipeline)

    ```
    raw row (11 features)
      └─ engineer      FunctionTransformer — fare_per_km, mins_per_km, wait_share_of_ride
      └─ prep          ColumnTransformer
      │   ├─ num       SimpleImputer(median) → StandardScaler   (7 raw + 3 engineered)
      │   └─ cat       SimpleImputer(most_frequent) → OneHotEncoder(handle_unknown="ignore")
      └─ clf           the regressor (model step)
      └─ predicted fare (₹)
    ```

    Because engineering and encoding live *inside* the pipeline, **inference needs only the raw
    inputs** — the pipeline re-derives features and rewrites categories on the spot.

    **Leakage note — why `surge_impact` is NOT engineered here.** EDA showed
    `booking_value ≈ base_fare × surge_multiplier` (r ≈ 0.9985): the fare is essentially the
    product of the two. Engineering `surge_impact = base_fare × surge_multiplier` would hand the
    model the answer, so the regression pipeline **omits it** and lets each learner (re)learn
    the interaction itself — that is what the baseline comparison actually tests.
    """)
    return


@app.cell
def _():
    train_csv = PROCESSED_DATA_DIR / "booking_value_train.csv"
    test_csv = PROCESSED_DATA_DIR / "booking_value_test.csv"
    model_file = "booking_value_model.joblib"
    return model_file, test_csv, train_csv


@app.cell
def _(test_csv, train_csv):
    train_full = pl.read_csv(train_csv)
    test_full = pl.read_csv(test_csv)
    print(
        f"train {train_full.shape[0]}x{train_full.shape[1]} | "
        f"test {test_full.shape[0]}x{test_full.shape[1]}"
    )
    train_full[BOOKING_VALUE_TARGET].describe().select(
        ["statistic", pl.col("value").round(2)]
    )
    return test_full, train_full


@app.cell
def _():
    mo.md(r"""
    **Design choice — holdout vs test.** The 80k training file is split again (random,
    20%) into *train* (64k) and *validation* (16k). Model selection and tuning use train + val;
    the separate `_test.csv` (20k) stays **untouched until the final generalization check**,
    so the reported generalization error is honest.
    """)
    return


@app.cell
def _(train_full):
    tr, val = train_test_split(
        train_full, test_size=0.2, random_state=42,
    )
    pl.DataFrame({"set": ["train", "valid"], "rows": [len(tr), len(val)]})
    return tr, val


@app.cell
def _():
    mo.md(r"""
    ### Engineered features (inside the pipeline)

    Three cheap, interpretable ride-efficiency features derived from the raw context columns,
    each `np.maximum(...)` guard keeping the transformer finite for any input.

    | Feature | Formula | Meaning |
    |---|---|---|
    | `fare_per_km` | `base_fare / max(dist, 0.1)` | price-per-km — higher = premium ride |
    | `mins_per_km` | `estimated_ride_time_min / max(dist, 0.1)` | ride "slowness" |
    | `wait_share_of_ride` | `avg_wait_time_min / max(est_min, 1)` | pick-up wait relative to ride time |

    `surge_impact` (the third classification-era feature) is intentionally **omitted** — see the
    leakage note above.
    """)
    return


@app.cell
def _(tr):
    eng_sample = engineer_booking_value_features(tr.head(5))[BOOKING_VALUE_ENG_FEATURES]
    prep_demo = booking_value_preprocessing()
    n_ml_features = prep_demo.fit_transform(
        engineer_booking_value_features(tr.head(500))
    ).shape[1]
    eng_sample
    return (n_ml_features,)


@app.cell
def _(n_ml_features):
    mo.md(f"""
    ### Baseline protocol

    Eight regressors, each wrapped by the same full pipeline. Slow models (SVR, Random Forest,
    HistGradientBoosting, XGBoost, CatBoost, LightGBM) fit on a **sample** of the training rows
    so the cell stays fast — flip the switch to run everything on the full 64k train.
    Every model is scored on the **full validation set** (16k rows) with the same metric battery.

    After engineering + encoding the pipeline feeds **{n_ml_features} features** to every learner.
    """)
    return


@app.cell
def _():
    BASELINES = {
        "SGDRegressor": lambda: SGDRegressor(
            max_iter=3000, tol=1e-3, penalty="elasticnet", random_state=42,
        ),
        "ElasticNet": lambda: ElasticNet(
            max_iter=3000, random_state=42,
        ),
        "SVR (RBF)": lambda: SVR(kernel="rbf", C=1.0),
        "RandomForest": lambda: RandomForestRegressor(
            n_estimators=200, n_jobs=-1, random_state=42,
        ),
        "HistGradientBoosting": lambda: HistGradientBoostingRegressor(
            random_state=42, max_iter=200,
        ),
        "XGBoost": lambda: XGBRegressor(
            n_estimators=200, n_jobs=-1, random_state=42, verbosity=0,
        ),
        "CatBoost": lambda: CatBoostRegressor(
            iterations=200, verbose=0, random_seed=42,
        ),
        "LightGBM": lambda: LGBMRegressor(
            n_estimators=200, n_jobs=-1, random_state=42, verbose=-1,
        ),
    }
    BASELINE_SAMPLE = {
        "SGDRegressor": 20_000,
        "ElasticNet": 20_000,
        "SVR (RBF)": 4_000,
        "RandomForest": 25_000,
        "HistGradientBoosting": 30_000,
        "XGBoost": 15_000,
        "CatBoost": 15_000,
        "LightGBM": 15_000,
    }
    return BASELINES, BASELINE_SAMPLE


@app.cell
def _():
    use_full = mo.ui.switch(
        value=False,
        label="Use FULL train data for all baselines (slower)",
    )
    use_full
    return (use_full,)


@app.cell
def _(BASELINES, BASELINE_SAMPLE, tr, use_full, val):
    baseline_rows = []
    baseline_fits = {}
    for _name, _cfg in BASELINES.items():
        _fit = tr if use_full.value else tr.sample(
            n=min(BASELINE_SAMPLE[_name], len(tr)), seed=42
        )
        _pipe = build_booking_value_pipeline(_cfg())
        _pipe.fit(_fit[BOOKING_VALUE_FEATURES], _fit[BOOKING_VALUE_TARGET])
        _yp = _pipe.predict(val[BOOKING_VALUE_FEATURES])
        baseline_rows.append(metrics_row_reg(_name, val[BOOKING_VALUE_TARGET], _yp))
        baseline_fits[_name] = _pipe
        logger.info(f"baseline done: {_name}")
    baseline_table = metrics_table_reg(baseline_rows)
    baseline_table
    return baseline_fits, baseline_table


@app.cell
def _(baseline_table):
    _r2 = metrics_bar_chart(baseline_table, "r2")
    _rmse = metrics_bar_chart(baseline_table, "rmse")
    mo.hstack([_r2, _rmse])
    return


@app.cell
def _():
    mo.md(r"""
    ### Reading the baseline table

    Check **`r2`** first (the scale-free selection metric), then **`rmse`/`mae`** (errors in ₹,
    where a good model should land well under ~₹100 typical error given the ₹335 mean) and
    **`mape`** for the relative view.

    Expectation: boosting/RF family (LightGBM / XGBoost / CatBoost / HistGradientBoosting /
    RandomForest) leads — the target is a **multiplicative interaction** (`base_fare × surge`),
    which trees approximate naturally — while linear SGD/ElasticNet will trail because a
    plain linear combination of `base_fare` and `surge_multiplier` cannot express the product.

    The next cell picks the winner automatically and fine-tunes **it**.
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ### Fine-tuning plan

    The best baseline is tuned with **RandomizedSearchCV** (`cv=3`, scoring `r2`). The search
    runs the **full pipeline** (engineering + preprocessing inside), tuning only the regressor
    step (`clf__…`).

    **Speed control (smoke tests):** the CV search fits on a **sample** of the train split by
    default (slider above) — the winner's hyper-parameters are then **refit on the full 64k
    train** when the switch is on, so the saved artifact uses all data while the search itself
    stays fast.
    """)
    return


@app.cell
def _():
    search_n_iter = mo.ui.slider(5, 60, value=15, step=5, label="RandomizedSearch iterations")
    search_cv = mo.ui.slider(2, 5, value=3, step=1, label="CV folds")
    tune_sample = mo.ui.slider(
        5_000, 40_000, value=20_000, step=5_000,
        label="Search sample rows (smoke-test speed)",
    )
    refit_full = mo.ui.switch(
        value=True,
        label="Refit tuned model on FULL train after the search",
    )
    mo.vstack([mo.hstack([search_n_iter, search_cv, tune_sample]), refit_full])
    return refit_full, search_cv, search_n_iter, tune_sample


@app.cell
def _():
    SEARCH_SPACES = {
        "SGDRegressor": {
            "clf__penalty": ["l2", "elasticnet"],
            "clf__alpha": loguniform(1e-5, 1e-1),
            "clf__l1_ratio": uniform(0, 1),
            "clf__learning_rate": ["invscaling", "adaptive"],
            "clf__eta0": loguniform(1e-4, 1e-1),
            "clf__max_iter": [2000, 4000],
        },
        "ElasticNet": {
            "clf__alpha": loguniform(1e-4, 1.0),
            "clf__l1_ratio": uniform(0, 1),
            "clf__max_iter": [5000],
        },
        "SVR (RBF)": {
            "clf__C": loguniform(1e-1, 1e2),
            "clf__gamma": ["scale", "auto"],
            "clf__epsilon": uniform(0.01, 0.6),
        },
        "RandomForest": {
            "clf__n_estimators": [200, 400, 600],
            "clf__max_depth": [None, 15, 30],
            "clf__min_samples_split": [2, 10, 50],
            "clf__min_samples_leaf": [1, 5, 20],
            "clf__max_features": ["sqrt", "log2", None],
        },
        "HistGradientBoosting": {
            "clf__learning_rate": loguniform(1e-3, 0.3),
            "clf__max_iter": [150, 250, 400],
            "clf__max_leaf_nodes": [15, 31, 63],
            "clf__max_depth": [None, 3, 8],
            "clf__l2_regularization": loguniform(1e-6, 10),
        },
        "XGBoost": {
            "clf__n_estimators": [100, 200, 400],
            "clf__learning_rate": loguniform(1e-3, 0.3),
            "clf__max_depth": [3, 6, 10],
            "clf__subsample": [0.7, 0.9, 1.0],
            "clf__colsample_bytree": [0.7, 0.9, 1.0],
            "clf__reg_lambda": loguniform(1e-3, 10),
        },
        "CatBoost": {
            "clf__iterations": [100, 200, 400],
            "clf__learning_rate": loguniform(1e-3, 0.3),
            "clf__depth": [4, 6, 8, 10],
            "clf__l2_leaf_reg": loguniform(1e-2, 10),
            "clf__bagging_temperature": [0.0, 1.0, 3.0],
        },
        "LightGBM": {
            "clf__n_estimators": [100, 200, 400],
            "clf__num_leaves": [15, 31, 63, 127],
            "clf__learning_rate": loguniform(1e-3, 0.3),
            "clf__subsample": [0.7, 1.0],
            "clf__colsample_bytree": [0.7, 1.0],
            "clf__reg_alpha": loguniform(1e-4, 10),
            "clf__min_child_samples": [10, 30, 60],
        },
    }
    return (SEARCH_SPACES,)


@app.cell
def _(baseline_table):
    best_name = str(baseline_table["model"][0])
    best_name
    return (best_name,)


@app.cell
def _(
    BASELINES,
    SEARCH_SPACES,
    best_name,
    refit_full,
    search_cv,
    search_n_iter,
    tr,
    tune_sample,
):
    _tune_tr = tr.sample(n=min(tune_sample.value, len(tr)), seed=42)
    best_search = RandomizedSearchCV(
        build_booking_value_pipeline(BASELINES[best_name]()),
        param_distributions=SEARCH_SPACES[best_name],
        n_iter=search_n_iter.value,
        cv=search_cv.value,
        scoring="r2",
        n_jobs=-1,
        random_state=42,
        refit=True,
        error_score="raise",
        verbose=0,
    )
    best_search.fit(_tune_tr[BOOKING_VALUE_FEATURES], _tune_tr[BOOKING_VALUE_TARGET])
    best_cv_score = round(float(best_search.best_score_), 4)
    best_params = dict(best_search.best_params_)
    if refit_full.value:
        best_pipe = clone(best_search.best_estimator_).fit(
            tr[BOOKING_VALUE_FEATURES], tr[BOOKING_VALUE_TARGET]
        )
    else:
        best_pipe = best_search.best_estimator_
    logger.info(
        f"tuned {best_name}: cv r2={best_cv_score}, refit_full={refit_full.value}"
    )
    best_pipe
    return best_cv_score, best_params, best_pipe


@app.cell
def _():
    mo.md(r"""
    ### Tuned vs baseline

    Compare **`r2`** (and RMSE/MAE in ₹) before → after tuning. A **licensed improvement**
    means the default hyper-parameters were leaving performance on the table; a flat or worse
    result means the defaults were already near-optimal and the CV search did not overfit.

    The tuned pipeline is saved next, together with its metadata.
    """)
    return


@app.cell
def _(baseline_fits, best_name, best_pipe, val):
    tuned_val = metrics_row_reg(
        f"{best_name} (tuned)",
        val[BOOKING_VALUE_TARGET],
        best_pipe.predict(val[BOOKING_VALUE_FEATURES]),
    )
    tuned_vs_baseline = metrics_table_reg(
        [
            metrics_row_reg(best_name, val[BOOKING_VALUE_TARGET], baseline_fits[best_name].predict(val[BOOKING_VALUE_FEATURES])),
            tuned_val,
        ]
    )
    tuned_vs_baseline
    return (tuned_val,)


@app.cell
def _():
    mo.md(r"""
    ### Save the tuned pipeline

    The winner is dumped with `joblib` and a sidecar `_meta.json` records its task, features,
    hyper-parameters, CV score and validation numbers — a self-describing artifact.
    """)
    return


@app.cell
def _(
    best_cv_score,
    best_name,
    best_params,
    best_pipe,
    model_file,
    search_cv,
    search_n_iter,
    tuned_val,
):
    _meta = {
        "project": "rapido_intelligent_system",
        "model": best_name,
        "task": "regression",
        "target": BOOKING_VALUE_TARGET,
        "features": BOOKING_VALUE_FEATURES,
        "engineered_features": list(BOOKING_VALUE_ENG_FEATURES),
        "best_params": {k.replace("clf__", "").replace("estimator__", ""): str(v) for k, v in best_params.items()},
        "cv_best_r2": best_cv_score,
        "val_rmse": tuned_val["rmse"],
        "val_mae": tuned_val["mae"],
        "val_r2": tuned_val["r2"],
        "val_mape": tuned_val["mape"],
        "search_n_iter": search_n_iter.value,
        "search_cv": search_cv.value,
    }
    model_path, meta_path = save_model(best_pipe, MODELS_DIR, model_file, _meta)
    (model_path, meta_path)
    return (model_path,)


@app.cell
def _():
    mo.md(r"""
    ### Generalisation check (on the untouched test split)

    The saved pipeline is **reloaded from disk** and scored on `booking_value_test.csv`
    (20k rows it never saw during selection or tuning). This is the honest estimate of how the
    model will behave on new bookings.
    """)
    return


@app.cell
def _(model_path, test_full):
    loaded_pipe = load_model(model_path)
    test_pred = loaded_pipe.predict(test_full[BOOKING_VALUE_FEATURES])
    test_metrics = metrics_row_reg("loaded on test", test_full[BOOKING_VALUE_TARGET], test_pred)
    test_metrics
    return loaded_pipe, test_metrics


@app.cell
def _(test_metrics):
    pl.DataFrame([test_metrics])
    return


@app.cell
def _():
    mo.md(r"""
    ### Reading the generalisation numbers

    - **`r2`** — should sit close to the validation R²; a big drop = overfit.
    - **`rmse`** — the headline business error (₹); on a ₹335 mean fare a two-digit RMSE is a
      solid model, a four-figure one is not.
    - **`mae`/`mape`** — the typical error and its relative share of the fare.

    Compare these to the validation numbers above — a small gap = healthy, a big one = overfit.
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ### Interactive prediction — raw inputs → fare

    The form below sends a **raw ride-context row** (nothing pre-processed) through the saved
    pipeline: engineering, imputation, scaling, one-hot encoding and the regressor all run
    *inside* the pipeline. Press **Predict** to see the estimated fare.
    """)
    return


@app.cell
def _(train_full):
    pickup_choices = train_full["pickup_location"].unique().sort().to_list()
    predict_form = mo.ui.form(
        mo.ui.dictionary(
            {
                "ride_distance_km": mo.ui.number(0.5, 60.0, value=8.0, step=0.5, label="Distance (km)"),
                "estimated_ride_time_min": mo.ui.number(2, 300, value=25, step=1, label="Est. ride time (min)"),
                "base_fare": mo.ui.number(20, 800, value=120, step=1, label="Base fare (₹)"),
                "surge_multiplier": mo.ui.number(1.0, 3.0, value=1.1, step=0.1, label="Surge multiplier"),
                "pickup_location": mo.ui.dropdown(pickup_choices, value=pickup_choices[0], label="Pickup location"),
                "traffic_level": mo.ui.dropdown(BOOKING_VALUE_CATEGORIES["traffic_level"], value="Low", label="Traffic"),
                "avg_wait_time_min": mo.ui.number(0, 120, value=8, step=1, label="Avg wait time (min)"),
                "avg_surge_multiplier": mo.ui.number(1.0, 3.0, value=1.5, step=0.05, label="Avg surge in area"),
                "vehicle_type": mo.ui.dropdown(BOOKING_VALUE_CATEGORIES["vehicle_type"], value="Bike", label="Vehicle"),
                "weather_condition": mo.ui.dropdown(BOOKING_VALUE_CATEGORIES["weather_condition"], value="Clear", label="Weather"),
                "hour_of_day": mo.ui.slider(0, 23, value=12, step=1, label="Hour of day"),
            }
        ),
        submit_button_label="Predict booking value",
    )
    predict_form
    return (predict_form,)


@app.cell
def _(loaded_pipe, predict_form):
    if predict_form.value:
        _row = pl.DataFrame([predict_form.value]).select(BOOKING_VALUE_FEATURES)
        _pred = float(loaded_pipe.predict(_row)[0])
        _input_view = _row.transpose(
            include_header=True, header_name="feature", column_names=["input"]
        )
        _ = mo.vstack(
            [
                mo.md(f"### **Predicted booking value: ₹ {_pred:,.2f}**"),
                _input_view,
            ]
        )
    else:
        _ = mo.md("Fill the form above and press **Predict**.")
    return


@app.cell
def _(best_name, test_metrics, tuned_val):
    mo.md(f"""
    ## Summary & next steps

    **Model built:** `{best_name}`, fine-tuned with randomized search (CV `r2`), saved as
    `models/booking_value_model.joblib` with metadata JSON alongside.

    **Final (test-set) metrics:**
    - R² `{test_metrics['r2']}` · RMSE `{test_metrics['rmse']}` ₹ · MAE `{test_metrics['mae']}` ₹
      · MAPE `{test_metrics['mape']}` (validation R² `{tuned_val['r2']}`).

    **What the pipeline owns** (fit AND inference): derived features (`fare_per_km`,
    `mins_per_km`, `wait_share_of_ride` — deliberately **not** `surge_impact`, which would leak
    the target), median/mode imputation, scaling, one-hot encoding — a raw ride row in, a fare out.

    **Next steps**
    1. Repeat this notebook for `customer_cancel_flag` and `driver_delay_flag` (binary
       classification), each in its own notebook.
    2. If deployed, pair the fare with the classification models (a cancellation or delay
       model can flag when the predicted fare is unreliable).
    """)
    return


if __name__ == "__main__":
    app.run()
