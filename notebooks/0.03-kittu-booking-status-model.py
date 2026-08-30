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
    from sklearn.linear_model import SGDClassifier, LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import (
        RandomForestClassifier, HistGradientBoostingClassifier,
    )
    from xgboost import XGBClassifier
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from loguru import logger
    from rapido_intelligent_system.config_mnb import (
        PROCESSED_DATA_DIR, MODELS_DIR,
    )
    from rapido_intelligent_system.modeling_mnb import (
        BOOKING_FEATURES, BOOKING_TARGET, BOOKING_ENG_FEATURES,
        BOOKING_CATEGORIES, TARGET_CLASSES,
        engineer_booking_features, booking_preprocessing,
        build_booking_pipeline, metrics_row, metrics_table,
        metrics_bar_chart, save_model, load_model,
        LabelsafeClassifier,
    )


@app.cell
def _():
    mo.md(r"""
    # 0.03 — booking_status Model (Multi-class)

    **Project:** Rapido Intelligent System · **Author:** kittu
    **Goal:** predict `booking_status` (Completed / Cancelled / Incomplete) from the raw
    booking-context features selected in notebook 0.02.

    ### Workflow in this notebook
    1. **Holdout** — a stratified validation set is carved from the training file; the
       untouched `data/processed/booking_status_test.csv` is kept for a **final generalization
       check only**.
    2. **Pipelines** — every experiment is one scikit-learn `Pipeline` that does **feature
       engineering → impute → scale / one-hot → classifier**, so the same object serves
       training *and* inference on raw inputs.
    3. **Baselines** — SGD, softmax (logistic regression), SVC, Random Forest,
       HistGradientBoosting, XGBoost, CatBoost and LightGBM. Slow learners get a **sample**
       of the data so cells pass the smoke test fast.
    4. **Select** the best baseline (metric chosen for the class imbalance), **fine-tune** it
       with randomized-search CV, **save** it, reload it, and estimate generalisation on the test split.
    5. **Interactive UI** — input ride context and see a prediction from the saved pipeline.
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ### Why these metrics?

    `booking_status` is **imbalanced multi-class**: Completed ≈ 68%, Cancelled ≈ 23%,
    Incomplete ≈ 8%. A model that always says "Completed" hits ~68% accuracy, so:

    - **accuracy / micro-F1** are dominated by the majority class (micro ≡ accuracy here),
    - **macro-F1** is brutally harsh: a struggling 8% class drags it down equally,
    - **weighted-F1** balances prevalence — the fair headline for *deployment*,
    - **macro-F1** is kept alongside as the *per-class-capability* counterpoint.

    → **Selection metric: weighted-F1**, reported together with accuracy and macro-F1 for skew sanity.
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ### Pipeline architecture (everything inside the pipeline)

    ```
    raw row (11 features)
      └─ engineer      FunctionTransformer  — fare_per_km, surge_impact, mins_per_km, wait_share_of_ride
      └─ prep          ColumnTransformer
      │   ├─ num       SimpleImputer(median) → StandardScaler   (6 raw + 4 engineered)
      │   └─ cat       SimpleImputer(most_frequent) → OneHotEncoder(handle_unknown="ignore")
      └─ clf           the classifier (model step)
      └─ prediction label + probabilities
    ```

    Because engineering and encoding live *inside* the pipeline, **inference needs only the raw
    inputs** — the pipeline re-derives features and rewrites categories on the spot.
    """)
    return


@app.cell
def _():
    train_csv = PROCESSED_DATA_DIR / "booking_status_train.csv"
    test_csv = PROCESSED_DATA_DIR / "booking_status_test.csv"
    model_file = "booking_status_model.joblib"
    model_path_t = MODELS_DIR / model_file
    return model_file, test_csv, train_csv


@app.cell
def _(test_csv, train_csv):
    train_full = pl.read_csv(train_csv)
    test_full = pl.read_csv(test_csv)
    balance = (
        train_full.group_by(BOOKING_TARGET)
        .len()
        .sort(BOOKING_TARGET)
        .with_columns((pl.col("len") / train_full.height * 100).round(1).alias("share_%"))
        .drop("len")
    )
    print(
        f"train {train_full.shape[0]}x{train_full.shape[1]} | "
        f"test {test_full.shape[0]}x{test_full.shape[1]}"
    )
    balance
    return test_full, train_full


@app.cell
def _():
    mo.md(r"""
    **Design choice — holdout vs test.** The 80k training file is split again (stratified)
    into *train* (64k) and *validation* (16k). Model selection and tuning use train + val;
    the separate `_test.csv` (20k) is **never touched until the final generalization check**,
    so the reported generalization error is honest.
    """)
    return


@app.cell
def _(train_full):
    X_cols = BOOKING_FEATURES
    tr, val = train_test_split(
        train_full, test_size=0.2,
        stratify=train_full[BOOKING_TARGET],
        random_state=42,
    )
    split_balance = pl.DataFrame(
        {
            "set": ["train", "valid"],
            "rows": [len(tr), len(val)],
            **{
                f"{_cls}%": [
                    round(100 * (tr[BOOKING_TARGET] == _cls).mean(), 1),
                    round(100 * (val[BOOKING_TARGET] == _cls).mean(), 1),
                ]
                for _cls in ["Completed", "Cancelled", "Incomplete"]
            },
        }
    )
    split_balance
    return tr, val


@app.cell
def _():
    mo.md(r"""
    ### Engineered features (inside the pipeline)

    Four cheap, interpretable ride-efficiency features derived from the raw context columns.
    Each `np.maximum(...)` guard keeps the transformer finite for any input:
    `÷` guarded by a small floor, no zero-division, no NaN explosions.

    | Feature | Formula | Meaning |
    |---|---|---|
    | `fare_per_km` | `base_fare / max(dist, 0.1)` | price-per-km — higher = premium ride |
    | `surge_impact` | `base_fare × surge_multiplier` | effective surge price |
    | `mins_per_km` | `estimated_ride_time_min / max(dist, 0.1)` | ride "slowness" |
    | `wait_share_of_ride` | `avg_wait_time_min / max(est_min, 1)` | pick-up wait relative to ride time |
    """)
    return


@app.cell
def _(tr):
    eng_sample = engineer_booking_features(tr.head(5))[BOOKING_ENG_FEATURES]
    prep_demo = booking_preprocessing()
    n_ml_features = prep_demo.fit_transform(
        engineer_booking_features(tr.head(500))
    ).shape[1]
    eng_sample
    return (n_ml_features,)


@app.cell
def _(n_ml_features):
    mo.md(f"""
    ### Baseline protocol

    Eight learners, each wrapped by the same full pipeline. Slow models (SVC, XGBoost,
    CatBoost, LightGBM, Random Forest) fit on a **sample** of the training rows so the cell
    stays fast — flip the switch to run everything on the full 64k train. Every model is
    scored on the **full validation set** (16k rows) with the same metric battery.

    After engineering + encoding the pipeline feeds **{n_ml_features} features** to every learner.
    """)
    return


@app.cell
def _():
    BASELINES = {
        "SGDClassifier (softmax)": lambda: SGDClassifier(
            loss="log_loss", max_iter=2000, tol=1e-3, random_state=42,
        ),
        "LogisticRegression (softmax)": lambda: LogisticRegression(
            solver="lbfgs", max_iter=2000, random_state=42,
        ),
        "SVC (RBF)": lambda: SVC(kernel="rbf", C=1.0, random_state=42),
        "RandomForest": lambda: RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=42,
        ),
        "HistGradientBoosting": lambda: HistGradientBoostingClassifier(
            random_state=42, max_iter=200,
        ),
        "XGBoost": lambda: LabelsafeClassifier(
            estimator=XGBClassifier(
                n_estimators=200, n_jobs=-1, random_state=42,
                eval_metric="mlogloss", verbosity=0,
            ),
        ),
        "CatBoost": lambda: CatBoostClassifier(
            iterations=200, verbose=0, random_seed=42,
        ),
        "LightGBM": lambda: LGBMClassifier(
            n_estimators=200, n_jobs=-1, random_state=42, verbose=-1,
        ),
    }
    BASELINE_SAMPLE = {
        "SGDClassifier (softmax)": 20_000,
        "LogisticRegression (softmax)": 20_000,
        "SVC (RBF)": 4_000,
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
        _pipe = build_booking_pipeline(_cfg())
        _pipe.fit(_fit[BOOKING_FEATURES], _fit[BOOKING_TARGET])
        _yp = _pipe.predict(val[BOOKING_FEATURES])
        baseline_rows.append(metrics_row(_name, val[BOOKING_TARGET], _yp))
        baseline_fits[_name] = _pipe
        logger.info(f"baseline done: {_name}")
    baseline_table = metrics_table(baseline_rows)
    baseline_table
    return baseline_fits, baseline_table


@app.cell
def _(baseline_table):
    _f1c = metrics_bar_chart(baseline_table, "f1_weighted")
    _mac = metrics_bar_chart(baseline_table, "f1_macro")
    mo.hstack([_f1c, _mac])
    return


@app.cell
def _():
    mo.md(r"""
    ### Reading the baseline table

    Check the **`f1_weighted`** column first (the deployment-headline), then **`f1_macro`**
    (per-class fairness) and the **accuracy** column for the majority-class baseline.

    Expectation: the boosting family (LightGBM / XGBoost / HistGradientBoosting) leads on
    weighted-F1 with Random Forest close behind; linear models (SGD, softmax) trail because
    the target separates on **non-linear interactions** (surge × distance × traffic). SVC is
    the wild-card — it can be strong but is the most sensitive to scaling and sample size.

    The next cell picks the winner automatically and fine-tunes **it**.
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ### Fine-tuning plan

    The best baseline is tuned with **RandomizedSearchCV** (`cv=3`, scoring `f1_weighted`).
    The search runs the **full pipeline** (engineering + preprocessing inside), tuning only the
    classifier step (`clf__…`).

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
        "SGDClassifier (softmax)": {
            "clf__loss": ["log_loss", "modified_huber"],
            "clf__alpha": loguniform(1e-5, 1e-1),
            "clf__l1_ratio": uniform(0, 1),
            "clf__learning_rate": ["optimal", "invscaling", "adaptive"],
            "clf__max_iter": [1000, 3000],
        },
        "LogisticRegression (softmax)": {
            "clf__C": loguniform(1e-3, 1e2),
            "clf__class_weight": [None, "balanced"],
            "clf__max_iter": [2000],
        },
        "SVC (RBF)": {
            "clf__C": loguniform(1e-1, 1e2),
            "clf__gamma": ["scale", "auto"],
            "clf__kernel": ["rbf", "linear"],
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
            "clf__max_iter": [100, 150, 250, 400],
            "clf__max_leaf_nodes": [15, 31, 63],
            "clf__max_depth": [None, 3, 8],
            "clf__l2_regularization": loguniform(1e-6, 10),
        },
        "XGBoost": {
            "clf__estimator__n_estimators": [100, 200, 400],
            "clf__estimator__learning_rate": loguniform(1e-3, 0.3),
            "clf__estimator__max_depth": [3, 6, 10],
            "clf__estimator__subsample": [0.7, 0.9, 1.0],
            "clf__estimator__colsample_bytree": [0.7, 0.9, 1.0],
            "clf__estimator__reg_lambda": loguniform(1e-3, 10),
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
        build_booking_pipeline(BASELINES[best_name]()),
        param_distributions=SEARCH_SPACES[best_name],
        n_iter=search_n_iter.value,
        cv=search_cv.value,
        scoring="f1_weighted",
        n_jobs=-1,
        random_state=42,
        refit=True,
        error_score="raise",
        verbose=0,
    )
    best_search.fit(_tune_tr[BOOKING_FEATURES], _tune_tr[BOOKING_TARGET])
    best_cv_score = round(float(best_search.best_score_), 4)
    best_params = dict(best_search.best_params_)
    if refit_full.value:
        best_pipe = clone(best_search.best_estimator_).fit(
            tr[BOOKING_FEATURES], tr[BOOKING_TARGET]
        )
    else:
        best_pipe = best_search.best_estimator_
    logger.info(
        f"tuned {best_name}: cv f1_weighted={best_cv_score}, refit_full={refit_full.value}"
    )
    best_pipe
    return best_cv_score, best_params, best_pipe


@app.cell
def _(baseline_fits, best_name, best_pipe, val):
    tuned_val = metrics_row(
        f"{best_name} (tuned)",
        val[BOOKING_TARGET],
        best_pipe.predict(val[BOOKING_FEATURES]),
    )
    tuned_vs_baseline = metrics_table(
        [
            metrics_row(best_name, val[BOOKING_TARGET], baseline_fits[best_name].predict(val[BOOKING_FEATURES])),
            tuned_val,
        ]
    )
    tuned_vs_baseline
    return (tuned_val,)


@app.cell
def _():
    mo.md(r"""
    ### Tuned vs baseline

    Compare `f1_weighted` (and macro-F1) before → after tuning. A **licensed improvement**
    means the default hyper-parameters were leaving performance on the table; a flat or worse
    result means the defaults were already near-optimal and the CV search did not overfit.

    The tuned pipeline is saved next, together with its metadata.
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
        "task": "multi-class classification",
        "target": BOOKING_TARGET,
        "classes": list(best_pipe.classes_),
        "features": BOOKING_FEATURES,
        "engineered_features": ["fare_per_km", "surge_impact", "mins_per_km", "wait_share_of_ride"],
        "best_params": {k.replace("clf__", "").replace("estimator__", ""): str(v) for k, v in best_params.items()},
        "cv_best_f1_weighted": best_cv_score,
        "val_f1_weighted": tuned_val["f1_weighted"],
        "val_f1_macro": tuned_val["f1_macro"],
        "val_accuracy": tuned_val["accuracy"],
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

    The saved pipeline is **reloaded from disk** and scored on `booking_status_test.csv`
    (20k rows it never saw during selection or tuning). This is the honest estimate of how the
    model will behave on new bookings.
    """)
    return


@app.cell
def _(model_path, test_full):
    loaded_pipe = load_model(model_path)
    test_pred = loaded_pipe.predict(test_full[BOOKING_FEATURES])
    test_metrics = metrics_row("loaded on test", test_full[BOOKING_TARGET], test_pred)
    confusion = (
        pl.DataFrame({"true": test_full[BOOKING_TARGET], "pred": test_pred})
        .group_by(["true", "pred"])
        .len()
        .join(
            pl.DataFrame({"true": TARGET_CLASSES}).join(
                pl.DataFrame({"pred": TARGET_CLASSES}), how="cross"
            ),
            on=["true", "pred"],
            how="left",
        )
        .fill_null(0)
        .rename({"len": "count"})
        .pivot(on="pred", index="true", values="count")
    )
    confusion
    return loaded_pipe, test_metrics


@app.cell
def _(test_metrics):
    val_test_view = pl.DataFrame([test_metrics])
    val_test_view
    return


@app.cell
def _():
    mo.md(r"""
    ### Reading the generalisation numbers

    - **accuracy** — beats the 68% majority-class baseline (translate: tonnes better than
      always predicting "Completed").
    - **`f1_weighted`** — the headline; sized for the real class mix.
    - **`f1_macro`** — the canary for the 8% Incomplete class. If it is much lower than
      weighted-F1, the model is weak at rare Incompletes; that is the expected price of rarity,
      but it is the metric to watch if Incomplete prediction becomes a business priority.

    Compare these to the validation numbers above — a small gap = healthy, a big one = overfit.
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ### Interactive prediction — raw inputs → label

    The form below sends a **raw ride-context row** (nothing pre-processed) through the saved
    pipeline: engineering, imputation, scaling, one-hot encoding and the classifier all run
    *inside* the pipeline. Press **Predict** to see the label and class probabilities.
    """)
    return


@app.cell
def _():
    predict_form = mo.ui.form(
        mo.ui.dictionary(
            {
                "ride_distance_km": mo.ui.number(0.5, 40.0, value=6.0, step=0.5, label="Distance (km)"),
                "estimated_ride_time_min": mo.ui.number(2, 180, value=18, step=1, label="Est. ride time (min)"),
                "base_fare": mo.ui.number(20, 600, value=75, step=1, label="Base fare (₹)"),
                "surge_multiplier": mo.ui.number(1.0, 3.0, value=1.1, step=0.1, label="Surge multiplier"),
                "avg_wait_time_min": mo.ui.number(0, 120, value=5, step=1, label="Avg wait time (min)"),
                "avg_surge_multiplier": mo.ui.number(1.0, 3.0, value=1.5, step=0.05, label="Avg surge in area"),
                "traffic_level": mo.ui.dropdown(BOOKING_CATEGORIES["traffic_level"], value="Medium", label="Traffic"),
                "weather_condition": mo.ui.dropdown(BOOKING_CATEGORIES["weather_condition"], value="Clear", label="Weather"),
                "demand_level": mo.ui.dropdown(BOOKING_CATEGORIES["demand_level"], value="Medium", label="Demand"),
                "season": mo.ui.dropdown(BOOKING_CATEGORIES["season"], value="Summer", label="Season"),
                "vehicle_type": mo.ui.dropdown(BOOKING_CATEGORIES["vehicle_type"], value="Bike", label="Vehicle"),
            }
        ),
        submit_button_label="Predict booking status",
    )
    predict_form
    return (predict_form,)


@app.cell
def _(loaded_pipe, predict_form):
    if predict_form.value:
        _row = pl.DataFrame([predict_form.value]).select(BOOKING_FEATURES)
        _pred_cls = str(loaded_pipe.predict(_row)[0])
        _prob_frame = pl.DataFrame(
            {
                "class": list(loaded_pipe.classes_),
                "probability": loaded_pipe.predict_proba(_row)[0],
            }
        )
        _prob_chart = (
            alt.Chart(_prob_frame, title="Class probabilities")
            .mark_bar()
            .encode(
                x=alt.X("class:N", sort=list(loaded_pipe.classes_)),
                y=alt.Y("probability:Q", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("class:N", legend=None),
                tooltip=["class", "probability"],
            )
            .properties(width=420, height=240)
        )
        _ = mo.vstack(
            [
                mo.md(f"### **Prediction: {_pred_cls}**"),
                mo.hstack([_prob_chart, _prob_frame]),
            ]
        )
    else:
        _ = mo.md("Fill the form above and press **Predict**.")
    return


@app.cell
def _(best_name, test_metrics, tuned_val):
    mo.md(f"""
    ## Summary & next steps

    **Model built:** `{best_name}`, fine-tuned with randomized search (CV `f1_weighted`),
    saved as `models/booking_status_model.joblib` with metadata JSON alongside.

    **Final (test-set) metrics:**
    - accuracy `{test_metrics['accuracy']}` · weighted-F1 `{test_metrics['f1_weighted']}`
      · macro-F1 `{test_metrics['f1_macro']}` (validation weighted-F1 `{tuned_val['f1_weighted']}`).

    Classes (in saved model order): {', '.join(TARGET_CLASSES)}.

    **What the pipeline owns** (fit AND inference): derived features
    (`fare_per_km`, `surge_impact`, `mins_per_km`, `wait_share_of_ride`), median/mode imputation,
    scaling, one-hot encoding — a raw ride row in, a label out.

    **Next steps**
    1. Repeat this notebook for `booking_value` (regression), `customer_cancel_flag` and
       `driver_delay_flag`, each in its own notebook.
    2. If Incomplete/F1-macro matters, switch selection to **macro-F1** or add class weighting
       in the tuned pipeline (see the search space — `class_weight` is already exposed).
    3. Consider a **probability threshold / business rule** layer on top of `predict_proba`
       (e.g. flag uncertain predictions for manual handling).
    """)
    return


if __name__ == "__main__":
    app.run()
