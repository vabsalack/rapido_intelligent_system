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
        DRIVER_FEATURES, DRIVER_TARGET, DRIVER_ENG_FEATURES,
        DRIVER_CATEGORIES, DRIVER_BINARY_CLASSES,
        engineer_driver_features, driver_preprocessing,
        build_driver_pipeline, metrics_row_bin, metrics_table_bin,
        metrics_bar_chart, save_model, load_model,
    )


@app.cell
def _():
    mo.md(r"""
    # 0.06 — driver_delay_flag Model (Binary Classification)

    **Project:** Rapido Intelligent System · **Author:** kittu
    **Goal:** predict whether a driver's ride is delayed (`driver_delay_flag` ∈ {0, 1}) from
    the driver-profile features selected in notebook 0.02.

    ### Workflow in this notebook
    1. **Holdout** — a stratified validation set is carved from the training file; the
       untouched `data/processed/driver_delay_flag_test.csv` is kept for a **final
       generalization check only**.
    2. **Pipelines** — every experiment is one scikit-learn `Pipeline` that does **feature
       engineering → impute → scale / one-hot → classifier**, so the same object serves
       training *and* inference on raw inputs.
    3. **Baselines** — SGD, logistic regression, SVC, Random Forest, HistGradientBoosting,
       XGBoost, CatBoost and LightGBM. Slow learners get a **sample** of the data so cells
       pass the smoke test fast.
    4. **Select** the best baseline, **fine-tune** it with randomized-search CV, **save** it,
       reload it, and estimate generalisation on the test split.
    5. **Interactive UI** — input driver attributes and see a prediction from the saved pipeline.

    Data is small (4k train / 1k test) and **imbalanced**: delay ≈ 13%, no-delay ≈ 87%.
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ### Why these metrics?

    Binary but **imbalanced** (87/13) — that changes the metric story:

    - **accuracy** — a naive "always no-delay" model already hits ~87%, so accuracy alone
      flatters; it is reported only as context,
    - **recall (pos)** — of true delays, how many we catch (do we miss delays?),
    - **precision (pos)** — of flagged delays, how many really happen (do we cry wolf?),
    - **F1 (pos)** — the harmonic mean; the **selection metric** for the minority class, where
      precision and recall must both be considered.

    → **Selection metric: F1 of the delayed class**, reported with recall, precision and accuracy.
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ### Pipeline architecture (everything inside the pipeline)

    ```
    raw row (10 features)
      └─ engineer      FunctionTransformer — incomplete_share, delay_x_incomplete, experience_share
      └─ prep          ColumnTransformer
      │   ├─ num       SimpleImputer(median) → StandardScaler   (8 raw + 3 engineered)
      │   └─ cat       SimpleImputer(most_frequent) → OneHotEncoder(handle_unknown="ignore")
      └─ clf           the classifier (model step)
      └─ prediction label + class probabilities
    ```

    Because engineering and encoding live *inside* the pipeline, **inference needs only the raw
    inputs** — the pipeline re-derives features and rewrites categories on the spot.
    """)
    return


@app.cell
def _():
    train_csv = PROCESSED_DATA_DIR / "driver_delay_flag_train.csv"
    test_csv = PROCESSED_DATA_DIR / "driver_delay_flag_test.csv"
    model_file = "driver_delay_flag_model.joblib"
    return model_file, test_csv, train_csv


@app.cell
def _(test_csv, train_csv):
    train_full = pl.read_csv(train_csv)
    test_full = pl.read_csv(test_csv)
    balance = (
        train_full.group_by(DRIVER_TARGET)
        .len()
        .sort(DRIVER_TARGET)
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
    **Design choice — holdout vs test.** The 4k training file is split again (stratified on the
    target) into *train* (3.2k) and *validation* (0.8k). Model selection and tuning use train +
    val; the separate `_test.csv` (1k) is **never touched until the final generalization check**,
    so the reported generalization error is honest. Resampling/class-weighting is *not* applied
    up front — the baselines stay realistic and F1 selection keeps the minority class honest.
    """)
    return


@app.cell
def _(train_full):
    tr, val = train_test_split(
        train_full, test_size=0.2,
        stratify=train_full[DRIVER_TARGET],
        random_state=42,
    )
    split_balance = pl.DataFrame(
        {
            "set": ["train", "valid"],
            "rows": [len(tr), len(val)],
            **{
                f"class_{_c}%": [
                    round(100 * (tr[DRIVER_TARGET] == _c).mean(), 1),
                    round(100 * (val[DRIVER_TARGET] == _c).mean(), 1),
                ]
                for _c in [0, 1]
            },
        }
    )
    split_balance
    return tr, val


@app.cell
def _():
    mo.md(r"""
    ### Engineered features (inside the pipeline)

    Three cheap, interpretable driver-behaviour features derived from the raw profile columns,
    guided by the strongest univariate signals (`incomplete_rides` r≈0.48, `avg_pickup_delay_min`
    r≈0.37 vs the flag). Each `clip` guard keeps the transformer finite for any input.

    | Feature | Formula | Meaning |
    |---|---|---|
    | `incomplete_share` | `incomplete_rides / max(assigned, 1)` | share of assigned rides left incomplete |
    | `delay_x_incomplete` | `avg_pickup_delay_min × max(incomplete, 0)` | delay burden weighted by track record |
    | `experience_share` | `experience_years / max(age, 1)` | experience as a fraction of life |
    """)
    return


@app.cell
def _(tr):
    eng_sample = engineer_driver_features(tr.head(5))[DRIVER_ENG_FEATURES]
    prep_demo = driver_preprocessing()
    n_ml_features = prep_demo.fit_transform(
        engineer_driver_features(tr.head(500))
    ).shape[1]
    eng_sample
    return (n_ml_features,)


@app.cell
def _(n_ml_features):
    mo.md(f"""
    ### Baseline protocol

    Eight learners, each wrapped by the same full pipeline. Slow models (SVC, Random Forest,
    HistGradientBoosting, XGBoost, CatBoost, LightGBM) fit on a **sample** of the training rows
    so the cell runs fast — flip the switch to use the full 3.2k train. Every model is scored on
    the **full validation set** (0.8k rows) with the same metric battery.

    After engineering + encoding the pipeline feeds **{n_ml_features} features** to every learner.
    """)
    return


@app.cell
def _():
    BASELINES = {
        "SGDClassifier (softmax)": lambda: SGDClassifier(
            loss="log_loss", max_iter=2000, tol=1e-3, random_state=42,
        ),
        "LogisticRegression": lambda: LogisticRegression(
            solver="lbfgs", max_iter=2000, random_state=42,
        ),
        "SVC (RBF)": lambda: SVC(kernel="rbf", C=1.0, random_state=42),
        "RandomForest": lambda: RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=42,
        ),
        "HistGradientBoosting": lambda: HistGradientBoostingClassifier(
            random_state=42, max_iter=200,
        ),
        "XGBoost": lambda: XGBClassifier(
            n_estimators=200, n_jobs=-1, random_state=42,
            eval_metric="logloss", verbosity=0,
        ),
        "CatBoost": lambda: CatBoostClassifier(
            iterations=200, verbose=0, random_seed=42,
        ),
        "LightGBM": lambda: LGBMClassifier(
            n_estimators=200, n_jobs=-1, random_state=42, verbose=-1,
        ),
    }
    BASELINE_SAMPLE = {
        "SGDClassifier (softmax)": 3_000,
        "LogisticRegression": 3_000,
        "SVC (RBF)": 2_000,
        "RandomForest": 3_000,
        "HistGradientBoosting": 3_000,
        "XGBoost": 3_000,
        "CatBoost": 3_000,
        "LightGBM": 3_000,
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
        _pipe = build_driver_pipeline(_cfg())
        _pipe.fit(_fit[DRIVER_FEATURES], _fit[DRIVER_TARGET])
        _yp = _pipe.predict(val[DRIVER_FEATURES])
        baseline_rows.append(metrics_row_bin(_name, val[DRIVER_TARGET], _yp))
        baseline_fits[_name] = _pipe
        logger.info(f"baseline done: {_name}")
    baseline_table = metrics_table_bin(baseline_rows)
    baseline_table
    return baseline_fits, baseline_table


@app.cell
def _(baseline_table):
    _f1 = metrics_bar_chart(baseline_table, "f1")
    _rec = metrics_bar_chart(baseline_table, "recall")
    _prec = metrics_bar_chart(baseline_table, "precision")
    mo.hstack([_f1, _rec, _prec])
    return


@app.cell
def _():
    mo.md(r"""
    ### Reading the baseline table

    Check **`f1`** first (the selection metric — it balances catching delays and not crying
    wolf), then **`recall`** (do we miss real delays?) and **`precision`** (are our flags
    trustworthy?). **`accuracy`** should be read against the ~87% always-no-delay baseline.

    Expectation: the boosting/tree family (LightGBM / XGBoost / HistGradientBoosting / CatBoost)
    leads — the delayed class lives on **non-linear interactions** (`incomplete_rides` ×
    `avg_pickup_delay_min`), which trees model naturally — while linear SGD/LR settle for
    conservative, high-precision/low-recall predictions on the minority class.

    The next cell picks the winner automatically and fine-tunes **it**.
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ### Fine-tuning plan

    The best baseline is tuned with **RandomizedSearchCV** (`cv=3`, scoring `f1`). The search
    runs the **full pipeline** (engineering + preprocessing inside), tuning only the classifier
    step (`clf__…`).

    **Speed control (smoke tests):** the CV search fits on a **sample** of the train split by
    default (slider above) — the winner's hyper-parameters are then **refit on the full 3.2k
    train** when the switch is on, so the saved artifact uses all data while the search itself
    stays fast.
    """)
    return


@app.cell
def _():
    search_n_iter = mo.ui.slider(5, 60, value=15, step=5, label="RandomizedSearch iterations")
    search_cv = mo.ui.slider(2, 5, value=3, step=1, label="CV folds")
    tune_sample = mo.ui.slider(
        1_000, 3_200, value=2_500, step=100,
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
        "LogisticRegression": {
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
        build_driver_pipeline(BASELINES[best_name]()),
        param_distributions=SEARCH_SPACES[best_name],
        n_iter=search_n_iter.value,
        cv=search_cv.value,
        scoring="f1",
        n_jobs=-1,
        random_state=42,
        refit=True,
        error_score="raise",
        verbose=0,
    )
    best_search.fit(_tune_tr[DRIVER_FEATURES], _tune_tr[DRIVER_TARGET])
    best_cv_score = round(float(best_search.best_score_), 4)
    best_params = dict(best_search.best_params_)
    if refit_full.value:
        best_pipe = clone(best_search.best_estimator_).fit(
            tr[DRIVER_FEATURES], tr[DRIVER_TARGET]
        )
    else:
        best_pipe = best_search.best_estimator_
    logger.info(
        f"tuned {best_name}: cv f1={best_cv_score}, refit_full={refit_full.value}"
    )
    best_pipe
    return best_cv_score, best_params, best_pipe


@app.cell
def _():
    mo.md(r"""
    ### Tuned vs baseline

    Compare **F1** (and recall/precision) before → after tuning. A **licensed improvement**
    means the default hyper-parameters were leaving performance on the table; a flat or worse
    result means the defaults were already near-optimal and the CV search did not overfit.

    The tuned pipeline is saved next, together with its metadata.
    """)
    return


@app.cell
def _(baseline_fits, best_name, best_pipe, val):
    tuned_val = metrics_row_bin(
        f"{best_name} (tuned)",
        val[DRIVER_TARGET],
        best_pipe.predict(val[DRIVER_FEATURES]),
    )
    tuned_vs_baseline = metrics_table_bin(
        [
            metrics_row_bin(best_name, val[DRIVER_TARGET], baseline_fits[best_name].predict(val[DRIVER_FEATURES])),
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
        "task": "binary classification",
        "target": DRIVER_TARGET,
        "classes": [0, 1],
        "features": DRIVER_FEATURES,
        "engineered_features": list(DRIVER_ENG_FEATURES),
        "best_params": {k.replace("clf__", "").replace("estimator__", ""): str(v) for k, v in best_params.items()},
        "cv_best_f1": best_cv_score,
        "val_f1": tuned_val["f1"],
        "val_precision": tuned_val["precision"],
        "val_recall": tuned_val["recall"],
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

    The saved pipeline is **reloaded from disk** and scored on `driver_delay_flag_test.csv`
    (1k rows it never saw during selection or tuning). This is the honest estimate of how the
    model will behave on new drivers.
    """)
    return


@app.cell
def _(model_path, test_full):
    loaded_pipe = load_model(model_path)
    test_pred = loaded_pipe.predict(test_full[DRIVER_FEATURES])
    test_metrics = metrics_row_bin("loaded on test", test_full[DRIVER_TARGET], test_pred)
    confusion = (
        pl.DataFrame({"true": test_full[DRIVER_TARGET], "pred": test_pred})
        .group_by(["true", "pred"])
        .len()
        .join(
            pl.DataFrame({"true": [0, 1]}).join(
                pl.DataFrame({"pred": [0, 1]}), how="cross"
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
    val_view = pl.DataFrame([test_metrics])
    val_view
    return


@app.cell
def _():
    mo.md(r"""
    ### Reading the generalisation numbers

    - **`f1`** — the headline for catching delays accurately on the minority class.
    - **`recall`** — fraction of real delays flagged; low recall = we set late to warnings.
    - **`precision`** — fraction of flags that are real delays; low precision = too many
      false alerts. The business can trade them by moving the decision threshold.
    - **`accuracy`** — context only; remember the ~87% majority-class baseline.

    Compare these to the validation numbers above — a small gap = healthy, a big one = overfit.
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ### Interactive prediction — raw inputs → label

    The form below sends a **raw driver-profile row** (nothing pre-processed) through the saved
    pipeline: engineering, imputation, scaling, one-hot encoding and the classifier all run
    *inside* the pipeline. Press **Predict** to see the label and class probabilities.
    """)
    return


@app.cell
def _():
    predict_form = mo.ui.form(
        mo.ui.dictionary(
            {
                "driver_age": mo.ui.number(22, 54, value=38, step=1, label="Driver age"),
                "driver_city": mo.ui.dropdown(DRIVER_CATEGORIES["driver_city"], value="Bangalore", label="City"),
                "vehicle_type": mo.ui.dropdown(DRIVER_CATEGORIES["vehicle_type"], value="Bike", label="Vehicle"),
                "driver_experience_years": mo.ui.number(1, 14, value=7, step=1, label="Experience (years)"),
                "total_assigned_rides": mo.ui.number(6, 36, value=20, step=1, label="Assigned rides"),
                "accepted_rides": mo.ui.number(0, 36, value=15, step=1, label="Accepted rides"),
                "incomplete_rides": mo.ui.number(0, 7, value=1, step=1, label="Incomplete rides"),
                "acceptance_rate": mo.ui.number(0.31, 1.0, value=0.77, step=0.01, label="Acceptance rate"),
                "avg_driver_rating": mo.ui.number(4.0, 5.0, value=4.5, step=0.1, label="Avg rating"),
                "avg_pickup_delay_min": mo.ui.number(1.0, 7.9, value=3.0, step=0.1, label="Avg pickup delay (min)"),
            }
        ),
        submit_button_label="Predict delay",
    )
    predict_form
    return (predict_form,)


@app.cell
def _(DRIVER_FEATURES, loaded_pipe, predict_form):
    if predict_form.value:
        _row = pl.DataFrame([predict_form.value]).select(DRIVER_FEATURES)
        _pred = int(loaded_pipe.predict(_row)[0])
        _pred_label = DRIVER_BINARY_CLASSES[_pred]
        _prob_frame = pl.DataFrame(
            {"class": list(loaded_pipe.classes_), "probability": loaded_pipe.predict_proba(_row)[0]}
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
                mo.md(f"### **Prediction: {_pred_label}**"),
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

    **Model built:** `{best_name}`, fine-tuned with randomized search (CV `f1`), saved as
    `models/driver_delay_flag_model.joblib` with metadata JSON alongside.

    **Final (test-set) metrics:**
    - F1 `{test_metrics['f1']}` · precision `{test_metrics['precision']}` · recall
      `{test_metrics['recall']}` · accuracy `{test_metrics['accuracy']}`
      (validation F1 `{tuned_val['f1']}`).

    **What the pipeline owns** (fit AND inference): derived features (`incomplete_share`,
    `delay_x_incomplete`, `experience_share`), median/mode imputation, scaling, one-hot
    encoding — a raw driver-profile row in, a delay/no-delay label out.

    **Next steps**
    1. All four targets now have notebooks and saved models — wire them into a shared
       inference entry point (`booking_status`, `booking_value`, `customer_cancel_flag`,
       `driver_delay_flag`).
    2. The delay flag is the minority (13%); a **decision-threshold** layer on `predict_proba`
       lets the business set the alert budget (e.g. "flag when delay-prob ≥ 0.6").
    3. The dataset is small (4k) — if more driver telemetry arrives, re-run the pipeline
       and watch the CV-vs-test gap.
    """)
    return


if __name__ == "__main__":
    app.run()