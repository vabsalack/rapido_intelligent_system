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
        CUSTOMER_FEATURES, CUSTOMER_TARGET, CUSTOMER_ENG_FEATURES,
        CUSTOMER_CATEGORIES, CUSTOMER_BINARY_CLASSES,
        engineer_customer_features, customer_preprocessing,
        build_customer_pipeline, metrics_row_bin, metrics_table_bin,
        metrics_bar_chart, save_model, load_model,
    )


@app.cell
def _():
    mo.md(r"""
    # 0.05 — customer_cancel_flag Model (Binary Classification)

    **Project:** Rapido Intelligent System · **Author:** kittu
    **Goal:** predict whether a customer's booking is cancelled (`customer_cancel_flag` ∈
    {0, 1}) from the customer-history features selected in notebook 0.02.

    ### Workflow in this notebook
    1. **Holdout** — a stratified validation set is carved from the training file; the
       untouched `data/processed/customer_cancel_flag_test.csv` is kept for a **final
       generalization check only**.
    2. **Pipelines** — every experiment is one scikit-learn `Pipeline` that does **feature
       engineering → impute → scale / one-hot → classifier**, so the same object serves
       training *and* inference on raw inputs.
    3. **Baselines** — SGD, logistic regression, SVC, Random Forest, HistGradientBoosting,
       XGBoost, CatBoost and LightGBM. Slow learners get a **sample** of the data so cells
       pass the smoke test fast.
    4. **Select** the best baseline, **fine-tune** it with randomized-search CV, **save** it,
       reload it, and estimate generalisation on the test split.
    5. **Interactive UI** — input customer attributes and see a prediction from the saved pipeline.

    Data is small (8k train / 2k test) and **near-balanced** (cancel ≈ 53%, no-cancel ≈ 47%).
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ### Why these metrics?

    Binary and **near-balanced** (53/47), so accuracy is meaningful — but the business cares
    about the **cancellation class**, so:

    - **accuracy** — overall correctness (fine when classes are balanced),
    - **precision (pos)** — of the flagged cancellations, how many really cancelled,
    - **recall (pos)** — of actual cancellations, how many we caught,
    - **F1 (pos)** — the harmonic mean; the **selection metric** when balanced precision/recall
      are both wanted.

    → **Selection metric: F1 of the cancellation class**, reported with accuracy, precision, recall.
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ### Pipeline architecture (everything inside the pipeline)

    ```
    raw row (9 features)
      └─ engineer      FunctionTransformer — completion_rate, incomplete_rate, bookings_per_year
      └─ prep          ColumnTransformer
      │   ├─ num       SimpleImputer(median) → StandardScaler   (6 raw + 3 engineered)
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
    train_csv = PROCESSED_DATA_DIR / "customer_cancel_flag_train.csv"
    test_csv = PROCESSED_DATA_DIR / "customer_cancel_flag_test.csv"
    model_file = "customer_cancel_flag_model.joblib"
    return model_file, test_csv, train_csv


@app.cell
def _(test_csv, train_csv):
    train_full = pl.read_csv(train_csv)
    test_full = pl.read_csv(test_csv)
    balance = (
        train_full.group_by(CUSTOMER_TARGET)
        .len()
        .sort(CUSTOMER_TARGET)
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
    **Design choice — holdout vs test.** The 8k training file is split again (stratified on the
    target) into *train* (6.4k) and *validation* (1.6k). Model selection and tuning use train +
    val; the separate `_test.csv` (2k) is **never touched until the final generalization check**,
    so the reported generalization error is honest.
    """)
    return


@app.cell
def _(train_full):
    tr, val = train_test_split(
        train_full, test_size=0.2,
        stratify=train_full[CUSTOMER_TARGET],
        random_state=42,
    )
    split_balance = pl.DataFrame(
        {
            "set": ["train", "valid"],
            "rows": [len(tr), len(val)],
            **{
                f"class_{_c}%": [
                    round(100 * (tr[CUSTOMER_TARGET] == _c).mean(), 1),
                    round(100 * (val[CUSTOMER_TARGET] == _c).mean(), 1),
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

    Three cheap, interpretable behaviour ratios derived from the customer's history columns,
    each `clip` guard keeping the transformer finite for any input.

    | Feature | Formula | Meaning |
    |---|---|---|
    | `completion_rate` | `completed_rides / max(total_bookings, 1)` | how reliably rides finish |
    | `incomplete_rate` | `incomplete_rides / max(total_bookings, 1)` | how often rides end incomplete |
    | `bookings_per_year` | `total_bookings / max(signup_days/365, 0.1)` | booking intensity |
    """)
    return


@app.cell
def _(tr):
    eng_sample = engineer_customer_features(tr.head(5))[CUSTOMER_ENG_FEATURES]
    prep_demo = customer_preprocessing()
    n_ml_features = prep_demo.fit_transform(
        engineer_customer_features(tr.head(500))
    ).shape[1]
    eng_sample
    return (n_ml_features,)


@app.cell
def _(n_ml_features):
    mo.md(f"""
    ### Baseline protocol

    Eight learners, each wrapped by the same full pipeline. Slow models (SVC, Random Forest,
    HistGradientBoosting, XGBoost, CatBoost, LightGBM) fit on a **sample** of the training rows
    so the cell stays fast — flip the switch to run everything on the full 6.4k train.
    Every model is scored on the **full validation set** (1.6k rows) with the same metric battery.

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
        "SGDClassifier (softmax)": 5_000,
        "LogisticRegression": 5_000,
        "SVC (RBF)": 4_000,
        "RandomForest": 6_000,
        "HistGradientBoosting": 6_000,
        "XGBoost": 6_000,
        "CatBoost": 6_000,
        "LightGBM": 6_000,
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
        _pipe = build_customer_pipeline(_cfg())
        _pipe.fit(_fit[CUSTOMER_FEATURES], _fit[CUSTOMER_TARGET])
        _yp = _pipe.predict(val[CUSTOMER_FEATURES])
        baseline_rows.append(metrics_row_bin(_name, val[CUSTOMER_TARGET], _yp))
        baseline_fits[_name] = _pipe
        logger.info(f"baseline done: {_name}")
    baseline_table = metrics_table_bin(baseline_rows)
    baseline_table
    return baseline_fits, baseline_table


@app.cell
def _(baseline_table):
    _f1 = metrics_bar_chart(baseline_table, "f1")
    _prec = metrics_bar_chart(baseline_table, "precision")
    _rec = metrics_bar_chart(baseline_table, "recall")
    mo.hstack([_f1, _prec, _rec])
    return


@app.cell
def _():
    mo.md(r"""
    ### Reading the baseline table

    Check **`f1`** first (the selection metric for the cancellation class), then **`recall`**
    (did we catch enough real cancellations?) and **`precision`** (are alerts trustworthy?).
    **`accuracy`** is a sanity check given the class balance.

    Expectation: the boosting/tree family (LightGBM / XGBoost / CatBoost / HistGradientBoosting /
    RandomForest) leads — cancellation behaviour is driven by **non-linear ratios**
    (completion rate × rating × intensity), which trees capture well; linear models (SGD,
    logistic) trail because the boundary is not a clean hyperplane.

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
    default (slider above) — the winner's hyper-parameters are then **refit on the full 6.4k
    train** when the switch is on, so the saved artifact uses all data while the search itself
    stays fast.
    """)
    return


@app.cell
def _():
    search_n_iter = mo.ui.slider(5, 60, value=15, step=5, label="RandomizedSearch iterations")
    search_cv = mo.ui.slider(2, 5, value=3, step=1, label="CV folds")
    tune_sample = mo.ui.slider(
        2_000, 6_400, value=4_000, step=400,
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
        build_customer_pipeline(BASELINES[best_name]()),
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
    best_search.fit(_tune_tr[CUSTOMER_FEATURES], _tune_tr[CUSTOMER_TARGET])
    best_cv_score = round(float(best_search.best_score_), 4)
    best_params = dict(best_search.best_params_)
    if refit_full.value:
        best_pipe = clone(best_search.best_estimator_).fit(
            tr[CUSTOMER_FEATURES], tr[CUSTOMER_TARGET]
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

    Compare **F1** (and precision/recall) before → after tuning. A **licensed improvement**
    means the default hyper-parameters were leaving performance on the table; a flat or worse
    result means the defaults were already near-optimal and the CV search did not overfit.

    The tuned pipeline is saved next, together with its metadata.
    """)
    return


@app.cell
def _(baseline_fits, best_name, best_pipe, val):
    tuned_val = metrics_row_bin(
        f"{best_name} (tuned)",
        val[CUSTOMER_TARGET],
        best_pipe.predict(val[CUSTOMER_FEATURES]),
    )
    tuned_vs_baseline = metrics_table_bin(
        [
            metrics_row_bin(best_name, val[CUSTOMER_TARGET], baseline_fits[best_name].predict(val[CUSTOMER_FEATURES])),
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
        "target": CUSTOMER_TARGET,
        "classes": [0, 1],
        "features": CUSTOMER_FEATURES,
        "engineered_features": list(CUSTOMER_ENG_FEATURES),
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

    The saved pipeline is **reloaded from disk** and scored on `customer_cancel_flag_test.csv`
    (2k rows it never saw during selection or tuning). This is the honest estimate of how the
    model will behave on new customers.
    """)
    return


@app.cell
def _(model_path, test_full):
    loaded_pipe = load_model(model_path)
    test_pred = loaded_pipe.predict(test_full[CUSTOMER_FEATURES])
    test_metrics = metrics_row_bin("loaded on test", test_full[CUSTOMER_TARGET], test_pred)
    confusion = (
        pl.DataFrame({"true": test_full[CUSTOMER_TARGET], "pred": test_pred})
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

    - **`f1`** — the headline for catching cancellations accurately.
    - **`recall`** — fraction of real cancellations flagged; low recall = we miss cancellations.
    - **`precision`** — fraction of flags that are real cancellations; low precision = we cry
      wolf. Together with recall it draws the operational trade-off.
    - **`accuracy`** — fine to read given the 53/47 balance.

    Compare these to the validation numbers above — a small gap = healthy, a big one = overfit.
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ### Interactive prediction — raw inputs → label

    The form below sends a **raw customer-history row** (nothing pre-processed) through the saved
    pipeline: engineering, imputation, scaling, one-hot encoding and the classifier all run
    *inside* the pipeline. Press **Predict** to see the label and class probabilities.
    """)
    return


@app.cell
def _():
    predict_form = mo.ui.form(
        mo.ui.dictionary(
            {
                "customer_gender": mo.ui.dropdown(CUSTOMER_CATEGORIES["customer_gender"], value="Male", label="Gender"),
                "customer_age": mo.ui.number(18, 64, value=40, step=1, label="Age"),
                "customer_city": mo.ui.dropdown(CUSTOMER_CATEGORIES["customer_city"], value="Bangalore", label="City"),
                "customer_signup_days_ago": mo.ui.number(30, 999, value=500, step=1, label="Days since signup"),
                "total_bookings": mo.ui.number(1, 23, value=10, step=1, label="Total bookings"),
                "completed_rides": mo.ui.number(0, 20, value=7, step=1, label="Completed rides"),
                "incomplete_rides": mo.ui.number(0, 5, value=1, step=1, label="Incomplete rides"),
                "avg_customer_rating": mo.ui.number(3.5, 5.0, value=4.3, step=0.1, label="Avg rating"),
                "preferred_vehicle_type": mo.ui.dropdown(CUSTOMER_CATEGORIES["preferred_vehicle_type"], value="Bike", label="Preferred vehicle"),
            }
        ),
        submit_button_label="Predict cancellation",
    )
    predict_form
    return (predict_form,)


@app.cell
def _(CUSTOMER_FEATURES, loaded_pipe, predict_form):
    if predict_form.value:
        _row = pl.DataFrame([predict_form.value]).select(CUSTOMER_FEATURES)
        _pred = int(loaded_pipe.predict(_row)[0])
        _pred_label = CUSTOMER_BINARY_CLASSES[_pred]
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
    `models/customer_cancel_flag_model.joblib` with metadata JSON alongside.

    **Final (test-set) metrics:**
    - F1 `{test_metrics['f1']}` · precision `{test_metrics['precision']}` · recall
      `{test_metrics['recall']}` · accuracy `{test_metrics['accuracy']}`
      (validation F1 `{tuned_val['f1']}`).

    **What the pipeline owns** (fit AND inference): derived features (`completion_rate`,
    `incomplete_rate`, `bookings_per_year`), median/mode imputation, scaling, one-hot
    encoding — a raw customer-history row in, a cancel/no-cancel label out.

    **Next steps**
    1. Repeat this notebook for `driver_delay_flag` (the last target), then wire all four
       models into a shared inference/API entry point.
    2. If the operational cost of false alerts matters, tune the **decision threshold** on
       `predict_proba` (a "risk band" UI) instead of the default 0.5.
    3. The small dataset (8k) makes CV noise a consideration — the refit-on-full-train step
       keeps the saved artifact robust.
    """)
    return


if __name__ == "__main__":
    app.run()