import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import polars as pl
    import altair as alt
    from rapido_intelligent_system.config_mnb import (
        RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR
    )
    from rapido_intelligent_system.dataset_mnb import (
        load_all_raw, save_dataframe, train_test_split_stratified,
        train_test_split_random,
    )
    from rapido_intelligent_system.plots_mnb import feature_bar_chart
    from rapido_intelligent_system.selection_mnb import (
        fit_label_maps, build_selection_matrix, mutual_info_ranking,
        pearson_ranking, rfe_selection, combine_selections,
        selection_compare_df,
    )


@app.cell
def _():
    mo.md(r"""
    # 0.02 — Feature Selection & 4 Model Datasets

    **Project:** Rapido Intelligent System · **Author:** kittu
    **Stage:** curated candidate features → filter (mutual information) + wrapper (RFE) → final per-target datasets.

    This is a **decision step**, not a modeling step: it picks the features that matter for each
    of the **4 modeling targets** and saves ready-to-model datasets.

    | Target | Task | Source |
    |---|---|---|
    | `booking_status` | multi-class | bookings + location demand + time flags |
    | `booking_value` | regression | bookings + location demand + time flags |
    | `customer_cancel_flag` | binary | customers profile |
    | `driver_delay_flag` | binary | drivers profile |

    ### Method
    1. **Build candidate feature matrices** — curated columns only, **no new engineering**
       (that belongs to the modeling notebooks).
    2. **Split train / test BEFORE selection** so test rows never influence feature choice
       (avoids selection leakage).
    3. **Filter**: Mutual Information (all targets) + Pearson correlation (regression reference).
    4. **Wrapper**: Recursive Feature Elimination with a random forest, on a **seeded sample**
       of the train split (wrappers are expensive on large data).
    5. **Combine**: wrapper's picks, padded with the top-MI features it missed.
    6. **Save**: candidate frames → `data/interim/`, selected-feature train/test → `data/processed/`.
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ## Leakage exclusions applied

    Candidate sets exclude anything derived from the target:

    - **`booking_status`** — drop `actual_ride_time_min` (only exists for completed rides) and
      `incomplete_ride_reason` (defines the class). Both carry the answer in their very presence.
    - **`booking_value`** — equals `base_fare × surge_multiplier` by construction; those two stay
      as legitimate drivers, but the near-determinism must be noted when reading feature rankings.
    - **`customer_cancel_flag`** — drop `cancelled_rides`, `cancellation_rate` (derived from the flag).
    - **`driver_delay_flag`** — drop `delay_count`, `delay_rate` (derived from the flag).
    - `location_demand`'s `cancelled_rides` / `completed_rides` aggregates dropped too.
    """)
    return


@app.cell
def _():
    mo.md(f"""
    ## Outputs

    - **`data/interim/{{target}}_candidates.csv`** — full candidate matrix (all curated features + target).
    - **`data/processed/{{target}}_train.csv`** / **`{{target}}_test.csv`** — selected features only, 80/20 split.

    - Interim: `{INTERIM_DATA_DIR}`
    - Processed: `{PROCESSED_DATA_DIR}`
    """)
    return


@app.cell
def _():
    BOOKING_FEATURES = [
        "hour_of_day", "is_weekend", "day_of_week",
        "ride_distance_km", "estimated_ride_time_min",
        "base_fare", "surge_multiplier",
        "city", "pickup_location", "drop_location", "vehicle_type",
        "traffic_level", "weather_condition",
        "total_requests", "avg_wait_time_min", "avg_surge_multiplier", "demand_level",
        "peak_time_flag", "season",
    ]
    CUSTOMER_FEATURES = [
        "customer_gender", "customer_age", "customer_city", "customer_signup_days_ago",
        "preferred_vehicle_type", "total_bookings", "completed_rides",
        "incomplete_rides", "avg_customer_rating",
    ]
    DRIVER_FEATURES = [
        "driver_age", "driver_city", "vehicle_type", "driver_experience_years",
        "total_assigned_rides", "accepted_rides", "incomplete_rides",
        "acceptance_rate", "avg_driver_rating", "avg_pickup_delay_min",
    ]
    TARGET_CONFIGS = {
        "booking_status": {"features": BOOKING_FEATURES, "task": "classification", "split": "stratified"},
        "booking_value": {"features": BOOKING_FEATURES, "task": "regression", "split": "random"},
        "customer_cancel_flag": {"features": CUSTOMER_FEATURES, "task": "classification", "split": "stratified"},
        "driver_delay_flag": {"features": DRIVER_FEATURES, "task": "classification", "split": "stratified"},
    }
    return (BOOKING_FEATURES, CUSTOMER_FEATURES, DRIVER_FEATURES, TARGET_CONFIGS)


@app.cell
def _():
    rfe_rows = mo.ui.slider(
        5_000, 30_000, value=15_000, step=5_000,
        label="RFE sample rows (wrapper step)",
    )
    rfe_features = mo.ui.slider(
        3, 12, value=8, step=1,
        label="RFE: features to keep",
    )
    seed = mo.ui.number(0, 999, value=42, label="Random seed")
    mo.hstack([rfe_rows, rfe_features, seed], justify="space-between")
    return (rfe_features, rfe_rows, seed)


@app.cell
def _():
    mo.md(r"""
    ## 1. Load raw data & build candidate matrices

    Four candidate sets. Bookings is enriched on:
    - `location_demand` (join on city, pickup_location, hour_of_day, vehicle_type),
    - `time_features` (join on booking date + hour).

    The static customer / driver tables are used directly.
    """)
    return


@app.cell
def _():
    raw = load_all_raw(RAW_DATA_DIR)
    raw
    return (raw,)


@app.cell
def _(pl, raw):
    loc_enrich = raw["location_demand"].select([
        "city", "pickup_location", "hour_of_day", "vehicle_type",
        "total_requests", "avg_wait_time_min", "avg_surge_multiplier", "demand_level",
    ])
    tf_enrich = (
        raw["time_features"]
        .with_columns(pl.col("datetime").str.slice(0, 10).alias("booking_date"))
        .select(["booking_date", "hour_of_day", "peak_time_flag", "season"])
    )
    book_features = (
        raw["bookings"]
        .join(loc_enrich, on=["city", "pickup_location", "hour_of_day", "vehicle_type"], how="left")
        .join(tf_enrich, on=["booking_date", "hour_of_day"], how="left")
    )
    return (book_features, loc_enrich, tf_enrich)


@app.cell
def _(BOOKING_FEATURES, CUSTOMER_FEATURES, DRIVER_FEATURES, TARGET_CONFIGS, book_features, pl, raw):
    candidate_sets = {
        "booking_status": book_features.select(BOOKING_FEATURES + ["booking_status"]),
        "booking_value": book_features.select(BOOKING_FEATURES + ["booking_value"]),
        "customer_cancel_flag": raw["customers"].select(CUSTOMER_FEATURES + ["customer_cancel_flag"]),
        "driver_delay_flag": raw["drivers"].select(DRIVER_FEATURES + ["driver_delay_flag"]),
    }
    cat_cols_by_target = {
        t: [c for c in cfg["features"] if candidate_sets[t][c].dtype == pl.String]
        for t, cfg in TARGET_CONFIGS.items()
    }
    return (candidate_sets, cat_cols_by_target)


@app.cell
def _(candidate_sets, pl):
    candidate_overview = pl.DataFrame(
        [
            {"target": t, "rows": df.height, "cols": df.width}
            for t, df in candidate_sets.items()
        ]
    )
    candidate_overview
    return (candidate_overview,)


@app.cell
def _():
    mo.md(r"""
    ## 2. Split train / test BEFORE selection

    Feature selection observes **only the train split**. This keeps the test set pristine:
    every feature choice and imputation value is a pure function of training data.
    `booking_status` and the two flags split **stratified**; `booking_value` (regression) splits
    randomly. Candidate frames also land in `data/interim/` for reproducibility.
    """)
    return


@app.cell
def _(INTERIM_DATA_DIR, candidate_sets, save_dataframe):
    interim_paths = {
        t: save_dataframe(df, INTERIM_DATA_DIR, f"{t}_candidates.csv")
        for t, df in candidate_sets.items()
    }
    interim_paths
    return (interim_paths,)


@app.cell
def _(TARGET_CONFIGS, candidate_sets, seed, train_test_split_random, train_test_split_stratified):
    splits = {}
    for _t, _cfg in TARGET_CONFIGS.items():
        _df = candidate_sets[_t]
        if _cfg["split"] == "stratified":
            _tr, _te = train_test_split_stratified(_df, target_col=_t, test_size=0.2, seed=seed.value)
        else:
            _tr, _te = train_test_split_random(_df, test_size=0.2, seed=seed.value)
        splits[_t] = (_tr, _te)
    return (splits,)


@app.cell
def _(pl, splits):
    split_sizes = pl.DataFrame(
        [{"target": _t, "train_rows": _tr.height, "test_rows": _te.height}
         for _t, (_tr, _te) in splits.items()]
    )
    split_sizes
    return (split_sizes,)


@app.cell
def _():
    mo.md(r"""
    ## 3. Filter method — Mutual Information

    Runs on the (label-encoded, median-imputed) train matrix. MI captures **non-linear**
    dependence, so it is used for every target. For `booking_value` we additionally report
    **Pearson correlation** as the linear reference. Categorical features are ordinal-coded
    before scoring; nulls in enriched columns are median-filled (train statistics only).
    """)
    return


@app.cell
def _(TARGET_CONFIGS, build_selection_matrix, cat_cols_by_target, fit_label_maps, mutual_info_ranking, pearson_ranking, splits):
    ranking_results = {}
    for _t, _cfg in TARGET_CONFIGS.items():
        _tr, _te = splits[_t]
        _cat_cols = cat_cols_by_target[_t]
        _enc_cols = _cat_cols + ([_t] if _cfg["task"] == "classification" else [])
        _maps = fit_label_maps(_tr, _enc_cols)
        X, y = build_selection_matrix(_tr, _cfg["features"], _enc_cols, _t, _maps)
        if _cfg["task"] == "classification":
            y = y.astype(int)
        mi = mutual_info_ranking(X, y, _cfg["features"], _cfg["task"])
        pear = pearson_ranking(X, y, _cfg["features"]) if _cfg["task"] == "regression" else None
        ranking_results[_t] = {"X": X, "y": y, "mi": mi, "pear": pear}
    return (ranking_results,)


@app.cell
def _(TARGET_CONFIGS, feature_bar_chart, mo, ranking_results):
    mi_charts = {
        _t: feature_bar_chart(_rr["mi"], "feature", "mi_score", title=f"Mutual information — {_t}")
        for _t, _rr in ranking_results.items()
    }
    mo.vstack(
        [
            mo.hstack([mi_charts["booking_status"], mi_charts["booking_value"]]),
            mo.hstack([mi_charts["customer_cancel_flag"], mi_charts["driver_delay_flag"]]),
        ]
    )
    return (mi_charts,)


@app.cell
def _(feature_bar_chart, mo, ranking_results):
    pear_chart = feature_bar_chart(
        ranking_results["booking_value"]["pear"], "feature", "pearson",
        title="Pearson correlation with target — booking_value",
    )
    pear_chart
    return (pear_chart,)


@app.cell
def _(mo, ranking_results):
    mi_head = {t: rr["mi"].head(5).to_dict(as_series=False) for t, rr in ranking_results.items()}
    mo.md(f"""
    ### Reading the filter results

    - **`booking_status`** — context rules: traffic/weather dominate (consistent with the EDA
      cancellation shares). Top-5 MI: `{mi_head['booking_status']['feature']}`.
    - **`booking_value`** — MI concentrated in `base_fare` / `ride_distance_km` /
      `estimated_ride_time_min` / `surge_multiplier` (confirmed by Pearson), then context.
    - **`customer_cancel_flag`** — MI is spread thinly; **no single demographic leaps out** —
      the strong historical signals were correctly excluded as leaks.
    - **`driver_delay_flag`** — `avg_pickup_delay_min` leads, then experience/rating; the
      near-leak `delay_rate` was excluded.
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ## 4. Wrapper method — Recursive Feature Elimination (RFE)

    RFE iteratively drops the least important feature from a **random forest** fit, so it models
    feature **interaction with the target jointly**. To keep runtime sane it runs on a seeded
    random sample of the train split (slider above). Full sample sizes: customers 10k, drivers 5k,
    bookings 20k (RFE caps at the slider value).
    """)
    return


@app.cell
def _(TARGET_CONFIGS, combine_selections, rfe_features, rfe_rows, rfe_selection, ranking_results, seed, selection_compare_df):
    selection_results = {}
    for _t, _cfg in TARGET_CONFIGS.items():
        _rr = ranking_results[_t]
        rank_df, wrapper = rfe_selection(
            _rr["X"], _rr["y"], _cfg["features"], _cfg["task"],
            n_features_to_select=rfe_features.value,
            sample_rows=rfe_rows.value,
            seed=seed.value,
        )
        final = combine_selections(_rr["mi"], wrapper, pad=3)
        compare = selection_compare_df(_rr["mi"], wrapper, final)
        selection_results[_t] = {"rank": rank_df, "wrapper": wrapper, "final": final, "compare": compare}
    return (selection_results,)


@app.cell
def _(mo, selection_results):
    blocks = []
    for _t, _r in selection_results.items():
        blocks.append(mo.md(f"### Selection summary — `{_t}`"))
        blocks.append(_r["compare"])
    mo.vstack(blocks)
    return


@app.cell
def _():
    mo.md(r"""
    ### Reading the wrapper results

    `wrapper_selected` marks the RFE picks; `final` marks what the model will actually see
    (wrapper picks + up to 3 top-MI extras it missed). Wrapper and filter rarely fully agree:
    MI is per-feature, RFE is joint — differences are expected and the combination is the point.
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ## 5. Decision — combined selection per target

    The **final** feature set per target = RFE picks, padded with the strongest top-MI features
    that RFE dropped. Rationale: RFE optimizes for joint predictive power; MI padding guards
    against dropping individually informative features that a single tree ensemble under-weights.
    """)
    return


@app.cell
def _(selection_results):
    final_features = {_t: _r["final"] for _t, _r in selection_results.items()}
    final_features
    return (final_features,)


@app.cell
def _(mo, selection_results):
    lines = []
    for _t, _r in selection_results.items():
        lines.append(f"- **`{_t}`** ({len(_r['final'])} features): " + ", ".join(f"`{f}`" for f in _r["final"]))
    mo.md("### Final feature sets\n\n" + "\n".join(lines))
    return


@app.cell
def _():
    mo.md(r"""
    ## 6. Save model-ready datasets

    Only the final features (+ target) survive to `data/processed/`, 80/20 train/test.
    Encoding is **not** applied here — the raw (label / numeric) columns are saved and the
    modeling notebooks own encoding, scaling and imputation.
    """)
    return


@app.cell
def _(PROCESSED_DATA_DIR, TARGET_CONFIGS, save_dataframe, selection_results, splits):
    processed_paths = {}
    for _t, _cfg in TARGET_CONFIGS.items():
        _tr, _te = splits[_t]
        keep = selection_results[_t]["final"] + [_t]
        processed_paths[_t] = {
            "train": save_dataframe(_tr.select(keep), PROCESSED_DATA_DIR, f"{_t}_train.csv"),
            "test": save_dataframe(_te.select(keep), PROCESSED_DATA_DIR, f"{_t}_test.csv"),
        }
    return (processed_paths,)


@app.cell
def _(TARGET_CONFIGS, pl, selection_results, splits):
    verify = pl.DataFrame(
        [
            {
                "target": _t,
                "task": _cfg["task"],
                "candidate_features": len(_cfg["features"]),
                "selected_features": len(selection_results[_t]["final"]),
                "train_rows": splits[_t][0].height,
                "test_rows": splits[_t][1].height,
                "cols_match": (
                    splits[_t][0].select(selection_results[_t]["final"] + [_t]).columns
                    == splits[_t][1].select(selection_results[_t]["final"] + [_t]).columns
                ),
            }
            for _t, _cfg in TARGET_CONFIGS.items()
        ]
    )
    verify
    return (verify,)


@app.cell
def _():
    mo.md(r"""
    ## 7. Summary & next steps

    **What was produced**
    - 4 candidate matrices → `data/interim/{{target}}_candidates.csv`.
    - 8 model-ready files → `data/processed/{{target}}_train.csv` / `_test.csv`,
      each holding only selected features + target, split before selection to avoid leakage.

    **Key decisions**
    1. Split before selection — test rows never influenced feature choice.
    2. Leakage columns excluded for every target.
    3. MI filter (all targets) + Pearson reference (regression) + RFE wrapper (sampled, seeded).
    4. Final = RFE picks + top-MI padding (`pad=3`).

    **Next steps**
    1. **Baseline models per target** — logistic regression / decision tree on the saved train
       files; compare against `booking_value`'s trivial `base_fare × surge` baseline.
    2. **Encoding & pipeline** — label encoding, scaling for linear models, imputation, all fit
       on train only (classes in `features_mnb.py` / future pipeline module).
    3. **Metric-driven tuning** — balanced accuracy / PR-AUC for `driver_delay_flag`
       (13% positives) and `booking_status` (68/23/8); RMSE/MAE for `booking_value`.
    4. Iterate selection if baselines disappoint — e.g. inspect RF importance, try
       RandomForest-SHAP or LASSO for the regression target.

    **Watch-outs**
    - `booking_value = base_fare × surge_multiplier` makes the regression target near-deterministic;
      a modeling notebook should quantify how "real" the learned relationship is.
    - RFE rankings are the joint view; MI is per-feature. Prefer interaction-aware metrics when
      the two disagree.
    """)
    return


if __name__ == "__main__":
    app.run()