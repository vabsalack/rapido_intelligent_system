import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    from pathlib import Path
    import numpy as np
    import polars as pl
    import typer
    from sklearn.feature_selection import (
        mutual_info_classif, mutual_info_regression, RFE,
    )
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from loguru import logger
    from tqdm import tqdm
    from rapido_intelligent_system.config_mnb import PROCESSED_DATA_DIR


@app.function
def fit_label_maps(df: pl.DataFrame, cat_cols: list[str]) -> dict[str, dict]:
    """Learn sorted label->code maps for categorical columns (fit on train only)."""
    return {
        c: {v: i for i, v in enumerate(df[c].drop_nulls().unique().sort().to_list())}
        for c in cat_cols
    }


@app.function
def apply_label_codes(
    df: pl.DataFrame, cat_cols: list[str], label_maps: dict[str, dict]
) -> pl.DataFrame:
    """Encode categorical columns to float codes; unseen values -> -1."""
    exprs = []
    for c in cat_cols:
        if c in label_maps:
            exprs.append(
                pl.col(c)
                .replace(label_maps[c])
                .cast(pl.Float64, strict=False)
                .fill_null(-1.0)
                .alias(c)
            )
    return df.with_columns(exprs) if exprs else df


@app.function
def build_selection_matrix(
    df: pl.DataFrame,
    feature_cols: list[str],
    cat_cols: list[str],
    target_col: str,
    label_maps: dict[str, dict],
) -> tuple[np.ndarray, np.ndarray]:
    """Encode + median-impute a numeric selection matrix for a target.

    Returns (X, y) as numpy arrays. Used only for filter/wrapper selection;
    saved datasets keep raw columns (encoding happens in the modeling pipeline).
    """
    num_cols = [c for c in feature_cols if c not in cat_cols]
    work = apply_label_codes(df.select(feature_cols + [target_col]), cat_cols, label_maps)
    medians = {c: float(work[c].median()) for c in num_cols}
    work = work.with_columns([pl.col(c).fill_null(medians[c]).alias(c) for c in num_cols])
    X = work.select(feature_cols).to_numpy()
    y = work[target_col].to_numpy()
    return X, y


@app.function
def mutual_info_ranking(
    X: np.ndarray, y: np.ndarray, feature_names: list[str], task_type: str = "classification"
) -> pl.DataFrame:
    """Filter method #1: rank features by mutual information with the target."""
    if task_type == "regression":
        scores = mutual_info_regression(X, y, random_state=42)
    else:
        scores = mutual_info_classif(X, y, random_state=42)
    return (
        pl.DataFrame({"feature": feature_names, "mi_score": np.round(scores, 4)})
        .sort("mi_score", descending=True)
    )


@app.function
def pearson_ranking(
    X: np.ndarray, y: np.ndarray, feature_names: list[str]
) -> pl.DataFrame:
    """Filter reference for regression: Pearson correlation with the target."""
    vals = [
        np.corrcoef(X[:, i], y)[0, 1] if np.std(X[:, i]) > 0 else np.nan
        for i in range(X.shape[1])
    ]
    return (
        pl.DataFrame({"feature": feature_names, "pearson": np.round(vals, 3)})
        .sort("pearson", descending=True)
    )


@app.function
def rfe_selection(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    task_type: str = "classification",
    n_features_to_select: int = 8,
    step: int = 2,
    sample_rows: int = 30_000,
    n_estimators: int = 40,
    seed: int = 42,
) -> tuple[pl.DataFrame, list[str]]:
    """Wrapper method: Recursive Feature Elimination with a random forest.

    Runs on a seeded random sample (wrapper methods are expensive on large data).
    Returns (ranking_df, selected_features).
    """
    idx = np.arange(X.shape[0])
    if X.shape[0] > sample_rows:
        rng = np.random.default_rng(seed)
        idx = rng.choice(X.shape[0], size=sample_rows, replace=False)
    Xs, ys = X[idx], y[idx]
    logger.info(f"RFE ({task_type}) on {Xs.shape[0]:,} rows x {Xs.shape[1]} features")
    if task_type == "regression":
        est = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=12, min_samples_leaf=10,
            n_jobs=-1, random_state=seed,
        )
    else:
        est = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=12, min_samples_leaf=10,
            n_jobs=-1, random_state=seed,
        )
    n_sel = max(2, min(n_features_to_select, Xs.shape[1] - 1))
    rfe = RFE(est, n_features_to_select=n_sel, step=step)
    rfe.fit(Xs, ys)
    ranking_df = (
        pl.DataFrame({"feature": feature_names, "rfe_rank": rfe.ranking_})
        .sort("rfe_rank")
    )
    selected = ranking_df.filter(pl.col("rfe_rank") == 1)["feature"].to_list()
    return ranking_df, selected


@app.function
def combine_selections(
    mi_rank: pl.DataFrame, rfe_selected: list[str], pad: int = 3
) -> list[str]:
    """Combine filter + wrapper: wrapper set, padded with top-MI features it missed."""
    wrapper = list(rfe_selected)
    mi_top = mi_rank.head(len(wrapper) + pad)["feature"].to_list()
    extra = [f for f in mi_top if f not in wrapper][:pad]
    return wrapper + extra


@app.function
def selection_compare_df(
    mi_rank: pl.DataFrame, rfe_selected: list[str], final: list[str]
) -> pl.DataFrame:
    """Side-by-side view: MI score, wrapper flag and final decision per feature."""
    return mi_rank.with_columns(
        pl.col("feature").is_in(rfe_selected).alias("wrapper_selected"),
        pl.col("feature").is_in(final).alias("final"),
    )


if __name__ == "__main__":
    app.run()