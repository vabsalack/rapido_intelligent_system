import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    from pathlib import Path
    import polars as pl
    import altair as alt
    import typer
    from loguru import logger
    from tqdm import tqdm
    from rapido_intelligent_system.config_mnb import FIGURES_DIR, PROCESSED_DATA_DIR


@app.cell
def _(mo):
    is_script_mode = mo.app_meta().mode == "script"
    return (is_script_mode,)


@app.cell
def _(
    FIGURES_DIR,
    PROCESSED_DATA_DIR,
    Path,
    is_script_mode,
    logger,
    tqdm,
    typer,
):
    typer_app = typer.Typer()

    @typer_app.command()
    def main(
        input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
        output_path: Path = FIGURES_DIR / "plot.png",
    ):
        logger.info("Generating plot from data...")
        for i in tqdm(range(10), total=10):
            if i == 5:
                logger.info("Something happened for iteration 5.")
        logger.success("Plot generation complete.")

    if is_script_mode:
        typer_app()
    return


@app.function
def _cap_chart_input(df: pl.DataFrame, n: int = 4000, seed: int = 42) -> pl.DataFrame:
    """Sample to at most n rows so altair stays under its 5k-row default limit."""
    if df.height <= n:
        return df
    return df.sample(n=n, seed=seed, shuffle=True)


@app.function
def missing_summary(df: pl.DataFrame) -> pl.DataFrame:
    """Return per-column null counts and percentages for a DataFrame."""
    counts = {c: df[c].null_count() for c in df.columns}
    out = pl.DataFrame(
        {"column": list(counts), "null_count": list(counts.values())}
    )
    return out.with_columns(
        (pl.col("null_count") / df.height * 100).round(2).alias("null_pct")
    ).sort("null_count", descending=True)


@app.function
def numeric_summary(df: pl.DataFrame) -> pl.DataFrame:
    """Describe numeric columns (count, nulls, mean, median, std, quartiles, min/max)."""
    num_cols = df.select(pl.selectors.numeric()).columns
    if not num_cols:
        return pl.DataFrame()
    out = []
    for c in num_cols:
        out.append(
            df.select(
                pl.lit(c).alias("column"),
                pl.col(c).count().alias("count"),
                pl.col(c).null_count().alias("null_count"),
                pl.col(c).mean().round(3).alias("mean"),
                pl.col(c).median().round(3).alias("median"),
                pl.col(c).std().round(3).alias("std"),
                pl.col(c).min().round(3).alias("min"),
                pl.col(c).quantile(0.25).round(3).alias("p25"),
                pl.col(c).quantile(0.75).round(3).alias("p75"),
                pl.col(c).max().round(3).alias("max"),
            )
        )
    out = [frame.with_columns(pl.selectors.numeric().cast(pl.Float64)) for frame in out]
    return pl.concat(out)


@app.function
def category_frequency_chart(
    df: pl.DataFrame, col: str, top_n: int = 20, title: str | None = None
) -> alt.Chart:
    """Horizontal bar chart of the top_n values of a column (input capped to 4k rows)."""
    df = _cap_chart_input(df)
    counts = (
        df.group_by(col)
        .len()
        .rename({"len": "count"})
        .sort("count", descending=True)
        .head(top_n)
    )
    return (
        alt.Chart(counts, title=title or f"Distribution of {col}")
        .mark_bar()
        .encode(
            x=alt.X("count:Q", title="Count"),
            y=alt.Y(f"{col}:N", title=col, sort="-x"),
            tooltip=[col, "count"],
        )
        .properties(width=420, height=300)
    )


@app.function
def numeric_histogram(
    df: pl.DataFrame, col: str, bins: int = 40, title: str | None = None
) -> alt.Chart:
    """Histogram of a numeric column with a dashed red mean line."""
    df = _cap_chart_input(df)
    hist = (
        alt.Chart(df, title=title or f"Distribution of {col}")
        .mark_bar(opacity=0.8)
        .encode(
            alt.X(f"{col}:Q", bin=alt.Bin(maxbins=bins), title=col),
            y=alt.Y("count()", title="Count"),
        )
    )
    mean_val = float(df[col].mean())
    rule = (
        alt.Chart(pl.DataFrame({"m": [mean_val]}))
        .mark_rule(color="red", strokeDash=[4, 4])
        .encode(x=alt.X("m:Q"))
    )
    return (hist + rule).properties(width=420, height=280)


@app.function
def stacked_target_chart(
    df: pl.DataFrame,
    cat_col: str,
    target_col: str,
    top_n: int = 15,
    title: str | None = None,
) -> alt.Chart:
    """Normalized stacked bars: within each category, the target composition."""
    df = _cap_chart_input(df)
    top_cats = (
        df.group_by(cat_col)
        .len()
        .sort("len", descending=True)
        .head(top_n)
        .select(cat_col)
    )
    counts = (
        df.join(top_cats, on=cat_col, how="inner")
        .group_by([cat_col, target_col])
        .len()
        .rename({"len": "count"})
    )
    return (
        alt.Chart(counts, title=title or f"{target_col} composition by {cat_col}")
        .mark_bar()
        .encode(
            x=alt.X(f"{cat_col}:N", title=cat_col, sort="-y"),
            y=alt.Y("count:Q", stack="normalize", axis=alt.Axis(format="%"), title="share"),
            color=alt.Color(f"{target_col}:N", title=target_col),
            tooltip=[cat_col, target_col, "count"],
        )
        .properties(width=420, height=280)
    )


@app.function
def feature_bar_chart(
    df: pl.DataFrame,
    cat_col: str,
    val_col: str,
    top_n: int = 15,
    title: str | None = None,
) -> alt.Chart:
    """Sorted bar chart of the top_n categories' mean `val_col` (e.g. MI scores)."""
    df = _cap_chart_input(df)
    top = (
        df.group_by(cat_col)
        .agg(pl.col(val_col).mean().alias(val_col))
        .sort(val_col, descending=True)
        .head(top_n)
    )
    return (
        alt.Chart(top, title=title or f"{val_col} by {cat_col}")
        .mark_bar()
        .encode(
            x=alt.X(f"{val_col}:Q", title=val_col),
            y=alt.Y(f"{cat_col}:N", title=cat_col, sort="-x"),
            tooltip=[cat_col, val_col],
        )
        .properties(width=420, height=300)
    )


if __name__ == "__main__":
    app.run()
