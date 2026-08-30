import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    from pathlib import Path
    import numpy as np
    import polars as pl
    import gdown
    import typer
    from sklearn.model_selection import train_test_split
    from rich import print
    from loguru import logger
    from tqdm import tqdm
    from rapido_intelligent_system.config_mnb import (
        PROCESSED_DATA_DIR, RAW_DATA_DIR, GDRIVE_LINK
    )


@app.cell
def _():
    is_script_mode = mo.app_meta().mode == "script"
    return (is_script_mode,)


@app.cell
def _(is_script_mode):
    typer_app = typer.Typer()

    @typer_app.command()
    def main(
        input_path: Path = RAW_DATA_DIR / "dataset.csv",
        output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    ):
        logger.info("Processing dataset...")
        for i in tqdm(range(10), total=10):
            if i == 5:
                logger.info("Something happened for iteration 5.")
        logger.success("Processing dataset complete.")

    @typer_app.command("download")
    def download_gdrive_folder(drive_web_url: str = GDRIVE_LINK,
                               output_dir: str = RAW_DATA_DIR) -> Path:
        """
        Downloads datasets from author's, keerhtivasan, Google drive to default RAW_DATA_DIR
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            files = gdown.download_folder(
                url=drive_web_url,
                output=str(output_dir),
                quiet=False,
                use_cookies=False,
            )
            print(files)
        except Exception as e:
            print(f"An error occurred: {e}")
            raise
        return output_dir

    if is_script_mode:
        typer_app()
    typer_app
    return


@app.function
def load_raw_csv(filename: str, raw_dir: Path = RAW_DATA_DIR) -> pl.DataFrame:
    """Load a raw CSV file with polars, inferring the schema from the full file."""
    path = raw_dir / filename
    logger.info(f"Loading {path.name} ({path.stat().st_size / 1e6:.1f} MB)")
    return pl.read_csv(path)


@app.function
def load_all_raw(raw_dir: Path = RAW_DATA_DIR) -> dict[str, pl.DataFrame]:
    """Load all 5 raw CSV files, keyed by dataset name."""
    file_names = {
        "bookings": "bookings.csv",
        "customers": "customers.csv",
        "drivers": "drivers.csv",
        "location_demand": "location_demand.csv",
        "time_features": "time_features.csv",
    }
    data = {key: load_raw_csv(fname, raw_dir) for key, fname in file_names.items()}
    shapes = ", ".join(f"{k} {v.shape[0]:,}x{v.shape[1]}" for k, v in data.items())
    logger.info(f"Loaded all raw files: {shapes}")
    return data


@app.function
def eda_sample(
    df: pl.DataFrame, n: int = 50_000, seed: int = 42, shuffle: bool = True
) -> pl.DataFrame:
    """Return a reproducible random sample of up to `n` rows (full frame if smaller)."""
    if df.height <= n:
        return df
    return df.sample(n=n, seed=seed, shuffle=shuffle)


@app.function
def shape_overview(file_data: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Build a compact (file, rows, cols) overview table from a dict of frames."""
    return pl.DataFrame(
        {
            "file": list(file_data),
            "rows": [df.height for df in file_data.values()],
            "cols": [df.width for df in file_data.values()],
        }
    )


@app.function
def save_dataframe(
    df: pl.DataFrame, output_dir: Path, filename: str
) -> Path:
    """Write a DataFrame to CSV, creating the directory if needed."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename
    df.write_csv(out_path)
    logger.success(f"Saved {df.height:,} rows x {df.width} cols -> {out_path}")
    return out_path


@app.function
def train_test_split_stratified(
    df: pl.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Stratified train/test split on `target_col` (use for classification)."""
    idx = np.arange(df.height)
    train_idx, test_idx = train_test_split(
        idx, test_size=test_size, random_state=seed,
        stratify=df[target_col].to_numpy(),
    )
    train, test = df.gather(train_idx), df.gather(test_idx)
    logger.info(
        f"Stratified split on {target_col}: train {train.height:,} / test {test.height:,}"
    )
    return train, test


@app.function
def train_test_split_random(
    df: pl.DataFrame, test_size: float = 0.2, seed: int = 42
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Plain random train/test split (use for regression targets)."""
    idx = np.arange(df.height)
    train_idx, test_idx = train_test_split(
        idx, test_size=test_size, random_state=seed
    )
    train, test = df.gather(train_idx), df.gather(test_idx)
    logger.info(f"Random split: train {train.height:,} / test {test.height:,}")
    return train, test


if __name__ == "__main__":
    app.run()
