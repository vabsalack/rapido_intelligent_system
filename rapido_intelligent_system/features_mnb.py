import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    from pathlib import Path
    import typer
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import LabelEncoder
    from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
    from loguru import logger
    from tqdm import tqdm
    from rapido_intelligent_system.config_mnb import PROCESSED_DATA_DIR


@app.cell
def _():
    is_script_mode = mo.app_meta().mode == "script"
    return (is_script_mode,)


@app.cell
def _(is_script_mode):
    typer_app = typer.Typer()

    @typer_app.command()
    def main(
        input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
        output_path: Path = PROCESSED_DATA_DIR / "features.csv",
    ):
        logger.info("Generating features from dataset...")
        for i in tqdm(range(10), total=10):
            if i == 5:
                logger.info("Something happened for iteration 5.")
        logger.success("Features generation complete.")

    if is_script_mode:
        typer_app()
    typer_app
    return


@app.function
def load_all_csvs(raw_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all 5 raw CSV files and return as a dict."""
    file_names = {
        "bookings": "bookings.csv",
        "customers": "customers.csv",
        "drivers": "drivers.csv",
        "location_demand": "location_demand.csv",
        "time_features": "time_features.csv",
    }
    data = {}
    for key, fname in file_names.items():
        path = raw_dir / fname
        data[key] = pd.read_csv(path)
        logger.info(f"  Loaded {key}: {data[key].shape}")
    return data


@app.function
def merge_datasets(
    bookings: pd.DataFrame,
    customers: pd.DataFrame,
    drivers: pd.DataFrame,
    location_demand: pd.DataFrame,
    time_features: pd.DataFrame,
) -> pd.DataFrame:
    """Merge all 5 datasets into a single DataFrame on appropriate keys."""
    # Rename conflicting columns before merge
    customers = customers.rename(columns={
        "completed_rides": "cust_completed_rides",
        "cancelled_rides": "cust_cancelled_rides",
        "incomplete_rides": "cust_incomplete_rides",
    })
    drivers = drivers.rename(columns={
        "completed_rides": "drv_completed_rides",
        "incomplete_rides": "drv_incomplete_rides",
    })
    drivers_merge = drivers.drop(columns=["vehicle_type"], errors="ignore")

    # bookings + customers
    df = bookings.merge(customers, on="customer_id", how="left")
    # + drivers
    df = df.merge(drivers_merge, on="driver_id", how="left")

    # Aggregate location_demand by (city, pickup_location, hour_of_day, vehicle_type)
    loc_agg = (
        location_demand
        .groupby(["city", "pickup_location", "hour_of_day", "vehicle_type"])
        .agg({
            "total_requests": "sum",
            "completed_rides": "sum",
            "cancelled_rides": "sum",
            "avg_wait_time_min": "mean",
            "avg_surge_multiplier": "mean",
        })
        .reset_index()
    )
    loc_agg.columns = [
        "city", "pickup_location", "hour_of_day", "vehicle_type",
        "loc_total_requests", "loc_completed_rides", "loc_cancelled_rides",
        "loc_avg_wait_time", "loc_avg_surge",
    ]
    df = df.merge(loc_agg, on=["city", "pickup_location", "hour_of_day", "vehicle_type"], how="left")

    # Merge time_features on date key
    time_features = time_features.copy()
    time_features["datetime"] = pd.to_datetime(time_features["datetime"])
    df["booking_date"] = pd.to_datetime(df["booking_date"])
    df["booking_datetime"] = pd.to_datetime(
        df["booking_date"].dt.strftime("%Y-%m-%d") + " " + df["booking_time"]
    )
    time_features["date_key"] = time_features["datetime"].dt.date
    df["date_key"] = df["booking_datetime"].dt.date

    tf_merge = time_features[["date_key", "is_holiday", "peak_time_flag", "season"]].drop_duplicates()
    df = df.merge(tf_merge, on="date_key", how="left")

    logger.info(f"  Merged shape: {df.shape}")
    return df


@app.function
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features from the merged DataFrame."""
    df = df.copy()

    # Cyclical time encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df["day_sin"] = np.sin(2 * np.pi * df["booking_date"].dt.dayofweek / 7)
    df["day_cos"] = np.cos(2 * np.pi * df["booking_date"].dt.dayofweek / 7)
    df["month"] = df["booking_date"].dt.month
    df["is_month_start"] = df["booking_date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["booking_date"].dt.is_month_end.astype(int)

    # Ride efficiency
    df["ride_time_diff"] = df["actual_ride_time_min"] - df["estimated_ride_time_min"]
    df["ride_time_ratio"] = df["actual_ride_time_min"] / df["estimated_ride_time_min"].replace(0, np.nan)
    df["speed_kmh"] = df["ride_distance_km"] / (df["actual_ride_time_min"] / 60).replace(0, np.nan)
    df["fare_per_km"] = df["base_fare"] / df["ride_distance_km"].replace(0, np.nan)
    df["surge_impact"] = df["base_fare"] * df["surge_multiplier"]

    # Location demand ratios
    df["loc_cancel_rate"] = df["loc_cancelled_rides"] / df["loc_total_requests"].replace(0, np.nan)
    df["loc_completion_rate"] = df["loc_completed_rides"] / df["loc_total_requests"].replace(0, np.nan)

    # Customer historical
    df["customer_booking_rate"] = df["total_bookings"] / df["customer_signup_days_ago"].replace(0, np.nan)
    df["customer_completion_ratio"] = df["cust_completed_rides"] / df["total_bookings"].replace(0, np.nan)
    df["customer_cancel_ratio"] = df["cust_cancelled_rides"] / df["total_bookings"].replace(0, np.nan)

    # Driver performance
    df["driver_acceptance_ratio"] = df["accepted_rides"] / df["total_assigned_rides"].replace(0, np.nan)
    df["driver_incomplete_ratio"] = df["drv_incomplete_rides"] / df["total_assigned_rides"].replace(0, np.nan)
    df["driver_delay_ratio"] = df["delay_count"] / df["total_assigned_rides"].replace(0, np.nan)

    # Same location flag
    df["same_location"] = (df["pickup_location"] == df["drop_location"]).astype(int)

    return df


@app.function
def encode_categoricals(
    df: pd.DataFrame, columns: list[str]
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """Label-encode categorical columns. Returns (df, dict of encoders)."""
    df = df.copy()
    encoders = {}
    for col in columns:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    return df, encoders


@app.function
def select_features_mutual_info(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    task_type: str = "classification",
    threshold: float = 0.0,
) -> list[str]:
    """Select features using mutual information. Returns list of selected feature names."""
    if task_type == "regression":
        mi_scores = mutual_info_regression(X, y, random_state=42)
    else:
        mi_scores = mutual_info_classif(X, y, random_state=42)

    mi_df = pd.DataFrame({
        "feature": feature_names,
        "mi_score": mi_scores,
    }).sort_values("mi_score", ascending=False).reset_index(drop=True)

    selected = mi_df[mi_df["mi_score"] > threshold]["feature"].tolist()

    # Fallback: if too few, take top 80%
    if len(selected) < len(feature_names) * 0.3:
        n_select = max(int(len(feature_names) * 0.5), 10)
        selected = mi_df.head(n_select)["feature"].tolist()

    return selected, mi_df


@app.function
def build_features_for_target(
    df: pd.DataFrame,
    target_col: str,
    task_type: str = "classification",
    leakage_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build feature set for a single target. Returns (out_df, mi_report)."""
    y = df[target_col].values
    exclude = (leakage_cols or []) + [target_col]
    feature_cols = [c for c in df.columns if c not in exclude]

    X_df = df[feature_cols].copy()
    # Convert all to numeric, coerce errors to NaN
    for col in X_df.columns:
        X_df[col] = pd.to_numeric(X_df[col], errors="coerce")
    # Fill NaN: median for numeric
    X_df = X_df.fillna(X_df.median())

    X = X_df.values

    selected, mi_df = select_features_mutual_info(
        X, y, feature_cols, task_type=task_type
    )

    out_df = df[selected + [target_col]].copy()
    return out_df, mi_df


@app.function
def save_features(
    df: pd.DataFrame, output_dir: Path, filename: str
) -> Path:
    """Save a feature DataFrame to CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename
    df.to_csv(out_path, index=False)
    logger.info(f"  Saved: {out_path} ({df.shape})")
    return out_path


if __name__ == "__main__":
    app.run()
