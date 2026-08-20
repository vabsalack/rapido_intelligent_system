"""
Feature Engineering Script for Rapido Intelligent System
=========================================================
Builds feature CSV files for 4 target variables:
1. booking_value (regression)
2. booking_status (classification)
3. customer_cancel_flag (classification)
4. driver_delay_flag (classification)

Pipeline:
  1. Load 5 raw CSVs
  2. Merge on appropriate keys
  3. Engineer features
  4. Encode categoricals
  5. Feature selection via mutual information
  6. Save to data/processed/
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from loguru import logger

# ── Paths ────────────────────────────────────────────────────────────────────
PROJ_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJ_ROOT / "data" / "raw"
PROC_DIR = PROJ_ROOT / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Load raw data ────────────────────────────────────────────────────────
logger.info("Loading raw CSV files...")
bookings = pd.read_csv(RAW_DIR / "bookings.csv")
customers = pd.read_csv(RAW_DIR / "customers.csv")
drivers = pd.read_csv(RAW_DIR / "drivers.csv")
location_demand = pd.read_csv(RAW_DIR / "location_demand.csv")
time_features = pd.read_csv(RAW_DIR / "time_features.csv")

logger.info(f"  bookings: {bookings.shape}")
logger.info(f"  customers: {customers.shape}")
logger.info(f"  drivers: {drivers.shape}")
logger.info(f"  location_demand: {location_demand.shape}")
logger.info(f"  time_features: {time_features.shape}")

# ── 2. Merge datasets ───────────────────────────────────────────────────────
logger.info("Merging datasets...")

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

# Drop duplicate vehicle_type from drivers (bookings already has it)
drivers_merge = drivers.drop(columns=["vehicle_type"], errors="ignore")

# Merge bookings with customers
df = bookings.merge(customers, on="customer_id", how="left")

# Merge with drivers (without vehicle_type conflict)
df = df.merge(drivers_merge, on="driver_id", how="left")

# Merge with location_demand (on city, pickup_location, hour_of_day, vehicle_type)
loc_agg = location_demand.groupby(
    ["city", "pickup_location", "hour_of_day", "vehicle_type"]
).agg({
    "total_requests": "sum",
    "completed_rides": "sum",
    "cancelled_rides": "sum",
    "avg_wait_time_min": "mean",
    "avg_surge_multiplier": "mean",
}).reset_index()

loc_agg.columns = [
    "city", "pickup_location", "hour_of_day", "vehicle_type",
    "loc_total_requests", "loc_completed_rides", "loc_cancelled_rides",
    "loc_avg_wait_time", "loc_avg_surge",
]
df = df.merge(loc_agg, on=["city", "pickup_location", "hour_of_day", "vehicle_type"], how="left")

# Merge with time_features (on datetime components)
time_features["datetime"] = pd.to_datetime(time_features["datetime"])
df["booking_date"] = pd.to_datetime(df["booking_date"])
df["booking_datetime"] = pd.to_datetime(df["booking_date"].dt.strftime("%Y-%m-%d") + " " + df["booking_time"])

# Extract date key for merging
time_features["date_key"] = time_features["datetime"].dt.date
df["date_key"] = df["booking_datetime"].dt.date

tf_merge = time_features[["date_key", "is_holiday", "peak_time_flag", "season"]].drop_duplicates()
df = df.merge(tf_merge, on="date_key", how="left")

logger.info(f"  Merged shape: {df.shape}")

# ── 3. Feature Engineering ──────────────────────────────────────────────────
logger.info("Engineering features...")

# Time-based features
df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
df["day_sin"] = np.sin(2 * np.pi * df["booking_date"].dt.dayofweek / 7)
df["day_cos"] = np.cos(2 * np.pi * df["booking_date"].dt.dayofweek / 7)
df["month"] = df["booking_date"].dt.month
df["is_month_start"] = df["booking_date"].dt.is_month_start.astype(int)
df["is_month_end"] = df["booking_date"].dt.is_month_end.astype(int)

# Ride efficiency features
df["ride_time_diff"] = df["actual_ride_time_min"] - df["estimated_ride_time_min"]
df["ride_time_ratio"] = df["actual_ride_time_min"] / df["estimated_ride_time_min"].replace(0, np.nan)
df["speed_kmh"] = df["ride_distance_km"] / (df["actual_ride_time_min"] / 60).replace(0, np.nan)
df["fare_per_km"] = df["base_fare"] / df["ride_distance_km"].replace(0, np.nan)
df["surge_impact"] = df["base_fare"] * df["surge_multiplier"]

# Location demand features
df["loc_cancel_rate"] = df["loc_cancelled_rides"] / df["loc_total_requests"].replace(0, np.nan)
df["loc_completion_rate"] = df["loc_completed_rides"] / df["loc_total_requests"].replace(0, np.nan)

# Customer historical features
df["customer_booking_rate"] = df["total_bookings"] / df["customer_signup_days_ago"].replace(0, np.nan)
df["customer_completion_ratio"] = df["cust_completed_rides"] / df["total_bookings"].replace(0, np.nan)
df["customer_cancel_ratio"] = df["cust_cancelled_rides"] / df["total_bookings"].replace(0, np.nan)

# Driver performance features
df["driver_acceptance_ratio"] = df["accepted_rides"] / df["total_assigned_rides"].replace(0, np.nan)
df["driver_incomplete_ratio"] = df["drv_incomplete_rides"] / df["total_assigned_rides"].replace(0, np.nan)
df["driver_delay_ratio"] = df["delay_count"] / df["total_assigned_rides"].replace(0, np.nan)

# Same pickup and drop flag
df["same_location"] = (df["pickup_location"] == df["drop_location"]).astype(int)

# ── 4. Encode categoricals ──────────────────────────────────────────────────
logger.info("Encoding categorical variables...")

label_encoders = {}
categorical_cols = [
    "city", "pickup_location", "drop_location", "vehicle_type",
    "traffic_level", "weather_condition", "day_of_week",
    "customer_gender", "customer_city", "preferred_vehicle_type",
    "driver_city", "season",
]

for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

# Target encodings
df["booking_status_enc"] = LabelEncoder().fit_transform(df["booking_status"].astype(str))

# ── 5. Drop columns not needed for features ─────────────────────────────────
drop_cols = [
    "booking_id", "booking_date", "booking_time", "booking_datetime",
    "date_key", "incomplete_ride_reason", "customer_id", "driver_id",
    "booking_status",  # will be used as target, kept for reference
]
df_features = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

# Fill NaN with median for numeric columns
numeric_cols = df_features.select_dtypes(include=[np.number]).columns
df_features[numeric_cols] = df_features[numeric_cols].fillna(df_features[numeric_cols].median())

logger.info(f"  Final feature matrix: {df_features.shape}")

# ── 6. Feature selection per target ─────────────────────────────────────────
TARGETS = {
    "booking_value": {"type": "regression"},
    "booking_status_enc": {"type": "classification", "display": "booking_status"},
    "customer_cancel_flag": {"type": "classification"},
    "driver_delay_flag": {"type": "classification"},
}

# Feature columns (exclude targets and target-related leakage)
leakage_cols = {
    "booking_value": ["booking_value"],
    "booking_status_enc": ["booking_status_enc"],
    "customer_cancel_flag": ["customer_cancel_flag"],
    "driver_delay_flag": ["driver_delay_flag"],
}

for target_col, meta in TARGETS.items():
    display_name = meta.get("display", target_col)
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing target: {display_name} ({meta['type']})")
    logger.info(f"{'='*60}")

    y = df_features[target_col].values
    exclude = leakage_cols.get(target_col, []) + [target_col]
    feature_cols = [c for c in df_features.columns if c not in exclude]
    X = df_features[feature_cols].values

    # Compute mutual information
    if meta["type"] == "regression":
        mi_scores = mutual_info_regression(X, y, random_state=42)
    else:
        mi_scores = mutual_info_classif(X, y, random_state=42)

    mi_df = pd.DataFrame({
        "feature": feature_cols,
        "mi_score": mi_scores,
    }).sort_values("mi_score", ascending=False).reset_index(drop=True)

    # Select features with MI > 0 (non-zero contribution)
    selected = mi_df[mi_df["mi_score"] > 0]["feature"].tolist()

    # If too few features, take top 80%
    if len(selected) < len(feature_cols) * 0.3:
        n_select = max(int(len(feature_cols) * 0.5), 10)
        selected = mi_df.head(n_select)["feature"].tolist()

    logger.info(f"  Selected {len(selected)}/{len(feature_cols)} features")
    logger.info(f"  Top 10 MI scores:")
    for _, row in mi_df.head(10).iterrows():
        logger.info(f"    {row['feature']}: {row['mi_score']:.4f}")

    # Build output dataframe
    out_df = df_features[selected + [target_col]].copy()
    out_path = PROC_DIR / f"features_{display_name}.csv"
    out_df.to_csv(out_path, index=False)
    logger.info(f"  Saved: {out_path} ({out_df.shape})")

# ── 7. Save MI scores for reference ─────────────────────────────────────────
logger.info("\nSaving MI score reference...")
mi_reference = {}
for target_col, meta in TARGETS.items():
    display_name = meta.get("display", target_col)
    y = df_features[target_col].values
    exclude = leakage_cols.get(target_col, []) + [target_col]
    feature_cols = [c for c in df_features.columns if c not in exclude]
    X = df_features[feature_cols].values

    if meta["type"] == "regression":
        mi_scores = mutual_info_regression(X, y, random_state=42)
    else:
        mi_scores = mutual_info_classif(X, y, random_state=42)

    mi_reference[display_name] = pd.DataFrame({
        "feature": feature_cols,
        "mi_score": mi_scores,
    }).sort_values("mi_score", ascending=False).reset_index(drop=True)

# Save combined MI report
with open(PROC_DIR / "mutual_information_report.txt", "w") as f:
    for target_name, mi_df in mi_reference.items():
        f.write(f"\n{'='*60}\n")
        f.write(f"Target: {target_name}\n")
        f.write(f"{'='*60}\n")
        f.write(mi_df.to_string(index=False))
        f.write("\n")

logger.info(f"Saved MI report to {PROC_DIR / 'mutual_information_report.txt'}")
logger.info("\nDone! All feature files saved to data/processed/")
