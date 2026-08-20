import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    from pathlib import Path
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import LabelEncoder
    from rapido_intelligent_system.config_mnb import RAW_DATA_DIR, PROCESSED_DATA_DIR
    from rapido_intelligent_system.features_mnb import (
        load_all_csvs,
        merge_datasets,
        engineer_features,
        encode_categoricals,
        build_features_for_target,
        save_features,
    )


@app.cell
def _():
    data = load_all_csvs(RAW_DATA_DIR)
    return (data,)


@app.cell
def _(data):
    df_merged = merge_datasets(
        data["bookings"],
        data["customers"],
        data["drivers"],
        data["location_demand"],
        data["time_features"],
    )
    return (df_merged,)


@app.cell
def _(df_merged):
    df_eng = engineer_features(df_merged)
    return (df_eng,)


@app.cell
def _(df_eng):
    categorical_cols = [
        "city", "pickup_location", "drop_location", "vehicle_type",
        "traffic_level", "weather_condition", "day_of_week",
        "customer_gender", "customer_city", "preferred_vehicle_type",
        "driver_city", "season",
    ]
    df_enc, _encoders = encode_categoricals(df_eng, categorical_cols)
    return (df_enc,)


@app.cell
def _(df_enc):
    le_status = LabelEncoder()
    df_enc["booking_status_enc"] = le_status.fit_transform(df_enc["booking_status"].astype(str))
    return


@app.cell
def _(df_enc):
    drop_cols = [
        "booking_id", "booking_date", "booking_time", "booking_datetime",
        "date_key", "incomplete_ride_reason", "customer_id", "driver_id",
        "booking_status",
    ]
    df_clean = df_enc.drop(columns=[c for c in drop_cols if c in df_enc.columns], errors="ignore")
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
    return (df_clean,)


@app.cell
def _(df_clean):
    out_bv, mi_bv = build_features_for_target(
        df_clean, target_col="booking_value", task_type="regression",
        leakage_cols=["booking_value"],
    )
    save_features(out_bv, PROCESSED_DATA_DIR, "features_booking_value.csv")
    return (mi_bv,)


@app.cell
def _(df_clean):
    out_bs, mi_bs = build_features_for_target(
        df_clean, target_col="booking_status_enc", task_type="classification",
        leakage_cols=["booking_status_enc"],
    )
    save_features(out_bs, PROCESSED_DATA_DIR, "features_booking_status.csv")
    return (mi_bs,)


@app.cell
def _(df_clean):
    out_cc, mi_cc = build_features_for_target(
        df_clean, target_col="customer_cancel_flag", task_type="classification",
        leakage_cols=["customer_cancel_flag"],
    )
    save_features(out_cc, PROCESSED_DATA_DIR, "features_customer_cancel_flag.csv")
    return (mi_cc,)


@app.cell
def _(df_clean):
    out_dd, mi_dd = build_features_for_target(
        df_clean, target_col="driver_delay_flag", task_type="classification",
        leakage_cols=["driver_delay_flag"],
    )
    save_features(out_dd, PROCESSED_DATA_DIR, "features_driver_delay_flag.csv")
    return (mi_dd,)


@app.cell
def _(mi_bs, mi_bv, mi_cc, mi_dd):
    for name, mi_df in [
        ("booking_value", mi_bv),
        ("booking_status", mi_bs),
        ("customer_cancel_flag", mi_cc),
        ("driver_delay_flag", mi_dd),
    ]:
        mo.md(f"### Top 10 MI Scores — {name}")
        mo.ui.table(mi_df.head(10))
    return


if __name__ == "__main__":
    app.run()
