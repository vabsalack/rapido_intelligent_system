import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import polars as pl
    import altair as alt
    from rapido_intelligent_system.config_mnb import (
        RAW_DATA_DIR, PROCESSED_DATA_DIR, FIGURES_DIR
    )
    from rapido_intelligent_system.dataset_mnb import (
        load_all_raw, eda_sample, shape_overview
    )
    from rapido_intelligent_system.plots_mnb import (
        missing_summary, numeric_summary, category_frequency_chart,
        numeric_histogram, stacked_target_chart
    )


@app.cell
def _():
    mo.md(r"""
    # 0.01 — EDA on Raw Files

    **Project:** Rapido Intelligent System — build predictive models from ride-hailing data.
    **Author:** kittu · **Stage:** initial exploration of the 5 raw datasets.

    This notebook explores `data/raw/` to understand **data quality**, **distributions** and
    **feature ↔ target relationships** before feature engineering and modeling.
    No models are trained here.

    **4 modeling targets** (reserved for later notebooks):

    | Target | Source dataset | Task |
    |---|---|---|
    | `booking_status` | bookings | classification (Completed / Cancelled / Incomplete) |
    | `booking_value` | bookings | regression (₹ fare) |
    | `customer_cancel_flag` | customers | binary classification |
    | `driver_delay_flag` | drivers | binary classification |

    Datasets: **bookings** (100k rides) · **customers** (10k) · **drivers** (5k)
    · **location_demand** (~18k demand cells) · **time_features** (8,760 hourly rows).

    > Each section records the steps taken and the insights observed. A consolidated
    > summary with next-step planning lives at the end of the notebook.
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ## Approach & tooling

    - **Polars** for fast, lazy-friendly data handling (replacing pandas).
    - **Altair** for declarative, interactive charts (replacing matplotlib/seaborn).
    - Charts built by `plots_mnb.py` reuse a **seeded sample capped at 4k rows**
      so they stay under Altair's 5k-row default limit and render quickly.
    - All raw tables are small enough to analyze in full; **sampling is applied only to
      chart inputs**, and nothing is written to disk during exploration.

    Reusable helpers live in the project's marimo modules:
    `dataset_mnb.py` (`load_all_raw`, `eda_sample`, `shape_overview`) and
    `plots_mnb.py` (`missing_summary`, `numeric_summary`, `category_frequency_chart`,
    `numeric_histogram`, `stacked_target_chart`).
    """)
    return


@app.cell
def _():
    mo.md(f"""
    ## Project paths

    - **Raw data:** `{RAW_DATA_DIR}`
    - **Processed data:** `{PROCESSED_DATA_DIR}`
    - **Figures:** `{FIGURES_DIR}`
    """)
    return


@app.cell
def _():
    use_sample = mo.ui.switch(
        value=True,
        label="Use random samples for chart inputs (turn OFF to plot full data)",
    )
    use_sample
    return (use_sample,)


@app.cell
def _():
    mo.md(r"""
    ## 1. Data loading & overview

    We load all five CSVs once into a dict-of-DataFrames, then inspect shapes,
    column types and missing values. This fixes the **data quality baseline** that
    every later modeling step depends on.
    """)
    return


@app.cell
def _():
    raw_files = load_all_raw(RAW_DATA_DIR)
    return (raw_files,)


@app.cell
def _(shape_overview, raw_files):
    overview = shape_overview(raw_files)
    overview
    return (overview,)


@app.cell
def _(pl, raw_files):
    profile_rows = []
    for name, df in raw_files.items():
        n_num = len(df.select(pl.selectors.numeric()).columns)
        n_cat = len(df.select(pl.selectors.string()).columns)
        profile_rows.append(
            {
                "file": name,
                "rows": df.height,
                "cols": df.width,
                "numeric_cols": n_num,
                "categorical_cols": n_cat,
                "other_cols": df.width - n_num - n_cat,
            }
        )
    profile = pl.DataFrame(profile_rows)
    profile
    return (profile,)


@app.cell
def _(missing_summary, pl, raw_files):
    missing_parts = [
        missing_summary(df).with_columns(pl.lit(name).alias("file"))
        for name, df in raw_files.items()
    ]
    missing_table = (
        pl.concat(missing_parts)
        .filter(pl.col("null_count") > 0)
        .sort("null_pct", descending=True)
    )
    missing_table
    return (missing_table,)


@app.cell
def _():
    mo.md(r"""
    ### Reading the overview

    - All five tables have **no structural surprises**: shapes are `100k / 10k / 5k / ~18k / 8,760`.
    - **Only two columns carry missing values**, both in `bookings`:
      - `actual_ride_time_min` — **31.7% null**: cancelled/incomplete rides never record a ride time.
      - `incomplete_ride_reason` — **91.6% null**: only the 8,370 `Incomplete` rides have a reason.
    - `booking_date`, `booking_time` and `time_features.datetime` are read as **strings**;
      they must be parsed to proper temporal types during feature engineering.
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ## 2. bookings — targets: `booking_status` and `booking_value`

    Rides at booking level. We study:
    1. the two target columns and their distributions,
    2. ride context (city / vehicle / traffic / weather),
    3. time patterns (hour / day-of-week / weekend),
    4. whether context drives the **outcome** (`booking_status`),
    5. what drives **fares** (`booking_value`),
    6. correlations among numeric features.

    A seeded sample of the rides is used for charts; full data is kept for table summaries.
    """)
    return


@app.cell
def _(eda_sample, raw_files, use_sample):
    bookings_full = raw_files["bookings"]
    bookings = (
        eda_sample(bookings_full, n=50_000, seed=42)
        if use_sample.value
        else bookings_full
    )
    return (bookings, bookings_full)


@app.cell
def _():
    mo.md(r"""
    ### 2.1 The target columns

    `booking_status` is the main click-away target — check the class balance. `booking_value`
    is the regression target — check for skew and the mean/median gap.
    """)
    return


@app.cell
def _(category_frequency_chart, bookings, mo, numeric_histogram):
    status_chart = category_frequency_chart(
        bookings, "booking_status", title="booking_status — overall balance"
    )
    value_hist = numeric_histogram(
        bookings, "booking_value", bins=40, title="booking_value — distribution (₹)"
    )
    mo.hstack([status_chart, value_hist])
    return (status_chart, value_hist)


@app.cell
def _(bookings, pl):
    value_by_status = (
        bookings.group_by("booking_status")
        .agg(
            pl.len().alias("n"),
            pl.col("booking_value").mean().round(2).alias("mean_value"),
            pl.col("booking_value").median().round(2).alias("median_value"),
        )
        .sort("mean_value", descending=True)
    )
    value_by_status
    return (value_by_status,)


@app.cell
def _():
    mo.md(r"""
    ### 2.2 Ride context

    Are cities, vehicle types, traffic and weather evenly represented, or is the data skewed?
    """)
    return


@app.cell
def _(bookings, category_frequency_chart, mo):
    city_chart = category_frequency_chart(bookings, "city", title="Bookings by city")
    vehicle_chart = category_frequency_chart(bookings, "vehicle_type", title="Bookings by vehicle type")
    traffic_chart = category_frequency_chart(bookings, "traffic_level", title="Bookings by traffic level")
    weather_chart = category_frequency_chart(bookings, "weather_condition", title="Bookings by weather")
    mo.vstack(
        [
            mo.hstack([city_chart, vehicle_chart]),
            mo.hstack([traffic_chart, weather_chart]),
        ]
    )
    return (city_chart, vehicle_chart, traffic_chart, weather_chart)


@app.cell
def _():
    mo.md(r"""
    ### 2.3 Time patterns

    Ride demand over the day, across the week, and on weekends. This informs the time-based
    features we may engineer (hour-of-day, day-of-week, weekend flag).
    """)
    return


@app.cell
def _(alt, bookings, mo, pl):
    hourly_counts = bookings.group_by("hour_of_day").len().sort("hour_of_day")
    hourly_chart = (
        alt.Chart(hourly_counts, title="Bookings by hour of day")
        .mark_line(point=True)
        .encode(
            x=alt.X("hour_of_day:Q", title="Hour of day"),
            y=alt.Y("len:Q", title="Bookings"),
        )
    )
    _dow_order = pl.DataFrame(
        {
            "day_of_week": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            "dow_num": [0, 1, 2, 3, 4, 5, 6],
        }
    )
    dow_counts = (
        bookings.group_by("day_of_week")
        .len()
        .join(_dow_order, on="day_of_week")
        .sort("dow_num")
    )
    dow_chart = (
        alt.Chart(dow_counts, title="Bookings by day of week")
        .mark_bar()
        .encode(
            x=alt.X("day_of_week:N", title="Day of week"),
            y=alt.Y("len:Q", title="Bookings"),
        )
    )
    weekend_counts = (
        bookings.with_columns(
            pl.when(pl.col("is_weekend") == 1)
            .then(pl.lit("Weekend"))
            .otherwise(pl.lit("Weekday"))
            .alias("day_type")
        )
        .group_by("day_type")
        .len()
    )
    weekend_chart = (
        alt.Chart(weekend_counts, title="Weekday vs weekend volume")
        .mark_bar()
        .encode(
            x=alt.X("day_type:N", title=""),
            y=alt.Y("len:Q", title="Bookings"),
        )
    )
    mo.vstack([hourly_chart, mo.hstack([dow_chart, weekend_chart])])
    return (hourly_chart, dow_chart, weekend_chart)


@app.cell
def _():
    mo.md(r"""
    ### 2.4 Does ride context drive the outcome?

    Compare the **composition of `booking_status`** within each category. A wide spread of
    "Cancelled" share across categories means the category is predictive.
    """)
    return


@app.cell
def _(bookings, mo, stacked_target_chart):
    status_traffic = stacked_target_chart(
        bookings, "traffic_level", "booking_status"
    )
    status_weather = stacked_target_chart(
        bookings, "weather_condition", "booking_status"
    )
    status_vehicle = stacked_target_chart(
        bookings, "vehicle_type", "booking_status"
    )
    mo.vstack(
        [
            mo.hstack([status_traffic, status_weather]),
            status_vehicle,
        ]
    )
    return (status_traffic, status_weather, status_vehicle)


@app.cell
def _():
    mo.md(r"""
    ### 2.5 What drives `booking_value`?

    Fares by vehicle type, surge, and distance. Note `booking_value` = `base_fare × surge_multiplier`
    by construction, so expect tight relationships.
    """)
    return


@app.cell
def _(alt, bookings, eda_sample, mo, pl):
    _plot_data = eda_sample(bookings, n=4_000, seed=42)
    bv_vehicle = (
        bookings.group_by("vehicle_type")
        .agg(pl.col("booking_value").median().round(2).alias("median_value"))
        .sort("median_value", descending=True)
    )
    vehicle_value = (
        alt.Chart(bv_vehicle, title="Median booking_value by vehicle type")
        .mark_bar()
        .encode(
            x=alt.X("vehicle_type:N", title="Vehicle type", sort="-y"),
            y=alt.Y("median_value:Q", title="Median ₹"),
        )
    )
    surge_scatter = (
        alt.Chart(_plot_data, title="surge_multiplier vs booking_value")
        .mark_circle(opacity=0.15, size=40)
        .encode(
            x=alt.X("surge_multiplier:Q", title="Surge multiplier"),
            y=alt.Y("booking_value:Q", title="Booking value (₹)"),
        )
    )
    distance_scatter = (
        alt.Chart(_plot_data, title="ride_distance_km vs booking_value")
        .mark_circle(opacity=0.15, size=40)
        .encode(
            x=alt.X("ride_distance_km:Q", title="Distance (km)"),
            y=alt.Y("booking_value:Q", title="Booking value (₹)"),
        )
    )
    mo.hstack([vehicle_value, surge_scatter, distance_scatter])
    return (vehicle_value, surge_scatter, distance_scatter)


@app.cell
def _():
    mo.md(r"""
    ### 2.6 Correlations among numeric features

    A compact view of linear relationships. `actual_ride_time_min` carries nulls, so its
    correlations are not reliable — treat accordingly during imputation.
    """)
    return


@app.cell
def _(alt, bookings, eda_sample, mo, pl):
    corr_data = (
        eda_sample(bookings, n=10_000, seed=42)
        .select(pl.selectors.numeric())
        .drop("is_weekend")
    )
    _numeric_cols = corr_data.columns
    corr = corr_data.corr()
    corr = corr.with_columns(pl.Series("x_names", _numeric_cols))
    corr_long = (
        corr.unpivot(
            index="x_names", on=_numeric_cols, variable_name="y", value_name="value"
        )
        .with_columns(pl.col("value").round(2))
    )
    corr_heat = (
        alt.Chart(corr_long, title="Correlation matrix — numeric bookings features")
        .mark_rect()
        .encode(
            x=alt.X("x_names:N", sort=_numeric_cols, title=""),
            y=alt.Y("y:N", sort=_numeric_cols, title=""),
            color=alt.Color(
                "value:Q",
                scale=alt.Scale(scheme="blueorange", domain=[-1, 1]),
                title="corr",
            ),
            tooltip=["x_names", "y", "value"],
        )
        .properties(width=540, height=540)
    )
    corr_heat
    return (corr_heat,)


@app.cell
def _():
    mo.md(r"""
    ### bookings — insights so far

    **Target `booking_status`** (full data): Completed **68.3%** (68,346) · Cancelled **23.3%**
    (23,284) · Incomplete **8.4%** (8,370). Imbalanced, but workable.

    **Target `booking_value`**: mean ₹336, median ₹290, range **₹27–1,266**, right-skewed.
    Cancelled rides tend to be pricier (mean ₹369 vs ₹323 for Completed) — high-fare, high-stress
    rides cancel more.

    **Context:** cities, vehicles, traffic and weather are each near-perfectly balanced
    (~33k / ~20k / ~33k across levels) — the synthetic generator is uniform.

    **Strong outcome signals** (cancellation share by category):
    - Traffic **High 39.3%** vs Low/Medium ≈ 19%.
    - Weather **Heavy Rain 36.8%**, Rain 28.4%, **Clear 10.9%**.
    - `vehicle_type` and `city` show **no** meaningful difference (≈ 25% everywhere).

    **Time:** demand is flat across hours (~4.1k/hour) and days; no meaningful weekend or
    time-of-day effect on cancellations.

    **Fare drivers:** `booking_value` correlates strongly with `base_fare` (0.92),
    `ride_distance_km` (0.67) and `estimated_ride_time_min` (0.65), mildly with
    `surge_multiplier` (0.33), and not at all with `hour_of_day`.

    **Data quality:** `actual_ride_time_min` (31.7% null) and `incomplete_ride_reason`
    (91.6% null) require imputation or explicit "missing" flags in modeling.
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ## 3. customers — target: `customer_cancel_flag`

    Static customer profiles (10k rows). The target is a **binary flag** built from a customer's
    historical cancellation behaviour — we check balance, demographics, and which historical
    features separate cancelling customers.
    """)
    return


@app.cell
def _():
    customers = raw_files["customers"]
    return (customers,)


@app.cell
def _():
    mo.md(r"""
    ### 3.1 Customer profile & target balance

    How balanced is the flag, and how are age, historical cancellation rate and rating distributed?
    """)
    return


@app.cell
def _(category_frequency_chart, customers, mo, numeric_histogram):
    cust_flag_chart = category_frequency_chart(
        customers, "customer_cancel_flag", title="customer_cancel_flag balance"
    )
    cust_age_hist = numeric_histogram(customers, "customer_age", bins=30, title="Customer age")
    cust_rate_hist = numeric_histogram(
        customers, "cancellation_rate", bins=30, title="Historical cancellation_rate"
    )
    cust_rating_hist = numeric_histogram(
        customers, "avg_customer_rating", bins=25, title="avg_customer_rating"
    )
    mo.vstack(
        [
            mo.hstack([cust_flag_chart, cust_age_hist]),
            mo.hstack([cust_rate_hist, cust_rating_hist]),
        ]
    )
    return (cust_flag_chart, cust_age_hist, cust_rate_hist, cust_rating_hist)


@app.cell
def _():
    mo.md(r"""
    ### 3.2 Demographics vs cancellation

    Does gender, city or preferred vehicle change the cancellation rate?
    """)
    return


@app.cell
def _(customers, mo, stacked_target_chart):
    gender_chart = stacked_target_chart(
        customers, "customer_gender", "customer_cancel_flag"
    )
    cust_city_chart = stacked_target_chart(
        customers, "customer_city", "customer_cancel_flag"
    )
    cust_vehicle_chart = stacked_target_chart(
        customers, "preferred_vehicle_type", "customer_cancel_flag"
    )
    mo.vstack([mo.hstack([gender_chart, cust_city_chart]), cust_vehicle_chart])
    return (gender_chart, cust_city_chart, cust_vehicle_chart)


@app.cell
def _():
    mo.md(r"""
    ### 3.3 Numeric separators

    Which numeric columns separate flag = 0 from flag = 1? This also reveals whether some
    columns are too close to the target (leakage risk).
    """)
    return


@app.cell
def _(alt, customers, mo, numeric_summary, pl):
    cust_num_overview = numeric_summary(customers)
    rate_by_flag = (
        customers.group_by("customer_cancel_flag")
        .agg(pl.col("cancellation_rate").mean().round(3).alias("mean_rate"))
        .sort("customer_cancel_flag")
        .with_columns(
            pl.when(pl.col("customer_cancel_flag") == 1)
            .then(pl.lit("flag=1"))
            .otherwise(pl.lit("flag=0"))
            .alias("group")
        )
    )
    rate_chart = (
        alt.Chart(rate_by_flag, title="Mean cancellation_rate by flag")
        .mark_bar()
        .encode(
            x=alt.X("group:N", title=""),
            y=alt.Y("mean_rate:Q", title="Mean cancellation_rate"),
        )
    )
    mo.hstack([rate_chart, cust_num_overview])
    return (cust_num_overview, rate_chart)


@app.cell
def _():
    mo.md(r"""
    ### customers — insights so far

    **Target balance:** flag = 1 at **53.4%** (5,343) vs 46.6% — nicely balanced.

    **What separates cancelling customers:**
    - `cancellation_rate` is the standout: mean **0.34** (flag=1) vs **0.11** (flag=0).
    - Median `cancelled_rides` 3.4 vs 1.2; `total_bookings` barely differs (10.2 vs 9.8).
    - Age, signup age, rating, gender, city and preferred vehicle show **no** meaningful separation.

    **⚠ Leakage warning:** `customer_cancel_flag` is almost certainly **derived from**
    `cancellation_rate` / `cancelled_rides`. Using those fields to predict the flag would give
    an unrealistically perfect model. Feature selection should **exclude** historical
    cancellation aggregates from the customer model.
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ## 4. drivers — target: `driver_delay_flag`

    Static driver profiles (5k rows). Binary target; only **13%** positive, so we also check
    class imbalance and which performance columns separate delayed drivers.
    """)
    return


@app.cell
def _():
    drivers = raw_files["drivers"]
    return (drivers,)


@app.cell
def _():
    mo.md(r"""
    ### 4.1 Fleet profile & target balance

    Age, experience and pickup-delay distributions; how imbalanced is `driver_delay_flag`?
    """)
    return


@app.cell
def _(category_frequency_chart, drivers, mo, numeric_histogram):
    drv_flag_chart = category_frequency_chart(
        drivers, "driver_delay_flag", title="driver_delay_flag balance"
    )
    drv_age_hist = numeric_histogram(drivers, "driver_age", bins=30, title="Driver age")
    drv_exp_hist = numeric_histogram(
        drivers, "driver_experience_years", bins=25, title="Experience (years)"
    )
    drv_pickup_hist = numeric_histogram(
        drivers, "avg_pickup_delay_min", bins=30, title="Avg pickup delay (min)"
    )
    mo.vstack(
        [
            mo.hstack([drv_flag_chart, drv_age_hist]),
            mo.hstack([drv_exp_hist, drv_pickup_hist]),
        ]
    )
    return (drv_flag_chart, drv_age_hist, drv_exp_hist, drv_pickup_hist)


@app.cell
def _():
    mo.md(r"""
    ### 4.2 Fleet context vs delay

    Is delay concentrated in certain cities or vehicle types?
    """)
    return


@app.cell
def _(drivers, mo, stacked_target_chart):
    drv_city_chart = stacked_target_chart(drivers, "driver_city", "driver_delay_flag")
    drv_vehicle_chart = stacked_target_chart(
        drivers, "vehicle_type", "driver_delay_flag"
    )
    mo.hstack([drv_city_chart, drv_vehicle_chart])
    return (drv_city_chart, drv_vehicle_chart)


@app.cell
def _():
    mo.md(r"""
    ### 4.3 Numeric separators

    Which driver-performance columns change with the flag?
    """)
    return


@app.cell
def _(alt, drivers, mo, numeric_summary, pl):
    drv_num_overview = numeric_summary(drivers)
    drv_stats = (
        drivers.group_by("driver_delay_flag")
        .agg(
            pl.col("delay_rate").mean().round(3).alias("mean_delay_rate"),
            pl.col("avg_pickup_delay_min").mean().round(2).alias("mean_pickup_delay"),
            pl.col("acceptance_rate").mean().round(3).alias("mean_acceptance"),
        )
        .sort("driver_delay_flag")
        .with_columns(
            pl.when(pl.col("driver_delay_flag") == 1)
            .then(pl.lit("flag=1"))
            .otherwise(pl.lit("flag=0"))
            .alias("group")
        )
    )
    delay_rate_chart = (
        alt.Chart(drv_stats, title="Mean delay_rate by flag")
        .mark_bar()
        .encode(
            x=alt.X("group:N", title=""),
            y=alt.Y("mean_delay_rate:Q", title="Mean delay_rate"),
        )
    )
    pickup_chart = (
        alt.Chart(drv_stats, title="Mean avg_pickup_delay by flag")
        .mark_bar()
        .encode(
            x=alt.X("group:N", title=""),
            y=alt.Y("mean_pickup_delay:Q", title="Mean pickup delay (min)"),
        )
    )
    mo.vstack([mo.hstack([delay_rate_chart, pickup_chart, drv_num_overview])])
    return (drv_num_overview, delay_rate_chart, pickup_chart)


@app.cell
def _():
    mo.md(r"""
    ### drivers — insights so far

    **Target balance:** flag = 1 at only **13.1%** (654 / 5,000) — imbalanced; modeling will
    need class-handling and careful metrics (PR-AUC over accuracy).

    **What separates delayed drivers:**
    - `delay_rate`: mean **0.14** (flag=1) vs **0.03** (flag=0) — near-leakage.
    - `avg_pickup_delay_min`: mean **4.5 min** vs **3.0 min** — meaningful but weaker.
    - `acceptance_rate`: 0.79 vs 0.76 — weak.
    - Age, experience, rating, city and vehicle show **no** material separation.

    **⚠ Leakage warning (same story as customers):** `delay_rate` is derived from
    `delay_count / total_assigned_rides` and closely mirrors the flag — exclude it (and the
    `delay_count`) from the driver model inputs.
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ## 5. location_demand — supporting table

    17,941 rows, one per (city, pickup_location, hour, vehicle) combo. No target here — it is
    a **pre-aggregated view of ride demand** we can later enrich bookings with.
    """)
    return


@app.cell
def _():
    loc_demand = raw_files["location_demand"]
    return (loc_demand,)


@app.cell
def _(alt, category_frequency_chart, loc_demand, mo, numeric_histogram, pl):
    demand_chart = category_frequency_chart(
        loc_demand, "demand_level", title="demand_level distribution"
    )
    ld_by_vehicle = (
        loc_demand.group_by("vehicle_type")
        .agg(pl.col("total_requests").sum().alias("requests"))
        .sort("requests", descending=True)
    )
    ld_vehicle_chart = (
        alt.Chart(ld_by_vehicle, title="Total requests by vehicle type")
        .mark_bar()
        .encode(
            x=alt.X("vehicle_type:N", sort="-y"),
            y=alt.Y("requests:Q"),
        )
    )
    ld_by_hour = (
        loc_demand.group_by("hour_of_day")
        .agg(pl.col("total_requests").sum().alias("requests"))
        .sort("hour_of_day")
    )
    ld_hour_chart = (
        alt.Chart(ld_by_hour, title="Requests by hour")
        .mark_line(point=True)
        .encode(
            x=alt.X("hour_of_day:Q"),
            y=alt.Y("requests:Q"),
        )
    )
    ld_surge_hist = numeric_histogram(
        loc_demand, "avg_surge_multiplier", bins=30, title="avg_surge_multiplier"
    )
    ld_wait_by_demand = (
        loc_demand.group_by("demand_level")
        .agg(
            pl.col("avg_wait_time_min").mean().round(1).alias("avg_wait_min"),
            pl.col("avg_surge_multiplier").mean().round(3).alias("avg_surge"),
        )
        .sort("demand_level")
    )
    mo.vstack(
        [
            mo.hstack([demand_chart, ld_vehicle_chart, ld_hour_chart]),
            mo.hstack([ld_surge_hist, ld_wait_by_demand]),
        ]
    )
    return (demand_chart, ld_vehicle_chart, ld_hour_chart, ld_surge_hist)


@app.cell
def _():
    mo.md(r"""
    ### location_demand — insights so far

    - Only **Low** (9,249) and **Medium** (8,692) demand levels exist — **no "High"**; demand
      cells carry 2–15 requests (mean ≈ 5.6).
    - `avg_wait_time_min` averages are strikingly high (mean ≈ 61 min); `avg_surge_multiplier`
      mean ≈ 1.59. Surge is **high where demand is high** — expected demand curve.
    - City-level totals in `location_demand` **exactly match** bookings counts per city
      (e.g. Delhi 20,161) — this table looks like a **derived aggregation of bookings**, not
      independent data. Model-time use: enrich bookings on (city, location, hour, vehicle).
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ## 6. time_features — supporting table

    8,760 hourly rows for the full year 2025 (24h × 365d). Provides season, holiday and peak-time
    flags to enrich bookings by date-time. No target.
    """)
    return


@app.cell
def _(alt, mo, pl, raw_files):
    time_features = raw_files["time_features"]
    tf_overview = pl.DataFrame(
        {
            "metric": [
                "rows",
                "start",
                "end",
                "peak_time share (%)",
                "holiday share (%)",
            ],
            "value": [
                str(time_features.height),
                str(time_features["datetime"].min()),
                str(time_features["datetime"].max()),
                str(round(time_features["peak_time_flag"].mean() * 100, 1)),
                str(round(time_features["is_holiday"].mean() * 100, 1)),
            ],
        }
    )
    season_counts = time_features.group_by("season").len().sort("len", descending=True)
    peak_by_hour = (
        time_features.group_by("hour_of_day")
        .agg(pl.col("peak_time_flag").mean().alias("peak_share"))
        .sort("hour_of_day")
    )
    peak_chart = (
        alt.Chart(peak_by_hour, title="Share of hours flagged as peak time")
        .mark_line(point=True)
        .encode(
            x=alt.X("hour_of_day:Q", title="Hour of day"),
            y=alt.Y("peak_share:Q", title="Peak-time share"),
        )
    )
    mo.hstack([tf_overview, season_counts, peak_chart])
    return (time_features, tf_overview, season_counts, peak_chart)


@app.cell
def _():
    mo.md(r"""
    ### time_features — insights so far

    - Full calendar year 2025; each hour appears exactly 365 times.
    - Seasons balanced: Summer 2,952 · Monsoon 2,928 · Winter 2,880.
    - `peak_time_flag` marks **29.2%** of hours and is identical across seasons (~0.29),
      but varies by hour (see chart) — a useful ride-time feature.
    - **`is_holiday` is all zeros — zero variance**, so it carries no predictive value;
      drop it during feature engineering. (`datetime` is also read as a string — parse it.)
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ## 7. Cross-dataset relationships

    All 10k customers and 5k drivers appear in bookings (full coverage), so we can join freely.
    We verify the strongest cross-signal: do customers already flagged as frequent cancellers
    actually cancel more of their bookings?
    """)
    return


@app.cell
def _(alt, bookings, customers, mo, pl):
    cancel_share = (
        bookings.join(
            customers.select(["customer_id", "customer_cancel_flag"]),
            on="customer_id",
            how="left",
        )
        .filter(pl.col("booking_status").is_in(["Completed", "Cancelled"]))
        .group_by(["customer_cancel_flag", "booking_status"])
        .len()
        .with_columns(
            (pl.col("len") / pl.col("len").sum().over("customer_cancel_flag") * 100)
            .round(1)
            .alias("share_pct")
        )
        .with_columns(
            pl.when(pl.col("customer_cancel_flag") == 1)
            .then(pl.lit("flag=1"))
            .otherwise(pl.lit("flag=0"))
            .alias("group")
        )
    )
    cancel_share_chart = (
        alt.Chart(cancel_share, title="Booking outcome by customer_cancel_flag")
        .mark_bar()
        .encode(
            x=alt.X("group:N", title="Customer flag"),
            y=alt.Y("share_pct:Q", title="Share of bookings (%)"),
            color=alt.Color("booking_status:N", title="Booking status"),
        )
    )
    cancel_share_chart
    return (cancel_share_chart,)


@app.cell
def _(pl, raw_files):
    coverage_table = pl.DataFrame(
        {
            "entity": ["customers referenced in bookings", "drivers referenced in bookings"],
            "unique_ids_in_bookings": [
                raw_files["bookings"]["customer_id"].n_unique(),
                raw_files["bookings"]["driver_id"].n_unique(),
            ],
            "rows_in_lookup_table": [
                raw_files["customers"].height,
                raw_files["drivers"].height,
            ],
        }
    )
    incomplete_reasons = (
        raw_files["bookings"]
        .filter(pl.col("booking_status") == "Incomplete")
        .group_by("incomplete_ride_reason")
        .len()
        .sort("len", descending=True)
        .rename({"len": "count"})
    )
    coverage_table
    return (coverage_table, incomplete_reasons)


@app.cell
def _():
    mo.md(r"""
    ### Cross-dataset — insights so far

    - **Flag = 1 customers cancel 35.3% of their (sampled) bookings vs 12.9% for flag = 0** —
      customer history clearly carries ride-level cancellation signal.
    - Full key coverage (10k/10k customers, 5k/5k drivers) means joins are loss-less;
      `drivers.driver_delay_flag` / `delay_rate` can enrich ride-level data too.
    - `incomplete_ride_reason` breakdown (of 8,370 incomplete rides): **Driver Delay 4,728**,
      Vehicle Issue 1,265, App Issue 1,221, Customer No-show 1,156 — the link to the driver
      model, though "Driver Delay" is only observable on incomplete rides.
    - Every ride row (even Cancelled) carries a `driver_id` and `customer_id`; missing-value
      analysis above already flagged the only true nulls.
    """)
    return


@app.cell
def _(pl, raw_files):
    b = raw_files["bookings"]
    _status = (
        b.group_by("booking_status")
        .len()
        .with_columns((pl.col("len") / b.height * 100).round(1).alias("pct"))
    )
    _customers = raw_files["customers"]
    _drivers = raw_files["drivers"]
    _cust = (
        _customers.group_by("customer_cancel_flag")
        .len()
        .with_columns((pl.col("len") / _customers.height * 100).round(1).alias("pct"))
    )
    _drv = (
        _drivers.group_by("driver_delay_flag")
        .len()
        .with_columns((pl.col("len") / _drivers.height * 100).round(1).alias("pct"))
    )

    def _pairs(df, key_col):
        return {
            r[key_col]: (int(r["len"]), float(r["pct"]))
            for r in df.iter_rows(named=True)
        }

    _bv_mean = round(float(b["booking_value"].mean()), 1)
    _cancel_high = round(
        float(
            b.filter(pl.col("traffic_level") == "High")
            .filter(pl.col("booking_status") == "Cancelled")
            .height
        )
        / b.filter(pl.col("traffic_level") == "High").height
        * 100,
        1,
    )
    _cancel_clear = round(
        float(
            b.filter(pl.col("weather_condition") == "Clear")
            .filter(pl.col("booking_status") == "Cancelled")
            .height
        )
        / b.filter(pl.col("weather_condition") == "Clear").height
        * 100,
        1,
    )
    snapshot = {
        "status": _pairs(_status, "booking_status"),
        "customer_flag": _pairs(_cust, "customer_cancel_flag"),
        "driver_flag": _pairs(_drv, "driver_delay_flag"),
        "bv_mean": _bv_mean,
        "cancel_high_traffic": _cancel_high,
        "cancel_clear_weather": _cancel_clear,
    }
    return (snapshot,)


@app.cell
def _(mo, snapshot):
    mo.md(f"""
    ## 8. Summary & next steps

    ### Raw-data snapshot (full data)
    - `booking_status`: Completed **{snapshot['status']['Completed'][0]:,}**
      ({snapshot['status']['Completed'][1]}%) · Cancelled **{snapshot['status']['Cancelled'][0]:,}**
      ({snapshot['status']['Cancelled'][1]}%) · Incomplete **{snapshot['status']['Incomplete'][0]:,}**
      ({snapshot['status']['Incomplete'][1]}%).
    - `customer_cancel_flag` = 1: **{snapshot['customer_flag'][1][0]:,}**
      ({snapshot['customer_flag'][1][1]}%) — balanced.
    - `driver_delay_flag` = 1: **{snapshot['driver_flag'][1][0]:,}**
      ({snapshot['driver_flag'][1][1]}%) — imbalanced (~1 in 8).
    - `booking_value` mean ≈ **₹{snapshot['bv_mean']}**.

    ### Headline insights
    1. **bookings `booking_status`** is the most workable target; driving context does the work —
       cancellation share **{snapshot['cancel_high_traffic']}%** under High traffic vs ≈19% Low/Medium,
       and {snapshot['cancel_clear_weather']}% in Clear weather vs 36.8% in Heavy Rain.
    2. **Fare** (`booking_value`) is a near-deterministic function of `base_fare`, distance and
       estimated time — strong regression signal, right-skewed target.
    3. **Customers & drivers** have clean (leak-prone) historical features and **no demographic
       signal**; class imbalance only matters for the driver model.
    4. `location_demand` / `time_features` are ready-made enrichment tables (demand cells,
       season, peak-time) with tight joins to bookings; `is_holiday` should be dropped
       (zero variance) and all date-time strings parsed.

    ### Proposed next step — **Feature selection & 4 modeling datasets**
    Build a feature pipeline that produces **four per-target datasets** from the raw sources:
    - **`booking_status` (multi-class)** — bookings + ride context + location demand + time flags;
      exclude leaks like `incomplete_ride_reason`.
    - **`booking_value` (regression)** — same base features; drop actual-time/null-heavy columns.
    - **`customer_cancel_flag` (binary)** — demographics + lifecycle features; **exclude**
      `cancellation_rate`, `cancelled_rides` (target-derived).
    - **`driver_delay_flag` (binary)** — fleet profile + performance; **exclude** `delay_rate`,
      `delay_count` (target-derived).

    After that: label-encoding, train/val split, baseline ML (logistic regression / decision tree),
    then metric-backed model selection per target.
    """)
    return


if __name__ == "__main__":
    app.run()