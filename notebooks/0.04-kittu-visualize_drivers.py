import marimo

__generated_with = "0.23.16"
app = marimo.App(width="wide")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go

    from rapido_intelligent_system.dataset import RAW_DATA_DIR

    px.defaults.template = "plotly_white"
    return RAW_DATA_DIR, mo, pd, px


@app.cell
def _(RAW_DATA_DIR, pd):
    drivers_path = RAW_DATA_DIR / "drivers.csv"
    drivers_df = pd.read_csv(drivers_path)

    drivers_df["delay_label"] = drivers_df["driver_delay_flag"].map(
        {1: "Delayed", 0: "On-time"}
    )
    return (drivers_df,)


@app.cell
def _():
    delay_colors = {"Delayed": "#d62728", "On-time": "#2ca02c"}
    return (delay_colors,)


@app.cell
def _(mo):
    mo.md(r"""
    # Rapido Drivers — EDA Dashboard

    Interactive exploratory analysis of 5k ride-hailing drivers across India.
    Use the sidebar filters to slice the data; every chart, card and table updates reactively.
    """)
    return


@app.cell
def _(drivers_df, mo):
    city_selector = mo.ui.multiselect(
        drivers_df["driver_city"].sort_values().unique().tolist(),
        value=drivers_df["driver_city"].sort_values().unique().tolist(),
        label="Cities",
        full_width=False,
    )
    vehicle_selector = mo.ui.multiselect(
        drivers_df["vehicle_type"].unique().tolist(),
        value=drivers_df["vehicle_type"].unique().tolist(),
        label="Vehicle types",
        full_width=False,
    )
    delay_selector = mo.ui.radio(
        ["All", "Delayed only", "On-time only"],
        value="All",
        label="Delay segment",
    )
    age_range = mo.ui.range_slider(
        int(drivers_df["driver_age"].min()),
        int(drivers_df["driver_age"].max()),
        value=(int(drivers_df["driver_age"].min()), int(drivers_df["driver_age"].max())),
        label="Driver age",
        full_width=True,
    )
    experience_range = mo.ui.range_slider(
        int(drivers_df["driver_experience_years"].min()),
        int(drivers_df["driver_experience_years"].max()),
        value=(
            int(drivers_df["driver_experience_years"].min()),
            int(drivers_df["driver_experience_years"].max()),
        ),
        label="Experience (years)",
        full_width=True,
    )
    rating_range = mo.ui.range_slider(
        float(drivers_df["avg_driver_rating"].min()),
        float(drivers_df["avg_driver_rating"].max()),
        value=(float(drivers_df["avg_driver_rating"].min()), float(drivers_df["avg_driver_rating"].max())),
        label="Avg driver rating",
        step=0.1,
        full_width=True,
    )

    sidebar_filters = mo.vstack(
        [
            mo.md("**Filters**"),
            city_selector,
            vehicle_selector,
            delay_selector,
            age_range,
            experience_range,
            rating_range,
        ],
        gap=0.7,
    )
    sidebar_filters
    return (
        age_range,
        city_selector,
        delay_selector,
        experience_range,
        rating_range,
        vehicle_selector,
    )


@app.cell
def _(
    age_range,
    city_selector,
    delay_selector,
    drivers_df,
    experience_range,
    rating_range,
    vehicle_selector,
):
    filtered_df = drivers_df
    filtered_df = filtered_df[filtered_df["driver_city"].isin(city_selector.value)]
    filtered_df = filtered_df[filtered_df["vehicle_type"].isin(vehicle_selector.value)]
    if delay_selector.value == "Delayed only":
        filtered_df = filtered_df[filtered_df["driver_delay_flag"] == 1]
    elif delay_selector.value == "On-time only":
        filtered_df = filtered_df[filtered_df["driver_delay_flag"] == 0]
    filtered_df = filtered_df[
        (filtered_df["driver_age"] >= age_range.value[0])
        & (filtered_df["driver_age"] <= age_range.value[1])
    ]
    filtered_df = filtered_df[
        (filtered_df["driver_experience_years"] >= experience_range.value[0])
        & (filtered_df["driver_experience_years"] <= experience_range.value[1])
    ]
    filtered_df = filtered_df[
        (filtered_df["avg_driver_rating"] >= rating_range.value[0])
        & (filtered_df["avg_driver_rating"] <= rating_range.value[1])
    ]
    return (filtered_df,)


@app.cell
def _(filtered_df):
    total_drivers = len(filtered_df)
    avg_acceptance_rate = filtered_df["acceptance_rate"].mean()
    avg_rating = filtered_df["avg_driver_rating"].mean()
    avg_delay_rate = filtered_df["delay_rate"].mean()
    avg_pickup_delay = filtered_df["avg_pickup_delay_min"].mean()
    delayed_drivers = int(filtered_df["driver_delay_flag"].sum())
    delayed_rate = delayed_drivers / total_drivers * 100 if total_drivers else 0
    return (
        avg_acceptance_rate,
        avg_delay_rate,
        avg_pickup_delay,
        avg_rating,
        delayed_drivers,
        delayed_rate,
        total_drivers,
    )


@app.cell
def _(
    avg_acceptance_rate,
    avg_delay_rate,
    avg_pickup_delay,
    avg_rating,
    delayed_drivers,
    delayed_rate,
    mo,
    total_drivers,
):
    kpi_cards = mo.hstack(
        [
            mo.stat(value=f"{total_drivers:,}", label="Total drivers", bordered=True),
            mo.stat(
                value=f"{avg_acceptance_rate:.0%}",
                label="Avg acceptance rate",
                bordered=True,
            ),
            mo.stat(value=f"{avg_rating:.2f}", label="Avg rating", bordered=True),
            mo.stat(
                value=f"{avg_delay_rate:.1%}",
                label="Avg delay rate",
                bordered=True,
            ),
            mo.stat(
                value=f"{avg_pickup_delay:.1f} min",
                label="Avg pickup delay",
                bordered=True,
            ),
            mo.stat(
                value=f"{delayed_drivers:,} ({delayed_rate:.0f}%)",
                label="Delay-flagged drivers",
                bordered=True,
            ),
        ],
        widths="equal",
    )
    kpi_cards
    return (kpi_cards,)


@app.cell
def _(filtered_df, px):
    _city_counts = (
        filtered_df.groupby("driver_city")
        .size()
        .reset_index(name="count")
        .sort_values("count")
    )
    driver_city_bar = px.bar(
        _city_counts,
        x="count",
        y="driver_city",
        orientation="h",
        title="Drivers by city",
        color_discrete_sequence=["#636efa"],
    )
    driver_city_bar.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=340)
    _ = driver_city_bar
    return (driver_city_bar,)


@app.cell
def _(filtered_df, px):
    _vehicle_counts = filtered_df["vehicle_type"].value_counts().reset_index()
    _vehicle_counts.columns = ["vehicle_type", "count"]
    driver_vehicle_bar = px.bar(
        _vehicle_counts,
        x="vehicle_type",
        y="count",
        title="Drivers by vehicle type",
        color_discrete_sequence=["#ff7f0e"],
    )
    driver_vehicle_bar.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=340)
    _ = driver_vehicle_bar
    return (driver_vehicle_bar,)


@app.cell
def _(filtered_df, px):
    driver_age_hist = px.histogram(
        filtered_df,
        x="driver_age",
        nbins=25,
        title="Driver age distribution",
        labels={"driver_age": "Age (years)"},
        color_discrete_sequence=["#1f77b4"],
    )
    driver_age_hist.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=340)
    _ = driver_age_hist
    return (driver_age_hist,)


@app.cell
def _(filtered_df, px):
    driver_exp_hist = px.histogram(
        filtered_df,
        x="driver_experience_years",
        nbins=14,
        title="Driver experience distribution",
        labels={"driver_experience_years": "Experience (years)"},
        color_discrete_sequence=["#e377c2"],
    )
    driver_exp_hist.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=340)
    _ = driver_exp_hist
    return (driver_exp_hist,)


@app.cell
def _(filtered_df, px):
    _flag_counts = (
        filtered_df["delay_label"].value_counts().rename_axis("delay_label").reset_index(name="count")
    )
    delay_flag_pie = px.pie(
        _flag_counts,
        names="delay_label",
        values="count",
        hole=0.5,
        color="delay_label",
        title="Drivers by delay flag",
    )
    delay_flag_pie.update_traces(textposition="inside", textinfo="percent+label")
    delay_flag_pie.update_layout(showlegend=False, margin=dict(l=10, r=10, t=40, b=10), height=340)
    _ = delay_flag_pie
    return (delay_flag_pie,)


@app.cell
def _(delay_colors, filtered_df, px):
    _flag_vehicle = (
        filtered_df.groupby(["vehicle_type", "delay_label"]).size().reset_index(name="count")
    )
    flag_vehicle_bar = px.bar(
        _flag_vehicle,
        x="vehicle_type",
        y="count",
        color="delay_label",
        color_discrete_map=delay_colors,
        barmode="stack",
        title="Delay flag by vehicle type",
    )
    flag_vehicle_bar.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=340)
    _ = flag_vehicle_bar
    return (flag_vehicle_bar,)


@app.cell
def _(filtered_df, px):
    acceptance_hist = px.histogram(
        filtered_df,
        x="acceptance_rate",
        nbins=30,
        title="Acceptance rate distribution",
        labels={"acceptance_rate": "Acceptance rate"},
        color_discrete_sequence=["#9467bd"],
    )
    acceptance_hist.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=340)
    _ = acceptance_hist
    return (acceptance_hist,)


@app.cell
def _(filtered_df, px):
    rating_hist = px.histogram(
        filtered_df,
        x="avg_driver_rating",
        nbins=25,
        title="Avg driver rating distribution",
        labels={"avg_driver_rating": "Avg rating"},
        color_discrete_sequence=["#17becf"],
    )
    rating_hist.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=340)
    _ = rating_hist
    return (rating_hist,)


@app.cell
def _(filtered_df, px):
    rating_acceptance_scatter = px.scatter(
        filtered_df,
        x="acceptance_rate",
        y="avg_driver_rating",
        color="vehicle_type",
        opacity=0.6,
        title="Rating vs acceptance rate",
        labels={"acceptance_rate": "Acceptance rate", "avg_driver_rating": "Avg rating"},
    )
    rating_acceptance_scatter.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=380)
    _ = rating_acceptance_scatter
    return (rating_acceptance_scatter,)


@app.cell
def _(filtered_df, px):
    _rating_city = (
        filtered_df.groupby("driver_city")["avg_driver_rating"]
        .mean()
        .reset_index(name="avg_rating")
        .sort_values("avg_rating")
    )
    rating_city_bar = px.bar(
        _rating_city,
        x="avg_rating",
        y="driver_city",
        orientation="h",
        title="Avg rating by city",
        color="avg_rating",
        color_continuous_scale="Blues",
        labels={"avg_rating": "Avg rating"},
    )
    rating_city_bar.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=340)
    _ = rating_city_bar
    return (rating_city_bar,)


@app.cell
def _(filtered_df, px):
    _acceptance_city = (
        filtered_df.groupby("driver_city")["acceptance_rate"]
        .mean()
        .reset_index(name="avg_acceptance")
        .sort_values("avg_acceptance")
    )
    acceptance_city_bar = px.bar(
        _acceptance_city,
        x="avg_acceptance",
        y="driver_city",
        orientation="h",
        title="Avg acceptance rate by city",
        color_discrete_sequence=["#2ca02c"],
        labels={"avg_acceptance": "Avg acceptance rate"},
    )
    acceptance_city_bar.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=340)
    _ = acceptance_city_bar
    return (acceptance_city_bar,)


@app.cell
def _(filtered_df, px):
    _rating_vehicle = (
        filtered_df.groupby("vehicle_type")["avg_driver_rating"]
        .mean()
        .reset_index(name="avg_rating")
    )
    rating_vehicle_bar = px.bar(
        _rating_vehicle,
        x="vehicle_type",
        y="avg_rating",
        title="Avg rating by vehicle type",
        color_discrete_sequence=["#ff7f0e"],
        labels={"vehicle_type": "Vehicle type", "avg_rating": "Avg rating"},
    )
    rating_vehicle_bar.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=340)
    _ = rating_vehicle_bar
    return (rating_vehicle_bar,)


@app.cell
def _(filtered_df, px):
    delay_rate_hist = px.histogram(
        filtered_df,
        x="delay_rate",
        nbins=30,
        title="Delay rate distribution",
        labels={"delay_rate": "Delay rate"},
        color_discrete_sequence=["#d62728"],
    )
    delay_rate_hist.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=340)
    _ = delay_rate_hist
    return (delay_rate_hist,)


@app.cell
def _(filtered_df, px):
    pickup_delay_hist = px.histogram(
        filtered_df,
        x="avg_pickup_delay_min",
        nbins=30,
        title="Avg pickup delay distribution",
        labels={"avg_pickup_delay_min": "Avg pickup delay (min)"},
        color_discrete_sequence=["#8c564b"],
    )
    pickup_delay_hist.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=340)
    _ = pickup_delay_hist
    return (pickup_delay_hist,)


@app.cell
def _(delay_colors, filtered_df, px):
    rating_delay_scatter = px.scatter(
        filtered_df,
        x="avg_pickup_delay_min",
        y="avg_driver_rating",
        color="delay_label",
        color_discrete_map=delay_colors,
        opacity=0.6,
        title="Rating vs pickup delay",
        labels={"avg_pickup_delay_min": "Avg pickup delay (min)", "avg_driver_rating": "Avg rating"},
    )
    rating_delay_scatter.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=380)
    _ = rating_delay_scatter
    return (rating_delay_scatter,)


@app.cell
def _(filtered_df, px):
    _pickup_city = (
        filtered_df.groupby("driver_city")["avg_pickup_delay_min"]
        .mean()
        .reset_index(name="avg_pickup_delay")
        .sort_values("avg_pickup_delay")
    )
    pickup_city_bar = px.bar(
        _pickup_city,
        x="avg_pickup_delay",
        y="driver_city",
        orientation="h",
        title="Avg pickup delay by city",
        color="avg_pickup_delay",
        color_continuous_scale="Reds",
        labels={"avg_pickup_delay": "Avg pickup delay (min)"},
    )
    pickup_city_bar.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=340)
    _ = pickup_city_bar
    return (pickup_city_bar,)


@app.cell
def _(filtered_df, px):
    exp_rating_scatter = px.scatter(
        filtered_df,
        x="driver_experience_years",
        y="avg_driver_rating",
        color="driver_city",
        opacity=0.6,
        title="Rating vs experience",
        labels={"driver_experience_years": "Experience (years)", "avg_driver_rating": "Avg rating"},
    )
    exp_rating_scatter.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=380)
    _ = exp_rating_scatter
    return (exp_rating_scatter,)


@app.cell
def _(filtered_df, mo):
    driver_explorer = mo.ui.data_explorer(filtered_df)
    driver_table = mo.ui.table(filtered_df.head(1000), page_size=20, selection=None)
    return driver_explorer, driver_table


@app.cell
def _(
    acceptance_city_bar,
    acceptance_hist,
    delay_flag_pie,
    delay_rate_hist,
    driver_age_hist,
    driver_city_bar,
    driver_exp_hist,
    driver_explorer,
    driver_table,
    driver_vehicle_bar,
    exp_rating_scatter,
    flag_vehicle_bar,
    kpi_cards,
    mo,
    pickup_city_bar,
    pickup_delay_hist,
    rating_acceptance_scatter,
    rating_city_bar,
    rating_delay_scatter,
    rating_hist,
    rating_vehicle_bar,
):
    overview_tab = mo.vstack(
        [
            mo.md("### Overview"),
            kpi_cards,
            mo.hstack([delay_flag_pie, driver_city_bar], widths=[1, 1]),
            mo.hstack([driver_vehicle_bar, flag_vehicle_bar], widths=[1, 1]),
        ],
        gap=1,
    )
    demographics_tab = mo.vstack(
        [
            mo.md("### Demographics"),
            mo.hstack([driver_age_hist, driver_exp_hist], widths=[1, 1]),
            driver_city_bar,
        ],
        gap=1,
    )
    performance_tab = mo.vstack(
        [
            mo.md("### Performance"),
            mo.hstack([acceptance_hist, rating_hist], widths=[1, 1]),
            rating_acceptance_scatter,
            mo.hstack([rating_city_bar, rating_vehicle_bar], widths=[1, 1]),
            acceptance_city_bar,
        ],
        gap=1,
    )
    reliability_tab = mo.vstack(
        [
            mo.md("### Reliability & risk"),
            mo.hstack([delay_rate_hist, pickup_delay_hist], widths=[1, 1]),
            rating_delay_scatter,
            mo.hstack([pickup_city_bar, flag_vehicle_bar], widths=[1, 1]),
            exp_rating_scatter,
        ],
        gap=1,
    )
    data_tab = mo.vstack(
        [
            mo.md("### Explore the filtered data"),
            driver_explorer,
            driver_table,
        ],
        gap=1,
    )
    tabs = mo.ui.tabs(
        {
            "Overview": overview_tab,
            "Demographics": demographics_tab,
            "Performance": performance_tab,
            "Reliability & Risk": reliability_tab,
            "Data": data_tab,
        },
        lazy=True,
    )
    main_content = mo.vstack([mo.md("### Dashboard"), tabs], gap=0.5)
    # mo.hstack([sidebar_filters, main_content], align="stretch", widths=[1, 4])
    main_content
    return


if __name__ == "__main__":
    app.run()
