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
    return RAW_DATA_DIR, go, mo, pd, px


@app.cell
def _(RAW_DATA_DIR, pd):
    locdemand_path = RAW_DATA_DIR / "location_demand.csv"
    locdemand_df = pd.read_csv(locdemand_path)

    locdemand_df["cancellation_rate"] = (
        locdemand_df["cancelled_rides"] / locdemand_df["total_requests"]
    )
    locdemand_df["completion_rate"] = (
        locdemand_df["completed_rides"] / locdemand_df["total_requests"]
    )
    return (locdemand_df,)


@app.cell
def _():
    demand_colors = {"Low": "#2ca02c", "Medium": "#ff7f0e"}
    return (demand_colors,)


@app.cell
def _(mo):
    mo.md(r"""
    # Rapido Location Demand — EDA Dashboard

    Interactive exploratory analysis of 18k city-level demand segments (city × pickup location × hour × vehicle type).
    Use the sidebar filters to slice the data; every chart, card and table updates reactively.
    """)
    return


@app.cell
def _(locdemand_df, mo):
    city_selector = mo.ui.multiselect(
        locdemand_df["city"].sort_values().unique().tolist(),
        value=locdemand_df["city"].sort_values().unique().tolist(),
        label="Cities",
        full_width=False,
    )
    vehicle_selector = mo.ui.multiselect(
        locdemand_df["vehicle_type"].unique().tolist(),
        value=locdemand_df["vehicle_type"].unique().tolist(),
        label="Vehicle types",
        full_width=False,
    )
    demand_selector = mo.ui.multiselect(
        locdemand_df["demand_level"].unique().tolist(),
        value=locdemand_df["demand_level"].unique().tolist(),
        label="Demand level",
        full_width=False,
    )
    location_selector = mo.ui.multiselect(
        locdemand_df["pickup_location"].sort_values().unique().tolist(),
        value=locdemand_df["pickup_location"].sort_values().unique().tolist(),
        label="Pickup locations",
        full_width=False,
    )
    hour_range = mo.ui.range_slider(
        int(locdemand_df["hour_of_day"].min()),
        int(locdemand_df["hour_of_day"].max()),
        value=(int(locdemand_df["hour_of_day"].min()), int(locdemand_df["hour_of_day"].max())),
        label="Hour of day",
        full_width=True,
    )

    sidebar_filters = mo.vstack(
        [
            mo.md("**Filters**"),
            city_selector,
            vehicle_selector,
            demand_selector,
            location_selector,
            hour_range,
        ],
        gap=0.7,
    )
    sidebar_filters
    return (
        city_selector,
        demand_selector,
        hour_range,
        location_selector,
        vehicle_selector,
    )


@app.cell
def _(
    city_selector,
    demand_selector,
    hour_range,
    location_selector,
    locdemand_df,
    vehicle_selector,
):
    filtered_df = locdemand_df
    filtered_df = filtered_df[filtered_df["city"].isin(city_selector.value)]
    filtered_df = filtered_df[filtered_df["vehicle_type"].isin(vehicle_selector.value)]
    filtered_df = filtered_df[filtered_df["demand_level"].isin(demand_selector.value)]
    filtered_df = filtered_df[filtered_df["pickup_location"].isin(location_selector.value)]
    filtered_df = filtered_df[
        (filtered_df["hour_of_day"] >= hour_range.value[0])
        & (filtered_df["hour_of_day"] <= hour_range.value[1])
    ]
    return (filtered_df,)


@app.cell
def _(filtered_df):
    segment_count = len(filtered_df)
    total_requests = int(filtered_df["total_requests"].sum())
    completed_sum = int(filtered_df["completed_rides"].sum())
    cancelled_sum = int(filtered_df["cancelled_rides"].sum())
    completion_rate = completed_sum / total_requests if total_requests else 0
    cancellation_rate = cancelled_sum / total_requests if total_requests else 0
    avg_wait_time = filtered_df["avg_wait_time_min"].mean()
    avg_surge = filtered_df["avg_surge_multiplier"].mean()
    return (
        avg_surge,
        avg_wait_time,
        cancellation_rate,
        completion_rate,
        segment_count,
        total_requests,
    )


@app.cell
def _(
    avg_surge,
    avg_wait_time,
    cancellation_rate,
    completion_rate,
    mo,
    segment_count,
    total_requests,
):
    kpi_cards = mo.hstack(
        [
            mo.stat(value=f"{total_requests:,}", label="Total ride requests", bordered=True),
            mo.stat(
                value=f"{completion_rate:.0%}",
                label="Completion rate",
                bordered=True,
            ),
            mo.stat(
                value=f"{cancellation_rate:.0%}",
                label="Cancellation rate",
                bordered=True,
            ),
            mo.stat(
                value=f"{avg_wait_time:.0f} min",
                label="Avg wait time",
                bordered=True,
            ),
            mo.stat(
                value=f"{avg_surge:.2f}x",
                label="Avg surge multiplier",
                bordered=True,
            ),
            mo.stat(value=f"{segment_count:,}", label="Demand segments", bordered=True),
        ],
        widths="equal",
    )
    kpi_cards
    return (kpi_cards,)


@app.cell
def _(filtered_df, px):
    _city_requests = (
        filtered_df.groupby("city")["total_requests"].sum().reset_index(name="requests").sort_values("requests")
    )
    city_bar = px.bar(
        _city_requests,
        x="requests",
        y="city",
        orientation="h",
        title="Ride requests by city",
        color_discrete_sequence=["#636efa"],
    )
    city_bar.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=340)
    _ = city_bar
    return (city_bar,)


@app.cell
def _(filtered_df, px):
    _vehicle_requests = (
        filtered_df.groupby("vehicle_type")["total_requests"].sum().reset_index(name="requests")
    )
    vehicle_bar = px.bar(
        _vehicle_requests,
        x="vehicle_type",
        y="requests",
        title="Ride requests by vehicle type",
        color_discrete_sequence=["#ff7f0e"],
    )
    vehicle_bar.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=340)
    _ = vehicle_bar
    return (vehicle_bar,)


@app.cell
def _(demand_colors, filtered_df, px):
    _demand_counts = (
        filtered_df["demand_level"].value_counts().rename_axis("demand_level").reset_index(name="count")
    )
    demand_level_pie = px.pie(
        _demand_counts,
        names="demand_level",
        values="count",
        hole=0.5,
        color="demand_level",
        color_discrete_map=demand_colors,
        title="Demand level distribution",
    )
    demand_level_pie.update_traces(textposition="inside", textinfo="percent+label")
    demand_level_pie.update_layout(showlegend=False, margin=dict(l=10, r=10, t=40, b=10), height=340)
    _ = demand_level_pie
    return (demand_level_pie,)


@app.cell
def _(demand_colors, filtered_df, px):
    _hourly = (
        filtered_df.groupby(["hour_of_day", "demand_level"])["total_requests"]
        .sum()
        .reset_index(name="requests")
    )
    hourly_demand_line = px.line(
        _hourly,
        x="hour_of_day",
        y="requests",
        color="demand_level",
        color_discrete_map=demand_colors,
        title="Ride requests by hour of day",
    )
    hourly_demand_line.update_xaxes(dtick=1)
    hourly_demand_line.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=340)
    _ = hourly_demand_line
    return (hourly_demand_line,)


@app.cell
def _(filtered_df, px):
    _hourly_city = (
        filtered_df.groupby(["hour_of_day", "city"])["total_requests"].sum().reset_index(name="requests")
    )
    hourly_city_line = px.line(
        _hourly_city,
        x="hour_of_day",
        y="requests",
        color="city",
        title="Ride requests by hour across cities",
    )
    hourly_city_line.update_xaxes(dtick=1)
    hourly_city_line.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=340)
    _ = hourly_city_line
    return (hourly_city_line,)


@app.cell
def _(demand_colors, filtered_df, px):
    _demand_hour = (
        filtered_df.groupby(["hour_of_day", "demand_level"])["total_requests"]
        .sum()
        .reset_index(name="requests")
    )
    demand_hour_bar = px.bar(
        _demand_hour,
        x="hour_of_day",
        y="requests",
        color="demand_level",
        color_discrete_map=demand_colors,
        barmode="stack",
        title="Demand level by hour of day",
    )
    demand_hour_bar.update_xaxes(dtick=1)
    demand_hour_bar.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=340)
    _ = demand_hour_bar
    return (demand_hour_bar,)


@app.cell
def _(filtered_df, go, px):
    _heat = (
        filtered_df.groupby(["hour_of_day", "city"])["total_requests"].sum().unstack(fill_value=0)
    )
    if _heat.empty:
        hour_city_heatmap = go.Figure()
    else:
        hour_city_heatmap = px.imshow(
            _heat,
            labels=dict(x="City", y="Hour of day", color="Requests"),
            title="Request volume: hour of day × city",
            color_continuous_scale="YlOrRd",
            aspect="auto",
        )
        hour_city_heatmap.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=420)
    _ = hour_city_heatmap
    return (hour_city_heatmap,)


@app.cell
def _(filtered_df, px):
    _top_locations = (
        filtered_df.groupby("pickup_location")["total_requests"]
        .sum()
        .reset_index(name="requests")
        .sort_values("requests")
        .tail(20)
    )
    top_location_bar = px.bar(
        _top_locations,
        x="requests",
        y="pickup_location",
        orientation="h",
        title="Top 20 pickup locations by ride requests",
        color_discrete_sequence=["#636efa"],
    )
    top_location_bar.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=460)
    _ = top_location_bar
    return (top_location_bar,)


@app.cell
def _(filtered_df, px):
    _top_wait = (
        filtered_df.groupby("pickup_location")["avg_wait_time_min"]
        .mean()
        .reset_index(name="avg_wait")
        .sort_values("avg_wait")
        .tail(20)
    )
    top_wait_bar = px.bar(
        _top_wait,
        x="avg_wait",
        y="pickup_location",
        orientation="h",
        title="Top 20 pickup locations by avg wait time",
        color="avg_wait",
        color_continuous_scale="Reds",
        labels={"avg_wait": "Avg wait (min)"},
    )
    top_wait_bar.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=460)
    _ = top_wait_bar
    return (top_wait_bar,)


@app.cell
def _(filtered_df, px):
    wait_hist = px.histogram(
        filtered_df,
        x="avg_wait_time_min",
        nbins=40,
        title="Avg wait time distribution",
        labels={"avg_wait_time_min": "Avg wait (min)"},
        color_discrete_sequence=["#17becf"],
    )
    wait_hist.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=340)
    _ = wait_hist
    return (wait_hist,)


@app.cell
def _(filtered_df, px):
    surge_hist = px.histogram(
        filtered_df,
        x="avg_surge_multiplier",
        nbins=40,
        title="Avg surge multiplier distribution",
        labels={"avg_surge_multiplier": "Avg surge multiplier"},
        color_discrete_sequence=["#9467bd"],
    )
    surge_hist.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=340)
    _ = surge_hist
    return (surge_hist,)


@app.cell
def _(filtered_df, px):
    cancel_rate_hist = px.histogram(
        filtered_df,
        x="cancellation_rate",
        nbins=40,
        title="Cancellation rate distribution",
        labels={"cancellation_rate": "Cancellation rate"},
        color_discrete_sequence=["#d62728"],
    )
    cancel_rate_hist.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=340)
    _ = cancel_rate_hist
    return (cancel_rate_hist,)


@app.cell
def _(filtered_df, px):
    _wait_city = (
        filtered_df.groupby("city")["avg_wait_time_min"].mean().reset_index(name="avg_wait").sort_values("avg_wait")
    )
    wait_city_bar = px.bar(
        _wait_city,
        x="avg_wait",
        y="city",
        orientation="h",
        title="Avg wait time by city",
        color="avg_wait",
        color_continuous_scale="Reds",
        labels={"avg_wait": "Avg wait (min)"},
    )
    wait_city_bar.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=340)
    _ = wait_city_bar
    return (wait_city_bar,)


@app.cell
def _(filtered_df, px):
    _surge_city = (
        filtered_df.groupby("city")["avg_surge_multiplier"]
        .mean()
        .reset_index(name="avg_surge")
        .sort_values("avg_surge")
    )
    surge_city_bar = px.bar(
        _surge_city,
        x="avg_surge",
        y="city",
        orientation="h",
        title="Avg surge multiplier by city",
        color="avg_surge",
        color_continuous_scale="Purples",
        labels={"avg_surge": "Avg surge"},
    )
    surge_city_bar.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=340)
    _ = surge_city_bar
    return (surge_city_bar,)


@app.cell
def _(filtered_df, px):
    _wait_vehicle = (
        filtered_df.groupby("vehicle_type")["avg_wait_time_min"]
        .mean()
        .reset_index(name="avg_wait")
    )
    wait_vehicle_bar = px.bar(
        _wait_vehicle,
        x="vehicle_type",
        y="avg_wait",
        title="Avg wait time by vehicle type",
        color_discrete_sequence=["#2ca02c"],
        labels={"vehicle_type": "Vehicle type", "avg_wait": "Avg wait (min)"},
    )
    wait_vehicle_bar.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=340)
    _ = wait_vehicle_bar
    return (wait_vehicle_bar,)


@app.cell
def _(filtered_df, px):
    _cancel_vehicle = (
        filtered_df.groupby("vehicle_type")["cancellation_rate"]
        .mean()
        .reset_index(name="avg_cancel")
    )
    cancel_vehicle_bar = px.bar(
        _cancel_vehicle,
        x="vehicle_type",
        y="avg_cancel",
        title="Avg cancellation rate by vehicle type",
        color_discrete_sequence=["#d62728"],
        labels={"vehicle_type": "Vehicle type", "avg_cancel": "Avg cancellation rate"},
    )
    cancel_vehicle_bar.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=340)
    _ = cancel_vehicle_bar
    return (cancel_vehicle_bar,)


@app.cell
def _(filtered_df, px):
    _sample_n = min(len(filtered_df), 8000)
    _sample = filtered_df.sample(_sample_n, random_state=42) if _sample_n else filtered_df
    requests_wait_scatter = px.scatter(
        _sample,
        x="avg_wait_time_min",
        y="total_requests",
        color="demand_level",
        opacity=0.6,
        title="Ride requests vs wait time",
        labels={"avg_wait_time_min": "Avg wait (min)", "total_requests": "Total requests"},
    )
    requests_wait_scatter.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=380)
    _ = requests_wait_scatter
    return (requests_wait_scatter,)


@app.cell
def _(filtered_df, pd, px):
    _surge_wait = filtered_df.copy()
    _surge_wait["wait_bucket"] = pd.cut(
        _surge_wait["avg_wait_time_min"],
        bins=[0, 30, 60, 90, 120, 200],
        labels=["<30", "30-60", "60-90", "90-120", "120+"],
    )
    _surge_wait_bucket = (
        _surge_wait.groupby("wait_bucket", observed=False)["avg_surge_multiplier"]
        .mean()
        .reset_index(name="avg_surge")
    )
    surge_wait_bar = px.bar(
        _surge_wait_bucket,
        x="wait_bucket",
        y="avg_surge",
        title="Avg surge by wait time bucket",
        color_discrete_sequence=["#ff7f0e"],
        labels={"wait_bucket": "Avg wait (min)", "avg_surge": "Avg surge multiplier"},
    )
    surge_wait_bar.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=340)
    _ = surge_wait_bar
    return (surge_wait_bar,)


@app.cell
def _(filtered_df, mo):
    loc_explorer = mo.ui.data_explorer(filtered_df)
    loc_table = mo.ui.table(filtered_df.head(1000), page_size=20, selection=None)
    return loc_explorer, loc_table


@app.cell
def _(
    cancel_rate_hist,
    cancel_vehicle_bar,
    city_bar,
    demand_hour_bar,
    demand_level_pie,
    hour_city_heatmap,
    hourly_city_line,
    hourly_demand_line,
    kpi_cards,
    loc_explorer,
    loc_table,
    mo,
    requests_wait_scatter,
    surge_city_bar,
    surge_hist,
    surge_wait_bar,
    top_location_bar,
    top_wait_bar,
    vehicle_bar,
    wait_city_bar,
    wait_hist,
    wait_vehicle_bar,
):
    overview_tab = mo.vstack(
        [
            mo.md("### Overview"),
            kpi_cards,
            mo.hstack([city_bar, vehicle_bar], widths=[1, 1]),
            mo.hstack([demand_level_pie, hourly_demand_line], widths=[1, 1]),
        ],
        gap=1,
    )
    temporal_tab = mo.vstack(
        [
            mo.md("### Temporal patterns"),
            hour_city_heatmap,
            hourly_city_line,
            mo.hstack([demand_hour_bar, hourly_demand_line], widths=[1, 1]),
        ],
        gap=1,
    )
    location_tab = mo.vstack(
        [
            mo.md("### Location hotspots"),
            mo.hstack([top_location_bar, top_wait_bar], widths=[1, 1]),
        ],
        gap=1,
    )
    service_tab = mo.vstack(
        [
            mo.md("### Service & pricing"),
            mo.hstack([wait_hist, surge_hist], widths=[1, 1]),
            mo.hstack([wait_city_bar, surge_city_bar], widths=[1, 1]),
            mo.hstack([wait_vehicle_bar, cancel_vehicle_bar], widths=[1, 1]),
            cancel_rate_hist,
            requests_wait_scatter,
            surge_wait_bar,
        ],
        gap=1,
    )
    data_tab = mo.vstack(
        [
            mo.md("### Explore the filtered data"),
            loc_explorer,
            loc_table,
        ],
        gap=1,
    )
    tabs = mo.ui.tabs(
        {
            "Overview": overview_tab,
            "Temporal": temporal_tab,
            "Locations": location_tab,
            "Service & Pricing": service_tab,
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
