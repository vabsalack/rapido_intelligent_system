# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo>=0.23.16",
# ]
# ///

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from emoji import emojize

    from rapido_intelligent_system.dataset import RAW_DATA_DIR

    px.defaults.template = "plotly_white"
    return RAW_DATA_DIR, go, mo, pd, px


@app.cell
def _(RAW_DATA_DIR, pd):
    bookings_path = RAW_DATA_DIR / "bookings.csv"
    booking_df = pd.read_csv(bookings_path)

    booking_df["booking_datetime"] = pd.to_datetime(
        booking_df["booking_date"] + " " + booking_df["booking_time"]
    )
    booking_df["date"] = booking_df["booking_datetime"].dt.date
    booking_df["month_number"] = booking_df["booking_datetime"].dt.month
    booking_df["month_name"] = booking_df["booking_datetime"].dt.month_name()
    return (booking_df,)


@app.cell
def _(mo):
    mo.md(r"""
    # Rapido Bookings — EDA Dashboard

    Interactive exploratory analysis of 100k ride-hailing bookings across India.
    Use the sidebar filters to slice the data; every chart, card and table updates reactively.
    """)
    return


@app.cell
def _(booking_df, mo):
    city_selector = mo.ui.multiselect(
        booking_df["city"].sort_values().unique().tolist(),
        value=booking_df["city"].sort_values().unique().tolist(),
        label="Cities",
        full_width=False,
    )
    vehicle_selector = mo.ui.multiselect(
        booking_df["vehicle_type"].unique().tolist(),
        value=booking_df["vehicle_type"].unique().tolist(),
        label="Vehicle types",
        full_width=False,
    )
    status_selector = mo.ui.multiselect(
        booking_df["booking_status"].unique().tolist(),
        value=booking_df["booking_status"].unique().tolist(),
        label="Booking status",
        full_width=False,
    )
    date_start = mo.ui.date(
        start=booking_df["date"].min(),
        stop=booking_df["date"].max(),
        value=booking_df["date"].min(),
        label="From date",
        full_width=False,
    )
    date_end = mo.ui.date(
        start=booking_df["date"].min(),
        stop=booking_df["date"].max(),
        value=booking_df["date"].max(),
        label="To date",
        full_width=False,
    )
    hour_range = mo.ui.range_slider(
        0,
        23,
        value=(0, 23),
        label="Hour of day",
        full_width=True,
    )

    sidebar_filters = mo.vstack(
        [
            mo.md("**Filters**"),
            city_selector,
            vehicle_selector,
            status_selector,
            mo.hstack([date_start, date_end], widths="equal"),
            hour_range,
        ],
        gap=0.7,
    )

    sidebar_filters
    return (
        city_selector,
        date_end,
        date_start,
        hour_range,
        status_selector,
        vehicle_selector,
    )


@app.cell
def _(
    booking_df,
    city_selector,
    date_end,
    date_start,
    hour_range,
    status_selector,
    vehicle_selector,
):
    filtered_df = booking_df
    filtered_df = filtered_df[filtered_df["city"].isin(city_selector.value)]
    filtered_df = filtered_df[filtered_df["vehicle_type"].isin(vehicle_selector.value)]
    filtered_df = filtered_df[filtered_df["booking_status"].isin(status_selector.value)]
    filtered_df = filtered_df[
        (filtered_df["date"] >= date_start.value) & (filtered_df["date"] <= date_end.value)
    ]
    filtered_df = filtered_df[
        (filtered_df["hour_of_day"] >= hour_range.value[0])
        & (filtered_df["hour_of_day"] <= hour_range.value[1])
    ]
    return (filtered_df,)


@app.cell
def _(filtered_df):
    total_bookings = len(filtered_df)
    completed_count = int((filtered_df["booking_status"] == "Completed").sum())
    cancelled_count = int((filtered_df["booking_status"] == "Cancelled").sum())
    incomplete_count = int((filtered_df["booking_status"] == "Incomplete").sum())
    completed_rate = completed_count / total_bookings * 100 if total_bookings else 0
    cancelled_rate = cancelled_count / total_bookings * 100 if total_bookings else 0
    avg_booking_value = filtered_df["booking_value"].mean()
    avg_distance = filtered_df["ride_distance_km"].mean()
    avg_surge = filtered_df["surge_multiplier"].mean()
    return (
        avg_booking_value,
        avg_distance,
        avg_surge,
        cancelled_rate,
        completed_rate,
        total_bookings,
    )


@app.cell
def _(
    avg_booking_value,
    avg_distance,
    avg_surge,
    cancelled_rate,
    completed_rate,
    mo,
    total_bookings,
):
    kpi_cards = mo.hstack(
        [
            mo.stat(
                value=f"{total_bookings:,}",
                label="Total bookings",
                bordered=True,
            ),
            mo.stat(
                value=f"{completed_rate:.1f}%",
                label="Completion rate",
                bordered=True,
            ),
            mo.stat(
                value=f"{cancelled_rate:.1f}%",
                label="Cancellation rate",
                bordered=True,
            ),
            mo.stat(
                value=f"Rs {avg_booking_value:,.1f}",
                label="Avg booking value",
                bordered=True,
            ),
            mo.stat(
                value=f"{avg_distance:.1f} km",
                label="Avg ride distance",
                bordered=True,
            ),
            mo.stat(
                value=f"{avg_surge:.2f} times",
                label="Avg surge multiplier",
                bordered=True,
            ),
        ],
        widths="equal",
    )
    kpi_cards
    return


@app.cell
def _():
    status_colors = {
        "Completed": "#2ca02c",
        "Cancelled": "#d62728",
        "Incomplete": "#ff7f0e",
    }
    return (status_colors,)


@app.cell
def _(filtered_df, px, status_colors):
    _daily_status = (
        filtered_df.groupby(["date", "booking_status"]).size().reset_index(name="bookings")
    )
    trend_fig = px.line(
        _daily_status,
        x="date",
        y="bookings",
        color="booking_status",
        color_discrete_map=status_colors,
        title="Bookings per day by status",
    )
    trend_fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=360)
    _ = trend_fig
    return (trend_fig,)


@app.cell
def _(filtered_df, px, status_colors):
    _hourly = (
        filtered_df.groupby(["hour_of_day", "booking_status"]).size().reset_index(name="bookings")
    )
    hour_fig = px.bar(
        _hourly,
        x="hour_of_day",
        y="bookings",
        color="booking_status",
        color_discrete_map=status_colors,
        barmode="stack",
        title="Bookings by hour of day",
    )
    hour_fig.update_xaxes(dtick=1)
    hour_fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=340)
    _ = hour_fig
    return (hour_fig,)


@app.cell
def _(filtered_df, px):
    _dow_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    _dow_counts = (
        filtered_df.groupby("day_of_week")
        .size()
        .reindex(_dow_order)
        .reset_index(name="bookings")
    )
    dow_fig = px.bar(
        _dow_counts,
        x="day_of_week",
        y="bookings",
        title="Bookings by day of week",
        color_discrete_sequence=["#636efa"],
    )
    dow_fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=340)
    _ = dow_fig
    return (dow_fig,)


@app.cell
def _(filtered_df, px):
    _month_counts = (
        filtered_df.groupby(["month_number", "month_name"])
        .size()
        .reset_index(name="bookings")
        .sort_values("month_number")
    )
    month_fig = px.bar(
        _month_counts,
        x="month_name",
        y="bookings",
        title="Bookings by month",
        color_discrete_sequence=["#ff7f0e"],
    )
    month_fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=340)
    _ = month_fig
    return (month_fig,)


@app.cell
def _(filtered_df, go, px):
    _dow_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    _heat = filtered_df.groupby(["hour_of_day", "day_of_week"]).size().unstack(fill_value=0)
    _heat = _heat.reindex(columns=_dow_order, fill_value=0)
    if _heat.empty:
        heatmap_fig = go.Figure()
    else:
        heatmap_fig = px.imshow(
            _heat,
            labels=dict(x="Day of week", y="Hour of day", color="Bookings"),
            title="Booking volume: hour of day × day of week",
            color_continuous_scale="YlOrRd",
            aspect="auto",
        )
        heatmap_fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=420)
    _ = heatmap_fig
    return (heatmap_fig,)


@app.cell
def _(filtered_df, px, status_colors):
    _status_counts = (
        filtered_df["booking_status"]
        .value_counts()
        .rename_axis("booking_status")
        .reset_index(name="count")
    )
    status_pie = px.pie(
        _status_counts,
        names="booking_status",
        values="count",
        hole=0.5,
        color="booking_status",
        color_discrete_map=status_colors,
        title="Booking status distribution",
    )
    status_pie.update_traces(textposition="inside", textinfo="percent+label")
    status_pie.update_layout(showlegend=False, margin=dict(l=10, r=10, t=40, b=10), height=340)
    _ = status_pie
    return (status_pie,)


@app.cell
def _(filtered_df, px):
    _city_counts = (
        filtered_df.groupby("city").size().reset_index(name="bookings").sort_values("bookings")
    )
    city_fig = px.bar(
        _city_counts,
        x="bookings",
        y="city",
        orientation="h",
        title="Bookings by city",
        color_discrete_sequence=["#636efa"],
    )
    city_fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=340)
    _ = city_fig
    return (city_fig,)


@app.cell
def _(filtered_df, px, status_colors):
    _vehicle_counts = (
        filtered_df.groupby(["vehicle_type", "booking_status"])
        .size()
        .reset_index(name="bookings")
    )
    vehicle_fig = px.bar(
        _vehicle_counts,
        x="vehicle_type",
        y="bookings",
        color="booking_status",
        color_discrete_map=status_colors,
        barmode="stack",
        title="Bookings by vehicle type",
    )
    vehicle_fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=340)
    _ = vehicle_fig
    return (vehicle_fig,)


@app.cell
def _(filtered_df, px):
    value_hist = px.histogram(
        filtered_df,
        x="booking_value",
        nbins=60,
        title="Booking value distribution",
        labels={"booking_value": "Booking value (₹)"},
        color_discrete_sequence=["#9467bd"],
    )
    value_hist.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=340)
    _ = value_hist
    return (value_hist,)


@app.cell
def _(filtered_df, px):
    distance_hist = px.histogram(
        filtered_df,
        x="ride_distance_km",
        nbins=60,
        title="Ride distance distribution",
        labels={"ride_distance_km": "Ride distance (km)"},
        color_discrete_sequence=["#17becf"],
    )
    distance_hist.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=340)
    _ = distance_hist
    return (distance_hist,)


@app.cell
def _(filtered_df, px):
    _sample_n = min(len(filtered_df), 15000)
    _sample = filtered_df.sample(_sample_n, random_state=42) if _sample_n else filtered_df
    scatter_fig = px.scatter(
        _sample,
        x="ride_distance_km",
        y="booking_value",
        color="vehicle_type",
        opacity=0.6,
        title="Booking value vs ride distance (sampled)",
        labels={"ride_distance_km": "Ride distance (km)", "booking_value": "Booking value (₹)"},
    )
    scatter_fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=420)
    _ = scatter_fig
    return (scatter_fig,)


@app.cell
def _(filtered_df, px):
    _surge = (
        filtered_df.groupby("surge_multiplier")["booking_value"]
        .agg(["mean", "count"])
        .reset_index()
    )
    _surge.columns = ["surge_multiplier", "avg_value", "count"]
    surge_fig = px.bar(
        _surge,
        x="surge_multiplier",
        y="avg_value",
        title="Average booking value by surge multiplier",
        color="count",
        color_continuous_scale="Blues",
        labels={"surge_multiplier": "Surge multiplier", "avg_value": "Avg booking value (₹)"},
    )
    surge_fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=340)
    _ = surge_fig
    return (surge_fig,)


@app.cell
def _(filtered_df, px, status_colors):
    _traffic = (
        filtered_df.groupby(["traffic_level", "booking_status"]).size().reset_index(name="bookings")
    )
    traffic_fig = px.bar(
        _traffic,
        x="traffic_level",
        y="bookings",
        color="booking_status",
        color_discrete_map=status_colors,
        barmode="stack",
        title="Bookings by traffic level",
        category_orders={"traffic_level": ["Low", "Medium", "High"]},
    )
    traffic_fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=340)
    _ = traffic_fig
    return (traffic_fig,)


@app.cell
def _(filtered_df, px, status_colors):
    _weather = (
        filtered_df.groupby(["weather_condition", "booking_status"])
        .size()
        .reset_index(name="bookings")
    )
    weather_fig = px.bar(
        _weather,
        x="weather_condition",
        y="bookings",
        color="booking_status",
        color_discrete_map=status_colors,
        barmode="stack",
        title="Bookings by weather condition",
        category_orders={"weather_condition": ["Clear", "Rain", "Heavy Rain"]},
    )
    weather_fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=340)
    _ = weather_fig
    return (weather_fig,)


@app.cell
def _(filtered_df, px):
    _pickup = (
        filtered_df["pickup_location"].value_counts().head(15).sort_values().reset_index()
    )
    _pickup.columns = ["pickup_location", "bookings"]
    pickup_fig = px.bar(
        _pickup,
        x="bookings",
        y="pickup_location",
        orientation="h",
        title="Top 15 pickup locations",
        color_discrete_sequence=["#636efa"],
    )
    pickup_fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=420)
    _ = pickup_fig
    return (pickup_fig,)


@app.cell
def _(filtered_df, px):
    _drop = filtered_df["drop_location"].value_counts().head(15).sort_values().reset_index()
    _drop.columns = ["drop_location", "bookings"]
    drop_fig = px.bar(
        _drop,
        x="bookings",
        y="drop_location",
        orientation="h",
        title="Top 15 drop locations",
        color_discrete_sequence=["#17becf"],
    )
    drop_fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=420)
    _ = drop_fig
    return (drop_fig,)


@app.cell
def _(filtered_df, px):
    _reasons = (
        filtered_df["incomplete_ride_reason"].value_counts().dropna().rename_axis("reason").reset_index(name="count")
    )
    reasons_fig = px.bar(
        _reasons,
        x="count",
        y="reason",
        orientation="h",
        title="Incomplete ride reasons",
        color_discrete_sequence=["#ff7f0e"],
    )
    reasons_fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=320)
    _ = reasons_fig
    return (reasons_fig,)


@app.cell
def _(filtered_df, go, px):
    _time_df = filtered_df.dropna(subset=["actual_ride_time_min"])
    _time_df = _time_df[_time_df["actual_ride_time_min"] > 0].copy()
    if _time_df.empty:
        time_hist = go.Figure()
    else:
        _time_df["time_diff"] = _time_df["actual_ride_time_min"] - _time_df["estimated_ride_time_min"]
        time_hist = px.histogram(
            _time_df,
            x="time_diff",
            nbins=50,
            title="Actual vs estimated ride time",
            labels={"time_diff": "Actual − estimated (min)"},
            color_discrete_sequence=["#2ca02c"],
        )
        time_hist.add_vline(
            x=0,
            line_dash="dash",
            line_color="red",
            annotation_text="on-time",
        )
    time_hist.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=340)
    _ = time_hist
    return (time_hist,)


@app.cell
def _(filtered_df, mo):
    booking_explorer = mo.ui.data_explorer(filtered_df)
    booking_table = mo.ui.table(
        filtered_df.drop(columns=["booking_datetime"]).head(1000),
        page_size=20,
        selection=None,
    )
    return booking_explorer, booking_table


@app.cell
def _(
    booking_explorer,
    booking_table,
    city_fig,
    distance_hist,
    dow_fig,
    drop_fig,
    heatmap_fig,
    hour_fig,
    mo,
    month_fig,
    pickup_fig,
    reasons_fig,
    scatter_fig,
    status_pie,
    surge_fig,
    time_hist,
    traffic_fig,
    trend_fig,
    value_hist,
    vehicle_fig,
    weather_fig,
):
    overview_tab = mo.vstack(
        [
            mo.md("### Overview"),
            mo.hstack([trend_fig, status_pie], widths=[2, 1]),
            mo.hstack([city_fig, vehicle_fig], widths=[1, 1]),
        ],
        gap=1,
    )
    temporal_tab = mo.vstack(
        [
            mo.md("### Temporal patterns"),
            heatmap_fig,
            mo.hstack([hour_fig, dow_fig], widths=[1, 1]),
            month_fig,
        ],
        gap=1,
    )
    pricing_tab = mo.vstack(
        [
            mo.md("### Pricing & distance"),
            mo.hstack([value_hist, distance_hist], widths=[1, 1]),
            scatter_fig,
            surge_fig,
        ],
        gap=1,
    )
    demand_tab = mo.vstack(
        [
            mo.md("### Demand drivers & service quality"),
            mo.hstack([traffic_fig, weather_fig], widths=[1, 1]),
            mo.hstack([pickup_fig, drop_fig], widths=[1, 1]),
            mo.hstack([reasons_fig, time_hist], widths=[1, 1]),
        ],
        gap=1,
    )
    data_tab = mo.vstack(
        [
            mo.md("### Explore the filtered data"),
            booking_explorer,
            booking_table,
        ],
        gap=1,
    )
    tabs = mo.ui.tabs(
        {
            "Overview": overview_tab,
            "Temporal": temporal_tab,
            "Pricing & Distance": pricing_tab,
            "Demand Drivers": demand_tab,
            "Data": data_tab,
        },
        value="Overview",
        orientation="horizontal",
        lazy=True,
    )
    main_content = mo.vstack([mo.md("### Dashboard"), tabs], gap=0.5)
    # mo.hstack([main_content], align="stretch", widths=[1, 4])
    main_content
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
