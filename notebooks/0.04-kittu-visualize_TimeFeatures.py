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
    timefeats_path = RAW_DATA_DIR / "time_features.csv"
    timefeats_df = pd.read_csv(timefeats_path)

    timefeats_df["datetime_dt"] = pd.to_datetime(timefeats_df["datetime"])
    timefeats_df["date"] = timefeats_df["datetime_dt"].dt.date
    timefeats_df["month"] = timefeats_df["datetime_dt"].dt.month
    timefeats_df["month_name"] = timefeats_df["datetime_dt"].dt.month_name()
    timefeats_df["peak_label"] = timefeats_df["peak_time_flag"].map({1: "Peak", 0: "Non-peak"})
    timefeats_df["weekend_label"] = timefeats_df["is_weekend"].map({1: "Weekend", 0: "Weekday"})
    return (timefeats_df,)


@app.cell
def _():
    peak_colors = {"Peak": "#ff7f0e", "Non-peak": "#636efa"}
    weekend_colors = {"Weekend": "#2ca02c", "Weekday": "#636efa"}
    season_colors = {
        "Winter": "#1f77b4",
        "Summer": "#ff7f0e",
        "Monsoon": "#2ca02c",
    }
    return peak_colors, season_colors, weekend_colors


@app.cell
def _(mo):
    mo.md(r"""
    # Rapido Time Features — EDA Dashboard

    Interactive exploration of the 2025 hourly calendar reference table (24 hours × 365 days).
    Use the sidebar filters to slice the data; every chart, card and table updates reactively.

    _Note: the `is_holiday` flag is constant (no holiday hours are recorded), so peak-time and weekend structure drive the analysis._
    """)
    return


@app.cell
def _(mo, timefeats_df):
    season_selector = mo.ui.multiselect(
        timefeats_df["season"].unique().tolist(),
        value=timefeats_df["season"].unique().tolist(),
        label="Seasons",
        full_width=False,
    )
    dow_selector = mo.ui.multiselect(
        [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ],
        value=[
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ],
        label="Days of week",
        full_width=False,
    )
    weekend_selector = mo.ui.radio(
        ["All", "Weekends only", "Weekdays only"],
        value="All",
        label="Weekend segment",
    )
    peak_selector = mo.ui.radio(
        ["All", "Peak only", "Non-peak only"],
        value="All",
        label="Peak segment",
    )
    hour_range = mo.ui.range_slider(
        int(timefeats_df["hour_of_day"].min()),
        int(timefeats_df["hour_of_day"].max()),
        value=(int(timefeats_df["hour_of_day"].min()), int(timefeats_df["hour_of_day"].max())),
        label="Hour of day",
        full_width=True,
    )
    month_range = mo.ui.range_slider(
        int(timefeats_df["month"].min()),
        int(timefeats_df["month"].max()),
        value=(int(timefeats_df["month"].min()), int(timefeats_df["month"].max())),
        label="Months (1-12)",
        full_width=True,
    )

    sidebar_filters = mo.vstack(
        [
            mo.md("**Filters**"),
            season_selector,
            dow_selector,
            weekend_selector,
            peak_selector,
            hour_range,
            month_range,
        ],
        gap=0.7,
    )
    sidebar_filters
    return (
        dow_selector,
        hour_range,
        month_range,
        peak_selector,
        season_selector,
        weekend_selector,
    )


@app.cell
def _(
    dow_selector,
    hour_range,
    month_range,
    peak_selector,
    season_selector,
    timefeats_df,
    weekend_selector,
):
    filtered_df = timefeats_df
    filtered_df = filtered_df[filtered_df["season"].isin(season_selector.value)]
    filtered_df = filtered_df[filtered_df["day_of_week"].isin(dow_selector.value)]
    if weekend_selector.value == "Weekends only":
        filtered_df = filtered_df[filtered_df["is_weekend"] == 1]
    elif weekend_selector.value == "Weekdays only":
        filtered_df = filtered_df[filtered_df["is_weekend"] == 0]
    if peak_selector.value == "Peak only":
        filtered_df = filtered_df[filtered_df["peak_time_flag"] == 1]
    elif peak_selector.value == "Non-peak only":
        filtered_df = filtered_df[filtered_df["peak_time_flag"] == 0]
    filtered_df = filtered_df[
        (filtered_df["hour_of_day"] >= hour_range.value[0])
        & (filtered_df["hour_of_day"] <= hour_range.value[1])
    ]
    filtered_df = filtered_df[
        (filtered_df["month"] >= month_range.value[0])
        & (filtered_df["month"] <= month_range.value[1])
    ]
    return (filtered_df,)


@app.cell
def _(filtered_df):
    total_hours = len(filtered_df)
    total_days = filtered_df["date"].nunique()
    peak_hours = int(filtered_df["peak_time_flag"].sum())
    peak_rate = peak_hours / total_hours * 100 if total_hours else 0
    weekend_hours = int(filtered_df["is_weekend"].sum())
    weekend_rate = weekend_hours / total_hours * 100 if total_hours else 0
    avg_peak_per_day = peak_hours / total_days if total_days else 0
    return avg_peak_per_day, peak_rate, total_days, total_hours, weekend_rate


@app.cell
def _(avg_peak_per_day, mo, peak_rate, total_days, total_hours, weekend_rate):
    kpi_cards = mo.hstack(
        [
            mo.stat(value=f"{total_hours:,}", label="Hours covered", bordered=True),
            mo.stat(value=f"{total_days:,}", label="Days covered", bordered=True),
            mo.stat(value=f"{peak_rate:.0f}%", label="Peak-time hours", bordered=True),
            mo.stat(value=f"{weekend_rate:.0f}%", label="Weekend hours", bordered=True),
            mo.stat(
                value=f"{avg_peak_per_day:.1f}",
                label="Avg peak hours / day",
                bordered=True,
            ),
        ],
        widths="equal",
    )
    kpi_cards
    return (kpi_cards,)


@app.cell
def _(filtered_df, px, season_colors):
    _season_counts = (
        filtered_df["season"].value_counts().rename_axis("season").reset_index(name="hours")
    )
    season_pie = px.pie(
        _season_counts,
        names="season",
        values="hours",
        hole=0.5,
        color="season",
        color_discrete_map=season_colors,
        title="Hours by season",
    )
    season_pie.update_traces(textposition="inside", textinfo="percent+label")
    season_pie.update_layout(showlegend=False, margin=dict(l=10, r=10, t=40, b=10), height=340)
    _ = season_pie
    return (season_pie,)


@app.cell
def _(filtered_df, peak_colors, px):
    _peak_counts = (
        filtered_df["peak_label"].value_counts().rename_axis("peak_label").reset_index(name="hours")
    )
    peak_pie = px.pie(
        _peak_counts,
        names="peak_label",
        values="hours",
        hole=0.5,
        color="peak_label",
        color_discrete_map=peak_colors,
        title="Peak vs non-peak hours",
    )
    peak_pie.update_traces(textposition="inside", textinfo="percent+label")
    peak_pie.update_layout(showlegend=False, margin=dict(l=10, r=10, t=40, b=10), height=340)
    _ = peak_pie
    return (peak_pie,)


@app.cell
def _(filtered_df, peak_colors, px):
    _peak_hour = (
        filtered_df.groupby(["hour_of_day", "peak_label"]).size().reset_index(name="hours")
    )
    peak_hour_bar = px.bar(
        _peak_hour,
        x="hour_of_day",
        y="hours",
        color="peak_label",
        color_discrete_map=peak_colors,
        barmode="stack",
        title="Peak-time hours by hour of day",
    )
    peak_hour_bar.update_xaxes(dtick=1)
    peak_hour_bar.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=340)
    _ = peak_hour_bar
    return (peak_hour_bar,)


@app.cell
def _(filtered_df, peak_colors, px):
    _dow_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    _peak_dow = (
        filtered_df.groupby(["day_of_week", "peak_label"]).size().reset_index(name="hours")
    )
    peak_dow_bar = px.bar(
        _peak_dow,
        x="day_of_week",
        y="hours",
        color="peak_label",
        color_discrete_map=peak_colors,
        barmode="stack",
        category_orders={"day_of_week": _dow_order},
        title="Peak-time hours by day of week",
    )
    peak_dow_bar.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=340)
    _ = peak_dow_bar
    return (peak_dow_bar,)


@app.cell
def _(filtered_df, px):
    _peak_month = (
        filtered_df.groupby(["month", "month_name"])["peak_time_flag"]
        .mean()
        .reset_index(name="peak_share")
        .sort_values("month")
    )
    peak_month_bar = px.bar(
        _peak_month,
        x="month_name",
        y="peak_share",
        title="Share of peak-time hours by month",
        color_discrete_sequence=["#ff7f0e"],
        labels={"month_name": "Month", "peak_share": "Peak share"},
    )
    peak_month_bar.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=340)
    _ = peak_month_bar
    return (peak_month_bar,)


@app.cell
def _(filtered_df, px, weekend_colors):
    _weekend_hour = (
        filtered_df.groupby(["hour_of_day", "weekend_label"]).size().reset_index(name="hours")
    )
    weekend_hour_bar = px.bar(
        _weekend_hour,
        x="hour_of_day",
        y="hours",
        color="weekend_label",
        color_discrete_map=weekend_colors,
        barmode="stack",
        title="Weekend vs weekday hours by hour of day",
    )
    weekend_hour_bar.update_xaxes(dtick=1)
    weekend_hour_bar.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=340)
    _ = weekend_hour_bar
    return (weekend_hour_bar,)


@app.cell
def _(filtered_df, px, weekend_colors):
    _dow_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    _weekend_dow = (
        filtered_df.groupby(["day_of_week", "weekend_label"]).size().reset_index(name="hours")
    )
    weekend_dow_bar = px.bar(
        _weekend_dow,
        x="day_of_week",
        y="hours",
        color="weekend_label",
        color_discrete_map=weekend_colors,
        barmode="stack",
        category_orders={"day_of_week": _dow_order},
        title="Weekend vs weekday hours by day of week",
    )
    weekend_dow_bar.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=340)
    _ = weekend_dow_bar
    return (weekend_dow_bar,)


@app.cell
def _(filtered_df, pd, px, season_colors):
    _season_month = (
        filtered_df.groupby(["month", "month_name", "season"]).size().reset_index(name="hours")
    )
    month_season_bar = px.bar(
        _season_month,
        x="month_name",
        y="hours",
        color="season",
        color_discrete_map=season_colors,
        barmode="stack",
        category_orders={"month_name": list(pd.date_range("2025-01-01", periods=12, freq="MS").strftime("%B"))},
        title="Season composition by month",
    )
    month_season_bar.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=340)
    _ = month_season_bar
    return (month_season_bar,)


@app.cell
def _(filtered_df, go, px):
    _peak_heat = (
        filtered_df.groupby(["hour_of_day", "month"])["peak_time_flag"].mean().unstack(fill_value=0)
    )
    if _peak_heat.empty:
        peak_heatmap = go.Figure()
    else:
        peak_heatmap = px.imshow(
            _peak_heat,
            labels=dict(x="Month", y="Hour of day", color="Peak share"),
            title="Peak-hour share: hour of day × month",
            color_continuous_scale="YlOrRd",
            zmin=0,
            zmax=1,
            aspect="auto",
        )
        peak_heatmap.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=440)
    _ = peak_heatmap
    return (peak_heatmap,)


@app.cell
def _(filtered_df, px):
    _weekend_month = (
        filtered_df.groupby(["month", "month_name"])["is_weekend"]
        .mean()
        .reset_index(name="weekend_share")
        .sort_values("month")
    )
    weekend_month_bar = px.bar(
        _weekend_month,
        x="month_name",
        y="weekend_share",
        title="Share of weekend hours by month",
        color_discrete_sequence=["#2ca02c"],
        labels={"month_name": "Month", "weekend_share": "Weekend share"},
    )
    weekend_month_bar.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=340)
    _ = weekend_month_bar
    return (weekend_month_bar,)


@app.cell
def _(filtered_df, mo):
    tf_explorer = mo.ui.data_explorer(filtered_df)
    tf_table = mo.ui.table(filtered_df.head(1000), page_size=20, selection=None)
    return tf_explorer, tf_table


@app.cell
def _(
    kpi_cards,
    mo,
    month_season_bar,
    peak_dow_bar,
    peak_heatmap,
    peak_hour_bar,
    peak_month_bar,
    peak_pie,
    season_pie,
    tf_explorer,
    tf_table,
    weekend_dow_bar,
    weekend_hour_bar,
    weekend_month_bar,
):
    overview_tab = mo.vstack(
        [
            mo.md("### Overview"),
            kpi_cards,
            mo.hstack([season_pie, peak_pie], widths=[1, 1]),
            mo.hstack([peak_hour_bar, peak_dow_bar], widths=[1, 1]),
        ],
        gap=1,
    )
    peak_tab = mo.vstack(
        [
            mo.md("### Peak-time structure"),
            peak_heatmap,
            peak_hour_bar,
            mo.hstack([peak_dow_bar, peak_month_bar], widths=[1, 1]),
        ],
        gap=1,
    )
    calendar_tab = mo.vstack(
        [
            mo.md("### Seasons & calendar"),
            month_season_bar,
            mo.hstack([weekend_hour_bar, weekend_dow_bar], widths=[1, 1]),
            weekend_month_bar,
        ],
        gap=1,
    )
    data_tab = mo.vstack(
        [
            mo.md("### Explore the filtered data"),
            tf_explorer,
            tf_table,
        ],
        gap=1,
    )
    tabs = mo.ui.tabs(
        {
            "Overview": overview_tab,
            "Peak Time": peak_tab,
            "Seasons & Calendar": calendar_tab,
            "Data": data_tab,
        },
        lazy=True,
    )
    main_content = mo.vstack([mo.md("### Dashboard"), tabs], gap=0.5)
    # mo.hstack([sidebar_filters, main_content], align="stretch", widths=[1, 4])
    main_content
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
