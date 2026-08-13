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

    from rapido_intelligent_system.dataset import RAW_DATA_DIR

    px.defaults.template = "plotly_white"
    return RAW_DATA_DIR, mo, pd, px


@app.cell
def _(RAW_DATA_DIR, pd):
    customer_path = RAW_DATA_DIR / "customers.csv"
    customer_df = pd.read_csv(customer_path)

    customer_df["risk_label"] = customer_df["customer_cancel_flag"].map(
        {1: "High risk", 0: "Normal"}
    )
    return (customer_df,)


@app.cell
def _():
    gender_colors = {"Male": "#1f77b4", "Female": "#e377c2", "Non-Binary": "#2ca02c"}
    risk_colors = {"High risk": "#d62728", "Normal": "#2ca02c"}
    return (risk_colors,)


@app.cell
def _(mo):
    mo.md(r"""
    # Rapido Customers — EDA Dashboard

    Interactive exploratory analysis of 10k ride-hailing customers across India.
    Use the sidebar filters to slice the data; every chart, card and table updates reactively.
    """)
    return


@app.cell
def _(customer_df, mo):
    city_selector = mo.ui.multiselect(
        customer_df["customer_city"].sort_values().unique().tolist(),
        value=customer_df["customer_city"].sort_values().unique().tolist(),
        label="Cities",
        full_width=False,
    )
    gender_selector = mo.ui.multiselect(
        customer_df["customer_gender"].unique().tolist(),
        value=customer_df["customer_gender"].unique().tolist(),
        label="Genders",
        full_width=False,
    )
    vehicle_selector = mo.ui.multiselect(
        customer_df["preferred_vehicle_type"].unique().tolist(),
        value=customer_df["preferred_vehicle_type"].unique().tolist(),
        label="Preferred vehicle",
        full_width=False,
    )
    risk_selector = mo.ui.radio(
        ["All", "High-risk only", "Normal only"],
        value="All",
        label="Risk segment",
    )
    age_range = mo.ui.range_slider(
        int(customer_df["customer_age"].min()),
        int(customer_df["customer_age"].max()),
        value=(int(customer_df["customer_age"].min()), int(customer_df["customer_age"].max())),
        label="Customer age",
        full_width=True,
    )
    rating_range = mo.ui.range_slider(
        1.0,
        5.0,
        value=(1.0, 5.0),
        label="Avg customer rating",
        step=0.1,
        full_width=True,
    )

    sidebar_filters = mo.vstack(
        [
            mo.md("**Filters**"),
            city_selector,
            gender_selector,
            vehicle_selector,
            risk_selector,
            age_range,
            rating_range,
        ],
        gap=0.7,
    )
    sidebar_filters
    return (
        age_range,
        city_selector,
        gender_selector,
        rating_range,
        risk_selector,
        vehicle_selector,
    )


@app.cell
def _(
    age_range,
    city_selector,
    customer_df,
    gender_selector,
    rating_range,
    risk_selector,
    vehicle_selector,
):
    filtered_df = customer_df
    filtered_df = filtered_df[filtered_df["customer_city"].isin(city_selector.value)]
    filtered_df = filtered_df[filtered_df["customer_gender"].isin(gender_selector.value)]
    filtered_df = filtered_df[
        filtered_df["preferred_vehicle_type"].isin(vehicle_selector.value)
    ]
    if risk_selector.value == "High-risk only":
        filtered_df = filtered_df[filtered_df["customer_cancel_flag"] == 1]
    elif risk_selector.value == "Normal only":
        filtered_df = filtered_df[filtered_df["customer_cancel_flag"] == 0]
    filtered_df = filtered_df[
        (filtered_df["customer_age"] >= age_range.value[0])
        & (filtered_df["customer_age"] <= age_range.value[1])
    ]
    filtered_df = filtered_df[
        (filtered_df["avg_customer_rating"] >= rating_range.value[0])
        & (filtered_df["avg_customer_rating"] <= rating_range.value[1])
    ]
    return (filtered_df,)


@app.cell
def _(filtered_df):
    total_customers = len(filtered_df)
    avg_age = filtered_df["customer_age"].mean()
    avg_bookings_per_customer = filtered_df["total_bookings"].mean()
    avg_cancellation_rate = filtered_df["cancellation_rate"].mean() * 100
    avg_rating = filtered_df["avg_customer_rating"].mean()
    high_risk_count = int(filtered_df["customer_cancel_flag"].sum())
    high_risk_rate = high_risk_count / total_customers * 100 if total_customers else 0
    return (
        avg_age,
        avg_bookings_per_customer,
        avg_cancellation_rate,
        avg_rating,
        high_risk_count,
        high_risk_rate,
        total_customers,
    )


@app.cell
def _(
    avg_age,
    avg_bookings_per_customer,
    avg_cancellation_rate,
    avg_rating,
    high_risk_count,
    high_risk_rate,
    mo,
    total_customers,
):
    kpi_cards = mo.hstack(
        [
            mo.stat(value=f"{total_customers:,}", label="Total customers", bordered=True),
            mo.stat(value=f"{avg_age:.1f} yrs", label="Avg age", bordered=True),
            mo.stat(
                value=f"{avg_bookings_per_customer:.1f}",
                label="Avg bookings / customer",
                bordered=True,
            ),
            mo.stat(
                value=f"{avg_cancellation_rate:.1f}%",
                label="Avg cancellation rate",
                bordered=True,
            ),
            mo.stat(value=f"{avg_rating:.2f}", label="Avg rating", bordered=True),
            mo.stat(
                value=f"{high_risk_count:,} ({high_risk_rate:.0f}%)",
                label="High-risk customers",
                bordered=True,
            ),
        ],
        widths="equal",
    )
    kpi_cards
    return (kpi_cards,)


@app.cell
def _(filtered_df, px):
    _gender_counts = (
        filtered_df["customer_gender"].value_counts().rename_axis("customer_gender").reset_index(name="count")
    )
    gender_pie = px.pie(
        _gender_counts,
        names="customer_gender",
        values="count",
        hole=0.5,
        color="customer_gender",
        title="Customers by gender",
    )
    gender_pie.update_traces(textposition="inside", textinfo="percent+label")
    gender_pie.update_layout(showlegend=False, margin=dict(l=10, r=10, t=40, b=10), height=340)
    _ = gender_pie
    return (gender_pie,)


@app.cell
def _(filtered_df, px):
    age_hist = px.histogram(
        filtered_df,
        x="customer_age",
        nbins=25,
        color="customer_gender",
        title="Age distribution by gender",
        labels={"customer_age": "Customer age (years)"},
    )
    age_hist.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=340)
    _ = age_hist
    return (age_hist,)


@app.cell
def _(filtered_df, px):
    age_box = px.box(
        filtered_df,
        x="customer_gender",
        y="customer_age",
        color="customer_gender",
        title="Age spread by gender",
        labels={"customer_gender": "Gender", "customer_age": "Age (years)"},
    )
    age_box.update_layout(showlegend=False, margin=dict(l=10, r=10, t=40, b=10), height=340)
    _ = age_box
    return (age_box,)


@app.cell
def _(filtered_df, px):
    _city_counts = (
        filtered_df.groupby("customer_city")
        .size()
        .reset_index(name="count")
        .sort_values("count")
    )
    city_bar = px.bar(
        _city_counts,
        x="count",
        y="customer_city",
        orientation="h",
        title="Customers by city",
        color_discrete_sequence=["#636efa"],
    )
    city_bar.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=340)
    _ = city_bar
    return (city_bar,)


@app.cell
def _(filtered_df, px):
    _vehicle_counts = filtered_df["preferred_vehicle_type"].value_counts().reset_index()
    _vehicle_counts.columns = ["preferred_vehicle_type", "count"]
    vehicle_bar = px.bar(
        _vehicle_counts,
        x="preferred_vehicle_type",
        y="count",
        title="Preferred vehicle type",
        color_discrete_sequence=["#ff7f0e"],
    )
    vehicle_bar.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=340)
    _ = vehicle_bar
    return (vehicle_bar,)


@app.cell
def _(filtered_df, px):
    bookings_hist = px.histogram(
        filtered_df,
        x="total_bookings",
        nbins=30,
        title="Total bookings per customer",
        labels={"total_bookings": "Total bookings"},
        color_discrete_sequence=["#9467bd"],
    )
    bookings_hist.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=340)
    _ = bookings_hist
    return (bookings_hist,)


@app.cell
def _(filtered_df, px):
    _booking_gender = (
        filtered_df.groupby("customer_gender")["total_bookings"].mean().reset_index(name="avg_bookings")
    )
    booking_gender_bar = px.bar(
        _booking_gender,
        x="customer_gender",
        y="avg_bookings",
        title="Avg bookings per customer by gender",
        color="customer_gender",
        labels={"avg_bookings": "Avg bookings"},
    )
    booking_gender_bar.update_layout(showlegend=False, margin=dict(l=10, r=10, t=40, b=10), height=340)
    _ = booking_gender_bar
    return (booking_gender_bar,)


@app.cell
def _(filtered_df, px, risk_colors):
    cancel_hist = px.histogram(
        filtered_df,
        x="cancellation_rate",
        nbins=40,
        color="risk_label",
        color_discrete_map=risk_colors,
        title="Cancellation rate distribution",
        labels={"cancellation_rate": "Cancellation rate"},
    )
    cancel_hist.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=340)
    _ = cancel_hist
    return (cancel_hist,)


@app.cell
def _(filtered_df, px, risk_colors):
    _flag_counts = (
        filtered_df["risk_label"].value_counts().rename_axis("risk_label").reset_index(name="count")
    )
    flag_pie = px.pie(
        _flag_counts,
        names="risk_label",
        values="count",
        hole=0.5,
        color="risk_label",
        color_discrete_map=risk_colors,
        title="High-risk customers",
    )
    flag_pie.update_traces(textposition="inside", textinfo="percent+label")
    flag_pie.update_layout(showlegend=False, margin=dict(l=10, r=10, t=40, b=10), height=340)
    _ = flag_pie
    return (flag_pie,)


@app.cell
def _(filtered_df, px):
    rating_hist = px.histogram(
        filtered_df,
        x="avg_customer_rating",
        nbins=40,
        title="Avg customer rating distribution",
        labels={"avg_customer_rating": "Avg rating"},
        color_discrete_sequence=["#17becf"],
    )
    rating_hist.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=340)
    _ = rating_hist
    return (rating_hist,)


@app.cell
def _(filtered_df, px, risk_colors):
    risk_scatter = px.scatter(
        filtered_df,
        x="avg_customer_rating",
        y="cancellation_rate",
        color="risk_label",
        color_discrete_map=risk_colors,
        opacity=0.6,
        title="Rating vs cancellation rate",
        labels={
            "avg_customer_rating": "Avg rating",
            "cancellation_rate": "Cancellation rate",
        },
    )
    risk_scatter.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=380)
    _ = risk_scatter
    return (risk_scatter,)


@app.cell
def _(filtered_df, px):
    tenure_scatter = px.scatter(
        filtered_df,
        x="customer_signup_days_ago",
        y="total_bookings",
        color="customer_city",
        opacity=0.6,
        title="Booking activity vs signup recency",
        labels={
            "customer_signup_days_ago": "Days since signup",
            "total_bookings": "Total bookings",
        },
    )
    tenure_scatter.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=380)
    _ = tenure_scatter
    return (tenure_scatter,)


@app.cell
def _(filtered_df, px):
    _cancel_city = (
        filtered_df.groupby("customer_city")["cancellation_rate"]
        .mean()
        .reset_index(name="avg_cancel_rate")
        .sort_values("avg_cancel_rate")
    )
    cancel_city_bar = px.bar(
        _cancel_city,
        x="avg_cancel_rate",
        y="customer_city",
        orientation="h",
        title="Avg cancellation rate by city",
        color="avg_cancel_rate",
        color_continuous_scale="Reds",
        labels={"avg_cancel_rate": "Avg cancellation rate"},
    )
    cancel_city_bar.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=340)
    _ = cancel_city_bar
    return (cancel_city_bar,)


@app.cell
def _(filtered_df, px):
    _rating_vehicle = (
        filtered_df.groupby("preferred_vehicle_type")["avg_customer_rating"]
        .mean()
        .reset_index(name="avg_rating")
    )
    rating_vehicle_bar = px.bar(
        _rating_vehicle,
        x="preferred_vehicle_type",
        y="avg_rating",
        title="Avg rating by preferred vehicle",
        color_discrete_sequence=["#2ca02c"],
        labels={"preferred_vehicle_type": "Vehicle type", "avg_rating": "Avg rating"},
    )
    rating_vehicle_bar.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=340)
    _ = rating_vehicle_bar
    return (rating_vehicle_bar,)


@app.cell
def _(filtered_df, px):
    _bookings_city = (
        filtered_df.groupby("customer_city")["total_bookings"]
        .mean()
        .reset_index(name="avg_bookings")
        .sort_values("avg_bookings")
    )
    bookings_city_bar = px.bar(
        _bookings_city,
        x="avg_bookings",
        y="customer_city",
        orientation="h",
        title="Avg bookings per customer by city",
        color_discrete_sequence=["#636efa"],
        labels={"avg_bookings": "Avg bookings"},
    )
    bookings_city_bar.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=340)
    _ = bookings_city_bar
    return (bookings_city_bar,)


@app.cell
def _(filtered_df, mo):
    customer_explorer = mo.ui.data_explorer(filtered_df)
    customer_table = mo.ui.table(filtered_df.head(1000), page_size=20, selection=None)
    return customer_explorer, customer_table


@app.cell
def _(
    age_box,
    age_hist,
    booking_gender_bar,
    bookings_city_bar,
    bookings_hist,
    cancel_city_bar,
    cancel_hist,
    city_bar,
    customer_explorer,
    customer_table,
    flag_pie,
    gender_pie,
    kpi_cards,
    mo,
    rating_hist,
    rating_vehicle_bar,
    risk_scatter,
    tenure_scatter,
    vehicle_bar,
):
    overview_tab = mo.vstack(
        [
            mo.md("### Overview"),
            kpi_cards,
            mo.hstack([gender_pie, city_bar], widths=[1, 1]),
            mo.hstack([vehicle_bar, flag_pie], widths=[1, 1]),
        ],
        gap=1,
    )
    demographics_tab = mo.vstack(
        [
            mo.md("### Demographics"),
            mo.hstack([age_hist, age_box], widths=[1, 1]),
            gender_pie,
        ],
        gap=1,
    )
    behavior_tab = mo.vstack(
        [
            mo.md("### Booking behavior"),
            mo.hstack([bookings_hist, booking_gender_bar], widths=[1, 1]),
            mo.hstack([cancel_hist, flag_pie], widths=[1, 1]),
            tenure_scatter,
        ],
        gap=1,
    )
    quality_tab = mo.vstack(
        [
            mo.md("### Quality & risk"),
            mo.hstack([rating_hist, risk_scatter], widths=[1, 1]),
            mo.hstack([cancel_city_bar, rating_vehicle_bar], widths=[1, 1]),
            bookings_city_bar,
        ],
        gap=1,
    )
    data_tab = mo.vstack(
        [
            mo.md("### Explore the filtered data"),
            customer_explorer,
            customer_table,
        ],
        gap=1,
    )
    tabs = mo.ui.tabs(
        {
            "Overview": overview_tab,
            "Demographics": demographics_tab,
            "Behavior": behavior_tab,
            "Quality & Risk": quality_tab,
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
