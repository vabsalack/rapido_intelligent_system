import marimo

__generated_with = "0.24.0"
app = marimo.App(auto_download=["ipynb"])


@app.cell
def _():
    import pandas as pd
    import seaborn as sns
    import marimo as mo
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    from rapido_intelligent_system.dataset_mnb import (classify_columns, 
                                                   initial_inspect_str_attr, 
                                                   validate_identifier_columns)


    return (
        classify_columns,
        initial_inspect_str_attr,
        mo,
        mpl,
        pd,
        plt,
        sns,
        validate_identifier_columns,
    )


@app.cell
def _(mo, mpl, pd, sns):
    from packaging.version import Version
    _libs_used = [pd, sns, mo, mpl]
    for _lib in _libs_used:
        print(_lib.__name__, Version(_lib.__version__))
    return


@app.cell
def _():
    from rapido_intelligent_system.config_mnb import RAW_DATA_DIR

    return (RAW_DATA_DIR,)


@app.cell
def _(RAW_DATA_DIR):
    # initialize raw dataset paths
    bookings_path = RAW_DATA_DIR / "bookings.csv"
    customers_path = RAW_DATA_DIR / "customers.csv"
    drivers_path = RAW_DATA_DIR / "drivers.csv"
    location_demand_path = RAW_DATA_DIR / "location_demand.csv"
    time_features_path = RAW_DATA_DIR / "time_features.csv"
    return (
        bookings_path,
        customers_path,
        drivers_path,
        location_demand_path,
        time_features_path,
    )


@app.cell
def _(
    bookings_path,
    customers_path,
    drivers_path,
    location_demand_path,
    time_features_path,
):
    # checks all the file exists
    dataset_path = [bookings_path, customers_path, drivers_path, location_demand_path, time_features_path]
    for item in dataset_path:
        assert item.is_file() == True
    else:
        print("All data files exists")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Quick look at Bookings.csv
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## overview of df only
    """)
    return


@app.cell
def _(bookings_path, pd):
    # load bookings file
    bookings = pd.read_csv(bookings_path)
    return (bookings,)


@app.cell
def _(bookings):
    bookings.info()
    return


@app.cell
def _(bookings, pd):
    pd.concat([bookings.head(5), bookings.tail(5)], ignore_index=False)
    return


@app.cell
def _(bookings):
    bookings.sample(n=20, ignore_index=False, random_state=55)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    classify colunms to id, num and str groups
    """)
    return


@app.cell
def _(bookings, classify_columns):
    bookings_id_cols, bookings_num_cols, bookings_str_cols, _is_match  = classify_columns(bookings)
    _is_match
    return bookings_id_cols, bookings_num_cols, bookings_str_cols


@app.cell
def _(bookings_id_cols, bookings_num_cols, bookings_str_cols):
    bookings_id_cols, bookings_num_cols, bookings_str_cols
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## stats look at nums attrbs
    """)
    return


@app.cell
def _(bookings, bookings_num_cols):
    bookings[bookings_num_cols]
    return


@app.cell
def _(bookings, sns):
    # Count Plot (most used in EDA)
    sns.countplot(data=bookings, x='is_weekend', hue="booking_status")

    # With target hue (powerful)
    # sns.countplot(data=df, x='department', hue='target')

    # Horizontal for many categories
    # sns.countplot(data=df, y='job_title')
    return


@app.cell
def _(bookings):
    bookings.describe().round(2)
    return


@app.cell
def _(bookings, plt):
    bookings.hist(bins=50, figsize=(12, 8))
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## stats look at strs attrbs
    """)
    return


@app.cell
def _(bookings, bookings_str_cols):
    bookings[bookings_str_cols]
    return


@app.cell
def _(bookings_str_cols):
    bookings_str_cols
    return


@app.cell
def _(bookings, initial_inspect_str_attr):
    initial_inspect_str_attr(bookings,"booking_date")
    return


@app.cell
def _(bookings, initial_inspect_str_attr):
    initial_inspect_str_attr(bookings, "booking_time")
    return


@app.cell
def _(bookings, initial_inspect_str_attr):
    initial_inspect_str_attr(bookings, "day_of_week")
    return


@app.cell
def _(bookings, initial_inspect_str_attr):
    initial_inspect_str_attr(bookings, "city")
    return


@app.cell
def _(bookings, initial_inspect_str_attr):
    initial_inspect_str_attr(bookings, "pickup_location")
    return


@app.cell
def _(bookings, initial_inspect_str_attr):
    initial_inspect_str_attr(bookings, "drop_location")
    return


@app.cell
def _(bookings, initial_inspect_str_attr):
    initial_inspect_str_attr(bookings, "vehicle_type")
    return


@app.cell
def _(bookings, initial_inspect_str_attr):
    initial_inspect_str_attr(bookings, "traffic_level")
    return


@app.cell
def _(bookings, initial_inspect_str_attr):
    initial_inspect_str_attr(bookings, "weather_condition")
    return


@app.cell
def _(bookings, initial_inspect_str_attr):
    initial_inspect_str_attr(bookings, "booking_status")
    return


@app.cell
def _(bookings, initial_inspect_str_attr):
    initial_inspect_str_attr(bookings, "incomplete_ride_reason")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## stats look at ids attrbs
    """)
    return


@app.cell
def _(bookings_id_cols):
    bookings_id_cols
    return


@app.cell
def _(bookings, bookings_id_cols):
    bookings[bookings_id_cols]
    return


@app.cell
def _(bookings, bookings_id_cols, validate_identifier_columns):
    validate_identifier_columns(bookings, bookings_id_cols)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Quick look at Customers.csv
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## overview of df only
    """)
    return


@app.cell
def _(customers_path, pd):
    # load customer file
    customers = pd.read_csv(customers_path)
    return (customers,)


@app.cell
def _(customers):
    customers.info()
    return


@app.cell
def _(customers, pd):
    pd.concat([customers.head(5), customers.tail(5)], ignore_index=False)
    return


@app.cell
def _(customers):
    customers.sample(n=20, ignore_index=False, random_state=55)
    return


@app.cell
def _(classify_columns, customers):
    customers_id_cols, customers_num_cols, customers_str_cols, _is_match  = classify_columns(customers)
    _is_match
    return customers_id_cols, customers_num_cols, customers_str_cols


@app.cell
def _(customers_id_cols, customers_num_cols, customers_str_cols):
    customers_id_cols, customers_num_cols, customers_str_cols
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## stats look at nums attrbs
    """)
    return


@app.cell
def _(customers):
    customers.describe().round(2)
    return


@app.cell
def _(customers, customers_num_cols):
    customers[customers_num_cols]
    return


@app.cell
def _(customers, plt):
    customers.hist(bins=50, figsize=(12, 8))
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## stats look at strs attrbs
    """)
    return


@app.cell
def _(customers, customers_str_cols):
    customers[customers_str_cols]
    return


@app.cell
def _(customers_str_cols):
    customers_str_cols
    return


@app.cell
def _(customers, initial_inspect_str_attr):
    initial_inspect_str_attr(customers, "customer_gender")
    return


@app.cell
def _(customers, initial_inspect_str_attr):
    initial_inspect_str_attr(customers, "customer_city")
    return


@app.cell
def _(customers, initial_inspect_str_attr):
    initial_inspect_str_attr(customers, "preferred_vehicle_type")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## stats look at ids attrbs
    """)
    return


@app.cell
def _(customers_id_cols):
    customers_id_cols
    return


@app.cell
def _(customers, customers_id_cols):
    customers[customers_id_cols]
    return


@app.cell
def _(customers, customers_id_cols, validate_identifier_columns):
    validate_identifier_columns(customers, customers_id_cols)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Quick look at drivers.csv
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## overview of df only
    """)
    return


@app.cell
def _(drivers_path, pd):
    # load drivers file
    drivers = pd.read_csv(drivers_path)
    return (drivers,)


@app.cell
def _(drivers):
    drivers.info()
    return


@app.cell
def _(drivers, pd):
    pd.concat([drivers.head(5), drivers.tail(5)], ignore_index=False)
    return


@app.cell
def _(drivers):
    drivers.sample(n=20, ignore_index=False, random_state=55)
    return


@app.cell
def _(classify_columns, drivers):
    drivers_id_cols, drivers_num_cols, drivers_str_cols, _is_match  = classify_columns(drivers)
    _is_match
    return drivers_id_cols, drivers_num_cols, drivers_str_cols


@app.cell
def _(drivers_id_cols, drivers_num_cols, drivers_str_cols):
    drivers_id_cols, drivers_num_cols, drivers_str_cols
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## stats look at nums attrbs
    """)
    return


@app.cell
def _(drivers, drivers_num_cols):
    drivers[drivers_num_cols]
    return


@app.cell
def _(drivers):
    drivers.describe().round(2)
    return


@app.cell
def _(drivers, plt):
    drivers.hist(bins=50, figsize=(12, 8))
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## stats look at strs attrbs
    """)
    return


@app.cell
def _(drivers, drivers_str_cols):
    drivers[drivers_str_cols]
    return


@app.cell
def _(drivers_str_cols):
    drivers_str_cols
    return


@app.cell
def _(drivers, initial_inspect_str_attr):
    initial_inspect_str_attr(drivers, "driver_city")
    return


@app.cell
def _(drivers, initial_inspect_str_attr):
    initial_inspect_str_attr(drivers, "vehicle_type")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## stats look at ids attrbs
    """)
    return


@app.cell
def _(drivers_id_cols):
    drivers_id_cols
    return


@app.cell
def _(drivers, drivers_id_cols):
    drivers[drivers_id_cols]
    return


@app.cell
def _(drivers, drivers_id_cols, validate_identifier_columns):
    validate_identifier_columns(drivers, drivers_id_cols)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Quick look at location_demand.csv
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## overview of df only
    """)
    return


@app.cell
def _(location_demand_path, pd):
    # load location demand file
    loc_demand = pd.read_csv(location_demand_path)
    return (loc_demand,)


@app.cell
def _(loc_demand):
    loc_demand.info()
    return


@app.cell
def _(loc_demand):
    loc_demand
    return


@app.cell
def _(loc_demand, pd):
    pd.concat([loc_demand.head(5), loc_demand.tail(5)], ignore_index=False)
    return


@app.cell
def _(loc_demand):
    loc_demand.sample(n=20, ignore_index=False, random_state=55)
    return


@app.cell
def _(classify_columns, loc_demand):
    loc_demand_id_cols, loc_demand_num_cols, loc_demand_str_cols, _is_match  = classify_columns(loc_demand)
    _is_match
    return loc_demand_id_cols, loc_demand_num_cols, loc_demand_str_cols


@app.cell
def _(loc_demand_id_cols, loc_demand_num_cols, loc_demand_str_cols):
    loc_demand_id_cols, loc_demand_num_cols, loc_demand_str_cols
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## stats look at nums attrbs
    """)
    return


@app.cell
def _(loc_demand, loc_demand_num_cols):
    loc_demand[loc_demand_num_cols]
    return


@app.cell
def _(loc_demand):
    loc_demand.describe().round(2)
    return


@app.cell
def _(loc_demand, plt):
    loc_demand.hist(bins=50, figsize=(12, 8))
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## stats look at strs attrbs
    """)
    return


@app.cell
def _(loc_demand, loc_demand_str_cols):
    loc_demand[loc_demand_str_cols]
    return


@app.cell
def _(loc_demand_str_cols):
    loc_demand_str_cols
    return


@app.cell
def _(initial_inspect_str_attr, loc_demand):
    initial_inspect_str_attr(loc_demand, "city")
    return


@app.cell
def _(initial_inspect_str_attr, loc_demand):
    initial_inspect_str_attr(loc_demand, "pickup_location")
    return


@app.cell
def _(initial_inspect_str_attr, loc_demand):
    initial_inspect_str_attr(loc_demand, "vehicle_type")
    return


@app.cell
def _(initial_inspect_str_attr, loc_demand):
    initial_inspect_str_attr(loc_demand, "demand_level")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## stats look at ids attrbs
    """)
    return


@app.cell
def _(loc_demand_id_cols):
    loc_demand_id_cols
    return


@app.cell
def _(loc_demand, loc_demand_id_cols):
    loc_demand[loc_demand_id_cols]
    return


@app.cell
def _(loc_demand, loc_demand_id_cols, validate_identifier_columns):
    validate_identifier_columns(loc_demand, loc_demand_id_cols)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Quick look at time_features.csv
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## overview of df only
    """)
    return


@app.cell
def _(pd, time_features_path):
    # load bookings file
    time_feat = pd.read_csv(time_features_path)
    return (time_feat,)


@app.cell
def _(time_feat):
    time_feat.info()
    return


@app.cell
def _(pd, time_feat):
    pd.concat([time_feat.head(5), time_feat.tail(5)], ignore_index=False)
    return


@app.cell
def _(time_feat):
    time_feat.sample(n=20, ignore_index=False, random_state=55)
    return


@app.cell
def _(classify_columns, time_feat):
    time_feat_id_cols, time_feat_num_cols, time_feat_str_cols, _is_match  = classify_columns(time_feat)
    _is_match
    return time_feat_id_cols, time_feat_num_cols, time_feat_str_cols


@app.cell
def _(time_feat_id_cols, time_feat_num_cols, time_feat_str_cols):
    time_feat_id_cols, time_feat_num_cols, time_feat_str_cols
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## stats look at nums attrbs
    """)
    return


@app.cell
def _(time_feat, time_feat_num_cols):
    time_feat[time_feat_num_cols]
    return


@app.cell
def _(time_feat):
    time_feat.describe().round(2)
    return


@app.cell
def _(plt, time_feat):
    time_feat.hist(bins=50, figsize=(12, 8))
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## stats look at strs attrbs
    """)
    return


@app.cell
def _(time_feat, time_feat_str_cols):
    time_feat[time_feat_str_cols]
    return


@app.cell
def _(time_feat_str_cols):
    time_feat_str_cols
    return


@app.cell
def _():
    # skipping this because it takes lots of time
    # initial_inspect_str_attr(time_feat, "datetime")
    return


@app.cell
def _(initial_inspect_str_attr, time_feat):
    initial_inspect_str_attr(time_feat, "day_of_week")
    return


@app.cell
def _(initial_inspect_str_attr, time_feat):
    initial_inspect_str_attr(time_feat, "season")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## stats look at ids attrbs
    """)
    return


@app.cell
def _(time_feat_id_cols):
    time_feat_id_cols
    return


@app.cell
def _(time_feat, time_feat_id_cols):
    time_feat[time_feat_id_cols]
    return


@app.cell
def _(time_feat, time_feat_id_cols, validate_identifier_columns):
    validate_identifier_columns(time_feat, time_feat_id_cols)
    return


if __name__ == "__main__":
    app.run()
