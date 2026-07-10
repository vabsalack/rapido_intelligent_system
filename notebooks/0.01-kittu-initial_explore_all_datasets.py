import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import pandas as pd
    import seaborn as sns
    import marimo as mo
    import matplotlib.pyplot as plt

    from rapido_intelligent_system.dataset import classify_columns
    from rapido_intelligent_system.dataset import initial_inspect_str_attr


    return classify_columns, initial_inspect_str_attr, mo, pd, plt, sns


@app.cell
def _(pd, sns):
    from packaging.version import Version
    print(Version(pd.__version__))
    print(Version(sns.__version__))
    return


@app.cell
def _():
    from rapido_intelligent_system.config import RAW_DATA_DIR

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
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### (bookings) take a quick look at the data structure
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    overview of bookings.info()

    0. 16.8 MB memory
    1. *10 Lakh*: rows
    2. *22*: columns
    3. Attributes inspect:
       1. 6: float64
       2. 2: int64
       3. 14: str
    4. 2: columns with nulls
       1. float64: 'actual_ride_time_min'
       2. str: 'incomplete_ride_reason'
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    population sample quick look (2 sample dfs)
    - understand nature of each attribute
    1. head 5 and tail 5; total 10 samples
    2. uniform sampling of 20 samples
    """)
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
    bookings_id_cols, bookings_num_cols, bookings_str_cols, _  = classify_columns(bookings)
    _
    return bookings_id_cols, bookings_num_cols, bookings_str_cols


@app.cell
def _(bookings_id_cols, bookings_num_cols, bookings_str_cols):
    bookings_id_cols, bookings_num_cols, bookings_str_cols
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    statistical quick look of num attrs
    """)
    return


@app.cell
def _(bookings):
    bookings.describe().round(2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    num attrs visual sense
    """)
    return


@app.cell
def _(bookings, plt):
    bookings.hist(bins=50, figsize=(12, 8))
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    statistical quick look and visual sense of str attributes
    """)
    return


@app.cell
def _(bookings, bookings_str_cols):
    bookings[bookings_str_cols]
    return


@app.cell
def _(bookings, bookings_str_cols, initial_inspect_str_attr):
    initial_inspect_str_attr(bookings, bookings_str_cols[2])
    return


@app.cell
def _(bookings, bookings_str_cols, initial_inspect_str_attr):
    initial_inspect_str_attr(bookings, bookings_str_cols[3])

    return


@app.cell
def _(bookings, bookings_str_cols, initial_inspect_str_attr):
    initial_inspect_str_attr(bookings, bookings_str_cols[4])

    return


@app.cell
def _(bookings, bookings_str_cols, initial_inspect_str_attr):
    initial_inspect_str_attr(bookings, bookings_str_cols[5])
    return


@app.cell
def _(bookings, bookings_str_cols, initial_inspect_str_attr):
    initial_inspect_str_attr(bookings, bookings_str_cols[6])
    return


@app.cell
def _(bookings, bookings_str_cols, initial_inspect_str_attr):
    initial_inspect_str_attr(bookings, bookings_str_cols[7])
    return


@app.cell
def _(bookings, bookings_str_cols, initial_inspect_str_attr):
    initial_inspect_str_attr(bookings, bookings_str_cols[8])
    return


@app.cell
def _(bookings, bookings_str_cols, initial_inspect_str_attr):
    initial_inspect_str_attr(bookings, bookings_str_cols[9])
    return


@app.cell
def _(bookings, bookings_str_cols, initial_inspect_str_attr):
    initial_inspect_str_attr(bookings, bookings_str_cols[10])
    return


@app.cell
def _(bookings, bookings_str_cols):
    bookings[bookings_str_cols][:5]
    return


if __name__ == "__main__":
    app.run()
