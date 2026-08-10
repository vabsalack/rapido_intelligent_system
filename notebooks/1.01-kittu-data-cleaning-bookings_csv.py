import marimo

__generated_with = "0.23.16"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    from rapido_intelligent_system.config import RAW_DATA_DIR

    return RAW_DATA_DIR, mo, pd


@app.cell
def _(RAW_DATA_DIR, pd):
    booking_df = pd.read_csv(RAW_DATA_DIR / "bookings.csv")
    return (booking_df,)


@app.cell
def _(booking_df):
    booking_df.columns
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    looking for missing values: MCAR, MAR and MNAR
    - from the dataset analysis, nulls values are in:
        1. "actual_ride_time_min" continuous ratio: 31654 = 31.654%
        2. "incomplete_ride_reason" str: 91630 = 91.630%
    """)
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    looking for exact duplicates
    """)
    return


@app.cell
def _(booking_df):
    # check duplicates on instance wise
    booking_df.duplicated().sum()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


if __name__ == "__main__":
    app.run()
