import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from rapido_intelligent_system.dataset import RAW_DATA_DIR
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly

    return RAW_DATA_DIR, pd


@app.cell
def _(RAW_DATA_DIR):
    bookings_path = RAW_DATA_DIR / "bookings.csv"
    return (bookings_path,)


@app.cell
def _(bookings_path, pd):
    booking_df = pd.read_csv(bookings_path)

    return (booking_df,)


@app.cell
def _(booking_df):
    booking_df
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
