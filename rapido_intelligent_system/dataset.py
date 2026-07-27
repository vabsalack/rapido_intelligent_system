from pathlib import Path

import gdown

from loguru import logger
from tqdm import tqdm
import typer

from rapido_intelligent_system.config import (PROCESSED_DATA_DIR, 
                                              RAW_DATA_DIR, 
                                              GDRIVE_LINK)


import pandas as pd
import seaborn as sns


from rich import print

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    # ----------------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Processing dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Processing dataset complete.")
    # -----------------------------------------


@app.command("download")
def download_gdrive_folder(drive_web_url: str = GDRIVE_LINK, 
                           output_dir: str = RAW_DATA_DIR) -> Path:
    """
    Downloads datasets from author's, keerhtivasan, Google drive to default RAW_DATA_DIR
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        files = gdown.download_folder(
            url=drive_web_url,
            output=str(output_dir),
            quiet=False,
            use_cookies=False,
        )
        
        print(files)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        raise
    
    return output_dir

def classify_columns(df):
    """
    input:
        df
    return
        id_col_names, num_col_names, text_col_names, bool
    function:
        classify and group columns names into identifiers, numerical and text types by their name only.
        It also check the match of no of columns input and no of columns output
    """

    identifiers = []
    numerical = []
    text = []

    og_cols_count = len(df.columns)

    for col in df.columns:
        if col.lower().endswith("_id") or col.lower() == "id":
            if col not in identifiers:
                identifiers.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            if col not in numerical:
                numerical.append(col)
        else:
            if col not in text:
                text.append(col)

    md_cols_count = sum([len(item) for item in [identifiers, numerical, text]])
    columsn_count_equal = "classified columns count = total columns count" if (og_cols_count == md_cols_count) else "classified columns count != total columns count"

    return identifiers, numerical, text, columsn_count_equal


def initial_inspect_str_attr(df: pd.DataFrame, 
                             attr_name: str, 
                             *,
                             value_count: bool = True, 
                             diagram: bool = True, 
                             miss_label: str = "<missing>"):
    """
    inputs: 
        df and str attr name
    returns:
        value_count series and count plot
    function: 
        perform value counts: number of unique categories and its frequency of occurences,
                horizontal count plot: for quick visual sense of frequncies across unique categoreis,
                for the given str/categorical attribute in given dataframe.
    """

    missing_label = miss_label
    plot_attr = df[attr_name].where(df[attr_name].notna(), missing_label)


    # freq_count = df[attr_name].value_counts(dropna=False, sort=True)
    freq_count = plot_attr.value_counts(sort=True)

    if diagram:
        order = freq_count.index

        plot = sns.countplot(y=plot_attr, order=order)
        plot.grid(True, axis="x", linestyle="--", alpha=0.7)
        plot.set_ylabel(attr_name)
        plot.set_xlabel("Count Frequencies")

    if value_count and diagram:
        return freq_count, plot

    if value_count:
        return freq_count

    if diagram:
        return plot

def validate_identifier_columns(df, id_columns):

    infos = ["id_columns is empty"]
    _flag = True
    for col in id_columns:
        if _flag:
            infos.pop()
            _flag = False
            
        # check if it exists
        if col not in df.columns:
            infos.append(f"{col}: not found in given dataframe")
            continue
        
        non_null_values = df[col].dropna()

        # check if all vals are null
        if non_null_values.empty:
            infos.append(f"{col}: all values are null")
            continue

        # check how many vals are null
        diff = len(df[col]) - len(non_null_values)
        if diff:
            infos.append(f"{col}: has {diff} null values")
        else:
            infos.append(f"{col}: has no null values")

        # check all non vals are unique
        if non_null_values.nunique() != len(non_null_values):
            infos.append(f"{col}: values are not unique")
        else:
            infos.append(f"{col}: all values are unique")

    # is_all_ids_valid = len(infos) == 0

    # if is_all_ids_valid:
    #     print("All given id columns are valid, does not contain null, all values are unquie in it")
    # else:
    #     print("Below id columsn has the errors")
    #     for info in infos:
    #         print(info)

    for info in infos:
        print(info)


if __name__ == "__main__":
    app()


