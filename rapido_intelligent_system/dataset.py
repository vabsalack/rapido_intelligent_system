from pathlib import Path

import gdown

# from loguru import logger
# from tqdm import tqdm
# import typer

# from rapido_intelligent_system.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from rapido_intelligent_system.config import RAW_DATA_DIR, GDRIVE_LINK


# app = typer.Typer()


# @app.command()
# def main(
#     # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
#     input_path: Path = RAW_DATA_DIR / "dataset.csv",
#     output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
#     # ----------------------------------------------
# ):
#     # ---- REPLACE THIS WITH YOUR OWN CODE ----
#     logger.info("Processing dataset...")
#     for i in tqdm(range(10), total=10):
#         if i == 5:
#             logger.info("Something happened for iteration 5.")
#     logger.success("Processing dataset complete.")
#     # -----------------------------------------



def download_gdrive_folder(folder_url: str, output_dir=None) -> Path:
    """
    Downloads all files from a public Google Drive folder link into output_dir.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        gdown.download_folder(
            url=folder_url,
            output=str(output_dir),
            quiet=False,
            use_cookies=False,
        )
        print(f"Download complete: {output_dir}")
    except Exception as e:
        print(f"An error occurred: {e}")
        raise

    return output_dir

def classify_columns(df, identifiers=None, numerical=None, str=None):
    """
    input:
        df
    function:
        groups columns names into identifiers, numerical and text lists.
        also check the match of no of columns input and no of columns output
    return
        id_col_names, num_col_names, text_col_names, bool
    interpret:
        classify attrbs to id, num and str by their names only
    """
    # import required libs
    import pandas as pd

    if identifiers is None:
        identifiers = []
    if numerical is None:
        numerical = []
    if str is None:
        str = []

    og_cols_count = len(df.columns)

    for col in df.columns:
        if col.lower().endswith("_id") or col.lower() == "id":
            if col not in identifiers:
                identifiers.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            if col not in numerical:
                numerical.append(col)
        else:
            if col not in str:
                str.append(col)
    md_cols_count = sum([len(item) for item in [identifiers, numerical, str]])
    is_count_match = og_cols_count == md_cols_count

    return identifiers, numerical, str, is_count_match


def initial_inspect_str_attr(df: pd.DataFrame, attr_name: str, value_count=True, diagram=True):
    """
    inputs: 
        df and str attr name
    function: 
        perform value counts and horizontal count plot for given str/categorical attribute
    returns:
        value_count series and count plot
    interpret:
        In value count series - the rows cat are no of unique values
        In count plot - it is for quick visual sense
    """
    # import related libs
    import pandas as pd
    import seaborn as sns

    freq_count = df[attr_name].value_counts(dropna=False, sort=True)

    missing_label = "<missing>"
    plot_attr = df[attr_name].where(df[attr_name].notna(), missing_label)
    order = plot_attr.value_counts(sort=True).index
    plot = sns.countplot(y=plot_attr, order=order)
    
    plot.grid(True, axis="x", linestyle="--", alpha=0.7)
    plot.set_ylabel(attr_name)

    if value_count and diagram:
        return freq_count, plot

    if value_count:
        return freq_count

    if diagram:
        return plot
    


if __name__ == "__main__":
    download_gdrive_folder(folder_url=GDRIVE_LINK, output_dir=RAW_DATA_DIR) # download the og dataset to raw data dir
    # app()


