import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    from pathlib import Path
    from dotenv import load_dotenv
    from loguru import logger
    from tqdm import tqdm

    load_dotenv()

    PROJ_ROOT = Path(__file__).resolve().parents[1]
    logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

    DATA_DIR = PROJ_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    INTERIM_DATA_DIR = DATA_DIR / "interim"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    EXTERNAL_DATA_DIR = DATA_DIR / "external"

    GDRIVE_LINK = "https://drive.google.com/drive/folders/1c4fPpkaZNIx73CxXMcjRX81Z4G4WUiOn?usp=sharing"

    MODELS_DIR = PROJ_ROOT / "models"
    REPORTS_DIR = PROJ_ROOT / "reports"
    FIGURES_DIR = REPORTS_DIR / "figures"

    try:
        logger.remove()
        logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
    except ModuleNotFoundError:
        pass


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Configuration Notebook

    This notebook displays project configuration and paths.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(f"""
    ## Project Configuration

    **Project Root:** `{PROJ_ROOT}`

    **GDRIVE Dataset:** `{GDRIVE_LINK}`

    **Raw Data Directory:** `{RAW_DATA_DIR}`

    **Processed Data Directory:** `{PROCESSED_DATA_DIR}`

    **Figures Directory:** `{FIGURES_DIR}`
    """)
    return


if __name__ == "__main__":
    app.run()
