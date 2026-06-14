# from pathlib import Path

# from loguru import logger
# from tqdm import tqdm
# import typer

# from rapido_intelligent_system.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

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

# if __name__ == "__main__":
#     app()


from pathlib import Path
import gdown
from rapido_intelligent_system.config import RAW_DATA_DIR

GDRIVE_LINK = "https://drive.google.com/drive/u/0/folders/1c4fPpkaZNIx73CxXMcjRX81Z4G4WUiOn"


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

if __name__ == "__main__":
    download_gdrive_folder(folder_url=GDRIVE_LINK, output_dir=RAW_DATA_DIR)


