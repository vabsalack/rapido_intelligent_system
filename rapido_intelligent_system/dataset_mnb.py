import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    from pathlib import Path
    import gdown
    import typer
    from rich import print
    from loguru import logger
    from tqdm import tqdm
    from rapido_intelligent_system.config_mnb import (
        PROCESSED_DATA_DIR, RAW_DATA_DIR, GDRIVE_LINK
    )


@app.cell
def _():
    is_script_mode = mo.app_meta().mode == "script"
    return (is_script_mode,)


@app.cell
def _(is_script_mode):
    typer_app = typer.Typer()

    @typer_app.command()
    def main(
        input_path: Path = RAW_DATA_DIR / "dataset.csv",
        output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    ):
        logger.info("Processing dataset...")
        for i in tqdm(range(10), total=10):
            if i == 5:
                logger.info("Something happened for iteration 5.")
        logger.success("Processing dataset complete.")

    @typer_app.command("download")
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

    if is_script_mode:
        typer_app()
    typer_app
    return


if __name__ == "__main__":
    app.run()
