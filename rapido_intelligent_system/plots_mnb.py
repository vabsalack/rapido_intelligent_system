import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    is_script_mode = mo.app_meta().mode == "script"
    return (is_script_mode,)


@app.cell
def _(
    FIGURES_DIR,
    PROCESSED_DATA_DIR,
    Path,
    is_script_mode,
    logger,
    tqdm,
    typer,
):
    typer_app = typer.Typer()

    @typer_app.command()
    def main(
        input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
        output_path: Path = FIGURES_DIR / "plot.png",
    ):
        logger.info("Generating plot from data...")
        for i in tqdm(range(10), total=10):
            if i == 5:
                logger.info("Something happened for iteration 5.")
        logger.success("Plot generation complete.")

    if is_script_mode:
        typer_app()
    return


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    import typer
    from loguru import logger
    from tqdm import tqdm
    from rapido_intelligent_system.config_mnb import FIGURES_DIR, PROCESSED_DATA_DIR

    return FIGURES_DIR, PROCESSED_DATA_DIR, Path, logger, mo, tqdm, typer


if __name__ == "__main__":
    app.run()
