# ------------------- scripts/cli.py -------------------
import typer
from src.data_scripts import download, process

app = typer.Typer()

@app.command()
def download_all():
    download.download_omdena()
    download.download_posadas()

@app.command()
def process_all():
    process.merge_datasets()

if __name__ == "__main__":
    app()
