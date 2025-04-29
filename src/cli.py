# ------------------- scripts/cli.py -------------------
# src/data_scripts/cli.py
import sys
from pathlib import Path

# Agrega el directorio src/ al sys.path para permitir importaciones relativas
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data_download import download, process
import typer


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
