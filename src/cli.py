# ------------------- scripts/cli.py -------------------
# src/data_scripts/cli.py
import sys
from pathlib import Path

# Agrega el directorio src/ al sys.path para permitir importaciones relativas
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data_download import download, process
from src.data_preprocess import preprocessing
from src.models.final_models import build_final_model
import typer


app = typer.Typer()

@app.command()
def download_all():
    download.download_omdena()
    download.download_posadas()

@app.command()
def process_all():
    process.merge_datasets()

@app.command()
def preprocess_all():
    preprocessing.main()

@app.command()
def build_model():
    """Entrena el modelo Voting final y guarda artefactos"""
    build_final_model.main()


if __name__ == "__main__":
    app()
