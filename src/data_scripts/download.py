# ------------------- scripts/download.py -------------------
from pathlib import Path
from datasets import load_dataset
import pandas as pd
import requests, re, time
from tqdm.auto import tqdm
from loguru import logger
from .utils import ensure_dir, download_file, clean_text
from config.settings import RAW_DATA_DIR

def download_omdena(output_path: Path = RAW_DATA_DIR / "omdena/fake_news_latam_omdena_combined.csv"):
    logger.info("Descargando dataset Omdena desde Hugging Face...")
    ensure_dir(output_path)
    dataset = load_dataset("IsaacRodgz/Fake-news-latam-omdena")
    df = pd.concat([dataset['train'].to_pandas(), dataset['test'].to_pandas()])
    df.columns = df.columns.str.strip().str.lower()
    df['split'] = ['train'] * len(dataset['train']) + ['test'] * len(dataset['test'])
    df.to_csv(output_path, index=False)
    logger.success(f"Dataset Omdena guardado en {output_path}")

def download_posadas(output_dir: Path = RAW_DATA_DIR / "FakeNewsCorpusSpanish"):
    urls = {
        "train": "https://github.com/jpposadas/FakeNewsCorpusSpanish/raw/master/train.xlsx",
        "test": "https://github.com/jpposadas/FakeNewsCorpusSpanish/raw/master/test.xlsx",
        "dev": "https://github.com/jpposadas/FakeNewsCorpusSpanish/raw/master/development.xlsx"
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    dfs = []
    for name, url in urls.items():
        path = output_dir / f"{name}.xlsx"
        if download_file(url, path):
            df = pd.read_excel(path)
            df.columns = df.columns.str.strip().str.lower()
            df['split'] = name
            dfs.append(df)
    if dfs:
        full_df = pd.concat(dfs, ignore_index=True)
        full_df.to_csv(output_dir / "fake_news_corpus_posadas_full.csv", index=False)
        logger.success(f"Dataset Posadas guardado en {output_dir}")
