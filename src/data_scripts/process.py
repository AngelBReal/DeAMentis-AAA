# ------------------- scripts/process.py -------------------
from config.settings import RAW_DATA_DIR, INTERIM_DATA_DIR
from .utils import clean_text
import pandas as pd
from pathlib import Path
from loguru import logger

def process_omdena_dataset(input_path: Path = RAW_DATA_DIR / "omdena/fake_news_latam_omdena_combined.csv"):
    df = pd.read_csv(input_path)
    df = df.rename(columns={
        'corrected_label': 'label',
        'prediction': 'label',
        'content': 'content',
        'title': 'title',
        'source': 'source'
    })
    for col in ['content', 'title']: df[col] = df[col].apply(clean_text)
    df['dataset_source'] = 'omdena'
    return df

def process_posadas_dataset(input_path: Path = RAW_DATA_DIR / "FakeNewsCorpusSpanish/fake_news_corpus_posadas_full.csv"):
    df = pd.read_csv(input_path)
    df = df.rename(columns={
        'category': 'label',
        'headline': 'title',
        'text': 'content',
        'source': 'source'
    })
    for col in ['content', 'title']: df[col] = df[col].apply(clean_text)
    df['label'] = df['label'].replace({'false': 'fake'}).str.lower()
    df['dataset_source'] = 'posadas'
    return df

def merge_datasets():
    posadas = process_posadas_dataset()
    omdena = process_omdena_dataset()
    combined = pd.concat([posadas, omdena], ignore_index=True)
    combined['id'] = [f"fn_{i:06d}" for i in range(len(combined))]
    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = INTERIM_DATA_DIR / "combined_fakenews_dataset.csv"
    combined.to_csv(path, index=False)
    logger.success(f"Dataset combinado guardado en {path}")