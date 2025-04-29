# ------------------- scripts/process.py -------------------
from data_download.config.settings import RAW_DATA_DIR, INTERIM_DATA_DIR
from data_download.utils import clean_text
import pandas as pd
from pathlib import Path
from loguru import logger

def process_omdena_dataset(input_path: Path = RAW_DATA_DIR / "omdena/fake_news_latam_omdena_combined.csv"):
    """
    Procesa el dataset de Omdena extrayendo solo las columnas relevantes y renombrando correctamente.
    """
    logger.info(f"Procesando dataset Omdena desde {input_path}")
    df = pd.read_csv(input_path)

    logger.info(f"Columnas detectadas en Omdena: {df.columns.tolist()}")

    # Eliminar columnas duplicadas
    df = df.loc[:, ~df.columns.duplicated()]

    # Buscar la columna más confiable para label
    label_col = None
    for col in ['corrected_label', 'prediction', 'label']:
        if col in df.columns:
            label_col = col
            break

    if not label_col:
        raise ValueError("No se encontró ninguna columna de etiquetas ('label', 'prediction', 'corrected_label')")

    # Construir dataframe con columnas renombradas
    renamed_df = pd.DataFrame()
    renamed_df['label'] = df[label_col].astype(str).str.lower()

    renamed_df['content'] = df['content'] if 'content' in df.columns else ''
    renamed_df['title'] = df['title'] if 'title' in df.columns else renamed_df['content'].apply(lambda x: ' '.join(str(x).split()[:8]) + '...')
    renamed_df['source'] = df['source'] if 'source' in df.columns else 'Sin Source'
    renamed_df['dataset_source'] = 'omdena'

    # Limpieza
    renamed_df['content'] = renamed_df['content'].apply(clean_text)
    renamed_df['title'] = renamed_df['title'].apply(clean_text)

    logger.success(f"Dataset Omdena procesado: {len(renamed_df)} filas")
    return renamed_df

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

    # Asegura que ambos DataFrames tengan las mismas columnas
    expected_cols = ['label', 'content', 'title', 'source', 'dataset_source']

    posadas = posadas[expected_cols]
    omdena = omdena[expected_cols]

    combined = pd.concat([posadas, omdena], ignore_index=True)

    combined['id'] = [f"fn_{i:06d}" for i in range(len(combined))]

    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = INTERIM_DATA_DIR / "combined_fakenews_dataset.csv"
    combined.to_csv(path, index=False)

    logger.success(f"Dataset combinado guardado en {path}")
    return combined