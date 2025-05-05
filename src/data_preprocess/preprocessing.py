import pandas as pd
import numpy as np
import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from src.data_preprocess.config import settings
from sklearn.model_selection import train_test_split

# ============================
# CONFIG PATHS
# ============================
BASE_DIR = settings.BASE_DIR
INTERIM_DATA_DIR = settings.INTERIM_DATA_DIR
PROCESSED_DATA_DIR = settings.PROCESSED_DATA_DIR

INPUT_FILE = INTERIM_DATA_DIR / "combined_fakenews_dataset.csv"
CLASSICAL_OUTPUT = PROCESSED_DATA_DIR / "classical_models.csv"
NN_OUTPUT = PROCESSED_DATA_DIR / "neural_networks.csv"
TRANSFORMERS_OUTPUT = PROCESSED_DATA_DIR / "transformers.csv"



# ============================
# GENERAL CLEANING FUNCTIONS
# ============================
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # eliminar URLs
    text = re.sub(r'<.*?>', '', text)  # eliminar etiquetas HTML
    text = re.sub(r'\d+', '[NUM]', text)  # reemplazar números
    text = re.sub(r'[^a-záéíóúñü\s]', '', text)  # quitar puntuación extraña
    text = re.sub(r'\s+', ' ', text).strip()  # limpiar espacios múltiples
    return text


# ============================
# FEATURE ENGINEERING
# ============================
def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    df['clean_title'] = df['title'].apply(clean_text)
    df['clean_content'] = df['content'].apply(clean_text)
    
    sensational_words = ['increíble', 'impactante', 'asombroso', 'urgente', 'viral',
                         'escándalo', 'secreto', 'exclusiva', 'prohibido', 'censurado']
    
    df['title_word_count'] = df['clean_title'].apply(lambda x: len(x.split()))
    df['content_word_count'] = df['clean_content'].apply(lambda x: len(x.split()))
    df['num_uppercase_words'] = df['content'].apply(lambda x: sum(1 for w in str(x).split() if w.isupper()))
    df['has_known_source'] = df['source'].apply(lambda x: 1 if pd.notnull(x) and str(x).strip() != '' else 0)
    df['exclam_density'] = df['content'].apply(lambda x: str(x).count('!') / (len(str(x)) + 1))
    df['title_length'] = df['title'].apply(lambda x: len(str(x)))
    df['content_length'] = df['content'].apply(lambda x: len(str(x)))

    # Otras features adicionales
    df['title_exclamation_count'] = df['clean_title'].apply(lambda x: x.count('!'))
    df['content_exclamation_count'] = df['clean_content'].apply(lambda x: x.count('!'))
    df['title_question_count'] = df['clean_title'].apply(lambda x: x.count('?'))
    df['content_question_count'] = df['clean_content'].apply(lambda x: x.count('?'))
    df['title_sensational_count'] = df['clean_title'].apply(lambda x: sum(1 for w in sensational_words if w in x))
    df['content_sensational_count'] = df['clean_content'].apply(lambda x: sum(1 for w in sensational_words if w in x))
    df['title_number_count'] = df['clean_title'].apply(lambda x: len(re.findall(r'\[NUM\]', x)))
    df['content_number_count'] = df['clean_content'].apply(lambda x: len(re.findall(r'\[NUM\]', x)))
    df['title_uppercase_ratio'] = df['title'].apply(lambda x: sum(1 for c in str(x) if c.isupper()) / len(str(x)) if len(str(x)) > 0 else 0)
    
    return df


# ============================
# PREPROCESSING PIPELINES
# ============================
def preprocess_classical(df: pd.DataFrame) -> None:
    df = add_text_features(df)
    df['combined_text'] = df['clean_title'] + ' ' + df['clean_content']

    # Save clean CSV (with all features, no split, no TF-IDF, no train/test separation)
    df.to_csv(PROCESSED_DATA_DIR / 'classical_models.csv', index=False)

    print(f"Clean classical dataset saved: {df.shape[0]} rows, {df.shape[1]} columns")

def preprocess_neural_networks(df: pd.DataFrame, max_length: int = 1000) -> pd.DataFrame:
    df['combined_text'] = df['title'].fillna('') + ' ' + df['content'].fillna('')
    df['combined_text'] = df['combined_text'].apply(lambda x: clean_text(x)[:max_length])
    return df[['label', 'combined_text']]

def preprocess_transformers(df: pd.DataFrame, max_length: int = 512) -> pd.DataFrame:
    df['combined_text'] = df['title'].fillna('') + ' ' + df['content'].fillna('')
    df['combined_text'] = df['combined_text'].apply(lambda x: str(x)[:max_length])
    return df[['label', 'combined_text']]

# ============================
# MAIN FUNCTION
# ============================
def main():
    df = pd.read_csv(INPUT_FILE)
    df = df.drop_duplicates(subset='id').dropna(subset=['label', 'content', 'title'])
    df['label'] = df['label'].apply(lambda x: 1 if x == 'fake' else 0)

    import os
    os.makedirs(os.path.dirname(CLASSICAL_OUTPUT), exist_ok=True)

    preprocess_classical(df)

    nn_df = preprocess_neural_networks(df)
    nn_df.to_csv(NN_OUTPUT, index=False)

    transformer_df = preprocess_transformers(df)
    transformer_df.to_csv(TRANSFORMERS_OUTPUT, index=False)

    print("Preprocessed datasets saved successfully.")

if __name__ == "__main__":
    
    main()
