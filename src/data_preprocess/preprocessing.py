import pandas as pd
import numpy as np
import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from src.data_preprocess.config import settings

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
    text = re.sub(r"http\\S+", "", text)  # eliminar URLs
    text = re.sub(r"\\*number\\*", "[NUM]", text)  # normalizar marcadores
    text = re.sub(r"[\"“”]", "", text)  # quitar comillas
    text = re.sub(r"[^a-záéíóúñü\\s]", "", text)  # quitar puntuación extraña
    text = re.sub(r"\\s+", " ", text).strip()  # limpiar espacios múltiples
    return text

# ============================
# FEATURE ENGINEERING
# ============================
def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    df['title_word_count'] = df['title'].apply(lambda x: len(str(x).split()))
    df['content_word_count'] = df['content'].apply(lambda x: len(str(x).split()))
    df['num_uppercase_words'] = df['content'].apply(lambda x: sum(1 for w in str(x).split() if w.isupper()))
    df['has_known_source'] = df['source'].apply(lambda x: 1 if pd.notnull(x) and str(x).strip() != '' else 0)
    df['fake_word_in_title'] = df['title'].apply(lambda x: 1 if 'fake' in str(x).lower() else 0)
    df['exclam_density'] = df['content'].apply(lambda x: str(x).count('!') / (len(str(x)) + 1))
    return df

# ============================
# PREPROCESSING PIPELINES
# ============================
def preprocess_classical(df: pd.DataFrame, tfidf_max_features: int = 500) -> pd.DataFrame:
    df = add_text_features(df)
    df['title_length'] = df['title'].apply(lambda x: len(str(x)))
    df['content_length'] = df['content'].apply(lambda x: len(str(x)))
    df['combined_text'] = df['title'].fillna('') + ' ' + df['content'].fillna('')
    tfidf = TfidfVectorizer(max_features=tfidf_max_features)
    tfidf_matrix = tfidf.fit_transform(df['combined_text'])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())

    # One-hot encode categorical fields
    df = pd.get_dummies(df, columns=['source', 'dataset_source'], drop_first=True)

    result_df = pd.concat([
        df[['label', 'title_length', 'content_length', 'title_word_count', 'content_word_count',
            'num_uppercase_words', 'has_known_source', 'fake_word_in_title', 'exclam_density']],
        df.filter(like='source_'),
        df.filter(like='dataset_source_'),
        tfidf_df
    ], axis=1)

    return result_df

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
    df['label'] = df['label'].apply(lambda x: 1 if x == 'FAKE' else 0)

    classical_df = preprocess_classical(df)
    classical_df.to_csv(CLASSICAL_OUTPUT, index=False)

    nn_df = preprocess_neural_networks(df)
    nn_df.to_csv(NN_OUTPUT, index=False)

    transformer_df = preprocess_transformers(df)
    transformer_df.to_csv(TRANSFORMERS_OUTPUT, index=False)

    print("Preprocessed datasets saved successfully.")

if __name__ == "__main__":
    main()
