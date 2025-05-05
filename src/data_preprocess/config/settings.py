from pathlib import Path

# Forza la raíz del proyecto con un path absoluto
BASE_DIR = Path(__file__).resolve().parents[3]
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
INTERIM_DATA_DIR = BASE_DIR / "data" / "interim"
PROCESSED_DATA_DIR = BASE_DIR/ "data" / "processed"


print(RAW_DATA_DIR)