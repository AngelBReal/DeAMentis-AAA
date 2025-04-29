from pathlib import Path
import os, re, requests, time
import pandas as pd
from tqdm.auto import tqdm
from loguru import logger

def ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def download_file(url: str, output_path: Path, timeout=30, retries=3, backoff_factor=0.5) -> bool:
    ensure_dir(output_path)
    if output_path.exists():
        logger.info(f"Ya existe: {output_path}")
        return True
    for attempt in range(retries):
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                with open(output_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            return True
        except Exception as e:
            logger.warning(f"Error {attempt+1}/{retries}: {e}")
            time.sleep(backoff_factor * (2 ** attempt))
    return False

def clean_text(text):
    if pd.isna(text): return ""
    text = str(text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text