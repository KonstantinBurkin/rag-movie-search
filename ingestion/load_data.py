"""Load raw movie data (CSV) and clean it into a processed DataFrame."""

import pandas as pd

from config import PROCESSED_DATA_DIR, RAW_DATA_DIR


def load_raw_movies(filename: str = "movies.csv") -> pd.DataFrame:
    path = RAW_DATA_DIR / filename
    return pd.read_csv(path)


def clean_movies(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["title", "overview"]).drop_duplicates(subset=["title"])
    df["overview"] = df["overview"].str.strip()
    return df.reset_index(drop=True)


def save_processed(df: pd.DataFrame, filename: str = "movies_clean.csv") -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_DIR / filename, index=False)


if __name__ == "__main__":
    raw_df = load_raw_movies()
    clean_df = clean_movies(raw_df)
    save_processed(clean_df)
    print(f"Processed {len(clean_df)} movies -> {PROCESSED_DATA_DIR / 'movies_clean.csv'}")
