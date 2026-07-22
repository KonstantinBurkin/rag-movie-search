"""Load raw movie data (CSV) and clean it into a processed DataFrame."""

import polars as pl

from config import PROCESSED_DATA_DIR, RAW_DATA_DIR


def load_raw_movies(filename: str = "summaries.csv") -> pl.DataFrame:
    path = RAW_DATA_DIR / filename
    return pl.read_csv(path)


def clean_movies(df: pl.DataFrame) -> pl.DataFrame:
    df = df.drop_nulls(subset=["title", "plot"]).unique(
        subset=["title"],
        keep="first",
        maintain_order=True,
    )
    df = df.with_columns(pl.col("plot").str.strip_chars())
    df = df.filter(
        c("release_year") > 1920,
        c("origin")
        .is_in(
            (
                "Bollywood",
                "Tamil",
                "Malayalam",
                "Telugu",
                "Bangladeshi",
                "Punjabi",
                "Assamese",
                "Marathi",
                "Malaysian",
                "Maldivian",
                "Bengali",
                "Kannada",
                "Egyptian",
            )
        )
        .not_(),
    )
    return df


def save_processed(df: pl.DataFrame, filename: str = "movies_clean.csv") -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.write_csv(PROCESSED_DATA_DIR / filename)


if __name__ == "__main__":
    raw_df = load_raw_movies()
    clean_df = clean_movies(raw_df)
    save_processed(clean_df)
    print(
        f"Processed {len(clean_df)} movies -> {PROCESSED_DATA_DIR / 'movies_clean.csv'}"
    )
