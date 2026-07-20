"""Download the movie-summaries dataset from Hugging Face"""

from datasets import load_dataset

from config import RAW_DATA_DIR, HF_DATASET_SUMMARIES_ID

SUMMARIES_SUMMARIES_COLUMN_MAP = {
    "Title": "title",
    "Plot": "plot",
    "Release Year": "release_year",
    "Genre": "genre",
    "Director": "director",
    "Origin/Ethnicity": "origin",
    # "Cast": "cast",
    # "PlotSummary": "overview",
    # "Wiki Page": "wiki_page",
}


def download(filename: str = "summaries.csv") -> None:
    dataset = load_dataset(HF_DATASET_SUMMARIES_ID, split="train")
    df = dataset.to_pandas().rename(columns=SUMMARIES_COLUMN_MAP)
    df = df[list(SUMMARIES_COLUMN_MAP.values())]

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DATA_DIR / filename
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} summaries -> {out_path}")


if __name__ == "__main__":
    download()
