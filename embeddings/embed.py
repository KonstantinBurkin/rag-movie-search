"""Build sentence embeddings for movies and store them in a Chroma collection."""

import polars as pl
import chromadb
from sentence_transformers import SentenceTransformer

from config import (
    CHROMA_DB_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    PROCESSED_DATA_DIR,
)


def get_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    return client.get_or_create_collection(name=COLLECTION_NAME)


def embed_movies(df: pl.DataFrame, model: SentenceTransformer, collection) -> None:
    documents = df["plot"].to_list()
    ids = [str(i) for i in range(df.height)]
    metadatas = df.drop("plot").to_dicts()

    embeddings = model.encode(documents, show_progress_bar=True).tolist()

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )


if __name__ == "__main__":
    df = pl.read_csv(PROCESSED_DATA_DIR / "movies_clean.csv")
    model = get_model()
    collection = get_collection()
    embed_movies(df, model, collection)
    print(f"Embedded {len(df)} movies into collection '{COLLECTION_NAME}'")
