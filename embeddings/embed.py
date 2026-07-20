"""Build sentence embeddings for movies and store them in a Chroma collection."""

import chromadb
import pandas as pd
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


def embed_movies(df: pd.DataFrame, model: SentenceTransformer, collection) -> None:
    documents = df["overview"].tolist()
    ids = df.index.astype(str).tolist()
    metadatas = df.drop(columns=["overview"]).to_dict(orient="records")

    embeddings = model.encode(documents, show_progress_bar=True).tolist()

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )


if __name__ == "__main__":
    df = pd.read_csv(PROCESSED_DATA_DIR / "movies_clean.csv")
    model = get_model()
    collection = get_collection()
    embed_movies(df, model, collection)
    print(f"Embedded {len(df)} movies into collection '{COLLECTION_NAME}'")
