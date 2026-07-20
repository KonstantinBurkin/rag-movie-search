"""Build sentence embeddings for movies and store them in a Chroma collection."""

import chromadb
import polars as pl
from sentence_transformers import SentenceTransformer

from config import (
    CHROMA_DB_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    PROCESSED_DATA_DIR,
)
from embeddings.schema import build_record


def get_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def get_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(CHROMA_DB_DIR))


def get_collection(client: chromadb.ClientAPI | None = None):
    return client.get_or_create_collection(name=COLLECTION_NAME)


def embed_movies(
    df: pl.DataFrame, model: SentenceTransformer, collection, max_batch_size: int
) -> None:
    records = [build_record(row) for row in df.to_dicts()]
    documents = [document for document, _ in records]
    metadatas = [metadata for _, metadata in records]
    ids = [str(i) for i in range(len(records))]

    embeddings = model.encode(documents, show_progress_bar=True).tolist()

    for start in range(0, len(ids), max_batch_size):
        end = start + max_batch_size
        collection.add(
            ids=ids[start:end],
            embeddings=embeddings[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )


if __name__ == "__main__":
    df = pl.read_csv(PROCESSED_DATA_DIR / "movies_clean.csv")
    model = get_model()
    client = get_client()
    collection = get_collection(client)
    embed_movies(df, model, collection, client.get_max_batch_size())
    print(f"Embedded {len(df)} movies into collection '{COLLECTION_NAME}'")
