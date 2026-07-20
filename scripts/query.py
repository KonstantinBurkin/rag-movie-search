"""Query the movie collection with a prompt."""

import argparse

from embeddings.embed import get_collection, get_model


def search(query: str, n_results: int = 5):
    model = get_model()
    collection = get_collection()

    query_embedding = model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=n_results)

    return results


def print_results(results) -> None:
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for rank, (doc, meta, dist) in enumerate(
        zip(documents, metadatas, distances), start=1
    ):
        title = meta.get("title", "Unknown")
        print(f"{rank}. {title} (distance: {dist:.4f})")
        print(f"   {doc}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search movies by query.")
    parser.add_argument("query", type=str, help="search query")
    parser.add_argument(
        "-n", "--n-results", type=int, default=5, help="Number of results to return"
    )
    args = parser.parse_args()

    results = search(args.query, args.n_results)
    print_results(results)
