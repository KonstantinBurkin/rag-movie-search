"""Query the movie collection with a prompt."""

import argparse
from typing import Any

from config import RERANK_CANDIDATE_POOL_SIZE, RERANK_TOP_K
from embeddings.embed import get_collection, get_model
from reranking.rerank import RankingResult, rank_movies
from scripts.history import append_history


def search(
    query: str,
    n_results: int = RERANK_CANDIDATE_POOL_SIZE,
    model=None,
    collection=None,
):
    model = model or get_model()
    collection = collection or get_collection()

    query_embedding = [vector.tolist() for vector in model.embed([query])]
    results = collection.query(query_embeddings=query_embedding, n_results=n_results)

    return results


def _to_candidate(document: str, metadata: dict[str, Any]) -> dict[str, Any]:
    title = metadata.get("title", "")
    plot = document.removeprefix(f"{title}\n\n")
    return {
        "title": title,
        "plot": plot,
        "metadata": {
            "genre": metadata.get("genre"),
            "release_year": metadata.get("release_year"),
            "director": metadata.get("director"),
            "origin": metadata.get("origin"),
        },
    }


def search_and_rerank(
    query: str,
    candidate_pool_size: int = RERANK_CANDIDATE_POOL_SIZE,
    top_k: int = RERANK_TOP_K,
    model=None,
    collection=None,
) -> RankingResult:
    """Vector search for candidates, then rerank +
    justify the top matches with Gemini."""
    results = search(query, n_results=candidate_pool_size, model=model, collection=collection)
    candidates = [
        _to_candidate(document, metadata)
        for document, metadata in zip(results["documents"][0], results["metadatas"][0])
    ]
    return rank_movies(query, candidates, top_k=top_k)


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


def print_ranked_results(ranking: RankingResult) -> None:
    for movie in ranking.rankings:
        print(f"{movie.rank}. {movie.title}")
        print(f"   {movie.justification}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Search movies by query, reranked and justified by Claude."
    )
    parser.add_argument("query", type=str, help="search query")
    parser.add_argument(
        "-n",
        "--top-k",
        type=int,
        default=RERANK_TOP_K,
        help="Number of ranked results to return",
    )
    parser.add_argument(
        "-c",
        "--candidates",
        type=int,
        default=RERANK_CANDIDATE_POOL_SIZE,
        help="Number of vector-search candidates to pass to the reranker",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Skip Claude reranking and show raw vector-search results",
    )
    args = parser.parse_args()

    if args.no_rerank:
        print_results(search(args.query, n_results=args.top_k))
    else:
        ranking = search_and_rerank(
            args.query, candidate_pool_size=args.candidates, top_k=args.top_k
        )
        print_ranked_results(ranking)
        append_history(args.query, ranking.model_dump())
