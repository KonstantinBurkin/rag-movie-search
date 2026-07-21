"""Rerank a shortlist of candidate movies with Gemini, given a user query."""

from typing import Any

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from config import RERANK_MODEL_NAME, RERANK_TOP_K

SYSTEM_PROMPT = """You are an expert movie curator.
Given a user's search query and a list of candidate movies,
select and rank the movies that best match the query.

For each selected movie, write a one-line justification that
references a specific detail from its plot — not a generic
restatement of the query. Rank strictly by relevance to the query,
best match first."""

RANKING_PROMPT_TEMPLATE = """User query: {query}

Candidate movies:
{candidates_block}

Select and rank the top {top_k} movies from this list that best
match the user query."""


class RankedMovie(BaseModel):
    rank: int = Field(..., ge=1, description="1 = best match")
    title: str
    justification: str = Field(..., description="One sentence, referencing the plot")
    release_year: int | None = None


class RankingResult(BaseModel):
    rankings: list[RankedMovie]


def _format_candidate(index: int, movie: dict[str, Any]) -> str:
    metadata = movie.get("metadata", {})
    meta_bits = ", ".join(
        f"{key}: {value}"
        for key, value in (
            ("genre", metadata.get("genre")),
            ("release_year", metadata.get("release_year")),
            ("director", metadata.get("director")),
            ("origin", metadata.get("origin")),
        )
        if value
    )
    return f"{index}. {movie['title']} ({meta_bits})\n   Plot: {movie['plot']}"


def build_prompt(
    query: str, candidates: list[dict[str, Any]], top_k: int = RERANK_TOP_K
) -> str:
    candidates_block = "\n".join(
        _format_candidate(i, movie) for i, movie in enumerate(candidates, start=1)
    )
    return RANKING_PROMPT_TEMPLATE.format(
        query=query, candidates_block=candidates_block, top_k=top_k
    )


def rank_movies(
    query: str,
    candidates: list[dict[str, Any]],
    top_k: int = RERANK_TOP_K,
    client: genai.Client | None = None,
) -> RankingResult:
    client = client or genai.Client()

    response = client.models.generate_content(
        model=RERANK_MODEL_NAME,
        contents=build_prompt(query, candidates, top_k=top_k),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=RankingResult,
        ),
    )

    if response.parsed is None:
        raise RuntimeError(
            "Gemini did not return a valid ranking (blocked or malformed response)"
        )

    return response.parsed
