"""Rerank a shortlist of candidate movies with Claude, given a user query."""

from typing import Any

import anthropic
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
    client: anthropic.Anthropic | None = None,
) -> RankingResult:
    client = client or anthropic.Anthropic()

    response = client.messages.parse(
        model=RERANK_MODEL_NAME,
        max_tokens=4096,
        thinking={"type": "adaptive"},
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": build_prompt(query, candidates, top_k=top_k)}
        ],
        output_format=RankingResult,
    )

    if response.stop_reason == "refusal":
        raise RuntimeError("Claude declined to rank this request")

    return response.parsed_output
