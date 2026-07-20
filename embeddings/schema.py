"""Document schema: movie -> embedding + metadata."""

from typing import Any


def build_document(title: str, plot: str) -> str:
    """Text that gets embedded."""
    return f"{title}\n\n{plot}"


def build_metadata(row: dict[str, Any]) -> dict[str, Any]:
    """Chroma metadata values must be str/int/float/bool - not None"""
    return {
        "title": row["title"] or "",
        "genre": row["genre"] or "unknown",
        "release_year": row["release_year"] if row["release_year"] is not None else 0,
        "director": row["director"] or "unknown",
        "origin": row["origin"] or "unknown",
    }


def build_record(row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    document = build_document(row["title"], row["plot"])
    metadata = build_metadata(row)
    return document, metadata
