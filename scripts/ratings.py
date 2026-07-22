"""Movie rating lookup via The Movie Database (TMDB) search API."""

import os

import requests

from config import TMDB_SEARCH_URL


def fetch_rating(title: str, year: int | None = None) -> float | None:
    """Return a TMDB vote average (0-10) for a movie title, or None if unavailable."""
    api_key = os.environ.get("TMDB_API_KEY")
    if not api_key:
        return None

    params = {"api_key": api_key, "query": title}
    if year:
        params["year"] = year

    try:
        response = requests.get(TMDB_SEARCH_URL, params=params, timeout=3)
        response.raise_for_status()
    except requests.RequestException:
        return None

    results = response.json().get("results") or []
    return results[0].get("vote_average") if results else None
