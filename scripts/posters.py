"""Movie poster lookup via The Movie Database (TMDB) search API."""

import os

import requests

TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w342"


def fetch_poster(title: str, year: int | None = None) -> str | None:
    """Return a poster image URL for a movie title, or None if unavailable."""
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
    poster_path = results[0].get("poster_path") if results else None
    return f"{TMDB_IMAGE_BASE_URL}{poster_path}" if poster_path else None
