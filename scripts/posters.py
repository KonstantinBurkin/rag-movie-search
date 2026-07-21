"""Best-effort movie poster lookup via the Wikipedia REST API (no API key needed)."""

from urllib.parse import quote

import requests

WIKIPEDIA_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"

# Wikipedia's robot policy rejects requests without an identifying User-Agent:
# https://meta.wikimedia.org/wiki/User-Agent_policy
REQUEST_HEADERS = {
    "User-Agent": "rag-movie-search/0.1 (https://github.com/KonstantinBurkin/rag-movie-search)"
}


def fetch_poster(title: str) -> str | None:
    """Return a thumbnail image URL for a movie title, or None if unavailable."""
    url = WIKIPEDIA_SUMMARY_URL.format(title=quote(title))
    try:
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=3)
        response.raise_for_status()
    except requests.RequestException:
        return None

    thumbnail = response.json().get("thumbnail")
    return thumbnail["source"] if thumbnail else None
