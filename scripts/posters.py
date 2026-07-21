"""Best-effort movie poster lookup via the Wikipedia REST API (no API key needed)."""

from urllib.parse import quote

import requests

WIKIPEDIA_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"


def fetch_poster(title: str) -> str | None:
    """Return a thumbnail image URL for a movie title, or None if unavailable."""
    url = WIKIPEDIA_SUMMARY_URL.format(title=quote(title))
    try:
        response = requests.get(url, timeout=3)
        response.raise_for_status()
    except requests.RequestException:
        return None

    thumbnail = response.json().get("thumbnail")
    return thumbnail["source"] if thumbnail else None
