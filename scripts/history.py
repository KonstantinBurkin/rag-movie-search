"""Persist the N most recent query/response pairs as JSON."""

import json
from datetime import datetime, timezone
from typing import Any

from config import QUERY_HISTORY_FILE, QUERY_HISTORY_MAX_ENTRIES


def load_history() -> list[dict[str, Any]]:
    if not QUERY_HISTORY_FILE.exists():
        return []
    return json.loads(QUERY_HISTORY_FILE.read_text())


def append_history(query: str, response: Any) -> None:
    QUERY_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    history = load_history()
    history.append(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "response": response,
        }
    )
    history = history[-QUERY_HISTORY_MAX_ENTRIES:]
    QUERY_HISTORY_FILE.write_text(json.dumps(history, indent=2))
