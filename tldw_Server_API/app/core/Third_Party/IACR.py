"""IACR conference metadata adapter.

Fetches metadata for a given venue and year.
Example: https://www.iacr.org/cryptodb/data/api/conf.php?year=2017&venue=crypto
"""
from __future__ import annotations

from typing import Any

from tldw_Server_API.app.core.http_client import fetch, fetch_json

BASE_URL = "https://www.iacr.org/cryptodb/data/api/conf.php"


def fetch_conference(venue: str, year: int) -> tuple[dict[str, Any] | None, str | None]:
    """Returns parsed JSON for the requested conference."""
    try:
        params = {"venue": venue, "year": str(year)}
        data = fetch_json(method="GET", url=BASE_URL, params=params, timeout=20)
        return data, None
    except TimeoutError:
        return None, "IACR request timed out."
    except Exception:
        return None, "IACR request failed."


def fetch_conference_raw(venue: str, year: int) -> tuple[bytes | None, str | None, str | None]:
    """Returns raw bytes and media type for the requested conference."""
    try:
        params = {"venue": venue, "year": str(year)}
        r = fetch(method="GET", url=BASE_URL, params=params, timeout=20)
        if r.status_code >= 400:
            return None, None, f"IACR HTTP error: {r.status_code}"
        ct = r.headers.get("content-type") or "application/json"
        return r.content, ct.split(";")[0], None
    except TimeoutError:
        return None, None, "IACR request timed out."
    except Exception:
        return None, None, "IACR request failed."
