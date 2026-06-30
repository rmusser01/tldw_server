"""Unpaywall client for OA PDF resolution by DOI.

Requires contact email via `UNPAYWALL_EMAIL`.
Returns (pdf_url, error_message).
"""
from __future__ import annotations

import os

from tldw_Server_API.app.core.http_client import fetch

BASE_URL = "https://api.unpaywall.org/v2"


def resolve_oa_pdf(doi: str) -> tuple[str | None, str | None]:
    """Resolve an OA PDF URL for the given DOI using Unpaywall.

    Returns (pdf_url, error_message). Missing email -> error; 404 -> (None, None).
    """
    email = os.getenv("UNPAYWALL_EMAIL")
    if not email:
        return None, "Unpaywall contact email not configured. Set UNPAYWALL_EMAIL."
    try:
        doi_clean = doi.strip()
        url = f"{BASE_URL}/{doi_clean}"
        r = fetch(method="GET", url=url, params={"email": email}, timeout=20)
        if r.status_code == 404:
            return None, None
        if r.status_code >= 400:
            return None, f"Unpaywall HTTP error: {r.status_code}"
        data = r.json() or {}
        # Prefer best_oa_location.url_for_pdf, then scan oa_locations
        best = data.get("best_oa_location") or {}
        pdf = best.get("url_for_pdf") or best.get("url")
        if not pdf:
            for loc in data.get("oa_locations") or []:
                pdf = loc.get("url_for_pdf") or loc.get("url")
                if pdf:
                    break
        return (pdf if pdf else None), None
    except TimeoutError:
        return None, "Unpaywall request timed out."
    except Exception:
        return None, "Unpaywall request failed."
