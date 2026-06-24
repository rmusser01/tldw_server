from __future__ import annotations

from typing import Any

from tldw_Server_API.app.core.exceptions import JSONDecodeError, ThirdPartyHTTPStatusError
from tldw_Server_API.app.core.http_client import fetch


def fetch_json_checked(
    *,
    method: str,
    url: str,
    require_json_ct: bool = True,
    max_bytes: int | None = None,
    **kwargs: Any,
) -> Any:
    response = fetch(method=method, url=url, **kwargs)
    try:
        if response.status_code >= 400:
            raise ThirdPartyHTTPStatusError(
                response.status_code,
                getattr(response, "reason_phrase", "") or None,
            )
        content_type = (response.headers.get("content-type") or "").lower()
        if require_json_ct and "application/json" not in content_type:
            raise JSONDecodeError("Response is not application/json")
        if max_bytes is not None and len(response.content) > max_bytes:
            raise JSONDecodeError("JSON response exceeds configured byte limit")
        return response.json()
    finally:
        response.close()
