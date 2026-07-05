"""Fetch runtime adapter for Web_Scraping."""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any, Protocol

from tldw_Server_API.app.core.http_client import fetch as http_fetch

from .requests import FetchRequest
from .responses import FetchResponse


class FetchClient(Protocol):
    """Synchronous fetch client used by runtime-aware scrape code."""

    def fetch(self, request: FetchRequest) -> FetchResponse:
        """Fetch a URL and return a normalized response."""


def _mutable_mapping_or_none(value: Mapping[str, Any]) -> dict[str, Any] | None:
    if not value:
        return None
    return {str(key): item for key, item in value.items()}


def _mutable_proxies(value: Mapping[str, str] | str | None) -> dict[str, str] | str | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {str(key): str(item) for key, item in value.items()}
    return str(value)


class DefaultFetchClient:
    """Default Web_Scraping fetch adapter over the central HTTP helper."""

    def fetch(self, request: FetchRequest) -> FetchResponse:
        if request.method != "GET":
            raise ValueError("DefaultFetchClient only supports GET requests in Phase 2")

        started = time.monotonic()
        if request.backend == "curl":
            raw = http_fetch(
                request.url,
                headers=_mutable_mapping_or_none(request.headers),
                cookies=_mutable_mapping_or_none(request.cookies),
                timeout=request.timeout,
                backend=request.backend,
                follow_redirects=request.allow_redirects,
                impersonate=request.impersonate,
                proxies=_mutable_proxies(request.proxies),
            )
            fallback_backend = "curl"
        else:
            raw = http_fetch(
                method=request.method,
                url=request.url,
                headers=_mutable_mapping_or_none(request.headers),
                cookies=_mutable_mapping_or_none(request.cookies),
                timeout=request.timeout,
                allow_redirects=request.allow_redirects,
                proxies=_mutable_proxies(request.proxies),
            )
            fallback_backend = "httpx"
        elapsed = max(0.0, time.monotonic() - started)
        return FetchResponse.from_raw(
            raw,
            fallback_url=request.url,
            fallback_backend=fallback_backend,
            elapsed_seconds=elapsed,
        )
