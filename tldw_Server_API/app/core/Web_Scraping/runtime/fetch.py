"""Fetch runtime adapter for Web_Scraping."""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any, Protocol

from tldw_Server_API.app.core.exceptions import BadRequestError
from tldw_Server_API.app.core.http_client import ResponseObserverAborted
from tldw_Server_API.app.core.http_client import fetch as http_fetch

from .controls import ParentControls
from .requests import FetchRequest
from .responses import FetchResponse


class FetchClient(Protocol):
    """Synchronous fetch client used by runtime-aware scrape code."""

    def fetch(self, request: FetchRequest) -> FetchResponse:
        """Fetch a URL and return a normalized response."""


def _mutable_mapping_or_none(value: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return a mutable copy for HTTP helpers that treat empty mappings as absent."""
    if not value:
        return None
    return {str(key): item for key, item in value.items()}


def _mutable_proxies(value: Mapping[str, str] | str | None) -> dict[str, str] | str | None:
    """Return proxies in the mutable shape expected by the central HTTP helper."""
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {str(key): str(item) for key, item in value.items()}
    return str(value)


class DefaultFetchClient:
    """Default Web_Scraping fetch adapter over the central HTTP helper."""

    def __init__(self, parent: ParentControls | None = None) -> None:
        self._parent = parent

    def fetch(self, request: FetchRequest) -> FetchResponse:
        """Fetch a normalized response for supported Web_Scraping requests."""
        if request.method != "GET":
            raise BadRequestError("DefaultFetchClient only supports GET requests in Phase 2")

        started = time.monotonic()
        if self._parent is not None:
            reservation = self._parent.reserve("network_requests")
            try:
                timeout = self._parent.cap_timeout(request.timeout)
                reservation.commit()
                try:
                    raw = http_fetch(
                        request.url,
                        headers=_mutable_mapping_or_none(request.headers),
                        cookies=_mutable_mapping_or_none(request.cookies),
                        timeout=timeout,
                        backend=request.backend,
                        follow_redirects=False,
                        impersonate=request.impersonate,
                        proxies=_mutable_proxies(request.proxies),
                        max_response_bytes=request.max_response_bytes,
                        on_response_chunk=self._parent.charge_bytes,
                    )
                except ResponseObserverAborted as exc:
                    raise exc._cause from None
                self._parent.check()
                return FetchResponse.from_raw(
                    raw,
                    fallback_url=request.url,
                    fallback_backend=request.backend,
                    elapsed_seconds=max(0.0, time.monotonic() - started),
                )
            finally:
                reservation.close()
        if request.max_response_bytes is not None:
            raw = http_fetch(
                request.url,
                headers=_mutable_mapping_or_none(request.headers),
                cookies=_mutable_mapping_or_none(request.cookies),
                timeout=request.timeout,
                backend=request.backend,
                follow_redirects=request.allow_redirects,
                impersonate=request.impersonate,
                proxies=_mutable_proxies(request.proxies),
                max_response_bytes=request.max_response_bytes,
            )
            fallback_backend = request.backend
        elif request.backend == "curl":
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
