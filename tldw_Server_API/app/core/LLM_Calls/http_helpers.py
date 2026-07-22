"""Helpers to construct a sync session facade for legacy streaming paths.

This module avoids importing chat_calls to prevent recursion. It uses the
centralized http_client underneath while preserving a minimal Session-like API.
Provider POSTs are deliberately single-attempt because no idempotency contract exists.
"""

import contextlib
from collections.abc import Iterable
from typing import Any, Optional

from tldw_Server_API.app.core.http_client import (
    RetryPolicy as _HC_RetryPolicy,
)
from tldw_Server_API.app.core.http_client import (
    create_client as _hc_create_client,
)
from tldw_Server_API.app.core.http_client import (
    fetch as _hc_fetch,
)


class _StreamResponse:
    def __init__(self, response: Any, ctx: Any) -> None:
        self._response = response
        self._ctx = ctx
        self._closed = False
        self.status_code = getattr(response, "status_code", None)
        self.headers = getattr(response, "headers", None)

    @property
    def text(self) -> str:
        return getattr(self._response, "text", "")

    def json(self) -> Any:
        return self._response.json()

    def raise_for_status(self) -> None:
        self._response.raise_for_status()

    def iter_lines(self, *args, **kwargs):
        return self._response.iter_lines(*args, **kwargs)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._response.close()
        finally:
            with contextlib.suppress(Exception):
                self._ctx.__exit__(None, None, None)


class _RetrySession:
    def __init__(
        self,
        *,
        total: int = 3,
        backoff_factor: float = 1.0,
        status_forcelist: Optional[Iterable[int]] = None,
        allowed_methods: Optional[Iterable[str]] = None,
    ) -> None:
        # Compatibility arguments are intentionally ignored: this facade only
        # exposes provider POSTs, which cannot be replayed without idempotency.
        _ = total, backoff_factor, status_forcelist, allowed_methods
        self._retry = _HC_RetryPolicy(attempts=1)
        self._client = None

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = _hc_create_client()
        return self._client

    def post(self, url, *, headers=None, json=None, stream: bool = False, timeout=None, **kwargs):
        if not stream:
            return _hc_fetch(
                method="POST",
                url=url,
                headers=headers,
                json=json,
                timeout=timeout,
                retry=self._retry,
                client=self._get_client(),
            )

        ctx = self._get_client().stream(
            "POST",
            url,
            headers=headers,
            json=json,
            timeout=timeout,
        )
        try:
            return _StreamResponse(ctx.__enter__(), ctx)
        except Exception:
            with contextlib.suppress(Exception):
                ctx.__exit__(None, None, None)
            raise

    def close(self) -> None:
        try:
            if self._client is not None:
                self._client.close()
        except Exception as close_error:
            _ = close_error  # best-effort client close


def create_session_with_retries(
    total: int = 3,
    backoff_factor: float = 1.0,
    status_forcelist: Optional[Iterable[int]] = None,
    allowed_methods: Optional[Iterable[str]] = None,
):
    """Return the legacy Session facade with single-attempt POST semantics."""
    return _RetrySession(
        total=total,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=allowed_methods,
    )
