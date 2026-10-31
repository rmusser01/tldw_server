"""Bounded, credential-scoped model discovery for configured TTS gateways."""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from tldw_Server_API.app.core.exceptions import (
    DownloadError,
    EgressPolicyError,
    JSONDecodeError,
    NetworkError,
    RetryExhaustedError,
    StreamingProtocolError,
)
from tldw_Server_API.app.core.http_client import RetryPolicy, afetch_json

from .gateway_config import GatewaySpec, build_gateway_url

MAX_DISCOVERY_BYTES = 1_048_576
MAX_DISCOVERY_MODELS = 1_000
MAX_DISCOVERY_MODEL_ID_LENGTH = 512

DiscoveryStatus = Literal["fresh", "stale", "disabled", "unavailable"]
CatalogSource = Literal["discovery", "stale_cache", "static"]
CacheKey = tuple[str, str, str]


class GatewayDiscoveryPayloadError(Exception):
    """Raised when an upstream model catalog violates the bounded shape."""


_DISCOVERY_ERRORS = (
    DownloadError,
    EgressPolicyError,
    GatewayDiscoveryPayloadError,
    JSONDecodeError,
    NetworkError,
    RetryExhaustedError,
    StreamingProtocolError,
)


@dataclass(frozen=True, slots=True)
class GatewayCatalogResult:
    """Authorized model IDs and sanitized discovery freshness metadata."""

    backend_id: str
    models: tuple[str, ...]
    discovery_status: DiscoveryStatus
    source: CatalogSource
    stale: bool
    fetched_at: float | None
    fresh_until: float | None
    stale_until: float | None
    discovered_model_count: int | None


@dataclass(frozen=True, slots=True)
class _CacheEntry:
    fetched_at: float
    fresh_until: float
    stale_until: float
    discovered_models: tuple[str, ...]


def _retrieve_owned_task_exception(task: asyncio.Task[Any]) -> None:
    """Retrieve an owned task failure without retaining or logging its details."""
    if not task.cancelled():
        task.exception()


class GatewayCatalog:
    """Discover models with a bounded LRU and per-scope refresh coalescing."""

    def __init__(
        self,
        *,
        max_entries: int = 128,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if isinstance(max_entries, bool) or not isinstance(max_entries, int) or max_entries <= 0:
            raise ValueError("max_entries must be a positive integer")
        self._max_entries = max_entries
        self._clock = clock
        self._lock = asyncio.Lock()
        self._cache: OrderedDict[CacheKey, _CacheEntry] = OrderedDict()
        self._inflight: dict[CacheKey, asyncio.Task[GatewayCatalogResult]] = {}

    async def get(
        self,
        spec: GatewaySpec,
        *,
        credential_scope_token: str,
        api_key: str | None,
    ) -> GatewayCatalogResult:
        """Return the authorized catalog for one effective credential scope."""
        if not isinstance(credential_scope_token, str) or not credential_scope_token:
            raise ValueError("credential_scope_token must be a non-empty string")

        if not spec.enabled or not spec.discovery.enabled:
            return self._static_result(spec, status="disabled")
        if not spec.models_path:
            return self._static_result(spec, status="unavailable")

        key = self._cache_key(spec, credential_scope_token)
        now = self._clock()
        async with self._lock:
            entry = self._cache.get(key)
            if entry is not None and now < entry.fresh_until:
                self._cache.move_to_end(key)
                return self._result(spec, entry, status="fresh")

            refresh = self._inflight.get(key)
            if refresh is None:
                refresh = asyncio.create_task(self._refresh(key, spec, api_key))
                refresh.add_done_callback(_retrieve_owned_task_exception)
                self._inflight[key] = refresh

        return await asyncio.shield(refresh)

    @staticmethod
    def _cache_key(spec: GatewaySpec, credential_scope_token: str) -> CacheKey:
        scope_digest = hashlib.sha256(
            credential_scope_token.encode("utf-8"),
            usedforsecurity=True,
        ).hexdigest()
        return spec.backend_id, spec.config_generation, scope_digest

    async def _refresh(
        self,
        key: CacheKey,
        spec: GatewaySpec,
        api_key: str | None,
    ) -> GatewayCatalogResult:
        current = asyncio.current_task()
        try:
            discovered = await self._fetch_models(spec, api_key)
            fetched_at = self._clock()
            entry = _CacheEntry(
                fetched_at=fetched_at,
                fresh_until=fetched_at + spec.discovery.ttl_seconds,
                stale_until=fetched_at + spec.discovery.stale_ttl_seconds,
                discovered_models=discovered,
            )
            async with self._lock:
                self._cache[key] = entry
                self._cache.move_to_end(key)
                while len(self._cache) > self._max_entries:
                    self._cache.popitem(last=False)
            return self._result(spec, entry, status="fresh")
        except asyncio.CancelledError:
            raise
        except _DISCOVERY_ERRORS:
            now = self._clock()
            async with self._lock:
                entry = self._cache.get(key)
                if entry is not None and now <= entry.stale_until:
                    self._cache.move_to_end(key)
                elif entry is not None:
                    self._cache.pop(key, None)
                    entry = None
            if entry is not None:
                return self._result(spec, entry, status="stale")
            return self._static_result(spec, status="unavailable")
        finally:
            async with self._lock:
                if self._inflight.get(key) is current:
                    self._inflight.pop(key, None)

    @staticmethod
    async def _fetch_models(spec: GatewaySpec, api_key: str | None) -> tuple[str, ...]:
        headers = dict(spec.headers)
        if api_key is not None and api_key.strip():
            headers["Authorization"] = f"Bearer {api_key}"
        payload = await afetch_json(
            method="GET",
            url=str(build_gateway_url(spec.base_url, spec.models_path or "")),
            params=dict(spec.discovery_query),
            headers=headers,
            timeout=spec.discovery.timeout_seconds,
            retry=RetryPolicy(),
            require_json_ct=True,
            max_bytes=MAX_DISCOVERY_BYTES,
            allow_redirects=False,
        )
        return _parse_discovered_models(payload)

    @classmethod
    def _result(
        cls,
        spec: GatewaySpec,
        entry: _CacheEntry,
        *,
        status: Literal["fresh", "stale"],
    ) -> GatewayCatalogResult:
        return GatewayCatalogResult(
            backend_id=spec.backend_id,
            models=cls._authorized_models(spec, entry.discovered_models),
            discovery_status=status,
            source="discovery" if status == "fresh" else "stale_cache",
            stale=status == "stale",
            fetched_at=entry.fetched_at,
            fresh_until=entry.fresh_until,
            stale_until=entry.stale_until,
            discovered_model_count=len(entry.discovered_models),
        )

    @classmethod
    def _static_result(
        cls,
        spec: GatewaySpec,
        *,
        status: Literal["disabled", "unavailable"],
    ) -> GatewayCatalogResult:
        return GatewayCatalogResult(
            backend_id=spec.backend_id,
            models=cls._authorized_models(spec, ()),
            discovery_status=status,
            source="static",
            stale=False,
            fetched_at=None,
            fresh_until=None,
            stale_until=None,
            discovered_model_count=None,
        )

    @staticmethod
    def _authorized_models(
        spec: GatewaySpec,
        discovered_models: tuple[str, ...],
    ) -> tuple[str, ...]:
        discovered = frozenset(discovered_models)
        candidates = [*discovered_models]
        if spec.default_model is not None:
            candidates.append(spec.default_model)
        candidates.extend(spec.model_overrides)
        if spec.allowed_models_configured:
            candidates.extend(sorted(spec.allowed_models))

        authorized: list[str] = []
        seen: set[str] = set()
        for model_id in candidates:
            if model_id not in seen and spec.allows_model(model_id, discovered):
                authorized.append(model_id)
                seen.add(model_id)
        return tuple(authorized)


def _parse_discovered_models(payload: Any) -> tuple[str, ...]:
    """Validate and bound the standard OpenAI model-list response shape."""
    if not isinstance(payload, Mapping):
        raise GatewayDiscoveryPayloadError("invalid discovery payload")
    data = payload.get("data")
    if not isinstance(data, list) or len(data) > MAX_DISCOVERY_MODELS:
        raise GatewayDiscoveryPayloadError("invalid discovery payload")

    models: list[str] = []
    seen: set[str] = set()
    for item in data:
        if not isinstance(item, Mapping):
            raise GatewayDiscoveryPayloadError("invalid discovery payload")
        model_id = item.get("id")
        if not isinstance(model_id, str) or not model_id.strip() or len(model_id) > MAX_DISCOVERY_MODEL_ID_LENGTH:
            raise GatewayDiscoveryPayloadError("invalid discovery payload")
        if model_id not in seen:
            models.append(model_id)
            seen.add(model_id)
    return tuple(models)


__all__ = [
    "MAX_DISCOVERY_BYTES",
    "MAX_DISCOVERY_MODEL_ID_LENGTH",
    "MAX_DISCOVERY_MODELS",
    "GatewayCatalog",
    "GatewayCatalogResult",
    "GatewayDiscoveryPayloadError",
]
