from __future__ import annotations

import asyncio
import copy
import json
import os
import threading
import time
import weakref
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.byok_runtime import (
    ByokResolutionError,
    ServerFallbackCredentials,
    merge_server_fallback_snapshot,
)
from tldw_Server_API.app.core.AuthNZ.database import DatabasePool, get_db_pool
from tldw_Server_API.app.core.AuthNZ.repos.llm_provider_overrides_repo import (
    AuthnzLLMProviderOverridesRepo,
)
from tldw_Server_API.app.core.AuthNZ.user_provider_secrets import (
    decrypt_byok_payload,
    fold_provider_credential_rows,
    loads_envelope,
)
from tldw_Server_API.app.core.LLM_Calls.adapter_registry import (
    canonical_builtin_llm_provider_name,
)
from tldw_Server_API.app.core.LLM_Calls.provider_identity import canonical_provider_name
from tldw_Server_API.app.core.testing import is_truthy


@dataclass(frozen=True, repr=False)
class LLMProviderOverride:
    provider: str
    is_enabled: bool | None = None
    allowed_models: list[str] | None = None
    config: dict[str, Any] = field(default_factory=dict)
    api_key: str | None = None
    credential_fields: dict[str, Any] = field(default_factory=dict)
    credentials_invalid: bool = False
    api_key_hint: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def __repr__(self) -> str:
        """Return a bounded representation without decrypted credential data."""
        return (
            "LLMProviderOverride("
            f"provider={self.provider!r}, credentials=[REDACTED])"
        )


class LLMProviderOverridesRefreshError(RuntimeError):
    """Raised when the provider-override store cannot publish a fresh snapshot."""

    def __init__(self) -> None:
        super().__init__("Provider credential storage is temporarily unavailable.")


class ProviderOverridePolicyError(ByokResolutionError):
    """Typed, sanitized provider-policy denial at the credential boundary."""

    def __init__(self, policy_code: str, provider: str) -> None:
        if policy_code not in {"provider_disabled", "model_not_allowed"}:
            raise ValueError("Unsupported provider override policy error code")
        self.policy_code = policy_code
        super().__init__("invalid_provider_credentials", provider)


def _copy_override_map(
    overrides: Mapping[str, LLMProviderOverride],
) -> dict[str, LLMProviderOverride]:
    """Return a transitive copy so mutable nested fields never cross callers."""
    return copy.deepcopy(dict(overrides))


@dataclass(frozen=True, repr=False)
class ProviderOverrideCallSnapshot:
    """One redacted provider override snapshot used for policy and fallback."""

    provider: str
    _override: LLMProviderOverride | None = field(repr=False)

    def __repr__(self) -> str:
        return (
            "ProviderOverrideCallSnapshot("
            f"provider={self.provider!r}, override={'present' if self._override else 'absent'}"
            ")"
        )

    def policy_error(self, model: str | None) -> dict[str, str] | None:
        """Return the bounded policy decision from this captured snapshot."""
        override = self._override
        if override is None:
            return None
        if override.is_enabled is False:
            return {
                "error_code": "provider_disabled",
                "message": f"Provider '{self.provider}' is disabled by admin override.",
            }
        if override.allowed_models and model and model not in override.allowed_models:
            return {
                "error_code": "model_not_allowed",
                "message": f"Model '{model}' is not allowed for provider '{self.provider}'.",
            }
        return None

    def enforce(self, model: str | None) -> None:
        """Fail closed when this snapshot denies the provider/model pair."""
        error = self.policy_error(model)
        if error is not None:
            raise ProviderOverridePolicyError(error["error_code"], self.provider)

    def ensure_healthy(self) -> None:
        """Reject this captured snapshot if its backing store became unhealthy."""
        with _OVERRIDE_LOCK:
            healthy = _OVERRIDE_CACHE_HEALTHY
        if not healthy:
            raise ByokResolutionError("credential_store_unavailable", self.provider)

    def test_model(self) -> str | None:
        """Return the admin-test model from this captured override."""
        override = self._override
        if override is None:
            return None
        default_model = override.config.get("default_model")
        if isinstance(default_model, str) and default_model.strip():
            return default_model.strip()
        if override.allowed_models:
            first_model = override.allowed_models[0]
            if isinstance(first_model, str) and first_model.strip():
                return first_model.strip()
        return None

    def server_fallback(
        self,
        base_fallback: ServerFallbackCredentials | None = None,
    ) -> ServerFallbackCredentials | None:
        """Build the server fallback from the same captured override."""
        self.ensure_healthy()
        return _server_fallback_from_override(
            self.provider,
            self._override,
            base_fallback=base_fallback,
        )


_OVERRIDE_CACHE: dict[str, LLMProviderOverride] = {}
_OVERRIDE_CACHE_HEALTHY = False
_OVERRIDE_CACHE_REFRESHED_AT = 0.0
_OVERRIDE_CACHE_TTL_DISABLED_FOR_TESTS = False
_OVERRIDE_LOCK = threading.Lock()
_OVERRIDE_REFRESH_GENERATION = 0
_OVERRIDE_COMPLETED_GENERATION = 0
_OVERRIDE_RECOVERY_IN_FLIGHT = False
_OVERRIDE_RECOVERY_TASK: asyncio.Task[None] | None = None
_OVERRIDE_REFRESH_SERVICE_TASK: asyncio.Task[None] | None = None
_OVERRIDE_RETIRED_TASKS: set[asyncio.Task[None]] = set()
_OVERRIDE_TASK_EPOCH = 0
_OVERRIDE_RECOVERY_FAILURES = 0
_OVERRIDE_RECOVERY_NEXT_RETRY_AT = 0.0
_OVERRIDE_RECOVERY_BACKOFF_INITIAL_SECONDS = 1.0
_OVERRIDE_RECOVERY_BACKOFF_MAX_SECONDS = 30.0
_OVERRIDE_REFRESH_INTERVAL_SECONDS = 5.0
_OVERRIDE_MAX_STALE_SECONDS = 30.0
_OVERRIDE_REFRESH_TIMEOUT_SECONDS = 10.0
_OVERRIDE_REFRESH_LOCKS: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop,
    asyncio.Lock,
] = weakref.WeakKeyDictionary()


def _configured_seconds(name: str, default: float, *, minimum: float) -> float:
    """Read a bounded positive duration without making bad env input fatal."""
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    if value < minimum or value > 3600.0:
        return default
    return value


_OVERRIDE_REFRESH_INTERVAL_SECONDS = _configured_seconds(
    "TLDW_LLM_PROVIDER_OVERRIDE_REFRESH_INTERVAL_SECONDS",
    _OVERRIDE_REFRESH_INTERVAL_SECONDS,
    minimum=0.1,
)
_OVERRIDE_MAX_STALE_SECONDS = max(
    _OVERRIDE_REFRESH_INTERVAL_SECONDS,
    _configured_seconds(
        "TLDW_LLM_PROVIDER_OVERRIDE_MAX_STALE_SECONDS",
        _OVERRIDE_MAX_STALE_SECONDS,
        minimum=1.0,
    ),
)
_OVERRIDE_REFRESH_TIMEOUT_SECONDS = _configured_seconds(
    "TLDW_LLM_PROVIDER_OVERRIDE_REFRESH_TIMEOUT_SECONDS",
    _OVERRIDE_REFRESH_TIMEOUT_SECONDS,
    minimum=0.1,
)


def _get_override_refresh_lock() -> asyncio.Lock:
    """Return the refresh mutex owned by the current event loop."""
    loop = asyncio.get_running_loop()
    with _OVERRIDE_LOCK:
        lock = _OVERRIDE_REFRESH_LOCKS.get(loop)
        if lock is None:
            lock = asyncio.Lock()
            _OVERRIDE_REFRESH_LOCKS[loop] = lock
        return lock


def _begin_override_refresh(*, task_epoch: int | None = None) -> int | None:
    """Reserve a refresh generation unless an owned worker was reset."""
    global _OVERRIDE_REFRESH_GENERATION
    with _OVERRIDE_LOCK:
        if task_epoch is not None and task_epoch != _OVERRIDE_TASK_EPOCH:
            return None
        _OVERRIDE_REFRESH_GENERATION += 1
        return _OVERRIDE_REFRESH_GENERATION


def _cancel_task_on_owning_loop(task: asyncio.Task[None]) -> bool:
    """Cancel ``task`` on its owner loop and report whether it is drained."""

    completion = threading.Event()

    def complete_task(completed_task: asyncio.Task[None]) -> None:
        with _OVERRIDE_LOCK:
            _OVERRIDE_RETIRED_TASKS.discard(completed_task)
        completion.set()

    if task.done():
        with _OVERRIDE_LOCK:
            _OVERRIDE_RETIRED_TASKS.discard(task)
        return True
    loop = task.get_loop()
    if loop.is_closed():
        if task.done():
            complete_task(task)
            return True
        raise RuntimeError(
            "Cannot drain pending provider override task because its owner loop is closed"
        )
    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        current_loop = None

    def cancel_on_owner_loop() -> None:
        if task.done():
            complete_task(task)
            return
        task.add_done_callback(complete_task)
        task.cancel()

    if current_loop is loop:
        # A synchronous reset cannot wait on its own running loop. Retired-task
        # ownership keeps the original observable until its callback runs.
        cancel_on_owner_loop()
        return task.done()

    timeout = _OVERRIDE_REFRESH_TIMEOUT_SECONDS
    try:
        loop.call_soon_threadsafe(cancel_on_owner_loop)
    except RuntimeError as exc:
        if task.done():
            complete_task(task)
            return True
        raise RuntimeError(
            "Could not schedule provider override task cancellation on its owner loop"
        ) from exc
    if not loop.is_running():
        if task.done():
            complete_task(task)
            return True
        raise RuntimeError(
            "Cannot drain pending provider override task because its owner loop is stopped"
        )
    if not completion.wait(timeout=timeout):
        if task.done():
            complete_task(task)
            return True
        raise RuntimeError("Timed out draining provider override task on its owner loop")
    if not task.done():
        raise RuntimeError("Provider override task drain completed while still pending")
    return True


def _complete_override_refresh_failure(
    generation: int,
) -> dict[str, LLMProviderOverride]:
    """Fail the latest generation closed without letting stale work mutate state."""
    global _OVERRIDE_CACHE_HEALTHY
    global _OVERRIDE_COMPLETED_GENERATION
    global _OVERRIDE_RECOVERY_FAILURES
    global _OVERRIDE_RECOVERY_NEXT_RETRY_AT

    with _OVERRIDE_LOCK:
        if generation == _OVERRIDE_REFRESH_GENERATION:
            _OVERRIDE_CACHE_HEALTHY = False
            _OVERRIDE_COMPLETED_GENERATION = generation
            _OVERRIDE_RECOVERY_FAILURES += 1
            exponent = min(max(_OVERRIDE_RECOVERY_FAILURES - 1, 0), 16)
            delay = min(
                _OVERRIDE_RECOVERY_BACKOFF_MAX_SECONDS,
                _OVERRIDE_RECOVERY_BACKOFF_INITIAL_SECONDS * (2**exponent),
            )
            _OVERRIDE_RECOVERY_NEXT_RETRY_AT = time.monotonic() + delay
        elif (
            _OVERRIDE_COMPLETED_GENERATION == _OVERRIDE_REFRESH_GENERATION
            and _OVERRIDE_CACHE_HEALTHY
        ):
            return _copy_override_map(_OVERRIDE_CACHE)

    raise LLMProviderOverridesRefreshError() from None


def _complete_override_refresh_success(
    generation: int,
    overrides: dict[str, LLMProviderOverride],
) -> dict[str, LLMProviderOverride]:
    """Publish only the latest-started successful refresh snapshot."""
    global _OVERRIDE_CACHE_HEALTHY
    global _OVERRIDE_CACHE_REFRESHED_AT
    global _OVERRIDE_CACHE_TTL_DISABLED_FOR_TESTS
    global _OVERRIDE_COMPLETED_GENERATION
    global _OVERRIDE_RECOVERY_FAILURES
    global _OVERRIDE_RECOVERY_NEXT_RETRY_AT

    with _OVERRIDE_LOCK:
        if generation == _OVERRIDE_REFRESH_GENERATION:
            _OVERRIDE_CACHE.clear()
            _OVERRIDE_CACHE.update(_copy_override_map(overrides))
            _OVERRIDE_CACHE_HEALTHY = True
            _OVERRIDE_CACHE_REFRESHED_AT = time.monotonic()
            _OVERRIDE_CACHE_TTL_DISABLED_FOR_TESTS = False
            _OVERRIDE_COMPLETED_GENERATION = generation
            _OVERRIDE_RECOVERY_FAILURES = 0
            _OVERRIDE_RECOVERY_NEXT_RETRY_AT = 0.0
            return _copy_override_map(_OVERRIDE_CACHE)
        if (
            _OVERRIDE_COMPLETED_GENERATION == _OVERRIDE_REFRESH_GENERATION
            and _OVERRIDE_CACHE_HEALTHY
        ):
            return _copy_override_map(_OVERRIDE_CACHE)
        if _OVERRIDE_COMPLETED_GENERATION < _OVERRIDE_REFRESH_GENERATION:
            # Another loop started a newer read but has not published it yet.
            # This successful caller may return its own coherent result, while
            # generation ordering still prevents it from replacing the cache.
            return _copy_override_map(overrides)

    raise LLMProviderOverridesRefreshError() from None


def _schedule_override_recovery() -> None:
    """Schedule one owned, demand-driven refresh after the retry delay."""
    global _OVERRIDE_RECOVERY_IN_FLIGHT
    global _OVERRIDE_RECOVERY_TASK

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return

    with _OVERRIDE_LOCK:
        if (
            _OVERRIDE_RECOVERY_IN_FLIGHT
            or time.monotonic() < _OVERRIDE_RECOVERY_NEXT_RETRY_AT
        ):
            return
        if (
            _OVERRIDE_CACHE_HEALTHY
            and time.monotonic() - _OVERRIDE_CACHE_REFRESHED_AT
            < _OVERRIDE_REFRESH_INTERVAL_SECONDS
        ):
            return
        _OVERRIDE_RECOVERY_IN_FLIGHT = True
        task_epoch = _OVERRIDE_TASK_EPOCH

    async def _recover() -> None:
        global _OVERRIDE_RECOVERY_IN_FLIGHT
        global _OVERRIDE_RECOVERY_TASK
        current_task = asyncio.current_task()
        try:
            with _OVERRIDE_LOCK:
                if (
                    task_epoch != _OVERRIDE_TASK_EPOCH
                    or _OVERRIDE_RECOVERY_TASK is not current_task
                ):
                    return
            await refresh_llm_provider_overrides(_task_epoch=task_epoch)
        except LLMProviderOverridesRefreshError:
            pass
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("Unexpected provider override recovery failure")
        finally:
            with _OVERRIDE_LOCK:
                if _OVERRIDE_RECOVERY_TASK is current_task:
                    _OVERRIDE_RECOVERY_IN_FLIGHT = False
                    _OVERRIDE_RECOVERY_TASK = None

    try:
        task = loop.create_task(_recover(), name="llm-provider-overrides-recovery")
    except Exception:
        with _OVERRIDE_LOCK:
            if task_epoch == _OVERRIDE_TASK_EPOCH:
                _OVERRIDE_RECOVERY_IN_FLIGHT = False
        return
    cancel_task = False
    with _OVERRIDE_LOCK:
        if task_epoch == _OVERRIDE_TASK_EPOCH and _OVERRIDE_RECOVERY_IN_FLIGHT:
            _OVERRIDE_RECOVERY_TASK = task
        else:
            if not task.done():
                _OVERRIDE_RETIRED_TASKS.add(task)
            cancel_task = True
    if cancel_task:
        _cancel_task_on_owning_loop(task)


async def shutdown_llm_provider_override_recovery() -> None:
    """Cancel and drain process-owned provider override refresh tasks."""
    global _OVERRIDE_RECOVERY_IN_FLIGHT
    global _OVERRIDE_RECOVERY_TASK
    global _OVERRIDE_REFRESH_SERVICE_TASK
    global _OVERRIDE_TASK_EPOCH

    current_loop = asyncio.get_running_loop()
    with _OVERRIDE_LOCK:
        active_tasks = (
            _OVERRIDE_RECOVERY_TASK,
            _OVERRIDE_REFRESH_SERVICE_TASK,
        )
        for task in active_tasks:
            if task is not None and not task.done():
                _OVERRIDE_RETIRED_TASKS.add(task)
        _OVERRIDE_TASK_EPOCH += 1
        _OVERRIDE_RECOVERY_IN_FLIGHT = False
        _OVERRIDE_RECOVERY_TASK = None
        _OVERRIDE_REFRESH_SERVICE_TASK = None
        tasks = list(
            dict.fromkeys(
                task
                for task in (*active_tasks, *_OVERRIDE_RETIRED_TASKS)
                if task is not None
            )
        )
    if not tasks:
        return

    local_tasks: list[asyncio.Task[None]] = []
    foreign_tasks: list[asyncio.Task[None]] = []
    for task in tasks:
        if task.done():
            with _OVERRIDE_LOCK:
                _OVERRIDE_RETIRED_TASKS.discard(task)
        elif task.get_loop() is current_loop:
            local_tasks.append(task)
        else:
            foreign_tasks.append(task)

    for task in local_tasks:
        _cancel_task_on_owning_loop(task)
    foreign_waiters = [
        asyncio.create_task(
            asyncio.to_thread(_cancel_task_on_owning_loop, task),
            name="llm-provider-overrides-shutdown-foreign-drain",
        )
        for task in foreign_tasks
    ]
    pending_local: set[asyncio.Task[None]] = set()
    if local_tasks:
        done_local, pending_local = await asyncio.wait(
            local_tasks,
            timeout=_OVERRIDE_REFRESH_TIMEOUT_SECONDS,
        )
        if done_local:
            await asyncio.gather(*done_local, return_exceptions=True)
    foreign_results = await asyncio.gather(
        *foreign_waiters,
        return_exceptions=True,
    )
    with _OVERRIDE_LOCK:
        for task in tasks:
            if task.done():
                _OVERRIDE_RETIRED_TASKS.discard(task)

    drain_failures: list[RuntimeError] = []
    if pending_local:
        drain_failures.append(
            RuntimeError("Timed out draining provider override task during shutdown")
        )
    drain_failures.extend(
        result for result in foreign_results if isinstance(result, RuntimeError)
    )
    if drain_failures:
        raise drain_failures[0]


def start_llm_provider_override_refresh_service() -> None:
    """Start the process-local periodic refresh loop once."""
    global _OVERRIDE_REFRESH_SERVICE_TASK

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return

    with _OVERRIDE_LOCK:
        existing = _OVERRIDE_REFRESH_SERVICE_TASK
        if existing is not None and not existing.done():
            return
        task_epoch = _OVERRIDE_TASK_EPOCH

    async def _run() -> None:
        global _OVERRIDE_REFRESH_SERVICE_TASK
        current_task = asyncio.current_task()
        try:
            while True:
                with _OVERRIDE_LOCK:
                    if (
                        task_epoch != _OVERRIDE_TASK_EPOCH
                        or _OVERRIDE_REFRESH_SERVICE_TASK is not current_task
                    ):
                        return
                await asyncio.sleep(_OVERRIDE_REFRESH_INTERVAL_SECONDS)
                with _OVERRIDE_LOCK:
                    if (
                        task_epoch != _OVERRIDE_TASK_EPOCH
                        or _OVERRIDE_REFRESH_SERVICE_TASK is not current_task
                    ):
                        return
                    retry_is_deferred = (
                        time.monotonic() < _OVERRIDE_RECOVERY_NEXT_RETRY_AT
                    )
                if retry_is_deferred:
                    continue
                try:
                    await refresh_llm_provider_overrides(_task_epoch=task_epoch)
                except LLMProviderOverridesRefreshError:
                    continue
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.warning("Unexpected periodic provider override refresh failure")
        finally:
            with _OVERRIDE_LOCK:
                if _OVERRIDE_REFRESH_SERVICE_TASK is asyncio.current_task():
                    _OVERRIDE_REFRESH_SERVICE_TASK = None

    task = loop.create_task(_run(), name="llm-provider-overrides-refresh-service")
    cancel_task = False
    with _OVERRIDE_LOCK:
        existing = _OVERRIDE_REFRESH_SERVICE_TASK
        if (
            task_epoch == _OVERRIDE_TASK_EPOCH
            and (existing is None or existing.done())
        ):
            _OVERRIDE_REFRESH_SERVICE_TASK = task
        else:
            if not task.done():
                _OVERRIDE_RETIRED_TASKS.add(task)
            cancel_task = True
    if cancel_task:
        _cancel_task_on_owning_loop(task)


def _parse_json_value(raw: Any, *, field_name: str, expected_type: type) -> Any | None:
    """Parse a stored JSON field and reject corrupt or shape-invalid values."""
    if raw is None:
        return None
    value = raw
    if isinstance(raw, str):
        if not raw.strip():
            return None
        try:
            value = json.loads(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid stored {field_name}") from exc
    if not isinstance(value, expected_type):
        raise ValueError(f"Invalid stored {field_name} shape")
    return value


def _normalize_models(raw: Any | None) -> list[str] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = [v.strip() for v in raw.split(",")]
    if not isinstance(raw, list):
        return None
    cleaned = [str(v).strip() for v in raw if isinstance(v, (str, int, float)) and str(v).strip()]
    return cleaned or None


def _normalize_optional_bool(raw: Any) -> bool | None:
    if raw is None:
        return None
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)) and raw in {0, 1}:
        return bool(raw)
    if isinstance(raw, str):
        lowered = raw.strip().lower()
        if is_truthy(lowered):
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    raise ValueError("Invalid stored is_enabled value")


def _parse_override_row(row: dict[str, Any]) -> LLMProviderOverride:
    provider = canonical_builtin_llm_provider_name(row.get("provider"))
    allowed_models_raw = _parse_json_value(
        row.get("allowed_models"),
        field_name="allowed_models",
        expected_type=list,
    )
    if allowed_models_raw is not None and any(
        not isinstance(value, (str, int, float)) for value in allowed_models_raw
    ):
        raise ValueError("Invalid stored allowed_models entry")
    allowed_models = _normalize_models(allowed_models_raw)
    config = _parse_json_value(
        row.get("config_json"),
        field_name="config_json",
        expected_type=dict,
    ) or {}

    api_key: str | None = None
    credential_fields: dict[str, Any] = {}
    secret_blob = row.get("secret_blob")
    credentials_invalid = row.get("api_key_hint") is not None and not secret_blob
    if secret_blob:
        try:
            payload = decrypt_byok_payload(loads_envelope(secret_blob))
            if not isinstance(payload, dict):
                raise ValueError("Provider override credential payload must be an object")
            api_key_raw = payload.get("api_key")
            if isinstance(api_key_raw, str) and api_key_raw.strip():
                api_key = api_key_raw
            else:
                credentials_invalid = True
            credential_fields_raw = payload.get("credential_fields")
            if credential_fields_raw is None:
                credential_fields = {}
            elif isinstance(credential_fields_raw, dict):
                credential_fields = credential_fields_raw
            else:
                credentials_invalid = True
        except Exception:
            credentials_invalid = True
            logger.warning("Provider override decrypt failed")

    return LLMProviderOverride(
        provider=provider,
        is_enabled=_normalize_optional_bool(row.get("is_enabled")),
        allowed_models=allowed_models,
        config=config,
        api_key=api_key,
        credential_fields=credential_fields,
        credentials_invalid=credentials_invalid,
        api_key_hint=row.get("api_key_hint"),
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
    )


def _get_healthy_override_snapshot(
    provider: str = "provider-overrides",
) -> dict[str, LLMProviderOverride]:
    """Return one verified snapshot or fail closed and trigger recovery."""
    global _OVERRIDE_CACHE_HEALTHY
    global _OVERRIDE_RECOVERY_NEXT_RETRY_AT

    now = time.monotonic()
    schedule_refresh = False
    with _OVERRIDE_LOCK:
        age = (
            0.0
            if _OVERRIDE_CACHE_TTL_DISABLED_FOR_TESTS
            else max(0.0, now - _OVERRIDE_CACHE_REFRESHED_AT)
        )
        if (
            _OVERRIDE_CACHE_HEALTHY
            and not _OVERRIDE_CACHE_TTL_DISABLED_FOR_TESTS
            and age >= _OVERRIDE_MAX_STALE_SECONDS
        ):
            _OVERRIDE_CACHE_HEALTHY = False
            _OVERRIDE_RECOVERY_NEXT_RETRY_AT = 0.0
        elif (
            _OVERRIDE_CACHE_HEALTHY
            and not _OVERRIDE_CACHE_TTL_DISABLED_FOR_TESTS
            and age >= _OVERRIDE_REFRESH_INTERVAL_SECONDS
        ):
            schedule_refresh = True
        cache_healthy = _OVERRIDE_CACHE_HEALTHY
        snapshot = _copy_override_map(_OVERRIDE_CACHE)

    if schedule_refresh or not cache_healthy:
        _schedule_override_recovery()
    if not cache_healthy:
        raise ByokResolutionError(
            "credential_store_unavailable",
            canonical_provider_name(provider),
        )
    return snapshot


def get_llm_provider_override(provider: str) -> LLMProviderOverride | None:
    provider_norm = canonical_provider_name(provider)
    return _get_healthy_override_snapshot(provider_norm).get(provider_norm)


def get_llm_provider_overrides_snapshot() -> dict[str, LLMProviderOverride]:
    return _get_healthy_override_snapshot()


def capture_provider_override_call_snapshot(provider: str) -> ProviderOverrideCallSnapshot:
    """Capture one provider override for policy and server credential fallback."""
    provider_norm = canonical_provider_name(provider)
    override = _get_healthy_override_snapshot(provider_norm).get(provider_norm)
    return ProviderOverrideCallSnapshot(provider=provider_norm, _override=override)


def set_llm_provider_overrides_cache_for_tests(
    overrides: dict[str, LLMProviderOverride] | None,
    *,
    healthy: bool = True,
    ttl_enabled: bool = False,
) -> None:
    global _OVERRIDE_CACHE_HEALTHY
    global _OVERRIDE_CACHE_REFRESHED_AT
    global _OVERRIDE_CACHE_TTL_DISABLED_FOR_TESTS
    global _OVERRIDE_REFRESH_GENERATION
    global _OVERRIDE_COMPLETED_GENERATION
    global _OVERRIDE_RECOVERY_IN_FLIGHT
    global _OVERRIDE_RECOVERY_TASK
    global _OVERRIDE_REFRESH_SERVICE_TASK
    global _OVERRIDE_TASK_EPOCH
    global _OVERRIDE_RECOVERY_FAILURES
    global _OVERRIDE_RECOVERY_NEXT_RETRY_AT
    with _OVERRIDE_LOCK:
        recovery_task = _OVERRIDE_RECOVERY_TASK
        refresh_service_task = _OVERRIDE_REFRESH_SERVICE_TASK
        for task in (recovery_task, refresh_service_task):
            if task is not None and not task.done():
                _OVERRIDE_RETIRED_TASKS.add(task)
        _OVERRIDE_TASK_EPOCH += 1
        _OVERRIDE_REFRESH_GENERATION += 1
        _OVERRIDE_COMPLETED_GENERATION = _OVERRIDE_REFRESH_GENERATION
        _OVERRIDE_CACHE.clear()
        if overrides:
            _OVERRIDE_CACHE.update(_copy_override_map(overrides))
        _OVERRIDE_CACHE_HEALTHY = healthy is True
        _OVERRIDE_CACHE_REFRESHED_AT = time.monotonic()
        # Explicit test snapshots are deterministic and must not start real DB
        # recovery tasks merely because a long randomized suite crosses the
        # production TTL. TTL regressions opt in explicitly.
        _OVERRIDE_CACHE_TTL_DISABLED_FOR_TESTS = not ttl_enabled
        _OVERRIDE_RECOVERY_IN_FLIGHT = False
        _OVERRIDE_RECOVERY_TASK = None
        _OVERRIDE_REFRESH_SERVICE_TASK = None
        _OVERRIDE_RECOVERY_FAILURES = 0
        _OVERRIDE_RECOVERY_NEXT_RETRY_AT = 0.0 if healthy else float("inf")
    drain_failures: list[RuntimeError] = []
    for task in (recovery_task, refresh_service_task):
        if task is None:
            continue
        try:
            _cancel_task_on_owning_loop(task)
        except RuntimeError as exc:
            drain_failures.append(exc)
    if drain_failures:
        raise drain_failures[0]


def apply_llm_provider_overrides_to_listing(
    payload: dict[str, Any],
    *,
    overrides: Mapping[str, LLMProviderOverride] | None = None,
) -> dict[str, Any]:
    if overrides is None:
        overrides = get_llm_provider_overrides_snapshot()
    if not overrides:
        return payload

    providers = payload.get("providers", [])
    if not isinstance(providers, list):
        return payload

    updated_providers = []
    for entry in providers:
        if not isinstance(entry, dict):
            updated_providers.append(entry)
            continue
        provider_name = canonical_provider_name(entry.get("name"))
        override = overrides.get(provider_name)
        if not override:
            entry.setdefault("enabled", True)
            updated_providers.append(entry)
            continue

        merged = dict(entry)
        merged["enabled"] = override.is_enabled if override.is_enabled is not None else merged.get("enabled", True)

        models = list(merged.get("models") or [])
        config_models = override.config.get("models")
        if isinstance(config_models, list):
            models = [str(v).strip() for v in config_models if str(v).strip()]
        if override.allowed_models:
            models = [m for m in models if m in override.allowed_models] if models else list(override.allowed_models)

        preferred_order = _get_model_priority_from_override(override, "highest_quality")
        if preferred_order:
            order_index = {name: index for index, name in enumerate(preferred_order)}
            models = sorted(
                models,
                key=lambda model_name: (order_index.get(model_name, len(order_index)), model_name),
            )
        merged["models"] = models

        models_info = merged.get("models_info")
        if isinstance(models_info, list) and (override.allowed_models or isinstance(config_models, list)):
            filtered_info = []
            for mi in models_info:
                if not isinstance(mi, dict):
                    continue
                if mi.get("name") in models:
                    filtered_info.append(mi)
            merged["models_info"] = filtered_info
        if isinstance(merged.get("models_info"), list) and preferred_order:
            order_index = {name: index for index, name in enumerate(preferred_order)}
            models_info_entries = [
                model_info for model_info in merged["models_info"]
                if isinstance(model_info, dict)
            ]
            merged["models_info"] = sorted(
                models_info_entries,
                key=lambda model_info: (
                    order_index.get(str(model_info.get("name") or ""), len(order_index)),
                    str(model_info.get("name") or ""),
                ),
            )

        default_model = override.config.get("default_model")
        if isinstance(default_model, str) and default_model.strip():
            merged["default_model"] = default_model.strip()

        updated_providers.append(merged)

    updated = dict(payload)
    updated["providers"] = updated_providers
    return updated


def validate_provider_override(provider: str, model: str | None) -> dict[str, str] | None:
    return capture_provider_override_call_snapshot(provider).policy_error(model)


def _get_model_priority_from_override(
    override: LLMProviderOverride,
    objective: str,
) -> list[str] | None:
    """Resolve model priority from an already captured provider override."""
    config = override.config if isinstance(override.config, dict) else {}

    routing_config = config.get("routing")
    if isinstance(routing_config, dict):
        model_rankings = routing_config.get("model_rankings")
        if isinstance(model_rankings, dict):
            ranked_models = _normalize_models(model_rankings.get(objective))
            if ranked_models:
                return ranked_models

    model_rankings = config.get("model_rankings")
    if isinstance(model_rankings, dict):
        ranked_models = _normalize_models(model_rankings.get(objective))
        if ranked_models:
            return ranked_models

    return None


def get_override_model_priority(
    provider: str,
    objective: str = "highest_quality",
    *,
    overrides: Mapping[str, LLMProviderOverride] | None = None,
) -> list[str] | None:
    override = (
        get_llm_provider_override(provider)
        if overrides is None
        else overrides.get(canonical_provider_name(provider))
    )
    if not override:
        return None
    return _get_model_priority_from_override(override, objective)


def get_override_default_model(provider: str) -> str | None:
    override = get_llm_provider_override(provider)
    if not override:
        return None
    default_model = override.config.get("default_model")
    if isinstance(default_model, str) and default_model.strip():
        return default_model.strip()
    return None


def get_override_credentials(provider: str) -> dict[str, Any] | None:
    override = get_llm_provider_override(provider)
    if not override:
        return None
    if not override.api_key and not override.credential_fields:
        return None
    return {
        "api_key": override.api_key,
        "credential_fields": override.credential_fields,
    }


def _server_fallback_from_override(
    provider: str,
    override: LLMProviderOverride | None,
    *,
    base_fallback: ServerFallbackCredentials | None = None,
) -> ServerFallbackCredentials | None:
    """Build one sanitized server fallback from an already captured override."""
    try:
        if not override:
            return None
        if override.credentials_invalid:
            raise ByokResolutionError("invalid_provider_credentials", provider)
        config = override.config if isinstance(override.config, dict) else {}
        raw_auth_source = config.get("auth_source") if "auth_source" in config else None
        if raw_auth_source is not None and (
            not isinstance(raw_auth_source, str) or not raw_auth_source.strip()
        ):
            raise ByokResolutionError("invalid_provider_credentials", provider)
        if override.api_key is not None and (
            not isinstance(override.api_key, str) or not override.api_key.strip()
        ):
            raise ByokResolutionError("invalid_provider_credentials", provider)
        if not isinstance(override.credential_fields, Mapping):
            raise ByokResolutionError("invalid_provider_credentials", provider)
        credential_fields = dict(override.credential_fields)
    except ByokResolutionError:
        raise
    except Exception:
        raise ByokResolutionError("invalid_provider_credentials", provider) from None

    frozen_base = base_fallback or ServerFallbackCredentials(
        api_key=None,
        credential_fields={},
        app_config={},
    )
    base_fields = frozen_base.credential_fields
    if not isinstance(base_fields, Mapping):
        raise ByokResolutionError("invalid_provider_credentials", provider)
    use_override_credentials = (
        override.api_key is not None
        or bool(credential_fields)
        or "auth_source" in config
    )
    effective_fields = (
        credential_fields if use_override_credentials else dict(base_fields)
    )
    effective_api_key = (
        override.api_key if override.api_key is not None else frozen_base.api_key
    )
    if "auth_source" in config:
        auth_source = raw_auth_source.strip() if isinstance(raw_auth_source, str) else None
    elif override.api_key is not None:
        auth_source = None
    else:
        auth_source = frozen_base.auth_source

    merged = merge_server_fallback_snapshot(
        provider,
        frozen_base,
        api_key=effective_api_key,
        credential_fields=effective_fields,
        auth_source=auth_source,
        provider_config=config,
        replace_credential_metadata=use_override_credentials,
    )
    if (
        base_fallback is None
        and merged.api_key is None
        and not merged.credential_fields
        and merged.auth_source is None
        and not merged.app_config
    ):
        return None
    return merged


def get_override_server_fallback(provider: str) -> ServerFallbackCredentials | None:
    """Return one atomic server fallback for a configured provider override."""
    return capture_provider_override_call_snapshot(provider).server_fallback()


async def refresh_llm_provider_overrides(
    pool: DatabasePool | None = None,
    *,
    force: bool | None = None,
    _task_epoch: int | None = None,
) -> dict[str, LLMProviderOverride]:
    """Load and atomically publish a bounded, serialized override snapshot."""
    with _OVERRIDE_LOCK:
        requested_completed_generation = _OVERRIDE_COMPLETED_GENERATION
    should_force = pool is not None if force is None else force
    refresh_lock = _get_override_refresh_lock()

    async with refresh_lock:
        if not should_force:
            with _OVERRIDE_LOCK:
                completed_after_request = (
                    requested_completed_generation < _OVERRIDE_COMPLETED_GENERATION
                )
                if (
                    _OVERRIDE_CACHE_HEALTHY
                    and completed_after_request
                ):
                    return _copy_override_map(_OVERRIDE_CACHE)

        generation = _begin_override_refresh(task_epoch=_task_epoch)
        if generation is None:
            with _OVERRIDE_LOCK:
                return _copy_override_map(_OVERRIDE_CACHE)

        async def _load_rows() -> list[dict[str, Any]]:
            db_pool = pool or await get_db_pool()
            repo = AuthnzLLMProviderOverridesRepo(db_pool)
            return await repo.list_overrides()

        load_failed = False
        try:
            rows = await asyncio.wait_for(
                _load_rows(),
                timeout=_OVERRIDE_REFRESH_TIMEOUT_SECONDS,
            )
        except asyncio.CancelledError:
            try:
                _complete_override_refresh_failure(generation)
            except LLMProviderOverridesRefreshError:
                pass
            raise
        except Exception:
            logger.warning("Failed to load provider overrides")
            load_failed = True
        if load_failed:
            return _complete_override_refresh_failure(generation)

        overrides: dict[str, LLMProviderOverride] = {}
        parse_failed = False
        try:
            for row in fold_provider_credential_rows(rows):
                override = _parse_override_row(row)
                overrides[override.provider] = override
        except Exception:
            logger.warning("Failed to parse provider override row")
            parse_failed = True
        if parse_failed:
            return _complete_override_refresh_failure(generation)

        return _complete_override_refresh_success(generation, overrides)
