"""Execution-scoped provider credential resolution and usage tracking."""

from __future__ import annotations

import asyncio
import copy
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, NoReturn

from tldw_Server_API.app.core.AuthNZ.byok_runtime import (
    ByokResolutionError,
    ByokResolutionStatus,
    ResolvedByokCredentials,
    ServerFallbackCredentials,
    resolve_byok_credentials,
)
from tldw_Server_API.app.core.LLM_Calls.provider_identity import canonical_provider_name

_Resolver = Callable[..., Awaitable[ResolvedByokCredentials]]
_FallbackResolver = Callable[[str], str | ServerFallbackCredentials | None]
USAGE_TASK_DRAIN_TIMEOUT_SECONDS = 0.25


def _serialization_error() -> TypeError:
    return TypeError("ProviderCallCredentials cannot be serialized")


def _copy_error() -> TypeError:
    return TypeError("ProviderCallCredentials cannot be copied")


def _reject_pydantic_serialization(_value: object) -> NoReturn:
    raise _serialization_error()


def _no_fallback(_provider: str) -> None:
    """Return an empty fallback after runtime cleanup."""
    return None


def _copy_app_config(
    app_config: dict[str, Any] | None,
) -> dict[str, Any] | None:
    copy_failed = False
    copied: dict[str, Any] | None = None
    try:
        copied = copy.deepcopy(app_config)
    except Exception:  # noqa: BLE001 - hostile credential metadata must fail closed
        copy_failed = True
    if copy_failed:
        raise RuntimeError("Provider credential configuration is unavailable")
    return copied


class ProviderCallCredentials:
    """Redacted, non-serializable credentials for one provider call."""

    __slots__ = (
        "provider",
        "api_key",
        "app_config",
        "auth_source",
        "endpoint_provenance",
        "credentials_resolved",
        "_runtime_generation",
        "_runtime_identity",
        "_credential_identity",
    )

    def __init__(
        self,
        *,
        provider: str,
        api_key: str | None,
        app_config: dict[str, Any] | None,
        auth_source: str | None,
        runtime_generation: int,
        runtime_identity: object,
        credential_identity: object,
        endpoint_provenance: str = "server_config",
    ) -> None:
        self.provider = provider
        self.api_key = api_key
        self.app_config = app_config
        self.auth_source = auth_source
        self.endpoint_provenance = endpoint_provenance
        self.credentials_resolved = True
        self._runtime_generation = runtime_generation
        self._runtime_identity = runtime_identity
        self._credential_identity = credential_identity

    def __repr__(self) -> str:
        return f"ProviderCallCredentials(provider={self.provider!r}, credentials=[REDACTED])"

    def __reduce__(self) -> NoReturn:
        raise _serialization_error()

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        del protocol
        raise _serialization_error()

    def __getstate__(self) -> NoReturn:
        raise _serialization_error()

    def __copy__(self) -> NoReturn:
        raise _copy_error()

    def __deepcopy__(self, memo) -> NoReturn:
        del memo
        raise _copy_error()

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler) -> Any:
        del source_type, handler
        from pydantic_core import core_schema

        return core_schema.is_instance_schema(
            cls,
            serialization=core_schema.plain_serializer_function_ser_schema(
                _reject_pydantic_serialization,
            ),
        )

    def model_dump(self, *args, **kwargs) -> NoReturn:
        del args, kwargs
        raise _serialization_error()

    def model_dump_json(self, *args, **kwargs) -> NoReturn:
        del args, kwargs
        raise _serialization_error()

    def dict(self, *args, **kwargs) -> NoReturn:
        del args, kwargs
        raise _serialization_error()

    def json(self, *args, **kwargs) -> NoReturn:
        del args, kwargs
        raise _serialization_error()

    def __json__(self, *args, **kwargs) -> NoReturn:
        del args, kwargs
        raise _serialization_error()


def reject_provider_call_credentials(value: object) -> None:
    """Reject provider credentials nested in supported persistence containers."""
    seen: set[int] = set()

    def visit(item: object) -> None:
        if isinstance(item, ProviderCallCredentials):
            raise _serialization_error()
        if not isinstance(item, (Mapping, list, tuple, set, frozenset)):
            return

        identity = id(item)
        if identity in seen:
            return
        seen.add(identity)

        if isinstance(item, Mapping):
            for key, nested in item.items():
                visit(key)
                visit(nested)
            return
        for nested in item:
            visit(nested)

    visit(value)


class _ResolvedEntry:
    __slots__ = ("generation", "identity", "resolution", "used")

    def __init__(
        self,
        generation: int,
        resolution: ResolvedByokCredentials,
    ) -> None:
        self.generation = generation
        self.identity = object()
        self.resolution = resolution
        self.used = False


class ProviderCredentialRuntime:
    """Resolve and cache provider credentials for one trusted execution."""

    __slots__ = (
        "_user_id",
        "_team_ids",
        "_org_ids",
        "_trusted_base_url_override",
        "_fallback_resolver",
        "_resolver",
        "_identity",
        "_cache",
        "_inflight",
        "_refresh_tasks",
        "_usage_tasks",
        "_refresh_locks",
        "_refreshed_generations",
        "_generations",
        "_closed",
        "_close_task",
    )

    def __init__(
        self,
        *,
        user_id: int | None,
        team_ids: Sequence[int] | None,
        org_ids: Sequence[int] | None,
        trusted_base_url_override: bool,
        fallback_resolver: _FallbackResolver,
        resolver: _Resolver | None = None,
    ) -> None:
        self._user_id = user_id
        self._team_ids = list(team_ids or ())
        self._org_ids = list(org_ids or ())
        self._trusted_base_url_override = trusted_base_url_override is True
        self._fallback_resolver = fallback_resolver
        self._resolver = resolver or resolve_byok_credentials
        self._identity = object()
        self._cache: dict[str, _ResolvedEntry] = {}
        self._inflight: dict[str, asyncio.Task[_ResolvedEntry]] = {}
        self._refresh_tasks: dict[str, asyncio.Task[_ResolvedEntry]] = {}
        self._usage_tasks: dict[_ResolvedEntry, asyncio.Task[None]] = {}
        self._refresh_locks: dict[str, asyncio.Lock] = {}
        self._refreshed_generations: dict[str, int] = {}
        self._generations: dict[str, int] = {}
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None

    async def resolve(
        self,
        provider: str,
        *,
        force_refresh: bool = False,
    ) -> ProviderCallCredentials:
        """Return explicit credentials for a normalized provider."""
        self._ensure_open()
        provider_norm = canonical_provider_name(provider)
        if force_refresh:
            return await self._refresh(provider_norm)

        while True:
            self._ensure_open()
            entry = self._cache.get(provider_norm)
            if entry is not None:
                return self._new_handle(provider_norm, entry)

            task = self._refresh_tasks.get(provider_norm)
            if task is None:
                task = self._inflight.get(provider_norm)
            if task is None:
                generation = self._generations.setdefault(provider_norm, 0)
                task = asyncio.create_task(
                    self._resolve_entry(
                        provider_norm,
                        generation=generation,
                        force_refresh=False,
                    )
                )
                self._inflight[provider_norm] = task
                task.add_done_callback(
                    lambda completed, name=provider_norm: self._forget_task(name, completed, refresh=False)
                )

            await self._await_owned(task)

    async def mark_used(self, handle: ProviderCallCredentials) -> None:
        """Record successful use once for a current handle generation."""
        self._ensure_open()
        if not isinstance(handle, ProviderCallCredentials):
            raise RuntimeError("Credential handle was not issued by this runtime")
        if handle._runtime_identity is not self._identity:
            raise RuntimeError("Credential handle was not issued by this runtime")

        entry = self._cache.get(handle.provider)
        if (
            entry is None
            or entry.generation != handle._runtime_generation
            or entry.identity is not handle._credential_identity
        ):
            return
        if entry.used:
            return

        task = self._usage_tasks.get(entry)
        if task is None:
            task = asyncio.create_task(self._mark_entry_used(handle.provider, entry))
            self._usage_tasks[entry] = task
            task.add_done_callback(
                lambda completed, owned_entry=entry: self._forget_usage_task(
                    owned_entry,
                    completed,
                )
            )

        await self._await_usage_task(task)
        if self._closed:
            raise RuntimeError("Provider credential runtime is closed")

    async def close(self) -> None:
        """Cancel owned work and release all execution-scoped references."""
        task = self._close_task
        if task is None:
            if self._closed:
                return
            self._closed = True
            task = asyncio.create_task(self._close_owned())
            self._close_task = task
            task.add_done_callback(self._close_task_done)

        await asyncio.shield(task)

    async def _refresh(self, provider: str) -> ProviderCallCredentials:
        lock = self._refresh_locks.setdefault(provider, asyncio.Lock())
        async with lock:
            self._ensure_open()
            task = self._refresh_tasks.get(provider)
            if task is None:
                entry = self._cache.get(provider)
                if entry is not None and self._refreshed_generations.get(provider) == entry.generation:
                    return self._new_handle(provider, entry)
                generation = self._generations.get(provider, 0) + 1
                self._generations[provider] = generation
                self._cache.pop(provider, None)
                task = asyncio.create_task(
                    self._resolve_entry(
                        provider,
                        generation=generation,
                        force_refresh=True,
                    )
                )
                self._refresh_tasks[provider] = task
                task.add_done_callback(
                    lambda completed, name=provider: self._forget_task(name, completed, refresh=True)
                )

        await self._await_owned(task)
        return await self.resolve(provider)

    async def _resolve_entry(
        self,
        provider: str,
        *,
        generation: int,
        force_refresh: bool,
    ) -> _ResolvedEntry:
        failure_code: str | None = None
        unexpected_failure = False
        resolution: ResolvedByokCredentials | None = None
        try:
            resolution = await self._resolver(
                provider,
                user_id=self._user_id,
                team_ids=list(self._team_ids),
                org_ids=list(self._org_ids),
                fallback_resolver=self._fallback_resolver,
                force_oauth_refresh=force_refresh,
                trusted_base_url_override=self._trusted_base_url_override,
            )
        except asyncio.CancelledError:
            raise
        except ByokResolutionError as exc:
            failure_code = exc.code
        except Exception:  # noqa: BLE001 - unexpected resolver failures are sanitized below
            unexpected_failure = True

        if failure_code is not None:
            raise ByokResolutionError(failure_code, provider)
        if unexpected_failure:
            raise RuntimeError("Provider credential resolution failed")
        if self._closed:
            raise RuntimeError("Provider credential runtime is closed")

        valid = False
        try:
            valid = (
                isinstance(resolution, ResolvedByokCredentials)
                and resolution.status in {ByokResolutionStatus.RESOLVED, ByokResolutionStatus.ABSENT}
                and canonical_provider_name(resolution.provider) == provider
            )
        except Exception:  # noqa: BLE001 - malformed resolver objects fail validation
            valid = False
        if not valid or resolution is None:
            raise RuntimeError("Provider credential resolution failed")

        entry = _ResolvedEntry(generation, resolution)
        if self._generations.get(provider, 0) == generation and not self._closed:
            self._cache[provider] = entry
            if force_refresh:
                self._refreshed_generations[provider] = generation
        return entry

    async def _mark_entry_used(
        self,
        provider: str,
        entry: _ResolvedEntry,
    ) -> None:
        """Persist usage before publishing the entry as used."""
        try:
            touch_callback = entry.resolution._touch_cb
            if touch_callback is not None:
                await touch_callback()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - usage persistence is best-effort and retryable
            return
        if self._cache.get(provider) is entry:
            entry.used = True

    async def _await_usage_task(self, task: asyncio.Task[None]) -> None:
        """Shield and drain usage persistence before cancellation escapes."""
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            await self._drain_usage_tasks({task})
            if self._closed:
                raise RuntimeError("Provider credential runtime is closed") from None
            raise

    async def _drain_usage_tasks(self, tasks: set[asyncio.Task[None]]) -> None:
        """Wait briefly for usage writes, then cancel and release owned references."""

        pending = {task for task in tasks if not task.done()}
        if not pending:
            return

        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(0.0, float(USAGE_TASK_DRAIN_TIMEOUT_SECONDS))
        while pending:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                _done, pending = await asyncio.wait(pending, timeout=remaining)
            except asyncio.CancelledError:
                # Caller cancellation stays authoritative, but cannot restart the deadline.
                continue

        if not pending:
            return

        for task in pending:
            task.cancel()
        for entry, owned_task in tuple(self._usage_tasks.items()):
            if owned_task in pending:
                self._usage_tasks.pop(entry, None)
        try:
            await asyncio.sleep(0)
        except asyncio.CancelledError:
            pass

    async def _await_owned(self, task: asyncio.Task[_ResolvedEntry]) -> _ResolvedEntry:
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            if self._closed:
                raise RuntimeError("Provider credential runtime is closed") from None
            raise

    def _new_handle(
        self,
        provider: str,
        entry: _ResolvedEntry,
    ) -> ProviderCallCredentials:
        resolution = entry.resolution
        return ProviderCallCredentials(
            provider=provider,
            api_key=resolution.api_key,
            app_config=_copy_app_config(resolution.app_config),
            auth_source=resolution.auth_source,
            runtime_generation=entry.generation,
            runtime_identity=self._identity,
            credential_identity=entry.identity,
            endpoint_provenance=(
                "byok"
                if isinstance(resolution.credential_fields.get("base_url"), str)
                and bool(resolution.credential_fields["base_url"].strip())
                else "server_config"
            ),
        )

    def _forget_task(
        self,
        provider: str,
        task: asyncio.Task[_ResolvedEntry],
        *,
        refresh: bool,
    ) -> None:
        tasks = self._refresh_tasks if refresh else self._inflight
        if tasks.get(provider) is task:
            tasks.pop(provider, None)
        if not task.cancelled():
            task.exception()

    def _forget_usage_task(
        self,
        entry: _ResolvedEntry,
        task: asyncio.Task[None],
    ) -> None:
        if self._usage_tasks.get(entry) is task:
            self._usage_tasks.pop(entry, None)
        if not task.cancelled():
            task.exception()

    async def _close_owned(self) -> None:
        cancellable_tasks = set(self._inflight.values()) | set(self._refresh_tasks.values())
        usage_tasks = set(self._usage_tasks.values())
        for task in cancellable_tasks:
            if not task.done():
                task.cancel()
        if cancellable_tasks:
            await asyncio.gather(*cancellable_tasks, return_exceptions=True)
        if usage_tasks:
            await self._drain_usage_tasks(usage_tasks)

        self._cache.clear()
        self._inflight.clear()
        self._refresh_tasks.clear()
        self._usage_tasks.clear()
        self._refresh_locks.clear()
        self._refreshed_generations.clear()
        self._generations.clear()
        self._team_ids.clear()
        self._org_ids.clear()
        self._user_id = None
        self._trusted_base_url_override = False
        self._fallback_resolver = _no_fallback
        self._resolver = resolve_byok_credentials
        self._identity = object()

    def _close_task_done(self, task: asyncio.Task[None]) -> None:
        """Release and safely observe the owned cleanup task."""
        try:
            if not task.cancelled():
                task.exception()
        except (asyncio.CancelledError, Exception):  # noqa: BLE001 - callback observes task failure
            return
        finally:
            if self._close_task is task:
                self._close_task = None

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Provider credential runtime is closed")
