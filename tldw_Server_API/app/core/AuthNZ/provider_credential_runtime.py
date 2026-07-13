"""Execution-scoped provider credential resolution and usage tracking."""

from __future__ import annotations

import asyncio
import copy
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, NoReturn

from tldw_Server_API.app.core.AuthNZ.byok_runtime import (
    ByokResolutionError,
    ByokResolutionStatus,
    ResolvedByokCredentials,
    resolve_byok_credentials,
)
from tldw_Server_API.app.core.AuthNZ.user_provider_secrets import (
    normalize_provider_name,
)


_Resolver = Callable[..., Awaitable[ResolvedByokCredentials]]
_FallbackResolver = Callable[[str], str | None]


def _serialization_error() -> TypeError:
    return TypeError("ProviderCallCredentials cannot be serialized")


def _copy_error() -> TypeError:
    return TypeError("ProviderCallCredentials cannot be copied")


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
    except Exception:
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
    ) -> None:
        self.provider = provider
        self.api_key = api_key
        self.app_config = app_config
        self.auth_source = auth_source
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
    def __get_pydantic_core_schema__(cls, source_type, handler) -> NoReturn:
        del source_type, handler
        raise _serialization_error()

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


class _ResolvedEntry:
    __slots__ = ("generation", "identity", "resolution", "used")

    def __init__(
        self,
        generation: int,
        resolution: ResolvedByokCredentials,
    ) -> None:
        touch_callback = resolution._touch_cb
        if touch_callback is not None:

            async def touch_without_leaking_failure() -> None:
                try:
                    await touch_callback()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    return

            resolution._touch_cb = touch_without_leaking_failure
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
        provider_norm = normalize_provider_name(provider)
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

        entry.used = True
        await entry.resolution.touch_last_used()

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
        except Exception:
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
                and normalize_provider_name(resolution.provider) == provider
            )
        except Exception:
            valid = False
        if not valid or resolution is None:
            raise RuntimeError("Provider credential resolution failed")

        entry = _ResolvedEntry(generation, resolution)
        if self._generations.get(provider, 0) == generation and not self._closed:
            self._cache[provider] = entry
            if force_refresh:
                self._refreshed_generations[provider] = generation
        return entry

    async def _await_owned(self, task: asyncio.Task[_ResolvedEntry]) -> _ResolvedEntry:
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            if self._closed:
                raise RuntimeError("Provider credential runtime is closed")
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

    async def _close_owned(self) -> None:
        tasks = set(self._inflight.values()) | set(self._refresh_tasks.values())
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        self._cache.clear()
        self._inflight.clear()
        self._refresh_tasks.clear()
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
        except (asyncio.CancelledError, Exception):
            return
        finally:
            if self._close_task is task:
                self._close_task = None

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Provider credential runtime is closed")
