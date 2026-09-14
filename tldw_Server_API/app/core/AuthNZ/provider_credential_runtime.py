"""Execution-scoped provider credential resolution and usage tracking."""

from __future__ import annotations

import asyncio
import copy
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NoReturn, Protocol

from tldw_Server_API.app.core.AuthNZ.byok_config import PROVIDER_APP_CONFIG_KEYS
from tldw_Server_API.app.core.AuthNZ.byok_helpers import load_server_config_snapshot
from tldw_Server_API.app.core.AuthNZ.byok_runtime import (
    ByokResolutionError,
    ByokResolutionStatus,
    ResolvedByokCredentials,
    ServerFallbackCredentials,
    resolve_byok_credentials,
    resolve_static_server_fallback_from_snapshot,
)
from tldw_Server_API.app.core.custom_openai_providers import (
    custom_openai_provider_number,
    custom_openai_section_name,
)
from tldw_Server_API.app.core.exceptions import raise_detached_error
from tldw_Server_API.app.core.LLM_Calls.provider_config_resolution import (
    TrustedProviderEndpoint,
    valid_provider_config_value,
)
from tldw_Server_API.app.core.LLM_Calls.provider_identity import canonical_provider_name
from tldw_Server_API.app.core.Security.egress import ConfiguredEndpointScope

_Resolver = Callable[..., Awaitable[ResolvedByokCredentials]]
_FallbackResolver = Callable[[str], str | ServerFallbackCredentials | None]


class _ProviderOverrideSnapshot(Protocol):
    """Policy and fallback operations captured from one override snapshot."""

    def enforce(self, model: str | None) -> None: ...

    def ensure_healthy(self) -> None: ...

    def server_fallback(
        self,
        base_fallback: ServerFallbackCredentials | None = None,
    ) -> ServerFallbackCredentials | None: ...


_OverrideSnapshotResolver = Callable[[str], _ProviderOverrideSnapshot]
USAGE_TASK_DRAIN_TIMEOUT_SECONDS = 0.25
RESOLUTION_TASK_CANCEL_DRAIN_TIMEOUT_SECONDS = 0.25
PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY = "_provider_call_credentials"

_RUNTIME_ISSUED_CREDENTIAL_TOKEN = object()
_SCOPED_ENDPOINT_FIELDS: dict[str, tuple[str, tuple[str, ...]]] = {
    "local-llm": ("local_llm", ("api_ip",)),
    "llama.cpp": ("llama_api", ("api_ip",)),
    "kobold": ("kobold_api", ("api_ip",)),
    "ooba": ("ooba_api", ("api_ip",)),
    "tabbyapi": ("tabby_api", ("api_ip",)),
    "vllm": ("vllm_api", ("api_ip",)),
    "ollama": ("ollama_api", ("api_url",)),
    "aphrodite": ("aphrodite_api", ("api_ip",)),
}


@dataclass(frozen=True, slots=True, repr=False)
class _ProviderCallTransportSnapshot:
    """Immutable key and exact endpoint capability issued for one call."""

    provider: str
    api_key: str | None
    trusted_endpoint: TrustedProviderEndpoint | None
    endpoint_provenance: str


def _trusted_endpoint_from_snapshot(
    provider: str,
    app_config: Mapping[str, Any] | None,
) -> TrustedProviderEndpoint | None:
    """Build an exact-origin capability solely from a captured config snapshot."""

    provider_norm = canonical_provider_name(provider)
    field_spec = _SCOPED_ENDPOINT_FIELDS.get(provider_norm)
    custom_number = custom_openai_provider_number(provider_norm)
    if field_spec is None and custom_number is not None:
        field_spec = (
            custom_openai_section_name(custom_number),
            ("api_ip", "api_base_url"),
        )
    if field_spec is None or not isinstance(app_config, Mapping):
        return None

    section_name, endpoint_fields = field_spec
    section = app_config.get(section_name)
    if not isinstance(section, Mapping):
        return None

    base_url = next(
        (
            candidate
            for field in endpoint_fields
            if (candidate := valid_provider_config_value(section.get(field)))
            is not None
        ),
        None,
    )
    if base_url is None:
        return None
    try:
        scope = ConfiguredEndpointScope.from_url(base_url)
    except ValueError:
        return None
    return TrustedProviderEndpoint(
        base_url=base_url.rstrip("/"),
        scope=scope,
    )


def _serialization_error() -> TypeError:
    """Build the stable error used for every serialization boundary."""
    return TypeError("ProviderCallCredentials cannot be serialized")


def _copy_error() -> TypeError:
    """Build the stable error used for copy and deepcopy boundaries."""
    return TypeError("ProviderCallCredentials cannot be copied")


def _reject_pydantic_serialization(_value: object) -> NoReturn:
    """Reject Pydantic serialization without inspecting credential contents."""
    raise _serialization_error()


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


def configured_provider_model_from_snapshot(
    provider: str,
    app_config: dict[str, Any] | None,
) -> str | None:
    """Read the frozen provider default model without consulting live state."""
    if not isinstance(app_config, dict):
        return None
    section = PROVIDER_APP_CONFIG_KEYS.get(provider)
    custom_number = custom_openai_provider_number(provider)
    if section is None and custom_number is not None:
        section = custom_openai_section_name(custom_number)
    if section is None:
        section = f"{provider.replace('.', '_').replace('-', '_')}_api"
    provider_config = app_config.get(section)
    if not isinstance(provider_config, dict):
        return None
    for key in ("model", "model_id", "model_path", "mlx_model_path"):
        value = provider_config.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


class ProviderCallCredentials:
    """Redacted, non-serializable credentials for one provider call."""

    _IMMUTABLE_SLOT_NAMES = frozenset(
        {
            "_transport_snapshot",
            "_issued_token",
            "_runtime_generation",
            "_runtime_identity",
            "_credential_identity",
        }
    )

    __slots__ = (
        "app_config",
        "auth_source",
        "_transport_snapshot",
        "_issued_token",
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
        trusted_endpoint: TrustedProviderEndpoint | None = None,
        _issued_token: object | None = None,
    ) -> None:
        self._transport_snapshot = _ProviderCallTransportSnapshot(
            provider=canonical_provider_name(provider),
            api_key=api_key,
            trusted_endpoint=trusted_endpoint,
            endpoint_provenance=endpoint_provenance,
        )
        self.app_config = app_config
        self.auth_source = auth_source
        self._issued_token = _issued_token
        self._runtime_generation = runtime_generation
        self._runtime_identity = runtime_identity
        self._credential_identity = credential_identity

    def __setattr__(self, name: str, value: object) -> None:
        if name in self._IMMUTABLE_SLOT_NAMES and hasattr(self, name):
            raise AttributeError("ProviderCallCredentials transport identity is immutable")
        object.__setattr__(self, name, value)

    @property
    def provider(self) -> str:
        """Return the canonical provider captured with the transport snapshot."""

        return self._transport_snapshot.provider

    @property
    def api_key(self) -> str | None:
        """Return the API key captured with the exact endpoint capability."""

        return self._transport_snapshot.api_key

    @property
    def trusted_endpoint(self) -> TrustedProviderEndpoint | None:
        """Return the frozen exact-origin endpoint capability, when applicable."""

        return self._transport_snapshot.trusted_endpoint

    @property
    def endpoint_provenance(self) -> str:
        """Return the bounded endpoint ownership class for observability."""

        return self._transport_snapshot.endpoint_provenance

    @property
    def credentials_resolved(self) -> bool:
        """Retain the transitional internal compatibility flag."""

        return True

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


def is_runtime_issued_provider_call_credentials(
    value: object,
    *,
    provider: str | None = None,
) -> bool:
    """Return whether *value* is an authentic runtime-issued credential handle."""

    if type(value) is not ProviderCallCredentials:
        return False
    if value._issued_token is not _RUNTIME_ISSUED_CREDENTIAL_TOKEN:
        return False
    if provider is None:
        return True
    try:
        return value.provider == canonical_provider_name(provider)
    except Exception:  # noqa: BLE001 - malformed provider values fail closed
        return False


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
    __slots__ = ("generation", "identity", "override_snapshot", "resolution", "used")

    def __init__(
        self,
        generation: int,
        resolution: ResolvedByokCredentials,
        override_snapshot: _ProviderOverrideSnapshot | None,
    ) -> None:
        self.generation = generation
        self.identity = object()
        self.override_snapshot = override_snapshot
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
        "_server_config_snapshot",
        "_uses_default_resolver",
        "_override_snapshot_resolver",
        "_resolver",
        "_identity",
        "_cache",
        "_issued_entries",
        "_inflight",
        "_refresh_tasks",
        "_usage_tasks",
        "_refresh_locks",
        "_refreshed_generations",
        "_generations",
        "_closed",
        "_closed_event",
        "_close_task",
    )

    def __init__(
        self,
        *,
        user_id: int | None,
        team_ids: Sequence[int] | None,
        org_ids: Sequence[int] | None,
        trusted_base_url_override: bool,
        fallback_resolver: _FallbackResolver | None = None,
        server_config_snapshot: Mapping[str, Any] | None = None,
        override_snapshot_resolver: _OverrideSnapshotResolver | None = None,
        resolver: _Resolver | None = None,
    ) -> None:
        uses_default_resolver = resolver is None or resolver is resolve_byok_credentials
        if uses_default_resolver and fallback_resolver is not None:
            raise ValueError(
                "Default credential resolution requires a frozen server config snapshot"
            )
        try:
            captured_server_config = copy.deepcopy(
                dict(
                    server_config_snapshot
                    if server_config_snapshot is not None
                    else load_server_config_snapshot()
                )
            )
        except Exception as exc:  # noqa: BLE001 - malformed snapshots fail closed
            del exc
            raise_detached_error(
                RuntimeError("Provider server configuration is unavailable")
            )
        self._user_id = user_id
        self._team_ids = list(team_ids or ())
        self._org_ids = list(org_ids or ())
        self._trusted_base_url_override = trusted_base_url_override is True
        self._fallback_resolver = fallback_resolver
        self._server_config_snapshot = captured_server_config
        self._uses_default_resolver = uses_default_resolver
        self._override_snapshot_resolver = override_snapshot_resolver
        self._resolver = resolver or resolve_byok_credentials
        self._identity = object()
        self._cache: dict[str, _ResolvedEntry] = {}
        self._issued_entries: dict[object, _ResolvedEntry] = {}
        self._inflight: dict[str, asyncio.Task[_ResolvedEntry]] = {}
        self._refresh_tasks: dict[str, asyncio.Task[_ResolvedEntry]] = {}
        self._usage_tasks: dict[_ResolvedEntry, asyncio.Task[bool]] = {}
        self._refresh_locks: dict[str, asyncio.Lock] = {}
        self._refreshed_generations: dict[str, int] = {}
        self._generations: dict[str, int] = {}
        self._closed = False
        self._closed_event = asyncio.Event()
        self._close_task: asyncio.Task[None] | None = None

    async def resolve(
        self,
        provider: str,
        *,
        model: str | None = None,
        force_refresh: bool = False,
    ) -> ProviderCallCredentials:
        """Return explicit credentials for a normalized provider."""
        self._ensure_open()
        provider_norm = canonical_provider_name(provider)
        if force_refresh:
            override_snapshot = self._capture_override_snapshot(provider_norm, model)
            return await self._refresh(provider_norm, model, override_snapshot)

        while True:
            self._ensure_open()
            override_snapshot = self._capture_override_snapshot(provider_norm, model)
            entry = self._cache.get(provider_norm)
            if entry is not None and entry.override_snapshot == override_snapshot:
                self._enforce_call_policy(
                    provider_norm,
                    model,
                    override_snapshot,
                    entry.resolution,
                )
                return self._new_handle(provider_norm, entry)

            task = self._refresh_tasks.get(provider_norm)
            if task is None:
                task = self._inflight.get(provider_norm)
            if task is None:
                if entry is None:
                    generation = self._generations.setdefault(provider_norm, 0)
                else:
                    generation = self._generations.get(provider_norm, 0) + 1
                    self._generations[provider_norm] = generation
                    self._cache.pop(provider_norm, None)
                task = asyncio.create_task(
                    self._resolve_entry(
                        provider_norm,
                        generation=generation,
                        force_refresh=False,
                        override_snapshot=override_snapshot,
                    )
                )
                self._inflight[provider_norm] = task
                task.add_done_callback(
                    lambda completed, name=provider_norm: self._forget_task(name, completed, refresh=False)
                )

            resolved_entry = await self._await_owned(task)
            if (
                self._cache.get(provider_norm) is resolved_entry
                and resolved_entry.override_snapshot == override_snapshot
            ):
                self._enforce_call_policy(
                    provider_norm,
                    model,
                    override_snapshot,
                    resolved_entry.resolution,
                )
                return self._new_handle(provider_norm, resolved_entry)

    async def mark_used(self, handle: ProviderCallCredentials) -> bool:
        """Record successful use and report whether persistence was confirmed."""
        self._ensure_open()
        if not is_runtime_issued_provider_call_credentials(handle):
            raise RuntimeError("Credential handle was not issued by this runtime")
        if handle._runtime_identity is not self._identity:
            raise RuntimeError("Credential handle was not issued by this runtime")

        entry = self._issued_entries.get(handle._credential_identity)
        if (
            entry is None
            or entry.generation != handle._runtime_generation
            or entry.identity is not handle._credential_identity
        ):
            return False
        if entry.used:
            return True

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

        persisted = await self._await_usage_task(task)
        if self._closed:
            raise RuntimeError("Provider credential runtime is closed")
        return persisted

    async def close(self) -> None:
        """Cancel owned work and release all execution-scoped references."""
        task = self._close_task
        if task is None:
            if self._closed:
                return
            self._closed = True
            self._closed_event.set()
            task = asyncio.create_task(self._close_owned())
            self._close_task = task
            task.add_done_callback(self._close_task_done)

        await asyncio.shield(task)

    async def _refresh(
        self,
        provider: str,
        model: str | None,
        override_snapshot: _ProviderOverrideSnapshot | None,
    ) -> ProviderCallCredentials:
        lock = self._refresh_locks.setdefault(provider, asyncio.Lock())
        async with lock:
            self._ensure_open()
            task = self._refresh_tasks.get(provider)
            if task is None:
                entry = self._cache.get(provider)
                if (
                    entry is not None
                    and entry.override_snapshot == override_snapshot
                    and self._refreshed_generations.get(provider) == entry.generation
                ):
                    self._enforce_call_policy(
                        provider,
                        model,
                        override_snapshot,
                        entry.resolution,
                    )
                    return self._new_handle(provider, entry)
                generation = self._generations.get(provider, 0) + 1
                self._generations[provider] = generation
                self._cache.pop(provider, None)
                if entry is not None:
                    self._issued_entries.pop(entry.identity, None)
                rejected_credential_generation = None
                if entry is not None:
                    try:
                        rejected_credential_generation = getattr(
                            entry.resolution,
                            "_credential_generation",
                            None,
                        )
                    except Exception:  # noqa: BLE001 - private metadata fails closed
                        raise RuntimeError(
                            "Provider credential generation is unavailable"
                        ) from None
                    if (
                        rejected_credential_generation is not None
                        and not isinstance(rejected_credential_generation, str)
                    ):
                        raise RuntimeError(
                            "Provider credential generation is unavailable"
                        )
                task = asyncio.create_task(
                    self._resolve_entry(
                        provider,
                        generation=generation,
                        force_refresh=True,
                        override_snapshot=override_snapshot,
                        rejected_credential_generation=rejected_credential_generation,
                    )
                )
                self._refresh_tasks[provider] = task
                task.add_done_callback(
                    lambda completed, name=provider: self._forget_task(name, completed, refresh=True)
                )

        resolved_entry = await self._await_owned(task)
        if (
            self._cache.get(provider) is resolved_entry
            and resolved_entry.override_snapshot == override_snapshot
        ):
            self._enforce_call_policy(
                provider,
                model,
                override_snapshot,
                resolved_entry.resolution,
            )
            return self._new_handle(provider, resolved_entry)
        return await self.resolve(provider, model=model, force_refresh=True)

    async def _resolve_entry(
        self,
        provider: str,
        *,
        generation: int,
        force_refresh: bool,
        override_snapshot: _ProviderOverrideSnapshot | None,
        rejected_credential_generation: str | None = None,
    ) -> _ResolvedEntry:
        resolution_failure: ByokResolutionError | None = None
        unexpected_failure = False
        resolution: ResolvedByokCredentials | None = None

        def captured_fallback(name: str) -> str | ServerFallbackCredentials | None:
            try:
                if canonical_provider_name(name) != provider:
                    raise ByokResolutionError("invalid_provider_credentials", provider)
                resolver = self._fallback_resolver
                base_fallback = (
                    resolver(name)
                    if resolver is not None
                    else resolve_static_server_fallback_from_snapshot(
                        name,
                        self._server_config_snapshot,
                    )
                )
                if override_snapshot is not None:
                    structured_base = (
                        ServerFallbackCredentials(
                            api_key=base_fallback,
                            credential_fields={},
                            app_config={},
                        )
                        if isinstance(base_fallback, str)
                        else base_fallback
                    )
                    override_fallback = override_snapshot.server_fallback(
                        structured_base
                        if isinstance(structured_base, ServerFallbackCredentials)
                        else None
                    )
                    if override_fallback is not None:
                        return override_fallback
                return base_fallback
            except ByokResolutionError:
                raise
            except Exception:  # noqa: BLE001 - fallback failures must be sanitized
                raise ByokResolutionError("invalid_provider_credentials", provider) from None

        try:
            resolver_kwargs: dict[str, Any] = {
                "user_id": self._user_id,
                "team_ids": list(self._team_ids),
                "org_ids": list(self._org_ids),
                "server_config_snapshot": copy.deepcopy(
                    self._server_config_snapshot
                ),
                "force_oauth_refresh": force_refresh,
                "trusted_base_url_override": self._trusted_base_url_override,
            }
            if self._uses_default_resolver:
                if override_snapshot is not None:
                    base_fallback = resolve_static_server_fallback_from_snapshot(
                        provider,
                        self._server_config_snapshot,
                    )
                    resolver_kwargs["fallback_override"] = (
                        override_snapshot.server_fallback(base_fallback)
                        or base_fallback
                    )
            elif (
                self._fallback_resolver is not None
                or override_snapshot is not None
            ):
                resolver_kwargs["fallback_resolver"] = captured_fallback
            if force_refresh and rejected_credential_generation is not None:
                resolver_kwargs["rejected_credential_generation"] = (
                    rejected_credential_generation
                )
            resolution = await self._resolver(
                provider,
                **resolver_kwargs,
            )
        except asyncio.CancelledError:
            raise
        except ByokResolutionError as exc:
            resolution_failure = exc
        except Exception:  # noqa: BLE001 - unexpected resolver failures are sanitized below
            unexpected_failure = True

        if resolution_failure is not None:
            raise_detached_error(resolution_failure)
        if unexpected_failure:
            raise RuntimeError("Provider credential resolution failed")
        if override_snapshot is not None:
            try:
                override_snapshot.ensure_healthy()
            except ByokResolutionError:
                raise
            except Exception:  # noqa: BLE001 - policy adapters fail closed
                raise RuntimeError("Provider policy resolution failed") from None
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

        entry = _ResolvedEntry(generation, resolution, override_snapshot)
        if self._generations.get(provider, 0) == generation and not self._closed:
            self._cache[provider] = entry
            if force_refresh:
                self._refreshed_generations[provider] = generation
        return entry

    async def _mark_entry_used(
        self,
        provider: str,
        entry: _ResolvedEntry,
    ) -> bool:
        """Persist usage before publishing the entry as used."""
        try:
            touch_callback = entry.resolution._touch_cb
            if touch_callback is not None:
                await touch_callback()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - usage persistence is best-effort and retryable
            return False
        if self._issued_entries.get(entry.identity) is entry:
            entry.used = True
        return True

    async def _await_usage_task(self, task: asyncio.Task[bool]) -> bool:
        """Bound normal waits and drain usage persistence before cancellation escapes."""
        try:
            done, _pending = await asyncio.wait(
                {task},
                timeout=max(0.0, float(USAGE_TASK_DRAIN_TIMEOUT_SECONDS)),
            )
            if task in done:
                return task.result()
            return False
        except asyncio.CancelledError:
            await self._drain_usage_tasks({task})
            if self._closed:
                raise RuntimeError("Provider credential runtime is closed") from None
            raise

    async def _drain_usage_tasks(self, tasks: set[asyncio.Task[bool]]) -> None:
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
        closed_waiter = asyncio.create_task(self._closed_event.wait())
        try:
            done, _pending = await asyncio.wait(
                {task, closed_waiter},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if self._closed or closed_waiter in done:
                raise RuntimeError("Provider credential runtime is closed")
            return task.result()
        except asyncio.CancelledError:
            if self._closed:
                raise RuntimeError("Provider credential runtime is closed") from None
            raise
        finally:
            if not closed_waiter.done():
                closed_waiter.cancel()
            await asyncio.gather(closed_waiter, return_exceptions=True)

    def _new_handle(
        self,
        provider: str,
        entry: _ResolvedEntry,
    ) -> ProviderCallCredentials:
        resolution = entry.resolution
        app_config = _copy_app_config(resolution.app_config)
        handle = ProviderCallCredentials(
            provider=provider,
            api_key=resolution.api_key,
            app_config=app_config,
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
            trusted_endpoint=_trusted_endpoint_from_snapshot(
                provider,
                app_config,
            ),
            _issued_token=_RUNTIME_ISSUED_CREDENTIAL_TOKEN,
        )
        self._issued_entries[entry.identity] = entry
        return handle

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
        task: asyncio.Task[bool],
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
        try:
            if cancellable_tasks:
                # Creation-time callbacks observe results that arrive after this deadline.
                await asyncio.wait(
                    cancellable_tasks,
                    timeout=max(
                        0.0,
                        float(RESOLUTION_TASK_CANCEL_DRAIN_TIMEOUT_SECONDS),
                    ),
                )
            if usage_tasks:
                await self._drain_usage_tasks(usage_tasks)
        finally:
            self._cache.clear()
            self._issued_entries.clear()
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
            self._fallback_resolver = None
            self._server_config_snapshot.clear()
            self._uses_default_resolver = True
            self._override_snapshot_resolver = None
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

    def _capture_override_snapshot(
        self,
        provider: str,
        model: str | None,
    ) -> _ProviderOverrideSnapshot | None:
        """Capture and enforce one provider policy snapshot, failing closed."""
        resolver = self._override_snapshot_resolver
        if resolver is None:
            return None
        try:
            snapshot = resolver(provider)
            snapshot.enforce(model)
            return snapshot
        except ByokResolutionError:
            raise
        except Exception:  # noqa: BLE001 - injected policy resolvers fail closed
            raise RuntimeError("Provider policy resolution failed") from None

    def _enforce_call_policy(
        self,
        provider: str,
        model: str | None,
        snapshot: _ProviderOverrideSnapshot | None,
        resolution: ResolvedByokCredentials,
    ) -> None:
        """Enforce policy against the explicit or frozen configured model per caller."""
        if snapshot is None:
            return
        effective_model = model or configured_provider_model_from_snapshot(
            provider,
            resolution.app_config,
        )
        try:
            snapshot.enforce(effective_model)
            snapshot.ensure_healthy()
        except ByokResolutionError:
            raise
        except Exception:  # noqa: BLE001 - policy adapters fail closed
            raise RuntimeError("Provider policy resolution failed") from None


async def mark_provider_credential_used(
    credential_runtime: ProviderCredentialRuntime,
    credentials: ProviderCallCredentials,
) -> bool:
    """Persist one credential touch, retrying one explicit in-flight failure."""

    persisted = await credential_runtime.mark_used(credentials)
    if persisted is False:
        # An explicit False can mean the runtime's bounded persistence task is
        # still being released. Yield once, then rejoin that request-owned work.
        await asyncio.sleep(0)
        persisted = await credential_runtime.mark_used(credentials)
    return persisted is not False
