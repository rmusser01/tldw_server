"""Supervisor for managed llama.cpp runtime profiles."""

from __future__ import annotations

import asyncio
import weakref
from datetime import UTC, datetime
from ipaddress import ip_address
from pathlib import Path
from typing import Any, Callable, Protocol
from uuid import uuid4

from tldw_Server_API.app.core.Local_LLM import handler_utils, llamacpp_inventory_service
from tldw_Server_API.app.core.Local_LLM.LLM_Inference_Exceptions import ServerError
from tldw_Server_API.app.core.Local_LLM.LLM_Inference_Schemas import LlamaCppConfig
from tldw_Server_API.app.core.Local_LLM.llamacpp_process_runner import (
    LlamaCppProcessRunner,
    validate_profile_server_args,
)
from tldw_Server_API.app.core.Local_LLM.llamacpp_profile_capabilities import (
    LlamaCppResolvedProfileLaunch,
    resolve_profile_launch,
)
from tldw_Server_API.app.core.Local_LLM.llamacpp_profile_store import (
    DEFAULT_PROFILE_ID,
    DEFAULT_PROFILE_NAME,
    JsonLlamaCppProfileStore,
    default_profile_store_path,
)
from tldw_Server_API.app.core.Local_LLM.llamacpp_runtime_models import (
    LlamaCppPortPolicy,
    LlamaCppProfile,
    LlamaCppProfileMode,
    LlamaCppProfileConflictError,
    LlamaCppProfileNotFoundError,
    LlamaCppRuntime,
    LlamaCppRuntimeState,
)

RunnerFactory = Callable[[LlamaCppConfig, str], Any]

_MAX_FAILURE_TEXT_LENGTH = 500


class LlamaCppProfileCreateInput(Protocol):
    """Structural input for creating a managed llama.cpp profile."""

    profile_id: str | None
    name: str
    enabled: bool
    mode: LlamaCppProfileMode
    model_id: str | None
    model_path: str | None
    mmproj_model_id: str | None
    host: str
    port: int
    port_policy: LlamaCppPortPolicy
    server_args: dict[str, object]
    autostart: bool
    restart_policy: dict[str, object]
    provider_alias: str | None
    tags: list[str]


class LlamaCppProfileUpdateInput(Protocol):
    """Structural input for partially updating a managed llama.cpp profile."""

    model_fields_set: set[str]


def _normalize_host_for_conflict(host: str | None) -> str:
    return handler_utils.strip_host_brackets(str(host or "127.0.0.1")).strip().lower()


def _is_wildcard_host(host: str) -> bool:
    try:
        return ip_address(host).is_unspecified
    except ValueError:
        return host in {"*", "all"}


def _hosts_conflict(first: str, second: str) -> bool:
    return first == second or _is_wildcard_host(first) or _is_wildcard_host(second)


class LlamaCppSupervisor:
    """Coordinate llama.cpp profiles and their owned process runners."""

    def __init__(
        self,
        *,
        config: LlamaCppConfig,
        store: JsonLlamaCppProfileStore,
        runner_factory: RunnerFactory = LlamaCppProcessRunner,
    ):
        self.config = config
        self.store = store
        self.runner_factory = runner_factory
        self._runners: dict[str, Any] = {}
        self._locks: weakref.WeakValueDictionary[str, asyncio.Lock] = weakref.WeakValueDictionary()
        self._store_lock = asyncio.Lock()
        self._start_lock = asyncio.Lock()
        self._paused: set[str] = set()

    @classmethod
    def from_manager(cls, manager: Any) -> "LlamaCppSupervisor":
        config = getattr(getattr(manager, "config", None), "llamacpp", None)
        if config is None:
            raise ServerError("Llama.cpp config is not available.")
        return cls(config=config, store=JsonLlamaCppProfileStore(default_profile_store_path()))

    def list_profiles(self) -> list[LlamaCppProfile]:
        return self.store.list_profiles()

    async def create_profile(self, request: LlamaCppProfileCreateInput) -> LlamaCppProfile:
        profile_id = request.profile_id or uuid4().hex
        async with self._lock_for(profile_id):
            return await self._create_profile_unlocked(profile_id, request)

    async def _create_profile_unlocked(
        self,
        profile_id: str,
        request: LlamaCppProfileCreateInput,
    ) -> LlamaCppProfile:
        profile = LlamaCppProfile(
            profile_id=profile_id,
            name=request.name,
            enabled=request.enabled,
            mode=request.mode,
            model_id=request.model_id,
            model_path=request.model_path,
            mmproj_model_id=request.mmproj_model_id,
            host=request.host,
            port=request.port,
            port_policy=request.port_policy,
            server_args=dict(request.server_args),
            autostart=request.autostart,
            restart_policy=dict(request.restart_policy),
            provider_alias=request.provider_alias,
            tags=list(request.tags),
        )
        self._validate_profile_launch_definition(profile)
        self._validate_runtime_port_available(profile)
        return await self._store_upsert(profile)

    async def update_profile(self, profile_id: str, request: LlamaCppProfileUpdateInput) -> LlamaCppProfile:
        async with self._lock_for(profile_id):
            return await self._update_profile_unlocked(profile_id, request)

    async def _update_profile_unlocked(
        self,
        profile_id: str,
        request: LlamaCppProfileUpdateInput,
    ) -> LlamaCppProfile:
        existing = self._require_profile(profile_id)
        updates = {field: getattr(request, field) for field in request.model_fields_set}
        if "server_args" in updates and updates["server_args"] is not None:
            updates["server_args"] = dict(updates["server_args"])
        if "restart_policy" in updates and updates["restart_policy"] is not None:
            updates["restart_policy"] = dict(updates["restart_policy"])
        if "tags" in updates and updates["tags"] is not None:
            updates["tags"] = list(updates["tags"])
        profile = LlamaCppProfile.model_validate(existing.model_dump(mode="python") | updates)
        self._validate_profile_launch_definition(profile)
        self._validate_runtime_port_available(profile)
        return await self._store_upsert(profile)

    async def delete_profile(self, profile_id: str) -> bool:
        async with self._lock_for(profile_id):
            return await self._delete_profile_unlocked(profile_id)

    async def _delete_profile_unlocked(self, profile_id: str) -> bool:
        runner = self._runners.get(profile_id)
        if runner is not None:
            await runner.stop()
            self._runners.pop(profile_id, None)
        self._paused.discard(profile_id)
        return await self._store_delete(profile_id)

    async def start_profile(self, profile_id: str) -> LlamaCppRuntime:
        async with self._lock_for(profile_id):
            return await self._start_profile_unlocked(profile_id)

    async def _start_profile_unlocked(self, profile_id: str, *, restart: bool = False) -> LlamaCppRuntime:
        profile = self._require_profile(profile_id)
        if not profile.enabled:
            profile = await self._store_upsert(profile.model_copy(update={"enabled": True}))
        self._paused.discard(profile_id)
        runner = self._runner_for(profile_id)
        async with self._start_lock:
            status = runner.status()
            if status.state == LlamaCppRuntimeState.RUNNING:
                if not restart:
                    return status
                await runner.stop()
            self._validate_runtime_port_available(profile)
            resolved = self._validate_profile_launch_definition(profile)
            launch_profile = profile.model_copy(update={"server_args": resolved.server_args})
            runtime = await runner.start(resolved.model_path, launch_profile)
            if profile.last_runtime_failure:
                await self._store_upsert(profile.model_copy(update={"last_runtime_failure": {}}))
            return runtime

    async def stop_profile(self, profile_id: str, disable: bool = False) -> LlamaCppRuntime:
        async with self._lock_for(profile_id):
            profile = self._require_profile(profile_id)
            if disable and profile.enabled:
                await self._store_upsert(profile.model_copy(update={"enabled": False}))
            self._paused.discard(profile_id)
            runner = self._runners.get(profile_id)
            if runner is None:
                return LlamaCppRuntime(profile_id=profile_id, state=LlamaCppRuntimeState.STOPPED, message="Stopped")
            return await runner.stop()

    async def pause_profile(self, profile_id: str) -> LlamaCppRuntime:
        async with self._lock_for(profile_id):
            profile = self._require_profile(profile_id)
            if profile.enabled:
                await self._store_upsert(profile.model_copy(update={"enabled": False}))
            runner = self._runners.get(profile_id)
            if runner is not None and runner.status().state == LlamaCppRuntimeState.RUNNING:
                await runner.stop()
            self._paused.add(profile_id)
            return LlamaCppRuntime(profile_id=profile_id, state=LlamaCppRuntimeState.PAUSED, message="Paused")

    async def resume_profile(self, profile_id: str) -> LlamaCppRuntime:
        async with self._lock_for(profile_id):
            profile = self._require_profile(profile_id)
            if not profile.enabled:
                await self._store_upsert(profile.model_copy(update={"enabled": True}))
            self._paused.discard(profile_id)
            return await self._start_profile_unlocked(profile_id)

    def list_runtimes(self) -> list[LlamaCppRuntime]:
        return [self.get_runtime(profile.profile_id) for profile in self.list_profiles()]

    def get_runtime(self, profile_id: str) -> LlamaCppRuntime:
        if profile_id in self._paused:
            return LlamaCppRuntime(profile_id=profile_id, state=LlamaCppRuntimeState.PAUSED, message="Paused")
        runner = self._runners.get(profile_id)
        if runner is not None:
            return runner.status()
        profile = self.store.get(profile_id)
        if profile is not None:
            if profile.last_runtime_failure:
                return self.runtime_from_last_failure(profile)
            return LlamaCppRuntime(profile_id=profile_id, state=LlamaCppRuntimeState.DEFINED)
        raise LlamaCppProfileNotFoundError(f"Llama.cpp profile '{profile_id}' was not found.")

    def is_profile_paused(self, profile_id: str) -> bool:
        """Return True when a profile is paused in the current process."""
        return profile_id in self._paused

    async def record_runtime_failure(
        self,
        profile_id: str,
        *,
        runtime: LlamaCppRuntime | None = None,
        error: BaseException | None = None,
        restart_count: int | None = None,
    ) -> LlamaCppRuntime:
        """Persist bounded failure metadata for one managed runtime profile."""
        async with self._lock_for(profile_id):
            profile = self._require_profile(profile_id)
            metadata = self._failure_metadata(
                profile=profile,
                runtime=runtime,
                error=error,
                restart_count=restart_count,
            )
            updated = await self._store_upsert(profile.model_copy(update={"last_runtime_failure": metadata}))
            return self.runtime_from_last_failure(updated)

    async def clear_runtime_failure(self, profile_id: str) -> LlamaCppProfile:
        """Clear durable failure metadata for one managed runtime profile."""
        async with self._lock_for(profile_id):
            profile = self._require_profile(profile_id)
            if not profile.last_runtime_failure:
                return profile
            return await self._store_upsert(profile.model_copy(update={"last_runtime_failure": {}}))

    def runtime_from_last_failure(self, profile: LlamaCppProfile) -> LlamaCppRuntime:
        """Build runtime status from a profile's durable failure metadata."""
        failure = profile.last_runtime_failure
        return LlamaCppRuntime(
            profile_id=profile.profile_id,
            state=LlamaCppRuntimeState.FAILED,
            host=profile.host,
            port=profile.port,
            endpoint=f"http://{profile.host}:{profile.port}" if profile.host and profile.port else None,
            model_id=profile.model_id,
            model_path=_str_or_none(failure.get("model_path")) or profile.model_path,
            stopped_at=_str_or_none(failure.get("stopped_at")),
            restart_count=_non_negative_int(failure.get("restart_count")),
            exit_code=_optional_int(failure.get("exit_code")),
            last_error=_str_or_none(failure.get("last_error")),
            health={"ready": False},
            message=_str_or_none(failure.get("last_error")) or "Last llama.cpp start failed.",
        )

    def tail_logs(self, profile_id: str, lines: int) -> dict[str, object]:
        self._require_profile(profile_id)
        runner = self._runners.get(profile_id)
        if runner is None:
            return {"lines": [], "truncated": False, "warnings": ["No active managed llama.cpp log file is available."]}
        return runner.tail_logs(lines)

    async def shutdown(self) -> None:
        failures: list[tuple[str, BaseException]] = []
        for profile_id in list(self._runners):
            try:
                async with self._lock_for(profile_id):
                    runner = self._runners.get(profile_id)
                    if runner is not None:
                        await runner.stop()
            except Exception as exc:  # noqa: BLE001 - shutdown should attempt every owned runner.
                failures.append((profile_id, exc))
        if failures:
            failed_ids = ", ".join(profile_id for profile_id, _exc in failures)
            raise RuntimeError(f"Failed to stop llama.cpp runner(s): {failed_ids}") from failures[0][1]

    def cleanup_sync(self) -> None:
        for runner in list(self._runners.values()):
            runner.cleanup_sync()

    async def ensure_default_profile_from_model(
        self,
        model_id: str,
        server_args: dict[str, object],
    ) -> LlamaCppProfile:
        async with self._lock_for(DEFAULT_PROFILE_ID):
            return await self._ensure_default_profile_from_model_unlocked(model_id, server_args)

    async def _ensure_default_profile_from_model_unlocked(
        self,
        model_id: str,
        server_args: dict[str, object],
    ) -> LlamaCppProfile:
        model_path = llamacpp_inventory_service.resolve_model_id(model_id)
        existing = self.store.get(DEFAULT_PROFILE_ID)
        host, port = self._resolve_default_host_port(server_args)
        if existing is None:
            profile = LlamaCppProfile(
                profile_id=DEFAULT_PROFILE_ID,
                name=DEFAULT_PROFILE_NAME,
                enabled=True,
                model_id=model_id,
                model_path=str(model_path),
                host=host,
                port=port,
                port_policy=LlamaCppPortPolicy.EXPLICIT,
                server_args=dict(server_args),
            )
        else:
            profile = existing.model_copy(
                update={
                    "enabled": True,
                    "model_id": model_id,
                    "model_path": str(model_path),
                    "host": host,
                    "port": port,
                    "port_policy": LlamaCppPortPolicy.EXPLICIT,
                    "server_args": dict(server_args),
                }
            )
        self._validate_profile_launch_definition(profile)
        self._validate_runtime_port_available(profile)
        return await self._store_upsert(profile)

    async def ensure_default_profile_from_path(
        self,
        model_path: Path,
        server_args: dict[str, object],
        *,
        model_label: str | None = None,
    ) -> LlamaCppProfile:
        async with self._lock_for(DEFAULT_PROFILE_ID):
            return await self._ensure_default_profile_from_path_unlocked(
                model_path,
                server_args,
                model_label=model_label,
            )

    async def _ensure_default_profile_from_path_unlocked(
        self,
        model_path: Path,
        server_args: dict[str, object],
        *,
        model_label: str | None = None,
    ) -> LlamaCppProfile:
        _ = model_label
        try:
            resolved_model_path = Path(model_path).expanduser().resolve()
        except (OSError, RuntimeError, ValueError) as exc:
            raise ServerError("Model path could not be resolved.") from exc
        existing = self.store.get(DEFAULT_PROFILE_ID)
        host, port = self._resolve_default_host_port(server_args)
        if existing is None:
            profile = LlamaCppProfile(
                profile_id=DEFAULT_PROFILE_ID,
                name=DEFAULT_PROFILE_NAME,
                enabled=True,
                model_id=None,
                model_path=str(resolved_model_path),
                host=host,
                port=port,
                port_policy=LlamaCppPortPolicy.EXPLICIT,
                server_args=dict(server_args),
            )
        else:
            profile = existing.model_copy(
                update={
                    "enabled": True,
                    "model_id": None,
                    "model_path": str(resolved_model_path),
                    "host": host,
                    "port": port,
                    "port_policy": LlamaCppPortPolicy.EXPLICIT,
                    "server_args": dict(server_args),
                }
            )
        self._validate_profile_launch_definition(profile)
        self._validate_runtime_port_available(profile)
        return await self._store_upsert(profile)

    async def start_default_by_model(self, model_id: str, server_args: dict[str, object]) -> LlamaCppRuntime:
        async with self._lock_for(DEFAULT_PROFILE_ID):
            profile = await self._ensure_default_profile_from_model_unlocked(model_id, server_args)
            return await self._start_profile_unlocked(profile.profile_id, restart=True)

    async def start_default_by_path(
        self,
        model_path: Path,
        server_args: dict[str, object],
        *,
        model_label: str | None = None,
    ) -> LlamaCppRuntime:
        async with self._lock_for(DEFAULT_PROFILE_ID):
            profile = await self._ensure_default_profile_from_path_unlocked(
                model_path,
                server_args,
                model_label=model_label,
            )
            return await self._start_profile_unlocked(profile.profile_id, restart=True)

    async def stop_default(self) -> LlamaCppRuntime:
        return await self.stop_profile(DEFAULT_PROFILE_ID)

    def default_status_compat(self) -> dict[str, object]:
        try:
            runtime = self.get_runtime(DEFAULT_PROFILE_ID)
        except LlamaCppProfileNotFoundError:
            runtime = LlamaCppRuntime(profile_id=DEFAULT_PROFILE_ID, state=LlamaCppRuntimeState.STOPPED)
        status = "running" if runtime.state == LlamaCppRuntimeState.RUNNING else runtime.state.value
        return {
            "status": status,
            "backend": "llamacpp",
            "model": runtime.model_path,
            "path": runtime.model_path,
            "host": runtime.host,
            "port": runtime.port,
            "pid": runtime.pid,
            "message": runtime.message,
        }

    def _lock_for(self, profile_id: str) -> asyncio.Lock:
        lock = self._locks.get(profile_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[profile_id] = lock
        return lock

    def _runner_for(self, profile_id: str) -> Any:
        runner = self._runners.get(profile_id)
        if runner is None:
            runner = self.runner_factory(self.config, profile_id)
            self._runners[profile_id] = runner
        return runner

    async def _store_upsert(self, profile: LlamaCppProfile) -> LlamaCppProfile:
        async with self._store_lock:
            return await asyncio.to_thread(self.store.upsert, profile)

    async def _store_delete(self, profile_id: str) -> bool:
        async with self._store_lock:
            return await asyncio.to_thread(self.store.delete, profile_id)

    def _resolve_default_host_port(self, server_args: dict[str, object]) -> tuple[str, int]:
        host = str(server_args.get("host") or self.config.default_host or "127.0.0.1")
        raw_port = server_args.get("port")
        if raw_port is None or raw_port == "":
            raw_port = self.config.default_port
        if raw_port is None or raw_port == "":
            raise ServerError("Llama.cpp default port is not configured.")
        try:
            return host, int(raw_port)
        except (TypeError, ValueError) as exc:
            raise ServerError("Llama.cpp default port must be an integer.") from exc

    def _require_profile(self, profile_id: str) -> LlamaCppProfile:
        profile = self.store.get(profile_id)
        if profile is None:
            raise LlamaCppProfileNotFoundError(f"Llama.cpp profile '{profile_id}' was not found.")
        return profile

    def _profile_for_launch_resolution(self, profile: LlamaCppProfile) -> LlamaCppProfile:
        """Return the profile shape used only for launch-time resolution.

        The default-profile compatibility path persists both `model_id` and a
        resolved `model_path` after `/start-by-model`. Clearing `model_id` here
        keeps start-by-model/start-by-path bridge launches path-based without
        mutating the stored profile. Other profiles keep their original asset
        selection so invalid IDs still fail before runner spawn.
        """
        if profile.profile_id == DEFAULT_PROFILE_ID and profile.model_path:
            return profile.model_copy(update={"model_id": None})
        return profile

    def _validate_profile_launch_definition(self, profile: LlamaCppProfile) -> LlamaCppResolvedProfileLaunch:
        """Resolve and validate launch assets and server args for a profile."""
        resolved = resolve_profile_launch(
            self._profile_for_launch_resolution(profile),
            path_resolver=self._resolve_launch_asset_path,
        )
        validate_profile_server_args(
            self.config,
            profile.model_copy(update={"server_args": resolved.server_args}),
            allowed_structured_args=self._allowed_structured_server_args(profile),
        )
        return resolved

    @staticmethod
    def _allowed_structured_server_args(profile: LlamaCppProfile) -> set[str]:
        if profile.profile_id == DEFAULT_PROFILE_ID:
            return {"host", "port"}
        return set()

    def _resolve_launch_asset_path(self, raw_path: str | Path, expected_kind: str, label: str) -> Path:
        """Resolve a launch asset path using the inventory service contract.

        The inventory service reads the current saved llama.cpp config for
        canonicalization, asset-kind validation, and allowlist enforcement. This
        avoids stale supervisor config snapshots and preserves the standard
        `ModelNotFoundError`/`ServerError` behavior used by Admin endpoints.
        """
        return llamacpp_inventory_service.resolve_asset_path(
            raw_path,
            expected_kind=expected_kind,
            label=label,
        )

    def _validate_runtime_port_available(self, profile: LlamaCppProfile) -> None:
        if not profile.enabled or profile.port_policy != LlamaCppPortPolicy.EXPLICIT:
            return
        host = _normalize_host_for_conflict(profile.host)
        for other_id, runner in self._runners.items():
            if other_id == profile.profile_id:
                continue
            runtime = runner.status()
            if runtime.state != LlamaCppRuntimeState.RUNNING or runtime.port != profile.port:
                continue
            runtime_host = _normalize_host_for_conflict(runtime.host)
            if _hosts_conflict(host, runtime_host):
                raise LlamaCppProfileConflictError(
                    f"Running llama.cpp profile '{other_id}' already uses {runtime.host}:{runtime.port}."
                )

    def _failure_metadata(
        self,
        *,
        profile: LlamaCppProfile,
        runtime: LlamaCppRuntime | None,
        error: BaseException | None,
        restart_count: int | None,
    ) -> dict[str, object]:
        existing = profile.last_runtime_failure
        last_error = (
            (runtime.last_error if runtime is not None else None)
            or (str(error) if error is not None else None)
            or "Llama.cpp runtime failed."
        )
        metadata: dict[str, object] = {
            "state": LlamaCppRuntimeState.FAILED.value,
            "last_error": last_error[:_MAX_FAILURE_TEXT_LENGTH],
            "restart_count": _non_negative_int(
                restart_count if restart_count is not None else _non_negative_int(existing.get("restart_count")) + 1
            ),
            "recorded_at": datetime.now(UTC).isoformat(),
        }
        if runtime is not None:
            if runtime.exit_code is not None:
                metadata["exit_code"] = runtime.exit_code
            if runtime.stopped_at:
                metadata["stopped_at"] = runtime.stopped_at
            if runtime.model_path:
                metadata["model_path"] = runtime.model_path
        return metadata


__all__ = ["LlamaCppSupervisor"]


def _non_negative_int(value: object) -> int:
    """Coerce a value to a non-negative integer, defaulting invalid values to zero."""
    try:
        parsed = int(value) if value is not None else 0
    except (TypeError, ValueError):
        return 0
    return max(0, parsed)


def _optional_int(value: object) -> int | None:
    """Coerce a value to int when possible, preserving missing or invalid values as None."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _str_or_none(value: object) -> str | None:
    """Coerce a non-empty value to string while keeping missing values as None."""
    if value in (None, ""):
        return None
    return str(value)
