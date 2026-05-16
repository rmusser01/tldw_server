"""Supervisor for managed llama.cpp runtime profiles."""

from __future__ import annotations

import asyncio
import weakref
from ipaddress import ip_address
from pathlib import Path
from typing import Any, Callable, Protocol
from uuid import uuid4

from tldw_Server_API.app.core.Local_LLM import handler_utils, llamacpp_inventory_service
from tldw_Server_API.app.core.Local_LLM.LLM_Inference_Exceptions import ServerError
from tldw_Server_API.app.core.Local_LLM.LLM_Inference_Schemas import LlamaCppConfig
from tldw_Server_API.app.core.Local_LLM.llamacpp_process_runner import LlamaCppProcessRunner
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
            model_path = self._resolve_profile_model_path(profile)
            return await runner.start(model_path, profile)

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
        if self.store.get(profile_id) is not None:
            return LlamaCppRuntime(profile_id=profile_id, state=LlamaCppRuntimeState.DEFINED)
        raise LlamaCppProfileNotFoundError(f"Llama.cpp profile '{profile_id}' was not found.")

    def tail_logs(self, profile_id: str, lines: int) -> dict[str, object]:
        self._require_profile(profile_id)
        runner = self._runners.get(profile_id)
        if runner is None:
            return {"lines": [], "truncated": False, "warnings": ["No active managed llama.cpp log file is available."]}
        return runner.tail_logs(lines)

    async def shutdown(self) -> None:
        for profile_id in list(self._runners):
            async with self._lock_for(profile_id):
                runner = self._runners.get(profile_id)
                if runner is not None:
                    await runner.stop()

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

    def _resolve_profile_model_path(self, profile: LlamaCppProfile) -> Path:
        if profile.model_path:
            return Path(profile.model_path)
        if profile.model_id:
            return llamacpp_inventory_service.resolve_model_id(profile.model_id)
        raise ServerError(f"Llama.cpp profile '{profile.profile_id}' does not have a model.")

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


__all__ = ["LlamaCppSupervisor"]
