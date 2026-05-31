from __future__ import annotations

import asyncio
import contextlib
import os
import secrets
import sys
import time
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

from tldw_Server_API.app.core.Local_LLM.handler_utils import build_base_url, is_port_free, resolve_client_host

from .omnivoice_sidecar_protocol import X_TLDW_SIDECAR_TOKEN_HEADER, build_sidecar_auth_headers
from .omnivoice_sidecar_server import validate_loopback_host


_READY_OR_REACHABLE_HEALTH_STATUSES = {
    "idle_stopped",
    "model_unavailable",
    "runtime_missing",
    "degraded",
}


def create_sidecar_async_client(*, timeout: float | None = None) -> httpx.AsyncClient:
    """Create an httpx client dedicated to loopback sidecar traffic."""
    kwargs: dict[str, Any] = {"trust_env": False}
    if timeout is not None:
        kwargs["timeout"] = timeout
    return httpx.AsyncClient(**kwargs)


class OmniVoiceSidecarSupervisor:
    """Own the lifecycle of the process-local OmniVoice sidecar."""

    def __init__(self, provider_config: dict[str, Any], repo_root: Path | None = None) -> None:
        self._provider_config = dict(provider_config or {})
        self._extra_params = dict(self._provider_config.get("extra_params") or {})
        self._repo_root = Path(repo_root or Path.cwd()).resolve()
        self._host = validate_loopback_host(self._extra_params.get("host"))
        self._start_port = int(self._coalesce_extra_param("port", 8039))
        self._autoselect_port = bool(self._extra_params.get("autoselect_port", True))
        self._port_probe_max = max(0, int(self._coalesce_extra_param("port_probe_max", 10)))
        self._healthcheck_timeout_seconds = float(self._coalesce_extra_param("healthcheck_timeout_seconds", 10.0))
        self._healthcheck_interval_seconds = float(self._coalesce_extra_param("healthcheck_interval_seconds", 0.25))
        self._startup_backoff_seconds = float(self._coalesce_extra_param("startup_backoff_seconds", 5.0))
        self._idle_shutdown_seconds = float(self._coalesce_extra_param("idle_shutdown_seconds", 900.0))
        self._model = self._resolve_non_empty_value(self._provider_config.get("model"))
        self._model_path = self._resolve_optional_path(self._extra_params.get("model_path"))
        self._runtime_path = self._resolve_optional_path(self._extra_params.get("runtime_path"))
        scratch_dir = self._resolve_optional_path(self._extra_params.get("scratch_dir"))
        if scratch_dir is None and self._runtime_path is not None:
            scratch_dir = self._runtime_path / "scratch"
        self._scratch_dir = scratch_dir
        self._device_map = self._resolve_non_empty_value(self._extra_params.get("device_map"))
        self._dtype = self._resolve_non_empty_value(self._extra_params.get("dtype"))
        self._closing = False
        self._token = secrets.token_urlsafe(32)
        self._client: httpx.AsyncClient | None = None
        self._process: asyncio.subprocess.Process | None = None
        self._port: int | None = None
        self._base_url: str | None = None
        self._last_failure_at: float | None = None
        self._last_activity_at: float | None = None
        self._lock = asyncio.Lock()

    def _coalesce_extra_param(self, key: str, default: Any) -> Any:
        value = self._extra_params.get(key)
        return default if value is None else value

    @staticmethod
    def _resolve_non_empty_value(value: Any) -> str | None:
        if value is None:
            return None
        resolved = str(value).strip()
        return resolved or None

    def _resolve_optional_path(self, value: Any) -> Path | None:
        resolved = self._resolve_non_empty_value(value)
        if resolved is None:
            return None
        path = Path(resolved).expanduser()
        if not path.is_absolute():
            path = self._repo_root / path
        return path.resolve()

    def _resolve_interpreter(self) -> str:
        configured = self._extra_params.get("python_path") or self._extra_params.get("interpreter_path")
        if configured:
            candidate = Path(str(configured)).expanduser()
            if not candidate.is_absolute():
                candidate = (self._repo_root / candidate).resolve()
            if candidate.exists():
                return str(candidate)
            logger.warning("Configured OmniVoice interpreter does not exist: {}", candidate)
        return sys.executable

    @property
    def sidecar_token(self) -> str:
        return self._token

    @property
    def port(self) -> int | None:
        return self._port

    @property
    def last_failure_at(self) -> float | None:
        return self._last_failure_at

    def mark_closing(self) -> None:
        self._closing = True

    async def ensure_started(self) -> str:
        if self._closing:
            raise RuntimeError("OmniVoice sidecar supervisor is closing")

        async with self._lock:
            if self._closing:
                raise RuntimeError("OmniVoice sidecar supervisor is closing")
            if self._is_process_running() and self._base_url:
                if await self._shutdown_if_idle_locked():
                    logger.debug("Restarting OmniVoice sidecar after idle timeout")
                else:
                    self._last_activity_at = time.time()
                    return self._base_url
            if self._last_failure_at is not None and (time.time() - self._last_failure_at) < self._startup_backoff_seconds:
                raise RuntimeError("OmniVoice sidecar startup is backing off after a recent failure")

            self._rotate_token()
            selected_port = self._select_port()
            self._process = await self._spawn_sidecar(selected_port)
            self._port = selected_port
            self._base_url = build_base_url(resolve_client_host(self._host), selected_port)

            try:
                await self._wait_for_ready()
            except Exception as exc:
                self._record_failure()
                await self._stop_process_locked()
                raise RuntimeError("OmniVoice sidecar did not reach /health") from exc

            self._last_activity_at = time.time()
            return self._base_url

    async def get_http_client(self, *, timeout: float | None = None) -> httpx.AsyncClient:
        if self._client is None or self._client_is_closed(self._client):
            self._client = create_sidecar_async_client(timeout=timeout)
        return self._client

    async def shutdown(self) -> None:
        self._closing = True
        async with self._lock:
            await self._stop_process_locked()
            client = self._client
            self._client = None
        if client is not None and not self._client_is_closed(client):
            await client.aclose()

    @staticmethod
    def _client_is_closed(client: httpx.AsyncClient) -> bool:
        return bool(getattr(client, "is_closed", False))

    def _is_process_running(self) -> bool:
        return self._process is not None and self._process.returncode is None

    def _is_idle(self) -> bool:
        if self._idle_shutdown_seconds <= 0:
            return False
        if self._last_activity_at is None:
            return False
        return (time.time() - self._last_activity_at) >= self._idle_shutdown_seconds

    def _rotate_token(self) -> None:
        self._token = secrets.token_urlsafe(32)

    def _build_subprocess_env(self, *, port: int) -> dict[str, str]:
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH")
        repo_pythonpath = str(self._repo_root)
        env["PYTHONPATH"] = (
            os.pathsep.join([repo_pythonpath, existing_pythonpath])
            if existing_pythonpath
            else repo_pythonpath
        )
        env.update(
            {
                "OMNIVOICE_SIDECAR_TOKEN": self._token,
                "OMNIVOICE_SIDECAR_HOST": self._host,
                "OMNIVOICE_SIDECAR_PORT": str(port),
            }
        )
        optional_env_values = {
            "OMNIVOICE_MODEL": self._model,
            "OMNIVOICE_MODEL_PATH": str(self._model_path) if self._model_path is not None else None,
            "OMNIVOICE_RUNTIME_PATH": str(self._runtime_path) if self._runtime_path is not None else None,
            "OMNIVOICE_SCRATCH_DIR": str(self._scratch_dir) if self._scratch_dir is not None else None,
            "OMNIVOICE_DEVICE_MAP": self._device_map,
            "OMNIVOICE_DTYPE": self._dtype,
        }
        env.update({key: value for key, value in optional_env_values.items() if value})
        return env

    def _prepare_runtime_directories(self) -> None:
        for directory in (self._runtime_path, self._scratch_dir):
            if directory is not None:
                directory.mkdir(parents=True, exist_ok=True)

    async def shutdown_if_idle(self) -> bool:
        async with self._lock:
            return await self._shutdown_if_idle_locked()

    async def _shutdown_if_idle_locked(self) -> bool:
        if self._idle_shutdown_seconds <= 0:
            return False
        if self._last_activity_at is None:
            return False
        if (time.time() - self._last_activity_at) < self._idle_shutdown_seconds:
            return False
        if self._process is None or self._process.returncode is not None:
            self._last_activity_at = None
            return False
        await self._stop_process_locked()
        self._last_activity_at = None
        return True

    def _record_failure(self) -> None:
        self._last_failure_at = time.time()

    def _select_port(self) -> int:
        if not self._autoselect_port:
            if not is_port_free(self._host, self._start_port):
                raise RuntimeError(f"OmniVoice sidecar port {self._start_port} is unavailable")
            return self._start_port

        for offset in range(self._port_probe_max + 1):
            candidate = self._start_port + offset
            if is_port_free(self._host, candidate):
                return candidate

        raise RuntimeError(
            f"OmniVoice sidecar could not find a free port in range "
            f"{self._start_port}-{self._start_port + self._port_probe_max}"
        )

    async def _spawn_sidecar(self, port: int) -> asyncio.subprocess.Process:
        self._prepare_runtime_directories()
        env = self._build_subprocess_env(port=port)
        logger.debug("Starting OmniVoice sidecar on {}:{}", self._host, port)
        return await asyncio.create_subprocess_exec(
            self._resolve_interpreter(),
            "-m",
            "tldw_Server_API.app.core.TTS.adapters.omnivoice_sidecar_server",
            cwd=str(self._repo_root),
            env=env,
        )

    async def _wait_for_ready(self) -> None:
        if not self._base_url:
            raise RuntimeError("OmniVoice sidecar base URL is not set")

        deadline = asyncio.get_running_loop().time() + self._healthcheck_timeout_seconds
        headers = build_sidecar_auth_headers(self._token)
        client = await self.get_http_client(timeout=self._healthcheck_interval_seconds)
        last_http_error: httpx.HTTPError | None = None

        while asyncio.get_running_loop().time() < deadline:
            if self._process is not None and self._process.returncode is not None:
                raise RuntimeError(
                    f"OmniVoice sidecar exited during startup with code {self._process.returncode}"
                )
            try:
                response = await client.get(
                    f"{self._base_url.rstrip('/')}/health",
                    headers=headers,
                    timeout=self._healthcheck_interval_seconds,
                )
                if response.status_code == 200:
                    payload = response.json()
                    if bool(payload.get("ready")) or payload.get("status") in _READY_OR_REACHABLE_HEALTH_STATUSES:
                        return
            except httpx.HTTPError as exc:
                last_http_error = exc
            await asyncio.sleep(self._healthcheck_interval_seconds)

        if last_http_error is not None:
            logger.debug(
                "OmniVoice sidecar health polling failed; last HTTP error: {}",
                last_http_error,
            )
            raise RuntimeError("OmniVoice sidecar did not reach /health") from last_http_error
        raise RuntimeError("OmniVoice sidecar did not reach /health")

    def _clear_process_state(self, process: asyncio.subprocess.Process | None = None) -> None:
        if process is not None and self._process is not None and self._process is not process:
            return
        self._process = None
        self._base_url = None
        self._port = None
        self._last_activity_at = None

    async def _stop_process(self) -> None:
        async with self._lock:
            await self._stop_process_locked()

    async def _stop_process_locked(self) -> None:
        process = self._process
        if process is None:
            self._clear_process_state()
            return
        if process.returncode is not None:
            self._clear_process_state(process)
            return
        with contextlib.suppress(ProcessLookupError):
            process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            with contextlib.suppress(Exception):
                await process.wait()
        finally:
            self._clear_process_state(process)


__all__ = ["OmniVoiceSidecarSupervisor", "X_TLDW_SIDECAR_TOKEN_HEADER"]
