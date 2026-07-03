"""Managed sidecar supervisor for audiocpp_server."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import socket
import time
from pathlib import Path
from typing import Any

from loguru import logger

from ..tts_exceptions import TTSError, TTSProviderInitializationError
from .audio_cpp_client import AudioCppClient
from .audio_cpp_config import PROVIDER_KEY, AudioCppConfig, validate_managed_host

_SUBPROCESS_ENV_ALLOWLIST = {
    "COMSPEC",
    "CUDA_HOME",
    "CUDA_PATH",
    "CUDA_VISIBLE_DEVICES",
    "DYLD_LIBRARY_PATH",
    "HIP_PATH",
    "LD_LIBRARY_PATH",
    "NVIDIA_DRIVER_CAPABILITIES",
    "NVIDIA_VISIBLE_DEVICES",
    "PATH",
    "PATHEXT",
    "ROCM_PATH",
    "SystemRoot",
    "TEMP",
    "TMP",
    "WINDIR",
}


def is_port_free(host: str, port: int) -> bool:
    """Return True when a TCP bind probe succeeds for host:port."""
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, int(port)))
        except OSError:
            return False
    return True


class AudioCppSidecarSupervisor:
    """Own the lifecycle of an optional loopback audiocpp_server process."""

    def __init__(self, provider_config: dict[str, Any], repo_root: Path | None = None) -> None:
        self._provider_config = dict(provider_config or {})
        self._repo_root = Path(repo_root or Path.cwd()).resolve(strict=False)
        self._audio_cpp_config = AudioCppConfig.from_provider_config(
            self._provider_config,
            repo_root=self._repo_root,
        )
        self._server = dict(self._audio_cpp_config.server or {})
        self._host = validate_managed_host(self._server.get("host"))
        self._start_port = int(self._server.get("port") or 8080)
        self._autoselect_port = self._as_bool(self._server.get("autoselect_port"), default=True)
        self._port_probe_max = max(0, int(self._server.get("port_probe_max") or 10))
        self._startup_timeout_seconds = float(self._server.get("startup_timeout_seconds") or 30.0)
        self._healthcheck_interval_seconds = float(self._server.get("healthcheck_interval_seconds") or 0.25)
        self._startup_backoff_seconds = float(self._server.get("startup_backoff_seconds") or 5.0)
        self._idle_shutdown_seconds = float(self._server.get("idle_shutdown_seconds") or 900.0)
        self._terminate_timeout_seconds = float(self._server.get("terminate_timeout_seconds") or 10.0)
        self._binary_path = self._resolve_repo_path(self._provider_config.get("binary_path"))
        self.server_config_path = self._resolve_server_config_path()
        self._process: asyncio.subprocess.Process | None = None
        self._client: Any | None = None
        self._port: int | None = None
        self._base_url: str | None = None
        self._last_failure_at: float | None = None
        self._last_activity_at: float | None = None
        self._lock = asyncio.Lock()

    @property
    def port(self) -> int | None:
        return self._port

    @property
    def base_url(self) -> str | None:
        return self._base_url

    @property
    def last_failure_at(self) -> float | None:
        return self._last_failure_at

    async def ensure_started(self) -> str:
        """Start the sidecar when needed and return its loopback base URL."""
        async with self._lock:
            if self._is_process_running() and self._base_url:
                if await self._shutdown_if_idle_locked():
                    logger.debug("Restarting audio.cpp sidecar after idle shutdown")
                else:
                    self._last_activity_at = time.time()
                    return self._base_url

            if self._last_failure_at is not None:
                elapsed = time.time() - self._last_failure_at
                if elapsed < self._startup_backoff_seconds:
                    raise TTSProviderInitializationError(
                        "audio.cpp sidecar startup is backing off after a recent failure",
                        provider=PROVIDER_KEY,
                        error_code="SIDECAR_BACKOFF",
                    )

            selected_port = self._select_port()
            self._write_server_config(selected_port)
            base_url = self._build_base_url(self._host, selected_port)
            try:
                self._port = selected_port
                self._base_url = base_url
                self._process = await self._spawn_sidecar()
                self._client = AudioCppClient(
                    base_url=base_url,
                    timeout=max(self._healthcheck_interval_seconds, 0.01),
                    allow_remote_base_url=False,
                )
                await self._wait_for_ready()
            except Exception as exc:
                self._record_failure()
                await self._stop_process_locked()
                await self._close_client()
                raise TTSProviderInitializationError(
                    "audio.cpp sidecar did not reach /health",
                    provider=PROVIDER_KEY,
                    error_code="SIDECAR_STARTUP_FAILED",
                ) from exc

            self._last_activity_at = time.time()
            return base_url

    async def shutdown_if_idle(self) -> bool:
        async with self._lock:
            return await self._shutdown_if_idle_locked()

    async def shutdown(self) -> None:
        async with self._lock:
            await self._stop_process_locked()
            await self._close_client()

    @staticmethod
    def _as_bool(value: Any, *, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
        return bool(value)

    def _resolve_repo_path(self, value: Any) -> Path:
        if value is None or str(value).strip() == "":
            raise TTSProviderInitializationError(
                "audio.cpp managed mode requires binary_path",
                provider=PROVIDER_KEY,
                error_code="MISSING_BINARY_PATH",
            )
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = self._repo_root / path
        return path.resolve(strict=False)

    def _resolve_server_config_path(self) -> Path:
        configured = self._server.get("server_config_path") or "models/audio_cpp/server.json"
        path = Path(str(configured)).expanduser()
        if not path.is_absolute():
            path = self._repo_root / path
        resolved = path.resolve(strict=False)
        models_root = self._audio_cpp_config.models_root
        try:
            resolved.relative_to(models_root)
        except ValueError as exc:
            raise TTSProviderInitializationError(
                "audio.cpp server_config_path must stay under the audio.cpp runtime root",
                provider=PROVIDER_KEY,
                error_code="SERVER_CONFIG_PATH_OUTSIDE_ROOT",
            ) from exc
        return resolved

    def _select_port(self) -> int:
        if not self._autoselect_port:
            if not is_port_free(self._host, self._start_port):
                raise TTSProviderInitializationError(
                    "audio.cpp sidecar configured port is unavailable",
                    provider=PROVIDER_KEY,
                    error_code="SIDECAR_PORT_UNAVAILABLE",
                )
            return self._start_port

        for offset in range(self._port_probe_max + 1):
            candidate = self._start_port + offset
            if is_port_free(self._host, candidate):
                return candidate

        raise TTSProviderInitializationError(
            "audio.cpp sidecar could not find a free loopback port",
            provider=PROVIDER_KEY,
            error_code="SIDECAR_PORT_UNAVAILABLE",
        )

    def _write_server_config(self, port: int) -> None:
        config = self._audio_cpp_config.render_server_config()
        config["host"] = self._host
        config["port"] = int(port)
        self.server_config_path.parent.mkdir(parents=True, exist_ok=True)
        self.server_config_path.write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    async def _spawn_sidecar(self) -> asyncio.subprocess.Process:
        logger.debug("Starting audio.cpp sidecar on {}:{}", self._host, self._port)
        return await asyncio.create_subprocess_exec(
            str(self._binary_path),
            "--config",
            str(self.server_config_path),
            cwd=str(self._repo_root),
            env=self._build_subprocess_env(),
        )

    def _build_subprocess_env(self) -> dict[str, str]:
        env: dict[str, str] = {}
        allowed_upper = {key.upper() for key in _SUBPROCESS_ENV_ALLOWLIST}
        for key, value in os.environ.items():
            if key.upper() in allowed_upper and value:
                env[key] = value
        return env

    async def _wait_for_ready(self) -> None:
        if self._client is None:
            raise RuntimeError("audio.cpp sidecar health client is not initialized")
        deadline = asyncio.get_running_loop().time() + self._startup_timeout_seconds

        while asyncio.get_running_loop().time() < deadline:
            if self._process is not None and self._process.returncode is not None:
                raise RuntimeError("audio.cpp sidecar exited during startup")
            health = await self._probe_health()
            if health:
                return
            await asyncio.sleep(self._healthcheck_interval_seconds)

        raise RuntimeError("audio.cpp sidecar did not reach /health")

    async def _probe_health(self) -> bool:
        try:
            payload = await self._client.health()
        except (TTSError, RuntimeError) as exc:
            logger.debug("audio.cpp sidecar health probe failed: {}", type(exc).__name__)
            return False
        if not isinstance(payload, dict):
            return False
        status = str(payload.get("status") or payload.get("state") or "ok").strip().lower()
        return status not in {"error", "failed", "unhealthy"}

    async def _shutdown_if_idle_locked(self) -> bool:
        if self._idle_shutdown_seconds <= 0 or self._last_activity_at is None:
            return False
        if (time.time() - self._last_activity_at) < self._idle_shutdown_seconds:
            return False
        if self._process is None or self._process.returncode is not None:
            self._clear_process_state()
            return False
        await self._stop_process_locked()
        return True

    def _is_process_running(self) -> bool:
        return self._process is not None and self._process.returncode is None

    def _record_failure(self) -> None:
        self._last_failure_at = time.time()

    async def _close_client(self) -> None:
        client = self._client
        self._client = None
        if client is None:
            return
        close = getattr(client, "close", None)
        if callable(close):
            maybe_close = close()
            if hasattr(maybe_close, "__await__"):
                await maybe_close

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
            await asyncio.wait_for(process.wait(), timeout=self._terminate_timeout_seconds)
        except asyncio.TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            with contextlib.suppress(Exception):
                await process.wait()
        finally:
            self._clear_process_state(process)

    def _clear_process_state(self, process: asyncio.subprocess.Process | None = None) -> None:
        if process is not None and self._process is not process:
            return
        self._process = None
        self._base_url = None
        self._port = None
        self._last_activity_at = None

    @staticmethod
    def _build_base_url(host: str, port: int) -> str:
        if ":" in host and not host.startswith("["):
            return f"http://[{host}]:{port}"
        return f"http://{host}:{port}"


__all__ = ["AudioCppSidecarSupervisor", "is_port_free"]
