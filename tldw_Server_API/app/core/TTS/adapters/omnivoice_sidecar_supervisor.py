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


def create_sidecar_async_client(*, timeout: float) -> httpx.AsyncClient:
    """Create an httpx client dedicated to loopback sidecar traffic."""
    return httpx.AsyncClient(trust_env=False, timeout=timeout)


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
        self._closing = False
        self._token = secrets.token_urlsafe(32)
        self._process: asyncio.subprocess.Process | None = None
        self._port: int | None = None
        self._base_url: str | None = None
        self._last_failure_at: float | None = None
        self._last_activity_at: float | None = None
        self._lock = asyncio.Lock()

    def _coalesce_extra_param(self, key: str, default: Any) -> Any:
        value = self._extra_params.get(key)
        return default if value is None else value

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

        if self._process is not None and self._process.returncode is None and self._base_url:
            self._last_activity_at = time.time()
            return self._base_url

        async with self._lock:
            if self._closing:
                raise RuntimeError("OmniVoice sidecar supervisor is closing")
            if self._process is not None and self._process.returncode is None and self._base_url:
                self._last_activity_at = time.time()
                return self._base_url
            if self._last_failure_at is not None and (time.time() - self._last_failure_at) < self._startup_backoff_seconds:
                raise RuntimeError("OmniVoice sidecar startup is backing off after a recent failure")

            selected_port = self._select_port()
            self._process = await self._spawn_sidecar(selected_port)
            self._port = selected_port
            self._base_url = build_base_url(resolve_client_host(self._host), selected_port)

            try:
                await self._wait_for_ready()
            except Exception as exc:
                self._record_failure()
                await self._stop_process()
                raise RuntimeError("OmniVoice sidecar did not reach /health") from exc

            self._last_activity_at = time.time()
            return self._base_url

    async def shutdown_if_idle(self) -> bool:
        if self._idle_shutdown_seconds <= 0:
            return False
        if self._last_activity_at is None:
            return False
        if (time.time() - self._last_activity_at) < self._idle_shutdown_seconds:
            return False
        if self._process is None or self._process.returncode is not None:
            self._last_activity_at = None
            return False
        await self._stop_process()
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
        env = os.environ.copy()
        env.update(
            {
                "OMNIVOICE_SIDECAR_TOKEN": self._token,
                "OMNIVOICE_SIDECAR_HOST": self._host,
                "OMNIVOICE_SIDECAR_PORT": str(port),
            }
        )
        logger.debug("Starting OmniVoice sidecar on {}:{}", self._host, port)
        return await asyncio.create_subprocess_exec(
            sys.executable,
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

        async with create_sidecar_async_client(timeout=self._healthcheck_interval_seconds) as client:
            while asyncio.get_running_loop().time() < deadline:
                if self._process is not None and self._process.returncode is not None:
                    raise RuntimeError(
                        f"OmniVoice sidecar exited during startup with code {self._process.returncode}"
                    )
                try:
                    response = await client.get(
                        f"{self._base_url.rstrip('/')}/health",
                        headers=headers,
                    )
                    if response.status_code == 200:
                        payload = response.json()
                        if bool(payload.get("ready", True)):
                            return
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(self._healthcheck_interval_seconds)

        raise RuntimeError("OmniVoice sidecar did not reach /health")

    async def _stop_process(self) -> None:
        process = self._process
        self._process = None
        self._base_url = None
        self._port = None
        if process is None:
            return
        if process.returncode is not None:
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


__all__ = ["OmniVoiceSidecarSupervisor", "X_TLDW_SIDECAR_TOKEN_HEADER"]
