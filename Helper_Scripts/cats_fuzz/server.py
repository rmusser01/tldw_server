"""Local uvicorn lifecycle and readiness helpers for CATS runtime blocks."""

from __future__ import annotations

import socket

# This module owns the local uvicorn subprocess lifecycle for the harness.
import subprocess  # nosec B404
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class UvicornServer:
    """Handle for a spawned uvicorn process and its optional log streams."""

    process: subprocess.Popen[str]
    url: str
    stdout_stream: TextIO | None = None
    stderr_stream: TextIO | None = None


def find_free_port() -> int:
    """Reserve and return an available loopback TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _poll_url(url: str, timeout: float = 2.0) -> int:
    """Return one HTTP status code for a constrained HTTP(S) health URL."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"Unsupported health check URL scheme: {parsed.scheme}")

    request = Request(url, method="GET")
    try:
        # URL scheme is constrained to HTTP(S) above before opening.
        with urlopen(request, timeout=timeout) as response:  # nosec B310
            return int(response.status)
    except HTTPError as exc:
        return int(exc.code)
    except URLError as exc:
        raise RuntimeError(str(exc)) from exc


def _startup_exit_message(return_code: int, stderr_path: Path | None) -> str:
    """Build an actionable startup failure message for an exited uvicorn child."""
    message = f"uvicorn exited during startup with code {return_code}"
    if stderr_path is not None:
        return f"{message}; stderr log: {stderr_path}"
    return f"{message}; stderr was not captured"


def wait_for_health(
    base_url: str,
    timeout_seconds: float = 30.0,
    *,
    process: subprocess.Popen[str] | None = None,
    stderr_path: Path | None = None,
) -> None:
    """Wait for the local server health endpoint and fail fast if uvicorn exits."""
    deadline = time.monotonic() + timeout_seconds
    health_url = f"{base_url.rstrip('/')}/health"
    last_error: Exception | str | None = None

    while time.monotonic() <= deadline:
        if process is not None:
            return_code = process.poll()
            if return_code is not None:
                raise RuntimeError(_startup_exit_message(return_code, stderr_path))
        try:
            status = _poll_url(health_url)
            if 200 <= status < 300:
                return
            last_error = f"{health_url} returned HTTP {status}"
        except Exception as exc:  # noqa: BLE001 - retain last transient startup error.
            last_error = exc
        time.sleep(0.25)

    raise TimeoutError(f"Timed out waiting for {health_url}; last error: {last_error}")


def wait_for_readiness(base_url: str, timeout_seconds: float = 30.0) -> None:
    """Wait for one of the supported readiness endpoints to return HTTP 2xx."""
    deadline = time.monotonic() + timeout_seconds
    readiness_urls = (
        f"{base_url.rstrip('/')}/ready",
        f"{base_url.rstrip('/')}/health/ready",
    )
    last_error: Exception | str | None = None

    while time.monotonic() <= deadline:
        for readiness_url in readiness_urls:
            try:
                status = _poll_url(readiness_url)
                if 200 <= status < 300:
                    return
                last_error = f"{readiness_url} returned HTTP {status}"
            except Exception as exc:  # noqa: BLE001 - retain last transient startup error.
                last_error = exc
        time.sleep(0.25)

    raise TimeoutError(
        f"Timed out waiting for readiness endpoints {', '.join(readiness_urls)}; " f"last error: {last_error}"
    )


def _close_server_streams(server: UvicornServer) -> None:
    """Close any log streams attached to a server handle."""
    for stream in (server.stdout_stream, server.stderr_stream):
        if stream is not None and not stream.closed:
            stream.close()


def start_server(
    env: Mapping[str, str],
    port: int | None = None,
    log_dir: Path | None = None,
) -> UvicornServer:
    """Start a loopback uvicorn server and wait until it is healthy."""
    selected_port = port if port is not None else find_free_port()
    url = f"http://127.0.0.1:{selected_port}"
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "tldw_Server_API.app.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(selected_port),
    ]
    stdout_target: int | TextIO
    stderr_target: int | TextIO
    stdout_stream: TextIO | None = None
    stderr_stream: TextIO | None = None
    stderr_path: Path | None = None
    if log_dir is None:
        stdout_target = subprocess.DEVNULL
        stderr_target = subprocess.DEVNULL
    else:
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout_stream = (log_dir / "uvicorn.stdout.log").open("w", encoding="utf-8")
        stderr_path = log_dir / "uvicorn.stderr.log"
        try:
            stderr_stream = stderr_path.open("w", encoding="utf-8")
        except Exception:
            stdout_stream.close()
            raise
        stdout_target = stdout_stream
        stderr_target = stderr_stream

    # Fixed argv, shell=False, local harness only.
    try:
        process = subprocess.Popen(  # nosec B603
            command,
            env=dict(env),
            stdout=stdout_target,
            stderr=stderr_target,
            text=True,
        )
    except Exception:
        for stream in (stdout_stream, stderr_stream):
            if stream is not None and not stream.closed:
                stream.close()
        raise
    server = UvicornServer(
        process=process,
        url=url,
        stdout_stream=stdout_stream,
        stderr_stream=stderr_stream,
    )
    try:
        wait_for_health(url, process=process, stderr_path=stderr_path)
    except Exception:
        stop_server(server)
        raise
    return server


def stop_server(server: UvicornServer) -> None:
    """Terminate a spawned uvicorn server and close any attached streams."""
    try:
        if server.process.poll() is not None:
            return

        server.process.terminate()
        try:
            server.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.process.kill()
            server.process.wait(timeout=10)
    finally:
        _close_server_streams(server)


__all__ = [
    "UvicornServer",
    "find_free_port",
    "start_server",
    "stop_server",
    "wait_for_health",
    "wait_for_readiness",
]
