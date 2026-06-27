from __future__ import annotations

import socket

# This module owns the local uvicorn subprocess lifecycle for the harness.
import subprocess  # nosec B404
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class UvicornServer:
    process: subprocess.Popen[str]
    url: str


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _poll_url(url: str, timeout: float = 2.0) -> int:
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


def wait_for_health(base_url: str, timeout_seconds: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    health_url = f"{base_url.rstrip('/')}/health"
    last_error: Exception | str | None = None

    while time.monotonic() <= deadline:
        try:
            status = _poll_url(health_url)
            if status < 500:
                return
            last_error = f"{health_url} returned HTTP {status}"
        except Exception as exc:  # noqa: BLE001 - retain last transient startup error.
            last_error = exc
        time.sleep(0.25)

    raise TimeoutError(f"Timed out waiting for {health_url}; last error: {last_error}")


def wait_for_readiness(base_url: str, timeout_seconds: float = 30.0) -> None:
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
                if status < 500:
                    return
                last_error = f"{readiness_url} returned HTTP {status}"
            except Exception as exc:  # noqa: BLE001 - retain last transient startup error.
                last_error = exc
        time.sleep(0.25)

    raise TimeoutError(
        f"Timed out waiting for readiness endpoints {', '.join(readiness_urls)}; " f"last error: {last_error}"
    )


def start_server(env: Mapping[str, str], port: int | None = None) -> UvicornServer:
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
    # Fixed argv, shell=False, local harness only.
    process = subprocess.Popen(  # nosec B603
        command,
        env=dict(env),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    server = UvicornServer(process=process, url=url)
    try:
        wait_for_health(url)
    except Exception:
        stop_server(server)
        raise
    return server


def stop_server(server: UvicornServer) -> None:
    if server.process.poll() is not None:
        return

    server.process.terminate()
    try:
        server.process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        server.process.kill()
        server.process.wait(timeout=10)


__all__ = [
    "UvicornServer",
    "find_free_port",
    "start_server",
    "stop_server",
    "wait_for_health",
    "wait_for_readiness",
]
