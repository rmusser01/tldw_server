"""Shared helpers for OCR runtime configuration, managed processes, and readiness probes."""

from __future__ import annotations

from dataclasses import dataclass, field
import http.client
import json
import os
import platform
import re
import signal
from threading import RLock
import time
from typing import Any, Mapping, Sequence


def _parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if not normalized:
        return default
    return normalized in {"1", "true", "yes", "on"}


def _parse_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return parsed


def _parse_argv_json(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value)
    raw = str(value).strip()
    if not raw:
        return ()
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return ()
    if not isinstance(parsed, list):
        return ()
    return tuple(str(item) for item in parsed)


def _pick_positive_cap(*caps: int | None) -> int | None:
    values = [cap for cap in caps if isinstance(cap, int) and cap > 0]
    if not values:
        return None
    return min(values)


def effective_page_concurrency(global_cap: int | None, backend_cap: int | None) -> int:
    """
    Resolve the page concurrency budget from the global and backend-specific caps.
    """

    return _pick_positive_cap(global_cap, backend_cap) or 1


def render_argv_template(
    argv: Sequence[str],
    *,
    model_path: str | None = None,
    image_path: str | None = None,
    prompt: str | None = None,
    host: str | None = None,
    port: int | str | None = None,
) -> list[str]:
    """
    Render argv placeholders without invoking a shell.
    """

    pattern = re.compile(r"\{model_path\}|\{image_path\}|\{prompt\}|\{host\}|\{port\}")
    replacements = {
        "{model_path}": "" if model_path is None else str(model_path),
        "{image_path}": "" if image_path is None else str(image_path),
        "{prompt}": "" if prompt is None else str(prompt),
        "{host}": "" if host is None else str(host),
        "{port}": "" if port is None else str(port),
    }

    rendered: list[str] = []
    for part in argv:
        value = str(part)
        rendered.append(pattern.sub(lambda match: replacements[match.group(0)], value))
    return rendered


def is_profile_available(profile: Any, backend_name: str | None = None) -> bool:
    """
    Cheap/local availability check for a runtime profile.

    This only consults in-memory/profile state and never performs reachability
    checks or subprocess probing.
    """

    if profile is None:
        return False

    if hasattr(profile, "is_available"):
        try:
            if backend_name is not None:
                return bool(profile.is_available(backend_name=backend_name))
            return bool(profile.is_available())
        except TypeError:
            return bool(profile.is_available())

    return False


@dataclass(frozen=True, slots=True)
class _BaseOCRProfile:
    mode: str
    allow_managed_start: bool = False
    max_page_concurrency: int | None = None
    argv: tuple[str, ...] = field(default_factory=tuple)

    def normalized_mode(self) -> str:
        return self.mode.strip().lower()


@dataclass(frozen=True, slots=True)
class RemoteOCRProfile(_BaseOCRProfile):
    host: str | None = None
    port: int | None = None
    model_path: str | None = None
    prompt: str | None = None

    def is_available(self) -> bool:
        return self.normalized_mode() == "remote" and bool(self.host) and self.port is not None


@dataclass(frozen=True, slots=True)
class ManagedOCRProfile(_BaseOCRProfile):
    host: str | None = None
    port: int | None = None
    model_path: str | None = None
    prompt: str | None = None

    def is_available(self, backend_name: str | None = None) -> bool:
        if _lookup_managed_process(backend_name) is not None:
            return True
        return (
            self.normalized_mode() == "managed"
            and self.allow_managed_start
            and bool(self.argv)
            and isinstance(self.port, int)
            and self.port > 0
        )


@dataclass(frozen=True, slots=True)
class CLIOCRProfile(_BaseOCRProfile):
    model_path: str | None = None
    prompt: str | None = None

    def is_available(self) -> bool:
        return self.normalized_mode() == "cli" and bool(self.argv)


@dataclass(frozen=True, slots=True)
class OCRRuntimeProfiles:
    remote: RemoteOCRProfile
    managed: ManagedOCRProfile
    cli: CLIOCRProfile
    active: _BaseOCRProfile


@dataclass(slots=True)
class ManagedProcessRecord:
    process: Any
    host: str | None = None
    port: int | None = None
    argv: tuple[str, ...] = field(default_factory=tuple)

    def is_running(self) -> bool:
        return is_process_running(self.process)

    @property
    def base_url(self) -> str | None:
        if not self.host or self.port is None:
            return None
        return f"http://{self.host}:{self.port}"


_MANAGED_PROCESS_REGISTRY: dict[str, ManagedProcessRecord] = {}
_MANAGED_PROCESS_LOCK = RLock()


def register_managed_process(
    backend_name: str,
    process: Any,
    *,
    host: str | None = None,
    port: int | None = None,
    argv: Sequence[str] | None = None,
) -> Any:
    with _MANAGED_PROCESS_LOCK:
        _MANAGED_PROCESS_REGISTRY[backend_name] = ManagedProcessRecord(
            process=process,
            host=host,
            port=port,
            argv=tuple(str(item) for item in (argv or ())),
        )
    return process


def get_managed_process(backend_name: str) -> Any | None:
    with _MANAGED_PROCESS_LOCK:
        record = _MANAGED_PROCESS_REGISTRY.get(backend_name)
    if record is None:
        return None
    if not record.is_running():
        clear_managed_process(backend_name)
        return None
    return record.process


def get_managed_process_record(backend_name: str) -> ManagedProcessRecord | None:
    with _MANAGED_PROCESS_LOCK:
        record = _MANAGED_PROCESS_REGISTRY.get(backend_name)
    if record is None:
        return None
    if not record.is_running():
        clear_managed_process(backend_name)
        return None
    return record


def clear_managed_process(backend_name: str) -> Any | None:
    with _MANAGED_PROCESS_LOCK:
        record = _MANAGED_PROCESS_REGISTRY.pop(backend_name, None)
    if record is None:
        return None
    return record.process


def reset_managed_process_registry() -> None:
    with _MANAGED_PROCESS_LOCK:
        _MANAGED_PROCESS_REGISTRY.clear()


def _lookup_managed_process(backend_name: str | None) -> Any | None:
    if backend_name is None:
        return None
    return get_managed_process(backend_name)


def is_process_running(process: Any) -> bool:
    if process is None:
        return False
    try:
        poll = getattr(process, "poll", None)
        if callable(poll):
            return poll() is None
        return getattr(process, "returncode", None) is None
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return False


def managed_process_running(backend_name: str) -> bool:
    return get_managed_process_record(backend_name) is not None


def cleanup_managed_process(backend_name: str, *, timeout: float = 5.0) -> bool:
    process = clear_managed_process(backend_name)
    return terminate_process(process, timeout=timeout)


def terminate_process(process: Any, *, timeout: float = 5.0) -> bool:
    if process is None or not is_process_running(process):
        return True

    is_windows = platform.system().lower().startswith("win")
    try:
        if is_windows:
            terminate = getattr(process, "terminate", None)
            if callable(terminate):
                terminate()
        else:
            pid = getattr(process, "pid", None)
            if pid:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
            else:
                terminate = getattr(process, "terminate", None)
                if callable(terminate):
                    terminate()
    except (AttributeError, OSError, ProcessLookupError, RuntimeError, ValueError):
        terminate = getattr(process, "terminate", None)
        if callable(terminate):
            try:
                terminate()
            except (AttributeError, OSError, RuntimeError, ValueError):
                pass

    wait = getattr(process, "wait", None)
    if callable(wait):
        deadline = time.monotonic() + max(timeout, 0.0)
        while time.monotonic() < deadline:
            try:
                wait(timeout=0.1)
                return True
            except TypeError:
                break
            except Exception:
                if not is_process_running(process):
                    return True
        try:
            wait(timeout=max(timeout, 0.0))
            return True
        except TypeError:
            pass
        except Exception:
            if not is_process_running(process):
                return True

    if is_process_running(process):
        kill = getattr(process, "kill", None)
        if callable(kill):
            try:
                kill()
            except (AttributeError, OSError, RuntimeError, ValueError):
                return False
        wait = getattr(process, "wait", None)
        if callable(wait):
            try:
                wait(timeout=max(timeout, 0.0))
                return True
            except (TypeError, Exception):
                pass
    return not is_process_running(process)


def wait_for_managed_http_ready(
    *,
    host: str,
    port: int,
    scheme: str = "http",
    timeout_total: float = 30.0,
    interval: float = 0.5,
    paths: tuple[str, ...] = ("/health", "/v1/models"),
) -> bool:
    deadline = time.monotonic() + max(timeout_total, 0.0)
    while time.monotonic() < deadline:
        for path in paths:
            connection_cls = http.client.HTTPSConnection if str(scheme).strip().lower() == "https" else http.client.HTTPConnection
            connection = connection_cls(host, port, timeout=min(5.0, max(interval, 0.1)))
            try:
                connection.request("GET", path)
                response = connection.getresponse()
                if 200 <= int(response.status) < 300:
                    response.read()
                    return True
                response.read()
            except (ConnectionError, OSError, TimeoutError, http.client.HTTPException):
                pass
            finally:
                connection.close()
        time.sleep(max(interval, 0.05))
    return False


def load_ocr_runtime_profiles(
    prefix: str,
    env: Mapping[str, Any] | None = None,
) -> OCRRuntimeProfiles:
    """
    Parse shared OCR runtime configuration for a backend namespace.
    """

    source = os.environ if env is None else env
    normalized_prefix = prefix.strip().upper().rstrip("_")

    def key(suffix: str) -> str:
        return f"{normalized_prefix}_OCR_{suffix}"

    mode = str(source.get(key("MODE"), "cli")).strip().lower() or "cli"
    allow_managed_start = _parse_bool(source.get(key("ALLOW_MANAGED_START")), False)
    max_page_concurrency = _parse_int(source.get(key("MAX_PAGE_CONCURRENCY")))
    if max_page_concurrency is None or max_page_concurrency < 1:
        max_page_concurrency = 1
    argv = _parse_argv_json(source.get(key("ARGV")))
    host = source.get(key("HOST"))
    port = _parse_int(source.get(key("PORT")))
    model_path = source.get(key("MODEL_PATH"))
    prompt = source.get(key("PROMPT"))

    remote = RemoteOCRProfile(
        mode=mode,
        allow_managed_start=allow_managed_start,
        max_page_concurrency=max_page_concurrency,
        argv=argv,
        host=None if host is None else str(host),
        port=port,
        model_path=None if model_path is None else str(model_path),
        prompt=None if prompt is None else str(prompt),
    )
    managed = ManagedOCRProfile(
        mode=mode,
        allow_managed_start=allow_managed_start,
        max_page_concurrency=max_page_concurrency,
        argv=argv,
        host=None if host is None else str(host),
        port=port,
        model_path=None if model_path is None else str(model_path),
        prompt=None if prompt is None else str(prompt),
    )
    cli = CLIOCRProfile(
        mode=mode,
        allow_managed_start=allow_managed_start,
        max_page_concurrency=max_page_concurrency,
        argv=argv,
        model_path=None if model_path is None else str(model_path),
        prompt=None if prompt is None else str(prompt),
    )

    if mode == "remote":
        active: _BaseOCRProfile = remote
    elif mode == "managed":
        active = managed
    else:
        active = cli

    return OCRRuntimeProfiles(remote=remote, managed=managed, cli=cli, active=active)
