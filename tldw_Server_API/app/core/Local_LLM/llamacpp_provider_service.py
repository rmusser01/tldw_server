from __future__ import annotations

import inspect
import os
import re
from pathlib import Path
from typing import Any

from tldw_Server_API.app.core.Local_LLM import handler_utils
from tldw_Server_API.app.core.Local_LLM.llamacpp_config_lock import LockAcquisitionError, llamacpp_config_write_lock
from tldw_Server_API.app.core.Setup import setup_manager
from tldw_Server_API.app.core.config import refresh_config_cache

PROVIDER_NAME = "llama"
PROVIDER_SECTION = "Local-API"
PROVIDER_ENDPOINT_FIELD = "llama_api_IP"

# No current config env override exists for Local-API.llama_api_IP. Keep this
# narrow so future env-backed provider endpoint aliases can be added explicitly.
LLAMA_PROVIDER_ENDPOINT_ENV_KEYS: tuple[str, ...] = ()

_MAX_LOG_LINES = 1000
_MAX_LOG_BYTES = 256 * 1024
_SECRET_PATTERNS = (
    "api_key",
    "token",
    "hf_token",
    "openai_api_key",
    "anthropic_api_key",
)


class ManagedServerNotRunningError(RuntimeError):
    """Raised when provider wiring needs an active managed llama.cpp server."""


class ProviderConfigWriteError(RuntimeError):
    """Raised when provider endpoint persistence fails."""


async def use_managed_server_in_chat(llm_manager: Any) -> dict[str, Any]:
    """Persist the active managed llama.cpp endpoint as the chat provider endpoint."""
    handler = _require_handler(llm_manager)
    status = await _get_status(handler)
    endpoint = _endpoint_from_status_or_handler(status, handler)

    try:
        with llamacpp_config_write_lock():
            setup_manager.update_config({PROVIDER_SECTION: {PROVIDER_ENDPOINT_FIELD: endpoint}})
            refresh_config_cache()
    except LockAcquisitionError as exc:
        raise ProviderConfigWriteError("Failed to update llama.cpp chat provider endpoint.") from exc
    except Exception as exc:
        raise ProviderConfigWriteError("Failed to update llama.cpp chat provider endpoint.") from exc

    warnings: list[str] = []
    effective = True
    env_override = get_provider_endpoint_env_override()
    if env_override:
        effective = False
        warnings.append(
            f"{env_override} is set, so updating {PROVIDER_SECTION}.{PROVIDER_ENDPOINT_FIELD} may not affect chat."
        )

    return {
        "provider": PROVIDER_NAME,
        "endpoint": endpoint,
        "updated": True,
        "effective": effective,
        "warnings": warnings,
    }


def tail_managed_log(llm_manager: Any, requested_lines: int) -> dict[str, Any]:
    """Return a bounded, redacted tail of the active handler's configured log file."""
    handler = _require_handler(llm_manager)
    line_count = max(1, min(int(requested_lines), _MAX_LOG_LINES))
    warnings: list[str] = []

    configured_log = getattr(getattr(handler, "config", None), "log_output_file", None)
    if not configured_log:
        return {"lines": [], "truncated": False, "warnings": ["No managed llama.cpp log file is configured."]}

    log_path = Path(str(configured_log)).expanduser()
    try:
        resolved_log_path = log_path.resolve(strict=False)
    except OSError:
        return {"lines": [], "truncated": False, "warnings": ["Managed llama.cpp log file is unavailable."]}

    if not resolved_log_path.is_file():
        return {"lines": [], "truncated": False, "warnings": ["Managed llama.cpp log file is unavailable."]}

    try:
        file_size = resolved_log_path.stat().st_size
        read_size = min(file_size, _MAX_LOG_BYTES)
        with resolved_log_path.open("rb") as handle:
            handle.seek(max(0, file_size - read_size))
            raw = handle.read(read_size)
    except OSError:
        return {"lines": [], "truncated": False, "warnings": ["Managed llama.cpp log file is unavailable."]}

    decoded = raw.decode("utf-8", errors="replace")
    all_lines = decoded.splitlines()
    selected = all_lines[-line_count:]
    truncated = file_size > read_size or len(all_lines) > len(selected)
    return {
        "lines": [_redact_log_line(line) for line in selected],
        "truncated": truncated,
        "warnings": warnings,
    }


def normalize_managed_base_url(host: str | None, port: int) -> str:
    """Build the provider base URL, mapping wildcard bind hosts to loopback."""
    client_host = handler_utils.resolve_client_host(host)
    return handler_utils.build_base_url(client_host, port)


def get_provider_endpoint_env_override() -> str | None:
    """Return the first known env override for the llama.cpp provider endpoint."""
    for env_key in LLAMA_PROVIDER_ENDPOINT_ENV_KEYS:
        if os.getenv(env_key):
            return env_key
    return None


def _require_handler(llm_manager: Any) -> Any:
    handler = getattr(llm_manager, "llamacpp", None)
    if handler is None:
        raise ManagedServerNotRunningError("Managed llama.cpp handler is not configured.")
    return handler


async def _get_status(handler: Any) -> dict[str, Any]:
    status_fn = getattr(handler, "get_server_status", None)
    if status_fn is None:
        raise ManagedServerNotRunningError("Managed llama.cpp server is not running.")
    status = status_fn()
    if inspect.isawaitable(status):
        status = await status
    if not isinstance(status, dict):
        raise ManagedServerNotRunningError("Managed llama.cpp server is not running.")
    return status


def _endpoint_from_status_or_handler(status: dict[str, Any], handler: Any) -> str:
    process = getattr(handler, "_active_server_process", None)
    if status.get("status") != "running" or process is None or getattr(process, "returncode", None) is not None:
        raise ManagedServerNotRunningError("Managed llama.cpp server is not running.")

    host = status.get("host") or getattr(handler, "_active_server_host", None)
    port = status.get("port") or getattr(handler, "_active_server_port", None)
    if port is None:
        raise ManagedServerNotRunningError("Managed llama.cpp server port is unavailable.")
    return normalize_managed_base_url(str(host or "127.0.0.1"), int(port))


def _redact_log_line(line: str) -> str:
    redacted = line
    for key in _SECRET_PATTERNS:
        redacted = re.sub(
            rf"(?i)\b{key}=([^\s]+)",
            f"{key}=[REDACTED]",
            redacted,
        )
    return redacted
