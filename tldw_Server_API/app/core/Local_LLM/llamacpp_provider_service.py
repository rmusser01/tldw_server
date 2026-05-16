from __future__ import annotations

import inspect
import os
import re
from pathlib import Path
from typing import Any

from starlette.concurrency import run_in_threadpool

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
_REDACTION_FIELD_PATTERN = r"(?:api_key|token|hf_token|openai_api_key|anthropic_api_key)"
_REDACTION_VALUE_PATTERN = r"(?:\"[^\"\r\n]*\"|'[^'\r\n]*'|[^\s\"',}\]]+)"


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


async def tail_managed_log(llm_manager: Any, requested_lines: int) -> dict[str, Any]:
    """Return a bounded, redacted tail of the active handler's configured log file."""
    handler = _require_handler(llm_manager)
    line_count = max(1, min(int(requested_lines), _MAX_LOG_LINES))

    configured_log = getattr(getattr(handler, "config", None), "log_output_file", None)
    if not configured_log:
        return {"lines": [], "truncated": False, "warnings": ["No managed llama.cpp log file is configured."]}

    status = await _get_status(handler)
    active_log = status.get("log_file")
    active_log_handle = getattr(handler, "_active_server_log_handle", None)
    if status.get("status") != "running" or not active_log or active_log_handle is None:
        return {"lines": [], "truncated": False, "warnings": ["No active managed llama.cpp log file is available."]}

    return await run_in_threadpool(_tail_managed_log_file, configured_log, active_log, line_count)


def _tail_managed_log_file(configured_log: Any, active_log: Any, line_count: int) -> dict[str, Any]:
    """Read and redact the managed log tail from a worker thread."""
    warnings: list[str] = []
    configured_log_path = Path(str(configured_log)).expanduser()
    active_log_path = Path(str(active_log)).expanduser()
    try:
        resolved_configured_log_path = configured_log_path.resolve(strict=False)
        resolved_active_log_path = active_log_path.resolve(strict=False)
    except OSError:
        return {"lines": [], "truncated": False, "warnings": ["Managed llama.cpp log file is unavailable."]}

    if resolved_configured_log_path != resolved_active_log_path:
        return {"lines": [], "truncated": False, "warnings": ["Active managed llama.cpp log file does not match configured log file."]}

    if not resolved_active_log_path.is_file():
        return {"lines": [], "truncated": False, "warnings": ["Managed llama.cpp log file is unavailable."]}

    try:
        file_size = resolved_active_log_path.stat().st_size
        read_size = min(file_size, _MAX_LOG_BYTES)
        with resolved_active_log_path.open("rb") as handle:
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
    def _redacted_value(match: re.Match[str]) -> str:
        prefix = match.group(1)
        value = match.group(2)
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            return f"{prefix}{value[0]}[REDACTED]{value[-1]}"
        return f"{prefix}[REDACTED]"

    redacted = line
    redacted = re.sub(
        rf"(?i)(Authorization\s*:\s*Bearer\s+)({_REDACTION_VALUE_PATTERN})",
        _redacted_value,
        redacted,
    )
    redacted = re.sub(
        rf"(?i)(--(?:api-key|hf-token|token)(?:\s*=\s*|\s+))({_REDACTION_VALUE_PATTERN})",
        _redacted_value,
        redacted,
    )
    redacted = re.sub(
        rf"(?i)(\b{_REDACTION_FIELD_PATTERN}\b\s*[:=]\s*)({_REDACTION_VALUE_PATTERN})",
        _redacted_value,
        redacted,
    )
    redacted = re.sub(
        rf"(?i)([\"']\b{_REDACTION_FIELD_PATTERN}\b[\"']\s*[:=]\s*)({_REDACTION_VALUE_PATTERN})",
        _redacted_value,
        redacted,
    )
    redacted = re.sub(
        r"\bsk-[A-Za-z0-9][A-Za-z0-9._-]{15,}\b",
        "[REDACTED_TOKEN]",
        redacted,
    )
    return redacted
