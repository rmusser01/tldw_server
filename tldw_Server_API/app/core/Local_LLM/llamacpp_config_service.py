from __future__ import annotations

import os
# Validation probes a user-selected local binary without shell.
import subprocess  # nosec B404
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from tldw_Server_API.app.core.Setup import setup_manager
from tldw_Server_API.app.core.config import load_comprehensive_config, refresh_config_cache


LLAMACPP_ENV_OVERRIDES = {
    "enabled": "LLAMACPP_ENABLED",
    "executable_path": "LLAMACPP_EXECUTABLE_PATH",
    "models_dir": "LLAMACPP_MODELS_DIR",
    "default_host": "LLAMACPP_HOST",
    "default_port": "LLAMACPP_PORT",
    "default_threads": "LLAMACPP_THREADS",
    "default_n_gpu_layers": "LLAMACPP_N_GPU_LAYERS",
    "default_ctx_size": "LLAMACPP_CTX_SIZE",
    "allow_unvalidated_args": "LLAMACPP_ALLOW_UNVALIDATED_ARGS",
    "allow_cli_secrets": "LLAMACPP_ALLOW_CLI_SECRETS",
    "port_autoselect": "LLAMACPP_PORT_AUTOSELECT",
    "port_probe_max": "LLAMACPP_PORT_PROBE_MAX",
    "allowed_paths": "LLAMACPP_ALLOWED_PATHS",
    "log_output_file": "LLAMACPP_LOG_OUTPUT_FILE",
}

RESTART_FIELDS = {
    "enabled",
    "executable_path",
    "models_dir",
    "allowed_paths",
    "log_output_file",
}

_BOOL_FIELDS = {
    "enabled",
    "allow_unvalidated_args",
    "allow_cli_secrets",
    "port_autoselect",
}
_INT_FIELDS = {
    "default_port",
    "default_threads",
    "default_n_gpu_layers",
    "default_ctx_size",
    "port_probe_max",
}
_LIST_FIELDS = {
    "allowed_paths",
    "registered_model_paths",
}
_SAVED_FIELDS = (
    "enabled",
    "executable_path",
    "models_dir",
    "default_host",
    "default_port",
    "default_threads",
    "default_n_gpu_layers",
    "default_ctx_size",
    "allow_unvalidated_args",
    "allow_cli_secrets",
    "port_autoselect",
    "port_probe_max",
    "allowed_paths",
    "registered_model_paths",
    "log_output_file",
)


def get_env_overrides() -> dict[str, bool]:
    """Return env override state for each typed llama.cpp config field."""
    return {field: os.getenv(env_name) is not None for field, env_name in LLAMACPP_ENV_OVERRIDES.items()}


def get_config_state(llm_manager: Any) -> dict[str, Any]:
    """Return saved config, active handler config, restart signals, and warnings."""
    saved_config = _read_saved_config()
    active_config = _read_active_config(llm_manager)
    restart_reasons = _restart_reasons(saved_config, active_config)
    warnings = _warnings(saved_config, active_config, restart_reasons)

    return {
        "saved_config": saved_config,
        "active_config": active_config,
        "restart_required": bool(restart_reasons),
        "restart_reasons": restart_reasons,
        "env_overrides": get_env_overrides(),
        "warnings": warnings,
    }


def update_config_state(payload: Any, llm_manager: Any) -> dict[str, Any]:
    """Persist typed llama.cpp config updates through the setup manager."""
    updates = _payload_to_updates(payload)
    if not updates:
        return get_config_state(llm_manager)

    env_overrides = get_env_overrides()
    locked_fields = sorted(field for field in updates if env_overrides.get(field, False))
    if locked_fields:
        raise HTTPException(
            status_code=409,
            detail={
                "message": "Some llama.cpp config fields are controlled by environment variables.",
                "locked_fields": locked_fields,
            },
        )

    setup_manager.update_config({"LlamaCpp": updates})
    refresh_config_cache()
    return get_config_state(llm_manager)


def validate_binary(binary_path: str, timeout_seconds: float = 3.0) -> dict[str, Any]:
    """Validate a llama.cpp binary path without starting a managed server."""
    warnings: list[str] = []
    raw_path = str(binary_path).strip()
    if not raw_path:
        return {
            "valid": False,
            "exists": False,
            "executable": False,
            "resolved_path": None,
            "version_output": None,
            "help_output": None,
            "warnings": ["Binary path is required."],
        }

    path = Path(raw_path).expanduser()
    exists = path.exists()
    executable = exists and os.access(path, os.X_OK)
    if not exists:
        warnings.append(f"Binary '{path.name or 'llama-server'}' was not found.")
    elif not executable:
        warnings.append(f"Binary '{path.name}' is not executable.")

    resolved_path = str(path.resolve()) if exists else None
    version_output = None
    help_output = None
    if exists and executable:
        version_output = _probe_binary(path, "--version", timeout_seconds, warnings)
        if version_output is None:
            help_output = _probe_binary(path, "--help", timeout_seconds, warnings)
        if version_output is None and help_output is None:
            warnings.append(f"Binary '{path.name}' did not return version or help output.")

    return {
        "valid": bool(exists and executable and (version_output or help_output)),
        "exists": bool(exists),
        "executable": bool(executable),
        "resolved_path": resolved_path,
        "version_output": version_output,
        "help_output": help_output,
        "warnings": warnings,
    }


def _read_saved_config() -> dict[str, Any]:
    parser = load_comprehensive_config()
    section = parser["LlamaCpp"] if parser and parser.has_section("LlamaCpp") else None
    saved: dict[str, Any] = {
        "enabled": False,
        "allowed_paths": [],
        "registered_model_paths": [],
    }
    if section is None:
        return saved

    for field in _SAVED_FIELDS:
        if field in _BOOL_FIELDS:
            saved[field] = section.getboolean(field, fallback=False)
        elif field in _INT_FIELDS:
            raw = section.get(field, fallback=None)
            saved[field] = _int_or_none(raw)
        elif field in _LIST_FIELDS:
            saved[field] = _split_list(section.get(field, fallback=""))
        else:
            saved[field] = _str_or_none(section.get(field, fallback=None))
    return saved


def _read_active_config(llm_manager: Any) -> dict[str, Any]:
    handler = getattr(llm_manager, "llamacpp", None)
    config = getattr(handler, "config", None) if handler is not None else None
    active: dict[str, Any] = {"handler_configured": config is not None}
    if config is None:
        return active

    active.update(
        {
            "enabled": getattr(config, "enabled", None),
            "executable_path": _path_to_str(getattr(config, "executable_path", None)),
            "models_dir": _path_to_str(getattr(config, "models_dir", None)),
            "default_host": getattr(config, "default_host", None),
            "default_port": getattr(config, "default_port", None),
            "allowed_paths": [_path_to_str(path) for path in (getattr(config, "allowed_paths", None) or [])],
            "log_output_file": _path_to_str(getattr(config, "log_output_file", None)),
            "active_model": getattr(handler, "_active_server_model", None),
            "active_host": getattr(handler, "_active_server_host", None),
            "active_port": getattr(handler, "_active_server_port", None),
            "active_pid": _active_pid(handler),
        }
    )
    return active


def _restart_reasons(saved_config: dict[str, Any], active_config: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if saved_config.get("enabled") and not active_config.get("handler_configured"):
        reasons.append("handler_not_configured")
        return reasons
    if not active_config.get("handler_configured"):
        return reasons

    for saved_field in sorted(RESTART_FIELDS - {"allowed_paths"}):
        if _normalize_compare(saved_config.get(saved_field)) != _normalize_compare(active_config.get(saved_field)):
            reasons.append(f"{saved_field}_changed")

    if _normalize_list(saved_config.get("allowed_paths")) != _normalize_list(active_config.get("allowed_paths")):
        reasons.append("allowed_paths_changed")
    return reasons


def _warnings(
    saved_config: dict[str, Any],
    active_config: dict[str, Any],
    restart_reasons: list[str],
) -> list[str]:
    warnings: list[str] = []
    if saved_config.get("enabled") and not active_config.get("handler_configured"):
        warnings.append("Saved llama.cpp config is enabled, but no active handler is configured.")
    if restart_reasons:
        warnings.append("An API server restart may be required for saved llama.cpp config changes to take effect.")
    return warnings


def _payload_to_updates(payload: Any) -> dict[str, Any]:
    if hasattr(payload, "model_dump"):
        raw = payload.model_dump(exclude_none=True)
    elif isinstance(payload, dict):
        raw = {k: v for k, v in payload.items() if v is not None}
    else:
        raw = {}

    updates: dict[str, Any] = {}
    for field, value in raw.items():
        if field in _LIST_FIELDS and value is not None:
            updates[field] = ", ".join(str(item).strip() for item in value if str(item).strip())
        else:
            updates[field] = value
    return updates


def _probe_binary(path: Path, flag: str, timeout_seconds: float, warnings: list[str]) -> str | None:
    try:
        # No shell, bounded args, executable path prechecked above.
        completed = subprocess.run(  # nosec B603
            [str(path), flag],
            capture_output=True,
            check=False,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        warnings.append(f"Binary '{path.name}' timed out while probing {flag}.")
        return None
    except OSError:
        warnings.append(f"Binary '{path.name}' could not be executed for {flag}.")
        return None

    output = "\n".join(part.strip() for part in (completed.stdout, completed.stderr) if part and part.strip()).strip()
    if output:
        return output[:4000]
    if completed.returncode == 0:
        return ""
    return None


def _split_list(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [part.strip() for part in str(raw).replace(os.pathsep, ",").split(",") if part.strip()]


def _int_or_none(raw: str | None) -> int | None:
    if raw is None or not str(raw).strip():
        return None
    try:
        return int(str(raw).strip())
    except ValueError:
        return None


def _str_or_none(raw: str | None) -> str | None:
    if raw is None:
        return None
    value = str(raw).strip()
    return value or None


def _path_to_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _active_pid(handler: Any) -> int | None:
    process = getattr(handler, "_active_server_process", None)
    if process is not None and getattr(process, "returncode", None) is None:
        return getattr(process, "pid", None)
    return None


def _normalize_compare(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _normalize_list(value: Any) -> list[str]:
    if not value:
        return []
    return sorted(str(item) for item in value)
