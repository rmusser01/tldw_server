from __future__ import annotations

import os
# Validation probes a user-selected local binary without shell.
import subprocess  # nosec B404
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from tldw_Server_API.app.core.Local_LLM.llamacpp_config_lock import LockAcquisitionError, llamacpp_config_write_lock
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
    "imported_asset_folders",
}
_LIST_VALUE_DELIMITERS = {",", os.pathsep}
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
    "imported_asset_folders",
    "log_output_file",
)


def get_env_overrides() -> dict[str, bool]:
    """Return env override state for each typed llama.cpp config field."""
    return {field: os.getenv(env_name) is not None for field, env_name in LLAMACPP_ENV_OVERRIDES.items()}


def get_config_state(llm_manager: Any) -> dict[str, Any]:
    """Return saved config, active handler config, restart signals, and warnings."""
    parse_warnings: list[str] = []
    saved_config = _read_saved_config(parse_warnings)
    active_config = _read_active_config(llm_manager)
    restart_reasons = _restart_reasons(saved_config, active_config)
    warnings = parse_warnings + _warnings(saved_config, active_config, restart_reasons)

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

    try:
        with llamacpp_config_write_lock():
            setup_manager.update_config({"LlamaCpp": updates})
            refresh_config_cache()
    except HTTPException:
        raise
    except LockAcquisitionError as exc:
        raise HTTPException(status_code=500, detail="Failed to update llama.cpp configuration.") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to update llama.cpp configuration.") from exc
    return get_config_state(llm_manager)


def validate_binary(
    binary_path: str,
    timeout_seconds: float = 3.0,
    *,
    llm_manager: Any | None = None,
    run_probe: bool = False,
) -> dict[str, Any]:
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
    probe_allowed = True
    if run_probe and exists and executable:
        probe_allowed = _matches_saved_or_active_executable(path, llm_manager)
        if not probe_allowed:
            warnings.append("Binary probe requires the path to be saved first.")
    if run_probe and exists and executable and probe_allowed:
        version_output = _probe_binary(path, "--version", timeout_seconds, warnings)
        if version_output is None:
            help_output = _probe_binary(path, "--help", timeout_seconds, warnings)
        if version_output is None and help_output is None:
            warnings.append(f"Binary '{path.name}' did not return version or help output.")

    probe_succeeded = version_output is not None or help_output is not None
    return {
        "valid": bool(exists and executable and (not run_probe or (probe_allowed and probe_succeeded))),
        "exists": bool(exists),
        "executable": bool(executable),
        "resolved_path": resolved_path,
        "version_output": version_output,
        "help_output": help_output,
        "warnings": warnings,
    }


def _read_saved_config(warnings: list[str] | None = None) -> dict[str, Any]:
    parser = load_comprehensive_config()
    section = parser["LlamaCpp"] if parser and parser.has_section("LlamaCpp") else None
    saved: dict[str, Any] = {
        "enabled": False,
        "allowed_paths": [],
        "registered_model_paths": [],
        "imported_asset_folders": [],
    }
    if section is None:
        return saved

    for field in _SAVED_FIELDS:
        if field in _BOOL_FIELDS:
            saved[field] = _bool_or_default(section.get(field, fallback=None), field, warnings, default=False)
        elif field in _INT_FIELDS:
            raw = section.get(field, fallback=None)
            saved[field] = _int_or_none(raw, field, warnings)
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
            "default_threads": getattr(config, "default_threads", None),
            "default_n_gpu_layers": getattr(config, "default_n_gpu_layers", None),
            "default_ctx_size": getattr(config, "default_ctx_size", None),
            "allow_unvalidated_args": getattr(config, "allow_unvalidated_args", None),
            "allow_cli_secrets": getattr(config, "allow_cli_secrets", None),
            "port_autoselect": getattr(config, "port_autoselect", None),
            "port_probe_max": getattr(config, "port_probe_max", None),
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
        fields_set = getattr(payload, "model_fields_set", set()) or getattr(payload, "__fields_set__", set())
        raw = {field: getattr(payload, field) for field in fields_set}
    elif isinstance(payload, dict):
        raw = {k: v for k, v in payload.items() if v is not None}
    else:
        raw = {}

    updates: dict[str, Any] = {}
    for field, value in raw.items():
        if value is None:
            updates[field] = ""
        elif field in _LIST_FIELDS:
            values = [value] if isinstance(value, str) else value
            updates[field] = ", ".join(
                item for item in (_validate_list_config_value(field, item) for item in values) if item
            )
        else:
            _validate_config_value(field, value)
            updates[field] = value
    return updates


def _validate_config_value(field: str, value: Any) -> str:
    try:
        return setup_manager.validate_config_value_single_line("LlamaCpp", field, value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _validate_list_config_value(field: str, value: Any) -> str:
    text = _validate_config_value(field, value).strip()
    if text and any(delimiter in text for delimiter in _LIST_VALUE_DELIMITERS):
        raise HTTPException(
            status_code=400,
            detail=f"LlamaCpp.{field} entries cannot contain comma or path separator characters.",
        )
    return text


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


def _bool_or_default(
    raw: str | None,
    field: str,
    warnings: list[str] | None,
    *,
    default: bool,
) -> bool:
    if raw is None or not str(raw).strip():
        return default
    lowered = str(raw).strip().lower()
    if lowered in {"true", "yes", "on", "1"}:
        return True
    if lowered in {"false", "no", "off", "0"}:
        return False
    if warnings is not None:
        warnings.append(f"Invalid boolean value for LlamaCpp.{field}.")
    return default


def _int_or_none(raw: str | None, field: str | None = None, warnings: list[str] | None = None) -> int | None:
    if raw is None or not str(raw).strip():
        return None
    try:
        return int(str(raw).strip())
    except ValueError:
        if warnings is not None and field is not None:
            warnings.append(f"Invalid integer value for LlamaCpp.{field}.")
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


def _matches_saved_or_active_executable(path: Path, llm_manager: Any | None) -> bool:
    requested = _resolve_for_compare(path)
    candidates: list[str] = []

    try:
        saved_config = _read_saved_config([])
    except Exception:
        saved_config = {}
    saved_executable = saved_config.get("executable_path")
    if saved_executable:
        candidates.append(str(saved_executable))

    handler = getattr(llm_manager, "llamacpp", None) if llm_manager is not None else None
    active_config = getattr(handler, "config", None) if handler is not None else None
    active_executable = getattr(active_config, "executable_path", None) if active_config is not None else None
    if active_executable:
        candidates.append(str(active_executable))

    return any(requested == _resolve_for_compare(Path(candidate).expanduser()) for candidate in candidates)


def _resolve_for_compare(path: Path) -> str:
    return str(path.expanduser().resolve())
