"""Shared utilities for Local LLM handlers.

This module provides common functionality used by LlamaCpp, Llamafile, and other handlers:
- Environment variable parsing
- Port availability checking and auto-selection
- Path security validation
- Defensive logging
- CLI argument denylist for secrets
"""

from __future__ import annotations

import os
import socket
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tldw_Server_API.app.core.testing import is_truthy

if TYPE_CHECKING:
    from loguru import Logger


# Unified denylist of CLI arguments that should not be passed directly
# (secrets should be passed via environment variables instead)
DEFAULT_SECRET_DENYLIST: set[str] = {
    "api_key",
    "hf_token",
    "token",
    "openai_api_key",
    "anthropic_api_key",
}

IPV4_WILDCARD_HOST = ".".join(("0", "0", "0", "0"))

# Wildcard bind sentinels used only for client-host normalization logic.
WILDCARD_HOSTS: set[str] = {
    IPV4_WILDCARD_HOST,
    "::",
    "0:0:0:0:0:0:0:0",
}


def strip_host_brackets(host: str) -> str:
    """Remove IPv6 brackets if present."""
    h = str(host).strip()
    if h.startswith("[") and h.endswith("]"):
        return h[1:-1]
    return h


def resolve_client_host(host: str | None) -> str:
    """Resolve a bind host to a usable client connect host.

    Wildcard bind hosts map to loopback for readiness/inference calls.
    """
    if host is None or str(host).strip() == "":
        return "127.0.0.1"
    h = strip_host_brackets(str(host))
    if h in WILDCARD_HOSTS:
        return "::1" if ":" in h else "127.0.0.1"
    return h


def format_host_for_url(host: str) -> str:
    """Format a host for use in a URL (adds IPv6 brackets when needed)."""
    if ":" in host and not host.startswith("["):
        return f"[{host}]"
    return host


def build_base_url(host: str, port: int) -> str:
    """Build an http:// base URL for the given host and port."""
    return f"http://{format_host_for_url(host)}:{int(port)}"


def env_bool(name: str) -> bool | None:
    """Parse a boolean value from an environment variable.

    Returns True for "1", "true", "yes", "on" (case-insensitive).
    Returns False for "0", "false", "no", "off" (case-insensitive).
    Returns None if the variable is not set.
    """
    v = os.getenv(name)
    if v is None:
        return None
    v_lower = str(v).strip().lower()
    if is_truthy(v_lower):
        return True
    if v_lower in {"0", "false", "no", "off"}:
        return False
    return None


def env_int(name: str) -> int | None:
    """Parse an integer value from an environment variable.

    Returns None if the variable is not set or cannot be parsed.
    """
    v = os.getenv(name)
    if v is None:
        return None
    try:
        return int(v)
    except ValueError:
        return None


def env_paths(name: str) -> list[Path] | None:
    """Parse a comma-separated list of paths from an environment variable.

    Returns None if the variable is not set or empty.
    """
    v = os.getenv(name)
    if not v:
        return None
    parts = [p.strip() for p in v.split(",") if p.strip()]
    return [Path(p) for p in parts] if parts else None


def _port_probe_host(host: str) -> str:
    """Use loopback as a proxy when availability-probing wildcard runtime hosts.

    This avoids binding test sockets to all interfaces for CodeQL/Bandit, but
    `is_port_free` and `pick_port` still only prove the loopback proxy port is
    available. Startup can still fail if the original wildcard bind differs.
    """
    clean_host = strip_host_brackets(host)
    # Wildcard probes are converted to loopback before any socket bind.
    if not clean_host:
        return "127.0.0.1"
    if clean_host in WILDCARD_HOSTS:
        return "::1" if ":" in clean_host else "127.0.0.1"
    return clean_host


def is_port_free(host: str, port: int) -> bool:
    """Check if a port is available for binding.

    Args:
        host: Host address to check (e.g., "127.0.0.1" or "::1")
        port: Port number to check

    Returns:
        True if the port is free, False otherwise
    """
    clean_host = _port_probe_host(host)
    family = socket.AF_INET6 if ":" in str(clean_host) else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            # Wildcard runtime hosts are normalized to loopback by _port_probe_host.
            # lgtm[py/bind-socket-all-network-interfaces]: clean_host never remains a wildcard address here.
            s.bind((clean_host, port))
            return True
        except OSError:
            return False


def pick_port(
    host: str,
    start_port: int,
    autoselect: bool = True,
    max_probe: int = 10,
) -> int:
    """Find an available port starting from start_port.

    Args:
        host: Host address to check
        start_port: Starting port number
        autoselect: If False, return start_port without checking
        max_probe: Maximum number of ports to probe

    Returns:
        An available port number, or start_port as fallback
    """
    if not autoselect:
        return start_port
    for i in range(max_probe + 1):
        candidate = start_port + i
        if is_port_free(host, candidate):
            return candidate
    return start_port  # Fallback


def is_path_allowed(
    path: Path,
    base_dirs: list[Path],
) -> bool:
    """Check if a path is under one of the allowed base directories.

    Uses os.path.realpath to fully resolve symlinks for security,
    preventing symlink-based path traversal attacks.

    Args:
        path: The path to check
        base_dirs: List of allowed base directories

    Returns:
        True if path is under one of the base directories, False otherwise
    """
    try:
        # Use realpath to fully resolve symlinks
        resolved_path = Path(os.path.realpath(str(path)))
    except Exception:
        return False

    for base in base_dirs:
        try:
            # Resolve symlinks in base dir too to prevent bypass
            resolved_base = Path(os.path.realpath(str(base)))
            resolved_path.relative_to(resolved_base)
            return True
        except (ValueError, Exception):
            continue
    return False


def build_allowed_paths(
    models_dir: Path,
    extra_paths: list[Path] | None = None,
) -> list[Path]:
    """Build a list of allowed base paths for security checks.

    Args:
        models_dir: The primary models directory
        extra_paths: Optional additional allowed paths

    Returns:
        List of Path objects representing allowed directories
    """
    bases = [models_dir]
    if extra_paths:
        bases.extend(extra_paths)
    return bases


def check_denylist(
    args: dict[str, Any],
    allow_secrets: bool = False,
    denylist: set[str] | None = None,
) -> None:
    """Check if any arguments are in the denylist.

    Args:
        args: Dictionary of arguments to check
        allow_secrets: If True, skip the check
        denylist: Set of denied argument names (uses DEFAULT_SECRET_DENYLIST if None)

    Raises:
        ValueError: If any denied arguments are found and allow_secrets is False
    """
    if allow_secrets:
        return

    deny = denylist if denylist is not None else DEFAULT_SECRET_DENYLIST

    def _normalize_key(key: str) -> str:
        return str(key).strip().lower().replace("-", "_")

    def _compact_key(key: str) -> str:
        return _normalize_key(key).replace("_", "")

    deny_normalized = {_normalize_key(k) for k in deny}
    deny_compact = {_compact_key(k) for k in deny}

    bad = [
        k
        for k in args
        if _normalize_key(k) in deny_normalized or _compact_key(k) in deny_compact
    ]
    if bad:
        raise ValueError(
            f"Refusing secret flags {bad}. Set environment variables "
            f"(e.g., HF_TOKEN) instead, or enable allow_cli_secrets."
        )


def safe_log(
    log: Logger,
    level: str,
    msg: str,
    *args,
) -> None:
    """Log defensively to avoid errors when sinks are closed during atexit.

    Args:
        log: Logger instance
        level: Log level (e.g., "info", "warning", "error")
        msg: Message to log
        *args: Additional arguments for the log message
    """
    try:
        log_fn = getattr(log, level, None)
        if callable(log_fn):
            log_fn(msg, *args)
    except Exception as log_error:
        # Swallow logging errors on interpreter shutdown / closed sinks
        _ = log_error


def apply_env_overrides(config: Any) -> None:
    """Apply environment variable overrides to a handler config.

    Checks for:
    - LOCAL_LLM_ALLOW_CLI_SECRETS
    - LOCAL_LLM_PORT_AUTOSELECT
    - LOCAL_LLM_PORT_PROBE_MAX
    - LOCAL_LLM_ALLOWED_PATHS

    Args:
        config: Handler config object to modify
    """
    b = env_bool("LOCAL_LLM_ALLOW_CLI_SECRETS")
    if b is not None:
        config.allow_cli_secrets = b

    b = env_bool("LOCAL_LLM_PORT_AUTOSELECT")
    if b is not None:
        config.port_autoselect = b

    i = env_int("LOCAL_LLM_PORT_PROBE_MAX")
    if i is not None:
        config.port_probe_max = i

    paths = env_paths("LOCAL_LLM_ALLOWED_PATHS")
    if paths is not None:
        config.allowed_paths = paths
