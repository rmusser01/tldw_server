from __future__ import annotations

import contextlib
import json
import os
import shlex
import shutil
from dataclasses import dataclass, field

from loguru import logger

from tldw_Server_API.app.core.config import get_config_section
from tldw_Server_API.app.core.config_paths import resolve_config_root
from tldw_Server_API.app.core.testing import is_truthy


@dataclass
class ACPRunnerConfig:
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    startup_timeout_sec: float = 10.0
    binary_path: str | None = None


@dataclass
class ACPSandboxConfig:
    enabled: bool = False
    runtime: str = "docker"
    base_image: str = "tldw/acp-agent:latest"
    network_policy: str = "deny_all"
    allowed_egress_hosts: list[str] = field(default_factory=list)
    run_as_root: bool = False
    read_only_root: bool = True
    ssh_enabled: bool = True
    ssh_user: str = "acp"
    ssh_host: str = "127.0.0.1"
    ssh_container_port: int = 2222
    ssh_port_min: int = 2222
    ssh_port_max: int = 2299
    agent_command: str = ""
    agent_args: list[str] = field(default_factory=list)
    agent_env: dict[str, str] = field(default_factory=dict)

    # Session TTL and quotas
    session_ttl_seconds: int = 86400  # 24h default
    max_concurrent_sessions_per_user: int = 5
    max_tokens_per_session: int = 1_000_000
    max_session_duration_seconds: int = 14400  # 4h default
    audit_retention_days: int = 30


def _get_config_file_dir() -> str:
    """Return the directory containing config.txt (the Config_Files dir)."""
    return str(resolve_config_root())


def _resolve_cwd(raw_cwd: str | None) -> str | None:
    """Resolve ``runner_cwd`` relative to the config file directory.

    Absolute paths are returned unchanged.  Relative paths (those that
    do not start with ``/`` or a Windows drive letter like ``C:\\``) are
    resolved against the directory that contains ``config.txt`` so the
    result is stable regardless of the process working directory.
    """
    if not raw_cwd:
        return None

    cwd = raw_cwd.strip()
    if not cwd:
        return None

    # Already absolute - nothing to do.
    if os.path.isabs(cwd):
        return cwd

    config_dir = _get_config_file_dir()
    resolved = os.path.normpath(os.path.join(config_dir, cwd))
    logger.debug(
        "ACP runner_cwd resolved: '{}' -> '{}' (config dir: {})",
        raw_cwd,
        resolved,
        config_dir,
    )
    return resolved


def validate_acp_config(config: ACPRunnerConfig) -> list[str]:
    """Validate an ACP runner configuration and return warning messages.

    Returns an empty list when everything looks good.  Each string in the
    returned list is a human-readable warning with an actionable hint.
    """
    warnings: list[str] = []

    if not config.command:
        warnings.append(
            "ACP runner_command is empty - ACP sessions will not work. "
            "Set runner_command in [ACP] section of config.txt or via "
            "ACP_RUNNER_COMMAND environment variable."
        )
    else:
        # Check if the command is available on PATH or as a file
        found = shutil.which(config.command)
        if found is None and not os.path.isfile(config.command):
            warnings.append(
                f"ACP runner_command '{config.command}' was not found on PATH "
                f"or as a file. Ensure the binary is installed and accessible."
            )

    if config.cwd and not os.path.isdir(config.cwd):
        warnings.append(
            f"ACP runner_cwd directory does not exist: {config.cwd}"
        )

    return warnings


def _parse_args(raw: str | None) -> list[str]:
    if not raw:
        return []
    text = raw.strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return shlex.split(text)
        if isinstance(data, list):
            return [str(item) for item in data]
        return []
    return shlex.split(text)


def _parse_env(raw: str | None) -> dict[str, str]:
    if not raw:
        return {}
    text = raw.strip()
    if not text:
        return {}
    if text.startswith("{"):
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
        return {}
    pairs = [seg.strip() for seg in text.split(",") if seg.strip()]
    env: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            continue
        key, value = pair.split("=", 1)
        key = key.strip()
        if not key:
            continue
        env[key] = value.strip()
    return env


def _resolve_runner_env_paths(
    raw_env: dict[str, str],
    *,
    resolve_relative_home: bool = True,
) -> dict[str, str]:
    """Normalize ACP runner env paths that are config-relative by convention."""
    if not raw_env:
        return {}

    env = dict(raw_env)
    home = env.get("HOME")
    if not resolve_relative_home or not home:
        return env

    home = home.strip()
    if not home or os.path.isabs(home):
        return env

    config_dir = _get_config_file_dir()
    resolved_home = os.path.normpath(os.path.join(config_dir, home))
    env["HOME"] = resolved_home
    logger.debug(
        "ACP runner_env HOME resolved: '{}' -> '{}' (config dir: {})",
        home,
        resolved_home,
        config_dir,
    )
    return env


def load_acp_runner_config() -> ACPRunnerConfig:
    section = get_config_section("ACP")
    binary_path = os.getenv("ACP_RUNNER_BINARY_PATH") or section.get("runner_binary_path")
    command = os.getenv("ACP_RUNNER_COMMAND") or section.get("runner_command", "")
    args_raw = os.getenv("ACP_RUNNER_ARGS") or section.get("runner_args", "")
    env_override = os.getenv("ACP_RUNNER_ENV")
    has_env_override = env_override is not None
    env_raw = env_override if has_env_override else section.get("runner_env", "")
    cwd_override = os.getenv("ACP_RUNNER_CWD")
    has_cwd_override = cwd_override is not None
    cwd = cwd_override if has_cwd_override else section.get("runner_cwd")

    # If binary_path is set, use it as the command directly (shortcut)
    if binary_path and not command:
        command = str(binary_path)
        args_raw = args_raw or ""
        cwd = cwd or None

    timeout_raw = os.getenv("ACP_RUNNER_STARTUP_TIMEOUT_MS") or section.get(
        "startup_timeout_ms"
    )
    timeout_sec = 10.0
    if timeout_raw:
        with contextlib.suppress(TypeError, ValueError):
            timeout_sec = float(timeout_raw) / 1000.0

    resolved_cwd = _resolve_cwd(str(cwd) if cwd else None)
    runner_env = _resolve_runner_env_paths(
        _parse_env(env_raw),
        resolve_relative_home=not has_env_override,
    )

    return ACPRunnerConfig(
        command=str(command or ""),
        args=_parse_args(args_raw),
        env=runner_env,
        cwd=resolved_cwd,
        startup_timeout_sec=timeout_sec,
        binary_path=str(binary_path) if binary_path else None,
    )


def _parse_bool(raw: str | None, default: bool = False) -> bool:
    if raw is None:
        return default
    return is_truthy(raw)


def _parse_int(raw: str | None, default: int) -> int:
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _parse_string_list(raw: str | None) -> list[str]:
    """Parse a comma-separated or JSON array string into a list of strings."""
    if not raw:
        return []
    text = raw.strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return [s.strip() for s in text.split(",") if s.strip()]
        if isinstance(data, list):
            return [str(item) for item in data]
        return []
    return [s.strip() for s in text.split(",") if s.strip()]


##############################################################################
# Permission policy templates
# ---
# Preset policy configurations that serve as base layers for the permission
# system.  Loaded by ``ACPRuntimePolicyService.build_snapshot()`` and merged
# with user customisations (user overrides take precedence).
##############################################################################

PERMISSION_POLICY_TEMPLATES: dict[str, dict] = {
    "read-only": {
        "tool_tier_overrides": {
            "Read(*)": "auto",
            "Glob(*)": "auto",
            "Grep(*)": "auto",
            "Bash(git:log*)": "auto",
            "Bash(git:status*)": "auto",
            "Bash(git:diff*)": "auto",
            "*": "individual",
        },
        "description": "Read-only access. All write/exec tools require individual approval.",
    },
    "developer": {
        "tool_tier_overrides": {
            "Bash(git:*)": "auto",
            "Bash(npm:*)": "auto",
            "Bash(yarn:*)": "auto",
            "Bash(pip:*)": "auto",
            "Read(*)": "auto",
            "Glob(*)": "auto",
            "Grep(*)": "auto",
            "Write(*)": "batch",
            "Edit(*)": "batch",
            "Bash(rm:*)": "individual",
            "Bash(*)": "individual",
        },
        "description": "Developer access. Git/npm auto, file writes batch, shell exec individual.",
    },
    "admin": {
        "tool_tier_overrides": {
            "Bash(rm:-rf*)": "individual",
            "Bash(git:push*--force*)": "individual",
            "Bash(git:reset*--hard*)": "individual",
            "Bash(drop*)": "individual",
            "*": "auto",
        },
        "description": "Admin access. Everything auto except destructive operations.",
    },
    "lockdown": {
        "tool_tier_overrides": {
            "*": "individual",
        },
        "description": "Full lockdown. Every tool requires individual approval.",
    },
}


def load_acp_sandbox_config() -> ACPSandboxConfig:
    section = get_config_section("ACP-SANDBOX")
    enabled = _parse_bool(os.getenv("ACP_SANDBOX_ENABLED") or section.get("enabled"), False)
    runtime = os.getenv("ACP_SANDBOX_RUNTIME") or section.get("runtime", "docker")
    base_image = os.getenv("ACP_SANDBOX_BASE_IMAGE") or section.get("base_image", "tldw/acp-agent:latest")
    network_policy = os.getenv("ACP_SANDBOX_NETWORK_POLICY") or section.get("network_policy", "deny_all")
    allowed_egress_raw = os.getenv("ACP_SANDBOX_ALLOWED_EGRESS_HOSTS") or section.get("allowed_egress_hosts", "")
    run_as_root = _parse_bool(os.getenv("ACP_SANDBOX_RUN_AS_ROOT") or section.get("run_as_root"), False)
    read_only_root = _parse_bool(os.getenv("ACP_SANDBOX_READ_ONLY_ROOT") or section.get("read_only_root"), True)
    ssh_enabled = _parse_bool(os.getenv("ACP_SSH_ENABLED") or section.get("ssh_enabled"), True)
    ssh_user = os.getenv("ACP_SSH_USER") or section.get("ssh_user", "acp")
    ssh_host = os.getenv("ACP_SSH_HOST") or section.get("ssh_host", "127.0.0.1")
    ssh_container_port = _parse_int(
        os.getenv("ACP_SSH_CONTAINER_PORT") or section.get("ssh_container_port"),
        2222,
    )
    ssh_port_min = _parse_int(os.getenv("ACP_SSH_PORT_MIN") or section.get("ssh_port_min"), 2222)
    ssh_port_max = _parse_int(os.getenv("ACP_SSH_PORT_MAX") or section.get("ssh_port_max"), 2299)
    agent_command = os.getenv("ACP_SANDBOX_AGENT_COMMAND") or section.get("agent_command", "")
    agent_args_raw = os.getenv("ACP_SANDBOX_AGENT_ARGS") or section.get("agent_args", "")
    agent_env_raw = os.getenv("ACP_SANDBOX_AGENT_ENV") or section.get("agent_env", "")

    # Session management config
    session_ttl = _parse_int(
        os.getenv("ACP_SESSION_TTL_SECONDS") or section.get("session_ttl_seconds"), 86400
    )
    max_concurrent = _parse_int(
        os.getenv("ACP_MAX_CONCURRENT_SESSIONS_PER_USER") or section.get("max_concurrent_sessions_per_user"), 5
    )
    max_tokens = _parse_int(
        os.getenv("ACP_MAX_TOKENS_PER_SESSION") or section.get("max_tokens_per_session"), 1_000_000
    )
    max_duration = _parse_int(
        os.getenv("ACP_MAX_SESSION_DURATION_SECONDS") or section.get("max_session_duration_seconds"), 14400
    )
    audit_retention = _parse_int(
        os.getenv("ACP_AUDIT_RETENTION_DAYS") or section.get("audit_retention_days"), 30
    )

    return ACPSandboxConfig(
        enabled=bool(enabled),
        runtime=str(runtime or "docker"),
        base_image=str(base_image or "tldw/acp-agent:latest"),
        network_policy=str(network_policy or "deny_all"),
        allowed_egress_hosts=_parse_string_list(allowed_egress_raw),
        run_as_root=bool(run_as_root),
        read_only_root=bool(read_only_root),
        ssh_enabled=bool(ssh_enabled),
        ssh_user=str(ssh_user or "acp"),
        ssh_host=str(ssh_host or "127.0.0.1"),
        ssh_container_port=int(ssh_container_port),
        ssh_port_min=int(ssh_port_min),
        ssh_port_max=int(ssh_port_max),
        agent_command=str(agent_command or ""),
        agent_args=_parse_args(agent_args_raw),
        agent_env=_parse_env(agent_env_raw),
        session_ttl_seconds=session_ttl,
        max_concurrent_sessions_per_user=max_concurrent,
        max_tokens_per_session=max_tokens,
        max_session_duration_seconds=max_duration,
        audit_retention_days=audit_retention,
    )
