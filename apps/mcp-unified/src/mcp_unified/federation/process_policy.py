"""Process execution policy helpers for upstream stdio MCP transports."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SHELL_EXECUTABLES = frozenset(
    {
        "sh",
        "bash",
        "zsh",
        "fish",
        "cmd",
        "cmd.exe",
        "powershell",
        "powershell.exe",
        "pwsh",
        "pwsh.exe",
    }
)


@dataclass(frozen=True, slots=True)
class StdioProcessPolicy:
    """Deployment-level process policy for upstream stdio MCP servers."""

    allowed_executables: tuple[str, ...] = ()
    allowed_cwd_roots: tuple[str | Path, ...] = ()
    allowed_env_names: tuple[str, ...] | None = None
    allow_path_lookup: bool = True
    reject_shell_executables: bool = True
    default_cwd: str | Path | None = None

    def __post_init__(self) -> None:
        """Normalize direct constructor inputs into immutable safe values."""

        object.__setattr__(
            self,
            "allowed_executables",
            _coerce_text_tuple(
                self.allowed_executables,
                field_name="allowed_executables",
            ),
        )
        object.__setattr__(
            self,
            "allowed_cwd_roots",
            _coerce_path_tuple(
                self.allowed_cwd_roots,
                field_name="allowed_cwd_roots",
            ),
        )
        object.__setattr__(
            self,
            "allowed_env_names",
            (
                None
                if self.allowed_env_names is None
                else _coerce_text_tuple(
                    self.allowed_env_names,
                    field_name="allowed_env_names",
                )
            ),
        )
        object.__setattr__(
            self,
            "allow_path_lookup",
            _coerce_bool(self.allow_path_lookup, field_name="allow_path_lookup"),
        )
        object.__setattr__(
            self,
            "reject_shell_executables",
            _coerce_bool(
                self.reject_shell_executables,
                field_name="reject_shell_executables",
            ),
        )
        object.__setattr__(
            self,
            "default_cwd",
            _coerce_optional_path(self.default_cwd, field_name="default_cwd"),
        )


@dataclass(frozen=True, slots=True)
class StdioProcessDecision:
    """Normalized process-policy decision consumed by the stdio transport."""

    cwd: str | None
    allowed_env_names: tuple[str, ...] | None


class StdioProcessPolicyViolation(ValueError):
    """Raised when process policy denies a stdio server before spawn."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.details = dict(details or {})


def coerce_stdio_process_policy(
    value: StdioProcessPolicy | Mapping[str, Any] | None,
) -> StdioProcessPolicy:
    """Return a normalized stdio process policy from config or caller input."""

    if value is None:
        return StdioProcessPolicy()
    if isinstance(value, StdioProcessPolicy):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("process_policy must be a mapping, StdioProcessPolicy, or None")
    supported = {
        "allowed_executables",
        "allowed_cwd_roots",
        "allowed_env_names",
        "allow_path_lookup",
        "reject_shell_executables",
        "default_cwd",
    }
    unknown = sorted(str(key) for key in set(value) - supported)
    if unknown:
        raise ValueError(f"process_policy has unsupported fields: {', '.join(unknown)}")
    return StdioProcessPolicy(**dict(value))


def validate_stdio_process_policy(
    *,
    server_id: str,
    command: tuple[str, ...],
    cwd: str | None,
    env_allowlist: Sequence[str],
    policy: StdioProcessPolicy,
) -> StdioProcessDecision:
    """Validate one stdio launch request against deployment process policy."""

    if not command:
        raise StdioProcessPolicyViolation(
            "External stdio process policy requires a command",
            reason_code="missing_command",
            details={"server_id": server_id, "field": "command"},
        )

    executable = command[0]
    executable_name = _executable_basename(executable)
    allowed_executable = _executable_allowed(
        executable,
        policy=policy,
        cwd=cwd,
    )
    if (
        policy.reject_shell_executables
        and executable_name.lower() in _SHELL_EXECUTABLES
        and not allowed_executable
    ):
        raise StdioProcessPolicyViolation(
            "External stdio process policy denied shell executable",
            reason_code="process_policy_shell_denied",
            details={
                "server_id": server_id,
                "field": "command",
                "executable_name": executable_name,
            },
        )

    if policy.allowed_executables and not allowed_executable:
        raise StdioProcessPolicyViolation(
            "External stdio process policy denied executable",
            reason_code="process_policy_executable_denied",
            details={
                "server_id": server_id,
                "field": "command",
                "executable_name": executable_name,
            },
        )

    bare_executable = _is_bare_executable(executable)
    normalized_env = _normalize_env_names(env_allowlist)
    allowed_env_names = policy.allowed_env_names
    if bare_executable and not policy.allow_path_lookup:
        raise StdioProcessPolicyViolation(
            "External stdio process policy denied PATH lookup",
            reason_code="process_policy_path_lookup_denied",
            details={
                "server_id": server_id,
                "field": "command",
                "executable_name": executable_name,
            },
        )
    if bare_executable:
        _require_allowed_env_name(
            "PATH",
            server_id=server_id,
            normalized_env=normalized_env,
            allowed_env_names=allowed_env_names,
        )

    if allowed_env_names is not None:
        allowed_set = set(allowed_env_names)
        for env_name in normalized_env:
            if env_name not in allowed_set:
                raise StdioProcessPolicyViolation(
                    "External stdio process policy denied environment inheritance",
                    reason_code="process_policy_env_denied",
                    details={
                        "server_id": server_id,
                        "field": "env_allowlist",
                        "env_name": env_name,
                    },
                )

    if policy.allowed_cwd_roots:
        if cwd is None:
            raise StdioProcessPolicyViolation(
                "External stdio process policy requires a bounded cwd",
                reason_code="process_policy_cwd_denied",
                details={"server_id": server_id, "field": "cwd"},
            )
        cwd_path = Path(cwd).resolve(strict=False)
        if not any(_path_is_relative_to(cwd_path, root) for root in policy.allowed_cwd_roots):
            raise StdioProcessPolicyViolation(
                "External stdio process policy denied cwd",
                reason_code="process_policy_cwd_denied",
                details={"server_id": server_id, "field": "cwd"},
            )

    return StdioProcessDecision(cwd=cwd, allowed_env_names=allowed_env_names)


def _require_allowed_env_name(
    env_name: str,
    *,
    server_id: str,
    normalized_env: tuple[str, ...],
    allowed_env_names: tuple[str, ...] | None,
) -> None:
    """Require an inherited env name in both server and policy allowlists."""

    if env_name not in normalized_env:
        raise StdioProcessPolicyViolation(
            "External stdio command requires PATH inheritance",
            reason_code="invalid_command",
            details={"server_id": server_id, "field": "env_allowlist", "env_name": env_name},
        )
    if allowed_env_names is not None and env_name not in allowed_env_names:
        raise StdioProcessPolicyViolation(
            "External stdio process policy denied PATH inheritance",
            reason_code="process_policy_env_denied",
            details={"server_id": server_id, "field": "env_allowlist", "env_name": env_name},
        )


def _executable_allowed(
    executable: str,
    *,
    policy: StdioProcessPolicy,
    cwd: str | None,
) -> bool:
    """Return whether the executable satisfies the configured allowlist.

    Bare allowlist entries only match bare command names. A command that names a
    relative or absolute path must match a path allowlist entry after resolution,
    which prevents basename-only entries from authorizing arbitrary binaries.
    """

    if not policy.allowed_executables:
        return False

    executable_has_path = _has_path_separator(executable)
    executable_name = _normcase(_executable_basename(executable))
    executable_path = (
        _resolve_executable_path(executable, cwd=cwd)
        if executable_has_path
        else None
    )
    for allowed in policy.allowed_executables:
        if _has_path_separator(allowed) or Path(allowed).is_absolute():
            if (
                executable_path is not None
                and executable_path == _resolve_path_key(allowed)
            ):
                return True
            continue
        if (
            not executable_has_path
            and executable_name == _normcase(_executable_basename(allowed))
        ):
            return True
    return False


def _resolve_executable_path(executable: str, *, cwd: str | None) -> str:
    """Resolve an executable path relative to cwd or the current process cwd."""

    path = Path(executable).expanduser()
    if not path.is_absolute():
        base = Path(cwd).expanduser() if cwd is not None else Path.cwd()
        path = base / path
    return _resolve_path_key(path)


def _resolve_path_key(path: str | Path) -> str:
    """Return the normalized comparison key for a process filesystem path."""

    return _normcase(str(Path(path).expanduser().resolve(strict=False)))


def _path_is_relative_to(child: Path, root: str | Path) -> bool:
    """Return whether child resolves inside root using platform path semantics."""

    child_key = _resolve_path_key(child)
    root_key = _resolve_path_key(root)
    try:
        return os.path.commonpath([child_key, root_key]) == root_key
    except ValueError:
        return False


def _normalize_env_names(values: Sequence[str]) -> tuple[str, ...]:
    """Normalize env allowlist names while dropping blank values."""

    return tuple(str(value).strip() for value in values if str(value).strip())


def _is_bare_executable(executable: str) -> bool:
    """Return true when an executable command relies on PATH lookup."""

    return not _has_path_separator(executable)


def _has_path_separator(value: str) -> bool:
    """Return true when value contains POSIX or Windows path separators."""

    return "/" in value or "\\" in value


def _executable_basename(executable: str) -> str:
    """Return the final path component for POSIX or Windows-style commands."""

    return executable.replace("\\", "/").rsplit("/", maxsplit=1)[-1].strip()


def _normcase(value: str) -> str:
    """Normalize path case only on platforms where paths are case-insensitive."""

    return os.path.normcase(value)


def _coerce_text_tuple(value: Any, *, field_name: str) -> tuple[str, ...]:
    """Coerce a config sequence into a tuple of non-empty text values."""

    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"{field_name} must be a list of non-empty strings")
    items: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{field_name} entries must be strings")
        text = item.strip()
        if not text:
            raise ValueError(f"{field_name} entries cannot be empty")
        items.append(text)
    return tuple(items)


def _coerce_path_tuple(value: Any, *, field_name: str) -> tuple[str | Path, ...]:
    """Coerce a config sequence into a tuple of non-empty path values."""

    if isinstance(value, (str, Path)) or not isinstance(value, Sequence):
        raise ValueError(f"{field_name} must be a list of non-empty paths")
    items: list[str | Path] = []
    for item in value:
        if not isinstance(item, (str, Path)):
            raise ValueError(f"{field_name} entries must be paths")
        if not str(item).strip():
            raise ValueError(f"{field_name} entries cannot be empty")
        items.append(item)
    return tuple(items)


def _coerce_optional_path(value: Any, *, field_name: str) -> str | Path | None:
    """Coerce an optional config value into a non-empty path or None."""

    if value is None:
        return None
    if not isinstance(value, (str, Path)):
        raise ValueError(f"{field_name} must be a path or None")
    if not str(value).strip():
        raise ValueError(f"{field_name} cannot be empty")
    return value


def _coerce_bool(value: Any, *, field_name: str) -> bool:
    """Coerce a config value that must already be a boolean."""

    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean")
    return value


__all__ = [
    "StdioProcessDecision",
    "StdioProcessPolicy",
    "StdioProcessPolicyViolation",
    "coerce_stdio_process_policy",
    "validate_stdio_process_policy",
]
