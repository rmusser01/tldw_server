"""Executable discovery for optional LSP backends."""

from __future__ import annotations

import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .errors import LspToolError
from .models import LspBackendStatus

_BACKEND_EXECUTABLES = {"ruff": "ruff", "pylsp": "pylsp"}
_BACKEND_DEFAULT_ARGS = {"ruff": ("server",), "pylsp": ()}
_DENIED_WRAPPERS = frozenset({"docker", "devbox", "npx"})
_CONTROL_TOKENS = ("&&", "||", ";", "|", "$(", "`", ">", "<")


@dataclass(frozen=True, slots=True)
class ResolvedLspExecutable:
    """Result of resolving one backend executable."""

    backend_id: str
    argv: tuple[str, ...]
    source: str
    available: bool
    executable_path: str | None = None
    reason_code: str | None = None
    install_hint: str | None = None

    def to_status(self, *, capabilities: Sequence[str] = ()) -> LspBackendStatus:
        """Return a backend-status view for degraded service status payloads."""

        return LspBackendStatus(
            name=self.backend_id,
            healthy=self.available,
            capabilities=tuple(capabilities) if self.available else (),
            detail=None if self.available else self.install_hint,
        )


class LspExecutableResolver:
    """Resolve Ruff and pylsp commands from explicit config, venv, then PATH."""

    def __init__(
        self,
        *,
        workspace_root: Path,
        explicit_commands: Mapping[str, Sequence[str] | str] | None = None,
        path_env: str | None = None,
    ):
        self.workspace_root = workspace_root
        self.explicit_commands = dict(explicit_commands or {})
        self.path_env = os.environ.get("PATH", "") if path_env is None else path_env

    def resolve(self, backend_id: str) -> ResolvedLspExecutable:
        """Resolve one backend command without launching it."""

        if backend_id not in _BACKEND_EXECUTABLES:
            raise LspToolError("config_error", f"unsupported LSP backend: {backend_id}")

        if backend_id in self.explicit_commands:
            argv = _validate_command(self.explicit_commands[backend_id])
            return ResolvedLspExecutable(
                backend_id=backend_id,
                argv=argv,
                source="explicit",
                available=True,
                executable_path=argv[0],
            )

        venv_executable = self._venv_executable(backend_id)
        if venv_executable is not None:
            return self._available(backend_id, venv_executable, source="venv")

        path_executable = shutil.which(_BACKEND_EXECUTABLES[backend_id], path=self.path_env)
        if path_executable is not None:
            return self._available(backend_id, Path(path_executable), source="path")

        return ResolvedLspExecutable(
            backend_id=backend_id,
            argv=(),
            source="missing",
            available=False,
            reason_code="backend_missing",
            install_hint=f"Run `pip install mcp-unified[lsp]` or configure the {backend_id} executable.",
        )

    def _available(self, backend_id: str, executable: Path, *, source: str) -> ResolvedLspExecutable:
        executable_text = str(executable)
        return ResolvedLspExecutable(
            backend_id=backend_id,
            argv=(executable_text, *_BACKEND_DEFAULT_ARGS[backend_id]),
            source=source,
            available=True,
            executable_path=executable_text,
        )

    def _venv_executable(self, backend_id: str) -> Path | None:
        executable_name = _BACKEND_EXECUTABLES[backend_id]
        venv_root = self.workspace_root / ".venv"
        candidates = [
            venv_root / "bin" / executable_name,
            venv_root / "Scripts" / executable_name,
            venv_root / "Scripts" / f"{executable_name}.exe",
        ]
        for candidate in candidates:
            if _is_executable_file(candidate):
                return candidate
        return None


def _validate_command(command: Sequence[str] | str) -> tuple[str, ...]:
    if isinstance(command, str):
        raise LspToolError("config_error", "explicit LSP command must be an argv list, not a shell string")
    if not command:
        raise LspToolError("config_error", "explicit LSP command must not be empty")
    argv = tuple(command)
    if not all(isinstance(token, str) and token for token in argv):
        raise LspToolError("config_error", "explicit LSP command must contain non-empty strings")

    executable = argv[0]
    executable_name = executable.replace("\\", "/").rsplit("/", maxsplit=1)[-1].lower()
    if executable_name in _DENIED_WRAPPERS:
        raise LspToolError("config_error", f"LSP executable wrapper is not supported: {executable_name}")
    if any(token in executable for token in _CONTROL_TOKENS):
        raise LspToolError("config_error", "LSP executable contains shell control syntax")
    if len(argv) == 1 and " " in executable and not _is_pathlike(executable):
        raise LspToolError("config_error", "LSP executable looks like a shell command string")
    if _is_pathlike(executable) and not _is_executable_file(Path(executable).expanduser()):
        raise LspToolError("config_error", "configured LSP executable path is not executable")
    return argv


def _is_pathlike(value: str) -> bool:
    return Path(value).is_absolute() or "/" in value or "\\" in value


def _is_executable_file(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)
