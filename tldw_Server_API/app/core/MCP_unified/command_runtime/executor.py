"""Raw command execution for the virtual CLI runtime."""

from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
import stat
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Protocol

from .models import (
    CommandChain,
    CommandExecutionResult,
    CommandExecutionStep,
    CommandSpillReference,
    CommandStepResult,
)

_DEFAULT_SPILL_ROOT_PREFIX = "mcp-command-execution-"
_DEFAULT_SPILL_ROOT_PRUNE_INTERVAL_SECONDS = 300.0


class CommandBackend(Protocol):
    """Backend contract used by the raw runtime executor."""

    async def execute(
        self,
        argv: list[str],
        stdin: Any,
        handler_context: Any | None = None,
    ) -> CommandStepResult:
        """Execute one command and return its raw stdout/stderr and exit code.

        Backends may receive raw text/bytes or a CommandSpillReference when a
        previous pipeline stage spilled oversized stdin to disk.
        """


@dataclass(frozen=True, slots=True)
class HandlerInvocationContext:
    """Execution metadata for one lexical command step."""

    runtime_context: Any | None = None
    segment_index: int = 0
    command_index: int = 0
    lexical_step_index: int = 0


@dataclass(slots=True)
class CommandRuntimeExecutor:
    """Execute parsed command chains with Unix-like chaining semantics."""

    backend: CommandBackend
    spill_dir: Path | str | None = None
    spill_threshold_bytes: int = 65_536
    preview_bytes: int = 4_096
    spill_retention_seconds: int = 3_600
    _default_spill_root_last_prune_by_parent: ClassVar[dict[str, float]] = {}
    _default_spill_root_prune_lock: ClassVar[threading.Lock] = threading.Lock()
    _default_spill_root: Path | None = field(init=False, default=None, repr=False)

    async def execute(self, chain: CommandChain) -> CommandExecutionResult:
        """Run a parsed command chain and preserve raw stream semantics."""

        start = time.perf_counter()
        step_indices = self._build_step_indices(chain)
        last_exit_code = 0
        last_stdout: Any = ""
        last_stderr: Any = ""
        last_stdout_spill: CommandSpillReference | None = None
        last_stderr_spill: CommandSpillReference | None = None
        last_stdout_binary = False
        last_stderr_binary = False
        steps: list[CommandExecutionStep] = []
        aggregate_stderr_parts: list[Any] = []
        aggregate_raw_stderr_parts: list[Any] = []
        aggregate_stderr_spills: list[CommandSpillReference] = []
        aggregate_stderr_binary = False
        aggregate_textual_stderr = False
        latest_stderr_spill: CommandSpillReference | None = None

        for segment_index, segment in enumerate(chain.segments):
            if segment_index > 0:
                operator = chain.links[segment_index - 1]
                if operator == "&&" and last_exit_code != 0:
                    continue
                if operator == "||" and last_exit_code == 0:
                    continue

            pipeline_stdin = ""
            segment_stdout_spill: CommandSpillReference | None = None
            segment_stderr_spill: CommandSpillReference | None = None
            segment_stdout_binary = False
            segment_exit_code = 0
            segment_terminal_stdout: Any = ""

            for command_index, command in enumerate(segment.commands):
                step_started = time.perf_counter()
                step_stdin = pipeline_stdin
                step_result = await self.backend.execute(
                    list(command.argv),
                    step_stdin,
                    HandlerInvocationContext(
                        segment_index=segment_index,
                        command_index=command_index,
                        lexical_step_index=step_indices[(segment_index, command_index)],
                    ),
                )
                step_stdout, step_stdout_binary = self._classify_stream(step_result.stdout)
                step_stderr, step_stderr_binary = self._classify_stream(step_result.stderr)
                step_stderr_contains_binary = step_stderr_binary
                preserved_binary_stderr_spill: CommandSpillReference | None = None
                if step_stdout_binary and command_index < len(segment.commands) - 1:
                    if step_stderr_binary and self._has_content(step_stderr):
                        preserved_binary_stderr_spill = await self._spill_payload_async(
                            step_stderr,
                            kind="stderr",
                            force=True,
                        )
                        step_stderr = self._binary_pipe_error("")
                    else:
                        step_stderr = self._binary_pipe_error(step_stderr)
                    step_stderr_binary = False
                    step_result = CommandStepResult(
                        stdout=step_result.stdout,
                        stderr=step_stderr,
                        exit_code=1,
                        metadata=step_result.metadata,
                )
                segment_exit_code = int(step_result.exit_code)
                segment_stdout_binary = step_stdout_binary
                segment_terminal_stdout = step_stdout
                segment_stdout_spill = await self._spill_payload_async(
                    step_stdout,
                    kind="stdout",
                )
                stored_stdout = self._empty_like(step_stdout) if segment_stdout_spill is not None else step_stdout
                pipeline_stdin = segment_stdout_spill if segment_stdout_spill is not None else step_stdout

                segment_stderr_spill = await self._spill_payload_async(
                    step_stderr,
                    kind="stderr",
                )
                if segment_stderr_spill is None and preserved_binary_stderr_spill is not None:
                    segment_stderr_spill = preserved_binary_stderr_spill
                keep_inline_stderr = preserved_binary_stderr_spill is not None and self._has_content(step_stderr)
                stored_stderr = step_stderr if keep_inline_stderr else (
                    self._empty_like(step_stderr) if segment_stderr_spill is not None else step_stderr
                )
                aggregate_stderr_binary = aggregate_stderr_binary or step_stderr_contains_binary
                if not step_stderr_binary and (self._has_content(step_stderr) or segment_stderr_spill is not None):
                    aggregate_textual_stderr = True
                if segment_stderr_spill is not None:
                    latest_stderr_spill = segment_stderr_spill
                    aggregate_stderr_spills.append(segment_stderr_spill)
                if self._has_content(stored_stderr):
                    aggregate_stderr_parts.append(stored_stderr)
                if self._has_content(step_stderr):
                    aggregate_raw_stderr_parts.append(step_stderr)

                steps.append(
                    CommandExecutionStep(
                        argv=list(command.argv),
                        stdin=step_stdin,
                        stdout=stored_stdout,
                        stderr=stored_stderr,
                        exit_code=segment_exit_code,
                        duration_ms=max(0.0, (time.perf_counter() - step_started) * 1000.0),
                        stdout_spill=segment_stdout_spill,
                        stderr_spill=segment_stderr_spill,
                        stdout_is_binary=step_stdout_binary,
                        stderr_is_binary=step_stderr_binary,
                        stderr_contains_binary=step_stderr_contains_binary,
                    )
                )

                if step_stdout_binary and command_index < len(segment.commands) - 1:
                    break

            last_exit_code = segment_exit_code
            last_stdout = segment_terminal_stdout
            last_stderr = self._combine_fragments(aggregate_raw_stderr_parts)
            last_stdout_spill = segment_stdout_spill
            last_stderr_spill = latest_stderr_spill
            last_stdout_binary = segment_stdout_binary
            last_stderr_binary = aggregate_stderr_binary and not aggregate_textual_stderr

        return CommandExecutionResult(
            stdout=last_stdout,
            stderr=last_stderr,
            exit_code=last_exit_code,
            duration_ms=max(0.0, (time.perf_counter() - start) * 1000.0),
            steps=steps,
            stdout_spill=last_stdout_spill,
            stderr_spill=last_stderr_spill,
            stderr_spills=aggregate_stderr_spills,
            stdout_is_binary=last_stdout_binary,
            stderr_is_binary=last_stderr_binary,
            stderr_contains_binary=aggregate_stderr_binary,
        )

    @staticmethod
    def _build_step_indices(chain: CommandChain) -> dict[tuple[int, int], int]:
        indices: dict[tuple[int, int], int] = {}
        step_index = 0
        for segment_index, segment in enumerate(chain.segments):
            for command_index, _command in enumerate(segment.commands):
                indices[(segment_index, command_index)] = step_index
                step_index += 1
        return indices

    async def _spill_payload_async(
        self,
        value: Any,
        *,
        kind: str,
        force: bool = False,
    ) -> CommandSpillReference | None:
        return await asyncio.to_thread(self._spill_payload, value, kind=kind, force=force)

    def _binary_pipe_error(self, existing_stderr: Any) -> str:
        parts: list[str] = []
        existing_text = self._coerce_text(existing_stderr)
        if existing_text:
            parts.append(existing_text)
        parts.append("Pipe payloads must be UTF-8 text in v1; binary output cannot flow into the next command.")
        return "\n".join(parts)

    def _classify_stream(self, value: Any) -> tuple[Any, bool]:
        if isinstance(value, bytes):
            try:
                return value.decode("utf-8"), False
            except UnicodeDecodeError:
                return value, True
        if value is None:
            return "", False
        if isinstance(value, str):
            return value, False
        return value, False

    def _empty_like(self, value: Any) -> Any:
        if isinstance(value, bytes):
            return b""
        return ""

    def _has_content(self, value: Any) -> bool:
        if isinstance(value, bytes):
            return len(value) > 0
        return bool(value)

    def _combine_fragments(self, fragments: list[Any]) -> Any:
        if not fragments:
            return ""
        if any(isinstance(fragment, bytes) for fragment in fragments):
            combined: list[bytes] = []
            for fragment in fragments:
                if isinstance(fragment, bytes):
                    combined.append(fragment)
                else:
                    combined.append(str(fragment).encode("utf-8"))
            return b"".join(combined)
        return "".join(str(fragment) for fragment in fragments)

    def _coerce_text(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            try:
                return value.decode("utf-8")
            except UnicodeDecodeError:
                return ""
        if isinstance(value, str):
            return value
        return str(value)

    def _spill_payload(self, value: Any, *, kind: str, force: bool = False) -> CommandSpillReference | None:
        payload = self._to_bytes(value)
        if not force and len(payload) <= self.spill_threshold_bytes:
            return None

        spill_root = self._resolve_spill_root()
        self._prune_stale_spills(spill_root)
        digest = hashlib.sha256(payload).hexdigest()[:16]
        path = self._write_unique_spill(payload, spill_root=spill_root, kind=kind, digest=digest)
        try:
            os.utime(path, None)
        except OSError:
            pass
        self._touch_spill_root(spill_root)

        return CommandSpillReference(
            path=str(path),
            bytes_written=len(payload),
            line_count=self._line_count(value),
            preview=self._preview_text(value, self.preview_bytes),
        )

    def _resolve_spill_root(self) -> Path:
        if self.spill_dir is not None:
            spill_root = Path(self.spill_dir)
            self._ensure_existing_or_created_spill_root(spill_root)
        else:
            if self._default_spill_root is None or not self._default_spill_root.exists():
                temp_parent = Path(tempfile.gettempdir())
                self._maybe_prune_stale_default_spill_roots(temp_parent)
                self._default_spill_root = Path(
                    tempfile.mkdtemp(
                        prefix=_DEFAULT_SPILL_ROOT_PREFIX,
                        dir=str(temp_parent),
                    )
                )
            spill_root = self._default_spill_root
            self._validate_existing_spill_root(spill_root)
        return spill_root

    def _ensure_existing_or_created_spill_root(self, spill_root: Path) -> None:
        try:
            spill_root.mkdir(parents=True, mode=0o700, exist_ok=False)
        except FileExistsError:
            pass
        self._validate_existing_spill_root(spill_root)

    def _validate_existing_spill_root(self, spill_root: Path) -> None:
        if spill_root.is_symlink():
            raise PermissionError(f"Refusing to use symlink spill directory: {spill_root}")
        if not spill_root.is_dir():
            raise PermissionError(f"Refusing to use non-directory spill path: {spill_root}")
        self._validate_spill_root_permissions(spill_root)

    def _validate_spill_root_permissions(self, spill_root: Path) -> None:
        try:
            info = spill_root.stat(follow_symlinks=False)
        except OSError as exc:
            raise PermissionError(f"Unable to inspect spill directory: {spill_root}") from exc

        getuid = getattr(os, "getuid", None)
        if callable(getuid) and info.st_uid != getuid():
            raise PermissionError(f"Refusing to use spill directory owned by another user: {spill_root}")

        if os.name != "nt" and stat.S_IMODE(info.st_mode) & 0o077:
            raise PermissionError(f"Refusing to use spill directory with non-private permissions: {spill_root}")

    def _ensure_spill_file(
        self,
        path: Path,
        payload: bytes,
        *,
        spill_root: Path,
        kind: str,
        digest: str,
    ) -> Path:
        if self._path_matches_payload(path, payload):
            return path

        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            if self._path_matches_payload(path, payload):
                return path
            return self._write_unique_spill(payload, spill_root=spill_root, kind=kind, digest=digest)

        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
        return path

    def _path_matches_payload(self, path: Path, payload: bytes) -> bool:
        try:
            return path.exists() and not path.is_symlink() and path.read_bytes() == payload
        except OSError:
            return False

    def _write_unique_spill(self, payload: bytes, *, spill_root: Path, kind: str, digest: str) -> Path:
        fd, raw_path = tempfile.mkstemp(prefix=f"mcp-command-{kind}-{digest}-", suffix=".txt", dir=str(spill_root))
        path = Path(raw_path)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
        except Exception:
            path.unlink(missing_ok=True)
            raise
        return path

    def _prune_stale_spills(self, spill_root: Path) -> None:
        if self.spill_retention_seconds <= 0:
            return

        cutoff = time.time() - float(self.spill_retention_seconds)
        for candidate in spill_root.glob("mcp-command-*.txt"):
            try:
                info = candidate.stat(follow_symlinks=False)
            except OSError:
                continue
            if not stat.S_ISREG(info.st_mode):
                continue
            if info.st_mtime >= cutoff:
                continue
            try:
                candidate.unlink()
            except OSError:
                continue

    def _prune_stale_default_spill_roots(self, temp_parent: Path) -> None:
        if self.spill_retention_seconds <= 0:
            return

        cutoff = time.time() - float(self.spill_retention_seconds)
        getuid = getattr(os, "getuid", None)
        for candidate in temp_parent.glob(f"{_DEFAULT_SPILL_ROOT_PREFIX}*"):
            try:
                info = candidate.stat(follow_symlinks=False)
            except OSError:
                continue
            if candidate.is_symlink() or not stat.S_ISDIR(info.st_mode):
                continue
            if callable(getuid) and info.st_uid != getuid():
                continue
            if info.st_mtime >= cutoff:
                continue
            try:
                shutil.rmtree(candidate)
            except OSError:
                continue

    def _maybe_prune_stale_default_spill_roots(self, temp_parent: Path) -> None:
        if self.spill_retention_seconds <= 0:
            return

        now = time.monotonic()
        temp_parent_key = os.fspath(temp_parent)

        with self._default_spill_root_prune_lock:
            last_prune = self._default_spill_root_last_prune_by_parent.get(temp_parent_key)
            if (
                last_prune is not None
                and now - last_prune < _DEFAULT_SPILL_ROOT_PRUNE_INTERVAL_SECONDS
            ):
                return
            self._prune_stale_default_spill_roots(temp_parent)
            self._default_spill_root_last_prune_by_parent[temp_parent_key] = now

    def _touch_spill_root(self, spill_root: Path) -> None:
        try:
            os.utime(spill_root, None)
        except OSError:
            pass

    def _preview_text(self, value: Any, preview_bytes: int) -> str:
        payload = self._to_bytes(value)
        return self._safe_utf8_prefix(payload, preview_bytes)

    def _to_bytes(self, value: Any) -> bytes:
        if isinstance(value, bytes):
            return value
        if isinstance(value, str):
            return value.encode("utf-8")
        if value is None:
            return b""
        return str(value).encode("utf-8")

    def _safe_utf8_prefix(self, payload: bytes, preview_bytes: int) -> str:
        if preview_bytes <= 0 or not payload:
            return ""
        prefix = payload[:preview_bytes]
        while prefix:
            try:
                return prefix.decode("utf-8")
            except UnicodeDecodeError:
                prefix = prefix[:-1]
        return ""

    def _line_count(self, value: Any) -> int:
        text = self._coerce_text(value)
        if not text:
            return 0
        return text.count("\n") + (0 if text.endswith("\n") else 1)
