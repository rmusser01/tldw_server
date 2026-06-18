"""Model-facing presentation helpers for command runtime results."""

from __future__ import annotations

import hashlib
import os
import stat
import tempfile
from pathlib import Path
from typing import Any

from .models import CommandExecutionResult, CommandSpillReference


def present_command_execution_result(
    result: CommandExecutionResult,
    *,
    preview_limit: int | None = None,
    byte_limit: int | None = None,
    line_limit: int = 200,
    spill_dir: Path | str | None = None,
    include_artifact_handles: bool = False,
) -> str:
    """Format a raw execution result for model consumption."""

    byte_limit = _resolve_byte_limit(preview_limit=preview_limit, byte_limit=byte_limit)
    footer = f"[exit:{result.exit_code} | {int(round(result.duration_ms))}ms]"

    stdout_text = _decode_stream(result.stdout)
    stderr_text = _collect_textual_stderr(result)
    text_stderr_spills = _collect_stderr_spills(result, binary=False)
    binary_stderr_spills = _collect_stderr_spills(result, binary=True)
    stderr_has_binary = result.stderr_contains_binary or result.stderr_is_binary

    if result.stdout_is_binary:
        lines = ["[binary output omitted]"]
        if result.stdout_spill is not None:
            lines.append(_binary_spill_notice("stdout", result.stdout_spill))
        if result.exit_code != 0:
            lines.extend(
                _render_stderr_block(
                    stderr_text,
                    text_stderr_spills,
                    binary_stderr_spills,
                    has_binary=stderr_has_binary,
                    byte_limit=byte_limit,
                    line_limit=line_limit,
                    spill_dir=spill_dir,
                    include_artifact_handles=include_artifact_handles,
                )
            )
        lines.append(footer)
        return "\n".join(lines)

    body_lines: list[str] = []
    stdout_block = _render_stdout(
        stdout_text,
        result.stdout_spill,
        byte_limit,
        line_limit,
        spill_dir,
        include_artifact_handles=include_artifact_handles,
    )
    if stdout_block:
        body_lines.append(stdout_block)

    if result.exit_code != 0:
        stderr_block = _render_stderr_block(
            stderr_text,
            text_stderr_spills,
            binary_stderr_spills,
            has_binary=stderr_has_binary,
            byte_limit=byte_limit,
            line_limit=line_limit,
            spill_dir=spill_dir,
            include_artifact_handles=include_artifact_handles,
        )
        if stderr_block:
            if body_lines:
                body_lines.append("")
            body_lines.extend(stderr_block)

    body_lines.append(footer)
    return "\n".join(line for line in body_lines if line is not None)


def _render_stdout(
    stdout_text: str,
    spill: CommandSpillReference | None,
    byte_limit: int,
    line_limit: int,
    spill_dir: Path | str | None,
    *,
    include_artifact_handles: bool,
) -> str:
    active_spill = spill
    if active_spill is not None:
        preview, truncated = _preview_spill_text(active_spill, byte_limit=byte_limit, line_limit=line_limit, fallback=stdout_text)
        line_count = active_spill.line_count
        byte_count = active_spill.bytes_written
    else:
        preview, truncated = _truncate_text(stdout_text, byte_limit=byte_limit, line_limit=line_limit)
        line_count = _count_lines(stdout_text)
        byte_count = len(stdout_text.encode("utf-8"))
    if truncated:
        notice = _overflow_notice(
            "stdout",
            line_count=line_count,
            byte_count=byte_count,
            spill=active_spill,
            include_artifact_handles=include_artifact_handles,
        )
        if preview:
            return "\n".join([preview, "", notice])
        return notice
    return preview


def _render_stderr(
    stderr_text: str,
    spills: list[CommandSpillReference],
    byte_limit: int,
    line_limit: int,
    spill_dir: Path | str | None,
    *,
    include_artifact_handles: bool,
) -> str:
    active_spills = list(spills)
    if stderr_text:
        preview, truncated = _truncate_text(stderr_text, byte_limit=byte_limit, line_limit=line_limit)
        line_count = _count_lines(stderr_text)
        byte_count = len(stderr_text.encode("utf-8"))
    elif active_spills:
        preview, truncated = _preview_spill_text(
            active_spills[0],
            byte_limit=byte_limit,
            line_limit=line_limit,
            fallback=active_spills[0].preview,
        )
        line_count = active_spills[0].line_count
        byte_count = active_spills[0].bytes_written
    else:
        preview, truncated = "", False
        line_count = 0
        byte_count = 0
    show_overflow = truncated or len(active_spills) > 1 or bool(stderr_text and active_spills)
    if active_spills and show_overflow:
        lines = [preview] if preview else []
        if lines:
            lines.append("")
        lines.append(
            _overflow_notice(
                "stderr",
                line_count=line_count,
                byte_count=byte_count,
                spill=active_spills[0],
                include_artifact_handles=include_artifact_handles,
            )
        )
        return "\n".join(lines)
    if truncated:
        notice = _overflow_notice(
            "stderr",
            line_count=line_count,
            byte_count=byte_count,
            spill=active_spills[0] if active_spills else None,
            include_artifact_handles=include_artifact_handles,
        )
        if preview:
            return "\n".join([preview, "", notice])
        return notice
    return preview


def _render_stderr_block(
    stderr_text: str,
    text_spills: list[CommandSpillReference],
    binary_spills: list[CommandSpillReference],
    *,
    has_binary: bool,
    byte_limit: int,
    line_limit: int,
    spill_dir: Path | str | None,
    include_artifact_handles: bool,
) -> list[str]:
    if not (stderr_text or text_spills or has_binary):
        return []

    lines = ["stderr:"]
    if stderr_text or text_spills:
        lines.append(
            _render_stderr(
                stderr_text,
                text_spills,
                byte_limit,
                line_limit,
                spill_dir,
                include_artifact_handles=include_artifact_handles,
            )
        )
    if has_binary:
        if len(lines) > 1:
            lines.append("")
        lines.append("[binary stderr omitted]")
        for spill in binary_spills:
            lines.append(_binary_spill_notice("stderr", spill))
    return lines


def _overflow_notice(
    stream_name: str,
    *,
    line_count: int,
    byte_count: int,
    spill: CommandSpillReference | None = None,
    include_artifact_handles: bool = False,
) -> str:
    lines = [
        f"--- {stream_name} truncated ({line_count} lines, {byte_count} bytes) ---",
        f"Full {stream_name} was stored internally because it exceeded the preview limit.",
    ]
    if include_artifact_handles and spill is not None:
        lines.append(f"artifact: mcp-run-output://{stream_name}/{Path(spill.path).name}")
    lines.append("Refine and rerun with: | grep <pattern>, | head <n>, or | tail <n>")
    return "\n".join(lines)


def _binary_spill_notice(stream_name: str, spill: CommandSpillReference) -> str:
    return "\n".join(
        [
            f"--- binary {stream_name} omitted ({spill.bytes_written} bytes) ---",
            f"Binary {stream_name} was stored internally.",
            "Use a binary-safe inspection command or text-rendering subcommand.",
        ]
    )


def _collect_stderr_spills(
    result: CommandExecutionResult,
    *,
    binary: bool,
) -> list[CommandSpillReference]:
    spills: list[CommandSpillReference] = []
    seen_paths: set[str] = set()
    if result.steps:
        for step in result.steps:
            if step.stderr_spill is None:
                continue
            step_has_binary = step.stderr_contains_binary or step.stderr_is_binary
            if step_has_binary is not binary:
                continue
            if step.stderr_spill.path in seen_paths:
                continue
            seen_paths.add(step.stderr_spill.path)
            spills.append(step.stderr_spill)
        return spills

    fallback_has_binary = result.stderr_contains_binary or result.stderr_is_binary
    if binary and not fallback_has_binary:
        return spills
    if not binary and fallback_has_binary:
        return spills

    candidates = list(result.stderr_spills)
    if result.stderr_spill is not None:
        candidates.append(result.stderr_spill)
    for spill in candidates:
        if spill.path in seen_paths:
            continue
        seen_paths.add(spill.path)
        spills.append(spill)
    return spills


def _collect_textual_stderr(result: CommandExecutionResult) -> str:
    if result.steps:
        fragments: list[str] = []
        saw_textual_stderr = False
        for step in result.steps:
            if step.stderr_is_binary:
                continue
            text = _decode_stream(step.stderr)
            if text:
                fragments.append(text)
                saw_textual_stderr = True
            if step.stderr_spill is not None and not (step.stderr_contains_binary or step.stderr_is_binary):
                saw_textual_stderr = True
        if saw_textual_stderr:
            return "".join(fragments)

    if result.stderr_is_binary:
        return ""
    return _decode_stream(result.stderr)

def _decode_stream(value: Any) -> str:
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value.decode("utf-8", errors="replace")
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _materialize_spill_reference(
    text: str,
    stream_name: str,
    *,
    spill_dir: Path | str | None,
) -> CommandSpillReference:
    payload = text.encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:16]
    spill_root = _resolve_spill_root(spill_dir)
    path = spill_root / f"{stream_name}-{digest}.txt"
    path = _ensure_materialized_spill(path, payload, spill_root=spill_root, stream_name=stream_name, digest=digest)
    return CommandSpillReference(
        path=str(path),
        bytes_written=len(payload),
        line_count=_count_lines(text),
        preview=_byte_safe_prefix(text, min(len(payload), 4_096)),
    )


def _resolve_byte_limit(*, preview_limit: int | None, byte_limit: int | None) -> int:
    if byte_limit is not None:
        return byte_limit
    if preview_limit is not None:
        return preview_limit
    return 4_000


def _resolve_spill_root(spill_dir: Path | str | None) -> Path:
    if spill_dir is not None:
        spill_root = Path(spill_dir)
    else:
        spill_root = Path(tempfile.gettempdir()) / f"mcp-command-presentation-{_default_spill_suffix()}"

    if spill_root.exists():
        if spill_root.is_symlink():
            raise PermissionError(f"Refusing to use symlink spill directory: {spill_root}")
        if not spill_root.is_dir():
            raise PermissionError(f"Refusing to use non-directory spill path: {spill_root}")
        _validate_spill_root_permissions(spill_root)
        return spill_root

    try:
        spill_root.mkdir(parents=True, mode=0o700)
    except FileExistsError as exc:
        if spill_root.is_symlink():
            raise PermissionError(f"Refusing to use symlink spill directory: {spill_root}") from exc
        if not spill_root.is_dir():
            raise PermissionError(f"Refusing to use non-directory spill path: {spill_root}") from exc
    _validate_spill_root_permissions(spill_root)
    return spill_root


def _default_spill_suffix() -> str:
    getuid = getattr(os, "getuid", None)
    if callable(getuid):
        return f"user-{getuid()}"
    return "local"


def _validate_spill_root_permissions(spill_root: Path) -> None:
    try:
        info = spill_root.stat(follow_symlinks=False)
    except OSError as exc:
        raise PermissionError(f"Unable to inspect spill directory: {spill_root}") from exc

    getuid = getattr(os, "getuid", None)
    if callable(getuid) and info.st_uid != getuid():
        raise PermissionError(f"Refusing to use spill directory owned by another user: {spill_root}")

    if os.name != "nt" and stat.S_IMODE(info.st_mode) & 0o077:
        raise PermissionError(f"Refusing to use spill directory with non-private permissions: {spill_root}")


def _truncate_text(text: str, *, byte_limit: int, line_limit: int) -> tuple[str, bool]:
    if not text:
        return "", False
    line_limited, line_truncated = _apply_line_limit(text, line_limit)
    preview = _byte_safe_prefix(line_limited, byte_limit)
    byte_truncated = len(line_limited.encode("utf-8")) > byte_limit
    return preview, line_truncated or byte_truncated


def _apply_line_limit(text: str, line_limit: int) -> tuple[str, bool]:
    if line_limit <= 0:
        return "", bool(text)
    lines = text.splitlines(keepends=True)
    if len(lines) <= line_limit:
        return text, False
    return "".join(lines[:line_limit]), True


def _count_lines(text: str) -> int:
    if not text:
        return 0
    return text.count("\n") + (0 if text.endswith("\n") else 1)


def _preview_spill_text(
    spill: CommandSpillReference,
    *,
    byte_limit: int,
    line_limit: int,
    fallback: str,
) -> tuple[str, bool]:
    metadata_truncated = spill.bytes_written > byte_limit or spill.line_count > line_limit
    if byte_limit <= 0 or line_limit <= 0:
        return "", spill.bytes_written > 0 or spill.line_count > 0

    try:
        payload = bytearray()
        newline_count = 0
        with Path(spill.path).open("rb") as handle:
            while len(payload) <= byte_limit and newline_count <= line_limit:
                chunk = handle.read(4096)
                if not chunk:
                    break
                payload.extend(chunk)
                newline_count += chunk.count(b"\n")
    except OSError:
        preview, truncated = _truncate_text(fallback or spill.preview, byte_limit=byte_limit, line_limit=line_limit)
        return preview, truncated or metadata_truncated

    preview, truncated = _truncate_text(_safe_decode_bytes(bytes(payload)), byte_limit=byte_limit, line_limit=line_limit)
    return preview, truncated or metadata_truncated


def _ensure_materialized_spill(
    path: Path,
    payload: bytes,
    *,
    spill_root: Path,
    stream_name: str,
    digest: str,
) -> Path:
    if _path_matches_payload(path, payload):
        return path

    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        if _path_matches_payload(path, payload):
            return path
        return _write_unique_spill(payload, spill_root=spill_root, stream_name=stream_name, digest=digest)

    with os.fdopen(fd, "wb") as handle:
        handle.write(payload)
    return path


def _path_matches_payload(path: Path, payload: bytes) -> bool:
    try:
        return path.exists() and not path.is_symlink() and path.read_bytes() == payload
    except OSError:
        return False


def _write_unique_spill(payload: bytes, *, spill_root: Path, stream_name: str, digest: str) -> Path:
    fd, raw_path = tempfile.mkstemp(prefix=f"{stream_name}-{digest}-", suffix=".txt", dir=str(spill_root))
    path = Path(raw_path)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return path


def _safe_decode_bytes(payload: bytes) -> str:
    if not payload:
        return ""
    prefix = payload
    while prefix:
        try:
            return prefix.decode("utf-8")
        except UnicodeDecodeError:
            prefix = prefix[:-1]
    return ""


def _byte_safe_prefix(text: str, byte_limit: int) -> str:
    if byte_limit <= 0:
        return ""
    payload = text.encode("utf-8")
    if len(payload) <= byte_limit:
        return text
    prefix = payload[:byte_limit]
    while prefix:
        try:
            return prefix.decode("utf-8")
        except UnicodeDecodeError:
            prefix = prefix[:-1]
    return ""
