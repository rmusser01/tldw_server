#!/usr/bin/env python3
from __future__ import annotations

import argparse
import errno
import html
import json
import os
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EXPECTED_EVIDENCE_FILES = (
    "host-smoke-evidence.json",
    "source-bundle-hashes-before.txt",
    "source-bundle-hashes-after.txt",
    "run-bundle-hashes.txt",
    "runtime-paths.txt",
    "cleanup-status.txt",
)
JSON_MAX_BYTES = 1024 * 1024
DISPLAY_MAX_CHARS = 240
INVALID_METADATA_VALUE = "invalid or unavailable"
RUNTIME_POINTER_KEYS = (
    "source_bundle_path",
    "run_bundle_path",
    "image_store_root",
    "socket_path",
    "serial_log_dir",
    "helper_pid_file",
    "evidence_dir",
)
MARKDOWN_ESCAPE_CHARS = frozenset("\\`[]()!|")
CLEANUP_KEYS = (
    "status",
    "helper_pid",
    "helper_running_after_cleanup",
    "socket_present_after_cleanup",
)


@dataclass(frozen=True)
class EvidenceFileStatus:
    name: str
    present: bool
    readable: bool
    reason: str
    size_bytes: int | None = None


@dataclass
class EvidenceDirHandle:
    path: Path
    fd: int | None
    warnings: list[str]

    def __enter__(self) -> "EvidenceDirHandle":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def close(self) -> None:
        if self.fd is None:
            return
        fd = self.fd
        self.fd = None
        os.close(fd)


def _display(value: object, *, max_chars: int = DISPLAY_MAX_CHARS) -> str:
    text = "" if value is None else str(value)
    text = " ".join(text.replace("\r", "\n").splitlines())
    text = html.escape(text, quote=False)
    text = "".join(
        f"\\{character}" if character in MARKDOWN_ESCAPE_CHARS else character
        for character in text
    )
    if len(text) > max_chars:
        return text[: max_chars - 1] + "..."
    return text


def _metadata_scalar(value: Any) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return INVALID_METADATA_VALUE


def _table(headers: tuple[str, ...], rows: list[tuple[object, ...]]) -> str:
    rendered = [
        "| " + " | ".join(_display(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        padded = row + ("",) * (len(headers) - len(row))
        rendered.append("| " + " | ".join(_display(cell) for cell in padded[: len(headers)]) + " |")
    return "\n".join(rendered)


def _probe_evidence_dir(evidence_dir: Path) -> tuple[bool, list[str]]:
    with _open_evidence_dir(evidence_dir) as evidence_root:
        return evidence_root.fd is not None, evidence_root.warnings


def _dir_fd_operations_available() -> bool:
    supports_dir_fd = getattr(os, "supports_dir_fd", set())
    supports_follow_symlinks = getattr(os, "supports_follow_symlinks", set())
    return (
        os.open in supports_dir_fd
        and os.stat in supports_dir_fd
        and os.access in supports_dir_fd
        and os.stat in supports_follow_symlinks
        and os.access in supports_follow_symlinks
        and hasattr(os, "O_NOFOLLOW")
    )


def _open_dir_flags() -> int:
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    return flags


def _missing_evidence_dir_warning(evidence_dir: Path) -> str:
    return (
        f"warning: evidence directory is missing: {evidence_dir}. "
        "This may indicate an early setup/preflight failure."
    )


def _classify_evidence_dir_open_error(
    evidence_dir: Path,
    exc: OSError,
    *,
    parent_fd: int | None = None,
    component: str | None = None,
) -> str:
    if parent_fd is not None and component is not None:
        try:
            metadata = os.stat(component, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return _missing_evidence_dir_warning(evidence_dir)
        except OSError:
            pass
        else:
            if stat.S_ISLNK(metadata.st_mode):
                return f"warning: evidence directory is a symlink or contains a symlink and was not read: {evidence_dir}"
            if not stat.S_ISDIR(metadata.st_mode):
                return f"warning: evidence path is not a directory and was not read: {evidence_dir}"
    if isinstance(exc, FileNotFoundError) or exc.errno == errno.ENOENT:
        return _missing_evidence_dir_warning(evidence_dir)
    if exc.errno == errno.ELOOP:
        return f"warning: evidence directory is a symlink or contains a symlink and was not read: {evidence_dir}"
    if exc.errno == errno.ENOTDIR:
        return f"warning: evidence path is not a directory and was not read: {evidence_dir}"
    if exc.errno in {errno.EACCES, errno.EPERM}:
        return f"warning: evidence directory is unreadable and was not read: {evidence_dir}"
    return f"warning: evidence directory path is unsafe or unavailable: {type(exc).__name__}: {exc}"


def _evidence_dir_components(evidence_dir: Path) -> tuple[str, list[str], str | None]:
    parts = evidence_dir.parts
    if evidence_dir.is_absolute():
        start_path = os.sep
        raw_components = list(parts[1:])
    else:
        start_path = "."
        raw_components = list(parts)

    components: list[str] = []
    for component in raw_components:
        if component in ("", "."):
            continue
        if component == "..":
            return start_path, [], "warning: evidence directory path contains unsafe '..' component and was not read"
        components.append(component)
    return start_path, components, None


def _open_evidence_dir(evidence_dir: Path) -> EvidenceDirHandle:
    if not _dir_fd_operations_available():
        return EvidenceDirHandle(
            path=evidence_dir,
            fd=None,
            warnings=[
                (
                    "warning: evidence directory was not read because safe directory "
                    "file descriptor operations are unavailable on this platform"
                )
            ],
        )

    start_path, components, component_warning = _evidence_dir_components(evidence_dir)
    if component_warning is not None:
        return EvidenceDirHandle(path=evidence_dir, fd=None, warnings=[component_warning])

    fd: int | None = None
    try:
        fd = os.open(start_path, _open_dir_flags())
        metadata = os.fstat(fd)
    except OSError as exc:
        if fd is not None:
            os.close(fd)
        return EvidenceDirHandle(
            path=evidence_dir,
            fd=None,
            warnings=[_classify_evidence_dir_open_error(evidence_dir, exc)],
        )
    if not stat.S_ISDIR(metadata.st_mode):
        os.close(fd)
        return EvidenceDirHandle(
            path=evidence_dir,
            fd=None,
            warnings=[f"warning: evidence path is not a directory and was not read: {evidence_dir}"],
        )
    for component in components:
        next_fd: int | None = None
        try:
            next_fd = os.open(component, _open_dir_flags(), dir_fd=fd)
        except OSError as exc:
            warning = _classify_evidence_dir_open_error(
                evidence_dir,
                exc,
                parent_fd=fd,
                component=component,
            )
            os.close(fd)
            return EvidenceDirHandle(
                path=evidence_dir,
                fd=None,
                warnings=[warning],
            )
        os.close(fd)
        fd = next_fd
        try:
            metadata = os.fstat(fd)
        except OSError as exc:
            os.close(fd)
            return EvidenceDirHandle(
                path=evidence_dir,
                fd=None,
                warnings=[f"warning: evidence directory cannot be safely opened: {type(exc).__name__}: {exc}"],
            )
        if not stat.S_ISDIR(metadata.st_mode):
            os.close(fd)
            return EvidenceDirHandle(
                path=evidence_dir,
                fd=None,
                warnings=[f"warning: evidence path is not a directory and was not read: {evidence_dir}"],
            )
    return EvidenceDirHandle(path=evidence_dir, fd=fd, warnings=[])


def _missing_file_statuses(reason: str) -> dict[str, EvidenceFileStatus]:
    return {
        name: EvidenceFileStatus(
            name=name,
            present=False,
            readable=False,
            reason=reason,
        )
        for name in EXPECTED_EVIDENCE_FILES
    }


def _probe_expected_file(evidence_root: EvidenceDirHandle, name: str) -> EvidenceFileStatus:
    if evidence_root.fd is None:
        return EvidenceFileStatus(
            name=name,
            present=False,
            readable=False,
            reason="missing: evidence directory was not inspected",
        )
    try:
        metadata = os.stat(name, dir_fd=evidence_root.fd, follow_symlinks=False)
    except FileNotFoundError:
        return EvidenceFileStatus(name=name, present=False, readable=False, reason="missing")
    except OSError as exc:
        return EvidenceFileStatus(
            name=name,
            present=False,
            readable=False,
            reason=f"cannot inspect: {type(exc).__name__}",
        )
    if stat.S_ISLNK(metadata.st_mode):
        return EvidenceFileStatus(
            name=name,
            present=True,
            readable=False,
            reason="symlink skipped",
            size_bytes=metadata.st_size,
        )
    if not stat.S_ISREG(metadata.st_mode):
        return EvidenceFileStatus(
            name=name,
            present=True,
            readable=False,
            reason="non-regular file skipped",
            size_bytes=metadata.st_size,
        )
    if not _can_read_child(evidence_root, name):
        return EvidenceFileStatus(
            name=name,
            present=True,
            readable=False,
            reason="unreadable",
            size_bytes=metadata.st_size,
        )
    return EvidenceFileStatus(
        name=name,
        present=True,
        readable=True,
        reason="ok",
        size_bytes=metadata.st_size,
    )


def _probe_expected_files(evidence_root: EvidenceDirHandle) -> dict[str, EvidenceFileStatus]:
    return {name: _probe_expected_file(evidence_root, name) for name in EXPECTED_EVIDENCE_FILES}


def _open_json_flags() -> int:
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    return flags


def _can_read_child(evidence_root: EvidenceDirHandle, name: str) -> bool:
    if evidence_root.fd is None:
        return False
    try:
        return os.access(name, os.R_OK, dir_fd=evidence_root.fd, follow_symlinks=False)
    except (NotImplementedError, OSError):
        return False


def _read_json_bytes_from_descriptor(evidence_root: EvidenceDirHandle) -> tuple[bytes | None, list[str]]:
    if evidence_root.fd is None:
        return None, ["warning: structured metadata unavailable: evidence directory was not inspected"]
    fd: int | None = None
    try:
        fd = os.open("host-smoke-evidence.json", _open_json_flags(), dir_fd=evidence_root.fd)
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode):
            return None, ["warning: structured metadata skipped: opened path is not a regular file"]
        if metadata.st_size > JSON_MAX_BYTES:
            return None, [f"warning: structured metadata skipped: exceeds {JSON_MAX_BYTES} bytes"]
        with os.fdopen(fd, "rb") as handle:
            fd = None
            raw_bytes = handle.read(JSON_MAX_BYTES + 1)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            return None, ["warning: structured metadata open failed: symlink skipped"]
        return None, [f"warning: structured metadata open/read failed: {type(exc).__name__}: {exc}"]
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass

    if len(raw_bytes) > JSON_MAX_BYTES:
        return None, [f"warning: structured metadata skipped: exceeds {JSON_MAX_BYTES} bytes"]
    return raw_bytes, []


def _load_evidence_json(
    evidence_root: EvidenceDirHandle,
    file_statuses: dict[str, EvidenceFileStatus],
) -> tuple[dict[str, Any] | None, list[str]]:
    json_status = file_statuses["host-smoke-evidence.json"]
    if not json_status.readable:
        return None, [f"warning: structured metadata unavailable: {json_status.reason}"]
    if json_status.size_bytes is not None and json_status.size_bytes > JSON_MAX_BYTES:
        return None, [f"warning: structured metadata skipped: exceeds {JSON_MAX_BYTES} bytes"]

    raw_bytes, read_warnings = _read_json_bytes_from_descriptor(evidence_root)
    if read_warnings:
        return None, read_warnings
    if raw_bytes is None:
        return None, ["warning: structured metadata unavailable: descriptor read returned no data"]

    try:
        raw_text = raw_bytes.decode("utf-8")
        payload = json.loads(raw_text)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, [f"warning: structured metadata parse failed: {type(exc).__name__}: {exc}"]
    if not isinstance(payload, dict):
        return None, ["warning: structured metadata parse failed: top-level JSON is not an object"]
    return payload, []


def _file_status_warnings(file_statuses: dict[str, EvidenceFileStatus]) -> list[str]:
    warnings: list[str] = []
    for status in file_statuses.values():
        if status.present and not status.readable:
            warnings.append(f"warning: {status.name} unavailable: {status.reason}")
    return warnings


def _render_warnings(warnings: list[str]) -> str:
    if not warnings:
        return ""
    lines = ["## Warnings"]
    lines.extend(f"- {_display(warning)}" for warning in warnings)
    return "\n".join(lines)


def _render_file_checklist(file_statuses: dict[str, EvidenceFileStatus]) -> str:
    rows = [
        (
            status.name,
            "yes" if status.present else "no",
            "yes" if status.readable else "no",
            status.reason,
            status.size_bytes if status.size_bytes is not None else "-",
        )
        for status in file_statuses.values()
    ]
    return "## Expected File Checklist\n\n" + _table(
        ("file", "present", "readable", "reason", "size bytes"),
        rows,
    )


def _render_run_overview(payload: dict[str, Any] | None) -> str:
    if payload is None:
        return "## Structured Run Metadata\n\nStructured metadata was not parsed; inspect the checklist and uploaded artifacts."

    rows: list[tuple[object, object]] = []
    for key in ("smoke_run_id", "final_exit_code", "created_at", "schema_version"):
        if key in payload:
            rows.append((key, _metadata_scalar(payload[key])))
    if not rows:
        return "## Structured Run Metadata\n\nStructured metadata was parsed, but no known run overview fields were present."
    return "## Structured Run Metadata\n\n" + _table(("Field", "Value"), rows)


def _render_phase_outcomes(payload: dict[str, Any] | None) -> str:
    if not payload:
        return ""
    phases = payload.get("phases")
    if not isinstance(phases, dict) or not phases:
        return ""

    rows: list[tuple[object, object, object, object]] = []
    for phase, details in phases.items():
        if isinstance(details, dict):
            rows.append(
                (
                    phase,
                    _metadata_scalar(details.get("status", "")),
                    _metadata_scalar(details.get("exit_code", "")),
                    _metadata_scalar(details.get("timestamp", "")),
                )
            )
        else:
            rows.append((phase, INVALID_METADATA_VALUE, "", ""))
    return "## Phase Outcomes\n\n" + _table(("Phase", "Status", "Exit code", "Timestamp"), rows)


def _render_cleanup(payload: dict[str, Any] | None) -> str:
    if not payload:
        return ""
    cleanup = payload.get("cleanup")
    if not isinstance(cleanup, dict) or not cleanup:
        return ""
    rows = [(key, _metadata_scalar(cleanup[key])) for key in CLEANUP_KEYS if key in cleanup]
    if not rows:
        return ""
    return "## Cleanup Status\n\n" + _table(("Field", "Value"), rows)


def _render_runtime_pointers(payload: dict[str, Any] | None) -> str:
    if not payload:
        return ""
    rows = [(key, _metadata_scalar(payload[key])) for key in RUNTIME_POINTER_KEYS if key in payload]
    if not rows:
        return ""
    return "## Runtime And Artifact Pointers\n\n" + _table(("Field", "Value"), rows)


def _render_recorded_evidence_paths(payload: dict[str, Any] | None) -> str:
    if not payload:
        return ""
    evidence_files = payload.get("evidence_files")
    if not isinstance(evidence_files, dict) or not evidence_files:
        return ""
    rows = [
        (name, _metadata_scalar(evidence_files.get(name, "")))
        for name in EXPECTED_EVIDENCE_FILES
        if name in evidence_files
    ]
    if not rows:
        return ""
    return "## Recorded Evidence File Paths\n\n" + _table(("File", "Recorded path"), rows)


def _render_log_artifacts(payload: dict[str, Any] | None) -> str:
    if not payload:
        return ""
    log_artifacts = payload.get("log_artifacts")
    if not isinstance(log_artifacts, list) or not log_artifacts:
        return ""

    rows: list[tuple[object, object, object]] = []
    for artifact in log_artifacts:
        if isinstance(artifact, dict):
            rows.append(
                (
                    _metadata_scalar(artifact.get("path", "")),
                    _metadata_scalar(artifact.get("size_bytes", "")),
                    _metadata_scalar(artifact.get("sha256", "")),
                )
            )
    if not rows:
        return ""
    return "## Log Artifact Metadata\n\n" + _table(("Path", "Size bytes", "SHA-256"), rows)


def render_summary(evidence_dir: Path) -> str:
    warnings: list[str] = []
    with _open_evidence_dir(evidence_dir) as evidence_root:
        warnings.extend(evidence_root.warnings)
        if evidence_root.fd is not None:
            file_statuses = _probe_expected_files(evidence_root)
            warnings.extend(_file_status_warnings(file_statuses))
            payload, json_warnings = _load_evidence_json(evidence_root, file_statuses)
            warnings.extend(json_warnings)
        else:
            file_statuses = _missing_file_statuses("missing: evidence directory was not inspected")
            payload = None

    sections = [
        "# VZ Linux Host Smoke Evidence Summary",
        "> Advisory only: this summary is diagnostic and does not determine the host smoke job result.",
        f"**Evidence directory:** `{_display(evidence_dir)}`",
        _render_warnings(warnings),
        _render_run_overview(payload),
        _render_file_checklist(file_statuses),
        _render_phase_outcomes(payload),
        _render_cleanup(payload),
        _render_runtime_pointers(payload),
        _render_recorded_evidence_paths(payload),
        _render_log_artifacts(payload),
        (
            "> Primary artifact: `vz-linux-host-gated-evidence`. "
            "Raw-log fallback: `vz-linux-host-gated-helper-logs`."
        ),
    ]
    return "\n\n".join(section for section in sections if section).rstrip() + "\n"


def _write_summary(markdown: str, summary_path: str | None) -> None:
    if not summary_path:
        sys.stdout.write(markdown)
        if not markdown.endswith("\n"):
            sys.stdout.write("\n")
        return
    try:
        with Path(summary_path).open("a", encoding="utf-8") as handle:
            handle.write(markdown)
            if not markdown.endswith("\n"):
                handle.write("\n")
    except OSError as exc:
        print(
            f"warning: unable to append to GITHUB_STEP_SUMMARY: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        sys.stdout.write(markdown)
        if not markdown.endswith("\n"):
            sys.stdout.write("\n")


def _parse_args(argv: list[str] | None) -> argparse.Namespace | None:
    parser = argparse.ArgumentParser(
        description="Render an advisory Markdown summary for VZ Linux host smoke evidence.",
    )
    parser.add_argument("--evidence-dir", required=True, help="Path to the evidence directory to summarize.")
    try:
        return parser.parse_args(argv)
    except SystemExit as exc:
        if exc.code not in (0, None):
            print("warning: invalid evidence summary arguments; no summary was rendered", file=sys.stderr)
        return None


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parse_args(argv)
        if args is None:
            return 0
        markdown = render_summary(Path(args.evidence_dir))
        _write_summary(markdown, os.environ.get("GITHUB_STEP_SUMMARY"))
    except Exception as exc:  # advisory CLI boundary; do not mask smoke failures
        print(f"warning: evidence summary unavailable: {type(exc).__name__}: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
