#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
RUNTIME_POINTER_KEYS = (
    "source_bundle_path",
    "run_bundle_path",
    "image_store_root",
    "socket_path",
    "serial_log_dir",
    "helper_pid_file",
    "evidence_dir",
)


@dataclass(frozen=True)
class EvidenceFileStatus:
    name: str
    present: bool
    readable: bool
    reason: str
    size_bytes: int | None = None


def _display(value: object, *, max_chars: int = DISPLAY_MAX_CHARS) -> str:
    text = "" if value is None else str(value)
    text = " ".join(text.replace("\r", "\n").splitlines())
    text = html.escape(text, quote=False)
    text = text.replace("|", "\\|")
    if len(text) > max_chars:
        return text[: max_chars - 1] + "..."
    return text


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
    try:
        metadata = evidence_dir.lstat()
    except FileNotFoundError:
        return False, [f"warning: evidence directory is missing: {evidence_dir}"]
    except OSError as exc:
        return False, [f"warning: evidence directory cannot be inspected: {type(exc).__name__}: {exc}"]
    if stat.S_ISLNK(metadata.st_mode):
        return False, [f"warning: evidence directory is a symlink and was not read: {evidence_dir}"]
    if not stat.S_ISDIR(metadata.st_mode):
        return False, [f"warning: evidence path is not a directory and was not read: {evidence_dir}"]
    if not os.access(evidence_dir, os.R_OK | os.X_OK):
        return False, [f"warning: evidence directory is unreadable and was not read: {evidence_dir}"]
    return True, []


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


def _probe_expected_file(evidence_dir: Path, name: str) -> EvidenceFileStatus:
    path = evidence_dir / name
    try:
        metadata = path.lstat()
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
    if not os.access(path, os.R_OK):
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


def _probe_expected_files(evidence_dir: Path) -> dict[str, EvidenceFileStatus]:
    return {name: _probe_expected_file(evidence_dir, name) for name in EXPECTED_EVIDENCE_FILES}


def _load_evidence_json(
    evidence_dir: Path,
    file_statuses: dict[str, EvidenceFileStatus],
) -> tuple[dict[str, Any] | None, list[str]]:
    json_status = file_statuses["host-smoke-evidence.json"]
    if not json_status.readable:
        return None, [f"warning: structured metadata unavailable: {json_status.reason}"]
    if json_status.size_bytes is not None and json_status.size_bytes > JSON_MAX_BYTES:
        return None, [f"warning: structured metadata skipped: exceeds {JSON_MAX_BYTES} bytes"]

    try:
        with (evidence_dir / "host-smoke-evidence.json").open("rb") as handle:
            raw_bytes = handle.read(JSON_MAX_BYTES + 1)
        if len(raw_bytes) > JSON_MAX_BYTES:
            return None, [f"warning: structured metadata skipped: exceeds {JSON_MAX_BYTES} bytes"]
        raw_text = raw_bytes.decode("utf-8")
        payload = json.loads(raw_text)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
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
            rows.append((key, payload[key]))
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
                    details.get("status", ""),
                    details.get("exit_code", ""),
                    details.get("timestamp", ""),
                )
            )
        else:
            rows.append((phase, details, "", ""))
    return "## Phase Outcomes\n\n" + _table(("Phase", "Status", "Exit code", "Timestamp"), rows)


def _render_cleanup(payload: dict[str, Any] | None) -> str:
    if not payload:
        return ""
    cleanup = payload.get("cleanup")
    if not isinstance(cleanup, dict) or not cleanup:
        return ""
    rows = [(key, value) for key, value in cleanup.items()]
    return "## Cleanup Status\n\n" + _table(("Field", "Value"), rows)


def _render_runtime_pointers(payload: dict[str, Any] | None) -> str:
    if not payload:
        return ""
    rows = [(key, payload[key]) for key in RUNTIME_POINTER_KEYS if key in payload]
    if not rows:
        return ""
    return "## Runtime And Artifact Pointers\n\n" + _table(("Field", "Value"), rows)


def _render_recorded_evidence_paths(payload: dict[str, Any] | None) -> str:
    if not payload:
        return ""
    evidence_files = payload.get("evidence_files")
    if not isinstance(evidence_files, dict) or not evidence_files:
        return ""
    rows = [(name, evidence_files.get(name, "")) for name in EXPECTED_EVIDENCE_FILES if name in evidence_files]
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
                    artifact.get("path", ""),
                    artifact.get("size_bytes", ""),
                    artifact.get("sha256", ""),
                )
            )
        else:
            rows.append((artifact, "", ""))
    return "## Log Artifact Metadata\n\n" + _table(("Path", "Size bytes", "SHA-256"), rows)


def render_summary(evidence_dir: Path) -> str:
    warnings: list[str] = []
    evidence_dir_ok, dir_warnings = _probe_evidence_dir(evidence_dir)
    warnings.extend(dir_warnings)

    if evidence_dir_ok:
        file_statuses = _probe_expected_files(evidence_dir)
        warnings.extend(_file_status_warnings(file_statuses))
        payload, json_warnings = _load_evidence_json(evidence_dir, file_statuses)
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
