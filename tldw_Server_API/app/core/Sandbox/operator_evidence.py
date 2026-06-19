"""Read bounded host-gated VZ smoke evidence for operator status."""

from __future__ import annotations

import errno
import json
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ENV_VZ_EVIDENCE_DIR = "TLDW_SANDBOX_VZ_EVIDENCE_DIR"
SOURCE_HOST_SMOKE_EVIDENCE = "host_smoke_evidence"
JSON_MAX_BYTES = 1024 * 1024
DISPLAY_MAX_CHARS = 240
MAX_PHASES = 16
STALE_AFTER_SECONDS = 7 * 24 * 60 * 60

EXPECTED_EVIDENCE_FILES = (
    "host-smoke-evidence.json",
    "source-bundle-hashes-before.txt",
    "source-bundle-hashes-after.txt",
    "run-bundle-hashes.txt",
    "runtime-paths.txt",
    "cleanup-status.txt",
)

RUNTIME_POINTER_KEYS = (
    "source_bundle_path",
    "run_bundle_path",
    "image_store_root",
    "socket_path",
    "serial_log_dir",
    "helper_pid_file",
    "evidence_dir",
)

CLEANUP_KEYS = (
    "status",
    "helper_pid",
    "helper_running_after_cleanup",
    "socket_present_after_cleanup",
)


@dataclass
class EvidenceDirHandle:
    """Descriptor-pinned evidence directory handle."""

    path: Path
    fd: int | None
    reasons: list[str]

    def close(self) -> None:
        """Close the directory descriptor if it is open."""

        if self.fd is None:
            return
        fd = self.fd
        self.fd = None
        os.close(fd)

    def __enter__(self) -> "EvidenceDirHandle":
        """Return this handle for context-manager use."""

        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        """Close the descriptor when leaving a context-manager block."""

        self.close()


def _dir_fd_operations_available() -> bool:
    """Return whether descriptor-relative no-follow reads are available."""

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


def _bounded_str(value: object, *, max_chars: int = DISPLAY_MAX_CHARS) -> str:
    """Return a single-line bounded string for API-safe metadata."""

    text = "" if value is None else str(value)
    text = " ".join(text.replace("\r", "\n").splitlines())
    if len(text) > max_chars:
        return text[: max_chars - 1] + "..."
    return text


def _safe_bool(value: object) -> bool | None:
    """Accept only real bool values."""

    return value if isinstance(value, bool) else None


def _safe_int(value: object) -> int | None:
    """Accept only real integer values, excluding bool."""

    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _open_dir_flags() -> int:
    """Build directory open flags that avoid following symlinks."""

    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    return flags


def _open_json_flags() -> int:
    """Build file open flags that avoid following symlinks."""

    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    return flags


def _evidence_dir_components(evidence_dir: Path) -> tuple[str, list[str], str | None]:
    """Split an evidence path into descriptor-walk components."""

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
            return start_path, [], "evidence_directory_unsafe_component"
        components.append(component)
    return start_path, components, None


def _normalize_macos_temp_alias(evidence_dir: Path) -> Path:
    """Rewrite macOS root temp aliases while preserving other symlink rejection."""

    if not evidence_dir.is_absolute():
        return evidence_dir
    parts = evidence_dir.parts
    if len(parts) < 3 or parts[1] not in {"tmp", "var"}:
        return evidence_dir

    alias_root = Path(os.sep) / parts[1]
    try:
        alias_metadata = os.lstat(alias_root)
    except OSError:
        return evidence_dir
    if not stat.S_ISLNK(alias_metadata.st_mode):
        return evidence_dir

    expected_target = Path(os.sep) / "private" / parts[1]
    try:
        resolved_target = alias_root.resolve(strict=True)
    except OSError:
        return evidence_dir
    if resolved_target != expected_target:
        return evidence_dir
    return resolved_target.joinpath(*parts[2:])


def _classify_open_error(exc: OSError) -> str:
    """Classify an evidence directory open failure into a stable reason."""

    if isinstance(exc, FileNotFoundError) or exc.errno == errno.ENOENT:
        return "evidence_directory_missing"
    if exc.errno == errno.ELOOP:
        return "evidence_directory_symlink"
    if exc.errno == errno.ENOTDIR:
        return "evidence_directory_not_directory"
    if exc.errno in {errno.EACCES, errno.EPERM}:
        return "evidence_directory_unreadable"
    return "evidence_directory_unavailable"


def _open_evidence_dir(evidence_dir: Path) -> EvidenceDirHandle:
    """Open the evidence directory using descriptor-safe traversal."""

    if not _dir_fd_operations_available():
        return EvidenceDirHandle(
            path=evidence_dir,
            fd=None,
            reasons=["evidence_safe_open_unavailable"],
        )

    physical_dir = _normalize_macos_temp_alias(evidence_dir)
    start_path, components, component_reason = _evidence_dir_components(physical_dir)
    if component_reason is not None:
        return EvidenceDirHandle(path=evidence_dir, fd=None, reasons=[component_reason])

    fd: int | None = None
    try:
        fd = os.open(start_path, _open_dir_flags())
        for component in components:
            next_fd = os.open(component, _open_dir_flags(), dir_fd=fd)
            os.close(fd)
            fd = next_fd
        metadata = os.fstat(fd)
    except OSError as exc:
        if fd is not None:
            os.close(fd)
        return EvidenceDirHandle(
            path=evidence_dir,
            fd=None,
            reasons=[_classify_open_error(exc)],
        )

    if not stat.S_ISDIR(metadata.st_mode):
        os.close(fd)
        return EvidenceDirHandle(
            path=evidence_dir,
            fd=None,
            reasons=["evidence_directory_not_directory"],
        )
    return EvidenceDirHandle(path=evidence_dir, fd=fd, reasons=[])


def _probe_expected_file(handle: EvidenceDirHandle, name: str) -> dict[str, object]:
    """Inspect one expected evidence file without following symlinks."""

    if handle.fd is None:
        return {
            "present": False,
            "readable": False,
            "reason": "evidence_directory_unavailable",
        }
    try:
        metadata = os.stat(name, dir_fd=handle.fd, follow_symlinks=False)
    except FileNotFoundError:
        return {"present": False, "readable": False, "reason": "missing"}
    except OSError:
        return {"present": False, "readable": False, "reason": "cannot_inspect"}

    if stat.S_ISLNK(metadata.st_mode):
        return {
            "present": True,
            "readable": False,
            "reason": "symlink",
            "size_bytes": metadata.st_size,
        }
    if not stat.S_ISREG(metadata.st_mode):
        return {
            "present": True,
            "readable": False,
            "reason": "non_regular",
            "size_bytes": metadata.st_size,
        }
    readable = os.access(name, os.R_OK, dir_fd=handle.fd, follow_symlinks=False)
    return {
        "present": True,
        "readable": bool(readable),
        "reason": "ok" if readable else "unreadable",
        "size_bytes": metadata.st_size,
    }


def _expected_files(handle: EvidenceDirHandle) -> dict[str, dict[str, object]]:
    """Return status for every known evidence file."""

    return {name: _probe_expected_file(handle, name) for name in EXPECTED_EVIDENCE_FILES}


def _read_json_bytes(handle: EvidenceDirHandle) -> tuple[bytes | None, list[str]]:
    """Read bounded structured metadata bytes from the evidence directory."""

    if handle.fd is None:
        return None, list(handle.reasons)

    fd: int | None = None
    try:
        fd = os.open("host-smoke-evidence.json", _open_json_flags(), dir_fd=handle.fd)
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode):
            return None, ["evidence_json_non_regular"]
        if metadata.st_size > JSON_MAX_BYTES:
            return None, ["evidence_json_oversized"]
        with os.fdopen(fd, "rb") as stream:
            fd = None
            raw = stream.read(JSON_MAX_BYTES + 1)
    except FileNotFoundError:
        return None, ["evidence_json_missing"]
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            return None, ["evidence_json_symlink"]
        return None, ["evidence_json_unavailable"]
    finally:
        if fd is not None:
            os.close(fd)

    if len(raw) > JSON_MAX_BYTES:
        return None, ["evidence_json_oversized"]
    return raw, []


def _parse_created_at(
    value: object,
    *,
    now: datetime,
) -> tuple[str | None, int | None, bool, list[str]]:
    """Parse created_at and return display value, age, stale flag, and reasons."""

    if not isinstance(value, str) or not value.strip():
        return None, None, False, ["evidence_created_at_malformed"]
    text = value.strip()
    parse_text = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        created_at = datetime.fromisoformat(parse_text)
    except ValueError:
        return _bounded_str(text), None, False, ["evidence_created_at_malformed"]
    if created_at.tzinfo is None:
        return _bounded_str(text), None, False, ["evidence_created_at_malformed"]
    age_seconds = int((now.astimezone(timezone.utc) - created_at.astimezone(timezone.utc)).total_seconds())
    if age_seconds < 0:
        return _bounded_str(text), 0, False, ["evidence_created_at_in_future"]
    return _bounded_str(text), age_seconds, age_seconds > STALE_AFTER_SECONDS, []


def _normalize_phases(value: object) -> dict[str, dict[str, object]]:
    """Normalize bounded phase metadata."""

    if not isinstance(value, Mapping):
        return {}
    phases: dict[str, dict[str, object]] = {}
    for raw_name, raw_details in list(value.items())[:MAX_PHASES]:
        name = _bounded_str(raw_name)
        if not name or not isinstance(raw_details, Mapping):
            continue
        phase: dict[str, object] = {}
        if "status" in raw_details:
            phase["status"] = _bounded_str(raw_details.get("status"))
        exit_code = _safe_int(raw_details.get("exit_code"))
        if exit_code is not None:
            phase["exit_code"] = exit_code
        if "timestamp" in raw_details:
            phase["timestamp"] = _bounded_str(raw_details.get("timestamp"))
        phases[name] = phase
    return phases


def _normalize_cleanup(value: object) -> dict[str, object]:
    """Normalize bounded cleanup metadata."""

    if not isinstance(value, Mapping):
        return {}
    cleanup: dict[str, object] = {}
    for key in CLEANUP_KEYS:
        raw = value.get(key)
        if isinstance(raw, bool):
            cleanup[key] = raw
        elif (integer := _safe_int(raw)) is not None:
            cleanup[key] = integer
        elif raw is not None:
            cleanup[key] = _bounded_str(raw)
    return cleanup


def _normalize_runtime_pointers(payload: Mapping[str, object]) -> dict[str, str]:
    """Return allowlisted bounded path pointers without dereferencing them."""

    pointers: dict[str, str] = {}
    for key in RUNTIME_POINTER_KEYS:
        value = payload.get(key)
        if value is not None and not isinstance(value, (dict, list, tuple, set)):
            pointers[key] = _bounded_str(value)
    return pointers


def _normalize_skip_flags(payload: Mapping[str, object]) -> tuple[dict[str, bool | None], list[str]]:
    """Return known skip flags, accepting only real bool values."""

    flags: dict[str, bool | None] = {}
    reasons: list[str] = []
    for key in ("skip_build", "skip_sign", "include_failure_drills"):
        value = _safe_bool(payload.get(key))
        flags[key] = value
        if value is None and key in payload:
            reasons.append("evidence_skip_flag_invalid")
    return flags, reasons


def _invalid_summary(
    *,
    evidence_dir: str | None,
    reasons: list[str],
    available: bool = False,
) -> dict[str, object]:
    """Build a configured invalid evidence summary."""

    summary: dict[str, object] = {
        "configured": True,
        "source": SOURCE_HOST_SMOKE_EVIDENCE,
        "available": available,
        "valid": False,
        "reasons": reasons,
    }
    if evidence_dir is not None:
        summary["evidence_dir"] = _bounded_str(evidence_dir)
    return summary


def collect_operator_evidence(
    *,
    environ: Mapping[str, str] | None = None,
    now: datetime | None = None,
) -> dict[str, object]:
    """Collect bounded operator evidence from the configured smoke bundle."""

    source_environ = os.environ if environ is None else environ
    evidence_dir_text = str(source_environ.get(ENV_VZ_EVIDENCE_DIR) or "").strip()
    if not evidence_dir_text:
        return {
            "configured": False,
            "source": SOURCE_HOST_SMOKE_EVIDENCE,
            "available": False,
            "valid": False,
            "reasons": ["evidence_not_configured"],
        }
    if "\x00" in evidence_dir_text:
        return _invalid_summary(
            evidence_dir=evidence_dir_text,
            reasons=["evidence_path_contains_nul"],
        )

    current_time = now or datetime.now(timezone.utc)
    evidence_dir = Path(evidence_dir_text)
    with _open_evidence_dir(evidence_dir) as handle:
        if handle.fd is None:
            return _invalid_summary(
                evidence_dir=evidence_dir_text,
                reasons=handle.reasons,
            )
        expected_files = _expected_files(handle)
        raw_bytes, read_reasons = _read_json_bytes(handle)
        if raw_bytes is None:
            return {
                **_invalid_summary(
                    evidence_dir=evidence_dir_text,
                    reasons=read_reasons,
                    available=True,
                ),
                "expected_files": expected_files,
            }

    try:
        payload = json.loads(raw_bytes.decode("utf-8"))
    except UnicodeDecodeError:
        return _invalid_summary(
            evidence_dir=evidence_dir_text,
            reasons=["evidence_json_malformed_utf8"],
            available=True,
        )
    except json.JSONDecodeError:
        return _invalid_summary(
            evidence_dir=evidence_dir_text,
            reasons=["evidence_json_malformed"],
            available=True,
        )
    if not isinstance(payload, Mapping):
        return _invalid_summary(
            evidence_dir=evidence_dir_text,
            reasons=["evidence_json_top_level_not_object"],
            available=True,
        )

    reasons: list[str] = []
    schema_version = _safe_int(payload.get("schema_version"))
    if schema_version is None:
        reasons.append("evidence_schema_version_missing")
    elif schema_version != 1:
        reasons.append("evidence_schema_version_unsupported")

    created_at, age_seconds, stale, created_reasons = _parse_created_at(
        payload.get("created_at"),
        now=current_time,
    )
    reasons.extend(created_reasons)

    final_exit_code = _safe_int(payload.get("final_exit_code"))
    if final_exit_code is None:
        reasons.append("evidence_final_exit_code_invalid")

    skip_flags, skip_reasons = _normalize_skip_flags(payload)
    reasons.extend(skip_reasons)

    summary: dict[str, object] = {
        "configured": True,
        "source": SOURCE_HOST_SMOKE_EVIDENCE,
        "available": True,
        "valid": not reasons,
        "evidence_dir": _bounded_str(evidence_dir_text),
        "schema_version": schema_version,
        "created_at": created_at,
        "age_seconds": age_seconds,
        "stale": stale,
        "smoke_run_id": _bounded_str(payload.get("smoke_run_id")),
        "final_exit_code": final_exit_code,
        "phases": _normalize_phases(payload.get("phases")),
        "cleanup": _normalize_cleanup(payload.get("cleanup")),
        "runtime_pointers": _normalize_runtime_pointers(payload),
        "expected_files": expected_files,
        "skip_flags": skip_flags,
        "reasons": reasons,
    }
    return summary
