"""Deterministic, redacted planning for canonical legacy webhook migration."""

from __future__ import annotations

import base64
import binascii
import errno
import hashlib
import hmac
import json
import os
import re
import secrets
import stat
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any

from tldw_Server_API.app.core.DB_Management.admin_webhooks_repository import (
    AdminWebhookRepository,
    LegacyImportDatabaseSnapshot,
    MigrationState,
    RegistrationInsert,
    RegistrationTarget,
)
from tldw_Server_API.app.core.exceptions import TransactionPassthroughError
from tldw_Server_API.app.services import admin_system_ops_service as system_ops

from .audit import (
    OperationalAction,
    OperationalAudit,
    OperationalAuditSink,
    OperationalOutcome,
    WebhookOperationalReasonCode,
    emit_mandatory_webhook_operation_audit,
)
from .catalog import EVENT_CATALOG, normalize_subscriptions
from .config import AdminWebhookMode, AdminWebhookSettings
from .crypto import (
    MIGRATION_DOMAIN_DATABASE_RECORD,
    MIGRATION_DOMAIN_DATABASE_TABLE,
    MIGRATION_DOMAIN_SYSTEM_OPS_RECORD,
    MIGRATION_DOMAIN_SYSTEM_OPS_SUBTREE,
    ProtectedValue,
    WebhookKeyRing,
)
from .domain import WebhookError, validate_webhook_target

MAX_CANONICAL_REGISTRATION_ID = 2**63 - 1
MAX_LEGACY_SOURCE_ROWS = 1_000
MAX_LEGACY_SECRET_BYTES = 16_384
REPORT_SCHEMA_VERSION = 1
INITIAL_PROTECTED_VERSION = 1

_SOURCE_KINDS = frozenset({"database", "system_ops"})
_SOURCE_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_REPORT_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_FINGERPRINT_PATTERN = re.compile(r"^hmac-sha256:[0-9a-f]{64}$")
_DIRECTORY_FSYNC_UNSUPPORTED_ERRNOS = frozenset(
    {
        errno.EINVAL,
        getattr(errno, "ENOTSUP", errno.EINVAL),
        getattr(errno, "EOPNOTSUPP", errno.EINVAL),
    }
)


class LegacyImportErrorCode(str, Enum):
    """Closed migration failures that never contain source material."""

    SOURCE_INVALID = "admin_webhook_legacy_source_invalid"
    SOURCE_CHANGED = "admin_webhook_legacy_source_changed"
    PATH_UNSAFE = "admin_webhook_legacy_path_unsafe"
    REPORT_INVALID = "admin_webhook_legacy_report_invalid"
    APPROVAL_MISMATCH = "admin_webhook_legacy_approval_mismatch"
    UNRESOLVED = "admin_webhook_legacy_unresolved"
    REGISTRATION_LIMIT = "admin_webhook_registration_limit"
    SEQUENCE_EXHAUSTED = "admin_webhook_sequence_exhausted"
    PRECONDITION_FAILED = "precondition_failed"
    AUDIT_UNAVAILABLE = "admin_webhook_audit_unavailable"
    ROLLBACK_WINDOW_CLOSED = "admin_webhook_rollback_window_closed"
    ROLLBACK_ARTIFACTS_ALREADY_RETIRED = "admin_webhook_rollback_artifacts_already_retired"
    ROLLBACK_ARTIFACTS_NOT_APPLICABLE = "admin_webhook_rollback_artifacts_not_applicable"
    OPERATION_FAILED = "admin_webhook_operation_failed"


class LegacyImportError(TransactionPassthroughError):
    """Sanitized expected failure at the offline migration boundary."""

    def __init__(self, code: LegacyImportErrorCode | str) -> None:
        self.code = LegacyImportErrorCode(code)
        super().__init__(self.code.value)


class LegacyRejectionReason(str, Enum):
    """Closed operator decisions that can suppress one exact source record."""

    RECEIVER_DECOMMISSIONED = "receiver_decommissioned"
    DUPLICATE_EXTERNAL_CONFIG = "duplicate_external_config"
    INVALID_LEGACY_RECORD = "invalid_legacy_record"
    OPERATOR_EXCLUDED = "operator_excluded"


@dataclass(frozen=True)
class LegacyImportRequest:
    """Operator-owned paths and explicit migration approval context."""

    report_path: Path
    backup_path: Path | None
    rollback_key_path: Path | None
    operator_id: int
    now: datetime
    allow_legacy_credential_decryption: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.report_path, Path):
            raise TypeError("report_path must be a Path")
        if self.backup_path is not None and not isinstance(self.backup_path, Path):
            raise TypeError("backup_path must be a Path")
        if self.rollback_key_path is not None and not isinstance(
            self.rollback_key_path,
            Path,
        ):
            raise TypeError("rollback_key_path must be a Path")
        if isinstance(self.operator_id, bool) or not isinstance(self.operator_id, int) or self.operator_id < 1:
            raise ValueError("operator_id must be a positive integer")
        if not isinstance(self.now, datetime) or self.now.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        if not isinstance(self.allow_legacy_credential_decryption, bool):
            raise TypeError("legacy credential decryption flag must be boolean")


@dataclass(frozen=True)
class LegacyAcceptedRecord:
    """Redacted projection of one source record accepted for import."""

    source_kind: str
    source_identity: str
    source_record_fingerprint: str
    canonical_id: int
    target_display: str
    target_hostname: str
    event_types: tuple[str, ...]
    timeout_seconds: int
    secret_rotation_required: bool = True


@dataclass(frozen=True)
class LegacyUnresolvedRecord:
    """Redacted source record that blocks apply until repaired or rejected."""

    source_kind: str
    source_identity: str
    source_record_fingerprint: str
    reason_code: str


@dataclass(frozen=True)
class LegacyRejectedRecord:
    """Current fingerprint-bound operator rejection included in a plan."""

    source_kind: str
    source_identity: str
    source_record_fingerprint: str
    reason_code: str
    operator_id: int


@dataclass(frozen=True)
class LegacyImportPlan:
    """Deterministic redacted migration plan reviewed before apply."""

    operation_id: str
    fingerprint_key_id: str
    legacy_credential_decryption_enabled: bool
    source_fingerprints: Mapping[str, str]
    accepted: tuple[LegacyAcceptedRecord, ...]
    unresolved: tuple[LegacyUnresolvedRecord, ...]
    explicitly_rejected: tuple[LegacyRejectedRecord, ...]
    projected_non_deleted_count: int
    source_mapping: Mapping[str, int]
    requires_system_ops_backup: bool
    report_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_fingerprints",
            MappingProxyType(dict(self.source_fingerprints)),
        )
        object.__setattr__(
            self,
            "source_mapping",
            MappingProxyType(dict(self.source_mapping)),
        )


@dataclass(frozen=True)
class _PreparedRecord:
    source_kind: str
    source_identity: str
    source_record_fingerprint: str
    requested_id: int | None
    url: str = field(repr=False)
    secret_bytes: bytes = field(repr=False)
    target_display: str
    target_hostname: str
    event_types: tuple[str, ...]
    description: str = field(repr=False)
    timeout_seconds: int

    @property
    def source_key(self) -> str:
        return f"{self.source_kind}:{self.source_identity}"


@dataclass(frozen=True)
class _RawRecord:
    source_kind: str
    source_identity: str
    source_record_fingerprint: str
    values: Mapping[str, object] = field(repr=False)

    @property
    def source_key(self) -> str:
        return f"{self.source_kind}:{self.source_identity}"


@dataclass(frozen=True)
class _ImportSnapshot:
    store: Mapping[str, object] = field(repr=False)
    store_bytes: bytes = field(repr=False)
    database: LegacyImportDatabaseSnapshot = field(repr=False)
    source_fingerprints: Mapping[str, str]
    fingerprint_key_id: str
    requires_system_ops_backup: bool


@dataclass(frozen=True)
class _NormalizedPaths:
    report: Path
    backup: Path | None
    rollback_key: Path | None


@dataclass(frozen=True)
class _FileEvidence:
    owner_id: int
    group_id: int
    mode: int
    identity: str


class _RecordIssue(Exception):
    def __init__(self, reason_code: str) -> None:
        self.reason_code = reason_code
        super().__init__(reason_code)


class LegacySecretDecryptor:
    """Migration-only wrapper around the historical unrelated-key fallback."""

    def decrypt(self, encrypted_blob: str) -> str:
        from tldw_Server_API.app.core.AuthNZ.admin_webhook_secrets import (
            decrypt_admin_webhook_secret,
        )

        try:
            plaintext = decrypt_admin_webhook_secret(encrypted_blob)
        except Exception:  # noqa: BLE001 - old helper errors must remain redacted
            raise _RecordIssue("legacy_credential_decryption_failed") from None
        if not isinstance(plaintext, str) or not plaintext:
            raise _RecordIssue("legacy_credential_decryption_failed")
        return plaintext


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError):
        raise LegacyImportError(LegacyImportErrorCode.SOURCE_INVALID) from None


def _json_safe(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        normalized = value
        if normalized.tzinfo is None:
            normalized = normalized.replace(tzinfo=timezone.utc)
        return normalized.astimezone(timezone.utc).isoformat()
    if isinstance(value, bytes):
        return {"bytes_b64": base64.b64encode(value).decode("ascii")}
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    return {"type": type(value).__name__}


def _source_identity(value: object, position: int) -> str:
    if isinstance(value, bool) or value is None:
        return f"row-{position + 1:06d}"
    if isinstance(value, int):
        candidate = str(value)
    elif isinstance(value, str):
        candidate = value.strip()
    else:
        return f"row-{position + 1:06d}"
    if _SOURCE_ID_PATTERN.fullmatch(candidate) is None:
        return f"row-{position + 1:06d}"
    return candidate


def _requested_numeric_id(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        candidate = value
    elif isinstance(value, str) and value.isdigit():
        candidate = int(value)
    else:
        return None
    return candidate if 1 <= candidate <= MAX_CANONICAL_REGISTRATION_ID else None


def _normalize_events(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        if value == "*":
            raw_events: object = ["*"]
        else:
            try:
                raw_events = json.loads(value)
            except (json.JSONDecodeError, RecursionError, TypeError, ValueError):
                raise _RecordIssue("event_types_invalid") from None
    else:
        raw_events = value
    if not isinstance(raw_events, list) or any(not isinstance(item, str) for item in raw_events):
        raise _RecordIssue("event_types_invalid")
    if "*" in raw_events:
        return tuple(item.event_type for item in EVENT_CATALOG)
    event_types = tuple(item for item in raw_events if isinstance(item, str))
    try:
        normalized: tuple[str, ...] = normalize_subscriptions(event_types)
        return normalized
    except WebhookError:
        raise _RecordIssue("event_types_invalid") from None


def _normalize_timeout(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 30:
        raise _RecordIssue("timeout_invalid")
    return value


def _normalize_description(value: object) -> str:
    if value is None:
        return ""
    if not isinstance(value, str) or len(value) > 500:
        raise _RecordIssue("description_invalid")
    return value


def _normalize_secret(value: object) -> bytes:
    if not isinstance(value, str) or not value:
        raise _RecordIssue("secret_missing")
    encoded = value.encode("utf-8")
    if len(encoded) > MAX_LEGACY_SECRET_BYTES:
        raise _RecordIssue("secret_invalid")
    return encoded


def _path_exists(path: Path) -> bool:
    return os.path.lexists(path)


def _normalize_output_path(path: Path) -> Path:
    expanded = Path(os.path.abspath(os.path.expanduser(str(path))))
    if _path_exists(expanded):
        metadata = os.lstat(expanded)
        if stat.S_ISLNK(metadata.st_mode):
            raise LegacyImportError(LegacyImportErrorCode.PATH_UNSAFE)
    try:
        parent = expanded.parent.resolve(strict=True)
        parent_metadata = parent.stat()
    except (OSError, RuntimeError):
        raise LegacyImportError(LegacyImportErrorCode.PATH_UNSAFE) from None
    if (
        not stat.S_ISDIR(parent_metadata.st_mode)
        or parent_metadata.st_uid != os.geteuid()
        or stat.S_IMODE(parent_metadata.st_mode) & 0o022
    ):
        raise LegacyImportError(LegacyImportErrorCode.PATH_UNSAFE)
    return parent / expanded.name


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        try:
            os.fsync(descriptor)
        except OSError as exc:
            if exc.errno not in _DIRECTORY_FSYNC_UNSUPPORTED_ERRNOS:
                raise
    finally:
        os.close(descriptor)


def _publish_private_report(path: Path, payload: bytes) -> None:
    if _path_exists(path):
        metadata = os.lstat(path)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.geteuid():
            raise LegacyImportError(LegacyImportErrorCode.PATH_UNSAFE)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    descriptor_open = True
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor_open = False
            if stream.write(payload) != len(payload):
                raise OSError("incomplete report write")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
        metadata = path.stat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or path.read_bytes() != payload
        ):
            raise LegacyImportError(LegacyImportErrorCode.PATH_UNSAFE)
    except LegacyImportError:
        raise
    except OSError:
        raise LegacyImportError(LegacyImportErrorCode.PATH_UNSAFE) from None
    finally:
        if descriptor_open:
            try:
                os.close(descriptor)
            except OSError:
                pass
        temporary.unlink(missing_ok=True)


def _read_private_file(path: Path, *, maximum_bytes: int = 70_000_000) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        raise LegacyImportError(LegacyImportErrorCode.PATH_UNSAFE) from None
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_size > maximum_bytes
        ):
            raise LegacyImportError(LegacyImportErrorCode.PATH_UNSAFE)
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1_048_576, maximum_bytes - total + 1))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum_bytes:
                raise LegacyImportError(LegacyImportErrorCode.PATH_UNSAFE)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _file_evidence(path: Path, *, expected_payload: bytes | None = None) -> _FileEvidence:
    payload = _read_private_file(path)
    if expected_payload is not None and not hmac.compare_digest(payload, expected_payload):
        raise LegacyImportError(LegacyImportErrorCode.PATH_UNSAFE)
    metadata = os.lstat(path)
    return _FileEvidence(
        owner_id=metadata.st_uid,
        group_id=metadata.st_gid,
        mode=stat.S_IMODE(metadata.st_mode),
        identity=f"{metadata.st_dev}:{metadata.st_ino}",
    )


def _staging_path(final_path: Path, operation_id: str) -> Path:
    return final_path.with_name(f".{final_path.name}.{operation_id}.staging")


def _publish_exclusive_artifact(
    final_path: Path,
    staging_path: Path,
    payload: bytes,
) -> _FileEvidence:
    if _path_exists(final_path) or _path_exists(staging_path):
        raise LegacyImportError(LegacyImportErrorCode.PATH_UNSAFE)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    linked = False
    try:
        descriptor = os.open(staging_path, flags, 0o600)
        os.fchmod(descriptor, 0o600)
        written = 0
        while written < len(payload):
            count = os.write(descriptor, payload[written:])
            if count < 1:
                raise OSError("incomplete artifact write")
            written += count
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        _file_evidence(staging_path, expected_payload=payload)
        os.link(staging_path, final_path, follow_symlinks=False)
        linked = True
        _fsync_directory(final_path.parent)
        evidence = _file_evidence(final_path, expected_payload=payload)
        staging_path.unlink()
        _fsync_directory(final_path.parent)
        return evidence
    except LegacyImportError:
        raise
    except OSError:
        raise LegacyImportError(LegacyImportErrorCode.PATH_UNSAFE) from None
    finally:
        if descriptor is not None:
            with suppress(OSError):
                os.close(descriptor)
        if not linked:
            with suppress(OSError):
                staging_path.unlink()


def _publish_or_resume_artifact(
    final_path: Path,
    staging_path: Path,
    payload: bytes,
) -> _FileEvidence:
    if _path_exists(final_path):
        final_evidence = _file_evidence(final_path, expected_payload=payload)
        if _path_exists(staging_path):
            staging_evidence = _file_evidence(staging_path, expected_payload=payload)
            if staging_evidence.identity != final_evidence.identity:
                raise LegacyImportError(LegacyImportErrorCode.PATH_UNSAFE)
            staging_path.unlink()
            _fsync_directory(final_path.parent)
        return final_evidence
    if _path_exists(staging_path):
        try:
            _file_evidence(staging_path, expected_payload=payload)
        except LegacyImportError:
            staging_path.unlink()
            _fsync_directory(staging_path.parent)
            return _publish_exclusive_artifact(final_path, staging_path, payload)
        try:
            os.link(staging_path, final_path, follow_symlinks=False)
            _fsync_directory(final_path.parent)
            evidence = _file_evidence(final_path, expected_payload=payload)
            staging_path.unlink()
            _fsync_directory(final_path.parent)
            return evidence
        except OSError:
            raise LegacyImportError(LegacyImportErrorCode.PATH_UNSAFE) from None
    return _publish_exclusive_artifact(final_path, staging_path, payload)


def _strict_json_object(payload: bytes, *, maximum_bytes: int) -> dict[str, object]:
    if len(payload) > maximum_bytes:
        raise LegacyImportError(LegacyImportErrorCode.OPERATION_FAILED)

    def object_hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate JSON key")
            value[key] = item
        return value

    try:
        value = json.loads(payload, object_pairs_hook=object_hook)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, TypeError, ValueError):
        raise LegacyImportError(LegacyImportErrorCode.OPERATION_FAILED) from None
    if not isinstance(value, dict):
        raise LegacyImportError(LegacyImportErrorCode.OPERATION_FAILED)
    return value


def _publish_exclusive_output(path: Path, payload: bytes) -> _FileEvidence:
    if _path_exists(path):
        raise LegacyImportError(LegacyImportErrorCode.PATH_UNSAFE)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    created = False
    try:
        descriptor = os.open(path, flags, 0o600)
        created = True
        os.fchmod(descriptor, 0o600)
        written = 0
        while written < len(payload):
            count = os.write(descriptor, payload[written:])
            if count < 1:
                raise OSError("incomplete output write")
            written += count
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        _fsync_directory(path.parent)
        return _file_evidence(path, expected_payload=payload)
    except LegacyImportError:
        if created:
            with suppress(OSError):
                path.unlink()
                _fsync_directory(path.parent)
        raise
    except OSError:
        if created:
            with suppress(OSError):
                path.unlink()
                _fsync_directory(path.parent)
        raise LegacyImportError(LegacyImportErrorCode.PATH_UNSAFE) from None
    finally:
        if descriptor is not None:
            with suppress(OSError):
                os.close(descriptor)


def _remove_published_output_if_same(path: Path, evidence: _FileEvidence) -> None:
    try:
        metadata = os.lstat(path)
    except OSError:
        return
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != evidence.owner_id
        or metadata.st_gid != evidence.group_id
        or stat.S_IMODE(metadata.st_mode) != evidence.mode
        or f"{metadata.st_dev}:{metadata.st_ino}" != evidence.identity
    ):
        return
    with suppress(OSError):
        path.unlink()
        _fsync_directory(path.parent)


def _record_payload(
    record: LegacyAcceptedRecord | LegacyUnresolvedRecord | LegacyRejectedRecord,
) -> dict[str, object]:
    if isinstance(record, LegacyAcceptedRecord):
        return {
            "source_kind": record.source_kind,
            "source_identity": record.source_identity,
            "source_record_fingerprint": record.source_record_fingerprint,
            "canonical_id": record.canonical_id,
            "target_display": record.target_display,
            "target_hostname": record.target_hostname,
            "event_types": list(record.event_types),
            "timeout_seconds": record.timeout_seconds,
            "secret_rotation_required": record.secret_rotation_required,
        }
    if isinstance(record, LegacyUnresolvedRecord):
        return {
            "source_kind": record.source_kind,
            "source_identity": record.source_identity,
            "source_record_fingerprint": record.source_record_fingerprint,
            "reason_code": record.reason_code,
        }
    return {
        "source_kind": record.source_kind,
        "source_identity": record.source_identity,
        "source_record_fingerprint": record.source_record_fingerprint,
        "reason_code": record.reason_code,
        "operator_id": record.operator_id,
    }


def _report_payload_object(plan: LegacyImportPlan) -> dict[str, object]:
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "operation_id": plan.operation_id,
        "fingerprint_key_id": plan.fingerprint_key_id,
        "legacy_credential_decryption_enabled": (plan.legacy_credential_decryption_enabled),
        "source_fingerprints": dict(sorted(plan.source_fingerprints.items())),
        "accepted": [_record_payload(record) for record in plan.accepted],
        "unresolved": [_record_payload(record) for record in plan.unresolved],
        "explicitly_rejected": [_record_payload(record) for record in plan.explicitly_rejected],
        "projected_non_deleted_count": plan.projected_non_deleted_count,
        "source_mapping": dict(sorted(plan.source_mapping.items())),
        "requires_system_ops_backup": plan.requires_system_ops_backup,
    }


def canonical_report_payload(plan: LegacyImportPlan) -> bytes:
    """Encode only the versioned, deterministic, redacted approval payload."""
    if not isinstance(plan, LegacyImportPlan):
        raise TypeError("LegacyImportPlan is required")
    return _canonical_json_bytes(_report_payload_object(plan))


def _plan_from_report_bytes(payload: bytes) -> LegacyImportPlan:
    envelope = _strict_json_object(payload, maximum_bytes=2_097_152)
    payload_keys = {
        "schema_version",
        "operation_id",
        "fingerprint_key_id",
        "legacy_credential_decryption_enabled",
        "source_fingerprints",
        "accepted",
        "unresolved",
        "explicitly_rejected",
        "projected_non_deleted_count",
        "source_mapping",
        "requires_system_ops_backup",
    }
    if set(envelope) != payload_keys | {"generated_at", "report_digest"}:
        raise LegacyImportError(LegacyImportErrorCode.REPORT_INVALID)
    report_digest = envelope.get("report_digest")
    operation_id = envelope.get("operation_id")
    fingerprint_key_id = envelope.get("fingerprint_key_id")
    credential_decryption = envelope.get("legacy_credential_decryption_enabled")
    source_fingerprints = envelope.get("source_fingerprints")
    source_mapping = envelope.get("source_mapping")
    projected_count = envelope.get("projected_non_deleted_count")
    requires_backup = envelope.get("requires_system_ops_backup")
    if (
        envelope.get("schema_version") != REPORT_SCHEMA_VERSION
        or not isinstance(operation_id, str)
        or re.fullmatch(r"whmig_[0-9a-f]{32}", operation_id) is None
        or not isinstance(fingerprint_key_id, str)
        or not fingerprint_key_id
        or not isinstance(credential_decryption, bool)
        or not isinstance(source_fingerprints, dict)
        or set(source_fingerprints) != _SOURCE_KINDS
        or any(
            not isinstance(value, str) or _FINGERPRINT_PATTERN.fullmatch(value) is None
            for value in source_fingerprints.values()
        )
        or not isinstance(source_mapping, dict)
        or any(
            not isinstance(key, str)
            or isinstance(value, bool)
            or not isinstance(value, int)
            or not 1 <= value <= MAX_CANONICAL_REGISTRATION_ID
            for key, value in source_mapping.items()
        )
        or isinstance(projected_count, bool)
        or not isinstance(projected_count, int)
        or projected_count < 0
        or not isinstance(requires_backup, bool)
        or not isinstance(report_digest, str)
        or _REPORT_DIGEST_PATTERN.fullmatch(report_digest) is None
    ):
        raise LegacyImportError(LegacyImportErrorCode.REPORT_INVALID)

    def common(record: object) -> tuple[dict[str, object], str, str, str]:
        if not isinstance(record, dict):
            raise LegacyImportError(LegacyImportErrorCode.REPORT_INVALID)
        source_kind = record.get("source_kind")
        source_identity = record.get("source_identity")
        source_record_fingerprint = record.get("source_record_fingerprint")
        if (
            not isinstance(source_kind, str)
            or source_kind not in _SOURCE_KINDS
            or not isinstance(source_identity, str)
            or _SOURCE_ID_PATTERN.fullmatch(source_identity) is None
            or not isinstance(source_record_fingerprint, str)
            or _FINGERPRINT_PATTERN.fullmatch(source_record_fingerprint) is None
        ):
            raise LegacyImportError(LegacyImportErrorCode.REPORT_INVALID)
        return record, source_kind, source_identity, source_record_fingerprint

    raw_accepted = envelope.get("accepted")
    raw_unresolved = envelope.get("unresolved")
    raw_rejected = envelope.get("explicitly_rejected")
    if not isinstance(raw_accepted, list):
        raise LegacyImportError(LegacyImportErrorCode.REPORT_INVALID)
    if not isinstance(raw_unresolved, list):
        raise LegacyImportError(LegacyImportErrorCode.REPORT_INVALID)
    if not isinstance(raw_rejected, list):
        raise LegacyImportError(LegacyImportErrorCode.REPORT_INVALID)
    accepted: list[LegacyAcceptedRecord] = []
    for item in raw_accepted:
        record, source_kind, source_identity, source_record_fingerprint = common(item)
        if set(record) != {
            "source_kind",
            "source_identity",
            "source_record_fingerprint",
            "canonical_id",
            "target_display",
            "target_hostname",
            "event_types",
            "timeout_seconds",
            "secret_rotation_required",
        }:
            raise LegacyImportError(LegacyImportErrorCode.REPORT_INVALID)
        canonical_id = record.get("canonical_id")
        target_display = record.get("target_display")
        target_hostname = record.get("target_hostname")
        event_types = record.get("event_types")
        timeout_seconds = record.get("timeout_seconds")
        if (
            isinstance(canonical_id, bool)
            or not isinstance(canonical_id, int)
            or not 1 <= canonical_id <= MAX_CANONICAL_REGISTRATION_ID
            or not isinstance(target_display, str)
            or not isinstance(target_hostname, str)
            or not isinstance(event_types, list)
            or any(not isinstance(value, str) for value in event_types)
            or isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, int)
            or not 1 <= timeout_seconds <= 30
            or record.get("secret_rotation_required") is not True
        ):
            raise LegacyImportError(LegacyImportErrorCode.REPORT_INVALID)
        normalized_event_types = tuple(value for value in event_types if isinstance(value, str))
        accepted.append(
            LegacyAcceptedRecord(
                source_kind=source_kind,
                source_identity=source_identity,
                source_record_fingerprint=source_record_fingerprint,
                canonical_id=canonical_id,
                target_display=target_display,
                target_hostname=target_hostname,
                event_types=normalized_event_types,
                timeout_seconds=timeout_seconds,
            )
        )
    unresolved: list[LegacyUnresolvedRecord] = []
    for item in raw_unresolved:
        record, source_kind, source_identity, source_record_fingerprint = common(item)
        reason_code = record.get("reason_code")
        if set(record) != {
            "source_kind",
            "source_identity",
            "source_record_fingerprint",
            "reason_code",
        } or not isinstance(reason_code, str):
            raise LegacyImportError(LegacyImportErrorCode.REPORT_INVALID)
        unresolved.append(
            LegacyUnresolvedRecord(
                source_kind=source_kind,
                source_identity=source_identity,
                source_record_fingerprint=source_record_fingerprint,
                reason_code=reason_code,
            )
        )
    rejected: list[LegacyRejectedRecord] = []
    for item in raw_rejected:
        record, source_kind, source_identity, source_record_fingerprint = common(item)
        reason_code = record.get("reason_code")
        rejection_operator_id = record.get("operator_id")
        if (
            set(record)
            != {
                "source_kind",
                "source_identity",
                "source_record_fingerprint",
                "reason_code",
                "operator_id",
            }
            or not isinstance(reason_code, str)
            or reason_code not in {value.value for value in LegacyRejectionReason}
            or isinstance(rejection_operator_id, bool)
            or not isinstance(rejection_operator_id, int)
            or rejection_operator_id < 1
        ):
            raise LegacyImportError(LegacyImportErrorCode.REPORT_INVALID)
        rejected.append(
            LegacyRejectedRecord(
                source_kind=source_kind,
                source_identity=source_identity,
                source_record_fingerprint=source_record_fingerprint,
                reason_code=reason_code,
                operator_id=rejection_operator_id,
            )
        )
    plan = LegacyImportPlan(
        operation_id=operation_id,
        fingerprint_key_id=fingerprint_key_id,
        legacy_credential_decryption_enabled=credential_decryption,
        source_fingerprints=source_fingerprints,
        accepted=tuple(accepted),
        unresolved=tuple(unresolved),
        explicitly_rejected=tuple(rejected),
        projected_non_deleted_count=projected_count,
        source_mapping=source_mapping,
        requires_system_ops_backup=requires_backup,
        report_digest=report_digest,
    )
    computed = "sha256:" + hashlib.sha256(canonical_report_payload(plan)).hexdigest()
    if not hmac.compare_digest(computed, report_digest):
        raise LegacyImportError(LegacyImportErrorCode.REPORT_INVALID)
    accepted_mapping = {
        f"{record.source_kind}:{record.source_identity}": record.canonical_id for record in plan.accepted
    }
    if accepted_mapping != dict(plan.source_mapping):
        raise LegacyImportError(LegacyImportErrorCode.REPORT_INVALID)
    return plan


class LegacyImportService:
    """Build and later apply one reviewed legacy webhook migration plan."""

    def __init__(
        self,
        *,
        repository: AdminWebhookRepository,
        key_ring: WebhookKeyRing,
        settings: AdminWebhookSettings,
        system_ops_path: Path | None = None,
        application_data_paths: tuple[Path, ...] = (),
        legacy_secret_decryptor: LegacySecretDecryptor | None = None,
        audit_sink: OperationalAuditSink = emit_mandatory_webhook_operation_audit,
        failure_injector: Callable[[str], None] | None = None,
    ) -> None:
        if not isinstance(repository, AdminWebhookRepository):
            raise TypeError("repository is required")
        if not isinstance(key_ring, WebhookKeyRing):
            raise TypeError("key_ring is required")
        if not isinstance(settings, AdminWebhookSettings):
            raise TypeError("settings are required")
        if any(not isinstance(path, Path) for path in application_data_paths):
            raise TypeError("application data paths must be Paths")
        if not callable(audit_sink):
            raise TypeError("audit_sink is required")
        if failure_injector is not None and not callable(failure_injector):
            raise TypeError("failure_injector must be callable")
        self._repository = repository
        self._key_ring = key_ring
        self._settings = settings
        self._system_ops_path = system_ops_path or system_ops._STORE_PATH
        self._application_data_paths = tuple(path.expanduser().resolve(strict=False) for path in application_data_paths)
        self._legacy_secret_decryptor = legacy_secret_decryptor
        self._audit_sink = audit_sink
        self._failure_injector = failure_injector

    def _checkpoint(self, name: str) -> None:
        if self._failure_injector is not None:
            self._failure_injector(name)

    def _normalize_paths(
        self,
        request: LegacyImportRequest,
        *,
        requires_backup: bool,
        resume_state: MigrationState | None = None,
    ) -> _NormalizedPaths:
        report = _normalize_output_path(request.report_path)
        backup = _normalize_output_path(request.backup_path) if request.backup_path is not None else None
        rollback_key = (
            _normalize_output_path(request.rollback_key_path) if request.rollback_key_path is not None else None
        )
        if requires_backup and (backup is None or rollback_key is None):
            raise LegacyImportError(LegacyImportErrorCode.PATH_UNSAFE)
        outputs = [path for path in (report, backup, rollback_key) if path is not None]
        if len(set(outputs)) != len(outputs):
            raise LegacyImportError(LegacyImportErrorCode.PATH_UNSAFE)

        source_paths = {self._system_ops_path.expanduser().resolve(strict=False)}
        if self._repository.database_path is not None:
            source_paths.add(self._repository.database_path.resolve(strict=False))
        if any(path in source_paths for path in outputs):
            raise LegacyImportError(LegacyImportErrorCode.PATH_UNSAFE)
        for path in (backup, rollback_key):
            if path is None:
                continue
            if any(_is_within(path, root) for root in self._application_data_paths):
                raise LegacyImportError(LegacyImportErrorCode.PATH_UNSAFE)
            claimed = resume_state is not None and (
                str(path) == resume_state.active_backup_path or str(path) == resume_state.active_key_path
            )
            if _path_exists(path) and not claimed:
                raise LegacyImportError(LegacyImportErrorCode.PATH_UNSAFE)
        return _NormalizedPaths(report=report, backup=backup, rollback_key=rollback_key)

    def _read_store(self) -> tuple[dict[str, Any], bytes]:
        try:
            with system_ops._STORE_LOCK, system_ops._store_file_lock(store_path=self._system_ops_path):
                store, raw = system_ops._read_store_strict(self._system_ops_path)
        except (OSError, RuntimeError, ValueError):
            raise LegacyImportError(LegacyImportErrorCode.SOURCE_INVALID) from None
        return store, raw

    def _fingerprint(
        self,
        domain: str,
        value: object,
    ) -> tuple[str, str]:
        return self._key_ring.fingerprint_migration_source(
            domain,
            _canonical_json_bytes(_json_safe(value)),
        )

    def _compose_snapshot(
        self,
        *,
        store: Mapping[str, object],
        store_bytes: bytes,
        database: LegacyImportDatabaseSnapshot,
    ) -> _ImportSnapshot:
        subtree = {
            "schema": 1,
            "webhooks_present": "webhooks" in store,
            "webhooks": store.get("webhooks"),
            "webhook_deliveries_present": "webhook_deliveries" in store,
            "webhook_deliveries": store.get("webhook_deliveries"),
        }
        key_id, system_fingerprint = self._fingerprint(
            MIGRATION_DOMAIN_SYSTEM_OPS_SUBTREE,
            subtree,
        )
        table = {
            "schema": 1,
            "table_present": database.table_present,
            "rows": [_json_safe(row.values) for row in database.rows],
        }
        database_key_id, database_fingerprint = self._fingerprint(
            MIGRATION_DOMAIN_DATABASE_TABLE,
            table,
        )
        if database_key_id != key_id:
            raise LegacyImportError(LegacyImportErrorCode.PRECONDITION_FAILED)
        return _ImportSnapshot(
            store=MappingProxyType(dict(store)),
            store_bytes=store_bytes,
            database=database,
            source_fingerprints=MappingProxyType(
                {
                    "database": database_fingerprint,
                    "system_ops": system_fingerprint,
                }
            ),
            fingerprint_key_id=key_id,
            requires_system_ops_backup=("webhooks" in store or "webhook_deliveries" in store),
        )

    async def _snapshot(self) -> _ImportSnapshot:
        store, store_bytes = self._read_store()
        database = await self._repository.get_legacy_import_snapshot()
        return self._compose_snapshot(
            store=store,
            store_bytes=store_bytes,
            database=database,
        )

    def _raw_records(self, snapshot: _ImportSnapshot) -> tuple[_RawRecord, ...]:
        raw_records: list[_RawRecord] = []
        raw_system = snapshot.store.get("webhooks", [])
        if not isinstance(raw_system, list):
            _, fingerprint = self._fingerprint(
                MIGRATION_DOMAIN_SYSTEM_OPS_RECORD,
                {"position": 0, "value": raw_system},
            )
            raw_records.append(
                _RawRecord(
                    source_kind="system_ops",
                    source_identity="collection",
                    source_record_fingerprint=fingerprint,
                    values=MappingProxyType({"_invalid_collection": raw_system}),
                )
            )
        else:
            for position, value in enumerate(raw_system):
                identity_value = value.get("id") if isinstance(value, Mapping) else None
                identity = _source_identity(identity_value, position)
                _, fingerprint = self._fingerprint(
                    MIGRATION_DOMAIN_SYSTEM_OPS_RECORD,
                    {"position": position, "value": value},
                )
                values = dict(value) if isinstance(value, Mapping) else {"_invalid_record": value}
                raw_records.append(
                    _RawRecord(
                        source_kind="system_ops",
                        source_identity=identity,
                        source_record_fingerprint=fingerprint,
                        values=MappingProxyType(values),
                    )
                )

        for row in snapshot.database.rows:
            _, fingerprint = self._fingerprint(
                MIGRATION_DOMAIN_DATABASE_RECORD,
                row.values,
            )
            raw_records.append(
                _RawRecord(
                    source_kind="database",
                    source_identity=row.source_identity,
                    source_record_fingerprint=fingerprint,
                    values=row.values,
                )
            )
        if len(raw_records) > MAX_LEGACY_SOURCE_ROWS:
            raise LegacyImportError(LegacyImportErrorCode.SOURCE_INVALID)
        return tuple(
            sorted(
                raw_records,
                key=lambda record: (
                    record.source_kind,
                    record.source_identity,
                    record.source_record_fingerprint,
                ),
            )
        )

    def _current_rejections(
        self,
        state: MigrationState,
    ) -> dict[tuple[str, str, str], LegacyRejectedRecord]:
        decisions: dict[tuple[str, str, str], LegacyRejectedRecord] = {}
        for value in state.source_rejections:
            if not isinstance(value, Mapping):
                raise LegacyImportError(LegacyImportErrorCode.SOURCE_INVALID)
            try:
                source_kind = value["source_kind"]
                source_identity = value["source_identity"]
                fingerprint = value["source_record_fingerprint"]
                reason_code = value["reason_code"]
                operator_id = value["operator_id"]
                fingerprint_key_id = value["fingerprint_key_id"]
            except KeyError:
                raise LegacyImportError(LegacyImportErrorCode.SOURCE_INVALID) from None
            if (
                source_kind not in _SOURCE_KINDS
                or not isinstance(source_identity, str)
                or _SOURCE_ID_PATTERN.fullmatch(source_identity) is None
                or not isinstance(fingerprint, str)
                or _FINGERPRINT_PATTERN.fullmatch(fingerprint) is None
                or reason_code not in {item.value for item in LegacyRejectionReason}
                or isinstance(operator_id, bool)
                or not isinstance(operator_id, int)
                or operator_id < 1
                or fingerprint_key_id != self._key_ring.primary_id
            ):
                raise LegacyImportError(LegacyImportErrorCode.SOURCE_INVALID)
            decision = LegacyRejectedRecord(
                source_kind=source_kind,
                source_identity=source_identity,
                source_record_fingerprint=fingerprint,
                reason_code=reason_code,
                operator_id=operator_id,
            )
            decisions[(source_kind, source_identity, fingerprint)] = decision
        return decisions

    def _prepare_record(
        self,
        raw: _RawRecord,
        *,
        allow_legacy_credential_decryption: bool,
    ) -> _PreparedRecord:
        values = raw.values
        if "_invalid_collection" in values:
            raise _RecordIssue("collection_invalid")
        if "_invalid_record" in values:
            raise _RecordIssue("record_invalid")
        url = values.get("url")
        if not isinstance(url, str):
            raise _RecordIssue("target_invalid")
        try:
            target = validate_webhook_target(
                url,
                allow_http_dev=self._settings.allow_http_dev,
                allow_e2e_loopback=self._settings.allow_e2e_loopback,
            )
        except WebhookError:
            raise _RecordIssue("target_rejected") from None

        if raw.source_kind == "system_ops":
            secret_bytes = _normalize_secret(values.get("secret"))
            events = _normalize_events(values.get("events"))
            timeout = _normalize_timeout(values.get("timeout_seconds", 10))
        else:
            plaintext = values.get("secret")
            encrypted = values.get("secret_encrypted")
            if isinstance(plaintext, str) and plaintext:
                if isinstance(encrypted, str) and encrypted:
                    raise _RecordIssue("secret_fields_ambiguous")
                secret_bytes = _normalize_secret(plaintext)
            elif isinstance(encrypted, str) and encrypted:
                if not allow_legacy_credential_decryption:
                    raise _RecordIssue("legacy_credential_decryption_required")
                decryptor = self._legacy_secret_decryptor or LegacySecretDecryptor()
                secret_bytes = _normalize_secret(decryptor.decrypt(encrypted))
            else:
                raise _RecordIssue("secret_missing")
            events = _normalize_events(values.get("event_types"))
            timeout = _normalize_timeout(values.get("timeout_seconds", 10))

        return _PreparedRecord(
            source_kind=raw.source_kind,
            source_identity=raw.source_identity,
            source_record_fingerprint=raw.source_record_fingerprint,
            requested_id=_requested_numeric_id(values.get("id")),
            url=target.url,
            secret_bytes=secret_bytes,
            target_display=target.target_display,
            target_hostname=target.hostname,
            event_types=events,
            description=_normalize_description(values.get("description")),
            timeout_seconds=timeout,
        )

    def _allocate_mapping(
        self,
        records: tuple[_PreparedRecord, ...],
        database: LegacyImportDatabaseSnapshot,
    ) -> dict[str, int]:
        existing = set(database.canonical_registration_ids)
        reserved = set(existing)
        mapping: dict[str, int] = {}
        deferred: list[_PreparedRecord] = []
        for record in records:
            requested = record.requested_id
            if requested is not None and requested not in reserved:
                mapping[record.source_key] = requested
                reserved.add(requested)
            else:
                deferred.append(record)

        cursor = database.next_registration_id
        for record in deferred:
            while cursor in reserved and cursor <= MAX_CANONICAL_REGISTRATION_ID:
                cursor += 1
            if cursor > MAX_CANONICAL_REGISTRATION_ID:
                raise LegacyImportError(LegacyImportErrorCode.SEQUENCE_EXHAUSTED)
            mapping[record.source_key] = cursor
            reserved.add(cursor)
            cursor += 1
        if mapping and max(mapping.values()) >= MAX_CANONICAL_REGISTRATION_ID:
            raise LegacyImportError(LegacyImportErrorCode.SEQUENCE_EXHAUSTED)
        return mapping

    async def _build_from_snapshot(
        self,
        request: LegacyImportRequest,
        snapshot: _ImportSnapshot,
        *,
        state: MigrationState | None = None,
    ) -> LegacyImportPlan:
        if state is None:
            state = await self._repository.get_migration_state()
        raw_records = self._raw_records(snapshot)
        duplicate_keys = {
            key for key, count in Counter(record.source_key for record in raw_records).items() if count > 1
        }
        rejections = self._current_rejections(state)
        prepared: list[_PreparedRecord] = []
        unresolved: list[LegacyUnresolvedRecord] = []
        rejected: list[LegacyRejectedRecord] = []
        for raw in raw_records:
            decision = rejections.get(
                (
                    raw.source_kind,
                    raw.source_identity,
                    raw.source_record_fingerprint,
                )
            )
            if decision is not None:
                rejected.append(decision)
                continue
            if raw.source_key in duplicate_keys:
                unresolved.append(
                    LegacyUnresolvedRecord(
                        source_kind=raw.source_kind,
                        source_identity=raw.source_identity,
                        source_record_fingerprint=raw.source_record_fingerprint,
                        reason_code="duplicate_source_identity",
                    )
                )
                continue
            try:
                prepared.append(
                    self._prepare_record(
                        raw,
                        allow_legacy_credential_decryption=(request.allow_legacy_credential_decryption),
                    )
                )
            except _RecordIssue as exc:
                unresolved.append(
                    LegacyUnresolvedRecord(
                        source_kind=raw.source_kind,
                        source_identity=raw.source_identity,
                        source_record_fingerprint=raw.source_record_fingerprint,
                        reason_code=exc.reason_code,
                    )
                )

        ordered_prepared = tuple(
            sorted(
                prepared,
                key=lambda record: (
                    record.source_kind,
                    record.source_identity,
                    record.source_record_fingerprint,
                ),
            )
        )
        mapping = self._allocate_mapping(ordered_prepared, snapshot.database)
        accepted = tuple(
            LegacyAcceptedRecord(
                source_kind=record.source_kind,
                source_identity=record.source_identity,
                source_record_fingerprint=record.source_record_fingerprint,
                canonical_id=mapping[record.source_key],
                target_display=record.target_display,
                target_hostname=record.target_hostname,
                event_types=record.event_types,
                timeout_seconds=record.timeout_seconds,
            )
            for record in ordered_prepared
        )
        operation_id = self._key_ring.derive_migration_operation_id(snapshot.source_fingerprints)
        provisional = LegacyImportPlan(
            operation_id=operation_id,
            fingerprint_key_id=snapshot.fingerprint_key_id,
            legacy_credential_decryption_enabled=(request.allow_legacy_credential_decryption),
            source_fingerprints=snapshot.source_fingerprints,
            accepted=accepted,
            unresolved=tuple(
                sorted(
                    unresolved,
                    key=lambda record: (
                        record.source_kind,
                        record.source_identity,
                        record.source_record_fingerprint,
                    ),
                )
            ),
            explicitly_rejected=tuple(
                sorted(
                    rejected,
                    key=lambda record: (
                        record.source_kind,
                        record.source_identity,
                        record.source_record_fingerprint,
                    ),
                )
            ),
            projected_non_deleted_count=(snapshot.database.canonical_non_deleted_count + len(accepted)),
            source_mapping=mapping,
            requires_system_ops_backup=snapshot.requires_system_ops_backup,
            report_digest="",
        )
        digest = "sha256:" + hashlib.sha256(canonical_report_payload(provisional)).hexdigest()
        return replace(provisional, report_digest=digest)

    async def build_plan(self, request: LegacyImportRequest) -> LegacyImportPlan:
        """Publish a deterministic mode-0600 redacted dry-run report."""
        if not isinstance(request, LegacyImportRequest):
            raise TypeError("LegacyImportRequest is required")
        snapshot = await self._snapshot()
        paths = self._normalize_paths(
            request,
            requires_backup=snapshot.requires_system_ops_backup,
        )
        plan = await self._build_from_snapshot(request, snapshot)
        envelope = _report_payload_object(plan)
        envelope["generated_at"] = request.now.astimezone(timezone.utc).isoformat()
        envelope["report_digest"] = plan.report_digest
        _publish_private_report(paths.report, _canonical_json_bytes(envelope))
        return plan

    def _validate_report(
        self,
        path: Path,
        *,
        approved_report_digest: str,
        fresh_plan: LegacyImportPlan,
    ) -> _FileEvidence:
        if _REPORT_DIGEST_PATTERN.fullmatch(approved_report_digest) is None:
            raise LegacyImportError(LegacyImportErrorCode.APPROVAL_MISMATCH)
        raw = _read_private_file(path, maximum_bytes=2_097_152)

        def object_hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
            value: dict[str, object] = {}
            for key, item in pairs:
                if key in value:
                    raise ValueError("duplicate report key")
                value[key] = item
            return value

        try:
            envelope = json.loads(raw, object_pairs_hook=object_hook)
        except (json.JSONDecodeError, RecursionError, TypeError, ValueError):
            raise LegacyImportError(LegacyImportErrorCode.REPORT_INVALID) from None
        if not isinstance(envelope, dict):
            raise LegacyImportError(LegacyImportErrorCode.REPORT_INVALID)
        allowed = set(_report_payload_object(fresh_plan)) | {
            "generated_at",
            "report_digest",
        }
        if set(envelope) != allowed:
            raise LegacyImportError(LegacyImportErrorCode.REPORT_INVALID)
        report_digest = envelope.get("report_digest")
        if not isinstance(report_digest, str) or _REPORT_DIGEST_PATTERN.fullmatch(report_digest) is None:
            raise LegacyImportError(LegacyImportErrorCode.REPORT_INVALID)
        payload = {key: value for key, value in envelope.items() if key not in {"generated_at", "report_digest"}}
        computed_digest = "sha256:" + hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
        if not hmac.compare_digest(computed_digest, report_digest):
            raise LegacyImportError(LegacyImportErrorCode.REPORT_INVALID)
        if not hmac.compare_digest(report_digest, approved_report_digest):
            raise LegacyImportError(LegacyImportErrorCode.APPROVAL_MISMATCH)
        if not hmac.compare_digest(report_digest, fresh_plan.report_digest):
            raise LegacyImportError(LegacyImportErrorCode.SOURCE_CHANGED)
        if payload != _report_payload_object(fresh_plan):
            raise LegacyImportError(LegacyImportErrorCode.SOURCE_CHANGED)
        return _file_evidence(path, expected_payload=raw)

    def _load_reviewed_plan(
        self,
        path: Path,
        *,
        approved_report_digest: str,
    ) -> tuple[LegacyImportPlan, _FileEvidence]:
        if _REPORT_DIGEST_PATTERN.fullmatch(approved_report_digest) is None:
            raise LegacyImportError(LegacyImportErrorCode.APPROVAL_MISMATCH)
        raw = _read_private_file(path, maximum_bytes=2_097_152)
        plan = _plan_from_report_bytes(raw)
        if not hmac.compare_digest(plan.report_digest, approved_report_digest):
            raise LegacyImportError(LegacyImportErrorCode.APPROVAL_MISMATCH)
        return plan, _file_evidence(path, expected_payload=raw)

    def _claimed_report_evidence(self, state: MigrationState) -> _FileEvidence:
        if (
            state.report_owner_id is None
            or state.report_group_id is None
            or state.report_mode is None
            or state.report_file_identity is None
        ):
            raise LegacyImportError(LegacyImportErrorCode.PRECONDITION_FAILED)
        return _FileEvidence(
            owner_id=state.report_owner_id,
            group_id=state.report_group_id,
            mode=state.report_mode,
            identity=state.report_file_identity,
        )

    def _validate_claimed_operation(
        self,
        *,
        state: MigrationState,
        request: LegacyImportRequest,
        plan: LegacyImportPlan,
        paths: _NormalizedPaths,
    ) -> None:
        if (
            state.import_operation_id != plan.operation_id
            or state.import_operator_id != request.operator_id
            or state.redacted_report_digest != plan.report_digest
            or state.fingerprint_key_id != plan.fingerprint_key_id
            or state.active_primary_key_id != self._key_ring.primary_id
            or state.system_ops_webhook_fingerprint != plan.source_fingerprints["system_ops"]
            or state.legacy_table_fingerprint != plan.source_fingerprints["database"]
            or dict(state.source_mapping) != dict(plan.source_mapping)
            or state.active_report_path != str(paths.report)
            or (
                plan.requires_system_ops_backup
                and (state.active_backup_path != str(paths.backup) or state.active_key_path != str(paths.rollback_key))
            )
            or plan.legacy_credential_decryption_enabled != request.allow_legacy_credential_decryption
        ):
            raise LegacyImportError(LegacyImportErrorCode.PRECONDITION_FAILED)

    @staticmethod
    def _state_rejections(plan: LegacyImportPlan) -> list[dict[str, object]]:
        return [
            {
                "source_kind": record.source_kind,
                "source_identity": record.source_identity,
                "source_record_fingerprint": record.source_record_fingerprint,
                "reason_code": record.reason_code,
                "operator_id": record.operator_id,
                "fingerprint_key_id": plan.fingerprint_key_id,
            }
            for record in plan.explicitly_rejected
        ]

    def _prepared_for_plan(
        self,
        request: LegacyImportRequest,
        snapshot: _ImportSnapshot,
        plan: LegacyImportPlan,
    ) -> tuple[_PreparedRecord, ...]:
        accepted_keys = {
            (
                record.source_kind,
                record.source_identity,
                record.source_record_fingerprint,
            )
            for record in plan.accepted
        }
        prepared = tuple(
            self._prepare_record(
                raw,
                allow_legacy_credential_decryption=(request.allow_legacy_credential_decryption),
            )
            for raw in self._raw_records(snapshot)
            if (
                raw.source_kind,
                raw.source_identity,
                raw.source_record_fingerprint,
            )
            in accepted_keys
        )
        if len(prepared) != len(plan.accepted):
            raise LegacyImportError(LegacyImportErrorCode.SOURCE_CHANGED)
        return prepared

    @staticmethod
    def _report_staging_path(path: Path, operation_id: str) -> Path:
        return path.with_name(f".{path.name}.{operation_id}.staging")

    async def _reserve_artifacts(
        self,
        *,
        request: LegacyImportRequest,
        plan: LegacyImportPlan,
        paths: _NormalizedPaths,
        report_evidence: _FileEvidence,
    ) -> MigrationState:
        if paths.backup is None or paths.rollback_key is None:
            raise LegacyImportError(LegacyImportErrorCode.PATH_UNSAFE)
        backup_staging = _staging_path(paths.backup, plan.operation_id)
        key_staging = _staging_path(paths.rollback_key, plan.operation_id)
        if _path_exists(backup_staging) or _path_exists(key_staging):
            raise LegacyImportError(LegacyImportErrorCode.PATH_UNSAFE)
        async with self._repository.transaction() as tx:
            state = await tx.lock_migration_state()
            if state.phase != "migration_pending" or state.rotation_phase not in {
                None,
                "complete",
            }:
                raise LegacyImportError(LegacyImportErrorCode.PRECONDITION_FAILED)
            return await tx.compare_and_set_migration_state(
                expected_revision=state.state_revision,
                updates={
                    "phase": "artifacts_pending",
                    "import_operation_id": plan.operation_id,
                    "import_operator_id": request.operator_id,
                    "import_started_at": request.now,
                    "import_approved_at": request.now,
                    "fingerprint_key_id": plan.fingerprint_key_id,
                    "active_primary_key_id": self._key_ring.primary_id,
                    "system_ops_webhook_fingerprint": plan.source_fingerprints["system_ops"],
                    "legacy_table_fingerprint": plan.source_fingerprints["database"],
                    "source_mapping_json": dict(plan.source_mapping),
                    "source_rejections_json": self._state_rejections(plan),
                    "redacted_report_digest": plan.report_digest,
                    "active_report_path": str(paths.report),
                    "active_backup_path": str(paths.backup),
                    "active_key_path": str(paths.rollback_key),
                    "staging_report_path": str(self._report_staging_path(paths.report, plan.operation_id)),
                    "staging_backup_path": str(backup_staging),
                    "staging_key_path": str(key_staging),
                    "report_owner_id": report_evidence.owner_id,
                    "report_group_id": report_evidence.group_id,
                    "report_mode": report_evidence.mode,
                    "report_file_identity": report_evidence.identity,
                    "backup_owner_id": os.geteuid(),
                    "backup_group_id": os.getegid(),
                    "backup_mode": 0o600,
                    "backup_file_identity": f"claim:{plan.operation_id}:backup",
                    "rollback_key_owner_id": os.geteuid(),
                    "rollback_key_group_id": os.getegid(),
                    "rollback_key_mode": 0o600,
                    "rollback_key_file_identity": f"claim:{plan.operation_id}:key",
                },
                at=request.now,
            )

    async def _publish_rollback_artifacts(
        self,
        *,
        request: LegacyImportRequest,
        plan: LegacyImportPlan,
        snapshot: _ImportSnapshot,
        paths: _NormalizedPaths,
        reserved: MigrationState,
    ) -> MigrationState:
        if (
            paths.backup is None
            or paths.rollback_key is None
            or reserved.staging_backup_path is None
            or reserved.staging_key_path is None
        ):
            raise LegacyImportError(LegacyImportErrorCode.PRECONDITION_FAILED)
        key_staging = Path(reserved.staging_key_path)
        backup_staging = Path(reserved.staging_backup_path)
        with system_ops._STORE_LOCK, system_ops._store_file_lock(store_path=self._system_ops_path):
            current_store = system_ops._load_store_strict(self._system_ops_path)
            current_fingerprint = self._system_subtree_fingerprint(current_store)
            if not hmac.compare_digest(
                current_fingerprint,
                plan.source_fingerprints["system_ops"],
            ):
                raise LegacyImportError(LegacyImportErrorCode.SOURCE_CHANGED)

            key_candidate = (
                paths.rollback_key
                if _path_exists(paths.rollback_key)
                else key_staging
                if _path_exists(key_staging)
                else None
            )
            if key_candidate is not None:
                try:
                    key_payload = _read_private_file(
                        key_candidate,
                        maximum_bytes=65_536,
                    )
                    rollback_key = self._rollback_key_from_payload(
                        key_payload,
                        plan=plan,
                    )
                except LegacyImportError:
                    if key_candidate == paths.rollback_key:
                        raise
                    key_candidate.unlink()
                    _fsync_directory(key_candidate.parent)
                    key_candidate = None
            if key_candidate is None:
                rollback_key = secrets.token_bytes(32)
                key_payload = _canonical_json_bytes(
                    {
                        "schema_version": 1,
                        "operation_id": plan.operation_id,
                        "source_fingerprint": plan.source_fingerprints["system_ops"],
                        "report_digest": plan.report_digest,
                        "key_b64": base64.b64encode(rollback_key).decode("ascii"),
                    }
                )
            key_evidence = _publish_or_resume_artifact(
                paths.rollback_key,
                key_staging,
                key_payload,
            )
            self._checkpoint("after_key_publish")

            rollback_ring = WebhookKeyRing(
                {"rollback": base64.b64encode(rollback_key).decode("ascii")},
                primary_id="rollback",
            )
            backup_candidate = (
                paths.backup if _path_exists(paths.backup) else backup_staging if _path_exists(backup_staging) else None
            )
            if backup_candidate is not None:
                try:
                    backup_payload = _read_private_file(backup_candidate)
                    self._verify_backup_payload(
                        payload=backup_payload,
                        ring=rollback_ring,
                        plan=plan,
                    )
                except LegacyImportError:
                    if backup_candidate == paths.backup:
                        raise
                    backup_candidate.unlink()
                    _fsync_directory(backup_candidate.parent)
                    backup_candidate = None
            if backup_candidate is None:
                protected_backup = rollback_ring.encrypt_bytes(
                    purpose="legacy.system_ops.backup",
                    identity={
                        "operation_id": plan.operation_id,
                        "source_fingerprint": plan.source_fingerprints["system_ops"],
                    },
                    plaintext=snapshot.store_bytes,
                )
                backup_payload = _canonical_json_bytes(
                    {
                        "schema_version": 1,
                        "key_id": protected_backup.key_id,
                        "ciphertext_json": protected_backup.ciphertext_json,
                    }
                )
            backup_evidence = _publish_or_resume_artifact(
                paths.backup,
                backup_staging,
                backup_payload,
            )
            self._checkpoint("after_backup_publish")

        ciphertext_digest = "sha256:" + hashlib.sha256(backup_payload).hexdigest()
        async with self._repository.transaction() as tx:
            state = await tx.lock_migration_state()
            if state.phase != "artifacts_pending" or state.import_operation_id != plan.operation_id:
                raise LegacyImportError(LegacyImportErrorCode.PRECONDITION_FAILED)
            return await tx.compare_and_set_migration_state(
                expected_revision=state.state_revision,
                updates={
                    "phase": "artifacts_ready",
                    "artifacts_ready_at": request.now,
                    "protected_backup_ciphertext_digest": ciphertext_digest,
                    "expected_ciphertext_digest": ciphertext_digest,
                    "backup_owner_id": backup_evidence.owner_id,
                    "backup_group_id": backup_evidence.group_id,
                    "backup_mode": backup_evidence.mode,
                    "backup_file_identity": backup_evidence.identity,
                    "rollback_key_owner_id": key_evidence.owner_id,
                    "rollback_key_group_id": key_evidence.group_id,
                    "rollback_key_mode": key_evidence.mode,
                    "rollback_key_file_identity": key_evidence.identity,
                },
                at=request.now,
            )

    def _rollback_key_from_payload(
        self,
        payload: bytes,
        *,
        plan: LegacyImportPlan,
    ) -> bytes:
        value = _strict_json_object(payload, maximum_bytes=65_536)
        if (
            set(value)
            != {
                "schema_version",
                "operation_id",
                "source_fingerprint",
                "report_digest",
                "key_b64",
            }
            or value.get("schema_version") != 1
            or value.get("operation_id") != plan.operation_id
            or value.get("source_fingerprint") != plan.source_fingerprints["system_ops"]
            or value.get("report_digest") != plan.report_digest
            or not isinstance(value.get("key_b64"), str)
        ):
            raise LegacyImportError(LegacyImportErrorCode.OPERATION_FAILED)
        encoded = value.get("key_b64")
        if not isinstance(encoded, str):
            raise LegacyImportError(LegacyImportErrorCode.OPERATION_FAILED)
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error):
            raise LegacyImportError(LegacyImportErrorCode.OPERATION_FAILED) from None
        if len(raw) != 32 or base64.b64encode(raw).decode("ascii") != encoded:
            raise LegacyImportError(LegacyImportErrorCode.OPERATION_FAILED)
        return raw

    def _verify_backup_payload(
        self,
        *,
        payload: bytes,
        ring: WebhookKeyRing,
        plan: LegacyImportPlan,
    ) -> None:
        value = _strict_json_object(payload, maximum_bytes=70_000_000)
        if (
            set(value) != {"schema_version", "key_id", "ciphertext_json"}
            or value.get("schema_version") != 1
            or value.get("key_id") != "rollback"
            or not isinstance(value.get("ciphertext_json"), str)
        ):
            raise LegacyImportError(LegacyImportErrorCode.OPERATION_FAILED)
        ciphertext_json = value.get("ciphertext_json")
        if not isinstance(ciphertext_json, str):
            raise LegacyImportError(LegacyImportErrorCode.OPERATION_FAILED)
        try:
            plaintext = ring.decrypt_bytes(
                purpose="legacy.system_ops.backup",
                identity={
                    "operation_id": plan.operation_id,
                    "source_fingerprint": plan.source_fingerprints["system_ops"],
                },
                protected=ProtectedValue(
                    ciphertext_json=ciphertext_json,
                    key_id="rollback",
                ),
            )
        except Exception:  # noqa: BLE001 - expose only a closed artifact failure
            raise LegacyImportError(LegacyImportErrorCode.OPERATION_FAILED) from None
        plaintext_store = _strict_json_object(
            plaintext,
            maximum_bytes=67_108_864,
        )
        if not hmac.compare_digest(
            self._system_subtree_fingerprint(plaintext_store),
            plan.source_fingerprints["system_ops"],
        ):
            raise LegacyImportError(LegacyImportErrorCode.OPERATION_FAILED)

    def _system_subtree_fingerprint(self, store: Mapping[str, object]) -> str:
        _, fingerprint = self._fingerprint(
            MIGRATION_DOMAIN_SYSTEM_OPS_SUBTREE,
            {
                "schema": 1,
                "webhooks_present": "webhooks" in store,
                "webhooks": store.get("webhooks"),
                "webhook_deliveries_present": "webhook_deliveries" in store,
                "webhook_deliveries": store.get("webhook_deliveries"),
            },
        )
        return fingerprint

    async def _commit_database(
        self,
        *,
        request: LegacyImportRequest,
        plan: LegacyImportPlan,
        snapshot: _ImportSnapshot,
        paths: _NormalizedPaths,
        report_evidence: _FileEvidence,
        artifact_state: MigrationState | None,
    ) -> tuple[MigrationState, tuple[_PreparedRecord, ...]]:
        async with self._repository.transaction() as tx:
            state = await tx.lock_migration_state()
            expected_phase = "artifacts_ready" if artifact_state is not None else "migration_pending"
            if state.phase != expected_phase:
                raise LegacyImportError(LegacyImportErrorCode.PRECONDITION_FAILED)
            if artifact_state is not None and (
                state.import_operation_id != plan.operation_id
                or state.redacted_report_digest != plan.report_digest
                or state.active_primary_key_id != self._key_ring.primary_id
            ):
                raise LegacyImportError(LegacyImportErrorCode.PRECONDITION_FAILED)

            locked_database = await tx.get_legacy_import_snapshot(lock=True)
            locked_snapshot = self._compose_snapshot(
                store=snapshot.store,
                store_bytes=snapshot.store_bytes,
                database=locked_database,
            )
            locked_plan = await self._build_from_snapshot(
                request,
                locked_snapshot,
                state=state,
            )
            if (
                locked_plan.report_digest != plan.report_digest
                or dict(locked_plan.source_mapping) != dict(plan.source_mapping)
                or locked_plan.source_fingerprints != plan.source_fingerprints
            ):
                raise LegacyImportError(LegacyImportErrorCode.SOURCE_CHANGED)
            prepared = self._prepared_for_plan(request, locked_snapshot, locked_plan)
            prepared_by_key = {
                (
                    record.source_kind,
                    record.source_identity,
                    record.source_record_fingerprint,
                ): record
                for record in prepared
            }
            for accepted in locked_plan.accepted:
                prepared_record = prepared_by_key[
                    (
                        accepted.source_kind,
                        accepted.source_identity,
                        accepted.source_record_fingerprint,
                    )
                ]
                target = self._key_ring.encrypt_text(
                    purpose="registration.target",
                    identity={
                        "registration_id": accepted.canonical_id,
                        "target_version": INITIAL_PROTECTED_VERSION,
                    },
                    plaintext=prepared_record.url,
                )
                secret = self._key_ring.encrypt_bytes(
                    purpose="registration.secret",
                    identity={
                        "registration_id": accepted.canonical_id,
                        "secret_version": INITIAL_PROTECTED_VERSION,
                    },
                    plaintext=prepared_record.secret_bytes,
                )
                await tx.insert_registration(
                    RegistrationInsert(
                        id=accepted.canonical_id,
                        description=prepared_record.description,
                        target=RegistrationTarget(
                            protected=target,
                            hostname=prepared_record.target_hostname,
                            display=prepared_record.target_display,
                        ),
                        event_types=prepared_record.event_types,
                        active=False,
                        timeout_seconds=prepared_record.timeout_seconds,
                        secret=secret,
                        secret_rotation_required=True,
                        actor_user_id=request.operator_id,
                        now=request.now,
                    )
                )
            if locked_plan.source_mapping:
                maximum_id = max(locked_plan.source_mapping.values())
                await tx.ensure_registration_sequence_above(maximum_id)

            updates: dict[str, object] = {
                "phase": "database_committed",
                "database_committed_at": request.now,
                "active_primary_key_id": self._key_ring.primary_id,
                "source_mapping_json": dict(locked_plan.source_mapping),
                "source_rejections_json": self._state_rejections(locked_plan),
                "rollback_retirement_phase": ("retained" if artifact_state is not None else "not_applicable"),
            }
            if artifact_state is None:
                updates.update(
                    {
                        "import_operation_id": locked_plan.operation_id,
                        "import_operator_id": request.operator_id,
                        "import_started_at": request.now,
                        "import_approved_at": request.now,
                        "fingerprint_key_id": locked_plan.fingerprint_key_id,
                        "system_ops_webhook_fingerprint": (locked_plan.source_fingerprints["system_ops"]),
                        "legacy_table_fingerprint": locked_plan.source_fingerprints["database"],
                        "redacted_report_digest": locked_plan.report_digest,
                        "active_report_path": str(paths.report),
                        "staging_report_path": str(
                            self._report_staging_path(
                                paths.report,
                                locked_plan.operation_id,
                            )
                        ),
                        "report_owner_id": report_evidence.owner_id,
                        "report_group_id": report_evidence.group_id,
                        "report_mode": report_evidence.mode,
                        "report_file_identity": report_evidence.identity,
                    }
                )
            committed = await tx.compare_and_set_migration_state(
                expected_revision=state.state_revision,
                updates=updates,
                at=request.now,
            )
        return committed, prepared

    async def _verify_canonical_records(
        self,
        plan: LegacyImportPlan,
        prepared: tuple[_PreparedRecord, ...],
    ) -> None:
        prepared_by_key = {
            (
                record.source_kind,
                record.source_identity,
                record.source_record_fingerprint,
            ): record
            for record in prepared
        }
        for accepted in plan.accepted:
            source = prepared_by_key[
                (
                    accepted.source_kind,
                    accepted.source_identity,
                    accepted.source_record_fingerprint,
                )
            ]
            stored = await self._repository.get_protected_registration(accepted.canonical_id)
            if stored is None or (
                stored.registration.active
                or not stored.registration.secret_rotation_required
                or stored.registration.event_types != source.event_types
                or stored.registration.timeout_seconds != source.timeout_seconds
                or stored.registration.target_hostname != source.target_hostname
                or stored.registration.target_display != source.target_display
            ):
                raise LegacyImportError(LegacyImportErrorCode.OPERATION_FAILED)
            target = self._key_ring.decrypt_text(
                purpose="registration.target",
                identity={
                    "registration_id": accepted.canonical_id,
                    "target_version": INITIAL_PROTECTED_VERSION,
                },
                protected=stored.target,
            )
            secret = self._key_ring.decrypt_bytes(
                purpose="registration.secret",
                identity={
                    "registration_id": accepted.canonical_id,
                    "secret_version": INITIAL_PROTECTED_VERSION,
                },
                protected=stored.secret,
            )
            if target != source.url or not hmac.compare_digest(secret, source.secret_bytes):
                raise LegacyImportError(LegacyImportErrorCode.OPERATION_FAILED)

    async def _verify_sanitize_and_complete(
        self,
        *,
        request: LegacyImportRequest,
        plan: LegacyImportPlan,
        prepared: tuple[_PreparedRecord, ...],
    ) -> MigrationState:
        await self._verify_canonical_records(plan, prepared)
        self._checkpoint("after_canonical_readback")
        with system_ops._STORE_LOCK, system_ops._store_file_lock(store_path=self._system_ops_path):
            try:
                current_store = system_ops._load_store_strict(self._system_ops_path)
            except (OSError, RuntimeError, ValueError):
                raise LegacyImportError(LegacyImportErrorCode.SOURCE_INVALID) from None
            if plan.requires_system_ops_backup:
                has_webhooks = "webhooks" in current_store
                has_deliveries = "webhook_deliveries" in current_store
                if has_webhooks or has_deliveries:
                    current_fingerprint = self._system_subtree_fingerprint(current_store)
                    if not hmac.compare_digest(
                        current_fingerprint,
                        plan.source_fingerprints["system_ops"],
                    ):
                        raise LegacyImportError(LegacyImportErrorCode.SOURCE_CHANGED)
                    sanitized = dict(current_store)
                    sanitized.pop("webhooks", None)
                    sanitized.pop("webhook_deliveries", None)
                    system_ops._atomic_write_store(self._system_ops_path, sanitized)
                    self._checkpoint("after_source_replace")
                    reread = system_ops._load_store_strict(self._system_ops_path)
                    if "webhooks" in reread or "webhook_deliveries" in reread:
                        raise LegacyImportError(LegacyImportErrorCode.OPERATION_FAILED)
            else:
                current_fingerprint = self._system_subtree_fingerprint(current_store)
                if not hmac.compare_digest(
                    current_fingerprint,
                    plan.source_fingerprints["system_ops"],
                ):
                    raise LegacyImportError(LegacyImportErrorCode.SOURCE_CHANGED)

        async with self._repository.transaction() as tx:
            state = await tx.lock_migration_state()
            if (
                state.phase != "database_committed"
                or state.import_operation_id != plan.operation_id
                or state.redacted_report_digest != plan.report_digest
            ):
                raise LegacyImportError(LegacyImportErrorCode.PRECONDITION_FAILED)
            updates: dict[str, object] = {
                "phase": "complete",
                "completed_at": request.now,
            }
            if plan.requires_system_ops_backup:
                updates["rollback_expires_at"] = request.now + timedelta(days=self._settings.rollback_window_days)
            completed = await tx.compare_and_set_migration_state(
                expected_revision=state.state_revision,
                updates=updates,
                at=request.now,
            )
        self._checkpoint("after_complete")
        return completed

    async def _emit_audit(
        self,
        *,
        request: LegacyImportRequest,
        plan: LegacyImportPlan,
        request_id: str,
        outcome: OperationalOutcome,
        reason_code: WebhookOperationalReasonCode | None = None,
    ) -> None:
        await self._audit_sink(
            OperationalAudit(
                operator_id=request.operator_id,
                action="admin_webhook.import.apply",
                operation_id=plan.operation_id,
                outcome=outcome,
                request_id=request_id,
                reason_code=reason_code,
            )
        )

    async def _emit_rollback_audit(
        self,
        *,
        operator_id: int,
        action: OperationalAction,
        operation_id: str,
        outcome: OperationalOutcome,
        request_id: str,
        reason_code: WebhookOperationalReasonCode | None = None,
    ) -> None:
        await self._audit_sink(
            OperationalAudit(
                operator_id=operator_id,
                action=action,
                operation_id=operation_id,
                outcome=outcome,
                request_id=request_id,
                reason_code=reason_code,
            )
        )

    def _read_rollback_backup(
        self,
        *,
        state: MigrationState,
        backup_path: Path,
        rollback_key_path: Path,
    ) -> bytes:
        key_payload = _strict_json_object(
            _read_private_file(rollback_key_path, maximum_bytes=65_536),
            maximum_bytes=65_536,
        )
        if set(key_payload) != {
            "schema_version",
            "operation_id",
            "source_fingerprint",
            "report_digest",
            "key_b64",
        }:
            raise LegacyImportError(LegacyImportErrorCode.OPERATION_FAILED)
        if (
            key_payload.get("schema_version") != 1
            or key_payload.get("operation_id") != state.import_operation_id
            or key_payload.get("source_fingerprint") != state.system_ops_webhook_fingerprint
            or key_payload.get("report_digest") != state.redacted_report_digest
            or not isinstance(key_payload.get("key_b64"), str)
        ):
            raise LegacyImportError(LegacyImportErrorCode.OPERATION_FAILED)
        encoded_key = key_payload.get("key_b64")
        if not isinstance(encoded_key, str):
            raise LegacyImportError(LegacyImportErrorCode.OPERATION_FAILED)
        try:
            raw_key = base64.b64decode(encoded_key, validate=True)
        except (ValueError, binascii.Error):
            raise LegacyImportError(LegacyImportErrorCode.OPERATION_FAILED) from None
        if len(raw_key) != 32 or base64.b64encode(raw_key).decode("ascii") != encoded_key:
            raise LegacyImportError(LegacyImportErrorCode.OPERATION_FAILED)

        backup_payload = _read_private_file(backup_path)
        backup_digest = "sha256:" + hashlib.sha256(backup_payload).hexdigest()
        if state.protected_backup_ciphertext_digest is None or not hmac.compare_digest(
            backup_digest,
            state.protected_backup_ciphertext_digest,
        ):
            raise LegacyImportError(LegacyImportErrorCode.OPERATION_FAILED)
        backup_envelope = _strict_json_object(
            backup_payload,
            maximum_bytes=70_000_000,
        )
        if (
            set(backup_envelope) != {"schema_version", "key_id", "ciphertext_json"}
            or backup_envelope.get("schema_version") != 1
            or backup_envelope.get("key_id") != "rollback"
            or not isinstance(backup_envelope.get("ciphertext_json"), str)
            or state.import_operation_id is None
            or state.system_ops_webhook_fingerprint is None
        ):
            raise LegacyImportError(LegacyImportErrorCode.OPERATION_FAILED)
        ciphertext_json = backup_envelope.get("ciphertext_json")
        if not isinstance(ciphertext_json, str):
            raise LegacyImportError(LegacyImportErrorCode.OPERATION_FAILED)
        ring = WebhookKeyRing(
            {"rollback": encoded_key},
            primary_id="rollback",
        )
        try:
            plaintext = ring.decrypt_bytes(
                purpose="legacy.system_ops.backup",
                identity={
                    "operation_id": state.import_operation_id,
                    "source_fingerprint": state.system_ops_webhook_fingerprint,
                },
                protected=ProtectedValue(
                    ciphertext_json=ciphertext_json,
                    key_id="rollback",
                ),
            )
        except Exception:  # noqa: BLE001 - expose only the closed rollback failure
            raise LegacyImportError(LegacyImportErrorCode.OPERATION_FAILED) from None
        _strict_json_object(plaintext, maximum_bytes=67_108_864)
        return plaintext

    async def extract_rollback_backup(
        self,
        *,
        backup_path: Path,
        rollback_key_path: Path,
        output_path: Path,
        operator_id: int,
        now: datetime,
        confirmed: bool,
        request_id: str,
    ) -> str:
        """Decrypt a retained backup to one new private operator-selected file."""
        if (
            not isinstance(backup_path, Path)
            or not isinstance(rollback_key_path, Path)
            or not isinstance(output_path, Path)
            or isinstance(operator_id, bool)
            or not isinstance(operator_id, int)
            or operator_id < 1
            or not isinstance(now, datetime)
            or now.tzinfo is None
        ):
            raise LegacyImportError(LegacyImportErrorCode.PRECONDITION_FAILED)
        state = await self._repository.get_migration_state()
        self._require_extractable_rollback_state(state=state, now=now)
        if not confirmed or state.import_operation_id is None:
            raise LegacyImportError(LegacyImportErrorCode.PRECONDITION_FAILED)
        operation_id = state.import_operation_id

        try:
            await self._emit_rollback_audit(
                operator_id=operator_id,
                action="admin_webhook.rollback.extract",
                operation_id=operation_id,
                outcome="accepted",
                request_id=request_id,
            )
        except Exception:  # noqa: BLE001 - mandatory audit fails closed
            raise LegacyImportError(LegacyImportErrorCode.AUDIT_UNAVAILABLE) from None
        normalized_output: Path | None = None
        published_output: _FileEvidence | None = None
        transaction_exited = False
        try:
            normalized_backup = _normalize_output_path(backup_path)
            normalized_key = _normalize_output_path(rollback_key_path)
            normalized_output = _normalize_output_path(output_path)
            if (
                str(normalized_backup) != state.active_backup_path
                or str(normalized_key) != state.active_key_path
                or len({normalized_backup, normalized_key, normalized_output}) != 3
                or normalized_output == self._system_ops_path.resolve(strict=False)
                or any(
                    _is_within(normalized_output, root)
                    for root in self._application_data_paths
                )
                or _path_exists(normalized_output)
            ):
                raise LegacyImportError(LegacyImportErrorCode.PATH_UNSAFE)
            async with self._repository.transaction() as tx:
                current = await tx.lock_migration_state()
                self._require_extractable_rollback_state(state=current, now=now)
                if (
                    current.import_operation_id != operation_id
                    or str(normalized_backup) != current.active_backup_path
                    or str(normalized_key) != current.active_key_path
                ):
                    raise LegacyImportError(LegacyImportErrorCode.PRECONDITION_FAILED)
                plaintext = self._read_rollback_backup(
                    state=current,
                    backup_path=normalized_backup,
                    rollback_key_path=normalized_key,
                )
                published_output = _publish_exclusive_output(
                    normalized_output,
                    plaintext,
                )
            transaction_exited = True
        except Exception as exc:  # noqa: BLE001 - sanitize operational boundary
            with suppress(Exception):
                await self._emit_rollback_audit(
                    operator_id=operator_id,
                    action="admin_webhook.rollback.extract",
                    operation_id=operation_id,
                    outcome="failed",
                    request_id=request_id,
                    reason_code=WebhookOperationalReasonCode.OPERATION_FAILED,
                )
            if isinstance(exc, LegacyImportError):
                raise
            raise LegacyImportError(LegacyImportErrorCode.OPERATION_FAILED) from None
        finally:
            if (
                published_output is not None
                and not transaction_exited
                and normalized_output is not None
            ):
                _remove_published_output_if_same(normalized_output, published_output)
        with suppress(Exception):
            await self._emit_rollback_audit(
                operator_id=operator_id,
                action="admin_webhook.rollback.extract",
                operation_id=operation_id,
                outcome="completed",
                request_id=request_id,
            )
        return "admin_webhook_rollback_backup_extracted"

    @staticmethod
    def _require_extractable_rollback_state(
        *,
        state: MigrationState,
        now: datetime,
    ) -> None:
        if (
            state.phase != "complete"
            or state.rollback_retirement_phase != "retained"
            or state.rollback_expires_at is None
            or now.astimezone(timezone.utc) >= state.rollback_expires_at
            or state.first_canonical_activity_at is not None
        ):
            raise LegacyImportError(LegacyImportErrorCode.ROLLBACK_WINDOW_CLOSED)

    async def _retire_artifact(
        self,
        *,
        state: MigrationState,
        path: Path,
        identity_column: str,
        expected_identity: str,
        kind: str,
        now: datetime,
        expected_digest: str | None = None,
    ) -> MigrationState:
        retiring_prefix = f"retiring:{kind}:"
        retired_prefix = f"retired:{kind}:"
        if expected_identity.startswith(retired_prefix):
            if _path_exists(path):
                raise LegacyImportError(LegacyImportErrorCode.PRECONDITION_FAILED)
            return state
        if expected_identity.startswith(retiring_prefix):
            original_identity = expected_identity[len(retiring_prefix) :]
        else:
            original_identity = expected_identity
            async with self._repository.transaction() as tx:
                current = await tx.lock_migration_state()
                if (
                    current.state_revision != state.state_revision
                    or getattr(current, identity_column) != expected_identity
                    or current.rollback_retirement_phase != "rollback_retirement_in_progress"
                ):
                    raise LegacyImportError(LegacyImportErrorCode.PRECONDITION_FAILED)
                state = await tx.compare_and_set_migration_state(
                    expected_revision=current.state_revision,
                    updates={identity_column: f"{retiring_prefix}{original_identity}"},
                    at=now,
                )

        if _path_exists(path):
            evidence = _file_evidence(path)
            if evidence.identity != original_identity:
                raise LegacyImportError(LegacyImportErrorCode.PRECONDITION_FAILED)
            if expected_digest is not None:
                actual_digest = "sha256:" + hashlib.sha256(_read_private_file(path)).hexdigest()
                if not hmac.compare_digest(actual_digest, expected_digest):
                    raise LegacyImportError(LegacyImportErrorCode.PRECONDITION_FAILED)
            path.unlink()
        _fsync_directory(path.parent)
        async with self._repository.transaction() as tx:
            current = await tx.lock_migration_state()
            marker = f"{retiring_prefix}{original_identity}"
            if current.state_revision != state.state_revision or getattr(current, identity_column) != marker:
                raise LegacyImportError(LegacyImportErrorCode.PRECONDITION_FAILED)
            return await tx.compare_and_set_migration_state(
                expected_revision=current.state_revision,
                updates={identity_column: f"{retired_prefix}{original_identity}"},
                at=now,
            )

    async def destroy_rollback_key(
        self,
        *,
        backup_path: Path,
        rollback_key_path: Path,
        operator_id: int,
        now: datetime,
        confirmed: bool,
        request_id: str,
    ) -> str:
        """Retire the one-time key and encrypted active backup after expiry."""
        state = await self._repository.get_migration_state()
        if state.rollback_retirement_phase == "not_applicable":
            return LegacyImportErrorCode.ROLLBACK_ARTIFACTS_NOT_APPLICABLE.value
        if state.rollback_retirement_phase == "retired":
            return LegacyImportErrorCode.ROLLBACK_ARTIFACTS_ALREADY_RETIRED.value
        if (
            state.phase != "complete"
            or state.rollback_expires_at is None
            or now.astimezone(timezone.utc) < state.rollback_expires_at
        ):
            raise LegacyImportError(LegacyImportErrorCode.ROLLBACK_WINDOW_CLOSED)
        if (
            not confirmed
            or isinstance(operator_id, bool)
            or not isinstance(operator_id, int)
            or operator_id < 1
            or state.import_operation_id is None
            or state.active_backup_path is None
            or state.active_key_path is None
            or state.protected_backup_ciphertext_digest is None
            or state.backup_file_identity is None
            or state.rollback_key_file_identity is None
        ):
            raise LegacyImportError(LegacyImportErrorCode.PRECONDITION_FAILED)
        operation_id = state.import_operation_id
        if operation_id is None:
            raise LegacyImportError(LegacyImportErrorCode.PRECONDITION_FAILED)
        try:
            await self._emit_rollback_audit(
                operator_id=operator_id,
                action="admin_webhook.rollback.destroy",
                operation_id=operation_id,
                outcome="accepted",
                request_id=request_id,
            )
        except Exception:  # noqa: BLE001 - mandatory audit fails closed
            raise LegacyImportError(LegacyImportErrorCode.AUDIT_UNAVAILABLE) from None
        try:
            normalized_backup = _normalize_output_path(backup_path)
            normalized_key = _normalize_output_path(rollback_key_path)
            if str(normalized_backup) != state.active_backup_path or str(normalized_key) != state.active_key_path:
                raise LegacyImportError(LegacyImportErrorCode.PATH_UNSAFE)
            if state.rollback_retirement_phase == "retained":
                self._read_rollback_backup(
                    state=state,
                    backup_path=normalized_backup,
                    rollback_key_path=normalized_key,
                )
                async with self._repository.transaction() as tx:
                    current = await tx.lock_migration_state()
                    if (
                        current.state_revision != state.state_revision
                        or current.rollback_retirement_phase != "retained"
                    ):
                        raise LegacyImportError(LegacyImportErrorCode.PRECONDITION_FAILED)
                    state = await tx.compare_and_set_migration_state(
                        expected_revision=current.state_revision,
                        updates={
                            "rollback_retirement_phase": ("rollback_retirement_in_progress"),
                            "rollback_retirement_operator_id": operator_id,
                            "rollback_retirement_started_at": now,
                        },
                        at=now,
                    )
            elif state.rollback_retirement_phase != "rollback_retirement_in_progress":
                raise LegacyImportError(LegacyImportErrorCode.PRECONDITION_FAILED)

            rollback_key_identity = state.rollback_key_file_identity
            if rollback_key_identity is None:
                raise LegacyImportError(LegacyImportErrorCode.PRECONDITION_FAILED)
            state = await self._retire_artifact(
                state=state,
                path=normalized_key,
                identity_column="rollback_key_file_identity",
                expected_identity=rollback_key_identity,
                kind="key",
                now=now,
            )
            backup_identity = state.backup_file_identity
            backup_digest = state.protected_backup_ciphertext_digest
            if backup_identity is None or backup_digest is None:
                raise LegacyImportError(LegacyImportErrorCode.PRECONDITION_FAILED)
            state = await self._retire_artifact(
                state=state,
                path=normalized_backup,
                identity_column="backup_file_identity",
                expected_identity=backup_identity,
                kind="backup",
                now=now,
                expected_digest=backup_digest,
            )
            async with self._repository.transaction() as tx:
                current = await tx.lock_migration_state()
                if (
                    current.state_revision != state.state_revision
                    or current.rollback_retirement_phase != "rollback_retirement_in_progress"
                ):
                    raise LegacyImportError(LegacyImportErrorCode.PRECONDITION_FAILED)
                await tx.compare_and_set_migration_state(
                    expected_revision=current.state_revision,
                    updates={
                        "rollback_retirement_phase": "retired",
                        "rollback_retirement_completed_at": now,
                    },
                    at=now,
                )
        except Exception as exc:  # noqa: BLE001 - sanitize operational boundary
            with suppress(Exception):
                await self._emit_rollback_audit(
                    operator_id=operator_id,
                    action="admin_webhook.rollback.destroy",
                    operation_id=operation_id,
                    outcome="failed",
                    request_id=request_id,
                    reason_code=WebhookOperationalReasonCode.OPERATION_FAILED,
                )
            if isinstance(exc, LegacyImportError):
                raise
            raise LegacyImportError(LegacyImportErrorCode.OPERATION_FAILED) from None
        with suppress(Exception):
            await self._emit_rollback_audit(
                operator_id=operator_id,
                action="admin_webhook.rollback.destroy",
                operation_id=operation_id,
                outcome="completed",
                request_id=request_id,
            )
        return "admin_webhook_rollback_artifacts_retired"

    async def reject_source(
        self,
        *,
        source_kind: str,
        source_identity: str,
        source_record_fingerprint: str,
        reason_code: LegacyRejectionReason,
        operator_id: int,
        now: datetime,
        request_id: str,
    ) -> MigrationState:
        """Persist one audited rejection bound to an exact current source record."""
        if (
            source_kind not in _SOURCE_KINDS
            or _SOURCE_ID_PATTERN.fullmatch(source_identity) is None
            or _FINGERPRINT_PATTERN.fullmatch(source_record_fingerprint) is None
            or not isinstance(reason_code, LegacyRejectionReason)
            or isinstance(operator_id, bool)
            or not isinstance(operator_id, int)
            or operator_id < 1
            or not isinstance(now, datetime)
            or now.tzinfo is None
        ):
            raise LegacyImportError(LegacyImportErrorCode.PRECONDITION_FAILED)
        snapshot = await self._snapshot()
        matching = [
            record
            for record in self._raw_records(snapshot)
            if record.source_kind == source_kind
            and record.source_identity == source_identity
            and hmac.compare_digest(
                record.source_record_fingerprint,
                source_record_fingerprint,
            )
        ]
        if len(matching) != 1:
            raise LegacyImportError(LegacyImportErrorCode.SOURCE_CHANGED)
        operation_id = self._key_ring.derive_migration_operation_id(snapshot.source_fingerprints)

        async def emit(
            outcome: OperationalOutcome,
            reason: WebhookOperationalReasonCode | None = None,
        ) -> None:
            await self._audit_sink(
                OperationalAudit(
                    operator_id=operator_id,
                    action="admin_webhook.import.reject_source",
                    operation_id=operation_id,
                    outcome=outcome,
                    request_id=request_id,
                    reason_code=reason,
                )
            )

        try:
            await emit("accepted")
        except Exception:  # noqa: BLE001 - mandatory audit fails closed
            raise LegacyImportError(LegacyImportErrorCode.AUDIT_UNAVAILABLE) from None
        try:
            async with self._repository.transaction() as tx:
                state = await tx.lock_migration_state()
                if state.phase != "migration_pending" or state.rotation_phase not in {
                    None,
                    "complete",
                }:
                    raise LegacyImportError(LegacyImportErrorCode.PRECONDITION_FAILED)
                decisions = [
                    dict(value)
                    for value in state.source_rejections
                    if isinstance(value, Mapping)
                    and not (
                        value.get("source_kind") == source_kind
                        and value.get("source_identity") == source_identity
                        and value.get("source_record_fingerprint") == source_record_fingerprint
                    )
                ]
                decisions.append(
                    {
                        "source_kind": source_kind,
                        "source_identity": source_identity,
                        "source_record_fingerprint": source_record_fingerprint,
                        "reason_code": reason_code.value,
                        "operator_id": operator_id,
                        "rejected_at": now.astimezone(timezone.utc).isoformat(),
                        "fingerprint_key_id": snapshot.fingerprint_key_id,
                    }
                )
                updated = await tx.compare_and_set_migration_state(
                    expected_revision=state.state_revision,
                    updates={"source_rejections_json": decisions},
                    at=now,
                )
        except Exception as exc:  # noqa: BLE001 - sanitize operational boundary
            with suppress(Exception):
                await emit("failed", WebhookOperationalReasonCode.OPERATION_FAILED)
            if isinstance(exc, LegacyImportError):
                raise
            raise LegacyImportError(LegacyImportErrorCode.OPERATION_FAILED) from None
        with suppress(Exception):
            await emit("completed")
        return updated

    def _recovery_plan_from_snapshot(
        self,
        *,
        request: LegacyImportRequest,
        state: MigrationState,
        snapshot: _ImportSnapshot,
    ) -> tuple[LegacyImportPlan, tuple[_PreparedRecord, ...]]:
        if (
            state.import_operation_id is None
            or state.fingerprint_key_id is None
            or state.system_ops_webhook_fingerprint is None
            or state.legacy_table_fingerprint is None
            or state.redacted_report_digest is None
            or snapshot.source_fingerprints
            != {
                "system_ops": state.system_ops_webhook_fingerprint,
                "database": state.legacy_table_fingerprint,
            }
        ):
            raise LegacyImportError(LegacyImportErrorCode.SOURCE_CHANGED)
        raw_by_key: dict[str, _RawRecord] = {}
        for raw in self._raw_records(snapshot):
            if raw.source_key in raw_by_key:
                raise LegacyImportError(LegacyImportErrorCode.SOURCE_CHANGED)
            raw_by_key[raw.source_key] = raw
        canonical_mapping: dict[str, int] = {}
        for source_key, canonical_id_value in state.source_mapping.items():
            if (
                not isinstance(source_key, str)
                or isinstance(canonical_id_value, bool)
                or not isinstance(canonical_id_value, int)
            ):
                raise LegacyImportError(LegacyImportErrorCode.PRECONDITION_FAILED)
            canonical_mapping[source_key] = canonical_id_value
        prepared: list[_PreparedRecord] = []
        accepted: list[LegacyAcceptedRecord] = []
        for source_key, canonical_id_value in sorted(canonical_mapping.items()):
            source_record = raw_by_key.get(source_key)
            if source_record is None:
                raise LegacyImportError(LegacyImportErrorCode.SOURCE_CHANGED)
            record = self._prepare_record(
                source_record,
                allow_legacy_credential_decryption=(request.allow_legacy_credential_decryption),
            )
            prepared.append(record)
            accepted.append(
                LegacyAcceptedRecord(
                    source_kind=record.source_kind,
                    source_identity=record.source_identity,
                    source_record_fingerprint=record.source_record_fingerprint,
                    canonical_id=canonical_id_value,
                    target_display=record.target_display,
                    target_hostname=record.target_hostname,
                    event_types=record.event_types,
                    timeout_seconds=record.timeout_seconds,
                )
            )
        rejected = tuple(
            sorted(
                self._current_rejections(state).values(),
                key=lambda record: (
                    record.source_kind,
                    record.source_identity,
                    record.source_record_fingerprint,
                ),
            )
        )
        accepted_source_keys = set(canonical_mapping)
        rejected_keys = {f"{record.source_kind}:{record.source_identity}" for record in rejected}
        if set(raw_by_key) - accepted_source_keys - rejected_keys:
            raise LegacyImportError(LegacyImportErrorCode.SOURCE_CHANGED)
        imported = len(accepted)
        projected_count = (
            snapshot.database.canonical_non_deleted_count
            if state.phase in {"database_committed", "complete"}
            else snapshot.database.canonical_non_deleted_count + imported
        )
        provisional = LegacyImportPlan(
            operation_id=state.import_operation_id,
            fingerprint_key_id=state.fingerprint_key_id,
            legacy_credential_decryption_enabled=(request.allow_legacy_credential_decryption),
            source_fingerprints=snapshot.source_fingerprints,
            accepted=tuple(accepted),
            unresolved=(),
            explicitly_rejected=rejected,
            projected_non_deleted_count=projected_count,
            source_mapping=canonical_mapping,
            requires_system_ops_backup=state.active_backup_path is not None,
            report_digest="",
        )
        digest = "sha256:" + hashlib.sha256(canonical_report_payload(provisional)).hexdigest()
        if not hmac.compare_digest(digest, state.redacted_report_digest):
            raise LegacyImportError(LegacyImportErrorCode.APPROVAL_MISMATCH)
        return replace(provisional, report_digest=digest), tuple(prepared)

    async def _snapshot_for_claimed_state(
        self,
        *,
        state: MigrationState,
        paths: _NormalizedPaths,
    ) -> _ImportSnapshot:
        current = await self._snapshot()
        if current.source_fingerprints["system_ops"] == state.system_ops_webhook_fingerprint:
            return current
        if (
            state.active_backup_path is None
            or paths.backup is None
            or paths.rollback_key is None
            or "webhooks" in current.store
            or "webhook_deliveries" in current.store
        ):
            raise LegacyImportError(LegacyImportErrorCode.SOURCE_CHANGED)
        original_bytes = self._read_rollback_backup(
            state=state,
            backup_path=paths.backup,
            rollback_key_path=paths.rollback_key,
        )
        original_store = _strict_json_object(
            original_bytes,
            maximum_bytes=67_108_864,
        )
        recovered = self._compose_snapshot(
            store=original_store,
            store_bytes=original_bytes,
            database=current.database,
        )
        if recovered.source_fingerprints != {
            "system_ops": state.system_ops_webhook_fingerprint,
            "database": state.legacy_table_fingerprint,
        }:
            raise LegacyImportError(LegacyImportErrorCode.SOURCE_CHANGED)
        return recovered

    async def apply_plan(
        self,
        request: LegacyImportRequest,
        *,
        approved_report_digest: str,
        request_id: str | None = None,
    ) -> MigrationState:
        """Apply one literal approved report through the durable import stages."""
        if not isinstance(request, LegacyImportRequest):
            raise TypeError("LegacyImportRequest is required")
        if self._settings.mode is not AdminWebhookMode.MIGRATE:
            raise LegacyImportError(LegacyImportErrorCode.PRECONDITION_FAILED)
        initial_state = await self._repository.get_migration_state()
        if initial_state.phase == "complete":
            if (
                initial_state.redacted_report_digest is None
                or not hmac.compare_digest(
                    initial_state.redacted_report_digest,
                    approved_report_digest,
                )
                or initial_state.import_operator_id != request.operator_id
                or initial_state.active_primary_key_id != self._key_ring.primary_id
            ):
                raise LegacyImportError(LegacyImportErrorCode.APPROVAL_MISMATCH)
            if _path_exists(request.report_path):
                reviewed, _ = self._load_reviewed_plan(
                    _normalize_output_path(request.report_path),
                    approved_report_digest=approved_report_digest,
                )
                if (
                    reviewed.operation_id != initial_state.import_operation_id
                    or reviewed.legacy_credential_decryption_enabled != request.allow_legacy_credential_decryption
                ):
                    raise LegacyImportError(LegacyImportErrorCode.APPROVAL_MISMATCH)
            return initial_state

        recovery_prepared: tuple[_PreparedRecord, ...] | None = None
        if initial_state.phase == "migration_pending":
            snapshot = await self._snapshot()
            paths = self._normalize_paths(
                request,
                requires_backup=snapshot.requires_system_ops_backup,
            )
            plan = await self._build_from_snapshot(request, snapshot)
            if plan.unresolved:
                raise LegacyImportError(LegacyImportErrorCode.UNRESOLVED)
            if (
                plan.projected_non_deleted_count > self._settings.registration_limit
                or plan.projected_non_deleted_count > MAX_LEGACY_SOURCE_ROWS
            ):
                raise LegacyImportError(LegacyImportErrorCode.REGISTRATION_LIMIT)
            if plan.source_mapping and max(plan.source_mapping.values()) >= (MAX_CANONICAL_REGISTRATION_ID):
                raise LegacyImportError(LegacyImportErrorCode.SEQUENCE_EXHAUSTED)
            report_evidence = self._validate_report(
                paths.report,
                approved_report_digest=approved_report_digest,
                fresh_plan=plan,
            )
        elif initial_state.phase in {
            "artifacts_pending",
            "artifacts_ready",
            "database_committed",
        }:
            requires_backup = initial_state.active_backup_path is not None
            paths = self._normalize_paths(
                request,
                requires_backup=requires_backup,
                resume_state=initial_state,
            )
            snapshot = await self._snapshot_for_claimed_state(
                state=initial_state,
                paths=paths,
            )
            recovered_plan, recovery_prepared = self._recovery_plan_from_snapshot(
                request=request,
                state=initial_state,
                snapshot=snapshot,
            )
            if _path_exists(paths.report):
                reviewed_plan, report_evidence = self._load_reviewed_plan(
                    paths.report,
                    approved_report_digest=approved_report_digest,
                )
                if reviewed_plan.report_digest != recovered_plan.report_digest or _report_payload_object(
                    reviewed_plan
                ) != _report_payload_object(recovered_plan):
                    raise LegacyImportError(LegacyImportErrorCode.SOURCE_CHANGED)
                plan = reviewed_plan
                if report_evidence.identity != initial_state.report_file_identity:
                    raise LegacyImportError(LegacyImportErrorCode.PRECONDITION_FAILED)
            else:
                plan = recovered_plan
                report_evidence = self._claimed_report_evidence(initial_state)
                if not hmac.compare_digest(
                    plan.report_digest,
                    approved_report_digest,
                ):
                    raise LegacyImportError(LegacyImportErrorCode.APPROVAL_MISMATCH)
            self._validate_claimed_operation(
                state=initial_state,
                request=request,
                plan=plan,
                paths=paths,
            )
        else:
            raise LegacyImportError(LegacyImportErrorCode.PRECONDITION_FAILED)

        active_request_id = request_id or f"whimp_{secrets.token_hex(12)}"
        try:
            await self._emit_audit(
                request=request,
                plan=plan,
                request_id=active_request_id,
                outcome="accepted",
            )
        except Exception:  # noqa: BLE001 - mandatory audit fails closed
            raise LegacyImportError(LegacyImportErrorCode.AUDIT_UNAVAILABLE) from None
        self._checkpoint("after_audit")

        try:
            artifact_state: MigrationState | None = None
            active_phase = initial_state.phase
            if active_phase == "migration_pending" and plan.requires_system_ops_backup:
                reserved = await self._reserve_artifacts(
                    request=request,
                    plan=plan,
                    paths=paths,
                    report_evidence=report_evidence,
                )
                self._checkpoint("after_artifacts_pending")
                artifact_state = await self._publish_rollback_artifacts(
                    request=request,
                    plan=plan,
                    snapshot=snapshot,
                    paths=paths,
                    reserved=reserved,
                )
                self._checkpoint("after_artifacts_ready")
            elif active_phase == "artifacts_pending":
                artifact_state = await self._publish_rollback_artifacts(
                    request=request,
                    plan=plan,
                    snapshot=snapshot,
                    paths=paths,
                    reserved=initial_state,
                )
                self._checkpoint("after_artifacts_ready")
            elif active_phase == "artifacts_ready":
                artifact_state = initial_state

            if active_phase == "database_committed":
                if recovery_prepared is None:
                    raise LegacyImportError(LegacyImportErrorCode.PRECONDITION_FAILED)
                prepared = recovery_prepared
            else:
                _, prepared = await self._commit_database(
                    request=request,
                    plan=plan,
                    snapshot=snapshot,
                    paths=paths,
                    report_evidence=report_evidence,
                    artifact_state=artifact_state,
                )
                self._checkpoint("after_database_commit")
            completed = await self._verify_sanitize_and_complete(
                request=request,
                plan=plan,
                prepared=prepared,
            )
        except Exception as exc:  # noqa: BLE001 - sanitize operational boundary
            reason = (
                WebhookOperationalReasonCode.LEGACY_SOURCE_CHANGED
                if isinstance(exc, LegacyImportError) and exc.code is LegacyImportErrorCode.SOURCE_CHANGED
                else WebhookOperationalReasonCode.PRECONDITION_FAILED
                if isinstance(exc, LegacyImportError) and exc.code is LegacyImportErrorCode.PRECONDITION_FAILED
                else WebhookOperationalReasonCode.OPERATION_FAILED
            )
            with suppress(Exception):
                await self._emit_audit(
                    request=request,
                    plan=plan,
                    request_id=active_request_id,
                    outcome="failed",
                    reason_code=reason,
                )
            if isinstance(exc, LegacyImportError):
                raise
            raise LegacyImportError(LegacyImportErrorCode.OPERATION_FAILED) from None

        with suppress(Exception):
            await self._emit_audit(
                request=request,
                plan=plan,
                request_id=active_request_id,
                outcome="completed",
            )
        return completed

    async def verify_and_sanitize(
        self,
        request: LegacyImportRequest,
        *,
        approved_report_digest: str,
        request_id: str | None = None,
    ) -> MigrationState:
        """Resume approved canonical readback and structural sanitization."""
        return await self.apply_plan(
            request,
            approved_report_digest=approved_report_digest,
            request_id=request_id,
        )
