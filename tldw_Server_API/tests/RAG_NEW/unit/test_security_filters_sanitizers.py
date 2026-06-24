"""Sanitizer coverage for RAG security filter fallback logs."""

import hashlib
import json
from pathlib import Path

import pytest

from tldw_Server_API.app.core.RAG.rag_service import security_filters
from tldw_Server_API.app.core.RAG.rag_service.security_filters import (
    AccessController,
    PIIDetector,
    SecurityAuditor,
    SecurityFilters,
    SensitivityLevel,
)


pytestmark = pytest.mark.unit


_SENSITIVE_SUBSTRINGS = (
    "4111111111111111",
    "/private/security",
    "secret-card-token",
    "secret-audit-token",
    "secret-check-token",
    "secret-delete-token",
    "luhn failed",
    "rotation failed",
    "check failed",
    "delete failed",
)


class _LoggerStub:
    def __init__(self) -> None:
        self.records: list[tuple[str, str]] = []

    def debug(self, message: str) -> None:
        self.records.append(("debug", str(message)))

    def error(self, message: str) -> None:
        self.records.append(("error", str(message)))

    def info(self, message: str) -> None:
        self.records.append(("info", str(message)))


class _MalformedCardNumber:
    def __iter__(self):
        raise ValueError(
            "luhn failed for 4111111111111111 /private/security/card?token=secret-card-token"
        )

    def __str__(self) -> str:
        return "4111111111111111 /private/security/card?token=secret-card-token"


def _joined_logs(logger_stub: _LoggerStub) -> str:
    return "\n".join(message for _level, message in logger_stub.records)


def _assert_logs_are_sanitized(logger_stub: _LoggerStub) -> None:
    joined = _joined_logs(logger_stub)
    for sensitive in _SENSITIVE_SUBSTRINGS:
        assert sensitive not in joined


def test_luhn_fallback_log_omits_card_value_and_exception_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()
    monkeypatch.setattr(security_filters, "logger", logger_stub)

    result = PIIDetector()._luhn_check(_MalformedCardNumber())  # type: ignore[arg-type]

    assert result is False
    assert logger_stub.records == [
        ("debug", "Luhn checksum validation failed: ValueError"),
    ]
    _assert_logs_are_sanitized(logger_stub)


def test_audit_rotation_fallback_log_omits_path_and_exception_details(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    logger_stub = _LoggerStub()
    monkeypatch.setattr(security_filters, "logger", logger_stub)
    auditor = SecurityAuditor(db_path=str(tmp_path / "audit.db"))

    def fail_rename(self, target) -> None:
        raise OSError(
            "rotation failed for /private/security/audit.db?token=secret-audit-token"
        )

    monkeypatch.setattr(security_filters.Path, "rename", fail_rename)

    auditor._rotate_by_size()

    assert logger_stub.records[-1] == ("error", "Error rotating audit log: OSError")
    _assert_logs_are_sanitized(logger_stub)


def test_audit_rotation_check_fallback_log_omits_path_and_exception_details(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    logger_stub = _LoggerStub()
    monkeypatch.setattr(security_filters, "logger", logger_stub)
    auditor = SecurityAuditor(db_path=str(tmp_path / "audit.db"))
    auditor._last_rotation_check = 0

    def fail_delete_old_records() -> None:
        raise RuntimeError(
            "check failed for /private/security/audit.db?token=secret-check-token"
        )

    monkeypatch.setattr(auditor, "_delete_old_records", fail_delete_old_records)

    auditor._check_rotation()

    assert logger_stub.records[-1] == ("error", "Error checking rotation: RuntimeError")
    _assert_logs_are_sanitized(logger_stub)


def test_audit_delete_fallback_log_omits_path_and_exception_details(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    logger_stub = _LoggerStub()
    monkeypatch.setattr(security_filters, "logger", logger_stub)
    auditor = SecurityAuditor(db_path=str(tmp_path / "audit.db"))

    def fail_connect(*_args, **_kwargs):
        raise RuntimeError(
            "delete failed for /private/security/audit.db?token=secret-delete-token"
        )

    monkeypatch.setattr(security_filters.sqlite3, "connect", fail_connect)

    auditor._delete_old_records()

    assert logger_stub.records[-1] == ("error", "Error deleting old records: RuntimeError")
    _assert_logs_are_sanitized(logger_stub)


def test_process_query_audit_metadata_omits_raw_query_and_pii_text(tmp_path: Path) -> None:
    """Verify query audit metadata excludes raw query and PII text."""
    audit_path = tmp_path / "audit.db"
    filters = SecurityFilters(audit_db_path=str(audit_path))
    raw_email = "jane.doe@example.com"
    query = f"Please find records for {raw_email} about confidential budget planning."

    processed_query, metadata = filters.process_query(
        query,
        user_id="user-123",
        mask_pii=True,
    )

    rendered_metadata = json.dumps(metadata)
    assert raw_email not in processed_query
    assert raw_email not in rendered_metadata
    assert query not in rendered_metadata
    assert "original_query" not in metadata
    assert metadata["query_hash"] == hashlib.sha256(query.encode("utf-8")).hexdigest()
    assert metadata["pii_detected"]
    assert all("text" not in match for match in metadata["pii_detected"])

    assert filters.auditor is not None
    rows = filters.auditor.get_audit_trail(user_id="user-123", limit=1)
    assert len(rows) == 1
    audit_metadata_text = rows[0]["metadata"]
    assert raw_email not in audit_metadata_text
    assert query not in audit_metadata_text
    audit_metadata = json.loads(audit_metadata_text)
    assert "original_query" not in audit_metadata
    assert all("text" not in match for match in audit_metadata["pii_detected"])


@pytest.mark.parametrize(
    ("role", "expected_access"),
    [
        (
            "admin",
            {
                SensitivityLevel.PUBLIC: True,
                SensitivityLevel.INTERNAL: True,
                SensitivityLevel.CONFIDENTIAL: True,
                SensitivityLevel.RESTRICTED: True,
            },
        ),
        (
            "manager",
            {
                SensitivityLevel.PUBLIC: True,
                SensitivityLevel.INTERNAL: True,
                SensitivityLevel.CONFIDENTIAL: True,
                SensitivityLevel.RESTRICTED: False,
            },
        ),
        (
            "employee",
            {
                SensitivityLevel.PUBLIC: True,
                SensitivityLevel.INTERNAL: True,
                SensitivityLevel.CONFIDENTIAL: False,
                SensitivityLevel.RESTRICTED: False,
            },
        ),
    ],
)
def test_role_sensitivity_permissions_are_cumulative(
    role: str,
    expected_access: dict[SensitivityLevel, bool],
) -> None:
    """Verify each role grants cumulative access to expected sensitivity levels."""
    controller = AccessController()
    user_id = f"{role}-user"
    controller.set_user_role(user_id, role)

    assert {
        level: controller.check_access(user_id, "doc", level)
        for level in SensitivityLevel
    } == expected_access
