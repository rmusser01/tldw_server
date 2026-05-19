"""Sanitizer coverage for RAG security filter fallback logs."""

from pathlib import Path

import pytest

from tldw_Server_API.app.core.RAG.rag_service import security_filters
from tldw_Server_API.app.core.RAG.rag_service.security_filters import PIIDetector, SecurityAuditor


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
