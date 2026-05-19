import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints.acp_triggers import (
    _get_trigger_manager,
    _sanitize_webhook_error_detail,
    _sanitize_webhook_success_result,
)


def test_sanitize_webhook_error_detail_preserves_known_client_errors() -> None:
    status_code, detail = _sanitize_webhook_error_detail(
        {"status": "rejected", "error": "verification_failed"}
    )

    assert status_code == 403
    assert detail == {"status": "rejected", "error": "verification_failed"}


def test_sanitize_webhook_error_detail_hides_internal_failures() -> None:
    status_code, detail = _sanitize_webhook_error_detail(
        {"status": "error", "error": "submission_failed: stack trace details"}
    )

    assert status_code == 503
    assert detail == {"status": "error", "error": "internal_error"}


def test_sanitize_webhook_error_detail_hides_secret_decryption_failures() -> None:
    status_code, detail = _sanitize_webhook_error_detail(
        {"status": "error", "error": "secret_decryption_failed: decrypt failed"}
    )

    assert status_code == 503
    assert detail == {"status": "error", "error": "internal_error"}


def test_sanitize_webhook_success_result_strips_internal_fields() -> None:
    detail = _sanitize_webhook_success_result(
        {
            "status": "accepted",
            "task_id": "task-123",
            "run_id": "internal-run-id",
            "debug": {"trace_id": "abc"},
            "error": "should-not-leak",
        }
    )

    assert detail == {"status": "accepted", "task_id": "task-123"}


def test_get_trigger_manager_sanitizes_secret_manager_setup_errors(monkeypatch) -> None:
    """Secret-manager setup failures should not expose local config details."""

    class FakeStore:
        def get_db(self):
            return object()

    class FailingSecretManager:
        def __init__(self):
            raise ValueError("fernet key missing at /private/config")

    monkeypatch.setattr(
        "tldw_Server_API.app.services.admin_acp_sessions_service._store",
        FakeStore(),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Agent_Client_Protocol.triggers.TriggerSecretManager",
        FailingSecretManager,
    )

    with pytest.raises(HTTPException) as exc_info:
        _get_trigger_manager()

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "Webhook trigger encryption not configured"
