import sys
import types
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints import connectors
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal


def test_extract_request_base_debug_log_is_sanitized(monkeypatch):
    class _BadRequest:
        @property
        def base_url(self):
            raise ValueError("request base leaked at /private/connectors-callback")

    fake_logger = MagicMock()
    monkeypatch.setattr(connectors, "logger", fake_logger)

    result = connectors._extract_request_base(_BadRequest())

    assert result == ""
    fake_logger.debug.assert_called_once_with("Failed to resolve base_url from request")


def test_load_active_job_warning_log_is_sanitized(monkeypatch):
    class _FailingJobManager:
        def get_job(self, job_id: int):
            assert job_id == 42
            raise RuntimeError("connectors job backend exploded at /private/connectors-jobs.db")

    fake_jobs_module = types.ModuleType("jobs_manager")
    fake_jobs_module.JobManager = _FailingJobManager
    fake_logger = MagicMock()

    monkeypatch.setitem(sys.modules, "tldw_Server_API.app.core.Jobs.manager", fake_jobs_module)
    monkeypatch.setattr(connectors, "logger", fake_logger)

    result = connectors._load_active_job({"active_job_id": "42"})

    assert result is None
    fake_logger.warning.assert_called_once_with("Failed to load active connectors job")


@pytest.mark.asyncio
async def test_queue_source_job_quota_log_is_sanitized(monkeypatch):
    fake_logger = MagicMock()
    monkeypatch.setattr(connectors, "logger", fake_logger)

    def _failing_counter(user_id: int) -> int:
        assert user_id == 7
        raise RuntimeError("quota backend leaked /private/connectors-quota.db")

    principal = AuthPrincipal(kind="user", user_id=7, roles=["member"])
    org_policy = {"quotas_per_role": {"member": {"max_jobs_per_day": 1}}}

    with pytest.raises(HTTPException) as exc_info:
        await connectors._queue_source_job(
            source_id=123,
            request=None,
            principal=principal,
            org_policy=org_policy,
            count_jobs_fn=_failing_counter,
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Daily import quota check failed"
    fake_logger.error.assert_called_once_with("Connectors quota check failed")


@pytest.mark.asyncio
async def test_browse_provider_sources_error_log_is_sanitized(monkeypatch):
    class _FailingConnector:
        async def list_files(self, *args, **kwargs):
            raise RuntimeError("browse backend leaked /private/connectors-browse.db")

    async def _get_tokens(db, user_id: int, account_id: int):
        assert user_id == 7
        assert account_id == 11
        return {"access_token": "token"}

    async def _get_account(db, user_id: int, account_id: int):
        assert user_id == 7
        assert account_id == 11
        return {"id": 11, "provider": "drive"}

    async def _get_email(db, user_id: int, account_id: int):
        assert user_id == 7
        assert account_id == 11
        return "user@example.test"

    fake_logger = MagicMock()
    monkeypatch.setattr(connectors, "logger", fake_logger)
    monkeypatch.setattr(connectors, "get_account_tokens", _get_tokens)
    monkeypatch.setattr(connectors, "get_account_for_user", _get_account)
    monkeypatch.setattr(connectors, "get_account_email", _get_email)
    monkeypatch.setattr(connectors, "get_connector_by_name", lambda provider: _FailingConnector())

    principal = AuthPrincipal(kind="user", user_id=7)

    with pytest.raises(HTTPException) as exc_info:
        await connectors.browse_provider_sources(
            provider="drive",
            account_id=11,
            db=object(),
            principal=principal,
        )

    assert exc_info.value.status_code == 502
    assert exc_info.value.detail == "Browse failed"
    fake_logger.error.assert_called_once_with("Connector browse failed")


@pytest.mark.asyncio
async def test_add_source_policy_error_log_is_sanitized(monkeypatch):
    async def _get_account(db, user_id: int, account_id: int):
        assert user_id == 7
        assert account_id == 11
        return {"id": 11, "provider": "drive"}

    def _failing_policy(*args, **kwargs):
        raise RuntimeError("policy backend leaked /private/connectors-policy.db")

    fake_logger = MagicMock()
    monkeypatch.setattr(connectors, "logger", fake_logger)
    monkeypatch.setattr(connectors, "get_account_for_user", _get_account)
    monkeypatch.setattr(connectors, "evaluate_policy_constraints", _failing_policy)

    payload = connectors.ConnectorSourceCreateRequest(
        account_id=11,
        provider="drive",
        remote_id="root",
        type="folder",
        path="/Team Drive",
    )
    principal = AuthPrincipal(kind="user", user_id=7)

    with pytest.raises(HTTPException) as exc_info:
        await connectors.add_source(
            request=object(),
            payload=payload,
            db=object(),
            principal=principal,
            org_policy={},
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Source denied: policy evaluation failed"
    fake_logger.error.assert_called_once_with("Connector source policy evaluation failed")


@pytest.mark.asyncio
async def test_add_source_webhook_warning_log_is_sanitized(monkeypatch):
    class _FailingWebhookConnector:
        redirect_base = "http://localhost:8000"

        async def subscribe_webhook(self, *args, **kwargs):
            raise RuntimeError("webhook backend leaked /private/connectors-webhooks.db")

    async def _get_account(db, user_id: int, account_id: int):
        assert user_id == 7
        assert account_id == 11
        return {"id": 11, "provider": "drive"}

    async def _create_source(db, **kwargs):
        return {
            "id": 123,
            "account_id": kwargs["account_id"],
            "provider": kwargs["provider"],
            "remote_id": kwargs["remote_id"],
            "type": kwargs["type_"],
            "path": kwargs["path"],
            "options": kwargs["options"],
            "enabled": kwargs["enabled"],
        }

    async def _get_tokens(db, user_id: int, account_id: int):
        assert user_id == 7
        assert account_id == 11
        return {"access_token": "token"}

    sync_updates = []

    async def _upsert_sync_state(db, **kwargs):
        sync_updates.append(kwargs)

    fake_logger = MagicMock()
    monkeypatch.setattr(connectors, "logger", fake_logger)
    monkeypatch.setattr(connectors, "get_account_for_user", _get_account)
    monkeypatch.setattr(connectors, "evaluate_policy_constraints", lambda *args, **kwargs: (True, None))
    monkeypatch.setattr(connectors, "create_source", _create_source)
    monkeypatch.setattr(connectors, "get_account_tokens", _get_tokens)
    monkeypatch.setattr(connectors, "get_connector_by_name", lambda provider: _FailingWebhookConnector())
    monkeypatch.setattr(connectors, "upsert_source_sync_state", _upsert_sync_state)

    payload = connectors.ConnectorSourceCreateRequest(
        account_id=11,
        provider="drive",
        remote_id="root",
        type="folder",
        path="/Team Drive",
    )
    principal = AuthPrincipal(kind="user", user_id=7)

    result = await connectors.add_source(
        request=None,
        payload=payload,
        db=object(),
        principal=principal,
        org_policy={},
    )

    assert result.id == 123
    assert sync_updates == [
        {
            "source_id": 123,
            "sync_mode": "hybrid",
            "webhook_status": "failed",
            "last_error": "webhook backend leaked /private/connectors-webhooks.db",
        }
    ]
    fake_logger.warning.assert_called_once_with("Connector webhook provisioning failed")
