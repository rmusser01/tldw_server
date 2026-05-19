"""
Tests for consent management endpoints.

Verifies that:
- GET /consent/preferences returns user consents
- POST /consent/preferences/{purpose} grants consent
- DELETE /consent/preferences/{purpose} withdraws consent
- Proper error handling for missing consent
"""
from __future__ import annotations

from datetime import datetime
import os

import pytest

from tldw_Server_API.app.api.v1.endpoints import consent as consent_endpoint
from tldw_Server_API.app.api.v1.endpoints.consent import (
    _get_consent_db_path,
    _get_consent_manager,
    _resolve_user_id,
)
from tldw_Server_API.app.api.v1.schemas.consent_schemas import ConsentRecordResponse
from tldw_Server_API.app.core.AuthNZ.consent_manager import ConsentManager
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.tests.helpers.app_main_state import reload_app_main

from fastapi import HTTPException


_CONSENT_SENSITIVE_MARKERS = (
    "consent backend leaked",
    "local-consent-db-path",
    "purpose-secret",
    "42",
)


class _LoggerStub:
    def __init__(self):
        self.error_calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def error(self, message: str, *args: object, **kwargs: object) -> None:
        self.error_calls.append((message, args, kwargs))


class _ExplodingConsentManager:
    def get_user_consents(self, user_id: int):
        raise RuntimeError("consent backend leaked local-consent-db-path 42")

    def grant_consent(
        self,
        user_id: int,
        purpose: str,
        *,
        ip_address: str | None = None,
        user_agent: str | None = None,
    ):
        raise RuntimeError("consent backend leaked local-consent-db-path purpose-secret 42")

    def withdraw_consent(self, user_id: int, purpose: str):
        raise RuntimeError("consent backend leaked local-consent-db-path purpose-secret 42")


def _assert_sanitized_error_log(logger_stub: _LoggerStub, expected_message: str) -> None:
    assert len(logger_stub.error_calls) == 1
    message, args, kwargs = logger_stub.error_calls[0]
    assert message == expected_message
    assert args == ()
    assert kwargs == {}
    rendered = " ".join([message, *(str(arg) for arg in args)])
    for marker in _CONSENT_SENSITIVE_MARKERS:
        assert marker not in rendered


@pytest.fixture()
def consent_db(tmp_path, monkeypatch):
    """Set up a temporary consent database."""
    db_path = str(tmp_path / "consent_test.db")
    monkeypatch.setenv("CONSENT_DB_PATH", db_path)
    return db_path


@pytest.fixture()
def principal():
    """Create a test auth principal."""
    return AuthPrincipal(kind="user", user_id=42, username="testuser")


class TestResolveUserId:
    def test_valid_principal(self, principal):
        assert _resolve_user_id(principal) == 42

    def test_none_user_id_raises(self):
        p = AuthPrincipal(kind="anonymous", user_id=None)
        with pytest.raises(HTTPException) as exc_info:
            _resolve_user_id(p)
        assert exc_info.value.status_code == 401


class TestGetConsentDbPath:
    def test_env_var_used(self, monkeypatch, tmp_path):
        db_path = str(tmp_path / "custom_consent.db")
        monkeypatch.setenv("CONSENT_DB_PATH", db_path)
        assert _get_consent_db_path() == db_path

    def test_fallback_path(self, monkeypatch):
        monkeypatch.delenv("CONSENT_DB_PATH", raising=False)
        path = _get_consent_db_path()
        assert "consent.db" in path

    def test_consent_manager_is_cached_per_db_path(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CONSENT_DB_PATH", str(tmp_path / "cached-consent.db"))

        first = _get_consent_manager()
        second = _get_consent_manager()

        assert first is second


class TestConsentSchemas:
    def test_consent_record_parses_datetime_fields(self):
        record = ConsentRecordResponse.model_validate(
            {
                "user_id": 42,
                "purpose": "analytics",
                "granted_at": "2026-03-17T12:34:56Z",
                "withdrawn_at": "2026-03-18T01:02:03Z",
            }
        )

        assert isinstance(record.granted_at, datetime)
        assert isinstance(record.withdrawn_at, datetime)


class TestConsentEndpointLogic:
    """Test the consent manager integration directly (unit-level)."""

    def test_grant_and_get_preferences(self, consent_db):
        mgr = _get_consent_manager()
        mgr.grant_consent(42, "analytics")
        mgr.grant_consent(42, "marketing")

        records = mgr.get_user_consents(42)
        assert len(records) == 2
        purposes = {r["purpose"] for r in records}
        assert purposes == {"analytics", "marketing"}

    def test_withdraw_consent(self, consent_db):
        mgr = _get_consent_manager()
        mgr.grant_consent(42, "analytics")

        result = mgr.withdraw_consent(42, "analytics")
        assert result is not None
        assert result["purpose"] == "analytics"

        # Verify withdrawn
        assert not mgr.check_consent(42, "analytics")

    def test_withdraw_nonexistent_returns_none(self, consent_db):
        mgr = _get_consent_manager()
        result = mgr.withdraw_consent(42, "nonexistent")
        assert result is None

    def test_grant_with_metadata(self, consent_db):
        mgr = _get_consent_manager()
        result = mgr.grant_consent(
            42, "tracking",
            ip_address="10.0.0.1",
            user_agent="TestBot/1.0",
        )
        assert result["purpose"] == "tracking"
        assert result["user_id"] == 42

    def test_get_empty_preferences(self, consent_db):
        mgr = _get_consent_manager()
        records = mgr.get_user_consents(999)
        assert records == []


class TestConsentEndpointAsync:
    """Test the async endpoint functions."""

    @pytest.mark.asyncio
    async def test_get_preferences_endpoint(self, consent_db, principal):
        from tldw_Server_API.app.api.v1.endpoints.consent import get_consent_preferences

        # Grant some consent first
        mgr = _get_consent_manager()
        mgr.grant_consent(42, "analytics")

        result = await get_consent_preferences(principal=principal)
        assert result.user_id == 42
        assert len(result.consents) == 1

    @pytest.mark.asyncio
    async def test_grant_consent_endpoint(self, consent_db, principal):
        from unittest.mock import MagicMock
        from tldw_Server_API.app.api.v1.endpoints.consent import grant_consent

        mock_request = MagicMock()
        mock_request.client = MagicMock()
        mock_request.client.host = "127.0.0.1"
        mock_request.headers = {"user-agent": "TestBot/1.0"}

        result = await grant_consent(
            purpose="analytics",
            request=mock_request,
            principal=principal,
        )
        assert result.user_id == 42
        assert result.purpose == "analytics"

    @pytest.mark.asyncio
    async def test_withdraw_consent_endpoint(self, consent_db, principal):
        from tldw_Server_API.app.api.v1.endpoints.consent import withdraw_consent

        # Grant first
        mgr = _get_consent_manager()
        mgr.grant_consent(42, "analytics")

        result = await withdraw_consent(purpose="analytics", principal=principal)
        assert result.purpose == "analytics"
        assert result.withdrawn_at is not None

    @pytest.mark.asyncio
    async def test_withdraw_missing_raises_404(self, consent_db, principal):
        from tldw_Server_API.app.api.v1.endpoints.consent import withdraw_consent

        with pytest.raises(HTTPException) as exc_info:
            await withdraw_consent(purpose="nonexistent", principal=principal)
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_get_preferences_sanitizes_backend_failure_log(self, monkeypatch, principal):
        logger_stub = _LoggerStub()
        monkeypatch.setattr(consent_endpoint, "logger", logger_stub)
        monkeypatch.setattr(
            consent_endpoint,
            "_get_consent_manager",
            lambda: _ExplodingConsentManager(),
        )

        with pytest.raises(HTTPException) as exc_info:
            await consent_endpoint.get_consent_preferences(principal=principal)

        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "Failed to retrieve consent preferences."
        _assert_sanitized_error_log(logger_stub, "Failed to get consent preferences")

    @pytest.mark.asyncio
    async def test_grant_consent_sanitizes_backend_failure_log(self, monkeypatch, principal):
        from unittest.mock import MagicMock

        logger_stub = _LoggerStub()
        monkeypatch.setattr(consent_endpoint, "logger", logger_stub)
        monkeypatch.setattr(
            consent_endpoint,
            "_get_consent_manager",
            lambda: _ExplodingConsentManager(),
        )
        mock_request = MagicMock()
        mock_request.client = MagicMock()
        mock_request.client.host = "127.0.0.1"
        mock_request.headers = {"user-agent": "TestBot/1.0"}

        with pytest.raises(HTTPException) as exc_info:
            await consent_endpoint.grant_consent(
                purpose="purpose-secret",
                request=mock_request,
                principal=principal,
            )

        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "Failed to record consent."
        _assert_sanitized_error_log(logger_stub, "Failed to grant consent")

    @pytest.mark.asyncio
    async def test_withdraw_consent_sanitizes_backend_failure_log(self, monkeypatch, principal):
        logger_stub = _LoggerStub()
        monkeypatch.setattr(consent_endpoint, "logger", logger_stub)
        monkeypatch.setattr(
            consent_endpoint,
            "_get_consent_manager",
            lambda: _ExplodingConsentManager(),
        )

        with pytest.raises(HTTPException) as exc_info:
            await consent_endpoint.withdraw_consent(
                purpose="purpose-secret",
                principal=principal,
            )

        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "Failed to withdraw consent."
        _assert_sanitized_error_log(logger_stub, "Failed to withdraw consent")


class TestConsentRouterWiring:
    def test_production_app_includes_consent_routes(self, monkeypatch):
        monkeypatch.setenv("MINIMAL_TEST_APP", "0")
        monkeypatch.setenv("ULTRA_MINIMAL_APP", "0")
        monkeypatch.delenv("ROUTES_DISABLE", raising=False)
        monkeypatch.delenv("ROUTES_ENABLE", raising=False)

        from tldw_Server_API.app.core import config as config_mod
        config_mod.clear_config_cache()
        reloaded = reload_app_main()
        route_paths = {getattr(route, "path", "") for route in reloaded.app.routes}

        assert "/api/v1/consent/preferences" in route_paths

    def test_production_app_omits_consent_routes_when_disabled(self, monkeypatch):
        monkeypatch.setenv("MINIMAL_TEST_APP", "0")
        monkeypatch.setenv("ULTRA_MINIMAL_APP", "0")
        monkeypatch.setenv("ROUTES_DISABLE", "consent")
        monkeypatch.delenv("ROUTES_ENABLE", raising=False)

        from tldw_Server_API.app.core import config as config_mod
        config_mod.clear_config_cache()
        reloaded = reload_app_main()
        route_paths = {getattr(route, "path", "") for route in reloaded.app.routes}

        assert "/api/v1/consent/preferences" not in route_paths
