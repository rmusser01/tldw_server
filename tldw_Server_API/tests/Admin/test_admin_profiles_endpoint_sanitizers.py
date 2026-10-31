from __future__ import annotations

from typing import Any

import pytest

from tldw_Server_API.app.api.v1.endpoints.admin import admin_profiles

pytestmark = pytest.mark.unit


class _LoggerStub:
    def __init__(self) -> None:
        self.warning_records: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.warning_records.append((message, args, kwargs))

    def opt(self, **_kwargs: Any) -> _LoggerStub:
        """Mirror loguru's `logger.opt(...)`, which returns a logger to call `.warning()` on."""
        return self


async def _raise_audit_error(*_args: Any, **_kwargs: Any) -> None:
    raise RuntimeError("profile audit backend exploded at /private/audit.db")


@pytest.mark.asyncio
async def test_list_user_profiles_sanitizes_audit_warning_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()
    response = object()

    async def _list_user_profiles(**_kwargs: Any):
        return response, {"event_type": "admin_profiles_listed"}

    monkeypatch.setattr(admin_profiles, "logger", logger_stub)
    monkeypatch.setattr(
        admin_profiles.admin_profiles_service,
        "list_user_profiles",
        _list_user_profiles,
    )
    monkeypatch.setattr(admin_profiles, "_get_emit_admin_audit_event", lambda: _raise_audit_error)

    result = await admin_profiles.admin_list_user_profiles(
        http_request=object(),
        sections=None,
        include_sources=False,
        include_raw=False,
        mask_secrets=True,
        user_ids=None,
        org_id=None,
        team_id=None,
        role=None,
        is_active=None,
        search=None,
        page=1,
        limit=25,
        principal=object(),
        session_manager=object(),
    )

    assert result is response
    assert logger_stub.warning_records == [("Admin audit emission failed", (), {})]


@pytest.mark.asyncio
async def test_get_user_profile_sanitizes_audit_warning_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()
    response = object()

    async def _get_user_profile(**_kwargs: Any):
        return response, {"event_type": "admin_profile_read"}

    monkeypatch.setattr(admin_profiles, "logger", logger_stub)
    monkeypatch.setattr(
        admin_profiles.admin_profiles_service,
        "get_user_profile",
        _get_user_profile,
    )
    monkeypatch.setattr(admin_profiles, "_get_emit_admin_audit_event", lambda: _raise_audit_error)

    result = await admin_profiles.admin_get_user_profile(
        user_id=42,
        http_request=object(),
        sections=None,
        include_sources=False,
        include_raw=False,
        mask_secrets=True,
        principal=object(),
        session_manager=object(),
    )

    assert result is response
    assert logger_stub.warning_records == [("Admin audit emission failed", (), {})]


@pytest.mark.asyncio
async def test_update_user_profile_sanitizes_audit_warning_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()
    response = object()

    async def _update_user_profile(**_kwargs: Any):
        return response, {"event_type": "admin_profile_updated"}

    monkeypatch.setattr(admin_profiles, "logger", logger_stub)
    monkeypatch.setattr(
        admin_profiles.admin_profiles_service,
        "update_user_profile",
        _update_user_profile,
    )
    monkeypatch.setattr(admin_profiles, "_get_emit_admin_audit_event", lambda: _raise_audit_error)

    result = await admin_profiles.admin_update_user_profile(
        user_id=42,
        payload=object(),
        http_request=object(),
        principal=object(),
        db=object(),
    )

    assert result is response
    assert logger_stub.warning_records == [("Admin audit emission failed", (), {})]


@pytest.mark.asyncio
async def test_bulk_update_user_profiles_sanitizes_audit_warning_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()
    response = object()

    async def _bulk_update_user_profiles(**_kwargs: Any):
        return response, {"event_type": "admin_profiles_bulk_updated"}

    monkeypatch.setattr(admin_profiles, "logger", logger_stub)
    monkeypatch.setattr(
        admin_profiles.admin_profiles_service,
        "bulk_update_user_profiles",
        _bulk_update_user_profiles,
    )
    monkeypatch.setattr(admin_profiles, "_get_emit_admin_audit_event", lambda: _raise_audit_error)

    result = await admin_profiles.admin_bulk_update_user_profiles(
        payload=object(),
        http_request=object(),
        principal=object(),
    )

    assert result is response
    assert logger_stub.warning_records == [("Admin audit emission failed", (), {})]


@pytest.mark.asyncio
async def test_group_override_audit_warning_log_has_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed group-override audit logs the group type/id, key and operation, plus the exception (Qodo Q6)."""
    logger_stub = _LoggerStub()

    async def _set_group_limit_override(**_kwargs: Any):
        return (
            {"scope": "team", "id": 9, "key": "limits.rag_queries_per_day", "value": 20},
            {"action": "team_profile_override.set"},
        )

    monkeypatch.setattr(admin_profiles, "logger", logger_stub)
    monkeypatch.setattr(
        admin_profiles.admin_profiles_service,
        "set_group_limit_override",
        _set_group_limit_override,
    )
    monkeypatch.setattr(admin_profiles, "_get_emit_admin_audit_event", lambda: _raise_audit_error)

    result = await admin_profiles._group_override(
        "team", 9, "limits.rag_queries_per_day", 20, object(), object()
    )

    assert result.value == 20
    assert len(logger_stub.warning_records) == 1
    message, args, _kwargs = logger_stub.warning_records[0]
    assert message == "Admin audit emission failed for {} {} override {} ({})"
    assert args == ("team", 9, "limits.rag_queries_per_day", "team_profile_override.set")
