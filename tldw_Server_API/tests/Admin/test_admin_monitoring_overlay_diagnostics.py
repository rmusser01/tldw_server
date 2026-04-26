from __future__ import annotations

import sys
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.api.v1.endpoints.admin import admin_monitoring as admin_monitoring_mod


class _LoggerStub:
    def __init__(self) -> None:
        self.errors: list[str] = []

    def error(self, message: str, *args, **kwargs) -> None:
        if args or kwargs:
            message = message.format(*args, **kwargs)
        self.errors.append(message)

    def exception(self, message: str, *args, **kwargs) -> None:
        if args or kwargs:
            message = message.format(*args, **kwargs)
        exc = sys.exc_info()[1]
        if exc is not None:
            message = f"{message}: {exc}"
        self.errors.append(message)


class _StubMonitoringDb:
    def __init__(self, row):
        self.row = row
        self.lookups: list[int] = []

    def get_alert(self, alert_id: int):
        self.lookups.append(alert_id)
        return self.row


def test_warn_if_runtime_alert_identity_missing_logs_warning(monkeypatch) -> None:
    db = _StubMonitoringDb(row=None)
    warnings: list[str] = []
    monkeypatch.setattr(
        admin_monitoring_mod.logger,
        "warning",
        lambda message, *args, **kwargs: warnings.append(str(message)),
    )

    admin_monitoring_mod._warn_if_overlay_identity_has_no_runtime_row("alert:77", db)

    assert db.lookups == [77]
    assert any("missing runtime alert" in msg for msg in warnings)


def test_warn_if_overlay_only_identity_logs_info_without_lookup(monkeypatch) -> None:
    db = _StubMonitoringDb(row=None)
    infos: list[str] = []
    monkeypatch.setattr(
        admin_monitoring_mod.logger,
        "info",
        lambda message, *args, **kwargs: infos.append(str(message)),
    )

    admin_monitoring_mod._warn_if_overlay_identity_has_no_runtime_row("fingerprint:abc", db)

    assert db.lookups == []
    assert any("overlay-only identity" in msg for msg in infos)


def test_warn_if_malformed_runtime_alert_identity_logs_warning(monkeypatch) -> None:
    db = _StubMonitoringDb(row=None)
    warnings: list[str] = []
    monkeypatch.setattr(
        admin_monitoring_mod.logger,
        "warning",
        lambda message, *args, **kwargs: warnings.append(str(message)),
    )

    admin_monitoring_mod._warn_if_overlay_identity_has_no_runtime_row("alert:not-an-int", db)

    assert db.lookups == []
    assert any("malformed runtime alert identity" in msg for msg in warnings)


@pytest.mark.asyncio
async def test_emit_overlay_identity_diagnostic_sanitizes_skip_log(monkeypatch) -> None:
    debug_records: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def _raise_path_resolution() -> str:
        raise RuntimeError("monitoring backend exploded at /private/monitoring.db")

    monkeypatch.setattr(
        admin_monitoring_mod,
        "_resolve_monitoring_alerts_db_path",
        _raise_path_resolution,
    )
    monkeypatch.setattr(
        admin_monitoring_mod.logger,
        "debug",
        lambda message, *args, **kwargs: debug_records.append((str(message), args, kwargs)),
    )

    await admin_monitoring_mod._emit_overlay_identity_diagnostic(
        "alert:private-alert-id",
    )

    assert debug_records == [("monitoring overlay diagnostic skipped", (), {})]


@pytest.mark.asyncio
async def test_delete_alert_rule_sanitizes_backend_error_log(monkeypatch) -> None:
    class _Repo:
        async def get_rule(self, rule_id: int):
            assert rule_id == 42
            return {"metric": "sensitive.metric"}

        async def delete_rule(self, rule_id: int):
            assert rule_id == 42
            raise RuntimeError("monitoring delete failed at /private/admin-monitoring.db")

    async def _fake_get_monitoring_repo():
        return _Repo()

    logger_stub = _LoggerStub()
    monkeypatch.setattr(admin_monitoring_mod, "_get_monitoring_repo", _fake_get_monitoring_repo)
    monkeypatch.setattr(admin_monitoring_mod, "logger", logger_stub)

    with pytest.raises(admin_monitoring_mod.HTTPException) as exc_info:
        await admin_monitoring_mod.delete_alert_rule(
            rule_id=42,
            request=SimpleNamespace(),
            principal=SimpleNamespace(user_id=7),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to delete alert rule"
    assert logger_stub.errors == ["Failed to delete admin alert rule"]
    assert "42" not in str(logger_stub.errors)
    assert "monitoring delete failed" not in str(logger_stub.errors)
    assert "/private/admin-monitoring.db" not in str(logger_stub.errors)


@pytest.mark.asyncio
async def test_create_alert_rule_sanitizes_backend_error_log(monkeypatch) -> None:
    from tldw_Server_API.app.api.v1.schemas.admin_schemas import AdminAlertRuleCreateRequest

    class _Repo:
        async def create_rule(self, **kwargs):
            assert kwargs["metric"] == "private.metric"
            raise RuntimeError("monitoring create failed at /private/admin-monitoring.db")

    async def _fake_get_monitoring_repo():
        return _Repo()

    logger_stub = _LoggerStub()
    monkeypatch.setattr(admin_monitoring_mod, "_get_monitoring_repo", _fake_get_monitoring_repo)
    monkeypatch.setattr(admin_monitoring_mod, "logger", logger_stub)

    with pytest.raises(admin_monitoring_mod.HTTPException) as exc_info:
        await admin_monitoring_mod.create_alert_rule(
            payload=AdminAlertRuleCreateRequest(
                metric="private.metric",
                operator=">",
                threshold=1.0,
                duration_minutes=5,
                severity="secret-severity",
            ),
            request=SimpleNamespace(),
            principal=SimpleNamespace(user_id=7),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to create alert rule"
    assert logger_stub.errors == ["Failed to create admin alert rule"]
    assert "private.metric" not in str(logger_stub.errors)
    assert "secret-severity" not in str(logger_stub.errors)
    assert "monitoring create failed" not in str(logger_stub.errors)
    assert "/private/admin-monitoring.db" not in str(logger_stub.errors)


@pytest.mark.asyncio
async def test_list_alert_rules_sanitizes_backend_error_log(monkeypatch) -> None:
    class _Repo:
        async def list_rules(self):
            raise RuntimeError("monitoring list failed at /private/admin-monitoring.db")

    async def _fake_get_monitoring_repo():
        return _Repo()

    logger_stub = _LoggerStub()
    monkeypatch.setattr(admin_monitoring_mod, "_get_monitoring_repo", _fake_get_monitoring_repo)
    monkeypatch.setattr(admin_monitoring_mod, "logger", logger_stub)

    with pytest.raises(admin_monitoring_mod.HTTPException) as exc_info:
        await admin_monitoring_mod.list_alert_rules(principal=SimpleNamespace(user_id=7))

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to list alert rules"
    assert logger_stub.errors == ["Failed to list admin alert rules"]
    assert "monitoring list failed" not in str(logger_stub.errors)
    assert "/private/admin-monitoring.db" not in str(logger_stub.errors)


@pytest.mark.asyncio
async def test_list_alert_history_sanitizes_backend_error_log(monkeypatch) -> None:
    class _Repo:
        async def list_alert_events(self, *, alert_identity, limit):
            assert alert_identity == "alert:private-history"
            assert limit == 25
            raise RuntimeError("monitoring history failed at /private/admin-monitoring.db")

    async def _fake_get_monitoring_repo():
        return _Repo()

    logger_stub = _LoggerStub()
    monkeypatch.setattr(admin_monitoring_mod, "_get_monitoring_repo", _fake_get_monitoring_repo)
    monkeypatch.setattr(admin_monitoring_mod, "logger", logger_stub)

    with pytest.raises(admin_monitoring_mod.HTTPException) as exc_info:
        await admin_monitoring_mod.list_alert_history(
            alert_identity="alert:private-history",
            limit=25,
            principal=SimpleNamespace(user_id=7),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to list alert history"
    assert logger_stub.errors == ["Failed to list admin alert history"]
    assert "alert:private-history" not in str(logger_stub.errors)
    assert "25" not in str(logger_stub.errors)
    assert "monitoring history failed" not in str(logger_stub.errors)
    assert "/private/admin-monitoring.db" not in str(logger_stub.errors)


@pytest.mark.asyncio
async def test_assign_alert_sanitizes_backend_error_log(monkeypatch) -> None:
    from tldw_Server_API.app.api.v1.schemas.admin_schemas import AdminAlertAssignRequest

    class _UsersRepo:
        async def get_user_by_id(self, user_id: int):
            assert user_id == 123
            return {"id": user_id}

    class _Repo:
        async def upsert_alert_state(self, **kwargs):
            assert kwargs["alert_identity"] == "alert:private-assign"
            assert kwargs["assigned_to_user_id"] == 123
            raise RuntimeError("monitoring assign failed at /private/admin-monitoring.db")

    async def _fake_get_users_repo():
        return _UsersRepo()

    async def _fake_get_monitoring_repo():
        return _Repo()

    async def _noop_overlay_diagnostic(alert_identity: str) -> None:
        assert alert_identity == "alert:private-assign"

    logger_stub = _LoggerStub()
    monkeypatch.setattr(admin_monitoring_mod, "_get_users_repo", _fake_get_users_repo)
    monkeypatch.setattr(admin_monitoring_mod, "_get_monitoring_repo", _fake_get_monitoring_repo)
    monkeypatch.setattr(admin_monitoring_mod, "_emit_overlay_identity_diagnostic", _noop_overlay_diagnostic)
    monkeypatch.setattr(admin_monitoring_mod, "logger", logger_stub)

    with pytest.raises(admin_monitoring_mod.HTTPException) as exc_info:
        await admin_monitoring_mod.assign_alert(
            alert_identity="alert:private-assign",
            payload=AdminAlertAssignRequest(assigned_to_user_id=123),
            request=SimpleNamespace(),
            principal=SimpleNamespace(user_id=7),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to assign alert"
    assert logger_stub.errors == ["Failed to assign monitoring alert"]
    assert "alert:private-assign" not in str(logger_stub.errors)
    assert "123" not in str(logger_stub.errors)
    assert "monitoring assign failed" not in str(logger_stub.errors)
    assert "/private/admin-monitoring.db" not in str(logger_stub.errors)


@pytest.mark.asyncio
async def test_snooze_alert_sanitizes_backend_error_log(monkeypatch) -> None:
    from tldw_Server_API.app.api.v1.schemas.admin_schemas import AdminAlertSnoozeRequest

    snoozed_until = datetime(2026, 4, 27, 10, 30, tzinfo=timezone.utc)

    class _Repo:
        async def upsert_alert_state(self, **kwargs):
            assert kwargs["alert_identity"] == "alert:private-snooze"
            assert kwargs["snoozed_until"] == snoozed_until.isoformat()
            raise RuntimeError("monitoring snooze failed at /private/admin-monitoring.db")

    async def _fake_get_monitoring_repo():
        return _Repo()

    async def _noop_overlay_diagnostic(alert_identity: str) -> None:
        assert alert_identity == "alert:private-snooze"

    logger_stub = _LoggerStub()
    monkeypatch.setattr(admin_monitoring_mod, "_get_monitoring_repo", _fake_get_monitoring_repo)
    monkeypatch.setattr(admin_monitoring_mod, "_emit_overlay_identity_diagnostic", _noop_overlay_diagnostic)
    monkeypatch.setattr(admin_monitoring_mod, "logger", logger_stub)

    with pytest.raises(admin_monitoring_mod.HTTPException) as exc_info:
        await admin_monitoring_mod.snooze_alert(
            alert_identity="alert:private-snooze",
            payload=AdminAlertSnoozeRequest(snoozed_until=snoozed_until),
            request=SimpleNamespace(),
            principal=SimpleNamespace(user_id=7),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to snooze alert"
    assert logger_stub.errors == ["Failed to snooze monitoring alert"]
    assert "alert:private-snooze" not in str(logger_stub.errors)
    assert snoozed_until.isoformat() not in str(logger_stub.errors)
    assert "monitoring snooze failed" not in str(logger_stub.errors)
    assert "/private/admin-monitoring.db" not in str(logger_stub.errors)


@pytest.mark.asyncio
async def test_escalate_alert_sanitizes_backend_error_log(monkeypatch) -> None:
    from tldw_Server_API.app.api.v1.schemas.admin_schemas import AdminAlertEscalateRequest

    class _Repo:
        async def upsert_alert_state(self, **kwargs):
            assert kwargs["alert_identity"] == "alert:private-escalate"
            assert kwargs["escalated_severity"] == "critical"
            raise RuntimeError("monitoring escalate failed at /private/admin-monitoring.db")

    async def _fake_get_monitoring_repo():
        return _Repo()

    async def _noop_overlay_diagnostic(alert_identity: str) -> None:
        assert alert_identity == "alert:private-escalate"

    logger_stub = _LoggerStub()
    monkeypatch.setattr(admin_monitoring_mod, "_get_monitoring_repo", _fake_get_monitoring_repo)
    monkeypatch.setattr(admin_monitoring_mod, "_emit_overlay_identity_diagnostic", _noop_overlay_diagnostic)
    monkeypatch.setattr(admin_monitoring_mod, "logger", logger_stub)

    with pytest.raises(admin_monitoring_mod.HTTPException) as exc_info:
        await admin_monitoring_mod.escalate_alert(
            alert_identity="alert:private-escalate",
            payload=AdminAlertEscalateRequest(severity="critical"),
            request=SimpleNamespace(),
            principal=SimpleNamespace(user_id=7),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to escalate alert"
    assert logger_stub.errors == ["Failed to escalate monitoring alert"]
    assert "alert:private-escalate" not in str(logger_stub.errors)
    assert "critical" not in str(logger_stub.errors)
    assert "monitoring escalate failed" not in str(logger_stub.errors)
    assert "/private/admin-monitoring.db" not in str(logger_stub.errors)
