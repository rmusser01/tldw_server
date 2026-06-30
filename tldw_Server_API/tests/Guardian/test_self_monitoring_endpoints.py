from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints import self_monitoring as selfmon_endpoint
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.config import settings as app_settings
from tldw_Server_API.app.core.DB_Management.Guardian_DB import GuardianDB
from tldw_Server_API.app.core.Monitoring import self_monitoring_service as selfmon_service
from tldw_Server_API.app.core.Monitoring.self_monitoring_service import SelfMonitoringService


def _user(user_id: str) -> User:
    return User(id=user_id, username=user_id, roles=["user"], permissions=[])


def test_partner_approval_endpoint_can_target_owner_guardian_db(tmp_path, monkeypatch):
    owner_db = GuardianDB(str(tmp_path / "owner.db"))
    partner_db = GuardianDB(str(tmp_path / "partner.db"))
    rule = owner_db.create_self_monitoring_rule(
        user_id="owner1",
        name="Owner Rule",
        patterns=["word"],
        notification_frequency="every_message",
        bypass_protection="partner_approval",
        bypass_partner_user_id="partner1",
        cooldown_minutes=99999,
    )
    request_result = SelfMonitoringService(owner_db).request_deactivation(rule.id, "owner1")
    token = request_result["confirmation_token"]

    app = FastAPI()
    app.include_router(selfmon_endpoint.router, prefix="/api/v1/self-monitoring")

    async def _current_partner():
        return _user("partner1")

    app.dependency_overrides[selfmon_endpoint.get_request_user] = _current_partner
    app.dependency_overrides[selfmon_endpoint.get_guardian_db_for_user] = lambda: partner_db

    def _existing_db_for_user_id(user_id):
        if str(user_id) == "owner1":
            return owner_db
        return partner_db

    monkeypatch.setattr(
        selfmon_service,
        "resolve_existing_guardian_db_for_user_id",
        _existing_db_for_user_id,
    )

    with TestClient(app) as client:
        resp = client.post(
            f"/api/v1/self-monitoring/rules/{rule.id}/approve-deactivation",
            json={"token": token, "owner_user_id": "owner1"},
        )

    assert resp.status_code == 200
    assert resp.json()["status"] == "disabled"
    assert owner_db.get_self_monitoring_rule(rule.id).enabled is False
    assert partner_db.get_self_monitoring_rule(rule.id) is None


def test_partner_approval_missing_owner_db_does_not_create_storage(tmp_path, monkeypatch):
    user_db_base = tmp_path / "user_databases"
    partner_db = GuardianDB(str(tmp_path / "partner.db"))

    app = FastAPI()
    app.include_router(selfmon_endpoint.router, prefix="/api/v1/self-monitoring")

    async def _current_partner():
        return _user("partner1")

    app.dependency_overrides[selfmon_endpoint.get_request_user] = _current_partner
    app.dependency_overrides[selfmon_endpoint.get_guardian_db_for_user] = lambda: partner_db
    monkeypatch.setenv("USER_DB_BASE_DIR", str(user_db_base))
    monkeypatch.setitem(app_settings, "USER_DB_BASE_DIR", str(user_db_base))

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/self-monitoring/rules/missing-rule/approve-deactivation",
            json={"token": "bad-token", "owner_user_id": "123456"},
        )

    assert resp.status_code == 400
    assert resp.json()["detail"] == "Owner Guardian DB not found"
    assert not user_db_base.exists()
