from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime

import pytest
from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.personal_context_deps import (
    get_personal_context_service,
    get_workspace_access_checker,
    personal_context_service_for_user,
)
from tldw_Server_API.app.api.v1.API_Deps.personalization_deps import (
    get_personalization_db_for_user,
)
from tldw_Server_API.app.api.v1.endpoints.personal_context import router
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.Personalization_DB import PersonalizationDB
from tldw_Server_API.tests.Personalization.personal_context_test_support import (
    encoded_master_key,
)

pytestmark = pytest.mark.unit


def _ids(prefix: str):
    counters: defaultdict[str, int] = defaultdict(int)

    def issue(label: str) -> str:
        counters[label] += 1
        return f"{prefix}-{label}-{counters[label]}"

    return issue


def test_authentication_failure_happens_before_personalization_db_open(monkeypatch) -> None:
    app = FastAPI()

    @app.get("/guarded")
    def guarded(_service=Depends(get_personal_context_service)):
        return {"ok": True}

    async def reject_user():
        raise HTTPException(status_code=401, detail="Not authenticated")

    opened = False

    def forbidden_open(_user_id):
        nonlocal opened
        opened = True
        raise AssertionError("storage opened before authentication")

    app.dependency_overrides[get_request_user] = reject_user
    monkeypatch.setattr(PersonalizationDB, "for_user", staticmethod(forbidden_open))

    with TestClient(app) as client:
        response = client.get("/guarded")

    assert response.status_code == 401
    assert opened is False


def test_cross_user_record_id_returns_same_not_found_without_fallback(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("TLDW_PERSONAL_CONTEXT_MASTER_KEY", encoded_master_key())
    databases = {
        "101": PersonalizationDB(str(tmp_path / "user-101.db")),
        "202": PersonalizationDB(str(tmp_path / "user-202.db")),
    }
    services = {
        user_id: personal_context_service_for_user(
            user_id,
            database=database,
            workspace_access=lambda _workspace_id: True,
            clock=lambda: datetime(2026, 8, 30, 20, 0, tzinfo=UTC),
            id_factory=_ids(user_id),
        )
        for user_id, database in databases.items()
    }
    for service in services.values():
        service.create_profile()
    scope_b = services["202"].list_scopes()[0]
    record_b = services["202"].create_manual_record(
        scope_id=scope_b.scope_id,
        payload={
            "schema_version": 1,
            "kind": "preference",
            "subject": "privacy",
            "polarity": "like",
            "value": "do not disclose",
        },
        semantic_key=None,
        controls={
            "sync_mode": "syncable",
            "agent_visibility": "user_only",
        },
    )

    selected_user = {"id": "101"}

    async def current_user():
        user_id = selected_user["id"]
        return User(id=int(user_id), username=f"user-{user_id}", is_active=True)

    def current_db():
        return databases[selected_user["id"]]

    app = FastAPI()
    app.include_router(router, prefix="/api/v1/personal-context")
    app.dependency_overrides[get_request_user] = current_user
    app.dependency_overrides[get_personalization_db_for_user] = current_db
    app.dependency_overrides[get_workspace_access_checker] = lambda: lambda _workspace_id: True

    with TestClient(app) as client:
        foreign = client.get(f"/api/v1/personal-context/records/{record_b.record_id}")
        missing = client.get("/api/v1/personal-context/records/does-not-exist")

    assert foreign.status_code == 404
    assert foreign.json() == {"detail": "Personal context record not found"}
    assert missing.status_code == 404
    assert missing.json() == foreign.json()
