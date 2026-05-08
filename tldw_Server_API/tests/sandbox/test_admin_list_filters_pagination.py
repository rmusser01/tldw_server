from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from tldw_Server_API.app.core.Sandbox.models import RunPhase, RunStatus, RuntimeType
from tldw_Server_API.app.main import app


def _client(monkeypatch) -> TestClient:


    monkeypatch.setenv("TEST_MODE", "1")
    # Use in-memory store by default (already defaulted in config)
    return TestClient(app)


def _admin_user_dep():


     # Override get_request_user to return admin
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
    return User(id=1, username="admin", roles=["admin"], is_admin=True)


def _seed_run(run_id: str, user_id: int, image_digest: str, started_offset_sec: int, phase: str = "completed") -> None:
    from tldw_Server_API.app.api.v1.endpoints import sandbox as sb
    st = RunStatus(
        id=run_id,
        phase=RunPhase(phase),
        spec_version="1.0",
        runtime=RuntimeType.docker,
        base_image="python:3.11-slim",
        exit_code=0,
        started_at=(datetime.now(timezone.utc) - timedelta(seconds=started_offset_sec)),
        finished_at=datetime.now(timezone.utc),
        message="ok",
        image_digest=image_digest,
        policy_hash="deadbeefcafebabe",
    )
    sb._service._orch._store.put_run(user_id, st)  # type: ignore[attr-defined]


def test_admin_list_filters_and_pagination(monkeypatch):


     # Override dependency for admin routes
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user
    app.dependency_overrides[get_request_user] = _admin_user_dep

    with _client(monkeypatch) as client:
        # Seed 3 runs: two with d1 digest, one with d2
        _seed_run("r1", 1, "d1", 300)
        _seed_run("r2", 1, "d1", 200)
        _seed_run("r3", 1, "d2", 100)

        # Filter by digest d1, page size 1
        r = client.get("/api/v1/sandbox/admin/runs", params={"image_digest": "d1", "limit": 1, "offset": 0})
        assert r.status_code == 200
        j = r.json()
        assert j["total"] == 2
        assert j["limit"] == 1
        assert j["offset"] == 0
        assert j["has_more"] is True
        assert j["pagination"]["total"] == 2
        assert j["pagination"]["limit"] == 1
        assert j["pagination"]["offset"] == 0
        assert j["pagination"]["has_more"] is True
        assert j["pagination"]["next_offset"] == 1
        assert len(j["items"]) == 1
        item_details = j["items"][0].get("status_reason_details")
        assert item_details["code"] == j["items"][0]["status_reason_code"]
        assert item_details["category"] == "success"

        # Next page
        r2 = client.get("/api/v1/sandbox/admin/runs", params={"image_digest": "d1", "limit": 1, "offset": 1})
        assert r2.status_code == 200
        j2 = r2.json()
        assert j2["total"] == 2
        assert j2["has_more"] is False
        assert j2["pagination"]["total"] == 2
        assert j2["pagination"]["limit"] == 1
        assert j2["pagination"]["offset"] == 1
        assert j2["pagination"]["has_more"] is False
        assert j2["pagination"]["next_offset"] is None
        assert len(j2["items"]) == 1

        # Date filter: only include recent (exclude r1 by from cutoff)
        recent_from = (datetime.now(timezone.utc) - timedelta(seconds=250)).isoformat()
        r3 = client.get("/api/v1/sandbox/admin/runs", params={"started_at_from": recent_from})
        assert r3.status_code == 200
        j3 = r3.json()
        # Should include r2 and r3 at least
        assert j3["total"] >= 2

    # Clear overrides
    app.dependency_overrides.clear()


def test_admin_list_filter_by_user_and_phase(monkeypatch):


     # Override dependency for admin routes
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user
    app.dependency_overrides[get_request_user] = _admin_user_dep

    with _client(monkeypatch) as client:
        # Seed runs for two users and phases
        _seed_run("u1_ok", 1, "d1", 300, phase="completed")
        _seed_run("u2_fail", 2, "d1", 200, phase="failed")
        _seed_run("u1_fail", 1, "d2", 100, phase="failed")

        # Filter: user_id=1 and phase=failed → expect only u1_fail
        r = client.get(
            "/api/v1/sandbox/admin/runs",
            params={"user_id": "1", "phase": "failed", "limit": 10, "offset": 0},
        )
        assert r.status_code == 200
        j = r.json()
        ids = [it["id"] for it in j.get("items", [])]
        assert "u1_fail" in ids
        assert "u2_fail" not in ids
        # Ensure totals align with filter (should be exactly 1 for this dataset)
        assert j.get("total") == 1

    app.dependency_overrides.clear()


def test_admin_list_sort_asc_desc(monkeypatch):


    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user
    app.dependency_overrides[get_request_user] = _admin_user_dep

    with _client(monkeypatch) as client:
        # Seed runs with precise ordering gaps
        _seed_run("s_desc_1", 1, "dA", 500)
        _seed_run("s_desc_2", 1, "dA", 300)
        _seed_run("s_desc_3", 1, "dA", 100)

        # Descending (default): newest first -> s_desc_3 first
        r_desc = client.get("/api/v1/sandbox/admin/runs", params={"image_digest": "dA", "limit": 3, "offset": 0, "sort": "desc"})
        assert r_desc.status_code == 200
        ids_desc = [it["id"] for it in r_desc.json().get("items", [])]
        assert ids_desc[:1] == ["s_desc_3"]

        # Ascending: oldest first -> s_desc_1 first
        r_asc = client.get("/api/v1/sandbox/admin/runs", params={"image_digest": "dA", "limit": 3, "offset": 0, "sort": "asc"})
        assert r_asc.status_code == 200
        ids_asc = [it["id"] for it in r_asc.json().get("items", [])]
        assert ids_asc[:1] == ["s_desc_1"]

    app.dependency_overrides.clear()
