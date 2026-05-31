import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from tldw_Server_API.app.main import app as fastapi_app
from tldw_Server_API.app.api.v1.endpoints import personalization as personalization_ep
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.api.v1.API_Deps.personalization_deps import get_personalization_db_for_user
from tldw_Server_API.app.core.DB_Management.Personalization_DB import PersonalizationDB


pytestmark = pytest.mark.unit


class _LoggerStub:
    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.warnings.append(message.format(*args) if args else message)

    def info(self, *args: Any, **kwargs: Any) -> None:
        return


class _UsageLogStub:
    user_id = "1"

    def log_event(self, *args: Any, **kwargs: Any) -> None:
        return


class _ProfileCountsDB:
    def topic_counts(self, _user_id: str) -> int:
        return 0

    def memory_counts(self, _user_id: str) -> int:
        return 0

    def session_count(self, _user_id: str) -> int:
        return 0


@pytest.fixture()
def client_with_personalization_db(tmp_path):
    db_path = tmp_path / "personalization.db"
    db = PersonalizationDB(str(db_path))

    async def override_user():
        return User(id=1, username="tester", email=None, is_active=True)

    def override_db_dep():

        return db

    fastapi_app.dependency_overrides[get_request_user] = override_user
    fastapi_app.dependency_overrides[get_personalization_db_for_user] = override_db_dep

    with TestClient(fastapi_app) as client:
        yield client

    fastapi_app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_personalization_opt_in_failure_log_is_sanitized(monkeypatch):
    class _FailingDB:
        def update_profile(self, **kwargs: Any):
            raise RuntimeError("personalization backend exploded at /private/personalization.db")

    logger = _LoggerStub()
    monkeypatch.setattr(personalization_ep, "logger", logger)
    monkeypatch.setattr(personalization_ep, "is_personalization_enabled", lambda: True)

    with pytest.raises(HTTPException) as exc_info:
        await personalization_ep.personalization_opt_in(
            payload=personalization_ep.OptInRequest(enabled=True),
            db=_FailingDB(),
            log=_UsageLogStub(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to update personalization profile"
    assert logger.warnings == ["Opt-in failed"]
    logged = "\n".join(logger.warnings)
    assert "personalization backend exploded" not in logged
    assert "/private/personalization.db" not in logged


@pytest.mark.asyncio
async def test_personalization_purge_failure_log_is_sanitized(monkeypatch):
    class _FailingDB:
        def purge_user(self, user_id: str):
            raise RuntimeError("personalization backend exploded at /private/personalization.db")

    logger = _LoggerStub()
    monkeypatch.setattr(personalization_ep, "logger", logger)
    monkeypatch.setattr(personalization_ep, "is_personalization_enabled", lambda: True)

    with pytest.raises(HTTPException) as exc_info:
        await personalization_ep.personalization_purge(
            db=_FailingDB(),
            log=_UsageLogStub(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to purge personalization data"
    assert logger.warnings == ["Purge failed"]
    logged = "\n".join(logger.warnings)
    assert "personalization backend exploded" not in logged
    assert "/private/personalization.db" not in logged


def test_profile_from_dict_invalid_updated_at_log_is_sanitized(monkeypatch):
    logger = _LoggerStub()
    monkeypatch.setattr(personalization_ep, "logger", logger)

    profile = personalization_ep._profile_from_dict(
        {
            "enabled": 1,
            "updated_at": {"raw": "/private/personalization-profile.json"},
        },
        _ProfileCountsDB(),
        "private-user",
    )

    assert profile.enabled is True
    assert logger.warnings == ["Profile updated_at missing or unparseable"]
    logged = "\n".join(logger.warnings)
    assert "private-user" not in logged
    assert "/private/personalization-profile.json" not in logged


def test_profile_roundtrip(client_with_personalization_db: TestClient):
    c = client_with_personalization_db
    # Get default profile
    r = c.get("/api/v1/personalization/profile")
    assert r.status_code == 200
    prof = r.json()
    assert "enabled" in prof
    # Verify new fields present
    assert "session_count" in prof
    assert "proactive_enabled" in prof
    assert "response_style" in prof
    assert "preferred_format" in prof

    # Opt in
    r2 = c.post("/api/v1/personalization/opt-in", json={"enabled": True})
    assert r2.status_code == 200
    prof2 = r2.json()
    assert prof2.get("enabled") is True

    # Update preferences - alpha=0.3 with existing beta=0.6 + gamma=0.2 sums to 1.1
    # Normalization kicks in: all three are scaled down proportionally
    r3 = c.post("/api/v1/personalization/preferences", json={"alpha": 0.3})
    assert r3.status_code == 200
    prof3 = r3.json()
    # After normalization: total should be <= 1.0
    total = prof3["alpha"] + prof3["beta"] + prof3["gamma"]
    assert total <= 1.0 + 1e-6
    # Alpha should be proportionally correct (0.3/1.1 ≈ 0.2727)
    assert prof3["alpha"] > 0.0


def test_explanations_response_includes_canonical_pagination(
    client_with_personalization_db: TestClient,
) -> None:
    """Explanation listing returns canonical offset pagination for the scaffolded collection."""
    response = client_with_personalization_db.get(
        "/api/v1/personalization/explanations",
        params={"limit": 7, "offset": 3},
    )

    assert response.status_code == 200
    assert response.json() == {
        "items": [],
        "total": 0,
        "pagination": {
            "mode": "offset",
            "limit": 7,
            "offset": 3,
            "total": 0,
            "has_more": False,
            "next_offset": None,
        },
    }


def test_memories_crud(client_with_personalization_db: TestClient):
    c = client_with_personalization_db
    # Add - no id field required (MemoryCreate schema)
    add = c.post(
        "/api/v1/personalization/memories",
        json={"type": "semantic", "content": "Remember this", "pinned": False},
    )
    assert add.status_code == 201
    mid = add.json()["id"]
    assert mid  # server-generated

    # Get specific memory
    get_r = c.get(f"/api/v1/personalization/memories/{mid}")
    assert get_r.status_code == 200
    assert get_r.json()["content"] == "Remember this"

    # List
    lst = c.get("/api/v1/personalization/memories")
    assert lst.status_code == 200
    data = lst.json()
    assert data["total"] >= 1
    assert data["pagination"] == {
        "mode": "page",
        "page": 1,
        "per_page": 50,
        "total": data["total"],
        "total_pages": 1,
        "has_more": False,
    }

    # Patch (update)
    patch_r = c.patch(
        f"/api/v1/personalization/memories/{mid}",
        json={"pinned": True, "content": "Updated memory"},
    )
    assert patch_r.status_code == 200
    assert patch_r.json()["pinned"] is True
    assert patch_r.json()["content"] == "Updated memory"

    # Delete
    dl = c.delete(f"/api/v1/personalization/memories/{mid}")
    assert dl.status_code == 200
    assert "deleted" in dl.json()["detail"]


def test_delete_memory_wrong_user(tmp_path):
    """Issue #1: delete_memory must verify user_id ownership."""
    db_path = tmp_path / "pdb.db"
    db = PersonalizationDB(str(db_path))
    from tldw_Server_API.app.core.DB_Management.Personalization_DB import SemanticMemory

    # Ensure profile exists for user_a before inserting memory (FK constraint)
    db.get_or_create_profile("user_a")
    db.get_or_create_profile("user_b")

    mem = SemanticMemory(user_id="user_a", content="Secret memory")
    mid = db.add_semantic_memory(mem)

    # user_b tries to delete user_a's memory
    assert db.delete_memory(mid, "user_b") is False
    # user_a can delete their own
    assert db.delete_memory(mid, "user_a") is True


def test_weight_clamping(client_with_personalization_db: TestClient):
    """Issue #8: weights should be clamped to [0,1] and normalized."""
    c = client_with_personalization_db

    # Negative value clamped to 0
    r = c.post("/api/v1/personalization/preferences", json={"alpha": -5.0})
    assert r.status_code == 200
    assert r.json()["alpha"] >= 0.0

    # Value > 1 clamped to 1
    r2 = c.post("/api/v1/personalization/preferences", json={"beta": 999.0})
    assert r2.status_code == 200
    assert r2.json()["beta"] <= 1.0

    # Sum > 1 gets normalized
    r3 = c.post(
        "/api/v1/personalization/preferences",
        json={"alpha": 0.8, "beta": 0.8, "gamma": 0.8},
    )
    assert r3.status_code == 200
    prof = r3.json()
    total = prof["alpha"] + prof["beta"] + prof["gamma"]
    assert total <= 1.0 + 1e-6


def test_purge_response_structured(client_with_personalization_db: TestClient):
    """Issue #7: purge response should be structured JSON, not a string."""
    c = client_with_personalization_db
    # Add a memory first
    c.post(
        "/api/v1/personalization/memories",
        json={"content": "to be purged"},
    )
    r = c.post("/api/v1/personalization/purge")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert isinstance(body["deleted_counts"], dict)
    assert body["enabled"] is False
    assert "purged_at" in body


def test_updated_at_from_db(client_with_personalization_db: TestClient):
    """Issue #4: updated_at should reflect DB value, not request time."""
    c = client_with_personalization_db
    # Opt in (sets updated_at in DB)
    r1 = c.post("/api/v1/personalization/opt-in", json={"enabled": True})
    t1 = r1.json()["updated_at"]

    # Get profile
    r2 = c.get("/api/v1/personalization/profile")
    t2 = r2.json()["updated_at"]

    # Both should be the same DB timestamp (not a new utcnow())
    assert t1 == t2


def test_post_memory_no_id_required(client_with_personalization_db: TestClient):
    """Issue #5: POST /memories should not require client to send 'id'."""
    c = client_with_personalization_db
    # Send without 'id' field
    r = c.post(
        "/api/v1/personalization/memories",
        json={"content": "No ID provided"},
    )
    assert r.status_code == 201
    assert r.json()["id"]  # Server generated


def test_validate_memories(client_with_personalization_db: TestClient):
    """Issue #13: POST /memories/validate should mark memories as validated."""
    c = client_with_personalization_db
    add = c.post(
        "/api/v1/personalization/memories",
        json={"content": "validate me"},
    )
    mid = add.json()["id"]

    r = c.post(
        "/api/v1/personalization/memories/validate",
        json={"memory_ids": [mid]},
    )
    assert r.status_code == 200
    assert "validated" in r.json()["detail"]


def test_import_export_memories(client_with_personalization_db: TestClient):
    """Issue #13: import/export endpoints."""
    c = client_with_personalization_db
    # Import
    r = c.post(
        "/api/v1/personalization/memories/import",
        json={"memories": [{"content": "imported 1"}, {"content": "imported 2", "pinned": True}]},
    )
    assert r.status_code == 201
    assert "imported 2" in r.json()["detail"]

    # Export
    r2 = c.get("/api/v1/personalization/memories/export")
    assert r2.status_code == 200
    body = r2.json()
    assert body["total"] >= 2
    assert len(body["memories"]) >= 2


def test_hidden_memory_filtering(client_with_personalization_db: TestClient):
    """Ensure hidden memories are excluded by default."""
    c = client_with_personalization_db
    add = c.post(
        "/api/v1/personalization/memories",
        json={"content": "visible"},
    )
    mid = add.json()["id"]

    # Mark as hidden
    c.patch(f"/api/v1/personalization/memories/{mid}", json={"hidden": True})

    # Default list should not include hidden
    lst = c.get("/api/v1/personalization/memories")
    assert lst.json()["total"] == 0

    # With include_hidden
    lst2 = c.get("/api/v1/personalization/memories?include_hidden=true")
    assert lst2.json()["total"] == 1


def test_preferences_new_fields(client_with_personalization_db: TestClient):
    """Issue #15: preferences should accept new Stage 2 fields."""
    c = client_with_personalization_db
    r = c.post(
        "/api/v1/personalization/preferences",
        json={
            "response_style": "concise",
            "preferred_format": "markdown",
            "proactive_enabled": False,
        },
    )
    assert r.status_code == 200
    prof = r.json()
    assert prof["response_style"] == "concise"
    assert prof["preferred_format"] == "markdown"
    assert prof["proactive_enabled"] is False


def test_recency_half_life_clamped(client_with_personalization_db: TestClient):
    """Issue #8: recency_half_life_days clamped to [1, 365]."""
    c = client_with_personalization_db
    r = c.post("/api/v1/personalization/preferences", json={"recency_half_life_days": 0})
    assert r.status_code == 200
    assert r.json()["recency_half_life_days"] >= 1

    r2 = c.post("/api/v1/personalization/preferences", json={"recency_half_life_days": 9999})
    assert r2.status_code == 200
    assert r2.json()["recency_half_life_days"] <= 365


def test_purged_at_set_on_purge_cleared_on_optin(tmp_path):
    """Issue #6: purged_at should be set on purge and cleared on opt-in."""
    db = PersonalizationDB(str(tmp_path / "purge.db"))
    db.get_or_create_profile("u1")

    # purge sets purged_at
    db.purge_user("u1")
    prof = db.get_or_create_profile("u1")
    assert prof.get("purged_at") is not None

    # Opt back in clears purged_at
    db.update_profile("u1", enabled=1)
    prof2 = db.get_or_create_profile("u1")
    assert prof2.get("purged_at") is None


def test_db_list_recent_events_thread_safe(tmp_path):
    """Issue #2: list_recent_events should use the lock (public method)."""
    from tldw_Server_API.app.core.DB_Management.Personalization_DB import UsageEvent

    db = PersonalizationDB(str(tmp_path / "events.db"))
    db.get_or_create_profile("u1")

    db.insert_usage_event(UsageEvent(user_id="u1", type="search", tags=["python"]))
    db.insert_usage_event(UsageEvent(user_id="u1", type="view", tags=["rust"]))

    events = db.list_recent_events("u1", limit=10)
    assert len(events) == 2
    assert all("type" in e for e in events)


def test_consolidation_non_numeric_user_id():
    """Issue #3: consolidation should handle non-numeric user IDs."""
    from tldw_Server_API.app.services.personalization_consolidation import _resolve_user_id_to_int

    # Numeric works
    assert _resolve_user_id_to_int("42") == 42
    # Non-numeric hashes to int without error
    result = _resolve_user_id_to_int("test_user_abc")
    assert isinstance(result, int)
    assert result >= 0


def test_enumerate_user_ids_with_dirs(tmp_path, monkeypatch):
    """_enumerate_user_ids scans user_databases/ for int-named subdirs."""
    from tldw_Server_API.app.services.personalization_consolidation import PersonalizationConsolidationService
    from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths

    # Create mock user directories
    (tmp_path / "3").mkdir()
    (tmp_path / "1").mkdir()
    (tmp_path / "2").mkdir()
    (tmp_path / "not_a_number").mkdir()  # should be skipped

    monkeypatch.setattr(DatabasePaths, "get_user_db_base_dir", staticmethod(lambda: tmp_path))

    result = PersonalizationConsolidationService._enumerate_user_ids()
    assert result == [1, 2, 3]  # sorted, unique ints


def test_enumerate_user_ids_fallback(tmp_path, monkeypatch):
    """When no user dirs exist, falls back to get_single_user_id()."""
    from tldw_Server_API.app.services.personalization_consolidation import PersonalizationConsolidationService
    from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths

    # Empty directory — no user subdirs
    monkeypatch.setattr(DatabasePaths, "get_user_db_base_dir", staticmethod(lambda: tmp_path))
    monkeypatch.setattr(DatabasePaths, "get_single_user_id", staticmethod(lambda: 1))

    result = PersonalizationConsolidationService._enumerate_user_ids()
    assert result == [1]


def test_enumerate_user_ids_base_dir_error(monkeypatch):
    """When get_user_db_base_dir raises, returns empty list."""
    from tldw_Server_API.app.services.personalization_consolidation import PersonalizationConsolidationService
    from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths

    def _boom():
        raise RuntimeError("no base dir")

    monkeypatch.setattr(DatabasePaths, "get_user_db_base_dir", staticmethod(_boom))

    result = PersonalizationConsolidationService._enumerate_user_ids()
    assert result == []


def test_get_status_includes_user_count():
    """get_status() should include user_count field."""
    from tldw_Server_API.app.services.personalization_consolidation import PersonalizationConsolidationService

    svc = PersonalizationConsolidationService()
    status = svc.get_status()
    assert "user_count" in status
    assert status["user_count"] == 0

    # Simulate a tick for a user
    svc._last_tick["1"] = "2025-01-01T00:00:00+00:00"
    status2 = svc.get_status()
    assert status2["user_count"] == 1
