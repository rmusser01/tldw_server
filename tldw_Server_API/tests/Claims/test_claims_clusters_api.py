import hashlib
import os
import tempfile
from typing import AsyncGenerator

from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal, AuthContext
from tldw_Server_API.app.core.AuthNZ.permissions import CLAIMS_REVIEW
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase


def _seed_cluster_db() -> tuple[str, int]:


    tmpdir = tempfile.mkdtemp(prefix="claims_clusters_")
    db_path = os.path.join(tmpdir, "media.db")
    db = MediaDatabase(db_path=db_path, client_id="1")
    db.initialize_db()
    content = "A. B. C."
    media_id, _, _ = db.add_media_with_keywords(title="Doc", media_type="text", content=content, keywords=None)
    chunk_hash = hashlib.sha256(content.encode()).hexdigest()
    db.upsert_claims([
        {
            "media_id": media_id,
            "chunk_index": 0,
            "span_start": None,
            "span_end": None,
            "claim_text": "A.",
            "confidence": 0.8,
            "extractor": "heuristic",
            "extractor_version": "v1",
            "chunk_hash": chunk_hash,
        }
    ])
    row = db.execute_query("SELECT id FROM Claims WHERE media_id = ? AND deleted = 0", (media_id,)).fetchone()
    claim_id = int(row["id"]) if isinstance(row, dict) else int(row[0])
    cluster = db.create_claim_cluster(user_id="1", canonical_claim_text="A.", representative_claim_id=claim_id)
    cluster_id = int(cluster.get("id"))
    db.add_claim_to_cluster(cluster_id=cluster_id, claim_id=claim_id, similarity_score=1.0)
    db.close_connection()
    return db_path, cluster_id


def _principal_override():


    async def _override(request=None):
        principal = AuthPrincipal(
            kind="user",
            user_id=1,
            api_key_id=None,
            subject="reviewer",
            token_type="access",
            jti=None,
            roles=["reviewer"],
            permissions=[CLAIMS_REVIEW],
            is_admin=False,
            org_ids=[],
            team_ids=[],
        )
        if request is not None:
            try:
                request.state.auth = AuthContext(
                    principal=principal,
                    ip=None,
                    user_agent=None,
                    request_id=None,
                )
            except Exception:
                _ = None
        return principal

    return _override


def test_claims_clusters_endpoints():

    from tldw_Server_API.app.main import app as fastapi_app

    class _User:
        def __init__(self) -> None:
            self.id = 1
            self.username = "reviewer"
            self.is_admin = False

    async def _override_user():
        return _User()

    db_path, cluster_id = _seed_cluster_db()

    async def _override_db() -> AsyncGenerator[MediaDatabase, None]:
        override_db = MediaDatabase(db_path=db_path, client_id="1")
        try:
            yield override_db
        finally:
            try:
                override_db.close_connection()
            except Exception:
                _ = None

    fastapi_app.dependency_overrides[get_auth_principal] = _principal_override()
    fastapi_app.dependency_overrides[get_request_user] = _override_user
    fastapi_app.dependency_overrides[get_media_db_for_user] = _override_db

    try:
        with TestClient(fastapi_app) as client:
            r = client.get("/api/v1/claims/clusters")
            assert r.status_code == 200, r.text
            clusters = r.json()
            assert any(int(item["id"]) == cluster_id for item in clusters)

            r_kw = client.get("/api/v1/claims/clusters?keyword=A")
            assert r_kw.status_code == 200, r_kw.text
            assert any(int(item["id"]) == cluster_id for item in r_kw.json())

            r_min = client.get("/api/v1/claims/clusters?min_size=2")
            assert r_min.status_code == 200, r_min.text
            assert all(int(item["id"]) != cluster_id for item in r_min.json())

            r_watch = client.get("/api/v1/claims/clusters?watchlisted=false")
            assert r_watch.status_code == 200, r_watch.text
            assert any(int(item["id"]) == cluster_id for item in r_watch.json())

            r_since = client.get("/api/v1/claims/clusters?since=2100-01-01")
            assert r_since.status_code == 200, r_since.text
            assert all(int(item["id"]) != cluster_id for item in r_since.json())

            r2 = client.get(f"/api/v1/claims/clusters/{cluster_id}")
            assert r2.status_code == 200, r2.text
            assert int(r2.json()["id"]) == cluster_id

            r3 = client.get(f"/api/v1/claims/clusters/{cluster_id}/members")
            assert r3.status_code == 200, r3.text
            assert len(r3.json()) >= 1

            r4 = client.get(f"/api/v1/claims/clusters/{cluster_id}/timeline")
            assert r4.status_code == 200, r4.text
            assert "timeline" in r4.json()

            r5 = client.get(f"/api/v1/claims/clusters/{cluster_id}/evidence")
            assert r5.status_code == 200, r5.text
            assert "counts" in r5.json()
    finally:
        fastapi_app.dependency_overrides.pop(get_auth_principal, None)
        fastapi_app.dependency_overrides.pop(get_request_user, None)
        fastapi_app.dependency_overrides.pop(get_media_db_for_user, None)


def test_default_cluster_rebuild_preserves_real_legacy_result(monkeypatch):
    from fastapi import FastAPI

    from tldw_Server_API.app.api.v1.API_Deps import DB_Deps
    from tldw_Server_API.app.api.v1.endpoints import claims
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
    from tldw_Server_API.app.core.Claims_Extraction import claims_service

    monkeypatch.delenv("CLAIMS_JOBS_ENABLED", raising=False)
    monkeypatch.delenv("CLAIMS_CLUSTER_REBUILD_JOBS_ENABLED", raising=False)
    path, _ = _seed_cluster_db()
    db = MediaDatabase(db_path=path, client_id="1")
    monkeypatch.setattr(claims_service, "_get_watchlists_db", lambda user: None)
    monkeypatch.setattr(DB_Deps, "_resolve_media_db_for_user", lambda user: db)
    app = FastAPI()
    app.include_router(claims.router, prefix="/api/v1")
    app.dependency_overrides[get_request_user] = lambda: User(
        id=1, username="cluster-admin", roles=["user"], permissions=["claims.admin"]
    )
    try:
        with TestClient(app) as client:
            response = client.post("/api/v1/claims/clusters/rebuild", params={"method": "exact", "min_size": 1})
        assert response.status_code == 200, response.text
        assert response.json() == {
            "clusters_created": 1,
            "claims_assigned": 1,
            "watchlist_notifications": {"status": "skipped", "reason": "watchlists_unavailable"},
        }
        assert "location" not in response.headers
    finally:
        db.close_connection()
