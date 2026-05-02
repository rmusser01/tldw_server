import hashlib
import os
import tempfile
from datetime import datetime
from typing import AsyncGenerator

from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal, AuthContext
from tldw_Server_API.app.core.AuthNZ.permissions import CLAIMS_ADMIN
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase


def _principal_override_admin():


    async def _override(request=None):
        principal = AuthPrincipal(
            kind="user",
            user_id=1,
            api_key_id=None,
            subject="admin",
            token_type="access",
            jti=None,
            roles=["admin"],
            permissions=[CLAIMS_ADMIN],
            is_admin=True,
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


def _seed_dashboard_db() -> str:


    tmpdir = tempfile.mkdtemp(prefix="claims_dashboard_")
    db_path = os.path.join(tmpdir, "media.db")
    db = MediaDatabase(db_path=db_path, client_id="1")
    db.initialize_db()
    content = "A. B. A."
    media_id, _, _ = db.add_media_with_keywords(title="Doc", media_type="text", content=content, keywords=None)
    chunk_hash = hashlib.sha256(content.encode()).hexdigest()
    db.upsert_claims(
        [
            {
                "media_id": media_id,
                "chunk_index": 0,
                "span_start": None,
                "span_end": None,
                "claim_text": "A.",
                "confidence": 0.9,
                "extractor": "heuristic",
                "extractor_version": "v1",
                "chunk_hash": chunk_hash,
            },
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
            },
            {
                "media_id": media_id,
                "chunk_index": 0,
                "span_start": None,
                "span_end": None,
                "claim_text": "B.",
                "confidence": 0.7,
                "extractor": "heuristic",
                "extractor_version": "v1",
                "chunk_hash": chunk_hash,
            },
        ]
    )
    rows = db.execute_query(
        "SELECT id FROM Claims WHERE media_id = ? AND deleted = 0 ORDER BY id ASC",
        (media_id,),
    ).fetchall()
    claim_ids = [int(r["id"]) if isinstance(r, dict) else int(r[0]) for r in rows]
    db.update_claim_review(
        claim_ids[0],
        review_status="approved",
        reviewer_id=1,
        review_notes="ok",
    )
    db.update_claim_review(
        claim_ids[1],
        review_status="flagged",
        reviewer_id=1,
        review_notes="needs check",
    )
    db.insert_claims_monitoring_event(
        user_id="1",
        event_type="unsupported_ratio",
        severity="warning",
        payload_json='{"provider": "test", "model": "mock"}',
    )
    db.rebuild_claim_clusters_exact(user_id="1", min_size=2)
    db.close_connection()
    return db_path


def test_claims_dashboard_analytics_and_export():


    from tldw_Server_API.app.main import app as fastapi_app

    class _User:
        def __init__(self) -> None:
            self.id = 1
            self.username = "admin"
            self.is_admin = True

    async def _override_user():
        return _User()

    db_path = _seed_dashboard_db()

    async def _override_db() -> AsyncGenerator[MediaDatabase, None]:
        override_db = MediaDatabase(db_path=db_path, client_id="1")
        try:
            yield override_db
        finally:
            try:
                override_db.close_connection()
            except Exception:
                _ = None

    fastapi_app.dependency_overrides[get_auth_principal] = _principal_override_admin()
    fastapi_app.dependency_overrides[get_request_user] = _override_user
    fastapi_app.dependency_overrides[get_media_db_for_user] = _override_db

    try:
        with TestClient(fastapi_app) as client:
            r = client.get("/api/v1/claims/analytics/dashboard")
            assert r.status_code == 200, r.text
            data = r.json()
            assert data["total_claims"] == 3
            assert "clusters" in data
            assert data["clusters"]["total_clusters"] >= 1
            assert "hotspots" in data["clusters"]
            assert isinstance(data["clusters"]["hotspots"], list)
            assert "review_throughput" in data
            today = datetime.utcnow().date().isoformat()
            daily_counts = {item["date"]: item["count"] for item in data["review_throughput"]["daily"]}
            assert daily_counts.get(today, 0) >= 2
            assert "review_status_trends" in data
            trend = data["review_status_trends"]
            trend_daily = {item["date"]: item for item in trend.get("daily", [])}
            today_trend = trend_daily.get(today)
            assert today_trend is not None
            assert today_trend["total"] >= 2
            assert "unsupported_ratios" in data
            assert "provider_usage" in data
            assert isinstance(data["provider_usage"], list)
            rebuild = data.get("rebuild_health")
            assert rebuild is None or rebuild.get("status") == "ok"

            r2 = client.post(
                "/api/v1/claims/analytics/export",
                json={
                    "format": "json",
                    "filters": {"event_type": "unsupported_ratio"},
                    "pagination": {"limit": 10, "offset": 0},
                },
            )
            assert r2.status_code == 200, r2.text
            export_meta = r2.json()
            assert export_meta["status"] == "ready"
            download_url = export_meta.get("download_url")
            assert download_url

            r3 = client.get(download_url)
            assert r3.status_code == 200, r3.text
            export_payload = r3.json()
            events = export_payload.get("events") or []
            assert events
            assert events[0]["event_type"] == "unsupported_ratio"

            r4 = client.get("/api/v1/claims/analytics/exports?limit=10&offset=0")
            assert r4.status_code == 200, r4.text
            list_payload = r4.json()
            export_ids = [item["export_id"] for item in list_payload.get("exports", [])]
            assert export_meta["export_id"] in export_ids
            assert list_payload["has_more"] is False
            assert list_payload["next_offset"] is None
            assert list_payload["pagination"] == {
                "mode": "offset",
                "limit": 10,
                "offset": 0,
                "total": list_payload["total"],
                "has_more": False,
                "next_offset": None,
            }
    finally:
        fastapi_app.dependency_overrides.pop(get_auth_principal, None)
        fastapi_app.dependency_overrides.pop(get_request_user, None)
        fastapi_app.dependency_overrides.pop(get_media_db_for_user, None)
