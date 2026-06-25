import hashlib
import os
import tempfile
from collections.abc import AsyncGenerator
from datetime import datetime

from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.core.AuthNZ.permissions import CLAIMS_ADMIN
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user
from tldw_Server_API.app.core.Claims_Extraction import claims_service
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


def test_claims_dashboard_analytics_and_export(monkeypatch):


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

    monkeypatch.setattr(
        claims_service.claims_jobs,
        "claims_jobs_summary",
        lambda **_kwargs: {"domain": "claims", "counts": {"queued": 2, "running": 1}},
    )

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
            assert "claims_jobs" in data
            assert "pause" not in data["claims_jobs"]
            assert "drain" not in data["claims_jobs"]
            assert "requeue" not in data["claims_jobs"]

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


def test_claims_analytics_scope_aggregate_widgets_to_owner(tmp_path):
    db_path = str(tmp_path / "claims-owner-analytics.db")
    db = MediaDatabase(db_path=db_path, client_id="1")
    db.initialize_db()
    owner_one_content = "Owner one alpha. Owner one beta."
    owner_two_content = "Owner two alpha. Owner two beta. Owner two gamma. Owner two delta. Owner two epsilon."
    owner_one_media_id, _, _ = db.add_media_with_keywords(
        title="Owner One",
        media_type="text",
        content=owner_one_content,
        keywords=None,
        owner_user_id=1,
    )
    owner_two_media_id, _, _ = db.add_media_with_keywords(
        title="Owner Two",
        media_type="text",
        content=owner_two_content,
        keywords=None,
        owner_user_id=2,
    )
    owner_one_hash = hashlib.sha256(owner_one_content.encode()).hexdigest()
    owner_two_hash = hashlib.sha256(owner_two_content.encode()).hexdigest()
    db.upsert_claims(
        [
            {
                "media_id": owner_one_media_id,
                "chunk_index": 0,
                "span_start": None,
                "span_end": None,
                "claim_text": "Owner one alpha.",
                "confidence": 0.9,
                "extractor": "heuristic",
                "extractor_version": "v1",
                "chunk_hash": owner_one_hash,
            },
            {
                "media_id": owner_one_media_id,
                "chunk_index": 0,
                "span_start": None,
                "span_end": None,
                "claim_text": "Owner one beta.",
                "confidence": 0.9,
                "extractor": "heuristic",
                "extractor_version": "v1",
                "chunk_hash": owner_one_hash,
            },
            *[
                {
                    "media_id": owner_two_media_id,
                    "chunk_index": 0,
                    "span_start": None,
                    "span_end": None,
                    "claim_text": f"Owner two claim {idx}.",
                    "confidence": 0.7,
                    "extractor": "heuristic",
                    "extractor_version": "v1",
                    "chunk_hash": owner_two_hash,
                }
                for idx in range(5)
            ],
        ]
    )
    owner_one_claim_ids = [
        int(row["id"])
        for row in db.execute_query(
            "SELECT id FROM Claims WHERE media_id = ? AND deleted = 0 ORDER BY id ASC",
            (owner_one_media_id,),
        ).fetchall()
    ]
    owner_two_claim_ids = [
        int(row["id"])
        for row in db.execute_query(
            "SELECT id FROM Claims WHERE media_id = ? AND deleted = 0 ORDER BY id ASC",
            (owner_two_media_id,),
        ).fetchall()
    ]
    db.update_claim_review(owner_one_claim_ids[0], review_status="approved", reviewer_id=1)
    db.update_claim_review(owner_one_claim_ids[1], review_status="flagged", reviewer_id=1)
    for claim_id in owner_two_claim_ids:
        db.update_claim_review(claim_id, review_status="rejected", reviewer_id=2)

    try:
        analytics = claims_service._build_claims_analytics(db, owner_user_id="1", window_days=1)
    finally:
        db.close_connection()

    assert analytics["total_claims"] == 2
    assert analytics["status_counts"] == {"approved": 1, "flagged": 1}
    assert analytics["review_throughput"]["total"] == 2
    assert analytics["review_status_trends"]["daily"][-1]["total"] == 2
    assert analytics["claims_per_media_top"] == [{"media_id": owner_one_media_id, "count": 2}]
    assert analytics["claims_per_media_stats"] == {"mean": 2.0, "p95": 2, "max": 2}
    assert analytics["clusters"]["orphan_claims"] == 2


def test_evaluate_claims_alerts_enqueues_jobs_when_enabled(monkeypatch):
    enqueued: list[dict[str, object]] = []

    class _Db:
        db_path_str = "/tmp/user-1/Media_DB_v2.db"

        def migrate_legacy_claims_monitoring_alerts(self, _user_id: str) -> None:
            return None

        def list_claims_monitoring_alerts(self, _user_id: str) -> list[dict[str, object]]:
            return [
                {
                    "id": 7,
                    "name": "Unsupported ratio",
                    "alert_type": "threshold_breach",
                    "enabled": True,
                    "threshold_ratio": 0.2,
                    "baseline_ratio": None,
                    "channels": {"slack": True, "webhook": True, "email": True},
                    "slack_webhook_url": "https://example.test/slack",
                    "webhook_url": "https://example.test/webhook",
                    "email_recipients": "ops@example.test",
                }
            ]

        def get_claims_monitoring_settings(self, _user_id: str) -> dict[str, object]:
            return {"enabled": True}

        def insert_claims_monitoring_event(self, **kwargs):
            assert kwargs["user_id"] == "1"
            assert kwargs["event_type"] == "unsupported_ratio"
            return {"id": 55, **kwargs}

    monkeypatch.setattr(claims_service.claims_jobs, "claims_jobs_enabled", lambda: True)
    monkeypatch.setattr(claims_service, "settings", {"CLAIMS_MONITORING_ENABLED": True})
    monkeypatch.setattr(
        claims_service.claims_jobs,
        "enqueue_claims_alert_delivery",
        lambda **kwargs: enqueued.append(kwargs) or {"id": len(enqueued) + 1},
    )
    monkeypatch.setattr(
        claims_service,
        "_dispatch_claims_alert_notifications",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("legacy dispatch should not run")),
    )
    monkeypatch.setattr(
        claims_service,
        "_compute_unsupported_ratios",
        lambda _window_sec, _baseline_sec: {"window_ratio": 0.5, "baseline_ratio": 0.1},
    )

    result = claims_service._evaluate_claims_alerts_for_user(
        target_user_id="1",
        db=_Db(),
        window_sec=3600,
        baseline_sec=86400,
    )

    assert result["results"][0]["triggered"] is True
    assert enqueued == [
        {"owner_user_id": "1", "event_id": 55, "alert_id": 7, "channel": "slack"},
        {"owner_user_id": "1", "event_id": 55, "alert_id": 7, "channel": "webhook"},
    ]
