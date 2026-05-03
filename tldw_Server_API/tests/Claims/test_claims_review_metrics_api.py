import hashlib
import os
import tempfile

from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user
from tldw_Server_API.app.core.AuthNZ.permissions import CLAIMS_ADMIN
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal
from tldw_Server_API.app.core.Claims_Extraction import claims_service
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase


def _seed_review_metrics_db() -> str:


    tmpdir = tempfile.mkdtemp(prefix="claims_review_metrics_api_")
    db_path = os.path.join(tmpdir, "media.db")
    db = MediaDatabase(db_path=db_path, client_id="1")
    db.initialize_db()
    content = "A. B."
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
        },
        {
            "media_id": media_id,
            "chunk_index": 1,
            "span_start": None,
            "span_end": None,
            "claim_text": "B.",
            "confidence": 0.7,
            "extractor": "llm",
            "extractor_version": "v2",
            "chunk_hash": chunk_hash,
        },
    ])
    rows = db.execute_query(
        "SELECT id FROM Claims WHERE media_id = ? ORDER BY id ASC",
        (media_id,),
    ).fetchall()
    claim_ids = [int(r["id"]) if isinstance(r, dict) else int(r[0]) for r in rows]

    db.update_claim_review(
        claim_ids[0],
        review_status="approved",
        reviewer_id=1,
        review_notes="ok",
        review_reason_code="typo",
        corrected_text="A1.",
    )
    db.update_claim_review(
        claim_ids[1],
        review_status="rejected",
        reviewer_id=1,
        review_notes="no",
        review_reason_code="spam",
    )

    log_rows = db.execute_query(
        "SELECT id FROM claims_review_log ORDER BY id ASC"
    ).fetchall()
    for row in log_rows:
        log_id = int(row["id"]) if isinstance(row, dict) else int(row[0])
        db.execute_query(
            "UPDATE claims_review_log SET created_at = ? WHERE id = ?",
            ("2024-01-10 01:00:00", log_id),
            commit=True,
        )

    claims_service.aggregate_claims_review_extractor_metrics_daily(
        db=db,
        target_user_id="1",
        report_date="2024-01-10",
    )
    db.close_connection()
    return db_path


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


def test_claims_review_metrics_endpoint():


    from tldw_Server_API.app.main import app as fastapi_app

    class _User:
        def __init__(self) -> None:
            self.id = 1
            self.username = "admin"
            self.is_admin = True

    async def _override_user():
        return _User()

    db_path = _seed_review_metrics_db()

    async def _override_db():
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
            r = client.get(
                "/api/v1/claims/review/metrics?start_date=2024-01-10&end_date=2024-01-10"
            )
            assert r.status_code == 200, r.text
            payload = r.json()
            assert payload["total"] == 2
            items = payload.get("items") or []
            metrics = {(item["extractor"], item["extractor_version"]): item for item in items}
            heuristic = metrics.get(("heuristic", "v1"))
            assert heuristic
            assert heuristic["approved_count"] == 1
            assert heuristic["edited_count"] == 1
            assert heuristic["reason_code_counts"]["typo"] == 1
            llm_metrics = metrics.get(("llm", "v2"))
            assert llm_metrics
            assert llm_metrics["rejected_count"] == 1
            assert llm_metrics["reason_code_counts"]["spam"] == 1
    finally:
        fastapi_app.dependency_overrides.pop(get_auth_principal, None)
        fastapi_app.dependency_overrides.pop(get_request_user, None)
        fastapi_app.dependency_overrides.pop(get_media_db_for_user, None)


def test_claims_review_metrics_endpoint_includes_canonical_pagination():
    """Review metrics pagination should expose the full filtered total."""

    from tldw_Server_API.app.main import app as fastapi_app

    class _User:
        def __init__(self) -> None:
            self.id = 1
            self.username = "admin"
            self.is_admin = True

    async def _override_user():
        return _User()

    db_path = _seed_review_metrics_db()

    async def _override_db():
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
            r = client.get(
                "/api/v1/claims/review/metrics"
                "?start_date=2024-01-10&end_date=2024-01-10&limit=1&offset=0"
            )
            assert r.status_code == 200, r.text
            payload = r.json()
            assert payload["total"] == 2
            assert len(payload["items"]) == 1
            assert payload["pagination"] == {
                "mode": "offset",
                "limit": 1,
                "offset": 0,
                "total": 2,
                "has_more": True,
                "next_offset": 1,
            }
            assert payload["has_more"] is True
            assert payload["next_offset"] == 1
    finally:
        fastapi_app.dependency_overrides.pop(get_auth_principal, None)
        fastapi_app.dependency_overrides.pop(get_request_user, None)
        fastapi_app.dependency_overrides.pop(get_media_db_for_user, None)
