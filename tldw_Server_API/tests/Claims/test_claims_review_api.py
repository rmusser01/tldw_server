import hashlib
import os
import tempfile
from typing import AsyncGenerator

from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal, AuthContext
from tldw_Server_API.app.core.AuthNZ.permissions import CLAIMS_ADMIN, CLAIMS_REVIEW
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase


def _seed_review_db() -> tuple[str, int, int]:


    tmpdir = tempfile.mkdtemp(prefix="claims_review_")
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
        },
        {
            "media_id": media_id,
            "chunk_index": 0,
            "span_start": None,
            "span_end": None,
            "claim_text": "B.",
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
            "claim_text": "C.",
            "confidence": 0.8,
            "extractor": "heuristic",
            "extractor_version": "v1",
            "chunk_hash": chunk_hash,
        },
    ])
    row = db.execute_query(
        "SELECT id FROM Claims WHERE media_id = ? AND deleted = 0 ORDER BY id ASC LIMIT 1",
        (media_id,),
    ).fetchone()
    claim_id = int(row["id"]) if isinstance(row, dict) else int(row[0])
    db.execute_query(
        "UPDATE Claims SET reviewer_id = ?, review_status = 'pending' WHERE media_id = ?",
        (1, media_id),
        commit=True,
    )
    db.close_connection()
    return db_path, claim_id, media_id


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
            permissions=[CLAIMS_REVIEW, CLAIMS_ADMIN],
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


def test_claims_review_flow():


    from tldw_Server_API.app.main import app as fastapi_app

    class _User:
        def __init__(self) -> None:
            self.id = 1
            self.username = "reviewer"
            self.is_admin = False

    async def _override_user():
        return _User()

    db_path, claim_id, _media_id = _seed_review_db()

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
            r = client.get("/api/v1/claims/review-queue")
            assert r.status_code == 200, r.text
            queue = r.json()
            assert any(int(item["id"]) == claim_id for item in queue)

            r_queue_legacy = client.get("/api/v1/claims/review-queue?limit=1&offset=0")
            assert r_queue_legacy.status_code == 200, r_queue_legacy.text
            assert isinstance(r_queue_legacy.json(), list)

            r_queue_page = client.get(
                "/api/v1/claims/review-queue?limit=2&offset=0&envelope=true"
            )
            assert r_queue_page.status_code == 200, r_queue_page.text
            queue_page = r_queue_page.json()
            assert len(queue_page["items"]) == 2
            assert queue_page["pagination"] == {
                "mode": "offset",
                "limit": 2,
                "offset": 0,
                "total": None,
                "has_more": True,
                "next_offset": 2,
            }
            assert queue_page["has_more"] is True
            assert queue_page["next_offset"] == 2

            payload = {
                "status": "approved",
                "review_version": 1,
                "notes": "Looks good",
            }
            r2 = client.patch(f"/api/v1/claims/{claim_id}/review", json=payload)
            assert r2.status_code == 200, r2.text
            data2 = r2.json()
            assert data2.get("review_status") == "approved"

            r3 = client.get(f"/api/v1/claims/{claim_id}/history")
            assert r3.status_code == 200, r3.text
            history = r3.json()
            assert len(history) >= 1

            rule_payload = {
                "priority": 10,
                "predicate_json": {"source": "example.com"},
            }
            r4 = client.post("/api/v1/claims/review/rules", json=rule_payload)
            assert r4.status_code == 200, r4.text
            rule = r4.json()
            assert rule.get("priority") == 10

            r5 = client.get("/api/v1/claims/review/rules")
            assert r5.status_code == 200, r5.text
            rules = r5.json()
            assert any(int(item["id"]) == int(rule["id"]) for item in rules)
    finally:
        fastapi_app.dependency_overrides.pop(get_auth_principal, None)
        fastapi_app.dependency_overrides.pop(get_request_user, None)
        fastapi_app.dependency_overrides.pop(get_media_db_for_user, None)


def test_claims_review_corrected_text_updates_span():


    from tldw_Server_API.app.main import app as fastapi_app

    class _User:
        def __init__(self) -> None:
            self.id = 1
            self.username = "reviewer"
            self.is_admin = False

    async def _override_user():
        return _User()

    db_path, claim_id, media_id = _seed_review_db()

    seed_db = MediaDatabase(db_path=db_path, client_id="1")
    try:
        seed_db.process_unvectorized_chunks(
            media_id=media_id,
            chunks=[
                {
                    "chunk_text": "A1. B. C.",
                    "chunk_index": 0,
                    "start_char": 0,
                    "end_char": len("A1. B. C."),
                    "chunk_type": "text",
                }
            ],
        )
    finally:
        try:
            seed_db.close_connection()
        except Exception:
            _ = None

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
            payload = {
                "status": "approved",
                "review_version": 1,
                "notes": "Corrected",
                "corrected_text": "A1.",
            }
            r = client.patch(f"/api/v1/claims/{claim_id}/review", json=payload)
            assert r.status_code == 200, r.text
            data = r.json()
            assert data.get("claim_text") == "A1."
            assert data.get("span_start") == 0
            assert data.get("span_end") == 3
    finally:
        fastapi_app.dependency_overrides.pop(get_auth_principal, None)
        fastapi_app.dependency_overrides.pop(get_request_user, None)
        fastapi_app.dependency_overrides.pop(get_media_db_for_user, None)
