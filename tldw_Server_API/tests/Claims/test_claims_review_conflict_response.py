"""Real-router conflict responses must serialize both supported database row types."""

from collections.abc import AsyncGenerator
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.core.AuthNZ.permissions import CLAIMS_ADMIN, CLAIMS_REVIEW
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("backend_name", ["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)])
def test_stale_claim_review_returns_json_conflict_without_changing_storage(
    backend_name, request, tmp_path, monkeypatch
):
    from tldw_Server_API.app.main import app

    backend = request.getfixturevalue("pg_restricted_backend") if backend_name == "postgres" else None
    path = ":memory:" if backend is not None else str(tmp_path / "claims.db")
    principal = AuthPrincipal(  # nosec B106 -- access is a token-type enum, not a credential.
        kind="user",
        user_id=1,
        subject="reviewer",
        token_type="access",
        roles=["admin"],
        permissions=[CLAIMS_REVIEW, CLAIMS_ADMIN],
        is_admin=True,
        org_ids=[],
        team_ids=[],
    )

    class User:
        id = 1
        username = "reviewer"
        is_admin = True

    async def principal_override():
        return principal

    async def user_override():
        return User()

    async def media_override() -> AsyncGenerator[MediaDatabase, None]:
        current = MediaDatabase(db_path=path, client_id="1", backend=backend)
        try:
            yield current
        finally:
            current.close_connection()

    monkeypatch.setitem(app.dependency_overrides, get_auth_principal, principal_override)
    monkeypatch.setitem(app.dependency_overrides, get_request_user, user_override)
    monkeypatch.setitem(app.dependency_overrides, get_media_db_for_user, media_override)
    with scoped_context(user_id=1, is_admin=True):
        db = MediaDatabase(db_path=path, client_id="1", backend=backend)
        try:
            db.initialize_db()
            media_id, _, _ = db.add_media_with_keywords(
                title="Conflict source",
                media_type="text",
                content="The telescope is named Selene.",
                keywords=None,
                owner_user_id=1,
            )
            db.upsert_claims(
                [
                    {
                        "media_id": media_id,
                        "chunk_index": 0,
                        "claim_text": "The telescope is named Selene.",
                        "extractor": "manual",
                        "extractor_version": "conflict-v1",
                        "chunk_hash": "conflict-source",
                    }
                ]
            )
            claim = db.get_claims_by_media(media_id, limit=10)[0]
            claim_id = int(claim["id"])
            db.update_claim_review(
                claim_id,
                review_status="approved",
                reviewer_id=1,
                review_notes="Saved concurrent review",
                expected_version=1,
                actor_user_id=1,
            )
            before = db.get_claim_with_media(claim_id, include_deleted=True)
            history = db.list_claim_review_history(claim_id)
            with TestClient(app, raise_server_exceptions=False) as client:
                response = client.patch(
                    f"/api/v1/claims/{claim_id}/review",
                    json={"status": "pending", "review_version": 1, "notes": "Stale draft"},
                )
            assert response.status_code == 409, response.text  # nosec B101 -- Assert the real HTTP conflict contract.
            detail = response.json()["detail"]
            current = detail["current"]
            assert detail["error"] == "version_conflict"  # nosec B101 -- Assert the public error discriminator.
            assert (current["id"], current["review_version"], current["review_status"], current["claim_text"]) == (
                claim_id,
                2,
                "approved",
                before["claim_text"],
            )  # nosec B101 -- Assert that the response contains the actual current row.
            for field in ("created_at", "reviewed_at"):
                if backend_name == "postgres":
                    assert isinstance(before[field], datetime)  # nosec B101 -- Ensure the causal PostgreSQL timestamp is real.
                    assert datetime.fromisoformat(current[field]) == before[field]  # nosec B101 -- Assert lossless JSON timestamps.
                else:
                    assert current[field] == before[field]  # nosec B101 -- Preserve SQLite's existing wire representation.
            assert db.get_claim_with_media(claim_id, include_deleted=True) == before  # nosec B101 -- Reject any stale-write effect.
            assert db.list_claim_review_history(claim_id) == history  # nosec B101 -- Reject duplicate or overwritten history.
        finally:
            db.close_connection()
