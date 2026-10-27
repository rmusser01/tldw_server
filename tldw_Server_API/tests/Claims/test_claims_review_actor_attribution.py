"""Persist the authenticated review actor independently of reviewer assignment."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.core.AuthNZ.permissions import CLAIMS_REVIEW
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.Claims_Extraction import claims_service
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase

pytestmark = pytest.mark.unit


@pytest.fixture
def review_db(tmp_path, monkeypatch):
    db = MediaDatabase(db_path=str(tmp_path / "claims-actor.db"), client_id="1")
    db.initialize_db()
    content = "Rowan Observatory opened in 2019. Its telescope is Selene."
    media_id, _, _ = db.add_media_with_keywords(
        title="Fictional actor control", media_type="text", content=content, keywords=None
    )
    db.upsert_claims(
        [
            {
                "media_id": media_id,
                "chunk_index": 0,
                "claim_text": text,
                "chunk_hash": hashlib.sha256(content.encode()).hexdigest(),
                "extractor": "heuristic",
                "extractor_version": "v1",
                "reviewer_id": reviewer,
            }
            for text, reviewer in [("Rowan Observatory opened in 2019.", None), ("Its telescope is Selene.", 7)]
        ]
    )
    monkeypatch.setattr(claims_service, "_enqueue_or_dispatch_claim_review_notifications", lambda **_kwargs: None)
    try:
        yield db, db.get_claims_by_media(media_id)
    finally:
        db.close_connection()


def principal(*, admin=True):
    return AuthPrincipal(
        kind="user",
        user_id=1,
        subject="actor-control",
        roles=["admin"] if admin else ["reviewer"],
        permissions=[] if admin else [CLAIMS_REVIEW],
        is_admin=admin,
    )


def user(*, admin=True):
    return SimpleNamespace(id=1, username="actor-control", is_admin=admin)


@pytest.mark.parametrize("assignment", [None, 7])
async def test_single_admin_history_records_actor_not_assignee(review_db, assignment):
    db, rows = review_db
    target = rows[0]
    untouched = db.get_claim_with_media(rows[1]["id"])
    result = await claims_service.review_claim(
        claim_id=target["id"],
        payload={
            "status": "approved",
            "review_version": target["review_version"],
            "reviewer_id": assignment,
            "notes": "Supported fictional source.",
        },
        user_id=None,
        principal=principal(),
        current_user=user(),
        db=db,
    )
    history = db.list_claim_review_history(target["id"])
    assert history[-1]["reviewer_id"] == 1  # nosec B101: pytest verifies review history and isolation.
    assert result["reviewer_id"] == assignment  # nosec B101: pytest verifies review history and isolation.
    assert db.get_claim_with_media(rows[1]["id"]) == untouched  # nosec B101: pytest verifies review history and isolation.


@pytest.mark.parametrize("assignment", [None, 7])
def test_bulk_admin_history_records_actor_not_assignee(review_db, assignment):
    db, rows = review_db
    target = rows[0]
    untouched = db.get_claim_with_media(rows[1]["id"])
    result = claims_service.bulk_review_claims(
        payload={
            "claim_ids": [target["id"]],
            "status": "approved",
            "reviewer_id": assignment,
            "notes": "Supported fictional batch.",
        },
        user_id=None,
        principal=principal(),
        current_user=user(),
        db=db,
    )
    history = db.list_claim_review_history(target["id"])
    assert history[-1]["reviewer_id"] == 1  # nosec B101: pytest verifies review history and isolation.
    assert result["updated"] == [target["id"]]  # nosec B101: pytest verifies review history and isolation.
    assert db.get_claim_with_media(target["id"])["reviewer_id"] == assignment  # nosec B101: pytest verifies review history and isolation.
    assert db.get_claim_with_media(rows[1]["id"]) == untouched  # nosec B101: pytest verifies review history and isolation.


async def test_assigned_reviewer_history_and_assignment_remain_consistent(review_db):
    db, rows = review_db
    target = db.update_claim_review(rows[0]["id"], reviewer_id=1)
    result = await claims_service.review_claim(
        claim_id=target["id"],
        payload={"status": "approved", "review_version": target["review_version"]},
        user_id=None,
        principal=principal(admin=False),
        current_user=user(admin=False),
        db=db,
    )
    assert result["reviewer_id"] == 1  # nosec B101: pytest verifies review history and isolation.
    assert db.list_claim_review_history(target["id"])[-1]["reviewer_id"] == 1  # nosec B101: pytest verifies review history and isolation.


async def test_stale_review_keeps_claim_and_history_unchanged(review_db):
    db, rows = review_db
    target = db.update_claim_review(rows[0]["id"], reviewer_id=1)
    history = db.list_claim_review_history(target["id"])
    with pytest.raises(HTTPException) as exc:
        await claims_service.review_claim(
            claim_id=target["id"],
            payload={"status": "approved", "review_version": 1},
            user_id=None,
            principal=principal(),
            current_user=user(),
            db=db,
        )
    assert exc.value.status_code == 409  # nosec B101: pytest verifies review history and isolation.
    assert db.get_claim_with_media(target["id"]) == target  # nosec B101: pytest verifies review history and isolation.
    assert db.list_claim_review_history(target["id"]) == history  # nosec B101: pytest verifies review history and isolation.


def test_direct_writer_retains_existing_assignee_audit_fallback(review_db):
    db, rows = review_db
    target = rows[1]
    updated = db.update_claim_review(target["id"], review_status="approved", expected_version=target["review_version"])
    assert updated["reviewer_id"] == 7  # nosec B101: pytest verifies review history and isolation.
    assert db.list_claim_review_history(target["id"])[-1]["reviewer_id"] == 7  # nosec B101: pytest verifies review history and isolation.


@pytest.mark.parametrize("actor_id, expected_actor", [(2, 2), (None, 1)])
async def test_history_prefers_principal_actor_and_retains_legacy_user_fallback(review_db, actor_id, expected_actor):
    db, rows = review_db
    target = rows[0]
    actor = principal()
    actor.user_id = actor_id
    await claims_service.review_claim(
        claim_id=target["id"],
        payload={"status": "approved", "review_version": target["review_version"], "reviewer_id": 7},
        user_id=None,
        principal=actor,
        current_user=user(),
        db=db,
    )
    assert db.list_claim_review_history(target["id"])[-1]["reviewer_id"] == expected_actor  # nosec B101: pytest verifies review history and isolation.
