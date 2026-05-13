from __future__ import annotations

import sqlite3

import pytest

from tldw_Server_API.app.core.Moderation.review_store import ModerationReviewStore


def _item_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "idempotency_key": "audit-item",
        "phase": "input",
        "source_type": "chat",
        "source_id": "conversation-42",
        "user_id": "user-7",
        "session_id": "session-9",
        "severity": "high",
        "category": "pii",
        "safe_fields": {
            "excerpt": True,
            "context": True,
            "matches": True,
            "effective_policy": True,
        },
        "excerpt": "private [REDACTED] sample",
        "context": {"conversation_id": "conversation-42", "turn": "3"},
        "effective_policy": {"enabled": True, "input_action": "block"},
        "matches": [
            {
                "rule_id": "pii-rule",
                "pattern_type": "pii",
                "category": "pii",
                "action": "block",
                "sample": "private [REDACTED] sample",
                "confidence": 0.91,
            }
        ],
        "recommended_action": "block",
    }
    payload.update(overrides)
    return payload


@pytest.mark.unit
def test_item_detail_includes_sanitized_decision_history_without_raw_undo_token(tmp_path):
    store = ModerationReviewStore(tmp_path / "review.db")
    item = store.upsert_item(_item_payload())

    decision = store.record_decision(
        item["id"],
        action="block",
        decided_by="principal:reviewer",
        reason="Contains private data",
    )

    detail = store.get_item(item["id"], include_history=True)

    assert detail is not None
    assert detail["decision_history"] == [
        {
            "id": decision["id"],
            "action": "block",
            "status": "blocked",
            "previous_status": "needs_review",
            "actor_id": "principal:reviewer",
            "reason": "Contains private data",
            "decided_at": decision["decided_at"],
            "undo_eligible": True,
            "undo_expires_at": decision["undo_expires_at"],
            "undone_at": None,
            "redaction_state": "not_redacted",
        }
    ]
    assert "undo_token" not in detail["decision_history"][0]


@pytest.mark.unit
def test_undo_tokens_are_hashed_expire_and_single_use(tmp_path):
    db_path = tmp_path / "review.db"
    store = ModerationReviewStore(db_path, undo_ttl_seconds=900)
    item = store.upsert_item(_item_payload())

    decision = store.record_decision(
        item["id"],
        action="approve",
        decided_by="principal:reviewer",
        reason="False positive",
    )
    raw_token = decision["undo_token"]
    assert raw_token

    with sqlite3.connect(db_path) as conn:
        columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(moderation_review_decisions)").fetchall()
        }
        row = conn.execute(
            "SELECT undo_token_hash, undo_expires_at FROM moderation_review_decisions WHERE id = ?",
            (decision["id"],),
        ).fetchone()

    assert "undo_token" not in columns
    assert row[0] != raw_token
    assert row[1] == decision["undo_expires_at"]

    undone = store.undo_decision(item["id"], undo_token=raw_token, actor_id="principal:reviewer")
    assert undone["status"] == "needs_review"

    with pytest.raises(KeyError):
        store.undo_decision(item["id"], undo_token=raw_token, actor_id="principal:reviewer")

    expired_store = ModerationReviewStore(tmp_path / "expired.db", undo_ttl_seconds=-1)
    expired_item = expired_store.upsert_item(_item_payload(idempotency_key="expired"))
    expired_decision = expired_store.record_decision(
        expired_item["id"],
        action="dismiss",
        decided_by="principal:reviewer",
    )
    with pytest.raises(ValueError, match="expired"):
        expired_store.undo_decision(
            expired_item["id"],
            undo_token=expired_decision["undo_token"],
            actor_id="principal:reviewer",
        )


@pytest.mark.unit
def test_undo_fails_when_later_decision_supersedes_original(tmp_path):
    store = ModerationReviewStore(tmp_path / "review.db")
    item = store.upsert_item(_item_payload())

    first = store.record_decision(item["id"], action="approve", decided_by="principal:one")
    store.record_decision(item["id"], action="block", decided_by="principal:two")

    with pytest.raises(ValueError, match="superseded"):
        store.undo_decision(item["id"], undo_token=first["undo_token"], actor_id="principal:one")

    assert store.get_item(item["id"])["status"] == "blocked"


@pytest.mark.unit
def test_audit_list_supports_decision_actor_action_and_date_filters(tmp_path):
    store = ModerationReviewStore(tmp_path / "review.db")
    item_a = store.upsert_item(_item_payload(idempotency_key="a", source_id="a"))
    item_b = store.upsert_item(_item_payload(idempotency_key="b", source_id="b"))
    decision_a = store.record_decision(
        item_a["id"],
        action="block",
        decided_by="principal:reviewer-a",
        reason="Private data",
    )
    store.record_decision(
        item_b["id"],
        action="dismiss",
        decided_by="principal:reviewer-b",
        reason="No issue",
    )

    by_decision = store.list_audit(decision_id=decision_a["id"], limit=10)
    assert [event["decision_id"] for event in by_decision["events"]] == [decision_a["id"]]

    by_actor = store.list_audit(actor_id="principal:reviewer-a", limit=10)
    assert [event["actor_id"] for event in by_actor["events"]] == ["principal:reviewer-a"]

    by_action = store.list_audit(action="decision.dismiss", limit=10)
    assert [event["action"] for event in by_action["events"]] == ["decision.dismiss"]

    by_date = store.list_audit(
        date_from=decision_a["decided_at"],
        date_to="9999-12-31T23:59:59Z",
        limit=10,
    )
    assert any(event["decision_id"] == decision_a["id"] for event in by_date["events"])


@pytest.mark.unit
def test_redact_decision_replaces_content_with_placeholders_and_preserves_audit(tmp_path):
    store = ModerationReviewStore(tmp_path / "review.db")
    item = store.upsert_item(_item_payload())

    decision = store.record_decision(
        item["id"],
        action="redact",
        decided_by="privacy-officer",
        reason="Source content deletion request",
    )
    redacted = store.get_item(item["id"], include_history=True)

    assert redacted["status"] == "redacted"
    assert redacted["excerpt"] == "[content redacted]"
    assert redacted["context"] == {
        "redacted": True,
        "message": "Context removed from moderation review item",
    }
    assert redacted["matches"][0]["sample"] == "[content redacted]"
    assert redacted["safe_fields"]["excerpt"] is False
    assert redacted["content_redacted_at"] is not None
    assert redacted["decision_history"][0]["id"] == decision["id"]
    assert redacted["decision_history"][0]["redaction_state"] == "redacted"

    audit = store.list_audit(item_id=item["id"], limit=10)
    assert {event["action"] for event in audit["events"][:2]} == {
        "content.redacted",
        "decision.redact",
    }
