from __future__ import annotations

import sqlite3

import pytest

from tldw_Server_API.app.core.Moderation.review_store import ModerationReviewStore


def _item_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "idempotency_key": "chat:42:input:user-7:pii:block:abc123",
        "phase": "input",
        "source_type": "chat",
        "source_id": "42",
        "user_id": "user-7",
        "session_id": "session-9",
        "severity": "high",
        "category": "pii",
        "safe_fields": {"excerpt": True, "context": True},
        "excerpt": "before [REDACTED] after",
        "context": {"conversation_id": "42"},
        "effective_policy": {"enabled": True, "input_action": "block"},
        "matches": [
            {
                "rule_id": "rule-1",
                "pattern_type": "regex",
                "category": "pii",
                "action": "block",
                "sample": "before [REDACTED] after",
                "confidence": 0.86,
            }
        ],
        "recommended_action": "block",
    }
    payload.update(overrides)
    return payload


@pytest.mark.unit
def test_review_store_creates_schema_and_idempotent_items(tmp_path):
    db_path = tmp_path / "moderation_review.db"
    store = ModerationReviewStore(db_path)

    first = store.upsert_item(_item_payload())
    second = store.upsert_item(_item_payload())

    assert first["id"] == second["id"]
    assert second["excerpt"] == "before [REDACTED] after"

    with sqlite3.connect(db_path) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
    assert {
        "moderation_review_items",
        "moderation_review_decisions",
        "moderation_review_audit_events",
    }.issubset(tables)


@pytest.mark.unit
def test_review_store_filters_paginates_decides_undoes_and_audits(tmp_path):
    store = ModerationReviewStore(tmp_path / "moderation_review.db")
    item_a = store.upsert_item(_item_payload(idempotency_key="a", category="pii", severity="high"))
    item_b = store.upsert_item(_item_payload(idempotency_key="b", category="toxicity", severity="medium"))

    page_1 = store.list_items(filters={"status": "needs_review"}, limit=1)
    assert [item["id"] for item in page_1["items"]] == [item_b["id"]]
    assert page_1["next_cursor"] is not None

    page_2 = store.list_items(filters={"status": "needs_review"}, limit=1, cursor=page_1["next_cursor"])
    assert [item["id"] for item in page_2["items"]] == [item_a["id"]]
    assert page_2["next_cursor"] is None

    filtered = store.list_items(filters={"category": "pii"}, limit=10)
    assert [item["id"] for item in filtered["items"]] == [item_a["id"]]

    decision = store.record_decision(
        item_a["id"],
        action="approve",
        decided_by="principal:user-7",
        reason="Looks safe after review",
    )
    assert decision["status"] == "approved"
    assert decision["previous_status"] == "needs_review"
    assert decision["undo_token"]
    assert store.get_item(item_a["id"])["status"] == "approved"

    undone = store.undo_decision(
        item_a["id"],
        undo_token=decision["undo_token"],
        actor_id="principal:user-7",
    )
    assert undone["status"] == "needs_review"

    audit = store.list_audit(limit=10)
    assert [event["action"] for event in audit["events"]] == [
        "decision.undo",
        "decision.approve",
        "item.created",
        "item.created",
    ]


@pytest.mark.unit
def test_review_store_redacts_content_but_keeps_audit_and_metadata(tmp_path):
    store = ModerationReviewStore(tmp_path / "moderation_review.db")
    item = store.upsert_item(_item_payload())

    redacted = store.redact_item_content(item["id"], actor_id="privacy-officer")

    assert redacted["excerpt"] == "[content redacted]"
    assert redacted["context"] == {}
    assert redacted["matches"][0]["sample"] == "[content redacted]"
    assert redacted["content_redacted_at"] is not None
    assert redacted["safe_fields"]["excerpt"] is False

    audit = store.list_audit(item_id=item["id"], limit=10)
    assert audit["events"][0]["action"] == "content.redacted"
