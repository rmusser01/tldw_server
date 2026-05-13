from __future__ import annotations

import pytest

from tldw_Server_API.app.core.Moderation.review_service import ModerationReviewService
from tldw_Server_API.app.core.Moderation.review_store import ModerationReviewStore


def _service(tmp_path) -> ModerationReviewService:
    return ModerationReviewService(store=ModerationReviewStore(tmp_path / "review.db"))


def _record(service: ModerationReviewService, *, key: str = "key-1", action: str = "block") -> dict:
    return service.record_item(
        {
            "idempotency_key": key,
            "phase": "input",
            "source_type": "chat",
            "source_id": "conversation-1",
            "user_id": "user-1",
            "severity": "high",
            "category": "pii",
            "safe_fields": {"excerpt": True},
            "excerpt": "safe [REDACTED] excerpt",
            "context": {"conversation_id": "conversation-1"},
            "effective_policy": {"enabled": True},
            "matches": [
                {
                    "pattern_type": "regex",
                    "category": "pii",
                    "action": action,
                    "sample": "safe [REDACTED] excerpt",
                    "confidence": 0.9,
                }
            ],
            "recommended_action": action,
        }
    )


@pytest.mark.unit
def test_review_service_maps_decisions_to_statuses_and_uses_actor_from_call(tmp_path):
    service = _service(tmp_path)
    item = _record(service)

    response = service.record_decision(
        item["id"],
        action="redact",
        actor_id="principal:reviewer",
        reason="Needs safer output",
    )

    assert response["item"]["status"] == "redacted"
    assert response["decision"]["decided_by"] == "principal:reviewer"
    assert response["undo_token"]

    audit = service.list_audit(item_id=item["id"], limit=10)
    assert audit["events"][0]["actor_id"] == "principal:reviewer"
    assert audit["events"][0]["action"] == "content.redacted"
    assert audit["events"][1]["action"] == "decision.redact"


@pytest.mark.unit
def test_review_service_bulk_decision_reports_partial_failures(tmp_path):
    service = _service(tmp_path)
    item = _record(service)

    result = service.bulk_decision(
        item_ids=[item["id"], "missing"],
        action="dismiss",
        actor_id="principal:reviewer",
        reason="Batch cleanup",
    )

    assert result["ok_count"] == 1
    assert result["error_count"] == 1
    assert result["results"][0]["ok"] is True
    assert result["results"][1]["ok"] is False
    assert result["results"][1]["error"] == "not_found"


@pytest.mark.unit
def test_review_service_rejects_raw_actor_in_decision_payload(tmp_path):
    service = _service(tmp_path)
    item = _record(service)

    response = service.record_decision(
        item["id"],
        action="approve",
        actor_id="principal:trusted",
        reason="Body actor must be ignored",
        request_actor_id="principal:spoofed",
    )

    assert response["decision"]["decided_by"] == "principal:trusted"


@pytest.mark.unit
def test_review_service_drops_raw_policy_patterns_from_recorded_items(tmp_path):
    service = _service(tmp_path)

    item = service.record_item(
        {
            "idempotency_key": "policy-safe",
            "phase": "input",
            "excerpt": "safe [REDACTED]",
            "effective_policy": {
                "enabled": True,
                "block_patterns": ["private-secret"],
                "rules": [
                    {
                        "pattern": "private-secret",
                        "action": "block",
                        "phase": "both",
                        "categories": "pii",
                        "replacement": "[MASK]",
                    }
                ],
            },
        }
    )

    assert "block_patterns" not in item["effective_policy"]
    assert "private-secret" not in str(item["effective_policy"])
    assert item["effective_policy"]["rules"] == [
        {
            "action": "block",
            "phase": "both",
            "categories": "pii",
            "has_replacement": True,
        }
    ]
