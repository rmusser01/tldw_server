"""Service helpers for sanitized moderation review workflows."""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.DB_Management.Moderation_Review_DB import ModerationReviewStore
from tldw_Server_API.app.core.testing import is_truthy


_ACTION_STATUS = {
    "approve": "approved",
    "block": "blocked",
    "redact": "redacted",
    "dismiss": "dismissed",
    "escalate": "escalated",
}

_CAPTURE_ACTIONS = {"block", "redact", "warn"}


def is_moderation_review_capture_enabled() -> bool:
    """Return whether moderation outcomes should be captured for review."""
    return is_truthy(os.getenv("MODERATION_REVIEW_CAPTURE_ENABLED", "false"))


def _severity_for_action(action: str) -> str:
    return {"block": "high", "redact": "medium", "warn": "low"}.get(action, "low")


def _decision_action_for_moderation_action(action: str) -> str:
    return {"block": "block", "redact": "redact", "warn": "dismiss"}.get(action, "dismiss")


def _pattern_type(pattern: str | None, category: str | None) -> str | None:
    if category == "pii":
        return "pii"
    if category:
        return "category"
    if pattern:
        return "regex"
    return None


def _safe_excerpt(excerpt: str | None) -> str:
    cleaned = str(excerpt or "").strip()
    if not cleaned:
        return "[content unavailable]"
    if len(cleaned) > 240:
        return cleaned[:237] + "..."
    return cleaned


def _sanitize_effective_policy(policy: dict[str, Any] | None) -> dict[str, Any]:
    """Return a review-safe policy snapshot without raw rule patterns."""
    if not isinstance(policy, dict):
        return {}
    safe = dict(policy)
    safe.pop("block_patterns", None)
    rules = safe.get("rules")
    if isinstance(rules, list):
        safe["rules"] = [
            {
                "action": str(rule.get("action") or ""),
                "phase": str(rule.get("phase") or ""),
                "categories": str(rule.get("categories") or ""),
                "has_replacement": bool(rule.get("replacement")),
            }
            for rule in rules
            if isinstance(rule, dict)
        ]
    return safe


def _idempotency_key(parts: list[str | None], excerpt: str) -> str:
    safe_parts = [str(part or "") for part in parts]
    excerpt_hash = hashlib.sha256(excerpt.encode("utf-8")).hexdigest()[:24]
    base = ":".join(safe_parts + [excerpt_hash])
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


class ModerationReviewService:
    """Application service over the moderation review store."""

    def __init__(self, store: ModerationReviewStore | None = None) -> None:
        self.store = store or ModerationReviewStore()

    def list_items(
        self,
        *,
        status: str | None = None,
        category: str | None = None,
        severity: str | None = None,
        source_type: str | None = None,
        source_id: str | None = None,
        user_id: str | None = None,
        q: str | None = None,
        sort: str | None = None,
        limit: int = 50,
        cursor: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        filters = {
            "status": status,
            "category": category,
            "severity": severity,
            "source_type": source_type,
            "source_id": source_id,
            "user_id": user_id,
            "q": q,
            "sort": sort,
        }
        return self.store.list_items(filters=filters, limit=limit, cursor=cursor)

    def get_item(self, item_id: str) -> dict[str, Any] | None:
        return self.store.get_item(item_id, include_history=True)

    def record_item(self, payload: dict[str, Any]) -> dict[str, Any]:
        safe_payload = dict(payload)
        safe_payload["effective_policy"] = _sanitize_effective_policy(
            safe_payload.get("effective_policy") if isinstance(safe_payload.get("effective_policy"), dict) else None
        )
        return self.store.upsert_item(safe_payload)

    def record_decision(
        self,
        item_id: str,
        *,
        action: str,
        actor_id: str,
        reason: str | None = None,
        request_actor_id: str | None = None,
    ) -> dict[str, Any]:
        if action not in _ACTION_STATUS:
            raise ValueError(f"unsupported moderation decision action: {action}")
        decision = self.store.record_decision(item_id, action=action, decided_by=actor_id, reason=reason)
        item = self.store.get_item(item_id, include_history=True)
        return {"item": item, "decision": decision, "undo_token": decision.get("undo_token")}

    def undo_decision(self, item_id: str, *, undo_token: str, actor_id: str) -> dict[str, Any]:
        self.store.undo_decision(item_id, undo_token=undo_token, actor_id=actor_id)
        item = self.store.get_item(item_id, include_history=True)
        if item is None:
            raise KeyError(item_id)
        return item

    def bulk_decision(
        self,
        *,
        item_ids: list[str],
        action: str,
        actor_id: str,
        reason: str | None = None,
    ) -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        ok_count = 0
        error_count = 0
        for item_id in item_ids:
            try:
                response = self.record_decision(item_id, action=action, actor_id=actor_id, reason=reason)
            except KeyError:
                error_count += 1
                results.append({"item_id": item_id, "ok": False, "error": "not_found"})
            except ValueError as exc:
                error_count += 1
                results.append({"item_id": item_id, "ok": False, "error": str(exc)})
            else:
                ok_count += 1
                results.append(
                    {
                        "item_id": item_id,
                        "ok": True,
                        "item": response["item"],
                        "decision": response["decision"],
                        "undo_token": response.get("undo_token"),
                    }
                )
        return {"results": results, "ok_count": ok_count, "error_count": error_count}

    def list_audit(
        self,
        *,
        item_id: str | None = None,
        decision_id: str | None = None,
        actor_id: str | None = None,
        action: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 50,
        cursor: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        return self.store.list_audit(
            item_id=item_id,
            decision_id=decision_id,
            actor_id=actor_id,
            action=action,
            date_from=date_from,
            date_to=date_to,
            limit=limit,
            cursor=cursor,
        )

    def redact_item_content(self, item_id: str, actor_id: str) -> dict[str, Any]:
        return self.store.redact_item_content(item_id, actor_id=actor_id)


@lru_cache(maxsize=1)
def get_moderation_review_service() -> ModerationReviewService:
    """Return the process-wide moderation review service."""
    return ModerationReviewService()


def build_review_item_from_moderation_outcome(
    *,
    phase: str,
    action: str,
    excerpt: str | None,
    category: str | None = None,
    matched_pattern: str | None = None,
    effective_policy: dict[str, Any] | None = None,
    source_type: str | None = None,
    source_id: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any] | None:
    """Build a sanitized review item payload from a moderation outcome."""
    normalized_action = str(action or "").strip().lower()
    if normalized_action not in _CAPTURE_ACTIONS:
        return None
    safe_excerpt = _safe_excerpt(excerpt)
    review_action = _decision_action_for_moderation_action(normalized_action)
    return {
        "idempotency_key": _idempotency_key(
            [source_type, source_id, phase, user_id, category, normalized_action],
            safe_excerpt,
        ),
        "phase": phase,
        "source_type": source_type,
        "source_id": source_id,
        "user_id": user_id,
        "session_id": session_id,
        "severity": _severity_for_action(normalized_action),
        "category": category,
        "safe_fields": {
            "excerpt": True,
            "context": True,
            "effective_policy": True,
            "matches": True,
        },
        "excerpt": safe_excerpt,
        "context": {
            key: str(value)
            for key, value in {
                "source_type": source_type,
                "source_id": source_id,
                "session_id": session_id,
            }.items()
            if value is not None
        },
        "effective_policy": _sanitize_effective_policy(effective_policy),
        "matches": [
            {
                "rule_id": None,
                "pattern_type": _pattern_type(matched_pattern, category),
                "category": category,
                "action": normalized_action,
                "sample": safe_excerpt,
                "confidence": None,
            }
        ],
        "recommended_action": review_action,
    }


def capture_moderation_review_item(**kwargs: Any) -> dict[str, Any] | None:
    """Persist a sanitized review item when capture is enabled."""
    if not is_moderation_review_capture_enabled():
        return None
    payload = build_review_item_from_moderation_outcome(**kwargs)
    if payload is None:
        return None
    try:
        return get_moderation_review_service().record_item(payload)
    except Exception as exc:
        logger.warning("Moderation review capture failed: {}: {}", type(exc).__name__, str(exc))
        return None
