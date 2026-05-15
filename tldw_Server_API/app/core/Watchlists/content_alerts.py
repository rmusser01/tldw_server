"""Content-match alert evaluation for first-class Watchlists."""

from __future__ import annotations

import json
import re
from typing import Any, Protocol

from loguru import logger

from tldw_Server_API.app.core.DB_Management.Watchlists_DB import (
    ScrapedItemRow,
    SourceRow,
    WatchlistContentAlertRow,
    WatchlistContentAlertRuleRow,
    WatchlistsDatabase,
)
from tldw_Server_API.app.core.Monitoring.notification_service import get_notification_service


_CONTENT_ALERT_NONCRITICAL_EXCEPTIONS = (
    OSError,
    RuntimeError,
    ValueError,
    TypeError,
    KeyError,
    AttributeError,
    json.JSONDecodeError,
    re.error,
)


class _Notifier(Protocol):
    def notify_or_batch(self, payload: dict[str, Any]) -> str:
        """Send or batch a notification payload."""


def _loads_object(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    data = json.loads(raw)
    return data if isinstance(data, dict) else {}


def _item_text(item: ScrapedItemRow) -> str:
    parts = [
        item.title,
        item.summary,
        item.content,
        item.url,
        " ".join(item.tags()),
    ]
    return "\n".join(str(part) for part in parts if part)


def _match_rule(rule: WatchlistContentAlertRuleRow, text: str) -> tuple[int, int, str] | None:
    if not text:
        return None
    pattern = str(rule.pattern or "").strip()
    if not pattern:
        return None
    if rule.rule_kind == "regex" or rule.match_mode == "regex":
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            return None
        return match.start(), match.end(), match.group(0)
    if rule.match_mode == "exact":
        if text == pattern:
            return 0, len(text), text
        return None
    lowered_text = text.lower()
    lowered_pattern = pattern.lower()
    start = lowered_text.find(lowered_pattern)
    if start < 0:
        return None
    end = start + len(pattern)
    return start, end, text[start:end]


def _snippet(text: str, start: int, end: int, *, max_len: int = 220) -> str:
    half_window = max(0, (max_len - (end - start)) // 2)
    snippet_start = max(0, start - half_window)
    snippet_end = min(len(text), end + half_window)
    out = text[snippet_start:snippet_end].strip()
    if snippet_start > 0:
        out = f"...{out}"
    if snippet_end < len(text):
        out = f"{out}..."
    return out


def _source_constraints_match(rule: WatchlistContentAlertRuleRow, source: SourceRow) -> bool:
    constraints = _loads_object(rule.source_constraints_json)
    if not constraints:
        return True
    source_ids = constraints.get("source_ids")
    if isinstance(source_ids, list) and source_ids:
        allowed_ids = {int(value) for value in source_ids if str(value).strip().isdigit()}
        if int(source.id) not in allowed_ids:
            return False
    source_types = constraints.get("source_types")
    if isinstance(source_types, list) and source_types:
        allowed_types = {str(value).strip().lower() for value in source_types if str(value).strip()}
        if str(source.source_type or "").strip().lower() not in allowed_types:
            return False
    source_tags = constraints.get("source_tags")
    if isinstance(source_tags, list) and source_tags:
        required_tags = {str(value).strip().lower() for value in source_tags if str(value).strip()}
        actual_tags = {str(value).strip().lower() for value in source.tags}
        if not required_tags.intersection(actual_tags):
            return False
    url_contains = constraints.get("url_contains")
    if isinstance(url_contains, list) and url_contains:
        source_url = str(source.url or "").lower()
        required_fragments = [str(value).strip().lower() for value in url_contains if str(value).strip()]
        if required_fragments and not any(fragment in source_url for fragment in required_fragments):
            return False
    return True


def _evidence_payload(
    *,
    item: ScrapedItemRow,
    source: SourceRow,
    rule: WatchlistContentAlertRuleRow,
    matched_text: str,
) -> dict[str, Any]:
    return {
        "url": item.url,
        "title": item.title,
        "summary": item.summary,
        "published_at": item.published_at,
        "source_id": int(source.id),
        "source_name": source.name,
        "source_url": source.url,
        "source_type": source.source_type,
        "source_tags": source.tags,
        "rule_kind": rule.rule_kind,
        "match_mode": rule.match_mode,
        "pattern": rule.pattern,
        "matched_text": matched_text,
    }


def _notification_payload(alert: WatchlistContentAlertRow, rule: WatchlistContentAlertRuleRow) -> dict[str, Any]:
    return {
        "type": "watchlist_content_alert",
        "user_id": alert.user_id,
        "watchlist_id": int(alert.watchlist_id),
        "rule_id": int(alert.rule_id),
        "item_id": int(alert.item_id),
        "run_id": int(alert.run_id),
        "job_id": int(alert.job_id),
        "source_id": int(alert.source_id),
        "severity": alert.severity,
        "rule_kind": rule.rule_kind,
        "pattern": rule.pattern,
        "title": alert.title,
        "snippet": alert.snippet,
        "matched_text": alert.matched_text,
        "route_tags": {"watchlist_id": int(alert.watchlist_id), "item_id": int(alert.item_id)},
    }


def evaluate_content_alert_rules_for_item(
    db: WatchlistsDatabase,
    *,
    watchlist_id: int,
    item: ScrapedItemRow,
    notifier: _Notifier | None = None,
) -> list[WatchlistContentAlertRow]:
    """Evaluate enabled content alert rules for a recorded Watchlist item."""
    try:
        source = db.get_source(int(item.source_id))
        rules, _ = db.list_content_alert_rules(int(watchlist_id), enabled=True, limit=1000, offset=0)
    except _CONTENT_ALERT_NONCRITICAL_EXCEPTIONS as exc:
        logger.debug("watchlist content alert setup skipped: {}", type(exc).__name__)
        return []

    text = _item_text(item)
    if not text:
        return []

    created: list[WatchlistContentAlertRow] = []
    active_notifier = notifier
    for rule in rules:
        try:
            if not _source_constraints_match(rule, source):
                continue
            match = _match_rule(rule, text)
            if match is None:
                continue
            start, end, matched_text = match
            dedupe_key = f"watchlist_content_alert:{int(watchlist_id)}:{int(rule.id)}:{int(item.id)}"
            if db.content_alert_duplicate_exists(dedupe_key):
                continue
            alert = db.create_content_alert(
                watchlist_id=int(watchlist_id),
                rule_id=int(rule.id),
                item_id=int(item.id),
                run_id=int(item.run_id),
                job_id=int(item.job_id),
                source_id=int(item.source_id),
                severity=rule.severity,
                title=item.title,
                snippet=_snippet(text, start, end),
                matched_text=matched_text,
                evidence=_evidence_payload(item=item, source=source, rule=rule, matched_text=matched_text),
                dedupe_key=dedupe_key,
            )
            created.append(alert)
            try:
                if active_notifier is None:
                    active_notifier = get_notification_service()
                active_notifier.notify_or_batch(_notification_payload(alert, rule))
            except _CONTENT_ALERT_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug("watchlist content alert notification skipped: {}", type(exc).__name__)
        except _CONTENT_ALERT_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug(
                "watchlist content alert rule {} skipped: {}",
                getattr(rule, "id", "?"),
                type(exc).__name__,
            )
            continue
    return created
