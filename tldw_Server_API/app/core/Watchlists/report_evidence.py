from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse


NEWS_STALE_AFTER_DAYS = 7


def _get_value(row: Any, key: str, default: Any = None) -> Any:
    if isinstance(row, Mapping):
        return row.get(key, default)
    return getattr(row, key, default)


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False


def _as_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_list_of_text(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except (TypeError, ValueError, json.JSONDecodeError):
            parsed = None
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
        return [stripped]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return _json_safe(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
        if isinstance(parsed, dict):
            return _json_safe(parsed)
    return {}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _parse_datetime(value: Any) -> datetime | None:
    text = _as_text(value)
    if not text:
        return None
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hostname(url: str | None) -> str | None:
    if not url:
        return None
    parsed = urlparse(url)
    return parsed.netloc or None


def _normalize_preset(preset: str | None) -> str:
    normalized = (preset or "general_research").strip().lower()
    if normalized == "auto":
        return "general_research"
    if normalized in {"cti_osint", "news_briefing", "general_research"}:
        return normalized
    return "general_research"


def _source_name_for(source: Any) -> str | None:
    return _as_text(_get_value(source, "name")) or _as_text(_get_value(source, "title"))


def _alert_to_payload(alert: Any) -> dict[str, Any]:
    return {
        "id": int(_as_int(_get_value(alert, "id")) or 0),
        "rule_id": int(_as_int(_get_value(alert, "rule_id")) or 0),
        "rule_name": _as_text(_get_value(alert, "rule_name")) or _as_text(_get_value(alert, "name")),
        "severity": _as_text(_get_value(alert, "severity")) or "medium",
        "status": _as_text(_get_value(alert, "status")) or "unread",
        "title": _as_text(_get_value(alert, "title")),
        "snippet": _as_text(_get_value(alert, "snippet")),
        "matched_text": _as_text(_get_value(alert, "matched_text")),
        "evidence": _as_dict(_get_value(alert, "evidence", _get_value(alert, "evidence_json"))),
        "created_at": _as_text(_get_value(alert, "created_at")),
    }


def _item_to_payload(
    item: Any,
    *,
    sources: Mapping[int, Any],
    alerts: Mapping[int, Sequence[Any]],
) -> dict[str, Any]:
    item_id = int(_as_int(_get_value(item, "id")) or 0)
    source_id = _as_int(_get_value(item, "source_id"))
    source = sources.get(source_id) if source_id is not None else None
    source_name = _source_name_for(source) if source is not None else None
    alert_payloads = [_alert_to_payload(alert) for alert in alerts.get(item_id, [])]
    return {
        "id": item_id,
        "title": _as_text(_get_value(item, "title")),
        "url": _as_text(_get_value(item, "url")),
        "source_id": source_id,
        "source_name": source_name,
        "published_at": _as_text(_get_value(item, "published_at")),
        "summary": _as_text(_get_value(item, "summary")),
        "tags": _as_list_of_text(_get_value(item, "tags")),
        "reviewed": _as_bool(_get_value(item, "reviewed")),
        "queued_for_briefing": _as_bool(_get_value(item, "queued_for_briefing")),
        "alerts": alert_payloads,
    }


def _excluded_reason(item: Any) -> str:
    explicit = _as_text(_get_value(item, "reason"))
    if explicit:
        return explicit
    status = _as_text(_get_value(item, "status"))
    if status and status != "ingested":
        return "filtered_or_error"
    if not _as_bool(_get_value(item, "queued_for_briefing")):
        return "not_queued_for_report"
    return "excluded_from_report"


def _excluded_item_to_payload(item: Any) -> dict[str, Any]:
    return {
        "id": int(_as_int(_get_value(item, "id")) or 0),
        "title": _as_text(_get_value(item, "title")),
        "url": _as_text(_get_value(item, "url")),
        "reason": _excluded_reason(item),
    }


def _source_summary(items: Sequence[dict[str, Any]], sources: Mapping[int, Any]) -> dict[str, Any]:
    source_counts: Counter[int] = Counter()
    missing_source_count = 0
    hosts: set[str] = set()

    for item in items:
        source_id = item.get("source_id")
        source_name = item.get("source_name")
        item_url = item.get("url")
        if source_id is None or not source_name or not item_url:
            missing_source_count += 1
        if isinstance(source_id, int):
            source_counts[source_id] += 1
            source = sources.get(source_id)
            host = _hostname(_as_text(_get_value(source, "url")) if source is not None else None)
            if host:
                hosts.add(host)
        item_host = _hostname(_as_text(item_url))
        if item_host:
            hosts.add(item_host)

    top_sources = []
    for source_id, count in source_counts.items():
        source = sources.get(source_id)
        top_sources.append(
            {
                "source_id": source_id,
                "source_name": _source_name_for(source) if source is not None else None,
                "count": count,
            }
        )
    top_sources.sort(
        key=lambda entry: (
            -int(entry["count"]),
            str(entry.get("source_name") or ""),
            int(entry["source_id"]),
        )
    )

    return {
        "unique_source_count": len(source_counts),
        "missing_source_count": missing_source_count,
        "hosts": sorted(hosts),
        "top_sources": top_sources,
    }


def _snapshot_id(
    *,
    watchlist_id: int | None,
    job_id: int,
    run_id: int,
    preset: str,
    generated_at: str,
    included_ids: Sequence[int],
    excluded_ids: Sequence[int],
) -> str:
    material = json.dumps(
        {
            "watchlist_id": watchlist_id,
            "job_id": job_id,
            "run_id": run_id,
            "preset": preset,
            "generated_at": generated_at,
            "included_ids": list(included_ids),
            "excluded_ids": list(excluded_ids),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]


def _warning(
    code: str,
    message: str,
    *,
    affected_item_ids: Sequence[int] | None = None,
    severity: str = "warning",
) -> dict[str, Any]:
    return {
        "code": code,
        "severity": severity,
        "message": message,
        "affected_item_ids": list(affected_item_ids or []),
    }


def evaluate_report_readiness(snapshot_like: Mapping[str, Any]) -> dict[str, Any]:
    included_items = list(snapshot_like.get("included_items") or [])
    preset = _normalize_preset(_as_text(snapshot_like.get("preset")))
    source_summary = snapshot_like.get("source_summary") if isinstance(snapshot_like.get("source_summary"), dict) else {}
    warnings: list[dict[str, Any]] = []

    if not included_items:
        return {
            "state": "blocked",
            "score": 0,
            "warnings": [
                _warning(
                    "no_included_items",
                    "No updates are included in this report.",
                    severity="blocking",
                )
            ],
        }

    score = 100
    unique_source_count = int(source_summary.get("unique_source_count") or 0)
    if unique_source_count < 2:
        warnings.append(
            _warning(
                "single_source",
                "Only one source is represented in this report.",
            )
        )
        score -= 15

    missing_provenance_ids = [
        int(item["id"])
        for item in included_items
        if not item.get("url") or item.get("source_id") is None or not item.get("source_name")
    ]
    if missing_provenance_ids:
        warnings.append(
            _warning(
                "missing_source_provenance",
                "One or more included updates are missing source provenance.",
                affected_item_ids=missing_provenance_ids,
            )
        )
        score -= 20

    alert_count = int(snapshot_like.get("alert_count") or 0)
    if preset == "cti_osint" and alert_count == 0:
        warnings.append(
            _warning(
                "no_alert_evidence",
                "This CTI report has no matching content alert evidence.",
            )
        )
        score -= 20

    if preset == "news_briefing":
        generated_at = _parse_datetime(snapshot_like.get("generated_at"))
        published_dates = [
            parsed
            for parsed in (_parse_datetime(item.get("published_at")) for item in included_items)
            if parsed is not None
        ]
        if generated_at is not None and published_dates:
            newest = max(published_dates)
            if (generated_at - newest).days >= NEWS_STALE_AFTER_DAYS:
                warnings.append(
                    _warning(
                        "stale_updates",
                        "The newest included update is older than the recency window for a news briefing.",
                    )
                )
                score -= 15

    unreviewed_queued_ids = [
        int(item["id"])
        for item in included_items
        if item.get("queued_for_briefing") and not item.get("reviewed")
    ]
    if unreviewed_queued_ids:
        warnings.append(
            _warning(
                "unreviewed_queued_items",
                "One or more queued updates have not been reviewed.",
                affected_item_ids=unreviewed_queued_ids,
            )
        )
        score -= 10

    return {
        "state": "warning" if warnings else "ready",
        "score": max(0, min(100, score)),
        "warnings": warnings,
    }


def build_legacy_live_only_readiness() -> dict[str, Any]:
    return {
        "state": "legacy_live_only",
        "score": 0,
        "warnings": [
            _warning(
                "legacy_live_only",
                "This older report was created before immutable evidence snapshots were available.",
                severity="info",
            )
        ],
    }


def build_report_evidence_snapshot(
    *,
    watchlist_id: int | None,
    job: Any,
    run: Any,
    included_items: Sequence[Any],
    excluded_items: Sequence[Any],
    sources: Mapping[int, Any],
    alerts: Mapping[int, Sequence[Any]],
    preset: str,
    generated_at: str | None = None,
) -> dict[str, Any]:
    effective_preset = _normalize_preset(preset)
    effective_generated_at = generated_at or _utcnow_iso()
    job_id = int(_as_int(_get_value(job, "id")) or 0)
    run_id = int(_as_int(_get_value(run, "id")) or 0)
    included_payloads = [
        _item_to_payload(item, sources=sources, alerts=alerts)
        for item in included_items
    ]
    excluded_payloads = [_excluded_item_to_payload(item) for item in excluded_items]
    summary = _source_summary(included_payloads, sources)
    alert_payloads = [
        alert
        for item in included_payloads
        for alert in item.get("alerts", [])
    ]
    critical_alert_count = sum(
        1 for alert in alert_payloads if str(alert.get("severity") or "").lower() == "critical"
    )
    snapshot: dict[str, Any] = {
        "schema_version": 1,
        "snapshot_id": _snapshot_id(
            watchlist_id=watchlist_id,
            job_id=job_id,
            run_id=run_id,
            preset=effective_preset,
            generated_at=effective_generated_at,
            included_ids=[int(item["id"]) for item in included_payloads],
            excluded_ids=[int(item["id"]) for item in excluded_payloads],
        ),
        "generated_at": effective_generated_at,
        "preset": effective_preset,
        "watchlist_id": watchlist_id,
        "job_id": job_id,
        "run_id": run_id,
        "output_id": None,
        "included_items": included_payloads,
        "excluded_items": excluded_payloads,
        "source_summary": summary,
        "included_count": len(included_payloads),
        "excluded_count": len(excluded_payloads),
        "alert_count": len(alert_payloads),
        "critical_alert_count": critical_alert_count,
    }
    snapshot["readiness"] = evaluate_report_readiness(snapshot)
    return _json_safe(snapshot)
