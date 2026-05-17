from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import time
from contextlib import contextmanager, suppress
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

from loguru import logger

from tldw_Server_API.app.core.Utils.Utils import get_database_dir

_FLAG_SCOPES = {"global", "org", "user"}
_INCIDENT_STATUSES = {"open", "investigating", "mitigating", "resolved"}
_INCIDENT_SEVERITIES = {"low", "medium", "high", "critical"}
_INCIDENT_ACTION_ITEM_LIMIT = 25
_INCIDENT_ACTION_ITEM_TEXT_MAX_LENGTH = 500
_UNSET = object()

_WEBHOOK_EVENTS = {
    "user.created",
    "user.deleted",
    "incident.created",
    "incident.updated",
    "incident.resolved",
}
_WEBHOOK_MAX_URL_LENGTH = 2048

_STORE_LOCK = Lock()
_STORE_PATH = Path(get_database_dir()) / "system_ops.json"
_LOCK_TIMEOUT_SECONDS = float(os.getenv("SYSTEM_OPS_LOCK_TIMEOUT", "5"))

_SYSTEM_OPS_NONCRITICAL_EXCEPTIONS = (
    AttributeError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
    json.JSONDecodeError,
)

try:
    import fcntl  # type: ignore

    _HAS_FCNTL = True
except ImportError:
    _HAS_FCNTL = False


@contextmanager
def _store_file_lock(timeout: float = _LOCK_TIMEOUT_SECONDS):
    lock_path = _STORE_PATH.with_suffix(_STORE_PATH.suffix + ".lock")
    lock_fd = None
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        start_time = time.time()
        if _HAS_FCNTL:
            lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
            while True:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError:
                    if time.time() - start_time > timeout:
                        raise RuntimeError(f"Failed to acquire system ops lock within {timeout}s") from None
                    time.sleep(0.05)
        else:
            while True:
                try:
                    lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
                    break
                except FileExistsError:
                    try:
                        lock_stat = os.stat(lock_path)
                        if time.time() - lock_stat.st_mtime > timeout * 2:
                            os.unlink(lock_path)
                            continue
                    except (OSError, FileNotFoundError):
                        pass
                    if time.time() - start_time > timeout:
                        raise RuntimeError(f"Failed to acquire system ops lock within {timeout}s") from None
                    time.sleep(0.05)
        yield
    finally:
        if lock_fd is not None:
            if _HAS_FCNTL:
                with suppress(_SYSTEM_OPS_NONCRITICAL_EXCEPTIONS):
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
            with suppress(_SYSTEM_OPS_NONCRITICAL_EXCEPTIONS):
                os.close(lock_fd)
        if not _HAS_FCNTL:
            with suppress(_SYSTEM_OPS_NONCRITICAL_EXCEPTIONS):
                lock_path.unlink(missing_ok=True)


@contextmanager
def _locked_store(write: bool = False):
    with _STORE_LOCK, _store_file_lock():
        store = _load_store()
        yield store
        if write:
            _save_store(store)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_store() -> dict[str, Any]:
    return {
        "maintenance": {
            "enabled": False,
            "message": "",
            "allowlist_user_ids": [],
            "allowlist_emails": [],
            "updated_at": None,
            "updated_by": None,
        },
        "feature_flags": [],
        "incidents": [],
        "webhooks": [],
        "webhook_deliveries": [],
        "invitations": [],
        "dependency_health_history": [],
        "email_delivery_log": [],
        "compliance_report_schedules": [],
        "digest_preferences": [],
    }


def _parse_iso(value: str | None) -> datetime:
    if not value:
        return datetime.min.replace(tzinfo=timezone.utc)
    raw = str(value).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)


def _normalize_incident_action_items(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []

    normalized: list[dict[str, Any]] = []
    for raw_item in value[:_INCIDENT_ACTION_ITEM_LIMIT]:
        if not isinstance(raw_item, dict):
            continue
        text = str(raw_item.get("text") or "").strip()
        if not text:
            continue
        normalized.append(
            {
                "id": str(raw_item.get("id") or f"ai_{uuid4().hex[:10]}"),
                "text": text[:_INCIDENT_ACTION_ITEM_TEXT_MAX_LENGTH],
                "done": bool(raw_item.get("done")),
            }
        )
    return normalized


def _normalize_incident_record(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("invalid_incident")

    incident = dict(value)
    incident["assigned_to_user_id"] = (
        int(incident["assigned_to_user_id"]) if incident.get("assigned_to_user_id") is not None else None
    )
    incident["assigned_to_label"] = (
        str(incident["assigned_to_label"]).strip() or None if incident.get("assigned_to_label") is not None else None
    )
    incident["root_cause"] = (
        str(incident["root_cause"]).strip() or None if incident.get("root_cause") is not None else None
    )
    incident["impact"] = str(incident["impact"]).strip() or None if incident.get("impact") is not None else None
    incident["action_items"] = _normalize_incident_action_items(incident.get("action_items"))
    incident.setdefault("acknowledged_at", None)

    # Preserve runbook_url if present
    incident.setdefault("runbook_url", None)

    # Compute SLA metrics (time to acknowledge, time to resolve)
    created_at_raw = incident.get("created_at")
    resolved_at_raw = incident.get("resolved_at")
    created_at = _parse_iso(created_at_raw) if created_at_raw else None
    resolved_at = _parse_iso(resolved_at_raw) if resolved_at_raw else None
    timeline = incident.get("timeline") or []

    # Time to acknowledge = time of first status change after creation
    first_event_at = None
    for event in timeline:
        event_time = _parse_iso(event.get("created_at") if isinstance(event, dict) else None)
        if created_at and event_time and event_time > created_at:
            first_event_at = event_time
            break
    incident["time_to_acknowledge_seconds"] = (
        int((first_event_at - created_at).total_seconds())
        if created_at and first_event_at and first_event_at >= created_at
        else None
    )
    incident["time_to_resolve_seconds"] = (
        int((resolved_at - created_at).total_seconds())
        if created_at and resolved_at and resolved_at >= created_at
        else None
    )

    return incident


def _load_store() -> dict[str, Any]:
    if not _STORE_PATH.exists():
        return _default_store()
    try:
        raw = _STORE_PATH.read_text(encoding="utf-8")
        data = json.loads(raw) if raw.strip() else {}
    except _SYSTEM_OPS_NONCRITICAL_EXCEPTIONS as exc:
        logger.warning("System ops store unreadable: {}", exc)
        return _default_store()
    if not isinstance(data, dict):
        return _default_store()
    data.setdefault("maintenance", _default_store()["maintenance"])
    data.setdefault("feature_flags", [])
    data.setdefault("incidents", [])
    data.setdefault("webhooks", [])
    data.setdefault("webhook_deliveries", [])
    data.setdefault("invitations", [])
    data.setdefault("dependency_health_history", [])
    data.setdefault("email_delivery_log", [])
    data.setdefault("compliance_report_schedules", [])
    data.setdefault("digest_preferences", [])
    return data


def _save_store(store: dict[str, Any]) -> None:
    _STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _STORE_PATH.write_text(json.dumps(store, indent=2, sort_keys=False), encoding="utf-8")


def _normalize_flag_scope(scope: str) -> str:
    value = (scope or "").strip().lower()
    if value not in _FLAG_SCOPES:
        raise ValueError("invalid_scope")
    return value


def _normalize_rollout_percent(value: Any, *, strict: bool) -> int:
    if value is None:
        return 100
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        if strict:
            raise ValueError("invalid_rollout_percent") from None
        return 0
    if 0 <= parsed <= 100:
        return parsed
    if strict:
        raise ValueError("invalid_rollout_percent")
    return 0


def _normalize_allowlist_ids(values: list[int] | None) -> list[int]:
    if not values:
        return []
    cleaned = []
    for val in values:
        try:
            cleaned.append(int(val))
        except (TypeError, ValueError):
            continue
    return sorted(set(cleaned))


def _normalize_allowlist_emails(values: list[str] | None) -> list[str]:
    if not values:
        return []
    cleaned = []
    for val in values:
        if not val:
            continue
        cleaned.append(str(val).strip().lower())
    return sorted({val for val in cleaned if val})


def _normalize_target_user_ids(values: list[int] | None) -> list[int]:
    if not isinstance(values, list):
        return []
    cleaned = _normalize_allowlist_ids(values)
    return [value for value in cleaned if value > 0]


def _normalize_variant_value(value: Any) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _build_flag_snapshot(flag: dict[str, Any]) -> dict[str, Any]:
    return {
        "scope": _normalize_flag_scope(str(flag.get("scope") or "global")),
        "enabled": bool(flag.get("enabled")),
        "org_id": flag.get("org_id"),
        "user_id": flag.get("user_id"),
        "target_user_ids": _normalize_target_user_ids(flag.get("target_user_ids")),
        "rollout_percent": _normalize_rollout_percent(flag.get("rollout_percent"), strict=False),
        "variant_value": _normalize_variant_value(flag.get("variant_value")),
    }


def _normalize_flag_snapshot(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    scope = value.get("scope")
    if scope is None:
        return None
    try:
        normalized_scope = _normalize_flag_scope(str(scope))
    except ValueError:
        return None
    return {
        "scope": normalized_scope,
        "enabled": bool(value.get("enabled")),
        "org_id": value.get("org_id"),
        "user_id": value.get("user_id"),
        "target_user_ids": _normalize_target_user_ids(value.get("target_user_ids")),
        "rollout_percent": _normalize_rollout_percent(value.get("rollout_percent"), strict=False),
        "variant_value": _normalize_variant_value(value.get("variant_value")),
    }


def _normalize_feature_flag_record(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("invalid_feature_flag")
    key = str(value.get("key") or "").strip()
    if not key:
        raise ValueError("invalid_feature_flag")
    scope = _normalize_flag_scope(str(value.get("scope") or "global"))
    normalized = {
        "key": key,
        "scope": scope,
        "enabled": bool(value.get("enabled")),
        "description": (str(value.get("description")).strip() if value.get("description") else None),
        "org_id": value.get("org_id"),
        "user_id": value.get("user_id"),
        "target_user_ids": _normalize_target_user_ids(value.get("target_user_ids")),
        "rollout_percent": _normalize_rollout_percent(value.get("rollout_percent"), strict=False),
        "variant_value": _normalize_variant_value(value.get("variant_value")),
        "created_at": value.get("created_at"),
        "updated_at": value.get("updated_at"),
        "updated_by": value.get("updated_by"),
        "history": [],
    }
    history: list[dict[str, Any]] = []
    for entry in value.get("history") or []:
        if not isinstance(entry, dict):
            continue
        history.append(
            {
                "timestamp": entry.get("timestamp") or normalized["updated_at"] or _now_iso(),
                "enabled": bool(entry.get("enabled", normalized["enabled"])),
                "actor": entry.get("actor"),
                "note": (str(entry.get("note")).strip() if entry.get("note") else None),
                "before": _normalize_flag_snapshot(entry.get("before")),
                "after": _normalize_flag_snapshot(entry.get("after")),
            }
        )
    normalized["history"] = history
    return normalized


def get_maintenance_state() -> dict[str, Any]:
    with _locked_store() as store:
        return dict(store["maintenance"])


def update_maintenance_state(
    *,
    enabled: bool,
    message: str | None,
    allowlist_user_ids: list[int] | None,
    allowlist_emails: list[str] | None,
    actor: str | None,
) -> dict[str, Any]:
    with _locked_store(write=True) as store:
        maintenance = store["maintenance"]
        maintenance["enabled"] = bool(enabled)
        maintenance["message"] = (message or "").strip()
        maintenance["allowlist_user_ids"] = _normalize_allowlist_ids(allowlist_user_ids)
        maintenance["allowlist_emails"] = _normalize_allowlist_emails(allowlist_emails)
        maintenance["updated_at"] = _now_iso()
        maintenance["updated_by"] = actor
        store["maintenance"] = maintenance
        return dict(maintenance)


def list_feature_flags(
    *,
    scope: str | None = None,
    org_id: int | None = None,
    user_id: int | None = None,
) -> list[dict[str, Any]]:
    with _locked_store() as store:
        flags_raw = list(store.get("feature_flags", []))
    flags = []
    for flag in flags_raw:
        try:
            flags.append(_normalize_feature_flag_record(flag))
        except ValueError:
            continue
    if scope:
        scope_norm = _normalize_flag_scope(scope)
        if scope_norm == "org" and org_id is None:
            raise ValueError("missing_org_id")
        if scope_norm == "user" and user_id is None:
            raise ValueError("missing_user_id")
        flags = [flag for flag in flags if flag.get("scope") == scope_norm]
    if org_id is not None:
        flags = [flag for flag in flags if flag.get("org_id") == org_id]
    if user_id is not None:
        flags = [flag for flag in flags if flag.get("user_id") == user_id]
    flags.sort(key=lambda item: (item.get("key") or "", item.get("scope") or ""))
    return flags


def upsert_feature_flag(
    *,
    key: str,
    scope: str,
    enabled: bool,
    description: str | None,
    org_id: int | None,
    user_id: int | None,
    target_user_ids: list[int] | None,
    rollout_percent: int | None,
    variant_value: str | None,
    actor: str | None,
    note: str | None,
) -> dict[str, Any]:
    normalized_key = (key or "").strip()
    if not normalized_key:
        raise ValueError("invalid_key")
    scope_norm = _normalize_flag_scope(scope)
    if scope_norm == "org" and org_id is None:
        raise ValueError("missing_org_id")
    if scope_norm == "user" and user_id is None:
        raise ValueError("missing_user_id")
    normalized_target_user_ids = _normalize_target_user_ids(target_user_ids)
    normalized_rollout_percent = _normalize_rollout_percent(rollout_percent, strict=True)
    normalized_variant_value = _normalize_variant_value(variant_value)

    now = _now_iso()
    with _locked_store(write=True) as store:
        flags = store.get("feature_flags", [])
        for flag in flags:
            if (
                flag.get("key") == normalized_key
                and flag.get("scope") == scope_norm
                and flag.get("org_id") == org_id
                and flag.get("user_id") == user_id
            ):
                before_state = _build_flag_snapshot(_normalize_feature_flag_record(flag))
                flag["enabled"] = bool(enabled)
                if description is not None:
                    flag["description"] = description.strip() or None
                flag["target_user_ids"] = normalized_target_user_ids
                flag["rollout_percent"] = normalized_rollout_percent
                flag["variant_value"] = normalized_variant_value
                flag["updated_at"] = now
                flag["updated_by"] = actor
                after_state = _build_flag_snapshot(flag)
                history_entry = {
                    "timestamp": now,
                    "enabled": bool(enabled),
                    "actor": actor,
                    "note": (note or "").strip() or None,
                    "before": before_state,
                    "after": after_state,
                }
                flag.setdefault("history", []).append(history_entry)
                return _normalize_feature_flag_record(flag)

        new_flag = {
            "key": normalized_key,
            "scope": scope_norm,
            "enabled": bool(enabled),
            "description": description.strip() if description else None,
            "org_id": org_id,
            "user_id": user_id,
            "target_user_ids": normalized_target_user_ids,
            "rollout_percent": normalized_rollout_percent,
            "variant_value": normalized_variant_value,
            "created_at": now,
            "updated_at": now,
            "updated_by": actor,
            "history": [],
        }
        history_entry = {
            "timestamp": now,
            "enabled": bool(enabled),
            "actor": actor,
            "note": (note or "").strip() or None,
            "before": None,
            "after": _build_flag_snapshot(new_flag),
        }
        new_flag["history"].append(history_entry)
        flags.append(new_flag)
        store["feature_flags"] = flags
        return _normalize_feature_flag_record(new_flag)


def delete_feature_flag(
    *,
    key: str,
    scope: str,
    org_id: int | None,
    user_id: int | None,
) -> None:
    normalized_key = (key or "").strip()
    scope_norm = _normalize_flag_scope(scope)
    if scope_norm == "org" and org_id is None:
        raise ValueError("missing_org_id")
    if scope_norm == "user" and user_id is None:
        raise ValueError("missing_user_id")
    with _locked_store(write=True) as store:
        flags = store.get("feature_flags", [])
        remaining = [
            flag
            for flag in flags
            if not (
                flag.get("key") == normalized_key
                and flag.get("scope") == scope_norm
                and flag.get("org_id") == org_id
                and flag.get("user_id") == user_id
            )
        ]
        if len(remaining) == len(flags):
            raise ValueError("not_found")
        store["feature_flags"] = remaining


def list_incidents(
    *,
    status: str | None,
    severity: str | None,
    tag: str | None,
    limit: int,
    offset: int,
) -> tuple[list[dict[str, Any]], int]:
    with _locked_store() as store:
        incidents = list(store.get("incidents", []))
    if status:
        status_norm = status.strip().lower()
        incidents = [item for item in incidents if item.get("status") == status_norm]
    if severity:
        severity_norm = severity.strip().lower()
        incidents = [item for item in incidents if item.get("severity") == severity_norm]
    if tag:
        tag_norm = tag.strip().lower()
        incidents = [item for item in incidents if tag_norm in {t.lower() for t in (item.get("tags") or [])}]
    incidents.sort(key=lambda item: _parse_iso(item.get("updated_at")), reverse=True)
    total = len(incidents)
    safe_offset = max(0, offset)
    safe_limit = max(1, limit)
    items = [_normalize_incident_record(item) for item in incidents[safe_offset : safe_offset + safe_limit]]
    for inc in items:
        inc["mtta_minutes"] = None
        inc["mttr_minutes"] = None
        created = inc.get("created_at")
        acknowledged = inc.get("acknowledged_at")
        resolved = inc.get("resolved_at")
        if created and acknowledged:
            try:
                c = datetime.fromisoformat(str(created))
                a = datetime.fromisoformat(str(acknowledged))
                val = (a - c).total_seconds() / 60
                if val >= 0:
                    inc["mtta_minutes"] = round(val, 1)
            except (ValueError, TypeError):
                pass
        if created and resolved:
            try:
                c = datetime.fromisoformat(str(created))
                r = datetime.fromisoformat(str(resolved))
                val = (r - c).total_seconds() / 60
                if val >= 0:
                    inc["mttr_minutes"] = round(val, 1)
            except (ValueError, TypeError):
                pass
    return items, total


def create_incident(
    *,
    title: str,
    status: str | None,
    severity: str | None,
    summary: str | None,
    tags: list[str] | None,
    actor: str | None,
) -> dict[str, Any]:
    title_norm = (title or "").strip()
    if not title_norm:
        raise ValueError("invalid_title")
    status_norm = (status or "open").strip().lower()
    severity_norm = (severity or "medium").strip().lower()
    if status_norm not in _INCIDENT_STATUSES:
        raise ValueError("invalid_status")
    if severity_norm not in _INCIDENT_SEVERITIES:
        raise ValueError("invalid_severity")
    now = _now_iso()
    incident_id = f"inc_{uuid4().hex[:10]}"
    resolved_at = now if status_norm == "resolved" else None
    timeline_entry = {
        "id": f"evt_{uuid4().hex[:10]}",
        "message": "Incident created",
        "created_at": now,
        "actor": actor,
    }
    acknowledged_at = now if status_norm != "open" else None
    incident = {
        "id": incident_id,
        "title": title_norm,
        "status": status_norm,
        "severity": severity_norm,
        "summary": (summary or "").strip() or None,
        "tags": tags or [],
        "created_at": now,
        "updated_at": now,
        "resolved_at": resolved_at,
        "acknowledged_at": acknowledged_at,
        "created_by": actor,
        "updated_by": actor,
        "timeline": [timeline_entry],
        "assigned_to_user_id": None,
        "assigned_to_label": None,
        "root_cause": None,
        "impact": None,
        "action_items": [],
    }
    with _locked_store(write=True) as store:
        store.setdefault("incidents", []).append(incident)
    return _normalize_incident_record(incident)


def update_incident(
    *,
    incident_id: str,
    title: str | None,
    status: str | None,
    severity: str | None,
    summary: str | None,
    tags: list[str] | None,
    assigned_to_user_id: Any = _UNSET,
    assigned_to_label: Any = _UNSET,
    root_cause: Any = _UNSET,
    impact: Any = _UNSET,
    runbook_url: Any = _UNSET,
    action_items: Any = _UNSET,
    update_message: str | None,
    actor: str | None,
) -> dict[str, Any]:
    now = _now_iso()
    note = (update_message or "").strip() or None
    with _locked_store(write=True) as store:
        incidents = store.get("incidents", [])
        for index, incident in enumerate(incidents):
            if incident.get("id") != incident_id:
                continue
            current = _normalize_incident_record(incident)
            updated_incident = dict(current)
            updated_incident["tags"] = list(current.get("tags") or [])
            updated_incident["timeline"] = list(current.get("timeline") or [])
            updated_incident["action_items"] = [dict(item) for item in current.get("action_items") or []]
            if title is not None:
                updated_incident["title"] = title.strip() or current.get("title")
            if status is not None:
                status_norm = status.strip().lower()
                if status_norm not in _INCIDENT_STATUSES:
                    raise ValueError("invalid_status")
                if current.get("status") == "open" and status_norm != "open" and not current.get("acknowledged_at"):
                    updated_incident["acknowledged_at"] = now
                updated_incident["status"] = status_norm
                updated_incident["resolved_at"] = now if status_norm == "resolved" else None
            if severity is not None:
                severity_norm = severity.strip().lower()
                if severity_norm not in _INCIDENT_SEVERITIES:
                    raise ValueError("invalid_severity")
                updated_incident["severity"] = severity_norm
            if summary is not None:
                updated_incident["summary"] = summary.strip() or None
            if tags is not None:
                updated_incident["tags"] = tags
            if assigned_to_user_id is not _UNSET:
                if assigned_to_user_id is None:
                    updated_incident["assigned_to_user_id"] = None
                    updated_incident["assigned_to_label"] = None
                else:
                    updated_incident["assigned_to_user_id"] = int(assigned_to_user_id)
                    updated_incident["assigned_to_label"] = (
                        str(assigned_to_label).strip() or None
                        if assigned_to_label is not None and assigned_to_label is not _UNSET
                        else None
                    )
            if root_cause is not _UNSET:
                updated_incident["root_cause"] = str(root_cause).strip() or None if root_cause is not None else None
            if impact is not _UNSET:
                updated_incident["impact"] = str(impact).strip() or None if impact is not None else None
            if runbook_url is not _UNSET:
                updated_incident["runbook_url"] = str(runbook_url).strip() or None if runbook_url is not None else None
            if action_items is not _UNSET:
                updated_incident["action_items"] = _normalize_incident_action_items(action_items)
            if note:
                updated_incident.setdefault("timeline", []).append(
                    {
                        "id": f"evt_{uuid4().hex[:10]}",
                        "message": note,
                        "created_at": now,
                        "actor": actor,
                    }
                )
            updated_incident["updated_at"] = now
            updated_incident["updated_by"] = actor
            incidents[index] = updated_incident
            return _normalize_incident_record(updated_incident)
    raise ValueError("not_found")


def add_incident_event(
    *,
    incident_id: str,
    message: str,
    actor: str | None,
) -> dict[str, Any]:
    note = (message or "").strip()
    if not note:
        raise ValueError("invalid_message")
    now = _now_iso()
    with _locked_store(write=True) as store:
        incidents = store.get("incidents", [])
        for incident in incidents:
            if incident.get("id") != incident_id:
                continue
            event = {
                "id": f"evt_{uuid4().hex[:10]}",
                "message": note,
                "created_at": now,
                "actor": actor,
            }
            incident.setdefault("timeline", []).append(event)
            incident["updated_at"] = now
            incident["updated_by"] = actor
            return _normalize_incident_record(incident)
    raise ValueError("not_found")


def get_incident(*, incident_id: str) -> dict[str, Any]:
    """Return a single incident by ID, or raise ``ValueError("not_found")``."""
    with _locked_store() as store:
        for incident in store.get("incidents", []):
            if incident.get("id") == incident_id:
                return _normalize_incident_record(incident)
    raise ValueError("not_found")


def delete_incident(*, incident_id: str) -> None:
    with _locked_store(write=True) as store:
        incidents = store.get("incidents", [])
        remaining = [item for item in incidents if item.get("id") != incident_id]
        if len(remaining) == len(incidents):
            raise ValueError("not_found")
        store["incidents"] = remaining


def notify_incident_stakeholders(
    *,
    incident_id: str,
    recipients: list[str],
    message: str | None = None,
    actor: str | None = None,
) -> dict[str, Any]:
    """Send email notification to stakeholders about an incident.

    Uses the ``EmailService`` to deliver a plain-text notification to each
    recipient.  Results are collected per-address and the notification is
    recorded as a timeline event on the incident.

    Returns a dict with ``incident_id`` and a ``notifications`` list of
    per-recipient delivery outcomes.
    """
    import asyncio

    from tldw_Server_API.app.core.AuthNZ.email_service import get_email_service

    incident = get_incident(incident_id=incident_id)
    email_service = get_email_service()

    subject = f"[Incident {incident['id']}] {incident['title']}" f" \u2014 {incident['status']}"
    body_parts = [
        f"Incident: {incident['title']}",
        f"Status: {incident['status']}",
        f"Severity: {incident['severity']}",
    ]
    custom_text = (message or "").strip()
    body_parts.append("")
    body_parts.append(custom_text or incident.get("summary") or "")
    text_body = "\n".join(body_parts)

    results: list[dict[str, Any]] = []
    for email_addr in recipients:
        email_addr = email_addr.strip()
        if not email_addr:
            continue
        try:
            # EmailService.send_email is async; run it in a sync context.
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop and loop.is_running():
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    fut = pool.submit(
                        asyncio.run,
                        email_service.send_email(
                            to_email=email_addr,
                            subject=subject,
                            html_body=f"<pre>{text_body}</pre>",
                            text_body=text_body,
                            _template="incident_notification",
                        ),
                    )
                    fut.result(timeout=30)
            else:
                asyncio.run(
                    email_service.send_email(
                        to_email=email_addr,
                        subject=subject,
                        html_body=f"<pre>{text_body}</pre>",
                        text_body=text_body,
                        _template="incident_notification",
                    )
                )
            results.append({"email": email_addr, "status": "sent"})
        except Exception as exc:  # noqa: BLE001
            logger.warning("Incident notification to {} failed: {}", email_addr, exc)
            results.append({"email": email_addr, "status": "failed", "error": str(exc)})

    # Record notification in the incident timeline
    sent_count = sum(1 for r in results if r["status"] == "sent")
    total_count = len(results)
    add_incident_event(
        incident_id=incident_id,
        message=f"Notification sent to {sent_count}/{total_count} stakeholder(s)",
        actor=actor or "system",
    )

    return {"incident_id": incident_id, "notifications": results}


# -----------------------------------------------------------------------------------------------------------------
# Webhooks
# -----------------------------------------------------------------------------------------------------------------


def _normalize_webhook_record(value: Any) -> dict[str, Any]:
    """Normalize a raw webhook dict from the store."""
    if not isinstance(value, dict):
        raise ValueError("invalid_webhook")
    return {
        "id": str(value.get("id") or ""),
        "url": str(value.get("url") or ""),
        "events": list(value.get("events") or []),
        "enabled": bool(value.get("enabled", True)),
        "created_at": value.get("created_at"),
        "updated_at": value.get("updated_at"),
    }


def _redact_webhook(webhook: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of the webhook with the secret removed (for list responses)."""
    result = dict(webhook)
    result.pop("secret", None)
    return result


def list_webhooks() -> list[dict[str, Any]]:
    """Return all webhooks with secrets redacted."""
    with _locked_store() as store:
        webhooks_raw = list(store.get("webhooks", []))
    webhooks = []
    for webhook in webhooks_raw:
        try:
            webhooks.append(_redact_webhook(_normalize_webhook_record(webhook)))
        except ValueError:
            continue
    webhooks.sort(key=lambda item: item.get("created_at") or "")
    return webhooks


def create_webhook(
    *,
    url: str,
    events: list[str],
    enabled: bool = True,
) -> dict[str, Any]:
    """Create a new webhook and return it with the secret included (shown once)."""
    url_norm = (url or "").strip()
    if not url_norm:
        raise ValueError("invalid_url")
    if len(url_norm) > _WEBHOOK_MAX_URL_LENGTH:
        raise ValueError("invalid_url")
    if not url_norm.startswith(("http://", "https://")):
        raise ValueError("invalid_url")

    events_norm = sorted({e.strip().lower() for e in (events or []) if e and e.strip()})
    if not events_norm:
        raise ValueError("invalid_events")
    invalid_events = set(events_norm) - _WEBHOOK_EVENTS
    if invalid_events:
        raise ValueError("invalid_events")

    now = _now_iso()
    webhook_id = f"wh_{uuid4().hex[:10]}"
    secret = secrets.token_hex(32)
    webhook = {
        "id": webhook_id,
        "url": url_norm,
        "secret": secret,
        "events": events_norm,
        "enabled": bool(enabled),
        "created_at": now,
        "updated_at": now,
    }
    with _locked_store(write=True) as store:
        store.setdefault("webhooks", []).append(webhook)
    # Return with secret so caller can show it once
    return dict(webhook)


def update_webhook(
    *,
    webhook_id: str,
    url: str | None = None,
    events: list[str] | None = None,
    enabled: bool | None = None,
) -> dict[str, Any]:
    """Update a webhook. Returns the webhook with secret redacted."""
    now = _now_iso()
    with _locked_store(write=True) as store:
        webhooks = store.get("webhooks", [])
        for webhook in webhooks:
            if webhook.get("id") != webhook_id:
                continue
            if url is not None:
                url_norm = url.strip()
                if not url_norm:
                    raise ValueError("invalid_url")
                if len(url_norm) > _WEBHOOK_MAX_URL_LENGTH:
                    raise ValueError("invalid_url")
                if not url_norm.startswith(("http://", "https://")):
                    raise ValueError("invalid_url")
                webhook["url"] = url_norm
            if events is not None:
                events_norm = sorted({e.strip().lower() for e in events if e and e.strip()})
                if not events_norm:
                    raise ValueError("invalid_events")
                invalid = set(events_norm) - _WEBHOOK_EVENTS
                if invalid:
                    raise ValueError("invalid_events")
                webhook["events"] = events_norm
            if enabled is not None:
                webhook["enabled"] = bool(enabled)
            webhook["updated_at"] = now
            return _redact_webhook(_normalize_webhook_record(webhook))
    raise ValueError("not_found")


def delete_webhook(*, webhook_id: str) -> None:
    """Delete a webhook by ID."""
    with _locked_store(write=True) as store:
        webhooks = store.get("webhooks", [])
        remaining = [item for item in webhooks if item.get("id") != webhook_id]
        if len(remaining) == len(webhooks):
            raise ValueError("not_found")
        store["webhooks"] = remaining


# ──────────────────────────────────────────────────────────────────────────────
# Webhook Delivery Log
# ──────────────────────────────────────────────────────────────────────────────

_WEBHOOK_DELIVERIES_CAP = 1000


def _get_webhook_with_secret(webhook_id: str) -> dict[str, Any]:
    """Return a webhook record including its secret. Raises ValueError if not found."""
    with _locked_store() as store:
        for webhook in store.get("webhooks", []):
            if webhook.get("id") == webhook_id:
                return dict(webhook)
    raise ValueError("not_found")


def record_webhook_delivery(
    *,
    webhook_id: str,
    event_type: str,
    status_code: int | None,
    response_time_ms: int | None,
    success: bool,
    error: str | None = None,
    payload_preview: str | None = None,
) -> dict[str, Any]:
    """Append a delivery record for a webhook. Prunes oldest entries beyond cap."""
    now = _now_iso()
    record = {
        "id": f"wd_{uuid4().hex[:12]}",
        "webhook_id": str(webhook_id),
        "event_type": str(event_type or ""),
        "status_code": int(status_code) if status_code is not None else None,
        "response_time_ms": int(response_time_ms) if response_time_ms is not None else None,
        "success": bool(success),
        "error": str(error)[:500] if error else None,
        "attempted_at": now,
        "payload_preview": str(payload_preview)[:500] if payload_preview else None,
    }
    with _locked_store(write=True) as store:
        deliveries = store.setdefault("webhook_deliveries", [])
        deliveries.append(record)
        # Keep only the most recent entries per webhook (prune oldest)
        wh_deliveries = [d for d in deliveries if d.get("webhook_id") == webhook_id]
        if len(wh_deliveries) > _WEBHOOK_DELIVERIES_CAP:
            excess = len(wh_deliveries) - _WEBHOOK_DELIVERIES_CAP
            # Remove the oldest excess entries for this webhook
            removed = 0
            pruned: list[dict[str, Any]] = []
            for d in deliveries:
                if d.get("webhook_id") == webhook_id and removed < excess:
                    removed += 1
                    continue
                pruned.append(d)
            store["webhook_deliveries"] = pruned
    return record


def list_webhook_deliveries(
    *,
    webhook_id: str,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[dict[str, Any]], int]:
    """Return delivery records for a webhook, newest first."""
    limit = max(1, min(limit, _WEBHOOK_DELIVERIES_CAP))
    offset = max(0, int(offset))
    with _locked_store() as store:
        deliveries = store.get("webhook_deliveries", [])
        wh_deliveries = [d for d in deliveries if d.get("webhook_id") == webhook_id]
    # Sort newest first
    wh_deliveries.sort(key=lambda d: d.get("attempted_at") or "", reverse=True)
    total = len(wh_deliveries)
    return wh_deliveries[offset: offset + limit], total


def _create_webhook_http_client(*, timeout: float):
    from tldw_Server_API.app.core.http_client import create_client

    return create_client(timeout=timeout, http2=False)


def send_test_webhook(*, webhook_id: str) -> dict[str, Any]:
    """Send a test payload to a webhook URL with HMAC signing and record the delivery.

    Returns the delivery record.
    """
    webhook = _get_webhook_with_secret(webhook_id)
    url = webhook.get("url", "")
    secret = webhook.get("secret", "")

    test_payload = {
        "event": "webhook.test",
        "webhook_id": webhook_id,
        "timestamp": _now_iso(),
        "data": {"message": "This is a test delivery from the admin panel."},
    }

    body_bytes = json.dumps(test_payload, separators=(",", ":")).encode("utf-8")

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if secret:
        sig = hmac.new(secret.encode("utf-8"), body_bytes, hashlib.sha256).hexdigest()
        headers["X-Webhook-Signature"] = f"sha256={sig}"

    status_code: int | None = None
    success = False
    error_msg: str | None = None
    start = time.monotonic()

    try:
        with _create_webhook_http_client(timeout=10.0) as client:
            resp = client.post(url, content=body_bytes, headers=headers)
            status_code = resp.status_code
            success = 200 <= resp.status_code < 300
            if not success:
                error_msg = f"HTTP {resp.status_code}"
    except _SYSTEM_OPS_NONCRITICAL_EXCEPTIONS as exc:
        error_msg = str(exc)[:500]
    except Exception as exc:
        error_msg = str(exc)[:500]

    elapsed_ms = int((time.monotonic() - start) * 1000)

    delivery = record_webhook_delivery(
        webhook_id=webhook_id,
        event_type="webhook.test",
        status_code=status_code,
        response_time_ms=elapsed_ms,
        success=success,
        error=error_msg,
        payload_preview=json.dumps(test_payload)[:500],
    )
    return delivery


# ──────────────────────────────────────────────────────────────────────────────
# User Invitations
# ──────────────────────────────────────────────────────────────────────────────

_INVITATION_STATUSES = {"pending", "accepted", "expired", "revoked"}
_INVITATION_ROLES = {"user", "admin", "service", "viewer"}
_INVITATION_DEFAULT_EXPIRY_DAYS = 7
_INVITATION_MAX_PENDING = 200


def _normalize_invitation_record(value: Any) -> dict[str, Any]:
    """Normalize and validate an invitation record."""
    if not isinstance(value, dict):
        raise ValueError("invalid_invitation")
    invitation = dict(value)
    invitation.setdefault("id", uuid4().hex[:16])
    invitation.setdefault("status", "pending")
    invitation.setdefault("created_at", _now_iso())
    invitation.setdefault("accepted_at", None)
    invitation.setdefault("email_sent", False)
    invitation.setdefault("email_error", None)
    invitation.setdefault("resend_count", 0)
    invitation.setdefault("last_resent_at", None)
    return invitation


def list_invitations(
    *,
    status: str | None = None,
) -> list[dict[str, Any]]:
    """List all user invitations, optionally filtered by status."""
    with _locked_store() as store:
        invitations_raw = list(store.get("invitations", []))

    invitations = []
    now = datetime.now(timezone.utc)
    for inv in invitations_raw:
        try:
            record = _normalize_invitation_record(inv)
        except ValueError:
            continue
        # Auto-expire pending invitations past their expiry date
        if record["status"] == "pending":
            expires_at = record.get("expires_at")
            if expires_at:
                expiry_dt = _parse_iso(expires_at)
                if expiry_dt < now:
                    record["status"] = "expired"
        invitations.append(record)

    if status:
        status_norm = status.strip().lower()
        if status_norm in _INVITATION_STATUSES:
            invitations = [inv for inv in invitations if inv.get("status") == status_norm]

    invitations.sort(key=lambda item: item.get("created_at") or "", reverse=True)
    return invitations


def create_invitation(
    *,
    email: str,
    role: str = "user",
    invited_by: str | None = None,
    expiry_days: int = _INVITATION_DEFAULT_EXPIRY_DAYS,
) -> dict[str, Any]:
    """Create a new user invitation."""
    email_norm = (email or "").strip().lower()
    if not email_norm or "@" not in email_norm:
        raise ValueError("invalid_email")

    role_norm = (role or "user").strip().lower()
    if role_norm not in _INVITATION_ROLES:
        raise ValueError("invalid_role")

    if expiry_days < 1 or expiry_days > 365:
        expiry_days = _INVITATION_DEFAULT_EXPIRY_DAYS

    token = secrets.token_urlsafe(32)
    now = _now_iso()
    now_dt = datetime.now(timezone.utc)
    expires_at = (now_dt + timedelta(days=expiry_days)).isoformat()

    invitation = {
        "id": uuid4().hex[:16],
        "email": email_norm,
        "role": role_norm,
        "status": "pending",
        "token": token,
        "invited_by": invited_by,
        "created_at": now,
        "expires_at": expires_at,
        "accepted_at": None,
        "email_sent": False,
        "email_error": None,
    }

    with _locked_store(write=True) as store:
        invitations = store.get("invitations", [])

        # Check for duplicate pending invitation to same email
        for existing in invitations:
            if existing.get("email") == email_norm and existing.get("status") == "pending":
                expires_at_existing = existing.get("expires_at")
                if expires_at_existing:
                    expiry_dt = _parse_iso(expires_at_existing)
                    if expiry_dt > now_dt:
                        raise ValueError("duplicate_pending_invitation")

        # Cap total pending invitations
        pending_count = sum(1 for inv in invitations if inv.get("status") == "pending")
        if pending_count >= _INVITATION_MAX_PENDING:
            raise ValueError("too_many_pending_invitations")

        invitations.append(invitation)
        store["invitations"] = invitations

    return invitation


def get_invitation_by_token(*, token: str) -> dict[str, Any] | None:
    """Look up an invitation by its token."""
    with _locked_store() as store:
        for inv in store.get("invitations", []):
            if inv.get("token") == token:
                return _normalize_invitation_record(inv)
    return None


def update_invitation_email_status(
    *,
    invitation_id: str,
    email_sent: bool,
    email_error: str | None = None,
) -> dict[str, Any] | None:
    """Update the email delivery status for an invitation."""
    with _locked_store(write=True) as store:
        invitations = store.get("invitations", [])
        for inv in invitations:
            if inv.get("id") == invitation_id:
                inv["email_sent"] = email_sent
                inv["email_error"] = email_error
                return _normalize_invitation_record(inv)
    return None


def revoke_invitation(*, invitation_id: str) -> dict[str, Any]:
    """Revoke a pending invitation."""
    with _locked_store(write=True) as store:
        invitations = store.get("invitations", [])
        for inv in invitations:
            if inv.get("id") == invitation_id:
                if inv.get("status") != "pending":
                    raise ValueError("not_pending")
                inv["status"] = "revoked"
                return _normalize_invitation_record(inv)
    raise ValueError("not_found")


def accept_invitation(*, invitation_id: str) -> dict[str, Any]:
    """Mark an invitation as accepted."""
    with _locked_store(write=True) as store:
        invitations = store.get("invitations", [])
        for inv in invitations:
            if inv.get("id") == invitation_id:
                if inv.get("status") != "pending":
                    raise ValueError("not_pending")
                inv["status"] = "accepted"
                inv["accepted_at"] = _now_iso()
                return _normalize_invitation_record(inv)
    raise ValueError("not_found")


# ---------------------------------------------------------------------------
# Dependency Health History
# ---------------------------------------------------------------------------

_HEALTH_HISTORY_RETENTION_DAYS = 90
_HEALTH_HISTORY_MAX_PER_DEPENDENCY = 10_000
_HEALTH_HISTORY_DEDUP_SECONDS = 3600  # hourly granularity


def _prune_health_history(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove entries older than retention and cap per-dependency."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=_HEALTH_HISTORY_RETENTION_DAYS)
    cutoff_iso = cutoff.isoformat()

    # First pass: drop expired entries
    fresh = [e for e in entries if (e.get("checked_at") or "") >= cutoff_iso]

    # Second pass: cap per dependency (keep newest)
    from collections import Counter

    counts: Counter[str] = Counter()
    for entry in fresh:
        counts[entry.get("dependency_name", "")] += 1

    over_limit = {name for name, count in counts.items() if count > _HEALTH_HISTORY_MAX_PER_DEPENDENCY}
    if not over_limit:
        return fresh

    # For over-limit deps, sort by checked_at descending and keep only the newest
    result: list[dict[str, Any]] = []
    kept: Counter[str] = Counter()
    for entry in reversed(fresh):
        dep_name = entry.get("dependency_name", "")
        if dep_name in over_limit:
            if kept[dep_name] >= _HEALTH_HISTORY_MAX_PER_DEPENDENCY:
                continue
            kept[dep_name] += 1
        result.append(entry)
    result.reverse()
    return result


def record_health_snapshot(results: list[dict[str, Any]]) -> int:
    """Append health check results to the history store.

    Each item in *results* should have at least ``name``, ``status``, and
    ``latency_ms`` keys (the shape returned by ``_check_dep`` in admin_ops).

    Returns the number of entries actually recorded (skipping duplicates
    that would violate hourly dedup).
    """
    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()
    recorded = 0

    with _locked_store(write=True) as store:
        history: list[dict[str, Any]] = store.get("dependency_health_history", [])

        # Build a lookup of the latest checked_at per dependency for dedup
        latest_by_dep: dict[str, str] = {}
        for entry in reversed(history):
            dep_name = entry.get("dependency_name", "")
            if dep_name not in latest_by_dep:
                latest_by_dep[dep_name] = entry.get("checked_at", "")

        for item in results:
            dep_name = str(item.get("name", "")).strip()
            if not dep_name:
                continue

            # Dedup: skip if last entry for this dep was within the dedup window
            last_ts = latest_by_dep.get(dep_name)
            if last_ts:
                try:
                    last_dt = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
                    if (now - last_dt).total_seconds() < _HEALTH_HISTORY_DEDUP_SECONDS:
                        continue
                except (ValueError, TypeError):
                    pass

            entry = {
                "dependency_name": dep_name,
                "status": str(item.get("status", "unknown")),
                "latency_ms": item.get("latency_ms"),
                "checked_at": now_iso,
            }
            history.append(entry)
            latest_by_dep[dep_name] = now_iso
            recorded += 1

        history = _prune_health_history(history)
        store["dependency_health_history"] = history

    return recorded


def get_uptime_stats(dependency_name: str, days: int = 30) -> dict[str, Any]:
    """Compute uptime statistics for a single dependency over the given window.

    Returns a dict with:
      - dependency_name, days, total_checks, healthy_checks, uptime_pct,
        avg_latency_ms, downtime_minutes, sparkline (hourly 0/1 for last 7 days)
    """
    if days < 1:
        days = 1
    if days > 90:
        days = 90

    now = datetime.now(timezone.utc)
    window_start = now - timedelta(days=days)
    window_start_iso = window_start.isoformat()

    with _locked_store() as store:
        history: list[dict[str, Any]] = store.get("dependency_health_history", [])

    # Filter to the requested dependency and time window
    entries = [
        e
        for e in history
        if e.get("dependency_name") == dependency_name and (e.get("checked_at") or "") >= window_start_iso
    ]

    total_checks = len(entries)
    healthy_checks = sum(1 for e in entries if e.get("status") == "healthy")

    uptime_pct = round((healthy_checks / total_checks) * 100, 2) if total_checks > 0 else 100.0

    latencies = [e["latency_ms"] for e in entries if isinstance(e.get("latency_ms"), (int, float))]
    avg_latency_ms = round(sum(latencies) / len(latencies), 1) if latencies else 0.0

    # Estimate downtime in minutes: each unhealthy check represents ~60 min (hourly granularity)
    unhealthy_checks = total_checks - healthy_checks
    downtime_minutes = unhealthy_checks * 60

    # Sparkline: hourly slots for the last 7 days (168 values)
    sparkline_hours = min(days, 7) * 24
    sparkline_start = now - timedelta(hours=sparkline_hours)
    slots: list[int | None] = [None] * sparkline_hours  # None = no data, 1 = healthy, 0 = not

    for e in entries:
        ts_str = e.get("checked_at", "")
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            continue
        if ts < sparkline_start:
            continue
        slot_index = int((ts - sparkline_start).total_seconds() / 3600)
        if 0 <= slot_index < sparkline_hours:
            is_healthy = 1 if e.get("status") == "healthy" else 0
            # If multiple checks in one hour, degrade wins (take worst)
            if slots[slot_index] is None:
                slots[slot_index] = is_healthy
            else:
                slots[slot_index] = min(slots[slot_index], is_healthy)

    # Replace None (no data) with 1 (assume healthy if no check was done)
    sparkline = [s if s is not None else 1 for s in slots]

    return {
        "dependency_name": dependency_name,
        "days": days,
        "total_checks": total_checks,
        "healthy_checks": healthy_checks,
        "uptime_pct": uptime_pct,
        "avg_latency_ms": avg_latency_ms,
        "downtime_minutes": downtime_minutes,
        "sparkline": sparkline,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Email Delivery Log
# ──────────────────────────────────────────────────────────────────────────────

_EMAIL_DELIVERY_LOG_CAP = 5000


def record_email_delivery(
    *,
    recipient: str,
    subject: str,
    template: str | None = None,
    status: str,  # "sent", "failed", "skipped"
    error: str | None = None,
) -> dict[str, Any]:
    """Record an email delivery attempt.

    Appends to the ``email_delivery_log`` list in the system-ops store.
    The log is capped at ``_EMAIL_DELIVERY_LOG_CAP`` entries (oldest pruned first).
    """
    entry: dict[str, Any] = {
        "id": f"edl_{uuid4().hex[:10]}",
        "recipient": str(recipient or ""),
        "subject": str(subject or ""),
        "template": str(template) if template else None,
        "status": str(status or "sent"),
        "error": str(error)[:500] if error else None,
        "sent_at": _now_iso(),
    }
    with _locked_store(write=True) as store:
        log = store.setdefault("email_delivery_log", [])
        log.append(entry)
        if len(log) > _EMAIL_DELIVERY_LOG_CAP:
            store["email_delivery_log"] = log[-_EMAIL_DELIVERY_LOG_CAP:]
    return entry


def list_email_deliveries(
    *,
    limit: int = 50,
    offset: int = 0,
    status: str | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """List email delivery log entries, newest first.

    Returns ``(items, total)`` where *total* is the count after any status
    filter (before offset/limit slicing).
    """
    limit = max(1, min(limit, 500))
    offset = max(0, offset)
    with _locked_store() as store:
        log: list[dict[str, Any]] = store.get("email_delivery_log", [])
    # Filter by status if requested
    if status:
        log = [entry for entry in log if entry.get("status") == status]
    total = len(log)
    # Newest first. Use append position as a deterministic tie-breaker for
    # platforms where rapid records can share identical timestamp strings.
    ordered = sorted(
        enumerate(log),
        key=lambda item: ((item[1].get("sent_at") or ""), item[0]),
        reverse=True,
    )
    log = [entry for _, entry in ordered]
    return log[offset : offset + limit], total


# ──────────────────────────────────────────────────────────────────────────────
# Compliance Report Schedules
# ──────────────────────────────────────────────────────────────────────────────

_REPORT_FREQUENCIES = {"daily", "weekly", "monthly"}
_REPORT_FORMATS = {"html", "json"}
_REPORT_MAX_RECIPIENTS = 20
_REPORT_MAX_SCHEDULES = 50


def _normalize_report_schedule(value: Any) -> dict[str, Any]:
    """Normalize and validate a compliance report schedule record."""
    if not isinstance(value, dict):
        raise ValueError("invalid_report_schedule")
    schedule = dict(value)
    schedule.setdefault("id", uuid4().hex[:16])
    schedule.setdefault("frequency", "weekly")
    schedule.setdefault("recipients", [])
    schedule.setdefault("format", "html")
    schedule.setdefault("enabled", True)
    schedule.setdefault("created_at", _now_iso())
    schedule.setdefault("last_sent_at", None)
    return schedule


def list_report_schedules() -> list[dict[str, Any]]:
    """List all compliance report schedules."""
    with _locked_store() as store:
        raw = list(store.get("compliance_report_schedules", []))
    schedules = []
    for item in raw:
        try:
            schedules.append(_normalize_report_schedule(item))
        except ValueError:
            continue
    schedules.sort(key=lambda s: s.get("created_at") or "", reverse=True)
    return schedules


def create_report_schedule(
    *,
    frequency: str,
    recipients: list[str],
    report_format: str = "html",
    enabled: bool = True,
) -> dict[str, Any]:
    """Create a new compliance report schedule."""
    freq_norm = (frequency or "").strip().lower()
    if freq_norm not in _REPORT_FREQUENCIES:
        raise ValueError("invalid_frequency")

    fmt_norm = (report_format or "html").strip().lower()
    if fmt_norm not in _REPORT_FORMATS:
        raise ValueError("invalid_format")

    if not isinstance(recipients, list) or len(recipients) == 0:
        raise ValueError("recipients_required")
    # Validate and normalize email recipients
    clean_recipients: list[str] = []
    for r in recipients[:_REPORT_MAX_RECIPIENTS]:
        email = str(r).strip().lower()
        if "@" not in email:
            raise ValueError("invalid_recipient_email")
        clean_recipients.append(email)

    schedule = {
        "id": uuid4().hex[:16],
        "frequency": freq_norm,
        "recipients": clean_recipients,
        "format": fmt_norm,
        "enabled": bool(enabled),
        "created_at": _now_iso(),
        "last_sent_at": None,
    }

    with _locked_store(write=True) as store:
        schedules = store.get("compliance_report_schedules", [])
        if len(schedules) >= _REPORT_MAX_SCHEDULES:
            raise ValueError("too_many_report_schedules")
        schedules.append(schedule)
        store["compliance_report_schedules"] = schedules

    return schedule


def update_report_schedule(
    *,
    schedule_id: str,
    frequency: str | None = None,
    recipients: list[str] | None = None,
    report_format: str | None = None,
    enabled: bool | None = None,
) -> dict[str, Any]:
    """Update an existing compliance report schedule."""
    with _locked_store(write=True) as store:
        schedules = store.get("compliance_report_schedules", [])
        for sched in schedules:
            if sched.get("id") == schedule_id:
                if frequency is not None:
                    freq_norm = frequency.strip().lower()
                    if freq_norm not in _REPORT_FREQUENCIES:
                        raise ValueError("invalid_frequency")
                    sched["frequency"] = freq_norm
                if recipients is not None:
                    if not isinstance(recipients, list) or len(recipients) == 0:
                        raise ValueError("recipients_required")
                    clean: list[str] = []
                    for r in recipients[:_REPORT_MAX_RECIPIENTS]:
                        email = str(r).strip().lower()
                        if "@" not in email:
                            raise ValueError("invalid_recipient_email")
                        clean.append(email)
                    sched["recipients"] = clean
                if report_format is not None:
                    fmt_norm = report_format.strip().lower()
                    if fmt_norm not in _REPORT_FORMATS:
                        raise ValueError("invalid_format")
                    sched["format"] = fmt_norm
                if enabled is not None:
                    sched["enabled"] = bool(enabled)
                return _normalize_report_schedule(sched)
    raise ValueError("not_found")


def delete_report_schedule(*, schedule_id: str) -> dict[str, Any]:
    """Delete a compliance report schedule."""
    with _locked_store(write=True) as store:
        schedules = store.get("compliance_report_schedules", [])
        for i, sched in enumerate(schedules):
            if sched.get("id") == schedule_id:
                removed = schedules.pop(i)
                store["compliance_report_schedules"] = schedules
                return _normalize_report_schedule(removed)
    raise ValueError("not_found")


def mark_report_schedule_sent(*, schedule_id: str) -> dict[str, Any]:
    """Update the last_sent_at timestamp for a report schedule."""
    with _locked_store(write=True) as store:
        schedules = store.get("compliance_report_schedules", [])
        for sched in schedules:
            if sched.get("id") == schedule_id:
                sched["last_sent_at"] = _now_iso()
                return _normalize_report_schedule(sched)
    raise ValueError("not_found")


# ──────────────────────────────────────────────────────────────────────────────
# Digest Preferences
# ──────────────────────────────────────────────────────────────────────────────

_DIGEST_FREQUENCIES = {"daily", "weekly", "off"}


def get_digest_preference(*, user_id: str) -> dict[str, Any] | None:
    """Get email digest preference for a user."""
    user_id_norm = str(user_id).strip()
    with _locked_store() as store:
        prefs = store.get("digest_preferences", [])
        for pref in prefs:
            if str(pref.get("user_id", "")).strip() == user_id_norm:
                return dict(pref)
    return None


def set_digest_preference(
    *,
    user_id: str,
    email: str,
    frequency: str = "off",
) -> dict[str, Any]:
    """Set or update the email digest preference for a user."""
    user_id_norm = str(user_id).strip()
    email_norm = str(email).strip().lower()
    if not email_norm or "@" not in email_norm:
        raise ValueError("invalid_email")

    freq_norm = (frequency or "off").strip().lower()
    if freq_norm not in _DIGEST_FREQUENCIES:
        raise ValueError("invalid_frequency")

    with _locked_store(write=True) as store:
        prefs = store.get("digest_preferences", [])
        for pref in prefs:
            if str(pref.get("user_id", "")).strip() == user_id_norm:
                pref["email"] = email_norm
                pref["frequency"] = freq_norm
                pref["enabled"] = freq_norm != "off"
                return dict(pref)
        # Create new entry
        new_pref = {
            "id": uuid4().hex[:16],
            "user_id": user_id_norm,
            "email": email_norm,
            "frequency": freq_norm,
            "enabled": freq_norm != "off",
            "created_at": _now_iso(),
        }
        prefs.append(new_pref)
        store["digest_preferences"] = prefs
        return dict(new_pref)


# ──────────────────────────────────────────────────────────────────────────────
# Resend Invitation
# ──────────────────────────────────────────────────────────────────────────────

_INVITATION_MAX_RESENDS = 3


def resend_invitation(*, invitation_id: str) -> dict[str, Any]:
    """Regenerate token and update expiry for a pending invitation.

    Rate-limited to ``_INVITATION_MAX_RESENDS`` resends per invitation.
    Returns the updated invitation record.

    Raises:
        ValueError: ``not_found`` if no invitation with that id exists.
        ValueError: ``not_pending`` if the invitation is not in pending status.
        ValueError: ``resend_limit_reached`` if max resends exceeded.
    """
    with _locked_store(write=True) as store:
        invitations = store.get("invitations", [])
        for inv in invitations:
            if inv.get("id") == invitation_id:
                # Check auto-expiry
                if inv.get("status") == "pending":
                    expires_at = inv.get("expires_at")
                    if expires_at:
                        expiry_dt = _parse_iso(expires_at)
                        if expiry_dt < datetime.now(timezone.utc):
                            inv["status"] = "expired"

                if inv.get("status") != "pending":
                    raise ValueError("not_pending")

                resend_count = int(inv.get("resend_count") or 0)
                if resend_count >= _INVITATION_MAX_RESENDS:
                    raise ValueError("resend_limit_reached")

                # Regenerate token and extend expiry
                inv["token"] = secrets.token_urlsafe(32)
                inv["expires_at"] = (
                    datetime.now(timezone.utc) + timedelta(days=_INVITATION_DEFAULT_EXPIRY_DAYS)
                ).isoformat()
                inv["resend_count"] = resend_count + 1
                inv["last_resent_at"] = _now_iso()
                # Reset email status so the caller can attempt re-delivery
                inv["email_sent"] = False
                inv["email_error"] = None

                return _normalize_invitation_record(inv)
    raise ValueError("not_found")


#######################################################################################################################
#
# Per-API-Key Usage Attribution
#
# Stores usage counters (request count, tokens, cost) per API key in the
# JSON ops store. Daily snapshots are capped at 90 entries per key.

_API_KEY_USAGE_DAILY_CAP = 90


def _default_key_usage(key_id: str) -> dict[str, Any]:
    return {
        "key_id": str(key_id),
        "request_count": 0,
        "total_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "estimated_cost_usd": 0.0,
        "last_used_at": None,
        "daily_snapshots": [],
    }


def record_api_key_usage(
    key_id: str,
    *,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    cost_usd: float = 0.0,
) -> dict[str, Any]:
    """Increment usage counters for a single API key and add a daily snapshot entry.

    Returns the updated usage record.
    """
    key_id = str(key_id)
    total_tokens = prompt_tokens + completion_tokens
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    with _locked_store(write=True) as store:
        usage_map: dict[str, Any] = store.setdefault("api_key_usage", {})
        entry = usage_map.get(key_id)
        if entry is None:
            entry = _default_key_usage(key_id)
            usage_map[key_id] = entry

        entry["request_count"] = entry.get("request_count", 0) + 1
        entry["total_tokens"] = entry.get("total_tokens", 0) + total_tokens
        entry["prompt_tokens"] = entry.get("prompt_tokens", 0) + prompt_tokens
        entry["completion_tokens"] = entry.get("completion_tokens", 0) + completion_tokens
        entry["estimated_cost_usd"] = round(entry.get("estimated_cost_usd", 0.0) + cost_usd, 6)
        entry["last_used_at"] = _now_iso()

        # Update or append daily snapshot
        snapshots: list[dict[str, Any]] = entry.setdefault("daily_snapshots", [])
        if snapshots and snapshots[-1].get("date") == today:
            snap = snapshots[-1]
            snap["requests"] = snap.get("requests", 0) + 1
            snap["tokens"] = snap.get("tokens", 0) + total_tokens
            snap["cost_usd"] = round(snap.get("cost_usd", 0.0) + cost_usd, 6)
        else:
            snapshots.append(
                {
                    "date": today,
                    "requests": 1,
                    "tokens": total_tokens,
                    "cost_usd": round(cost_usd, 6),
                }
            )

        # Cap daily snapshots at 90 days
        if len(snapshots) > _API_KEY_USAGE_DAILY_CAP:
            entry["daily_snapshots"] = snapshots[-_API_KEY_USAGE_DAILY_CAP:]

        return dict(entry)


def get_api_key_usage(key_id: str) -> dict[str, Any]:
    """Return the usage summary for a single API key.

    Returns a default (zeroed) record if no usage has been recorded.
    """
    key_id = str(key_id)
    with _locked_store() as store:
        usage_map: dict[str, Any] = store.get("api_key_usage", {})
        entry = usage_map.get(key_id)
        if entry is None:
            return _default_key_usage(key_id)
        return dict(entry)


def list_api_key_usage(*, limit: int = 10) -> list[dict[str, Any]]:
    """Return top API keys ranked by total token consumption.

    Args:
        limit: Maximum number of entries to return (default 10).
    """
    with _locked_store() as store:
        usage_map: dict[str, Any] = store.get("api_key_usage", {})
        items = list(usage_map.values())

    items.sort(key=lambda item: item.get("total_tokens", 0), reverse=True)
    return items[: max(1, limit)]
