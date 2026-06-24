from __future__ import annotations

import asyncio
import json
import os
import sqlite3
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from html import escape
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.Collections.reading_service import ReadingService
from tldw_Server_API.app.core.Collections.utils import is_supported_reading_url
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.exceptions import ReadingDigestJobError
from tldw_Server_API.app.core.testing import is_truthy
from tldw_Server_API.app.services.outputs_service import (
    _build_output_filename,
    _outputs_dir_for_user,
    _resolve_output_path_for_user,
    build_items_context_from_content_items,
    render_output_template,
)

READING_DIGEST_DOMAIN = "reading"
READING_DIGEST_JOB_TYPE = "reading_digest"
LOGGER = logger.bind(module="reading_digest_jobs")
_READING_DIGEST_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = (
    AttributeError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    sqlite3.Error,
    json.JSONDecodeError,
)

def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip()
    if raw == "":
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        logger.warning("Invalid {}; using default={}", name, default)
        return default


READING_DIGEST_DEFAULT_LIMIT = _env_int("READING_DIGEST_DEFAULT_LIMIT", 50)
READING_DIGEST_MAX_LIMIT = _env_int("READING_DIGEST_MAX_LIMIT", 500)
READING_DIGEST_RETENTION_DAYS = _env_int("READING_DIGEST_RETENTION_DAYS", 30)
READING_DIGEST_SUGGESTIONS_DEFAULT_LIMIT = _env_int("READING_DIGEST_SUGGESTIONS_DEFAULT_LIMIT", 5)
READING_DIGEST_SUGGESTIONS_MAX_LIMIT = _env_int("READING_DIGEST_SUGGESTIONS_MAX_LIMIT", 50)
READING_DIGEST_SUGGESTIONS_MAX_CANDIDATES = _env_int("READING_DIGEST_SUGGESTIONS_MAX_CANDIDATES", 200)

_SUGGESTION_ALLOWED_STATUSES = {"saved", "reading", "read", "archived"}


def reading_digest_queue() -> str:
    queue = (os.getenv("READING_DIGEST_JOBS_QUEUE") or "reading-digest").strip()
    return queue or "reading-digest"


def _parse_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as exc:
            LOGGER.debug("Failed to parse reading digest payload {}: {}", payload, exc)
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _resolve_user_id(job: dict[str, Any], payload: dict[str, Any]) -> int:
    owner = job.get("owner_user_id") or payload.get("user_id")
    if owner is None or str(owner).strip() == "":
        return int(DatabasePaths.get_single_user_id())
    return int(owner)


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        LOGGER.debug("Failed to coerce int from {}: {}", value, exc)
        return default


def _parse_iso_datetime(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        LOGGER.debug("Failed to parse ISO datetime from {}: {}", raw, exc)
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _normalize_filters(filters: Any) -> dict[str, Any]:
    if isinstance(filters, dict):
        return filters
    return {}


def _normalize_tag_list(tags: Any) -> list[str]:
    if isinstance(tags, str):
        tags = [tags]
    if not isinstance(tags, list):
        return []
    return [str(tag).strip() for tag in tags if str(tag).strip()]


def _normalize_suggestions_config(filters: dict[str, Any]) -> dict[str, Any] | None:
    raw = filters.get("suggestions")
    if not isinstance(raw, dict):
        return None
    enabled = raw.get("enabled")
    if isinstance(enabled, str):
        enabled = is_truthy(enabled.strip())
    if not enabled:
        return None

    limit = _safe_int(raw.get("limit"), READING_DIGEST_SUGGESTIONS_DEFAULT_LIMIT)
    limit = max(1, min(READING_DIGEST_SUGGESTIONS_MAX_LIMIT, limit))

    include_read = raw.get("include_read", False)
    if isinstance(include_read, str):
        include_read = is_truthy(include_read.strip())
    else:
        include_read = bool(include_read)

    include_archived = raw.get("include_archived", False)
    if isinstance(include_archived, str):
        include_archived = is_truthy(include_archived.strip())
    else:
        include_archived = bool(include_archived)

    statuses = raw.get("status")
    if isinstance(statuses, str):
        statuses = [statuses]
    statuses = [s for s in statuses if s in _SUGGESTION_ALLOWED_STATUSES] if isinstance(statuses, list) else []
    if not statuses:
        statuses = ["saved", "reading"]

    if not include_read:
        statuses = [s for s in statuses if s != "read"]
    elif "read" not in statuses:
        statuses.append("read")

    if not include_archived:
        statuses = [s for s in statuses if s != "archived"]
    elif "archived" not in statuses:
        statuses.append("archived")

    if not statuses:
        statuses = ["saved", "reading"]

    exclude_tags = _normalize_tag_list(raw.get("exclude_tags"))
    max_age_days = raw.get("max_age_days")
    if max_age_days is not None:
        max_age_days = _safe_int(max_age_days, 0)
    if max_age_days is not None and max_age_days <= 0:
        max_age_days = None

    return {
        "enabled": True,
        "limit": limit,
        "status": statuses,
        "exclude_tags": exclude_tags,
        "max_age_days": max_age_days,
        "include_read": include_read,
        "include_archived": include_archived,
    }


def _recency_score(when: str | None, now: datetime) -> float:
    parsed = _parse_iso_datetime(when)
    if parsed is None:
        return 0.0
    age_days = max(0.0, (now - parsed).total_seconds() / 86400.0)
    if age_days >= 30.0:
        return 0.0
    return (30.0 - age_days) / 30.0


def _score_suggestion_candidate(
    row: Any,
    digest_tags: set[str],
    now: datetime,
) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []

    recency = _recency_score(getattr(row, "updated_at", None) or getattr(row, "created_at", None), now)
    if recency:
        score += recency
        if recency >= 0.5:
            reasons.append("recent")

    if bool(getattr(row, "favorite", False)):
        score += 0.6
        reasons.append("favorite")

    status = str(getattr(row, "status", "") or "").lower()
    status_bonus = {"reading": 0.4, "saved": 0.25, "read": 0.1, "archived": 0.0}.get(status, 0.0)
    if status_bonus:
        score += status_bonus
        reasons.append(f"status:{status}")

    tags = getattr(row, "tags", None)
    if not isinstance(tags, list):
        tags = []
    normalized_tags = {str(tag).strip().lower() for tag in tags if str(tag).strip()}
    overlap = len(digest_tags.intersection(normalized_tags))
    if overlap:
        overlap_bonus = min(1.0, 0.25 * overlap)
        score += overlap_bonus
        reasons.append(f"tag_overlap:{overlap}")

    word_count = getattr(row, "word_count", None)
    if isinstance(word_count, int) and word_count >= 8000:
        score -= 0.2
        reasons.append("long_read")

    return score, reasons


def _select_suggestion_candidates(
    *,
    service: ReadingService,
    digest_rows: Iterable[Any],
    digest_tags: list[str],
    suggestions_config: dict[str, Any],
) -> tuple[list[Any], dict[str, Any]]:
    digest_item_ids = {getattr(row, "id", None) for row in digest_rows if getattr(row, "id", None) is not None}
    normalized_digest_tags = {tag.strip().lower() for tag in digest_tags if isinstance(tag, str) and tag.strip()}
    exclude_tags = {tag.strip().lower() for tag in suggestions_config.get("exclude_tags", [])}
    max_age_days = suggestions_config.get("max_age_days")
    now = datetime.now(tz=timezone.utc)

    candidate_limit = READING_DIGEST_SUGGESTIONS_MAX_CANDIDATES
    statuses = suggestions_config.get("status")
    if not isinstance(statuses, list):
        statuses = ["saved", "reading"]

    rows, _total = service.list_items(
        status=statuses,
        page=1,
        size=candidate_limit,
        offset=0,
        limit=candidate_limit,
        sort="updated_desc",
    )

    filtered: list[Any] = []
    for row in rows:
        row_id = getattr(row, "id", None)
        if row_id is not None and row_id in digest_item_ids:
            continue
        tags = getattr(row, "tags", None)
        if not isinstance(tags, list):
            tags = []
        normalized_tags = {str(tag).strip().lower() for tag in tags if str(tag).strip()}
        if exclude_tags and normalized_tags.intersection(exclude_tags):
            continue
        if max_age_days:
            updated = _parse_iso_datetime(getattr(row, "updated_at", None) or getattr(row, "created_at", None))
            if updated and (now - updated).days > int(max_age_days):
                continue
        filtered.append(row)

    scored: list[tuple[float, Any, list[str]]] = []
    for row in filtered:
        score, reasons = _score_suggestion_candidate(row, normalized_digest_tags, now)
        scored.append((score, row, reasons))

    def _sort_key(entry: tuple[float, Any, list[str]]) -> tuple[float, float, int]:
        score, row, _reasons = entry
        updated = _parse_iso_datetime(getattr(row, "updated_at", None) or getattr(row, "created_at", None))
        timestamp = updated.timestamp() if updated else 0.0
        return (score, timestamp, int(getattr(row, "id", 0) or 0))

    scored.sort(key=_sort_key, reverse=True)
    limit = suggestions_config.get("limit") or 0
    selected = scored[: int(limit)] if limit else []

    selected_rows = [row for _score, row, _reasons in selected]
    scores = {int(getattr(row, "id", 0) or 0): round(score, 4) for score, row, _reasons in selected}
    reasons = {int(getattr(row, "id", 0) or 0): rs for _score, row, rs in selected}
    meta = {"count": len(selected_rows), "scores": scores, "reasons": reasons}
    return selected_rows, meta


def _resolve_retention_until(retention_days: int | None) -> str | None:
    days = retention_days
    if days is None:
        days = max(0, int(READING_DIGEST_RETENTION_DAYS))
    if days <= 0:
        return None
    return (datetime.now(tz=timezone.utc) + timedelta(days=days)).isoformat()


def _render_default_markdown(
    title: str,
    items: list[dict[str, Any]],
    suggestions: list[dict[str, Any]] | None = None,
) -> str:
    lines = [f"# {title}", ""]
    for idx, itm in enumerate(items, 1):
        entry_title = itm.get("title") or f"Item {idx}"
        url = str(itm.get("url") or "").strip()
        line = f"{idx}. [{entry_title}]({url})" if url else f"{idx}. {entry_title}"
        if url and not is_supported_reading_url(url):
            line = f"{idx}. {entry_title}"
        summary = itm.get("summary") or ""
        if summary:
            line += f" - {summary}"
        lines.append(line)
    if suggestions:
        lines.extend(["", "## Suggested reads", ""])
        for idx, itm in enumerate(suggestions, 1):
            entry_title = itm.get("title") or f"Suggestion {idx}"
            url = str(itm.get("url") or "").strip()
            line = f"{idx}. [{entry_title}]({url})" if url else f"{idx}. {entry_title}"
            if url and not is_supported_reading_url(url):
                line = f"{idx}. {entry_title}"
            summary = itm.get("summary") or ""
            if summary:
                line += f" - {summary}"
            lines.append(line)
    return "\n".join(lines)


def _render_default_html(
    title: str,
    items: list[dict[str, Any]],
    suggestions: list[dict[str, Any]] | None = None,
) -> str:
    body_parts = [f"<h1>{escape(title)}</h1>", "<ol>"]
    for idx, itm in enumerate(items, 1):
        entry_title = escape(itm.get("title") or f"Item {idx}")
        summary = escape(itm.get("summary") or "")
        url = str(itm.get("url") or "").strip()
        entry = f'<li><a href="{escape(url)}">{entry_title}</a>' if is_supported_reading_url(url) else f"<li>{entry_title}"
        if summary:
            entry += f" - {summary}"
        entry += "</li>"
        body_parts.append(entry)
    body_parts.append("</ol>")
    if suggestions:
        body_parts.extend(["<h2>Suggested reads</h2>", "<ol>"])
        for idx, itm in enumerate(suggestions, 1):
            entry_title = escape(itm.get("title") or f"Suggestion {idx}")
            summary = escape(itm.get("summary") or "")
            url = str(itm.get("url") or "").strip()
            entry = f'<li><a href="{escape(url)}">{entry_title}</a>' if is_supported_reading_url(url) else f"<li>{entry_title}"
            if summary:
                entry += f" - {summary}"
            entry += "</li>"
            body_parts.append(entry)
        body_parts.append("</ol>")
    return "\n".join(body_parts)


def _build_reading_items_context(rows: list[Any]) -> list[dict[str, Any]]:
    items = build_items_context_from_content_items(rows)
    for idx, row in enumerate(rows):
        try:
            items[idx]["status"] = getattr(row, "status", None)
            items[idx]["favorite"] = bool(getattr(row, "favorite", False))
            items[idx]["notes"] = getattr(row, "notes", None) or ""
            items[idx]["read_at"] = getattr(row, "read_at", None)
            items[idx]["updated_at"] = getattr(row, "updated_at", None)
        except _READING_DIGEST_NONCRITICAL_EXCEPTIONS as exc:
            row_id = None
            try:
                row_id = getattr(row, "id", None) or getattr(row, "pk", None)
                if row_id is None and isinstance(row, dict):
                    row_id = row.get("id") or row.get("pk")
            except _READING_DIGEST_NONCRITICAL_EXCEPTIONS:
                row_id = None
            logger.debug(
                "reading_digest: failed to enrich item context idx={} row_id={}: {}",
                idx,
                row_id,
                exc,
                exc_info=True,
            )
            continue
    return items


async def handle_reading_digest_job(job: dict[str, Any]) -> dict[str, Any]:
    payload = _parse_payload(job.get("payload"))
    user_id = _resolve_user_id(job, payload)
    schedule_id = payload.get("schedule_id")
    if not schedule_id:
        raise ReadingDigestJobError("reading_digest_missing_schedule", retryable=False)

    collections_db = CollectionsDatabase.for_user(user_id)
    try:
        schedule = collections_db.get_reading_digest_schedule(str(schedule_id))
    except KeyError as exc:
        raise ReadingDigestJobError("reading_digest_schedule_not_found", retryable=False) from exc

    if not schedule.enabled:
        collections_db.set_reading_digest_history(schedule.id, last_status="skipped_disabled")
        return {"status": "skipped", "reason": "disabled"}

    try:
        collections_db.set_reading_digest_history(
            schedule.id,
            last_run_at=datetime.now(timezone.utc).isoformat(),
            last_status="running",
        )
    except _READING_DIGEST_NONCRITICAL_EXCEPTIONS as exc:
        logger.debug(
            "reading_digest: failed to mark schedule as running (schedule_id={}, user_id={}): {}",
            schedule.id,
            user_id,
            exc,
            exc_info=True,
        )

    try:
        filters = _normalize_filters(schedule.filters_json)
        if isinstance(schedule.filters_json, str):
            try:
                filters = json.loads(schedule.filters_json) if schedule.filters_json else {}
            except _READING_DIGEST_NONCRITICAL_EXCEPTIONS:
                filters = {}

        status = filters.get("status")
        if isinstance(status, str):
            status = [status]
        tags = filters.get("tags")
        if isinstance(tags, str):
            tags = [tags]
        favorite = filters.get("favorite")
        domain = filters.get("domain")
        q = filters.get("q")
        date_from = filters.get("date_from")
        date_to = filters.get("date_to")
        sort = filters.get("sort")
        limit = _safe_int(filters.get("limit"), READING_DIGEST_DEFAULT_LIMIT)
        limit = max(1, min(READING_DIGEST_MAX_LIMIT, limit))

        service = ReadingService(user_id)
        rows, _total = await asyncio.to_thread(
            lambda: service.list_items(
                status=status if isinstance(status, list) else None,
                tags=tags if isinstance(tags, list) else None,
                favorite=favorite if isinstance(favorite, bool) else None,
                q=q if isinstance(q, str) else None,
                domain=domain if isinstance(domain, str) else None,
                date_from=str(date_from) if date_from else None,
                date_to=str(date_to) if date_to else None,
                page=1,
                size=limit,
                offset=0,
                limit=limit,
                sort=sort if isinstance(sort, str) else None,
            )
        )

        if isinstance(status, list):
            filters["status"] = status
        if isinstance(tags, list):
            filters["tags"] = tags

        items_context = _build_reading_items_context(rows)
        suggestions_context: list[dict[str, Any]] = []
        suggestions_meta: dict[str, Any] | None = None
        suggestions_config = _normalize_suggestions_config(filters)
        if suggestions_config:
            suggestions_rows, suggestions_meta = _select_suggestion_candidates(
                service=service,
                digest_rows=rows,
                digest_tags=tags if isinstance(tags, list) else [],
                suggestions_config=suggestions_config,
            )
            suggestions_context = _build_reading_items_context(suggestions_rows)

        generated_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        title_base = schedule.name or "Reading Digest"
        title = f"{title_base} ({generated_at})"

        output_template = None
        if schedule.template_id is not None:
            try:
                output_template = collections_db.get_output_template(int(schedule.template_id))
            except _READING_DIGEST_NONCRITICAL_EXCEPTIONS:
                output_template = None
        if output_template is None and schedule.template_name:
            try:
                output_template = collections_db.get_output_template_by_name(str(schedule.template_name))
            except _READING_DIGEST_NONCRITICAL_EXCEPTIONS:
                output_template = None

        output_format = (schedule.format or "").lower() or (getattr(output_template, "format", "") or "md")
        if output_format not in {"md", "html"}:
            output_format = "md"

        if output_template and output_template.format not in {"md", "html"}:
            logger.warning("reading_digest: invalid template format for {}", output_template.name)
            output_template = None

        if output_template is None:
            fallback_type = "newsletter_html" if output_format == "html" else "newsletter_markdown"
            try:
                output_template = collections_db.get_default_output_template_by_type(fallback_type)
            except _READING_DIGEST_NONCRITICAL_EXCEPTIONS:
                output_template = None

        context: dict[str, Any] = {
            "title": title,
            "generated_at": generated_at,
            "items": items_context,
            "item_count": len(items_context),
            "schedule_id": schedule.id,
            "schedule_name": schedule.name,
            "filters": filters,
        }
        if suggestions_config is not None:
            context["suggestions"] = suggestions_context
            context["suggestions_meta"] = suggestions_meta or {"count": len(suggestions_context)}

        if output_template:
            context["template_name"] = output_template.name
            if output_template.description:
                context["template_description"] = output_template.description
            content = render_output_template(output_template.body, context)
            output_format = (output_template.format or output_format or "md").lower()
            if output_format not in {"md", "html"}:
                output_format = "md"
        else:
            if output_format == "html":
                content = _render_default_html(title, items_context, suggestions_context)
            else:
                content = _render_default_markdown(title, items_context, suggestions_context)

        try:
            out_dir = _outputs_dir_for_user(user_id)
            await asyncio.to_thread(out_dir.mkdir, parents=True, exist_ok=True)
        except Exception as exc:
            raise ReadingDigestJobError("reading_digest_storage_unavailable", retryable=False) from exc

        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = _build_output_filename(title, "digest", ts, output_format)
        path = _resolve_output_path_for_user(user_id, filename)

        try:
            await asyncio.to_thread(path.write_text, content or "", encoding="utf-8")
        except Exception as exc:
            raise ReadingDigestJobError("reading_digest_write_failed", retryable=False) from exc

        retention_until = _resolve_retention_until(schedule.retention_days)
        item_ids = [
            item_id
            for item in items_context
            for item_id in [item.get("content_item_id") or item.get("id")]
            if item_id is not None
        ]
        metadata = {
            "schedule_id": schedule.id,
            "schedule_name": schedule.name,
            "item_count": len(items_context),
            "item_ids": item_ids,
            "filters": filters,
            "format": output_format,
            "type": "reading_digest",
        }
        if suggestions_config is not None:
            suggestions_item_ids = [
                item_id
                for item in suggestions_context
                for item_id in [item.get("content_item_id") or item.get("id")]
                if item_id is not None
            ]
            metadata.update(
                {
                    "suggestions_count": len(suggestions_context),
                    "suggestions_item_ids": suggestions_item_ids,
                    "suggestions_config": suggestions_config,
                }
            )
        if output_template:
            metadata.update(
                {
                    "template_id": output_template.id,
                    "template_name": output_template.name,
                    "template_source": "outputs_templates",
                }
            )
            if output_template.description:
                metadata["template_description"] = output_template.description

        try:
            row = await asyncio.to_thread(
                lambda: collections_db.create_output_artifact(
                    type_="reading_digest",
                    title=title,
                    format_=output_format,
                    storage_path=filename,
                    metadata_json=json.dumps(metadata),
                    job_id=job.get("id"),
                    retention_until=retention_until,
                )
            )
        except Exception as exc:
            try:
                await asyncio.to_thread(path.unlink)
            except FileNotFoundError as cleanup_exc:
                logger.debug(
                    "reading_digest: cleanup missing file after db insert failure "
                    "(schedule_id={}, job_id={}, path={}): {}",
                    schedule.id,
                    job.get("id"),
                    path,
                    cleanup_exc,
                )
            except _READING_DIGEST_NONCRITICAL_EXCEPTIONS as cleanup_exc:
                logger.debug(
                    "reading_digest: cleanup unlink failed after db insert failure "
                    "(schedule_id={}, job_id={}, path={}): {}",
                    schedule.id,
                    job.get("id"),
                    path,
                    cleanup_exc,
                    exc_info=True,
                )
            raise ReadingDigestJobError("reading_digest_db_insert_failed", retryable=False) from exc

        try:
            collections_db.set_reading_digest_history(schedule.id, last_status="succeeded")
        except _READING_DIGEST_NONCRITICAL_EXCEPTIONS:
            logger.exception(
                "reading_digest: failed to call set_reading_digest_history for schedule {}",
                schedule.id,
            )

        return {
            "status": "succeeded",
            "output_id": row.id,
            "item_count": len(items_context),
        }
    except Exception:
        try:
            collections_db.set_reading_digest_history(schedule.id, last_status="error")
        except _READING_DIGEST_NONCRITICAL_EXCEPTIONS as inner_exc:
            logger.debug(
                "reading_digest: failed to mark schedule error (schedule_id={}, user_id={}): {}",
                schedule.id,
                user_id,
                inner_exc,
                exc_info=True,
            )
        raise
