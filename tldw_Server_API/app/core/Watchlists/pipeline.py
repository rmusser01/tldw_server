"""
Watchlists fetch→ingest pipeline

Executes a watchlist job for a given user:
- Select sources based on job scope (sources/groups/tags)
- For each source:
  - If `rss`: fetch feed items, then fetch each linked page and ingest
  - If `site`: fetch page and ingest
- Apply job-level filters (include/exclude/flag) before ingestion, short-circuiting on the
  highest-priority matching rule. Filter decisions and tallies are recorded into run stats
  (filters_matched, filters_actions, filter_tallies) and filtered items are recorded into
  `scraped_items` with status="filtered".
- Persist per-run stats for each run, upsert Collections content_items for ingested
  items regardless of Media DB persistence, and append media IDs to scrape_run_items
  only when Media DB persistence is enabled

Include-only gating semantics:
- A job may set `require_include=true` in its filters payload. When any include rules exist and
  this is true, only include-matched candidates are ingested; others are treated as filtered.
- If the job does not set `require_include`, the pipeline checks the organization default
  via organizations.metadata.watchlists.require_include_default (or flat key
  watchlists_require_include_default). If neither is set, it falls back to the environment
  variable `WATCHLISTS_REQUIRE_INCLUDE_DEFAULT`. Include gating only applies when include rules exist.

Notes:
- In tests (TEST_MODE=1), RSS fetch returns a fake item and site fetch may be bypassed.
  We count items but avoid network.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import inspect
import json
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
from tldw_Server_API.app.core.Collections.embedding_queue import enqueue_embeddings_job_for_item
from tldw_Server_API.app.core.Collections.utils import hash_text_sha256, truncate_text, word_count
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
from tldw_Server_API.app.core.DB_Management.media_db.api import get_media_repository
from tldw_Server_API.app.core.DB_Management.media_db.api import managed_media_database
from tldw_Server_API.app.core.DB_Management.Personalization_DB import PersonalizationDB
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.DB_Management.scope_context import get_scope
from tldw_Server_API.app.core.DB_Management.Watchlists_DB import SourceRow, WatchlistsDatabase
from tldw_Server_API.app.core.Personalization.companion_activity import build_watchlist_item_added_activity
from tldw_Server_API.app.core.feature_flags import is_personalization_enabled
from tldw_Server_API.app.core.Watchlists.fetchers import (
    fetch_rss_feed,
    fetch_rss_feed_history,
    fetch_site_article_async,
    fetch_site_items_with_rules,
)
from tldw_Server_API.app.core.Watchlists.content_alerts import evaluate_content_alert_rules_for_item
from tldw_Server_API.app.core.Watchlists.filters import evaluate_filters, normalize_filters
from tldw_Server_API.app.core.testing import env_flag_enabled, is_test_mode

try:
    import bleach as _bleach  # type: ignore
except ImportError:  # pragma: no cover
    _bleach = None  # type: ignore[assignment]

_WATCHLISTS_PIPELINE_NONCRITICAL_EXCEPTIONS = (
    asyncio.CancelledError,
    asyncio.TimeoutError,
    AssertionError,
    AttributeError,
    ConnectionError,
    FileNotFoundError,
    ImportError,
    IndexError,
    KeyError,
    LookupError,
    OSError,
    PermissionError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    UnicodeDecodeError,
    json.JSONDecodeError,
)


def _utcnow_iso() -> str:
    return datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()


def _forums_enabled() -> bool:
    return env_flag_enabled("WATCHLIST_FORUMS_ENABLED")


def _forum_default_top_n() -> int:
    try:
        parsed = int(os.getenv("WATCHLIST_FORUM_DEFAULT_TOP_N", "20") or 20)
    except _WATCHLISTS_PIPELINE_NONCRITICAL_EXCEPTIONS:
        parsed = 20
    if parsed <= 0:
        return 20
    return min(parsed, 200)


def _forum_delay_seconds() -> float:
    try:
        delay_ms = int(os.getenv("WATCHLIST_FORUMS_DELAY_MS", "1500") or 1500)
    except _WATCHLISTS_PIPELINE_NONCRITICAL_EXCEPTIONS:
        delay_ms = 1500
    if delay_ms < 0:
        delay_ms = 0
    return delay_ms / 1000.0


def _normalize_tz(tz: str | None) -> str:
    if not tz or tz.upper() == "UTC":
        return "UTC"
    t = tz.strip().upper()
    if t.startswith("UTC+") or t.startswith("UTC-"):
        try:
            sign = 1 if t[3] == "+" else -1
            hours = int(t[4:])
            etc_offset = -sign * hours
            return f"Etc/GMT{('+' if etc_offset>0 else '')}{etc_offset}" if etc_offset != 0 else "Etc/GMT"
        except _WATCHLISTS_PIPELINE_NONCRITICAL_EXCEPTIONS:
            return "UTC"
    return tz


def fetch_site_article(url: str):
    """Wrapper for article fetch to allow test monkeypatching with sync or async callables."""
    return fetch_site_article_async(url)


async def _maybe_await(value):
    if inspect.isawaitable(value):
        return await value
    return value


_SENSITIVE_ERROR_TOKEN_RE = re.compile(
    r"(?i)(?:"
    r"['\"](?:api[_-]?key|auth[_-]?token|access[_-]?token|refresh[_-]?token|"
    r"token|secret|password|credential)['\"]\s*:\s*['\"][^'\"]+['\"]"
    r"|(?:^|[?&\s,;])(?:api[_-]?key|auth[_-]?token|access[_-]?token|"
    r"refresh[_-]?token|token|secret|password|credential)\s*[:=]\s*"
    r"(?:['\"][^'\"]*['\"]|[^'\"\s,;&]+)"
    r")"
)
_AUTHORIZATION_BEARER_RE = re.compile(
    r"(?i)(?:['\"]authorization['\"]\s*:\s*['\"]bearer\s+[^'\"]+['\"]|"
    r"\bauthorization\s*:\s*bearer\s+(?:['\"][^'\"]+['\"]|[^'\"\s,;&]+))"
)
_BEARER_SECRET_RE = re.compile(
    r"(?i)(?:['\"]bearer['\"]\s*:\s*['\"]bearer\s+[^'\"]+['\"]|"
    r"\bbearer\s+(?:['\"][^'\"]+['\"]|[^'\"\s,;&]+))"
)
_STANDALONE_HTTP_URL_RE = re.compile(r"(?i)^https?://\S+$")


def _source_error_url_without_credentials(text: str) -> str | None:
    """Return a URL without userinfo, query, or fragment when text is a standalone URL."""
    if not _STANDALONE_HTTP_URL_RE.fullmatch(text):
        return None
    parsed = urlparse(text)
    if not parsed.scheme or not parsed.netloc:
        return None
    hostname = parsed.hostname or parsed.netloc.rsplit("@", 1)[-1]
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    netloc = f"{hostname}:{parsed.port}" if parsed.port is not None else hostname
    return urlunparse((parsed.scheme, netloc, parsed.path, "", "", ""))


def _safe_source_error_text(value: Any) -> str:
    """Return bounded source-error text with common credentials and local paths removed."""
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        safe_url = _source_error_url_without_credentials(text)
        if safe_url:
            return safe_url[:500]
    except _WATCHLISTS_PIPELINE_NONCRITICAL_EXCEPTIONS as exc:
        logger.debug(
            "watchlists.source_error_url_parse_failed: error_type={}",
            type(exc).__name__,
        )
    text = _AUTHORIZATION_BEARER_RE.sub("Authorization: Bearer [redacted]", text)
    text = _BEARER_SECRET_RE.sub("Bearer [redacted]", text)
    text = _SENSITIVE_ERROR_TOKEN_RE.sub("[redacted]", text)
    return text[:500]


def _ingest_watchlist_media(
    *,
    media_db: Any,
    url: str,
    title: str,
    media_type: str,
    content: str,
    author: str | None,
    keywords: list[str],
    overwrite: bool = False,
) -> tuple[Any, Any, Any]:
    """Route watchlist media ingest through the repository API for real DB sessions."""
    media_writer = get_media_repository(media_db)
    return media_writer.add_media_with_keywords(
        url=url,
        title=title,
        media_type=media_type,
        content=content,
        author=author,
        keywords=keywords,
        overwrite=overwrite,
    )


def _compute_next_run(cron: str | None, timezone_str: str | None) -> str | None:
    if not cron:
        return None
    try:
        from apscheduler.triggers.cron import CronTrigger
        tz = _normalize_tz(timezone_str) or "UTC"
        trigger = CronTrigger.from_crontab(cron, timezone=tz)
        now = datetime.now(trigger.timezone)
        nxt = trigger.get_next_fire_time(None, now)
        return nxt.isoformat() if nxt else None
    except _WATCHLISTS_PIPELINE_NONCRITICAL_EXCEPTIONS:
        return None


def _run_is_cancelled(db: WatchlistsDatabase, run_id: int) -> bool:
    try:
        run = db.get_run(run_id)
    except _WATCHLISTS_PIPELINE_NONCRITICAL_EXCEPTIONS:
        return False
    return str(getattr(run, "status", "") or "").strip().lower() == "cancelled"


_truncate = truncate_text
_hash_content = hash_text_sha256
_word_count = word_count

# ---------------------------------------------------------------------------
# Feed content sanitization
# ---------------------------------------------------------------------------
_FEED_ALLOWED_TAGS = [
    "a", "abbr", "b", "blockquote", "br", "code", "dd", "del", "dl", "dt",
    "em", "h1", "h2", "h3", "h4", "h5", "h6", "hr", "i", "img", "li",
    "ol", "p", "pre", "strong", "sub", "sup", "table", "tbody", "td",
    "tfoot", "th", "thead", "tr", "ul",
]
_FEED_ALLOWED_ATTRS: dict[str, list[str]] = {
    "a": ["href", "title", "rel"],
    "img": ["src", "alt", "title", "width", "height"],
}
_FEED_ALLOWED_PROTOCOLS = ["http", "https", "mailto"]


import re as _re

# Pre-strip patterns: remove entire tag+content for dangerous elements
_STRIP_CONTENT_RE = [
    _re.compile(r"<\s*script[^>]*>.*?</\s*script\s*>", _re.IGNORECASE | _re.DOTALL),
    _re.compile(r"<\s*script[^>]*/?\s*>", _re.IGNORECASE),
    _re.compile(r"<\s*style[^>]*>.*?</\s*style\s*>", _re.IGNORECASE | _re.DOTALL),
    _re.compile(r"<\s*iframe[^>]*>.*?</\s*iframe\s*>", _re.IGNORECASE | _re.DOTALL),
    _re.compile(r"<\s*iframe[^>]*/?\s*>", _re.IGNORECASE),
    _re.compile(r"<\s*object[^>]*>.*?</\s*object\s*>", _re.IGNORECASE | _re.DOTALL),
    _re.compile(r"<\s*embed[^>]*/?\s*>", _re.IGNORECASE),
]
_EVENT_HANDLER_RE = _re.compile(r"\bon\w+\s*=\s*[\"'][^\"']*[\"']", _re.IGNORECASE)


def _sanitize_feed_html(value: str | None) -> str | None:
    """Sanitize HTML from feed content, stripping scripts/iframes/event handlers."""
    if not value:
        return value
    # Phase 1: remove dangerous elements and their content (regex)
    cleaned = value
    for pattern in _STRIP_CONTENT_RE:
        cleaned = pattern.sub("", cleaned)
    cleaned = _EVENT_HANDLER_RE.sub("", cleaned)
    # Phase 2: allowlist remaining tags via bleach (if available)
    if _bleach is not None:
        cleaned = _bleach.clean(
            cleaned,
            tags=_FEED_ALLOWED_TAGS,
            attributes=_FEED_ALLOWED_ATTRS,
            protocols=_FEED_ALLOWED_PROTOCOLS,
            strip=True,
            strip_comments=True,
        )
    return cleaned


# ---------------------------------------------------------------------------
# Feed health tracking helpers
# ---------------------------------------------------------------------------
_FEED_HEALTH_MAX_CONSEC_ERRORS = int(os.getenv("COLLECTIONS_FEED_MAX_CONSEC_ERRORS", "10") or "10")
_FEED_HEALTH_BACKOFF_BASE_HOURS = 1
_FEED_HEALTH_BACKOFF_CAP_HOURS = 24


def _compute_feed_backoff(consec_errors: int) -> timedelta:
    """Exponential backoff: 1h, 2h, 4h, 8h, ... capped at 24h."""
    hours = min(_FEED_HEALTH_BACKOFF_BASE_HOURS * (2 ** (consec_errors - 1)), _FEED_HEALTH_BACKOFF_CAP_HOURS)
    return timedelta(hours=hours)


def _feed_health_status(consec_errors: int, active: bool) -> str:
    """Derive health status string from error count."""
    if not active:
        return "disabled"
    if consec_errors >= 5:
        return "failing"
    if consec_errors >= 1:
        return "degraded"
    return "healthy"


# ---------------------------------------------------------------------------
# Feed retention defaults
# ---------------------------------------------------------------------------
_FEED_RETENTION_MAX_ITEMS = int(os.getenv("COLLECTIONS_FEED_MAX_ITEMS", "0") or "0")  # 0 = unlimited
_FEED_RETENTION_DAYS = int(os.getenv("COLLECTIONS_FEED_RETENTION_DAYS", "0") or "0")  # 0 = unlimited


def _apply_feed_retention(collections_db: CollectionsDatabase, origin: str, src) -> None:
    """Apply per-source retention policy after successful ingestion."""
    try:
        src_settings = json.loads(getattr(src, "settings_json", None) or "{}") if getattr(src, "settings_json", None) else {}
    except _WATCHLISTS_PIPELINE_NONCRITICAL_EXCEPTIONS:
        src_settings = {}
    retention = src_settings.get("retention") if isinstance(src_settings, dict) else None
    max_items = (retention.get("max_items") if isinstance(retention, dict) else None) or _FEED_RETENTION_MAX_ITEMS
    retention_days = (retention.get("retention_days") if isinstance(retention, dict) else None) or _FEED_RETENTION_DAYS
    if not max_items and not retention_days:
        return
    try:
        pruned = collections_db.prune_content_items_for_source(
            origin=origin,
            origin_id=int(src.id),
            max_items=int(max_items) if max_items else None,
            retention_days=int(retention_days) if retention_days else None,
        )
        if pruned:
            logger.debug(f"watchlists.retention: pruned {pruned} items for source {src.id}")
    except _WATCHLISTS_PIPELINE_NONCRITICAL_EXCEPTIONS as exc:
        logger.debug(f"watchlists.retention: failed for source {getattr(src, 'id', '?')}: {exc}")


# Tracking query parameters to strip during URL normalization
_TRACKING_PARAMS = frozenset({
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "utm_id", "utm_source_platform", "utm_creative_format",
    "fbclid", "gclid", "gclsrc", "dclid", "msclkid",
    "mc_cid", "mc_eid", "oly_anon_id", "oly_enc_id",
    "vero_id", "twclid", "igshid", "s_cid",
    "_hsenc", "_hsmi", "hsa_cam", "hsa_grp", "hsa_mt", "hsa_src",
    "hsa_ad", "hsa_acc", "hsa_net", "hsa_ver", "hsa_la", "hsa_ol", "hsa_kw",
    "ref", "ref_src",
})


def _normalize_url(url: str) -> str:
    """Normalize a URL for dedup: lowercase scheme/host, strip www, remove tracking params, strip trailing slash."""
    try:
        parsed = urlparse(url)
        scheme = (parsed.scheme or "https").lower()
        hostname = (parsed.hostname or "").lower()
        if hostname.startswith("www."):
            hostname = hostname[4:]
        port = parsed.port
        if port and ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
            port = None
        netloc = f"{hostname}:{port}" if port else hostname
        path = parsed.path.rstrip("/") or "/"
        # Filter out tracking query params, keep the rest sorted for consistency
        if parsed.query:
            params = parse_qs(parsed.query, keep_blank_values=True)
            filtered = {k: v for k, v in params.items() if k.lower() not in _TRACKING_PARAMS}
            query = urlencode(sorted(filtered.items()), doseq=True) if filtered else ""
        else:
            query = ""
        return urlunparse((scheme, netloc, path, "", query, ""))
    except _WATCHLISTS_PIPELINE_NONCRITICAL_EXCEPTIONS:
        return url


def _resolve_collections_origin(
    source_settings: dict[str, Any] | None,
    job_output_prefs: dict[str, Any] | None,
    *,
    default: str = "watchlist",
) -> str:
    for cfg in (source_settings, job_output_prefs):
        if not isinstance(cfg, dict):
            continue
        origin = cfg.get("collections_origin")
        if isinstance(origin, str) and origin.strip():
            return origin.strip()
        nested = cfg.get("collections")
        if isinstance(nested, dict):
            origin = nested.get("origin")
            if isinstance(origin, str) and origin.strip():
                return origin.strip()
    return default


def _maybe_promote_feed_schedule(
    *,
    db: WatchlistsDatabase,
    job,
    job_output_prefs: dict[str, Any],
) -> None:
    schedule_cfg = job_output_prefs.get("collections_schedule")
    if not isinstance(schedule_cfg, dict):
        return
    if schedule_cfg.get("mode") != "hourly_then_daily":
        return
    if schedule_cfg.get("promoted") is True:
        return
    daily_expr = str(schedule_cfg.get("daily_expr") or "0 0 * * *").strip()
    if not daily_expr:
        return
    try:
        promote_after = int(schedule_cfg.get("promote_after_hours", 24) or 24)
    except _WATCHLISTS_PIPELINE_NONCRITICAL_EXCEPTIONS:
        promote_after = 24
    if promote_after <= 0:
        promote_after = 24
    created_raw = schedule_cfg.get("created_at") or getattr(job, "created_at", None)
    if not created_raw:
        return
    try:
        created_dt = datetime.fromisoformat(str(created_raw))
        if created_dt.tzinfo is None:
            created_dt = created_dt.replace(tzinfo=timezone.utc)
    except _WATCHLISTS_PIPELINE_NONCRITICAL_EXCEPTIONS:
        return
    now_utc = datetime.utcnow().replace(tzinfo=timezone.utc)
    if now_utc < created_dt + timedelta(hours=promote_after):
        return

    schedule_cfg["promoted"] = True
    schedule_cfg["promoted_at"] = _utcnow_iso()
    job_output_prefs["collections_schedule"] = schedule_cfg
    patch = {"output_prefs_json": json.dumps(job_output_prefs)}
    if str(job.schedule_expr or "").strip() != daily_expr:
        patch["schedule_expr"] = daily_expr
    try:
        updated = db.update_job(int(job.id), patch)
    except _WATCHLISTS_PIPELINE_NONCRITICAL_EXCEPTIONS as exc:
        logger.debug(f"collections schedule promote failed for job {getattr(job, 'id', '?')}: {exc}")
        return
    try:
        next_run = _compute_next_run(updated.schedule_expr, updated.schedule_timezone)
        db.set_job_history(job_id=int(updated.id), next_run_at=next_run)
    except _WATCHLISTS_PIPELINE_NONCRITICAL_EXCEPTIONS:
        pass
    try:
        if updated.wf_schedule_id:
            from tldw_Server_API.app.services.workflows_scheduler import get_workflows_scheduler
            svc = get_workflows_scheduler()
            svc.update(
                updated.wf_schedule_id,
                {
                    "cron": updated.schedule_expr or "* * * * *",
                    "timezone": _normalize_tz(updated.schedule_timezone) or "UTC",
                    "enabled": bool(updated.active),
                },
            )
    except _WATCHLISTS_PIPELINE_NONCRITICAL_EXCEPTIONS as exc:
        logger.debug(f"collections schedule promote sync failed for job {getattr(job, 'id', '?')}: {exc}")


def _select_sources_for_scope(db: WatchlistsDatabase, scope: dict[str, Any]) -> list[SourceRow]:
    """Resolve sources given a job scope.

    Scope semantics (minimal):
    - sources: explicit source IDs (ints)
    - tags: list of tag names (AND semantics)
    - groups: list of group IDs (OR semantics across groups)
    """
    selected: dict[int, SourceRow] = {}
    # Explicit IDs
    for sid in map(int, scope.get("sources", []) or []):
        try:
            r = db.get_source(sid)
            if int(r.active or 0) == 1:
                selected[int(r.id)] = r
        except _WATCHLISTS_PIPELINE_NONCRITICAL_EXCEPTIONS:
            continue
    # Tags
    tag_names = scope.get("tags") or []
    if tag_names:
        rows, _ = db.list_sources(q=None, tag_names=tag_names, limit=10000, offset=0)
        for r in rows:
            if int(r.active or 0) == 1:
                selected[int(r.id)] = r
    # Groups
    group_ids = scope.get("groups") or []
    if group_ids:
        try:
            rows = db.list_sources_by_group_ids(group_ids)
            for r in rows:
                if int(r.active or 0) == 1:
                    selected[int(r.id)] = r
        except _WATCHLISTS_PIPELINE_NONCRITICAL_EXCEPTIONS:
            pass
    return list(selected.values())


async def _maybe_auto_generate_output(
    *,
    db: WatchlistsDatabase,
    collections_db: CollectionsDatabase,
    user_id: int,
    run,
    job,
    job_output_prefs: dict[str, Any],
    stats: dict[str, Any],
) -> int | None:
    """Generate a briefing output automatically if configured in job output_prefs.

    Returns the output artifact ID, or None if skipped.
    """
    auto_cfg = job_output_prefs.get("auto_output")
    if not isinstance(auto_cfg, dict) or not auto_cfg.get("enabled"):
        return None
    if stats.get("items_ingested", 0) <= 0:
        return None

    # Lazy imports to avoid circular dependencies
    from tldw_Server_API.app.core.Watchlists import template_store
    from tldw_Server_API.app.services.outputs_service import (
        _build_output_filename,
        _outputs_dir_for_user,
        _resolve_output_path_for_user,
        build_items_context_from_content_items,
        render_output_template,
    )

    items_rows, _ = db.list_items(run_id=run.id, status="ingested", limit=1000, offset=0)
    if not items_rows:
        return None

    output_type = str(auto_cfg.get("type", "briefing_markdown"))
    template_name = auto_cfg.get("template_name")
    if not template_name:
        template_defaults = job_output_prefs.get("template") or {}
        template_name = template_defaults.get("default_name")

    # Resolve template content
    template_content: str | None = None
    template_format = "md"
    if template_name:
        try:
            tpl = template_store.load_template(str(template_name))
            template_content = tpl.content
            template_format = tpl.format or "md"
        except _WATCHLISTS_PIPELINE_NONCRITICAL_EXCEPTIONS:
            logger.debug(f"auto-output: template {template_name!r} not found, using default")

    # Build context (reuses shared helper for consistent dict shape)
    job_name = getattr(job, "name", None) or f"Job-{getattr(job, 'id', '?')}"
    title = f"{job_name}-Auto-{run.id}"
    items_payload = build_items_context_from_content_items(items_rows)

    if template_content:
        context = {
            "title": title,
            "generated_at": _utcnow_iso(),
            "items": items_payload,
            "item_count": len(items_payload),
        }
        rendered = render_output_template(template_content, context)
    else:
        # Default markdown briefing
        lines = [f"# {title}", ""]
        for idx, itm in enumerate(items_payload, 1):
            item_title = itm.get("title") or "Untitled"
            item_url = itm.get("url") or ""
            item_summary = itm.get("summary") or ""
            if item_url:
                lines.append(f"{idx}. [{item_title}]({item_url})")
            else:
                lines.append(f"{idx}. {item_title}")
            if item_summary:
                lines.append(f"   {item_summary[:200]}")
            lines.append("")
        rendered = "\n".join(lines)

    # Write file
    out_dir = _outputs_dir_for_user(user_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = _build_output_filename(title, "auto", ts, template_format)
    path = _resolve_output_path_for_user(user_id, filename)
    path.write_text(rendered, encoding="utf-8")

    # Persist artifact
    metadata = {
        "item_count": len(items_payload),
        "format": template_format,
        "type": output_type,
        "origin": "watchlists",
        "generation_mode": "auto_output",
        "auto_output": True,
        "run_id": run.id,
        "job_id": getattr(job, "id", None),
    }
    artifact = collections_db.create_output_artifact(
        type_=output_type,
        title=title,
        format_=template_format,
        storage_path=filename,
        metadata_json=json.dumps(metadata),
        job_id=getattr(job, "id", None),
        run_id=run.id,
    )
    return artifact.id


def _update_auto_output_audio_metadata(
    collections_db: CollectionsDatabase,
    output_id: int,
    audio_result: Any,
) -> None:
    """Attach audio trigger state to an auto-generated output artifact."""
    from tldw_Server_API.app.core.Watchlists.audio_briefing_workflow import (
        apply_audio_briefing_result_metadata,
    )

    row = collections_db.get_output_artifact(output_id)
    metadata: dict[str, Any] = {}
    raw_metadata = getattr(row, "metadata_json", None)
    if raw_metadata:
        try:
            parsed = json.loads(raw_metadata)
            if isinstance(parsed, dict):
                metadata = parsed
        except _WATCHLISTS_PIPELINE_NONCRITICAL_EXCEPTIONS:
            metadata = {}
    apply_audio_briefing_result_metadata(metadata, audio_result, requested=True)
    collections_db.update_output_artifact_metadata(
        output_id,
        metadata_json=json.dumps(metadata),
    )


async def run_watchlist_job(
    user_id: int,
    job_id: int,
    *,
    source_ids_override: list[int] | None = None,
    capture_companion_activity: bool = False,
    companion_route: str | None = None,
) -> dict[str, Any]:
    """Run the watchlist fetch→ingest pipeline for this user/job.

    Returns minimal stats: { run_id, items_found, items_ingested }.
    """
    db = WatchlistsDatabase.for_user(user_id)
    collections_db = CollectionsDatabase.for_user(user_id)
    job = db.get_job(job_id)
    is_first_run = bool(not getattr(job, "last_run_at", None))
    run = db.create_run(job_id=job_id, status="running")

    job_output_prefs: dict[str, Any] = {}
    try:
        if getattr(job, "output_prefs_json", None):
            job_output_prefs = json.loads(job.output_prefs_json or "{}")
            if not isinstance(job_output_prefs, dict):
                job_output_prefs = {}
    except _WATCHLISTS_PIPELINE_NONCRITICAL_EXCEPTIONS as e:
        logger.exception(
            "Failed to parse job.output_prefs_json; job_id={} job_record_id={} job_repr={} "
            "output_prefs_json={!r} error={}",
            job_id,
            getattr(job, "id", None),
            repr(job),
            getattr(job, "output_prefs_json", None),
            e,
        )
        job_output_prefs = {}
    ingest_cfg = job_output_prefs.get("ingest") if isinstance(job_output_prefs, dict) else None
    persist_to_media_db = False
    if isinstance(ingest_cfg, dict) and "persist_to_media_db" in ingest_cfg:
        persist_to_media_db = bool(ingest_cfg.get("persist_to_media_db"))
    elif isinstance(job_output_prefs, dict) and isinstance(job_output_prefs.get("persist_to_media_db"), bool):
        persist_to_media_db = bool(job_output_prefs.get("persist_to_media_db"))

    stack = contextlib.ExitStack()
    try:
        # Resolve per-user media DB path and instantiate when requested
        mdb = None
        if persist_to_media_db:
            media_db_path = str(DatabasePaths.get_media_db_path(int(user_id)))
            mdb = stack.enter_context(
                managed_media_database(
                    client_id=str(user_id),
                    db_path=media_db_path,
                    initialize=True,
                    suppress_close_exceptions=_WATCHLISTS_PIPELINE_NONCRITICAL_EXCEPTIONS,
                )
            )

        normalized_user_id = str(user_id)
        companion_activity_db: PersonalizationDB | None = None
        companion_capture_enabled = False
        if capture_companion_activity and companion_route and is_personalization_enabled():
            companion_activity_db = PersonalizationDB.for_user(user_id)
            companion_profile = companion_activity_db.get_or_create_profile(normalized_user_id)
            companion_capture_enabled = bool(companion_profile.get("enabled", 0))

        # Fetch scope and sources
        scope = {}
        try:
            scope = json.loads(job.scope_json or "{}") if job.scope_json else {}
        except _WATCHLISTS_PIPELINE_NONCRITICAL_EXCEPTIONS:
            scope = {}
        override_ids: list[int] = []
        if source_ids_override:
            seen_override: set[int] = set()
            for raw_id in source_ids_override:
                try:
                    source_id = int(raw_id)
                except _WATCHLISTS_PIPELINE_NONCRITICAL_EXCEPTIONS:
                    continue
                if source_id <= 0 or source_id in seen_override:
                    continue
                seen_override.add(source_id)
                override_ids.append(source_id)

        if override_ids:
            sources = _select_sources_for_scope(db, {"sources": override_ids})
        else:
            sources = _select_sources_for_scope(db, scope or {})

        items_found = 0
        items_ingested = 0
        # Allow tags on source to flow into ingestion keywords
        def _keywords_for_source(sr: SourceRow) -> list[str]:
            try:
                return [t for t in (sr.tags or []) if t]
            except _WATCHLISTS_PIPELINE_NONCRITICAL_EXCEPTIONS:
                return []

        # TEST_MODE short-circuit for offline tests: do not perform network
        test_mode = is_test_mode()

        # Load job-level filters + gating toggle (bridge from SUBS Import Rules)
        job_filters: list[dict[str, Any]] = []
        job_require_include: bool | None = None
        try:
            if getattr(job, "job_filters_json", None):
                raw = json.loads(job.job_filters_json or "{}")
                job_filters = normalize_filters(raw)
                if isinstance(raw, dict) and "require_include" in raw:
                    try:
                        job_require_include = bool(raw.get("require_include"))
                    except _WATCHLISTS_PIPELINE_NONCRITICAL_EXCEPTIONS:
                        job_require_include = None
        except _WATCHLISTS_PIPELINE_NONCRITICAL_EXCEPTIONS:
            job_filters = []
            job_require_include = None

        async def _org_require_include_default() -> bool:
            # Read from active org metadata when available; fallback to env var
            try:
                scope = get_scope()
                org_id = getattr(scope, "effective_org_id", None) if scope else None
                if org_id is not None:
                    pool = await get_db_pool()
                    row = await pool.fetchone("SELECT metadata FROM organizations WHERE id = ?", int(org_id))
                    if row is not None:
                        meta = row.get("metadata")
                        try:
                            import json as _json
                            if isinstance(meta, str):
                                meta_dict = _json.loads(meta)
                            elif isinstance(meta, (dict,)):
                                meta_dict = meta
                            else:
                                meta_dict = None
                            if isinstance(meta_dict, dict):
                                # Accept either nested or flat key
                                if isinstance(meta_dict.get("watchlists"), dict):
                                    val = meta_dict.get("watchlists", {}).get("require_include_default")
                                    if isinstance(val, bool):
                                        return val
                                flat = meta_dict.get("watchlists_require_include_default")
                                if isinstance(flat, bool):
                                    return flat
                        except _WATCHLISTS_PIPELINE_NONCRITICAL_EXCEPTIONS:
                            pass
            except _WATCHLISTS_PIPELINE_NONCRITICAL_EXCEPTIONS:
                pass
            try:
                return env_flag_enabled("WATCHLISTS_REQUIRE_INCLUDE_DEFAULT")
            except _WATCHLISTS_PIPELINE_NONCRITICAL_EXCEPTIONS:
                return False

        include_rules_exist = any((str(f.get("action")) == "include") for f in job_filters)
        org_default = await _org_require_include_default()
        effective_require_include = job_require_include if (job_require_include is not None) else org_default
        include_gating_active = bool(effective_require_include and include_rules_exist)

        # Filter evaluation statistics
        filter_stats: dict[str, Any] = {
            "filters_matched": 0,
            "filters_actions": {"include": 0, "exclude": 0, "flag": 0},
            "filter_tallies": {},
        }

        # Bounded debug logging for filter decisions
        try:
            _max_debug = int(os.getenv("WATCHLISTS_FILTER_DEBUG_MAX", "100") or 100)
        except _WATCHLISTS_PIPELINE_NONCRITICAL_EXCEPTIONS:
            _max_debug = 100
        _debug_count = 0

        # History/backfill counters for the run
        history_pages_total = 0
        history_any_stop = False
        history_used = False
        source_statuses: list[dict[str, Any]] = []

        for src in sources:
            source_companion_events: list[dict[str, Any]] = []
            source_items_found_start = items_found
            source_items_ingested_start = items_ingested
            source_status = "ok"
            source_error: str | None = None
            if _run_is_cancelled(db, run.id):
                logger.info(f"watchlists.run_cancelled: stopping run {run.id} before source {getattr(src, 'id', '?')}")
                break
            try:
                src_type = (src.source_type or "").lower()
                if src_type == "forum":
                    if not _forums_enabled():
                        with contextlib.suppress(_WATCHLISTS_PIPELINE_NONCRITICAL_EXCEPTIONS):
                            db.update_source_scrape_meta(int(src.id), last_scraped_at=_utcnow_iso(), status="forum_disabled")
                        source_status = "forum_disabled"
                        continue
                    if not test_mode:
                        await asyncio.sleep(_forum_delay_seconds())
                # Defer source if Retry-After previously set and not elapsed
                if getattr(src, "defer_until", None):
                    try:
                        from datetime import datetime as _dt
                        now_utc = datetime.utcnow().replace(tzinfo=timezone.utc)
                        defer_dt = _dt.fromisoformat(str(src.defer_until))
                        if now_utc < defer_dt:
                            # still deferred
                            source_status = "deferred"
                            continue
                        # past due: clear defer
                        with contextlib.suppress(_WATCHLISTS_PIPELINE_NONCRITICAL_EXCEPTIONS):
                            db.clear_source_defer_until(int(src.id))
                    except _WATCHLISTS_PIPELINE_NONCRITICAL_EXCEPTIONS:
                        # if parse fails, continue normal flow
                        pass

                def _record_scraped(
                    *,
                    status: str,
                    url: str | None,
                    title: str | None,
                    summary: str | None,
                    media_id: int | None,
                    media_uuid: str | None,
                    content: str | None = None,
                    published_at: str | None = None,
                    _src=src,
                    ) -> None:
                    try:
                        item = db.record_scraped_item(
                            run_id=run.id,
                            job_id=job_id,
                            source_id=int(_src.id),
                            media_id=media_id,
                            media_uuid=media_uuid,
                            url=url,
                            title=title,
                            summary=_truncate(summary),
                            content=_sanitize_feed_html(content) if content else None,
                            published_at=published_at,
                            tags=_keywords_for_source(_src),
                            status=status,
                        )
                        if companion_capture_enabled and status == "ingested":
                            source_companion_events.append(
                                build_watchlist_item_added_activity(
                                    item=item,
                                    route=str(companion_route),
                                )
                            )
                        if status == "ingested":
                            try:
                                target_watchlist_id = getattr(job, "watchlist_id", None)
                                if target_watchlist_id is None:
                                    source_watchlist_ids = db.list_source_watchlist_ids(int(_src.id))
                                    target_watchlist_id = source_watchlist_ids[0] if source_watchlist_ids else None
                                if target_watchlist_id is not None:
                                    evaluate_content_alert_rules_for_item(
                                        db,
                                        watchlist_id=int(target_watchlist_id),
                                        item=item,
                                    )
                            except _WATCHLISTS_PIPELINE_NONCRITICAL_EXCEPTIONS as alert_err:
                                logger.debug(
                                    "watchlist content alert evaluation failed "
                                    f"(source_id={getattr(_src, 'id', '?')}, item_id={getattr(item, 'id', '?')}): {alert_err}"
                                )
                    except _WATCHLISTS_PIPELINE_NONCRITICAL_EXCEPTIONS as rec_err:
                        logger.debug(f"record_scraped_item failed (source_id={getattr(_src, 'id', '?')}): {rec_err}")

                if src_type == "rss":
                    rss_items: list[dict[str, Any]]
                    # Per-source settings
                    settings = {}
                    try:
                        settings = json.loads(src.settings_json or "{}") if getattr(src, "settings_json", None) else {}
                    except _WATCHLISTS_PIPELINE_NONCRITICAL_EXCEPTIONS:
                        settings = {}
                    collections_origin = _resolve_collections_origin(settings, job_output_prefs)
                    rss_limit = int(settings.get("limit", 50)) if isinstance(settings.get("limit", 50), int) else 50
                    # Effective history/backfill options: merge job output_prefs.history over source.settings.history
                    history_cfg = settings.get("history") if isinstance(settings.get("history"), dict) else {}
                    job_hist = (
                        job_output_prefs.get("history")
                        if isinstance(job_output_prefs, dict) and isinstance(job_output_prefs.get("history"), dict)
                        else {}
                    )
                    if job_hist:
                        # shallow-merge, job overrides source
                        merged = dict(history_cfg)
                        merged.update(job_hist)
                        history_cfg = merged
                    hist_strategy = str(history_cfg.get("strategy", "auto")).lower() if isinstance(history_cfg, dict) else "auto"
                    try:
                        hist_max_pages = int(history_cfg.get("max_pages", 1)) if isinstance(history_cfg, dict) else 1
                    except _WATCHLISTS_PIPELINE_NONCRITICAL_EXCEPTIONS:
                        hist_max_pages = 1
                    if hist_max_pages < 1:
                        hist_max_pages = 1
                    hist_on_304 = bool(history_cfg.get("on_304", False)) if isinstance(history_cfg, dict) else False
                    try:
                        hist_per_page = int(history_cfg.get("per_page_limit")) if isinstance(history_cfg.get("per_page_limit"), int) else None
                    except _WATCHLISTS_PIPELINE_NONCRITICAL_EXCEPTIONS:
                        hist_per_page = None
                    hist_stop_on_seen = bool(history_cfg.get("stop_on_seen", False)) if isinstance(history_cfg, dict) else False
                    # Load DB seen keys for boundary stop mode
                    seen_keys: list[str] = []
                    if hist_stop_on_seen:
                        try:
                            limit_base = rss_limit if rss_limit > 0 else 50
                            if isinstance(hist_per_page, int) and hist_per_page > 0:
                                limit_base = max(limit_base, hist_per_page)
                            limit = limit_base * max(hist_max_pages, 1)
                            seen_keys = db.list_seen_item_keys(int(src.id), limit=limit)
                        except _WATCHLISTS_PIPELINE_NONCRITICAL_EXCEPTIONS:
                            seen_keys = []
                    if test_mode:
                        res = {"status": 200, "items": [{"title": "Test Item", "url": "https://example.com/x", "summary": "Test"}]}
                    else:
                        # Use history-aware fetcher when configured
                        use_history = hist_max_pages > 1 or hist_strategy in {"auto", "atom", "wordpress"}
                        if use_history:
                            res = await fetch_rss_feed_history(
                                src.url,
                                etag=getattr(src, "etag", None),
                                last_modified=getattr(src, "last_modified", None),
                                timeout=8.0,
                                tenant_id="default",
                                strategy=hist_strategy,
                                max_pages=hist_max_pages,
                                per_page_limit=hist_per_page,
                                on_304=hist_on_304,
                                stop_on_seen=hist_stop_on_seen,
                                seen_keys=seen_keys,
                            )
                        else:
                            res = await fetch_rss_feed(
                                src.url,
                                etag=getattr(src, "etag", None),
                                last_modified=getattr(src, "last_modified", None),
                                timeout=8.0,
                                tenant_id="default",
                            )
                    status = int(res.get("status", 0) or 0)
                    if status == 304:
                        # nothing new
                        # Increment consecutive not-modified count and optionally apply adaptive backoff
                        try:
                            curr = int(getattr(src, "consec_not_modified", 0) or 0)
                        except _WATCHLISTS_PIPELINE_NONCRITICAL_EXCEPTIONS:
                            curr = 0
                        new_count = curr + 1
                        # Adaptive backoff parameters
                        import os as _os
                        try:
                            threshold = int(_os.getenv("WATCHLISTS_304_BACKOFF_THRESHOLD", "3") or 3)
                        except _WATCHLISTS_PIPELINE_NONCRITICAL_EXCEPTIONS:
                            threshold = 3
                        try:
                            base_sec = int(_os.getenv("WATCHLISTS_304_BACKOFF_BASE_SEC", "3600") or 3600)
                        except _WATCHLISTS_PIPELINE_NONCRITICAL_EXCEPTIONS:
                            base_sec = 3600
                        try:
                            max_sec = int(_os.getenv("WATCHLISTS_304_BACKOFF_MAX_SEC", "21600") or 21600)
                        except _WATCHLISTS_PIPELINE_NONCRITICAL_EXCEPTIONS:
                            max_sec = 21600
                        try:
                            jitter_pct = float(_os.getenv("WATCHLISTS_304_BACKOFF_JITTER_PCT", "0.1") or 0.1)
                        except _WATCHLISTS_PIPELINE_NONCRITICAL_EXCEPTIONS:
                            jitter_pct = 0.1

                        defer_until_val = None
                        if new_count >= threshold:
                            exp = new_count - threshold
                            raw = base_sec * (2 ** exp)
                            secs = min(raw, max_sec)
                            # apply ± jitter
                            import random as _rnd
                            j = int(secs * jitter_pct)
                            if j > 0:
                                secs = max(0, secs + _rnd.randint(-j, j))  # nosec B311
                            from datetime import timedelta as _td
                            defer_until_val = (datetime.utcnow().replace(tzinfo=timezone.utc) + _td(seconds=secs)).isoformat()
                        with contextlib.suppress(_WATCHLISTS_PIPELINE_NONCRITICAL_EXCEPTIONS):
                            db.update_source_scrape_meta(
                                int(src.id),
                                last_scraped_at=_utcnow_iso(),
                                status=("not_modified" if defer_until_val is None else f"not_modified_backoff:{secs}"),
                                defer_until=defer_until_val,
                                consec_not_modified=new_count,
                            )
                        source_status = "not_modified" if defer_until_val is None else f"not_modified_backoff:{secs}"
                        continue
                    if status == 429:
                        # Defer per Retry-After
                        ra = res.get("retry_after")
                        if isinstance(ra, int) and ra > 0:
                            from datetime import timedelta
                            until = (datetime.utcnow().replace(tzinfo=timezone.utc) + timedelta(seconds=ra)).isoformat()
                            with contextlib.suppress(_WATCHLISTS_PIPELINE_NONCRITICAL_EXCEPTIONS):
                                db.update_source_scrape_meta(int(src.id), status="deferred", defer_until=until)
                        source_status = "deferred"
                        continue
                    if status // 100 != 2:
                        # error path — track consecutive errors and apply backoff
                        source_status = f"error:{status}"
                        source_error = f"HTTP {status}"
                        with contextlib.suppress(_WATCHLISTS_PIPELINE_NONCRITICAL_EXCEPTIONS):
                            prev_errors = int(getattr(src, "consec_errors", 0) or 0)
                            new_errors = prev_errors + 1
                            backoff_until = (datetime.utcnow().replace(tzinfo=timezone.utc) + _compute_feed_backoff(new_errors)).isoformat()
                            health = _feed_health_status(new_errors, bool(getattr(src, "active", 1)))
                            auto_disable = new_errors >= _FEED_HEALTH_MAX_CONSEC_ERRORS
                            db.update_source_scrape_meta(
                                int(src.id),
                                last_scraped_at=_utcnow_iso(),
                                status=f"error:{status}",
                                consec_errors=new_errors,
                                defer_until=backoff_until,
                                active=0 if auto_disable else None,
                            )
                            if auto_disable:
                                logger.warning(f"watchlists.health: auto-disabled source {src.id} after {new_errors} consecutive errors")
                        continue

                    # 200 OK — reset error tracking
                    if res.get("etag") is not None or res.get("last_modified") is not None:
                        with contextlib.suppress(_WATCHLISTS_PIPELINE_NONCRITICAL_EXCEPTIONS):
                            db.update_source_scrape_meta(int(src.id), etag=res.get("etag"), last_modified=res.get("last_modified"), consec_not_modified=0, consec_errors=0)
                    # Accumulate history counters for run stats when applicable
                    try:
                        if isinstance(res.get("pages_fetched"), int):
                            history_pages_total += int(res.get("pages_fetched"))
                            history_used = True
                        if bool(res.get("stop_on_seen_triggered")):
                            history_any_stop = True
                    except _WATCHLISTS_PIPELINE_NONCRITICAL_EXCEPTIONS:
                        pass
                    rss_items = list(res.get("items") or [])
                    if isinstance(rss_limit, int) and rss_limit > 0:
                        rss_items = rss_items[:rss_limit]
                    items_found += len(rss_items)
                    for it in rss_items:
                        if _run_is_cancelled(db, run.id):
                            logger.info(f"watchlists.run_cancelled: stopping run {run.id} during RSS item processing")
                            break
                        link = it.get("url") or it.get("link")
                        if not link:
                            continue
                        # Item-level dedup check
                        item_key = (it.get("guid") or _normalize_url(link) or (it.get("title") or ""))
                        if not item_key:
                            # Fallback: hash content to avoid empty-key collisions
                            _raw = json.dumps(
                                {k: it.get(k) for k in ("title", "summary", "content", "url", "link", "published")},
                                sort_keys=True, default=str,
                            )
                            item_key = f"sha256:{hashlib.sha256(_raw.encode()).hexdigest()[:32]}"
                            logger.warning(f"watchlists.dedup: empty key for item in source {getattr(src, 'id', '?')}, using content hash")
                        # Skip dedup on the very first run (TEST_MODE only) to stabilize offline tests
                        skip_dedup = test_mode and is_first_run
                        if not skip_dedup:
                            try:
                                if db.has_seen_item(int(src.id), item_key):
                                    # Already seen; skip ingestion but count as found
                                    _record_scraped(
                                        status="duplicate",
                                        url=link,
                                        title=it.get("title"),
                                        summary=it.get("summary"),
                                        content=(it.get("content") or it.get("summary")),
                                        media_id=None,
                                        media_uuid=None,
                                        published_at=it.get("published"),
                                    )
                                    continue
                            except _WATCHLISTS_PIPELINE_NONCRITICAL_EXCEPTIONS:
                                pass

                        # Evaluate filters before fetching article
                        decision = None
                        flagged = False
                        try:
                            candidate = {
                                "title": it.get("title"),
                                "summary": it.get("summary"),
                                "content": None,
                                "author": it.get("author"),
                                "published_at": it.get("published"),
                            }
                            decision, meta = evaluate_filters(job_filters, candidate)
                            if _debug_count < _max_debug:
                                logger.debug(
                                    f"watchlists.filter:rss source={getattr(src,'id',None)} decision={decision} gating={include_gating_active} title={(it.get('title') or '')[:60]}"
                                )
                                _debug_count += 1
                            if decision is not None:
                                filter_stats["filters_matched"] += 1
                                if decision in filter_stats["filters_actions"]:
                                    filter_stats["filters_actions"][decision] += 1
                                key = meta.get("key") if isinstance(meta, dict) else None
                                if key:
                                    filter_stats["filter_tallies"][key] = filter_stats["filter_tallies"].get(key, 0) + 1
                            if decision == "exclude":
                                _record_scraped(
                                    status="filtered",
                                    url=link,
                                    title=it.get("title"),
                                    summary=it.get("summary"),
                                    content=(it.get("content") or it.get("summary")),
                                    media_id=None,
                                    media_uuid=None,
                                    published_at=it.get("published"),
                                )
                                continue
                            # Include-only gating: if active and no include matched, filter out
                            if include_gating_active and decision != "include":
                                _record_scraped(
                                    status="filtered",
                                    url=link,
                                    title=it.get("title"),
                                    summary=it.get("summary"),
                                    content=(it.get("content") or it.get("summary")),
                                    media_id=None,
                                    media_uuid=None,
                                    published_at=it.get("published"),
                                )
                                continue
                            flagged = (decision == "flag")
                        except _WATCHLISTS_PIPELINE_NONCRITICAL_EXCEPTIONS:
                            flagged = False

                        # Optional: if feed provides full text, skip article fetch
                        rss_cfg = settings.get("rss") if isinstance(settings.get("rss"), dict) else {}
                        prefer_feed = bool(rss_cfg.get("use_feed_content_if_available", False)) if isinstance(rss_cfg, dict) else False
                        try:
                            min_chars = int(rss_cfg.get("feed_content_min_chars", 400)) if isinstance(rss_cfg, dict) else 400
                        except _WATCHLISTS_PIPELINE_NONCRITICAL_EXCEPTIONS:
                            min_chars = 400
                        article = None
                        if prefer_feed:
                            feed_text = (it.get("summary") or "").strip()
                            if feed_text and len(feed_text) >= max(0, min_chars):
                                article = {
                                    "title": it.get("title") or "Untitled",
                                    "url": link,
                                    "content": feed_text,
                                    "author": it.get("author"),
                                }
                        if article is None:
                            article = None if test_mode else await _maybe_await(fetch_site_article(link))
                        if article is None and test_mode:
                            # In tests, fall back to summary as content
                            article = {
                                "title": it.get("title") or "Untitled",
                                "url": link,
                                "content": it.get("summary") or "",
                                "author": None,
                            }
                        if not article:
                            continue
                        ingested_media_id: int | None = None
                        ingested_media_uuid: str | None = None
                        summary_text = article.get("content") or it.get("summary") or ""
                        if persist_to_media_db and mdb is not None:
                            try:
                                media_id, media_uuid, _msg = _ingest_watchlist_media(
                                    media_db=mdb,
                                    url=article.get("url") or link,
                                    title=article.get("title") or (it.get("title") or "Untitled"),
                                    media_type="article",
                                    content=article.get("content") or (it.get("summary") or ""),
                                    author=article.get("author"),
                                    keywords=(_keywords_for_source(src) + (["flagged"] if flagged else [])),
                                    overwrite=False,
                                )
                                if media_id:
                                    ingested_media_id = int(media_id)
                                    ingested_media_uuid = media_uuid
                                    db.append_run_item(run.id, ingested_media_id, source_id=int(src.id))
                            except _WATCHLISTS_PIPELINE_NONCRITICAL_EXCEPTIONS as e:
                                logger.debug(f"Media DB ingestion failed for {link}: {e}")

                        content_text = article.get("content") or summary_text or ""
                        # Sanitize feed HTML before storing
                        summary_text = _sanitize_feed_html(summary_text) or ""
                        content_text = _sanitize_feed_html(content_text) or ""
                        tags_for_item = _keywords_for_source(src)
                        if flagged and "flagged" not in tags_for_item:
                            tags_for_item = tags_for_item + ["flagged"]
                        metadata_payload = {
                            "source_id": int(src.id),
                            "source_name": getattr(src, "name", None),
                            "job_id": job_id,
                            "run_id": run.id,
                            "media_uuid": ingested_media_uuid,
                            "tags": tags_for_item,
                            "origin": collections_origin,
                        }
                        item_row = None
                        try:
                            item_row = collections_db.upsert_content_item(
                                origin=collections_origin,
                                origin_type=str(src.source_type or ""),
                                origin_id=int(src.id),
                                url=article.get("url") or link,
                                canonical_url=article.get("url") or link,
                                domain=None,
                                title=article.get("title") or (it.get("title") or "Untitled"),
                                summary=_truncate(summary_text, 600),
                                content_hash=_hash_content(content_text),
                                word_count=_word_count(content_text),
                                published_at=it.get("published"),
                                status="new",
                                favorite=False,
                                metadata=metadata_payload,
                                media_id=ingested_media_id,
                                job_id=job_id,
                                run_id=run.id,
                                source_id=int(src.id),
                                read_at=None,
                                tags=tags_for_item,
                            )
                        except _WATCHLISTS_PIPELINE_NONCRITICAL_EXCEPTIONS as exc:
                            logger.debug(f"Collections upsert failed (rss) for {link}: {exc}")
                        if item_row:
                            items_ingested += 1
                            if item_row.is_new or item_row.content_changed:
                                try:
                                    await enqueue_embeddings_job_for_item(
                                        user_id=user_id,
                                        item_id=item_row.id,
                                        content=content_text,
                                        metadata={
                                            "origin": collections_origin,
                                            "job_id": job_id,
                                            "run_id": run.id,
                                            "tags": tags_for_item,
                                        },
                                    )
                                except _WATCHLISTS_PIPELINE_NONCRITICAL_EXCEPTIONS as exc:
                                    logger.debug(f"Embedding enqueue failed for watchlist item {item_row.id}: {exc}")
                            try:
                                db.mark_seen_item(
                                    int(src.id),
                                    item_key,
                                    etag=None,
                                    last_modified=(it.get("published") or None),
                                )
                            except _WATCHLISTS_PIPELINE_NONCRITICAL_EXCEPTIONS as exc:
                                logger.debug(
                                    f"watchlists: failed to mark seen item for source {src.id}: {exc}"
                                )
                            _record_scraped(
                                status="ingested",
                                url=article.get("url") or link,
                                title=article.get("title") or (it.get("title") or "Untitled"),
                                summary=summary_text,
                                content=content_text,
                                media_id=ingested_media_id,
                                media_uuid=ingested_media_uuid,
                                published_at=it.get("published"),
                            )
                        else:
                            _record_scraped(
                                status="error",
                                url=article.get("url") or link,
                                title=article.get("title") or (it.get("title") or "Untitled"),
                                summary=summary_text,
                                content=content_text,
                                media_id=ingested_media_id,
                                media_uuid=ingested_media_uuid,
                                published_at=it.get("published"),
                            )
                            continue

                    # Update last_scraped_at/status for source
                        with contextlib.suppress(_WATCHLISTS_PIPELINE_NONCRITICAL_EXCEPTIONS):
                            db.update_source_scrape_meta(int(src.id), last_scraped_at=_utcnow_iso(), status="ok", consec_errors=0)

                    # Apply retention policy if configured
                        _apply_feed_retention(collections_db, collections_origin, src)

                elif src_type in {"site", "forum"}:
                    # Determine discovery preferences
                    settings = {}
                    try:
                        settings = json.loads(src.settings_json or "{}") if getattr(src, "settings_json", None) else {}
                    except _WATCHLISTS_PIPELINE_NONCRITICAL_EXCEPTIONS:
                        settings = {}
                    collections_origin = _resolve_collections_origin(settings, job_output_prefs)

                    scrape_rules = settings.get("scrape_rules") if isinstance(settings.get("scrape_rules"), dict) else None
                    prefetched_by_url: dict[str, dict[str, Any]] = {}
                    urls_to_fetch: list[str] = []

                    if scrape_rules:
                        try:
                            scraped_items = await fetch_site_items_with_rules(
                                base_url=str(scrape_rules.get("list_url") or src.url),
                                rules=scrape_rules,
                                tenant_id="default",
                            )
                        except _WATCHLISTS_PIPELINE_NONCRITICAL_EXCEPTIONS as exc:
                            logger.debug(f"Scrape rules fetch failed for source {getattr(src, 'id', '?')}: {exc}")
                            scraped_items = []
                        for entry in scraped_items:
                            link = (entry.get("url") or "").strip()
                            if not link:
                                continue
                            if link not in prefetched_by_url:
                                prefetched_by_url[link] = entry
                                urls_to_fetch.append(link)
                        if "top_n" in settings:
                            try:
                                top_limit = max(0, int(settings.get("top_n", 0)))
                            except _WATCHLISTS_PIPELINE_NONCRITICAL_EXCEPTIONS:
                                top_limit = None
                            if top_limit == 0:
                                urls_to_fetch = []
                            elif top_limit is not None and top_limit < len(urls_to_fetch):
                                urls_to_fetch = urls_to_fetch[:top_limit]
                        if urls_to_fetch:
                            prefetched_by_url = {url: prefetched_by_url[url] for url in urls_to_fetch}
                        if not urls_to_fetch:
                            urls_to_fetch = [src.url]
                    else:
                        default_top_n = _forum_default_top_n() if src_type == "forum" else 1
                        top_n = default_top_n
                        try:
                            top_n = int(settings.get("top_n", default_top_n))
                        except _WATCHLISTS_PIPELINE_NONCRITICAL_EXCEPTIONS:
                            top_n = default_top_n
                        if top_n <= 0:
                            top_n = default_top_n
                        discover_method = str(settings.get("discover_method", "auto")).lower()
                        if top_n > 1:
                            try:
                                from tldw_Server_API.app.core.Watchlists.fetchers import fetch_site_top_links

                                urls_to_fetch = await fetch_site_top_links(src.url, top_n=top_n, method=discover_method)
                            except _WATCHLISTS_PIPELINE_NONCRITICAL_EXCEPTIONS:
                                urls_to_fetch = [src.url]
                        else:
                            urls_to_fetch = [src.url]

                    if not urls_to_fetch:
                        source_status = "skipped:no_urls"
                        continue

                    skip_article_fetch = bool(scrape_rules.get("skip_article_fetch")) if isinstance(scrape_rules, dict) else False

                    items_found += len(urls_to_fetch)
                    extraction_failure_count = 0
                    extraction_success_count = 0
                    for page_url in urls_to_fetch:
                        if _run_is_cancelled(db, run.id):
                            logger.info(f"watchlists.run_cancelled: stopping run {run.id} during site item processing")
                            break
                        prefetch = prefetched_by_url.get(page_url)
                        item_key = (prefetch.get("guid") if prefetch and prefetch.get("guid") else _normalize_url(page_url))
                        skip_dedup = test_mode and is_first_run
                        if not skip_dedup:
                            try:
                                if db.has_seen_item(int(src.id), item_key):
                                    _record_scraped(
                                        status="duplicate",
                                        url=page_url,
                                        title=(prefetch.get("title") if prefetch and prefetch.get("title") else src.name),
                                        summary=(prefetch.get("summary") if prefetch else None),
                                        content=(prefetch.get("content") or prefetch.get("summary")) if prefetch else None,
                                        media_id=None,
                                        media_uuid=None,
                                        published_at=(prefetch.get("published") or prefetch.get("published_raw")) if prefetch else None,
                                    )
                                    continue
                            except _WATCHLISTS_PIPELINE_NONCRITICAL_EXCEPTIONS:
                                pass

                        article: dict[str, Any] | None = None
                        if skip_article_fetch and prefetch:
                            article = {
                                "title": prefetch.get("title") or src.name or "Untitled",
                                "url": page_url,
                                "content": prefetch.get("content") or prefetch.get("summary") or "",
                                "author": prefetch.get("author"),
                            }
                        if article is None:
                            if test_mode:
                                article = {"title": src.name or "Untitled", "url": page_url, "content": "", "author": None}
                            else:
                                article = await _maybe_await(fetch_site_article(page_url))
                        if (not article or not article.get("content")) and prefetch:
                            article = article or {}
                            article["title"] = article.get("title") or prefetch.get("title") or src.name
                            article["url"] = article.get("url") or page_url
                            article["content"] = article.get("content") or prefetch.get("content") or prefetch.get("summary") or ""
                            if prefetch.get("author") and not article.get("author"):
                                article["author"] = prefetch.get("author")
                        if not article:
                            extraction_failure_count += 1
                            _record_scraped(
                                status="error",
                                url=page_url,
                                title=prefetch.get("title") if prefetch and prefetch.get("title") else src.name,
                                summary=prefetch.get("summary") if prefetch else None,
                                content=(prefetch.get("content") or prefetch.get("summary")) if prefetch else None,
                                media_id=None,
                                media_uuid=None,
                                published_at=prefetch.get("published") if prefetch else None,
                            )
                            continue

                        extraction_success_count += 1
                        article["url"] = article.get("url") or page_url
                        if not article.get("title"):
                            article["title"] = prefetch.get("title") if prefetch and prefetch.get("title") else src.name

                        ingested_media_id: int | None = None
                        ingested_media_uuid: str | None = None
                        summary_text = article.get("content") or ""
                        if not summary_text and prefetch:
                            summary_text = prefetch.get("content") or prefetch.get("summary") or ""
                        # Evaluate filters combining article + prefetch metadata
                        decision = None
                        flagged = False
                        try:
                            candidate = {
                                "title": article.get("title") or (prefetch.get("title") if prefetch else None) or src.name,
                                "summary": (prefetch.get("summary") if prefetch else None),
                                "content": article.get("content"),
                                "author": article.get("author") or (prefetch.get("author") if prefetch else None),
                                "published_at": (prefetch.get("published") if prefetch else None),
                            }
                            decision, meta = evaluate_filters(job_filters, candidate)
                            if _debug_count < _max_debug:
                                logger.debug(
                                    f"watchlists.filter:site source={getattr(src,'id',None)} decision={decision} gating={include_gating_active} url={(page_url or '')[:120]}"
                                )
                                _debug_count += 1
                            if decision is not None:
                                filter_stats["filters_matched"] += 1
                                if decision in filter_stats["filters_actions"]:
                                    filter_stats["filters_actions"][decision] += 1
                                key = meta.get("key") if isinstance(meta, dict) else None
                                if key:
                                    filter_stats["filter_tallies"][key] = filter_stats["filter_tallies"].get(key, 0) + 1
                            if decision == "exclude":
                                _record_scraped(
                                    status="filtered",
                                    url=article.get("url") or page_url,
                                    title=article.get("title") or src.name,
                                    summary=(prefetch.get("summary") if prefetch else None),
                                    content=(
                                        article.get("content")
                                        or (prefetch.get("content") if prefetch else None)
                                        or (prefetch.get("summary") if prefetch else None)
                                    ),
                                    media_id=None,
                                    media_uuid=None,
                                    published_at=(prefetch.get("published") if prefetch else None),
                                )
                                continue
                            # Include-only gating
                            if include_gating_active and decision != "include":
                                _record_scraped(
                                    status="filtered",
                                    url=article.get("url") or page_url,
                                    title=article.get("title") or src.name,
                                    summary=(prefetch.get("summary") if prefetch else None),
                                    content=(
                                        article.get("content")
                                        or (prefetch.get("content") if prefetch else None)
                                        or (prefetch.get("summary") if prefetch else None)
                                    ),
                                    media_id=None,
                                    media_uuid=None,
                                    published_at=(prefetch.get("published") if prefetch else None),
                                )
                                continue
                            flagged = (decision == "flag")
                        except _WATCHLISTS_PIPELINE_NONCRITICAL_EXCEPTIONS:
                            flagged = False
                        if persist_to_media_db and mdb is not None:
                            try:
                                media_id, media_uuid, _msg = _ingest_watchlist_media(
                                    media_db=mdb,
                                    url=article.get("url") or page_url,
                                    title=article.get("title") or src.name,
                                    media_type="article",
                                    content=article.get("content") or summary_text or "",
                                    author=article.get("author"),
                                    keywords=(_keywords_for_source(src) + (["flagged"] if flagged else [])),
                                    overwrite=False,
                                )
                                if media_id:
                                    ingested_media_id = int(media_id)
                                    ingested_media_uuid = media_uuid
                                    db.append_run_item(run.id, ingested_media_id, source_id=int(src.id))
                            except _WATCHLISTS_PIPELINE_NONCRITICAL_EXCEPTIONS as e:
                                logger.debug(f"Media DB ingestion failed for site {page_url}: {e}")

                        content_text = article.get("content") or summary_text or ""
                        # Sanitize feed HTML before storing
                        summary_text = _sanitize_feed_html(summary_text) or ""
                        content_text = _sanitize_feed_html(content_text) or ""
                        tags_for_item = _keywords_for_source(src)
                        if flagged and "flagged" not in tags_for_item:
                            tags_for_item = tags_for_item + ["flagged"]
                        metadata_payload = {
                            "source_id": int(src.id),
                            "source_name": getattr(src, "name", None),
                            "job_id": job_id,
                            "run_id": run.id,
                            "media_uuid": ingested_media_uuid,
                            "tags": tags_for_item,
                            "origin": collections_origin,
                        }
                        if prefetch and (prefetch.get("published") or prefetch.get("published_raw")):
                            metadata_payload["prefetch_published"] = prefetch.get("published") or prefetch.get("published_raw")
                        item_row = None
                        try:
                            item_row = collections_db.upsert_content_item(
                                origin=collections_origin,
                                origin_type=str(src.source_type or ""),
                                origin_id=int(src.id),
                                url=article.get("url") or page_url,
                                canonical_url=article.get("url") or page_url,
                                domain=None,
                                title=article.get("title") or src.name,
                                summary=_truncate(summary_text, 600),
                                content_hash=_hash_content(content_text),
                                word_count=_word_count(content_text),
                                published_at=(prefetch.get("published") if prefetch else None),
                                status="new",
                                favorite=False,
                                metadata=metadata_payload,
                                media_id=ingested_media_id,
                                job_id=job_id,
                                run_id=run.id,
                                source_id=int(src.id),
                                read_at=None,
                                tags=tags_for_item,
                            )
                        except _WATCHLISTS_PIPELINE_NONCRITICAL_EXCEPTIONS as exc:
                            logger.debug(f"Collections upsert failed (site) for {page_url}: {exc}")
                        if item_row:
                            items_ingested += 1
                            if item_row.is_new or item_row.content_changed:
                                try:
                                    await enqueue_embeddings_job_for_item(
                                        user_id=user_id,
                                        item_id=item_row.id,
                                        content=content_text,
                                        metadata={
                                            "origin": collections_origin,
                                            "job_id": job_id,
                                            "run_id": run.id,
                                            "tags": tags_for_item,
                                        },
                                    )
                                except _WATCHLISTS_PIPELINE_NONCRITICAL_EXCEPTIONS as exc:
                                    logger.debug(f"Embedding enqueue failed for watchlist item {item_row.id}: {exc}")
                            with contextlib.suppress(_WATCHLISTS_PIPELINE_NONCRITICAL_EXCEPTIONS):
                                db.mark_seen_item(
                                    int(src.id),
                                    item_key,
                                    etag=None,
                                    last_modified=(prefetch.get("published") or prefetch.get("published_raw")) if prefetch else None,
                                )
                            _record_scraped(
                                status="ingested",
                                url=article.get("url") or page_url,
                                title=article.get("title") or src.name,
                                summary=summary_text or (prefetch.get("summary") if prefetch else None),
                                content=content_text,
                                media_id=ingested_media_id,
                                media_uuid=ingested_media_uuid,
                                published_at=(prefetch.get("published") or prefetch.get("published_raw")) if prefetch else None,
                            )
                        else:
                            _record_scraped(
                                status="error",
                                url=article.get("url") or page_url,
                                title=article.get("title") or src.name,
                                summary=summary_text or (prefetch.get("summary") if prefetch else None),
                                content=content_text,
                                media_id=ingested_media_id,
                                media_uuid=ingested_media_uuid,
                                published_at=(prefetch.get("published") or prefetch.get("published_raw")) if prefetch else None,
                            )
                    if extraction_failure_count:
                        if extraction_success_count:
                            source_status = "partial:extraction"
                            source_error = (
                                f"{extraction_failure_count} of "
                                f"{extraction_failure_count + extraction_success_count} URLs failed extraction"
                            )
                        else:
                            source_status = "error:extraction"
                            source_error = f"No article content extracted from {extraction_failure_count} URL(s)"
                    with contextlib.suppress(_WATCHLISTS_PIPELINE_NONCRITICAL_EXCEPTIONS):
                        db.update_source_scrape_meta(int(src.id), last_scraped_at=_utcnow_iso(), status="ok", consec_errors=0)
                    # Apply retention policy if configured
                    _apply_feed_retention(collections_db, collections_origin, src)
                else:
                    # Unknown type - skip
                    source_status = "skipped:unknown_type"
                    continue
            except _WATCHLISTS_PIPELINE_NONCRITICAL_EXCEPTIONS as e:
                logger.debug(
                    "Source processing failed (id={}) error_type={}",
                    getattr(src, "id", "?"),
                    type(e).__name__,
                )
                source_status = "error"
                source_error = _safe_source_error_text(e) or type(e).__name__
                with contextlib.suppress(_WATCHLISTS_PIPELINE_NONCRITICAL_EXCEPTIONS):
                    prev_errors = int(getattr(src, "consec_errors", 0) or 0)
                    new_errors = prev_errors + 1
                    backoff_until = (datetime.utcnow().replace(tzinfo=timezone.utc) + _compute_feed_backoff(new_errors)).isoformat()
                    auto_disable = new_errors >= _FEED_HEALTH_MAX_CONSEC_ERRORS
                    db.update_source_scrape_meta(
                        int(src.id),
                        last_scraped_at=_utcnow_iso(),
                        status="error",
                        consec_errors=new_errors,
                        defer_until=backoff_until,
                        active=0 if auto_disable else None,
                    )
                    if auto_disable:
                        logger.warning(f"watchlists.health: auto-disabled source {src.id} after {new_errors} consecutive errors")
            finally:
                source_statuses.append(
                    {
                        "source_id": int(getattr(src, "id", 0) or 0),
                        "name": getattr(src, "name", None),
                        "status": source_status,
                        "error": source_error,
                        "items_found": max(0, items_found - source_items_found_start),
                        "items_ingested": max(0, items_ingested - source_items_ingested_start),
                    }
                )
                if companion_activity_db is not None and source_companion_events:
                    try:
                        companion_activity_db.insert_companion_activity_events_bulk(
                            user_id=normalized_user_id,
                            events=source_companion_events,
                        )
                    except _WATCHLISTS_PIPELINE_NONCRITICAL_EXCEPTIONS as exc:
                        logger.debug(
                            "watchlists companion batch insert skipped for source {}: {}",
                            getattr(src, "id", "?"),
                            exc,
                        )

        source_errors = sum(
            1
            for source_status_entry in source_statuses
            if str(source_status_entry.get("status") or "").startswith(("error", "partial:"))
        )
        stats = {
            "items_found": items_found,
            "items_ingested": items_ingested,
            "source_statuses": source_statuses,
            "source_errors": source_errors,
        }
        try:
            if filter_stats["filters_matched"]:
                stats["filters_matched"] = int(filter_stats["filters_matched"])  # type: ignore[assignment]
                stats["filters_actions"] = filter_stats["filters_actions"]
                stats["filter_tallies"] = filter_stats["filter_tallies"]
        except _WATCHLISTS_PIPELINE_NONCRITICAL_EXCEPTIONS:
            pass
        # Attach history/backfill counters when used
        try:
            if history_used:
                stats["history"] = {
                    "pages_fetched": int(history_pages_total),
                    "stop_on_seen_triggered": bool(history_any_stop),
                }
        except _WATCHLISTS_PIPELINE_NONCRITICAL_EXCEPTIONS:
            pass

        if _run_is_cancelled(db, run.id):
            cancelled_error = "cancelled_by_user"
            try:
                cancelled_run = db.get_run(run.id)
                if getattr(cancelled_run, "error_msg", None):
                    cancelled_error = str(cancelled_run.error_msg)
            except _WATCHLISTS_PIPELINE_NONCRITICAL_EXCEPTIONS:
                pass
            db.update_run(
                run.id,
                status="cancelled",
                finished_at=_utcnow_iso(),
                stats_json=json.dumps(stats),
                error_msg=cancelled_error,
            )
            db.set_job_history(job_id=job_id, last_run_at=_utcnow_iso(), next_run_at=_compute_next_run(job.schedule_expr, job.schedule_timezone))
            return {"run_id": run.id, "status": "cancelled", **stats}

        db.update_run(run.id, status="succeeded", finished_at=_utcnow_iso(), stats_json=json.dumps(stats))

        # Update job history
        next_run = _compute_next_run(job.schedule_expr, job.schedule_timezone)
        db.set_job_history(job_id=job_id, last_run_at=_utcnow_iso(), next_run_at=next_run)
        try:
            if isinstance(job_output_prefs, dict):
                _maybe_promote_feed_schedule(db=db, job=job, job_output_prefs=job_output_prefs)
        except _WATCHLISTS_PIPELINE_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug(f"collections schedule auto-promote failed for job {job_id}: {exc}")

        # Auto-generate output if configured
        try:
            auto_output_id = await _maybe_auto_generate_output(
                db=db,
                collections_db=collections_db,
                user_id=user_id,
                run=run,
                job=job,
                job_output_prefs=job_output_prefs,
                stats=stats,
            )
            if auto_output_id:
                stats["auto_output_id"] = auto_output_id
        except _WATCHLISTS_PIPELINE_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug(f"auto-output generation failed for run {run.id}: {exc}")

        # Trigger audio briefing workflow if configured
        try:
            if isinstance(job_output_prefs, dict) and job_output_prefs.get("generate_audio"):
                from tldw_Server_API.app.core.Watchlists.audio_briefing_workflow import (
                    apply_audio_briefing_result_metadata,
                    trigger_audio_briefing,
                )

                audio_result = await trigger_audio_briefing(
                    user_id=user_id,
                    job_id=job_id,
                    run_id=run.id,
                    output_prefs=job_output_prefs,
                    db=db,
                )
                apply_audio_briefing_result_metadata(stats, audio_result)
                if auto_output_id:
                    try:
                        _update_auto_output_audio_metadata(collections_db, int(auto_output_id), audio_result)
                    except _WATCHLISTS_PIPELINE_NONCRITICAL_EXCEPTIONS as exc:
                        logger.debug(f"auto-output audio metadata update failed for output {auto_output_id}: {exc}")
        except _WATCHLISTS_PIPELINE_NONCRITICAL_EXCEPTIONS as exc:
            logger.warning(
                "Audio briefing trigger failed for job {} (error_type={})",
                job_id,
                type(exc).__name__,
            )

        # Persist post-run augmentation fields (e.g., auto_output_id, audio_briefing_task_id).
        try:
            db.update_run(run.id, stats_json=json.dumps(stats))
        except _WATCHLISTS_PIPELINE_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug(f"post-run stats persistence failed for run {run.id}: {exc}")

        return {"run_id": run.id, **stats}
    finally:
        stack.close()
