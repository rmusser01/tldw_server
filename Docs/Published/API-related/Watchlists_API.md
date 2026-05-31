# Watchlists API Cheat Sheet

Quick reference for first-class Watchlists, recurring scraping (RSS or site lists), runs, items, and outputs.

Auth:
- Single-user: `X-API-KEY: <key>`
- Multi-user: `Authorization: Bearer <token>`

Base path: `/api/v1/watchlists`

## Core concepts

- Watchlist: project-like tracking container for intent, sources, monitors, runs, items, and outputs.
- Source: a target to scrape (`rss` or `site`; `forum` is feature-flagged).
- Job: schedule + scope + filters that produce runs.
- Run: one execution of a job.
- Item: a scraped candidate (ingested, filtered, duplicate, or error).
- Output: rendered report from run items.
- Template: named output template (md/html).

Watchlist fields:
- `name`, `description`, and `objective`
- `domain`: `general`, `cti_osint`, or `news`
- `status`: `active`, `paused`, or `archived`
- `priority`: `low`, `medium`, `high`, or `critical`
- `tags`
- lifecycle timestamps including `created_at`, `updated_at`, `archived_at`, `deleted_at`, and `restore_expires_at`

Existing unscoped data is associated with a default `Imported Watchlist`. New source and job create calls that omit `watchlist_id` are attached to that default container for backward compatibility.

## Endpoints

Watchlists:
- `GET /`
- `POST /`
- `GET /{watchlist_id}`
- `PATCH /{watchlist_id}`
- `DELETE /{watchlist_id}` (soft delete with restore window)
- `POST /{watchlist_id}/restore`

Sources:
- `POST /sources`
- `GET /sources`
- `GET /sources/{source_id}`
- `GET /sources/{source_id}/seen` (dedup/seen counters + optional recent keys)
- `PATCH /sources/{source_id}`
- `DELETE /sources/{source_id}`
- `DELETE /sources/{source_id}/seen` (clear dedup/seen state, optional backoff reset)
- `POST /sources/{source_id}/test` (preview without ingestion)
- `POST /sources/bulk` (bulk create)
- `GET /sources/export` (OPML)
- `POST /sources/import` (OPML)

Tags and groups:
- `GET /tags`
- `POST /groups`
- `GET /groups`
- `PATCH /groups/{group_id}`
- `DELETE /groups/{group_id}`

Jobs and filters:
- `POST /jobs`
- `GET /jobs`
- `GET /jobs/{job_id}`
- `PATCH /jobs/{job_id}`
- `DELETE /jobs/{job_id}`
- `POST /jobs/{job_id}/preview`
- `PATCH /jobs/{job_id}/filters` (replace)
- `POST /jobs/{job_id}/filters:add` (append)

Runs:
- `POST /jobs/{job_id}/run`
- `GET /jobs/{job_id}/runs`
- `GET /runs`
- `GET /runs/{run_id}`
- `GET /runs/{run_id}/details`
- `GET /runs/{run_id}/audio`
- `GET /runs/{run_id}/tallies.csv`
- `GET /runs/export.csv`
- `WS /runs/{run_id}/stream`

Items:
- `GET /items`
- `GET /items/smart-counts`
- `POST /items/batch-update` (batch triage flags)
- `GET /items/{item_id}`
- `PATCH /items/{item_id}` (flagging)

Item saved views:
- `GET /{watchlist_id}/item-views`
- `POST /{watchlist_id}/item-views`
- `PATCH /{watchlist_id}/item-views/{view_id}`
- `DELETE /{watchlist_id}/item-views/{view_id}`

Outputs:
- `POST /outputs`
- `GET /outputs`
- `GET /outputs/{output_id}`
- `GET /outputs/{output_id}/download`
- `GET /outputs/{output_id}/evidence`
- `GET /outputs/{output_id}/readiness`

Content alerts:
- `GET /{watchlist_id}/content-alert-rules`
- `POST /{watchlist_id}/content-alert-rules`
- `PATCH /{watchlist_id}/content-alert-rules/{rule_id}`
- `DELETE /{watchlist_id}/content-alert-rules/{rule_id}`
- `GET /{watchlist_id}/alerts`
- `GET /{watchlist_id}/alerts/{alert_id}`
- `PATCH /{watchlist_id}/alerts/{alert_id}` (review state)

Templates:
- `GET /templates`
- `GET /templates/{template_name}`
- `POST /templates` (create or update)
- `POST /templates/validate` (syntax check only)
- `POST /templates/preview` (render against run context)
- `POST /templates/compose/section` (manual inline section generation)
- `POST /templates/compose/flow-check` (manual final flow-check diff)
- `DELETE /templates/{template_name}`

Scoped list filters:
- `GET /sources?watchlist_id=<id>`
- `GET /jobs?watchlist_id=<id>`
- `GET /runs?watchlist_id=<id>`
- `GET /runs/export.csv?watchlist_id=<id>`
- `GET /items?watchlist_id=<id>`
- `GET /items/smart-counts?watchlist_id=<id>`
- `GET /outputs?watchlist_id=<id>`

`watchlist_id` filters return `404` when the Watchlist does not exist or has been deleted. The legacy `/{watchlist_id}/clusters` route currently uses a job id despite the path segment name; this route is preserved for compatibility.

## Minimal payloads

Create a Watchlist:
```json
{
  "name": "Healthcare ransomware",
  "description": "Track hospital impact reports",
  "objective": "Find new ransomware reports affecting hospitals in Germany",
  "domain": "cti_osint",
  "priority": "high",
  "tags": ["ransomware", "healthcare"]
}
```

Update a Watchlist lifecycle or metadata:
```json
{
  "status": "archived",
  "priority": "critical",
  "tags": ["cti", "ransomware"]
}
```

Create a source (RSS):
```json
{
  "name": "Example Feed",
  "url": "https://example.com/feed.xml",
  "source_type": "rss",
  "tags": ["news"],
  "watchlist_id": 42
}
```

Create a source (site + scrape rules):
```json
{
  "name": "Docs Changelog",
  "url": "https://docs.example.com/changelog",
  "source_type": "site",
  "settings": {
    "top_n": 10,
    "discover_method": "auto",
    "scrape_rules": {
      "list_url": "https://docs.example.com/changelog",
      "item_selector": ".entry",
      "title_selector": "h2 a",
      "link_selector": "h2 a",
      "summary_selector": ".summary",
      "date_selector": "time"
    }
  }
}
```

Create a job:
```json
{
  "name": "Docs updates",
  "watchlist_id": 42,
  "scope": {"sources": [123]},
  "schedule_expr": "0 */6 * * *",
  "timezone": "UTC",
  "ingest_prefs": {"persist_to_media_db": true},
  "output_prefs": {
    "deliveries": {
      "email": {
        "enabled": true,
        "subject": "Daily Watchlist Digest"
      }
    }
  }
}
```

Create a content alert rule:
```json
{
  "name": "Active exploitation",
  "rule_kind": "cve",
  "match_mode": "contains",
  "pattern": "CVE-2026-1234",
  "severity": "critical",
  "source_constraints": {
    "source_tags": ["advisory"]
  },
  "metadata": {
    "descriptor": "active exploitation"
  }
}
```

Supported rule kinds are `keyword`, `regex`, `descriptor`, `classification`, `entity`, `ioc`, and `cve`. The first implementation is deterministic text matching; richer extractors can later populate descriptors, classifications, entities, and IOCs before matching. `source_constraints` is structured JSON and may include `source_ids`, `source_types`, `source_tags`, or `url_contains`.

Review a content alert:
```json
{
  "status": "read"
}
```

Alert review states are `unread`, `read`, and `dismissed`.

List Watchlist Updates with server-backed triage filters:
```bash
curl "$BASE/api/v1/watchlists/items?watchlist_id=42&sort=alert_severity_desc&has_alert=true&include_alert_summary=true&size=50" \
  -H "Authorization: Bearer $TOKEN"
```

Batch mark all matching unread alert Updates as reviewed:
```json
{
  "watchlist_id": 42,
  "scope": {
    "reviewed": false,
    "has_alert": true,
    "alert_status": "unread"
  },
  "reviewed": true,
  "limit": 500
}
```

Save an Updates review view for a Watchlist:
```json
{
  "name": "Unread critical alerts",
  "filters": {
    "reviewed": false,
    "has_alert": true,
    "alert_severity": "critical"
  },
  "sort": "alert_severity_desc",
  "is_default": false
}
```

Trigger a run:
```json
{}
```

Preview candidates without ingestion:
```json
{}
```

Create/update a template with visual composer metadata (dual storage):
```json
{
  "name": "newsletter_template",
  "format": "md",
  "content": "# {{ title }}\n{% for item in items %}\n- {{ item.title }}\n{% endfor %}",
  "overwrite": true,
  "composer_ast": {
    "schema_version": "1.0.0",
    "nodes": [
      {"id": "header-1", "type": "HeaderBlock", "source": "# {{ title }}"},
      {"id": "loop-1", "type": "ItemLoopBlock", "source": "{% for item in items %}\n- {{ item.title }}\n{% endfor %}"}
    ]
  },
  "composer_schema_version": "1.0.0",
  "composer_sync_hash": "abc123",
  "composer_sync_status": "in_sync"
}
```

Manually generate a section draft (preview only):
```json
{
  "run_id": 42,
  "block_id": "intro-1",
  "prompt": "Write a concise introduction in 2 sentences.",
  "input_scope": "all_items",
  "style": "neutral",
  "length_target": "short"
}
```

Run final flow-check (manual suggest/auto modes):
```json
{
  "run_id": 42,
  "mode": "suggest_only",
  "sections": [
    {"id": "intro-1", "content": "Intro paragraph"},
    {"id": "body-1", "content": "Body paragraph"}
  ]
}
```

## Filters (job-level)

Payload shape:
```json
{
  "filters": [
    {"type": "keyword", "action": "include", "value": {"keywords": ["llm", "ai"]}},
    {"type": "regex", "action": "exclude", "value": {"pattern": "sponsored"}}
  ],
  "require_include": true
}
```

Filter types: `keyword`, `author`, `date_range`, `regex`, `all`  
Actions: `include`, `exclude`, `flag`

## Include-only quick semantics

`require_include` can come from:
- Job filters payload (`job_filters.require_include`)
- Org/admin default (when the job value is unset)

Decision summary:

| Effective `require_include` | Any active include rule | Include match found | Result |
| --- | --- | --- | --- |
| `false` | no | n/a | Item may ingest unless excluded |
| `false` | yes | no | Item may ingest unless excluded |
| `false` | yes | yes | Item ingests (and can be flagged) |
| `true` | no | n/a | Item is filtered (include-only guard) |
| `true` | yes | no | Item is filtered |
| `true` | yes | yes | Item ingests (unless excluded by higher-priority rule) |

Notes:
- Exclude rules still win when they match.
- Invalid regex filters fail safely (no crash); include-only mode can still filter all items when no include matches.

## Template Composer Notes

- Runtime rendering still uses `content` as source of truth.
- Visual composer metadata is persisted in parallel (`composer_ast`, schema, sync status/hash).
- Unsupported Jinja syntax is preserved as `RawCodeBlock` in visual mode; it is never dropped.
- Inline section generation and final flow-check are manual-only authoring/preview helpers in v1.
- `compose/flow-check` does not schedule jobs or auto-run future outputs; it only returns suggestions/patches.

## OPML import/export examples

Import OPML with optional defaults:
```bash
curl -X POST "$BASE/api/v1/watchlists/sources/import" \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@subscriptions.opml" \
  -F "active=true" \
  -F "group_id=12" \
  -F "tags=research" \
  -F "tags=ai"
```

Export all sources as OPML:
```bash
curl "$BASE/api/v1/watchlists/sources/export" \
  -H "Authorization: Bearer $TOKEN" \
  -o watchlists_sources.opml
```

Export with filters (`group` supports repeated params for OR semantics):
```bash
curl "$BASE/api/v1/watchlists/sources/export?type=rss&tag=ai&group=12&group=19" \
  -H "Authorization: Bearer $TOKEN" \
  -o watchlists_rss_ai_groups_12_19.opml
```

Filter semantics for OPML export:
- Repeated `group` values are OR-ed.
- `tag` + `group` are combined with AND semantics.
- Unknown `group` ids return HTTP 200 with an empty OPML source list.

## Dedup/seen tools

Inspect per-source seen state:
```bash
curl "$BASE/api/v1/watchlists/sources/123/seen?keys_limit=20" \
  -H "Authorization: Bearer $TOKEN"
```

Clear seen state for a source (also clears source backoff by default):
```bash
curl -X DELETE "$BASE/api/v1/watchlists/sources/123/seen?clear_backoff=true" \
  -H "Authorization: Bearer $TOKEN"
```

Admin-only inspect/reset for another user:
- Add `target_user_id=<user_id>` query param.
- Non-admin calls with `target_user_id` return `403` (`watchlists_admin_required_for_target_user`).

Cross-user read support (`target_user_id`) is also available for admin users on:
- `GET /sources`
- `GET /sources/{source_id}`
- `GET /jobs`
- `GET /jobs/{job_id}`
- `GET /jobs/{job_id}/runs`
- `GET /runs`
- `GET /runs/export.csv`
- `GET /runs/{run_id}`
- `GET /runs/{run_id}/details`
- `GET /runs/{run_id}/audio`
- `GET /runs/{run_id}/tallies.csv`
- `GET /items`
- `GET /items/{item_id}`

Sharing policy is controlled by `WATCHLIST_SHARING_MODE`:
- `admin_cross_user` (default): admin can read cross-user watchlists data
- `admin_same_org`: admin cross-user reads require overlapping org membership
- `private_only`: cross-user reads blocked

## Alerts boundary

Watchlist content alerts are user-facing notifications created when newly collected Watchlist items match user-defined descriptors, classifications, entities, keywords, CVEs, IOCs, or source constraints. Alert records preserve matched evidence, item/run/job/source IDs, snippets, and review state.

Existing run-stat alert rules remain pipeline health behavior. Conditions such as no items, high error rate, item thresholds, or run failure should be shown as health issues, not unqualified content alerts. Topic Monitoring remains an internal dependency/reference for matching and notification patterns; product Watchlists do not store user-facing rules in the separate `monitoring_watchlists` model.

## Updates triage

The selected Watchlist review queue is user-facing "Updates" copy in the WebUI while the API keeps the existing `items` route names for compatibility.

`GET /items` supports these Stage 4 triage query parameters:
- `sort`: `created_desc`, `created_asc`, `published_desc`, `published_asc`, `unread_first`, `source_asc`, or `alert_severity_desc`. Default behavior remains `created_desc`.
- `reviewed`, `queued_for_briefing`, `status`, `run_id`, `job_id`, `source_id`, `watchlist_id`, `q`, `since`, and `until`.
- `has_alert`, `alert_status`, `alert_severity`, and `alert_rule_id` for content-alert-aware review queues.
- `include_alert_summary=true` to attach compact alert context to each returned item.

`GET /items/smart-counts` accepts the same Watchlist, source, search, date, status, and alert filters. It also supports `queue_run_id` for queued-count scoping.

When `include_alert_summary=true`, each item may include:
- `total`, `unread`, `read`, and `dismissed` alert counts.
- `highest_severity`, `latest_alert_id`, `latest_alert_status`, and `latest_alert_created_at`.
- `latest_matched_text`, `rule_ids`, and `severities`.

Batch triage uses `POST /items/batch-update`. Provide exactly one of:
- `item_ids`: explicit selected/page item IDs.
- `scope`: all-filtered item criteria using `run_id`, `job_id`, `source_id`, `status`, `reviewed`, `queued_for_briefing`, `q`, `search`, `since`, `until`, `has_alert`, `alert_status`, `alert_severity`, or `alert_rule_id`.

The batch payload may set `reviewed`, `status`, and/or `queued_for_briefing`. `limit` defaults to `500` and is capped at `5000`. Responses include `matched`, `changed`, `unchanged`, `failed`, ID lists, `capped`, `exhausted`, and the effective `limit` so clients can explain partial results.

Saved item views are Watchlist-scoped and persist via `/{watchlist_id}/item-views`. Each view stores `name`, `filters`, `sort`, `is_default`, timestamps, and the owning `watchlist_id`. Filters are intentionally the same server-backed item filters used by the Updates queue.

Report/briefing handoff remains the `queued_for_briefing` flag on items. Stage 4 supports selecting and queueing evidence; it does not create defensible report artifacts, immutable evidence snapshots, weak-evidence warnings, or a full report builder. Those are Stage 5 responsibilities.

## Ingestion and persistence

- Watchlists always store run stats and scraped items in the Watchlists DB.
- Set `ingest_prefs.persist_to_media_db=true` to persist items into the Media DB.
- Items track status: `ingested`, `filtered`, `duplicate`, or `error`.

## Outputs

Create a base report from a run:
```json
{
  "run_id": 456,
  "title": "Docs digest",
  "format": "md",
  "ingest_to_media_db": true
}
```

Outputs can render templates (Markdown or HTML). Set `template_name` to use a named template.

Output provenance:
- Generated Watchlists outputs include `metadata.origin="watchlists"`.
- New outputs include `metadata.watchlist_id`, `metadata.job_id`, `metadata.run_id`, `metadata.item_ids`, and `metadata.version`.
- `GET /outputs?watchlist_id=<id>` filters by the Watchlist's jobs. Older outputs that do not yet carry `metadata.watchlist_id` are still returned when their stored `job_id` belongs to the requested Watchlist.
- `GET /outputs?run_id=<id>` and `GET /outputs?job_id=<id>` remain supported and can be combined with `watchlist_id`.

Stage 5 defensible report fields on `POST /outputs`:
- `item_ids`: explicit included item IDs from the run. Use queued Updates (`queued_for_briefing=true`) when generating a defensible report.
- `report_preset`: `auto`, `cti_osint`, `news_briefing`, or `general_research`. `auto` resolves from the Watchlist domain when available.
- `include_evidence_table`: include immutable evidence rows in the render context. Default `true`.
- `include_excluded_items`: capture same-run excluded/unselected items in the evidence snapshot. Default `true`.
- `require_reviewed_items`: warn when included Updates have not been reviewed. Default `false`.
- `allow_weak_evidence`: allow generation when readiness is `warning`. Set `false` to reject weak evidence with `422 report_readiness_warning`.
- `template_name` and `template_version`: render a stored Watchlists template. Built-in Stage 5 templates include `cti_osint_report_markdown` and `news_briefing_markdown`.

Create a CTI/OSINT report from queued Updates:
```json
{
  "run_id": 456,
  "item_ids": [1001, 1002],
  "title": "Healthcare ransomware activity - May 15",
  "format": "md",
  "report_preset": "cti_osint",
  "template_name": "cti_osint_report_markdown",
  "include_evidence_table": true,
  "include_excluded_items": true,
  "allow_weak_evidence": true
}
```

Create a news briefing with source-diversity and recency context:
```json
{
  "run_id": 789,
  "item_ids": [2101, 2104, 2108],
  "title": "Election litigation briefing",
  "format": "md",
  "report_preset": "news_briefing",
  "template_name": "news_briefing_markdown",
  "include_evidence_table": true,
  "include_excluded_items": true
}
```

Stage 5 output metadata includes:
- `report_preset`
- `report_schema_version`
- `report_snapshot_path`
- `report_readiness` with `state`, `score`, and `warnings`
- `included_item_count`
- `excluded_item_count`
- `excluded_item_total_count`
- `excluded_items_truncated`
- `source_count`
- `missing_source_count`
- `alert_count`
- `critical_alert_count`
- `weak_evidence_warning_count`

Readiness states:
- `ready`: report has included evidence and no readiness warnings.
- `warning`: report can be generated, but users should review caveats before relying on it.
- `blocked`: report cannot be generated without included evidence.
- `legacy_live_only`: older output without an immutable snapshot; the server can only expose current metadata/readiness.

Readiness warning codes currently include:
- `no_included_items`: no items were selected for the report.
- `single_source`: only one source is represented.
- `missing_source_provenance`: one or more included items lack source provenance.
- `unreviewed_items`: included items were queued but not reviewed.
- `no_alert_evidence`: CTI report has no matching content alert evidence.
- `stale_news`: news briefing contains stale included updates.
- `legacy_live_only`: output predates immutable evidence snapshots.

Fetch immutable report evidence:
```bash
curl "$BASE/api/v1/watchlists/outputs/9001/evidence" \
  -H "Authorization: Bearer $TOKEN"
```

Response shape:
```json
{
  "output_id": 9001,
  "immutable_snapshot": true,
  "readiness": {
    "state": "warning",
    "score": 76,
    "warnings": [
      {
        "code": "single_source",
        "severity": "warning",
        "message": "Only one source is represented.",
        "affected_item_ids": [1001]
      }
    ]
  },
  "snapshot": {
    "schema_version": 1,
    "snapshot_id": "watchlist-report-...",
    "generated_at": "2026-05-15T12:00:00Z",
    "preset": "cti_osint",
    "included_items": [
      {
        "id": 1001,
        "title": "Vendor advisory confirms exploitation",
        "url": "https://vendor.example/cve",
        "source_name": "Vendor Advisory",
        "published_at": "2026-05-15T10:00:00Z",
        "summary": "Active exploitation against edge devices.",
        "reviewed": true,
        "queued_for_briefing": true,
        "alerts": [
          {
            "rule_name": "Critical CVE",
            "severity": "critical",
            "matched_text": "CVE-2026-1234"
          }
        ]
      }
    ],
    "excluded_items": [
      {
        "id": 1003,
        "title": "Background market note",
        "reason": "not_queued_for_report"
      }
    ],
    "source_summary": {"unique_source_count": 2, "missing_source_count": 0},
    "included_count": 1,
    "excluded_count": 1,
    "alert_count": 1,
    "critical_alert_count": 1
  }
}
```

`GET /outputs/{output_id}/readiness` returns only the `readiness` object. For legacy outputs without `metadata.report_snapshot_path`, `/evidence` returns `immutable_snapshot=false`, `snapshot=null`, and `readiness.state="legacy_live_only"`.

Create a report plus TTS variant output (`type=tts_audio`, `format=mp3`):
```json
{
  "run_id": 456,
  "title": "Docs digest",
  "format": "md",
  "generate_tts": true,
  "tts_model": "kokoro",
  "tts_voice": "af_heart",
  "tts_speed": 1.0
}
```

Create a report and enqueue multi-voice audio briefing workflow:
```json
{
  "run_id": 456,
  "title": "Docs digest",
  "generate_audio": true,
  "target_audio_minutes": 8,
  "audio_model": "kokoro",
  "audio_voice": "af_heart",
  "audio_speed": 1.0,
  "background_audio_uri": "file:///absolute/path/to/background-bed.mp3",
  "background_volume": 0.2,
  "background_delay_ms": 500,
  "background_fade_seconds": 2.5,
  "llm_provider": "openai",
  "llm_model": "gpt-4o-mini",
  "persona_summarize": true,
  "persona_id": "analyst",
  "persona_provider": "openai",
  "persona_model": "gpt-4o-mini",
  "voice_map": {"HOST": "af_bella", "REPORTER": "am_adam"}
}
```

When `generate_audio=true`, output metadata includes:
- `audio_briefing_requested`
- `audio_briefing_task_id`
- `audio_briefing_status` (`pending`, `skipped`, or `enqueue_failed`)

Poll audio briefing status/artifact for the run:
```bash
curl "$BASE/api/v1/watchlists/runs/456/audio" \
  -H "Authorization: Bearer $TOKEN"
```

`GET /runs/{run_id}/audio` typically returns:
- `status: pending` while workflow/artifact is not ready
- `status: completed` with `download_url` when audio artifact is available

## Scrape rules quick notes

Supported fields include:
- `item_selector` / `entry_selector` (list entries)
- `title_selector` / `title_xpath`
- `link_selector` / `url_selector`
- `summary_selector` / `summary_xpath`
- `content_selector` / `content_xpath`
- `date_selector` / `published_xpath`
- `limit`, `pagination`, `alternates`

Full rules live in `tldw_Server_API/app/core/Watchlists/fetchers.py`.
