# Watchlists core

This document describes the feed/site monitoring core: sources and check runs, filters with include-gating, noise controls (dedup, backoff, auto-disable, retention), content alerts, evidence-backed outputs and templates, briefings with audio and delivery, OPML and WebSub — and the scheduled-run wiring. It is **distinct from `core/External_Sources/`** (OAuth file-sync connectors, see [ingestion.md](./ingestion.md)): Watchlists fetches public feeds/pages under egress policy.

## Authoritative files

All paths under `tldw_Server_API/` unless noted.

| Concern | File | Key symbols |
| --- | --- | --- |
| Pipeline | `app/core/Watchlists/pipeline.py` | `run_watchlist_job` — the run engine: scope resolution, per-source/per-item cancellation, feed sanitization, backoff/retention, schedule promotion |
| Fetchers | `fetchers.py` | RSS (+ paginated history with auto/atom/wordpress strategies), site articles, scrape-rules item extraction, frontpage/forum discovery, WebSub hub discovery; every fetch passes the egress policy |
| Filters | `filters.py` | keyword/author/regex/date/all matchers with regex guardrails; exclude/flag/include actions; include-gating precedence: job flag > org metadata > env, applied only when include rules exist |
| DB | `app/core/DB_Management/Watchlists_DB.py` | per-user store following the content-backend axis (SQLite default, shared Postgres when configured); sources/watchlists/jobs/runs/items/seen/alerts/templates/websub tables; soft delete + restore |
| Scheduling | `app/core/Scheduler/handlers/watchlists.py` + `app/services/workflows_scheduler.py` | the `@task watchlist_run` handler on queue `watchlists`; APScheduler schedules routed by `inputs.watchlist_job_id`; **90 s jitter** by default to avoid on-the-hour herds |
| Routes | `app/api/v1/endpoints/watchlists.py` (~100 routes) + schemas | sources CRUD (+OPML import/export, test/check-now, seen, bulk, tags/groups), schedule preview, jobs CRUD + run, runs (+cancel, details, WS stream, CSV exports, diagnostics), briefings (+ per-stage retry, audio), items/views, outputs (+evidence/readiness), templates (+validate/preview/compose), content-alert rules, containers, claim-cluster subscriptions |
| Briefings | `briefing_{contract,fulfillment,delivery}.py`, `audio_briefing_workflow.py` | occurrence-keyed **idempotent stage machine**: select → render text → persist (+audio) → delivery; fulfillment failure never fails the run |
| Outputs | `output_enrichment_handler.py`, `report_evidence.py`, `template_store.py` (+ AST composer) | Collections-DB artifacts with evidence snapshots and readiness evaluation; versioned on-disk templates |
| Alerts | `content_alerts.py`, `alert_rules.py`, `watchlist_alert_rules_db.py` | per-item content alert evaluation; see gotcha 1 on the dead legacy rules surface |
| Interop | `opml.py`, `websub.py` + the collections websub callback router | OPML parse/generate; HMAC-verified WebSub push with renewal loop |

## Scheduled and on-demand runs (dataflow)

1. A job with a `schedule_expr` gets a WorkflowSchedule row; the APScheduler rescan fires it, routed to the internal Scheduler task `watchlist_run` (the ADR-003 rule — not Jobs) with jitter.
2. On demand: `/jobs/{id}/run` runs the pipeline inline; `/sources/check-now` lazily materializes a hidden "[System] Manual Source Refresh" job with a source override.
3. Downstream Scheduler tasks: deferred output enrichment and per-occurrence briefing delivery (optionally gated on an audio-dependency task).

## Run pipeline (dataflow, condensed)

1. Open the per-user Watchlists + Collections DBs; create a running row; resolve scope, tenant, filters, include-gating.
2. Per source (cancellation checked between sources and between items): honor `defer_until` backoffs; RSS uses conditional GET; **304 → adaptive backoff** (threshold 3, 1 h doubling to a 6 h cap, jittered); **429 → Retry-After**; errors → exponential backoff and **auto-disable at 10 consecutive**; success resets counters.
3. Per item: dedup key (guid / normalized URL with tracking params stripped / title / content hash); filters evaluated **before** article fetch with short-circuit actions and tallies; include-gating filters non-matches as `filtered`; article fetch via the feed-content shortcut or site fetch under egress + selector guardrails.
4. Persistence: Media-DB ingest only when configured; **Collections upsert always** (sanitized HTML allow-list) + embedding enqueue when new/changed; seen-marking; scraped-item recording for every status (ingested/filtered/duplicate/error); per-item content alerts best-effort; retention applied per origin.
5. Finalize: cancelled/succeeded with job history and next-run computation; then briefing fulfillment (only after durable success, occurrence-idempotent, failures recorded not raised). Non-cancel exceptions mark the run failed with a **credential-sanitized** error and re-raise.

## Failure behaviors (verified)

| Case | Behavior |
| --- | --- |
| Repeated 304 / 429 / errors | Adaptive deferral / Retry-After / backoff + auto-disable |
| Run exception | Run failed with sanitized error; re-raised to the endpoint |
| Per-item media ingest failure | Item continues via Collections; only a Collections failure records `error` |
| Briefing/audio/delivery failure | Non-fatal to the run; occurrence stages record retries; per-stage retry endpoints exist |
| Preview/test routes | Bounded and side-effect-free (no run rows, no state mutation) |

## Config keys (selection; exact)

Env: the `WATCHLISTS_304_BACKOFF_*` family (base 3600 / jitter 0.1 / cap 21600 / threshold 3), `WATCHLISTS_SCHEDULER_JITTER_SEC` (90), the WS stream family (poll 1.0 s, tail 64 KiB, chunk 8 KiB), `WATCHLIST_FORUMS_ENABLED/_DELAY_MS`, `WATCHLIST_FORUM_DEFAULT_TOP_N`, `WATCHLIST_BRIEFING_MAX_ITEMS`, `WATCHLIST_OUTPUT_*_TTL_SECONDS`, `WATCHLIST_TEMPLATE_DIR`, `COLLECTIONS_FEED_MAX_CONSEC_ERRORS` (10) / `_MAX_ITEMS` / `_RETENTION_DAYS`, `WATCHLISTS_REQUIRE_INCLUDE_DEFAULT`, the `WEBSUB_*` family. Org metadata: `watchlists.require_include_default`. Per-job JSON: output prefs (ingest/history/rss/briefing/delivery) and job filters.

## Verified gotchas

1. `endpoints/watchlist_alert_rules.py` is **not mounted** — no router spec references it; the legacy rules store is effectively dead surface. Live alerting is the content-alert routes and Claims alerts. (Worth a deletion/mounting decision.)
2. Filtered and duplicate items are still written to `scraped_items` — raw item counts include noise; smart counts distinguish.
3. Collections upsert happens regardless of the media-persist flag; run-items/media ids exist only when persistence is on.
4. Run streaming is DB-polling (stats + log tail), not event push.
5. Include-gating precedence (job > org > env) plus the "only when include rules exist" condition is easy to misstate.
6. `TEST_MODE=1` short-circuits fetches with a fake item and skips first-run dedup — pipeline tests rely on it.
7. Watchlists state follows the content-backend axis: a shared Postgres content backend moves these tables too.
8. WebSub push arrives on the collections callback router (HMAC-verified), not under `/watchlists/*`.

## Governing decisions and docs

No Watchlists-specific ADR; related: ADR-003 (Scheduler routing), ADR-026 (egress), ADR-041 (recurring scheduling feasibility — Watchlists keeps its contract). Current: the module README (accurate map); no `Code_Documentation` guide exists — this doc is the reference. `Docs/Monitoring/WATCHLISTS_*` policies are ops snapshots.

## Related docs

- [jobs-background.md](./jobs-background.md) — the Scheduler queues and APScheduler feeds
- [claims-extraction.md](./claims-extraction.md) — claim-cluster subscriptions and alerts
- [ingestion.md](./ingestion.md) — the External_Sources boundary
