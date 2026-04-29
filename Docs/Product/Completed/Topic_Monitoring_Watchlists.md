# Topic Monitoring & Watchlists (Design)

Goal: Provide a configurable, privacy-respecting content monitoring feature that detects when specified topics are mentioned in user activity (chat input/output, ingestion, notes, RAG queries) and emits non-blocking alerts for admins/owners to review.

## Scope (Phase 1)
- Rule-based “watchlists” (literals/regex) with optional categories and severities.
- Scopes: global (all users), per-user; basic support for per-team and per-org when caller provides membership.
- Integration points: chat input and output; ingestion; notes (create/update/bulk); RAG search (unified/simple/advanced). Hooks emit alerts but NEVER block content.
- Alert storage and retrieval API with mark-as-read.
- Operational endpoints to manage watchlists and list alerts, gated by monitoring permissions.

## Non-Goals (Phase 1)
- Email/SMS/Slack/webhook delivery (planned in Phase 2).
- Real-time WS push to WebUI (planned in Phase 2).
- ML classifiers or external moderation APIs (local-first only for now).

## Requirements
- Opt-in and auditable: explicit configuration enabling; record the alert but do not alter the content flow.
- Enablement gate: `MONITORING_ENABLED=true` (or `monitoring.enabled` in config) must be set to activate scanning.
- Safe by default: local regex engine, bounded scan length, DoS-safe pattern validation (re-use checks from ModerationService).
- Matching policy must be robust: define normalization, case folding, word-boundary behavior, and max scan length for literals and regex.
- Transparent: Admins can inspect effective configuration and rules.
- Source of truth: DB table; `monitoring_watchlists.json` is an optional seed/import. Reload is an idempotent upsert and does not delete existing watchlists without explicit flagging.
- Evaluation order: apply `scope_type=global` watchlists before user-specific scopes (alerts can fire from both).

## Matching Policy (Phase 1)
- Preprocessing: none; use raw input text as-is to mirror ModerationService (no normalization or zero-width stripping).
- Case handling: all rules are case-insensitive by default (always `re.IGNORECASE`). Regex flags are additive and cannot disable ignorecase.
- Word boundaries (literals): none. Literal patterns are substring matches via `re.escape(...)`. Use regex rules for boundary-sensitive behavior.
- Regex format: `/pattern/flags` where flags may include `i` (ignorecase), `m` (multiline), `s` (dotall), `x` (verbose). Unknown flags are ignored. If no trailing slash is present, treat as literal.
- Regex safety: reuse ModerationService heuristics (reject if length > 2000, nested quantifiers, or >100 groups).
- Max scan length: use ModerationService chunking with `MODERATION_MAX_SCAN_CHARS` / `moderation.max_scan_chars` (default 200000). Scans cover the full text in overlapping chunks (10% overlap, min 32, max 1024).
- Future parity: any changes to normalization, boundary handling, or case rules must be implemented in both ModerationService and MonitoringService together to avoid mismatched behavior.

## Data Model
- WatchlistRule
  - `rule_id` (optional stable id; used for updates and dedupe)
  - `pattern` (literal or `/regex/`)
  - `category` (e.g., `self_harm`, `adult`, `violence`, `custom`)
  - `severity` (`info|warning|critical`)
  - `note` (free-text)
  - Optional per-rule `tags` set

- Watchlist
  - `id`, `name`, `description`
  - `enabled`
  - `scope_type` (`global|user|team|org`)
  - `scope_id` (string; null for `global`)
  - `managed_by` (`config|api`) to control reload behavior
  - `rules: List[WatchlistRule]`

- Alert (SQLite table `topic_alerts`)
  - `id` (PK), `created_at`, `user_id`
  - `scope_type`, `scope_id`
  - `source` (`chat.input|chat.output|ingestion|notes|rag`)
  - `watchlist_id`, `rule_id`, `rule_category`, `rule_severity`, `pattern`
  - `source_id` (message/note/ingestion id), `chunk_id` (optional for streams)
  - `text_snippet` (truncated), `metadata` (JSON; include similarity/dedupe hash)
  - `is_read` (bool), `read_at`

## Files & Placement
- Core: `tldw_Server_API/app/core/Monitoring/topic_monitoring_service.py`
- Schemas: `tldw_Server_API/app/api/v1/schemas/monitoring_schemas.py`
- Endpoints: `tldw_Server_API/app/api/v1/endpoints/monitoring.py`
- Config (optional): `tldw_Server_API/Config_Files/monitoring_watchlists.json`

## API Endpoints (Phase 1)
- `GET  /api/v1/monitoring/watchlists`        list watchlists (requires `system.logs`)
- `POST /api/v1/monitoring/watchlists`        create/update watchlist (requires `system.logs`)
- `DELETE /api/v1/monitoring/watchlists/{id}` delete watchlist (requires `system.logs`)
- `GET  /api/v1/monitoring/alerts`            list alerts (requires `system.logs`; filters: user_id, since, unread)
- `POST /api/v1/monitoring/alerts/{id}/read`  mark alert as read (requires `system.logs`)
- `POST /api/v1/monitoring/alerts/{id}/acknowledge` acknowledge alert (requires `system.logs`)
- `DELETE /api/v1/monitoring/alerts/{id}` dismiss alert (requires `system.logs`)
- `POST /api/v1/monitoring/reload`            reload config file (requires `system.logs`)

## Permission Model
- `/api/v1/monitoring/*` routes require `system.logs`. These are operational monitoring routes under the non-`/admin` prefix, not anonymous or end-user public routes. Admin-role principals also pass through the shared AuthNZ admin bypass.
- `/api/v1/admin/monitoring/*` routes inherit the admin role gate from the `/api/v1/admin` router. They are the control-plane surface for shared alert rules, overlay mutations, and alert history.
- There is no unauthenticated public monitoring surface. Public monitoring means non-`/admin` prefix, not anonymous access.

## Reload Semantics (Phase 1)
- Default mode is `upsert` only. No deletes or disables unless explicitly requested.
- Watchlist identity: use `id` when provided; otherwise use the natural key `(name, scope_type, scope_id)`.
- Rules: use `rule_id` when provided; otherwise compute a stable `rule_id` hash from `(pattern, category, severity, note, tags)` to prevent duplicates on reload.
- Managed scope: reload only touches watchlists with `managed_by=config` unless a request flag opts in to unmanaged items.
- Optional flags (request body or query): `delete_missing=true` (delete config-managed watchlists absent from config), `disable_missing=true` (set `enabled=false` instead of deleting). Flags are mutually exclusive.

## Integration (Phase 1)
At moderation/processing sites in endpoints, MonitoringService is called for:
- chat input (pre-LLM) with `source=chat.input`
- chat output (stream and non-stream) with `source=chat.output` (streaming: emit per chunk; similarity-based dedupe)
- ingestion pipeline with `source=ingestion`
- notes creation/update/bulk with `source=notes.*`
- RAG queries with `source=rag.*`

Monitoring emits alerts without changing moderation behavior or endpoint results.

## Streaming Dedupe (Phase 1)
- Dedupe is per stream (`source_id`) and per rule (`rule_id`) across a sliding window of recent chunks.
- Similarity uses the same raw text passed to matching. Default algorithm: SimHash over word 3-grams; treat Hamming distance <= 3 as a duplicate.
- Suggested metadata fields: `stream_id`, `chunk_id`, `chunk_seq`, `dedupe_hash`, `dedupe_algo=simhash`, `dedupe_similarity`, `dedupe_window_ms`, `scan_truncated`.
- If a chunk is deduped, skip alert creation; otherwise store the similarity metrics in `metadata`.

## Security & Privacy
- Monitoring APIs are AuthNZ claim-gated. Extend to org/team leads later.
- Opt-in via config or explicit creation of watchlists.
- Store minimal snippets (e.g., first 200 chars around the match).
- Local-first by default; webhook/email delivery is attempted only when operators configure those channels.

## Notifications (Phase 1 scaffolding)
- Local JSONL file sink gated by severity threshold.
- Topic-alert notifications may also make best-effort webhook/email attempts when configured.
- Generic notifications use the JSONL sink plus optional webhook dispatch; they do not send email in the current batch.
- Topic-alert notifications sent through `notify()` remain immediate. Digest mode applies to generic/guardian payloads routed through `notify_or_batch()`.
- Digest modes buffer generic/guardian items in memory by recipient. `hourly` and `daily` select the batching bucket; they do not run their own scheduler, so callers invoke `flush_digest()` at the intended cadence.
- On flush, each selected recipient gets one compiled `monitoring_digest` payload per recipient through the generic notification path. Generic notifications use the JSONL sink plus optional webhook dispatch, so digest delivery follows that same local-first path.
- `flush_digest()` returns the number of buffered items successfully processed. Failed digest deliveries are requeued for a later flush instead of being dropped; webhook dispatch keeps the existing best-effort retry behavior behind the generic notification path.
- Configure via env or config:
  - `MONITORING_NOTIFY_ENABLED`, `MONITORING_NOTIFY_MIN_SEVERITY`, `MONITORING_NOTIFY_FILE`, `MONITORING_NOTIFY_DIGEST_MODE`
  - `MONITORING_NOTIFY_WEBHOOK_URL`, `MONITORING_NOTIFY_EMAIL_TO`, `MONITORING_NOTIFY_SMTP_HOST`, `MONITORING_NOTIFY_EMAIL_FROM`

## Alert Lifecycle
- Mutation responses include the authoritative merged alert state as `{status, id, item}`.
- `read` marks the runtime alert as read without setting `acknowledged_at`.
- `acknowledge` marks the runtime alert as read and records `acknowledged_at`.
- `dismiss` marks the runtime alert as read and records `dismissed_at`.

## Admin Overlay Identity Contract
- Admin overlay mutation endpoints (`assign`, `snooze`, and `escalate`) only accept runtime-backed `alert:<id>` identities.
- The referenced runtime alert row must exist in the monitoring alerts database before overlay state or history events are written.
- Overlay-only identities such as `fingerprint:*` are not a public mutation contract; operators should create or locate the runtime alert first, then mutate its `alert:<id>` identity.

## Phase 2 (planned)
- Delivery channels: email, webhook, Slack.
- WebSocket push to WebUI for admins.
- ML topic classifiers & customizable taxonomies.
- Org/team scoping UI in WebUI.

## Tests
- Unit tests for rule parsing, safe regex checks, and alert creation.
- Endpoint tests for list/create/reload/alerts.
