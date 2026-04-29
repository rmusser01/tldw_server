# Monitoring

## 1. Descriptive of Current Feature Set

- Purpose: Topic Monitoring and lightweight notifications for watchlist‑based text scanning, plus admin APIs to manage watchlists and view alerts. (Metrics/tracing are covered in the Metrics module.)
- Capabilities:
  - Topic Monitoring Service: scans text against configured watchlists; creates alerts; de‑duplicates within a time window; supports `global`/`user`/`team`/`org` scopes.
  - Notification Service (Phase 1): local‑first JSONL sink; optional webhook/email stubs with retries; severity threshold.
  - Admin endpoints to CRUD watchlists, list/mark alerts, and inspect/update notification settings.
- Inputs/Outputs:
  - Input: free‑text strings from various sources (chat input/output, ingestion summaries, notes, RAG results).
  - Output: persisted `topic_alerts` rows and JSONL notification records; optional webhook/email sends.
- Related Endpoints (mounted under `/api/v1`):
  - List watchlists: tldw_Server_API/app/api/v1/endpoints/monitoring.py:37
  - Upsert watchlist: tldw_Server_API/app/api/v1/endpoints/monitoring.py:43
  - Delete watchlist: tldw_Server_API/app/api/v1/endpoints/monitoring.py:54
  - Reload watchlists: tldw_Server_API/app/api/v1/endpoints/monitoring.py:63
  - List alerts: tldw_Server_API/app/api/v1/endpoints/monitoring.py:70
  - Mark alert read: tldw_Server_API/app/api/v1/endpoints/monitoring.py:95
  - Get notification settings: tldw_Server_API/app/api/v1/endpoints/monitoring.py:104
  - Update notification settings: tldw_Server_API/app/api/v1/endpoints/monitoring.py:110
  - Send test notification: tldw_Server_API/app/api/v1/endpoints/monitoring.py:118
  - Tail recent notifications: tldw_Server_API/app/api/v1/endpoints/monitoring.py:141
- Related Schemas/DB
  - Schemas: tldw_Server_API/app/api/v1/schemas/monitoring_schemas.py:1
  - DB layer: tldw_Server_API/app/core/DB_Management/TopicMonitoring_DB.py:1

## 2. Technical Details of Features

- Architecture & Data Flow
  - TopicMonitoringService: loads watchlists JSON, compiles rules, evaluates text, deduplicates, persists alerts, and calls `NotificationService`. Singleton accessor: tldw_Server_API/app/core/Monitoring/topic_monitoring_service.py:305
  - Compiled rules hold `regex`, `category`, `severity`; dangerous regex patterns are rejected; snippets are bounded to avoid large payloads.
  - Alerts persistence via `TopicMonitoringDB` (SQLite, WAL, indexes on timestamps/users/watchlists); shape defined in module docstring: tldw_Server_API/app/core/DB_Management/TopicMonitoring_DB.py:1
  - NotificationService: JSONL file sink for topic alerts with optional best-effort webhook/email attempts: tldw_Server_API/app/core/Monitoring/notification_service.py:1
  - NotificationService also serves **guardian alert dispatch** — `dispatch_guardian_notification()` in `supervised_policy.py` calls `notify_or_batch()` to route guardian alerts through the same JSONL/webhook/email pipeline.
  - Generic notifications only use the JSONL sink plus optional webhook dispatch; they do not send email in the current implementation.
  - `notify_or_batch()` buffers payloads in `hourly`/`daily` digest modes, and `flush_digest()` currently clears buffered items and returns the count only.

- Configuration (env or config file `monitoring.*`)
  - Topic monitor:
    - `MONITORING_WATCHLISTS_FILE`: path to watchlists JSON (default `tldw_Server_API/Config_Files/monitoring_watchlists.json`).
    - `MONITORING_ALERTS_DB`: path to alerts SQLite DB (default `Databases/monitoring_alerts.db`).
    - `TOPIC_MONITOR_MAX_SCAN_CHARS`: max scanned characters per text (default `200000`).
    - `TOPIC_MONITOR_DEDUP_SECONDS`: duplicate suppression window (default `300`).
  - Notifications:
    - `MONITORING_NOTIFY_ENABLED`: `true|false` (default `false`).
    - `MONITORING_NOTIFY_MIN_SEVERITY`: `info|warning|critical` (default `critical`).
    - `MONITORING_NOTIFY_FILE`: JSONL sink path (default `Databases/monitoring_notifications.log`).
    - `MONITORING_NOTIFY_WEBHOOK_URL`: optional webhook URL.
    - Optional email (Phase 1 best‑effort): `MONITORING_NOTIFY_SMTP_HOST`, `MONITORING_NOTIFY_SMTP_PORT`, `MONITORING_NOTIFY_SMTP_STARTTLS`, `MONITORING_NOTIFY_SMTP_USER`, `MONITORING_NOTIFY_SMTP_PASSWORD`, `MONITORING_NOTIFY_EMAIL_TO`, `MONITORING_NOTIFY_EMAIL_FROM`.

- Concurrency & Performance
  - RLocks protect in‑memory watchlists and DB operations; scanning is bounded by `_max_scan_chars`.
  - Webhook/email sends use daemon threads and tenacity retries; failures are non‑blocking.
  - JSONL writes use a file lock; tails are bounded in the API.

- Error Handling & Safety
  - Rule compilation is guarded; invalid rules are skipped with warnings.
  - Alert metadata is JSON‑encoded/decoded with fallback on parse failures.
  - Admin endpoints use claim-first authorization (`get_auth_principal` + `RequireRole("admin")` / `RequirePermission(...)`) and validate inputs.

## 3. Developer-Related/Relevant Information for Contributors

- Folder Structure
  - `topic_monitoring_service.py` — load/compile rules, evaluate text, persist alerts, call notifier.
  - `notification_service.py` — JSONL sink + optional webhook/email stubs with retries and severity thresholding.
  - DB: `DB_Management/TopicMonitoring_DB.py` — SQLite wrapper for `topic_alerts`.
- Extension Points
  - Add new delivery channels by extending `NotificationService` (e.g., queue, provider SDK). Keep sends best‑effort and non‑blocking.
  - Expand scopes/team/org support by enriching `_applicable_watchlists(...)`.
- Tests
  - Topic monitoring flow: tldw_Server_API/tests/Monitoring/test_topic_monitoring.py:1
  - Notification thresholds and file sink: tldw_Server_API/tests/Monitoring/test_notification_service.py:1
  - Metrics JSON/Prometheus shape (observability): tldw_Server_API/tests/Monitoring/test_metrics_endpoints.py:1
- Local Dev Tips
  - Enable notifications locally with `MONITORING_NOTIFY_ENABLED=true` and set `MONITORING_NOTIFY_FILE` to a temp path to inspect JSONL.
  - Use the admin APIs to manage watchlists; reload via `/api/v1/monitoring/reload` after file edits.
- Pitfalls & Gotchas
  - Webhook/email sends are best‑effort and may be disabled in restricted environments; rely on JSONL for auditability.
  - Public alert mutation endpoints return minimal acknowledgements; re-list alerts for authoritative merged state.
  - Large texts are truncated for scanning; test your rules with realistic snippets.
  - Regex complexity can impact performance; prefer literals or well‑scoped regexes.
