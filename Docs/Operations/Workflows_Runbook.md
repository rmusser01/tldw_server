# Workflows Runbook

Operational guidance for running, migrating, backing up, and troubleshooting the Workflows module.

## Migrations

- Postgres
  - Schema management: created/updated by `WorkflowsDatabase._initialize_schema_backend()` on startup.
  - Check current schema: `SELECT version FROM workflow_schema_version;` Compare to `_CURRENT_SCHEMA_VERSION` in `Workflows_DB.py`.
  - Known migration (v3): convert `workflow_events.payload_json` from `TEXT` to `JSONB` and add GIN index; ensures efficient JSON filtering. Validate with `\d+ workflow_events`.
  - Dry-run checklist:
    - Take snapshot (see Backups) and validate restore.
    - Put the app in maintenance mode to avoid concurrent writes.
    - Apply DB migrations by starting a single server instance; verify logs for schema bump completion.
  - Rollback: stop app, restore snapshot, and restart a previous version of the app binary.

- SQLite
  - Schema is applied idempotently at startup. WAL mode and busy timeouts are enabled; write bursts use backoff.
  - If custom migration is needed, back up the DB file and run statements within a transaction (`BEGIN IMMEDIATE; ... COMMIT;`).

## Backups

- SQLite
  - Cold snapshot: stop the server and copy `Databases/workflows.db` (and `-wal` if present).
  - Hot snapshot: `sqlite3 Databases/workflows.db '.backup Databases/workflows.backup-YYYYMMDD.db'`
  - Verify: run a test instance with the backup DB mounted; check `GET /api/v1/workflows/runs`.

- Postgres
  - Logical dump: `pg_dump -Fc -f workflows-YYYYMMDD.dump $DATABASE_URL_WORKFLOWS`
  - PITR: enable WAL archiving and base backups per org policy.
  - Verify: `pg_restore -l workflows.dump` and test restore into a staging DB.

## Retention

- Artifact GC worker (optional)
  - Enable: `WORKFLOWS_ARTIFACT_GC_ENABLED=true`.
  - Configure: `WORKFLOWS_ARTIFACT_RETENTION_DAYS` (default 30), `WORKFLOWS_ARTIFACT_GC_INTERVAL_SEC`.
  - Policy: delete `file://` artifacts from disk if present and remove DB rows older than cutoff.
  - Evidence: each successful deletion appends an `artifact_gc` event to the parent run.

- Run/event retention (manual)
  - Consider periodic deletion by age for `workflow_events` and old `workflow_runs` in high-volume deployments.
  - Example (Postgres): `DELETE FROM workflow_events WHERE created_at < NOW() - INTERVAL '90 days';` (test in staging first).

## Incident Triage

1) Failed run / repeated retries
   - Start with `GET /api/v1/workflows/runs/{run_id}/investigation`.
   - Use `primary_failure.reason_code_core`, `category`, `blame_scope`, `retryable`, and `recommended_actions` to classify the incident.
   - If the investigation identifies a failed step, follow with `GET /api/v1/workflows/runs/{run_id}/steps/{step_id}/attempts` to inspect retry history.
   - Decide rerun policy from both the failure metadata and the step replay metadata:
     - Safe replay: transient failure on a replay-safe step.
     - Conditional replay: transient failure on a side-effecting or human-reviewed step.
     - Fix-before-rerun: definition, input, or policy failures. Run `POST /api/v1/workflows/preflight` after the change before retrying.
   - If the derived investigation looks stale or incomplete, fall back to `GET /runs/{run_id}/events` and artifact/log inspection.
2) User cannot see run
   - Check tenant and owner constraints; admins can filter with `owner=`.
   - Confirm DB instance: the API and tests should use the same DB path/URL; check `DATABASE_URL_WORKFLOWS`.
3) Webhook not firing
  - Confirm `WORKFLOWS_DISABLE_COMPLETION_WEBHOOKS` is not set.
  - Check `webhook_delivery` events for `blocked|failed`.
  - Validate egress policy and allowlists; inspect DLQ with `SELECT * FROM workflow_webhook_dlq`.
  - DLQ replay appends `webhook_delivery` evidence for both delivered and failed attempts.
  - Exhausted DLQ rows are parked with a future `next_attempt_at` once `WORKFLOWS_WEBHOOK_DLQ_MAX_ATTEMPTS` is reached.
  - If using signatures, ensure receiver computes HMAC over `"{ts}.{body}"` with header `X-Signature-Timestamp`.
4) Event stream out of order / missing
  - Validate uniqueness of `(run_id, event_seq)` and counters table health.
  - If workflow evidence appears sparse, remember the GC and replay paths now record `artifact_gc` and `webhook_delivery` events respectively.
5) “database is locked” (SQLite)
  - WAL mode and busy timeout should handle most cases; reduce concurrency or move to Postgres.
6) Artifact download 404/403
  - Verify artifact exists for the run (`GET /runs/{run_id}/artifacts`).
  - Check path containment vs recorded `workdir` (strict mode may block).
  - In tests, ensure the route uses the same DB instance as the test fixture.

## Scheduler Notes

- Recurring schedule discovery scans all tenant rows within each per-user scheduler DB; it no longer assumes `tenant_id="default"`.
- Manual `POST /api/v1/scheduler/workflows/{schedule_id}/run-now` uses the same routing rule as recurring execution.
- Watchlist-backed schedules must land on `watchlist_run` in the `watchlists` queue; other schedules use `workflow_run` in the `workflows` queue.
- `run_adhoc` requires the same `workflows` scope as saved-definition runs.

## Authoring Checks

- Use `POST /api/v1/workflows/preflight` before rollout, after definition edits, and before retrying a failed run from a changed workflow revision.
- Treat `definition_invalid` as a release blocker.
- Treat `unsafe_replay_step` warnings as rollout review items, not cosmetic lint.
- For persistent incidents, compare the failing run’s `status_reason` and latest attempt `reason_code_core` before deciding whether to rollback or replay.

## Maintenance

- Verify readiness endpoints `/readyz` and `/healthz` in monitoring.
- Review schema version and DLQ queue depth during upgrades.
- Periodically audit allow/deny lists and retention policy according to compliance.

## Debugging

Enable targeted debug logs to aid triage:
- `WORKFLOWS_DEBUG=1` - enable broad Workflows debug logs.
- `WORKFLOWS_ARTIFACTS_DEBUG=1` - log artifact endpoints (IDs, paths, Range headers, containment decisions).
- `WORKFLOWS_DLQ_DEBUG=1` - log DLQ list/replay requests and worker processing decisions.
