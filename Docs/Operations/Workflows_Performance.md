# Workflows Performance Tuning

Guidance for sizing and tuning the Workflows module under different backends.

## SQLite (Development / Light Workloads)

- WAL mode enabled with busy timeouts; inserts retry with backoff on contention.
- Keep concurrent writers modest. For heavier workloads, prefer Postgres.
- Use SSD/NVMe storage; avoid network filesystems.
- Tune busy timeout via `PRAGMA busy_timeout=5000` (default in code) and keep batch writes small.
- Consider Litestream or sqlite-backup for continuous backups; ensure I/O doesn’t starve the main process.

### Workflow statement caching

Reusable SQLite workflow connections deliberately use `cached_statements=0`: concurrent cached reads have returned incorrect run/owner identities. Keep this correctness safeguard enabled. Borrowed pool connections are returned even when contention temporarily empties the pool.

The [paired cache benchmark and raw results](../Performance/WORKFLOWS_SQLITE_CACHE_2026_10_16_TASK_13819.md) measure the adapter cost, validate every read, and separate valid timings from corrupt cached controls. Reproduce locally with `python Helper_Scripts/benchmarks/workflows_sqlite_cache.py --output /tmp/workflows-cache.json` after activating the project environment. It creates disposable databases and does not change server configuration.

## PostgreSQL (Recommended for Production)

- Connection Pooling
  - The Postgres backend uses `psycopg` with pooling; size pools according to CPU and workload.
  - Tune: min/max pool size, connection lifetime, and idle timeouts per environment.

- Indexes & Constraints
  - Unique `(run_id, event_seq)` guarantees ordered events per run.
  - JSONB payloads with GIN on `workflow_events.payload_json` (migrated in v3) enable JSON filtering when needed.
  - Add B-tree indexes on `workflow_runs(tenant_id, status, created_at)` to speed up filtered listings.

- Vacuum/Analyze
  - Use autovacuum; configure aggressive settings for high-churn tables like `workflow_events`.

## Engine & Queues

- Tenant/workflow concurrency caps:
  - `WORKFLOWS_TENANT_CONCURRENCY` and `WORKFLOWS_WORKFLOW_CONCURRENCY`.

- Quotas/Rate Limits
  - Keep endpoint rate limits and per-user quotas enabled outside tests; adjust for bursty clients.
  - Prefer batched submissions where possible; avoid thundering herds of ad-hoc runs.

## Monitoring

- Scrape metrics and build dashboards/duty alerts (examples under `monitoring/`).
- Track p95 step durations, queue depth, and webhook delivery outcomes per host.
- Watch DB metrics: lock waits, autovacuum activity, connection saturation, row churn in `workflow_events`.
