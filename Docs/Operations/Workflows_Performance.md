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

### End-to-end WebSocket load

The [real-network WebSocket benchmark and raw results](../Performance/WORKFLOWS_WEBSOCKET_LOAD_2026_10_16_TASK_13820.md) measure connection setup, initial snapshot, history drain, live-event latency and server event-loop lag. After activating the project environment, run:

```bash
python Helper_Scripts/benchmarks/workflows_websocket_load.py --output /tmp/workflows-ws-load.json
```

Run from a Git checkout with the backend dependencies installed. The harness copies tracked working source into disposable storage and starts the production app on loopback TCP with real JWT verification and SQLite. Lifespan workers are disabled. It validates ordered delivery and authorization refusals; a nonzero exit means the run is invalid. It never connects to an existing server. Avoid other tests or benchmarks while collecting timings.

Workflow WebSockets offload synchronous owner/event reads to Starlette's bounded shared threadpool so disk waits do not block unrelated ASGI work. Readers can still queue behind other threadpool users. History delivery yields between events so writable sockets cannot monopolize the event loop, and normal peer departure finishes without an unhandled ASGI exception. The existing one-second polling interval remains part of event-delivery latency; offloading reads is a responsiveness repair, not a low-latency push protocol. Measure on deployment hardware before setting capacity or latency targets.

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
