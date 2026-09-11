# Long-Term Admin Guide (Operations & Maintenance)

Version: v0.1.0
Audience: Operators and administrators running tldw_server in production

This guide covers day-2 operations: upgrades, backups, monitoring, capacity and cost management, security operations, and troubleshooting. Pair this with the Production Hardening checklist and Metrics Cheatsheet.

Related documents
- [Production reference deployment](Production_Reference_Deployment.md)
- First-time production setup: `Docs/Deployment/First_Time_Production_Setup.md`
- Production hardening checklist: `Docs/Published/User_Guides/Server/Production_Hardening_Checklist.md`
- Reverse proxy examples: `Docs/Deployment/Reverse_Proxy_Examples.md`
- Postgres migration: `Docs/Deployment/Postgres_Migration_Guide.md`
- Metrics & Grafana: `Docs/Deployment/Monitoring/Metrics_Cheatsheet.md`
- Environment reference: `Env_Vars.md`

## 1) Service Management

Docker Compose
- Production operations: use `make production-preflight`,
  `make production-deploy`, and `make production-rollback` exactly as documented
  in the production reference deployment.
- Generic `docker compose up -d`, rebuild commands, and overlays below are
  non-production examples; they do not provide verified backups or rollback.
- Start (non-production): `docker compose up -d`
- Stop: `docker compose down`
- Logs: `docker compose logs -f app`
- Rebuild: `docker compose build app && docker compose up -d`
- Sidecar workers: `docker compose -f Dockerfiles/docker-compose.yml -f Dockerfiles/docker-compose.workers.yml up -d --build` (see `Docs/Deployment/Sidecar_Workers.md`).
- Scale workers (CPU bound): set `UVICORN_WORKERS` env and rebuild or override at runtime.
 - The legacy `docker-compose.override.yml` is a non-production customization
   example, not the production reference profile.

systemd (bare-metal)
- Status: `sudo systemctl status tldw`
- Logs: `sudo journalctl -u tldw -f`
- Restart: `sudo systemctl restart tldw`
- Sidecar worker units/timers: `Docs/Deployment/systemd/` (see `Docs/Deployment/Sidecar_Workers.md`).

launchd (macOS)
- LaunchAgents/LaunchDaemons examples: `Docs/Deployment/launchd/` (see `Docs/Deployment/Sidecar_Workers.md`).

## 2) Upgrades & Rollbacks

Recommended (Compose deployments)
1. Set immutable `TLDW_APP_IMAGE` and `TLDW_ROLLBACK_IMAGE` values.
2. Run `make production-preflight PRODUCTION_ENV_FILE=/absolute/path`.
3. Run `make production-deploy PRODUCTION_ENV_FILE=/absolute/path`; retain its
   verified manifest.
4. Verify exact public `/health`, public 404 denials, and authenticated operator
   diagnostics and metrics.
5. If state or migration compatibility is uncertain, stop writes and run
   `make production-rollback` with that manifest. Do not use binary-only rollback.

Bare-metal
- Update code then reinstall: `pip install --upgrade .`.
- Restart the service.

Database migrations
- AuthNZ: startup runs migrations automatically; check logs for “Ensured AuthNZ migrations”.
- Content/Workflows: see `Postgres_Migration_Guide.md` for SQLite→Postgres migration and validation.

## 3) Backups & Restore

What to back up
- AuthNZ DB: Postgres (`pg_dump`) or SQLite file at `Databases/users.db`.
- Content DBs: for per-user SQLite, back up the complete `<USER_DB_BASE_DIR>/` tree; also back up any explicit content SQLite path set via `TLDW_CONTENT_SQLITE_PATH` or `[Database].sqlite_path`. For PostgreSQL, back up the migrated content database.
- Per-user data: `<USER_DB_BASE_DIR>/<user_id>/ChaChaNotes.db` and associated files.
- Chroma/vector data: volume/directory you configured (default under `Databases/user_databases`).
- Config: `.env`, `tldw_Server_API/Config_Files/config.txt`.
- Optional: `logs/`, `server.log`, `Samples/` if you modified.

`USER_DB_BASE_DIR` is defined in `tldw_Server_API.app.core.config` (defaults to `Databases/user_databases/` under the project root). Override via environment variable or `Config_Files/config.txt` as needed.

PostgreSQL (example)
```bash
# Backup
PG_DB=tldw_users
PG_USER=tldw_user
PG_HOST=localhost
pg_dump -U "$PG_USER" -h "$PG_HOST" -F c -d "$PG_DB" > /backups/${PG_DB}_$(date +%F).dump

# Restore
pg_restore -U "$PG_USER" -h "$PG_HOST" -d "$PG_DB" -c /backups/${PG_DB}_YYYY-MM-DD.dump
```

SQLite backup
1. For a raw file copy, fully stop the application and every other process that can write to its SQLite databases. For a live backup, use SQLite's backup API or `VACUUM INTO` instead of copying open database files.
2. For the default layout, copy the complete `Databases/` tree while preserving its directory structure and any `-wal`, `-shm`, or `-journal` sidecars. If `USER_DB_BASE_DIR` points outside that tree, copy the complete configured directory too.
3. Copy each explicit content SQLite path configured through `TLDW_CONTENT_SQLITE_PATH` or `[Database].sqlite_path` if it is outside those trees, together with any matching journal sidecars.
4. Verify that the backup contains every configured database path before restarting the application.

Disaster recovery (Compose)
1. Provision a new host with Docker/Compose, same `.env` and volumes.
2. Restore DB dumps/files and persistent volumes.
3. Bring up stack and validate with `/health` and smoke tests.

## 4) Monitoring & Alerting

Endpoints
- Public liveness: `GET /health` returns only `{"status":"ok"}`.
- Container-local readiness: `GET /internal/ready`; it is not a remote
  anonymous endpoint. Legacy `/ready` is operator-protected and publicly denied.
- Metrics (text): authenticated `GET /api/v1/metrics/text` with `system.logs`.
- JSON metrics: authenticated `GET /api/v1/metrics/json`.
- Chat/LLM cost and tokens: `GET /api/v1/metrics/chat`.
- Unified circuit breaker status (admin): `GET /api/v1/admin/circuit-breakers` (requires admin role + `system.logs` permission).

Grafana + Prometheus
- Use the sample dashboards and alerts referenced in `Docs/Deployment/Monitoring/Metrics_Cheatsheet.md`.
- Set `TLDW_METRICS_API_KEY_FILE` to a mode-0600 file containing an API key for
  a dedicated principal whose only diagnostic permission is `system.logs`.
- Suggested alerts: HTTP 5xx error rate, p95 latency, Postgres connection saturation, token/cost spikes, user storage near quota.

Logs
- Container logs (stdout/stderr) via `docker compose logs -f app`.
- Bare-metal via journal: `journalctl -u tldw -f`.
- Adjust verbosity with `LOG_LEVEL`.
 - Kubernetes: `kubectl logs -n tldw deploy/tldw-app -f`.

Circuit breaker observability and tuning
- Use `GET /api/v1/admin/circuit-breakers` for consolidated breaker state across in-memory and persisted rows. Filters: `state`, `category`, `service`, `name_prefix`.
- In persistent mode, active in-process breakers typically report `source="mixed"` (present in memory and persisted storage). Treat this as normal.
- Monitor persistence contention:
  - `circuit_breaker_persist_conflicts_total` should usually stay near zero.
  - Sustained growth indicates write contention on shared breaker rows; tune `CIRCUIT_BREAKER_PERSIST_MAX_RETRIES` and investigate high-churn breaker patterns.
- For multi-worker deployments, run with `CIRCUIT_BREAKER_REGISTRY_MODE=persistent` so HALF_OPEN probe limits are coordinated across workers.
- Tune `CIRCUIT_BREAKER_HALF_OPEN_LEASE_TTL_SECONDS` to exceed normal probe runtime while still recovering abandoned slots promptly.

## 5) Capacity, Performance & Cost

Application
- CPU workers: set `UVICORN_WORKERS` (default 4 in Dockerfile.prod). Monitor latency and CPU to tune.
- Caching: ensure embeddings/model caches reside on fast disk; configure per module where available.
- Background jobs: Chatbooks worker enabled by default (core backend). Control via `CHATBOOKS_CORE_WORKER_ENABLED`. Media ingest jobs worker follows the `media` route policy by default; control via `MEDIA_INGEST_JOBS_WORKER_ENABLED`, or use sidecar workers for multi-worker deployments.
- SQLite deployments: avoid multiple Uvicorn workers with in-process jobs. Use sidecar workers or Postgres for higher concurrency.

Database
- Prefer Postgres for multi-user. Tune pool sizes with `TLDW_DB_POOL_SIZE`, `TLDW_DB_MAX_OVERFLOW`, `TLDW_DB_POOL_TIMEOUT`.
- Postgres maintenance: schedule VACUUM/ANALYZE, monitor autovacuum, and size indices.
- SQLite: enable WAL; avoid concurrent heavy writers.

LLM providers & cost
- Track provider usage and cost via metrics and admin endpoints listed in the README under Admin Reporting.
- Use Virtual Keys and per-provider allowlists where applicable to constrain usage.
- Consider separate API keys per environment/user group for accountability.

RAG & embeddings
- For high concurrency, consider the enterprise embeddings worker/orchestrator topology (see Embeddings Deployment Guide).
- Place vector stores on persistent, fast storage; monitor `embedding_cache_*` metrics.
 - Kubernetes samples are provided under `Samples/Kubernetes`.

## 6) Security Operations

Secrets and auth
- Rotate `SINGLE_USER_API_KEY` and `JWT_SECRET_KEY` on a schedule; rolling restarts will invalidate old sessions as configured.
- Never print keys in production. Keep `SHOW_API_KEY_ON_STARTUP` unset/false with `tldw_production=true`.

Registration controls (multi-user)
- Toggle with `ENABLE_REGISTRATION=true|false` and `REQUIRE_REGISTRATION_CODE=true|false`.
- Default storage quota per user: `DEFAULT_STORAGE_QUOTA_MB`.

Network
- Enforce TLS at the proxy and restrict `ALLOWED_ORIGINS`.
- Ensure WebSocket upgrade rules for `/api/v1/audio/stream/transcribe` and `/api/v1/mcp/*`.
 - Caddy example: `Helper_Scripts/Samples/Caddy/Caddyfile.compose`.

Rate limiting
- Keep global and module-specific rate limiters enabled; adjust per your user base.
- Consider reverse-proxy limits as an extra control plane.

Auditing
- Centralize and retain logs. Export audit logs periodically (see Multi-User Deployment Guide examples).

## 7) Routine Tasks (Checklist)

Weekly
- Review metrics dashboards for error spikes and slow endpoints.
- Check Postgres for bloat/long-running queries.
- Validate backups (restore test in staging where possible).
- Rotate logs and prune old container images/volumes.

Monthly
- Patch OS, Docker, and dependencies (rebuild images).
- Rotate API/JWT secrets as policy dictates.
- Review CORS and proxy configs; verify TLS renewal.

## 8) Troubleshooting

Common issues and fixes
- 502/504 via proxy during long requests
  - Increase proxy timeouts; ensure WebSocket upgrade where required.
- Auth errors in production
  - Verify `AUTH_MODE` and secrets; confirm `.env` is loaded. In single-user, ensure header `X-API-KEY` is set.
- “Database is locked” (SQLite)
  - Switch to Postgres for multi-user; enable WAL mode if staying on SQLite.
- Circuit breaker status differs across workers
  - Ensure `CIRCUIT_BREAKER_REGISTRY_MODE=persistent`. `memory` mode is process-local and does not coordinate state or HALF_OPEN probes across workers.
- Repeated HALF_OPEN probe starvation after worker crashes
  - Reduce `CIRCUIT_BREAKER_HALF_OPEN_LEASE_TTL_SECONDS` if recovery is too slow, or increase it if long-running probes routinely outlive the lease.
- Frequent persistence conflict retries
  - Check `circuit_breaker_persist_conflicts_total` and increase `CIRCUIT_BREAKER_PERSIST_MAX_RETRIES` when contention is expected.
- Postgres connection saturation
  - Increase pool size (`TLDW_DB_POOL_SIZE`, `TLDW_DB_MAX_OVERFLOW`) and Postgres `max_connections`. Inspect `pg_stat_activity`.
- High cost/usage spikes
  - Check metrics `/api/v1/metrics/chat`, enforce provider/model allowlists, and rotate/limit keys.
- Slow embeddings/LLM
  - Use smaller models, enable GPU where available, or scale out orchestrator.

Direct-host diagnostics (non-production profiles only; for the production
reference, use the in-container operator commands in
`Docs/Deployment/Production_Reference_Deployment.md`)
```bash
# Basic health
curl -sS http://127.0.0.1:8000/health | jq .

# Providers
curl -sS -H "X-API-KEY: $SINGLE_USER_API_KEY" http://127.0.0.1:8000/api/v1/llm/providers | jq .

# Metrics snapshot
curl -sS -H "Authorization: Bearer $TLDW_OPERATOR_TOKEN" \
  http://127.0.0.1:8000/api/v1/metrics/text | head -n 50
```

## 9) Change Management

Recommended practice
- Maintain environments: dev → staging → prod.
- Tag releases and build images from tags; pin deployments to tags, not `main`.
- Run smoke tests after deploy; roll back quickly on failures.

## 10) References

- README admin endpoints and usage reporting: `README.md`
- Registration & AuthNZ configuration: `Docs/User_Guides/Server/Authentication_Setup.md`
- Multi-User deployment patterns: `Docs/User_Guides/Server/Multi-User_Deployment_Guide.md`
- Reverse proxy and TLS: `Docs/Deployment/Reverse_Proxy_Examples.md`
- Production-safe Compose operations: `Docs/Deployment/Production_Reference_Deployment.md`
- Postgres/SQLite backends: `Docs/Code_Documentation/Database-Backends.md`
- Metrics and dashboards: `Docs/Deployment/Monitoring/Metrics_Cheatsheet.md`
