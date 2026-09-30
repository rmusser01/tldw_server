# Production Hardening Checklist

Version: 1.0.0

This checklist helps you run tldw_server safely in production. It assumes you deploy behind a reverse proxy with TLS.

Authentication & Secrets
- Set `tldw_production=true` in the environment for production deployments.
- Single-user mode:
  - Set `SINGLE_USER_API_KEY` to a strong value (>= 24 chars).
  - Startup fails in production if the key is missing/weak/default.
  - Do not set `SHOW_API_KEY_ON_STARTUP` in production (leave unset or `false`).
- Multi-user mode:
  - Set `JWT_SECRET_KEY` via environment (>= 32 chars; not a template/default value).
  - Startup fails in production if the secret is missing/weak/default.
- Masking:
  - In production, the API key is masked in startup logs.

Database & Storage
- Multi-user production: Use PostgreSQL. SQLite is not supported when `tldw_production=true`.
- Set `DATABASE_URL=postgresql://<user>:<pass>@<host>:5432/<db>`.
- Configure connection pool sizes and resource limits as appropriate.
- Ensure backups and retention policies for databases and the `Databases/` directory.

Reverse Proxy & TLS
- Terminate TLS at your reverse proxy (Nginx/Traefik) and forward to the app.
- Ensure WebSocket upgrade is configured for:
  - `/api/v1/audio/stream/transcribe`
  - `/api/v1/mcp/*` (if using MCP)
- Set appropriate timeouts and keep-alive settings for long-running requests.
- If exposing the Next.js WebUI, prefer serving it behind the same origin as the API (reverse proxy) to avoid CORS complexity.
 - See reverse proxy examples: `../Deployment/Reverse_Proxy_Examples.md`
- For browser-based multi-user deployments:
  - Set `PUBLIC_WEB_BASE_URL` to the public web origin used in auth emails.
  - Prefer serving the WebUI and API behind the same origin instead of relying on browser-stored bearer tokens.
  - Require `PUBLIC_WEB_BASE_URL` to use `https://`.
  - Require `BILLING_REDIRECT_REQUIRE_HTTPS=true`.
  - Require `BILLING_REDIRECT_ALLOWLIST_REQUIRED=true`.
  - Set `BILLING_ALLOWED_REDIRECT_HOSTS` to the public app host so checkout and portal redirects cannot escape the intended customer surface.

CORS & CSRF
- Restrict CORS to trusted origins only (avoid wildcard in production).
- CSRF: Enabled by default in multi-user mode. If running browser-based clients in single-user mode, consider `CSRF_ENABLED=true` via the csurf settings (see AuthNZ docs) and tighten CORS accordingly.
 - Set CORS via env: `ALLOWED_ORIGINS=https://your.domain.com,https://admin.your.domain.com` or JSON array `ALLOWED_ORIGINS='["https://your.domain.com", "https://admin.your.domain.com"]'`.

Rate Limiting & Abuse Prevention
- RG (rate gating) global rate limiter is enabled by default in production unless you disable it with `RG_ENABLED=0` or in `config.txt`; in test environments, RG is disabled by default unless you explicitly enable it (for example via `RG_ENABLED=1` or `config.txt`).
- Treat `RG_ENABLED=0` as an emergency/debug-only setting. With RG disabled, only the auth brute-force floor remains (`check_auth_rate_limit`, 30/min, `AUTH_DEPS_AUTH_FALLBACK_RATE_LIMIT`); RG enforcement and the general `check_rate_limit` fallback are off (ADR-057). Layers outside RG (RBAC rate classes, billing plan limits, virtual-key budgets) have their own switches; see `Docs/Operations/Rate_Limits_Troubleshooting.md`.
- Tune per-module rate limiters (Chat/RAG/Evals) via their respective settings.
- Consider a network-level rate limit at the reverse proxy for additional protection.

Observability
- Metrics: Expose Prometheus metrics as needed; secure the endpoint.
- Tracing: Configure OpenTelemetry exporters if required.
- Logs: Centralize logs; avoid logging sensitive data. Set appropriate log levels.
- Dashboards: Use `Helper_Scripts/Samples/Grafana/README.md` for provisioning examples and `Docs/Deployment/Monitoring/Metrics_Cheatsheet.md` for metric-to-panel guidance.
- Request IDs: The app sets/propagates `X-Request-ID` on each response. Configure your proxy to pass it through.
- Tracing headers: Forward `traceparent` and `tracestate` headers at the proxy if using OpenTelemetry tracing.
 - Alerting: Use Prometheus alert rules from `Samples/Prometheus/alerts.yml` to notify when users approach storage quotas.
   - Warning at >90% for 15m, Critical at >98% for 5m (per sample). Mount rules into Prometheus and reference from prometheus.yml.

App Server
- Use uvicorn workers suitable for your CPU and workload (`UVICORN_WORKERS`, default 4 in Dockerfile.prod).
- Monitor CPU/RAM and tune workers/threads accordingly.
- Ensure the container runs as non-root (Dockerfile.prod creates `appuser`).

Security Scanning & CI/CD
- Run lint and security checks in CI (Ruff, Black, Bandit, pip-audit). Make them blocking gradually as the codebase is normalized.
- Consider container image scanning (e.g., Trivy) for Docker images.

Docker & Compose
- Use the provided `docker-compose.yml` for app + postgres + redis.
- Mount volumes for databases and user data.
- Ensure a secure `.env` is provided via secrets or environment.

Preflight Report
- On startup, the app logs a non-sensitive preflight report summarizing key settings (mode, database engine, CSRF, CORS size, providers, rate limiter, OTEL availability). Review logs for warnings or errors.

Incident Response
- Have a process to rotate keys (`SINGLE_USER_API_KEY`, `JWT_SECRET_KEY`) and revoke sessions.
- Configure backups and validate restore procedures.
- Run deployment-specific preflight checks as part of staging and pre-release verification.

References
- Multi-User Deployment Guide: `./Multi-User_Deployment_Guide.md`
- README sections: Authentication Setup, Configuration Options
