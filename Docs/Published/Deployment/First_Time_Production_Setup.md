# First-Time Production Setup

Version: v0.1.0
Audience: DevOps/SREs and self-hosters deploying tldw_server for the first time

This guide walks you through a secure, production-ready first deployment of tldw_server. It covers Docker Compose (recommended) and a bare-metal alternative, plus the initial setup wizard, TLS, CORS, and basic verification.

Related documents
- [Production-safe reference deployment](Production_Reference_Deployment.md)
- Reverse proxy examples (Nginx/Traefik): `Docs/Deployment/Reverse_Proxy_Examples.md`
- Postgres migration: `Docs/Deployment/Postgres_Migration_Guide.md`
- Sidecar workers (systemd/launchd): `Docs/Deployment/Sidecar_Workers.md`
- Metrics and Grafana: `Docs/Deployment/Monitoring/Metrics_Cheatsheet.md`
- Environment variables reference: `Env_Vars.md`
- General installation (local/dev): `Docs/Getting_Started/README.md`
- Production hardening checklist: `Docs/Published/User_Guides/Server/Production_Hardening_Checklist.md`

## 1) Prerequisites

- OS: Linux (Ubuntu 22.04 LTS or similar recommended). macOS/Windows supported for small installs.
- CPU/RAM: Minimum 2 vCPU / 4 GB RAM; recommended 4+ vCPU / 8+ GB.
- Storage: 50 GB+ SSD for media, databases, and models.
- FFmpeg installed (Docker image includes it; bare-metal must install via package manager).
- DNS configured for your domain (if exposing over the internet).
- TLS via reverse proxy (Nginx/Traefik/Caddy). See reverse proxy guide.
- Database: SQLite is fine for single-user; Postgres recommended for production multi-user.

Security preflight
- Decide auth mode: `single_user` (API key) or `multi_user` (JWT).
- Generate strong secrets:
  - API key: use the canonical profile guide steps in `Docs/Getting_Started/` for your selected deployment mode.
  - JWT secret: `openssl rand -base64 64`
- Restrict CORS to your site(s) with `ALLOWED_ORIGINS`.
- In production, set `tldw_production=true` to mask secrets in logs and harden defaults.

## 2) Quick Decision Matrix

- Want the fastest secure start, one host? Choose Docker Compose (recommended).
- Need package-managed services and systemd? Use bare-metal + Nginx.
- Expect multiple users/teams? Prefer Postgres and reverse proxy TLS from day one.

## 3) Option A - Production reference Docker Compose (recommended)

Use the fail-closed workflow in
`Docs/Deployment/Production_Reference_Deployment.md`. It provides the standalone
Compose topology, TLS proxy, raw environment contract, offline preflight,
verified deployment, and restore-backed rollback.

The following quickstart profile guides are for local evaluation and are
non-production starting points:

- Single-user Docker: `Docs/Getting_Started/Profile_Docker_Single_User.md`
- Multi-user Docker + Postgres: `Docs/Getting_Started/Profile_Docker_Multi_User_Postgres.md`

Do not assemble a production deployment by combining the legacy quickstart or
proxy overlays. For the reference profile:

- Keep Postgres volumes backed up and tested for restore.
- Terminate TLS at your reverse proxy and forward to `app:8000`.
- Ensure WebSocket upgrade support for `/api/v1/audio/stream/transcribe` and `/api/v1/mcp/*`.
- Configure `ALLOWED_ORIGINS` explicitly for your public domain(s).

## 4) Option B - Bare-Metal (systemd + Nginx)

Use the local profile guide for canonical non-Docker setup:

- `Docs/Getting_Started/Profile_Local_Single_User.md`

Then apply production controls in this order:

1. Add process supervision for the API service (systemd/launchd/service manager).
2. Place a reverse proxy in front of the API with TLS termination.
3. Lock down CORS origins and upload limits.
4. Apply backup and restore procedures for your selected database.

Reference implementations:

- Reverse proxy examples: `Docs/Deployment/Reverse_Proxy_Examples.md`
- Sidecar workers: `Docs/Deployment/Sidecar_Workers.md`
- Postgres migration: `Docs/Deployment/Postgres_Migration_Guide.md`

## 5) Configuration essentials

- `AUTH_MODE`: `single_user` or `multi_user`.
- `SINGLE_USER_API_KEY` (single-user) or `JWT_SECRET_KEY` (multi-user).
- `DATABASE_URL`: SQLite for dev; Postgres URL recommended in production multi-user.
- `ALLOWED_ORIGINS`: Comma-separated or JSON array of trusted origins.
- `tldw_production`: `true` in production to mask secrets and enable production guards.
- Provider keys: e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.

See `Env_Vars.md` for the complete list and `Docs/AuthNZ/AUTHNZ_DATABASE_CONFIG.md` for AuthNZ DB details.

## 6) Verify and smoke test

Use the [production reference verification
steps](Production_Reference_Deployment.md#5-verify-public-and-operator-surfaces)
for public liveness and private readiness checks.

Public `/health` returns only `{"status":"ok"}`. Detailed health, readiness,
and metrics require an authenticated operator; the proxy returns 404 for
internal and legacy readiness paths.

The remaining direct-host single-user examples are non-production quickstart
checks; they do not apply to the reference topology, which does not publish the
app port.

Auth check (non-production single-user example)
```bash
curl -s -H "X-API-KEY: $SINGLE_USER_API_KEY" http://127.0.0.1:8000/api/v1/llm/providers | jq .
```

Media test (small file)
```bash
curl -s -H "X-API-KEY: $SINGLE_USER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"url":"https://example.com/sample.pdf"}' \
  http://127.0.0.1:8000/api/v1/media/add | jq .status
```

## 7) Production checklist (summary)

- Secrets:
  - Strong `SINGLE_USER_API_KEY` (single-user) or `JWT_SECRET_KEY` (multi-user).
  - Don’t print keys on startup in production; keep `SHOW_API_KEY_ON_STARTUP` unset/false.
- Database:
  - Use Postgres for multi-user; back up volumes regularly.
  - If migrating from SQLite, follow `Postgres_Migration_Guide.md`.
- Network:
  - TLS via reverse proxy; enable WebSocket upgrades.
  - Restrict `ALLOWED_ORIGINS`.
- Observability:
  - Enable authenticated Prometheus scraping with a `system.logs` principal;
    import Grafana dashboards (see Metrics Cheatsheet).
  - Centralize logs; set `LOG_LEVEL=info`.
- Rate limits:
  - Keep global and module-specific rate limiters enabled and tuned for your users.

For a comprehensive list, see `Docs/Published/User_Guides/Server/Production_Hardening_Checklist.md`.
