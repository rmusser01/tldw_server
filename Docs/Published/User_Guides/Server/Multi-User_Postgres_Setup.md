# Multi-User Postgres Setup (AuthNZ)

**Last Updated**: 2026-02-04

## What this guide covers
This guide configures the AuthNZ user database on PostgreSQL for multi-user mode.
For robust multi-user setups, PostgreSQL is the default and recommended backend.
It does not move per-user content, notes, or workflow databases to Postgres automatically.
Those remain per-user SQLite under `Databases/user_databases/<user_id>` unless you enable Postgres Content Mode.

See `Docs/Deployment/Postgres_Migration_Guide.md` if you want to migrate content databases to Postgres.

## How multi-user + Postgres works
- `AUTH_MODE=multi_user` switches authentication to JWTs.
- `DATABASE_URL` points to the AuthNZ database (users, sessions, RBAC, orgs/teams, API keys, audit logs, usage).
- `python -m tldw_Server_API.app.core.AuthNZ.initialize` bootstraps/updates the schema and seeds baseline RBAC.
- Users authenticate via `Authorization: Bearer <jwt>`; the JWT is validated against AuthNZ tables in Postgres.
- Per-user content DBs stay on disk unless you migrate them separately.

## Prerequisites
- Python 3.11+
- PostgreSQL 13+ (16+ recommended)
- tldw_server installed with Postgres extras: `pip install -e ".[multiplayer]"`

## Quick start (Postgres-first, Docker Compose)

1) Create `.env`:
```bash
cp tldw_Server_API/Config_Files/.env.example tldw_Server_API/Config_Files/.env
```

2) Set environment variables:
```bash
AUTH_MODE=multi_user
DATABASE_URL=postgresql://tldw_user:StrongPassword@postgres:5432/tldw_users
JWT_SECRET_KEY=...
SESSION_ENCRYPTION_KEY=...
MCP_JWT_SECRET=...
MCP_API_KEY_SALT=...
tldw_production=true
```

3) Start services:
```bash
docker compose -f Dockerfiles/docker-compose.yml up -d --build
```

4) Initialize AuthNZ (creates tables + admin):
```bash
docker compose -f Dockerfiles/docker-compose.yml exec app \
  python -m tldw_Server_API.app.core.AuthNZ.initialize
```

## Option B: Postgres-only container + local app

If you want to run the API locally but still use a Postgres container:

1) Start Postgres:
```bash
docker compose -f Dockerfiles/docker-compose.postgres.yml up -d
```

2) Configure `.env` (or export env vars):
```bash
AUTH_MODE=multi_user
DATABASE_URL=postgresql://tldw_user:StrongPassword@localhost:5432/tldw_users
JWT_SECRET_KEY=...
SESSION_ENCRYPTION_KEY=...
MCP_JWT_SECRET=...
MCP_API_KEY_SALT=...
```

3) Initialize AuthNZ + run the server:
```bash
python -m tldw_Server_API.app.core.AuthNZ.initialize
python -m uvicorn tldw_Server_API.app.main:app --reload
```

## Option C: Existing Postgres (local/remote)

1) Create database and user (as a Postgres superuser):
```sql
CREATE DATABASE tldw_users;
CREATE USER tldw_user WITH ENCRYPTED PASSWORD 'StrongPassword';
GRANT ALL PRIVILEGES ON DATABASE tldw_users TO tldw_user;
```

2) Confirm the PostgreSQL version:
PostgreSQL 13+ provides `gen_random_uuid()` as a built-in function. `tldw_server` does not install database extensions during startup, so the application role does not need extension-management privileges.
```sql
SHOW server_version;
```

3) Configure `.env` (or export env vars):
```bash
AUTH_MODE=multi_user
DATABASE_URL=postgresql://tldw_user:StrongPassword@localhost:5432/tldw_users
JWT_SECRET_KEY=...
SESSION_ENCRYPTION_KEY=...
MCP_JWT_SECRET=...
MCP_API_KEY_SALT=...
```

4) Initialize AuthNZ:
```bash
python -m tldw_Server_API.app.core.AuthNZ.initialize
```

5) Start the server:
```bash
python -m uvicorn tldw_Server_API.app.main:app --reload
```

## Verify
1) Login to get a token:
```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"your-password"}'
```

2) Use the token:
```bash
curl -H "Authorization: Bearer <jwt>" \
  http://localhost:8000/api/v1/media/search
```

## Troubleshooting
- `JWT_SECRET_KEY must be set`: set a 32+ char key.
- `SQLite is not supported in production`: set `DATABASE_URL` to Postgres and `tldw_production=true`.
- `function gen_random_uuid() does not exist`: upgrade to PostgreSQL 13 or newer.
- Login fails after setup: re-run `AuthNZ.initialize` and confirm the admin user was created.

## Related docs
- `Docs/User_Guides/Server/Authentication_Setup.md`
- `Docs/User_Guides/Server/Production_Hardening_Checklist.md`
- `Docs/Deployment/Postgres_Migration_Guide.md`
