# Minimal Single-User Deployment

## Overview

The maintained `local-single` profile runs the API directly from a Python
virtual environment. It uses SQLite and does not require external Redis or
PostgreSQL services. This is the smallest supported profile for development,
local research, and resource-constrained single-user installations.

The maintained Docker single-user profile is also a one-host deployment, but
its production Compose file includes Redis. Use that checked-in Compose file
instead of copying an inline service definition.

## Requirements

- Python 3.10 or newer
- 2 CPU cores
- 4 GB RAM minimum; 8 GB recommended for transcription or local models
- 10 GB disk plus storage for ingested media and models
- `ffmpeg` for audio or video processing
- Git and Make on macOS/Linux; use WSL2 or the documented PowerShell commands
  on Windows

## Local SQLite Profile

From the repository root:

```bash
make install-local
make setup-local-single
make start-local-single
```

In another terminal, verify the running API:

```bash
make verify-local-single
```

These targets keep setup and startup separate:

- `make install-local` creates `.venv` and installs the project into it.
- `make setup-local-single` creates or updates
  `tldw_Server_API/Config_Files/.env` and generates required secrets.
- `make start-local-single` starts Uvicorn with the configured environment.
- `make verify-local-single` checks the running profile.

See [Local Single-User Setup](../Getting_Started/Profile_Local_Single_User.md)
for the WebUI and platform-specific equivalents.

### Manual Equivalent

Use the virtual environment's interpreter explicitly; do not assume `python`
or `pip` is available globally:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -e .
.venv/bin/python -m tldw_Server_API.cli.wizard.cli \
  init --profile local-single \
  --env-file tldw_Server_API/Config_Files/.env \
  --default --yes
TLDW_ENV_FILE=tldw_Server_API/Config_Files/.env \
  .venv/bin/python -m uvicorn tldw_Server_API.app.main:app \
  --host 127.0.0.1 --port 8000
```

The lower-level AuthNZ initializer is a one-time bootstrap or repair command,
not part of every server launch. Automation should make its non-interactive
intent explicit:

```bash
TLDW_ENV_FILE=tldw_Server_API/Config_Files/.env \
  .venv/bin/python -m tldw_Server_API.app.core.AuthNZ.initialize \
  --non-interactive
```

If an interactive initializer reaches closed stdin, it now uses each prompt's
displayed default instead of raising `EOFError`.

## Core Configuration

The setup wizard manages these values. For a manual configuration, the core
profile is:

```bash
AUTH_MODE=single_user
DATABASE_URL=sqlite:///./Databases/users.db
REDIS_ENABLED=false
LOG_LEVEL=INFO
```

`SINGLE_USER_API_KEY`, `MCP_JWT_SECRET`, and `MCP_API_KEY_SALT` must contain
real generated secrets, not values copied unchanged from `.env.example`.
`LOG_LEVEL=WARNING` provides quieter startup output; `DEBUG` restores verbose
diagnostics.

## Observe Startup

First startup can take longer while databases are migrated and optional
components are imported. Stream both output channels to the terminal and a
file so the final exception and exit status are not lost with a closing shell:

```bash
set -o pipefail
TLDW_ENV_FILE=tldw_Server_API/Config_Files/.env \
  .venv/bin/python -m uvicorn tldw_Server_API.app.main:app \
  --host 127.0.0.1 --port 8000 \
  2>&1 | tee /tmp/tldw-startup.log
```

Do not place startup output in an undrained `subprocess.PIPE`; the application
can fill a small pipe buffer before binding its port. Stream it continuously or
write it to a file.

Probe health separately:

```bash
curl -fsS http://127.0.0.1:8000/health
```

## Docker Single-User Profile

Use the maintained profile helpers:

```bash
make setup-docker-single
make start-docker-single
make verify-docker-single
```

The equivalent Compose command uses the checked-in production image and
health checks:

```bash
docker compose \
  --env-file tldw_Server_API/Config_Files/.env \
  -f Dockerfiles/docker-compose.single-user.yml \
  up -d --build --wait
```

Follow container startup with:

```bash
docker compose \
  --env-file tldw_Server_API/Config_Files/.env \
  -f Dockerfiles/docker-compose.single-user.yml \
  logs -f app
```

This Docker profile includes Redis. The direct local profile is the supported
Redis-free option.

## Troubleshooting Early Exit

### `python` or `pip` is not found

Run `make install-local`, then use `.venv/bin/python` as shown above. On
Windows PowerShell, use `.venv\Scripts\python.exe`.

### Default or missing API key

Re-run `make setup-local-single`. For a lower-level bootstrap, run the AuthNZ
initializer with `--non-interactive`. Do not start the server with
`SINGLE_USER_API_KEY=CHANGE_ME_TO_SECURE_API_KEY`.

### `Single-user bootstrap invariant check failed`

The initializer fails closed when a single-user AuthNZ database contains extra
active users or unexpected active non-virtual API keys. It does not deactivate
accounts or revoke keys automatically.

Stop the server and preserve the configured AuthNZ database before repair:

```bash
cp Databases/users.db \
  "Databases/users.db.backup-$(date +%Y%m%d-%H%M%S)"
```

If the conflicting AuthNZ records are disposable, move the original aside and
bootstrap a fresh AuthNZ database:

```bash
mv Databases/users.db \
  "Databases/users.db.pre-repair-$(date +%Y%m%d-%H%M%S)"
TLDW_ENV_FILE=tldw_Server_API/Config_Files/.env \
  .venv/bin/python -m tldw_Server_API.app.core.AuthNZ.initialize \
  --non-interactive
```

This resets authentication state, not the separate per-user media and notes
databases. If existing accounts or API keys must be preserved, keep the backup
and repair them through supported admin controls rather than editing SQLite
with raw SQL.

### The launcher still appears silent

Run the command under **Observe Startup**, then inspect the end of the captured
file:

```bash
tail -n 100 /tmp/tldw-startup.log
```

For Docker, use `docker compose ... ps -a` and `docker compose ... logs app` to
distinguish an application exit from an out-of-memory kill or health-check
failure.

## Moving Beyond the Minimal Profile

- Use [Docker Single-User Setup](../Getting_Started/Profile_Docker_Single_User.md)
  for a bundled production-shaped single-host deployment.
- Use [Docker Multi-User PostgreSQL Setup](../Getting_Started/Profile_Docker_Multi_User_Postgres.md)
  when multiple users, PostgreSQL, or distributed operation is required.
