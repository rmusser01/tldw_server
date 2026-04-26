# Docker Multi-User + Postgres Setup

Use this profile for team-style deployments with JWT auth mode, a first admin account, and bundled PostgreSQL.

> **Windows:** Use WSL2 for the documented make commands. If you prefer PowerShell, run the equivalent tldw-setup command shown under each step and start Docker Desktop before Docker profiles.

## Prepare

Prerequisites:

- Docker Engine or Docker Desktop
- Docker Compose
- Git
- Python 3.10+ for the lightweight `tldw-setup` helper used by the Makefile

```bash
git clone https://github.com/rmusser01/tldw_server.git
cd tldw_server
export ADMIN_USERNAME=tldw-admin
export ADMIN_PASSWORD="$(python3 -c 'import secrets; print(secrets.token_urlsafe(24))')"
make setup-docker-multi
```

PowerShell / no-`make` equivalent:

```powershell
$env:ADMIN_USERNAME="tldw-admin"
$env:ADMIN_PASSWORD = py -3.12 -c "import secrets; print(secrets.token_urlsafe(24))"
py -3.12 -m venv .setup-venv
.\.setup-venv\Scripts\python -m pip install --upgrade pip setuptools wheel
.\.setup-venv\Scripts\python -m pip install "typer>=0.12.0" "loguru>=0.7.0" "httpx>=0.24.0" "python-dotenv>=1.0.0" "cryptography>=41.0.0"
.\.setup-venv\Scripts\python -m tldw_Server_API.cli.wizard.cli init --profile docker-multi-postgres --env-file tldw_Server_API/Config_Files/.env --default --yes
```

`make setup-docker-multi` creates or updates `tldw_Server_API/Config_Files/.env` for `AUTH_MODE=multi_user`, bundled Postgres, required JWT/MCP/BYOK secrets, and the first admin bootstrap values. Keep that `.env` file with your backups.

## Start

```bash
make start-docker-multi
```

PowerShell / no-`make` equivalent:

```powershell
$env:TLDW_ENV_FILE=(Resolve-Path "tldw_Server_API/Config_Files/.env").Path
docker compose -f Dockerfiles/docker-compose.multi-user-postgres.yml up -d --wait
```

The API starts at http://127.0.0.1:8000. This profile uses the bundled Postgres and Redis services from `Dockerfiles/docker-compose.multi-user-postgres.yml`.

Default persistence uses Docker named volumes:

- `app-data` backs `/app/Databases`, including per-user databases, uploads, first-run markers, and vector stores.
- `postgres_data` backs the bundled Postgres database.
- `redis_data` backs the bundled Redis container.
- No nested named volume is mounted under /app/Databases/user_databases.

`docker compose down` keeps named volumes. `docker compose down -v` deletes the named volumes and removes persisted data.

## Verify

```bash
make verify-docker-multi
```

PowerShell / no-`make` equivalent:

```powershell
.\.setup-venv\Scripts\python -m tldw_Server_API.cli.wizard.cli verify --profile docker-multi-postgres --env-file tldw_Server_API/Config_Files/.env --base-url http://127.0.0.1:8000 --first-value
```

Manual spot checks:

```bash
curl -sS http://127.0.0.1:8000/health
curl -sS http://127.0.0.1:8000/docs > /dev/null && echo "docs-ok"
curl -sS http://127.0.0.1:8000/api/v1/config/quickstart
```

## First Value

Run the provider-independent first-value ingest/search check. `make verify-docker-multi` logs in with the generated admin credentials, posts a small Markdown document to `/api/v1/media/add`, then searches for `tldw-onboarding-verification-unique` through `/api/v1/media/search`.

```bash
make verify-docker-multi
```

If the first-value check fails, confirm `ADMIN_USERNAME` and `ADMIN_PASSWORD` are still exported in the shell running verification.

## Audio Path

Stock Docker CPU/default audio works with the bundled container dependencies. If you edit host-side audio config or add host-side model files, rebuild the image or use the documented host-storage/custom image path so those changes are visible inside the container.

After this profile is running, continue with one of:

- [First-Time Audio Setup: CPU Systems](./First_Time_Audio_Setup_CPU.md)
- [First-Time Audio Setup: GPU/Accelerated Systems](./First_Time_Audio_Setup_GPU_Accelerated.md)

## Troubleshoot

- If Docker is not reachable on Windows, start Docker Desktop and run the documented commands from WSL2.
- If setup fails, confirm both `ADMIN_USERNAME` and `ADMIN_PASSWORD` are set in the shell that runs setup.
- If startup fails, check the bundled `postgres` service logs and only inspect `TLDW_DATABASE_URL_OVERRIDE` / `TLDW_JOBS_DB_URL_OVERRIDE` if you intentionally configured external databases.
- If login fails, check app logs with `docker compose -f Dockerfiles/docker-compose.multi-user-postgres.yml logs --tail=200 app`.
- If port `8000` is already in use, stop the conflicting process or change the host port mapping in the compose file.
- If you need a full reset, stop the stack and remove volumes with `docker compose -f Dockerfiles/docker-compose.multi-user-postgres.yml down -v`.

## Optional Add-ons

- Add the WebUI overlay after the API profile is healthy:
  ```bash
  TLDW_ENV_FILE=$(pwd)/tldw_Server_API/Config_Files/.env \
    docker compose -f Dockerfiles/docker-compose.multi-user-postgres.yml \
    -f Dockerfiles/docker-compose.webui.yml up -d --wait
  ```
- Advanced external Postgres: keep the public bundled Postgres default unless you already operate Postgres. To use external auth/jobs databases, set these in `tldw_Server_API/Config_Files/.env` before `make start-docker-multi`:
  ```bash
  TLDW_DATABASE_URL_OVERRIDE=postgresql://your_user:your_pass@your-host:5432/tldw_users
  TLDW_JOBS_DB_URL_OVERRIDE=postgresql://your_user:your_pass@your-host:5432/tldw_jobs
  ```
- Follow `Docs/User_Guides/Server/Multi-User_Postgres_Setup.md` for production hardening, account policy, and operational details.
