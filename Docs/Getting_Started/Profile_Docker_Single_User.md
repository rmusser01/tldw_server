# Docker Single-User + WebUI Setup

Use this profile when you want the quickest self-hosted path: one operator, single-user API-key auth, Docker-managed persistence, and the Next.js WebUI.

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
make setup-docker-single
```

PowerShell / no-`make` equivalent:

```powershell
py -3.12 -m venv .setup-venv
.\.setup-venv\Scripts\python -m pip install --upgrade pip setuptools wheel
.\.setup-venv\Scripts\python -m pip install "typer>=0.12.0" "loguru>=0.7.0" "httpx>=0.24.0" "python-dotenv>=1.0.0" "cryptography>=41.0.0"
.\.setup-venv\Scripts\python -m tldw_Server_API.cli.wizard.cli init --profile docker-single-webui --env-file tldw_Server_API/Config_Files/.env --default --yes
```

`make setup-docker-single` creates or updates `tldw_Server_API/Config_Files/.env` for `AUTH_MODE=single_user` and generates a strong `SINGLE_USER_API_KEY` when needed. Keep that `.env` file with your backups.

`make quickstart` is the shortest alias for this same Docker single-user + WebUI lifecycle. It runs setup, start, and verification in order.

## Start

```bash
make start-docker-single
```

PowerShell / no-`make` equivalent:

```powershell
docker compose --env-file tldw_Server_API/Config_Files/.env -f Dockerfiles/docker-compose.single-user.yml -f Dockerfiles/docker-compose.webui.yml up -d --wait
```

The API starts at http://127.0.0.1:8000 and the WebUI starts at http://127.0.0.1:8080. The default browser path uses same-origin browser API requests through the WebUI proxy. Treat LAN/custom-host browser access as advanced configuration; if you intentionally need that path, set `NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE=advanced` together with `NEXT_PUBLIC_API_URL`.

Default persistence uses Docker named volumes:

- `app-data` backs `/app/Databases`, including the default SQLite AuthNZ DB, per-user databases, uploads, first-run markers, and vector stores.
- `redis_data` backs the bundled Redis container.
- No nested named volume is mounted under /app/Databases/user_databases.

`docker compose down` keeps named volumes. `docker compose down -v` deletes them and removes persisted databases, user files, and vector stores.

If you prefer host-visible storage for inspection or external backups, use `Dockerfiles/docker-compose.host-storage.yml` instead of the default profile compose file:

```bash
docker compose --env-file tldw_Server_API/Config_Files/.env \
  -f Dockerfiles/docker-compose.host-storage.yml \
  -f Dockerfiles/docker-compose.webui.yml up -d --wait
```

## Verify

```bash
make verify-docker-single
```

PowerShell / no-`make` equivalent:

```powershell
.\.setup-venv\Scripts\python -m tldw_Server_API.cli.wizard.cli verify --profile docker-single-webui --env-file tldw_Server_API/Config_Files/.env --base-url http://127.0.0.1:8000 --webui-url http://127.0.0.1:8080 --first-value
```

Manual spot checks:

```bash
curl -sS http://127.0.0.1:8000/health
curl -sS http://127.0.0.1:8000/docs > /dev/null && echo "docs-ok"
curl -sS http://127.0.0.1:8000/api/v1/config/quickstart
curl -sS http://127.0.0.1:8080 > /dev/null && echo "webui-ok"
```

## First Value

Open the WebUI at http://127.0.0.1:8080, or run the provider-independent first-value ingest/search verification. The verify command posts a small Markdown document to `/api/v1/media/add`, then searches for `tldw-onboarding-verification-unique` through `/api/v1/media/search`.

```bash
make verify-docker-single
```

This does not require an LLM provider key. Add provider keys to `tldw_Server_API/Config_Files/.env` later when you are ready to use chat or hosted model features.

## Audio Path

Stock Docker CPU/default audio works with the bundled container dependencies. If you edit host-side audio config or add host-side model files, rebuild the image or use the documented host-storage/custom image path so those changes are visible inside the container.

After this profile is running, continue with one of:

- [First-Time Audio Setup: CPU Systems](./First_Time_Audio_Setup_CPU.md)
- [First-Time Audio Setup: GPU/Accelerated Systems](./First_Time_Audio_Setup_GPU_Accelerated.md)

## Troubleshoot

- If Docker is not reachable on Windows, start Docker Desktop and run the documented commands from WSL2.
- If containers do not start, run `docker compose -f Dockerfiles/docker-compose.single-user.yml -f Dockerfiles/docker-compose.webui.yml logs --tail=200`.
- If the API is unavailable, verify no other process is using port `8000`.
- If the WebUI is unavailable, verify no other process is using port `8080`.
- If direct API calls return `401`, confirm `SINGLE_USER_API_KEY` in `tldw_Server_API/Config_Files/.env` and use it as `X-API-KEY`.
- If you need a full reset, stop the stack and remove volumes with `docker compose -f Dockerfiles/docker-compose.single-user.yml -f Dockerfiles/docker-compose.webui.yml down -v`.

## Optional Add-ons

- Add provider API keys to `tldw_Server_API/Config_Files/.env`, then restart the stack.
- Use `Dockerfiles/docker-compose.host-storage.yml` for host-visible data under `docker-data/`.
- Use the advanced WebUI environment variables only when browsers must reach a LAN host, reverse proxy, or custom domain.
