# Getting Started with tldw

One page to get you from zero to a running tldw server. Pick your path and follow the steps.

## Choose your setup path

```
Are you using Docker?
|
+-- Yes --> Are you setting up for one person or a team?
|   |
|   +-- One person ----> Section A: Docker single-user + WebUI (recommended)
|   |
|   +-- Team / org ----> Section B: Docker multi-user + Postgres
|
+-- No ---------------> Section C: Local single-user
```

> After your server is running, skip to [Adding LLM Providers](#adding-llm-providers) and [Verify Your Setup](#verify-your-setup).

---

## Section A: Docker single-user + WebUI (Recommended)

**Time: ~2 minutes**

### Prerequisites

- Docker Engine + Docker Compose (Docker Desktop includes both)
- Git

> **Windows:** Use WSL2 for the documented make commands. If you prefer PowerShell, run the equivalent tldw-setup command shown under each step and start Docker Desktop before Docker profiles.

### Steps

```bash
git clone https://github.com/rmusser01/tldw_server.git
cd tldw_server
make setup-docker-single
make start-docker-single
make verify-docker-single
```

`make quickstart` is the shortest alias for the same Docker single-user + WebUI lifecycle.

### What you get

| Service | URL |
| --- | --- |
| API server | http://localhost:8000 |
| API docs (Swagger) | http://localhost:8000/docs |
| WebUI | http://localhost:8080 |

### Retrieve your API key

The API key is needed for direct API access (curl, scripts, browser extension). The WebUI uses a same-origin proxy and does not require it.

```bash
make show-api-key
```

### Data storage

By default, application data lives in Docker named volumes (`app-data`, `redis_data`). No nested named volume is mounted under `/app/Databases/user_databases`. `docker compose down` preserves named volumes; `docker compose down -v` deletes them.

For host-visible storage instead, see the `docker-compose.host-storage.yml` variant in the Docker Single-User guide.

---

## Section B: Docker multi-user + Postgres

**Time: ~10 minutes**

### Prerequisites

- Docker Engine + Docker Compose
- Git

### Steps

```bash
git clone https://github.com/rmusser01/tldw_server.git
cd tldw_server
export ADMIN_USERNAME=tldw-admin
export ADMIN_PASSWORD="$(python3 -c 'import secrets; print(secrets.token_urlsafe(24))')"
make setup-docker-multi
make start-docker-multi
make verify-docker-multi
```

### What you get

| Service | URL |
| --- | --- |
| API server | http://localhost:8000 |
| API docs | http://localhost:8000/docs |
| WebUI (if overlay added) | http://localhost:8080 |

### Next: create the first admin user

Set `ADMIN_USERNAME` and the generated `ADMIN_PASSWORD` before `make setup-docker-multi` to create the first admin automatically. Keep the variables in the same shell so the manual login example can reuse them:

```bash
JWT=$(
  curl -sS -X POST http://127.0.0.1:8000/api/v1/auth/login \
    -H "Content-Type: application/x-www-form-urlencoded" \
    -d "username=$ADMIN_USERNAME" \
    -d "password=$ADMIN_PASSWORD" | jq -r '.access_token'
)

curl -sS http://127.0.0.1:8000/api/v1/auth/me \
  -H "Authorization: Bearer $JWT"
```

For full details see `Docs/User_Guides/Server/Multi-User_Postgres_Setup.md`.

### External Postgres

The public profile uses the bundled Postgres service by default. Advanced operators who already run Postgres can set the override URLs in `tldw_Server_API/Config_Files/.env` before `make start-docker-multi`:

```bash
TLDW_DATABASE_URL_OVERRIDE=postgresql://your_user:your_pass@your-host:5432/tldw_users
TLDW_JOBS_DB_URL_OVERRIDE=postgresql://your_user:your_pass@your-host:5432/tldw_jobs
```

The user must have `CREATE TABLE` permissions.

---

## Section C: Local single-user (No Docker)

**Time: ~15 minutes**

### Prerequisites

- Python 3.10+
- FFmpeg (`ffmpeg -version` to check)
- Git

> **Windows:** Use WSL2 for the documented make commands. If you prefer PowerShell, run the equivalent tldw-setup command shown under each step and start Docker Desktop before Docker profiles.

### Steps

```bash
git clone https://github.com/rmusser01/tldw_server.git
cd tldw_server
make install-local
make setup-local-single
make start-local-single
```

The install target creates a `.venv`, installs dependencies, and `make setup-local-single` configures single-user auth. The server starts at http://127.0.0.1:8000. In another terminal, run:

```bash
make verify-local-single
```

If your default `python3` is older than 3.10:

```bash
make install-local PYTHON=python3.12
```

### What you get

| Service | URL |
| --- | --- |
| API server | http://127.0.0.1:8000 |
| API docs | http://127.0.0.1:8000/docs |

The local profile does not include the WebUI by default. To add it, see the [Local Profile: Add the WebUI](../../README.md#local-profile-add-the-webui) section in the main README.

### Manual alternative (no Make)

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -e .
cp tldw_Server_API/Config_Files/.env.example tldw_Server_API/Config_Files/.env
python -m uvicorn tldw_Server_API.app.main:app --reload
```

---

## Adding LLM Providers

Works for all setup paths. You need at least one provider key to use Chat features.

**1. Edit `tldw_Server_API/Config_Files/.env` and add one or more keys:**

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AI...
GROQ_API_KEY=gsk_...
COHERE_API_KEY=...
```

**2. Restart:**

```bash
# Docker single-user + WebUI
make start-docker-single

# Local
# Stop the server (Ctrl+C) and re-run:
make start-local-single
```

**3. Verify the provider is available:**

```bash
API_KEY=$(make show-api-key)
curl -H "X-API-Key: $API_KEY" http://localhost:8000/api/v1/config/providers
```

---

## Verify Your Setup

Run these checks after any setup path:

```bash
# 1. Server health
curl -sS http://localhost:8000/health

# 2. API docs load
curl -sS http://localhost:8000/docs > /dev/null && echo "docs-ok"

# 3. Quickstart info
curl -sS http://localhost:8000/api/v1/config/quickstart

# 4. (Docker + WebUI) WebUI loads
curl -sS http://localhost:8080 > /dev/null && echo "webui-ok"
```

### Try first-value ingest/search

```bash
API_KEY=$(make show-api-key)

printf '# tldw onboarding verification\n\nThis sample verifies ingest and search with tldw-onboarding-verification-unique.\n' > /tmp/tldw-onboarding-verification.md

curl -sS -X POST http://localhost:8000/api/v1/media/add \
  -H "X-API-Key: $API_KEY" \
  -F "media_type=document" \
  -F "title=tldw onboarding verification" \
  -F "keywords=onboarding,verification" \
  -F "perform_analysis=false" \
  -F "perform_chunking=true" \
  -F "files=@/tmp/tldw-onboarding-verification.md;type=text/markdown"

curl -sS -X POST http://localhost:8000/api/v1/media/search \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "tldw-onboarding-verification-unique", "fields": ["title", "content"]}'
```

---

## Guided Setup Wizard (Optional)

For a visual configuration experience, edit `tldw_Server_API/Config_Files/config.txt`:

```ini
[Setup]
enable_first_time_setup = true
setup_completed = false
```

Restart the server, then visit http://localhost:8000/setup. The wizard walks you through provider configuration, audio setup, and more.

---

## Troubleshooting

### Docker containers do not start

```bash
docker compose -f Dockerfiles/docker-compose.single-user.yml -f Dockerfiles/docker-compose.webui.yml logs --tail=200
```

### Port 8000 or 8080 already in use

Stop the conflicting process, or change the host port mapping in the compose file (e.g., `"9000:8000"`).

### Auth errors (401/403)

- Confirm `AUTH_MODE` and `SINGLE_USER_API_KEY` are set in `tldw_Server_API/Config_Files/.env`
- For Docker, the entrypoint auto-generates a key if the placeholder is unchanged. Retrieve it with `make show-api-key`.

### Local install fails on audio dependencies

- Verify FFmpeg: `ffmpeg -version`
- Verify Python version: `python3 --version` (must be 3.10+)

### Multi-user: cannot connect to Postgres

- Confirm the bundled Postgres container is healthy, or verify `TLDW_DATABASE_URL_OVERRIDE` / `TLDW_JOBS_DB_URL_OVERRIDE` if you intentionally use external databases
- Check: `docker compose -f Dockerfiles/docker-compose.multi-user-postgres.yml logs postgres --tail=50`

### Docker ignores host config changes

The stock Docker image bakes in `Config_Files` at build time. After editing files on the host, rebuild:

```bash
docker compose --env-file tldw_Server_API/Config_Files/.env \
  -f Dockerfiles/docker-compose.single-user.yml \
  -f Dockerfiles/docker-compose.webui.yml up -d --build
```

---

## What's Next?

- **Chat**: Open the WebUI and send a message, or use the `/api/v1/chat/completions` endpoint
- **Ingest media**: Upload a PDF, paste a YouTube URL, or use the `/api/v1/media/process` endpoint
- **Speech**: Set up audio with the [CPU](./First_Time_Audio_Setup_CPU.md) or [GPU/Accelerated](./First_Time_Audio_Setup_GPU_Accelerated.md) audio guide
- **Setup wizard**: Try the guided wizard at http://localhost:8000/setup
- **API reference**: Browse the full API at http://localhost:8000/docs

---

## Detailed Guides

This quickstart covers the essentials. For deeper configuration, see:

| Topic | Guide |
| --- | --- |
| Docker single-user (full details) | [Profile_Docker_Single_User.md](./Profile_Docker_Single_User.md) |
| Docker multi-user + Postgres | [Profile_Docker_Multi_User_Postgres.md](./Profile_Docker_Multi_User_Postgres.md) |
| Local single-user (full details) | [Profile_Local_Single_User.md](./Profile_Local_Single_User.md) |
| Audio setup (CPU) | [First_Time_Audio_Setup_CPU.md](./First_Time_Audio_Setup_CPU.md) |
| Audio setup (GPU/Accelerated) | [First_Time_Audio_Setup_GPU_Accelerated.md](./First_Time_Audio_Setup_GPU_Accelerated.md) |
| Multi-user admin setup | [Multi-User_Postgres_Setup.md](../User_Guides/Server/Multi-User_Postgres_Setup.md) |
