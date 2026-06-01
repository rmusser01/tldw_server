# Local Single-User Setup

Use this profile for development, local debugging, or contributor workflows where you want the API running directly on your machine with a Python virtual environment. It is a peer solo choice to Docker single-user; once the WebUI is reachable, both paths use the same first-chat completion gate.

> **Windows:** Use WSL2 for the documented make commands. If you prefer PowerShell, run the equivalent tldw-setup command shown under each step and start Docker Desktop before Docker profiles.

## Prepare

Prerequisites:

- Python 3.10+
- `ffmpeg`
- Git

```bash
git clone https://github.com/rmusser01/tldw_server.git
cd tldw_server
make install-local
make setup-local-single
```

No-`make` shortcut scripts from the repository root:

```bash
# macOS/Linux terminal
./quick-launch.sh
```

```powershell
# Windows PowerShell
.\quick-launch.ps1
```

On macOS, you can also double-click `quick-launch.command` from Finder. These shortcuts create or update `.venv`, run the `local-single` setup wizard, and start the API at `http://127.0.0.1:8000`.

PowerShell / manual no-`make` equivalent:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python -m pip install -e .
.\.venv\Scripts\python -m tldw_Server_API.cli.wizard.cli init --profile local-single --env-file tldw_Server_API/Config_Files/.env --default --yes
```

`make install-local` installs dependencies into `.venv`. `make setup-local-single` creates or updates `tldw_Server_API/Config_Files/.env` for `AUTH_MODE=single_user` and generates a strong `SINGLE_USER_API_KEY` when needed.

`make quickstart-install` remains a compatibility alias for local installation only. It does not start the server.

## Start

```bash
make start-local-single
```

PowerShell / manual no-`make` equivalent:

```powershell
.\.venv\Scripts\python -m uvicorn tldw_Server_API.app.main:app --host 127.0.0.1 --port 8000
```

The API starts at http://127.0.0.1:8000.

Start the WebUI in a second terminal when you are ready for the cohesive first-time setup flow:

```bash
cd apps/tldw-frontend
bun run dev -- -p 8080
```

Then open http://127.0.0.1:8080 to open the WebUI.

## Verify

In another terminal:

```bash
make verify-local-single
```

PowerShell / no-`make` equivalent:

```powershell
.\.venv\Scripts\python -m tldw_Server_API.cli.wizard.cli verify --profile local-single --env-file tldw_Server_API/Config_Files/.env --base-url http://127.0.0.1:8000 --first-value
```

Manual spot checks:

```bash
curl -sS http://127.0.0.1:8000/health
curl -sS http://127.0.0.1:8000/docs > /dev/null && echo "docs-ok"
curl -sS http://127.0.0.1:8000/api/v1/config/quickstart
curl -sS http://127.0.0.1:8080 > /dev/null && echo "webui-ok"
```

## First Value

Open http://127.0.0.1:8080 to open the WebUI and complete first-time setup there. The setup completion gate is the first successful chat response from your selected hosted API key or local OpenAI-compatible provider. Immediately after that, add your first source so chat can use your own material.

The CLI verify command still runs a provider-independent first-value ingest/search verification. It posts a small Markdown document to `/api/v1/media/add`, then searches for `tldw-onboarding-verification-unique` through `/api/v1/media/search`.

```bash
make verify-local-single
```

This runtime check does not require an LLM provider key. Normal first-time chat/provider setup happens in the WebUI wizard; no manual config-file changes are required for the standard solo path.

## Audio Path

Local audio setup can use host-side config and model files directly. After this profile is running, continue with one of:

- [First-Time Audio Setup: CPU Systems](./First_Time_Audio_Setup_CPU.md)
- [First-Time Audio Setup: GPU/Accelerated Systems](./First_Time_Audio_Setup_GPU_Accelerated.md)

## Troubleshoot

- If install fails due to Python version, rerun with `PYTHON=python3.12 make install-local` or another supported interpreter.
- If startup fails on audio/video dependencies, verify `ffmpeg -version`.
- If port `8000` is in use, stop the conflicting process or run `uvicorn` on another port.
- If direct API calls return `401`, confirm `SINGLE_USER_API_KEY` in `tldw_Server_API/Config_Files/.env` and use it as `X-API-KEY`.
- If WebUI setup cannot save provider settings, use `/setup` as the backend/operator recovery surface and inspect `tldw_Server_API/Config_Files/.env` only as a troubleshooting step.
- On Windows, use WSL2 for the Makefile path or run the PowerShell equivalents.

## Optional Add-ons

- Add the WebUI after the API is healthy: see [Local Profile: Add the WebUI](../../README.md#local-profile-add-the-webui).
- Keep provider setup in the WebUI first-run wizard for the normal path; use file-based configuration only for recovery, automation, or advanced deployments.
- Install development extras with `source .venv/bin/activate && pip install -e ".[dev]"`.
