<div align="center">
  <h1>tldw Server</h1>
  <p>Too Long; Didn't Watch - API-first media analysis and research platform</p>

  <p>
    <a href="https://github.com/rmusser01/tldw_server">
      <img alt="Made with love" src="https://img.shields.io/badge/made_with-love-red?style=for-the-badge&labelColor=orange" />
    </a>
    <a href="https://www.gnu.org/licenses/old-licenses/gpl-3.0.en.html">
      <img alt="License: GPLv3" src="https://img.shields.io/badge/license-GPLv3-blue.svg" />
    </a>
    <a href="https://github.com/rmusser01/tldw_server/actions/workflows/e2e-smoke.yml">
      <img alt="E2E Critical Smoke" src="https://github.com/rmusser01/tldw_server/actions/workflows/e2e-smoke.yml/badge.svg" />
    </a>
  </p>

  <p>Process media with 16+ LLM providers and OpenAI-compatible APIs for Chat, Embeddings, and Evals.</p>
  <p>Browser extension support is included in-repo; hosted deployment tooling is still evolving.</p>
  <p><strong>Your local open-source platform for media analysis, knowledge work, and LLM-backed creation.</strong></p>
</div>

---

## Table of Contents
<details>
<summary>Expand table of contents</summary>

- [Overview](#overview)
- [Start Here](#start-here)
- [Current Status](#current-status)
- [What's New (in the last few releases)](#whats-new-in-the-last-few-releases)
- [Privacy & Security](#privacy--security)
- [Highlights](#highlights)
- [Feature Status](#feature-status)
- [Quickstart](#quickstart)
  - [Preflight Check (Recommended)](#preflight-check-recommended)
  - [At-a-Glance Commands](#at-a-glance-commands)
  - [No-Docker Path (Makefile, Development)](#no-docker-path-makefile-development)
  - [No-Make Path (Windows-Friendly)](#no-make-path-windows-friendly)
  - [Manual Setup](#manual-setup)
  - [Local Profile: Add the WebUI](#local-profile-add-the-webui)
  - [Run the Web UI (WIP)](#run-the-web-ui-wip)
  - [Docker Compose](#docker-compose)
  - [Supporting Services via Docker](#supporting-services-via-docker)
- [Usage Examples](#usage-examples)
- [Key Endpoints](#key-endpoints)
- [Architecture & Repo Layout](#architecture--repo-layout)
- [Architecture Diagram](#architecture-diagram)
- [Networking & Limits](#networking--limits)
- [Running Tests](#running-tests)
- [CI Status & Smoke Tests](#ci-status--smoke-tests)
- [Documentation & Resources](#documentation--resources)
  - [Resource Governor Config](#resource-governor-config)
  - [OpenAI-Compatible Strict Mode (Local Providers)](#openai-compatible-strict-mode-local-providers)
  - [Chatbook Tools Guide](#chatbook-tools-guide)
- [Deployment](#deployment)
- [Monitoring](#monitoring)
  - [PostgreSQL Content Mode](#postgresql-content-mode)
- [Troubleshooting](#troubleshooting)
- [Contributing & Support](#contributing--support)
- [Developer Guides](#developer-guides)
  - [Ingestion & Media Processing Docs](#ingestion--media-processing-docs)
- [About](#about)
  - [Getting Help](#getting-help)
  - [Security Disclosures](#security-disclosures)
  - [Project Guidelines](#project-guidelines)
- [Appendix (Optional)](#appendix-optional)
- [License](#license)
- [Credits](#credits)
</details>

## Overview
**tldw_server** is an open-source, API-first platform for ingesting media, transcribing, analyzing, and retrieving knowledge from video, audio, documents, websites, and more.
It runs a FastAPI server with OpenAI-compatible Chat, Audio, Embeddings, and Evals APIs, a unified RAG pipeline, and integrations with local or hosted LLM providers.
The primary clients are the Next.js WebUI and Admin UI, with browser-extension support in the repo as well.
Long-term vision: a personal research assistant inspired by "The Young Lady's Illustrated Primer" that helps people learn, reason about, and retain what they watch or read.

Good fit for:
- Turning videos, podcasts, and documents into searchable, citable knowledge.
- Running local or hosted LLMs behind a consistent OpenAI-compatible API.
- Building research workflows with RAG, evaluation, and prompt tooling.

## Start Here

1. **Check prerequisites:** `make quickstart-prereqs` (or verify Python 3.10+ and ffmpeg manually; Docker is only required for the Docker-based profiles)
2. **Pick your setup profile:**

| Profile | Best for | Command |
|---------|----------|---------|
| [Docker single-user + WebUI](Docs/Getting_Started/Profile_Docker_Single_User.md) | Most users (Recommended) | `make quickstart` |
| [Docker multi-user + Postgres](Docs/Getting_Started/Profile_Docker_Multi_User_Postgres.md) | Teams, public deployments | See profile guide |
| [Local single-user](Docs/Getting_Started/Profile_Local_Single_User.md) | Development, debugging | `make quickstart-install` |

1. **Follow your profile guide** end-to-end — it covers install, run, verify, and next steps.

Developers working on the WebUI, extension, or shared app packages should also start with [apps/DEVELOPMENT.md](apps/DEVELOPMENT.md).

Optional add-ons (apply AFTER your base profile is healthy):
- [First-time audio setup: CPU systems](Docs/Getting_Started/First_Time_Audio_Setup_CPU.md)
- [First-time audio setup: GPU/accelerated systems](Docs/Getting_Started/First_Time_Audio_Setup_GPU_Accelerated.md)


## Current Status

Current release line:
- `0.1.31` Beta status. Expect rough edges and please report issues.
- Primary client surfaces are the Next.js WebUI, Admin UI, and browser extension.
- The `dev` branch currently contains additional unreleased work beyond `0.1.31`; see [CHANGELOG.md](CHANGELOG.md) for branch-level detail and [Docs/Published/RELEASE_NOTES.md](Docs/Published/RELEASE_NOTES.md) for the published release entry point.

<details>
<summary>Current focus and migration notes from the old Gradio version</summary>

### Active Work-in-Progress (not in order)
- Multi-User setup
- Workflows
- [Web App frontend & Browser Extension UIs](./apps/)
- Unified Admin Dashboard ([admin-ui](./admin-ui))
- Collections (read-it-later)
- Better TTS support
- Better Documentation

### Migrating From Gradio Version (pre-0.1.0)
- Backup:
    - `cp -a ./Databases ./Databases.backup`
- Update configuration:
    - Copy provider keys to `.env`.
    - For AuthNZ setup: `cp tldw_Server_API/Config_Files/.env.example tldw_Server_API/Config_Files/.env && python -m tldw_Server_API.app.core.AuthNZ.initialize`
- Database migration:
    - Inspect: `python -m tldw_Server_API.app.core.DB_Management.migrate_db status`
    - Migrate: `python -m tldw_Server_API.app.core.DB_Management.migrate_db migrate`
    - Optional: `--db-path /path/to/media.db` if not using defaults
    - If migrating content to Postgres later, use the tools under `tldw_Server_API/app/core/DB_Management/` (e.g., migration_tools.py)
- API changes:
    - Use FastAPI routes; see http://127.0.0.1:8000/docs. OpenAI-compatible endpoints are available (e.g., `/api/v1/chat/completions`).
- Frontend:
    - Next.js WebUI: `apps/tldw-frontend/` (run separately)
    - Or integrate directly against the API
</details>

## What's New (in the last few releases)

Recently shipped:
- Evaluations Recipe Framework for guided retrieval tuning, answer-quality runs, and recipe-first eval workflows.
- MCP Virtual CLI plus follow-on hardening for governed, workspace-bounded command execution.
- Writing Suite Phase 1 with manuscript CRUD, TipTap editor surfaces, tree/focus mode, and reorder support.
- Study Packs Phase 1 for jobs-backed study-material generation and remediation handoffs.
- Container snapshot publishing to GHCR with a unified `container-build-check` rollup job.
- FTUE / FTUX expansion across onboarding, LLM setup, chat, watchlists, MCP Hub, moderation, quiz, and flashcards.
- Deep Research with jobs-backed runs, SSE progress, checkpoint review, and chat handoffs.
- Shared workspace cloning plus "Shared With Me" flows and privilege-aware sharing rules.
- Companion Home dashboard, notifications surfaces, and integrations / scheduled-task management.
- MCP Hub governance pack management and broader ACP workspace discovery/health support.
- Audio installer and bundle follow-through, including curated `kitten_tts` and `pocket_tts_cpp` paths.

Currently landing on `dev` (post-`0.1.31` branch work):
- Writing Suite Phases 2-4 with characters, world info, plot/research tools, AI analysis, agent chat, and live writing feedback.
- Persona-routed onboarding, Mission Control home flows, and storage quota warnings.
- Browser web clipper support for capturing pages into the research workflow.
- Study suggestions for quiz and flashcard workflows.
- Additional sandbox / ACP ergonomics and `llama.cpp` / `chatllm` OCR backend expansion.

See [CHANGELOG.md](CHANGELOG.md) for the full running history and [Docs/Published/RELEASE_NOTES.md](Docs/Published/RELEASE_NOTES.md) for published release notes.

## Privacy & Security

- Self-hosted by design; no telemetry or data collection.
- Users own and control their data; see hardening guidance for production.
- Auth modes: single-user API key or multi-user JWT.
- Security reporting and hardening docs: `SECURITY.md`, `Docs/Published/User_Guides/Server/Production_Hardening_Checklist.md`.
- Outbound URL egress policy blocks SSRF to private networks and disallowed ports for media downloads (audio/video/doc URLs), with test-mode DNS relaxations for hostnames.

## Highlights

- Ingest and process media (video, audio, docs, web) with OCR and chunking.
- OpenAI-compatible APIs for Chat, Audio (STT/TTS), Embeddings, and Evals.
- Hybrid search + RAG (BM25 + vector + rerank) with ChromaDB or pgvector.
- Multi-provider LLM and TTS/STT support, including local models.
- MCP server with JWT/RBAC, tool execution, WebSockets, and metrics.
- Next.js WebUI and Admin UI (both in active development).

<details>
<summary>Full highlights (expanded)</summary>

- Media ingestion & processing: video, audio, PDFs, EPUB, DOCX, HTML, Markdown, XML, MediaWiki dumps; OCR for PDFs/images; metadata extraction; configurable chunking.
- Custom-built Chunking library, tldw_Chunker, supporting token, word, sentence, paragraph, semantic, hierarchical and template chunking approaches.
- Audio & speech: real-time and file STT via faster_whisper, NVIDIA NeMo (Canary/Parakeet), Qwen2Audio; diarization/VAD; TTS backends — commercial: OpenAI, ElevenLabs; local: Kokoro, PocketTTS (Python/ONNX), PocketTTS.cpp (native binary), LuxTTS, Higgs, Chatterbox, Dia, VibeVoice, VibeVoice Realtime, NeuTTS, IndexTTS2, Supertonic, Supertonic2, Qwen3-TTS, EchoTTS; voice catalog + audio jobs queue.
- Audiobooks: parse + chapter detection, per-chapter voice settings, optional TTS provider overrides (alignment/subtitles Kokoro-only), and M4B packaging (API-only).
- Search & retrieval (RAG): hybrid BM25 + vector (ChromaDB/pgvector), re-ranking, contextual retrieval, OpenAI-compatible embeddings, vector stores API, and media embeddings ingestion. 50+ optional parameters available for tuning.
- Chat & providers: `/api/v1/chat/completions` (OpenAI-compatible), 16+ providers (commercial + self-hosted), character chat, budgets/allowlists, moderation endpoint.
- Knowledge management: notes, prompt library, character cards, soft-delete with recovery, Chatbooks import/export, flashcards (.apkg), reading items/highlights.
- Prompt Studio & evaluations: projects, prompt testing/optimization, unified evaluation APIs (G-Eval, RAG, OCR, embeddings A/B tests, batch metrics). Full evaluations and prompt management.
- Research & web scraping: multi-provider web search, academic paper search (arXiv/PubMed/etc.), scraping jobs with cookies/progress, aggregation/final answers.
- Connectors: Google Drive + Notion OAuth import; connector policies/quotas.
- Outputs & artifacts: templated outputs (Markdown/HTML/MP3 via TTS), persisted artifacts with retention controls.
- Automation: watchlists (sources/groups/tags, OPML import/export, scheduled runs) and workflows engine/scheduler (WIP).
- MCP Server: production MCP with JWT/RBAC, tool execution, WebSockets, metrics, and health endpoints. Use existing tools or add your own using a handy guide. Setup tool categories and collections, allowing for easier context management.
- Admin/Ops: orgs/teams/roles, API key management, audit logging, usage/LLM cost reporting, Prometheus metrics + Grafana dashboards, PostgreSQL content mode + backup/restore helpers.

</details>


## Feature Status

See the full [Feature Status Matrix at `Docs/Published/Overview/Feature_Status.md`](./Docs/Published/Overview/Feature_Status.md).

## Quickstart

`pip install tldw_server` is not generally available yet (PyPI publishing is in progress).
For now, use this repository checkout and the setup paths below (`make` and no-`make`).

### Preflight Check (Recommended)

If you plan to use Makefile quickstart targets:
```bash
make quickstart-prereqs
```

If `make` is unavailable (common on Windows), run the equivalent checks manually:
```powershell
py -3.12 --version  # or py -3.13 / -3.11 / -3.10
ffmpeg -version
docker --version    # only if using Docker paths
```

### At-a-Glance Commands

Choose one install path:

| Goal | Command |
|------|---------|
| API only (local Python, no Docker) | `make quickstart-install` |
| API only (Docker) | `make quickstart-docker` |
| API + WebUI (Docker) | `make quickstart-docker-webui` |
| No `make` available (common on Windows) | See [No-Make Path (Windows-Friendly)](#no-make-path-windows-friendly) |

```bash
git clone https://github.com/rmusser01/tldw_server.git && cd tldw_server

# Recommended default: Docker single-user + WebUI
make quickstart

# API-only Docker path:
# make quickstart-docker

# Explicit full-stack Docker path:
# make quickstart-docker-webui

# Local development path (API only):
# make quickstart-install
# If `python3` is older than 3.10 on your machine:
# make quickstart-install PYTHON=python3.13  # or python3.12 / python3.11 / python3.10

# Force a full image rebuild when needed:
# make quickstart-docker DOCKER_BUILD=true
# make quickstart-docker-webui DOCKER_BUILD=true
```

If `make` is unavailable, use [No-Make Path (Windows-Friendly)](#no-make-path-windows-friendly).

### Default Docker Path (Makefile)

```bash
# from repo root
make quickstart
```

This target:
- Starts the Docker single-user + WebUI setup.
- Uses the existing `quickstart-docker-webui` flow under the hood.
- Brings up the API at `http://localhost:8000` and WebUI at `http://localhost:8080`.
- Keeps the default browser path on same-origin browser API requests through the WebUI proxy.

Want a more advanced deployment?
- Team/public deployment: [Docker Multi-User + Postgres Profile](Docs/Getting_Started/Profile_Docker_Multi_User_Postgres.md)
- API-only Docker deployment: `make quickstart-docker`
- Use the advanced/custom-host path for LAN, reverse-proxy, or custom-domain browser access only when you need browsers to call a non-default host.

### No-Docker Path (Makefile, Development)

```bash
# from repo root
make quickstart-install
# If `python3` is older than 3.10 on your machine:
# make quickstart-install PYTHON=python3.13  # or python3.12 / python3.11 / python3.10
```

This target:
- Creates `.venv` if missing and installs dependencies.
- Creates `tldw_Server_API/Config_Files/.env` from `.env.example` if missing.
- Initializes AuthNZ (non-interactive).
- Starts the local API server at `http://127.0.0.1:8000`.

Verify with:
```bash
curl http://localhost:8000/health  # No auth needed!
```

Already have dependencies installed and a Python 3.10+ interpreter selected? Use `make quickstart-local` (or set `PYTHON=python3.13` / `PYTHON=python3.12` / `PYTHON=.venv/bin/python`).

### No-Make Path (Windows-Friendly)

Use these paths when `make` is not available.

API only (local Python, no Docker):
- Follow [Manual Setup](#manual-setup) below (includes PowerShell commands).

API only (Docker):
```powershell
# from repo root
if (!(Test-Path "tldw_Server_API/Config_Files/.env")) { Copy-Item "tldw_Server_API/Config_Files/.env.example" "tldw_Server_API/Config_Files/.env" }
docker compose --env-file tldw_Server_API/Config_Files/.env -f Dockerfiles/docker-compose.yml up -d --build
curl http://localhost:8000/health
```

API + WebUI (Docker):
```powershell
# from repo root
if (!(Test-Path "tldw_Server_API/Config_Files/.env")) { Copy-Item "tldw_Server_API/Config_Files/.env.example" "tldw_Server_API/Config_Files/.env" }
# Optional for non-localhost deployments:
# $env:NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE="advanced"
# $env:NEXT_PUBLIC_API_URL="http://YOUR_HOST_OR_DOMAIN:8000"
docker compose --env-file tldw_Server_API/Config_Files/.env -f Dockerfiles/docker-compose.yml -f Dockerfiles/docker-compose.webui.yml up -d --build
```

### Manual Setup

Supported Python versions:
- Minimum: Python 3.10+
- CI-tested: Python 3.11, 3.12, and 3.13
- Recommended for local development: Python 3.12

Prerequisites:
- Required: Python + ffmpeg
- Optional (for audio capture paths): PyAudio/PortAudio

Platform setup examples:
```bash
# macOS (Homebrew)
brew install ffmpeg portaudio

# Ubuntu/Debian
sudo apt update
sudo apt install -y ffmpeg portaudio19-dev python3-pyaudio

# Fedora/RHEL (RPM Fusion may be needed for ffmpeg)
sudo dnf install -y ffmpeg portaudio-devel python3-pyaudio
```

Windows notes:
- Install ffmpeg (for example with `winget` or Chocolatey).
- For PyAudio, run `pip install pyaudio` in the activated venv. If wheel/build fails, install Microsoft C++ Build Tools and retry.

1) **Create and activate a virtual environment**
```bash
# macOS/Linux: choose a supported interpreter explicitly (3.12 recommended)
python3.12 -m venv .venv  # or python3.13 / python3.11 / python3.10
source .venv/bin/activate

# Windows (PowerShell)
py -3.12 -m venv .venv  # or -3.13 / -3.11 / -3.10
.venv\Scripts\Activate.ps1

# Confirm venv interpreter version
python --version

# Install project dependencies
pip install -e .
```

2) **Configure environment**
```bash
# macOS/Linux
cp tldw_Server_API/Config_Files/.env.example tldw_Server_API/Config_Files/.env

# Windows (PowerShell)
# Copy-Item tldw_Server_API/Config_Files/.env.example tldw_Server_API/Config_Files/.env

# Edit tldw_Server_API/Config_Files/.env and set:
# SINGLE_USER_API_KEY=<at least 16 chars>
```

3) **Initialize auth** (use `--non-interactive` to skip prompts)
```bash
python -m tldw_Server_API.app.core.AuthNZ.initialize --non-interactive
```

4) **Start the server**
```bash
python -m uvicorn tldw_Server_API.app.main:app --reload
```

5) **Verify**
```bash
curl http://localhost:8000/health
# Expected: {"status":"ok",...}
```

- API docs: http://127.0.0.1:8000/docs
- Quickstart redirect: http://127.0.0.1:8000/api/v1/config/quickstart

<details>
<summary>Optional extras and pyaudio setup</summary>

```bash
# Optional pip extras
pip install -e ".[multiplayer]"   # multi-user/PostgreSQL features
pip install -e ".[tooling]"       # unified tooling smoke helpers/scripts
pip install -e ".[dev]"           # tests and linters
pip install -e ".[otel]"          # OpenTelemetry metrics/tracing

# pyaudio (needed for audio capture and some processing paths)
# Linux (Ubuntu/Debian): sudo apt install -y portaudio19-dev python3-pyaudio
# Linux (Fedora/RHEL):   sudo dnf install -y portaudio-devel python3-pyaudio
# macOS: brew install portaudio && pip install pyaudio
# Windows (PowerShell):   pip install pyaudio  # if build fails, install Microsoft C++ Build Tools
```

MCP secrets (`MCP_JWT_SECRET`, `MCP_API_KEY_SALT`) are auto-generated per-process
when MCP runs with local-only defaults (loopback IP/origin restrictions).
These ephemeral secrets do not persist across restarts.

For shared or production deployments (when MCP is accessible beyond localhost),
set these explicitly in `tldw_Server_API/Config_Files/.env` and rotate regularly.
Run these commands in your shell and paste the resulting values into `tldw_Server_API/Config_Files/.env`
(do not paste the command templates themselves):
```bash
# Generate secure secrets — run each command, then copy the output into tldw_Server_API/Config_Files/.env
openssl rand -base64 32   # paste the output as the value for MCP_JWT_SECRET
openssl rand -base64 32   # paste the output as the value for MCP_API_KEY_SALT
```
See [MCP System Admin Guide](Docs/MCP/Unified/System_Admin_Guide.md) for details.
</details>

### Next Steps by Goal

| I want to... | Guide |
|--------------|-------|
| Choose the right onboarding path | [Getting Started Index](Docs/Getting_Started/README.md) |
| Start the default Docker + WebUI setup | `make quickstart` |
| Start from local single-user, then launch the WebUI | [Local Profile: Add the WebUI](#local-profile-add-the-webui) |
| Build apps against the API locally | [Local Single-User Profile](Docs/Getting_Started/Profile_Local_Single_User.md) |
| Run on my home server with Docker | [Docker Single-User Profile](Docs/Getting_Started/Profile_Docker_Single_User.md) |
| Deploy for a team with proper security | [Docker Multi-User + Postgres Profile](Docs/Getting_Started/Profile_Docker_Multi_User_Postgres.md) |
| Develop the WebUI or browser extension | [Extension & Web UI Development Guide](apps/DEVELOPMENT.md) |

### Local Profile: Add the WebUI

If you already completed the [Local Single-User Profile](Docs/Getting_Started/Profile_Local_Single_User.md) and your API is running at `http://127.0.0.1:8000`, this is the shortest add-on path:

```bash
# from the repo root
cd apps/tldw-frontend
cp .env.local.example .env.local
bun install
bun run dev -- -p 8080
```

Confirm `.env.local` contains:
```bash
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
NEXT_PUBLIC_API_VERSION=v1
# Optional in single-user mode:
# NEXT_PUBLIC_X_API_KEY=your_single_user_api_key
```

Open `http://localhost:8080`.

Need Bun installation steps, npm fallback, or LAN/mobile access setup? See [Run the Web UI (WIP)](#run-the-web-ui-wip).

<details>
<summary>Sidecar workers (optional)</summary>

Run the API and Jobs workers as separate processes (recommended for heavier workloads or when using multiple Uvicorn workers):
```bash
./start-sidecars.sh
```
Notes:
- Sets `TLDW_WORKERS_SIDECAR_MODE=true` to keep the API process from starting in-process workers.
- Default worker list is documented in `Docs/Deployment/Sidecar_Workers.md` (regen via `python Helper_Scripts/Deployment/generate_sidecar_files.py`).
- Customize worker list with `TLDW_SIDECAR_WORKERS=chatbooks,files,data_tables,prompt_studio,privilege_snapshots,audio,media_ingest,evals_abtest` or point to a different manifest with `TLDW_WORKERS_MANIFEST=/path/to/manifest.json`.
- Logs are written to `.logs/sidecars/` by default.
- Compose overlay: `docker compose -f Dockerfiles/docker-compose.yml -f Dockerfiles/docker-compose.workers.yml up -d --build`.
- systemd and launchd examples: `Docs/Deployment/Sidecar_Workers.md`.
In-process workers are fine for light dev use, but SQLite + multiple Uvicorn workers can increase lock contention; use sidecars or Postgres for heavier workloads.
</details>

<details>
<summary>Model container cycling (optional, Docker)</summary>

For low-VRAM dev rigs, you can cycle model containers instead of running all model stacks at once:

```bash
make model-cycle \
  MODEL_CYCLE_FIRST=your-llm-container \
  MODEL_CYCLE_SECOND=your-tts-container \
  MODEL_CYCLE_EXCLUDED=postgres,redis \
  MODEL_CYCLE_FIRST_BOOT_WAIT=30 \
  MODEL_CYCLE_SECOND_BOOT_WAIT=10
```

Notes:
- Runs `Helper_Scripts/model_container_cycle.py`.
- Workflow is: stop all running containers except exclusions, start first model container, stop it, then start second model container.
- Use `MODEL_CYCLE_DRY_RUN=true` to print Docker commands without executing them.
- Container names are whatever Docker reports via `docker ps --format '{{.Names}}'`.
</details>

### Run the Web UI (WIP)

The current Next.js UI is a work in progress and may be unstable, buggy, or rough around the edges.
Use Bun (recommended) or Node.js with npm/yarn/pnpm.

Quickest full-stack path (containerized API + WebUI):
```bash
make quickstart-docker-webui
# API:   http://localhost:8000
# WebUI: http://localhost:8080
# Optional advanced/custom-host path for LAN, reverse-proxy, or custom-domain browser access:
# make quickstart-docker-webui NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE=advanced NEXT_PUBLIC_API_URL=http://YOUR_HOST_OR_DOMAIN:8000
# Optional (force image rebuild instead of cached layers):
# make quickstart-docker-webui DOCKER_BUILD=true
```
No-`make` equivalent: use [No-Make Path (Windows-Friendly)](#no-make-path-windows-friendly).

Default quickstart behavior keeps same-origin browser API requests through the WebUI proxy. When you intentionally move the browser onto an advanced/custom-host path for LAN, reverse-proxy, or custom-domain browser access, set both `NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE=advanced` and `NEXT_PUBLIC_API_URL`.

Local WebUI development (API should already be running, for example via `make quickstart`):

If Bun is not installed yet, install it first:
```bash
# macOS/Linux
curl -fsSL https://bun.sh/install | bash

# Windows (PowerShell)
powershell -c "irm bun.sh/install.ps1 | iex"

# Open a new terminal and verify:
bun --version
```

1) From the repo root:
```bash
cd apps/tldw-frontend
cp .env.local.example .env.local
```
2) Set your API URL (defaults shown):
```bash
# .env.local
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
NEXT_PUBLIC_API_VERSION=v1
# Optional for single-user mode:
# NEXT_PUBLIC_X_API_KEY=your_api_key
```
3) Install and run the dev server (use port 8080 to match the server's built-in local browser defaults):
```bash
bun install
bun run dev -- -p 8080

# npm fallback:
# npm install
# npm run dev -- -p 8080
```
Open http://localhost:8080

<details>
<summary>LAN/mobile access setup (optional)</summary>

Access from another device on your network (for example, phone/tablet):
```bash
# API (from repo root)
python -m uvicorn tldw_Server_API.app.main:app --host 0.0.0.0 --port 8000

# tldw_Server_API/Config_Files/.env
# Only needed when the browser connects from a non-localhost origin:
# ALLOWED_ORIGINS=http://YOUR_SERVER_IP:8080

# apps/tldw-frontend/.env.local
# NEXT_PUBLIC_API_URL=http://YOUR_SERVER_IP:8000

# WebUI
bun run dev -- -H 0.0.0.0 -p 8080
```
Then open `http://YOUR_SERVER_IP:8080` from your mobile device on the same network.
</details>

Tip: http://127.0.0.1:8000/api/v1/config/quickstart redirects to your configured quickstart target.

Security note: avoid exposing this quickstart setup directly to the public internet. For internet-facing access, use HTTPS with a reverse proxy and follow `Docs/Deployment/First_Time_Production_Setup.md`.

### Docker Compose

For the fastest Docker path, use the quickstart targets in [At-a-Glance Commands](#at-a-glance-commands).
This section is the canonical manual/no-`make` compose reference.
Quickstart targets skip forced rebuilds by default; pass `DOCKER_BUILD=true` to force a rebuild.

Or manually:
```bash
# Single-user mode (simplest)
docker compose --env-file tldw_Server_API/Config_Files/.env -f Dockerfiles/docker-compose.yml up -d --build
curl http://localhost:8000/health  # Verify
```

On first start in `single_user` mode, the app container now:
- Ensures `SINGLE_USER_API_KEY` is secure (generates one if missing/placeholder).
- Runs `AuthNZ.initialize --non-interactive` once per persisted Docker data volume.
- View the configured key with:
```bash
grep '^SINGLE_USER_API_KEY=' tldw_Server_API/Config_Files/.env
```

<details>
<summary>More Docker options (multi-user, overlays)</summary>

```bash
# Multi-user (Postgres users DB)
export AUTH_MODE=multi_user
export DATABASE_URL=postgresql://tldw_user:TestPassword123!@postgres:5432/tldw_users
docker compose -f Dockerfiles/docker-compose.postgres.yml up -d

# Dev overlay — unified streaming (non-prod)
docker compose -f Dockerfiles/docker-compose.yml -f Dockerfiles/docker-compose.dev.yml up -d --build

# WebUI overlay (Next.js container on :8080)
docker compose -f Dockerfiles/docker-compose.yml -f Dockerfiles/docker-compose.webui.yml up -d --build

# Check status
docker compose -f Dockerfiles/docker-compose.yml ps
docker compose -f Dockerfiles/docker-compose.yml logs -f app

# Proxy overlays
#   - Dockerfiles/docker-compose.proxy.yml (Caddy)
#   - Dockerfiles/docker-compose.proxy-nginx.yml (Nginx)

# pgvector + pgbouncer for Postgres
docker compose -f Dockerfiles/docker-compose.yml -f Dockerfiles/docker-compose.pg.yml up -d --build
```
</details>

Notes
- Run compose commands from the repository root. The base compose file at `Dockerfiles/docker-compose.yml` builds with context at the repo root and includes Postgres and Redis services.
- For `Dockerfiles/docker-compose.webui.yml`, the default quickstart leaves `NEXT_PUBLIC_API_URL` empty so browsers stay on same-origin browser API requests through the WebUI proxy. Set `NEXT_PUBLIC_API_URL` only for the advanced/custom-host path for LAN, reverse-proxy, or custom-domain browser access.
- `NEXT_PUBLIC_API_VERSION` and `NEXT_PUBLIC_X_API_KEY` are also build-time public values in the client bundle; set them explicitly for your target deployment/auth mode.
- If you need per-environment API URLs without rebuilding the WebUI image, switch to a runtime env-substitution strategy instead of compile-time `NEXT_PUBLIC_*` build args.
- The primary UI is the Next.js WebUI in `apps/tldw-frontend/`.
  - Run it separately for development, or use `Dockerfiles/docker-compose.webui.yml` for a containerized WebUI.
  - For unified streaming validation in non-prod, prefer the dev overlay above. You can also export `STREAMS_UNIFIED=1` directly in your environment.

### Supporting Services via Docker

<details>
<summary>Supporting services (Postgres, Redis, Prometheus, Grafana)</summary>

Run only infrastructure services without the app.

Postgres + Redis (base compose)
```bash
docker compose -f Dockerfiles/docker-compose.yml up -d postgres redis
```

Prometheus + Grafana (embeddings compose, monitoring profile)
```bash
docker compose -f Dockerfiles/docker-compose.embeddings.yml --profile monitoring up -d prometheus grafana
```

All four together
```bash
docker compose -f Dockerfiles/docker-compose.yml up -d postgres redis
docker compose -f Dockerfiles/docker-compose.embeddings.yml --profile monitoring up -d prometheus grafana
```

Manage and verify
```bash
# Status
docker compose -f Dockerfiles/docker-compose.yml ps
docker compose -f Dockerfiles/docker-compose.embeddings.yml ps

# Logs
docker compose -f Dockerfiles/docker-compose.yml logs -f postgres redis
docker compose -f Dockerfiles/docker-compose.embeddings.yml logs -f prometheus grafana

# Stop
docker compose -f Dockerfiles/docker-compose.yml stop postgres redis
docker compose -f Dockerfiles/docker-compose.embeddings.yml stop prometheus grafana

# Remove
docker compose -f Dockerfiles/docker-compose.yml down
docker compose -f Dockerfiles/docker-compose.embeddings.yml down
```

Ports
- Postgres: 5432
- Redis: 6379
- Prometheus: 9091 (container listens on 9090)
- Grafana: 3000

Prometheus config
- Create `Config_Files/prometheus.yml` to define scrape targets. Minimal self-scrape example:
```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
```
See Docs/Operations/monitoring/README.md for examples that scrape the API and worker orchestrator.

Tip: See multi-user setup and production hardening in Docs/User_Guides/Server/Authentication_Setup.md and Docs/Published/Deployment/First_Time_Production_Setup.md.

</details>

## Usage Examples

<details>
<summary>Usage Examples</summary>

Use the single-user API key with the `X-API-KEY` header.

Chat (OpenAI-compatible)
```bash
curl -s http://127.0.0.1:8000/api/v1/chat/completions \
  -H "X-API-KEY: $SINGLE_USER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'
```

Embeddings (OpenAI-compatible)
```bash
curl -s http://127.0.0.1:8000/api/v1/embeddings \
  -H "X-API-KEY: $SINGLE_USER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "text-embedding-3-small",
    "input": "hello world"
  }'
```

Media Ingest (URL)
```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/media/add \
  -H "X-API-KEY: $SINGLE_USER_API_KEY" \
  -F "media_type=document" \
  -F "title=Example Article" \
  -F "keywords=demo,quickstart" \
  -F "urls=https://www.example.com/some-article"
```

Media Search
```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/media/search \
  -H "X-API-KEY: $SINGLE_USER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "keyword"
  }'
```

Audio Transcription (file)
```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/audio/transcriptions \
  -H "X-API-KEY: $SINGLE_USER_API_KEY" \
  -F "file=@sample.wav" -F "model=whisper-1"
```
</details>

## Key Endpoints
<details>
<summary>Key Endpoints</summary>

- Media: `POST /api/v1/media/add` - ingest/process media (URLs/files) with DB persistence ([docs](Docs/Code_Documentation/Ingestion_Media_Processing.md))
- Media Search: `POST /api/v1/media/search` - search ingested content ([docs](Docs/API-related/API_Design.md))
- Chat: `POST /api/v1/chat/completions` - OpenAI-compatible chat ([docs](Docs/API-related/Chat_API_Documentation.md))
- Chat Commands: `GET /api/v1/chat/commands` - list available slash commands ([docs](Docs/API-related/Chatbook_Features_API_Documentation.md#chat-tools-slash-commands))
- Chat Dictionary Validate: `POST /api/v1/chat/dictionaries/validate` - validate a chat dictionary ([docs](Docs/API-related/Chatbook_Features_API_Documentation.md#chat-dictionary-api))
- Embeddings: `POST /api/v1/embeddings` - OpenAI-compatible embeddings ([docs](Docs/API-related/Embeddings_API_Documentation.md))
- RAG: `POST /api/v1/rag/search` - unified RAG search ([docs](Docs/API-related/RAG-API-Guide.md))
- Audio STT: `POST /api/v1/audio/transcriptions` - file-based transcription ([docs](Docs/API-related/Audio_Transcription_API.md))
- Audio STT (WS): `WS /api/v1/audio/stream/transcribe` - real-time transcription ([docs](Docs/API-related/Audio_Transcription_API.md))
- Audio TTS: `POST /api/v1/audio/speech` - text-to-speech (streaming and non-streaming) ([docs](Docs/API-related/TTS_API.md))
- TTS Voices: `GET /api/v1/audio/voices/catalog` - voice catalog across providers ([docs](Docs/API-related/TTS_API.md))
- Voice Assistant: `POST /api/v1/voice/command` (REST) and `WS /api/v1/voice/assistant` (real-time) ([docs](Docs/API/Voice_Assistant.md))
- Audiobooks Parse: `POST /api/v1/audiobooks/parse` - parse + chapter preview ([docs](Docs/Product/Completed/Audiobook_Creation_PRD.md))
- Audiobooks Jobs: `POST /api/v1/audiobooks/jobs` - create audiobook job; `GET /api/v1/audiobooks/jobs/{job_id}` - status ([docs](Docs/Product/Completed/Audiobook_Creation_PRD.md))
- Audiobooks Subtitles: `POST /api/v1/audiobooks/subtitles` - render subtitle export ([docs](Docs/Product/Completed/Audiobook_Creation_PRD.md))
- Vector Stores: `POST /api/v1/vector_stores` - create; `POST /api/v1/vector_stores/{id}/query` - query ([docs](Docs/API-related/Vector_Stores_Admin_and_Query.md))
- OCR Backends: `GET /api/v1/ocr/backends` - available OCR providers ([docs](Docs/API-related/OCR_API_Documentation.md))
- VLM Backends: `GET /api/v1/vlm/backends` - available VLM providers ([docs](Docs/Code_Documentation/VLM_Backends.md))
- Connectors: `GET /api/v1/connectors/providers` - Drive/Notion providers ([docs](Docs/Product/External_Connectors_PRD.md))
- Outputs: `POST /api/v1/outputs` - generate output artifact (md/html/mp3) ([docs](Docs/Product/Completed/Content_Collections_PRD.md))
- Tasks: `POST /api/v1/tasks` + `GET /api/v1/tasks` - reminder task create/list ([docs](Docs/API-related/Reminder_Notifications_API.md))
- Notifications: `GET /api/v1/notifications` + `GET /api/v1/notifications/stream` - inbox list + realtime SSE ([docs](Docs/API-related/Reminder_Notifications_API.md))
- Metrics: `GET /api/v1/metrics/text` - Prometheus metrics (text format) ([docs](Docs/Deployment/Monitoring/Metrics_Cheatsheet.md))
- Providers: `GET /api/v1/llm/providers` - provider/models list ([docs](Docs/API-related/Providers_API_Documentation.md))
- MCP: `GET /api/v1/mcp/status` - MCP server status ([docs](Docs/MCP/Unified/System_Admin_Guide.md))
- MCP Hub Management: `GET /api/v1/mcp/hub/acp-profiles`, `GET /api/v1/mcp/hub/external-servers`, plus mutation endpoints for authorized principals ([docs](Docs/MCP/mcp_hub_management.md))

Admin maintenance
- Chat model aliases cache reload: `POST /api/v1/admin/chat/model-aliases/reload`
  - Single-user (API key)
    ```bash
    curl -s -X POST http://127.0.0.1:8000/api/v1/admin/chat/model-aliases/reload \
      -H "X-API-KEY: $SINGLE_USER_API_KEY"
    ```
  - Multi-user (JWT)
    ```bash
    curl -s -X POST http://127.0.0.1:8000/api/v1/admin/chat/model-aliases/reload \
      -H "Authorization: Bearer $JWT"
    ```

Examples
- GET `/api/v1/chat/commands` response
  ```json
  {
    "commands": [
      {"name": "time", "description": "Show the current time (optional TZ).", "required_permission": "chat.commands.time"},
      {"name": "weather", "description": "Show current weather for a location.", "required_permission": "chat.commands.weather"}
    ]
  }
  ```
- POST `/api/v1/chat/dictionaries/validate` request
  ```json
  {
    "data": {
      "name": "Example",
      "entries": [
        {"type": "literal", "pattern": "today", "replacement": "It is {{ now('%B %d') }}."},
        {"type": "regex", "pattern": "User:(\\w+)", "replacement": "Hello, {{ match.group(1) }}!"}
      ]
    },
    "schema_version": 1,
    "strict": false
  }
  ```
  Minimal success response
  ```json
  {
    "ok": true,
    "schema_version": 1,
    "errors": [],
    "warnings": [],
    "entry_stats": {"total": 2, "regex": 1, "literal": 1},
    "suggested_fixes": []
  }
  ```

</details>

## Architecture & Repo Layout

<details>
<summary>Architecture & Repo Layout</summary>

```text
<repo_root>/
├── tldw_Server_API/              # Main API server implementation
│   ├── app/
│   │   ├── api/v1/
│   │   │   ├── endpoints/        # REST endpoints (media, chat, audio, rag, evals, etc.)
│   │   │   ├── schemas/          # Pydantic models
│   │   │   └── API_Deps/         # Shared dependencies (auth, DB, rate limits)
│   │   ├── core/                 # Core logic (AuthNZ, RAG, LLM, DB, TTS, MCP, etc.)
│   │   ├── services/             # Background services
│   │   ├── Setup_UI/             # Setup UI assets
│   │   ├── static/               # Static assets
│   │   └── main.py               # FastAPI entry point
│   ├── cli/                      # CLI entry points
│   ├── Config_Files/             # config.txt, example YAMLs, migration helpers
│   ├── Databases/                # Default DBs (runtime data; some are gitignored)
│   └── tests/                    # Pytest suite
├── admin-ui/                     # Admin dashboard
├── apps/tldw-frontend/                # Next.js WebUI (primary web client)
├── Docs/                         # Documentation (API, Development, RAG, AuthNZ, TTS, etc.)
├── Helper_Scripts/               # Utilities (installers, prompt tools, doc generators)
├── mock_openai_server/           # Mock OpenAI-compatible API server for tests/dev
├── Dockerfiles/                  # Docker images and compose files
│   └── Monitoring/               # Monitoring stack assets
├── Databases/                    # DBs (AuthNZ defaults here; content DBs per-user under user_databases/)
├── models/                       # Optional model assets (if used)
├── pyproject.toml                # Project configuration
├── README.md                     # Project README (this file)
├── start-sidecars.sh             # Run API + Jobs workers as sidecar processes
└── Project_Guidelines.md         # Development philosophy
```

Notes
- The primary UI is the Next.js WebUI in `apps/tldw-frontend/` (run separately).
- SQLite is default for local dev; PostgreSQL supported for AuthNZ and content DBs.
- `mock_openai_server/` is handy for local OpenAI-compatible API testing.

</details>

## Architecture Diagram

```mermaid
flowchart LR
  subgraph CLIENTS [Clients]
    WebUI[Next.js WebUI (primary web client)]:::client
    MCPClients[MCP Clients (IDE/tools)]:::client
    APIClients[CLI/HTTP Clients]:::client
  end

  subgraph API_STACK [FastAPI App]
    API[FastAPI App /api/v1]:::api
    Endpoints[Endpoints + Schemas]:::module
    Dependencies[API Deps (Auth, DB, rate limits, resource governor)]:::module
    Services[Background Services/Jobs]:::module
  end

  subgraph CORE [Core Modules]
    AuthNZ[AuthNZ]:::core
    RAG[RAG]:::core
    LLM[LLM Calls]:::core
    Embeddings[Embeddings]:::core
    Media[Ingestion & Media Processing]:::core
    Chunking[Chunking]:::core
    Chat[Chat/Characters]:::core
    Audio[Audio STT/TTS]:::core
    Evaluations[Evaluations]:::core
    PromptStudio[Prompt Studio]:::core
    Knowledge[Notes/Prompts/Chatbooks]:::core
    MCP[MCP Unified]:::core
    Research[Research/Web Search]:::core
  end

  subgraph STORAGE [Storage]
    UsersDB[(AuthNZ DB: SQLite/PostgreSQL)]:::db
    ContentDB[(Content DBs: Media/Notes/Chats)]:::db
    EvalsDB[(Evaluations DB: SQLite/PostgreSQL)]:::db
    VectorDB[(Vector DB: ChromaDB/pgvector)]:::db
  end

  subgraph EXTERNAL [External Providers]
    LLMCloud[LLM APIs (OpenAI, Anthropic, etc.)]:::ext
    LLMOnPrem[Local LLMs (vLLM, Ollama, llama.cpp, ...)]:::ext
    AudioProv[STT/TTS Providers]:::ext
    OCRVLM[OCR/VLM (tesseract, dots, points)]:::ext
    MediaDL[yt-dlp / ffmpeg]:::ext
    WebSearch[Web Search/Scrapers]:::ext
  end

  %% Client to API
  WebUI -->|HTTP| API
  MCPClients -->|HTTP/WebSocket| API
  APIClients -->|HTTP/WebSocket| API

  %% Inside API stack
  API --> Endpoints
  API --> Dependencies
  API --> Services

  %% Endpoints to core modules
  Endpoints --> AuthNZ
  Endpoints --> RAG
  Endpoints --> LLM
  Endpoints --> Embeddings
  Endpoints --> Media
  Endpoints --> Chunking
  Endpoints --> Chat
  Endpoints --> Audio
  Endpoints --> Evaluations
  Endpoints --> PromptStudio
  Endpoints --> Knowledge
  Endpoints --> MCP
  Endpoints --> Research

  %% Core to storage
  AuthNZ --> UsersDB
  Media --> ContentDB
  Knowledge --> ContentDB
  Chat --> ContentDB
  Evaluations --> EvalsDB
  RAG --> ContentDB
  RAG --> VectorDB
  Embeddings --> VectorDB

  %% Core to external services
  LLM --> LLMCloud
  LLM --> LLMOnPrem
  Audio --> AudioProv
  Media --> MediaDL
  Media --> OCRVLM
  Research --> WebSearch

  classDef client fill:#e8f3ff,stroke:#5b8def,color:#1f3b6e;
  classDef api fill:#fff4e6,stroke:#ff9800,color:#5d3d00;
  classDef module fill:#f4f6f8,stroke:#9aa5b1,color:#2d3748;
  classDef core fill:#eefbea,stroke:#34a853,color:#1e4620;
  classDef db fill:#f0eaff,stroke:#8e6cf1,color:#3a2a87;
  classDef ext fill:#fff0f0,stroke:#e57373,color:#7b1f1f;
```

## Networking & Limits

- HTTP client and TLS/pinning configuration: `tldw_Server_API/Config_Files/README.md` (timeouts, retries, redirects/proxies, JSON limits, TLS min version, cert pinning, SSE/download helpers).
- Egress/SSRF policy and security middleware: `tldw_Server_API/app/core/Security/README.md`.
- Resource Governor (rate limits, tokens, streams; Redis backend optional): `tldw_Server_API/app/core/Resource_Governance/README.md`.

### API Rate Limits (High-Level)

- **Characters & Character Chat**
  - Per-user caps (configurable via env/settings; defaults documented in `Docs/API-related/CHARACTER_CHAT_API_DOCUMENTATION.md`):
    - Operations per hour (character operations).
    - Per-user character limit.
    - Concurrent chats per user.
    - Messages per chat.
    - Per-minute limits for chat completions and message sends.
  - Status endpoint:
    - `GET /api/v1/characters/rate-limit-status` → returns a simple snapshot:
      - `operations_used`, `operations_remaining`, `reset_time` (Unix timestamp or `null`).
  - Enforcement is handled by `CharacterRateLimiter` with Redis-backed ZSETs when `REDIS_ENABLED=true`, or per-process in-memory counters otherwise.

- **Core Chat / RAG / Embeddings**
  - Per-user RPM/TPM limits enforced via the Resource Governor (optional Redis backend).
  - See:
    - `tldw_Server_API/app/core/Chat/rate_limiter.py`
    - `tldw_Server_API/app/core/Resource_Governance/README.md`

All limits are designed to be conservative by default and can be tuned using the various `*_RATE_LIMIT_*`, `MAX_*`, and RG policy settings in `Config_Files/` and environment variables.

## Running Tests

<details>
<summary>Running tests locally</summary>

- `python -m pytest -v` - full test suite (skips heavy optional suites by default).
- `python -m pytest --cov=tldw_Server_API --cov-report=term-missing` - coverage report.
- Use markers (`unit`, `integration`, `e2e`, `external_api`, `performance`) to focus specific areas.
- Enable optional suites with environment flags such as `RUN_MCP_TESTS=1`, `TLDW_TEST_POSTGRES_REQUIRED=1`, or `RUN_MOCK_OPENAI=1`.
- `make tooling-smoke` - runs unified tooling smoke checks (`streaming_unified_smoke.py` + `watchlists_audio_smoke.py`) against a running local API (`http://127.0.0.1:8000` by default).

</details>

## Frontend Integration Testing

Use the helper script to run frontend unit tests plus smoke checks against a live backend:

```bash
cd apps/tldw-frontend
npm run test:integration
```

Notes:
- Starts the backend (uvicorn) by default and runs `pytest -m integration`, then `npm run test:run` + `npm run smoke`.
- Set `TLDW_X_API_KEY=...` for single-user mode (a temporary key is generated if missing).
- Use `--backend-docker` to start the backend via Docker Compose, or `--skip-backend` if you already have it running.
- Use `--no-backend-tests` to skip backend integration tests.

## CI Status & Smoke Tests

<details>
<summary>CI status and smoke tests</summary>

| Workflow | Status |
| --- | --- |
| E2E Critical Smoke (In-Process) | [![E2E Critical Smoke](https://github.com/rmusser01/tldw_server/actions/workflows/e2e-smoke.yml/badge.svg)](https://github.com/rmusser01/tldw_server/actions/workflows/e2e-smoke.yml) |

Required gate policy and check-name contract:
- `Docs/Development/CI_REQUIRED_GATES.md`

Run locally

- In-process (no open port):
  - `export E2E_INPROCESS=1 AUTH_MODE=single_user TEST_MODE=1`
  - `export SINGLE_USER_API_KEY=test-api-key-for-e2e-testing-12345`
  - `export SINGLE_USER_TEST_API_KEY=$SINGLE_USER_API_KEY`
  - `python -m pytest tldw_Server_API/tests/e2e/ --critical-only -q`
- Live server (normal):
  - `python -m uvicorn tldw_Server_API.app.main:app --reload`
  - `export E2E_TEST_BASE_URL=http://localhost:8000`
  - `python -m pytest tldw_Server_API/tests/e2e/ --critical-only -q`

</details>

## Documentation & Resources

<details>
<summary>Documentation and resources</summary>

**Getting Started Guides:**
- [Getting Started Index](Docs/Getting_Started/README.md) - choose the right setup path
- [Local Single-User Profile](Docs/Getting_Started/Profile_Local_Single_User.md) - local API development path
- [Docker Single-User Profile](Docs/Getting_Started/Profile_Docker_Single_User.md) - self-host with Docker
- [Docker Multi-User + Postgres Profile](Docs/Getting_Started/Profile_Docker_Multi_User_Postgres.md) - team deployment baseline
- [First-time audio setup: CPU systems](Docs/Getting_Started/First_Time_Audio_Setup_CPU.md) - local-first STT and TTS for CPU boxes
- [First-time audio setup: GPU/accelerated systems](Docs/Getting_Started/First_Time_Audio_Setup_GPU_Accelerated.md) - NVIDIA and Apple Silicon speech setup
- [GPU/STT Add-on](Docs/Getting_Started/GPU_STT_Addon.md) - legacy pointer to the accelerated guide

**Reference:**
- `Docs/Documentation.md` - documentation index and developer guide links
- `Docs/About.md` - project background and philosophy
- Module deep dives: `Docs/Code_Documentation/AuthNZ-Developer-Guide.md`, `Docs/Code_Documentation/RAG-Developer-Guide.md`, `Docs/MCP/Unified/Developer_Guide.md`
- Packaging and releases: `Docs/Development/PyPI_Publishing.md`
- Distribution strategy (API + WebUI): `Docs/Development/Packaging_and_Distribution_Strategy.md`
- API references: `Docs/API-related/RAG-API-Guide.md`, `Docs/API-related/OCR_API_Documentation.md`, `Docs/API-related/Prompt_Studio_API.md`
- Deployment/Monitoring: `Docs/Published/Deployment/First_Time_Production_Setup.md`, `Docs/Published/Deployment/Reverse_Proxy_Examples.md`, `Docs/Deployment/Monitoring/`
- Speech quickstart (STT + TTS): `Docs/User_Guides/WebUI_Extension/Getting-Started-STT_and_TTS.md`
- TTS provider onboarding: `Docs/User_Guides/WebUI_Extension/TTS_Getting_Started.md`
- TTS deep runbooks index: `Docs/User_Guides/WebUI_Extension/TTS-SETUP-GUIDE.md`
- Design notes (WIP features): `Docs/Design/` - e.g., `Docs/Design/Audio_Pipeline.md`

### Resource Governor Config

For complete Resource Governor setup and examples (env, DB store bootstrap, YAML policy, middleware, diagnostics, and tests), see `tldw_Server_API/app/core/Resource_Governance/README.md`.

### OpenAI-Compatible Strict Mode (Local Providers)

Some self-hosted OpenAI-compatible servers reject unknown fields (like `top_k`). For local providers you can enable a strict mode that filters non-standard keys from chat payloads.

- Set `strict_openai_compat: true` in the relevant provider section (`local_llm`, `llama_api`, `ooba_api`, `tabby_api`, `vllm_api`, `aphrodite_api`, `ollama_api`).
- For `local_llm`, you can also use `LOCAL_LLM_STRICT_OPENAI_COMPAT=1`.
- When enabled, only standard OpenAI Chat Completions parameters are sent:
  `messages, model, temperature, top_p, max_tokens, n, stop, presence_penalty, frequency_penalty, logit_bias, seed, response_format, tools, tool_choice, logprobs, top_logprobs, user, stream`.

### Chatbook Tools Guide

- Getting started: `Docs/User_Guides/WebUI_Extension/Chatbook_Tools_Getting_Started.md`
- Product spec (PRD): `Docs/Product/Completed/Chatbook-Tools-PRD.md`
- Related endpoints (also listed above under Key Endpoints):
  - `GET /api/v1/chat/commands` — list slash commands (RBAC-filtered when enabled; returns empty list when disabled)
  - `POST /api/v1/chat/dictionaries/validate` — validate chat dictionaries (schema, regex, templates)

</details>

## Deployment

<details>
<summary>Deployment resources</summary>
- Dockerfiles and compose templates live under `Dockerfiles/` (see `Dockerfiles/README.md`).
- Reverse proxy samples: `Helper_Scripts/Samples/Nginx/`, `Helper_Scripts/Samples/Caddy/`.
- Monitoring: `Docs/Deployment/Monitoring/` and `Helper_Scripts/Samples/Grafana/`.
- Prometheus metrics exposed at `/metrics` and `/api/v1/metrics`.
- Production hardening: `Docs/Published/User_Guides/Server/Production_Hardening_Checklist.md`.
</details>

## Monitoring

<details>
<summary>Monitoring resources</summary>

- Monitoring docs and setup: `Docs/Deployment/Monitoring/README.md`
- Grafana dashboards and samples: `Helper_Scripts/Samples/Grafana/README.md`
- Prometheus scrape endpoints: `GET /metrics` and `GET /api/v1/mcp/metrics/prometheus` (both require authenticated principals with appropriate permissions; MCP Prometheus uses `system.logs`)

### PostgreSQL Content Mode

- Content DBs (Media, ChaChaNotes, Workflows) can run on Postgres.
- See: `Docs/Published/User_Guides/Server/Multi-User_Postgres_Setup.md`, `Docs/Published/Deployment/Postgres_Migration_Guide.md`, and `Docs/Published/User_Guides/Server/Backups_Using_Litestream.md`.

</details>

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `Connection refused` | Server not running - check `uvicorn` or `docker compose ps` |
| `401 Unauthorized` | API key mismatch - verify `.env` matches your `X-API-KEY` header |
| `API key too short` | Key must be at least 16 characters |
| `ffmpeg not found` | Install: `brew install ffmpeg` (macOS) or `apt install ffmpeg` (Linux) |
| `ModuleNotFoundError` | Activate venv: `source .venv/bin/activate` |
| Torch/CUDA mismatch | Install CUDA-compatible PyTorch or use CPU wheels |
| SQLite locks | Use short-lived transactions; consider Postgres for concurrency |
| Docker issues | Check `docker compose ps` and `docker compose logs -f` |

Quick verify: `curl http://localhost:8000/health` (no auth needed)

## Contributing & Support

- Read `CONTRIBUTING.md`, `Project_Guidelines.md`, and `AGENTS.md` before submitting changes.
- File bugs or feature requests via GitHub Issues; longer-form discussions live in GitHub Discussions.
- Respect the project philosophy: incremental progress, clear intent, and kindness toward contributors.

---

## Developer Guides

<details>
<summary>Developer Guides + Links</summary>

- Documentation index: `Docs/Documentation.md` (see the "Developer Guides" section)
- Core module guides:
  - Chat Module: `Docs/Code_Documentation/Chat_Developer_Guide.md`
  - Chunking Module: `Docs/Code_Documentation/Chunking-Module.md`
  - RAG Module: `tldw_Server_API/app/core/RAG/README.md` and `tldw_Server_API/app/core/RAG/API_DOCUMENTATION.md`

### Ingestion & Media Processing Docs
- Overview: `Docs/Code_Documentation/Ingestion_Media_Processing.md`
- Pipelines:
  - Audio: `Docs/Code_Documentation/Ingestion_Pipeline_Audio.md`
  - Video: `Docs/Code_Documentation/Ingestion_Pipeline_Video.md`
  - PDF: `Docs/Code_Documentation/Ingestion_Pipeline_PDF.md`
  - EPUB: `Docs/Code_Documentation/Ingestion_Pipeline_Ebooks.md`
  - Documents: `Docs/Code_Documentation/Ingestion_Pipeline_Documents.md`
  - MediaWiki: `Docs/Code_Documentation/Ingestion_Pipeline_MediaWiki.md`

</details>

## About

tldw_server started as a tool to transcribe and summarize YouTube videos and has evolved into a comprehensive media analysis and knowledge management platform for researchers, students, and professionals.
For extended background and vision, see `Docs/About.md`.

### Getting Help
- API Documentation: `http://localhost:8000/docs`
- GitHub Issues: [Report bugs or request features](https://github.com/rmusser01/tldw_server/issues)
- Discussions: [Community forum](https://github.com/rmusser01/tldw_server/discussions)

### Security Disclosures
See `SECURITY.md` for reporting guidelines and disclosures.

### Project Guidelines
See [Project_Guidelines.md](Project_Guidelines.md) for development philosophy and contribution guidelines.

## Appendix (Optional)

<details>
<summary>Background and reading list</summary>

Optional background reading for deeper context; not required to use the project.

- See `Docs/About.md` for the extended project background, vision, and notes.
- https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5049562
- Purpose of this section is to help bring awareness to certain concepts and terms that are used in the field of AI/ML/NLP, as well as to provide some resources for learning more about them.
- Also because some of those things are extremely relevant and important to know if you care about accuracy and the effectiveness of the LLMs you're using.
- Some of this stuff may be 101 level, but I'm going to include it anyways. This repo is aimed at people from a lot of different fields, so I want to make sure everyone can understand what's going on. Or at least has an idea.
- LLMs 101(coming from a tech background): https://vinija.ai/models/LLM/
- LLM Fundamentals / LLM Scientist / LLM Engineer courses(Free): https://github.com/mlabonne/llm-course
- **Phrases & Terms**
  - **LLM** - Large Language Model - A type of neural network that can generate human-like text.
  - **API** - Application Programming Interface - A set of rules and protocols that allows one software application to communicate with another.
  - **API Wrapper** - A set of functions that provide a simplified interface to a larger body of code.
  - **API Key** - A unique identifier that is used to authenticate a user, developer, or calling program to an API.
  - **GUI** - Graphical User Interface
  - **CLI** - Command Line Interface
  - **DB** - Database
  - **SQLite** - A C-language library that implements a small, fast, self-contained, high-reliability, full-featured, SQL database engine.
  - **Prompt Engineering** - The process of designing prompts that are used to guide the output of a language model.
  - **Quantization** - The process of converting a continuous range of values into a finite range of discrete values.
  - **GGUF Files** - GGUF is a binary format that is designed for fast loading and saving of models, and for ease of reading. Models are traditionally developed using PyTorch or another framework, and then converted to GGUF for use in GGML. https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
  - **Inference Engine** - A software system that is designed to execute a model that has been trained by a machine learning algorithm. Llama.cpp and Kobold.cpp are examples of inference engines.
- **Papers & Concepts**
  1. Lost in the Middle: How Language Models Use Long Contexts(2023)
    - https://arxiv.org/abs/2307.03172
    - `We analyze the performance of language models on two tasks that require identifying relevant information in their input contexts: multi-document question answering and key-value retrieval. We find that performance can degrade significantly when changing the position of relevant information, indicating that current language models do not robustly make use of information in long input contexts. In particular, we observe that performance is often highest when relevant information occurs at the beginning or end of the input context, and significantly degrades when models must access relevant information in the middle of long contexts, even for explicitly long-context models`
  2. [Same Task, More Tokens: the Impact of Input Length on the Reasoning Performance of Large Language Models(2024)](https://arxiv.org/abs/2402.14848)
     - `Our findings show a notable degradation in LLMs' reasoning performance at much shorter input lengths than their technical maximum. We show that the degradation trend appears in every version of our dataset, although at different intensities. Additionally, our study reveals that the traditional metric of next word prediction correlates negatively with performance of LLMs' on our reasoning dataset. We analyse our results and identify failure modes that can serve as useful guides for future research, potentially informing strategies to address the limitations observed in LLMs.`
  3. Why Does the Effective Context Length of LLMs Fall Short?(2024)
    - https://arxiv.org/abs/2410.18745
    - `     Advancements in distributed training and efficient attention mechanisms have significantly expanded the context window sizes of large language models (LLMs). However, recent work reveals that the effective context lengths of open-source LLMs often fall short, typically not exceeding half of their training lengths. In this work, we attribute this limitation to the left-skewed frequency distribution of relative positions formed in LLMs pretraining and post-training stages, which impedes their ability to effectively gather distant information. To address this challenge, we introduce ShifTed Rotray position embeddING (STRING). STRING shifts well-trained positions to overwrite the original ineffective positions during inference, enhancing performance within their existing training lengths. Experimental results show that without additional training, STRING dramatically improves the performance of the latest large-scale models, such as Llama3.1 70B and Qwen2 72B, by over 10 points on popular long-context benchmarks RULER and InfiniteBench, establishing new state-of-the-art results for open-source LLMs. Compared to commercial models, Llama 3.1 70B with \method even achieves better performance than GPT-4-128K and clearly surpasses Claude 2 and Kimi-chat.`
  4. [RULER: What's the Real Context Size of Your Long-Context Language Models?(2024)](https://arxiv.org/abs/2404.06654)
    - `The needle-in-a-haystack (NIAH) test, which examines the ability to retrieve a piece of information (the "needle") from long distractor texts (the "haystack"), has been widely adopted to evaluate long-context language models (LMs). However, this simple retrieval-based test is indicative of only a superficial form of long-context understanding. To provide a more comprehensive evaluation of long-context LMs, we create a new synthetic benchmark RULER with flexible configurations for customized sequence length and task complexity. RULER expands upon the vanilla NIAH test to encompass variations with diverse types and quantities of needles. Moreover, RULER introduces new task categories multi-hop tracing and aggregation to test behaviors beyond searching from context. We evaluate ten long-context LMs with 13 representative tasks in RULER. Despite achieving nearly perfect accuracy in the vanilla NIAH test, all models exhibit large performance drops as the context length increases. While these models all claim context sizes of 32K tokens or greater, only four models (GPT-4, Command-R, Yi-34B, and Mixtral) can maintain satisfactory performance at the length of 32K. Our analysis of Yi-34B, which supports context length of 200K, reveals large room for improvement as we increase input length and task complexity.`
  5. Abliteration (Uncensoring LLMs)
     - [Uncensor any LLM with abliteration - Maxime Labonne(2024)](https://huggingface.co/blog/mlabonne/abliteration)
  6. Retrieval-Augmented-Generation
        - [Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997)
          - https://arxiv.org/abs/2312.10997
          - `Retrieval-Augmented Generation (RAG) has emerged as a promising solution by incorporating knowledge from external databases. This enhances the accuracy and credibility of the generation, particularly for knowledge-intensive tasks, and allows for continuous knowledge updates and integration of domain-specific information. RAG synergistically merges LLMs' intrinsic knowledge with the vast, dynamic repositories of external databases. This comprehensive review paper offers a detailed examination of the progression of RAG paradigms, encompassing the Naive RAG, the Advanced RAG, and the Modular RAG. It meticulously scrutinizes the tripartite foundation of RAG frameworks, which includes the retrieval, the generation and the augmentation techniques. The paper highlights the state-of-the-art technologies embedded in each of these critical components, providing a profound understanding of the advancements in RAG systems. Furthermore, this paper introduces up-to-date evaluation framework and benchmark. At the end, this article delineates the challenges currently faced and points out prospective avenues for research and development. `
  7. Prompt Engineering
     - Prompt Engineering Guide: https://www.promptingguide.ai/ & https://github.com/dair-ai/Prompt-Engineering-Guide
     - 'The Prompt Report' - https://arxiv.org/abs/2406.06608
  8. Bias and Fairness in LLMs
     - [ChatGPT Doesn't Trust Chargers Fans: Guardrail Sensitivity in Context](https://arxiv.org/abs/2407.06866)
       - `While the biases of language models in production are extensively documented, the biases of their guardrails have been neglected. This paper studies how contextual information about the user influences the likelihood of an LLM to refuse to execute a request. By generating user biographies that offer ideological and demographic information, we find a number of biases in guardrail sensitivity on GPT-3.5. Younger, female, and Asian-American personas are more likely to trigger a refusal guardrail when requesting censored or illegal information. Guardrails are also sycophantic, refusing to comply with requests for a political position the user is likely to disagree with. We find that certain identity groups and seemingly innocuous information, e.g., sports fandom, can elicit changes in guardrail sensitivity similar to direct statements of political ideology. For each demographic category and even for American football team fandom, we find that ChatGPT appears to infer a likely political ideology and modify guardrail behavior accordingly.`
- **Tools & Libraries**
  1. `llama.cpp` - A C++ inference engine. Highly recommend.
     * https://github.com/ggerganov/llama.cpp
  2. `kobold.cpp` - A C++ inference engine. GUI wrapper of llama.cpp with some tweaks.
     * https://github.com/LostRuins/koboldcpp
  3. `sillytavern` - A web-based interface for text generation models. Supports inference engines. Ignore the cat girls and weebness. This software is _powerful_ and _useful_. Also supports just about every API you could want.
     * https://github.com/SillyTavern/SillyTavern
  4. `llamafile` - A wrapper for llama.cpp that allows for easy use of local LLMs.
     * Uses libcosomopolitan for cross-platform compatibility.
     * Can be used to run LLMs on Windows, Linux, and MacOS with a single binary wrapper around Llama.cpp.
  5. `pytorch` - An open-source machine learning library based on the Torch library.
  6. `ffmpeg` - A free software project consisting of a large suite of libraries and programs for handling video, audio, and other multimedia files and streams.
  7. `pandoc` - A free and open-source document converter, widely used as a writing tool (especially by scholars) and as a basis for publishing workflows.
     * https://pandoc.org/
  8. `marker` - A tool for converting PDFs(and other document types) to markdown.
     * https://github.com/VikParuchuri/marker
  9. `faster_whisper` - A fast, lightweight, and accurate speech-to-text model.
      * https://github.com/SYSTRAN/faster-whisper

</details>

<details>
<summary>Local model recommendations (opinionated)</summary>

Personal recommendations; optional reading.

- These are just the 'standard smaller' models I recommend, there are many more out there, and you can use any of them with this project.
  - One should also be aware that people create 'fine-tunes' and 'merges' of existing models, to create new models that are more suited to their needs.
  - This can result in models that may be better at some tasks but worse at others, so it's important to test and see what works best for you.
- Mistral Nemo Instruct 2407 - https://huggingface.co/QuantFactory/Mistral-Nemo-Instruct-2407-GGUF
- Magistral Small: https://huggingface.co/mistralai/Magistral-Small-2509-GGUF
- Qwen 3/VL Series
  - Qwen/Qwen3-VL-4B-Instruct (Qwen3 4B+Vision): https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF
  - Qwen3-30B-A3B-Instruct-2507: https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF

For commercial API usage for use with this project: GPT-5.1, Anthropic's models(The Temu of AI services/models), Kimi k2, DeepSeek
Originally written: Flipside I would say none, honestly. The (largest players) will gaslight you and charge you money for it. Fun.  That being said they obviously can provide help/be useful(helped me make this app), but it's important to remember that they're not your friend, and they're not there to help you. They are there to make money not off you, but off large institutions and your data.  You are just a stepping stone to their goals.
2025 Nov: I would say service quality has improved enough to the point where it can make sense to use a 'premium' subscription/usage of API services without expecting to be screwed 7-8/10 times.
2025 Dec: I spoke too soon. I have the opinion of GPT-5.1 High > Sonnet/Opus. The Anthropic models are just too 'independent/lazy'. I personally have never had gpt5 fill out a PRD with completely hallucinated bullshit, or gaslight me repeatedly across multiple different sessions. Sonnet/Opus 4.0-4.5? Complete opposite.


From @nrose 05/08/2024 on Threads:
```
No, it’s a design. First they train it, then they optimize it. Optimize it for what- better answers?
  No. For efficiency.
Per watt. Because they need all the compute they can get to train the next model.So it’s a sawtooth.
The model declines over time, then the optimization makes it somewhat better, then in a sort of
  reverse asymptote, they dedicate all their “good compute” to the next bigger model.Which they then
  trim down over time, so they can train the next big model… etc etc.
None of these companies exist to provide AI services in 2024. They’re only doing it to finance the
 things they want to build in 2025 and 2026 and so on, and the goal is to obsolete computing in general
  and become a hidden monopoly like the oil and electric companies.
2024 service quality is not a metric they want to optimize, they’re forced to, only to maintain some
  directional income
```

</details>

<details>
<summary>Quotes</summary>

- `I like the lies-to-children motif, because it underlies the way we run our society and resonates nicely with Discworld. Like the reason for Unseen being a storehouse of knowledge - you arrive knowing everything and leave realising that you know practically nothing, therefore all the knowledge you had must be stored in the university. But it's like that in "real Science", too. You arrive with your sparkling A-levels all agleam, and the first job of the tutors is to reveal that what you thought was true is only true for a given value of "truth". Most of us need just "enough" knowledge of the sciences, and it's delivered to us in metaphors and analogies that bite us in the bum if we think they're the same as the truth.`
    * Terry Pratchett
- `The first principle is that you must not fool yourself - and you are the easiest person to fool.`
    * Richard Feynman

</details>

---

## License

GNU General Public License v3.0 - see `LICENSE` for details.

---

## Credits

- [The original version of tldw by @the-crypt-keeper](https://github.com/the-crypt-keeper/tldw/tree/main/tldw-original-scripts)
- [yt-dlp](https://github.com/yt-dlp/yt-dlp)
- [ffmpeg](https://github.com/FFmpeg/FFmpeg)
- [faster_whisper](https://github.com/SYSTRAN/faster-whisper)
- [pyannote](https://github.com/pyannote/pyannote-audio)
- Thank you cognitivetech for the summarization system prompt: https://github.com/cognitivetech/llm-long-text-summarization/tree/main?tab=readme-ov-file#one-shot-prompting
- [Fabric](https://github.com/danielmiessler/fabric)
- [Llamafile](https://github.com/Mozilla-Ocho/llamafile) - For the local LLM inference engine
- [Mikupad](https://github.com/lmg-anon/mikupad) - Because I'm not going to write a whole new frontend for non-chat writing.
- The people who have helped me get to this point(SC & CS), and especially for those not around to see it(DT & CC).
