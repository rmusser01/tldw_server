# tldw_Server_API - API Guide

This document covers the API layout, how to run the server, authentication, and how to add or work with endpoints inside `tldw_Server_API`.

## Overview

The server is FastAPI-based with an OpenAI-compatible Chat and Audio API, a unified RAG module, and a unified Evaluations module. The Next.js WebUI lives under `apps/tldw-frontend/`.

Key directories:
- `app/main.py` - FastAPI app, router includes, middleware
- `app/api/v1/endpoints/` - Endpoint modules (media, chat, audio, rag, evals, prompts, notes, etc.)
- `app/api/v1/schemas/` - Pydantic request/response models
- `app/api/v1/API_Deps/` - Shared dependencies (auth, DB, rate limits)
- `app/core/` - Business logic (AuthNZ, RAG, LLM, DB, TTS, MCP, etc.)
- `app/Setup_UI/` - Setup UI HTML (served at `/setup`)

## Running the Server

```bash
python -m uvicorn tldw_Server_API.app.main:app --reload
# API docs: http://127.0.0.1:8000/docs
# Quickstart: http://127.0.0.1:8000/api/v1/config/quickstart
# Setup UI:  http://127.0.0.1:8000/setup (if required)
```

### Authentication
- Single-user mode (personal): Use `X-API-KEY` header.
- Multi-user mode (team): Use `Authorization: Bearer <JWT>`.

Quick setup:
```bash
cp tldw_Server_API/Config_Files/.env.example tldw_Server_API/Config_Files/.env
python -m tldw_Server_API.app.core.AuthNZ.initialize
```

Configure provider keys in `.env` or `tldw_Server_API/Config_Files/config.txt`.

## Endpoint Overview

- Chat: `POST /api/v1/chat/completions` (OpenAI compatible)
- Embeddings: `POST /api/v1/embeddings` (OpenAI compatible)
- Audio STT/TTS:
  - `POST /api/v1/audio/transcriptions` (file-based STT)
  - `WS  /api/v1/audio/stream/transcribe` (real-time STT)
  - `POST /api/v1/audio/speech` (TTS; streaming and non-streaming)
- Media: `POST /api/v1/media/process`, `GET /api/v1/media/search`, and related
- RAG: `POST /api/v1/rag/search` (unified pipeline)
- Evaluations (unified): `/api/v1/evaluations/...` (geval, rag, batch, metrics)
- LLM Providers: `GET /api/v1/llm/providers`
- Chatbooks: `POST /api/v1/chatbooks/export`, `POST /api/v1/chatbooks/import`
- MCP Unified: `GET /api/v1/mcp/status` and related
- OCR: `GET /api/v1/ocr/backends` (list available OCR backends and health hints)

See `app/main.py` for router includes and full route namespaces.

## Further API Docs

- Chat API: `Docs/API-related/Chat_API_Documentation.md`
- Character Chat API: `Docs/API-related/CHARACTER_CHAT_API_DOCUMENTATION.md`
- OCR Providers: `Docs/OCR/OCR_Providers.md`

## Adding a New Endpoint

1. Implement handler(s) in `app/api/v1/endpoints/<feature>.py` and schemas in `app/api/v1/schemas/`.
2. Place core logic under `app/core/<area>/` (avoid heavy logic in endpoint files).
3. Include the router in `app/main.py` with `app.include_router(...)`.
4. Add tests under `tldw_Server_API/tests/<feature>/`.

## Providers and Configuration

- Central provider configuration lives in `Config_Files/config.txt` (plus `.env`).
- RAG defaults can be tuned via env or config file. In particular, you can set the default FTS granularity for retrieval:
  - Environment (highest precedence): `RAG_DEFAULT_FTS_LEVEL=media|chunk`
  - Config file: `tldw_Server_API/Config_Files/config.txt`
    - Under `[RAG]`: `default_fts_level = media` (or `chunk`)
  - Requests can still override with `fts_level` in the unified RAG API payload.
- The `GET /api/v1/llm/providers` endpoint reflects configured providers and models. For commercial providers,
  the list is now seeded from `Config_Files/model_pricing.json` (pricing catalog) and merged with any models
  explicitly listed in `config.txt`. This makes `model_pricing.json` the primary reference for discoverable
  models; add entries there (with per‑1K prompt/completion rates) to expose them system‑wide.
  - Reload without restart: `POST /api/v1/admin/llm-usage/pricing/reload`.
- Chat request validation is in `app/api/v1/schemas/chat_request_schemas.py` and related modules.

### Chatbooks Job Backend Configuration

- `CHATBOOKS_JOBS_BACKEND`: Selects job backend for Chatbooks. Values: `core` (default) or `prompt_studio`.
- `TLDW_JOBS_BACKEND`: Module-wide default job backend (domain overrides take precedence).
- Deprecated: `TLDW_USE_PROMPT_STUDIO_QUEUE` - use `CHATBOOKS_JOBS_BACKEND=prompt_studio` instead.

- `CHATBOOKS_CORE_WORKER_ENABLED`: Enable/disable the shared core worker when using the `core` backend (default `true`). Set to `false` to skip starting the background worker even if `core` is selected.

- `JOBS_DB_URL`: Optional. If set to a PostgreSQL DSN (e.g., `postgresql://user:pass@host:5432/dbname`), the core Jobs backend uses PostgreSQL instead of SQLite.
  - Requirements: install extras `db_postgres` to pull `psycopg` and `psycopg-pool`.
  - Schema: created automatically via `app/core/Jobs/pg_migrations.py` on first use.
  - Notes: JSON fields (`payload`, `result`) are stored as JSONB in Postgres and as TEXT in SQLite.

When `core` is selected, a shared background worker starts at app startup (unless heavy startup is disabled) to process Chatbooks jobs across users.

Lease behavior and backoff
- Leases: the worker acquires jobs with a lease (`JOBS_LEASE_SECONDS`, default 60) and renews periodically (`JOBS_LEASE_RENEW_SECONDS`, default 30).
- Lease limits: cap lease/renew duration via `JOBS_LEASE_MAX_SECONDS` (default 3600).
- Renew jitter: add jitter to renewal timing via `JOBS_LEASE_RENEW_JITTER_SECONDS` (default 5 seconds) to avoid herd effects.
- Reclaim: expired processing leases are reclaimed fairly by priority and enqueue time.
- Backoff: retryable failures use exponential backoff with jitter; the next attempt’s `available_at` advances with each retry.

Signed download URLs (optional):
- `CHATBOOKS_SIGNED_URLS=true|false` - enable HMAC-signed download URLs.
- `CHATBOOKS_SIGNING_SECRET` - shared secret used to sign download links.
- `CHATBOOKS_ENFORCE_EXPIRY=true|false` - enforce job `expires_at` with `410` when expired.
- `CHATBOOKS_URL_TTL_SECONDS` - default expiry TTL for generated links (default 86400 seconds).

## Running Tests

```bash
pip install pytest httpx
python -m pytest -v
python -m pytest --cov=tldw_Server_API --cov-report=term-missing

# Markers
python -m pytest -m "unit" -v
python -m pytest -m "integration" -v
```

## OCR Backends (dots.ocr)

- The PDF pipeline supports pluggable OCR backends via `ocr_backend`.
- Built-in: `tesseract` (CLI). Optional: `dots` (dots.ocr project).

Using dots.ocr
- Select via API: set `enable_ocr=true` and `ocr_backend=dots` on PDF ingestion or OCR evaluation endpoints.
- Prompt control: set `DOTS_OCR_PROMPT` (default: `prompt_ocr`) to switch between text-only and layout prompts (see dots.ocr docs for options like `prompt_layout_only_en`).
- Installation (summary; see upstream for details):
  - Create env (Python 3.12), install a compatible PyTorch build (CUDA/CPU), then `pip install -e .` from the dots.ocr repo.
  - Download model weights: `python3 tools/download_model.py` (save directory name should not contain periods; e.g., `DotsOCR`).
  - Recommended: run a vLLM server with the downloaded weights for best performance and consistency.
- Manual install: `dots.ocr` is not bundled in `tldw-server` package metadata because PyPI rejects direct Git dependencies. Install it from the upstream repo before selecting `ocr_backend=dots`.
- Backend behavior:
  - The `dots` backend shells out to `python -m dots_ocr.parser` per page image.
  - If dots.ocr is not installed/available, auto-detect falls back to `tesseract` unless `ocr_backend=dots` is explicitly forced.
  - `ocr_mode=fallback` OCRs only pages without selectable text; use `always` to force OCR for all pages.

Integration tests
- A unit test validates backend availability handling.
- An integration test (skipped unless `dots_ocr` is importable) exercises `/api/v1/evaluations/ocr-pdf` with a tiny generated PDF and `ocr_backend=dots`.

### OCR Backends (POINTS-Reader)

- Backend name: `points`.
- Usage: set `enable_ocr=true` and `ocr_backend=points`.

Two integration modes
- Transformers (local model):
  - Install WePOINTS manually from source and use the HF model `tencent/POINTS-Reader` locally.
  - Env (optional): `POINTS_MODEL_PATH` to override model id/path.
  - Requirements: `transformers`, `torch` (CUDA recommended for performance).
- SGLang server (recommended for production):
  - Launch SGLang with `--model-path tencent/POINTS-Reader --trust-remote-code --chat-template points-v15-chat` on port 8081 (per upstream docs).
  - Env: `POINTS_SGLANG_URL` (default `http://127.0.0.1:8081/v1/chat/completions`), `POINTS_SGLANG_MODEL` (default `WePoints`).

Mode selection
- `POINTS_MODE` selects behavior: `auto` (default), `sglang`, or `transformers`.
  - `auto` prefers SGLang if `POINTS_SGLANG_URL` is set; otherwise uses transformers if libraries exist.

Prompt and generation
- Set `POINTS_PROMPT` to override the default extraction prompt (tables in HTML, others in Markdown).
- Optional generation envs: `POINTS_MAX_NEW_TOKENS`, `POINTS_TEMPERATURE`, `POINTS_REPETITION_PENALTY`, `POINTS_TOP_P`, `POINTS_TOP_K`, `POINTS_DO_SAMPLE`.

Install via extras/manual source setup (optional)
- Local transformers path: `pip install .[ocr_points_transformers]`, then install WePOINTS manually from the upstream repo.
- SGLang client only: `pip install .[ocr_points_sglang]`

Notes
- Known issues from upstream: repeated/missing content on complex layouts, difficulties with handwriting, English/Chinese focus.

See `Docs/OCR/POINTS-Reader.md` for a complete setup and usage guide.

Auto vs explicit backend
- The `auto` selection uses registry order by default (tesseract → dots → points). For the highest quality, explicitly set `ocr_backend` to `dots` or `points`, or use the new `auto_high_quality` alias which prefers ML backends (points → dots → tesseract). You can also set a custom priority via config (`[OCR] backend_priority = points, dots, tesseract`).

## Notes

- CORS and security middleware are configured in `app/main.py`.
- Single-user mode prints the API key and URLs on startup.
- The WebUI consumes the API; set `NEXT_PUBLIC_X_API_KEY` in the frontend to prefill single-user auth.

---

## Maintainer Notes (Original)

The following section preserves the original personal notes that were previously in this README for quick reference.

### Code Calling Pipeline

- You write the logic in `/app/core/<library>` and any backgroundable service processing in `/app/services`
- Then you call into the library/ies via `/api/v1/<route>`
- Which the routes themselves are defined in `main.py`

So to add a new route/API Endpoint:
- Write the endpoint in `main.py`
- Write the handling logic of the endpoint in `/api/v1/<route>`
- Write the majority/main of the processing logic in `/app/core/<library>`
- Write any background-able service processing in `/app/services`

### FastAPI on Windows (process stop)

- FastAPI has a bug, which is caused by starlette, caused by python.
- The gist is that you can't kill the python server on Windows without killing the process itself, or issuing a 'stop' command from within the process.

### Quick Commands

- Launch the API:
  - `python -m uvicorn tldw_Server_API.app.main:app --reload`
  - Visit the API via `127.0.0.1:8000/docs`

- Launching tests
  - `pip install pytest httpx`
  - `python -m pytest test_media_versions.py -v`
  - `python -m pytest .\\tldw_Server_API\\tests\\Media_Ingestion_Modification\\test_media_versions.py -v`

### Notes

- API Providers/Key checks defined in `/app/api/v1/schemas/chat_request_schemas.py`
