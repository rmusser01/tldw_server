# Code Map

This code map gives a concise, current view of the tldw_server backend. It focuses on how requests flow through FastAPI endpoints into core modules and databases, and where key features live in the repository.

## High-Level Architecture

```mermaid
%%{ init: { 'theme': 'forest', 'flowchart': { 'curve': 'stepBefore' } } }%%
flowchart LR
    Client[Web UI / API Client]
    subgraph API[FastAPI App (tldw_Server_API/app/main.py)]
      E1[api/v1 endpoints]
    end
    subgraph CORE[Core Modules]
      C1[AuthNZ]
      C2[Media Ingestion]
      C3[Chunking]
      C4[Embeddings]
      C5[RAG]
      C6[LLM Providers]
      C7[Audio STT/TTS]
      C8[MCP Unified]
    end
    subgraph DATA[Storage]
      D1[SQLite/PostgreSQL]
      D2[ChromaDB]
      D3[User DB]
    end
    Client -->|HTTP/WebSocket| API --> CORE --> DATA
```

For a deeper Mermaid atlas of request lifecycle, router groups, storage ownership, and subsystem data flows, see `Docs/Code_Documentation/Data_Flow_Atlas.md`.

## Top-Level Layout

- Server entry: `tldw_Server_API/app/main.py`
- API endpoints: `tldw_Server_API/app/api/v1/endpoints/`
- Core logic: `tldw_Server_API/app/core/`
- Web UI (Next.js): `apps/tldw-frontend/`
- Setup UI: `tldw_Server_API/app/Setup_UI/` (assets under `tldw_Server_API/app/static/setup/`)
- Config files: `tldw_Server_API/Config_Files/`
- Databases: `Databases/` (top-level) and `tldw_Server_API/Databases/` (runtime)
- Tests: `tldw_Server_API/tests/`

## API Surface (Selected)

- Auth & Users: `auth.py`, `users.py`
- Media: `media.py`, `media_embeddings.py`
- Audio: `audio.py` (OpenAI-compatible STT + WebSocket streaming)
- Chunking: `chunking.py`, `chunking_templates.py`
- Embeddings: `embeddings_v5_production_enhanced.py`, `vector_stores_openai.py`
- RAG: `rag_unified.py`, `rag_health.py`, `workflows.py`
- Chat: `chat.py`, characters (`characters_endpoint.py`, `character_*`)
- Evaluations: `evaluations_unified.py`
- Research: `research.py`, `paper_search.py`, `web_scraping.py`
- OCR: `ocr.py`
- MCP Unified: `mcp_unified_endpoint.py`
- Utilities: `metrics.py`, `setup.py`, `sync.py`, `tools.py`

Routers are mounted in `main.py` with prefix `/api/v1`.

## Core Modules (Selected)

- AuthNZ: `core/AuthNZ/` (single-user API key and multi-user JWT)
- Media Ingestion: `core/Ingestion_Media_Processing/`
  - Audio: STT (faster_whisper, NeMo Parakeet/Canary), diarization, streaming
  - Video: `Video_DL_Ingestion_Lib.py` (yt-dlp), merges, extraction
  - Documents: `PDF/`, `Books/`, `Plaintext/`, `XML/`, `OCR/`, `MediaWiki/`
- Chunking: `core/Chunking/`
  - Entry: `chunker.py`, config/types in `base.py`
  - Strategies: `strategies/` (words, sentences, paragraphs, tokens, json/xml, structure_aware, propositions, semantic, ebook_chapters, rolling_summarize)
  - Templates: `templates.py`, `template_initialization.py`, `template_library/`
- Embeddings: `core/Embeddings/` (batching, workers, vector store adapters)
- RAG: `core/RAG/` (hybrid FTS5 + vector + rerank), `rag_service/`
- LLM Calls: `core/LLM_Calls/` (providers, routing, streaming)
- MCP Unified: `core/MCP_unified/` (production-ready server + tooling)
- TTS: `core/TTS/` (OpenAI-compatible and local adapters)
- DB Management: `core/DB_Management/` (media_db package, migrations)
  - MediaDatabase docs: Docs/Code_Documentation/Databases/Media_DB_v2.md (Claims API, Chunking Templates API, Troubleshooting)
- Metrics: `core/Metrics/` (Prometheus/OpenTelemetry)

## Databases & Stores

- Media DB (content):
  - Default dev path via manager: `<USER_DB_BASE_DIR>/<user_id>/Media_DB_v2.db` (SQLite, FTS5)
  - Root-level path `Databases/Media_DB_v2.db` is deprecated
  - Backends layer wired for PostgreSQL but SQLite is default
- AuthNZ (Users):
  - `DATABASE_URL` (env) - default resolves to centralized `sqlite:///./Databases/users.db` for AuthNZ unless configured otherwise; PostgreSQL recommended for multi-user mode.
- Evaluations DB: `<USER_DB_BASE_DIR>/<user_id>/evaluations/evaluations.db` (per-user unified schema + audit; root `Databases/evaluations.db` is legacy/fallback)
- Per-user notes/chats: `<USER_DB_BASE_DIR>/<user_id>/ChaChaNotes.db`
- Per-user prompts: `<USER_DB_BASE_DIR>/<user_id>/prompts_user_dbs/user_prompts_v2.sqlite`
- Prompt Studio DB: `<USER_DB_BASE_DIR>/<user_id>/prompt_studio_dbs/prompt_studio.db`
- Vector store (per user): ChromaDB at `<USER_DB_BASE_DIR>/<user_id>/chroma_storage/` with meta/jobs SQLite under `vector_store/`
- Per-user storage (non-DB assets): outputs (`<USER_DB_BASE_DIR>/<user_id>/outputs/`), voices (`<USER_DB_BASE_DIR>/<user_id>/voices/`), rewrite cache (`<USER_DB_BASE_DIR>/<user_id>/Rewrite_Cache/rewrite_cache.jsonl`), personalization (`<USER_DB_BASE_DIR>/<user_id>/rag_personalization.json`)

Note: All paths can be overridden by environment variable or `Config_Files/config.txt`. `USER_DB_BASE_DIR` is defined in `tldw_Server_API.app.core.config` and controls the per-user root (defaults to `Databases/user_databases/`); `USER_DB_BASE` is deprecated and only used as an alias for rewrite cache resolution.

## Key Flows

1) Media Ingestion → Chunking → Embedding → RAG Index
   - Endpoint: `POST /api/v1/media/add` (preferred) and typed `POST /api/v1/media/process-*` routes → `core/Ingestion_Media_Processing/*`
   - Chunking via `core/Chunking/chunker.py` (optionally hierarchical/templates)
   - Embeddings via `core/Embeddings/` → ChromaDB and FTS5 entries in Media DB

2) Audio Transcription (file or stream)
   - Endpoints: `POST /api/v1/audio/transcriptions`, `WS /api/v1/audio/stream/transcribe`
   - Uses `core/Ingestion_Media_Processing/Audio/*` (faster_whisper/NeMo)
   - Outputs transcripts for storage/search; optional diarization

3) Chat with Retrieval
   - Endpoint: `POST /api/v1/chat/completions` (OpenAI-compatible)
   - Provider routing in `core/Chat/` and `core/LLM_Calls/`
   - Retrieval via `core/RAG/` on embedded/chunked content

4) Chunking Templates
   - Endpoints: `/api/v1/chunking/templates/...`
   - DB-backed templates via `core/Chunking/templates.py` and seeding in `template_initialization.py`

5) Evaluations
   - Unified endpoints under `/api/v1/evaluations/...`
   - Managers and metrics under `core/Evaluations/`

## Where to Add or Modify

- New endpoint: `app/api/v1/endpoints/<feature>.py` + schema under `schemas/`
- Core logic: `app/core/<Feature>/`
- DB changes: migrate via `core/DB_Management/migrations/`
- Chunking changes: `core/Chunking/` (see Chunking Module docs)
- Tests: `tldw_Server_API/tests/<feature>/`

## Notes

- CORS configured in `main.py` (adjust for deployment)
- Auth mode and keys set via env or `Config_Files/`
- Web UI: Next.js client in `apps/tldw-frontend/` interacts with the same API; setup flow served at `/setup`.
- Self-hosting setup commands are centralized in `Docs/Getting_Started/README.md` and profile guides.
