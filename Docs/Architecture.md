# Architecture Overview

This document gives new contributors a fast, opinionated tour of how tldw_server is structured and how the main pieces fit together at runtime. It complements the top-level `README.md` (high-level overview) and `Docs/Code_Documentation/Code_Map.md` (detailed code map).

If you read **this file**, then **Code_Map.md**, and skim the module-specific developer guides, you will have a solid mental model of the system.

- High-level intro and mental model
- Repository and directory layout
- Runtime architecture and request flow
- Core modules and data flows (media, RAG, chat, audio, MCP)
- Databases and storage
- Auth modes and multi-tenancy
- Patterns, conventions, and where to start when adding features

---

## 1. Mental Model

At a high level, tldw_server is:

- A **FastAPI app** exposing REST and WebSocket APIs under `/api/v1`.
- A set of **core domain modules** under `tldw_Server_API/app/core/` (AuthNZ, Media Ingestion, Chunking, Embeddings, RAG, Chat, Evaluations, MCP, etc.).
- A **storage layer** using SQLite by default (PostgreSQL supported) plus ChromaDB for vectors, with per-user content and metadata.
- A **provider layer** for commercial/local LLMs, STT/TTS backends, OCR, and connectors.
- Optional **Next.js WebUI** at `apps/tldw-frontend/` and external clients (CLI tools, MCP-aware IDE integrations).

Think of the architecture as:

> Clients → FastAPI endpoints → Core domain services → Databases / Vector stores / External providers

The goal is to keep endpoints thin, push logic into core modules, and keep storage access centralized via `core/DB_Management/` and the vector store adapters.

For a visual diagram, see `README.md` (Architecture Diagram) and `Docs/Code_Documentation/Code_Map.md`. For detailed backend data flow and process diagrams, see `Docs/Code_Documentation/Data_Flow_Atlas.md`.

---

## 2. Repository Layout (High Level)

From the repo root:

```text
<repo_root>/
├── tldw_Server_API/              # Main API server implementation
│   ├── app/
│   │   ├── api/v1/
│   │   │   ├── endpoints/        # REST endpoints (media, chat, audio, rag, evals, etc.)
│   │   │   ├── schemas/          # Pydantic models
│   │   │   └── API_Deps/         # Shared dependencies (auth, DB, rate limits)
│   │   ├── core/                 # Core logic (AuthNZ, RAG, LLM, DB, TTS, MCP, etc.)
│   │   ├── services/             # Background services and workers
│   │   └── main.py               # FastAPI entry point
│   ├── Config_Files/             # config.txt, MCP configs, helpers
│   ├── Databases/                # Runtime DBs (some paths deprecated)
│   ├── tests/                    # Pytest suite (mirrors app structure)
├── apps/tldw-frontend/                # Next.js WebUI (primary web client)
├── Docs/                         # Architecture, API, design, and developer docs
├── Dockerfiles/                  # Docker images and compose files
├── Databases/                    # AuthNZ + per-user content DB roots
├── Helper_Scripts/               # Utilities (installers, doc ingestion, etc.)
├── models/                       # Optional model assets
├── pyproject.toml                # Project configuration and extras
├── Env_Vars.md                   # Environment variable reference
├── Project_Guidelines.md         # Development philosophy and standards
└── README.md                     # High-level overview and quickstart
```

For a file-by-file code map of the backend, see `Docs/Code_Documentation/Code_Map.md`.

---

## 3. Runtime Architecture

### 3.1 Components

#### Clients
- Next.js WebUI at `apps/tldw-frontend/` (primary web client).
- Any HTTP client (curl, Postman, other backends) and MCP-aware tools.

#### FastAPI app
- Entry point: `tldw_Server_API/app/main.py`.
- Routers mounted under `/api/v1` from `app/api/v1/endpoints/`.
- Shared dependencies (auth, DB sessions, rate limiting): `app/api/v1/API_Deps/`.
- Background services and tasks: `app/services/` (jobs, schedulers, maintenance).

#### Core modules (`app/core/`)
- Domain-specific packages: AuthNZ, media ingestion, chunking, embeddings, RAG, chat, audio STT/TTS, MCP, evaluations, metrics, resource governance, etc.
- Each module is responsible for its own business logic and typically exposes pure-ish Python APIs used by endpoints.

#### Storage
- Relational databases (SQLite or PostgreSQL) for auth, jobs, evaluations, chats/notes, and media metadata.
- Per-user vector stores via ChromaDB (or pgvector when configured).
- File-based media and temporary assets (e.g., downloads, transcodes, embeddings cache).

#### External providers
- Commercial LLMs (OpenAI, Anthropic, Google, Groq, etc.).
- Local/self-hosted LLMs (Ollama, vLLM, llama.cpp, TabbyAPI, etc.).
- STT/TTS providers (faster_whisper, NeMo, Qwen2Audio, OpenAI-compatible TTS, local Kokoro ONNX).
- OCR engines, web scrapers, and other external tools.

### 3.2 High-Level Flow

Typical flow for an HTTP request:

1. **Client** calls an endpoint (e.g., `POST /api/v1/chat/completions`).
2. **FastAPI router** in `app/api/v1/endpoints/` parses/validates the request using Pydantic schemas from `app/api/v1/schemas/`.
3. **Dependencies** (`API_Deps`) inject:
   - Auth context (single-user API key or multi-user JWT).
   - Database connections (AuthNZ DB, content DBs, vector stores).
   - Rate limiting and resource governance guards.
4. The endpoint calls into one or more **core modules** (e.g., `core/RAG/`, `core/LLM_Calls/`, `core/Chat/`), which:
   - Read or write to databases via `core/DB_Management/`.
   - Call external providers via pluggable adapters.
   - Orchestrate pipelines (chunking → embeddings → search → generation).
5. The endpoint returns a response, optionally streaming via SSE/WebSocket.

For deeper diagrams and call graphs per subsystem, see:
- `Docs/Code_Documentation/Code_Map.md`
- `Docs/Code_Documentation/Embeddings-Documentation.md`
- `Docs/Code_Documentation/RAG-Developer-Guide.md`
- `Docs/MCP/Unified/Developer_Guide.md`

---

## 4. Core Modules (Backend)

Most feature work touches one or more of these directories under `tldw_Server_API/app/core/`. This list is intentionally selective; see `Code_Map.md` for a more exhaustive view.

- `AuthNZ/`
  - Auth modes (`single_user` API key vs `multi_user` JWT) and user management.
  - Initialization CLI (`python -m tldw_Server_API.app.core.AuthNZ.initialize`) for setting up DBs and keys.
  - Integration with FastAPI dependencies and security scopes.

- `Ingestion_Media_Processing/`
  - Pipelines for ingesting video, audio, documents, and web content.
  - Uses `ffmpeg`, `yt-dlp`, PDF/e-book libraries, OCR, etc.
  - Normalizes content into chunks + metadata and writes to Media DB v2.

- `Chunking/`
  - Generic chunking engine (`chunker.py`) and strategies (`strategies/`).
  - Template system (`templates.py`, `template_library/`) for hierarchical and domain-specific chunking.
  - Powers both ingestion and evaluations workflows.

- `Embeddings/`
  - Embedding pipeline (synchronous and worker-based).
  - Adapters for OpenAI-compatible and local embedding models.
  - Integrates with ChromaDB / pgvector and Media DB v2.

- `RAG/`
  - Unified retrieval pipeline combining FTS5/BM25 + vectors + re-ranking.
  - Service layer for `/api/v1/rag/*` endpoints and chat retrieval.
  - Handles scoring, ranking, and answer assembly.

- `Chat/` and `Character_Chat/`
  - OpenAI-compatible `/chat/completions` orchestration.
  - Character cards, chat sessions, and history management.
  - Provider routing and streaming orchestration (via `LLM_Calls/`).

- `LLM_Calls/`
  - Provider abstraction for 16+ LLM backends (commercial and local).
  - Handles API key usage, rate limits, error handling, and streaming.
  - Central place to add new providers or tweak provider behavior.

- `TTS/` and audio-related modules
  - Text-to-speech and speech-to-text pipelines.
  - File-based transcription (`/audio/transcriptions`) and streaming transcription (`/audio/stream/transcribe`).
  - Voice catalog and multi-provider TTS abstraction.

- `MCP_unified/`
  - Production-ready Model Context Protocol server + HTTP/WebSocket endpoints.
  - Modules (`media`, `knowledge`, `notes`, etc.) mapped to tools for agentic clients.
  - Metrics, health checks, and RBAC integration.

- `Evaluations/`
  - Unified evaluations engine (G-Eval, RAG metrics, batch scoring).
  - Integrates with embeddings, chunking, and LLM providers.
  - Backed by its own evaluations DB.

- `DB_Management/`
  - Media DB v2, notes/chats DB, migrations, and helpers.
  - Abstractions for SQLite/PostgreSQL; **no raw SQL in endpoints**.

- `Resource_Governance/` and `RateLimiting/`
  - Centralized resource governor (tokens, concurrency, quota) with Redis support.
  - Endpoint-level rate limiting and policy enforcement.

Other important areas:

- `Monitoring/`, `Metrics/`: Prometheus/OpenTelemetry exporters and metrics collection.
- `Search_and_Research/`, `WebSearch/`, `Web_Scraping/`: web search, scraping, and research helpers.
- `Notes/`, `Chatbooks/`, `Prompt_Management/`: knowledge management and artifacts.

---

## 5. Key Data Flows

This section highlights common flows a new contributor will likely touch.

### 5.1 Media Ingestion → Chunking → Embeddings → RAG

1. Client calls one of the `POST /api/v1/media/process-*` endpoints (e.g., `/process-documents`, `/process-videos`, `/process-audios`) or `/api/v1/media/add` when also persisting to the Media DB.
2. Endpoint in `app/api/v1/endpoints/media.py`:
   - Validates input and resolves user/context.
   - Calls into `core/Ingestion_Media_Processing/`.
3. Ingestion module:
   - Downloads/transcodes media if necessary (`yt-dlp`, `ffmpeg`, etc.).
   - Extracts raw text/transcripts + metadata.
   - Writes media and basic metadata into Media DB v2 via `DB_Management/`.
4. Chunking module (`core/Chunking/`):
   - Splits content by strategy and/or templates.
   - Assigns chunk IDs and hierarchy.
5. Embeddings module (`core/Embeddings/`):
   - Computes embeddings for chunks.
   - Writes vectors and metadata to ChromaDB / pgvector and updates Media DB.
6. RAG module (`core/RAG/`):
   - Exposes search endpoints (`/api/v1/rag/*`).
   - Uses both text and vector indexes when serving queries.

### 5.2 Chat with Retrieval

1. Client calls `POST /api/v1/chat/completions` with messages and optional retrieval settings.
2. Endpoint in `app/api/v1/endpoints/chat.py`:
   - Resolves provider/model (from config, aliases, or request).
   - Optionally calls `core/RAG/` to fetch context for retrieval-augmented replies.
3. `core/Chat/` orchestrates:
   - System/instruction messages.
   - Context windows and truncation/compaction.
   - Conversation persistence.
4. `core/LLM_Calls/` sends the final request to the chosen provider and streams the response back to the client.

### 5.3 Audio STT/TTS and Streaming

1. STT (file): `POST /api/v1/audio/transcriptions`.
2. STT (streaming): `WS /api/v1/audio/stream/transcribe`.
3. TTS: `POST /api/v1/audio/speech`.
4. Endpoints delegate to:
   - `core/Ingestion_Media_Processing/Audio/*` for STT.
   - `core/TTS/` for TTS and voice management.
5. Outputs can be:
   - Persisted as media items for search and RAG.
   - Streamed directly to clients.

For subsystem-level diagrams and details, see:
- `Docs/Code_Documentation/Ingestion_Media_Processing.md`
- `Docs/User_Guides/WebUI_Extension/TTS_Getting_Started.md`
- `Docs/Development/Audio-Multi-User-Architecture.md`

---

## 6. Databases and Storage

Database design is covered in depth in:
- `Docs/Code_Documentation/Database.md`
- the media DB code documentation page
- `Docs/Code_Documentation/Databases/ChaChaNotes_DB.md`

This section gives the quick mental model.

Note: `<USER_DB_BASE_DIR>` is defined in `tldw_Server_API.app.core.config`, defaults to `Databases/user_databases/` under the project root, and can be overridden via environment variable or `Config_Files/config.txt`.

### AuthNZ DB
- Centralized in all auth modes.
- Default (single-user): SQLite file configured by `DATABASE_URL` (defaults to `sqlite:///./Databases/users.db`).
- Multi-user: centralized PostgreSQL instance (e.g., `postgresql://user:password@host:5432/tldw_users`).
- Unlike per-user Content/Media DBs under `<USER_DB_BASE_DIR>/<user_id>/`, AuthNZ data remains centralized.
- Stores users, credentials, permissions, and related auth data.

### Content / Media DB
- Per-user SQLite DB under `<USER_DB_BASE_DIR>/<user_id>/<content-db>.db`.
- Stores media items, chunks, metadata, and FTS indexes.
- Root-level single-file content DB paths are deprecated; always go through the DB helpers.
- Replace `<content-db>.db` with your configured per-user content DB filename.

### Notes / Chats / Characters
- Per-user `ChaChaNotes.db` under `<USER_DB_BASE_DIR>/<user_id>/ChaChaNotes.db`.
- Stores notes, chat history, and character data.

### Prompt Studio and related artifacts
- Per-user prompts DB under `<USER_DB_BASE_DIR>/<user_id>/prompts_user_dbs/user_prompts_v2.sqlite`.

### Evaluations DB
- Per-user SQLite DB under `<USER_DB_BASE_DIR>/<user_id>/evaluations/evaluations.db`.
- Stores evaluations, metrics, and audit logs for the resolved user context.
- Root-level `Databases/evaluations.db` may exist as a legacy/fallback path; use `DatabasePaths.get_evaluations_db_path(user_id)` for normal access.

### Vector Store
- Default: ChromaDB, usually per-user under `<USER_DB_BASE_DIR>/<user_id>/chroma_storage/`.
- Optionally: PostgreSQL with pgvector, configured via `config.txt` and env vars.

All DB access should go through the abstractions in `core/DB_Management/` and the vector store wrappers in `core/Embeddings/` and `core/RAG/`.

---

## 7. Auth Modes and Multi-Tenancy

tldw_server supports two primary auth modes:

- `AUTH_MODE=single_user`
  - Simple API key authentication via `X-API-KEY` header.
  - Intended for personal/local deployments and single-user setups.
  - Content and notes are still organized per logical user ID, but the AuthNZ layer is simpler.

- `AUTH_MODE=multi_user`
  - JWT-based auth with signup/login flows and permissions.
  - Recommended for multi-tenant deployments and hosted environments.
  - Typically paired with PostgreSQL for AuthNZ DB and Job DB.

Per-user data:

- User identity (from API key or JWT) is mapped to a **user_id**.
- Per-user DB paths are derived from this user_id under `<USER_DB_BASE_DIR>/` (defaults to `Databases/user_databases/` unless configured).
- RAG, notes, prompts, and vector stores all use these per-user roots to keep content logically isolated.

See:
- `Env_Vars.md` for environment variable reference.
- `Docs/Code_Documentation/AuthNZ-Developer-Guide.md` for implementation details.

---

## 8. Frontend and Clients

### Next.js WebUI (`apps/tldw-frontend/`)
- Primary web client, talking to the same FastAPI APIs (`/api/v1`).
- Focused on interactive media ingestion, search, chat, and evaluations.

#### Programmatic clients
- Any HTTP client can call the OpenAI-compatible Chat, Embeddings, Audio, and RAG endpoints.
- MCP clients (IDEs, agents) use the MCP Unified APIs at `/api/v1/mcp/*`.

Key documentation:
- `Docs/API-related/API_README.md`
- `Docs/MCP/Unified/Developer_Guide.md`
- `Docs/MCP/Unified/Documentation_Ingestion_Playbook.md`

---

## 9. Patterns, Conventions, and How to Add Features

The project guidelines in `Project_Guidelines.md` and `AGENTS.md` cover philosophy in detail. This section summarizes the most important patterns for contributors.

#### Coding patterns
- Prefer **thin endpoints** and **fat core modules**:
  - Endpoint: parse/validate, call core, shape response.
  - Core: domain logic, side effects, DB + provider integration.
- Rely on **Pydantic models** for all API inputs/outputs (`app/api/v1/schemas/`).
- Keep functions focused on single responsibilities and fully type hinted.
- Prefer **async/await** for I/O-bound code (HTTP calls, DB, file I/O).
- Centralize DB access via `core/DB_Management/`; avoid raw SQL in endpoints.

#### Adding a new feature
1. **Design first**: Sketch the feature and data flow. For larger features, add a design doc under `Docs/Design/`.
2. **Core implementation**: Add business logic under `app/core/<Feature>/` or extend an existing module.
3. **API layer**: Add or update endpoints under `app/api/v1/endpoints/` and Pydantic models under `app/api/v1/schemas/`.
4. **Dependencies**: If you need shared dependencies (auth, DB, rate limits), wire them in `API_Deps/`.
5. **Tests**: Add tests under `tldw_Server_API/tests/<feature>/` mirroring the app structure.
6. **Config and docs**: Wire any knobs into `Config_Files/config.txt` and update docs under `Docs/`.

#### Testing and local dev
- Run tests via `python -m pytest -v` from the repo root.
- Use markers (`unit`, `integration`, `e2e`, `external_api`, `performance`) to scope suites.
- For DB-intensive features, prefer existing fixtures (e.g., AuthNZ Postgres fixture) over custom setups.

---

## 10. Where to Go Next

If you are new to the project, a good path is:

1. Read `README.md` (Overview, Architecture & Repo Layout, Quickstart).
2. Read this file (`Docs/Architecture.md`) to internalize the mental model.
3. Open `Docs/Code_Documentation/Code_Map.md` and skim:
   - High-Level Architecture
   - Top-Level Layout
   - Key Flows
4. Jump into module guides for the area you care about:
   - RAG: `Docs/Code_Documentation/RAG-Developer-Guide.md`
   - AuthNZ: `Docs/Code_Documentation/AuthNZ-Developer-Guide.md`
   - Embeddings: `Docs/Code_Documentation/Embeddings-Documentation.md`
   - Chat & Chatbooks: `Docs/Code_Documentation/Chat_Developer_Guide.md`, `Docs/Code_Documentation/Chatbook_Developer_Guide.md`
   - MCP: `Docs/MCP/Unified/Developer_Guide.md`
5. Review `Project_Guidelines.md` and `Env_Vars.md` before making substantial changes.

With those pieces in place, you should be able to:
- Trace any request from client → endpoint → core module → database/provider.
- Identify where to plug in new functionality.
- Confidently navigate the codebase without being overwhelmed by its size.
