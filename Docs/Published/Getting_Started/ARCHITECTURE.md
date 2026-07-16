# tldw Architecture

This document describes how tldw's components connect and communicate. All diagrams use [Mermaid](https://mermaid.js.org/) syntax and render natively on GitHub.

## Component Overview

```mermaid
graph TB
    subgraph Clients
        WebUI["WebUI<br/>(Next.js :8080)"]
        Admin["Admin Panel<br/>(Next.js :3001)"]
        Ext["Browser Extension<br/>(Chrome / Firefox)"]
    end

    subgraph "tldw API Server (FastAPI :8000)"
        API["REST API<br/>/api/v1/*"]
        Auth["AuthNZ<br/>(JWT + API Key)"]
        Chat["Chat / LLM"]
        RAG["RAG Pipeline"]
        Ingest["Media Ingestion"]
        STT["Audio STT / TTS"]
        MCP["MCP Unified"]
    end

    subgraph Databases
        SQLite["SQLite<br/>(content, notes, chat)"]
        Chroma["ChromaDB<br/>(vector embeddings)"]
        PG["PostgreSQL<br/>(auth — multi-user)"]
        Redis["Redis<br/>(cache, rate limits)"]
    end

    subgraph External
        LLM["LLM Providers<br/>(OpenAI, Anthropic,<br/>Groq, local, ...)"]
        Media["Media Sources<br/>(YouTube, URLs, files)"]
        FFmpeg["FFmpeg"]
    end

    WebUI -- "same-origin rewrite" --> API
    Admin -- "server-side proxy<br/>(httpOnly cookies)" --> API
    Ext -- "direct calls<br/>(X-API-KEY)" --> API

    API --> Auth
    API --> Chat
    API --> RAG
    API --> Ingest
    API --> STT
    API --> MCP

    Chat --> LLM
    RAG --> Chroma
    RAG --> SQLite
    Ingest --> Media
    Ingest --> FFmpeg
    STT --> FFmpeg

    Auth --> SQLite
    Auth -.-> PG
    API --> Redis
    Chat --> SQLite
```

## How Components Connect

### WebUI (port 8080)

The primary user interface is a Next.js application that shares UI code with the browser extension via a monorepo (`apps/packages/ui/src/`). It runs on port 8080 (mapped from container port 3000 in Docker).

**Proxy pattern:** In Docker "quickstart" mode, the WebUI uses Next.js rewrites to forward `/api/*` requests to the API server's internal Docker hostname (`http://app:8000`). The browser never contacts the API server directly -- all requests go through the same origin and are rewritten server-side. This avoids CORS issues and keeps API keys out of the browser.

In local development, the frontend runs on port 8080 and can be pointed at a local API server via `NEXT_PUBLIC_API_URL`.

**Key files:**
- `apps/tldw-frontend/next.config.mjs` -- rewrites configuration
- `apps/tldw-frontend/pages/` -- thin page wrappers (SSR disabled)
- `apps/packages/ui/src/` -- shared components, hooks, services, stores

### Admin Panel (port 3001)

The administration dashboard is a separate Next.js application (`admin-ui/`) that runs on port 3001. It provides user management, monitoring, audit logs, data operations, and system configuration for multi-user deployments.

**Proxy pattern:** The Admin Panel uses a server-side API proxy route (`admin-ui/app/api/proxy/[...path]/route.ts`). All API calls from the browser go to `/api/proxy/*`, which the Next.js server forwards to the backend with authentication headers extracted from httpOnly cookies. The middleware (`admin-ui/middleware.ts`) validates JWT or API key cookies before allowing access to any page.

**Key files:**
- `admin-ui/app/api/proxy/[...path]/route.ts` -- server-side proxy
- `admin-ui/middleware.ts` -- authentication gate with JWT verification
- `admin-ui/lib/server-auth.ts` -- backend auth header injection

### Browser Extension

The browser extension (`apps/extension/`) is a WXT-based Chrome/Firefox/Edge extension that provides a sidepanel interface. It shares the same UI codebase (`apps/packages/ui/src/`) as the WebUI through the monorepo workspace.

**Communication:** The extension communicates directly with the tldw API server using an `X-API-KEY` header stored in extension storage. API calls are routed through the extension's background service worker (`bgRequest()`) to keep credentials out of content scripts.

**Key files:**
- `apps/extension/wxt.config.ts` -- build configuration and manifest
- `apps/extension/entrypoints/` -- background, sidepanel, options entry points
- `apps/packages/ui/src/services/background-proxy.ts` -- streaming proxy

### API Server (port 8000)

The FastAPI backend (`tldw_Server_API/`) is the core of the system. It exposes a REST API under `/api/v1/` and handles all business logic.

**Major subsystems:**

| Subsystem | Location | Purpose |
|-----------|----------|---------|
| AuthNZ | `app/core/AuthNZ/` | JWT + API key auth, RBAC, MFA, sessions |
| Chat / LLM | `app/core/LLM_Calls/` | Unified interface to 16+ LLM providers |
| RAG | `app/core/RAG/` | Hybrid search (FTS5 + vector + rerank) |
| Media Ingestion | `app/core/Ingestion_Media_Processing/` | Video, audio, PDF, EPUB, HTML, etc. |
| Audio STT/TTS | `app/api/v1/endpoints/audio/` + `app/core/TTS/` | Transcription and speech synthesis |
| Embeddings | `app/core/Embeddings/` | ChromaDB vector storage, batching |
| MCP | `app/core/MCP_unified/` | Model Context Protocol server |

**LLM provider adapters** live in `app/core/LLM_Calls/providers/` and include: OpenAI, Anthropic, Google, Cohere, DeepSeek, Groq, HuggingFace, Mistral, OpenRouter, Qwen, Moonshot, Z.AI, Bedrock, MLX, and custom OpenAI-compatible endpoints (Ollama, llama.cpp, vLLM, etc.).

### Databases

| Database | Engine | Scope | Notes |
|----------|--------|-------|-------|
| Content DB | SQLite | Per-user under `Databases/user_databases/<user_id>/` | Media items, transcripts, chunks, FTS5 index |
| ChaChaNotes | SQLite | Per-user | Chat history, notes, character sessions |
| Auth DB | SQLite or PostgreSQL | Shared | Users, roles, sessions, API keys |
| Vector Store | ChromaDB | Per-user | Embedding vectors for RAG |
| Evaluations | SQLite | Shared | Eval runs, metrics, recipes |
| Redis | Redis | Shared (optional) | Caching, rate limiting, job queues |

In single-user mode, SQLite is used for everything. Multi-user deployments typically use PostgreSQL for the auth database (configured via `DATABASE_URL`).

## Authentication Flow

```mermaid
sequenceDiagram
    participant B as Browser
    participant W as WebUI / Admin
    participant A as API Server
    participant DB as Auth DB

    Note over B,DB: Multi-user JWT flow
    B->>W: POST /login (username, password)
    W->>A: POST /api/v1/auth/login
    A->>DB: Verify credentials
    DB-->>A: User record
    A-->>W: JWT access + refresh tokens
    W-->>B: Set httpOnly cookies

    Note over B,DB: Subsequent requests
    B->>W: GET /api/proxy/v1/media/search
    W->>W: Middleware validates JWT cookie
    W->>A: Forward with Authorization header
    A-->>W: Response
    W-->>B: Response

    Note over B,DB: Extension flow (single-user)
    B->>A: GET /api/v1/chat/completions<br/>X-API-KEY: <key>
    A->>DB: Validate API key
    A-->>B: Response
```

## Data Flow: Media Ingestion

```mermaid
sequenceDiagram
    participant U as User
    participant UI as WebUI
    participant API as API Server
    participant DL as Downloader<br/>(yt-dlp)
    participant FF as FFmpeg
    participant STT as STT Engine
    participant DB as SQLite
    participant VDB as ChromaDB

    U->>UI: Submit YouTube URL
    UI->>API: POST /api/v1/media/process
    API->>DL: Download video/audio
    DL-->>API: Media file
    API->>FF: Extract audio track
    FF-->>API: Audio file
    API->>STT: Transcribe audio
    STT-->>API: Transcript text
    API->>API: Chunk transcript
    API->>DB: Store media record + chunks
    API->>VDB: Generate & store embeddings
    API-->>UI: Processing complete
    UI-->>U: Display transcript
```

## Deployment Topologies

### Single-user local (no Docker)

The simplest setup: run the API server with Python and the WebUI with Node.js/Bun.

```
Python venv                      Node.js / Bun
+--------------------------+     +--------------------+
| tldw API Server (:8000)  |     | WebUI (:8080)      |
| SQLite for everything     |<----| next.config rewrite|
| ChromaDB (embedded)      |     +--------------------+
+--------------------------+
```

**Start:**
```bash
# Terminal 1 - API server
python -m uvicorn tldw_Server_API.app.main:app --reload

# Terminal 2 - WebUI
cd apps/tldw-frontend && bun run dev -- -p 8080
```

### Single-user Docker

One `docker compose` command brings up the API, WebUI, PostgreSQL, and Redis.

```bash
docker compose -f Dockerfiles/docker-compose.yml \
               -f Dockerfiles/docker-compose.webui.yml \
               up --build
```

Services: API (:8000), WebUI (:8080), PostgreSQL (:5432), Redis (:6379).

### Multi-user Docker

Add PostgreSQL as the auth backend and optionally put Caddy in front as a reverse proxy with automatic HTTPS.

```bash
export AUTH_MODE=multi_user
export DATABASE_URL=postgresql://tldw_user:password@postgres:5432/tldw_users

docker compose -f Dockerfiles/docker-compose.yml \
               -f Dockerfiles/docker-compose.webui.yml \
               -f Dockerfiles/docker-compose.proxy.yml \
               up --build
```

Services: Caddy (:80/:443) -> API (:8000) + WebUI (:8080), PostgreSQL, Redis.

### With Admin Panel

The Admin Panel runs separately and connects to the same API server:

```bash
cd admin-ui && npm run dev  # port 3001
```

Configure it to point at the API server via environment variables. The admin panel authenticates using the same JWT/API-key credentials as any other client.
