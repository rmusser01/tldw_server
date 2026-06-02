# tldw_Server_API Data Flow Atlas

This atlas maps how data moves through `tldw_Server_API`. It is written for new contributors and maintainers who need to trace requests across FastAPI endpoints, dependencies, core modules, storage, providers, and background workers.

## Table Of Contents

- [How To Read This Atlas](#how-to-read-this-atlas)
- [System Context](#system-context)
- [Request Lifecycle](#request-lifecycle)
- [Router Group Map](#router-group-map)
- [Data Store Map](#data-store-map)
- [Core Flow Diagrams](#core-flow-diagrams)
- [Extended Domain Maps](#extended-domain-maps)
- [Router Coverage Matrix](#router-coverage-matrix)
- [How To Update This Atlas](#how-to-update-this-atlas)

## How To Read This Atlas

Use this atlas as a flow map, not as an OpenAPI replacement. Route names, module names, and storage paths should be verified against the code before edits.

| Shape or Group | Meaning |
| --- | --- |
| Clients | WebUI, admin UI, extension, HTTP clients, MCP clients, or other callers |
| FastAPI app | `app/main.py`, middleware, lifecycle, router registration |
| Endpoint groups | Routers under `app/api/v1/endpoints/`, grouped by `router_groups/*.py` |
| API dependencies | Auth, user context, DB handles, rate limits, resource governance, request validation |
| Core modules | Domain logic under `app/core/` |
| Storage | SQLite/PostgreSQL DBs, ChromaDB/pgvector, file storage, Redis/job backends |
| Providers | LLM, STT, TTS, OCR, web/media, and other external or local providers |
| Workers | Jobs, Scheduler, APScheduler bridges, background services, lifecycle workers |
| Optional routes | Feature-gated, lazy-imported, or optional dependency routes |

## System Context

```mermaid
flowchart LR
    subgraph Clients
        WebUI[Next.js WebUI]
        AdminUI[Admin UI]
        Extension[Browser extension]
        HTTP[HTTP clients]
        MCPClients[MCP clients]
    end

    subgraph FastAPI["FastAPI app"]
        Main["app/main.py"]
        Lifespan[Middleware and lifespan]
        Registry[Router registry]
    end

    subgraph Deps["API dependencies"]
        AuthDeps[Auth and user context]
        Validation[Pydantic validation]
        RateLimit[Rate limiting]
        Governance[Resource governance]
        DBDeps[DB and vector dependencies]
    end

    subgraph Endpoints["Endpoint groups"]
        Core[Core specs]
        Content[Content specs]
        Admin[Admin specs]
        Optional[Optional and minimal specs]
    end

    subgraph CoreModules["Core modules"]
        AuthNZ[AuthNZ]
        Ingestion[Ingestion]
        Chunking[Chunking]
        Embeddings[Embeddings]
        RAG[RAG]
        ChatLLM[Chat and LLM]
        AudioTTS[Audio and TTS]
        Evaluations[Evaluations]
        MCP[MCP Unified]
        JobsScheduler[Jobs and Scheduler]
        StorageCore[Storage and DB Management]
    end

    subgraph Storage["Storage"]
        AuthDB[AuthNZ DB]
        MediaDB[Per-user Media DB]
        NotesDB[Per-user ChaChaNotes DB]
        PromptDB[Prompt and Prompt Studio DBs]
        EvalDB[Per-user Evaluations DB]
        VectorStore[ChromaDB or pgvector]
        Files[Files, outputs, voices, cache]
        RedisJobs[Redis or Jobs backend]
    end

    subgraph Providers
        LLMProviders[LLM providers]
        STTProviders[STT providers]
        TTSProviders[TTS providers]
        ExternalSources[Web, media, OCR, connectors]
    end

    subgraph Workers
        JobWorkers[Jobs workers]
        SchedulerWorkers[Scheduler workers]
        APScheduler[APScheduler bridges]
        BackgroundServices[Lifecycle services]
    end

    WebUI --> Main
    AdminUI --> Main
    Extension --> Main
    HTTP --> Main
    MCPClients --> Main
    Main --> Lifespan
    Main --> Registry
    Registry --> Core
    Registry --> Content
    Registry --> Admin
    Registry --> Optional
    Core --> Deps
    Content --> Deps
    Admin --> Deps
    Optional --> Deps
    Deps --> AuthNZ
    Deps --> StorageCore
    Core --> AuthNZ
    Core --> ChatLLM
    Core --> MCP
    Content --> Ingestion
    Content --> Chunking
    Content --> Embeddings
    Content --> RAG
    Content --> AudioTTS
    Content --> Evaluations
    Content --> JobsScheduler
    Admin --> AuthNZ
    Admin --> JobsScheduler
    Admin --> StorageCore
    AuthNZ --> AuthDB
    Ingestion --> MediaDB
    Chunking --> MediaDB
    Embeddings --> VectorStore
    RAG --> MediaDB
    RAG --> VectorStore
    ChatLLM --> NotesDB
    AudioTTS --> Files
    Evaluations --> EvalDB
    MCP --> AuthDB
    JobsScheduler --> RedisJobs
    StorageCore --> AuthDB
    StorageCore --> MediaDB
    StorageCore --> NotesDB
    StorageCore --> PromptDB
    ChatLLM --> LLMProviders
    AudioTTS --> STTProviders
    AudioTTS --> TTSProviders
    Ingestion --> ExternalSources
    JobsScheduler --> JobWorkers
    JobsScheduler --> SchedulerWorkers
    APScheduler --> JobsScheduler
    BackgroundServices --> JobsScheduler
```

## Request Lifecycle

```mermaid
sequenceDiagram
    participant Client
    participant Main as app/main.py
    participant Registry as router_registry.py
    participant Spec as RouterSpec
    participant Endpoint as Endpoint router
    participant Deps as API dependencies
    participant Core as Core module
    participant Store as Storage/provider/worker

    Main->>Registry: register_all_routers or minimal register_router_specs
    Registry->>Registry: register_router_specs(specs)
    loop each RouterSpec
        alt spec has route_key
            Registry->>Registry: route_enabled(route_key, default_stable)
            alt route disabled or gating fails
                Registry-->>Main: skip router
            else route enabled
                Registry->>Spec: RouterSpec.resolve_router
            end
        else unkeyed spec
            Registry->>Spec: RouterSpec.resolve_router
        end
        Note over Spec: Lazy imported routers resolve through factories from append_imported_router_spec.
        Spec-->>Registry: APIRouter or skippable optional import error
        Registry->>Main: include_router_idempotent(router, prefix, tags)
    end

    Client->>Main: HTTP request, streaming request, or WebSocket connect
    Main->>Endpoint: route match after middleware and lifespan readiness
    Endpoint->>Deps: schema validation and dependency resolution
    Deps->>Deps: auth and user context
    Deps->>Deps: rate limit and resource governance
    alt auth, rate, governance, or validation failure
        Deps-->>Client: error response without core work
    else dependencies accepted
        Deps->>Core: request model, user context, DB handles
        Core->>Store: read/write DBs, call provider, or enqueue work
        alt normal response
            Store-->>Core: result data
            Core-->>Endpoint: response model
            Endpoint-->>Client: JSON or file response
        else streaming or WebSocket
            Store-->>Core: chunks or events
            Core-->>Client: StreamingResponse or WebSocket messages
        else async job
            Store-->>Core: job id and status handle
            Core-->>Client: job handle response
        end
    end
```

## Router Group Map

```mermaid
flowchart TB
    Main["app/main.py"]
    Mode{App mode}
    Ultra[Ultra minimal control-plane health only]
    Minimal[MINIMAL_TEST_APP]
    Full[Full app]

    Main --> Mode
    Mode --> Ultra
    Mode --> Minimal
    Mode --> Full

    subgraph MinimalPath["Minimal test registration path"]
        MinRequired[iter_minimal_test_router_specs]
        MinOptional[iter_minimal_optional_router_specs]
        MinRegister[register_router_specs]
        MinRequired --> MinRegister
        MinOptional --> MinRegister
    end

    subgraph FullPath["Full app registration path"]
        RegisterAll[register_all_routers]
        CoreSpecs[iter_core_router_specs]
        ContentSpecs[iter_content_router_specs]
        AdminSpecs[iter_admin_router_specs]
        RegisterAll --> CoreSpecs
        RegisterAll --> ContentSpecs
        RegisterAll --> AdminSpecs
    end

    subgraph CoreGroup["Core specs"]
        Infrastructure[health, moderation, monitoring, metrics, audit, consent, setup]
        Identity[auth, users, user keys, config, sync]
        ChatProviders[chat, chat loop, tools, ACP, LLM, VLM, MCP Unified]
    end

    subgraph ContentGroup["Content specs"]
        Retrieval[RAG, research, paper search]
        Processing[embeddings, media embeddings, evaluations, OCR, media, audio]
        DataWorkflows[chunking, vector stores, prompts, workflows, scheduler]
        Experience[notes, prompt studio, workspaces, characters, outputs, chatbooks]
        Integrations[connectors, ingestion sources, web scraping, Slack, Discord, Telegram, meetings]
    end

    subgraph AdminGroup["Admin specs"]
        AdminOps[admin, config admin, resource governor, jobs admin]
        OrgBilling[orgs, scoped keys, privileges, billing, invites]
        SafetyOps[guardian, self monitoring, sandbox, benchmarks, MCP catalogs and hub]
    end

    subgraph SpecFlow["Registration and gating flow"]
        Imported[append_imported_router_spec]
        RouterSpecNode[RouterSpec]
        Gate{route_enabled for route_key}
        Resolve[RouterSpec.resolve_router]
        OptionalSkip[Skip optional missing module or attribute]
        Include[include_router_idempotent]
        Registered[Router included once per router, prefix, tags]
    end

    Minimal --> MinRequired
    Full --> RegisterAll
    CoreSpecs --> CoreGroup
    ContentSpecs --> ContentGroup
    AdminSpecs --> AdminGroup
    CoreGroup --> Imported
    ContentGroup --> Imported
    AdminGroup --> Imported
    MinRegister --> RouterSpecNode
    Imported --> RouterSpecNode
    RouterSpecNode --> Gate
    Gate -->|disabled| OptionalSkip
    Gate -->|enabled or unkeyed| Resolve
    Resolve -->|optional import failure| OptionalSkip
    Resolve --> Include
    Include --> Registered
    Include -->|duplicate signature| OptionalSkip
```

## Data Store Map

```mermaid
flowchart LR
    subgraph Shared["Shared or deployment-level storage"]
        AuthDB[AuthNZ DB: Databases/users.db or PostgreSQL]
        JobDB[Jobs DB: SQLite or PostgreSQL when configured]
        Redis[Redis: queues, locks, rate/backpressure, optional job backend]
    end

    subgraph UserRoot["Per-user root: USER_DB_BASE_DIR/<user_id>/"]
        MediaDB[Media DB: Media_DB_v2.db]
        ChaCha[ChaChaNotes: ChaChaNotes.db]
        Prompts[Prompts DB and prompt libraries]
        PromptStudio[Prompt Studio DB: prompt_studio_dbs/prompt_studio.db]
        EvalDB[Per-user evaluations storage: evaluations/evaluations.db]
        Vector[ChromaDB: chroma_storage plus vector_store metadata]
        Outputs[outputs/ generated artifacts]
        Voices[voices/ custom voices and provider runtime cache]
        Rewrite[Rewrite_Cache/rewrite_cache.jsonl]
        Personalization[rag_personalization.json]
    end

    subgraph Owners["Typical owners"]
        AuthNZ[core/AuthNZ]
        DBMgmt[core/DB_Management and API_Deps]
        Ingestion[Ingestion and media endpoints]
        NotesChat[Notes, chat, characters, workspaces]
        PromptCore[Prompts and Prompt Studio]
        EvalCore[Evaluations]
        EmbedRAG[Embeddings and RAG]
        FileCore[Storage, outputs, TTS]
        JobsScheduler[Jobs, Scheduler, APScheduler]
    end

    AuthNZ --> AuthDB
    DBMgmt --> MediaDB
    DBMgmt --> ChaCha
    Ingestion --> MediaDB
    NotesChat --> ChaCha
    PromptCore --> Prompts
    PromptCore --> PromptStudio
    EvalCore --> EvalDB
    EmbedRAG --> Vector
    EmbedRAG --> MediaDB
    FileCore --> Outputs
    FileCore --> Voices
    FileCore --> Rewrite
    EmbedRAG --> Personalization
    JobsScheduler --> JobDB
    JobsScheduler --> Redis
    Ingestion --> JobsScheduler
    FileCore --> JobsScheduler
```

## Core Flow Diagrams

Placeholder: this section will collect the primary backend flow diagrams for core contributor workflows.

### Auth And User Context

Placeholder: this flow will trace single-user API key and multi-user JWT context resolution through API dependencies and AuthNZ storage.

### Media Ingestion

Placeholder: this flow will trace media requests through ingestion, metadata extraction, transcript handling, chunk persistence, and optional embedding.

### Audio STT/TTS

Placeholder: this flow will trace file and streaming transcription plus speech synthesis through audio endpoints, local providers, external providers, and output handling.

### Chunking And Embeddings

Placeholder: this flow will trace chunking templates, chunk creation, embedding generation, vector-store writes, and related metadata.

### RAG/Search

Placeholder: this flow will trace search inputs through FTS/vector retrieval, reranking, context assembly, and response construction.

### Chat And LLM Provider Calls

Placeholder: this flow will trace chat requests through conversation state, optional retrieval, provider routing, streaming, and persistence.

### Jobs And Scheduler

Placeholder: this flow will show how user-visible Jobs and internal Scheduler tasks differ, including worker and APScheduler handoffs.

## Extended Domain Maps

Placeholder: this section will collect additional domain maps once the foundation and core flows are in place.

### Evaluations

Placeholder: this flow will trace evaluation runs, recipes, metrics, audit records, and batch execution.

### MCP Unified

Placeholder: this flow will trace MCP status, tool execution, WebSocket handling, auth context, and core MCP services.

### Prompt Studio

Placeholder: this flow will trace prompt project, prompt version, test, optimization, and persistence paths.

### Notes And Chatbooks

Placeholder: this flow will trace notes, chats, character sessions, chatbook export/import, and background job handling.

### Research And Web Scraping

Placeholder: this flow will trace research and scraping requests through provider selection, extraction, aggregation, and storage.

### Storage, Files, And Outputs

Placeholder: this flow will trace upload handling, generated outputs, per-user file storage, temporary files, and cleanup responsibilities.

### Admin, Ops, And Governance

Placeholder: this flow will trace admin routes, monitoring, metrics, resource governance, rate limits, and operational controls.

### Characters And Workspaces

Placeholder: this flow will trace character card data, workspace state, chat/session links, and related per-user storage.

### Integrations And Connectors

Placeholder: this flow will trace connector routes, external integrations, optional dependency behavior, and provider handoffs.

## Router Coverage Matrix

Placeholder: this section will track every major router group or domain, representative modules, the atlas section that covers it, and any known coverage limits.

## How To Update This Atlas

Placeholder: this section will define the maintenance checklist for keeping diagrams, route groups, storage paths, and verification commands current.
