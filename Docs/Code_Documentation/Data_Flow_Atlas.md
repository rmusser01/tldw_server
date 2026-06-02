# tldw_Server_API Data Flow Atlas

This atlas maps how data moves through `tldw_Server_API`. It is written for new contributors and maintainers who need to trace requests across FastAPI endpoints, dependencies, core modules, storage, providers, and background workers.

File path: `Docs/Code_Documentation/Data_Flow_Atlas.md`

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

These flows trace the backend paths most likely to matter when a newcomer asks where data goes after an API call. They are intentionally grouped by process rather than by every route handler.

### Auth And User Context

**Purpose:** Resolve the caller, enforce auth policy, and turn identity into the user-scoped paths used by content modules.

**Primary entrypoints:** Most protected endpoints through `get_current_user`, `get_request_user`, `AuthPrincipal`, `TokenScopeGuard`, `RequireRole`, and related dependencies in `app/api/v1/API_Deps/auth_deps.py`.

```mermaid
flowchart LR
    subgraph Caller["Caller credentials"]
        APIKey[X-API-KEY single-user or API key]
        Bearer[Authorization bearer JWT]
        Cookie[Session or browser context]
    end

    subgraph Deps["API auth dependencies"]
        AuthDep[get_current_user and get_request_user]
        Principal[AuthPrincipal and user dict]
        Guards[Role, scope, rate, quota guards]
    end

    subgraph IdentityStore["Identity and auth storage"]
        AuthNZ[core/AuthNZ]
        AuthDB[AuthNZ DB: users, sessions, API keys, RBAC, MFA]
        JWTService[JWT service and session manager]
        APIKeyMgr[API key manager]
    end

    subgraph UserContext["Resolved user context"]
        SingleUser[Fixed single-user principal]
        MultiUser[DB-backed user principal]
        UserId[user_id and permissions]
    end

    subgraph UserStorage["Per-user content storage selection"]
        DBPaths[DatabasePaths and API_Deps DB helpers]
        UserRoot["USER_DB_BASE_DIR/<user_id>/"]
        MediaDB[Media DB, FTS, chunks]
        NotesDB[ChaChaNotes]
        VectorRoot[ChromaDB and vector metadata]
        EvalDB[Per-user evaluations DB]
    end

    APIKey --> AuthDep
    Bearer --> AuthDep
    Cookie --> AuthDep
    AuthDep --> AuthNZ
    AuthNZ --> APIKeyMgr
    AuthNZ --> JWTService
    APIKeyMgr --> AuthDB
    JWTService --> AuthDB
    AuthNZ --> Principal
    Principal --> Guards
    Guards -->|single_user mode| SingleUser
    Guards -->|multi_user mode| MultiUser
    SingleUser --> UserId
    MultiUser --> UserId
    UserId --> DBPaths
    DBPaths --> UserRoot
    UserRoot --> MediaDB
    UserRoot --> NotesDB
    UserRoot --> VectorRoot
    UserRoot --> EvalDB
```

**Key storage/provider touchpoints:** AuthNZ DB stores identity, sessions, API keys, RBAC, quotas, and MFA state. Per-user content storage is selected only after user context resolves; it lives under `USER_DB_BASE_DIR/<user_id>/` and includes Media DB, ChaChaNotes, ChromaDB/vector metadata, prompts, outputs, and per-user evaluations storage.

**Where to look in code:** `app/api/v1/API_Deps/auth_deps.py`, `app/core/AuthNZ/`, `app/core/DB_Management/db_path_utils.py`, `app/core/DB_Management/Users_DB.py`, and the per-domain DB dependency modules under `app/api/v1/API_Deps/`.

### Media Ingestion

**Purpose:** Convert files, documents, URLs, web pages, audio, and video into normalized records, chunks, search indexes, and optional embeddings so content is searchable and RAG-ready.

**Primary entrypoints:** `POST /api/v1/media/add`, `POST /api/v1/media/process-documents`, `POST /api/v1/media/process-videos`, `POST /api/v1/media/process-audios`, `POST /api/v1/media/process-pdfs`, `POST /api/v1/media/process-ebooks`, web scraping and ingestion-source routes.

```mermaid
flowchart LR
    subgraph Inputs
        Files[Uploaded files and documents]
        URLs[URL, video, audio, feed inputs]
        Web[Web scraping and article extraction]
    end

    subgraph EndpointLayer["Media endpoints"]
        Add["/media/add persistent ingest"]
        Process["process-* no-persistence helpers"]
        JobsPath[Optional Jobs or background path]
    end

    subgraph Processing["core/Ingestion_Media_Processing"]
        Dispatch[Media type dispatch]
        Download[Download with yt-dlp or URL fetch]
        Transcode[ffmpeg transcode or audio extraction]
        OCR[PDF or image OCR branch]
        STT[Audio/video STT branch]
        Parse[Document, ebook, HTML, XML parsing]
        Normalize[Normalize text, metadata, transcript segments]
        Chunk[Chunking strategies and templates]
    end

    subgraph Searchable["Search and RAG readiness"]
        Persist[Persist primary item and metadata]
        MediaDB[Per-user Media DB]
        FTS[FTS5 media and keyword indexes]
        EmbedOpt{generate_embeddings?}
        Embed[Embedding provider/model]
        Vector[Per-user ChromaDB or vector backend]
    end

    Files --> Add
    URLs --> Add
    Web --> Add
    Files --> Process
    URLs --> Process
    Add --> Dispatch
    Process --> Dispatch
    Add --> JobsPath
    JobsPath --> Dispatch
    Dispatch --> Download
    Dispatch --> Parse
    Download --> Transcode
    Transcode --> STT
    Parse --> OCR
    OCR --> Normalize
    STT --> Normalize
    Parse --> Normalize
    Normalize --> Chunk
    Chunk --> Persist
    Persist --> MediaDB
    MediaDB --> FTS
    Persist --> EmbedOpt
    EmbedOpt -->|yes| Embed
    Embed --> Vector
    EmbedOpt -->|no| FTS
```

**Key storage/provider touchpoints:** Media DB stores content, transcripts, metadata, chunks, keywords, and FTS state. Embedding generation writes per-user vector records and vector metadata. Providers include yt-dlp, ffmpeg, OCR backends, STT backends, web extractors, embedding providers, and optional Jobs workers.

**Where to look in code:** `app/api/v1/endpoints/media/`, `app/core/Ingestion_Media_Processing/`, `app/core/DB_Management/Media_DB_v2.py`, `app/core/DB_Management/media_db/`, `app/core/Embeddings/`, `Docs/Code_Documentation/Pieces.md`, and `Docs/Code_Documentation/Ingestion_Pipeline_Video.md`.

### Audio STT/TTS

**Purpose:** Handle file transcription, real-time streaming transcription, and speech synthesis while keeping the file, WebSocket, and TTS paths distinct.

**Primary entrypoints:** `POST /api/v1/audio/transcriptions`, `WS /api/v1/audio/stream/transcribe`, `POST /api/v1/audio/speech`, `GET /api/v1/audio/voices/catalog`, audio history and audio job/status endpoints.

```mermaid
flowchart TB
    subgraph FileSTT["File transcription path"]
        FileReq["/audio/transcriptions upload"]
        ValidateAudio[Validate file and options]
        STTBackend[Select STT backend: faster_whisper, NeMo, Qwen, local]
        Transcript[Transcript, segments, SRT/VTT/JSON]
        STTResponse[Return transcript response]
        UploadRetention[Uploaded audio retained only by STT policy]
    end

    subgraph StreamSTT["Streaming transcription path"]
        WSReq["WebSocket /audio/stream/transcribe"]
        StreamAuth[Token or auth context]
        StreamConfig[Streaming model config]
        AudioChunks[Incoming audio chunks]
        PartialFinal[Partial and final transcript frames]
        PersistGate{persist_transcript and media_id?}
        NoPersist[No Media DB transcript write]
    end

    subgraph TTSPath["TTS path"]
        SpeechReq["/audio/speech text request"]
        VoiceCatalog[Voice catalog and settings]
        TTSBackend[Select TTS backend: OpenAI-compatible or Kokoro/local]
        AudioOut[Audio bytes or file output]
    end

    subgraph OptionalPersistence["Optional persistence and background tracking"]
        TTSHistory[TTS history and audio job records]
        MediaPersist[upsert_transcript writes media transcript]
        ChunkSearch[Optional chunk and index transcript]
        MediaDB[Per-user Media DB and FTS]
        Vector[Optional embeddings and vector store]
        Files[Per-user outputs, voices, retained artifacts]
        Jobs[Audio Jobs/background workers]
    end

    FileReq --> ValidateAudio --> STTBackend --> Transcript
    FileReq --> UploadRetention --> Files
    WSReq --> StreamAuth --> StreamConfig --> AudioChunks --> PartialFinal
    SpeechReq --> VoiceCatalog --> TTSBackend --> AudioOut
    Transcript --> STTResponse
    PartialFinal --> PersistGate
    PersistGate -->|yes| MediaPersist
    PersistGate -->|no| NoPersist
    AudioOut --> TTSHistory
    MediaPersist --> ChunkSearch
    ChunkSearch --> MediaDB
    ChunkSearch --> Vector
    AudioOut --> Files
    TTSHistory --> Jobs
```

**Key storage/provider touchpoints:** STT and TTS providers may be local runtimes or external OpenAI-compatible services. File STT usually returns transcript responses; uploaded audio may be retained according to STT policy. Streaming transcript persistence is opt-in and requires `persist_transcript` plus `media_id` before `upsert_transcript` writes to the Media DB. TTS has history/audio jobs, and generated or uploaded artifacts may be retained by policy. Media transcript persistence is optional and conditional; only persisted transcripts can later be chunked, indexed with FTS, and embedded for RAG.

**Where to look in code:** `app/api/v1/endpoints/audio/`, especially `audio.py`, `audio_transcriptions.py`, `audio_streaming.py`, `audio_tts.py`, `audio_history.py`, and `audio_jobs.py`; also `app/core/Ingestion_Media_Processing/Audio/`, `app/core/TTS/`, `Docs/STT-TTS/`, and media persistence helpers when transcription is saved as content.

### Chunking And Embeddings

**Purpose:** Produce stable text pieces from raw content and attach embedding vectors so chunks can be retrieved by FTS, BM25, vector search, or hybrid RAG.

**Primary entrypoints:** `POST /api/v1/chunking/chunk_text`, chunk template routes, ingestion-triggered chunking in media/process endpoints, embedding endpoints, media embedding jobs, and vector-store admin routes.

```mermaid
flowchart LR
    subgraph Triggers
        APIChunk[API-triggered chunk_text]
        IngestChunk[Ingestion-triggered chunking]
        Batch[Batch or worker-triggered embedding]
    end

    subgraph Chunking["core/Chunking"]
        Options[Resolve strategy/template/options]
        Strategies[words, sentences, paragraphs, tokens, semantic, template]
        Pieces[Chunk objects with text, offsets, metadata]
    end

    subgraph Metadata["Media DB relationship"]
        MediaItem[Media item or transcript]
        Unvectorized[UnvectorizedMediaChunks]
        MediaChunks[MediaChunks or claims/propositions]
        FTS[FTS5 text and keyword indexes]
    end

    subgraph Embeddings["core/Embeddings"]
        Provider[Embedding provider/model selection]
        Queue[Batch/job metadata]
        Vectors[Vector records]
        VectorStore[Per-user ChromaDB or pgvector]
    end

    APIChunk --> Options
    IngestChunk --> Options
    Options --> Strategies --> Pieces
    Pieces --> MediaItem
    Pieces --> Unvectorized
    Pieces --> MediaChunks
    MediaItem --> FTS
    MediaChunks --> FTS
    Unvectorized --> Batch
    Batch --> Queue
    APIChunk --> Provider
    IngestChunk --> Provider
    Queue --> Provider
    Provider --> Vectors --> VectorStore
    Vectors --> MediaChunks
```

**Key storage/provider touchpoints:** Chunk metadata and FTS state live in the per-user Media DB. Vector payloads and embedding job/batch metadata live under the per-user vector store path. Embedding providers and models are resolved from request/config, and chunking can be invoked directly by API callers or indirectly by ingestion.

**Where to look in code:** `app/api/v1/endpoints/chunking.py`, embedding endpoints, `app/core/Chunking/`, `app/core/Ingestion_Media_Processing/chunking_options.py`, `app/core/Embeddings/ChromaDB_Library.py`, vector metadata/job DB modules, `Docs/Code_Documentation/Pieces.md`, and `Docs/Code_Documentation/Database.md`.

### RAG/Search

**Purpose:** Normalize search/RAG requests, retrieve candidate chunks from lexical and vector paths, rerank and post-process them, then assemble results or generation context.

**Primary entrypoints:** `POST /api/v1/rag/search`, `POST /api/v1/rag/search/stream`, RAG settings/backends endpoints, media search endpoints, and chat flows that request RAG context before generation.

```mermaid
flowchart LR
    subgraph Request
        Standalone[Standalone RAG/Search endpoint]
        ChatUse[Chat asks for optional RAG context]
        Normalize[resolve_rag_request and settings]
    end

    subgraph Retrieval["Hybrid retrieval"]
        Plan[Retrieval plan]
        FTS[FTS/BM25 retrieval from Media DB]
        Vector[Vector retrieval from ChromaDB or pgvector]
        Merge[Score normalization and merge]
    end

    subgraph RankContext["Rank and context assembly"]
        Rerank[rerank: FlashRank, cross-encoder, hybrid, llama.cpp, or none]
        Filters[Security filters, citations, highlighting]
        Context[Result/context assembly]
        Stream[Optional event stream]
    end

    subgraph Consumers
        SearchResponse[RAG search response]
        ChatPrompt[Context passed to chat prompt]
        Feedback[Feedback and analytics]
    end

    Standalone --> Normalize
    ChatUse --> Normalize
    Normalize --> Plan
    Plan --> FTS
    Plan --> Vector
    FTS --> Merge
    Vector --> Merge
    Merge --> Rerank
    Rerank --> Filters
    Filters --> Context
    Context --> SearchResponse
    Context --> ChatPrompt
    Context --> Stream
    SearchResponse --> Feedback
```

**Key storage/provider touchpoints:** FTS/BM25 reads from the per-user Media DB and its FTS tables. Vector retrieval reads per-user ChromaDB or pgvector collections populated by embeddings. Rerankers may use local models or provider-backed adapters. Feedback and analytics attach to the RAG service path.

**Where to look in code:** `app/api/v1/endpoints/rag_unified.py`, `app/core/RAG/rag_service/request_resolution.py`, `retrieval_plan.py`, `database_retrievers.py`, `unified_pipeline.py`, `response_mapping.py`, `streaming_executor.py`, and embedding/vector-store modules.

### Chat And LLM Provider Calls

**Purpose:** Accept OpenAI-compatible chat requests, optionally enrich them with retrieval context, resolve a provider/model, call the adapter, and persist conversation state separately from retrieval.

**Primary entrypoints:** `POST /api/v1/chat/completions`, chat session/conversation routes, chat document/workflow routes, `/api/v1/llm/providers`, and provider metadata/model routing routes.

```mermaid
flowchart LR
    subgraph ChatRequest["Chat generation"]
        Endpoint["/chat/completions"]
        Validate[OpenAI-compatible request validation]
        Session[Optional conversation or session state]
        PersistIn[Persist user message when configured]
    end

    subgraph RetrievalContext["Optional retrieval"]
        NeedRAG{RAG requested?}
        RAGFlow["RAG/Search flow"]
        PromptContext[Prompt context and citations]
    end

    subgraph ProviderCall["LLM provider call"]
        Resolve[Provider/model resolution and BYOK/config lookup]
        Adapter[LLM adapter registry]
        External[Commercial or local provider]
    end

    subgraph ResponsePaths["Response paths"]
        NonStream[Non-streaming JSON response]
        Stream[Streaming SSE chunks]
        PersistOut[Persist assistant message and metadata]
        NotesDB[Per-user ChaChaNotes chat/session DB]
    end

    Endpoint --> Validate --> Session --> PersistIn
    PersistIn --> NeedRAG
    NeedRAG -->|yes| RAGFlow --> PromptContext --> Resolve
    NeedRAG -->|no| Resolve
    Resolve --> Adapter --> External
    External -->|complete response| NonStream
    External -->|delta events| Stream
    NonStream --> PersistOut
    Stream --> PersistOut
    PersistOut --> NotesDB
```

**Key storage/provider touchpoints:** Chat/session state persists in the per-user ChaChaNotes database when configured. RAG context is assembled from Media DB and vector-store reads but remains separable from generation. Provider resolution can use config, BYOK/user provider secrets, model routing, and adapter registry entries for OpenAI-compatible, commercial, and local providers.

**Where to look in code:** chat endpoints under `app/api/v1/endpoints/`, `app/core/Chat/`, `app/core/LLM_Calls/adapter_registry.py`, `app/core/LLM_Calls/providers/`, `app/core/LLM_Calls/routing/`, `app/core/AuthNZ/byok_helpers.py`, and `app/core/DB_Management/ChaChaNotes_DB.py`.

### Jobs And Scheduler

**Purpose:** Distinguish user-visible Jobs from internal Scheduler orchestration and show how recurring APScheduler services bridge into the chosen backend.

**Primary entrypoints:** Jobs admin/status endpoints, domain workers that enqueue Jobs, Scheduler workflow endpoints, `@task`-registered scheduler handlers, APScheduler-backed workflow and digest services.

```mermaid
flowchart LR
    subgraph Producers
        UserAction[User-visible long work]
        InternalFlow[Internal orchestration]
        Recurring[Recurring APScheduler trigger]
    end

    subgraph JobsPath["Jobs backend"]
        JobCreate[Create Job with owner, domain, quota]
        JobDB[Jobs DB or Redis-backed state]
        Admin[Admin status, pause, resume, drain, retry]
        WorkerSDK[Jobs WorkerSDK or domain worker]
        JobResult[Result, failure, retry, audit]
    end

    subgraph SchedulerPath["Core Scheduler backend"]
        TaskReg[@task handler registration]
        TaskCreate[Create task with dependency and idempotency key]
        SchedulerDB[Scheduler persistence]
        Dependency[Dependency resolution]
        SchedulerWorker[Scheduler worker pool]
        TaskResult[Task result and workflow state]
    end

    subgraph Bridge["APScheduler bridges"]
        APS[APScheduler service]
        Choose{Chosen backend}
    end

    UserAction --> JobCreate --> JobDB --> Admin
    JobDB --> WorkerSDK --> JobResult --> Admin
    InternalFlow --> TaskReg --> TaskCreate --> SchedulerDB --> Dependency --> SchedulerWorker --> TaskResult
    Recurring --> APS --> Choose
    Choose -->|user-visible or ops-controlled| JobCreate
    Choose -->|dependency orchestration| TaskCreate
```

**Key storage/provider touchpoints:** Jobs use a Jobs backend for owner/domain state, retries, admin controls, quotas, worker leases, and status summaries. Scheduler uses its own persistence for task registration, dependencies, idempotency, and workflow execution. APScheduler services should enqueue into Jobs or Scheduler according to the workflow they support.

**Where to look in code:** `app/api/v1/endpoints/jobs_admin.py`, `app/core/Jobs/`, `app/services/*jobs_worker*.py`, `app/api/v1/endpoints/scheduler_workflows.py`, `app/core/Scheduler/`, workflow/watchlist scheduler services, and APScheduler startup/lifecycle services.

**Decision note:** Use Jobs for new user-visible features or work needing admin/ops status, pause/resume/drain, retries, quotas, or RLS. Use Scheduler for internal orchestration where registered handlers, task dependencies, and idempotency keys are central. Recurring schedules should use APScheduler to enqueue into whichever backend the feature needs.

## Extended Domain Maps

These maps cover the remaining router domains at group level. They avoid endpoint inventory detail, but each section names the route families, core services, storage, providers, and handoff points needed to trace a domain end to end.

### Evaluations

**Purpose:** Manage evaluation recipes, datasets, runs, model-graded checks, RAG evaluations, metrics, and result persistence without mixing evaluation state into chat or media storage.

**Primary entrypoints:** `/api/v1/evaluations`, `/api/v1/evaluations/datasets`, `/api/v1/evaluations/{eval_id}/runs`, recipes, synthetic datasets, RAG pipeline evaluation, embeddings A/B tests, benchmarks, webhooks, and evaluation history/status routes.

```mermaid
flowchart LR
    subgraph Routes["Evaluation routes"]
        CRUD[Recipes and evaluation CRUD]
        Datasets[Datasets and samples]
        Runs[Runs, cancel, history, status]
        RAGHooks[RAG eval and benchmark hooks]
    end

    subgraph Services["core/Evaluations"]
        Unified[UnifiedEvaluationService]
        Runner[Evaluation runner and recipe executors]
        Judge[GEval, response quality, LLM judge]
        Metrics[Metrics, audit, webhooks]
    end

    subgraph External["Inputs and providers"]
        RAG[RAG/Search results and traces]
        LLM[LLM provider or BYOK judge call]
        Embed[Embedding provider for A/B tests]
    end

    subgraph Storage["Per-user evaluation storage"]
        EvalDB["USER_DB_BASE_DIR/<user_id>/evaluations/evaluations.db"]
        Audit[Unified audit events]
        Results[Metrics, outputs, run state]
    end

    CRUD --> Unified
    Datasets --> Unified
    Runs --> Runner
    RAGHooks --> Runner
    Unified --> EvalDB
    Runner --> RAG
    Runner --> Judge
    Judge --> LLM
    Runner --> Embed
    Runner --> Results --> EvalDB
    Metrics --> Audit
    Metrics --> EvalDB
```

**Key storage/provider touchpoints:** Evaluations use per-user evaluation storage where user context is available, including recipes, datasets, runs, idempotency keys, metrics, and results. LLM judge calls go through configured provider/BYOK resolution; RAG evaluation reads RAG outputs and persists evaluation metrics rather than changing RAG storage directly.

**Where to look in code:** `app/api/v1/endpoints/evaluations/`, `app/core/Evaluations/`, `app/core/DB_Management/Evaluations_DB.py`, `app/core/DB_Management/db_path_utils.py`, `Docs/Code_Documentation/Evaluations_Developer_Guide.md`, and RAG evaluation helpers under `app/core/RAG/`.

### MCP Unified

**Purpose:** Expose MCP over HTTP and WebSocket with AuthNZ/RBAC, module/tool discovery, domain dispatch, health, metrics, and tool execution responses.

**Primary entrypoints:** `/api/v1/mcp`, `/api/v1/mcp/request/batch`, `/api/v1/mcp/ws`, `/api/v1/mcp/status`, `/api/v1/mcp/metrics`, `/api/v1/mcp/tools`, `/api/v1/mcp/tools/execute`, `/api/v1/mcp/modules`, `/api/v1/mcp/resources`, `/api/v1/mcp/prompts`, MCP token routes, hub routes, and scoped tool catalog routes.

```mermaid
flowchart LR
    subgraph Entrypoints["MCP entrypoints"]
        HTTP[MCP JSON-RPC HTTP]
        Batch[Batch request]
        WS[WebSocket session]
        Status[Status, metrics, health]
        Tools[Tools, modules, resources, prompts]
    end

    subgraph Security["Auth and governance"]
        Auth[API key, JWT, or MCP JWT]
        RBAC[Permissions and RBAC]
        Catalogs[Tool catalogs and org/team scope]
    end

    subgraph Server["core/MCP_unified"]
        ServerCore[Unified MCP server]
        Registry[Module and tool registry]
        Dispatch[Domain dispatch]
        Monitor[Metrics and monitoring]
    end

    subgraph Domains["Tool domains"]
        Content[Content, RAG, notes, media]
        Admin[Admin and configuration tools]
        External[External MCP servers and hub]
    end

    subgraph Output["Responses and telemetry"]
        ToolResult[Tool execution result]
        Lists[Filtered discovery lists]
        Health[Status and Prometheus metrics]
    end

    HTTP --> Auth
    Batch --> Auth
    WS --> Auth
    Tools --> Auth
    Status --> RBAC
    Auth --> RBAC --> Catalogs --> ServerCore
    ServerCore --> Registry --> Dispatch
    Dispatch --> Content
    Dispatch --> Admin
    Dispatch --> External
    Dispatch --> ToolResult
    Registry --> Lists
    Monitor --> Health
    ServerCore --> Monitor
```

**Key storage/provider touchpoints:** AuthNZ stores identities, permissions, org/team membership, provider secrets, and tool catalog metadata. MCP runtime state, metrics, external server settings, and tool/module health live in MCP unified services. Tool execution then touches the target domain storage or provider through the dispatched module.

**Where to look in code:** `app/api/v1/endpoints/mcp_unified_endpoint.py`, `app/api/v1/endpoints/mcp_hub_management.py`, `app/api/v1/endpoints/mcp_catalogs_manage.py`, `app/core/MCP_unified/`, `app/services/admin_tool_catalog_service.py`, `Docs/MCP/`, and `Docs/MCP/Unified/`.

### Prompt Studio

**Purpose:** Manage prompt projects, prompt versions, test cases, evaluations, optimization jobs, live status, and WebSocket progress around provider-backed prompt execution.

**Primary entrypoints:** Prompt Studio project, prompt, test case, evaluation, optimization, status, and WebSocket routers under `/api/v1/prompt-studio`.

```mermaid
flowchart LR
    subgraph Routes["Prompt Studio routes"]
        Projects[Projects]
        Prompts[Prompts and versions]
        Cases[Test cases]
        Eval[Evaluations]
        Opt[Optimization]
        Status[Status and WebSocket]
    end

    subgraph Services["prompt_studio core"]
        DBDep[Prompt Studio DB dependency]
        Executor[Prompt executor]
        TestRunner[Test runner]
        Optimizer[Optimization strategies]
        JobsAdapter[Jobs adapter]
    end

    subgraph Providers
        LLM[LLM provider calls]
        Jobs[Core Jobs backend and worker]
    end

    subgraph Storage
        PromptDB["USER_DB_BASE_DIR/<user_id>/prompt_studio_dbs/prompt_studio.db"]
        Results[Test, evaluation, optimization results]
    end

    Projects --> DBDep
    Prompts --> DBDep
    Cases --> DBDep
    Eval --> TestRunner
    Opt --> Optimizer
    Status --> JobsAdapter
    DBDep --> PromptDB
    TestRunner --> Executor --> LLM
    Optimizer --> TestRunner
    Eval --> JobsAdapter --> Jobs
    Opt --> JobsAdapter --> Jobs
    Jobs --> Results --> PromptDB
```

**Key storage/provider touchpoints:** Prompt Studio persists projects, signatures, prompts, versions, test cases, evaluation runs, optimization runs, and job metadata in the per-user Prompt Studio DB. Prompt execution and optimization call LLM providers through the existing provider layer. Longer evaluation, generation, and optimization work can run through the core Jobs backend and prompt-studio worker.

**Where to look in code:** `app/api/v1/endpoints/prompt_studio/`, `app/api/v1/API_Deps/prompt_studio_deps.py`, `app/core/Prompt_Management/prompt_studio/`, `app/core/Prompt_Management/prompt_studio/services/jobs_worker.py`, and `Docs/Code_Documentation/Database.md`.

### Notes And Chatbooks

**Purpose:** Store notes and graph links, support web clipper style captures, and export/import portable Chatbooks that can include notes, conversations, characters, and related artifacts.

**Primary entrypoints:** `/api/v1/notes`, `/api/v1/notes/graph`, web clipper/capture paths where enabled, `/api/v1/chatbooks/export`, `/api/v1/chatbooks/import`, preview, continuation, download, and export/import job status routes.

```mermaid
flowchart LR
    subgraph NotesRoutes["Notes routes"]
        Notes[Notes CRUD and search]
        Graph[Graph and links]
        Clip[Web clipper or captured content]
    end

    subgraph ChatbookRoutes["Chatbook routes"]
        Export[Export selection]
        Import[Import ZIP]
        Preview[Preview and continuation]
        Jobs[Export/import jobs]
    end

    subgraph Core["Core services"]
        NotesCore[Notes service]
        GraphCore[Notes graph service]
        ChatbookSvc[ChatbookService]
        Validator[ChatbookValidator and quotas]
    end

    subgraph Storage
        ChaCha[Per-user ChaChaNotes DB]
        Temp[Per-user chatbooks temp]
        Archives[Generated chatbook archives]
        Audit[Audit and metrics]
    end

    Notes --> NotesCore --> ChaCha
    Graph --> GraphCore --> ChaCha
    Clip --> NotesCore
    Export --> Validator --> ChatbookSvc
    Import --> Validator --> ChatbookSvc
    Preview --> ChatbookSvc
    ChatbookSvc --> ChaCha
    ChatbookSvc --> Temp
    ChatbookSvc --> Archives
    Jobs --> ChatbookSvc
    ChatbookSvc --> Audit
```

**Key storage/provider touchpoints:** Notes, graph edges, chats, and characters are stored in the per-user ChaChaNotes DB. Chatbook import/export uses per-user chatbook temp/export directories, validates archive content, tracks quotas/jobs, and writes audit/metrics events. Generated archives are returned through job-backed download metadata.

**Where to look in code:** `app/api/v1/endpoints/notes.py`, `app/api/v1/endpoints/notes_graph.py`, `app/api/v1/endpoints/chatbooks.py`, `app/core/Notes/`, `app/core/Notes_Graph/`, `app/core/WebClipper/`, `app/core/Chatbooks/`, and `app/core/DB_Management/ChaChaNotes_DB.py`.

### Research And Web Scraping

**Purpose:** Search papers, perform multi-provider web search, scrape web content, run deeper research sessions, and hand useful results to ingestion, Media DB, or RAG-ready storage.

**Primary entrypoints:** `/api/v1/research/websearch`, preferred `/api/v1/paper-search/*` routes, deprecated research shims, `/api/v1/research/runs`, web scraping service/job/progress routes, media web scraping process routes, and optional ingestion handoff routes.

```mermaid
flowchart LR
    subgraph Routes["Research and scrape routes"]
        Paper[Paper search]
        WebSearch[Web search and aggregation]
        Scrape[Web scraping service]
        Deep[Deep research runs]
        Process[Media web scrape process]
    end

    subgraph Sources["External sources"]
        PaperSrc[arXiv, Semantic Scholar, PubMed, OSF, Zenodo]
        SearchSrc[Searx, Tavily, Serper, Google-like providers]
        Web[Target web pages and feeds]
        LLM[LLM aggregation and relevance calls]
    end

    subgraph Core["Research core"]
        Normalize[Normalize and rank results]
        Policy[Egress, robots, rate, dedupe policy]
        Extract[Article extraction and scraping]
        Bundle[Research bundles and artifacts]
    end

    subgraph Handoff["Persistence and handoff"]
        Ingest[Ingestion handoff]
        MediaDB[Per-user Media DB]
        RAG[RAG/Search availability]
        Outputs[Research outputs/artifacts]
    end

    Paper --> PaperSrc --> Normalize
    WebSearch --> SearchSrc --> Normalize
    WebSearch --> LLM
    Scrape --> Policy --> Web --> Extract
    Deep --> Bundle
    Process --> Extract
    Normalize --> Ingest
    Extract --> Ingest
    Bundle --> Outputs
    Ingest --> MediaDB --> RAG
```

**Key storage/provider touchpoints:** Paper and web providers are external and may require API keys or configured endpoints. Scraping applies outbound/robots/rate/dedupe policy before extraction. Ingestion handoff writes normalized content to the per-user Media DB and can make content available for FTS, embeddings, and RAG; deep research can also produce allowlisted output artifacts.

**Where to look in code:** `app/api/v1/endpoints/research.py`, `app/api/v1/endpoints/research_runs.py`, `app/api/v1/endpoints/paper_search.py`, `app/api/v1/endpoints/web_scraping.py`, `app/api/v1/endpoints/media/process_web_scraping.py`, `app/core/Search_and_Research/README.md`, `app/core/Web_Scraping/`, `app/core/WebSearch/`, and `app/core/Research/`.

### Storage, Files, And Outputs

**Purpose:** Track generated files, user folders, trash, downloads, quotas, output templates, and generated artifacts consistently across features.

**Primary entrypoints:** `/api/v1/storage/files`, storage usage, folders, trash, download routes, admin storage quota routes, `/api/v1/outputs`, output template routes, and feature-specific generated file registration helpers.

```mermaid
flowchart LR
    subgraph Routes["Storage and output routes"]
        Files[User files]
        Folders[Virtual folders]
        Trash[Trash and restore]
        Download[Download and signed access]
        Usage[Usage and quotas]
        Outputs[Outputs and templates]
    end

    subgraph Services["Storage services"]
        Quota[StorageQuotaService]
        Repo[Generated files repo]
        Helpers[Generated file helpers]
        Guard[Storage quota guard]
    end

    subgraph Producers["Artifact producers"]
        TTS[TTS and voice clones]
        Chatbooks[Chatbooks]
        Research[Research artifacts]
        Media[Media and ingestion outputs]
    end

    subgraph Storage
        FileStore["USER_DB_BASE_DIR/<user_id>/outputs and voices"]
        Metadata[Generated files metadata]
        Templates[Output templates]
        Quotas[User, team, org quotas]
    end

    Producers --> Helpers --> Quota
    Files --> Repo
    Folders --> Repo
    Trash --> Repo
    Download --> Repo
    Usage --> Quota
    Outputs --> Templates
    Guard --> Quota
    Quota --> Repo --> Metadata
    Repo --> FileStore
    Quota --> Quotas
```

**Key storage/provider touchpoints:** Generated file metadata, access times, soft delete state, folders, and quota accounting are stored through AuthNZ/generated-file repositories and storage services, while bytes live under per-user outputs/voices or feature-specific directories. Download routes verify ownership and path containment; signed download behavior is documented for job-backed/generated artifacts where the feature exposes expiring download URLs.

**Where to look in code:** `app/api/v1/endpoints/storage.py`, `storage_user_files.py`, `storage_user_folders.py`, `storage_trash.py`, `storage_usage.py`, `storage_download.py`, `storage_admin_quotas.py`, `outputs.py`, `outputs_templates.py`, `app/services/storage_quota_service.py`, `app/api/v1/API_Deps/storage_quota_guard.py`, and `app/core/Storage/`.

### Admin, Ops, And Governance

**Purpose:** Centralize operator controls for users, RBAC, monitoring, audit, orgs, billing, config, jobs, resource limits, usage, and operational safety surfaces.

**Primary entrypoints:** Admin route group under `/api/v1/admin/*`, jobs admin routes, config admin routes, monitoring/metrics/audit routes, org/team/billing/privilege routes, resource governor and quota routes, MCP catalog/hub admin routes, and startup/system diagnostics.

```mermaid
flowchart LR
    subgraph AdminRoutes["Admin routes"]
        Users[Users, sessions, MFA, API keys]
        RBAC[RBAC, privileges, orgs, billing]
        Ops[Monitoring, metrics, audit, system]
        Config[Config admin and profiles]
        JobsAdmin[Jobs admin]
        Governor[Resource governor and quotas]
    end

    subgraph Deps["Admin dependencies"]
        Role[RequireRole admin]
        Perm[Permission and scope guards]
        Rate[Rate limits]
        AuditDep[Audit context]
    end

    subgraph Core["Admin core services"]
        AuthNZ[AuthNZ services and repos]
        Metrics[Metrics manager]
        Jobs[JobManager and RLS/domain controls]
        ConfigSvc[Config/profile stores]
        Governance[Moderation, resource, policy services]
    end

    subgraph Storage
        AuthDB[AuthNZ users, roles, orgs, billing, BYOK]
        Usage[Usage, audit, metrics, quotas]
        JobsDB[Jobs DB or archive]
        ConfigFiles[Config files and snapshots]
    end

    AdminRoutes --> Role --> Perm --> Rate --> AuditDep
    AuditDep --> AuthNZ
    Users --> AuthNZ --> AuthDB
    RBAC --> AuthNZ
    Ops --> Metrics --> Usage
    Config --> ConfigSvc --> ConfigFiles
    JobsAdmin --> Jobs --> JobsDB
    Governor --> Governance --> Usage
    AuditDep --> Usage
```

**Key storage/provider touchpoints:** Admin surfaces primarily touch shared AuthNZ/usage/audit storage, org/team/billing/privilege tables, config snapshots/files, resource governor quota state, and Jobs persistence/archive state. Domain-scoped admin controls may apply RBAC and RLS context before listing, mutating, or sweeping operational records.

**Where to look in code:** `app/api/v1/endpoints/admin/`, `app/api/v1/endpoints/jobs_admin.py`, `app/api/v1/endpoints/config_admin.py`, `app/core/AuthNZ/`, `app/core/Jobs/`, `app/core/Metrics/`, `app/core/Moderation/`, `app/services/*admin*`, and `Docs/API-related/User_Registration_API_Documentation.md`.

### Characters And Workspaces

**Purpose:** Manage character cards, character chat sessions/messages/memory, workspace sources/artifacts/notes, workspace migrations, and their handoff into chat and LLM generation.

**Primary entrypoints:** Character endpoints, character session/message/memory routes, workspace CRUD, workspace sources/artifacts/notes/capabilities/status routes, workspace migration session/chunk/finalize/client-delete-ack routes, and prototype workspace/session routes.

```mermaid
flowchart LR
    subgraph Routes["Character and workspace routes"]
        Characters[Character CRUD and cards]
        Sessions[Character sessions/messages/memory]
        Workspaces[Workspaces]
        Sources[Workspace sources, artifacts, notes]
        Migrations[Workspace migrations]
        Prototype[Prototype workspaces and branch sessions]
    end

    subgraph Core["Core services"]
        CharCore[Character_Chat modules]
        WorkspaceCore[Workspace capability and DB helpers]
        MigrationCore[Migration session and chunk protocol]
        ProtoCore[Prototype workspace orchestration]
        ChatHandoff[Chat orchestration handoff]
    end

    subgraph Storage
        ChaCha[Per-user ChaChaNotes DB]
        MigrationTables[Workspace migration sessions and chunks]
        AuthDB[AuthNZ prototype workspace repos]
        Jobs[Jobs for branch/source bootstrap]
    end

    subgraph Providers
        LLM[LLM providers]
        RAG[RAG context from workspace sources]
    end

    Characters --> CharCore --> ChaCha
    Sessions --> CharCore --> ChaCha
    Workspaces --> WorkspaceCore --> ChaCha
    Sources --> WorkspaceCore --> ChaCha
    Migrations --> MigrationCore --> MigrationTables --> ChaCha
    MigrationCore --> WorkspaceCore
    Prototype --> ProtoCore --> AuthDB
    Prototype --> Jobs
    CharCore --> ChatHandoff
    WorkspaceCore --> ChatHandoff
    ChatHandoff --> RAG
    ChatHandoff --> LLM
```

**Key storage/provider touchpoints:** Characters, sessions, messages, memories, workspaces, workspace sources, artifacts, notes, and workspace migration records live primarily in the per-user ChaChaNotes DB. Workspace migrations create or reuse a target workspace, record migration sessions and declared chunks, accept idempotent chunk receipts, finalize only after all chunks are present, and track client legacy-delete acknowledgement state. Prototype workspace collaboration uses AuthNZ repository storage and Jobs for branch/session bootstrap. Character and workspace context can be passed to chat orchestration, which then calls RAG and LLM providers.

**Where to look in code:** `app/api/v1/endpoints/characters_endpoint.py`, `app/api/v1/endpoints/workspaces.py`, `app/api/v1/endpoints/workspace_migrations.py`, `app/api/v1/endpoints/prototype_workspaces.py`, `app/core/Character_Chat/`, `app/core/Workspaces/`, `app/core/Prototype_Workspaces/`, workspace migration schema/methods in `app/core/DB_Management/ChaChaNotes_DB.py`, and chat orchestration modules.

### Integrations And Connectors

**Purpose:** Connect external systems, ingestion sources, and chat/meeting integrations to the same ingestion, research, Jobs, AuthNZ, and provider-secret paths used by internal workflows.

**Primary entrypoints:** `/api/v1/connectors`, `/api/v1/ingestion-sources`, Slack events/commands/OAuth/admin routes, Discord routes/OAuth/admin helpers, Telegram admin/webhook routes, meetings routes, and optional connector or integration routers gated by configuration/dependencies.

```mermaid
flowchart LR
    subgraph Routes["Integration routes"]
        Connectors[Connectors and OAuth]
        Sources[Ingestion sources and sync]
        ChatOps[Slack, Discord, Telegram]
        Meetings[Meetings]
        Optional[Optional gated routes]
    end

    subgraph AuthConfig["Auth, secrets, and gating"]
        Auth[User, org, team identity]
        Secrets[Provider secrets and installs]
        Gates[Feature/config gates]
        Verify[Webhook signatures and policies]
    end

    subgraph Work["Processing path"]
        Queue[Connector or ingestion Jobs]
        Normalize[Normalize external payloads]
        Ingest[Ingestion handoff]
        Research[Research/search handoff]
        Chat[Chat/LLM handoff]
    end

    subgraph StorageProviders["Storage and providers"]
        AuthDB[AuthNZ secrets, installs, approvals]
        MediaDB[Media DB]
        NotesDB[ChaChaNotes]
        External[External APIs and webhooks]
    end

    Connectors --> Auth
    Sources --> Auth
    ChatOps --> Verify
    Meetings --> Auth
    Optional --> Gates
    Auth --> Secrets --> AuthDB
    Verify --> Secrets
    Connectors --> External
    ChatOps --> External
    Sources --> Queue
    Connectors --> Queue
    Queue --> Normalize
    Normalize --> Ingest --> MediaDB
    Normalize --> Research
    Normalize --> Chat
    Chat --> NotesDB
```

**Key storage/provider touchpoints:** AuthNZ stores user/org/team identities, provider secrets, OAuth installs, linked actors, approvals, and connector metadata. Ingestion-source syncs and connector jobs enqueue work, normalize external payloads, and hand content to ingestion/Media DB, notes, research, or chat/LLM providers. Optional routes may be gated by config, dependency availability, or explicit feature flags.

**Where to look in code:** `app/api/v1/endpoints/connectors.py`, `app/api/v1/endpoints/ingestion_sources.py`, Slack/Discord/Telegram endpoint and support files, `app/api/v1/endpoints/meetings.py`, `app/core/Ingestion_Sources/`, connector services under `app/core/External_Sources/` where present, provider-secret repos under `app/core/AuthNZ/`, and media/research ingestion handoff modules.

## Router Coverage Matrix

This table groups routers by the way they are registered and maintained, not by every concrete endpoint path. Use it to audit whether new router domains have a corresponding atlas entry and whether related diagrams were updated together.

| Router group or domain | Representative routes/modules | Atlas section | Coverage note |
| --- | --- | --- | --- |
| Core/infrastructure | `main.py`, `router_registry.py`, `router_groups/core.py`, `router_groups/minimal.py`, setup, health, metrics, OpenAPI helpers | [System Context](#system-context), [Request Lifecycle](#request-lifecycle), [Router Group Map](#router-group-map) | Covers app startup, router registration, middleware/dependencies, and operational surfaces. Individual health/setup variants are grouped under infrastructure rather than listed one by one. |
| Identity/config/sync | `auth.py`, `users.py`, `config_info.py`, `config_admin.py`, `sync.py`, AuthNZ dependencies, provider-secret helpers | [Auth And User Context](#auth-and-user-context), [Admin, Ops, And Governance](#admin-ops-and-governance) | Groups identity, user context, configuration, sync, and provider-secret flows because they share AuthNZ/user-scope ownership. |
| Chat/LLM | `chat.py`, OpenAI-compatible chat routes, `core/Chat/`, `core/LLM_Calls/`, provider routing | [Chat And LLM Provider Calls](#chat-and-llm-provider-calls) | Covers request shaping, optional RAG context, conversation persistence, provider selection, and streaming responses. Character-specific chat is cross-linked through the characters/workspaces row. |
| ACP/MCP | `mcp_unified_endpoint.py`, ACP endpoints where enabled, `core/MCP_unified/` | [MCP Unified](#mcp-unified), [Router Group Map](#router-group-map) | Groups MCP and ACP-style tool/client protocols as external tool-control surfaces with shared auth, RBAC, and execution concerns. |
| Content/RAG/media/audio/embeddings/evaluations/OCR | `media/` endpoint package, `media_embeddings.py`, `rag_unified.py`, `rag_health.py`, `audio/` endpoint package, `embeddings_*`, `evaluations_unified.py`, `ocr.py`, ingestion/chunking/embedding/RAG/evaluation core modules | [Media Ingestion](#media-ingestion), [Audio STT/TTS](#audio-stttts), [Chunking And Embeddings](#chunking-and-embeddings), [RAG/Search](#ragsearch), [Evaluations](#evaluations) | Groups high-volume content processing domains that move data between uploads/providers, Media DB, vector stores, per-user evaluations storage, and response-first audio paths. |
| Workflows/scheduler/jobs | `workflows.py`, jobs endpoints, Scheduler handlers, APScheduler bridges, WorkerSDK/background services | [Jobs And Scheduler](#jobs-and-scheduler), [Admin, Ops, And Governance](#admin-ops-and-governance) | Separates user-visible Jobs from internal Scheduler orchestration while showing where recurring schedules enqueue into each backend. |
| Notes/prompts/prompt studio/workspaces/characters | notes/chatbook endpoints, prompt endpoints, `prompt_studio/` endpoint package, workspace routes including migrations, character endpoints and card/session helpers | [Prompt Studio](#prompt-studio), [Notes And Chatbooks](#notes-and-chatbooks), [Characters And Workspaces](#characters-and-workspaces) | Groups user-authored knowledge, conversation artifacts, prompt assets, workspace migration state, and character/session data because they primarily persist through ChaChaNotes and related per-user stores. |
| Collections/reading and learning tools | `collections_feeds.py`, `collections_websub.py`, `reading.py`, `reading_highlights.py`, `translate.py`, `slides.py`, `flashcards.py`, `quizzes.py`, `study_suggestions.py`, writing/manuscript routes | [Media Ingestion](#media-ingestion), [Notes And Chatbooks](#notes-and-chatbooks), [Characters And Workspaces](#characters-and-workspaces) | Explicitly groups registered lightweight content and learning routers that organize, annotate, transform, or study content. The atlas covers their storage/provider handoffs at the domain level, not every route. |
| Application content tools | Kanban modules, `data_tables.py`, `items.py`, `reminders.py`, `notifications.py`, `watchlists.py`, scheduled tasks control plane, VN assets/play routes | [Storage, Files, And Outputs](#storage-files-and-outputs), [Jobs And Scheduler](#jobs-and-scheduler), [Admin, Ops, And Governance](#admin-ops-and-governance) | Covers app-level boards, data tables, items, tasks/reminders, notifications, watchlists, scheduled tasks, and VN routes as registered content/application surfaces with storage, scheduling, and governance touchpoints. |
| Persona/companion personalization | `persona.py`, `personalization.py`, `companion.py`, `archetype_endpoints.py`, voice assistant routes | [Chat And LLM Provider Calls](#chat-and-llm-provider-calls), [Characters And Workspaces](#characters-and-workspaces), [Audio STT/TTS](#audio-stttts) | Groups persona, companion, archetype, personalization, and voice assistant routes because they bridge user profile state, character/workspace context, chat/LLM calls, and audio streams. |
| Storage/files/outputs/sharing | file upload/download routes, outputs/artifacts, local storage helpers, sharing/export/import handlers, chatbooks | [Storage, Files, And Outputs](#storage-files-and-outputs), [Notes And Chatbooks](#notes-and-chatbooks), [Data Store Map](#data-store-map) | Covers storage ownership and file/output lifecycles at the domain level; concrete file routes are intentionally summarized by storage responsibility. |
| Research/web scraping/connectors/integrations | `research.py`, `paper_search.py`, `web_scraping.py`, connectors, ingestion sources, Slack/Discord/Telegram/meeting routes where enabled | [Research And Web Scraping](#research-and-web-scraping), [Integrations And Connectors](#integrations-and-connectors) | Groups external-source ingestion and integration callbacks because both normalize provider/web payloads before handing off to media, notes, research, chat, or jobs. |
| Admin/orgs/billing/resource governance/monitoring | admin routers, org/team routes, billing/subscription routes, resource governance, rate limits, metrics, audit/ops endpoints | [Admin, Ops, And Governance](#admin-ops-and-governance), [Request Lifecycle](#request-lifecycle) | Covers governance, quotas, RBAC, observability, and administrative controls as cross-cutting policy layers rather than feature-specific endpoint lists. |

## How To Update This Atlas

- Check `router_groups/*.py` and `router_registry.py` for router additions, removals, lazy imports, optional routes, or registration changes.
- Check changed endpoint and core modules for new storage ownership, provider calls, background workers, queue paths, or persistence gates.
- Update the relevant Mermaid diagram and the [Router Coverage Matrix](#router-coverage-matrix) together so coverage remains auditable.
- Re-run Markdown/Mermaid text checks for changed headings, diagram syntax anchors, and required terms.
- Record verification commands and results in the relevant Backlog task.
