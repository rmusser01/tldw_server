# C4 level 3 — FastAPI server components

The [system context and container views](./c4.md) show how clients and services
connect. This view opens the **FastAPI server** container and follows the
components that handle a request, access data, or dispatch background work.
The [architecture index](./README.md) links to the remaining subsystems.

```mermaid
flowchart TB
    clients["Containers and systems · API clients<br/>Browser apps, API clients and MCP clients"]

    subgraph server["FastAPI server container"]
        routes["Component · Request edge and routers<br/>FastAPI and Pydantic<br/>Validate requests and shape responses"]
        auth["Component · AuthNZ and governance<br/>FastAPI dependencies and ASGI middleware<br/>Resolve users and enforce access policies"]
        media["Component · Media and audio<br/>Python, FFmpeg and speech adapters<br/>Extract, chunk and synthesize content"]
        chat["Component · Chat orchestration<br/>Python and SSE<br/>Build prompts and manage conversations"]
        rag["Component · Retrieval and embeddings<br/>Python retrieval pipeline<br/>Index, search and rank content"]
        mcp["Component · MCP Unified<br/>Python JSON-RPC server<br/>Dispatch tools from the module registry"]
        llm["Component · LLM gateway<br/>Python provider adapters<br/>Resolve and route model calls"]
        persistence["Component · Persistence adapters<br/>Python and SQL<br/>Resolve stores and owner scope"]
        background["Component · Background execution<br/>Jobs, Scheduler and APScheduler<br/>Queue work and run enabled pollers"]
    end

    data[("Data stores · Local data<br/>SQLite, embedded ChromaDB and files")]
    providers["External systems · Model providers<br/>Hosted or self-hosted inference endpoints"]

    clients -->|"Call HTTP or WebSocket APIs"| routes
    routes -->|"Resolve principal and apply controls"| auth
    routes -->|"Process media or speech"| media
    routes -->|"Handle completions"| chat
    routes -->|"Serve search and embeddings"| rag
    routes -->|"Handle MCP tool requests"| mcp
    routes -->|"Enqueue jobs and read status"| background

    media -->|"Stores media"| persistence
    media -.->|"Request indexing when enabled"| rag
    chat -->|"Persists conversations"| persistence
    chat -->|"Send completion requests"| llm
    mcp -->|"Run retrieval tools"| rag
    rag -->|"Reads lexical indexes"| persistence
    rag -->|"Read and write embedded vectors"| data
    rag -.->|"Request generation or query rewriting"| llm
    llm -->|"Call configured endpoints over HTTP"| providers

    auth -->|"Persists identity"| persistence
    background -->|"Persist queue state"| persistence
    persistence -->|"Open local databases and files"| data
```

Boxes inside the FastAPI boundary are in-process components. Solid arrows show
the main interaction paths. Dotted arrows show paths that depend on the feature
or deployment configuration. SQLite and ChromaDB use embedded clients with
filesystem storage. The [container view](./c4.md#c4-level-2--containers-and-data-stores) shows optional network stores.

AuthNZ accepts API keys and browser sessions in single-user mode and credentials
such as JWTs and sessions in multi-user mode. Recurring APScheduler services submit to Jobs or
to the internal Scheduler according to the feature. Optional Jobs worker
processes lease work from the same SQL-backed queue as the API. [Jobs and Scheduler decision](../ADR/003-jobs-vs-scheduler-default.md)

## Start in the code

| Component | Source entry point | Deeper guide |
| --- | --- | --- |
| Request edge and routers | [FastAPI entry point](../../tldw_Server_API/app/main.py), [router registry](../../tldw_Server_API/app/api/v1/router_registry.py) | [App shell](./app-shell.md) |
| AuthNZ and governance | [Auth dependencies](../../tldw_Server_API/app/api/v1/API_Deps/auth_deps.py), [ingress governor](../../tldw_Server_API/app/core/Resource_Governance/middleware_simple.py) | [AuthNZ](./authnz.md), [governance](./governance-billing.md) |
| Media and audio | [Media routes](../../tldw_Server_API/app/api/v1/endpoints/media/), [audio routes](../../tldw_Server_API/app/api/v1/endpoints/audio/audio.py) | [Ingestion](./ingestion.md), [audio](./audio.md) |
| Chat orchestration | [Chat routes](../../tldw_Server_API/app/api/v1/endpoints/chat.py), [completion pipeline](../../tldw_Server_API/app/core/Chat/completion_pipeline.py) | [Chat and characters](./chat-character.md) |
| Retrieval and embeddings | [Unified RAG pipeline](../../tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py), [embeddings core](../../tldw_Server_API/app/core/Embeddings/) | [RAG](./rag.md), [chunking and embeddings](./chunking-embeddings.md) |
| MCP Unified | [MCP routes](../../tldw_Server_API/app/api/v1/endpoints/mcp_unified_endpoint.py), [embedded server](../../tldw_Server_API/app/core/MCP_unified/server.py) | [MCP Unified](./mcp-unified.md) |
| LLM gateway | [LLM call entry point](../../tldw_Server_API/app/core/LLM_Calls/gateway.py) | [LLM providers](./llm-providers.md) |
| Persistence adapters | [Database paths](../../tldw_Server_API/app/core/DB_Management/db_path_utils.py), [content backend](../../tldw_Server_API/app/core/DB_Management/content_backend.py), [AuthNZ database](../../tldw_Server_API/app/core/AuthNZ/database.py) | [Database layer](./database-layer.md) |
| Background execution | [Lifespan startup](../../tldw_Server_API/app/services/lifespan_startup_sequence.py), [JobManager](../../tldw_Server_API/app/core/Jobs/manager.py) | [Jobs and background work](./jobs-background.md) |

## Trace a grounded chat reply

1. The WebUI RAG mode calls `POST /api/v1/rag/search` with generation disabled. The route resolves the user and their data sources. The RAG pipeline retrieves and ranks matching content. [RAG route](../../tldw_Server_API/app/api/v1/endpoints/rag_unified.py)
2. When it has evidence to use, the WebUI formats the retrieved passages into the prompt and sends a separate `POST /api/v1/chat/completions` request. [WebUI RAG mode](../../apps/packages/ui/src/hooks/chat-modes/ragMode.ts)
3. Chat assembles the completion request and calls the configured model through the LLM gateway. Streaming replies return as SSE. [Chat service](../../tldw_Server_API/app/core/Chat/chat_service.py)
