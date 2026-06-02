# tldw_Server_API Data Flow Atlas Design

**Date:** 2026-06-02
**Surface:** Backend architecture and contributor documentation
**Status:** Approved in-session
**Backlog:** TASK-502

---

## Goal

Create a dedicated, newcomer- and maintainer-oriented data flow atlas for `tldw_Server_API`.

The atlas should make it easier to:

- understand how requests enter the FastAPI app and reach core modules;
- reason about how data moves between endpoints, dependencies, services, databases, vector stores, workers, and external providers;
- find the right code area when debugging cross-module behavior;
- see how major subsystems relate without reading every endpoint file first.

The implementation target is:

- `Docs/Code_Documentation/Data_Flow_Atlas.md`
- links from `Docs/Architecture.md` and `Docs/Code_Documentation/Code_Map.md`

All diagrams should be Mermaid embedded in Markdown. Do not produce PNG, SVG, or other generated image assets for the first pass.

## Audience

The primary audiences are:

1. New contributors trying to understand where to start and how requests flow.
2. Maintainers debugging behavior that crosses endpoint, core, storage, provider, and worker boundaries.

The atlas can still help operators, but deployment and configuration diagrams are not the main goal. Existing Getting Started and deployment docs remain the right place for operator-first topology material.

## Product Decision

Use a **layered atlas** instead of one oversized architecture diagram.

The atlas should start broad, then progressively reveal detail:

1. top-level system context;
2. request lifecycle;
3. router group map;
4. data store map;
5. subsystem flow diagrams;
6. code/documentation anchors.

This structure gives newcomers a top-down path while giving maintainers direct entry points for specific flows.

The implementation should be staged so the atlas remains reviewable:

1. foundation maps: system context, request lifecycle, router group map, and data-store map;
2. core flow diagrams: auth, media ingestion, audio, chunking/embeddings, RAG, chat/LLM, jobs/scheduler;
3. extended domain maps: evaluations, MCP, prompt studio, notes/chatbooks, research/web scraping, storage/files, admin/ops, characters/workspaces, integrations/connectors;
4. final linking, coverage matrix, and verification pass.

## Non-Goals

- No generated OpenAPI replacement.
- No endpoint-by-endpoint route inventory for every concrete path.
- No rendered diagram assets in the first pass.
- No implementation changes to backend code.
- No broad rewrite of existing architecture docs.
- No attempt to diagram every internal function call.
- No speculative future architecture.

The atlas should be exhaustive by **groups, domains, and flows**, not by every route handler or private helper.

## Documentation Placement

Create:

- `Docs/Code_Documentation/Data_Flow_Atlas.md`

Update:

- `Docs/Architecture.md`
- `Docs/Code_Documentation/Code_Map.md`

The new atlas should complement existing documents rather than duplicate them wholesale:

- `Docs/Architecture.md` remains the concise contributor mental model.
- `Docs/Code_Documentation/Code_Map.md` remains the compact code-location reference.
- `Docs/Getting_Started/ARCHITECTURE.md` remains the deployment/client topology guide.
- `Data_Flow_Atlas.md` becomes the detailed flow and process map.

## Diagram Conventions

The atlas should define a small legend before the diagrams:

| Shape or Group | Meaning |
| --- | --- |
| Clients | WebUI, external HTTP clients, MCP clients, browser extension, admin UI, or other callers |
| FastAPI app | `app/main.py`, router registration, middleware, lifecycle, OpenAPI surface |
| Endpoint groups | Routers under `app/api/v1/endpoints/` grouped by `router_groups/*.py` |
| API dependencies | Auth, user context, database handles, rate limits, resource governance, request validation |
| Core modules | Domain logic under `app/core/` |
| Storage | SQLite/PostgreSQL databases, ChromaDB/pgvector, file storage, Redis/job backends |
| Providers | LLM, STT, TTS, OCR, web, media, and other external/local providers |
| Workers | Jobs, Scheduler, APScheduler bridges, background services, and app lifecycle workers |
| Optional routes | Feature-gated or optional dependency routes |

Prefer:

- `flowchart` diagrams for component and data-store maps;
- `sequenceDiagram` diagrams for ordered request/process flows;
- clear labels over dense implementation detail;
- grouped subgraphs that match repository boundaries.

## Atlas Structure

### 1. How To Read This Atlas

Explain the target audience, conventions, and scope. Make clear that the atlas maps current architecture and should be checked against code before future edits.

### 2. System Context

Add one high-level Mermaid diagram showing:

- WebUI, admin UI, browser extension, HTTP clients, MCP clients;
- FastAPI app;
- router groups;
- API dependencies;
- core modules;
- storage;
- external providers;
- background workers.

This diagram should establish the system-wide mental model:

`Clients -> FastAPI -> API deps/endpoints -> core modules -> storage/providers/workers`

### 3. Request Lifecycle

Add an ordered process diagram for a typical HTTP/WebSocket request:

1. router registration from `router_groups`;
2. middleware/lifecycle readiness checks where relevant;
3. request reaches an endpoint;
4. Pydantic/schema validation;
5. auth and user context resolution;
6. rate limit/resource governance checks;
7. database/vector-store/provider dependencies;
8. core service call;
9. persistence/provider/worker interaction;
10. response, streaming response, WebSocket message, or job handle.

This diagram should help contributors reason about where cross-cutting behavior belongs.

### 4. Router Group Map

Base the router map on the current files:

- `tldw_Server_API/app/api/v1/router_registry.py`
- `tldw_Server_API/app/api/v1/router_groups/spec.py`
- `tldw_Server_API/app/api/v1/router_groups/core.py`
- `tldw_Server_API/app/api/v1/router_groups/content.py`
- `tldw_Server_API/app/api/v1/router_groups/admin.py`
- `tldw_Server_API/app/api/v1/router_groups/minimal.py`
- `tldw_Server_API/app/api/v1/router_groups/conditional.py`
- `tldw_Server_API/app/api/v1/router_groups/factories.py`

Verify the map against these concrete registration anchors:

- `include_router_idempotent`
- `register_router_specs`
- `register_all_routers`
- `RouterSpec.resolve_router`
- `append_imported_router_spec`
- `iter_core_router_specs`
- `iter_content_router_specs`
- `iter_admin_router_specs`
- `iter_minimal_test_router_specs`
- `iter_minimal_optional_router_specs`
- grouped and minimal registration calls in `tldw_Server_API/app/main.py`

The map should show major groups rather than every individual route:

- core/infrastructure;
- identity/config/sync;
- chat and LLM providers;
- ACP and MCP;
- content, RAG, media, audio, embeddings, evaluations, OCR;
- workflows, scheduler, jobs;
- notes, prompts, prompt studio, workspaces, characters;
- storage, files, outputs, sharing;
- research, web scraping, connectors, integrations;
- admin, orgs, billing, resource governance, monitoring.

When a route is optional, lazy-imported, or feature-gated, label it as such.

Include a compact router coverage table after the map:

| Router group or domain | Representative routes/modules | Atlas section | Coverage note |
| --- | --- | --- | --- |

This table should make grouping decisions explicit. It does not need every concrete path, but it should include every major router group/domain and representative modules from the current router specs.

### 5. Data Store Map

Add a data-store map showing the major persisted surfaces and their owners:

- AuthNZ DB: users, sessions, API keys, RBAC, MFA;
- per-user Media DB: media items, transcripts, chunks, metadata, FTS;
- per-user ChaChaNotes DB: notes, chats, character data;
- prompt and Prompt Studio databases;
- evaluations DB;
- vector store: ChromaDB or pgvector;
- file storage and generated artifacts;
- Redis when configured;
- Jobs DB and Scheduler persistence where applicable.

The map should distinguish shared storage from per-user storage.

### 6. Subsystem Flow Atlas

Each subsystem entry should follow this format:

1. Purpose
2. Primary entrypoints
3. Mermaid diagram
4. Key storage/provider touchpoints
5. Where to look in code

Include these flows:

- Auth and user context
- Media ingestion
- Audio STT/TTS
- Chunking and embeddings
- RAG/search
- Chat and LLM provider calls
- Evaluations
- MCP Unified
- Prompt Studio
- Notes and Chatbooks
- Jobs and Scheduler
- Research and web scraping
- Storage/files/outputs
- Admin/ops/governance
- Characters and workspaces
- Integrations and connectors

Smaller domains can be grouped into map diagrams instead of full sequence diagrams when the ordered process would not add clarity.

## Subsystem Details

### Auth And User Context

Show both major auth paths:

- single-user `X-API-KEY`;
- multi-user JWT.

The flow should include:

- auth dependency;
- `core/AuthNZ`;
- Auth DB;
- resolved user context;
- per-user storage root selection.

### Media Ingestion

Show the main ingestion branches:

- file/document input;
- URL/video/audio input;
- web scraping input;
- OCR/STT branches where applicable;
- normalization;
- chunking;
- Media DB write;
- embedding/vector-store update;
- optional job/background handling.

### Audio STT/TTS

Show:

- file transcription;
- streaming transcription over WebSocket;
- text-to-speech;
- provider/local backend selection;
- optional transcript persistence as media/searchable content;
- audio history/jobs where relevant.

### Chunking And Embeddings

Show:

- chunking endpoints and ingestion-triggered chunking;
- chunking strategies/templates;
- embedding provider/model selection;
- vector store write;
- Media DB metadata and FTS relationships;
- worker/job path for batch embeddings.

### RAG/Search

Show:

- unified RAG request;
- request normalization and settings;
- FTS/BM25 path;
- vector retrieval path;
- reranking/post-processing;
- context/result assembly;
- use from standalone RAG endpoints and chat.

### Chat And LLM Providers

Show:

- OpenAI-compatible chat entrypoint;
- optional RAG context;
- chat/session persistence;
- provider/model resolution;
- provider adapter call;
- streaming and non-streaming responses.

### Evaluations

Show:

- recipes/datasets/runs;
- evaluator services;
- RAG eval hooks;
- LLM judge/provider calls;
- metrics and results persistence.

### MCP Unified

Show:

- MCP HTTP/WebSocket entrypoints;
- auth/RBAC;
- tool/module registry;
- dispatch to domain modules;
- status/metrics/tool execution outputs.

### Jobs And Scheduler

Show both paths and label the decision point:

- Jobs for user-visible work, admin status, retries, quotas, and worker processing;
- Scheduler for internal orchestration with dependencies, task registration, and idempotency;
- APScheduler bridges that enqueue into the chosen backend.

This should align with the repository's Jobs vs Scheduler decision guide.

## Implementation Anchors

The implementation plan should inspect and use these current sources before writing final diagrams:

- `Docs/Architecture.md`
- `Docs/Getting_Started/ARCHITECTURE.md`
- `Docs/Code_Documentation/Code_Map.md`
- `tldw_Server_API/README.md`
- `tldw_Server_API/app/main.py`
- `tldw_Server_API/app/api/v1/router_groups/*.py`
- selected endpoint modules under `tldw_Server_API/app/api/v1/endpoints/`
- selected core modules under `tldw_Server_API/app/core/`
- module guides under `Docs/Code_Documentation/`
- MCP docs under `Docs/MCP/Unified/`
- audio docs under `Docs/STT-TTS/`

Use real module names, route prefixes, and storage paths from code and existing docs.

## Error Handling And Edge Cases

The atlas should show these cross-cutting behaviors where they materially affect flow:

- optional route import and feature gating;
- auth failure and missing permission paths;
- rate-limit/resource-governance blocking;
- provider/backend failure;
- job enqueue versus synchronous processing;
- streaming/WebSocket disconnects;
- per-user storage isolation;
- shared versus per-user databases.

Do not turn every diagram into an error-state diagram. Add error/edge labels only where they explain control flow.

## Verification

Implementation verification should be documentation-focused:

- confirm the new doc exists at `Docs/Code_Documentation/Data_Flow_Atlas.md`;
- confirm `Docs/Architecture.md` links to it;
- confirm `Docs/Code_Documentation/Code_Map.md` links to it;
- run a text check for Mermaid fences and obvious Markdown link problems;
- render-check Mermaid when a local renderer is already available, without committing generated PNG/SVG assets;
- grep for key source anchors to ensure diagrams cite real module paths;
- manually compare the router map against `router_groups/*.py`;
- confirm the router coverage table accounts for the major router groups/domains called out in this spec;
- record verification results in `TASK-502`.

No backend unit tests are required unless implementation uncovers a code/docs mismatch that requires code changes.

Bandit is not required for documentation-only changes. The Backlog task should record this as a non-code skip.

## Maintenance Guidance

Keep the atlas maintainable:

- prefer grouped diagrams over huge diagrams;
- keep diagram labels stable and code-grounded;
- update the atlas when router groups, storage ownership, or major process flows change;
- link to deeper module docs rather than duplicating all module internals;
- avoid speculative architecture language unless clearly labeled as future work.
- include a brief "How to update this atlas" checklist at the end of the final doc.

## Implementation Decisions

Use these decisions when writing the implementation plan:

- `Data_Flow_Atlas.md` should be one long page with a table of contents.
- Admin/ops should use one primary map plus smaller diagrams only when needed for clarity.
- The router group map should include domain buckets, and the router coverage table should list representative router spec names/modules so maintainers can audit coverage without reading a huge diagram.
- The final doc should include a short "How to update this atlas" checklist.
