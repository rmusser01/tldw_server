# Architecture docs

This folder is the **living internals reference** for tldw_server: one document per subsystem, describing how it actually works — authoritative files, dataflows, boundaries, failure behavior, and verified gotchas — anchored to the current code rather than to point-in-time plans. It follows the same conventions as the chatbook's `Docs/Architecture/` reference.

It complements the other doc families:

| Family | Question it answers |
| --- | --- |
| `Docs/Architecture/` (this folder) | How does it work inside, and where is the code? |
| `Docs/Code_Documentation/` | Per-module developer guides (varying freshness — see the cross-link notes below) |
| `Docs/ADR/` | Why was it decided this way? (index in `Docs/ADR/README.md`) |
| `Docs/Design/` | Design exploration and collections (untouched by this reference; separately maintained) |
| `Docs/superpowers/`, `Docs/Plans/` | Point-in-time specs, plans, and evidence |

## Index

### Spine

- [app-shell.md](./app-shell.md) — the FastAPI app, lifespan sequences, declarative router registry and route gating, v1/v2 versioning, the middleware ordering contract, config resolution, deployment surface
- [database-layer.md](./database-layer.md) — the backend abstraction (SQLite/PostgreSQL), the three independent selection axes, per-user paths, migrations, RLS
- [jobs-background.md](./jobs-background.md) — the Jobs substrate (leases, fencing tokens, fair share, quarantine), worker pollers and sidecar mode, the internal Scheduler, Jobs-vs-Scheduler rule, scheduling/automations/reminders/notifications

### Identity and access

- [authnz.md](./authnz.md) — dual-mode auth (single-user key vs multi-user JWT+sessions+MFA), claim-first principals, API keys, the admin-services pattern, billing-as-limits, BYOK, governance/audit cores

### Content pipelines

- [ingestion.md](./ingestion.md) — `/media/add` orchestration, upload/SSRF gates, persistent ingestion sources, OAuth external connectors, the persistence seam, embeddings dispatch
- [chunking-embeddings.md](./chunking-embeddings.md) — the strategy-based chunker (and the three-passes trap), the embeddings engine, per-user ChromaDB, the Jobs/Redis staged pipeline
- [rag.md](./rag.md) — the unified retrieval pipeline, hybrid RRF fusion, cross-source ordering, reranking, guarded generation, deep-search webresearch

### Conversation surfaces

- [llm-providers.md](./llm-providers.md) — the adapter registry, accounted adapter base class, BYOK and base-URL policy, SSE normalization, routing, admin overrides, the llama.cpp control plane
- [chat-character.md](./chat-character.md) — general chat (history selection, branch fencing), character sessions (streaming + separate persist), character/world-book CRUD, chatbook ZIP packs, macros

### Cross-cutting

- [notes-sync.md](./notes-sync.md) — Sync v2: capabilities/devices/datasets, envelope model and encryption policies, push/pull, batched conflicts, blobs, personal-context domains
- [research.md](./research.md) — the phased research engine on Jobs (fencing-token leases), checkpoints and review, broker lanes, the discovery catalog, artifacts
- [mcp-unified.md](./mcp-unified.md) — the embedded unified MCP server: transports, two-phase prepared tool calls, module registry with failover, prompt continuation cursors, federation status
- [evaluations.md](./evaluations.md) — unified eval service and runner, webhook delivery, recipe runs, the persona judge contract, core-only WordBench
- [audio.md](./audio.md) — STT/TTS: normalization layer, TTS v2 adapter stack and gateway fallback, long-form jobs, OpenAI-compatible transcriptions, realtime streaming

## Cross-links into `Docs/Code_Documentation/` (freshness audit)

These existing guides were spot-checked against code while writing this reference:

| Guide | Status |
| --- | --- |
| `Jobs_Manager.md`, `Jobs_Module.md` | **Current** |
| `Database-Backends.md` | Broadly current (minor value-vocabulary drift) |
| `Database.md` | Partly stale (AuthNZ default URL claim, line anchors) |
| `Code_Map.md` | Partly stale (router-registry model, no `/api/v2`, ~120 missing endpoint modules) |
| `Data_Flow_Atlas.md` | Mostly current, but presents Redis as a jobs backend — Jobs is strictly SQLite/Postgres |
| `Ingestion_Media_Processing.md`, `Ingestion_Pipeline_Video.md` | Current |
| `Ingestion_Pipeline_{Audio,PDF}.md` | Current on signatures, otherwise older |
| `Ingestion_Pipeline_{Documents,Ebooks,MediaWiki}.md` | Older; signatures still accurate |
| `Embeddings-Developer-Guide.md`, `Embeddings-Documentation.md` | Current |
| `Chunking-Module.md` | Stale-ish (predates timecode/frontmatter/propositions; claims a nonexistent template library) |
| `RAG-Developer-Guide.md` (+ `Docs/API-related/RAG_API_Documentation.md`) | Current; prefer over `Docs/RAG/` research notes |
| `AuthNZ-Developer-Guide.md` | **Stale** (Feb; predates claim-first API) — use `app/core/AuthNZ/README.md` |
| `Chat_Developer_Guide.md` | Partly stale (dispatch entrypoint wording) |
| `CHARACTER_CHAT_API_DOCUMENTATION.md` | Stale (missing `/complete-v2`, `/completions/persist`) |
| `Chatbook_Developer_Guide.md`, `Guides/Chatbooks_Code_Guide.md`, `Guides/Evaluations_Code_Guide.md` | Current (Oct) |
| `Evaluations_Developer_Guide.md` | **Stale** (Feb; missing recipes/wordbench/persona-judge) — use the in-module guides |
| `Local_LLM.md` | **Stale** — use `app/core/Local_LLM/README.md` |

In-module READMEs (`app/core/*/README.md`) were consistently current and are the preferred deep references. `Docs/API/sync-v2.md` is the living sync reference with two known stale spots (conflict-action vocabulary; a single-conflict resolve route that only exists batched).

## Conventions

- Each doc names its **authoritative files**. When code and any other doc disagree, code wins and the doc should be fixed.
- Dataflows are numbered and verified against the code; **gotchas** are invariants verified in source, not folklore.
- Decisions link to `Docs/ADR/`; per-module depth links into current `Code_Documentation` guides or module READMEs.
- These docs describe the current state. When a subsystem changes materially, update its doc in the same change.

## Not yet covered

Areas without a doc here yet (candidates): the admin-webui frontend (`admin-ui/`), the SDKs (`sdks/`), Watchlists core (feed/site monitoring — distinct from external connectors), Claims Extraction core, CodeGraph, Image Generation, Buddy, Governance deep-dive, Billing deep-dive, agent orchestration/ACP, and the mock OpenAI server. Until a doc exists here, the ADR set, module READMEs, and `Docs/Code_Documentation/` are the reference for those areas.
