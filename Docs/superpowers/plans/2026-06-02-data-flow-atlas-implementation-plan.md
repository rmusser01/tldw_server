# tldw_Server_API Data Flow Atlas Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `Docs/Code_Documentation/Data_Flow_Atlas.md`, a Mermaid-only backend data flow atlas for newcomers and maintainers.

**Architecture:** Add one dedicated atlas document, then link it from the existing architecture and code-map docs. Keep the atlas layered: foundation maps first, core flows second, extended domain maps third, then coverage and verification. The implementation is documentation-only and must stay grounded in current router registration, endpoint modules, core modules, storage docs, and subsystem guides.

**Tech Stack:** Markdown, Mermaid flowcharts, Mermaid sequence diagrams, `rg`, shell text checks, optional Mermaid renderer when locally available.

---

## Source Spec And Tracking

- Approved spec: `Docs/superpowers/specs/2026-06-02-data-flow-atlas-design.md`
- Atlas implementation and verification task: `TASK-502`
- Implementation planning task: `TASK-503` (planning artifact only)
- Implementation target doc: `Docs/Code_Documentation/Data_Flow_Atlas.md`

## File Structure

Create:

- `Docs/Code_Documentation/Data_Flow_Atlas.md`
  - Owns all atlas content: legend, system context, request lifecycle, router group map, data-store map, subsystem flow diagrams, router coverage matrix, and update checklist.

Modify:

- `Docs/Architecture.md`
  - Add a concise link to the atlas near existing visual/code-map references.
- `Docs/Code_Documentation/Code_Map.md`
  - Add a concise link to the atlas near high-level architecture/key flows.
- `backlog/tasks/task-503 - Plan-tldw-Server-API-data-flow-atlas-implementation.md`
  - Record plan verification and handoff notes.
- `backlog/tasks/task-502 - Design-tldw-Server-API-data-flow-atlas.md`
  - Record atlas implementation touched files, verification results, docs-only Bandit skip, and final summary.

Do not modify backend code.

## Stage 0: Preflight And Worktree Hygiene

**Goal:** Confirm the implementation worker has the approved plan/spec, uses `TASK-502` for implementation tracking, and has an isolated staged file set.

**Files:**
- Read: `Docs/superpowers/specs/2026-06-02-data-flow-atlas-design.md`
- Read: `Docs/superpowers/plans/2026-06-02-data-flow-atlas-implementation-plan.md`
- Update: `backlog/tasks/task-502 - Design-tldw-Server-API-data-flow-atlas.md`

- [ ] **Step 1: Read the approved design spec**

Run:

```bash
sed -n '1,440p' Docs/superpowers/specs/2026-06-02-data-flow-atlas-design.md
```

Expected: the spec describes the layered atlas, phased delivery, router coverage table, concrete router registration anchors, and verification requirements.

- [ ] **Step 2: Check working tree state**

Run:

```bash
git status --short
```

Expected: unrelated dirty files may exist. Do not revert or stage unrelated changes.

- [ ] **Step 3: Confirm Backlog tracking task**

Use the Backlog MCP workflow. View `TASK-502`:

```bash
backlog task TASK-502 --plain
```

Expected: `TASK-502` exists and remains the authoritative task for atlas implementation verification. Do not create a duplicate implementation task. `TASK-503` is only for this plan document.

- [ ] **Step 4: Commit checkpoint**

No commit is required for read-only preflight unless a new Backlog task file was created.

If `TASK-502` metadata is updated:

```bash
git add 'backlog/tasks/task-502 - Design-tldw-Server-API-data-flow-atlas.md'
git commit -m "docs: update data flow atlas tracking"
```

Expected: commit only the Backlog task update.

## Stage 1: Foundation Atlas Skeleton

**Goal:** Create the atlas document with a stable table of contents, reading guide, legend, and section placeholders.

**Files:**
- Create: `Docs/Code_Documentation/Data_Flow_Atlas.md`
- Update: `backlog/tasks/task-502 - Design-tldw-Server-API-data-flow-atlas.md`

- [ ] **Step 1: Inspect nearby docs for tone and link style**

Run:

```bash
sed -n '1,140p' Docs/Architecture.md
sed -n '1,140p' Docs/Code_Documentation/Code_Map.md
sed -n '1,220p' Docs/Getting_Started/ARCHITECTURE.md
```

Expected: existing docs use concise Markdown, Mermaid fences, and relative doc links.

- [ ] **Step 2: Create the atlas skeleton**

Create `Docs/Code_Documentation/Data_Flow_Atlas.md` with these top-level sections:

```markdown
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
```

Expected: the file exists and has the agreed structure.

- [ ] **Step 3: Add section placeholders**

Add placeholder headings for all later stages:

```markdown
## System Context
## Request Lifecycle
## Router Group Map
## Data Store Map
## Core Flow Diagrams
### Auth And User Context
### Media Ingestion
### Audio STT/TTS
### Chunking And Embeddings
### RAG/Search
### Chat And LLM Provider Calls
### Jobs And Scheduler
## Extended Domain Maps
### Evaluations
### MCP Unified
### Prompt Studio
### Notes And Chatbooks
### Research And Web Scraping
### Storage, Files, And Outputs
### Admin, Ops, And Governance
### Characters And Workspaces
### Integrations And Connectors
## Router Coverage Matrix
## How To Update This Atlas
```

Expected: later stages can fill sections without changing the overall shape.

- [ ] **Step 4: Run a skeleton text check**

Run:

```bash
rg -n "Data Flow Atlas|System Context|Router Coverage Matrix|How To Update This Atlas" Docs/Code_Documentation/Data_Flow_Atlas.md
```

Expected: all required section names appear.

- [ ] **Step 5: Commit foundation skeleton**

```bash
git add Docs/Code_Documentation/Data_Flow_Atlas.md 'backlog/tasks/task-502 - Design-tldw-Server-API-data-flow-atlas.md'
git commit -m "docs: add data flow atlas skeleton"
```

Expected: commit only the atlas skeleton and related Backlog task update.

## Stage 2: Foundation Maps

**Goal:** Fill the system context, request lifecycle, router group map, and data-store map with code-grounded Mermaid diagrams.

**Files:**
- Modify: `Docs/Code_Documentation/Data_Flow_Atlas.md`
- Update: `backlog/tasks/task-502 - Design-tldw-Server-API-data-flow-atlas.md`
- Read: `tldw_Server_API/app/main.py`
- Read: `tldw_Server_API/app/api/v1/router_registry.py`
- Read: `tldw_Server_API/app/api/v1/router_groups/spec.py`
- Read: `tldw_Server_API/app/api/v1/router_groups/core.py`
- Read: `tldw_Server_API/app/api/v1/router_groups/content.py`
- Read: `tldw_Server_API/app/api/v1/router_groups/admin.py`
- Read: `tldw_Server_API/app/api/v1/router_groups/minimal.py`
- Read: `tldw_Server_API/app/api/v1/router_groups/conditional.py`
- Read: `tldw_Server_API/app/api/v1/router_groups/factories.py`
- Read: `Docs/Architecture.md`
- Read: `Docs/Code_Documentation/Code_Map.md`

- [ ] **Step 1: Inspect router registration anchors**

Run:

```bash
rg -n "def (include_router_idempotent|register_router_specs|register_all_routers)|def resolve_router|def append_imported_router_spec|def iter_.*router_specs|register_router_specs\\(" tldw_Server_API/app/api/v1/router_registry.py tldw_Server_API/app/api/v1/router_groups/*.py tldw_Server_API/app/main.py
```

Expected: output includes `include_router_idempotent`, `register_router_specs`, `register_all_routers`, `RouterSpec.resolve_router`, `append_imported_router_spec`, router group iterator functions, and minimal registration calls in `main.py`.

- [ ] **Step 2: Add the System Context flowchart**

Add a Mermaid `flowchart LR` showing:

- Clients: WebUI, Admin UI, browser extension, HTTP clients, MCP clients.
- FastAPI app: `main.py`, middleware/lifespan, router registry.
- API deps: auth/user context, validation, rate limiting, resource governance, DB dependencies.
- Endpoint groups: core, content, admin, optional/minimal.
- Core modules: AuthNZ, Ingestion, Chunking, Embeddings, RAG, Chat/LLM, Audio/TTS, Evaluations, MCP, Jobs/Scheduler, Storage.
- Storage/providers/workers.

Expected: the diagram keeps labels short and uses subgraphs.

- [ ] **Step 3: Add the Request Lifecycle sequence diagram**

Add a Mermaid `sequenceDiagram` from client to router registration, endpoint, dependencies, core module, storage/provider/worker, and response.

Must include these branches or notes:

- `route_enabled` gating for keyed router specs;
- lazy router resolution through `RouterSpec.resolve_router`;
- auth failure/rate-limit/resource-governance short-circuit;
- streaming/WebSocket response possibility;
- job handle response possibility.

Expected: contributors can identify where cross-cutting concerns belong.

- [ ] **Step 4: Add the Router Group Map**

Add a Mermaid `flowchart TB` showing:

- `main.py` registration path;
- full app path through `register_all_routers`;
- minimal test path through `iter_minimal_test_router_specs` and `iter_minimal_optional_router_specs`;
- grouped specs: core, content, admin;
- optional/lazy import behavior from `append_imported_router_spec`;
- idempotent include behavior.

Expected: the map shows registration/gating flow, not only filenames.

- [ ] **Step 5: Add the Data Store Map**

Add a Mermaid `flowchart LR` showing shared versus per-user storage:

- AuthNZ DB;
- per-user Media DB;
- per-user ChaChaNotes DB;
- prompt and Prompt Studio DBs;
- evaluations DB;
- vector store;
- file outputs/storage/voices/rewrite cache;
- Redis/job backend when configured.

Expected: storage ownership is clear enough for debugging data isolation issues.

- [ ] **Step 6: Run foundation checks**

Run:

```bash
rg -n "flowchart|sequenceDiagram|include_router_idempotent|register_router_specs|RouterSpec.resolve_router|AuthNZ DB|Media DB|ChaChaNotes|ChromaDB|Redis" Docs/Code_Documentation/Data_Flow_Atlas.md
```

Expected: all foundation map anchors appear.

- [ ] **Step 7: Commit foundation maps**

```bash
git add Docs/Code_Documentation/Data_Flow_Atlas.md 'backlog/tasks/task-502 - Design-tldw-Server-API-data-flow-atlas.md'
git commit -m "docs: map backend data flow foundations"
```

Expected: commit only atlas updates and related Backlog task update.

## Stage 3: Core Flow Diagrams

**Goal:** Fill the core process flows most useful to newcomers and maintainers.

**Files:**
- Modify: `Docs/Code_Documentation/Data_Flow_Atlas.md`
- Update: `backlog/tasks/task-502 - Design-tldw-Server-API-data-flow-atlas.md`
- Read selected files under `tldw_Server_API/app/api/v1/endpoints/`
- Read selected files under `tldw_Server_API/app/core/`
- Read module docs under `Docs/Code_Documentation/`, `Docs/STT-TTS/`, and `Docs/MCP/Unified/`

- [ ] **Step 1: Inspect core flow anchors**

Run:

```bash
rg -n "chat/completions|audio/transcriptions|stream/transcribe|media/add|process-|rag|embeddings|Job|Scheduler|APScheduler" tldw_Server_API/app/api/v1/endpoints tldw_Server_API/app/core Docs/Code_Documentation Docs/STT-TTS Docs/MCP/Unified
```

Expected: output identifies representative endpoint/core/doc anchors for the core flow diagrams.

- [ ] **Step 2: Add Auth And User Context**

Add `Purpose`, `Primary entrypoints`, Mermaid diagram, storage/provider touchpoints, and code links.

Required content:

- single-user `X-API-KEY`;
- multi-user JWT;
- auth dependency;
- `core/AuthNZ`;
- Auth DB;
- resolved user context;
- per-user storage root selection.

Expected: the diagram distinguishes identity/auth storage from per-user content storage.

- [ ] **Step 3: Add Media Ingestion**

Required content:

- file/document, URL/video/audio, web scraping inputs;
- downloader/transcode/OCR/STT branches;
- normalization;
- chunking;
- Media DB write;
- embeddings/vector store update;
- optional job/background path.

Expected: the diagram explains how content becomes searchable/RAG-ready.

- [ ] **Step 4: Add Audio STT/TTS**

Required content:

- file transcription;
- streaming transcription over WebSocket;
- TTS;
- provider/local backend selection;
- optional persistence as media/searchable transcript;
- audio history/jobs.

Expected: file, streaming, and TTS paths are visually distinct.

- [ ] **Step 5: Add Chunking And Embeddings**

Required content:

- chunking endpoint and ingestion-triggered chunking;
- strategies/templates;
- embedding provider/model selection;
- vector store write;
- Media DB metadata and FTS relationship;
- batch/worker path.

Expected: the diagram makes clear that chunking and embeddings can be API-triggered or ingestion-triggered.

- [ ] **Step 6: Add RAG/Search**

Required content:

- unified RAG request;
- request normalization/settings;
- FTS/BM25 retrieval;
- vector retrieval;
- reranking/post-processing;
- result/context assembly;
- standalone RAG endpoint and chat usage.

Expected: hybrid retrieval flow is clear.

- [ ] **Step 7: Add Chat And LLM Provider Calls**

Required content:

- OpenAI-compatible chat endpoint;
- optional RAG context;
- chat/session persistence;
- provider/model resolution;
- provider adapter call;
- streaming and non-streaming response paths.

Expected: chat generation and retrieval are shown as related but separable paths.

- [ ] **Step 8: Add Jobs And Scheduler**

Required content:

- Jobs path for user-visible work, admin status, retries, quotas, worker processing;
- Scheduler path for internal orchestration, dependencies, task registration, idempotency;
- APScheduler bridges to chosen backend;
- decision note matching the repository Jobs vs Scheduler guide.

Expected: contributors understand when to use Jobs versus Scheduler.

- [ ] **Step 9: Run core flow checks**

Run:

```bash
rg -n "Auth And User Context|Media Ingestion|Audio STT/TTS|Chunking And Embeddings|RAG/Search|Chat And LLM Provider Calls|Jobs And Scheduler|X-API-KEY|JWT|FTS|BM25|rerank|APScheduler" Docs/Code_Documentation/Data_Flow_Atlas.md
```

Expected: every core flow section and key term appears.

- [ ] **Step 10: Commit core flows**

```bash
git add Docs/Code_Documentation/Data_Flow_Atlas.md 'backlog/tasks/task-502 - Design-tldw-Server-API-data-flow-atlas.md'
git commit -m "docs: add core backend flow diagrams"
```

Expected: commit only atlas updates and related Backlog task update.

## Stage 4: Extended Domain Maps

**Goal:** Add grouped diagrams for the remaining domains so the atlas is exhaustive by group/domain/flow without becoming an endpoint inventory.

**Files:**
- Modify: `Docs/Code_Documentation/Data_Flow_Atlas.md`
- Update: `backlog/tasks/task-502 - Design-tldw-Server-API-data-flow-atlas.md`
- Read selected endpoint/core docs for each domain

- [ ] **Step 1: Inspect extended domain anchors**

Run:

```bash
rg -n "evaluations|mcp|prompt_studio|chatbooks|notes|research|web_scraping|storage|outputs|admin|governance|characters|workspaces|connectors|integrations" tldw_Server_API/app/api/v1/endpoints tldw_Server_API/app/core Docs/Code_Documentation Docs/MCP Docs/API-related
```

Expected: output identifies representative endpoint/core/doc anchors for extended maps.

- [ ] **Step 2: Add Evaluations map**

Show recipes, datasets, runs, evaluator services, RAG eval hooks, LLM judge/provider calls, metrics/results persistence.

- [ ] **Step 3: Add MCP Unified map**

Show HTTP/WebSocket entrypoints, auth/RBAC, tool/module registry, domain dispatch, status/metrics/tool execution outputs.

- [ ] **Step 4: Add Prompt Studio map**

Show projects, prompts, test cases, optimization/evaluation/status/WebSocket routes, prompt studio DB, provider calls, and job/background path where applicable.

- [ ] **Step 5: Add Notes And Chatbooks map**

Show notes/graph/web clipper, ChaChaNotes DB, chatbooks import/export, background job path, and generated artifacts/storage.

- [ ] **Step 6: Add Research And Web Scraping map**

Show research routes, paper search, web search/scraping, external sources/connectors, ingestion handoff, Media DB/RAG handoff.

- [ ] **Step 7: Add Storage, Files, And Outputs map**

Show storage routes, user files/folders/trash/downloads, outputs/templates, generated file helpers, file storage, quotas, and signed download behavior where documented.

- [ ] **Step 8: Add Admin, Ops, And Governance map**

Show admin route group, monitoring/metrics/audit, orgs/billing/privileges, resource governor, jobs admin, config admin, and shared AuthNZ/usage storage.

- [ ] **Step 9: Add Characters And Workspaces map**

Show character endpoints, character sessions/messages/memory, workspace routes/migrations, ChaChaNotes DB, chat/LLM handoff.

- [ ] **Step 10: Add Integrations And Connectors map**

Show connectors, ingestion sources, Slack/Discord/Telegram/meetings, external providers, ingestion/research handoff, optional route gating.

- [ ] **Step 11: Run extended domain checks**

Run:

```bash
rg -n "Evaluations|MCP Unified|Prompt Studio|Notes And Chatbooks|Research And Web Scraping|Storage, Files, And Outputs|Admin, Ops, And Governance|Characters And Workspaces|Integrations And Connectors" Docs/Code_Documentation/Data_Flow_Atlas.md
```

Expected: all extended domain maps appear.

- [ ] **Step 12: Commit extended maps**

```bash
git add Docs/Code_Documentation/Data_Flow_Atlas.md 'backlog/tasks/task-502 - Design-tldw-Server-API-data-flow-atlas.md'
git commit -m "docs: add extended backend domain maps"
```

Expected: commit only atlas updates and related Backlog task update.

## Stage 5: Router Coverage Matrix And Links

**Goal:** Make coverage auditable and link the atlas from existing docs.

**Files:**
- Modify: `Docs/Code_Documentation/Data_Flow_Atlas.md`
- Modify: `Docs/Architecture.md`
- Modify: `Docs/Code_Documentation/Code_Map.md`
- Update: `backlog/tasks/task-502 - Design-tldw-Server-API-data-flow-atlas.md`

- [ ] **Step 1: Build the router coverage matrix**

In `Data_Flow_Atlas.md`, add a compact table:

```markdown
| Router group or domain | Representative routes/modules | Atlas section | Coverage note |
| --- | --- | --- | --- |
| Core infrastructure | `health`, `monitoring`, `metrics`, `audit`, `setup` | System Context, Request Lifecycle, Admin/Ops | Grouped because these share infrastructure/diagnostic flow. |
```

Include representative rows for:

- core/infrastructure;
- identity/config/sync;
- chat/LLM;
- ACP/MCP;
- content/RAG/media/audio/embeddings/evaluations/OCR;
- workflows/scheduler/jobs;
- notes/prompts/prompt studio/workspaces/characters;
- storage/files/outputs/sharing;
- research/web scraping/connectors/integrations;
- admin/orgs/billing/resource governance/monitoring.

Expected: every domain group from the approved spec has a row.

- [ ] **Step 2: Add the update checklist**

Add a short checklist under `How To Update This Atlas`:

```markdown
- Check `router_groups/*.py` and `router_registry.py` for router changes.
- Check changed endpoint/core modules for new storage/provider/worker paths.
- Update the relevant diagram and the router coverage matrix together.
- Re-run Markdown/Mermaid text checks.
- Record verification in the relevant Backlog task.
```

Expected: future maintainers have a low-friction update path.

- [ ] **Step 3: Link from Architecture.md**

Modify `Docs/Architecture.md` near the existing visual diagram/code-map sentence:

```markdown
For detailed backend data flow and process diagrams, see `Docs/Code_Documentation/Data_Flow_Atlas.md`.
```

Expected: `Docs/Architecture.md` points readers to the atlas without becoming longer.

- [ ] **Step 4: Link from Code_Map.md**

Modify `Docs/Code_Documentation/Code_Map.md` near the high-level architecture or key flows section:

```markdown
For a deeper Mermaid atlas of request lifecycle, router groups, storage ownership, and subsystem data flows, see `Docs/Code_Documentation/Data_Flow_Atlas.md`.
```

Expected: code-map readers can jump to the detailed atlas.

- [ ] **Step 5: Run link checks**

Run:

```bash
rg -n "Data_Flow_Atlas.md" Docs/Architecture.md Docs/Code_Documentation/Code_Map.md Docs/Code_Documentation/Data_Flow_Atlas.md
```

Expected: all three docs reference the atlas path.

- [ ] **Step 6: Commit coverage and links**

```bash
git add Docs/Code_Documentation/Data_Flow_Atlas.md Docs/Architecture.md Docs/Code_Documentation/Code_Map.md 'backlog/tasks/task-502 - Design-tldw-Server-API-data-flow-atlas.md'
git commit -m "docs: link data flow atlas"
```

Expected: commit only atlas, link docs, and related Backlog task update.

## Stage 6: Verification And Final Backlog Update

**Goal:** Verify the documentation and record final evidence.

**Files:**
- Read: `Docs/Code_Documentation/Data_Flow_Atlas.md`
- Modify: `backlog/tasks/task-502 - Design-tldw-Server-API-data-flow-atlas.md`

- [ ] **Step 1: Check required sections and source anchors**

Run:

```bash
rg -n "System Context|Request Lifecycle|Router Group Map|Data Store Map|Router Coverage Matrix|How To Update This Atlas|include_router_idempotent|register_router_specs|RouterSpec.resolve_router|append_imported_router_spec|ChaChaNotes|ChromaDB|APScheduler" Docs/Code_Documentation/Data_Flow_Atlas.md
```

Expected: all required sections and key anchors appear.

- [ ] **Step 2: Check Mermaid fence count**

Run:

```bash
awk '
  /^```mermaid$/ {
    if (in_mermaid) {
      print "Nested Mermaid fence at line " NR
      exit 1
    }
    in_mermaid = 1
    open_count++
    next
  }
  /^```$/ && in_mermaid {
    in_mermaid = 0
    close_count++
    next
  }
  END {
    if (in_mermaid) {
      print "Unclosed Mermaid fence"
      exit 1
    }
    printf "mermaid_fences_open=%d close=%d\n", open_count, close_count
    if (open_count != close_count) {
      exit 1
    }
  }
' Docs/Code_Documentation/Data_Flow_Atlas.md
```

Expected: command exits `0` and prints equal open/close counts.

- [ ] **Step 3: Optionally render-check Mermaid if a renderer exists**

Run:

```bash
command -v mmdc
```

Expected: if this prints a path, run a local render check into `/tmp` and do not commit generated assets. If unavailable, record that Mermaid render-check was skipped because no local renderer was available.

- [ ] **Step 4: Verify atlas links**

Run:

```bash
rg -n "Data_Flow_Atlas.md" Docs/Architecture.md Docs/Code_Documentation/Code_Map.md Docs/Code_Documentation/Data_Flow_Atlas.md
```

Expected: all intended docs link to the atlas.

- [ ] **Step 5: Verify router coverage against router group files**

Run:

```bash
rg -n "ImportedRouterSpec|RouterSpec\\(|iter_.*router_specs|route_key=|log_name=" tldw_Server_API/app/api/v1/router_groups/*.py
```

Expected: manually compare the output to the router coverage matrix. Every major group/domain should be represented or explicitly grouped.

- [ ] **Step 6: Record docs-only security handling**

Update the Backlog task final summary:

```markdown
Bandit skipped: documentation-only change; no Python code modified.
```

Expected: security validation skip is explicit and justified.

- [ ] **Step 7: Final status check**

Run:

```bash
git status --short Docs/Code_Documentation/Data_Flow_Atlas.md Docs/Architecture.md Docs/Code_Documentation/Code_Map.md 'backlog/tasks/task-502 - Design-tldw-Server-API-data-flow-atlas.md'
```

Expected: only intended files are modified or staged.

- [ ] **Step 8: Commit final verification update**

```bash
git add 'backlog/tasks/task-502 - Design-tldw-Server-API-data-flow-atlas.md'
git commit -m "docs: record data flow atlas verification"
```

Expected: commit only final Backlog task update if all doc changes were already committed.

## Final Acceptance Criteria

- [ ] `Docs/Code_Documentation/Data_Flow_Atlas.md` exists.
- [ ] Atlas is Mermaid-only and contains no generated image assets.
- [ ] Atlas includes system context, request lifecycle, router group map, data-store map, core flows, extended domain maps, router coverage matrix, and update checklist.
- [ ] Atlas cites real code/doc anchors for router registration, endpoint groups, core modules, storage, providers, and workers.
- [ ] `Docs/Architecture.md` links to the atlas.
- [ ] `Docs/Code_Documentation/Code_Map.md` links to the atlas.
- [ ] Router coverage matrix accounts for all major router groups/domains in the approved spec.
- [ ] Markdown/Mermaid text checks were run.
- [ ] Mermaid render-check was run when a local renderer existed, or the skip was recorded.
- [ ] Bandit skip was recorded as documentation-only.
- [ ] Backlog task records touched files, verification, known skips, and final summary.
