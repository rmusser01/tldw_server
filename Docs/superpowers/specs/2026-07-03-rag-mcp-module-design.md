# RAG MCP Module Design

Date: 2026-07-03
Status: Draft for spec review
Backlog: TASK-12118

## Summary

Add a proper `rag.*` MCP module for tldw_server's existing RAG capabilities.

The module should expose retrieval, grounded answer generation, citations, source health, and capability discovery through a curated MCP surface. It should not create a new `research.*` facade or a separate research layer. Research workflows should be curated through catalogs and client guidance by grouping existing module-owned tools such as `rag.*`, `knowledge.*`, `media.*`, and `notes.*`.

The first implementation slice is deliberately narrow:

- `rag.capabilities`
- `rag.source_health`
- `rag.search`
- `rag.answer`

These tools should reuse the existing RAG request-resolution, pipeline, response-mapping, quota, source-scoping, and usage-accounting paths rather than calling the HTTP endpoint internally or duplicating endpoint logic.

## Goals

- Make core RAG functionality available through MCP for internal chat agents and external MCP clients.
- Keep the MCP surface curated and task-oriented without hiding the module ownership model.
- Preserve existing HTTP RAG behavior and endpoint contracts.
- Share RAG request resolution, source health, response mapping, quota checks, and usage accounting between HTTP and MCP.
- Keep `knowledge.*` as lightweight cross-source FTS discovery and `rag.*` as retrieval/generation/citation tooling.
- Provide bounded, citation-aware, machine-readable MCP results.
- Support curated workflow catalogs for existing-library research without using catalogs as a security boundary.

## Non-Goals

- No `research.*` MCP module or new research facade.
- No new research layer over existing modules.
- No batch RAG tool in the first slice.
- No streaming RAG tool in the first slice.
- No ablation, feedback, ingestion, note-writing, export, or chatbook workflows in the first slice.
- No silent web fallback, URL scraping, image search, video search, or external provider research loop in the first slice.
- No replacement of `knowledge.search` with RAG behavior.
- No direct HTTP call from MCP back into the FastAPI route.

## Current State

MCP Unified already has a production module architecture under `tldw_Server_API/app/core/MCP_unified/`.

Default enabled modules include:

- `media`
- `mcp_discovery`
- `governance`
- `notes`
- `prompts`
- `knowledge`
- `kanban`
- `quizzes`
- `flashcards`
- `slides`

Implemented but not default-enabled modules include browser, git, sandbox, web, characters, and chats tooling.

The current `knowledge.*` module is an FTS-style fan-out aggregator over source modules. It owns discovery and fetch behavior across notes, media, chats, characters, and prompts. It already applies source tool permission checks and persona/source scopes before fan-out.

The HTTP RAG API currently exposes:

- `POST /api/v1/rag/search`
- `GET /api/v1/rag/source-health`
- `GET /api/v1/rag/capabilities`
- batch, stream, ablation, feedback, and other adjacent endpoints

The RAG endpoint currently owns FastAPI concerns such as auth dependencies, rate limiting, token scopes, quota checks, usage logging, database dependency injection, request resolution, source health construction, and response conversion. The MCP module must not bypass those controls by calling `unified_rag_pipeline()` directly with ad hoc arguments.

Existing RAG service seams include:

- `rag_service/request_resolution.py`
- request bundle helpers used by `rag_unified.py`
- `rag_service/response_mapping.py`
- `unified_rag_pipeline()`
- source health helpers
- trust and citation metadata such as `metadata.knowledge_trust` and `metadata.hard_citations.coverage`

There is also an existing direct RAG consumer in `slides.generate.from_rag` that calls the pipeline directly. The `rag.*` work should create or reuse a shared helper that makes direct RAG consumers safer over time instead of adding another bypass.

## Approved Approach

Add a real `RagModule` under:

`tldw_Server_API/app/core/MCP_unified/modules/implementations/rag_module.py`

This module owns the `rag.*` MCP tools. It should validate curated MCP arguments, map them into constrained `UnifiedRAGRequest` objects, call shared RAG service helpers, and compact the result into MCP-safe payloads.

The module should not call the HTTP endpoint internally. HTTP and MCP should share service-level helpers under the RAG service package.

## Module Ownership

`knowledge.*` remains lightweight discovery:

- `knowledge.search`
- `knowledge.get`
- FTS fan-out across notes, media, chats, characters, and prompts
- source fetch and inspection routing

`rag.*` owns grounded retrieval and generation:

- source readiness for RAG
- runtime RAG capabilities
- retrieval-only search
- grounded answer generation
- citations, chunk citations, trust metadata, hard-citation coverage

`media.*` owns media inspection:

- transcript/content/metadata inspection
- media ingestion and updates where already supported

`notes.*` owns bounded note writes:

- note search and retrieval
- future workflow outputs may create/update notes through `notes.*`, not `rag.*`

Catalogs own workflow curation:

- group existing tools into research-friendly sets
- reduce discovery noise
- never replace RBAC, source checks, quotas, or module enablement

## Initial Tools

### `rag.capabilities`

Return MCP-safe RAG capability metadata aligned with `/api/v1/rag/capabilities`.

This is informational. The HTTP endpoint appears informational and broadly accessible, but the MCP tool may still be gated by `tools.execute:rag.capabilities` unless the project deliberately grants it broadly.

### `rag.source_health`

Return safe pre-query source readiness aligned with `/api/v1/rag/source-health`.

The tool should use the same source health helper as HTTP and should respect the current user/source context.

### `rag.search`

Run retrieval without generation.

Defaults:

- `enable_generation=false`
- citations enabled where the retrieval path can provide them
- local-library sources only
- bounded `top_k`
- bounded returned documents
- bounded per-document content

### `rag.answer`

Run grounded answer generation over retrieved sources.

Defaults:

- `enable_generation=true`
- citations enabled
- conservative grounding behavior
- hard-citation support enabled when profile/defaults request it
- abstain or mark partial when evidence is weak

`rag.answer` is more expensive and riskier than `rag.search` because it can use an LLM provider. It should have a distinct MCP category and rate/cost posture from ordinary read-only retrieval.

## Curated Input Surface

The tools should not expose the full `UnifiedRAGRequest` schema directly. That schema is an HTTP power-user interface. MCP should expose a smaller curated surface with a controlled `advanced` escape hatch only for options intentionally supported in this slice.

Common arguments:

- `query`
- `sources`
- `search_mode`
- `top_k`
- `min_score`
- `rag_profile`
- `include_documents`
- `max_documents`
- `max_content_chars`
- `allow_partial`
- citation options
- `advanced`

Default source behavior:

- implicit/default source selection may return partial results with warnings
- explicit source requests are strict by default
- if the caller requests `["media", "notes"]` and notes are unavailable, return `ok:false` unless `allow_partial=true`

Every tool schema should use `additionalProperties=false`.

## Architecture

The target flow is:

1. MCP client calls `rag.search` or `rag.answer`.
2. `RagModule.execute_tool()` validates curated arguments and maps them into a constrained `UnifiedRAGRequest`.
3. A shared RAG service request-bundle helper resolves:
   - `ResolvedRAGRequest`
   - retrieval plan
   - canonical sources
   - profile defaults
   - search-agent defaults
   - user id
   - feedback user id where relevant
   - pipeline kwargs
4. The helper accepts database paths/handles from either:
   - FastAPI dependencies for HTTP
   - `RequestContext.db_paths` for MCP
5. Source scopes from the MCP request context are applied before retrieval:
   - `media_id`
   - `note_id`
   - `conversation_id`
   - `character_id`
   - `prompt_id`
   - workspace/session metadata where available
6. The module executes the shared RAG pipeline through the shared bundle.
7. Existing response mapping converts core output into the common RAG response.
8. `RagModule` compacts the response for MCP payload caps while preserving citations and trust metadata.

The shared helper should be a service-level seam. It should not import FastAPI request objects or MCP protocol classes directly. Transport adapters should pass explicit user/context values into it.

## Security And Governance

The MCP module must preserve the controls currently layered on HTTP RAG routes.

Required parity:

- top-level MCP tool execution permission: `tools.execute:rag.<tool>`
- RAG-specific RBAC/resource posture equivalent to `rbac_rate_limit("rag.search")`
- media/source read entitlement equivalent to `MEDIA_READ`
- token-scope constraints equivalent to `TokenScopeGuard`
- RAG query daily quota equivalent to `LimitCategory.RAG_QUERIES_DAY`
- MCP per-tool/category rate limiting
- source scope filtering
- usage logging/accounting
- safe config and request metadata propagation

If any of those controls are not available as reusable service helpers today, the implementation should extract reusable helpers rather than copying endpoint-local logic into the module.

Catalog membership must not be treated as security. Catalogs shape discovery only. Tool execution permissions, module enablement, source entitlements, quotas, and source scopes are the security boundary.

## Result Contract

`rag.*` defines only the inner structured tool payload. It does not define a new MCP transport envelope.

For JSON-RPC `tools/call`, the existing MCP runtime wraps dict results under:

`result.content[0].json`

The runtime also returns `module`, `tool`, and `eval` alongside the content. The HTTP `/tools/execute` facade can expose the same inner payload under its existing `result` field.

Canonical inner payload:

```json
{
  "ok": true,
  "query": "...",
  "mode": "search",
  "documents": [],
  "citations": [],
  "chunk_citations": [],
  "metadata": {
    "sources_requested": ["media", "notes"],
    "sources_used": ["media"],
    "sources_unavailable": [],
    "allow_partial": false,
    "search_mode": "hybrid",
    "top_k": 10,
    "returned_documents": 6,
    "documents_truncated": false,
    "max_documents": 6,
    "max_content_chars": 2000,
    "knowledge_trust": {},
    "hard_citation_coverage": null,
    "timings_ms": {},
    "warnings": []
  },
  "errors": []
}
```

`rag.search` omits `answer`.

`rag.answer` includes:

```json
{
  "answer": {
    "text": "...",
    "status": "answered",
    "reason_code": null
  }
}
```

Allowed answer statuses:

- `answered`
- `partial`
- `abstained`

An uncited or weakly grounded answer must be `partial` or `abstained`, not silently treated as fully answered.

`citations` is the normalized compact citation list for agents. `chunk_citations` remains available when requested so span-level provenance is not lost. Existing RAG citation fields are preserved rather than replaced.

`hard_citation_coverage` is derived from existing RAG metadata, specifically `metadata.hard_citations.coverage` when present. It is not a second source of truth.

Result caps are explicit:

- each document has `content_truncated`
- response metadata has `documents_truncated`
- response metadata has `max_documents`
- response metadata has `max_content_chars`

The module may include internal `eval` metadata in the returned dict to feed existing MCP tool-use reporting. Clients should treat `metadata` as RAG-domain metadata and `eval` as execution observability.

## Error Handling

Transport and policy failures use the existing MCP error path:

- invalid schema
- auth/RBAC denial
- source entitlement denial
- token-scope denial
- rate limits
- quota limits
- disabled modules
- module timeouts

Where the existing MCP server provides `error.data.reason_code` and `error.data.next_action`, `rag.*` should preserve that behavior.

RAG-domain failures return structured tool payloads:

```json
{
  "ok": false,
  "reason_code": "no_results",
  "message": "No matching documents were found.",
  "metadata": {},
  "errors": []
}
```

Examples:

- `no_results`
- `source_unavailable`
- `retrieval_failed`
- `generation_failed`
- `weak_evidence`
- `citation_coverage_insufficient`

Partial success is allowed only when `allow_partial=true` or when the source selection was implicit/default. Explicit source requests fail closed by default.

Public error payloads must not leak secrets, raw provider keys, config paths, or prompt bodies. Internal logs can retain operational context through existing redaction rules.

## Testing

Test coverage should prove three things:

1. `rag.*` exposes a curated MCP surface.
2. `rag.*` calls the shared RAG service path.
3. `rag.*` preserves HTTP RAG controls.

Required tests:

- tool schema tests for all four tools, including strict schemas and caps
- argument mapping tests from curated MCP args to constrained `UnifiedRAGRequest`
- shared-control parity tests for:
  - `check_rate_limit`
  - `rbac_rate_limit("rag.search")`
  - `MEDIA_READ`
  - `TokenScopeGuard`
  - `LimitCategory.RAG_QUERIES_DAY`
  - source scoping
  - usage logging
- result compaction tests for bounded documents, truncation flags, normalized citations, preserved chunk citations, and derived hard-citation coverage
- error contract tests for invalid args, auth/RBAC/source denial, quota denial, RAG-domain failures, and partial source behavior
- permission tests for `tools.execute:rag.capabilities`, `rag.source_health`, `rag.search`, and `rag.answer`
- catalog tests proving visibility filtering does not grant execution rights
- MCP category/rate tests for `rag.search` and `rag.answer`
- config tests proving the module loads from `mcp_modules.yaml`
- regression tests proving `knowledge.search` remains FTS discovery
- follow-up or regression coverage for direct RAG consumers such as `slides.generate.from_rag`

Implementation verification should include:

- targeted MCP/RAG pytest scope
- Bandit on touched MCP/RAG/config files
- a manual or automated smoke call through JSON-RPC `tools/call` for `rag.search` and `rag.answer`

## Rollout

1. Add `RagModule` with the four initial tools.
2. Enable it by default only if shared RAG controls are enforced through MCP; otherwise ship disabled until parity is complete.
3. Add `rag.search` and `rag.answer` to MCP tool category config, with `rag.answer` in a costlier category than ordinary read tools.
4. Add curated workflow catalogs for existing-library research, but treat catalogs as discovery curation only.
5. Keep batch, stream, ablation, feedback, ingestion, note-writing, and export out of the first slice.
6. Document the HTTP-to-MCP crosswalk:
   - `/api/v1/rag/search` maps to `rag.search` and `rag.answer`
   - `/api/v1/rag/source-health` maps to `rag.source_health`
   - `/api/v1/rag/capabilities` maps to `rag.capabilities`
7. Document that `rag.capabilities` may be MCP-permission-gated even though the HTTP capabilities endpoint is informational.
8. Run targeted tests plus Bandit on touched MCP/RAG/config scopes before implementation is considered complete.

## Acceptance Criteria

- MCP clients can discover and call `rag.search` and `rag.answer`.
- Results are citation-aware, bounded, and machine-readable.
- Implementation reuses shared RAG request resolution, response mapping, quota checks, source checks, and usage accounting.
- Tool permissions and source permissions are enforced independently of catalog membership.
- Catalogs reduce discovery noise without introducing a `research.*` facade.
- `knowledge.search` remains FTS discovery.
- The first slice does not include batch, stream, ablation, feedback, ingestion, note-writing, or export workflows.

## Follow-Ups

- Evaluate whether `slides.generate.from_rag` should migrate to the same shared RAG helper in the first implementation or a follow-up task.
- Consider a later `rag.batch` only after single-query control parity is stable.
- Consider a later streaming-adjacent MCP design only after the HTTP streaming executor is cleaner.
- Consider note-writing workflows through `notes.*`, not `rag.*`.
- Consider feedback and evaluation workflows after `rag.answer` has stable citation and trust contracts.
