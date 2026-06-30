# RAG Remediation Design

Date: 2026-04-08
Status: Approved design
Scope: `tldw_Server_API` unified RAG request handling, orchestration, retrieval ownership, evidence ownership, cache invalidation ownership, and API response mapping

## Background

This remediation design follows the six-stage RAG architecture review completed on 2026-04-07:

- [Stage 1 Architecture Survey and Inventory](/Users/appledev/Documents/GitHub/tldw_server/.worktrees/rag-module-review/Docs/superpowers/reviews/rag/2026-04-07-stage1-architecture-survey-and-inventory.md)
- [Stage 2 Unified Pipeline Orchestration](/Users/appledev/Documents/GitHub/tldw_server/.worktrees/rag-module-review/Docs/superpowers/reviews/rag/2026-04-07-stage2-unified-pipeline-orchestration.md)
- [Stage 3 API Schema and Request Boundaries](/Users/appledev/Documents/GitHub/tldw_server/.worktrees/rag-module-review/Docs/superpowers/reviews/rag/2026-04-07-stage3-api-schema-and-request-boundaries.md)
- [Stage 4 Retrieval Boundaries and Data Sources](/Users/appledev/Documents/GitHub/tldw_server/.worktrees/rag-module-review/Docs/superpowers/reviews/rag/2026-04-07-stage4-retrieval-boundaries-and-data-sources.md)
- [Stage 5 Reranking and Post-Retrieval Composition](/Users/appledev/Documents/GitHub/tldw_server/.worktrees/rag-module-review/Docs/superpowers/reviews/rag/2026-04-07-stage5-reranking-and-post-retrieval-composition.md)
- [Stage 6 Test Gaps and Synthesis](/Users/appledev/Documents/GitHub/tldw_server/.worktrees/rag-module-review/Docs/superpowers/reviews/rag/2026-04-07-stage6-test-gaps-and-synthesis.md)

The review found four primary architectural defects:

1. Request policy ownership is split across endpoint helpers, schemas, profile helpers, and ad hoc branch logic.
2. Retrieval policy and namespace ownership are distributed across orchestration code, concrete retrievers, and API utilities.
3. Post-retrieval evidence can be transformed or replaced without one authoritative owner.
4. API response mapping duplicates internal knowledge and already misses declared response fields.

The remediation goal is long-term stability with pragmatic delivery. That rules out a big-bang rewrite. The design therefore uses a staged core rewrite behind compatibility adapters.

## Goals

1. Preserve the existing external `/api/v1/rag/search` request and response contract unless a change is additive or a deprecation-marked cleanup.
2. Replace the current RAG center of gravity with explicit internal contracts for resolved requests, retrieval plans, evidence states, and final results.
3. Make standard, agentic, batch, and streaming paths consume the same resolved request semantics.
4. Move retrieval and invalidation ownership into core RAG services.
5. Make post-retrieval evidence transformation explicit and centrally coordinated.
6. Add structural tests that protect boundary behavior instead of only leaf behavior.

## Non-Goals

1. No breaking HTTP contract redesign in this cycle.
2. No full removal of existing pipeline modules in the first implementation cycle.
3. No speculative cleanup outside the reviewed RAG request, retrieval, evidence, and response seams.

## Compatibility Policy

External compatibility will be preserved as follows:

- Existing request and response JSON shapes remain accepted and returned.
- Existing aliases such as `corpus -> index_namespace` continue to work.
- Existing response fields that are already declared in the schema become more consistently populated.
- Additive internal metadata cleanup is allowed when an explicit response field already exists.
- Deprecation-marked cleanup is allowed for duplicate internal contract owners such as `UnifiedBatchRequest` drift, but not as a hard removal in the first cycle.
- `UnifiedBatchRequest` remains accepted externally, but during migration it becomes a transport wrapper around shared RAG request semantics plus `queries`, not a second canonical request contract.

This means the remediation may correct behavior where different paths currently disagree, but it will not intentionally remove existing API inputs or outputs in the first pass.

## Architecture Overview

The target internal contract chain is:

`API request -> adapter -> ResolvedRAGRequest -> RetrievalPlan -> RetrievedEvidence -> DerivedEvidence -> RAGResult -> API response`

The API layer remains a compatibility shell. The new core RAG seams own policy and state transitions.

### 1. ResolvedRAGRequest

`ResolvedRAGRequest` is the single authoritative internal request contract.

Responsibilities:

- apply request precedence once
- resolve `rag_profile` defaults once
- normalize aliases such as `corpus -> index_namespace`
- resolve implicit user and feedback identifiers
- carry branch-agnostic policy for retrieval, generation, verification, and response shaping

This replaces endpoint-local resolution logic such as `_build_effective_request_payload()` and the hand-built agentic kwargs path in [rag_unified.py](/Users/appledev/Documents/GitHub/tldw_server/tldw_Server_API/app/api/v1/endpoints/rag_unified.py).

### 2. RetrievalPlan

`RetrievalPlan` is the core retrieval policy contract.

Responsibilities:

- source routing
- retrieval mode selection
- namespace and collection naming
- retriever execution setup
- invalidation namespace ownership

This prevents `MultiDatabaseRetriever` from acting as both coordinator and concrete policy owner in [database_retrievers.py](/Users/appledev/Documents/GitHub/tldw_server/tldw_Server_API/app/core/RAG/rag_service/database_retrievers.py).

### 3. RetrievedEvidence

`RetrievedEvidence` represents the authoritative output of retrieval before any generation-side or post-retrieval transforms occur.

Responsibilities:

- store the canonical retrieved document set
- record retrieval provenance and routing decisions
- separate raw retrieval evidence from later derived artifacts

This creates one boundary that later stages may consume but not silently replace.

### 4. DerivedEvidence

`DerivedEvidence` represents the output of coordinated post-retrieval work.

Responsibilities:

- reranking outputs
- citation structures
- guardrail outputs
- verification outputs
- answer-supporting derived artifacts
- agentic synthetic chunks, span assemblies, and corroboration artifacts derived from retrieved sources

Only a new post-retrieval coordinator may produce `DerivedEvidence`. Existing modules such as [guardrails.py](/Users/appledev/Documents/GitHub/tldw_server/tldw_Server_API/app/core/RAG/rag_service/guardrails.py), [citations.py](/Users/appledev/Documents/GitHub/tldw_server/tldw_Server_API/app/core/RAG/rag_service/citations.py), [generation.py](/Users/appledev/Documents/GitHub/tldw_server/tldw_Server_API/app/core/RAG/rag_service/generation.py), [response_writer.py](/Users/appledev/Documents/GitHub/tldw_server/tldw_Server_API/app/core/RAG/rag_service/response_writer.py), and agentic logic may still execute, but only through this coordinator.

For the agentic path, query-time synthetic chunk assembly is treated as derived evidence built from retrieved evidence. It is not a separate retrieval contract and must not reintroduce a parallel request-policy path.

### 5. RAGResult

`RAGResult` is the single authoritative internal result model.

Responsibilities:

- expose explicit response fields such as `verification_report`, `chunk_citations`, `query_classification`, `reformulated_query`, and `research_summary`
- separate result fields from generic metadata
- provide one mapping target for the API layer

This replaces endpoint-owned response-field inference in `convert_result_to_response()`.

## Module Boundaries

The staged rewrite should introduce or extract the following core modules under `tldw_Server_API/app/core/RAG/rag_service/`:

- `request_resolution.py`
  Builds `ResolvedRAGRequest` from endpoint input and compatibility aliases.
- `retrieval_plan.py`
  Builds `RetrievalPlan` and owns namespace and collection decisions.
- `evidence_models.py`
  Defines `RetrievedEvidence` and `DerivedEvidence`.
- `post_retrieval_coordinator.py`
  Owns all transitions from retrieved evidence to derived evidence.
- `result_model.py`
  Defines `RAGResult`.
- `response_mapping.py`
  Maps `RAGResult` to the API schema.
- `cache_invalidation.py`
  Owns semantic cache, vector-store, and agentic invalidation in core RAG.

Existing modules continue to exist initially, but their ownership narrows:

- [rag_unified.py](/Users/appledev/Documents/GitHub/tldw_server/tldw_Server_API/app/api/v1/endpoints/rag_unified.py)
  Becomes an adapter and transport layer, not a request policy or response policy owner.
- [unified_pipeline.py](/Users/appledev/Documents/GitHub/tldw_server/tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py)
  Becomes a compatibility shell over new orchestration seams instead of remaining the orchestrator.
- [database_retrievers.py](/Users/appledev/Documents/GitHub/tldw_server/tldw_Server_API/app/core/RAG/rag_service/database_retrievers.py)
  Shrinks toward execution responsibilities instead of policy ownership.
- [agentic_chunker.py](/Users/appledev/Documents/GitHub/tldw_server/tldw_Server_API/app/core/RAG/rag_service/agentic_chunker.py)
  Moves onto the same resolved request and evidence contracts instead of owning a parallel policy path.
- [app/api/v1/utils/rag_cache.py](/Users/appledev/Documents/GitHub/tldw_server/tldw_Server_API/app/api/v1/utils/rag_cache.py)
  Has its responsibilities migrated into core `cache_invalidation.py`.

## Staged Migration

The rewrite is intentionally staged to reduce blast radius and preserve rollback points.

Each migrated path should be independently switchable during rollout so standard, agentic, batch, and streaming routes can fall back to legacy orchestration without reverting unrelated migrated paths.

### Stage 1: Introduce Core Contracts Without Behavior Change

Deliverables:

- add `ResolvedRAGRequest`, `RetrievalPlan`, `RetrievedEvidence`, `DerivedEvidence`, and `RAGResult`
- add request resolution and response mapping modules
- add thin planner and coordinator adapters that preserve existing behavior while making the new contracts executable
- keep endpoints and pipelines behaviorally equivalent while switching them to the new internal types where feasible

Primary objective:

- stop adding new logic to endpoint-local payload assembly and result mapping

### Stage 2: Migrate the Standard Path First

Deliverables:

- convert the standard `/search` path to use:
  `request -> resolver -> retrieval plan -> retrieval -> post-retrieval coordinator -> result -> response mapper`
- make [unified_pipeline.py](/Users/appledev/Documents/GitHub/tldw_server/tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py) a compatibility wrapper over the new orchestrator

Primary objective:

- replace the highest-risk path first while keeping transport complexity low

### Stage 3: Move Retrieval and Invalidation Ownership Into Core

Deliverables:

- complete retrieval planning ownership so source routing, namespace decisions, and retriever setup stop living in endpoint and retriever glue
- migrate cache invalidation from [rag_cache.py](/Users/appledev/Documents/GitHub/tldw_server/tldw_Server_API/app/api/v1/utils/rag_cache.py) into core RAG
- reduce `MultiDatabaseRetriever` to execution concerns

Primary objective:

- remove convention-based retrieval and invalidation ownership

### Stage 4: Migrate Agentic, Batch, and Streaming Paths

Deliverables:

- move non-streaming agentic search onto the same request resolver and result mapper
- move batch handling onto the same resolved request semantics, with `UnifiedBatchRequest` acting only as a compatibility wrapper over shared request fields plus `queries`
- move streaming pre-resolution and agentic prefetch paths onto the same internal request contract while keeping transport-specific SSE or chunk emission isolated to the endpoint layer

Primary objective:

- remove branch-specific policy drift

### Stage 5: Cleanup Compatibility Shims

Deliverables:

- remove dead adapter code once all entry points use the new contracts
- narrow metadata-only escape hatches where explicit fields now exist
- retain deprecation markers for any transitional request or batch aliases still accepted externally

Primary objective:

- finish the migration without keeping duplicate internal owners

## Confirmed Issues Addressed In This Design

### Request Contract Drift

This design fixes:

- non-streaming agentic `/search` bypassing shared request resolution
- batch processing acting as a second effective request contract
- profile, namespace, and implicit user resolution drifting by path

### Response Contract Drift

This design fixes:

- missing `verification_report` mapping despite the field existing in [UnifiedRAGResponse](/Users/appledev/Documents/GitHub/tldw_server/tldw_Server_API/app/api/v1/schemas/rag_schemas_unified.py)
- endpoint-owned metadata unpacking as a second schema authority
- inconsistent ownership of explicit response fields versus metadata-only fields

### API-to-Core Boundary Inversion

This design fixes:

- core workers or jobs depending on an API utility for RAG cache invalidation
- invalidation rules being spread across API helpers and core modules

### Retrieval Ownership Drift

This design fixes:

- retrieval policy hardening inside concrete retrievers
- `MultiDatabaseRetriever` acting like a mixed coordinator and executor
- namespace and collection ownership remaining convention-based

### Evidence Ownership Drift

This design fixes:

- post-retrieval stages changing the working evidence set without one owner
- citations, guardrails, verification, and response writing composing through ad hoc orchestration rather than a single post-retrieval coordinator

## Testing Strategy

Structural tests are part of the remediation, not follow-up work.

### Request Parity Tests

Add representative tests proving standard, agentic, batch, and streaming paths resolve the same effective request semantics for shared fields:

- `rag_profile`
- `index_namespace`
- implicit user and feedback resolution
- verification and generation flags

### Response Parity Tests

Add tests proving explicit response fields are mapped from `RAGResult` rather than inferred ad hoc from metadata:

- `verification_report`
- `chunk_citations`
- `query_classification`
- `reformulated_query`
- `research_summary`

### Retrieval Ownership Tests

Add focused tests around:

- namespace and collection naming
- invalidation routing
- retrieval plan generation

PGVector parity tests should remain behind existing environment gates when external services are required, but the contract itself must be explicitly tested.

### Evidence-State Tests

Add tests that prove:

- retrieved evidence and derived evidence are distinct contracts
- post-retrieval transforms cannot replace retrieved evidence without going through the coordinator
- result mapping reflects the coordinated final state

### Compatibility Tests

Retain endpoint-level compatibility tests for `/api/v1/rag/search` so internal rewrites can proceed without silent contract drift.

Streaming compatibility tests should also prove that transport framing remains stable while resolved request semantics and terminal structured payloads stay aligned with the non-streaming contract.

## Risks And Mitigations

### Risk: staged rewrite leaves half-migrated logic in place too long

Mitigation:

- each migration stage must leave one path fully converted
- avoid introducing new features into old orchestration seams during migration
- keep per-path rollout switches until the migrated path and its compatibility tests are stable

### Risk: compatibility shims become permanent

Mitigation:

- each stage must identify which temporary adapter paths are expected to be removed in the next stage
- cleanup is an explicit migration stage, not optional follow-up

### Risk: evidence contracts are introduced but bypassed

Mitigation:

- post-retrieval modules are only called through the coordinator for migrated paths
- tests should fail when explicit result fields diverge from coordinated evidence state

## Success Criteria

The remediation is successful when all of the following are true:

1. Standard, agentic, batch, and streaming paths resolve shared request semantics through one core request contract.
2. Retrieval planning and cache invalidation ownership reside in core RAG services, not endpoint helpers or concrete retrievers.
3. Retrieved evidence and derived evidence are explicit internal contracts with one transition owner.
4. API response mapping is driven by `RAGResult` and populates declared explicit fields consistently.
5. Compatibility tests for `/api/v1/rag/search` remain green while the migration proceeds.
