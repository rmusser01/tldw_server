# RAG Thin Endpoint Follow-Up Design

## Goal

Finish the next architectural cleanup pass after the core-stability refactor by making [`rag_unified.py`](/Users/appledev/Documents/GitHub/tldw_server/.worktrees/rag-module-review/tldw_Server_API/app/api/v1/endpoints/rag_unified.py) a thin transport adapter across the remaining weak seams.

This follow-up covers three dependent slices, executed in order under one umbrella design:

1. Standard core contract threading
2. Agentic shell untangling
3. Streaming executor extraction

The desired end state is:

`HTTP route -> request_bundle.build_request_bundle -> core executor (standard | agentic | streaming) -> core post-retrieval / response mapping -> HTTP response`

## Scope

### In Scope

- [`tldw_Server_API/app/api/v1/endpoints/rag_unified.py`](/Users/appledev/Documents/GitHub/tldw_server/.worktrees/rag-module-review/tldw_Server_API/app/api/v1/endpoints/rag_unified.py)
- [`tldw_Server_API/app/core/RAG/rag_service/request_bundle.py`](/Users/appledev/Documents/GitHub/tldw_server/.worktrees/rag-module-review/tldw_Server_API/app/core/RAG/rag_service/request_bundle.py)
- [`tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py`](/Users/appledev/Documents/GitHub/tldw_server/.worktrees/rag-module-review/tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py)
- [`tldw_Server_API/app/core/RAG/rag_service/retrieval_executor.py`](/Users/appledev/Documents/GitHub/tldw_server/.worktrees/rag-module-review/tldw_Server_API/app/core/RAG/rag_service/retrieval_executor.py)
- [`tldw_Server_API/app/core/RAG/rag_service/generation_executor.py`](/Users/appledev/Documents/GitHub/tldw_server/.worktrees/rag-module-review/tldw_Server_API/app/core/RAG/rag_service/generation_executor.py)
- [`tldw_Server_API/app/core/RAG/rag_service/post_retrieval_coordinator.py`](/Users/appledev/Documents/GitHub/tldw_server/.worktrees/rag-module-review/tldw_Server_API/app/core/RAG/rag_service/post_retrieval_coordinator.py)
- [`tldw_Server_API/app/core/RAG/rag_service/agentic_execution.py`](/Users/appledev/Documents/GitHub/tldw_server/.worktrees/rag-module-review/tldw_Server_API/app/core/RAG/rag_service/agentic_execution.py)
- [`tldw_Server_API/app/core/RAG/rag_service/agentic_chunker.py`](/Users/appledev/Documents/GitHub/tldw_server/.worktrees/rag-module-review/tldw_Server_API/app/core/RAG/rag_service/agentic_chunker.py)
- New core streaming executor module(s) under [`tldw_Server_API/app/core/RAG/rag_service/`](/Users/appledev/Documents/GitHub/tldw_server/.worktrees/rag-module-review/tldw_Server_API/app/core/RAG/rag_service)
- RAG tests that pin request resolution, executor ownership, agentic parity, streaming delegation, batch/resume parity, and endpoint cleanup
- [`tldw_Server_API/app/core/RAG/rag_service/README.md`](/Users/appledev/Documents/GitHub/tldw_server/.worktrees/rag-module-review/tldw_Server_API/app/core/RAG/rag_service/README.md)

### Out of Scope

- Breaking removal of existing external `/api/v1/rag/search`, `/stream`, batch, or resume request/response fields
- New end-user RAG features unrelated to ownership cleanup
- Full `unified_pipeline.py` replacement beyond what is necessary to make the standard core contracts authoritative
- Non-RAG endpoint cleanup

## Constraints

- Preserve existing external HTTP behavior by default.
- Additive or deprecation-marked API cleanup is allowed if it directly reduces endpoint-local orchestration.
- Maintain the existing review worktree isolation.
- Keep changes incremental and phaseable; no big-bang rewrite.
- Treat current `dev` as the integration baseline, not the earlier review snapshot.
- Before Phase 1, port or explicitly re-verify any `dev`-side RAG changes that landed after the review branch diverged.

## Current Problems

The prior refactor materially improved the architecture, but the remaining weak seams are:

1. The standard core path still does not consume the canonical `ResolvedRAGRequest` contract end-to-end.
2. `agentic_execution.py` still depends back on `agentic_chunker.py`, leaving agentic ownership muddy.
3. `/api/v1/rag/search/stream` still owns too much business logic in the endpoint.
4. Cleanup tests still focus more on exposed helper names and behavior parity than on delegation to stable core execution boundaries.
5. The review branch is already diverged from current `dev` in live RAG code, including `analytics_db.py`, `unified_pipeline.py`-adjacent behavior, and RAG test coverage. The next plan must account for that drift explicitly rather than assuming the review branch is the only source of truth.

## Target Architecture

### Endpoint Boundary

[`rag_unified.py`](/Users/appledev/Documents/GitHub/tldw_server/.worktrees/rag-module-review/tldw_Server_API/app/api/v1/endpoints/rag_unified.py) should only own:

- HTTP request parsing and dependency injection
- auth/rate-limit/usage logging
- route selection between standard, agentic, streaming, batch, and resume
- HTTP and NDJSON response formatting

It should not own:

- internal request normalization beyond calling `build_request_bundle(...)`
- execution policy construction
- agentic config construction
- post-retrieval evidence shaping
- result/response adaptation logic

### Core Boundaries

#### 1. Request Contract

[`request_bundle.py`](/Users/appledev/Documents/GitHub/tldw_server/.worktrees/rag-module-review/tldw_Server_API/app/core/RAG/rag_service/request_bundle.py) remains the only transport-to-core handoff seam.

Every path must hand off:

- `ResolvedRAGRequest`
- `RetrievalPlan`
- canonical pipeline kwargs carrying those exact objects

Core consumers must use those objects directly instead of rebuilding ad hoc namespaces or request-like structures.

#### 2. Standard Execution Contract

The standard path should have one authoritative core flow that owns:

- retrieval
- generation
- post-retrieval coordination
- internal result construction

The endpoint must not post-process standard results after `unified_rag_pipeline(...)` returns.

If necessary, [`unified_pipeline.py`](/Users/appledev/Documents/GitHub/tldw_server/.worktrees/rag-module-review/tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py) becomes a compatibility shell over a smaller standard executor/orchestrator, but the core request contract must remain authoritative all the way through retrieval, generation, and coordination.

#### 3. Agentic Execution Contract

[`agentic_execution.py`](/Users/appledev/Documents/GitHub/tldw_server/.worktrees/rag-module-review/tldw_Server_API/app/core/RAG/rag_service/agentic_execution.py) should fully own:

- `AgenticConfig`
- effective agentic execution payload/context construction
- execution-only helpers needed by standard and streaming agentic paths

[`agentic_chunker.py`](/Users/appledev/Documents/GitHub/tldw_server/.worktrees/rag-module-review/tldw_Server_API/app/core/RAG/rag_service/agentic_chunker.py) should become a compatibility facade at most. There should be no reverse dependency from core execution back into the chunker shell.

#### 4. Streaming Contract

Streaming should gain a dedicated core executor boundary. That core executor should own:

- retrieval prefetch
- optional agentic pre-execution
- generation context preparation
- internal stream-event sequencing

The endpoint should only translate stable internal events into the existing NDJSON HTTP contract.

## Recommended Execution Strategy

Use a sequential core-first extraction.

### Phase 0: `dev` Reconciliation Gate

#### Objective

Reconcile the review branch with current `dev` RAG deltas before further architectural cleanup.

#### Required Changes

- Diff current review-branch RAG scope against current `dev`.
- Port or explicitly re-verify `dev`-side RAG changes that matter to the same ownership seams, especially:
  - [`analytics_db.py`](/Users/appledev/Documents/GitHub/tldw_server/.worktrees/rag-module-review/tldw_Server_API/app/core/RAG/rag_service/analytics_db.py)
  - [`unified_pipeline.py`](/Users/appledev/Documents/GitHub/tldw_server/.worktrees/rag-module-review/tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py)
  - [`agentic_chunker.py`](/Users/appledev/Documents/GitHub/tldw_server/.worktrees/rag-module-review/tldw_Server_API/app/core/RAG/rag_service/agentic_chunker.py)
  - RAG tests that were added, restored, or tightened on `dev`
- Record any intentionally deferred `dev`-side differences so later phases do not silently regress them.

#### Success Criteria

- The follow-up implementation starts from a branch state that is explicitly reconciled against current `dev` RAG changes.
- Any remaining branch-vs-`dev` differences are deliberate, documented, and covered by tests.

### Phase 1: Standard Core Contract Threading

#### Objective

Make the standard path consume `ResolvedRAGRequest` and `RetrievalPlan` as first-class inputs end-to-end.

#### Required Changes

- Remove ad hoc internal request reconstruction inside [`unified_pipeline.py`](/Users/appledev/Documents/GitHub/tldw_server/.worktrees/rag-module-review/tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py).
- Ensure retrieval and generation executors receive the canonical `ResolvedRAGRequest` and `RetrievalPlan`.
- Move standard-path evidence coordination fully into core.
- Remove endpoint-side standard result post-processing in [`rag_unified.py`](/Users/appledev/Documents/GitHub/tldw_server/.worktrees/rag-module-review/tldw_Server_API/app/api/v1/endpoints/rag_unified.py).

#### Success Criteria

- The standard endpoint path builds one request bundle and hands it off once.
- Core retrieval, generation, and post-retrieval coordination all see the same canonical request and plan objects.
- The endpoint no longer coordinates standard evidence after the pipeline returns.
- A branch-vs-`dev` checkpoint confirms Phase 1 did not drop newer `dev`-side RAG behavior in the same files.

### Phase 2: Agentic Shell Untangling

#### Objective

Make core agentic execution independent from the chunker compatibility shell.

#### Required Changes

- Move `AgenticConfig` and execution-context building into [`agentic_execution.py`](/Users/appledev/Documents/GitHub/tldw_server/.worktrees/rag-module-review/tldw_Server_API/app/core/RAG/rag_service/agentic_execution.py).
- Replace reverse lookups or patch seams that point back through [`agentic_chunker.py`](/Users/appledev/Documents/GitHub/tldw_server/.worktrees/rag-module-review/tldw_Server_API/app/core/RAG/rag_service/agentic_chunker.py) with core-owned seams.
- Make both standard and streaming agentic paths call the same core execution-context builder.
- Reduce `agentic_chunker.py` to compatibility-only behavior or remove its ownership role from the main execution path.
- Preserve and explicitly test the structure-DB failure fallback path currently guarded on `dev`, so retiring shell ownership does not drop heuristic fallback behavior when structure lookup fails.

#### Success Criteria

- `agentic_execution.py` no longer imports runtime ownership back from `agentic_chunker.py`.
- Standard and streaming agentic flows share the same core execution-context builder.
- Tests pin multiple config knobs across both paths, not just one flag.
- The agentic structure-DB error path still falls back to heuristics and is covered by an explicit regression test.
- A branch-vs-`dev` checkpoint confirms Phase 2 did not drop active `dev` agentic safeguards or tests.

### Phase 3: Streaming Executor Extraction

#### Objective

Move streaming execution ownership out of the endpoint into core.

#### Required Changes

- Introduce a core streaming executor module under [`tldw_Server_API/app/core/RAG/rag_service/`](/Users/appledev/Documents/GitHub/tldw_server/.worktrees/rag-module-review/tldw_Server_API/app/core/RAG/rag_service).
- Move streaming retrieval prefetch, agentic pre-execution, generation setup, and event ordering into that executor.
- Make [`rag_unified.py`](/Users/appledev/Documents/GitHub/tldw_server/.worktrees/rag-module-review/tldw_Server_API/app/api/v1/endpoints/rag_unified.py) choose the streaming route, hand off the canonical bundle, and emit the returned internal events as NDJSON.

#### Success Criteria

- `/search/stream` no longer contains large inline business-logic blocks for retrieval/generation orchestration.
- Streaming behavior remains externally compatible.
- A stable core stream-event boundary exists and is tested.
- A branch-vs-`dev` checkpoint confirms Phase 3 did not regress newer `dev` streaming or adjacent RAG behavior.

## Testing Strategy

Tests should pin delegation and ownership boundaries, not just output shape.

### Phase 1 Tests

- Standard-path tests proving the same `ResolvedRAGRequest` and `RetrievalPlan` objects flow through retrieval, generation, and post-retrieval coordination.
- Endpoint-level test proving `/rag/search` delegates standard evidence coordination to the core helper.
- Response mapping tests proving endpoint code uses core mapping directly.
- A reconciliation check against current `dev` for touched standard-path files after the phase lands.

### Phase 2 Tests

- Unit tests for the core agentic execution-context builder.
- Parity tests proving standard and streaming agentic paths use the same builder.
- Tests covering multiple agentic config knobs across both paths.
- Tests proving `agentic_execution.py` no longer depends back on `agentic_chunker.py`.
- Explicit regression test for structure-DB lookup failure falling back to heuristics.
- A reconciliation check against current `dev` for touched agentic files and tests after the phase lands.

### Phase 3 Tests

- Endpoint-level tests proving streaming delegates to the core streaming executor.
- Stream-event contract tests that pin event ordering and event translation at the endpoint boundary.
- Parity tests ensuring the streaming path still honors canonical request bundle resolution.
- A reconciliation check against current `dev` for touched streaming/RAG transport files after the phase lands.

### Compatibility Tests

Retain explicit compatibility coverage for:

- standard `/api/v1/rag/search`
- `/api/v1/rag/search/stream`
- batch
- batch resume
- agentic standard vs streaming parity

### Phase Checkpoints

After each phase, run a focused branch-vs-`dev` review for the touched RAG files before starting the next phase. At minimum, compare:

- [`rag_unified.py`](/Users/appledev/Documents/GitHub/tldw_server/.worktrees/rag-module-review/tldw_Server_API/app/api/v1/endpoints/rag_unified.py)
- [`unified_pipeline.py`](/Users/appledev/Documents/GitHub/tldw_server/.worktrees/rag-module-review/tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py)
- [`agentic_chunker.py`](/Users/appledev/Documents/GitHub/tldw_server/.worktrees/rag-module-review/tldw_Server_API/app/core/RAG/rag_service/agentic_chunker.py)
- [`agentic_execution.py`](/Users/appledev/Documents/GitHub/tldw_server/.worktrees/rag-module-review/tldw_Server_API/app/core/RAG/rag_service/agentic_execution.py)
- [`analytics_db.py`](/Users/appledev/Documents/GitHub/tldw_server/.worktrees/rag-module-review/tldw_Server_API/app/core/RAG/rag_service/analytics_db.py)
- RAG tests changed in that phase

## Risks

### Risk 1: Hidden reliance on ad hoc request shapes

The current core may still rely on reconstructed request-like objects in more places than the current review captured.

Mitigation:

- add request-object identity tests before removing reconstruction
- migrate standard path first before touching agentic and streaming

### Risk 1b: Review branch drift from current `dev`

The review branch is no longer a complete proxy for the current application state. `dev` has already landed RAG-adjacent fixes and tests after the branch split.

Mitigation:

- add the explicit Phase 0 `dev` reconciliation gate
- run branch-vs-`dev` checkpoints after every phase
- treat restored `dev` tests as part of the living contract unless deliberately replaced

### Risk 2: Compatibility seams hidden inside agentic shell behavior

Some tests or runtime patch seams still rely on the chunker module’s exported surface.

Mitigation:

- preserve compatibility aliases temporarily if necessary
- move ownership first, then retire aliases with explicit tests

### Risk 3: Streaming extraction can break event order or response semantics

Streaming is the most behavior-sensitive seam.

Mitigation:

- extract only after standard and agentic contracts are stable
- keep explicit stream parity tests and event-order tests

## Artifacts

Expected follow-up plan artifact:

- [`Docs/superpowers/plans/2026-04-24-rag-thin-endpoint-followup-plan.md`](/Users/appledev/Documents/GitHub/tldw_server/.worktrees/rag-module-review/Docs/superpowers/plans/2026-04-24-rag-thin-endpoint-followup-plan.md)

## Definition Of Done

This umbrella follow-up is complete when:

- standard core execution uses canonical request contracts end-to-end
- agentic execution no longer depends back on `agentic_chunker.py`
- streaming execution has a core executor boundary
- `rag_unified.py` is transport-oriented across standard, agentic, streaming, batch, and resume
- cleanup, parity, and regression tests pin the new ownership seams
- external HTTP contracts remain compatible aside from any explicit additive or deprecation-marked cleanup
