# Political PDF remaining retrieval and vector acceptance

Task: TASK-15505.7. Approved design: [retrieval/vector follow-up](../../Design/POLITICAL_PDF_VIP_FIXES_2026_11_15.md#remaining-multifacet-retrieval-and-local-vector-acceptance-task-155057).

ADR assessment: reuse ADR-022/060 for Embeddings and Jobs, ADR-053 for fusion, and ADR-017/052 for existing authority. No new ADR or storage owner is required.

### Live vector adapter correction

The first real MiniLM index completed with 431 vectors. Public vector search exposed three translation errors: chunk IDs became the parent Media ID, stored `chunk_start`/`chunk_end` were not normalized into common offsets, and L2 distances were treated as cosine distances. Fix these boundaries before accepting vector/hybrid results. Keep raw distances, honor the collection's actual distance metric, and return bounded similarity scores. Preserve vector chunk identity separately from `media_id`. Normalize offset aliases and use the existing scoped canonical context verifier to derive page spans; when expansion is disabled, verify with a zero-character budget. Never trust stale vector page metadata or change access rules/context caps. Test a real local Chroma collection, exact canonical Media pages, disabled/enabled expansion, distinct chunks from one source, stale/hidden controls, and the full public endpoint. Parent owns runtime restarts and model-backed acceptance.

## Stage 1: Preserve separate question facets
**Goal**: Recover short-acronym evidence without increasing context or search limits.
**Success Criteria**: The original FBI question retains both review and receiving-office passages through the public pipeline; unrelated source rows and explicit search syntax remain protected.
**Tests**: RED then GREEN for normalized citation directives, uppercase acronym controls, bounded multifacet persisted retrieval, full endpoint evidence/page/security output and cache-policy rollover.
**Status**: Complete — RED 7; combined 165 GREEN, then 9 final parser/property/public-pipeline checks; independent review and live post-restart replay coordinated by parent.

## Stage 2: Verify existing local embeddings
**Goal**: Use the cached MiniLM model through existing local embedding and Jobs paths.
**Success Criteria**: Offline loader/inference succeeds; coordinated isolated configuration indexes the same PDF and reports vector readiness with actual vector evidence.
**Tests**: Local model smoke without downloads, durable job completion/count/status checks, source-scoped vector and hybrid public queries.
**Status**: In Progress — cached MiniLM offline loader/inference passed (2 × 384 finite vectors, 3.237 seconds); parent owns runtime configuration and index mutation.

## Stage 3: Evaluate coverage and record limits
**Goal**: Distinguish fixed regressions, held-out retrieval coverage and generated claim faithfulness.
**Success Criteria**: Frozen source-audited held-out cases report per-facet/all-facet coverage and latency by text/vector/hybrid mode; actual answer checks remain separate.
**Tests**: Public pipeline replay of original failures and held-out questions; parent-coordinated real generation only after evidence acceptance. Focused lint/Bandit and independent review precede handoff.
**Status**: Not Started
