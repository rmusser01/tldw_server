# RAG Module Review Design

Date: 2026-04-07
Topic: RAG module architecture and maintainability review
Status: Approved design

## Objective

Review the RAG module in `tldw_server` for architecture and maintainability issues with a broad-first, sequential audit that leaves a written record for each review slice.

The review should prioritize substantive structural findings over generic style commentary, keep evidence-backed issues separate from softer risks, and include concrete refactor or action guidance for each stage record.

## Scope

This review is centered on:

- `tldw_Server_API/app/core/RAG`
- `tldw_Server_API/app/api/v1/endpoints/rag_unified.py`
- `tldw_Server_API/app/api/v1/endpoints/rag_health.py`
- `tldw_Server_API/app/api/v1/schemas/rag_schemas_unified.py`
- RAG-focused tests and only those callers needed to confirm intended ownership boundaries or architectural assumptions

This includes:

- the broad RAG package structure and dependency concentration
- unified pipeline orchestration and its adjacent feature seams
- endpoint, schema, profile-default, and request-mapping boundaries
- retrieval composition across databases, vector stores, and source adapters
- reranking and post-retrieval composition including citations, guardrails, grading, and verification touchpoints
- architectural test coverage and where the current test layout masks structural risk
- direct boundary or support modules that materially control RAG request defaults, type contracts, or cache behavior when they are on an active RAG call path

This review excludes:

- unrelated storage, billing, admin, or evaluation code except where a boundary must be inspected to confirm RAG ownership
- remediation or refactoring during the review itself unless explicitly requested later
- equal-depth treatment of every low-risk helper module in the RAG tree

## Approaches Considered

### Recommended: Staged review ledger

Run a broad architecture survey across the full RAG tree first, then audit the highest-risk boundaries in ordered slices with one markdown record per slice and a final synthesis.

Why this is preferred:

- matches the requested broad-first then deep-sequential workflow
- creates a durable review history instead of one blended report
- keeps findings and suggested actions attributable to a specific slice
- reduces the chance that later deep-dive findings overwrite or blur earlier broad-architecture conclusions

### Alternative: Monolithic review report

Produce one large architecture review document and append deeper findings into the same file.

Trade-offs:

- faster to start
- weaker traceability for sequential review slices
- higher risk of duplicated or contradictory findings as the audit deepens

### Alternative: Dependency-map-first audit

Build a formal dependency and coupling map before reviewing any individual hotspot deeply.

Trade-offs:

- strongest up-front systems picture
- slower path to first useful review artifact
- more front-loaded documentation than necessary for the requested audit

## Chosen Method

Use the recommended staged review ledger with six ordered stages:

1. `Architecture survey and inventory`
   Review the full RAG tree, identify major seams, hotspot files, dependency concentration, recent churn, and early architectural concerns.
2. `Unified pipeline orchestration`
   Review `unified_pipeline.py` as the central orchestration layer, focusing on orchestration overload, parameter sprawl, hidden phase coupling, and response-shaping leakage, along with adjacent shared type or contract modules when they materially shape orchestration behavior.
3. `API, schema, and request boundaries`
   Review `rag_unified.py`, `rag_health.py`, `rag_schemas_unified.py`, `rag_schemas_simple.py`, and `profiles.py` for ownership drift across schema defaults, endpoint defaults, profile resolution, request construction, and response mapping.
4. `Retrieval boundaries and data sources`
   Review retriever composition, query-expansion hooks, data-source abstractions, vector-store seams, and database-facing boundaries for mixed responsibilities and weak extension points.
5. `Reranking and post-retrieval composition`
   Review reranking, generation, citations, guardrails, grading, verification, response writing, and related post-retrieval modules for feature tangling and ambiguous ownership.
6. `Test gaps and synthesis`
   Compare the reviewed architecture against the test surface, identify missing structural protections, and produce one prioritized synthesis.

## Evidence Model

The review will rely on:

- direct source inspection across the scoped RAG module and direct API boundaries
- direct inspection of relevant RAG-focused tests
- targeted caller inspection where ownership or architectural intent is otherwise unclear
- recent git history when it helps explain churn, regression patterns, or likely hotspots

The default git-history baseline should start with the last 20 commits touching:

- `tldw_Server_API/app/core/RAG`
- `tldw_Server_API/app/api/v1/endpoints/rag_unified.py`
- `tldw_Server_API/app/api/v1/schemas/rag_schemas_unified.py`

The review is primarily static and read-first. Runtime verification may be used selectively when it materially strengthens or falsifies a concrete architectural claim, such as hidden coupling enforced only by runtime behavior or tests that do not reflect the actual ownership boundary.

## Findings Model

The stage reports and final synthesis should include only substantive items.

Each item should contain:

- severity
- confidence
- concise statement of the issue
- why it matters
- concrete file and line references

Observations should be classified as:

### Confirmed finding

A maintainability or architectural defect supported directly by source, tests, or control-flow evidence.

### Probable risk

A likely structural problem whose impact depends on assumptions not fully proven within the scoped review.

### Improvement

A concrete refactor or simplification opportunity that is not strong evidence of a current defect but would reduce coupling, churn, or regression risk.

Each stage report should write findings before suggested actions. Suggested actions should stay concrete and localized rather than turning into a broad rewrite plan.

## Severity Model

Severity should balance these architecture-focused axes:

- blast radius across RAG features or request paths
- ownership ambiguity between layers
- change friction and cost of safe modification
- regression likelihood from the current structure
- difficulty of testing the affected boundary safely

When issues are otherwise similar, rank higher the one that affects more layers, spreads configuration or control-flow knowledge across more modules, or makes future changes harder to isolate.

## Review Focus Areas

The audit should bias toward:

- oversized orchestrators that mix coordination with leaf behavior
- leakage between endpoint, schema, pipeline, retrieval, reranking, generation, and response-shaping layers
- parameter and configuration sprawl that obscures effective behavior
- duplicated or drifting data models across schemas, internal types, and response mappers
- optional-feature tangles where new features increase coupling across unrelated phases
- helper modules that silently take on orchestrator responsibilities
- docs, examples, or tests that no longer match the current architecture
- structural test gaps caused by over-mocking or only validating happy-path wiring

## Representative Selection and Routing Rule

Because `tldw_Server_API/app/core/RAG` is too large for equal-depth review, the audit should use an explicit seed set and a routing rule for newly discovered hotspots.

The initial seed set should include at minimum:

- `tldw_Server_API/app/core/RAG/README.md`
- `tldw_Server_API/app/core/RAG/rag_service/README.md`
- `tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py`
- `tldw_Server_API/app/core/RAG/rag_service/database_retrievers.py`
- `tldw_Server_API/app/core/RAG/rag_service/advanced_reranking.py`
- `tldw_Server_API/app/core/RAG/rag_service/generation.py`
- `tldw_Server_API/app/core/RAG/rag_service/guardrails.py`
- `tldw_Server_API/app/core/RAG/rag_service/citations.py`
- `tldw_Server_API/app/core/RAG/rag_service/profiles.py`
- `tldw_Server_API/app/core/RAG/rag_service/types.py`
- `tldw_Server_API/app/core/RAG/rag_service/semantic_cache.py`
- `tldw_Server_API/app/core/RAG/rag_service/vector_stores/factory.py`
- `tldw_Server_API/app/core/RAG/rag_service/agentic_chunker.py`
- `tldw_Server_API/app/core/RAG/rag_service/research_agent.py`
- `tldw_Server_API/app/api/v1/endpoints/rag_unified.py`
- `tldw_Server_API/app/api/v1/endpoints/rag_health.py`
- `tldw_Server_API/app/api/v1/schemas/rag_schemas_unified.py`
- `tldw_Server_API/app/api/v1/schemas/rag_schemas_simple.py`
- `tldw_Server_API/app/api/v1/utils/rag_cache.py`

If Stage 1 identifies a hotspot outside this seed set, that module must be assigned explicitly to one later stage with a short rationale recorded in the Stage 1 exit note. No important secondary seam should be left unowned between stages.

## Sequential Review Artifacts

The review should be recorded in:

- `Docs/superpowers/reviews/rag/README.md`
- `Docs/superpowers/reviews/rag/2026-04-07-stage1-architecture-survey-and-inventory.md`
- `Docs/superpowers/reviews/rag/2026-04-07-stage2-unified-pipeline-orchestration.md`
- `Docs/superpowers/reviews/rag/2026-04-07-stage3-api-schema-and-request-boundaries.md`
- `Docs/superpowers/reviews/rag/2026-04-07-stage4-retrieval-boundaries-and-data-sources.md`
- `Docs/superpowers/reviews/rag/2026-04-07-stage5-reranking-and-post-retrieval-composition.md`
- `Docs/superpowers/reviews/rag/2026-04-07-stage6-test-gaps-and-synthesis.md`

Each stage record should use this structure:

- `Scope`
- `Code Paths Reviewed`
- `Tests Reviewed`
- `Validation Commands`
- `Findings`
- `Suggested Refactor/Actions`
- `Coverage Gaps`
- `Exit Note`

## Execution Boundaries

- The review remains inside the RAG subsystem and its direct API boundaries.
- Cross-module behavior may be noted only when a local RAG file clearly depends on it.
- The review remains non-invasive and should not silently turn into remediation work.
- The final output is a review and action ledger, not an implementation plan.

## Final Deliverable

The canonical review record will be the six stage files above plus a final synthesis stage. The final response to the user will summarize the highest-signal findings in code-review style, ordered by severity, and point back to the stage artifacts rather than replacing them.

Each finding will include:

- severity and confidence
- a short explanation of the structural problem or maintainability risk
- why it matters in terms of blast radius, ownership, or regression risk
- file and line references

The report may include confirmed findings, probable risks, and improvements. If a reviewed area appears healthy, the review may say so briefly rather than inventing debt. If uncertainty remains, it should be labeled as an assumption or open question rather than overstated as a confirmed defect.

## Success Criteria

The review is successful when:

- the broad survey covers the full RAG tree without pretending every file received equal depth
- each deep slice has its own stable, written record
- findings are evidence-backed and separated from softer risks
- each stage includes concrete suggested refactor or action guidance after the findings
- severity reflects architecture and maintainability impact rather than style preferences
- the final synthesis is usable as a triage artifact for follow-up remediation planning

## Constraints

- Do not broaden this run into a whole-application review.
- Do not silently convert the review into implementation work.
- Do not present speculation as a confirmed structural defect.
- Do not over-index on naming, formatting, or low-signal cleanup concerns.

## Expected Outcome

This design yields a broad, high-signal architecture and maintainability review of the RAG subsystem, followed by sequential deep audits of its highest-risk seams, with a durable per-slice record and concrete follow-up guidance that can later feed implementation planning.
