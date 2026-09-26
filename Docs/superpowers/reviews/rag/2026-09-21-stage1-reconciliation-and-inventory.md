# Stage 1 (2026-09-21) Prior-Findings Reconciliation and Refreshed Inventory

## Scope

Re-open the 2026-04-07 RAG ledger against today's code. Two jobs:

1. Re-verify every 2026-04-07 finding and classify it **still-live** or **already-addressed**. The prior ledger is ~5 months old and `unified_pipeline.py` has taken 42 commits since.
2. Refresh the size/churn baseline and state what the 2026-04-07 review could not have covered, so the two new stage files do not re-litigate settled ground.

This pass runs the five-axis rubric from the 2026-09-21 repo-wide briefing. **Axis 5 (efficiency) did not exist in the 2026-04-07 review** — that review ran the three-axis Codeslop rubric with performance explicitly excluded. Efficiency findings in this module are therefore unclaimed territory by construction, and Stage 2 (2026-09-21) is mostly that.

This review is read-only. No source file was modified.

## Code Paths Reviewed

- `tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py:unified_rag_pipeline (2051-9077)`
- `tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py:_build_cache_namespace (2825-2908)`
- `tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py:_explicit_include_data_sources (1818-1830)` and its consumer at `(3046-3086)`
- `tldw_Server_API/app/core/RAG/rag_service/retrieval_executor.py:execute_retrieval_phase (13-90)`
- `tldw_Server_API/app/core/RAG/rag_service/database_retrievers.py:MultiDatabaseRetriever.retrieve (4724-4932)`
- `tldw_Server_API/app/core/RAG/rag_service/database_retrievers.py:MediaDBRetriever.retrieve (1240-1285)`
- `tldw_Server_API/app/core/RAG/rag_service/advanced_reranking.py:create_reranker (1782-1820)`
- `tldw_Server_API/app/core/RAG/rag_service/resilience.py:RetryPolicy (230-290)`
- `tldw_Server_API/app/core/RAG/rag_service/batch_utils.py:run_batch (84-188)`, `run_batch_indexed (190-277)`
- `tldw_Server_API/app/core/RAG/rag_service/utils.py:TokenCounter (16-48)`, `normalize_scores (162-195)`
- `tldw_Server_API/app/core/RAG/rag_service/web_fallback.py:_truncate_content_by_tokens (50-91)`
- `tldw_Server_API/app/api/v1/utils/rag_cache.py` (ownership re-check only)
- `tldw_Server_API/app/core/DB_Management/scope_context.py:content_authorization_cache_scope (59-80)`

## Tests Reviewed

Located by import-grep, never by path, per the briefing.

- `tldw_Server_API/tests/RAG_NEW/unit/test_retrieval_plan_usage.py` — proves the pipeline routes through `retrieval_plan`/`execute_retrieval_phase` and degrades when `batch_utils` import fails (`:333-334`). Downgrades the risk that the new retrieval seam is decorative.
- `tldw_Server_API/tests/RAG_NEW/unit/test_rag_unified_search_agent_defaults.py` — still the main precedence guard named by the 2026-04-07 Stage 3.
- `tldw_Server_API/tests/RAG/test_batch_utils.py` — the **only** caller of `run_batch_indexed` anywhere in the repo.
- `tldw_Server_API/tests/RAG/test_resilience_sanitizers.py:70-71` — the only direct exerciser of `resilience.RetryPolicy`.
- `tldw_Server_API/tests/RAG/conftest.py:dual_backend_env (141-199)` — the only sqlite/postgres-parametrized RAG fixture; used by exactly two test files.
- `tldw_Server_API/tests/RAG_NEW/conftest.py:42-107` — four autouse isolation fixtures that do **not** apply to `tests/RAG`.

This is import-grep reachability, not measured coverage. The suite was not executed.

## Validation Commands

```bash
find tldw_Server_API/app/core/RAG -name '*.py' | xargs wc -l | sort -rn | head -3
```
Observed: `58804 total`; `9586 .../unified_pipeline.py`; `5280 .../database_retrievers.py`.
The 2026-04-07 baseline recorded 6977 and 3590 for the same two files. Delta: **+2609 and +1690 lines in ~5 months.**

```bash
git log --oneline --since='2026-04-07' -- tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py | wc -l
```
Observed: `42`.

```bash
awk 'NR>=2051 && NR<9077' tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py | grep -cE '^[[:space:]]+(async )?def '
```
Observed: `49` (nested function definitions inside the single `unified_rag_pipeline` body).

```bash
grep -c "except ImportError" tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py
```
Observed: `52`.

```bash
grep -rl "core\.RAG" tldw_Server_API/tests | wc -l
find tldw_Server_API/tests/RAG -name 'test_*.py' | wc -l
find tldw_Server_API/tests/RAG_NEW -name 'test_*.py' | wc -l
```
Observed: `222`, `47`, `143`.

```bash
git log --since='12 months ago' --oneline -- tldw_Server_API/tests/RAG | wc -l
git log --since='12 months ago' --oneline -- tldw_Server_API/tests/RAG_NEW | wc -l
```
Observed: `98`, `255`. Both trees are actively maintained; neither is a frozen legacy tree.

```bash
ls Docs/ADR/ | grep -iE 'rag|rerank|retriev|vector'
```
Observed: no output. There is no binding ADR governing RAG retrieval, reranking, or scoring, so no finding in this ledger contradicts one.

## Findings

### A. Prior findings re-verified as STILL-LIVE

- **2026-04-07 Stage 2 #1 — `unified_rag_pipeline()` is policy engine, phase controller, and recovery loop. STILL-LIVE and materially worse.** The claim was made against a 6977-line file. Today `unified_rag_pipeline` is a **single function spanning `unified_pipeline.py:2051-9077` — 7,026 lines — containing 49 nested `def`s**. The function body alone is now larger than the whole file was when the prior review called it a god module. Post-verification still recursively re-enters the pipeline (`unified_pipeline.py:5957-6072` region), and numeric-fidelity retry still re-retrieves and merges (`unified_pipeline.py:5742-5749`). Confidence: confirmed.
- **2026-04-07 Stage 2 #3 / Stage 3 #3 — metadata-derived result contract. STILL-LIVE.** `UnifiedSearchResult` is still declared in the orchestrator (`unified_pipeline.py:1360`), and endpoint-side mapping still re-derives response fields from `result.metadata`. Confidence: confirmed.
- **2026-04-07 Stage 4 #1/#2 — retrieval policy fixes inside the concrete retriever, and `MultiDatabaseRetriever` dispatches on concrete retriever classes and private methods. STILL-LIVE.** `MultiDatabaseRetriever.retrieve (4724-4932)` still `isinstance`-switches on `MediaDBRetriever`, `NotesDBRetriever`, `CharacterCardsRetriever` and reaches into `_retrieve_vector`/`_retrieve_fts` at `database_retrievers.py:4807-4880`. Confidence: confirmed.
- **2026-04-07 Stage 4 #3 — `user_{user_id}_media_embeddings` rebuilt by convention in several layers. STILL-LIVE.** Still hardcoded at `database_retrievers.py:2141`. Confidence: confirmed.
- **2026-04-07 Stage 3 #4 — `app/api/v1/utils/rag_cache.py` owns core RAG cache semantics and is imported by core worker code. STILL-LIVE.** Still a core -> `app/api/v1/utils` inversion. Confidence: confirmed. Note this is the *mild* shape in the briefing's layering taxonomy (a utility, not an endpoint), but it is a utility that owns domain policy, so it is worse than a schema-only import.
- **2026-04-07 Stage 5 #1/#2 — post-retrieval stages replace the working document set; guardrails are leaf heuristics but hidden orchestrators at their call sites. STILL-LIVE.** Confidence: confirmed.
- **2026-04-07 Stage 6 Coverage Gap #1 — no test pins a one-way phase graph. STILL-LIVE.** No structural ownership test appeared in the 42 intervening commits.

### B. Prior findings that are PARTIALLY ADDRESSED

- **2026-04-07 Stage 6 Action #1 ("define one canonical contract chain: resolved request -> retrieval plan -> retrieved evidence") — PARTIALLY SHIPPED.** Four modules now exist that did not appear anywhere in the 2026-04-07 inventory: `rag_service/request_resolution.py` (`ResolvedRAGRequest`), `rag_service/retrieval_plan.py` (`RetrievalPlan`), `rag_service/evidence_models.py` (`RetrievedEvidence`), and `rag_service/retrieval_executor.py:execute_retrieval_phase (13-90)`. The main pipeline genuinely routes base retrieval and all four fan-out variants through `execute_retrieval_phase` (`unified_pipeline.py:4348-4380`). **The contract chain exists; the orchestrator did not shrink.** The extraction added a seam without removing the surface behind it — the recommended fix was applied additively, which is why the file grew 2,609 lines while gaining the abstraction that was supposed to shrink it. That inversion is the single most useful thing to know before planning the next round.
- **2026-04-07 Stage 5 #1 (agentic fork) — PARTIALLY ADDRESSED.** `agentic_execution.py` (1030 lines) now exists as a separate module, so the agentic path is no longer only inside `agentic_chunker.py`. The synthetic-document evidence model the prior stage objected to is unchanged.

### C. Prior concern that is now ADDRESSED — do not re-report

- **Per-request scope filters versus the semantic cache.** I opened this as a suspected cache-key omission: `_build_cache_namespace (2825-2908)` keys on owner, workspace, and a `retrieval_scope` of sources/search-mode/top_k/min_score/fts_level/date_range/late-chunking/namespace/collections, and `content_authorization_cache_scope` (`DB_Management/scope_context.py:59-80`) keys only on identity and RBAC. Neither includes `include_media_ids`/`include_note_ids`. **It is not a defect:** `unified_pipeline.py:3060-3065` sets `enable_cache = False` and `retrieval_cache_eligible = False` whenever an explicit include-list is present, recording `cache_bypassed: {"reason": "explicit_source_selection"}`. The scope filter cannot be served from cache. Recorded here so the next reviewer does not spend the same pass on it.

### D. What the 2026-04-07 review could not have covered

- **Axis 5 (efficiency) was excluded from that review's rubric.** Every efficiency finding in `2026-09-21-stage2-efficiency-and-correctness.md` is new by construction, not by oversight.
- **Axis 1 (duplication with a named canonical destination) was not the prior review's frame.** That review produced ownership and boundary findings, not `adoption-gap` / `divergent-copies` classifications. `2026-09-21-stage3-duplication-and-test-topology.md` covers that axis.
- **The prior Stage 1 noted that `tests/RAG` and `tests/RAG_NEW` "should both remain visible" but never assessed the split itself.** Stage 3 does.

### E. Refreshed inventory (facts, not findings)

- Module: 95 Python files, 58,804 LOC.
- Top five by size: `unified_pipeline.py` 9,586; `database_retrievers.py` 5,280; `advanced_reranking.py` 2,075; `research_agent.py` 1,681; `analytics_system.py` 1,282.
- Top five by 12-month churn: `unified_pipeline.py` 105; `database_retrievers.py` 66; `agentic_chunker.py` 37; `generation.py` 28; `semantic_cache.py` 27.
- Size x churn puts `unified_pipeline.py` and `database_retrievers.py` far ahead of everything else. Both new stage files concentrate there, with `advanced_reranking.py` third because it is where the per-request cost lives.
- Full captures: [`2026-09-21-stage1-hotspot-sizes.txt`](./2026-09-21-stage1-hotspot-sizes.txt), [`2026-09-21-stage1-churn-baseline.txt`](./2026-09-21-stage1-churn-baseline.txt).

## Suggested Refactor/Actions

1. Treat the 2026-04-07 Stage 6 Action #1 as **half-done, not open**. The contract chain shipped; the follow-through — moving phase bodies *behind* `execute_retrieval_phase` and friends and deleting them from `unified_rag_pipeline` — did not. Re-scoping the remaining work as "migrate callers off the monolith" is a different and much more tractable task than the original "define a contract chain."
2. Do not re-file the still-live Stage 2/4/5 ownership findings as new work. Point any new Backlog task at the 2026-04-07 stage files as the canonical statement and use this section only as the re-verification date stamp.
3. The `Media_DB_v2.py` -> `media_db/` package split is the in-repo template for what to do with `unified_pipeline.py`: same team, same layer, already shipped, a monolith with comparable churn. Read `core/DB_Management/media_db/` package boundaries before proposing new ones, rather than inventing a shape.

## Exit Note

Stage 1 (2026-09-21) does not overturn any 2026-04-07 finding. It re-dates them, records one that is now genuinely closed (the scope-filter/cache interaction), records one that is half-shipped in a way that made the headline metric worse, and hands the two unclaimed axes — efficiency and duplication — to the two following stage files.
