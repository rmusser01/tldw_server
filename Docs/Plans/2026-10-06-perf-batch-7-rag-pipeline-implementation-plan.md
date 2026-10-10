# Backend Performance Remediation — Batch 7: RAG Pipeline Depth Implementation Plan (2026-10-06)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan stage-by-stage. Steps use checkbox (`- [ ]`) syntax for tracking.

**Backlog task:** TASK-13519 · **Index:** [2026-10-06-perf-remediation-coordination-index.md](2026-10-06-perf-remediation-coordination-index.md)

**Goal:** Fix the RAG-specific hot-path costs: the per-request reranker model load (the single biggest per-search win — Stage 1 is independently pull-forwardable), sequential expansion retrieval, redundant query embeddings, MMR/LLM reranker inefficiencies, linear semantic-cache scans, and the late-chunking / dedup quadratics.

**Architecture:** Lifecycle management for expensive objects (singleton registry), concurrency for independent retrieval variants, memoization for repeated embeddings, and algorithm swaps for the quadratic inner loops. Search result quality must be unchanged — every stage that touches ranking has a parity test.

**Tech Stack:** `asyncio.gather`, `functools.lru_cache`, NumPy matrix ops, SimHash banding, existing reranker classes.

## Global constraints

- Backend only; activate venv; one commit per stage referencing TASK-13519.
- Ranking-parity: any stage touching rerank/fusion order gets a test asserting identical document ordering pre/post on a fixed fixture (record golden from pre-change code first).
- Singleton rerankers must be stateless across `rerank()` calls — Stage 1 verifies this before sharing instances.
- External calls (embeddings, LLM) mocked in tests; flashrank/cross-encoder weights cached from the repo's existing cache dir (no network in CI — skip model-loading assertions if weights are absent, via `pytest.mark.skipif` on cache-dir existence).
- Bandit: `python -m bandit -r tldw_Server_API/app/core/RAG tldw_Server_API/app/core/Embeddings/ChromaDB_Library.py -f json -o /tmp/bandit_perf_b7.json`.
- Line numbers from the 2026-10-06 review; re-locate by symbol.

---

## Stage 1: Reranker singleton registry ⭐ (independently pull-forwardable)

**Goal:** `create_reranker` stops deserializing the model from disk on every search.

**Files:**
- Modify: `tldw_Server_API/app/core/RAG/rag_service/advanced_reranking.py` (`create_reranker` ~1796-1820)
- Modify: `tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py` (call sites ~6293, ~1561, ~6333)
- Test: `tldw_Server_API/tests/RAG/test_reranker_registry.py` (new)

**Change:**
1. Audit each reranker class for per-call mutable state in `rerank()` (grep for `self.` assignments inside `rerank` and helpers). Any stateful reranker gets its state reset at `rerank()` entry instead of relying on `__init__`.
2. Module-level `_RERANKER_REGISTRY: dict[tuple, BaseReranker]` keyed `(strategy, model_name, cache_dir, device)`; `create_reranker` returns the shared instance (double-checked under a `threading.Lock` — construction may run in threads).
3. `reset_reranker_registry()` for tests and for config-change invalidation (call it from the config refresh path if rerank model config is editable at runtime — grep `refresh_config_cache` consumers).

**Tests:**
- [ ] `test_create_reranker_returns_singleton` — same key → `is` same instance; different model → different instance.
- [ ] `test_rerank_output_parity` — fixed fixture of 20 documents + query; ordering identical to a freshly-constructed reranker's output (skipif weights not cached).
- [ ] `test_registry_reset` — after reset, next create constructs a new instance (constructor counter).

**Status:** Not Started

## Stage 2: Parallel query-expansion retrieval

**Goal:** Expansion variants stop executing sequentially (2-5× wall-time multiplier when enabled).

**Files:**
- Modify: `tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py` (~4491-4508)
- Test: `tldw_Server_API/tests/RAG/test_expansion_gather.py` (new)

**Change:** build coroutines per variant and `asyncio.gather` them — mirror the followup-search pattern at ~5229-5234 (including its exception handling: a failed variant should degrade the same way the sequential loop degrades, i.e., partial results, logged, not fatal).

**Tests:**
- [ ] `test_expansion_variants_concurrent` — mocked retrieval with 100ms latency × 3 variants → wall time < 250ms.
- [ ] `test_variant_failure_partial_results` — one variant raises → others' documents present, exception logged (parity with sequential semantics).
- [ ] `test_merged_results_identical` — same merged doc set/order as sequential reference on fixture.

**Status:** Not Started

## Stage 3: Query-embedding memoization

**Goal:** The same query string stops being re-embedded multiple times per request (main pass + variants + followups + HyDE) and across requests.

**Files:**
- Modify: `tldw_Server_API/app/core/RAG/rag_service/database_retrievers.py` (~2189-2206 embedding call; `retrieve_hybrid` ~2553)
- Test: `tldw_Server_API/tests/RAG/test_query_embedding_lru.py` (new)

**Change:**
1. Per-request memo dict threaded through the retrieval context (variants hit it within one pipeline run).
2. Process-level bounded LRU (128 entries) keyed `(model_id, hashlib.sha256(text).digest())` → vector; `clear_query_embedding_cache()` exported for tests/rotation. LRU stores numpy arrays; mind memory (128 × 3072 floats ≈ 1.5MB — acceptable).
3. Do **not** cache failed embeddings.

**Tests:**
- [ ] `test_same_query_embedded_once_per_request` — mocked embed call count == 1 across main + 2 variants + followup.
- [ ] `test_cache_hit_across_requests` — two requests, same query/model → 1 embed call; different model → 2.
- [ ] `test_failures_not_cached` — failing embed retried on next call.

**Status:** Not Started

## Stage 4: MMR word-set precompute + LLM-reranker batching

**Goal:** DiversityReranker stops re-tokenizing both documents per pairwise comparison; LLMReranker stops scoring one passage per round trip.

**Files:**
- Modify: `tldw_Server_API/app/core/RAG/rag_service/advanced_reranking.py` (`DiversityReranker._mmr`-style selection loop ~1228-1278; `LLMReranker._score_batch` ~1646-1679)
- Test: `tldw_Server_API/tests/RAG/test_reranker_inner_loops.py` (new)

**Change:**
1. Precompute `word_sets = [frozenset(d.content.lower().split()) for d in documents]` once before the selection loop; `_compute_similarity` overload accepting sets; replace `remaining_indices.remove(idx)` list scans with set-based bookkeeping.
2. `LLMReranker`: one prompt listing all passages (numbered, each truncated per current 1500-char rule) requesting a numeric score per passage; parse per-passage scores; on any parse failure fall back to the existing sequential path. Keep the 20-doc cap and the overall time budget.

**Tests:**
- [ ] `test_mmr_selection_parity` — fixture 50 docs, identical selected order vs reference implementation (copy of old loop in test).
- [ ] `test_llm_reranker_single_call` — 10 passages → 1 LLM call; scores match a mocked response.
- [ ] `test_llm_parse_failure_falls_back` — malformed LLM output → sequential fallback invoked, results still returned.

**Status:** Not Started

## Stage 5: Semantic cache — vectorized similarity

**Goal:** `find_similar` stops looping `np.linalg.norm` over per-entry vectors.

**Files:**
- Modify: `tldw_Server_API/app/core/RAG/rag_service/semantic_cache.py` (~295-302, ~325-332)
- Test: `tldw_Server_API/tests/RAG/test_semantic_cache_vectorized.py` (new)

**Change:** maintain a stacked `(size × d)` matrix alongside the dict (numpy vstack on insert, drop rows on evict); lookup = `sims = matrix @ embedding` (vectors are unit-normalized at insert — verify with an insert-time assert); best = `argmax` above threshold. Keep eviction semantics identical.

**Tests:**
- [ ] `test_find_similar_parity` — 500 cached entries, random queries: best key and similarity equal the loop implementation's (tolerance 1e-9).
- [ ] `test_insert_normalization_assert` — non-unit vector normalized at insert (existing behavior, now asserted).

**Status:** Not Started

## Stage 6: Late-chunking fallback — bounded, cached, cheaper fuzzy

**Goal:** The late-chunking fallback path stops re-chunking 20 full documents synchronously with quadratic fuzzy scoring.

**Files:**
- Modify: `tldw_Server_API/app/core/RAG/rag_service/database_retrievers.py` (`Chunker()` per request ~1430-1448; chunk loop ~1462-1500; `_score_media_chunk_text` ~373-382)
- Test: `tldw_Server_API/tests/RAG/test_late_chunking_bounds.py` (new)

**Change:**
1. Module-level shared `Chunker` (lazy singleton under a lock; `reset_chunker()` for tests).
2. Move the chunking loop to `asyncio.to_thread` (it is sync CPU work inside async retrieval).
3. Cap late-chunked docs per query at a config default of 5 (`RAG_LATE_CHUNK_MAX_DOCS`), document the behavior change in task notes.
4. `_score_media_chunk_text`: compute the chunk's token set once and pass it across all query terms (per-chunk cache dict); replace `SequenceMatcher` with a length-gated early-exit check (the existing `abs(len(doc_term) - len(term)) > 2` skip stays; add a cheap prefix/hash gate before SequenceMatcher, and cap SequenceMatcher to terms ≤ 32 chars). Preserve scoring output within tolerance (parity test at 1e-6).

**Tests:**
- [ ] `test_shared_chunker_reused` — 2 searches → 1 Chunker construction (mock counter).
- [ ] `test_late_chunk_docs_capped` — 20 matched docs → ≤ 5 chunked.
- [ ] `test_scoring_parity` — fixture chunks × terms: new scores == old scores (reference copy) within 1e-6.

**Status:** Not Started

## Stage 7: Ingest dedup — SimHash banding

**Goal:** `_dedupe_text_chunks` stops comparing every chunk against every retained chunk.

**Files:**
- Modify: `tldw_Server_API/app/core/Embeddings/ChromaDB_Library.py` (~1698-1718)
- Test: `tldw_Server_API/tests/Embeddings/test_dedup_banding.py` (new)

**Change:**
1. Bucket retained chunks by SimHash bands (4 × 16-bit prefixes); candidate pairs = band collisions only; Jaccard + hamming checks run within collision buckets (same thresholds and tie-breaking as today).
2. Chunks whose SimHash is 0 (non-simhash mode) keep the current path but are gated behind the existing `use_simhash` flag — if simhash is disabled by config, keep exact old behavior and skip banding (document this).
3. Preserve duplicate-detection parity: near-identical chunks (Jaccard ≥ threshold within hamming gate) must be flagged identically.

**Tests:**
- [ ] `test_deduup_parity_on_fixture` — 1,000 synthetic chunks (10% near-dupes planted): flagged set identical to old implementation.
- [ ] `test_banding_beats_quadratic_counter` — candidate-pair comparison counter ≤ banded bound (assert counter < n²/10 on the fixture).

**Status:** Not Started

## Stage 8: Cap rerank candidates at top-k

**Goal:** The reranker never scores more candidates than it can return.

**Files:**
- Modify: `tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py` (pre-rerank slice, ~6293 block)
- Test: extend `tldw_Server_API/tests/RAG/test_reranker_registry.py`

**Change:** before `reranker.rerank`, slice `result.documents[: max(rerank_top_k or top_k, rerank_candidate_floor)]` where floor defaults to `top_k` (verify config semantics first: if `rerank_top_k` can exceed the retrieved count this is a no-op; if profiles rely on reranking the full 50-100 hybrid candidates to *select* top-k, set the cap to `max(rerank_top_k * 2, top_k)` and record the chosen rule in task notes — the parity test guards it).

**Tests:**
- [ ] `test_rerank_input_capped` — mock reranker records len(documents): ≤ configured cap.
- [ ] `test_final_results_parity` — with the cap rule chosen, final top-k output identical to uncapped run on fixture.

**Status:** Not Started

---

## Batch verification

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/RAG tldw_Server_API/tests/Embeddings -v -m "not external_api and not local_llm_service"
python -m tldw_Server_API.tests.perf.benchmarks.bench_reranker_instantiation   # expect ms_second_instantiation ≈ 0
python -m bandit -r tldw_Server_API/app/core/RAG tldw_Server_API/app/core/Embeddings/ChromaDB_Library.py -f json -o /tmp/bandit_perf_b7.json
```

Record search-path deltas in `Docs/Reviews/PERF_BASELINE_2026_10.md`; update TASK-13519 (notes, touched files, verification, final summary, DOD — document the Stage 6/8 behavior changes).
