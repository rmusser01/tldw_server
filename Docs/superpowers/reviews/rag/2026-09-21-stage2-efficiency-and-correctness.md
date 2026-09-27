# Stage 2 (2026-09-21) Efficiency and Correctness

## Scope

Axis 5 (efficiency) and Axis 4 (correctness / latent bugs) over the RAG hot set: `unified_pipeline.py`, `database_retrievers.py`, `advanced_reranking.py`. Axis 5 was excluded from the 2026-04-07 rubric, so nothing here overlaps that ledger. Axis 4 findings here are concrete-scenario defects, which the 2026-04-07 review did not produce for this module.

Read-only. No source file was modified.

## Code Paths Reviewed

- `advanced_reranking.py:create_reranker (1782-1820)`
- `advanced_reranking.py:FlashRankReranker.__init__ (436-490)` — model construction at `:469`
- `advanced_reranking.py:TransformersCrossEncoderReranker.__init__ (824-902)` — model construction at `:870` and `:890`
- `advanced_reranking.py:TransformersCrossEncoderReranker.rerank (904-990)` — inference at `:931` and `:941-955`
- `advanced_reranking.py:HybridReranker.__init__ (1377-1409)` — `:1397-1402`
- `advanced_reranking.py:DiversityReranker.rerank (1189-1260)`, `_compute_similarity (1261-1278)`
- `advanced_reranking.py:TwoTierReranker.__init__ (1868-1885)` — `:1876`
- `unified_pipeline.py` reranker construction at `:6265` and `:6305`
- `unified_pipeline.py:_execute_retrieval_variant (4337-4366)`; expansion fan-out `(4472-4500)`; decomposition fan-out `(4681-4740)`
- `unified_pipeline.py:_resilient_call (2962-3003)`
- `database_retrievers.py:MediaDBRetriever.retrieve (1240-1285)`
- `database_retrievers.py:_retrieve_via_backend (1739-1788)`, `_search_media_db (1790-1821)`, `_build_media_documents (1823-1885)`
- `database_retrievers.py:BaseRetriever._execute_query (886-889)`
- `database_retrievers.py:NotesDBRetriever.retrieve (2938-3006)`, `_retrieve_allowed_notes_via_chacha (3008-3026)`, `_retrieve_allowed_notes_via_sql (3028-3053)`, `_row_to_document (3055-3077)`
- `database_retrievers.py:MultiDatabaseRetriever.retrieve (4724-4932)` — global sort at `:4930`
- `database_retrievers.py:_retrieve_vector (2097-2523)` — query embedding at `:2184-2201`
- `utils.py:normalize_scores (162-195)`

## Tests Reviewed

By import-grep.

- `tests/RAG_NEW/unit/test_flashrank_reranker_init.py`, `test_preinstalled_local_reranker.py`, `test_reranker_trust_remote_code.py` — cover reranker *construction correctness* (model id, cache dir, trust flags). They do not assert that construction is amortized across requests, so they neither catch nor block rag-1.
- `tests/RAG_NEW/unit/test_two_tier_reranker.py`, `test_pipeline_two_tier_gate.py`, `test_reranker_metrics.py`, `test_llamacpp_reranker_topk_full_input.py` — reranker gating and metadata with fake rerankers. They construct rerankers directly, so they bypass `create_reranker` and would stay green through any caching change. That makes rag-1 **cheap** to fix.
- `tests/RAG_NEW/unit/test_retrieval.py` — the strongest retrieval test. Covers Media DB fallback, chunk-level late chunking, scoped vector filters, bounded-term retry, allowed-media filtering, and some `MultiDatabaseRetriever` dispatch. It asserts *which documents* come back, not their scores or their relative order across sources, so it does not protect rag-2 or rag-5, but it does mean a normalization fix has a real regression net underneath it.
- `tests/RAG_NEW/unit/test_knowledge_source_retrieval_coverage.py`, `test_kanban_retriever.py`, `test_chunk_fts_integration.py`, `test_media_chunk_fts_metadata.py` — per-source retrieval behavior; none asserts cross-source ranking.
- `tests/RAG/test_restricted_postgres_media_retrieval.py`, `tests/RAG/test_dual_backend_end_to_end.py` — the only Postgres-backed media retrieval coverage; both live in the older tree (see Stage 3).
- No test anywhere asserts event-loop responsiveness or reranker construction count. rag-1 and rag-3 are unprotected in both directions: nothing catches them, and nothing will break when they are fixed.

Import-grep reachability, not measured coverage. The suite was not executed.

## Validation Commands

```bash
grep -rn "create_reranker(" tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py
```
Observed: two call sites, `:6265` and `:6305`, both inside the `unified_rag_pipeline` request body.

```bash
grep -n "lru_cache\|_CACHE\|functools" tldw_Server_API/app/core/RAG/rag_service/advanced_reranking.py
```
Observed: one line, `113: cache_dir = os.getenv("RAG_FLASHRANK_CACHE_DIR")`. There is no memoization of any kind in the reranker factory or in any reranker class.

```bash
grep -n "_retrieve_chunk_fts\|_retrieve_via_backend\|_retrieve_via_chacha\|to_thread" tldw_Server_API/app/core/RAG/rag_service/database_retrievers.py
```
Observed: `to_thread` at `:888`, `:2943`, `:2948`, `:3543`, `:3965`. `_retrieve_via_backend` called unwrapped at `:1251`, `:1268`, `:1285`; `_retrieve_chunk_fts_with_stats` unwrapped at `:1265`; `_retrieve_via_chacha` unwrapped at `:2955`.

```bash
grep -n "score=" tldw_Server_API/app/core/RAG/rag_service/database_retrievers.py
```
Observed 40 assignment sites. Representative scales: min-max-normalized `[0,1]` at `:1386`, `:1729`, `:1880`; constant `1.0` at `:3025`, `:3054`; constant `0.5` at `:3957`, `:3990`, `:4309`, `:4406`; `matched_fields/5.0` at `:4174`; constants `0.6`/`0.4` at `:5123`, `:5147`, `:5171`.

```bash
grep -n "documents.sort(key=lambda d: getattr(d, \"score\", 0.0), reverse=True)" tldw_Server_API/app/core/RAG/rag_service/database_retrievers.py
```
Observed: `:3428` and `:4930`.

```bash
grep -n "for eq in extra_queries" tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py
grep -n "sem = asyncio.Semaphore(max_workers)" tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py
```
Observed: `4477` (serial loop) and `4711` (bounded-concurrent fan-out) — two fan-out policies ~230 lines apart in one function.

```bash
ls Docs/ADR/ | grep -iE 'rag|rerank|retriev|vector'
```
Observed: no output. No ADR governs any of this.

## Findings

---

### FINDING rag-1 — a transformer reranker model is loaded from disk on every RAG request

```
axis:        efficiency
class:       n/a
severity:    High
sites:       unified_pipeline.py:unified_rag_pipeline (6265) and (6305) -> advanced_reranking.py:create_reranker (1782-1820)
             -> advanced_reranking.py:FlashRankReranker.__init__ (469)
             -> advanced_reranking.py:TransformersCrossEncoderReranker.__init__ (870, 890)
             -> advanced_reranking.py:HybridReranker.__init__ (1397-1402)
             -> advanced_reranking.py:TwoTierReranker.__init__ (1876)
canonical:   NONE
destination: a `rag_service/reranker_registry.py` owning exactly one responsibility — process-lifetime
             caching of reranker instances keyed by (strategy, model_name, device, revision,
             local_files_only, trust_remote_code). `create_reranker` becomes a thin lookup into it.
             NOT a new knob on RerankingConfig, and nothing added to Utils.py or http_client.py.
knowledge:   n/a
scenario:    n/a (efficiency axis)
cost-driver: One model materialization per RAG request that enables reranking. `create_reranker` has no
             memoization (`grep lru_cache` over advanced_reranking.py returns only an unrelated env-var
             line), and every reranker class loads its model in `__init__`:
             `FlashRankReranker.__init__` calls `Ranker(...)` at :469;
             `TransformersCrossEncoderReranker.__init__` calls `CrossEncoder(...)` at :870 or
             `AutoModelForSequenceClassification.from_pretrained(...)` at :890.
             This is on the default path, not an exotic one: `enable_reranking` defaults `True` and
             `reranking_strategy` defaults to `flashrank` in `rag_schemas_unified.py:931-940` and
             `:2016-2017`; the `balanced` profile uses `hybrid` (`profiles.py:246-247`) and
             `HybridReranker.__init__` itself constructs a `FlashRankReranker` at :1399; the
             research-grade profile uses `two_tier` (`profiles.py:349-350`) and `TwoTierReranker`
             constructs a `TransformersCrossEncoderReranker` at :1876. Scales linearly with request
             rate, and the constant is a weight-deserialization, not a cheap allocation — for a
             bge-reranker-class cross-encoder that is hundreds of MB read and materialized into torch
             tensors per request. Under concurrency it also multiplies resident memory by the number of
             in-flight requests, because no instance is shared.
             Degradation makes it worse, not better: the two-tier failure path at :6305 constructs a
             *second* reranker inside the same request.
impact:      Model load dominates P95 latency for the default configuration and is the single largest
             avoidable per-request cost in the module.
tests:       tests/RAG_NEW/unit/test_flashrank_reranker_init.py,
             tests/RAG_NEW/unit/test_preinstalled_local_reranker.py,
             tests/RAG_NEW/unit/test_reranker_trust_remote_code.py (construction correctness — these
             pin the keying inputs a cache must respect);
             tests/RAG_NEW/unit/test_two_tier_reranker.py, test_pipeline_two_tier_gate.py,
             test_reranker_metrics.py (gating, via injected fake rerankers — unaffected by caching).
effort:      cheap. The construction inputs are already explicit and already asserted by three tests, so
             the cache key is derivable from existing covered behavior, and the gating tests inject
             rerankers directly and will not notice. The only care needed is that
             `trust_remote_code` / `local_files_only` / `revision` are part of the key so a config change
             cannot serve a model loaded under different trust settings.
owner-only:  no
confidence:  confirmed
```

---

### FINDING rag-2 — cross-source result ranking is meaningless because each retriever normalizes to its own scale before a global sort

```
axis:        correctness
class:       n/a
severity:    High
sites:       database_retrievers.py:MultiDatabaseRetriever.retrieve (4930-4932)  <- the global sort+cap
             producers with incompatible scales:
               database_retrievers.py:_build_media_documents (1849-1856, 1880)   min-max over the page
               database_retrievers.py:_retrieve_chunk_fts_with_stats (1367, 1386) min-max over the page
               database_retrievers.py:_retrieve_vector (1653-1655, 1729)          min-max over the page
               database_retrievers.py:_retrieve_allowed_notes_via_chacha (3025)   constant 1.0
               database_retrievers.py:_retrieve_allowed_notes_via_sql (3054)      constant 1.0
               database_retrievers.py:ChatHistoryRetriever.retrieve (3957, 3990)  constant 0.5
               database_retrievers.py:WorldBooksRetriever.retrieve (4082)         matched_fields/5 + priority/100
               database_retrievers.py:ChatDictionariesRetriever.retrieve (4174)   matched_fields/5
               database_retrievers.py:CharacterCardsRetriever.retrieve (4309)     constant 0.5
               database_retrievers.py:SQLRetriever.retrieve (4406)                constant 0.5
               database_retrievers.py:ClaimsRetriever.retrieve (5123, 5147, 5171, 5209) constants 0.6/0.4
canonical:   NONE
destination: a `rag_service/score_fusion.py` owning one responsibility — turning per-source result lists
             into one comparably-ranked list. Rank-based fusion (RRF) is the boring correct answer
             because it needs no cross-source score calibration at all, and the module already has a
             rank-fusion entry point in `MultiDatabaseRetriever.retrieve_with_fusion (4949)` that the
             main path does not use. The cheapest correct fix is to route the main path through fusion
             rather than to invent a calibration scheme.
knowledge:   "what a relevance score means" is currently answered eleven different ways inside one file,
             and `MultiDatabaseRetriever.retrieve` silently assumes all eleven answers are the same
             number line. Any new retriever must guess a scale, and there is no place where the
             convention is written down, so each new source drifts further.
scenario:    A user searches with `sources=["media_db","notes"]`, `top_k=10`, and an explicit
             `include_note_ids` list of 20 notes. `_retrieve_allowed_notes_via_sql (3028-3053)` returns
             those notes ordered by `last_modified DESC` — with no text match required, as its own
             docstring says — and `_row_to_document` stamps every one of them `score=1.0` (:3054).
             Media results come back min-max normalized, so exactly one media document scores 1.0 and
             the rest fall below it. The global sort at :4930 therefore ranks 20 constant-1.0 notes at
             or above every media document, and the `documents[:max_results]` cap on the next line
             returns a top-10 of notes only. Zero media documents survive, including high-bm25 matches.
             The generation stage then answers from notes that were never scored for relevance.
             A second, scale-free instance of the same defect: because min-max always maps the best
             item of a page to 1.0 and the worst to 0.0, the top hit of *every* source ties at 1.0, and
             `list.sort` being stable resolves that tie by `self.retrievers` iteration order — so which
             source wins the top slot is decided by dict insertion order, not by relevance.
impact:      Wrong documents reach generation, which is the failure mode RAG exists to prevent. It is
             silent — no error, no metadata flag, and the response looks well-formed.
tests:       tests/RAG_NEW/unit/test_retrieval.py (asserts which documents are returned per source, not
             their relative order across sources); tests/RAG_NEW/unit/test_knowledge_source_retrieval_coverage.py;
             tests/RAG_NEW/unit/test_learned_fusion_simple.py (fusion helper in isolation, not on the
             main path). No test asserts cross-source ordering, which is why this survived.
effort:      moderate. The per-source retrieval behavior is well covered by test_retrieval.py, so the
             producers are safe to touch; the risk is that switching the main path to rank fusion changes
             returned ordering for every existing caller, so it needs a design note and a staged rollout
             rather than a patch.
owner-only:  no
confidence:  confirmed (the eleven incompatible scales and the global sort over them);
             probable-risk (the exact top-10 composition in the scenario depends on how many notes the
             allowlist carries relative to top_k)
```

---

### FINDING rag-3 — synchronous SQLite FTS and torch inference run on the event loop in the two hottest paths

```
axis:        efficiency
class:       n/a
severity:    High
sites:       database_retrievers.py:MediaDBRetriever.retrieve (1251, 1265, 1268, 1285) — calls the
               synchronous `_retrieve_via_backend (1739-1788)` / `_retrieve_chunk_fts_with_stats (1520)`
               directly from an `async def`, and those reach SQLite via `_search_media_db (1790-1821)`
             database_retrievers.py:NotesDBRetriever.retrieve (2955) — same, via `_retrieve_via_chacha (3079)`
             advanced_reranking.py:TransformersCrossEncoderReranker.rerank (931) — `self._ce.predict(...)`
             advanced_reranking.py:TransformersCrossEncoderReranker.rerank (941-955) — the raw-transformers
               tokenize + `model(**enc)` loop
             advanced_reranking.py:DiversityReranker.rerank (1189-1260) — pure-CPU MMR loop, see rag-4
canonical:   database_retrievers.py:BaseRetriever._execute_query (886-889) — `await asyncio.to_thread(...)`.
             The same file already does it correctly at :2943, :2948, :3543, :3965.
destination: n/a — this is an adoption gap against an in-file pattern, not a new module.
knowledge:   n/a
cost-driver: The process runs one event loop. A full-text scan over a user's media corpus, and a
             cross-encoder forward pass over up to `top_k` query/document pairs, both execute on the loop
             thread with nothing yielding. Retrieval cost scales with corpus size; rerank cost scales
             with `top_k x max_length` transformer forward passes (512 tokens default, `:948`). While
             either runs, *every* other in-flight request in the process — every other RAG query, every
             chat completion, every health check — is stalled. Concurrency does not help because the two
             paths that would benefit most from it are the two that block.
             The inconsistency is the tell: the same file wraps Notes FTS (`:2943`), Prompts (`:3543`),
             and chat messages (`:3965`) in `asyncio.to_thread`, but leaves the primary Media DB path —
             the default source — unwrapped at all four of its call sites. Two sibling call sites in
             `MediaDBRetriever.retrieve` are even sequential blocking queries in one request: chunk FTS
             at :1265, then, if it returned nothing, a second media-level query at :1268.
impact:      Head-of-line blocking on the shared loop. This is a throughput ceiling for the whole server,
             not a slow endpoint.
tests:       tests/RAG_NEW/unit/test_retrieval.py, test_chunk_fts_integration.py,
             test_media_chunk_fts_metadata.py (Media DB retrieval behavior — they call the async
             `retrieve()` and assert results, so they survive a to_thread wrap unchanged);
             tests/RAG_NEW/unit/test_two_tier_reranker.py, test_reranker_metrics.py (rerank, with fakes).
             No test observes loop responsiveness.
effort:      cheap for the retriever sites — four call sites, the in-file idiom already exists, and the
             behavior is covered. Moderate for the reranker: `_ce.predict` is thread-safe for
             sentence-transformers but the raw-transformers branch mutates `enc` device placement, so a
             to_thread wrap there wants the model-cache from rag-1 in place first so one model is not
             entered from several threads.
owner-only:  no
confidence:  confirmed
```

---

### FINDING rag-4 — MMR diversity reranking re-tokenizes every document body O(top_k^2) times

```
axis:        efficiency
class:       n/a
severity:    Medium
sites:       advanced_reranking.py:DiversityReranker.rerank (1224-1245) — the
               `while remaining` / `for idx in remaining_indices` / `for selected_idx in selected_indices`
               triple loop, calling `_compute_similarity` at :1235
             advanced_reranking.py:DiversityReranker._compute_similarity (1261-1278)
             reached on the default path via advanced_reranking.py:HybridReranker.__init__ (1400)
canonical:   NONE
destination: n/a — the fix is local: hoist the tokenization out of the loop.
knowledge:   n/a
cost-driver: `_compute_similarity(text1, text2)` does `set(text.lower().split())` on **both full
             document bodies** on every call (`:1269-1270`), and nothing is memoized. The MMR loop calls
             it once per (remaining document x already-selected document) pair, on every outer
             iteration, so the call count is O(top_k^2 x n_documents) and each call re-lowercases and
             re-splits two complete document contents. With `top_k=10` over 50 candidate documents of
             ~4 KB each, that is on the order of 2,500 calls, ~5,000 full-string lowercase+split+set
             constructions, and tens of MB of transient string churn — per request, synchronously on the
             event loop (see rag-3).
             The whole cost is avoidable in one line: build `word_sets = [set(d.content.lower().split())
             for d in documents]` once before the loop (n set constructions instead of O(top_k^2 x n))
             and compute Jaccard over the cached sets.
             Worth noting alongside: the docstring at :1265-1266 says "Simple Jaccard similarity for
             demonstration. In production, use embeddings or more sophisticated methods." It is on the
             production default path — `HybridReranker` composes it at :1400 and `hybrid` is the
             `balanced` profile's strategy (`profiles.py:246-247`). That is a separate product question,
             not a finding; the cost driver above stands either way.
impact:      Medium rather than High only because it is bounded by `top_k` and does not load anything
             from disk. It is on the default path and the fix is a few lines.
tests:       tests/RAG_NEW/unit/test_advanced_reranking_sanitizers.py,
             tests/RAG_NEW/unit/test_two_tier_reranker.py (reranker behavior; MMR output ordering is
             unchanged by memoizing the word sets, so a fix is verifiable against existing assertions).
effort:      cheap. Pure hoist, no behavior change, existing tests are the net.
owner-only:  no
confidence:  confirmed
```

---

### FINDING rag-5 — page-relative min-max normalization makes the `min_score` filter drop results by rank rather than by relevance

```
axis:        correctness
class:       n/a
severity:    Medium
sites:       database_retrievers.py:_build_media_documents (1849-1856) — min-max over the page
             database_retrievers.py:_build_media_documents (1882-1884) — `if float(score_val) < min_score: continue`
             database_retrievers.py:_retrieve_chunk_fts_with_stats (1367) — same idiom
             database_retrievers.py:_retrieve_vector (1653-1655) — same idiom
             utils.py:normalize_scores (177-183) — the `max == min -> [0.5] * n` degenerate branch
             public knob: rag_schemas_unified.py:min_score (220-232) and (1945)
canonical:   NONE
destination: n/a — either apply `min_score` to the raw retriever score before normalization, or document
             `min_score` as a page-relative percentile knob. Both are one-line decisions; the bug is
             that the code does neither deliberately.
knowledge:   n/a
scenario:    `min_score` is a public request field (`rag_schemas_unified.py:220`, `ge=0.0, le=1.0`), and
             `_build_media_documents` compares it against the *normalized* score (:1882), not the raw
             bm25 rank. Min-max maps the best row of the returned page to 1.0 and the worst to 0.0.
             So: (a) with any `min_score > 0`, the lowest-ranked row of every page is dropped
             unconditionally, however relevant it actually is — the filter is a "drop the tail of this
             page" operation wearing a relevance-threshold name; (b) with a query that matches exactly
             one media item, `normalize_scores` takes the `max == min` branch at utils.py:181 and
             returns `[0.5]`, so a request with `min_score=0.6` returns **zero results for an exact
             single match**; (c) the same `[0.5]` branch fires when every row shares a bm25 rank, so a
             whole page of equally-good matches passes or fails as one block depending on which side of
             0.5 the caller's threshold sits.
             Default `min_score` is 0.0 everywhere (`RetrievalConfig.min_score`, `advanced_config.py:107`,
             `request_resolution.py:303`), so this only bites callers who set the knob — but it is an
             advertised, validated, documented knob.
impact:      Silently wrong result sets for any caller who uses `min_score`, with the worst case (b)
             returning nothing at all for a perfect match. Medium rather than High because the default
             value avoids it.
tests:       tests/RAG_NEW/unit/test_retrieval.py (Media DB document construction — closest existing
             coverage, does not vary min_score); tests/RAG/test_rag_selection_filters.py.
             `utils.normalize_scores` itself has no direct test: `grep -rln "normalize_scores" tldw_Server_API/tests`
             returns only tests/Slides/test_standalone_html_sources.py, which is incidental.
effort:      moderate. The decision (raw-score threshold vs documented percentile) is a product call, and
             `normalize_scores` has no test of its own, so a fix should add one first.
owner-only:  no
confidence:  confirmed (the normalized-vs-raw comparison and the `[0.5]` degenerate branch);
             probable-risk (how many deployments actually set min_score > 0)
```

---

### FINDING rag-6 — the query-expansion fan-out is serial and unbudgeted while the decomposition fan-out 230 lines below it is bounded-concurrent and budgeted

```
axis:        efficiency
class:       n/a
severity:    Medium
sites:       unified_pipeline.py (4475-4500) — `for eq in extra_queries:` with `await` inside, no
               concurrency, no time budget
             unified_pipeline.py (4681-4740) — the decomposition fan-out: semaphore at :4711,
               `asyncio.create_task` at :4717, `asyncio.wait(..., timeout=remaining)` at :4721,
               pending-task cancellation at :4722-4723, doc budget at :4742
             both route through unified_pipeline.py:_execute_retrieval_variant (4337-4366)
canonical:   rag_service/batch_utils.py:run_batch (84-188) — the module's own bounded-concurrency helper,
             already used by unified_pipeline.py:9346-9360 for the batch pipeline.
destination: n/a — adopt the existing `run_batch`.
knowledge:   n/a
cost-driver: Each iteration of the serial loop is a complete retrieval pass through
             `execute_retrieval_phase` -> `MultiDatabaseRetriever.retrieve` -> per-source queries, and for
             any vector-enabled source it includes a fresh query embedding: `_retrieve_vector` calls
             `create_embeddings_batch([query], ...)` at database_retrievers.py:2184-2201 on every
             invocation, and the `query_vector` passthrough that would let a caller reuse one embedding
             (`:2120`, `:2144`) is supplied from exactly one place in the whole repo — the HyDE branch at
             unified_pipeline.py:4449. So N expanded query variants cost N sequential retrievals and N
             sequential embedding round-trips. With a remote embedding provider at ~100 ms per call and
             3-4 variants, that is 300-400 ms of pure serialized network wait that the decomposition path
             — doing the same shape of work — already avoids. `expand_query` defaults on in the `accurate`
             and `balanced` profiles (`profiles.py:166`, `:213`).
             Two further consequences of the asymmetry: the expansion loop has no time budget, so it
             cannot be cut short the way decomposition is at :4721; and the variants could be embedded in
             one batched `create_embeddings_batch(all_variants, ...)` call rather than N calls of one.
impact:      Adds latency proportional to the expansion factor on a default-on path, for no behavioral
             benefit — the results are merged unordered at :4492 either way.
tests:       tests/RAG_NEW/unit/test_retrieval_plan_usage.py (proves the pipeline routes through
             `execute_retrieval_phase` and tolerates a forced `batch_utils` ImportError at :333-334 —
             so the adoption path is already exercised);
             tests/RAG/test_batch_utils.py (run_batch semantics).
effort:      cheap. `run_batch` is already imported in this file, already has a fallback path, and its
             semantics are tested. The merge at :4492 is order-independent.
owner-only:  no
confidence:  confirmed
```

---

### FINDING rag-7 — `_resilient_call` retries every exception type, including permanent ones, inside the request's own timeout budget

```
axis:        correctness
class:       n/a
severity:    Low
sites:       unified_pipeline.py:_resilient_call (2995-3000) — `RetryPolicy(RetryConfig(max_attempts=...))`
             resilience.py:RetryConfig (87-96) — `initial_delay=1.0`, `exponential_base=2.0`,
               `retry_on=[Exception]`, `dont_retry_on=[]`
             resilience.py:RetryPolicy._should_retry (274-279)
             resilience.py:RetryPolicy._calculate_delay (281-289)
canonical:   core/http_client.py:_should_retry (2340-2357) — the repo's adopted retry classifier, which
             separates retriable transport failures from permanent ones and treats DNS resolution
             failures as terminal.
destination: see rag-13 in stage 3 — a cohesive `core/Utils/backoff.py` owning retry *scheduling*, seeded
             from the http_client implementation rather than this one.
knowledge:   n/a
scenario:    `_resilient_call` wraps retrieval, reranking, and generation. `RetryConfig` is constructed
             with only `max_attempts` (:2996), so every other field takes its default — including
             `retry_on=[Exception]` with an empty `dont_retry_on`. A deterministic failure therefore
             retries: a malformed `RetrievalPlan` raising `TypeError`, a bad filter raising `ValueError`,
             or a provider returning a permanent 401 all run `max_attempts` times with 1 s and 2 s sleeps
             between them. Those sleeps are spent inside the request's own `timeout_seconds` budget
             (`_with_timeout` at :3001 wraps the whole retry chain), so on a short timeout the caller
             receives an `asyncio.TimeoutError` instead of the actual `TypeError` — the real cause is
             erased and the request takes 3 s longer to fail.
impact:      Low: it degrades error reporting and adds latency on already-failing requests rather than
             producing wrong results. Worth fixing when `dont_retry_on` is populated, which is a
             one-field change at the construction site.
tests:       tests/RAG/test_resilience_sanitizers.py:70-71 (constructs RetryPolicy with an explicit
             `retry_on=[RuntimeError]`, so it never exercises the default-everything config the pipeline
             actually uses).
effort:      cheap — populate `dont_retry_on` at the single construction site, or better, adopt the
             http_client classifier per rag-13.
owner-only:  no
confidence:  confirmed (the configuration); probable-risk (how often a permanent error reaches this
             wrapper in practice)
```

---

### FINDING rag-8 — 52 optional-import shims convert a module-load failure into a silent feature-disable with no error surfaced

```
axis:        encapsulation
class:       n/a
severity:    Low
sites:       unified_pipeline.py — 52 `except ImportError:` blocks, each followed by `_X = None` and a
               module-level re-export; ~107 such `_X = None` assignments in the first 1500 lines.
               Representative: the reranking import block at (871-886) and its consumer guard at (6122).
             Contrast the one site that does report degradation: (6093-6100), which records
               `_record_profile_degradation(reason="unavailable_dependency")` for `two_tier` only.
canonical:   NONE
destination: n/a — the proportionate change is to make the guard at :6122 record a degradation the same
             way :6093 already does, not to restructure 52 imports.
knowledge:   n/a
scenario:    If `from .advanced_reranking import create_reranker` fails for any reason — a transitive
             ImportError, a circular import during partial package initialization — `create_reranker`
             becomes `None`. The guard at :6122 (`if create_reranker and RerankingStrategy and
             RerankingConfig:`) then skips the entire reranking phase. For `flashrank`, `hybrid`, and
             `cross_encoder` there is no `else`, no error appended to `result.errors`, and no metadata
             flag: the request returns un-reranked documents, HTTP 200, with `enable_reranking: true`
             still echoed in the response. Only `two_tier` gets a degradation record, at :6093.
             The same shape repeats for every one of the 52 shimmed imports.
impact:      Low as a confirmed defect because `advanced_reranking.py`'s own top-level imports are
             first-party plus numpy (`:12-35`), with flashrank / sentence-transformers / transformers all
             imported lazily inside the constructors — so a spontaneous ImportError is unlikely today.
             The risk is that the pattern makes "feature silently off" the default failure mode for 52
             dependencies, and the one site that records it proves the authors consider it worth
             recording.
tests:       tests/RAG_NEW/unit/test_pipeline_two_tier_gate.py (covers only the two_tier branch that
             *does* record degradation); tests/RAG_NEW/unit/test_retrieval_plan_usage.py:333-334
             (the only test that forces an import failure, for batch_utils).
effort:      cheap for the reranking site — mirror the existing `_record_profile_degradation` call.
             Expensive and not recommended for all 52.
owner-only:  no
confidence:  confirmed (the pattern and the missing degradation record at :6122);
             assumption (that a real-world ImportError reaches it)
```

## Suggested Refactor/Actions

Ordered by value over cost, not by severity.

1. **rag-1 first.** A `rag_service/reranker_registry.py` with a keyed process-lifetime cache is the single highest-value change in this ledger: High severity, cheap, well-covered construction inputs, and it unblocks the safe version of the rag-3 reranker fix. Small enough to skip the design-doc treatment — one module, one responsibility, existing tests.
2. **rag-3 retriever half.** Wrap the four unwrapped Media DB call sites and the one Notes call site in `asyncio.to_thread`, matching `BaseRetriever._execute_query (888)`. Four lines, in-file precedent, covered by `test_retrieval.py`. Do the reranker half after rag-1.
3. **rag-4 and rag-6.** Both are hoists. rag-4 is a local memoization; rag-6 is adopting `batch_utils.run_batch`, which the same file already imports 4,900 lines further down.
4. **rag-2 needs the design-first treatment** (`Docs/Design/YYYY-MM-DD-rag-score-fusion-design.md`, an ADR entry, a Backlog task linking both, and a staged `IMPLEMENTATION_PLAN_rag_score_fusion.md`). It changes result ordering for every existing caller, and the choice between rank fusion and per-source calibration is a decision, not a fix. Note that `retrieve_with_fusion (4949)` already exists — the plan is probably "route the main path through it," not "build one."
5. **rag-5 wants a test before a fix.** `utils.normalize_scores` has no direct coverage. Add one that pins the `max == min -> [0.5]` contract, then decide whether `min_score` is a raw threshold or a documented percentile.
6. **rag-7 and rag-8 are one-line opportunistic fixes** at their single sites. Neither justifies a task of its own; attach them to whichever change touches `_resilient_call` or the rerank block next.
