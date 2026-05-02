# Stage 4 Retrieval Boundaries and Data Sources

## Scope

Review retrieval composition across database retrievers, query expansion, caches, vector stores, and source adapters.

## Code Paths Reviewed

- Reviewed: `tldw_Server_API/app/core/RAG/rag_service/database_retrievers.py`
  - Retrieval seam owners: `BaseRetriever` (`447+`), `MediaDBRetriever.retrieve()` (`683+`), `MediaDBRetriever._retrieve_vector()` (`1562+`), `MediaDBRetriever.retrieve_hybrid()` (`1881+`), `NotesDBRetriever.retrieve()` (`2036+`), `KanbanDBRetriever.retrieve()` (`2244+`), `CharacterCardsRetriever.retrieve()` (`2635+`), `MultiDatabaseRetriever.retrieve()` (`3133+`), `MultiDatabaseRetriever.retrieve_with_fusion()` (`3249+`), `retrieve_from_databases()` (`3540+`).
- Reviewed: `tldw_Server_API/app/core/RAG/rag_service/query_expansion.py`
  - Expansion seams: compatibility helpers and corpus-aware `multi_strategy_expansion()` (`546+`), wrapper seam `QueryExpansionRetriever` (`679+`).
- Reviewed: `tldw_Server_API/app/core/RAG/rag_service/hyde.py`
  - Utility-only seam: `generate_hypothetical_answer()` (`56+`) and `embed_text()` (`74+`).
- Reviewed: `tldw_Server_API/app/core/RAG/rag_service/query_classifier.py`
  - Routing seam: `classify_query()` (`246+`), `reformulate_query()` (`359+`), `classify_and_reformulate()` (`450+`).
- Reviewed: `tldw_Server_API/app/core/RAG/rag_service/semantic_cache.py`
  - Cache ownership seam: `SemanticCache` core (`41+`), namespace and path normalization (`618+`), shared cache constructor `get_shared_cache()` (`803+`), shared invalidation `clear_shared_caches()` (`840+`).
- Reviewed: `tldw_Server_API/app/api/v1/utils/rag_cache.py`
  - API-side invalidation seam cited for namespace collection and hardcoded media collection naming in `delete_media_vectors()` and `invalidate_rag_caches()`.
- Reviewed: `tldw_Server_API/app/core/RAG/rag_service/media_search.py`
  - Side-source retrieval seam: `search_images()` (`122+`), `search_videos()` (`194+`).
- Reviewed: `tldw_Server_API/app/core/RAG/rag_service/web_fallback.py`
  - Fallback seam: `web_search_fallback()` (`94+`), `merge_web_results()` (`273+`), `fallback_to_web_search()` (`317+`).
- Reviewed: `tldw_Server_API/app/core/RAG/rag_service/vector_stores/base.py`
  - Adapter contract seam: `VectorStoreAdapter` (`46+`), `search()` (`142+`), `multi_search()` (`166+`), `delete_by_filter()` (`128+`).
- Reviewed: `tldw_Server_API/app/core/RAG/rag_service/vector_stores/factory.py`
  - Adapter creation seam: `VectorStoreFactory.create_from_settings()` (`103+`), `create_from_settings_for_user()` (`176+`).
- Reviewed: `tldw_Server_API/app/core/RAG/rag_service/vector_stores/chromadb_adapter.py`
  - Chroma ownership seam: `initialize()` (`61+`), `create_collection()` (`85+`), `delete_by_filter()` (`232+`), `search()` (`336+`), `multi_search()` (`400+`).
- Reviewed: `tldw_Server_API/app/core/RAG/rag_service/vector_stores/pgvector_adapter.py`
  - PGVector ownership seam: `initialize()` (`96+`), `create_collection()` (`281+`), `delete_by_filter()` (`394+`), `_build_where_from_filter()` (`432+`), `search()` (`650+`), `multi_search()` (`707+`).

Boundary map:
- The authoritative retrieval outputs in this stage are the `Document` lists emitted by the source retrievers, especially `MediaDBRetriever.retrieve()` and `MediaDBRetriever._retrieve_vector()`, then globally merged by `MultiDatabaseRetriever.retrieve()` or `retrieve_with_fusion()`.
- Retrieval policy becomes concrete in retriever code, not in the adapter contract. `MediaDBRetriever._retrieve_vector()` resolves collection names, generates query embeddings, applies allowed-media and metadata filters, decides HyDE execution, and converts adapter results back into `Document`.
- Source-specific behavior is exposed directly by concrete retrievers. `NotesDBRetriever`, `CharacterCardsRetriever`, and `KanbanDBRetriever` each encode their own adapter-vs-SQL behavior, ranking normalization, and source metadata layout.
- Vector-store ownership is split: `factory.py` chooses adapter type and connection params, adapters own backend mechanics, but retrievers and API utilities still own collection naming and namespace semantics.
- Semantic cache internals are cohesive on similarity, persistence, and namespacing, but caller identity and invalidation policy remain external.

## Tests Reviewed

- `tldw_Server_API/tests/RAG_NEW/unit/test_retrieval.py`
  - Protects `RetrievalConfig`, media-db retrieval, bounded-term fallback, chunk-level late chunking, allowed-media filtering, vector scoped-model lookup, shared DB attachment, and multi-database orchestration.
  - Constrains `MediaDBRetriever`, `ClaimsRetriever`, and `MultiDatabaseRetriever`.
  - Mixes happy-path, fallback-path, adapter-path, and filter-path coverage; multi-tenant coverage is limited to user-scoped collection naming via mocks.
- `tldw_Server_API/tests/RAG_NEW/unit/test_vector_store_parity.py`
  - Protects basic adapter parity for search ordering, filter behavior, Chroma user DB base resolution, and PGVector JSONB filter operators plus `multi_search()`.
  - Constrains the `VectorStoreAdapter` contract and adapter implementations.
  - Mostly happy-path parity coverage; the real PGVector cases depend on external Postgres availability.
- `tldw_Server_API/tests/RAG_NEW/unit/test_vector_store_admin_guardrails.py`
  - Protects admin-only vector-store endpoints and rejects empty `delete_by_filter` requests.
  - Constrains the admin surface over adapter operations rather than retriever internals.
  - Explicit failure-path coverage.
- `tldw_Server_API/tests/RAG_NEW/unit/test_vector_retriever_hyde.py`
  - Protects HyDE-only result return, score-preferential HyDE merge, and chunk-level parent dedupe.
  - Constrains `MediaDBRetriever._retrieve_vector()`.
  - Happy-path merge coverage only; no backend failure coverage.
- `tldw_Server_API/tests/RAG_NEW/unit/test_hyde_retrieval_merge.py`
  - Protects media-vs-chunk merge modes, HyDE early exit, and score reordering from HyDE weighting.
  - Constrains `MediaDBRetriever._retrieve_vector()`.
  - Covers one important branch (`HYDE_ONLY_IF_NEEDED`) in addition to happy-path merge behavior.
- `tldw_Server_API/tests/RAG_NEW/unit/test_semantic_cache_tenant_scoping.py`
  - Protects namespace reporting, shared-cache isolation by namespace, and namespace-scoped clearing.
  - Constrains `get_shared_cache()` and `clear_shared_caches()`.
  - Explicit multi-tenant coverage.
- `tldw_Server_API/tests/RAG_NEW/unit/test_semantic_cache_persistence.py`
  - Protects save/load fidelity for timestamps and semantic-match recovery after reload.
  - Constrains `SemanticCache.save()`, `load()`, and `find_similar()`.
  - Happy-path persistence coverage only.
- `tldw_Server_API/tests/RAG_NEW/unit/test_semantic_cache_path_sanitization.py`
  - Protects relative-path anchoring under the cache root and rejection of out-of-root absolute paths.
  - Constrains `_sanitize_persist_path()` and `get_shared_cache()`.
  - Explicit failure-path coverage.
- `tldw_Server_API/tests/RAG_NEW/unit/test_corpus_synonyms_expansion.py`
  - Protects corpus-specific synonym loading from config-root discovery and its use in `multi_strategy_expansion()`.
  - Constrains `synonyms_registry.py` and query-expansion compatibility helpers.
  - Happy-path coverage with config-env routing.
- `tldw_Server_API/tests/RAG_NEW/unit/test_query_classifier.py`
  - Protects lenient parsing of fenced or list-wrapped classifier JSON.
  - Constrains `_parse_classification_response()` only.
  - Happy-path parsing coverage only.
- `tldw_Server_API/tests/RAG_NEW/unit/test_media_search.py`
  - Protects `asyncio.to_thread` usage, normalized result shaping, and YouTube thumbnail derivation.
  - Constrains `search_images()` and `search_videos()`.
  - Happy-path coverage only.
- `tldw_Server_API/tests/RAG_NEW/integration/test_retriever_pgvector_multi_search.py`
  - Protects PGVector-backed `multi_search()` plus JSONB filter propagation through `MediaDBRetriever._retrieve_vector()`.
  - Constrains the retriever-to-adapter integration seam.
  - True integration coverage when Postgres is available; otherwise skipped.
- `tldw_Server_API/tests/RAG_NEW/integration/test_adapter_guards.py`
  - Protects production-mode refusal of raw SQL fallback and success of adapter-backed retrieval for notes and character data.
  - Constrains source-specific adapter-vs-SQL boundaries in `NotesDBRetriever` and `CharacterCardsRetriever`.
  - Covers both failure and success paths.

## Validation Commands

- Seam inventory:
  - `rg -n "class (.*Retriever|.*Adapter)|def (retrieve|search|expand|initialize|create_|delete_by_filter|get_shared_cache|lookup|store|merge)|DataSource|index_namespace|vector_store_type" tldw_Server_API/app/core/RAG/rag_service/database_retrievers.py tldw_Server_API/app/core/RAG/rag_service/query_expansion.py tldw_Server_API/app/core/RAG/rag_service/hyde.py tldw_Server_API/app/core/RAG/rag_service/query_classifier.py tldw_Server_API/app/core/RAG/rag_service/semantic_cache.py tldw_Server_API/app/core/RAG/rag_service/media_search.py tldw_Server_API/app/core/RAG/rag_service/web_fallback.py tldw_Server_API/app/core/RAG/rag_service/vector_stores/base.py tldw_Server_API/app/core/RAG/rag_service/vector_stores/factory.py tldw_Server_API/app/core/RAG/rag_service/vector_stores/chromadb_adapter.py tldw_Server_API/app/core/RAG/rag_service/vector_stores/pgvector_adapter.py`
- Additional ownership trace:
  - `rg -n "generate_hypothetical_answer|embed_text|HYDE_|classify_query|classify_and_reformulate|multi_strategy_expansion|get_shared_cache|clear_shared_caches|collection_prefix|create_from_settings_for_user|fallback_to_web_search|search_images|search_videos" tldw_Server_API/app/core/RAG/rag_service tldw_Server_API/app/api/v1/utils/rag_cache.py`
- Targeted retrieval tests:
  - `source ../../.venv/bin/activate && python -m pytest tldw_Server_API/tests/RAG_NEW/unit/test_retrieval.py tldw_Server_API/tests/RAG_NEW/unit/test_vector_store_parity.py tldw_Server_API/tests/RAG_NEW/unit/test_vector_store_admin_guardrails.py tldw_Server_API/tests/RAG_NEW/unit/test_vector_retriever_hyde.py tldw_Server_API/tests/RAG_NEW/unit/test_hyde_retrieval_merge.py tldw_Server_API/tests/RAG_NEW/unit/test_semantic_cache_tenant_scoping.py tldw_Server_API/tests/RAG_NEW/unit/test_semantic_cache_persistence.py tldw_Server_API/tests/RAG_NEW/unit/test_semantic_cache_path_sanitization.py tldw_Server_API/tests/RAG_NEW/unit/test_corpus_synonyms_expansion.py tldw_Server_API/tests/RAG_NEW/unit/test_query_classifier.py tldw_Server_API/tests/RAG_NEW/unit/test_media_search.py tldw_Server_API/tests/RAG_NEW/integration/test_retriever_pgvector_multi_search.py tldw_Server_API/tests/RAG_NEW/integration/test_adapter_guards.py -v`
  - Result in this worktree: `52 passed, 4 skipped, 117 warnings in 7.21s`.
  - Concrete skips:
    - `test_vector_store_parity.py::test_parity_basic_search_and_filter[pgvector]`
    - `test_vector_store_parity.py::test_parity_in_and_numeric[pgvector]`
    - `test_vector_store_parity.py::test_parity_boolean_and_multi_search[pgvector]`
    - `test_retriever_pgvector_multi_search.py::test_retriever_multi_search_with_jsonb_filter`
  - Skip reason observed in output: connection to `127.0.0.1:5432` failed with `Operation not permitted`, so the PGVector parity and integration seams were not exercised end-to-end in this environment.
  - Fixture context: DSN resolution for those tests is owned by `tldw_Server_API/tests/helpers/pgvector.py`, which resolves an env-driven Postgres DSN (`PG_TEST_DSN`, `PGVECTOR_DSN`, `TEST_DATABASE_URL`, `DATABASE_URL`, container-style vars, etc.), performs a connectivity check, and skips PG-backed tests when no reachable DSN is available.
- Docs-scope security check:
  - `source ../../.venv/bin/activate && python -m bandit -r Docs/superpowers/reviews/rag -f json -o /tmp/bandit_stage4_rag.json`
  - Result in this worktree: `jq '{errors: (.errors | length), results: (.results | length)}' /tmp/bandit_stage4_rag.json` returned `{"errors":0,"results":0}`.

## Findings

1. High severity, high confidence: retrieval policy becomes fixed inside `MediaDBRetriever`, not at a single upstream handoff.
   - `MediaDBRetriever.retrieve()` owns the first hard split between media-level FTS, chunk-level FTS, and late chunking (`683+`), while `MediaDBRetriever._retrieve_vector()` owns collection selection, scoped embedding-model lookup, allowed-media filters, metadata filter composition, HyDE branching, HyDE merge policy, and FTS fallback (`1562+`).
   - This means the authoritative retrieval output for media is the `Document` list emitted by retriever code before later pipeline stages touch it. The policy is not merely “search the vector store,” it is “resolve the collection, decide the embedding model, merge filters, maybe run HyDE, maybe fall back, then materialize documents.”
   - Stage 2 left open where retrieval policy actually becomes fixed; this stage confirms the answer is “inside the concrete retriever,” especially for Media DB.

2. High severity, high confidence: `MultiDatabaseRetriever` is not a source-agnostic coordinator; it knows concrete retriever classes and their private retrieval modes.
   - `MultiDatabaseRetriever.retrieve()` switches on retriever type, mutates per-call config on the retriever instance, and directly chooses `retrieve_hybrid()`, `_retrieve_vector()`, `_retrieve_fts()`, or source-specific kwargs such as `allowed_media_ids` and `allowed_note_ids` (`3133+`).
   - New source types therefore require changes in two places: the source retriever and the coordinator that dispatches it.
   - This is the clearest place where database retrievers expose source-specific behavior across the extension boundary instead of behind one public retrieval contract.

3. Medium severity, high confidence: vector-store factory and adapter ownership stop at instantiation; namespace and collection naming are owned elsewhere and are only loosely consistent.
   - `VectorStoreFactory.create_from_settings()` chooses adapter type, DSN/embedding config, and even records `collection_prefix` in `VectorStoreConfig` (`103+`, `160+`), but Stage 4 review found no retrieval-side consumer of `collection_prefix`.
   - The actual retrieval path hardcodes `user_{user_id}_media_embeddings` when no `index_namespace` is provided, treats `index_namespace` as either a concrete collection name or a wildcard/pattern list, and passes those names straight to adapter search methods (`1596+`, `1694+`, `1703+`).
   - The same hardcoded collection contract appears again in API-side invalidation (`app/api/v1/utils/rag_cache.py`), which constructs `user_{user_id}_media_embeddings` directly before calling `delete_by_filter()`.
   - Result: user namespaces, collection names, and adapter ownership are coupled by convention rather than by one owned resolver.

4. Medium severity, medium confidence: query shaping helpers leak retrieval-policy knowledge instead of remaining neutral preprocessing utilities.
   - `classify_query()` and `classify_and_reformulate()` decide whether local DB, web, academic, or discussion retrieval should run (`246+`, `450+`), which makes classifier output part of retrieval-source policy rather than just query analysis.
   - `multi_strategy_expansion()` reuses `corpus` for synonym resolution and the pipeline passes `index_namespace` as that corpus key, so vector-store namespace semantics bleed into synonym expansion (`546+`; Stage 2 traced the `index_namespace` call site).
   - `QueryExpansionRetriever` exists as a wrapper contract (`679+`), but the main pipeline relies on compatibility helpers instead of this wrapper, so the wrapper is not the authoritative retrieval-expansion seam.
   - `web_fallback.py` is a cleaner utility, but it is still a retrieval-source branch whose activation depends on upstream retrieval policy rather than on an isolated source-routing layer.
   - `media_search.py` stays reviewed in this stage only as a side-source utility surface. Its activation path is through the Stage 5 research-agent side path, not a claim this report proves through Stage 4 retrieval-policy evidence.

5. Medium severity, medium confidence: semantic cache internals are relatively cohesive, but tenant identity and invalidation policy still bleed across core and API layers.
   - Inside `semantic_cache.py`, ownership is clear: namespace normalization, persist-path sanitization, shared-instance keys, and similarity lookup all live together (`618+`, `739+`, `803+`, `840+`).
   - Outside the module, caller code still decides what a namespace means. The API invalidation helper derives namespaces from both username and user ID, then clears caches by those values rather than by one canonical tenant identifier.
   - The cache module therefore owns safe cache persistence and singleton scoping, but not the identity policy that determines which caller maps to which cache namespace.

6. Low severity, high confidence: `hyde.py` is not the operational owner of HyDE retrieval behavior.
   - The actual Stage 4 HyDE retrieval behavior that tests protect is the settings-driven `kind="hyde_q"` branch and merge logic inside `database_retrievers.py` (`1669+` through the merge branches), not the helper functions in `hyde.py`.
   - `hyde.py` is currently used by `unified_pipeline.py` and `post_generation_verifier.py`, not by `MediaDBRetriever._retrieve_vector()`.
   - The module boundary therefore reads cleaner than the real behavior boundary: the HyDE utility module exists, but retriever policy is independently implemented elsewhere.

## Suggested Refactor/Actions

- Introduce a small retrieval-plan object built once after Stage 3 request shaping. It should carry normalized sources, canonical tenant/namespace, collection targets, filter payloads, expansion mode, and HyDE mode so retrievers consume resolved policy instead of re-deciding it.
- Replace `MultiDatabaseRetriever` instance-type switching with one public retriever contract. If a source needs special fields, pass a source-scoped request object rather than calling private methods like `_retrieve_vector()` and `_retrieve_fts()` from the coordinator.
- Centralize collection and namespace resolution behind one core resolver used by retrievers, vector-store invalidation helpers, and adapter factory callers. Either remove `collection_prefix` from `VectorStoreConfig` or make it authoritative everywhere.
- Move HyDE merge behavior behind a dedicated retrieval helper or service that is explicitly shared by retrievers and later verification flows. `hyde.py` should either own HyDE retrieval inputs/outputs or remain a pure utility module with no duplicated policy elsewhere.
- Keep semantic cache persistence and similarity logic where it is, but move namespace derivation and invalidation policy into a core cache coordinator. API code should pass canonical tenant identifiers, not assemble alias sets ad hoc.
- If `QueryExpansionRetriever` is intended to be the extension seam, route pipeline usage through it. If not, remove the wrapper and make the compatibility helper path the documented contract.

## Coverage Gaps

- The requested test slice does not prove end-to-end PGVector parity in this environment because all PG-backed parity and integration tests skipped on `127.0.0.1:5432` connectivity failure.
- No reviewed test proves that collection naming is centrally owned or that `collection_prefix` in `VectorStoreConfig` has any effect. The current naming convention is protected only indirectly through hardcoded expectations.
- No reviewed test exercises `QueryExpansionRetriever` as the authoritative pipeline seam, and no Stage 4 test covers `fallback_to_web_search()` or `merge_web_results()`.
- `test_query_classifier.py` only protects response parsing. It does not cover heuristic routing, LLM fallback behavior, or the source-routing decisions that most affect retrieval ownership.
- `test_media_search.py` only covers happy-path normalization. It does not cover failure behavior or the policy question of who is allowed to trigger media-side retrieval.
- The semantic-cache tests cover tenant isolation and path safety inside the cache module, but not the cross-layer namespace mapping chosen by API invalidation helpers.
- `hyde.py` itself has no direct Stage 4 test coverage. The protected HyDE behavior is retriever-owned, not utility-owned.

## Exit Note

Stage 4 settles the retrieval-side boundary left open by Stages 2 and 3: the authoritative retrieval outputs are the `Document` lists emitted by concrete retrievers and fused by `MultiDatabaseRetriever`, with media retrieval policy concretized inside `MediaDBRetriever` before later pipeline stages see the data. Stage 5 should verify that reranking, post-verification, research-agent media/web paths, and response-writing logic do not silently create a second authoritative document set or re-open collection/cache ownership that Stage 4 found to belong upstream.
