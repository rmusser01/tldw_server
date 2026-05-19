# Stage 4 Exemplars, World Books, and Search/Retrieval Behavior

## Scope

- Reviewed exemplar CRUD, hybrid search, selector packing, embedding sync, telemetry helpers, world-book attach/detach, world-book context processing, and DB-level exemplar persistence/search.
- Focused on fallback behavior, ownership boundaries, deleted-record visibility, ordering/pagination semantics, and whether best-effort recovery can hide ranking or integrity problems.
- Analysis-only review. No source code changes were made.

## Code Paths Reviewed

- `tldw_Server_API/app/api/v1/endpoints/characters_endpoint.py`
  - exemplar payload flattening and response conversion helpers
  - best-effort embedding sync/delete helpers
  - hybrid exemplar search fallback at `_search_character_exemplars_hybrid_best_effort()`
  - exemplar create/get/update/delete/search/select-debug endpoints
  - world-book list/create/get/update/delete/import/export/statistics endpoints
  - character/world-book attach, detach, list, and process-context endpoints
- `tldw_Server_API/app/core/Character_Chat/world_book_manager.py`
  - world-book CRUD
  - entry CRUD
  - `attach_to_character()`, `detach_from_character()`, `get_character_world_books()`
  - `process_context()`
  - world-book search and import/export helpers
- `tldw_Server_API/app/core/Character_Chat/modules/persona_exemplar_selector.py`
  - turn classification, candidate retrieval, safety gating, scoring, MMR, and budget packing
- `tldw_Server_API/app/core/Character_Chat/modules/persona_exemplar_embeddings.py`
  - vector index scoring, embedding fallback scoring, upsert, and delete helpers
- `tldw_Server_API/app/core/Character_Chat/modules/persona_exemplar_telemetry.py`
  - IOO/IOR/LCS telemetry and refusal/copy-ratio flags
- `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
  - character exemplar create/get/list/update/delete/search persistence
  - FTS-backed search behavior and deleted-row filtering
  - persona exemplar persistence/lookup helpers used by the selector module

## Tests Reviewed

- `tldw_Server_API/tests/Character_Chat_NEW/integration/test_character_exemplars_api.py`
  - strong coverage for exemplar CRUD/search/debug-selection happy paths
  - medium coverage for embedding fallback and sync hooks
  - weak coverage for fallback pagination totals and error observability
- `tldw_Server_API/tests/ChaChaNotesDB/test_character_exemplars_db.py`
  - strong coverage for DB CRUD/search and deleted-row exclusion
  - weak coverage for hybrid-search fallback totals and cross-branch ownership checks
- `tldw_Server_API/tests/Character_Chat_NEW/unit/test_persona_exemplar_selector.py`
  - strong coverage for budget, MMR, safety gating, and malformed search/list fallback
  - no direct coverage for DB-backed pagination totals or ranking metadata on fallback
- `tldw_Server_API/tests/Character_Chat_NEW/unit/test_persona_exemplar_embeddings.py`
  - strong coverage for vector scoring, merge behavior, and nonfatal error swallowing
- `tldw_Server_API/tests/Character_Chat_NEW/unit/test_persona_exemplar_telemetry.py`
  - narrow helper-level coverage only
- `tldw_Server_API/tests/Character_Chat_NEW/unit/test_world_book_manager.py`
  - strong coverage for world-book CRUD, entry filters, cache behavior, and process-context behavior
- `tldw_Server_API/tests/Characters/test_characters_world_book_permissions_unit.py`
  - strong coverage for attach/detach/list permission mapping and entry-count optimization
- `tldw_Server_API/tests/Character_Chat/test_world_book_negatives_and_new_endpoint.py`
  - useful negative-path coverage for world-book endpoints and chat completions
  - the full-suite run hit order-dependent `503` failures here; see Validation Results
- `tldw_Server_API/tests/RAG/test_dual_backend_characters_retriever.py`
  - useful smoke coverage for retriever parity
  - postgres variants were skipped in the validation environment because the local Postgres backend was unavailable

## Validation Commands

- Route and DB surface mapping:
```bash
rg -n "exemplar|world-book|world_books|process_context|attach_to_character|detach_from_character|get_character_world_books|search_character_exemplars|select_character_exemplars" \
  tldw_Server_API/app/api/v1/endpoints/characters_endpoint.py \
  tldw_Server_API/app/core/Character_Chat/world_book_manager.py \
  tldw_Server_API/app/core/Character_Chat/modules/persona_exemplar_selector.py \
  tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py
```
- Targeted review run:
```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Character_Chat_NEW/integration/test_character_exemplars_api.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_character_exemplars_db.py \
  tldw_Server_API/tests/Character_Chat_NEW/unit/test_persona_exemplar_selector.py \
  tldw_Server_API/tests/Character_Chat_NEW/unit/test_persona_exemplar_embeddings.py \
  tldw_Server_API/tests/Character_Chat_NEW/unit/test_persona_exemplar_telemetry.py \
  tldw_Server_API/tests/Character_Chat_NEW/unit/test_world_book_manager.py \
  tldw_Server_API/tests/Characters/test_characters_world_book_permissions_unit.py \
  tldw_Server_API/tests/Character_Chat/test_world_book_negatives_and_new_endpoint.py \
  tldw_Server_API/tests/RAG/test_dual_backend_characters_retriever.py -v
```
- Isolated rerun for the failing world-book file:
```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Character_Chat/test_world_book_negatives_and_new_endpoint.py -vv
```
- Validation results:
  - Full targeted suite: `100 passed, 3 skipped, 4 failed`
  - All four failures were in `tldw_Server_API/tests/Character_Chat/test_world_book_negatives_and_new_endpoint.py` and returned `503 Service Unavailable`
  - Isolated world-book rerun: `4 passed, 1 skipped`

## Findings

- Medium | correctness | Hybrid exemplar search falls back to lexical slices but reports a truncated `total` when embedding scoring is unavailable.
  - `_search_character_exemplars_hybrid_best_effort()` fetches a capped lexical candidate pool, then on missing `user_id` or embedding-scoring failure returns `lexical_candidates[offset:offset + limit]` with `total = len(lexical_candidates)` at `tldw_Server_API/app/api/v1/endpoints/characters_endpoint.py:635-724`.
  - That `total` is only the size of the prefetched window, not the DB-side total from `search_character_exemplars()`, so pagination metadata can under-report the real match count.
  - Because the function logs and falls back instead of surfacing the ranking failure, callers can get a normal 200 response while the ranking backend is actually degraded.

- Low | contract | World-book deletion/detach responses write `world_book_id` into `DeletionResponse.character_id`.
  - `detach_world_book_from_character()` constructs `DeletionResponse(message=..., character_id=world_book_id)` instead of returning the character id at `tldw_Server_API/app/api/v1/endpoints/characters_endpoint.py:2530-2551`.
  - The same field reuse appears in the world-book delete path at `tldw_Server_API/app/api/v1/endpoints/characters_endpoint.py:2147-2149`.
  - This does not break persistence, but it is a response-shape mismatch that can mislead client code and makes the contract harder to reason about.
- Low | performance | Hybrid exemplar search expands and rescoring candidate pools aggressively for paginated requests.
  - `_search_character_exemplars_hybrid_best_effort()` scales `candidate_pool_size` with `limit + offset`, then runs an FTS search and may backfill additional rows from `list_character_exemplars()` before embedding scoring the full merged pool at `tldw_Server_API/app/api/v1/endpoints/characters_endpoint.py:654-709`.
  - That work is bounded by `_EXEMPLAR_SEARCH_HYBRID_CANDIDATE_CAP`, but large offsets still increase the amount of DB and embedding work required to serve a single page.
  - Impact: exemplar search latency can degrade disproportionately for deeper pagination or repeated fallback scenarios, and the current API does not surface whether the response was served from the more expensive hybrid path.

## Coverage Gaps

- No test asserts hybrid exemplar fallback pagination totals when embeddings are unavailable or raise.
- No test asserts the world-book delete/detach response field semantics.
- No direct test covers cached world-book visibility after an external delete/disable, though the request-scoped service pattern reduces exposure.
- Telemetry helper coverage is unit-only; no endpoint currently consumes it directly in this stage.

## Improvements

- Add a regression test for hybrid exemplar search when embedding scoring is disabled or fails, and assert the returned `total` semantics explicitly.
- Add a contract test for world-book detach/delete responses so `character_id` cannot silently drift to `world_book_id`.
- Consider surfacing embedding-ranking degradation more explicitly if callers need to distinguish lexical fallback from ranked search.
- If exemplar search pagination becomes latency-sensitive, consider a cheaper fallback path or explicit pagination caps so deeper pages do not require repeated hybrid rescoring of oversized candidate pools.
- If telemetry is intended to be user-facing, wire it into an endpoint or document that it remains an internal diagnostic helper.

## Exit Note

- The broad targeted validation run completed with `100 passed, 3 skipped, 4 failed`.
- The four failures were all in `tldw_Server_API/tests/Character_Chat/test_world_book_negatives_and_new_endpoint.py` and all returned `503 Service Unavailable` from `tldw_Server_API/app/api/v1/API_Deps/ChaCha_Notes_DB_Deps.py`'s initialization path.
- An isolated rerun of that same file passed cleanly with `4 passed, 1 skipped`, which makes the earlier 503s an order-dependent environment/state issue rather than a stable product regression.
- Stage 4 review is complete and the only modified file is this report.
