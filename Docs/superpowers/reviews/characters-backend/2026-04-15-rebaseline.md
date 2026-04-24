# Wave 2 Characters Backend Rebaseline

- Date: 2026-04-15
- Scope: Current-tree rebaseline for the Wave 2 ChaChaNotes and character/chat lifecycle surface
- Basis: Re-ran the previously identified mixed-suite failure slices on `dev` and checked whether the highest-priority March findings were still live in source or tests

## Summary

The March 2026 characters-backend review is no longer a safe list of assumed-live defects. Some of its highest-confidence lifecycle findings are already closed in the current tree, while the order-dependent ChaChaNotes bootstrap failures are still live and now have two independently reproduced mixed-suite manifestations:

- Stage 5 mixed suite still fails in the SSE bootstrap path with six `503 Service Unavailable` responses.
- Stage 4 mixed suite still fails in the world-book path with four `503 Service Unavailable` responses.
- The isolated SSE rerun still passes, so the current failure shape remains order-dependent lifecycle poisoning rather than a stable endpoint regression.

## Current-Tree Validation

### Stage 5 Mixed Suite

Command:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Character_Chat/test_character_chat_endpoints.py \
  tldw_Server_API/tests/Character_Chat_NEW/integration/test_character_chat_auto_routing.py \
  tldw_Server_API/tests/Character_Chat_NEW/integration/test_character_chat_stream_and_persist.py \
  tldw_Server_API/tests/Character_Chat/unit/test_chat_session_character_scope_api.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_conversation_character_scope_filters.py \
  tldw_Server_API/tests/Character_Chat_NEW/unit/test_character_chat_default_provider.py \
  tldw_Server_API/tests/Character_Chat_NEW/unit/test_character_chat_manager.py \
  tldw_Server_API/tests/Character_Chat_NEW/unit/test_character_memory.py \
  tldw_Server_API/tests/unit/test_character_rate_limiter.py \
  tldw_Server_API/tests/Streaming/test_character_chat_sse_unified_flag.py \
  tldw_Server_API/tests/e2e/test_chats_and_characters.py \
  tldw_Server_API/tests/server_e2e_tests/test_character_chats_workflow.py -v
```

Observed result:

- `100 passed`
- `5 skipped`
- `6 failed`
- `1 error`

Failure shape:

- All six product failures were in `tldw_Server_API/tests/Streaming/test_character_chat_sse_unified_flag.py`
- The failing bootstrap request was `GET /api/v1/characters/`
- Each failing case returned `503 Service Unavailable`
- The one additional error was the known sandbox-only server e2e port-binding failure

### Isolated SSE Rerun

Command:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Streaming/test_character_chat_sse_unified_flag.py -v
```

Observed result:

- `6 passed`
- `5 warnings`

Interpretation:

- The SSE surface is still healthy in isolation.
- The current bug remains a mixed-suite lifecycle/state leak.

### Stage 4 Mixed World-Book Suite

Command:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Character_Chat/test_world_book_and_limits.py \
  tldw_Server_API/tests/Character_Chat/test_world_book_negatives.py \
  tldw_Server_API/tests/Character_Chat/test_world_book_manager_legacy.py \
  tldw_Server_API/tests/Character_Chat/test_character_chat_endpoints.py \
  tldw_Server_API/tests/Characters/test_characters_endpoint.py \
  tldw_Server_API/tests/Characters/test_characters_world_book_permissions_unit.py \
  tldw_Server_API/tests/Character_Chat/test_world_book_negatives_and_new_endpoint.py \
  tldw_Server_API/tests/RAG/test_dual_backend_characters_retriever.py -v
```

Observed result:

- `103 passed`
- `4 skipped`
- `4 failed`
- `7 warnings`

Failure shape:

- All four product failures were in `tldw_Server_API/tests/Character_Chat/test_world_book_negatives_and_new_endpoint.py`
- Those failures also returned `503 Service Unavailable`
- The failing requests were again character bootstrap or world-book requests that depend on ChaChaNotes initialization
- PostgreSQL retriever skips remained environmental rather than product regressions

## Closed Since March

These findings should not be reimplemented blindly:

- Active-row restore no longer succeeds spuriously.
  - Current tests now encode the conflict behavior:
    - `tldw_Server_API/tests/Characters/test_character_functionality_db.py::test_restore_character_card_raises_conflict_when_row_is_already_active`
    - `tldw_Server_API/tests/Characters/test_characters_endpoint.py::test_restore_character_active_row_returns_409`
- Empty update payloads are now explicitly treated as a deliberate no-op contract rather than an unreviewed accidental behavior.
  - Current tests encode that behavior in:
    - `tldw_Server_API/tests/Characters/test_character_functionality_db.py::test_empty_update_payload_remains_a_noop`

## Still Live

### Highest Priority

- Order-dependent ChaChaNotes bootstrap poisoning remains live across both the SSE and world-book mixed suites.
  - Evidence: Stage 5 reproduced six SSE `503`s, while the isolated SSE rerun passed.
  - Evidence: Stage 4 reproduced four world-book `503`s in a separate mixed suite.
  - Likely center of gravity remains `tldw_Server_API/app/api/v1/API_Deps/ChaCha_Notes_DB_Deps.py`, especially shared init, shutdown, and reset behavior.

### Lower Priority But Still Live In Source

- World-book delete and detach responses still overload `DeletionResponse.character_id` for non-character resources.
  - This is misleading but lower risk than the bootstrap failures.

## Rebaseline Conclusion

Wave 2 should start with the ChaChaNotes dependency lifecycle, not with the March restore/empty-update findings. The active contract to fix is:

1. mixed-suite bootstrap must not poison later request-path initialization
2. both reproduced mixed suites must go green after the fix
3. once the lifecycle leak is fixed, clean up the remaining misleading world-book response ids inside the same bounded subsystem wave
