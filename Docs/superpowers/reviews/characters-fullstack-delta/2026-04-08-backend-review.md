# Characters Backend Delta Review

## Scope

- backend Character lifecycle and persistence
- backend import/export and retrieval
- backend chat-coupled behavior

## Baseline Artifacts Checked

- Docs/superpowers/specs/2026-03-23-characters-backend-review-design.md
- Docs/superpowers/reviews/characters-backend/2026-03-23-stage2-api-crud-versioning.md
- Docs/superpowers/reviews/characters-backend/2026-03-23-stage3-import-validation-export.md
- Docs/superpowers/reviews/characters-backend/2026-03-23-stage4-exemplars-worldbooks-search.md
- Docs/superpowers/reviews/characters-backend/2026-03-23-stage5-chat-rate-limit-synthesis.md
- Docs/superpowers/specs/2026-04-07-characters-backend-remediation-design.md

## Code Paths Reviewed

- tldw_Server_API/app/api/v1/endpoints/characters_endpoint.py: lifecycle, import/export, versions, exemplars, and world-book routes
- tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py: Character chat session creation, completion prep, greeting injection, memory injection, and routing
- tldw_Server_API/app/api/v1/endpoints/character_messages.py: Character message persistence and message retrieval paths
- tldw_Server_API/app/core/Character_Chat/modules/character_db.py: persistence normalization and lifecycle helpers
- tldw_Server_API/app/core/Character_Chat/modules/character_io.py: import, export, and file-format handling
- tldw_Server_API/app/core/Character_Chat/modules/character_chat.py: chat-coupled Character behavior
- tldw_Server_API/app/core/Character_Chat/character_rate_limiter.py and character_limits.py: quota and throttle behavior
- tldw_Server_API/app/core/Character_Chat/world_book_manager.py and app/core/Chat/chat_characters.py: retrieval and cross-module Character coupling
- tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py: Character DB invariants, restore, revert, and version history
- tldw_Server_API/app/api/v1/schemas/character_schemas.py and character_memory_schemas.py: request and response contracts

## Tests Reviewed

- tldw_Server_API/tests/Characters/test_characters_endpoint.py
- tldw_Server_API/tests/Characters/test_character_functionality_db.py
- tldw_Server_API/tests/Characters/test_character_chat_lib.py
- tldw_Server_API/tests/Characters/test_character_chat_greetings_api.py
- tldw_Server_API/tests/Characters/test_ccv3_parser.py
- tldw_Server_API/tests/Character_Chat_NEW/integration/test_character_api.py
- tldw_Server_API/tests/Character_Chat_NEW/integration/test_character_chat_auto_routing.py
- tldw_Server_API/tests/Character_Chat_NEW/integration/test_character_chat_stream_and_persist.py
- tldw_Server_API/tests/Character_Chat_NEW/integration/test_character_exemplars_api.py
- tldw_Server_API/tests/Character_Chat_NEW/unit/test_character_memory.py
- tldw_Server_API/tests/Character_Chat_NEW/unit/test_world_book_manager.py
- tldw_Server_API/tests/unit/test_character_rate_limiter.py

## Validation Commands

- `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Characters/test_characters_endpoint.py tldw_Server_API/tests/Characters/test_character_functionality_db.py tldw_Server_API/tests/Characters/test_character_chat_lib.py tldw_Server_API/tests/Characters/test_ccv3_parser.py tldw_Server_API/tests/Character_Chat_NEW/integration/test_character_api.py tldw_Server_API/tests/Character_Chat_NEW/unit/test_png_export.py -q` — fail
- Failing test: `tldw_Server_API/tests/Characters/test_character_functionality_db.py::TestCharacterCardUpdate::test_empty_update_payload_remains_a_noop` (`TypeError: CharactersRAGDB.get_character_card_by_id() got an unexpected keyword argument 'include_deleted'`). The rest of the lifecycle/import-export slice completed cleanly enough to show no adjacent breakage around the current review findings.
- `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Character_Chat_NEW/integration/test_character_exemplars_api.py tldw_Server_API/tests/Character_Chat_NEW/unit/test_world_book_manager.py tldw_Server_API/tests/ChaChaNotesDB/test_character_exemplars_db.py tldw_Server_API/tests/RAG/test_dual_backend_characters_retriever.py -q` — pass
- Confidence gained: the targeted retrieval and world-book slice passed (`67 passed, 2 skipped`), so no new retrieval-sensitive Character regression was reproduced in this Stage 3 sample.
- `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Characters/test_character_chat_greetings_api.py tldw_Server_API/tests/Character_Chat_NEW/integration/test_character_chat_auto_routing.py tldw_Server_API/tests/Character_Chat_NEW/integration/test_character_chat_stream_and_persist.py tldw_Server_API/tests/Character_Chat/unit/test_chat_session_character_scope_api.py tldw_Server_API/tests/unit/test_character_rate_limiter.py tldw_Server_API/tests/Streaming/test_character_chat_sse_unified_flag.py -q` — fail
- Failing tests: `tldw_Server_API/tests/Streaming/test_character_chat_sse_unified_flag.py::test_character_chat_streaming_unified_sse`, `::test_character_chat_streaming_unified_sse_slow_async_heartbeat`, `::test_character_chat_streaming_unified_sse_provider_duplicate_done`, `::test_character_chat_streaming_unified_sse_chunk_limit`, `::test_character_chat_streaming_unified_sse_byte_limit`, and `::test_character_chat_streaming_unified_sse_provider_error_emits_done`, each bootstrapping into `GET /api/v1/characters/` returning `503`. The adjacent persist and rate-limit tests in that slice otherwise passed.

## Findings

### Confirmed finding: manual character-memory extraction checks a conversation owner field that the conversations table does not store

- Severity: Medium
- Type: correctness
- Novelty: net-new
- Baseline artifacts checked: Docs/superpowers/reviews/characters-backend/2026-03-23-stage5-chat-rate-limit-synthesis.md; Docs/superpowers/specs/2026-04-07-characters-backend-remediation-design.md
- Why it matters: The newer manual memory-extraction path can reject valid, owned chats before any extraction work starts, which leaves the character-memory feature unusable from its API surface even though the underlying storage and extraction helpers exist.
- Current evidence: Direct-edge inspection included `tldw_Server_API/app/api/v1/endpoints/character_memory.py:231-235`, where manual extraction authorizes against `conversation.get("user_id")`, and the adjacent unit coverage in `tldw_Server_API/tests/Character_Chat_NEW/unit/test_character_memory.py`, which does not exercise the endpoint ownership check. That read was compared against the conversations schema and conversation write/read paths, which store and expose conversation ownership through `client_id` rather than `user_id` (`tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py:744-756`, `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py:19310-19399`). None of the reviewed Character backend tests exercise `/characters/{character_id}/memories/extract`, so this mismatch is currently unguarded.
- Targeted validation note: None of the executed Stage 3 pytest slices covered `/api/v1/characters/{character_id}/memories/extract` or the endpoint ownership branch in `character_memory.py`, so runtime confirmation was not feasible within this task's prescribed test set.
- Validation status: environment-limited, confidence unchanged

### Probable risk: streamed assistant persistence skips message-cap enforcement when quota checks fail internally

- Severity: Medium
- Type: correctness
- Novelty: net-new
- Baseline artifacts checked: Docs/superpowers/reviews/characters-backend/2026-03-23-stage5-chat-rate-limit-synthesis.md; Docs/superpowers/specs/2026-04-07-characters-backend-remediation-design.md
- Why it matters: This candidate is narrower than the March Stage 5 quota issue. The historical finding was about non-atomic count-then-check enforcement under concurrency; this delta item is about a distinct fail-open exception path in the newer streamed-persist write flow, where internal quota-check failures do not stop persistence.
- Current evidence: `persist_streamed_assistant_message()` catches non-HTTP exceptions around `count_messages_for_conversation()` and `check_message_limit()` and only logs before continuing at `tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py:6098-6105`. That is a separate failure mode from the March Stage 5 non-atomic enforcement review and it differs from the direct message-send path, which converts the same counting failure into `503` at `tldw_Server_API/app/api/v1/endpoints/character_messages.py:263-281`. The reviewed streaming persistence tests only cover happy-path persistence and speaker metadata, not quota-failure behavior (`tldw_Server_API/tests/Character_Chat_NEW/integration/test_character_chat_stream_and_persist.py`).
- Targeted validation note: The executed chat slice did hit `/api/v1/chats/{chat_id}/completions/persist` happy-path coverage and nearby limiter coverage, but it did not inject a `count_messages_for_conversation()` or `check_message_limit()` exception through the persist endpoint. The same slice also produced unrelated SSE bootstrap failures from `test_character_chat_sse_unified_flag.py`, where `GET /api/v1/characters/` returned `503` before the streaming assertions ran.
- Validation status: not reproduced by targeted pytest slice, downgraded to probable risk

## Residual Risks / Open Questions

- The March review set did not cover the newer character-memory API surface directly, so any broader claims about memory CRUD beyond the ownership mismatch above still need targeted runtime validation.
- `character_chat_sessions.py` now carries process-local memory-extraction counters, and the static pass cannot tell whether their lifecycle is acceptable under multi-worker or long-running workloads without Stage 3 runtime checks.

## Coverage Gaps

- No reviewed test exercises `/api/v1/characters/{character_id}/memories/extract`, so the conversation-ownership contract for manual character-memory extraction is currently untested.
- No reviewed test forces `count_messages_for_conversation()` or rate-limiter failures through `/api/v1/chats/{chat_id}/completions/persist`, so the persist endpoint's fail-open branch is unguarded.

## Improvements

- Add an endpoint regression test for manual character-memory extraction that uses a real chat session and asserts ownership is checked against the conversation field actually stored by `ChaChaNotes_DB`.
- Add a persist-endpoint regression test that injects a message-count or limiter failure and asserts the API returns `503` instead of persisting the assistant reply.

## Exit Note

Backend validation complete. All surviving confirmed findings now have either executed support or a clearly stated reason why runtime confirmation was not feasible in this environment.
