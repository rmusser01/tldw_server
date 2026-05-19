# Stage 5 Chat Coupling, Rate Limits, and Final Synthesis

## Scope
- Reviewed how character state and ChaChaNotes dependency behavior affect chat/session APIs, including scope filtering, default-provider resolution, rate limiting, message caps, streamed persistence, and shutdown/init behavior.
- Consolidated Stage 2 through Stage 4 into one ranked backend synthesis so maintainers can fix the highest-impact issues first.
- Analysis-only stage. No source code changes were made.

## Code Paths Reviewed
- `tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py`
  - default provider resolution at `:212-257`
  - chat create quota check at `:3077-3102`
  - completion message-cap pre-check at `:4649-4663`
  - participant/turn-context resolution at `:5981-6112`
- `tldw_Server_API/app/api/v1/endpoints/character_messages.py`
  - send-message guardrails and per-chat message-cap check at `:260-282`
- `tldw_Server_API/app/core/Character_Chat/modules/character_chat.py`
  - `load_chat_and_character()` at `:613-734`
  - `start_new_chat_session()` at `:737-920`
- `tldw_Server_API/app/core/Character_Chat/character_rate_limiter.py`
  - request-throttling behavior at `:139-182`
- `tldw_Server_API/app/core/Character_Chat/character_limits.py`
  - hard chat/message guardrails at `:89-168`
- `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
  - conversation scope filtering and character-scoped counts at `:19120-19850`
- `tldw_Server_API/app/api/v1/API_Deps/ChaCha_Notes_DB_Deps.py`
  - shared init wait path at `:373-418`
  - executor-backed DB init at `:420-457`
  - warmup/default-character task scheduling at `:460-537`
  - shutdown draining at `:570-611`
- Cross-stage synthesis references:
  - `Docs/superpowers/reviews/characters-backend/2026-03-23-stage2-api-crud-versioning.md`
  - `Docs/superpowers/reviews/characters-backend/2026-03-23-stage3-import-validation-export.md`
  - `Docs/superpowers/reviews/characters-backend/2026-03-23-stage4-exemplars-worldbooks-search.md`

## Tests Reviewed
- `tldw_Server_API/tests/Character_Chat/test_character_chat_endpoints.py`
  - useful integration smoke coverage for chat/session endpoints
  - medium coverage for quota/error semantics
- `tldw_Server_API/tests/Character_Chat_NEW/integration/test_character_chat_auto_routing.py`
  - strong coverage for routing/default-provider selection paths
- `tldw_Server_API/tests/Character_Chat_NEW/integration/test_character_chat_stream_and_persist.py`
  - strong coverage for streamed persistence and active-speaker identity preservation
- `tldw_Server_API/tests/Character_Chat/unit/test_chat_session_character_scope_api.py`
  - strong coverage for conversation scope filtering
- `tldw_Server_API/tests/ChaChaNotesDB/test_conversation_character_scope_filters.py`
  - strong DB-level coverage for character/workspace/global scope filters
- `tldw_Server_API/tests/Character_Chat_NEW/unit/test_character_chat_default_provider.py`
  - strong, narrow coverage for default-provider resolution behavior in test mode
- `tldw_Server_API/tests/Character_Chat_NEW/unit/test_character_chat_manager.py`
  - useful unit coverage for chat manager helper behavior
- `tldw_Server_API/tests/Character_Chat_NEW/unit/test_character_memory.py`
  - useful unit coverage for memory extraction/injection helpers
- `tldw_Server_API/tests/unit/test_character_rate_limiter.py`
  - strong unit coverage for Resource Governor enabled/disabled request-throttling semantics
  - weak coverage for endpoint-level enforcement behavior
- `tldw_Server_API/tests/Streaming/test_character_chat_sse_unified_flag.py`
  - good streaming-path coverage
  - exposed order-dependent `503` failures in a larger mixed suite, but passed cleanly in isolation
- `tldw_Server_API/tests/e2e/test_chats_and_characters.py`
  - environment-limited; skipped when no local server is available
- `tldw_Server_API/tests/server_e2e_tests/test_character_chats_workflow.py`
  - environment-limited; server fixture failed in this sandbox while binding a free port

## Validation Commands
- Chat-coupling and rate-limit surface map:
```bash
rg -n "character_id|conversation_character_scope|load_chat_and_character|get_conversations_for_character|count_conversations_for_user_by_character|first_message|default provider|auto_routing" \
  tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py \
  tldw_Server_API/app/api/v1/endpoints/character_messages.py \
  tldw_Server_API/app/core/Character_Chat/modules/character_chat.py \
  tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py
```
- Rate-limit and dependency-init surface map:
```bash
rg -n "check_rate_limit|check_character_limit|resource governor|legacy deprecation|character_chat.default|operation|initialization timed out|default-character" \
  tldw_Server_API/app/core/Character_Chat/character_rate_limiter.py \
  tldw_Server_API/app/core/Character_Chat/character_limits.py \
  tldw_Server_API/app/api/v1/API_Deps/ChaCha_Notes_DB_Deps.py
```
- Targeted Stage 5 validation run:
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
- Isolated SSE rerun:
```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Streaming/test_character_chat_sse_unified_flag.py -v
```
- Validation results:
  - Full targeted suite: `94 passed, 5 skipped, 6 failed, 1 error in 79.19s`
  - The 6 failures were all in `tldw_Server_API/tests/Streaming/test_character_chat_sse_unified_flag.py` and all surfaced `503 Service Unavailable`
  - The failing bootstrap request in those SSE cases was `GET /api/v1/characters/`
  - Isolated SSE rerun: `6 passed in 7.14s`
  - `tldw_Server_API/tests/server_e2e_tests/test_character_chats_workflow.py` failed in setup with `PermissionError: [Errno 1] Operation not permitted` while binding a free port
  - `tldw_Server_API/tests/e2e/test_chats_and_characters.py` was skipped because the local server was unavailable in this sandbox
  - Stage 4 also saw order-dependent `503` failures in `tldw_Server_API/tests/Character_Chat/test_world_book_negatives_and_new_endpoint.py`: the mixed suite failed with `4` `503`s, while an isolated rerun passed with `4 passed, 1 skipped`

## Findings
- Findings are grouped by impact, and each item carries an explicit confidence label.

- High | high confidence | correctness | `restore_character_card()` accepts restore requests for already-active rows and returns success without enforcing `expected_version`.
  - The DB restore path returns early when `deleted == 0` at `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py:18220-18224`, before any optimistic-concurrency check.
  - The endpoint exposes that behavior directly through `restore_character_endpoint()` at `tldw_Server_API/app/api/v1/endpoints/characters_endpoint.py:1876-1896`.
  - Impact: a logically invalid restore request can succeed with `200 OK`, which weakens the lifecycle concurrency contract.

- Medium | medium confidence | availability | Character-adjacent endpoints can fail with order-dependent `503 Service Unavailable` responses when ChaChaNotes initialization state leaks across broader backend test slices.
  - `_get_or_init_db_instance()` returns `503` on shared-init wait timeout and executor-backed init timeout at `tldw_Server_API/app/api/v1/API_Deps/ChaCha_Notes_DB_Deps.py:386-418` and `:423-437`.
  - The same dependency layer also schedules background default-character work during warmup and request resolution at `tldw_Server_API/app/api/v1/API_Deps/ChaCha_Notes_DB_Deps.py:460-490` and `:533-536`, then drains those tasks/futures during shutdown at `:570-611`.
  - Evidence: Stage 4 mixed-suite world-book tests failed with four `503`s but passed in isolation, and the Stage 5 mixed suite failed six SSE tests with `503`s during `GET /api/v1/characters/` bootstrap while the isolated SSE rerun passed.
  - Inference: the exact branch still needs targeted debugging, but the current evidence points to order-dependent initialization or shutdown state in the ChaChaNotes dependency layer rather than a stable endpoint regression.

- Medium | high confidence | correctness | Empty updates bypass optimistic concurrency entirely.
  - `CharacterUpdate` permits an empty payload, the endpoint forwards `{}` to the DB layer, and `update_character_card()` returns `True` immediately for empty `card_data` at `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py:17924-17928`.
  - Impact: callers can submit a stale or fabricated `expected_version` with `{}` and still receive success, which weakens the versioning contract even though no row changes.

- Medium | high confidence | operations | Request-level character chat throttling is effectively fail-open unless Resource Governor enforcement is explicitly enabled and returns a decision.
  - `check_rate_limit()` allows the request when the limiter itself is disabled, when `_rg_character_enabled()` is false, when request enforcement is disabled, and when the Resource Governor decision is unavailable at `tldw_Server_API/app/core/Character_Chat/character_rate_limiter.py:146-164`.
  - `tldw_Server_API/tests/unit/test_character_rate_limiter.py:29-37` explicitly encodes the RG-disabled allow-all behavior.
  - Impact: operators can believe request throttling is active because `CharacterRateLimiter.enabled` is true, while production behavior still admits all requests unless the RG path is fully configured and healthy.

- Medium | medium confidence | correctness | Chat and message caps are enforced with count-then-check patterns that remain non-atomic under concurrency.
  - Chat creation counts existing conversations before enforcement at `tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py:3080-3092`.
  - Message sends and streamed-persist paths do the same for per-chat message counts at `tldw_Server_API/app/api/v1/endpoints/character_messages.py:263-268`, `tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py:4651-4658`, and `:6097-6101`.
  - Inference: because the limit check happens before the later insert/persist work and is not backed by an atomic DB constraint in these paths, concurrent requests can overshoot configured caps.

- Medium | high confidence | correctness | Malformed YAML and other non-JSON text imports are normalized into synthetic characters instead of being rejected.
  - The import path accepts text-ish content types at `tldw_Server_API/app/api/v1/endpoints/characters_endpoint.py:91-126` and `:809-817`.
  - `load_character_card_from_string_content()` falls back to a generated plain-text character when JSON and YAML parsing fail at `tldw_Server_API/app/core/Character_Chat/modules/character_io.py:745-801`.
  - Impact: typoed or partially malformed card files can import successfully as unexpected synthetic characters.

- Medium | high confidence | contract | Image-file imports bypass the DB image-normalization path, so equivalent avatars are stored differently depending on import format.
  - File import copies uploaded bytes into `parsed_card["image"]` and removes `image_base64` at `tldw_Server_API/app/core/Character_Chat/modules/character_io.py:1001-1011`.
  - The DB storage path only runs resize and WEBP normalization when `_prepare_character_data_for_db_storage()` sees `image_base64` at `tldw_Server_API/app/core/Character_Chat/modules/character_db.py:68-145`.
  - Impact: JSON/base64 imports and image-file imports do not converge to one storage format, which makes footprint and downstream export behavior transport-dependent.

- Medium | high confidence | contract | PNG export can generate files that the PNG importer later rejects on metadata-size limits.
  - `_encode_png_with_chara_metadata()` embeds the full card JSON into the `chara` text chunk without a matching export-side size cap at `tldw_Server_API/app/api/v1/endpoints/characters_endpoint.py:2936-2944` and `:3148-3151`.
  - The import path rejects metadata that exceeds `MAX_PNG_METADATA_BYTES` at `tldw_Server_API/app/core/Character_Chat/modules/character_io.py:231-238`, `:268-275`, and `:427-434`.
  - Impact: sufficiently large characters can export successfully as PNG and then fail re-import, so PNG is not a reliable round-trip format near the limit.

- Medium | high confidence | correctness | Version history and revert omit avatar state, so image-only edits are not fully diffable or reversible.
  - `_CHARACTER_REVERT_FIELDS` excludes `image` / `image_base64` at `tldw_Server_API/app/api/v1/endpoints/characters_endpoint.py:238-253`.
  - The sync-log snapshot consumed by `get_character_version_history()` also omits the avatar payload at `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py:1213-1251` and `:30792-30838`.
  - Impact: image-only character edits can advance the version while `/versions`, `/versions/diff`, and `/revert` still lack enough state to represent or restore the change.

- Medium | high confidence | correctness | Hybrid exemplar search falls back to lexical slices but reports a truncated `total` when embedding scoring is unavailable.
  - `_search_character_exemplars_hybrid_best_effort()` returns `lexical_candidates[offset:offset + limit]` with `total = len(lexical_candidates)` at `tldw_Server_API/app/api/v1/endpoints/characters_endpoint.py:635-724`.
  - That `total` reflects the prefetched fallback window rather than the DB-side total match count.
  - Impact: callers can receive a normal `200` while ranking is degraded and pagination metadata under-reports the actual result set.

- Low | high confidence | contract | World-book delete and detach responses write `world_book_id` into `DeletionResponse.character_id`.
  - The world-book delete and detach handlers both populate `character_id` with the world-book id at `tldw_Server_API/app/api/v1/endpoints/characters_endpoint.py:2147-2149` and `:2530-2551`.
  - Impact: this does not corrupt data, but it does make the response contract misleading for clients.

- Low | medium confidence | performance | Import/export image handling duplicates large buffers and validates images without an explicit pixel-count ceiling in the import path.
  - The import endpoint reads the full upload into memory before finishing validation at `tldw_Server_API/app/api/v1/endpoints/characters_endpoint.py:794-812`.
  - The image-validation and base64-normalization paths decode, verify, reopen, and sometimes re-encode large payloads at `tldw_Server_API/app/core/Character_Chat/modules/character_io.py:856-875` and `tldw_Server_API/app/core/Character_Chat/modules/character_db.py:80-124`.
  - Impact: request-size caps bound the worst case, but large allowed images still incur repeated full-buffer work.

- Low | high confidence | performance | Hybrid exemplar fallback grows the candidate pool with `offset + limit`, which makes deeper pages more expensive to serve.
  - `_search_character_exemplars_hybrid_best_effort()` expands its candidate pool before optional embedding rescoring at `tldw_Server_API/app/api/v1/endpoints/characters_endpoint.py:654-709`.
  - Impact: deeper pagination or repeated fallback scenarios can drive more DB and embedding work than the API shape suggests.

### Open questions and assumptions
- Open question: should restore of an already-active row ever be treated as idempotent success, or should restore require a deleted tombstone plus a matching version?
- Open question: should empty `{}` updates be rejected outright rather than treated as successful no-ops?
- Open question: is fail-open Resource Governor behavior intentional operational policy, or should request throttling fail closed once `CharacterRateLimiter` is enabled?
- Open question: should avatar bytes become part of version snapshots so history and revert are actually lossless for user-visible card changes?
- Assumption: the order-dependent `503` issue is in the ChaChaNotes dependency lifecycle, likely around shared init or default-character background work, but the exact root cause still needs a focused reproduction.

## Coverage Gaps
- No endpoint-level regression test forces the Resource Governor deny/unavailable paths through the live chat/session APIs.
- No concurrency test asserts chat-create or message-send caps under simultaneous requests, so the non-atomic quota paths are currently unguarded.
- No regression test currently covers the order-dependent `503` failures seen in mixed Stage 4 and Stage 5 suites.
- `tldw_Server_API/tests/server_e2e_tests/test_character_chats_workflow.py` could not run in this sandbox because the server fixture failed to bind a free port.
- `tldw_Server_API/tests/e2e/test_chats_and_characters.py` remained skipped because the local server was unavailable.
- Earlier-stage coverage gaps remain relevant here:
  - no test for restore on an already-active row
  - no test for empty update payloads with stale `expected_version`
  - no test for avatar-only version-history and revert behavior
  - no regression guard for PNG export near `MAX_PNG_METADATA_BYTES`
  - no regression test for hybrid exemplar fallback `total` semantics

## Improvements
- Investigate the ChaChaNotes dependency lifecycle first and add one mixed-suite regression that reproduces the order-dependent `503` failures before changing behavior.
- Close the optimistic-locking holes in restore and empty-update paths so lifecycle endpoints consistently enforce `expected_version`.
- Decide whether request throttling should stay fail-open; if not, tighten the `CharacterRateLimiter` contract and add endpoint-level deny/unavailable tests.
- If chat/message caps are intended to be strict, move enforcement to an atomic DB-side mechanism or a transactional reservation flow rather than read-then-write checks.
- Align image-file imports with the same normalization policy as base64 imports, or document the transport-dependent storage behavior explicitly.
- Make PNG export respect the same metadata ceiling that import enforces if round-trip safety is a goal.
- Fix hybrid exemplar fallback totals and the world-book response field mismatch, then lock both with targeted contract tests.

## Exit Note
- Stage 5 review completed against the requested chat/session, rate-limit, and streaming persistence surfaces.
- The requested mixed-suite validation produced order-dependent `503` failures in the SSE slice, but the isolated SSE rerun passed cleanly, so this stage records an availability risk rather than a stable endpoint regression.
- This report is the final ranked synthesis for the backend Characters review. Stage-specific deep dives remain in the Stage 2, Stage 3, and Stage 4 reports linked above.
