# Cycle4 UAT107 / UAT108 read-only diagnosis

Repository: `/Users/macbook-dev/Documents/GitHub/tldw_server2`.
Scope: source reads, captured public synthetic request/response, private offline probes, official task notes only. No application/test-source edits, browser actions, API/model calls, runtime changes, or commits.

## UAT107 / TASK13260.48: stale model inventory after setup save

**Confirmed cause: Chat validates the configured custom model against an import-time configuration parser. The picker alias is correct.**

- The captured ordinary request selects `api_provider: custom-openai-api` and the exact Gemma model advertised by the configured provider. The response is 400 `model_not_available` for that same pair.
- `provider_readiness.py:17-27,65-77` deliberately maps catalog `custom_openai_api` to Chat `custom-openai-api`; `chat_service.py:608-640` has its configured model field. Changing this alias would be the wrong repair.
- `chat_service.py:569` captures `_config = load_comprehensive_config()` on import. `_configured_models_for_provider_cached` (957-974) reads that captured parser. `known_models_for_provider_cached` (978-998) adds pricing models, and the explicit availability validator uses that inventory (1001 onward; endpoint `chat.py:2779-2798,4470`).
- The setup save writes its own file parser (`Setup/setup_manager.py:572-635`), then clears central configuration loaders (`endpoints/setup.py:895-905,1467-1486`; `core/config.py:5935-5953`). This does not replace Chat's captured parser or clear Chat inventory caches. Even `invalidate_model_alias_caches` (`chat_service.py:1685-1705`) only clears the inventory LRUs; rebuilding still reads the stale parser.
- `/llm/providers` loads configuration afresh (`llm_providers.py:1467`) and can advertise the new model. The successful setup test at 00:38:25.844Z called `/setup/first-run/first-chat`, a separate verification path; it does not establish ordinary `/chat/completions` passed.

### Offline confirmation

`uat107-offline-inventory-probe.py` executes exact AST-extracted production inventory/cache-clear functions without importing the application. It reads only model fields from the actual isolated pre-setup backup `config.txt.pre-setup-20260916003245.bak` and current configuration. The startup model is `gpt-4.1-2025-04-14`; the saved model is the exact Gemma path in the captured request.

All five assertions passed:

1. Central fresh loader sees the saved Gemma model.
2. Gemma remains rejected after the actual setup refresh function.
3. Gemma remains rejected even after additional Chat model-LRU clearing.
4. Replacing Chat's startup parser and clearing model LRUs accepts Gemma (restart analogue).
5. An unrelated model remains rejected after refresh.

The pricing catalog is an empty controlled fixture; the live 400 independently establishes that the actual inventory did not contain this model. This is not an in-process inspection of the running server. Parent may restart only the isolated API18300 once to continue the matrix, documenting the result as a workaround. A restart is not a product fix or no-restart acceptance.

### Minimal repair boundary after freeze

Use the currently loaded provider configuration for configured-model validation and invalidate/key/remove both dependent caches so neither retains pre-save fields. Preserve canonical aliases, strict explicit-model rejection, provider availability/admin policy, and credential routing. Do not merely add a spelling alias or clear only the existing LRUs. Prefer the existing fresh configuration/runtime resolver patterns over another global configuration abstraction.

Likely source: `core/Chat/chat_service.py`; only add a setup invalidation integration seam if the selected cache design requires it. Existing fixtures: `tests/Chat/unit/test_chat_service_normalization.py`, `tests/Setup/test_setup_provider_validation.py`, and `tests/Chat/integration/test_chat_endpoint_simplified.py` / `test_chat_provider_override_store_boundary.py`.

Meaningful regression: pre-import/cache old custom model, save a new model through the real setup save boundary, then verify fresh catalog and ordinary Chat strict validation agree without restarting. Cover warm and initially cold caches, repeated saves, numbered custom slots, canonical aliases, unrelated/disabled models, and environment precedence. Mock provider inference, not model validation. Native follow-up must include first and second ordinary turns plus reload on the unchanged process.

## UAT108 / TASK13260.49: display error serialized as an assistant answer

**Confirmed request construction defect; the UAT107 request was rejected before inference, so no claim that the model consumed this specific payload.**

- Captured request has roles `user, assistant, user`; the two user texts are equal, and assistant content is a valid `__tldw_error__:` envelope with summary/hint/detail.
- `PlaygroundForm.tsx:3038-3068` Retry selects the last user text and invokes ordinary `sendMessage` without excluding/reusing the failed turn.
- `chat-helper/index.ts:153-168` deliberately preserves the failed user and error bubble in local history. That display/persistence behavior is useful and should remain truthful.
- `normalChatMode.ts:525` passes model history through `utils/generate-history.ts`. That function skips image-generation rows but serializes every assistant row as `AIMessage`, including valid error envelopes.
- Backend `chat_service.py:3948-3973` recognizes diagnostic envelopes, but `4248` applies it only to DB-loaded history. The subsequent request-message loop `4252` does not filter them. Existing backend retry tests submit one new user message, so they do not establish safety for the actual browser's user/error/user request.

### Controlled serializer reproduction

`uat108-history-probe.config.ts` appends a private in-memory test to the existing history serializer suite; repository source is untouched. It feeds the captured original pair to the real serializer and real decoder.

`uat108-history-probe-red.log`: **1 expected failure / 3 passes**. The valid displayed failure becomes an AI message (expected model history length1, actual2). Controls preserve normal image filtering, a user quotation plus malformed assistant marker, and legitimate partial assistant text returned by `buildAssistantErrorContent`.

An initial invocation pointed to a nonexistent top-level Vitest module. Its startup-only failure is retained separately as `uat108-history-probe-startup-error.log`; it is not a product RED. The completed probe uses the installed Bun-store Vitest entrypoint.

### Minimal repair boundary after freeze

Filter only recognized valid assistant display envelopes at model-history projection, using the existing decoder. Do not strip arbitrary strings, user quotations, malformed examples, partial answers, successful text, or the visible failure history. Exercise the actual Retry action through pipeline request capture: merely filtering the assistant leaves duplicate user input, so the intended failed-turn reuse must be explicit rather than indiscriminately deduplicating equal user text. Preserve a user intentionally submitting the same question twice. Review the existing regeneration/submit-history override seam rather than creating a separate Chat pipeline.

Potential source paths: `utils/generate-history.ts`, actual Retry handler in `PlaygroundForm.tsx`, and only the existing submit-history seam if needed. Existing tests: `utils/__tests__/generate-history.image-generation.test.ts`, real Form integration harness (the banner test alone only mocks onRetry), `hooks/chat-modes/__tests__/chatModePipeline.conversation-id.test.ts`, and normal Chat request/persistence integration harness. Backend defense, if included, can reuse `_is_saved_chat_error_envelope` for request projection and extend `tests/Chat/unit/test_chat_history_and_streaming.py` plus `integration/test_persona_backed_chat_conversations.py`; do not change legitimate request roles or stored diagnostic visibility.

Required cases: empty failed answer → Retry another model; failure → later distinct turn; partial streamed answer then failure; valid/malformed/user-quoted envelope controls; intentional repeated successful user turn; saved/temporary flow and account/history replacement during async Retry. Record actual request and server/local row identities so this repair does not compound UAT103 user acknowledgment duplication.

## Reproduction commands

```sh
source .venv/bin/activate && python /private/tmp/uat107-offline-inventory-probe.py
node apps/node_modules/.bun/vitest@4.0.18+08ee8852a9d25cb0/node_modules/vitest/vitest.mjs run --config /private/tmp/uat108-history-probe.config.ts
```

No product fix, live recovery, or completed UAT acceptance is claimed. UAT106 initial wrong default selection is separate from both findings.
