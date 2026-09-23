---
id: TASK-13340
title: Remove 614 lines of dead chat orchestration that still takes commits
status: Done
assignee: []
created_date: '2026-09-22 05:00'
updated_date: '2026-09-23 23:42'
labels:
  - dead-code
  - chat
dependencies: []
references:
  - 'tldw_Server_API/app/core/Chat/chat_orchestrator.py:869'
  - 'tldw_Server_API/app/core/Chat/chat_orchestrator.py:1379'
  - 'tldw_Server_API/app/core/Chat/Workflows.py:34'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
_chat_sync_impl (869-1190, 322 lines) and achat (1379-1670, 292 lines) are near-twins implementing the entire multimodal turn-assembly policy twice: slash-command parsing, chat-dictionary pre/post-gen replacement, image-history handling, RAG prefix construction, custom-prompt placement, empty-message placeholder.

REACHABILITY: the only importer of chat_orchestrator.chat is core/Chat/Workflows.py:34, whose only importer is its own test, and which loads ./App_Function_Libraries/Workflows/Workflows.json - a path that does not exist in this repo. achat has no non-test importer outside chat_orchestrator.py. So this is 614 lines of dead production code that still takes commits. (chat_api_call in the same file IS live with 5 core callers, so this is a partial-file deletion.)

ONE REAL DIVERGENCE the duplication hides: achat wraps rag_text_prefix construction in try/except _CHAT_ORCHESTRATOR_NONCRITICAL_EXCEPTIONS (which includes AttributeError and TypeError); _chat_sync_impl has the identical expression with NO try. Same input, two outcomes - the async path silently drops retrieval context and answers UNGROUNDED with no error and no log, while the sync path raises and returns a 500. Second divergence: _chat_sync_impl logs a masked API key under ALLOW_MASKED_KEY_LOG, achat has no such block.

Also blocks TASK-13336-adjacent work: the orchestrator CommandContext derives is_single_user_owner from the SERVER env rather than the caller, which is a latent auth footgun reachable only through this dead path.

Source: synthesis F39
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Dead orchestration paths deleted, chat_api_call retained
- [x] #2 Workflows.py resolved (deleted or repointed)
- [x] #3 The divergent try/except is not carried into whatever remains
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Commit 0c5d266050. Reachability re-verified by grep across app/ and tests/: chat()/achat()/_chat_sync_impl/_run_achat_sync/_run_coro_sync/_build_command_context had no production importer; the only importer of chat was core/Chat/Workflows.py, imported only by tests/Chat/unit/test_chat_workflows.py, and App_Function_Libraries/ does not exist. chat_api_call/chat_api_call_async kept (5+ live core callers). Deleted with the dead code: the sync-bridge ThreadPoolExecutor + atexit hook (only used by _run_coro_sync), _build_command_context (the server-env is_single_user_owner footgun), 11 imports that only the dead path used. chat_orchestrator.py 1660 -> 598 lines. Workflows.py deleted (AC2) plus its pyproject per-file-ignore. AC3: both copies of the rag_text_prefix expression are gone; nothing that remains builds a RAG prefix. Test-only users deleted with the code: tests/Chat/unit/test_chat_workflows.py, tests/Chat_NEW/unit/test_chat_sync_wrapper.py, tests/Chat_NEW/unit/test_chat_command_injection.py, and the executor-shutdown case in test_phase3_3_sanitizers.py. Live /chat/completions slash-command injection remains covered by Chat_NEW/integration test_chat_command_{perf,concurrency,replace_mode,audit}.py and test_chat_skill_commands_injection.py. Docs: Env_Vars.md drops CHAT_COMMANDS_ASYNC_ONLY and CHAT_SYNC_CORO_TIMEOUT_SECONDS (only read by the dead path). The core->api import ratchet (tests/lint/test_core_to_api_import_boundary.py) shrinks by chat_orchestrator.py because its test_baselines_only_shrink check requires it. Tests: tests/Chat + Chat_NEW + LLM_Adapters + LLM_Calls + lint, -n 8. Before: 19 failed / 3974 passed / 44 skipped. After: 19 failed / 3958 passed / 44 skipped (-16 = deleted tests). FAILED sets are the same apart from 2 entries. Chat_NEW/integration/test_moderation.py::test_input_block_fails_closed_when_audit_service_missing failed before and passed after under xdist, and fails in isolation on both (pre-existing: credential_store_unavailable 503). lint baseline_only_shrink failed after until the baseline entry was removed, and now passes (3/3). The other 18 failures are pre-existing on 4a84d02b55: persona_backed_chat_conversations x6, chat_image_recovery x2, chat_helpers x1, LLM_Calls strict_filter/ollama x9. Bandit -ll on chat_orchestrator.py: no findings. No new regression test: this is a pure deletion with no behaviour to pin.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Deleted the dead chat()/achat() multimodal orchestration (about 1060 lines incl. the sync bridge) and the unreachable core/Chat/Workflows.py. chat_api_call/chat_api_call_async stay. The RAG-prefix try/except divergence and the server-env is_single_user_owner CommandContext both went away with the deleted code. Tests that only exercised the dead code were deleted; the live command-injection path is still covered by the Chat_NEW integration tests.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
