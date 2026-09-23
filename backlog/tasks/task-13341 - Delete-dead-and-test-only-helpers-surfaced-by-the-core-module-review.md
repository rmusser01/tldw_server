---
id: TASK-13341
title: Delete dead and test-only helpers surfaced by the core-module review
status: Done
assignee: []
created_date: '2026-09-22 05:00'
updated_date: '2026-09-23 20:07'
labels:
  - dead-code
  - cleanup
dependencies: []
references:
  - 'tldw_Server_API/app/core/Utils/Utils.py:383'
  - 'tldw_Server_API/app/core/LLM_Calls/streaming.py:280'
  - 'tldw_Server_API/app/core/DB_Management/PromptStudioDatabase.py:2420'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Zero-risk deletions, each verified by symbol grep across app/ and tests/. Deletion is the cheapest win available and is deliberately separated from consolidation.

ZERO REFERENCES:
- core/Utils/Utils.py:truncate_content (383). NOTE it is a CHARACTER-count truncator, so it is NOT the canonical for the token-budget pair - do not conflate.
- core/Utils/Utils.py:generate_unique_identifier (601). The review prompt cited generate_unique_id, which does not exist.
- core/Utils/Utils.py:is_valid_url (625).
- LLM_Calls/streaming.py:aiter_normalized_sse (280-314) - and it is the ONLY helper pairing SSE normalisation with egress policy and retries.
- RAG/rag_service/batch_utils.py:run_batch_indexed (190-277), already drifted (omits the fail_fast abort log its twin has).
- TTS/tts_validation.py:ProviderLimits.get_max_text_length (219).
- TTS/tts_config.py:ProviderConfig.max_retries (66) - DOCUMENTED TO OPERATORS at TTS-DEPLOYMENT.md:165 and read by nothing.
- TTS/adapters/base.py:convert_audio_format source_format param - in the signature and docstring, never in the body, 11 callers pass it.

UNREACHABLE:
- PromptStudioDatabase.py:list_optimization_iterations (2420-2470), 51 lines shadowed by the redefinition at :2472. Both carry # noqa: F811, a rule on NEITHER the global ignore list NOR the per-file block.
- Sync/v2/service.py:resolve_conflict (6925-6935) plus two dead parameters - an unreachable SECURITY check, which reads as coverage.

TEST-ONLY (the test keeps it green while nothing ships it):
- core/Utils/Utils.py:save_temp_file (826), sole consumer tests/Utils/test_utils_general.py:147.
- LLM_Calls/streaming.py:aiter_sse_lines_httpx (120-156).

STALE DOC: core/Chat/REFACTORING_PLAN.md points contributors at the wrong module and a deleted test file, and states Current Status (May 2025) against a 2026 codebase.

NOT INCLUDED: core/Sync/Sync_Client.py (1,112 LOC, zero production importers) - still named in two design docs, so deleting it is a product decision, not a cleanup.

Source: synthesis F40 / section 5
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Listed symbols deleted and the suite still passes
- [x] #2 save_temp_file and its test removed together, or the helper given a real consumer
- [x] #3 REFACTORING_PLAN.md deleted or folded into the module README
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
PARTIAL. Deleted the four helpers with genuinely ZERO references (72 lines): Utils.py truncate_content, generate_unique_identifier, is_valid_url; LLM_Calls/streaming.py aiter_normalized_sse. Both modules still import; re-verified references immediately before deleting.

CONFIRMED but NOT deleted - each is TEST-ONLY, so removing the helper means removing its test, which is a product call about whether the capability is wanted rather than a mechanical cleanup:
- Utils.py save_temp_file (1 ref: tests/Utils/test_utils_general.py:147)
- LLM_Calls/streaming.py aiter_sse_lines_httpx (4 refs, all tests/LLM_Calls/test_llm_streaming_and_security.py)
- RAG batch_utils.run_batch_indexed (11 refs, all tests/RAG/test_batch_utils.py)
Note is_valid_url was safe to delete: its 2 apparent references are a NESTED local function in Web_Scraping/Article_Extractor_Lib.py:1243, not the Utils one.

STILL OPEN: the unreachable PromptStudioDatabase.list_optimization_iterations duplicate, the dead Sync resolve_conflict branch, the TTS dead knobs, and Chat/REFACTORING_PLAN.md.

2026-09-23 reconciliation: no ACs met. AC1 NOT met. Done: Utils.truncate_content, generate_unique_identifier and is_valid_url, and streaming.aiter_normalized_sse (0 references remain; the modules import). The duplicate PromptStudioDatabase.list_optimization_iterations is also gone, done under TASK-13318 (dee169a794; one def remains at :1937). Still present: RAG batch_utils.run_batch_indexed (:190), TTS tts_validation ProviderLimits.get_max_text_length (:220), TTS tts_config ProviderConfig.max_retries (:66), TTS adapters/base.py convert_audio_format source_format param (still unused in the body), Sync/v2/service.py resolve_conflict unreachable personal-context require_active_exchange branch (~:7036-7046, still shadowed by the unconditional raise above it) plus the dead require_personal_context_conflict and personal_context_exchange params, and streaming.aiter_sse_lines_httpx (:120). The suite has not been re-run for the finished set. AC2 NOT met - Utils.save_temp_file (:789) and its test are both still there. AC3 NOT met - core/Chat/REFACTORING_PLAN.md still exists.

2026-09-23 (c8a570b21e): deleted TTS convert_audio_format source_format (13 kwarg call sites + 3 test fakes), ProviderConfig.max_retries (+ tts_providers_config.yaml and TTS-DEPLOYMENT.md lines; pydantic extra=ignore keeps old configs loading), ProviderLimits.get_max_text_length (kokoro test reads get_limits directly); Sync resolve_conflict unreachable require_active_exchange branch + require_personal_context_conflict / personal_context_exchange / _verified_personal_context_exchange params (sole internal caller updated, no external callers); Utils.save_temp_file + temp_files + cleanup_temp_files with its test (AC2: removed together); RAG run_batch_indexed + TestRunBatchIndexed; core/Chat/REFACTORING_PLAN.md (AC3: all phases done, README module map already covers the inventory; README pointer updated). AC1 AMENDMENT: streaming.aiter_sse_lines_httpx deliberately KEPT - Docs/Development/LLM_Adapters_Authoring_Guide.md directs new async adapters to it and its tests pin secret-bounded error frames; deleting it would send authors to hand-rolled loops. Verification: TTS, TTS_NEW, Utils, RAG batch_utils, Sync, Chat/unit on HEAD vs change: 102 failed both; -7 passed = the 7 deleted tests; the one differing failure each way is a load flake (tts history perf sanity passes 3/3 alone). Bandit -ll on touched modules: no findings. Found in passing, separate fix: TTS/vendors/kittentts_compat.py closures reference the except-bound exc after the block (F821).
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Deleted the confirmed dead and test-only code (TTS knobs and params, unreachable Sync check, Utils temp-file helpers, run_batch_indexed, the finished Chat refactoring plan); kept aiter_sse_lines_httpx as the documented adapter helper.
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
