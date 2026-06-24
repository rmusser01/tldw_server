---
id: TASK-2425
title: Harden LLM_Calls review findings
status: Done
assignee: []
created_date: '2026-06-23 14:41'
updated_date: '2026-06-23 14:58'
labels:
  - security
  - llm
dependencies: []
references: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verify and address validated findings from the current-code review of `tldw_Server_API/app/core/LLM_Calls`. Scope includes safe upstream 400 logging, sync-to-async streaming lifecycle and backpressure, MLX load race handling, Hugging Face GGUF destination validation, and documenting the summarization path-read finding as not currently active after verification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Validated findings are backed by focused failing tests before production edits.
- [x] #2 Upstream 400 logging avoids writing prompt, message, request body, or secret values to logs by default.
- [x] #3 Sync-to-async streaming bridge applies bounded backpressure and closes sync iterators on cancellation.
- [x] #4 MLX load failure from an older overlapping load cannot restore over a newer successful load.
- [x] #5 Hugging Face GGUF downloads reject path traversal or path-component filenames.
- [x] #6 Targeted tests and Bandit pass for touched scope.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Backlog MCP tools were unavailable and the Backlog CLI failed on stale internal task filename references. The user approved manual task creation as the fallback. Verification result before edits: the summarization arbitrary file-read finding is not active through `analyze()` because the file-reading helper is shadowed by a later `extract_text_from_input()` definition, and the remaining file-reading metadata helper has no callers.

RED verification: focused regression tests failed before implementation for raw upstream 400-body logging, missing bounded stream bridge API, stale MLX session restore after an older failed load, and missing Hugging Face GGUF filename validation.

Implemented safe upstream HTTP error metadata logging, bounded/cancellable `wrap_sync_stream`, OpenAI/Anthropic/Cohere async-stream delegation to the shared bridge, an MLX load generation guard, and GGUF filename validation before destination path construction.

GREEN verification: focused four-test regression run passed with 4 passed and 18 warnings. Broader targeted suite passed with 62 passed and 134 warnings:
`tldw_Server_API/tests/LLM_Calls/test_llm_streaming_and_security.py`,
`tldw_Server_API/tests/LLM_Calls/test_mlx_provider.py`,
`tldw_Server_API/tests/LLM_Calls/test_llm_providers.py::TestHuggingFaceAPI`, and
`tldw_Server_API/tests/LLM_Adapters/unit/test_adapter_stream_error_normalization.py`.

Bandit verification on touched production files exited 0 with zero findings. `git diff --check` on touched files exited 0.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Hardened the validated `LLM_Calls` review findings. Provider 400 logging now records safe metadata instead of raw request/error bodies, sync-to-async streaming has bounded backpressure and cancellation cleanup, duplicated adapter stream bridges delegate to the shared helper, overlapping MLX load failures cannot restore stale sessions over newer successful loads, and Hugging Face GGUF download filenames are constrained to local `.gguf` filenames without path components.
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
