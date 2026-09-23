---
id: TASK-2425
title: Harden LLM_Calls review findings
status: Done
assignee: []
created_date: '2026-06-23 14:41'
updated_date: '2026-09-23 15:28'
labels:
  - security
  - llm
dependencies: []
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

Final Summary:
--------------------------------------------------
Rebased the PR branch onto latest `origin/dev` and addressed the validated PR review comments. The sync-to-async streaming bridge no longer consumes default-executor workers for chunk delivery and now logs iterator close failures at debug level. Superseded MLX loads now emit a distinct `superseded` metric status instead of being counted as successful applied loads. GGUF filename coverage was split into focused invalid and valid scenarios, with new regression tests for stream delivery and MLX metrics.

SUPERSEDED IN PART, 2026-09-23, by TASK-13288.

This task's conclusion -- that the summarization path-read finding was 'not currently
active' -- was correct as an observation and unsafe as a resting state. It was true only
because Summarization_General_Lib defined extract_text_from_input twice at module scope
and the second, non-file-reading definition won by definition order, with the first
carrying os.path.isfile(input_data) -> open(input_data).read() behind an inline F811
suppression. So an arbitrary file read reachable from analyze() was held inactive by
nothing but which 'def' came last.

Anyone deleting the *second* definition as a duplicate -- the obvious tidy-up, and what
the F811 warning invites -- would have silently re-armed it.

TASK-13288 removed the shadowed definition, so the behaviour that already ran is now the
only behaviour. It also found and deleted a second, entirely separate copy of the same
hazard in the same module: extract_metadata_and_content did os.path.exists(input_data)
-> open(input_data) -> json.load(file) and had no callers anywhere in the repo, so this
task's review did not reach it.

The property is now pinned by tests that fail if either read returns, rather than
inferred from definition order: tests/LLM_Calls/test_summarization_input_extraction.py,
including an AST ratchet over the whole module.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Rebased the PR branch onto latest `origin/dev` and addressed the validated PR review comments. The sync-to-async streaming bridge no longer consumes default-executor workers for chunk delivery and now logs iterator close failures at debug level. Superseded MLX loads now emit a distinct `superseded` metric status instead of being counted as successful applied loads. GGUF filename coverage was split into focused invalid and valid scenarios, with new regression tests for stream delivery and MLX metrics.
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
