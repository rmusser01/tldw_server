---
id: TASK-12892
title: Isolate HuggingFace handler unit tests from optional transformers runtime
status: Done
created_date: 2026-07-04 19:42
labels:
- tests
- local-llm
priority: Medium
modified_files:
- tldw_Server_API/tests/Local_LLM/test_huggingface_handler.py
updated_date: 2026-07-04 21:58
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The LLM-to-Notifications slice stops in a HuggingFace handler unit test because the test imports real transformers dependencies and the local optional dependency chain fails while importing safetensors.torch.storage_ptr. The behavior under test can be exercised with dependency stubs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 HuggingFace handler unit tests do not import real transformers/torch optional runtime.
- [x] #2 The focused HuggingFace handler tests pass in the current environment.
- [x] #3 The LLM-to-Notifications slice progresses past the HuggingFace handler tests.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Remove module-level runtime probes/importorskip for real transformers and torch from this unit test file.
2. Stub HuggingFaceHandler._ensure_hf_dependencies in tests so path traversal and cache-key behavior are tested without optional HF imports.
3. Run the focused HuggingFace handler tests and the LLM-to-Notifications slice.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Removed module-level optional transformers/torch import probes from the HuggingFace handler unit tests and stubbed only the dependencies needed by each test. Focused verification: `python -m pytest -q tldw_Server_API/tests/Local_LLM/test_huggingface_handler.py` -> 2 passed, 10 warnings. Changed-scope slice passed later: 1838 passed, 54 skipped, 1 xfailed, 2 xpassed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Isolated HuggingFace handler unit tests from optional runtime packages so traversal and cache-key behavior can be verified without requiring transformers/torch imports.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Focused HuggingFace handler test output captured.
- [x] #2 Slice verification output captured.
- [x] #3 Task updated with final summary.
<!-- DOD:END -->
