---
id: TASK-12891
title: Stabilize Llama.cpp acquisition worker DNS safety test
status: Done
created_date: 2026-07-04 19:38
labels:
- tests
- llm-local
priority: Medium
modified_files:
- tldw_Server_API/tests/LLM_Local/test_llamacpp_acquisition_jobs_worker.py
updated_date: 2026-07-04 21:57
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The LLM-to-Notifications slice stops in the Llama.cpp acquisition worker happy-path test because the production URL safety validator resolves example.com before the mocked download stream runs. In sandboxed or offline test environments DNS resolution can fail, making the test depend on external network state.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The focused worker happy-path test no longer depends on real DNS resolution.
- [x] #2 The LLM-to-Notifications slice progresses past the Llama.cpp acquisition worker test.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a local getaddrinfo stub for example.com in the worker happy-path test, matching neighboring acquisition service tests.
2. Re-run the focused worker test.
3. Re-run the LLM-to-Notifications slice.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented local DNS stubbing for example.com in the Llama.cpp acquisition worker tests. Focused verification: `python -m pytest -q tldw_Server_API/tests/LLM_Local/test_llamacpp_acquisition_jobs_worker.py` -> 7 passed, 20 warnings. Changed-scope slice passed later: 1838 passed, 54 skipped, 1 xfailed, 2 xpassed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Stabilized the Llama.cpp acquisition worker tests by removing their dependency on real DNS while preserving the production URL safety path under test.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Focused worker test output captured.
- [x] #2 Slice verification output captured.
- [x] #3 Task updated with final summary.
<!-- DOD:END -->
