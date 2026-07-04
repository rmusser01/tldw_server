---
id: TASK-12154
title: Stabilize media process-pdfs auto chunking route test
status: Done
created_date: 2026-07-04 19:54
labels:
- tests
- media
priority: Medium
modified_files:
- tldw_Server_API/tests/LLM_Adapters/conftest.py
- tldw_Server_API/tests/Media/test_auto_chunking_process_endpoints.py
updated_date: 2026-07-04 21:59
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The LLM-to-Notifications slice now reaches Media and stops in test_process_pdfs_auto_chunking_adds_plan_metadata with a 404 for /api/v1/media/process-pdfs. The endpoint module exists, so investigate route registration/import conditions and make the test deterministic without losing auto-chunking coverage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The focused process-pdfs auto chunking test no longer receives 404 in the current environment.
- [x] #2 Auto-chunking plan metadata assertions remain covered.
- [x] #3 The LLM-to-Notifications slice progresses past this media test.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Root cause: LLM_Adapters/conftest.py mutates ROUTES_DISABLE at module import, adding media/audio route disables for the entire pytest process. In broad slices this can leave the shared app without /api/v1/media/process-pdfs before Media tests run, while the focused Media test passes because that conftest is not imported.
Scoped the LLM adapter tests' ROUTES_DISABLE mutation to an autouse fixture and changed the auto-chunking process-pdfs route test to patch the boundary assistant instead of route globals. Focused verification: auto-chunking route test -> 1 passed; adapter+media reproducer -> 5 passed. Changed-scope slice passed later: 1838 passed, 54 skipped, 1 xfailed, 2 xpassed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Prevented adapter-test route configuration from leaking into media route tests and kept the process-pdfs fallback behavior under direct assistant-level test control.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Focused media auto chunking test output captured.
- [x] #2 Slice verification output captured.
- [x] #3 Task updated with final summary.
<!-- DOD:END -->
