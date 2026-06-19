---
id: TASK-2335
title: Fix Chatbook API preview TestClient hang
status: Done
labels:
- chatbooks
- tests
- bug
modified_files:
- tldw_Server_API/tests/Chatbooks/test_chatbooks_api_preview.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Investigate and fix the full-app TestClient lifecycle hang in tldw_Server_API/tests/Chatbooks/test_chatbooks_api_preview.py that blocks Chatbook v1.1 final endpoint verification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Root cause: `test_chatbooks_api_preview.py` imported the full `tldw_Server_API.app.main.app` and opened a full-lifespan `TestClient` per test. The fixture set `TEST_MODE` after importing the app, so the module-level app test-mode flag could not reliably suppress full startup/shutdown workers. After one preview test, later TestClient setup/shutdown could block in the Starlette/AnyIO portal while full-app background services were being cancelled. A one-off minimal FastAPI app including only the Chatbooks router reproduced the same endpoint behavior without the hang.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Changed `tldw_Server_API/tests/Chatbooks/test_chatbooks_api_preview.py` to build a small FastAPI app with the Chatbooks router, an isolated `CharactersRAGDB`, auth override, and audit stub instead of using the global full application. This preserves endpoint coverage for real Chatbook preview behavior while avoiding unrelated application lifespan workers.

Verification:
- Existing failing/hanging evidence before fix: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Chatbooks/test_chatbooks_api_preview.py -k 'not test_preview_manifest_version_ok' --timeout=60 -v` timed out after the first selected test while setting up the next full-app TestClient.
- Hypothesis check: a one-off minimal app using the Chatbooks router handled legacy, canonical, and invalid preview archives without hanging.
- `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Chatbooks/test_chatbooks_api_preview.py --timeout=60 -v` passed with 5 tests.
- Original endpoint verification command now passes: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Chatbooks/test_chatbooks_api_preview.py tldw_Server_API/tests/Chatbooks/test_chatbooks_api_error_and_preview_mapping.py -v` passed with 19 tests and 7 warnings.
- Focused v1.1 regression suite passed: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Chatbooks/test_chatbooks_manifest_contract.py tldw_Server_API/tests/Chatbooks/test_chatbooks_manifest_v1_1_contract.py tldw_Server_API/tests/Chatbooks/test_chatbooks_v1_1_file_inventory.py tldw_Server_API/tests/Chatbooks/test_chatbooks_v1_1_preview.py tldw_Server_API/tests/Chatbooks/test_explainer_session_content_type.py tldw_Server_API/tests/Chatbooks/test_chatbooks_import_validation.py -v` passed with 46 tests and 6 warnings.
- Task-scoped `git diff --check` passed for the touched test and Backlog files.
- Bandit skipped because this was a test-only fixture change and no production code changed.
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
