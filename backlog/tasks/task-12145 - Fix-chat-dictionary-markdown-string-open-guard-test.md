---
id: TASK-12145
title: Fix chat dictionary markdown string open guard test
status: Done
created_date: 2026-07-04 07:30
labels:
- tests
- chat-dictionary
- stability
priority: High
modified_files:
- tldw_Server_API/tests/Character_Chat_NEW/unit/test_chat_dictionary_unit.py
updated_date: 2026-07-04 17:24
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The broad AuthNZ/chat pytest slice fails at teardown for `TestImportExport.test_import_from_markdown_string_does_not_open_file` because the test patches global `builtins.open`. The service call itself treats string markdown as content, but unrelated teardown/config cleanup can call `open` before pytest restores the monkeypatch, making the guard too broad and order-sensitive.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The test still verifies string markdown content does not call the chat dictionary module's file-open path.
- [x] #2 The focused chat dictionary unit test file passes.
- [x] #3 The broad AuthNZ/chat fail-fast slice gets past this failure or its next result is recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Narrow the open guard to the `chat_dictionary` module lookup used by `import_from_markdown` instead of patching `builtins.open` globally.
2. Run the focused test and full chat dictionary unit file.
3. Run Bandit on the touched test file and `git diff --check`.
4. Rerun the broader fail-fast slice enough to confirm the previous teardown blocker is cleared or record the next blocker.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Root cause: `test_import_from_markdown_string_does_not_open_file` patched `builtins.open` globally. The service call itself treats string markdown as content, but in the broad suite unrelated pytest/app teardown can call `open` before the monkeypatch fixture restores it, so the broad run failed at teardown with `AssertionError("open should not be called for string content")`.

Fix: patched `tldw_Server_API.app.core.Character_Chat.chat_dictionary.open` with `raising=False` instead of patching `builtins.open`. That keeps the guard on the exact module lookup used by `ChatDictionaryService.import_from_markdown` for Path inputs and avoids interfering with unrelated teardown file access. Also annotated the existing deterministic `random.Random(0)` probability-test RNG with `# nosec B311` so touched-file Bandit is clean.

Verification:
- Prior RED: broad AuthNZ/chat slice stopped at `tldw_Server_API/tests/Character_Chat_NEW/unit/test_chat_dictionary_unit.py::TestImportExport::test_import_from_markdown_string_does_not_open_file` teardown after `1808 passed, 36 skipped`.
- Focused guard: `/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q --tb=short tldw_Server_API/tests/Character_Chat_NEW/unit/test_chat_dictionary_unit.py::TestImportExport::test_import_from_markdown_string_does_not_open_file` passed, `1 passed, 8 warnings`.
- Full chat dictionary unit file: `/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q --tb=short tldw_Server_API/tests/Character_Chat_NEW/unit/test_chat_dictionary_unit.py` passed, `39 passed, 86 warnings`.
- Security: `/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m bandit -r tldw_Server_API/tests/Character_Chat_NEW/unit/test_chat_dictionary_unit.py -s B101 -f json -o /tmp/bandit_task_12145.json` produced `errors: []`, `results: []`.
- Whitespace: `git diff --check` passed.
- Broad confirmation: `TLDW_TEST_NO_DOCKER=1 /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q -x --tb=short tldw_Server_API/tests/AuthNZ_Federation tldw_Server_API/tests/AuthNZ_Postgres tldw_Server_API/tests/AuthNZ_SQLite tldw_Server_API/tests/AuthNZ_Unit tldw_Server_API/tests/Billing tldw_Server_API/tests/CI tldw_Server_API/tests/ChaChaNotesDB tldw_Server_API/tests/Character_Chat tldw_Server_API/tests/Character_Chat_NEW tldw_Server_API/tests/Characters tldw_Server_API/tests/Chat` passed the previous chat dictionary failure and later stopped at `tldw_Server_API/tests/Chat/integration/test_chat_integration.py::test_chat_completion_integration` because the sandbox denied binding a local mock HTTP server, after `2443 passed, 55 skipped`.
- Environment check: reran the focused Chat integration test outside the sandbox and it passed, `1 passed, 61 warnings`, confirming the new broad stop is sandbox-related rather than a code failure.
Post-commit verification update: reran the full broad AuthNZ/chat slice outside the sandbox so local mock HTTP server fixtures could bind to `127.0.0.1`. Command: `TLDW_TEST_NO_DOCKER=1 /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q -x --tb=short tldw_Server_API/tests/AuthNZ_Federation tldw_Server_API/tests/AuthNZ_Postgres tldw_Server_API/tests/AuthNZ_SQLite tldw_Server_API/tests/AuthNZ_Unit tldw_Server_API/tests/Billing tldw_Server_API/tests/CI tldw_Server_API/tests/ChaChaNotesDB tldw_Server_API/tests/Character_Chat tldw_Server_API/tests/Character_Chat_NEW tldw_Server_API/tests/Characters tldw_Server_API/tests/Chat`. Result: `3146 passed, 40 skipped, 35518 warnings in 1613.32s`.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Narrowed the markdown string test's file-open guard to the chat dictionary module so it still proves string content is not treated as a path without replacing global `builtins.open` during teardown. Added a test-only Bandit suppression for the existing deterministic RNG used by a probability test.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Root cause recorded in task notes.
- [x] #2 Verification commands and outcomes recorded.
- [x] #3 Bandit result recorded for touched Python test scope.
- [x] #4 Final summary explains what changed and why.
<!-- DOD:END -->
