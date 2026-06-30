---
id: TASK-233.6
title: Implement OpenWebUI webui.db adapter and SQLite filename validation
status: Done
assignee: []
created_date: '2026-05-10 22:04'
updated_date: '2026-05-10 22:12'
labels:
  - chatbooks
  - openwebui
  - backend
dependencies: []
documentation:
  - >-
    Docs/superpowers/plans/2026-05-10-openwebui-db-chat-import-implementation-plan.md
  - Docs/superpowers/specs/2026-05-10-openwebui-db-chat-import-design.md
parent_task_id: TASK-233
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Stage 1 of the OpenWebUI webui.db import plan: add a safe read-only SQLite adapter for uploaded OpenWebUI databases and extend Chatbook filename validation for .db/.sqlite uploads. This task is intentionally limited to adapter/validator behavior and focused tests; API/service/frontend wiring remains later stages.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Adapter validates SQLite magic bytes, required OpenWebUI tables and required columns before extracting preview data.
- [x] #2 Adapter previews multiple OpenWebUI users with per-user chat, message, folder, branch, duplicate, archived/pinned, attachment-reference and warning counts without returning raw chat/message content.
- [x] #3 Adapter extracts only a selected OpenWebUI user into normalized OpenWebUI conversation/message plans and preserves folder/metadata plans for later service import.
- [x] #4 Adapter treats chat.folder_id as authoritative and reports folder.items drift, missing parents or cycles as warnings.
- [x] #5 ChatbookValidator accepts safe .db/.sqlite filenames for DB imports while rejecting traversal, unsupported extensions and double-extension bypasses.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented Stage 1 adapter and validator changes. Red verification: focused pytest collection failed on missing tldw_Server_API.app.core.Chatbooks.import_adapters.openwebui_db module before production code. Worktree does not have its own .venv, so subsequent verification used /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv.

Verification:
- python -m pytest tldw_Server_API/tests/Chatbooks/test_openwebui_db_import_adapter.py tldw_Server_API/tests/Chatbooks/test_chatbook_security.py -q -> 20 passed, 5 warnings
- python -m bandit -r tldw_Server_API/app/core/Chatbooks/import_adapters/openwebui_db.py tldw_Server_API/app/core/Chatbooks/chatbook_validators.py -f json -o /private/tmp/bandit_openwebui_db_adapter.json -> 0 findings
- git diff --check -> clean

Scope note: service/API/frontend/user docs wiring remains deferred to later implementation stages by plan; no blocker for this adapter-only slice.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added the OpenWebUI webui.db Stage 1 adapter and SQLite filename validation. The adapter validates SQLite magic bytes and required schema, opens the database read-only, previews users without raw chat/message content, extracts only the selected source user, preserves DB/folder metadata for later import, and warns on folder.items drift, missing parents, and cycles. ChatbookValidator now accepts safe .db/.sqlite import filenames while rejecting unsupported and double-extension bypasses.
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
