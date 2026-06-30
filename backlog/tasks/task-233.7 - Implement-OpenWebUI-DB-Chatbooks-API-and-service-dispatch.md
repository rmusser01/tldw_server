---
id: TASK-233.7
title: Implement OpenWebUI DB Chatbooks API and service dispatch
status: Done
assignee: []
created_date: '2026-05-10 22:14'
updated_date: '2026-05-10 23:00'
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
Implement Stage 2 of the OpenWebUI webui.db import plan: add source_format=openwebui_db and selected_openwebui_user_id through schemas, Chatbooks preview/import endpoints, and ChatbookService dispatch. Frontend, Jobs, folder mirroring and user-facing docs remain later stages.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Schemas expose openwebui_db source format, selected_openwebui_user_id, DB preview payloads and DB import result payloads.
- [x] #2 Preview endpoint accepts safe .db/.sqlite uploads for openwebui_db, skips archive validation, returns openwebui_db_preview, and cleans up temp preview files.
- [x] #3 Import endpoint rejects openwebui_db without selected_openwebui_user_id before service import and rejects wrong extensions for each source format.
- [x] #4 ChatbookService dispatch accepts openwebui_db, previews DB uploads, extracts only the selected OpenWebUI user, and returns openwebui_db_result with selected user metadata.
- [x] #5 Existing chatbook and openwebui_json preview/import behavior remains covered and unchanged.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented Stage 2 Chatbooks schema, endpoint and service dispatch for source_format=openwebui_db. Red verification: pytest failed with unknown enum openwebui_db and missing preview_openwebui_db/import_openwebui_db service methods before production changes.

Verification:
- python -m pytest tldw_Server_API/tests/Chatbooks/test_chatbooks_openwebui_db_api.py tldw_Server_API/tests/Chatbooks/test_openwebui_import_service.py -q -> 17 passed, 5 warnings
- python -m pytest tldw_Server_API/tests/Chatbooks/test_chatbooks_api_error_and_preview_mapping.py tldw_Server_API/tests/Chatbooks/test_chatbooks_openwebui_db_api.py tldw_Server_API/tests/Chatbooks/test_openwebui_import_service.py tldw_Server_API/tests/Chatbooks/test_openwebui_db_import_adapter.py tldw_Server_API/tests/Chatbooks/test_chatbook_security.py -q -> 55 passed, 5 warnings
- python -m bandit -r tldw_Server_API/app/api/v1/endpoints/chatbooks.py tldw_Server_API/app/api/v1/schemas/chatbook_schemas.py tldw_Server_API/app/core/Chatbooks/chatbook_service.py tldw_Server_API/app/core/Chatbooks/import_adapters/openwebui_db.py tldw_Server_API/app/core/Chatbooks/chatbook_validators.py -f json -o /private/tmp/bandit_openwebui_db_api_service.json -> 0 findings
- git diff --check -> clean

Scope note: async Jobs worker dispatch, visible folder mirroring, frontend controls and user-facing docs remain deferred to later plan stages.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Wired OpenWebUI database imports through Chatbooks schemas, preview/import endpoints and ChatbookService dispatch. The API now accepts source_format=openwebui_db, validates .db/.sqlite filenames, returns openwebui_db_preview, requires selected_openwebui_user_id for DB imports before service work, and returns openwebui_db_result for sync imports. The service now previews DB uploads and imports the selected OpenWebUI user through the Stage 1 adapter while preserving existing JSON/chatbook behavior.
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
