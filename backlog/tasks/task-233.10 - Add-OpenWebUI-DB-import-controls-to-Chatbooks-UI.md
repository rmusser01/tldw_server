---
id: TASK-233.10
title: Add OpenWebUI DB import controls to Chatbooks UI
status: Done
assignee: []
created_date: '2026-05-10 23:18'
updated_date: '2026-05-10 23:34'
labels:
  - chatbooks
  - openwebui
  - frontend
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
Implement Stage 5 frontend support for OpenWebUI database imports in the Chatbooks import UI and API client. Users should be able to select the DB source format, preview detected users, choose one source user, see destination/folder/attachment caveats, and submit selected_openwebui_user_id with import requests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 API client upload paths support source_format=openwebui_db and selected_openwebui_user_id multipart fields.
- [x] #2 Chatbooks playground import UI exposes OpenWebUI database mode alongside existing chatbook and JSON import modes.
- [x] #3 Preview results render detected OpenWebUI DB users and require exactly one selected user before import.
- [x] #4 UI communicates destination namespace plus metadata-only attachment/file hydration caveat without adding unsupported binary import controls.
- [x] #5 Existing chatbook and OpenWebUI JSON import UI behavior remains unchanged.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
RED: local package Vitest initially failed before the UI exposed OpenWebUI database mode; this confirmed the new DB-mode UI test captured the missing behavior.

GREEN: cd apps/packages/ui && ./node_modules/.bin/vitest run src/services/__tests__/tldw-api-client.chatbooks-openwebui.test.ts src/components/Option/Chatbooks/__tests__/ChatbooksPlaygroundPage.openwebui-import.test.tsx -> 2 files passed, 7 tests passed. git diff --check clean.

Bandit not applicable for this frontend-only Stage 5 slice; no Python code was touched in TASK-233.10.

Re-ran the same focused Vitest command after adding explicit DB-mode assertions for hidden media/embedding controls; result remained 2 files passed, 7 tests passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added OpenWebUI database import controls to the Chatbooks playground and client: source_format=openwebui_db upload support, selected_openwebui_user_id serialization, DB preview user rendering, selected-user gating, destination namespace copy, and metadata-only file caveat. Reused the existing OpenWebUI import test file for JSON and DB coverage to avoid duplicate setup while preserving stale-preview coverage.
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
