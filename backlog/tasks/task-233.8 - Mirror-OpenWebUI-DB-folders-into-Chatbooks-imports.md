---
id: TASK-233.8
title: Mirror OpenWebUI DB folders into Chatbooks imports
status: Done
assignee: []
created_date: '2026-05-10 23:00'
updated_date: '2026-05-10 23:11'
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
Implement Stage 3 folder mirroring for selected-user OpenWebUI database chat imports. Reuse tldw keyword collections/conversation keyword links to create visible folders under an OpenWebUI/user namespace. Jobs, frontend and user docs remain later stages.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Folder mirroring creates or reuses an OpenWebUI/user namespace and source folder path using existing keyword collection APIs.
- [x] #2 Imported DB conversations are linked to mirrored visible folders idempotently without duplicating collection-keyword or conversation-keyword links.
- [x] #3 Global keyword collection name collisions are handled deterministically with stable source-aware disambiguation and warnings instead of merging unrelated folders.
- [x] #4 Invalid or empty OpenWebUI path segments are sanitized while original source names remain preserved in metadata/settings or warnings.
- [x] #5 OpenWebUI DB import results report mirrored folder and folder link counts while preserving existing JSON import behavior.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented Stage 3 folder mirroring in tldw_Server_API/app/core/Chatbooks/openwebui_folders.py and wired selected-user DB imports to mirror OpenWebUI / <user> / <source folder path> using keyword collections and conversation keyword links. Added deterministic global collection-name disambiguation, path-segment sanitization warnings, folder path metadata preservation in conversation settings, and mirrored_folders/folder_links result counts.

Verification: focused Stage 3 pytest passed 16 tests; overlapping Chatbooks regression pytest passed 59 tests; Bandit over openwebui_folders.py, chatbook_service.py, and chatbook_schemas.py reported 0 findings; git diff --check passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed Stage 3 folder mirroring for OpenWebUI database imports. DB imports now create/reuse visible OpenWebUI/user/source-folder keyword collection paths, link imported conversations idempotently through shared keywords, preserve folder path/source metadata, warn on sanitization or deterministic disambiguation, and report mirrored folder/link counts without changing JSON import behavior.
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
