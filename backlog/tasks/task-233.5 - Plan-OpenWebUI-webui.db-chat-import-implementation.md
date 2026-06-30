---
id: TASK-233.5
title: Plan OpenWebUI webui.db chat import implementation
status: Done
assignee: []
created_date: '2026-05-10 21:55'
updated_date: '2026-05-10 22:02'
labels:
  - chatbooks
  - openwebui
  - planning
dependencies: []
documentation:
  - Docs/superpowers/plans/2026-05-10-openwebui-db-chat-import-implementation-plan.md
  - Docs/superpowers/specs/2026-05-10-openwebui-db-chat-import-design.md
  - 'https://docs.openwebui.com/reference/database-schema/'
parent_task_id: TASK-233
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a repo-grounded implementation plan for the approved v2 OpenWebUI webui.db chat import feature. The plan should turn TASK-233.4's design spec into staged, testable backend/frontend/docs work while preserving the approved constraints: uploaded DB only, explicit selected OpenWebUI user, reuse Chatbooks source_format=openwebui_db, mirror folders through existing tldw visible folder support, and preserve attachments/files/artifacts as metadata only.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Implementation plan references the approved webui.db import design spec and current Chatbooks/OpenWebUI import code paths.
- [x] #2 Plan decomposes backend adapter, Chatbooks API/service/jobs integration, folder helper, frontend, docs, and verification into reviewable stages.
- [x] #3 Plan includes concrete file paths, focused tests, commands, and expected verification outcomes.
- [x] #4 Plan calls out security controls for uploaded SQLite databases and Bandit requirements for touched Python code.
- [x] #5 Backlog task records verification and final summary for the planning artifact.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect current Chatbooks OpenWebUI JSON import, Chatbooks API/service/jobs flow, frontend import UI, and visible folder support. 2. Create a staged implementation plan covering adapter, schemas/endpoints/service dispatch, import/folder mirroring, Jobs, frontend, docs, and verification. 3. Manually review the plan against TASK-233.4 constraints and record verification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created `Docs/superpowers/plans/2026-05-10-openwebui-db-chat-import-implementation-plan.md`.

The plan is grounded in the current `openwebui.py` adapter, `ChatbookService.import_chatbook`, `chatbooks.py` preview/import endpoints, `jobs_worker.py`, `chatbook_schemas.py`, Chatbooks frontend import UI, and the visible folder system backed by keyword collections plus conversation-keyword links.

Manual review note: the plan accounts for the current `keyword_collections.name` global uniqueness constraint by requiring deterministic folder-name disambiguation instead of a schema migration in this slice.

Verification: targeted `rg` plan coverage check confirmed DB source format, selected-user field, DB preview/result schemas, folder stages, frontend paths, Jobs dispatch, SQLite safety, and Bandit coverage.

Verification: `git diff --cached --check` completed with no whitespace errors.

Bandit skip: docs/backlog-only planning slice; implementation plan includes Bandit command for the future Python code slice.

Known skips/blockers: runtime implementation remains future work.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added a staged implementation plan for uploaded OpenWebUI `webui.db` chat import. The plan breaks the feature into adapter, API/service dispatch, folder mirroring, async Jobs, frontend, docs/OpenAPI, and final verification stages with concrete file paths, focused tests, commands, security checks, and commit checkpoints.
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
