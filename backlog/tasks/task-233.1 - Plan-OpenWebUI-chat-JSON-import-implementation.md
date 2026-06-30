---
id: TASK-233.1
title: Plan OpenWebUI chat JSON import implementation
status: Done
assignee: []
created_date: '2026-05-10 16:31'
updated_date: '2026-05-10 16:36'
labels: []
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-05-10-openwebui-chat-import-design.md
parent_task_id: TASK-233
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a repo-grounded implementation plan for importing normal OpenWebUI Export Chats JSON files through the existing Chatbooks import flow. The plan should carry forward the approved design decisions from TASK-233 and split the work into reviewable stages for backend parsing/import, Chatbooks API/job integration, ChaCha metadata helpers, frontend UX/client updates, tests, docs, and verification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Implementation plan document is created under Docs/superpowers/plans with a task-specific filename.
- [x] #2 Plan references the approved design decisions from TASK-233 and does not reopen out-of-scope direct database or live-server import work for v1.
- [x] #3 Plan covers backend OpenWebUI JSON adapter, Chatbooks endpoint/schema/job branching, ChaCha duplicate/metadata helpers, frontend import controls, and documentation updates.
- [x] #4 Plan includes concrete test and verification checkpoints for parser behavior, duplicate handling, message tree preservation, API preview/import flows, frontend behavior, and security validation.
- [x] #5 Backlog task notes are updated with the plan path and review outcome before committing.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect current Chatbooks backend import/preview paths, async worker payload handling, and ZIP validation boundaries.
2. Inspect ChaCha conversation/message storage helpers for duplicate lookup and metadata persistence extension points.
3. Inspect WebUI Chatbooks import UI and tldw client helpers for multipart option handling.
4. Draft a staged implementation plan under Docs/superpowers/plans with tests, docs, and verification gates.
5. Review the plan for ambiguity, missed edge cases, and implementation-order risks before committing.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created implementation plan at Docs/superpowers/plans/2026-05-10-openwebui-chat-import-implementation-plan.md.

Review pass completed against the approved TASK-233 design and current code paths in Chatbooks endpoints/service/worker, ChatbookValidator, ChaCha conversation/message stores, and the WebUI Chatbooks import tab.

Plan review improvements added: multipart form parsing for source_format, JSON branching before ZIP validation, forcing unsupported OpenWebUI media/embedding options false despite current WebUI defaults, hiding Chatbook content selection for OpenWebUI v1, generalizing JSON path resolution wording, and updating static API docs if required.

Verification: git diff --check passed. No pytest/Bandit run because this task only adds docs and Backlog planning metadata, not implementation code.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created and reviewed the implementation plan for OpenWebUI chat JSON import. The plan keeps v1 bounded to normal OpenWebUI Export Chats JSON, extends the existing Chatbooks import path, preserves full message trees, covers duplicate/rename behavior, details backend/frontend/test/security stages, and records review-driven corrections for multipart source_format parsing and OpenWebUI option normalization.
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
