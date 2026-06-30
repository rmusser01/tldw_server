---
id: TASK-233.2.1
title: 'Address PR #1542 OpenWebUI import review comments'
status: Done
assignee: []
created_date: '2026-05-10 19:33'
updated_date: '2026-05-10 19:49'
labels: []
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1542'
documentation:
  - >-
    Docs/superpowers/plans/2026-05-10-openwebui-chat-import-implementation-plan.md
parent_task_id: TASK-233.2
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve actionable review feedback on PR #1542 after rebasing the OpenWebUI chat JSON import branch onto dev.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 OpenWebUI preview rejects unsafe JSON paths with a user-facing error instead of surfacing a server error.
- [x] #2 OpenWebUI imports do not count or parent-link messages when DB insertion fails, and per-chat failures do not leave retry-blocking partial conversations.
- [x] #3 OpenWebUI import Jobs mark claimed validation failures as failed before raising.
- [x] #4 ChaCha source-ref lookup safely maps tuple-style Postgres rows.
- [x] #5 Frontend preview state ignores stale async preview responses after the selected file/source changes.
- [x] #6 Review-only test comments are addressed without asserting internal staged file suffixes or leaking test files into shared service temp directories.
- [x] #7 Targeted backend/frontend tests, git diff --check, and Bandit are rerun.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Review-fix pass for PR #1542: add failing regression tests for each actionable review issue, patch the service/worker/store/WebUI code narrowly, rerun targeted backend and frontend checks plus git diff --check and Bandit, then resolve stale or fixed review threads.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Review fixes implemented for PR #1542: safe OpenWebUI preview path errors, full per-chat rollback for failed message/settings/metadata writes, claimed import-job validation failure status, DB-owned cross-backend title lookup, tuple-style Postgres source-ref row conversion, stale WebUI preview response guard, multipart content_selections validation, non-UTF8 JSON rejection, and test hygiene cleanup.

Verification: latest run passed 32 Chatbooks backend tests, 65 selected ChaCha conversation/metadata tests, and 5 targeted UI tests. git diff --check was clean. Bandit JSON results=0 and errors=0 at /tmp/bandit_openwebui_chat_import_review_fixes.json.

Follow-up after the initial review-fix commit: multipart content_selections validation still returned FastAPI 422 before the endpoint parser for invalid form JSON. Added a default ImportChatbookRequest dependency so multipart fields are parsed explicitly by the endpoint and return HTTP 400.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed the actionable PR #1542 review feedback for OpenWebUI chat import. The importer now handles unsafe preview paths as user-facing validation errors, rolls back failed OpenWebUI chat imports instead of leaving duplicate-blocking partial conversations, persists claimed async import validation failures as failed, delegates exact title checks to a DB-owned cross-backend helper, maps tuple-style Postgres source-ref rows safely, and ignores stale WebUI preview responses. Tests were expanded for these regressions and review-only test assertions were cleaned up.

Follow-up verification after the multipart dependency fix: 44 focused backend tests passed, the targeted ChatbooksPlaygroundPage OpenWebUI Vitest file passed, git diff --check was clean, and Bandit returned results=[] for touched backend files in /tmp/bandit_openwebui_chat_import_review.json.
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
