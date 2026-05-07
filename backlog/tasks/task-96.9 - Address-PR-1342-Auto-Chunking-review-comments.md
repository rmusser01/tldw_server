---
id: TASK-96.9
title: Address PR 1342 Auto Chunking review comments
status: Done
assignee: []
created_date: '2026-05-07 00:35'
updated_date: '2026-05-07 01:12'
labels:
  - review-fix
  - auto-chunking
  - backend
  - frontend
dependencies:
  - TASK-96.7
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1342'
parent_task_id: TASK-96
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve all actionable review comments on PR #1342 for the Auto Chunking branch. Scope includes backend Auto Chunking correctness, persistence consistency, web scraping runtime/logging fixes, frontend stale manual-field filtering, and review hygiene items from CodeRabbit, Qodo, and Gemini.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 All still-valid inline and summary review comments on PR #1342 are implemented or explicitly documented as not applicable.
- [x] #2 Backend Auto Chunking persisted ingest paths use resolved Auto chunk options consistently with stored chunking_plan metadata.
- [x] #3 Frontend Auto mode does not submit or compare stale Manual-only chunking fields.
- [x] #4 Focused backend/frontend tests, OpenAPI verification where relevant, Bandit on touched backend scope, and git diff checks are run and recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented PR #1342 review fixes: shared chunking form coercion, mapping-aware Auto resolver, resolved Auto chunk options in persistence, finalized job chunking_plan preference, per-result hierarchical chunking decisions, web scraping logging/scope fixes, and Quick Ingest Auto/Manual stale-field filtering.

Verification: backend focused suite 36 passed; frontend focused Vitest suite 69 passed; OpenAPI internal refs test passed; Bandit touched backend scope reported 0 findings in /tmp/bandit_auto_chunking_review.json; git diff --check passed.

Follow-up PR review pass: verified live PR #1342 unresolved threads. Most are already fixed at branch tip; adding explicit /media/add Auto chunking regression coverage and removing the stale prepare_chunking_options_dict helper alias before resolving remaining review threads.

Follow-up verification: tldw_Server_API/tests/MediaIngestion_NEW/unit/test_persistence_original_storage.py passed 12 tests; focused Auto Chunking backend suite passed 48 tests; targeted media/add regression tests passed 2 tests; Bandit on persistence.py reported 0 findings in /tmp/bandit_auto_chunking_media_add_followup.json; git diff --check passed.

Frontend review verification rerun: direct root-level bunx vitest invocation failed before collection because UI alias config was not loaded; reran from apps/packages/ui with package vitest config and Quick Ingest focused suite passed 69 tests.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Follow-up review issue addressed: /api/v1/media/add no longer carries the stale prepare_chunking_options_dict helper alias, and new regression tests prove Auto mode routes resolved planner chunk options into both document-like and audio/video batch processors instead of stale manual options. Existing unresolved PR threads were rechecked against the branch tip before resolution.
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
