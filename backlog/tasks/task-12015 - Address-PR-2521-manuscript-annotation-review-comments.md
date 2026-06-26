---
id: TASK-12015
title: Address PR 2521 manuscript annotation review comments
status: Done
labels:
- review
- webui
- backend
- writing-playground
- manuscripts
references:
- PR #2521
modified_files:
- apps/extension/tests/e2e/writing-playground-mode-parity.spec.ts
- apps/packages/ui/src/components/Option/WritingPlayground/WritingAnnotationCard.tsx
- apps/packages/ui/src/components/Option/WritingPlayground/WritingAnnotationList.tsx
- apps/packages/ui/src/components/Option/WritingPlayground/WritingAnnotationMarginRail.tsx
- apps/packages/ui/src/components/Option/WritingPlayground/WritingAnnotationsTab.tsx
- apps/packages/ui/src/components/Option/WritingPlayground/WritingTipTapEditor.tsx
- apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingAnnotationMarginRail.test.tsx
- apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingAnnotationsTab.test.tsx
- apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingPlayground.manuscript-api-shapes.guard.test.ts
- apps/packages/ui/src/components/Option/WritingPlayground/__tests__/writing-annotation-anchor-utils.test.ts
- apps/packages/ui/src/components/Option/WritingPlayground/index.tsx
- apps/packages/ui/src/components/Option/WritingPlayground/writing-annotation-anchor-utils.ts
- backlog/tasks/task-12013 - Fix-writing-annotation-review-job-acquisition-filter.md
- backlog/tasks/task-12014 - Fix-Task-5-writing-annotation-review-code-quality-issues.md
- tldw_Server_API/app/api/v1/endpoints/writing_manuscripts.py
- tldw_Server_API/app/api/v1/schemas/writing_manuscript_schemas.py
- tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py
- tldw_Server_API/app/core/DB_Management/ManuscriptDB.py
- tldw_Server_API/app/core/Jobs/manager.py
- tldw_Server_API/app/core/Writing/manuscript_annotation_jobs.py
- tldw_Server_API/app/core/Writing/manuscript_annotations.py
- tldw_Server_API/app/core/exceptions.py
- tldw_Server_API/app/services/writing_annotation_review_jobs_worker.py
- tldw_Server_API/tests/Writing/test_manuscript_annotation_review_jobs.py
- tldw_Server_API/tests/Writing/test_manuscript_annotations_db.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address reviewer and CI follow-up items on PR #2521 after rebasing the manuscript annotations branch onto latest dev. Track validation, touched files, and final PR comment summary here.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Rebase PR branch onto latest `origin/dev`.
- [x] Address current PR review comments that are valid against the rebased branch.
- [x] Add or update focused regression coverage for changed behavior.
- [x] Run focused frontend/backend verification and Bandit on touched backend scope.
- [x] Record any known verification skips or unrelated failures.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Rebasing onto latest `origin/dev` completed with autostash; unrelated local dirty files were preserved and left unstaged.
- Addressed frontend review issues for UTF-16/code-point offset conversion in the margin rail, async mutation error handling, exact service guard assertions, TipTap scene-content sync, TipTap local echo handling, and E2E fixture cleanup on seed failure.
- Addressed backend review issues for bulk scene-row loading during annotation listing, tag element validation, static update assignment fragments, scene-review duplicate suppression before max cap, empty model-output diagnostics, writing queue allowlisting, centralized writing annotation exceptions, selected-text review endpoint extraction, and PostgreSQL `manuscript_annotations` sync-log triggers.
- Marked TASK-12013 and TASK-12014 Definition of Done checklists complete to match their Done status and final summaries.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Rebased `codex/writing-manuscript-annotations-design` onto the latest `origin/dev` and addressed the PR #2521 review comments. The follow-up commit hardens the Writing Playground manuscript annotations flow across UI, API, DB, Jobs, and task metadata:

- Fixed margin rail offset conversion for persisted code-point anchors, with emoji regression coverage.
- Added UI mutation rejection handling and toast routing for annotation list/tab actions.
- Tightened manuscript annotation service guard tests so type/function assertions target the exact exports.
- Prevented TipTap scene content from re-seeding over local edits and skipped hot-path full-document comparison for editor-originated echoes.
- Cleaned up the E2E margin rail seed fixture if setup fails after project creation.
- Bulk-loaded scene rows for annotation list anchor derivation to avoid N+1 reads.
- Rejected non-string annotation tags and preserved static SQL assignment fragments for annotation updates.
- Added PostgreSQL sync-log trigger/function coverage for `manuscript_annotations`.
- Made scene-review processing deduplicate before `max_comments`, report empty model output, and use the real Jobs queue allowlist.
- Moved writing annotation review exceptions into the shared exceptions module and extracted selected-text AI review persistence out of the API route.

Verification:
- `git diff --check`: passed.
- `bunx vitest run src/components/Option/WritingPlayground/__tests__/WritingPlayground.manuscript-api-shapes.guard.test.ts src/components/Option/WritingPlayground/__tests__/WritingTipTapEditor.external-sync.test.tsx src/components/Option/WritingPlayground/__tests__/writing-annotation-anchor-utils.test.ts src/components/Option/WritingPlayground/__tests__/WritingAnnotationMarginRail.test.tsx src/components/Option/WritingPlayground/__tests__/WritingAnnotationsTab.test.tsx`: 5 files passed, 42 tests passed.
- `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Writing/test_manuscript_annotations_db.py tldw_Server_API/tests/Writing/test_manuscript_annotation_review_jobs.py tldw_Server_API/tests/Writing/test_manuscript_annotations_api.py tldw_Server_API/tests/Services/test_writing_annotation_review_jobs_worker.py -q`: 59 passed, 5 warnings.
- `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/DB_Management/ManuscriptDB.py tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py tldw_Server_API/app/core/Writing/manuscript_annotations.py tldw_Server_API/app/core/Writing/manuscript_annotation_jobs.py tldw_Server_API/app/api/v1/endpoints/writing_manuscripts.py tldw_Server_API/app/services/writing_annotation_review_jobs_worker.py tldw_Server_API/app/core/Jobs/manager.py tldw_Server_API/app/core/exceptions.py -f json -o /tmp/bandit_pr2521_review.json`: passed, zero findings.

Known verification note:
- `NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false` remains red on pre-existing unrelated package errors in Notes tests, AudioStudio, ScheduledTasks, Setup tests, Dexie audiobook migration, background response narrowing, scheduled-tasks control-plane params, and voice cloning. No touched Writing Playground annotation files were reported.
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
