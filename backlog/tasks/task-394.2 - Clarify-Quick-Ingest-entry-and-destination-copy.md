---
id: TASK-394.2
title: Clarify Quick Ingest entry and destination copy
status: Done
assignee: []
created_date: '2026-05-16 00:42'
updated_date: '2026-05-28 06:06'
labels:
  - quick-ingest
  - ux
  - task-2
dependencies: []
documentation:
  - >-
    Docs/superpowers/plans/2026-05-16-quick-ingest-ux-remediation-implementation-plan.md
parent_task_id: TASK-394
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute implementation plan Task 2: improve first-time clarity, destination expectations, entry consistency, and accessible labels for the active Quick Ingest wizard.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Entry labels and wizard copy clarify Quick Ingest purpose and where items go
- [x] #2 First-time and returning-user flows keep the same quick path with better recognition over recall
- [x] #3 Accessible labels and keyboard/focus behavior are preserved or improved
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Started from merged dev 66c8bec4d in isolated worktree .worktrees/notes-post-merge-next. Scope: execute plan Task 2 only, focused Quick Ingest launcher/Add step copy and tests.

- Merged dev already included the first-open Add step purpose/destination copy and the Quick Ingest launcher wording/aria-label baseline.
- Added a regression test for Review step estimate copy to prevent duplicate approximation markers (for example, `~~21 min estimated`).
- Patched Review step summary and long-duration warning templates so `formatEstimate()` owns the `~` marker.
- Verification: local UI dependencies were repaired with `bun install` under `apps/` because copied worktree symlinks pointed at a missing node_modules store.
- Verification: `./node_modules/.bin/vitest run src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.integration.test.tsx --maxWorkers=1 --no-file-parallelism` passed after the red failure was observed.
- Verification: `./node_modules/.bin/vitest run src/components/Layouts/__tests__/QuickIngestButton.resume.test.tsx --maxWorkers=1 --no-file-parallelism` passed.
- Verification: `git diff --check` passed.
- Browser smoke: Next dev server started at http://127.0.0.1:18082 with local test API-key env; `/media` rendered the Quick Ingest launcher, but an unrelated dev runtime overlay from `GET /api/v1/media?page=1&results_per_page=20&include_keywords=true` blocked modal interaction. Dev server was stopped.
- Bandit: skipped because this slice only changes frontend TS/TSX and Backlog task metadata.
- PR review follow-up after rebasing on origin/dev: added explicit Quick Ingest review estimate locale entries without a leading approximation marker in the i18next and extension locale bundles, and loosened the duplicate-marker regression to accept every unit emitted by `formatEstimate()`.
- Review-fix verification: `./node_modules/.bin/vitest run src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.integration.test.tsx --maxWorkers=1 --no-file-parallelism`, `./node_modules/.bin/vitest run src/components/Layouts/__tests__/QuickIngestButton.resume.test.tsx --maxWorkers=1 --no-file-parallelism`, locale JSON/static no-`~{{time}}` check, and `git diff --check` passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed Task 2 focused slice. The merged baseline already had the first-open Add step destination copy and launcher terminology, so this slice fixed the remaining Review step estimate-copy defect and added regression coverage. Quick Ingest review now renders a single approximation marker from the formatter, avoiding strings like `~~21 min estimated` in both the summary and long-duration warning.
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
