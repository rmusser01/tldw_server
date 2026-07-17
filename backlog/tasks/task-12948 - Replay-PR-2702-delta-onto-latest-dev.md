---
id: TASK-12948
title: Replay PR 2702 delta onto latest dev
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-07-12 05:08'
labels:
  - release
  - rebase
  - review
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a fresh dev-targeted branch from current origin/dev, replay only the PR #2702-specific commits, resolve conflicts in favor of newer dev behavior where appropriate, validate the resulting net delta, and open a PR targeting dev. Do not modify or open a PR against main.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Branch is based on the latest origin/dev and contains no main-targeted correction
- [x] #2 Only PR #2702-specific changes still missing from current dev are included
- [x] #3 Conflicts are resolved against newer dev behavior with minimal changes
- [x] #4 Focused backend/frontend/workflow validation passes
- [x] #5 A pull request is opened targeting dev
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created codex/pr-2702-dev-rebase from origin/dev d28c16bfa3 and replayed the eight PR #2702-specific commits while preserving newer dev changes. Eight tracking/review commits bring the branch to 16 commits and 43 changed files relative to dev. Independent and GitHub review fixes cover snapshot archive-first deletion, logical-session quota aggregation across current/legacy directories, shared maintenance locking, deferral of unidentified snapshot directories that cannot be safely locked, SQLite draft read/connection behavior, abortable duplicate/stale document preparation with accepted ingest-job cancellation, strict recovery capability checks, accurate localized statuses/errors, visible completed-import scope refresh coverage, and narrow test corrections. GitHub Actions queue: max was rejected because the concurrency schema does not support it. Fresh validation: 70 frontend tests and 78 backend tests passed; extension compile, Playwright discovery, workflow validation, Bandit zero findings, and git diff checks passed. Full WebUI typecheck still reports only the pre-existing untouched QuickIngestWizardModal.tsx overflowY type error. Branch is 0 behind origin/dev.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
PR #2712 targets dev from current origin/dev and contains only the PR #2702 replay plus validated review fixes; no main-targeted correction exists. The final branch is 0 behind dev, with 8 replayed commits plus 8 tracking/review commits across 43 files. Validation passed for 70 focused frontend tests, 78 focused backend tests, extension TypeScript compile, Playwright discovery, workflow parsing/concurrency, Bandit, and diff checks. The only full WebUI typecheck failure is the pre-existing untouched QuickIngestWizardModal.tsx overflowY typing issue.
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
