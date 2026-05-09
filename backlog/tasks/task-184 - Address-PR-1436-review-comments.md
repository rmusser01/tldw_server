---
id: TASK-184
title: Address PR-1436 review comments
status: In Progress
assignee:
  - codex
created_date: '2026-05-09 19:39'
updated_date: '2026-05-09 19:40'
labels:
  - webui
  - dependencies
  - issue-1346
  - review-fix
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1436'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address actionable reviewer comments on PR #1436 for the Media FilterPanel dayjs cleanup. Scope is limited to the muted-text Tailwind token typo, timezone-stable date fixture, deterministic dayjs import scan root/path normalization, verification, and resolving the addressed review threads.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 FilterPanel native date labels use the existing text-text-muted token instead of text-textMuted.
- [x] #2 The existing date-range display test uses timezone-stable local ISO fixtures instead of fixed UTC-noon strings.
- [x] #3 FilterPanel.dayjs-imports.test.ts computes the scanned source root from the test file location and normalizes relative paths to forward slashes.
- [x] #4 Focused Media FilterPanel tests pass and diff whitespace check passes after the review fixes.
- [ ] #5 PR #1436 review threads for these findings are resolved or made outdated by the pushed fix commit.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Mirror TASK-184 into the PR branch worktree before code edits.
2. Verify the reviewer claims against local code and token naming.
3. Patch FilterPanel labels from text-textMuted to text-text-muted.
4. Patch the date display fixture to use isoForLocalDate for local-time deterministic ISO values.
5. Patch the dayjs import scan test to derive srcRoot from the test file directory and normalize relative paths to forward slashes.
6. Run the focused Media FilterPanel Vitest suite, exact dayjs import scan, WebUI lint, targeted TypeScript filter, and git diff --check.
7. Commit and push the review fixes, then resolve or confirm outdated the addressed PR review threads and recheck PR status.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Inspected PR #1436 review surface. Actionable findings are: duplicated text-textMuted label-token comments, timezone-sensitive date display fixture, and process.cwd()/path-separator sensitivity in FilterPanel.dayjs-imports.test.ts. CodeRabbit status is passing; GitHub Actions checks are still pending.

Implemented the three review fixes on PR #1436: replaced both text-textMuted classes with text-text-muted, changed the date display fixture to isoForLocalDate local-boundary fixtures, and made FilterPanel.dayjs-imports.test.ts derive srcRoot from __dirname with forward-slash relative path normalization.

Verification: focused Media FilterPanel Vitest suite passed with 12 tests; exact dayjs import scan still returns only the four deferred ReadingList/Items imports; text-textMuted scan returned no matches; git diff --check exited 0; WebUI lint exited 0 with the existing 131-warning baseline.

TypeScript baseline note: `node_modules/.bin/tsc --noEmit --project tsconfig.json --pretty false` still exits 2 on existing EmbeddingsModelSelectionConfig.tsx and lib/api/vnPlay.ts errors; filtering /tmp/task184_tsc.log for FilterPanel, components/Media, Media/__tests__, task-184, and task-176 returned no matches.

Bandit skipped because this review-fix slice changed TypeScript/tests and Backlog metadata only; no Python files were modified.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
