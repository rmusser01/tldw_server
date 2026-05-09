---
id: TASK-164
title: Remove dayjs from Flashcards Review and Manage display formatting
status: Done
assignee:
  - codex
created_date: '2026-05-09 15:32'
updated_date: '2026-05-09 15:58'
labels:
  - webui
  - dependencies
  - cleanup
  - dayjs
  - flashcards
dependencies:
  - TASK-158
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1346'
  - 'https://github.com/rmusser01/tldw_server/pull/1411'
  - 'https://github.com/rmusser01/tldw_server/pull/1417'
documentation:
  - Docs/Design/WebUI_Dependency_Audit.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue GitHub issue #1346 after PR #1411 by replacing the remaining display-only dayjs usage in Flashcards ReviewTab and ManageTab with native date/time helpers. Scope is limited to review-result due labels, next-due labels, card-list due badges, and expanded due metadata. Leave Ant Design Dayjs value-contract surfaces outside Flashcards for separate design work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Flashcards ReviewTab no longer imports dayjs or dayjs/plugin/relativeTime for review-result or next-due display labels.
- [x] #2 Flashcards ManageTab no longer imports dayjs or dayjs/plugin/relativeTime for due-status and due label display formatting.
- [x] #3 Native helper coverage preserves representative dayjs-compatible relative labels and absolute display labels used by ReviewTab and ManageTab.
- [x] #4 The WebUI dependency audit records the shared UI dayjs import count reduction and remaining Ant Design Dayjs value-contract surfaces.
- [x] #5 Focused Vitest, exact Flashcards tab dayjs import scan, broader shared UI dayjs import scan, lint or baseline rationale, git diff hygiene, and Bandit skip/run rationale are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Create isolated worktree from current origin/dev. [done]
2. Add focused failing tests around shared Flashcards date-display helper coverage. [done]
3. Replace ReviewTab and ManageTab display-only dayjs calls with shared native helpers. [done]
4. Update dependency audit and task metadata. [done]
5. Run focused Vitest, import scans, lint, diff hygiene, and PR packaging. [done]
6. Address PR #1417 review comments and rerun focused verification. [done]
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
PR #1411 was verified merged into dev at 2026-05-09T15:29:29Z. Current origin/dev audit shows 11 remaining shared UI dayjs import lines: ReviewTab and ManageTab display formatting plus Ant Design Dayjs value/type surfaces in Media, ReadingList, Items, DataTables, and Kanban. This task targets only the remaining Flashcards display-only imports.

RED verified with `bunx vitest run src/components/Flashcards/utils/__tests__/date-display.test.ts` from `apps/packages/ui`: 5 failures because `formatFlashcardLongDateTime` and `isFlashcardTimestampBefore` were missing.

Implemented native long absolute date formatting, timestamp-before checks, and replaced display-only `dayjs` calls in Flashcards ReviewTab and ManageTab.

Focused helper test now passes and exact Flashcards tabs package-import scan returns no direct `dayjs` imports.

Broader shared UI package-import scan now finds 7 remaining `dayjs` lines, all in Ant Design `Dayjs` value/type surfaces.

Verification: `bunx vitest run src/components/Flashcards/utils/__tests__/date-display.test.ts src/components/Flashcards/tabs/__tests__/ManageTab.scheduling-metadata.test.tsx src/components/Flashcards/tabs/__tests__/ReviewTab.create-cta.test.tsx` passed with 49 tests.

Verification: exact Flashcards tabs `dayjs` package-import scan returned no matches; broader shared UI scan returned 7 remaining matches in Ant Design value-contract surfaces.

Verification: `bun run lint` from `apps/tldw-frontend` exited 0 with the existing 131-warning baseline and no touched-file lint errors.

Verification: `git diff --check` passed.

TypeScript baseline: `bunx tsc -p tsconfig.json --noEmit --pretty false` from `apps/packages/ui` still exits 2 on existing repo-wide test/service type errors outside this slice; no errors were reported in touched ReviewTab, ManageTab, or date-display files in the observed output.

Bandit skipped because this slice changed TypeScript, documentation, and Backlog metadata only; no Python files were modified.

PR #1417 review pass: Gemini left two inline comments on `date-display.ts` asking to replace hardcoded weekday/month arrays and manual long-date assembly with `Intl.DateTimeFormat` to avoid locale-specific lookup tables. Verified as actionable for this codebase; applying a narrow helper change and preserving deterministic en-US coverage via test options.

PR #1417 review fix implemented: replaced the manual long-date weekday/month arrays and 12-hour assembly with `Intl.DateTimeFormat`, with deterministic `en-US` test coverage for the prior display label.

Post-review verification: focused Flashcards Vitest command passed again with 49 tests; exact Flashcards tabs `dayjs` scan returned no matches; broader shared UI `dayjs` scan remains 7 Ant Design value-contract surfaces; `git diff --check` passed; `bun run lint` still exits 0 with the existing 131-warning baseline; package TypeScript still exits 2 on existing repo-wide baseline errors outside this slice with no touched-file errors observed.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed the remaining display-only dayjs usage from Flashcards ReviewTab and ManageTab by extending the shared native date-display helpers with long absolute formatting and timestamp-before checks. The PR review follow-up replaced manual English weekday/month arrays with `Intl.DateTimeFormat` while keeping deterministic en-US helper coverage. The dependency audit now records the shared UI dayjs import count reduction from 11 to 7, with all remaining imports limited to Ant Design Dayjs value/type surfaces. Verification passed for focused Flashcards Vitest coverage, exact import scans, WebUI lint, and diff hygiene; the shared UI TypeScript command still fails on existing repo-wide baseline errors outside this slice, and Bandit was skipped because no Python files changed.
<!-- SECTION:FINAL_SUMMARY:END -->
