---
id: TASK-162
title: Address PR 1411 date-display review comments
status: In Progress
assignee:
  - codex
created_date: '2026-05-09 06:15'
updated_date: '2026-05-09 06:31'
labels:
  - webui
  - dependencies
  - cleanup
  - dayjs
  - flashcards
  - review-fix
dependencies:
  - TASK-158
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1411'
  - 'https://github.com/rmusser01/tldw_server/pull/1411#discussion_r3212561415'
  - 'https://github.com/rmusser01/tldw_server/pull/1411#discussion_r3212561417'
documentation:
  - Docs/Design/WebUI_Dependency_Audit.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address actionable review feedback on PR #1411 for the FlashcardEditDrawer dayjs cleanup. Scope is limited to fixing the scheduling metadata test's vacuous timestamp assertion and hardening Flashcards date-display numeric timestamp parsing/combined formatting so millisecond timestamps before 2001 are not reinterpreted as seconds.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 FlashcardEditDrawer scheduling metadata test no longer falls back to an empty expected timestamp string before includes() assertions.
- [x] #2 Flashcards date-display helpers correctly treat pre-2001 millisecond timestamps as milliseconds and avoid re-parsing an already-normalized timestamp in the combined absolute/relative helper.
- [x] #3 Focused date-display and FlashcardEditDrawer scheduling metadata tests cover the review regressions and pass.
- [ ] #4 PR #1411 actionable review threads are replied to and resolved after the fix is pushed.
- [x] #5 Diff hygiene, focused import scan, lint or baseline rationale, and Bandit skip/run rationale are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed the review feedback by removing the FlashcardEditDrawer scheduling metadata test's empty-string fallback, asserting formatter outputs are non-null before includes() assertions, lowering the numeric epoch-seconds cutoff to 10_000_000_000, and formatting combined absolute/relative labels from the already-normalized millisecond timestamp.

Verification so far: red Vitest run failed on the new pre-2001 millisecond timestamp regression before implementation; after the fix, `bunx vitest run src/components/Flashcards/utils/__tests__/date-display.test.ts src/components/Flashcards/components/__tests__/FlashcardEditDrawer.scheduling-metadata.test.tsx` passed with 2 files and 23 tests. `git diff --check` passed. Flashcards component/utils dayjs import scan returned no matches; broader shared-ui dayjs scan remains the known 11-line baseline. `bun run lint` in apps/tldw-frontend exited 0 with the existing 131-warning baseline. Full apps/packages/ui TypeScript check still exits 2 on unrelated baseline diagnostics; filtered diagnostics for `date-display` and `FlashcardEditDrawer.scheduling-metadata` returned no matches. Bandit is not applicable because this review fix touches TypeScript/frontend test files and a Backlog task file, not Python code.

Additional CodeRabbit review pass: removed `React.useMemo` around `dueAtDisplay` and `lastReviewedDisplay` so relative labels are recomputed on render, changed numeric timestamp detection to compare `Math.abs(value)` against the seconds cutoff, and added regression coverage for pre-1970 numeric second and millisecond timestamps. Verification after this pass: red Vitest failed for the new pre-1970 millisecond case before implementation; after the fix, focused Vitest passed with 2 files and 25 tests, `git diff --check` passed, Flashcards component/utils dayjs scan returned no matches, broader shared-ui dayjs scan remains the known 11-line baseline, `bun run lint` exited 0 with the existing 131-warning baseline, and the narrow TypeScript diagnostic filter for `date-display`, `FlashcardEditDrawer.tsx`, and `FlashcardEditDrawer.scheduling-metadata` returned no matches.
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
