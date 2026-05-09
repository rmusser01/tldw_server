---
id: TASK-148
title: Address PR 1396 Qodo review comment on humanizeMilliseconds tests
status: Done
assignee: []
created_date: '2026-05-09 04:01'
updated_date: '2026-05-09 04:02'
labels:
  - webui
  - dependencies
  - cleanup
  - review-fix
dependencies:
  - TASK-147
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1396'
  - 'https://github.com/rmusser01/tldw_server/pull/1396#discussion_r3212394989'
documentation:
  - Docs/Design/WebUI_Dependency_Audit.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address the actionable Qodo review thread on PR #1396 by restructuring the humanizeMilliseconds threshold coverage so each threshold scenario is localized to a single assertion while preserving the dependency guard.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The threshold coverage in apps/packages/ui/src/utils/__tests__/humanize-milliseconds.test.ts uses table-driven or split test cases with one assertion per threshold scenario.
- [x] #2 The dayjs source guard remains covered.
- [x] #3 Focused Vitest and git diff hygiene checks pass; Bandit skip rationale is recorded if no Python files change.
- [x] #4 The PR review thread is answered and, if possible, resolved after the fix is pushed.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verified the Qodo finding against the test file and addressed it by changing the threshold coverage to table-driven Vitest cases. Each generated test case has one assertion and failures are localized per threshold scenario. The dayjs source guard remains as a separate test.

Verification: bunx vitest run src/utils/__tests__/humanize-milliseconds.test.ts exited 0 from apps/packages/ui; git diff --check exited 0; bun run lint exited 0 from apps/tldw-frontend with existing unrelated warnings only. Bandit skipped because no Python files changed. PR review thread will be answered and resolved after the fix is pushed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Restructured the humanizeMilliseconds threshold test into table-driven single-assertion cases while keeping the dayjs source guard intact. This addresses the Qodo maintainability review comment on PR #1396 without changing runtime code.
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
