---
id: TASK-151
title: Address PR 1398 review comments on WorldBooks relative time helper
status: In Progress
assignee: []
created_date: '2026-05-09 04:38'
updated_date: '2026-05-09 04:48'
labels:
  - webui
  - dependencies
  - cleanup
  - dayjs
  - review-fix
dependencies:
  - TASK-149
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1398'
  - 'https://github.com/rmusser01/tldw_server/pull/1398#discussion_r3212418048'
  - 'https://github.com/rmusser01/tldw_server/pull/1398#discussion_r3212422167'
documentation:
  - Docs/Design/WebUI_Dependency_Audit.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address the actionable Gemini/Qodo review feedback on PR #1398 by preventing singular-unit pluralization at relative-time rounding boundaries and tightening the WorldBooks no-dayjs guard test to detect imports instead of any incidental substring.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Focused tests cover relative-time rounding boundaries that previously produced labels such as 1 minutes ago or 1 hours ago.
- [x] #2 worldBookListUtils formats singular boundary labels with dayjs-compatible wording for representative seconds, minutes, hours, days, and years cases.
- [x] #3 The no-dayjs guard detects dayjs import statements instead of any incidental dayjs substring.
- [x] #4 Focused Vitest and git diff hygiene checks pass; lint result and Bandit skip rationale are recorded if applicable.
- [ ] #5 The actionable PR review threads are answered and resolved after the fix is pushed.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
TDD red check: focused WorldBooks utility test failed before the production fix with pluralized singular labels including 1 minutes ago, 1 hours ago, 1 days ago, and 1 years ago.

Implemented review fix in worldBookListUtils by keeping dayjs-compatible raw millisecond thresholds for singular boundary ranges and routing counted units through a singular/plural helper.

Tightened the no-dayjs test guard to match dayjs import statements instead of any incidental substring.

Verification: bunx vitest run src/components/Option/WorldBooks/__tests__/worldBookListUtils.test.ts passed with 19 tests; bun run test:worldbooks passed with 62 files, 208 passed, 6 skipped; git diff --check passed; bun run lint in apps/tldw-frontend exited 0 with the existing 131 warnings baseline and no touched-file warnings; Bandit skipped because this review fix only touches TypeScript/test/Backlog files.
<!-- SECTION:NOTES:END -->
