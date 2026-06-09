---
id: TASK-481.18
title: Fix notes AI title strategy selector test
status: Done
labels:
- notes
- tests
- webui
parent_task_id: TASK-481
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix the deterministic Notes Stage 10 AI-title test failure where the strategy selector test cannot find the `LLM (quality)` AntD option. Keep the fix scoped to the root cause and restore the full Notes component sweep baseline.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Focused AI-title test file passes.
- [x] #2 Full Notes component sweep no longer fails on `NotesManagerPage.stage10.ai-title.test.tsx`.
- [x] #3 No production behavior is changed unless root-cause evidence shows a product bug.
- [x] #4 Verification and known skips are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Root cause: the test was stale, not the product behavior. It opened the AntD `Select` through the root element instead of the rendered `.ant-select-content` trigger used by other Notes tests, and it searched for the obsolete label `LLM (quality)` while the current UI label is `AI-powered`.
- Changed only the test harness interaction and option label assertion in `NotesManagerPage.stage10.ai-title.test.tsx`.
- Verification:
  - Red: `bunx vitest run src/components/Notes/__tests__/NotesManagerPage.stage10.ai-title.test.tsx` failed with `Unable to find an element with the text: LLM (quality)`.
  - Green focused: same command passed 1 file / 6 tests.
  - Full Notes sweep: `bunx vitest run src/components/Notes/__tests__` passed 67 files / 207 tests.
- Bandit: Not applicable; no Python/backend files changed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed the deterministic Stage 10 AI-title selector test by aligning it with current AntD Select markup and the current `AI-powered` strategy label. No production code changed; the full Notes component sweep now passes.
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
