---
id: TASK-15651
title: Flaky WorkspaceHeader ACP artifact tests in shared-ui-coverage
status: To Do
labels:
- webui
- tests
- ci
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/WorkspaceHeader.test.tsx failed in CI three times across two runs of PR #3615 on 2026-11-25 ("creates distinct ACP artifact versions when a completed run is saved repeatedly": expected mock called 2 times, got 1; "preserves a competing ACP version and requires another explicit save": expected 1, got 0). It passed on rerun each time and 3/3 locally. Not fixed here.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The tests pass reliably in shared-ui-coverage CI
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
