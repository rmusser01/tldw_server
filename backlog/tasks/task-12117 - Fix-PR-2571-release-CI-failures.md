---
id: TASK-12117
title: Fix PR 2571 release CI failures
status: In Progress
labels:
- ci
- release
- pr-2571
priority: high
references:
- https://github.com/rmusser01/tldw_server/pull/2571
modified_files:
- Helper_Scripts/release.py
- tldw_Server_API/tests/Docs/test_release_docs_contract.py
- apps/packages/ui/src/components/Option/Playground/__tests__/Playground.responsive-parity.guard.test.ts
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed current PR #2571 CI failures by allowing release README version updates to use the current combined beyond/post-release status line and by updating the Playground responsive parity guard to the current mobile/focus-mode condition. Validation: docs suite 117 passed; playground device-matrix 15 passed; Bandit on Helper_Scripts/release.py reported 0 results; git diff --check clean.
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
