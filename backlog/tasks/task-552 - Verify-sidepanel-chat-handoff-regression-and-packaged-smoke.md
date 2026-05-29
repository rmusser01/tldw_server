---
id: TASK-552
title: Verify sidepanel chat handoff regression and packaged smoke
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-05-29 07:49'
labels:
  - chat
  - extension
  - verification
dependencies: []
references:
  - TASK-546
  - TASK-547
  - TASK-548
  - TASK-549
  - TASK-551
documentation:
  - Docs/superpowers/specs/2026-05-29-sidepanel-chat-webui-handoff-design.md
  - >-
    Docs/superpowers/plans/2026-05-29-sidepanel-chat-webui-handoff-implementation.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute Task 4 from the sidepanel chat WebUI handoff plan: run focused unit regressions, relevant existing playground/sidepanel tests, UI type/build sanity, packaged/browser smoke where available, record evidence and skips, and close the sidepanel chat handoff implementation verification slice.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Focused sidepanel handoff unit regression set passes or any unrelated baseline failures are documented.
- [ ] #2 Existing relevant playground/sidepanel tests pass or failures are documented with exact failing tests.
- [ ] #3 UI type/build sanity is run and recorded.
- [ ] #4 Packaged/browser smoke is run when a harness is available, otherwise the skip reason is recorded.
- [ ] #5 Bandit skip reason is documented for UI-only TypeScript/markdown scope.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->
<!-- SECTION:FINAL_SUMMARY:END -->

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
