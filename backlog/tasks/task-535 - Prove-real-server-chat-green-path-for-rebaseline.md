---
id: TASK-535
title: Prove real-server chat green path for rebaseline
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-28 02:42'
labels:
  - chat
  - ux
  - e2e
  - webui
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Expand the /chat rebaseline from rail and sidepanel proof into a focused real-server green-path suite and refreshed UX evidence. Keep scope to /chat and direct sidepanel handoff: configured provider first send, visible streaming/loading or recoverable response state, stop/retry/regenerate affordance evidence where deterministic, model switch, Web search/status feedback, assistant select/clear, and refreshed first-run/mobile screenshots. Do not implement draft, page-context, or thread transfer from the extension in this task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Real-server /chat proof covers configured provider selection and first send against the local backend/mock provider.
- [x] #2 The suite or evidence captures loading/streaming and stop/retry/regenerate behavior, or documents the exact deterministic blocker and leaves a focused follow-up.
- [x] #3 Web search toggle/status-strip behavior is verified in /chat without requiring a live third-party search provider call.
- [x] #4 Assistant/persona select and clear remain proven in the same real-server workflow or through a linked focused test.
- [x] #5 Post-rebase first-time and mobile /chat screenshots/evidence JSON are refreshed and the rebaseline review doc is updated.
- [x] #6 Extension sidepanel full-screen entry remains route-only to /chat and is referenced as already covered by TASK-534 rather than reworked.
- [x] #7 Verification commands, skips, blockers, and Bandit applicability are recorded in the task.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-28-chat-green-path-proof.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Started TASK-535 after TASK-534 packaged sidepanel proof. Scope is a focused real-server /chat green-path suite and refreshed evidence only; route-only extension handoff remains covered by TASK-534 and draft/page/thread transfer stays out of scope.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed TASK-535. Added strict real-server /chat streaming stop/regenerate coverage, refreshed first-time/configured desktop/mobile evidence assets plus evidence.json and the rebaseline review, and fixed the mobile cockpit composer overlap regression found by the focused E2E subset. Verification: focused real-server subset passed 5 tests (desktop cockpit/focus, mobile cockpit, mobile send, model-provider confidence, streaming stop/regenerate); mobile cockpit single test passed; cockpit guard passed 6 tests; extra cockpit a11y/maturity/sticky-composer Vitest batch passed 18 tests. Bandit skipped because no Python files were changed. TASK-534 remains the route-only packaged sidepanel handoff reference.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched Python scope or documented non-Python skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
