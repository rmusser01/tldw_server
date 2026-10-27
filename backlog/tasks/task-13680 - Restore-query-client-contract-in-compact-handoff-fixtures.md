---
id: TASK-13680
title: Restore query client contract in compact handoff fixtures
status: Done
labels:
- chat
- tests
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The final eight-suite handoff qualification reproduced 47 failures in two ControlRow fixtures before any handoff assertions: their react-query mocks omit useQueryClient, which the shared model catalog now requires. Preserve declared query I/O doubles and provide an actual isolated QueryClient; do not alter production ownership or handoff behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Both ControlRow suites and all eight handoff suites pass in native and WebUI profiles with original behavioral assertions intact.
- [x] #2 Only fixture query bindings change, with independent review and no production capability fabrication.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
PR #3229; original run 150 passed/47 failed across eight suites. ADR assessment: no durable architecture change; fixture contract correction under existing shared catalog and ADR-049 ownership. Bandit N/A TypeScript tests only. Master TASK-13458 records final contract coverage and packaged-shell limits.
Only two compact ControlRow fixture query bindings change: real QueryClient is exposed alongside the declared query-data I/O adapter. All eight handoff suites197/profile pass with original ownership/conflict/expiry/omission/failure assertions; production capability is unchanged. PR #3229; final bounded source review by two independent reviewers. TypeScript/JSON/Markdown-only unit: Bandit N/A locally; fresh full PR67-production-Python scan has0 introduced findings and0 scan errors vs exact merge-base. Installed extension/physical AT/touch/native-toolbar zoom/participants remain unrun. Overall merge and current Scene stability remain separate gates.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Only two compact ControlRow fixture query bindings change: real QueryClient is exposed alongside the declared query-data I/O adapter. All eight handoff suites197/profile pass with original ownership/conflict/expiry/omission/failure assertions; production capability is unchanged.
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
