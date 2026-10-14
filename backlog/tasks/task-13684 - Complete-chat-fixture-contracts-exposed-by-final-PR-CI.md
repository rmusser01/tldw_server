---
id: TASK-13684
title: Complete chat fixture contracts exposed by final PR CI
status: Done
labels:
- chat
- testing
- ci
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Repair incomplete shared-hook fixture exports and diagnose the remaining Scene/reset test failures on the rebased PR3229 head, preserving production ownership guards and behavioral assertions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Affected fixture modules retain actual shared hook and query client contracts
- [x] #2 Scene conflict and reset-refusal regressions are diagnosed and pass without relaxed assertions or raised deadlines
- [x] #3 Affected native UI and client profiles pass and independent reviewers inspect the final delta
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR assessment: no new durable architecture decision; existing ADR049/050 ownership and ADR059 task editing govern this test fixture repair. CI RED: head716e8cd shards3/4/6 show missing useQueryClient / useActorSceneHistoryId exports, one Scene read-count failure, and one reset-refusal timeout. Plan: inventory caller mocks; reproduce under native UI CI profile; minimal fixture repair; verify affected suites and independent review; publish and rerun required gates.
Final actual provider-failure journey initially received200because the owned mock was still in local-success mode; this is fixture configuration, not a relaxed401 assertion. Native654affected fixture cases pass. Both WebUI/extension profiles are running all20 affected/connected suites (1121cases perprofile expected). Protected provider-failure qualification will use the existing CI mock configuration and retain401/receipt/canonical-row/explicitRetry checks.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed all caller fixture exports through actual QueryClient and partial Scene-hook modules; restored timer spies before uninstalling fake time; preserved Scene CAS and source/receipt assertions; supplied the compact admission gate and corrected only dispatched uncertainty expectations. Native654fixture cases pass; complete WebUI and extension20-suite profiles each pass1121cases. Protected401/explicitRetry/canonical-row journey passes13.9s with the existing CI failure mock; original successful mock restored. Final independent reviewers report no actionable finding. Parent TASK13458 tracks remote-dev integration and exact-head landing.
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
