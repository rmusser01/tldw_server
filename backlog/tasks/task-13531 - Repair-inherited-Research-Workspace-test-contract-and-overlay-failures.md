---
id: TASK-13531
title: Repair inherited Research Workspace test contract and overlay failures
status: To Do
labels:
- knowledge
- research
- tests
- maintenance
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Burn down the ten named Research/Chat/storage assertions found during TASK-13530.1 final validation. An isolated canonical dev2c5f19d export reproduces all ten exact diagnostics: quota warning1, source-grounded provider1, workspace hook scope1 and SourceViewControls keyboard/dialog lifecycle7. Diagnose real current behavior before choosing fixture or product fixes; retain meaningful keyboard, focus, busy-state, server-confirmation, ownership and stale-submit protections. This is a separate reviewable maintenance unit rather than an inaccurate broad TASK12116 assignment.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 All ten canonical-dev failures have diagnosed causes and preserve or strengthen their intended behavior checks without skips or weakened security assertions.
- [ ] #2 Four owning suites pass under the documented native-resolution environment; focused default behavior and any inherited environment qualification are recorded accurately.
- [ ] #3 Any actual product defect is fixed at its existing shared root with meaningful RED/GREEN and independent review; types, touched lint/hooks and browser checks where applicable pass.
- [ ] #4 Tracking links canonical comparison receipts, exact final commits/checks, original related tasks and remaining native/device limits.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Scope before edits. Canonical dev comparison: /private/tmp/knowledge-capture-dev-baseline-four-suites-receipt.json,233cases223passed10failed0pending/exit1/46.65s; exact named and diagnostic match to feature621 run. Existing search found no open exact maintenance owner: TASK478.4,13260.268 and13394 are historical Done units; TASK12116 broader strict/lint/dependency work is not assigned all ten. ADR assessment: no new ADR required for fixture/contract repair under existing Research selection, saved normal Chat provider/ownership and dialog accessibility rules. New durable behavior/policy changes require controller assessment. User has authorized burning down followups; separate sequential Task7 is added to the approved execution plan. No new feature, dependency, broad refactor or assertion disabling.
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
