---
id: TASK-576
title: Address unified solo onboarding UAT blockers
status: In Progress
labels:
- onboarding
- uat
- webui
- quick-ingest
documentation:
- Docs/superpowers/specs/2026-05-31-unified-solo-onboarding-uat-repair-design.md
- Docs/superpowers/plans/2026-05-31-unified-solo-onboarding-uat-repair-implementation-plan.md
modified_files:
- Docs/superpowers/specs/2026-05-31-unified-solo-onboarding-uat-repair-design.md
- Docs/superpowers/plans/2026-05-31-unified-solo-onboarding-uat-repair-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Plan and implement repairs required for a clean first-time solo-user walkthrough: root setup entry, WebUI auth handoff, first chat completion, first-source ingest, and fresh-install UAT verification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Design spec created and reviewed: Docs/superpowers/specs/2026-05-31-unified-solo-onboarding-uat-repair-design.md. Implementation plan created: Docs/superpowers/plans/2026-05-31-unified-solo-onboarding-uat-repair-implementation-plan.md. Current preflight result: focused implementation can start after the current ~2.1GiB free-space check, but Docker rebuild/fresh-install UAT remains blocked until at least 10GiB is available. Next step after user approval is targeted cleanup if heavy UAT/build work is needed, otherwise Stage 1 red-green implementation can begin.
<!-- SECTION:PLAN:END -->

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
