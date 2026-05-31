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
modified_files:
- Docs/superpowers/specs/2026-05-31-unified-solo-onboarding-uat-repair-design.md
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
Design spec created and reviewed: Docs/superpowers/specs/2026-05-31-unified-solo-onboarding-uat-repair-design.md. Self-review tightened local-only auth handoff, inline auth readiness, deterministic first-source UAT fixture, and durable web-ingest semantics. Spec review then found one visibility/searchability wording conflict; it was corrected and the final review approved with no blocking issues. Implementation planning should start after the user review gate passes.
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
