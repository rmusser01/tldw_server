---
id: TASK-15670
title: Design outbound destination inventory and privacy regression checks
status: Done
labels:
- security
- privacy
- ci
- design
references:
- Docs/Design/2026-11-24-outbound-destination-inventory-design.md
- Docs/ADR/097-reviewed-outbound-destinations-and-privacy-modes.md
- Docs/superpowers/plans/2026-11-24-outbound-privacy-increment-1.md
- TASK-15647
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Specify a required CI and local check for new outbound destinations and assets during setup and runtime, with reviewed product declarations and a staged path to instance-local approvals and enforcement.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Describe required and locally runnable setup and runtime checks, their scope, and their failure criteria.
- [x] #2 Define the reviewed inventory, runtime mode semantics, coverage criteria, and privacy-preserving contact history.
- [x] #3 Record ADR assessment and obtain review of the written spec before implementation planning.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required: yes. Proposed ADR-097 adds a durable CI declaration and strict-mode release policy alongside ADR-026, ADR-030, and ADR-090. The written design records the approved order: CI/local gate first, broader coverage second, deployment-wide strict modes last. Awaiting review of the written spec before implementation planning. Bandit is not applicable to this documentation-only scope.
Written-spec review findings addressed at the requester’s direction. Increment 1 now requires the shipped browser setup wizard, transport observation for one opt-in download, and an exact local fixture allowlist. Later increments specify approval authority, ledger access and unknown ownership, and denial of new sends on open sessions after revocation or expiry. The proposed ADR records these boundaries. Work in this revision is documentation only.
Verification for the revision: reviewed all five changes against the spec and ADR, checked Markdown links and placeholders, and passed Backlog canonical-format and Git whitespace checks. No runtime code changed.
Requester approved the revised written spec on 2026-11-24 and requested the Increment 1 implementation plan. No runtime implementation was included in the design task.
Renumbered during the PR #3612 CI follow-up because dev already uses the original ID.
Renumbered after the backend change detector found that dev assigned the previous ID to the desktop chat sidebar task.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Approved design and proposed ADR-097 define the required outbound regression check and scope the later runtime modes. Increment 1 implementation is tracked independently in TASK-15647 with the linked five-stage plan.
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
