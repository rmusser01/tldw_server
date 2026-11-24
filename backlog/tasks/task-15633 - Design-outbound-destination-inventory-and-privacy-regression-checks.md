---
id: TASK-15633
title: Design outbound destination inventory and privacy regression checks
status: In Progress
labels:
- security
- privacy
- ci
- design
references:
- Docs/Design/2026-11-24-outbound-destination-inventory-design.md
- Docs/ADR/097-reviewed-outbound-destinations-and-privacy-modes.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Specify a required CI and local check for new outbound destinations and assets during setup and runtime, with reviewed product declarations and a staged path to instance-local approvals and enforcement.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Describe required and locally runnable setup and runtime checks, their scope, and their failure criteria.
- [x] #2 Define the reviewed inventory, runtime mode semantics, coverage criteria, and privacy-preserving contact history.
- [ ] #3 Record ADR assessment and obtain review of the written spec before implementation planning.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required: yes. Proposed ADR-097 adds a durable CI declaration and strict-mode release policy alongside ADR-026, ADR-030, and ADR-090. The written design records the approved order: CI/local gate first, broader coverage second, deployment-wide strict modes last. Awaiting review of the written spec before implementation planning. Bandit is not applicable to this documentation-only scope.
Written-spec review findings addressed at the requester’s direction. Increment 1 now requires the shipped browser setup wizard, transport observation for one opt-in download, and an exact local fixture allowlist. Later increments specify approval authority, ledger access and unknown ownership, and denial of new sends on open sessions after revocation or expiry. The proposed ADR records these boundaries. Work in this revision is documentation only.
Verification for the revision: reviewed all five changes against the spec and ADR, checked Markdown links and placeholders, and passed Backlog canonical-format and Git whitespace checks. No runtime code changed.
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
