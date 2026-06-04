---
id: TASK-2234
title: Backfill Resource Governance endpoint policy ADR
status: To Do
assignee: []
created_date: ''
updated_date: '2026-06-04 01:53'
labels:
  - docs
  - process
  - adr
  - resource-governance
  - security
dependencies:
  - TASK-2233
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Backfill a single accepted ADR from confirmed INV-028/TASK-2233 evidence. Scope the ADR to Resource Governance new-endpoint policy: claim-first auth gate expectations, deciding Resource Governor applicability for latency/cost-sensitive endpoints, route-map ownership for ingress coverage, DB policy-store plus YAML route_map merge behavior, and request-ingress fail-closed behavior when a route_map references a missing request policy. Exclude claims that all existing endpoints are already covered, non-request categories are middleware-enforced, or Redis outage handling is globally fail-closed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Create Docs/ADR/018-resource-governance-endpoint-policy-and-route-map.md as an accepted ADR using the standard ADR template and TASK-2233 evidence.
- [ ] #2 Keep accepted claims scoped to confirmed current behavior: claim-first new-endpoint auth guidance, Resource Governor applicability decision, route-map coverage ownership, DB policy-store/YAML route_map merge precedence, and request-ingress missing-policy denial.
- [ ] #3 Record caveats as consequences or follow-up, including middleware request-only scope, no blanket all-endpoint coverage claim, and Redis outage fail mode configurability.
- [ ] #4 Update the decision inventory after ADR creation to mark INV-028 backfilled and link the ADR/task.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Create one accepted ADR using the existing template and next ADR number. Link the source README, TASK-2233 audit, and inventory row. Keep the decision scoped to confirmed current behavior and capture caveats as consequences/follow-up, not accepted claims. Update the inventory row from confirmed candidate to backfilled accepted ADR after verification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

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
