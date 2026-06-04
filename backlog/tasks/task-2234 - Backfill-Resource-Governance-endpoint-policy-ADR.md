---
id: TASK-2234
title: Backfill Resource Governance endpoint policy ADR
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-04 02:39'
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
- [x] #1 Create Docs/ADR/018-resource-governance-endpoint-policy-and-route-map.md as an accepted ADR using the standard ADR template and TASK-2233 evidence.
- [x] #2 Keep accepted claims scoped to confirmed current behavior: claim-first new-endpoint auth guidance, Resource Governor applicability decision, route-map coverage ownership, DB policy-store/YAML route_map merge precedence, and request-ingress missing-policy denial.
- [x] #3 Record caveats as consequences or follow-up, including middleware request-only scope, no blanket all-endpoint coverage claim, and Redis outage fail mode configurability.
- [x] #4 Update the decision inventory after ADR creation to mark INV-028 backfilled and link the ADR/task.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Create one accepted ADR using the existing template and next ADR number. Link the source README, TASK-2233 audit, and inventory row. Keep the decision scoped to confirmed current behavior and capture caveats as consequences/follow-up, not accepted claims. Update the inventory row from confirmed candidate to backfilled accepted ADR after verification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Created Docs/ADR/018-resource-governance-endpoint-policy-and-route-map.md as the accepted Resource Governance endpoint policy ADR. Updated Docs/ADR/README.md, the decision inventory, and the Resource Governance README backlink.

Verification: git diff --check passed. ADR/link grep passed for ADR-018, the inventory row, and the Resource Governance README backlink. Targeted pytest passed: python -m pytest -q tldw_Server_API/tests/Resource_Governance/test_policy_loader_route_map_db_store.py tldw_Server_API/tests/Resource_Governance/test_policy_loader_reload_db_store.py tldw_Server_API/tests/Resource_Governance/test_middleware_simple.py tldw_Server_API/tests/Resource_Governance/test_slowapi_decorated_routes_mapped.py tldw_Server_API/tests/Resource_Governance/test_auth_route_map_coverage.py tldw_Server_API/tests/AuthNZ_Unit/test_claim_first_single_user_mode_guardrail.py (14 passed).

Bandit: not run because this slice only changed Markdown documentation and Backlog task records; no Python/source code changed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Backfilled INV-028 as accepted ADR-018, updated the ADR index and inventory, and linked the Resource Governance README to the covering ADR. Verification passed with git diff --check, ADR/link grep, and the targeted Resource Governance/AuthNZ pytest subset (14 passed). Bandit was not applicable because no Python/source code changed.
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
