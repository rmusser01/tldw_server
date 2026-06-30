---
id: TASK-2233
title: Confirm Resource Governance ADR candidate for backfill
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-04 01:56'
labels:
  - docs
  - process
  - adr
  - resource-governance
  - security
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Audit INV-028 from the ADR decision inventory against current Resource Governance docs, implementation, route-map/policy-store behavior, endpoint usage, and tests before promoting it to an accepted ADR. Confirm whether claim-first auth guidance, latency/cost policy selection, route-map coverage, DB/file policy merge behavior, and fail-closed missing-policy behavior are current governing behavior. Create a confirmation audit under Docs/ADR/inventory/, update the decision inventory with the bounded disposition, and create a follow-up backfill task only if the decision is current enough for accepted ADR backfill. Do not create accepted ADRs during this confirmation audit.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Confirm INV-028 against current Resource Governance docs, implementation, route-map/policy-store behavior, endpoint usage, and tests with concrete file-path evidence.
- [x] #2 Create Docs/ADR/inventory/2026-06-04-resource-governance-confirmation-audit.md with disposition, evidence, caveats, and next action.
- [x] #3 Update Docs/ADR/inventory/2026-06-03-decision-inventory.md so INV-028 and the provider/integration slice reflect the confirmation result.
- [x] #4 Create a bounded follow-up Backlog task only if INV-028 is confirmed current enough for accepted ADR backfill.
- [x] #5 Record docs-only verification and Bandit applicability in TASK-2233.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Review INV-028 source docs and current Resource Governance code paths, policy-store/route-map behavior, endpoint integration points, and focused tests. Record concrete evidence and caveats in a new confirmation audit. Update the decision inventory row and recommended slice status. If current, create a bounded follow-up Backlog task for accepted ADR backfill; otherwise leave inventory-only with rationale. Run docs-only verification and record Bandit skip if no Python/source files are touched.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Confirmed INV-028 as current for a bounded Resource Governance ADR: claim-first new-endpoint guidance, Resource Governor applicability decisions for latency/cost-sensitive endpoints, route-map ownership, DB policy-store plus YAML route_map merge precedence, and request-ingress denial when a route_map resolves to a missing request policy.

Created Docs/ADR/inventory/2026-06-04-resource-governance-confirmation-audit.md and updated Docs/ADR/inventory/2026-06-03-decision-inventory.md. Created TASK-2234 for the accepted ADR backfill.

Verification: git diff --check passed. Targeted pytest passed: python -m pytest -q tldw_Server_API/tests/Resource_Governance/test_policy_loader_route_map_db_store.py tldw_Server_API/tests/Resource_Governance/test_policy_loader_reload_db_store.py tldw_Server_API/tests/Resource_Governance/test_middleware_simple.py tldw_Server_API/tests/Resource_Governance/test_slowapi_decorated_routes_mapped.py tldw_Server_API/tests/Resource_Governance/test_auth_route_map_coverage.py tldw_Server_API/tests/AuthNZ_Unit/test_claim_first_single_user_mode_guardrail.py (14 passed).

Bandit: not run because this slice touched documentation and Backlog task records only; no Python/source code changed.

PR review follow-up: normalized Backlog FINAL_SUMMARY markers after review feedback.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Confirmed INV-028 as current enough for a bounded Resource Governance ADR backfill, documented evidence and caveats in the confirmation audit, updated the decision inventory, and created TASK-2234 for the accepted ADR follow-up. Verification passed with git diff --check and the targeted Resource Governance/AuthNZ pytest subset (14 passed). Bandit was not applicable because no Python/source code changed.
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
