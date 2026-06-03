---
id: TASK-520
title: Backfill ACP persistence and scoped RBAC ADRs
status: To Do
labels:
- docs
- process
- adr
- acp
- authnz
- rbac
modified_files:
- Docs/ADR/
- Docs/ADR/inventory/2026-06-03-decision-inventory.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create bounded ADRs for confirmed current ACP persistence and scoped Org/Team RBAC decisions from TASK-519 after owner review and explicit sign-off are recorded. The ACP ADR should cover shared ACP session/registry persistence plus per-user orchestration persistence, while avoiding unverified registry setup-guide consolidation claims. The RBAC ADR should cover core scoped permission semantics only: feature-flagged propagation, require_active default, admin-level denylist, JWT/default/API-key scope sources, and tools/MCP eligibility; it must not claim missing admin mapping endpoints, resolver metrics, or the older invalid-claim fallback behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Record owner review and explicit sign-off before creating the accepted ACP persistence ADR; then add one ADR using INV-023/TASK-519 evidence, scoped to current implemented persistence behavior, with `**Status:** Accepted` and `**Backfilled from:** Docs/Plans/2026-03-08-acp-persistence-registry-expansion-design.md` metadata.
- [ ] #2 Record owner review and explicit sign-off before creating the accepted scoped Org/Team RBAC ADR; then add one ADR using INV-024/TASK-519 evidence, with caveats for implementation gaps excluded from the decision, with `**Status:** Accepted` and `**Backfilled from:** Docs/Design/Org_Team_RBAC_Propagation_V2.md` metadata.
- [ ] #3 Update the ADR inventory and source docs with covering ADR links/status, owner-review status, sign-off evidence, and the accepted/backfilled metadata recorded for each ADR.
- [ ] #4 Record docs-only verification and Bandit skip.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Review the confirmation audit from TASK-519, obtain and record owner sign-off for the ACP persistence and scoped RBAC backfill scope, choose the next ADR numbers, draft two immutable accepted ADRs under Docs/ADR/ with `Status: Accepted` and `Backfilled from:` metadata, update inventory/source documentation links with the sign-off evidence, and verify with docs-only checks. Keep implementation gaps as consequences or follow-up notes rather than accepted claims.
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
