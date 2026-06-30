---
id: TASK-519
title: Confirm ACP AuthNZ RBAC ADR candidates for backfill
status: Done
labels:
- docs
- process
- adr
- acp
- authnz
- rbac
modified_files:
- Docs/ADR/inventory/2026-06-03-acp-rbac-confirmation-audit.md
- Docs/ADR/inventory/2026-06-03-decision-inventory.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Audit ACP persistence and Org/Team RBAC candidate decisions from INV-023 and INV-024 against current implementation, docs, schemas, and tests before promoting any of them into canonical ADRs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Review INV-023 and INV-024 source decisions against current implementation evidence.
- [x] #2 Classify each inventory row with concrete evidence and next action.
- [x] #3 Update the ADR decision inventory with ACP/AuthNZ/RBAC dispositions.
- [x] #4 Create a follow-up ADR backfill task only for confirmed current decisions, or document why no backfill task is safe yet.
- [x] #5 Record documentation-only verification and Bandit skip.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Review INV-023 source Docs/Plans/2026-03-08-acp-persistence-registry-expansion-design.md and INV-024 source Docs/Design/Org_Team_RBAC_Propagation_V2.md against current ACP/AuthNZ/RBAC code, module docs, schemas, migrations, and tests. Create a confirmation audit under Docs/ADR/inventory/ with concrete evidence and classify each row as current governing, superseded, stale, duplicate, or still needing owner review. Update the decision inventory with dispositions and create a follow-up ADR backfill task only for confirmed current decisions. Do not create accepted ADRs during this confirmation audit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Created `Docs/ADR/inventory/2026-06-03-acp-rbac-confirmation-audit.md`.
- Classified `INV-023` as current governing for implemented ACP persistence: shared ACP sessions/registry data in `Databases/acp_sessions.db` and per-user orchestration data in `Databases/user_databases/<id>/orchestration.db` by default, with the user DB base directory overrideable by configuration.
- Classified `INV-024` as current governing for core scoped Org/Team RBAC semantics: feature-flagged propagation, `require_active` default, admin-level denylist, JWT/API-key/default-membership scope sources, and MCP/tool permission eligibility.
- Updated `Docs/ADR/inventory/2026-06-03-decision-inventory.md` with dispositions and `TASK-520` as the bounded backfill task.
- Known caveats for `TASK-520`: do not claim unverified ACP setup-guide consolidation, missing scoped-RBAC admin mapping endpoints, missing resolver metrics/failure flag, or the older invalid-claim fallback behavior.
- PR #2249 review follow-up corrected the default orchestration DB path wording and updated `TASK-520` to require owner sign-off plus `Status: Accepted` and `Backfilled from:` ADR metadata before accepted backfill ADRs are created.
- Verification: `git diff --check`, targeted `rg` content checks, trailing-whitespace `rg`, and `git diff --cached --check` passed. Bandit is skipped because no Python/source code changed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Confirmed INV-023 and INV-024 as bounded current-governing ADR candidates, updated the ADR inventory, and created TASK-520 to backfill the accepted ADRs without overstating implementation gaps. Verification passed with `git diff --check`, targeted `rg` content checks, trailing-whitespace `rg`, and `git diff --cached --check`. This was a documentation-only task; Bandit is not applicable.
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
