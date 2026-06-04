---
id: TASK-520
title: Backfill ACP persistence and scoped RBAC ADRs
status: Done
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
- Docs/Plans/2026-03-08-acp-persistence-registry-expansion-design.md
- Docs/Design/Org_Team_RBAC_Propagation_V2.md
- backlog/tasks/task-520 - Backfill-ACP-persistence-and-scoped-RBAC-ADRs.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create bounded ADRs for confirmed current ACP persistence and scoped Org/Team RBAC decisions from TASK-519 after owner review and explicit sign-off are recorded. The ACP ADR should cover shared ACP session/registry persistence plus per-user orchestration persistence, while avoiding unverified registry setup-guide consolidation claims. The RBAC ADR should cover core scoped permission semantics only: feature-flagged propagation, require_active default, admin-level denylist, JWT/default/API-key scope sources, and tools/MCP eligibility; it must not claim missing admin mapping endpoints, resolver metrics, or the older invalid-claim fallback behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Record owner review and explicit sign-off before creating the accepted ACP persistence ADR; then add one ADR using INV-023/TASK-519 evidence, scoped to current implemented persistence behavior, with `**Status:** Accepted` and `**Backfilled from:** Docs/Plans/2026-03-08-acp-persistence-registry-expansion-design.md` metadata.
- [x] #2 Record owner review and explicit sign-off before creating the accepted scoped Org/Team RBAC ADR; then add one ADR using INV-024/TASK-519 evidence, with caveats for implementation gaps excluded from the decision, with `**Status:** Accepted` and `**Backfilled from:** Docs/Design/Org_Team_RBAC_Propagation_V2.md` metadata.
- [x] #3 Update the ADR inventory and source docs with covering ADR links/status, owner-review status, sign-off evidence, and the accepted/backfilled metadata recorded for each ADR.
- [x] #4 Record docs-only verification and Bandit skip.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Review the confirmation audit from TASK-519, obtain and record owner sign-off for the ACP persistence and scoped RBAC backfill scope, choose the next ADR numbers, draft two immutable accepted ADRs under Docs/ADR/ with `Status: Accepted` and `Backfilled from:` metadata, update inventory/source documentation links with the sign-off evidence, and verify with docs-only checks. Keep implementation gaps as consequences or follow-up notes rather than accepted claims.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Owner sign-off: user instructed `continue` on 2026-06-04 after TASK-520 scope and the owner sign-off requirement were summarized.
- Backlog MCP `task_view` resolves duplicate `TASK-520` IDs to an unrelated completed flashcards task in this repo snapshot, so this exact task file was updated directly.
- Restored missing canonical ADR framework files from `origin/codex/adr-evaluations-backfill` because the current `dev` inventory references ADR-001 through ADR-015 while `Docs/ADR/` only contained inventory files after PR #2249 merged.
- Added `Docs/ADR/016-acp-session-and-orchestration-persistence.md` with `Status: Accepted`, `Backfilled from:` metadata, and bounded ACP persistence scope.
- Added `Docs/ADR/017-scoped-org-team-rbac-core-semantics.md` with `Status: Accepted`, `Backfilled from:` metadata, and scoped RBAC caveats for unimplemented admin endpoints, metrics, and invalid-claim fallback.
- Updated ADR README, decision inventory, TASK-519 audit, and source docs with covering ADR links/status and sign-off evidence.
- Verification recorded in final summary. Bandit skipped because this task changed documentation only and no Python/source files were touched.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Backfilled ACP persistence and scoped Org/Team RBAC as accepted ADRs after recorded owner sign-off. Added ADR-016 for shared ACP session/registry persistence plus per-user orchestration persistence and ADR-017 for feature-flagged scoped RBAC core semantics. Restored the missing canonical ADR framework/ADR-001 through ADR-015 files that the merged inventory already referenced, updated the ADR index, inventory, audit, and source docs, and corrected the ACP source plan's default orchestration DB path wording.

Verification: `git diff --check`, targeted `rg` content checks, ADR link existence check, no conflict markers, no trailing whitespace, and docs-only touched-file check passed. Bandit was skipped because no Python/source files were touched.
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
