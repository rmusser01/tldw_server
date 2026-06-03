---
id: TASK-518
title: Backfill confirmed Evaluations ADRs from audit
status: Done
labels:
- docs
- process
- adr
- evaluations
modified_files:
- Docs/ADR/012-evaluations-resource-id-prefixes.md
- Docs/ADR/013-evaluations-deletion-lifecycle.md
- Docs/ADR/014-evaluations-openai-compatible-schemas.md
- Docs/ADR/015-evaluations-existing-evaluator-integration.md
- Docs/ADR/README.md
- Docs/ADR/inventory/2026-06-03-decision-inventory.md
- Docs/ADR/inventory/2026-06-03-evaluations-confirmation-audit.md
- Docs/Evals/Evals-Plan-1.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Backfill only the Evaluations decisions confirmed by TASK-517 as current governing behavior. Use the confirmation audit as source evidence and keep superseded or partially current embedded ADR text out of accepted ADRs unless it is rewritten as a replacement decision with owner approval.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Create canonical ADRs only for confirmed current Evaluations decisions from TASK-517.
- [x] #2 Keep one decision per ADR; split resource ID conventions, deletion lifecycle, API schema shape, and evaluator integration strategy unless owner approves a narrower grouping.
- [x] #3 Do not backfill the old SQLite-only persistence ADR, SQLite JSON TEXT ADR, or broad async-background ADR as accepted without a replacement/split review.
- [x] #4 Update ADR index, inventory recommended actions, and relevant source links for the ADRs created.
- [x] #5 Record docs-only verification and Bandit skip.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Use Docs/ADR/inventory/2026-06-03-evaluations-confirmation-audit.md to draft one-decision-per-ADR records for confirmed current rows INV-010, INV-011, INV-013, and INV-015. Exclude INV-009, INV-012, and INV-014 from direct accepted backfill because current evidence shows the old text is superseded or needs a split/replacement decision. Update ADR index, inventory links/statuses, and source documentation links as needed.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Created ADR-012 through ADR-015 for the confirmed current Evaluations decisions from TASK-517. Left INV-009, INV-012, and INV-014 unresolved because current evidence requires replacement or split decisions before accepted ADR backfill. Verification: `git diff --check`, targeted `rg` checks for ADR links/inventory mappings, and file existence checks for ADR-012 through ADR-015. Bandit skipped because only documentation and task-record files changed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Backfilled the confirmed Evaluations ADR slice with ADR-012 through ADR-015 and updated the ADR index, decision inventory, Evals source note, and TASK-517 audit follow-up links. The stale/partial SQLite-only and broad async/background entries were not accepted as ADRs. Verification was docs-only: git diff --check plus targeted rg/link/file checks; Bandit was skipped because no Python/code paths changed.
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
