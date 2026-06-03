---
id: TASK-518
title: Backfill confirmed Evaluations ADRs from audit
status: To Do
labels:
- docs
- process
- adr
- evaluations
modified_files:
- Docs/ADR/README.md
- Docs/ADR/inventory/2026-06-03-decision-inventory.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Backfill only the Evaluations decisions confirmed by TASK-517 as current governing behavior. Use the confirmation audit as source evidence and keep superseded or partially current embedded ADR text out of accepted ADRs unless it is rewritten as a replacement decision with owner approval.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Create canonical ADRs only for confirmed current Evaluations decisions from TASK-517.
- [ ] #2 Keep one decision per ADR; split resource ID conventions, deletion lifecycle, API schema shape, and evaluator integration strategy unless owner approves a narrower grouping.
- [ ] #3 Do not backfill the old SQLite-only persistence ADR, SQLite JSON TEXT ADR, or broad async-background ADR as accepted without a replacement/split review.
- [ ] #4 Update ADR index, inventory recommended actions, and relevant source links for the ADRs created.
- [ ] #5 Record docs-only verification and Bandit skip.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Use Docs/ADR/inventory/2026-06-03-evaluations-confirmation-audit.md to draft one-decision-per-ADR records for confirmed current rows INV-010, INV-011, INV-013, and INV-015. Exclude INV-009, INV-012, and INV-014 from direct accepted backfill because current evidence shows the old text is superseded or needs a split/replacement decision. Update ADR index, inventory links/statuses, and source documentation links as needed.
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
