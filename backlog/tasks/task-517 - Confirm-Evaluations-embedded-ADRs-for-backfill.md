---
id: TASK-517
title: Confirm Evaluations embedded ADRs for backfill
status: Done
labels:
- docs
- process
- adr
- evaluations
modified_files:
- Docs/ADR/inventory/2026-06-03-evaluations-confirmation-audit.md
- Docs/ADR/inventory/2026-06-03-decision-inventory.md
- backlog/tasks/task-518 - Backfill-confirmed-Evaluations-ADRs-from-audit.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Audit embedded ADR-style decisions in Docs/Evals/Evals-Plan-1.md against current Evaluations code, docs, schemas, and tests before promoting any evaluations decision into canonical ADRs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Review Docs/Evals/Evals-Plan-1.md embedded ADRs INV-009 through INV-015 against current Evaluations implementation evidence.
- [ ] #2 Classify each inventory row as current governing, superseded, stale, duplicate, or still needing owner review with concrete evidence.
- [ ] #3 Update the ADR decision inventory with concrete Evaluations dispositions and next actions.
- [ ] #4 Create a follow-up ADR backfill task only for confirmed current Evaluations decisions, or document why no backfill task is safe yet.
- [ ] #5 Record documentation-only verification and Bandit skip.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Create an Evaluations confirmation audit under Docs/ADR/inventory/. Review current code/docs around Evaluations_DB, unified evaluation schemas, CRUD/dataset endpoints, eval runner, unified service, recipe/jobs paths, and tests. Update Docs/ADR/inventory/2026-06-03-decision-inventory.md rows INV-009 through INV-015 with concrete dispositions. Do not create accepted ADRs in this confirmation audit unless the row is unambiguously current and owner-approved; prefer a separate follow-up backfill task for confirmed current decisions.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added the Evaluations ADR confirmation audit, updated the decision inventory with concrete dispositions for INV-009 through INV-015, and created TASK-518 for the confirmed current backfill candidates. No canonical ADRs were created in this audit. Bandit was skipped as docs-only; verification was recorded with git diff --check and targeted rg checks.
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
