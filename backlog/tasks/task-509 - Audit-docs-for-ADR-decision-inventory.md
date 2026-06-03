---
id: TASK-509
title: Audit docs for ADR decision inventory
status: In Progress
labels:
- docs
- process
- adr
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Audit existing decision sources and produce Docs/ADR/inventory/YYYY-MM-DD-decision-inventory.md with current, superseded, stale, duplicate, and needs-owner-review classifications.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Inventory covers Docs/Design/**, Docs/Plans/**, Docs/superpowers/specs/**, Docs/superpowers/plans/**, embedded ADRs, and module docs with decision language.
- [ ] #2 Inventory records source path, decision summary, candidate status, recommended action, and owner-review need.
- [ ] #3 No accepted ADR is created for ambiguous or contradicted decisions without owner review.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Plan created: Docs/superpowers/plans/2026-06-03-adr-follow-up-sprint-implementation-plan.md. Execution order starts here; complete TASK-509 before TASK-510/TASK-511.
Started execution in worktree .worktrees/adr-follow-up-plan on branch codex/adr-follow-up-plan. Verified .worktrees is ignored and task records are visible.
Draft inventory created at Docs/ADR/inventory/2026-06-03-decision-inventory.md. Verification: enumerated 1,646 reviewable Markdown/RST files; broad decision-language search found 1,464 candidate files across required scopes; coverage matrix records scope counts, reviewed files, skipped-file rationale, and coverage result. Ran inventory checks for coverage matrix rows, decision table rows, and classification vocabulary. Bandit skipped: documentation-only task; no Python/code paths touched. Status remains In Progress pending owner review of candidate rows.
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
