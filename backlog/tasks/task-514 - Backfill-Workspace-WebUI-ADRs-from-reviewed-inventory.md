---
id: TASK-514
title: Backfill Workspace/WebUI ADRs from reviewed inventory
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-06-03 04:08'
labels:
  - docs
  - process
  - adr
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Backfill only the owner-approved Workspace/WebUI inventory rows from Docs/ADR/inventory/2026-06-03-decision-inventory.md.

Scope:
- INV-017: Docs/Design/Workspace_Canonical_Model_Decision_2026_05.md
- INV-018: Docs/Design/Workspace_Persistence_Architecture.md
- INV-020: Docs/Design/Quick_Chat_Docs_Assistant.md
- Context only: INV-019 from Docs/superpowers/specs/2026-05-06-tldw-product-roadmap-design.md and Docs/superpowers/plans/2026-05-06-tldw-product-roadmap-first-slice-implementation-plan.md

Expected outputs:
- Accepted backfilled ADR for the first-slice canonical workspace shell/route boundary.
- Accepted backfilled ADR for workspace split-key persistence and IndexedDB offload.
- Accepted backfilled ADR for Quick Chat helper modes and docs-scoped retrieval/browse guidance.

Prerequisites:
- TASK-509 owner-review defaults approved by requester.
- Do not convert stale, superseded, duplicate, ambiguous, or unresolved rows.

Backfilled ADR output rules:
- Use Docs/ADR/000-template.md.
- Use Status: Accepted only for still-governing owner-approved decisions.
- Set Backfilled from: <source path>.
- Set Related task to this child task ID.
- Keep stale, superseded, duplicate, and ambiguous decisions classified in the inventory; do not silently convert them.

Source-doc link policy:
- Where practical, update high-value source docs to link to the covering or superseding ADR.
- Do not churn low-value historical docs solely to add links.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Create accepted backfilled ADRs for INV-017, INV-018, and INV-020 using Docs/ADR/000-template.md metadata rules.
- [ ] #2 Update Docs/ADR/README.md index with the new ADRs and statuses.
- [ ] #3 Update high-value source docs for INV-017, INV-018, and INV-020 to link to the covering ADRs where practical.
- [ ] #4 Update Docs/ADR/inventory/2026-06-03-decision-inventory.md to record the backfilled ADR links and TASK-514 output.
- [ ] #5 Record verification and Bandit docs-only skip; do not backfill unresolved or duplicate rows.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Started Workspace/WebUI pilot backfill after TASK-509 owner-review defaults were approved. This child task is the TASK-511 evidence-gate pilot slice.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->
<!-- SECTION:FINAL_SUMMARY:END -->

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
