---
id: TASK-514
title: Backfill Workspace/WebUI ADRs from reviewed inventory
status: Done
assignee: []
created_date: ''
updated_date: 2026-06-03 04:08
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
- [x] #1 Create accepted backfilled ADRs for INV-017, INV-018, and INV-020 using Docs/ADR/000-template.md metadata rules.
- [x] #2 Update Docs/ADR/README.md index with the new ADRs and statuses.
- [x] #3 Update high-value source docs for INV-017, INV-018, and INV-020 to link to the covering ADRs where practical.
- [x] #4 Update Docs/ADR/inventory/2026-06-03-decision-inventory.md to record the backfilled ADR links and TASK-514 output.
- [x] #5 Record verification and Bandit docs-only skip; do not backfill unresolved or duplicate rows.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Started Workspace/WebUI pilot backfill after TASK-509 owner-review defaults were approved. This child task is the TASK-511 evidence-gate pilot slice.
Verification: confirmed ADR-007/008/009 have Status: Accepted, Backfilled from metadata, and Related task: TASK-514; confirmed ADR README index entries; confirmed source docs link to covering ADRs; confirmed inventory rows INV-017/018/020 record backfilled ADR links and INV-019 remains duplicate context. git diff --check passed. Bandit skipped: documentation-only task; no Python/code paths touched.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Backfilled the Workspace/WebUI pilot slice from reviewed inventory rows INV-017, INV-018, and INV-020. Created ADR-007 for the canonical ResearchWorkspace first-slice shell, ADR-008 for workspace split-key persistence and IndexedDB offload, and ADR-009 for Quick Chat helper modes. Updated the ADR index, source docs, and inventory mappings. Bandit skipped because no Python/code paths were touched.
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
