---
id: TASK-2273
title: Backfill Data Tables backend ADR
status: To Do
assignee: []
created_date: '2026-06-07 02:54'
labels:
  - docs
  - process
  - adr
  - data-tables
dependencies:
  - TASK-2272
references:
  - Docs/ADR/inventory/2026-06-07-data-tables-confirmation-audit.md
  - Docs/Design/Data_Tables_Backend.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Backfill a bounded Data Tables ADR from TASK-2272 evidence. Scope the accepted decision to Media DB ownership for metadata, source snapshots, columns, and rows; UUID public table identity with numeric job ID caveat; Jobs-backed generation/regeneration with the Data_Tables worker; stored source snapshots for regeneration/RAG reproducibility; and server-side exports through direct adapter rendering or File Artifacts. Keep frontend editing, all-source ownership proof, File Artifacts storage internals, and synchronous wait/direct-download caveats explicit unless separately confirmed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Create the next accepted ADR under Docs/ADR using the standard template and TASK-2272 confirmation evidence.
- [ ] #2 Keep claims scoped to Data Tables backend storage, Jobs generation/regeneration, source snapshots, exports, and table UUID identity with explicit caveats.
- [ ] #3 Update Docs/ADR/README.md, INV-025 inventory disposition, and the Data_Tables README/source doc backlinks to the new ADR.
- [ ] #4 Record verification and Bandit applicability in this task.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
