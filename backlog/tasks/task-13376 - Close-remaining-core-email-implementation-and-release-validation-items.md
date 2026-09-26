---
id: TASK-13376
title: Close remaining core email implementation and release validation items
status: In Progress
assignee: []
created_date: '2026-09-26 16:01'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
User explicitly requested handling all remaining items: ingestion metrics, sensitive logging, attachment MIME policies, sustained50msg/s ingestion,1M search benchmark SQLite then PG, target deployment/parity/flags, documentation and owner-ready evidence. Synthetic data only, no personal Gmail or model requests. Coordinate reviewable child tasks and keep performance runs serialized.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Metrics, logging and attachment policy PRD gaps are implemented or verified with tests
- [ ] #2 Sustained ingestion and1M search measured on SQLite then PostgreSQL and failures addressed
- [ ] #3 Chosen-environment flag,endpoint,parity and delegation checks complete with rollback evidence
- [ ] #4 PRD and release evidence reconciled, touched-code lint/security clean, resources cleaned and commits recorded
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
