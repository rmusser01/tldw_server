---
id: TASK-13318
title: >-
  PromptStudioDatabase implements 59 methods twice plus a third delegating
  facade
status: To Do
assignee: []
created_date: '2026-09-22 04:55'
labels:
  - duplication
  - db
  - dual-backend
dependencies: []
references:
  - 'tldw_Server_API/app/core/DB_Management/PromptStudioDatabase.py:722'
  - 'tldw_Server_API/app/core/DB_Management/PromptStudioDatabase.py:3848'
  - 'tldw_Server_API/app/core/DB_Management/PromptStudioDatabase.py:7144'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A 7,426-line file where _BackendPromptStudioDatabase (722-3843, 3,121 lines) and _SQLitePromptStudioDatabase (3848-7141, 3,293 lines) implement 59 identically-named methods over parallel SQL, and PromptStudioDatabase (7144-7426) is a third *args/**kwargs delegating facade redeclaring 43 of them.

Observed drift rate already: 1 missing method (TASK-13290), 7 signature mismatches, and a retry policy on one side only. Three signature mismatches reproduced at runtime; _format_test_case differs in ARITY, (row) vs (cursor, row), so any shared helper calling it polymorphically is wrong on one backend by construction. The facade *args/**kwargs erases all of this from mypy and every IDE.

No cross-backend parity test exists over the 59 methods. Of 34 test files, 9 touch PostgreSQL and all are plumbing-level.

Destination: a prompt_studio_db/ package, one module per aggregate, ONE backend-neutral implementation over DatabaseBackend with dialect SQL isolated - following the already-shipped core/DB_Management/media_db/ split of Media_DB_v2.py. Needs design doc + ADR + staged plan.

Source: synthesis F20
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Design doc and ADR recorded before code changes
- [ ] #2 A signature-parity test covers all paired methods
- [ ] #3 Business logic exists once, with only SQL differing per backend
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
