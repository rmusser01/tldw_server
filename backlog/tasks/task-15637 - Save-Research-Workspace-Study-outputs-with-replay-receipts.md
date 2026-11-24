---
id: TASK-15637
title: Save Research Workspace Study outputs with replay receipts
status: Done
labels:
- research-workspace
- study
- api
references:
- Docs/Design/research-workspace-study-publication.md
- Docs/ADR/097-research-workspace-study-publication.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let a qualified Research Workspace workflow save generated flashcards or a quiz to the native Study owner and recover when the save response is lost. Companion client task tldw_chatbook TASK-21510.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each accepted operation commits one workspace-scoped native deck or quiz, its entries, and bounded generation provenance atomically.
- [x] #2 Concurrent and later identical requests return the same owner. Changed requests, stale sources, wrong owners, and deleted or moved outputs cannot create duplicates or disclose data.
- [x] #3 Authenticated clients can discover support, list and reopen exact generated owners, and use the ordinary Study routes to work with them.
- [x] #4 SQLite and PostgreSQL schema upgrades and applicable contract checks cover the new persistence and API.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required: yes. ADR path: Docs/ADR/097-research-workspace-study-publication.md. Native Study publication adds an authenticated workspace API and durable receipt boundary.

The Workspace Study contract publishes a canonical deck with cards or quiz with questions and cites captured, selected source associations. The native owner and its bounded receipt commit together. A matching operation retries against the same scoped owner. A changed request, stale source, moved or deleted owner, or mismatched account fails without another write. Exact native deck reads let Study reopen records outside its first page. SQLite v87 and PostgreSQL v93 install the receipt store after Document Workspace links. Both prior schema variants upgrade in place, and PostgreSQL applies forced owner RLS.

Verified 100 targeted Workspace, migration, and RLS contract tests and 19 tests on the final publication contract, including both live database backends. Ruff, touched-scope Bandit, and the generated OpenAPI checks passed. Chatbook TASK-21510 verified both kinds over HTTP with selected Media, a stub provider, responses dropped after commit, and fresh native Study reads after reconnecting.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added retry-safe native Research Studio publication for flashcards and quizzes in one authenticated Workspace transaction.
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
