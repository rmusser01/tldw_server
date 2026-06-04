---
id: TASK-2250
title: Backfill DB Management per-user paths and content backend ADR
status: To Do
dependencies:
- TASK-2249
labels:
- docs
- process
- adr
- db-management
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Backfill a bounded DB Management ADR from TASK-2249 evidence. Scope the accepted decision to DB_Management ownership of per-user database paths under USER_DB_BASE_DIR/default Databases/user_databases, SQLite as the default per-user content storage mode, PostgreSQL as the shared content backend option with startup validation, and explicit caveats for AuthNZ/users DB separation, explicit SQLite path overrides, test fallback paths, legacy aliases, and non-universal PostgreSQL support across every DB family.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Create the next accepted ADR under Docs/ADR/ using the standard ADR template and TASK-2249 evidence.
- [ ] #2 Keep accepted claims scoped to per-user database path ownership, SQLite default behavior, PostgreSQL content backend option, startup validation, and documented caveats.
- [ ] #3 Update Docs/ADR/README.md, INV-030 inventory row, and relevant DB_Management README backlink after ADR creation.
- [ ] #4 Record verification and Bandit applicability in this task.
<!-- AC:END -->

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
