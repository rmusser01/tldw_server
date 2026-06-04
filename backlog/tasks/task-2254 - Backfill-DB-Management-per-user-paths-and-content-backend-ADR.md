---
id: TASK-2254
title: Backfill DB Management per-user paths and content backend ADR
status: Done
dependencies:
- TASK-2253
labels:
- docs
- process
- adr
- db-management
modified_files:
- Docs/ADR/020-db-management-per-user-paths-and-content-backend.md
- Docs/ADR/README.md
- Docs/ADR/inventory/2026-06-03-decision-inventory.md
- tldw_Server_API/app/core/DB_Management/README.md
- backlog/tasks/task-2254 - Backfill-DB-Management-per-user-paths-and-content-backend-ADR.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Backfill a bounded DB Management ADR from TASK-2253 evidence. Scope the accepted decision to DB_Management ownership of per-user database paths under USER_DB_BASE_DIR (defaulting to Databases/user_databases), SQLite as the default per-user content storage mode, PostgreSQL as the shared content backend option with startup validation, and explicit caveats for AuthNZ/users DB separation, explicit SQLite path overrides, test fallback paths, legacy aliases, and non-universal PostgreSQL support across every DB family.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Create the next accepted ADR under Docs/ADR/ using the standard ADR template and TASK-2253 evidence.
- [x] #2 Keep accepted claims scoped to per-user database path ownership, SQLite default behavior, PostgreSQL content backend option, startup validation, and documented caveats.
- [x] #3 Update Docs/ADR/README.md, INV-030 inventory row, and relevant DB_Management README backlink after ADR creation.
- [x] #4 Record verification and Bandit applicability in this task.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Created `Docs/ADR/020-db-management-per-user-paths-and-content-backend.md` as the accepted backfill ADR for INV-030.
- Scoped the ADR to DB_Management per-user path ownership, SQLite default content mode, explicit PostgreSQL content mode, startup validation, and caveats from TASK-2253.
- Updated `Docs/ADR/README.md`, the INV-030 inventory row/default disposition, and `tldw_Server_API/app/core/DB_Management/README.md`.
- Verification:
  - `git diff --check` passed.
  - `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Docs/test_docs_index_path_hygiene_script.py tldw_Server_API/tests/Docs/test_readme_docs_path_hygiene_script.py tldw_Server_API/tests/Docs/test_top_guides_docs_path_hygiene_script.py` passed: 3 passed.
  - ADR/reference grep confirmed ADR-020 index, inventory, and DB_Management README references.
  - Stale queued-phrasing grep found no matches.
- Bandit: not applicable because this task touched only Markdown documentation and Backlog task metadata, not Python/code files.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Backfilled INV-030 as ADR-020 for the bounded DB Management path/content-backend decision. The ADR records DatabasePaths per-user ownership, SQLite default content mode, explicit PostgreSQL content mode with startup validation, and caveats for explicit SQLite overrides, test fallback paths, deprecated aliases, AuthNZ/users DB separation, historical compatibility paths, and non-universal PostgreSQL support. Updated the ADR index, inventory row/defaults, and DB_Management README backlink. Verification: git diff --check passed; docs path hygiene pytest target passed 3 tests; ADR/reference greps confirmed ADR-020 links and no stale queued phrasing. Bandit: not applicable because the touched scope is Markdown documentation and Backlog task metadata only.
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
