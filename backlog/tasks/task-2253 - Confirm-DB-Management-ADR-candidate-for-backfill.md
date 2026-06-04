---
id: TASK-2253
title: Confirm DB Management ADR candidate for backfill
status: Done
labels:
- docs
- process
- adr
- db-management
modified_files:
- Docs/ADR/inventory/2026-06-04-db-management-confirmation-audit.md
- Docs/ADR/inventory/2026-06-03-decision-inventory.md
- backlog/tasks/task-2253 - Confirm-DB-Management-ADR-candidate-for-backfill.md
- backlog/tasks/task-2254 - Backfill-DB-Management-per-user-paths-and-content-backend-ADR.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Confirm whether INV-030 from Docs/ADR/inventory/2026-06-03-decision-inventory.md is current and bounded enough for ADR backfill. Verify DB_Management module docs and representative code/tests for per-user database path ownership under Databases/user_databases, content backend defaults, PostgreSQL option boundaries, caveats, and any scope that should remain inventory-only.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Create a confirmation audit under Docs/ADR/inventory/ for the DB Management ADR candidate using current origin/dev evidence.
- [x] #2 Classify INV-030 as ready for a bounded ADR backfill, needing split/owner review, or inventory-only, with explicit caveats.
- [x] #3 Update the inventory row and recommended backfill/default sections only if the confirmation result changes the tracked next action.
- [x] #4 Record verification and Bandit applicability in the Backlog task.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Created `Docs/ADR/inventory/2026-06-04-db-management-confirmation-audit.md`.
- Classified INV-030 as current governing and ready for one bounded ADR backfill, with scope limited to DB_Management per-user path ownership, SQLite default content-storage behavior, PostgreSQL shared content backend option, startup validation, and caveats.
- Updated `Docs/ADR/inventory/2026-06-03-decision-inventory.md` to record TASK-2253 confirmation and queue TASK-2254 for the ADR backfill.
- Created TASK-2254 as the follow-up ADR implementation task.
- Verification:
  - `git diff --check` passed.
  - ADR/backlog reference grep passed for TASK-2253, TASK-2254, INV-030, PostgreSQL, SQLite, AuthNZ, and Bandit references.
  - `backlog task TASK-2253 --plain` and `backlog task TASK-2254 --plain` rendered correctly.
  - Targeted DB/startup verification command ran 97 tests: 93 passed and 4 failed in unchanged current-base tests under `tldw_Server_API/tests/DB_Management/test_content_backend_cache.py`.
  - Reproduced the 4 failing current-base tests directly: analytics backend tuple/bootstrapped-target expectations and watchlists default watchlist lookup fail in unchanged code. This task changed docs/backlog only and did not modify Python code.
- Bandit: skipped because the touched scope is documentation and Backlog task metadata only; no Python/code paths were changed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Confirmed INV-030 as ready for a bounded DB Management ADR backfill. The confirmation audit now captures current evidence, caveats, and a scoped TASK-2254 follow-up for the actual ADR. Verification found unrelated current-base DB cache test failures; they are documented here and not caused by this docs-only task.
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
