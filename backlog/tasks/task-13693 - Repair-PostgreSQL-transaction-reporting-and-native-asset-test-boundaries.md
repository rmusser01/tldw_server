---
id: TASK-13693
title: Repair PostgreSQL transaction reporting and native asset test boundaries
status: Done
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve PR #3229 db-privileges failures at their root without weakening ownership, outermost-transaction, exact bytes, quota or fencing guarantees.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Transaction reporting preserves SQLite and reflects PostgreSQL raw status plus managed depths with fail-closed unknown status
- [x] #2 Native asset fixture reads, post-commit fences and redacted constraint expectations match backend contracts
- [x] #3 Both backend regressions, security validation and independent review pass
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required: no. Existing backend contracts (Docs/ADR/020-database-backend-selection.md; verify exact path in plan) remain unchanged; ADR-059 governs task tracking. Source-incarnation ADR-071 remains proposed and identity guards are unchanged. Linked PR https://github.com/rmusser01/tldw_server/pull/3229. Separate historical migration fixture task TASK-13694.
ADR path correction: governing backend choice is Docs/ADR/020-db-management-per-user-paths-and-content-backend.md; ADR-059 tracking, ADR-071 remains Proposed. Plan Docs/Design/IMPLEMENTATION_PLAN_chat_postgres_ci_13693.md. Status table RED1fail20pass to GREEN21; independent A found legacy empty-managed-depth omission, new case RED1fail23pass then final GREEN24. A and B now clear source/fixture supplement; actual backend suite running. Ruff 0 baseline/current across6files. Bandit150baseline158current: only8 additional ordinary pytest assertions (B101), zero new non-test-assert findings, no scan errors. No guard/ownership/quota/schema/identity changes.
Qualification complete: seven-file asset/read/role/historical/error/status suite266passed1intentionalSQLite-onlyskip (PostgreSQL row-lock case),12warnings,959.27s. Final property legacy-depth/status/actualemptytransaction/operationownership/PGFTS/bootstrap qualification152passed0skips,30warnings,726.49s. All actual PostgreSQL fixtures required. Final24status cases pass; both independent reviewers A/B CLEAR after correcting A's managed legacy-depth finding. All configured source hooks and Python syntax pass. Ruff0 across all6changedPythonfiles; newtest and3cleanfixtures Black pass; legacy source/historical formatter deltas exactly identical to baseline. Bandit150baseline158current, no scan errors or new production/security finding:8 additions are ordinary pytest asserts. Test output artifacts contain full passing pytest summaries; shell exit-capture wrapper used zsh readonly status and was corrected for subsequent commands, without rerunning successful suites.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Corrected shared PostgreSQL transaction reporting to include captured operation or supported legacy depth, backend depth and raw driver status; only proven IDLE is false. SQLite/bootstrap ownership behavior and native outermost guards unchanged. Fixture snapshots no longer introduce implicit transactions, reclaim events signal after commit, and invalid roles assert typed redacted backend errors. Qualification266passed/1expectedSQLite-onlyskip plus final152passed; two independent reviews clear; no new security issue.
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
