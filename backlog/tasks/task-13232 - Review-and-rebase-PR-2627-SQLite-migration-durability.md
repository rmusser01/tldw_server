---
id: TASK-13232
title: Review and rebase PR 2627 SQLite migration durability
status: Done
assignee: []
created_date: '2026-09-10 00:43'
updated_date: '2026-09-10 01:56'
labels:
  - db
  - migrations
  - review
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/2627'
documentation:
  - Docs/Database_Migrations.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Complete the migration durability work formerly tracked as TASK-12142, resolve current-dev compatibility and review findings, and assess merge usefulness.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Packaged SQLite upgrades retain original checksums and pass on current dev.
- [x] #2 Migration SQL, success ledger, and schema version commit atomically; hidden transaction control cannot bypass rollback.
- [x] #3 Unsupported legacy Media DB versions fail with actionable data-preserving recovery guidance.
- [x] #4 Focused tests, security checks, independent review, and PR readiness are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
# PR 2627 migration durability review

Tracking: TASK-13232, created through Backlog CLI with the old PR task history because TASK-12142 collides with a newer dev task.

## Stage 1: Rebase and applicability
**Goal**: Rebase onto dev 40345571a2cfc8b3a8893545836097d27e4ee86c and review current callers.
**Success Criteria**: Preserve dev idempotent SQL metadata and classification fixes; identify current regressions.
**Tests**: Focused migration/bootstrap baseline.
**Status**: Complete (109 passed, 11 failed on packaged transaction wrappers, 7 skipped).

## Stage 2: Address durability and compatibility findings
**Goal**: Retain shipped SQL/checksums, unwrap outer legacy transactions, enforce body transaction ownership with SQLite authorizer, and document legacy recovery.
**Success Criteria**: Current packaged upgrades run; commented transaction control cannot commit partial DDL; version and ledger failures roll back SQL.
**Tests**: Real SQLite regressions for wrappers, transaction comments, triggers/literals, bookkeeping failure and retry; existing v22-to-current integration tests.
**Status**: Complete

## Stage 3: Verify and assess merge readiness
**Goal**: Validate final diff and prepare the reviewed branch for commit and an exact force-with-lease push; fresh remote CI remains a merge gate.
**Success Criteria**: Focused tests, syntax/lint/security checks, independent review; report current CI and human change-summary gate.
**Tests**: Migration/bootstrap suite, Bandit touched production scope, pre-commit checks, git diff --check, remote SHA/base verification.
**Status**: Complete

## Stage 4: Qodo follow-up on latest dev
**Goal**: Resolve all six Qodo findings while preserving original SQL integrity.
**Success Criteria**: BOM-prefixed wrapped files execute, source/checksum remain unchanged, new tests meet repository conventions, and public SQL effects replace parser call-count assertions.
**Tests**: Three BOM cases fail before the execution-only normalization and pass after; unit selection, expanded migration/CLI/bootstrap/backup suite, lint/security and independent review.
**Status**: Complete (171 passed, 7 PostgreSQL-unavailable skips; final remote review/CI are merge gates).

## Stage 5: Boundary compatibility and legacy failure cleanup
**Goal**: Accept comments on legacy wrappers and FK boundaries while retaining transaction ownership.
**Success Criteria**: SQL comments outside quoted tokens normalize for classification; body SQL and checksum source stay original; modified internal test participates in unit selection.
**Tests**: Eight public file migration cases for comment locations with commit/rollback, seven lexer output cases, expanded migration/CLI/bootstrap/backup suite and independent SQLite comparisons.
**Status**: Complete (186 passed, 7 PostgreSQL-unavailable skips; after unrelated latest-dev rebase, 39 selected unit tests pass).
Legacy rejection follow-up: use the existing pool invalidation API to close/remove only the rejected startup thread connection. Real factory-backed regressions verify empty pool stats, closed old handle, fresh usable acquisition, and preserved data/version. Complete: three failures before the fix; four legacy tests and 186 expanded tests pass afterward, with seven PostgreSQL-unavailable skips.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implementation complete on branch codex/audit-db-migration-durability-2026-07-04 from origin/dev fd5c152b065c408e4e8ee5f08da41589f21cb7f5. Changes: DatabaseMigrator now splits SQL statements with sqlite3.complete_statement and executes migration SQL, success ledger updates, and schema_version updates inside one owned BEGIN IMMEDIATE transaction; failed multi-statement migrations roll back partial DDL/DML before recording a failed schema_migrations row; migration scripts with transaction-control statements are rejected because the migrator owns transactions; v23 transcript history migration no longer embeds BEGIN/COMMIT; Media DB bootstrap now explicitly rejects file-backed schema versions below v22 with a backup/export/rebuild remediation message instead of attempting an impossible packaged migration chain. Validation: focused regressions passed (2 passed), broader DB migration/bootstrap validation passed (97 passed, 201 warnings), full Media DB schema bootstrap passed within that set, Bandit on touched Python production files reported 0 findings, and git diff --check passed.
Draft PR opened against dev: https://github.com/rmusser01/tldw_server/pull/2627. This PR is intentionally draft pending the required human-written Change summary for AI-generated PRs.
2026-07-04 review follow-up before the later rebase from origin/dev fd5c152b065c408e4e8ee5f08da41589f21cb7f5: addressed PR #2627 comments on DB migration statement handling. Added red/green regressions for extracting SQLite function-style PRAGMA foreign_keys(OFF/ON) statements and for limiting sqlite3.complete_statement calls to semicolon statement boundaries. Updated DatabaseMigrator to recognize both assignment-style and function-style foreign_keys pragmas and to avoid O(N^2) per-character completeness checks. Verified: two new regressions failed before production change and passed after; migration-focused suite passed (99 passed, 211 warnings); Bandit over tldw_Server_API/app/core/DB_Management/db_migration.py reported 0 findings; git diff --check passed. Reviewed the pathlib.Path annotation comment and found no code issue because the test imports pathlib as a module; will reply with technical no-change rationale.
Post-rebase validation on current origin/dev 4c1ca5d8358bff2a5a7fb5c75d60d1bd6728e702: rebased codex/audit-db-migration-durability-2026-07-04 so merge-base equals current origin/dev. Fresh verification after rebase: migration-focused DB suite passed (99 passed, 211 warnings); Bandit over tldw_Server_API/app/core/DB_Management/db_migration.py reported 0 findings in /tmp/bandit_db_migration_review_rebased_dev.json; git diff --check HEAD~1..HEAD passed.
2026-07-04 current-dev refresh: rebased `codex/audit-db-migration-durability-2026-07-04` onto `origin/dev` `09d9ec901e1d4548f7924f1c6bcefa963fadd9bd`; merge-base matches `origin/dev`. Validation: `python -m pytest tldw_Server_API/tests/DB_Management/test_db_migration_planning.py tldw_Server_API/tests/DB_Management/test_db_migration_loader.py tldw_Server_API/tests/DB_Management/test_db_migration_path_validation.py tldw_Server_API/tests/DB_Management/test_db_migration_verification.py tldw_Server_API/tests/DB_Management/test_migration_tools.py tldw_Server_API/tests/DB_Management/test_migration_cli_integration.py tldw_Server_API/tests/DB_Management/test_media_db_migration_missing_scripts_error.py tldw_Server_API/tests/DB_Management/test_media_db_schema_bootstrap.py -q` passed with 99 tests; `python -m bandit -r tldw_Server_API/app/core/DB_Management/db_migration.py -f json -o /tmp/bandit_db_migration_origin_dev_09d9ec.json` reported 0 findings over 766 LOC; `git diff --check HEAD~1..HEAD` passed.
2026-07-04 latest-dev refresh: rebased and validated PR #2627 on origin/dev 6b727b221e55646eba663a03571e38302f7fafc2. Tested head 08981bb92e48. Verification: focused DB migration pytest suite => 99 passed, 211 warnings; bandit -r tldw_Server_API/app/core/DB_Management/db_migration.py => 0 findings over 766 LOC; git diff --check HEAD~1..HEAD => clean.

2026-09-09: Reopened PR #2627 review on dev 40345571a2cfc8b3a8893545836097d27e4ee86c. Original TASK-12142 identifier collides with an unrelated Research Workspace task now on dev; this replacement retains the migration task history. Rebase completed in isolated worktree. Baseline: 109 passed, 11 failed (packaged wrappers 024-026), 7 skipped. Added regressions: 8 failed, 9 passed before fix. Fix preserves shipped migration checksums, unwraps outer legacy transactions, and uses SQLite authorizer to deny body transaction controls and foreign_keys writes. Plan: IMPLEMENTATION_PLAN_pr2627_migration_review.md.

2026-09-09 final review: retained dev SQL-idempotence behavior, restored all shipped migration SQL bytes (zero diff vs dev), unwrapped legacy outer BEGIN/COMMIT in the runner, and denied body transaction/savepoint/FK-setting operations through SQLite authorizer. Added referenced recovery guidance. Corrected one stale internal test only after reproducing its identical failure against unchanged origin/dev. Independent review found no remaining substantive issues, including a second review of the stale-test adjustment.
Verification: source the project .venv, then python -m pytest tldw_Server_API/tests/DB_Management/test_db_migration_planning.py tldw_Server_API/tests/DB_Management/test_db_migration_loader.py tldw_Server_API/tests/DB_Management/test_db_migration_path_validation.py tldw_Server_API/tests/DB_Management/test_db_migration_verification.py tldw_Server_API/tests/DB_Management/test_migration_tools.py tldw_Server_API/tests/DB_Management/test_migration_cli_integration.py tldw_Server_API/tests/DB_Management/test_media_db_migration_missing_scripts_error.py tldw_Server_API/tests/DB_Management/test_media_db_schema_bootstrap.py tldw_Server_API/tests/DB_Management/test_media_db_bootstrap_lifecycle_ops.py tldw_Server_API/tests/DB_Management/test_db_backup_integrity.py tldw_Server_API/app/core/DB_Management/test_migrations.py -q -rs => 167 passed, 7 skipped (PostgreSQL unavailable), 4 warnings. Regression red/green: 8 failed/9 passed before fix, 17 passed after fix; additional property, downgrade and ledger tests pass in the final suite.
Security/style: python -m bandit -r tldw_Server_API/app/core/DB_Management/db_migration.py tldw_Server_API/app/core/DB_Management/media_db/schema/backends/sqlite_helpers.py tldw_Server_API/app/core/DB_Management/test_migrations.py -f json -o /tmp/pr2627-bandit.json => 0 findings over 1297 LOC. Ruff, py_compile, guard_no_nonempty_legacy_complete.py, guard_http_client_patching.py, and git diff --check origin/dev pass. Temporary plan is retained in this task plan section and removed from the working tree at completion.
Touched: Docs/Database_Migrations.md; db_migration.py; media_db/schema/backends/sqlite_helpers.py; DB_Management/test_migrations.py; tests/DB_Management/test_db_migration_planning.py; tests/DB_Management/test_media_db_migration_missing_scripts_error.py; replacement Backlog record. TASK-12142 on dev is left untouched. No packaged SQL remains changed. Fetch immediately before publication still reports dev 40345571a2cfc8b3a8893545836097d27e4ee86c and PR head 3fae4675ae7e36c3217aea0986d4864685c48ac3, which is the exact force-with-lease expectation.
Merge assessment: still useful because dev retains the partial-DDL durability defect. Local fixes/review are complete. Fresh remote checks and a requester-owned human-written Change summary are still required before merge; this work does not merge the PR.

2026-09-10 Qodo follow-up: current-head review on c346aea33e reported six actionable items: BOM-prefixed legacy outer wrappers are not recognized; test classification markers, docstrings, shared fixtures, descriptive case IDs, and observable parser behavior need improvement. Reopened for fixes and rebase onto dev 456eafb7a6 (unrelated VZ guest buffering changes). Plan: reproduce BOM failure with file-backed public migration API and verify checksums; remove only the leading BOM from execution input; restructure new tests around shared fixtures/public behavior; run focused and expanded regression checks plus lint/security; reply to all six threads, request final review, then merge only after required checks and final-head review pass. Existing requester-written Change summary remains approved and preserved.

2026-09-10 Qodo remediation verified on dev 456eafb7a603449722ba8db806071a5e2aa5e7d6. Rebase range-diff showed all four prior commits unchanged. Removed one leading BOM only from execution input with str.removeprefix; loader source and checksum calculation remain unchanged. Added on-disk migration regressions for BOM before BEGIN, a comment, and foreign-key PRAGMA; all three failed with not-authorized errors before the fix and passed afterward. They verify original file/up_sql, ledger checksum, schema version, and embedded BOM literal preservation. Existing embedded-BOM COMMIT rejection still passes.
Addressed the five test-maintenance findings with shared initialized/versioned SQLite fixtures, exactly one unit classification marker and docstrings on each added planning test, descriptive parameter IDs, and public database-effects tests replacing the parser spy and FK extraction assertion. Independent review found all six Qodo findings addressed and no remaining substantive correctness/security issues.
Validation after activating the project .venv: python -m pytest on the same 11 migration/CLI/bootstrap/backup modules recorded above -q -rs => 171 passed, 7 skipped (PostgreSQL unavailable), 4 warnings in 36.20s. Marker-selected planning and legacy tests => 23 passed, 5 pre-existing unclassified tests deselected. Ruff, py_compile, both repository guards and git diff --check pass; Bandit on the same touched application scope => 0 findings, 1297 LOC. Packaged SQL remains unchanged against latest dev. Results: /tmp/pr2627-qodo-full-tests.log and /tmp/pr2627-qodo-bandit.json.
The requester supplied the human-written Change summary, now preserved verbatim in the PR description, and explicitly authorized final rebase, Qodo remediation, review replies and merge. Local implementation is complete; a new review covering the published final head and all required GitHub checks remain necessary before merging.

2026-09-10 final-head Qodo review on 6e5fefe258 reported two follow-ups: the modified internal missing-chain regression needs a unit marker, and valid comments around outer transaction/FK-PRAGMA boundaries prevent compatibility classification. Plan: reproduce accepted boundary-comment failures and rollback behavior with public file-backed migrations; normalize only comments outside quoted SQL tokens for classification while retaining original executable SQL/checksums; add the marker; rerun relevant regression, lint/security and independent review before publication and another final-head review.

2026-09-10 boundary-comment follow-up: Qodo review on 6e5fefe258 had two findings. Added the unit marker to the modified internal missing-chain regression. Replaced full-line-only comment stripping with quote-aware comment normalization for classification: quoted strings/identifiers remain intact, comments become whitespace, and original SQL still executes and supplies checksums. Added eight public file-backed cases for leading/embedded block comments and inline comments before/after semicolons on outer BEGIN/COMMIT and FK PRAGMAs. Each comment mode verifies success or forced schema-version-write rollback, ledger/version consistency and original source/checksum integrity. Seven lexer output cases protect single/double/backtick/bracket quoting and escaped quotes. Red run: 13 failed, 2 passed; green unit selection: 39 passed, 16 pre-existing unclassified tests deselected.
Expanded 11-module migration/CLI/bootstrap/backup suite (same command above) => 186 passed, 7 PostgreSQL-unavailable skips, 4 warnings in 35.15s. Ruff, compilation, repository guards and whitespace checks pass. Bandit touched application scope => 0 findings over 1299 LOC. Logs: /tmp/pr2627-comments-full-tests.log and /tmp/pr2627-comments-bandit.json. Independent review found no substantive issues and confirmed SQLite semantics with eight executed quote/comment comparisons.
Dev advanced to f0248aaa00047d2ffcc3bde295d9fbb8296add8a (audio test and task records only). Rebased with no conflicts; range-diff preserves all five prior patches. Post-rebase unit selection again passes 39 tests, with 16 pre-existing unclassified tests deselected; whitespace and shipped-SQL equality checks pass. Log: /tmp/pr2627-comments-rebased-unit.log. Human Change summary remains accepted and unchanged. Publish fixes and resolve both findings, then require completed final-head Qodo review and required GitHub gates before authorized merge.

2026-09-10 Qodo review on 23f6ee25e5 reported one actionable legacy-startup cleanup issue: unsupported-version rejection leaves the current thread connection in the shared SQLite pool. Plan: extend the real legacy rejection regression to observe pool statistics, closed connection behavior and fresh acquisition; reproduce first, then use the existing clear_thread_local_connection API only on this rejection path, preserving successful/in-memory bootstrap ownership; run focused/full validation and independent review before publication.

2026-09-10 legacy-pool cleanup: Qodo review on 23f6ee25e5 identified a retained connection on unsupported legacy rejection. Extended the existing real 1/8/21 cases through the shared backend factory: all three failed before the change with one active connection remaining. The rejection branch now uses SQLiteConnectionPool.clear_thread_local_connection(), which closes and invalidates only the current thread handle while leaving the pool reusable. Tests verify active count zero, ProgrammingError on the old closed handle, a distinct usable replacement, preserved version/data and no migrator invocation. Four focused legacy tests pass. Successful migration and in-memory lifecycles are untouched; independent review found no actionable issues.
Fresh expanded migration/CLI/bootstrap/backup validation (same 11-module command above): 186 passed, 7 PostgreSQL-unavailable skips, 4 warnings in 35.64s. Ruff, py_compile, repository guards, whitespace and shipped-SQL equality checks pass. Bandit touched application scope: zero findings over 1303 LOC. Logs: /tmp/pr2627-cleanup-red.log, /tmp/pr2627-cleanup-green.log, /tmp/pr2627-cleanup-full-tests.log, /tmp/pr2627-cleanup-bandit.json. Remote dev remains f0248aaa00047d2ffcc3bde295d9fbb8296add8a and observed PR head is 23f6ee25e53f28e098f59c8c1df40706440da860. The human Change summary is preserved; final-head Qodo review and required checks remain merge gates.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Rebased PR #2627 onto dev f0248aaa00 and addressed Qodo findings through reviewed head 23f6ee25e5, including closing/invalidation of pooled connections on legacy rejection. Migration execution/bookkeeping stay atomic, BOM/commented wrappers remain compatible, SQL/checksums are preserved, and regression tests cover rollback, retries, quoting and cleanup. Latest expanded verification: 186 passed, 7 PostgreSQL-unavailable skips; zero Bandit findings and no substantive independent-review issues. The requester-owned Change summary is present. Await completed final-head Qodo review and required GitHub checks before the authorized merge.
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
