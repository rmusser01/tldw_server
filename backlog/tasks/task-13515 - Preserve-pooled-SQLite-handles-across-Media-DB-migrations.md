---
id: TASK-13515
title: Preserve pooled SQLite handles across Media DB migrations
status: In Progress
created_date: 2026-10-07 06:46
priority: high
modified_files:
- tldw_Server_API/app/core/DB_Management/media_db/schema/backends/sqlite_helpers.py
- tldw_Server_API/tests/DB_Management/test_media_db_schema_bootstrap.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Supported SQLite Media DB upgrade closes a connection borrowed from the shared pool directly, then opens an unrelated raw replacement. A subsequent same-path database instance receives the cached closed handle and fails schema-version lookup. Use existing pool ownership APIs; do not suppress normal startup or weaken migration checks.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Supported migration leaves an immediately borrowed pooled connection usable and a second same-path MediaDatabase can initialize.
- [x] #2 Migration failure invalidates the borrowed handle, preserves the original error, and allows a fresh usable connection to be borrowed.
- [x] #3 Existing schema bootstrap and SQLite pool regression tests pass; lint and security checks recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce supported upgrade and failed-upgrade handle reuse with synthetic SQLite data. 2. Replace direct pooled close with invalidate_connection and reborrow through db.get_connection after migration; retain existing validation. 3. Run focused and neighboring tests, Ruff and Bandit, obtain independent review, submit narrow PR against latest dev. ADR required: no; restores the existing pool ownership contract without changing public APIs or persistence schema.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Base dev e3c345b76f2d93527488b2015b9c3a2649c1f981. MCP search/resource calls did not return; CLI fallback search and read-only exact code/task searches found no existing task for this defect. No hosted operational records, credentials or customer data may be included in this public change.
TDD: all three synthetic supported-upgrade/shared-backend/failure-retry cases failed before production edits with Cannot operate on a closed database, then passed after replacing raw close with existing pool invalidation and raw reopen with db.get_connection. Independent Python 3.12.13 run: 138 passed, 5 PostgreSQL-dependent skipped (Docker disabled; PostgreSQL unavailable), 83 warnings. Paths: schema bootstrap, shared SQLite backend registry, pool pruning, factory logging, owner logging. Scoped Ruff passed; production Bandit returned zero findings; git diff --check passed. Independent review in progress. No schema/API changes, startup suppression, new dependency or broader pool fallback.
Independent final review: one P2 test-only cleanup finding addressed with existing clear_thread_local_connection in both new finally blocks; final scoped review has no remaining P1/P2 findings. Post-review Python 3.12.13 regression rerun: 138 passed, 5 PostgreSQL-dependent skipped, 83 existing warnings, 2.92 seconds. Ruff and Bandit on touched files passed (B101 assertions excluded for tests; production alone also zero findings). Requester explicitly waived the human-written Change summary merge requirement; do not represent an AI-authored summary as human-written. No ADR or user-facing documentation change required for restoring an existing internal ownership contract. Public CI/review feedback and merge remain pending.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Supported SQLite Media DB migrations now detach the borrowed connection from its pool before migration and reborrow a properly configured pooled handle afterward. Three causal regressions cover immediate shared-pool reuse, same-path construction and failed migration recovery. Focused neighboring suite: 138 passed with five PostgreSQL-only skips. No schema, public API, dependency or startup-policy changes.
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
