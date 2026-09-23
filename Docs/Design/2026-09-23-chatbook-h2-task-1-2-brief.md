# Task 1.2 brief: durable native fork operations and deletion

Tracking: TASK-13261.4; prerequisite integration TASK-13261.5. Work only in the isolated `codex/chatbook-h2-native-fork` worktree after the H1 merge, preserving `apps/tldw-frontend/.next-live-tier-h1-production/` and main/UAT. The accepted Task 1.1 projection is already present and its integrated focused suite passed 128 tests on 2026-09-23 after metadata regression fixes. The initial v70/v74 schema and receipt-store increment is reviewed in [the storage review](../Reviews/CHATBOOK_H2_NATIVE_STORE_INCREMENT_REVIEW_2026_09_23.md); the remaining Task 1.2 interfaces and lifecycle work below are not qualified.

## Pinned inputs and first check

H1 PR #2968 integrated commit `c13bad37fefd85ad15e3d0e9145db308a2a4723e`, server dev `91e8bbf84c25d3afbba2bb53ed06280d44c35307`, Chatbook dev `4d3e2d380e6ebb7a8e465d2d90a7564a82e7f6ef`. Recheck remote heads before writing migrations. H1's catalog-aware repair has SQLite v69 and PostgreSQL v73; use v70/v74 only if those remain the current versions. Never infer the meaning of an old v68 database from its number alone. Verify clean initialization, current-dev-v68/PG-v72 upgrade, prior H1-v68 upgrade, repeated initialization and rollback using the existing repository fixtures.

## Scope and invariants

Implement the plan's Task 1.2 store/migration/deletion surfaces, including native operation receipts, candidate/claim/reference and quota-intent schema, direct-owner PostgreSQL RLS, protected child binding, and workspace native-admission closure. No byte transfer, provider call, new quota authority, public fork execution, or `tldw-agent` integration belongs in this task. Keep SQL in DB_Management and external I/O outside the short final transaction.

- Reserve by authenticated owner, operation kind and ID with immutable canonical request digest. A changed digest conflicts; same accepted key never becomes a fresh creation after source/child deletion, receipt compaction or retry.
- Lock operation before workspace before conversation before assets. Committed result or permanent gone/expired outcome is resolved before source lookup. A failed transaction leaves no partial child, claim or receipt transition.
- Direct-owner RLS must work under the actual app role and `FORCE ROW LEVEL SECURITY`; do not rely on live conversation rows for receipt visibility. Keep server workspace admission and operation scope checks together.
- Begin workspace close before enumeration. Reject native admission into a closed workspace; keep closure on partial failure and allow deletion retry at the current public version. Preserve existing saved-view delete-first barriers after moving them inside the short final transition. Nested staged deletion rejects before mutation.
- H1 strict ordered image reads, image details, saved-turn retry and general browser multi-image guard remain intact. H2's representation-aware cold reopen is a later client/asset task; do not weaken reads to make store tests pass.

## Red/green sequence

1. Write migration and tenancy tests first in `test_native_fork_migration.py`, `test_native_fork_transactions.py`, and `test_native_fork_tenancy.py`, following `test_history_selection_transactions.py` and the project's PostgreSQL fixture. Record RED failures for missing v70/v74 schema, durable same-key state and direct-owner RLS.
2. Add the smallest database schema/store methods from the plan. Prove deterministic same-key concurrency, failed commit rollback, child soft/hard/bulk deletion, source deletion, receipt compaction, wrong owner and cold reopen on SQLite and live PostgreSQL.
3. Add workspace-close and residual-protected-row barriers, including admission-before-close and close-before-admission. Re-run the two existing saved-view deletion concurrency suites with their original waiting and error assertions.
4. Run the new suites and affected H1 migration/history tests, compile/Ruff and touched Python Bandit. Obtain independent P1/P2 review and fix all findings before the Task 1.2 commit.

The interface names and complete cases remain authoritative in `IMPLEMENTATION_PLAN_chatbook_h2_native_fork.md` Task 1.2 and `Docs/Design/2026-09-18-chatbook-h2-native-fork-design.md`; this brief adds the refreshed source/migration constraints and does not narrow those acceptance criteria.
