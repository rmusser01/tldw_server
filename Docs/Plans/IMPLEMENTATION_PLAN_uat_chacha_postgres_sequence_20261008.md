# ChaCha PostgreSQL sequence monotonicity implementation plan

**Task:** TASK-13260.278.18.83.45 (root-owned; separate source-confirmed sequence note recorded and read back before edits).

**Goal:** Normal PostgreSQL Character creation must not collide with the tenant-hidden bootstrap row, and ChaCha sequence maintenance must preserve allocated IDs across ownership, rollback and concurrent allocation.

**Design:** Extract the existing Media fixed-inventory forward-only sequence loop into a shared backend helper. Preserve Media's public function, inventory and callers. ChaCha delegates using its existing inventory. The helper retains sequence last_value/is_called, uses the existing rollback-only ALTER SEQUENCE OWNED BY savepoint to serialize allocations, and advances only for a higher visible explicit ID or an equal visible ID whose sequence is uncalled. No MAX0 reset, RLS bypass, caller commit, global transaction change or invented allocator.

**ADR check:** ADR required: no new ADR. [ADR-020](../ADR/020-db-management-per-user-paths-and-content-backend.md) governs the supported shared PostgreSQL backend and migration isolation. This applies Media's existing monotonic sequence rule to ChaCha; it changes no ownership or public persistence contract.

**Scope:** Only the existing ChaChaNotes_DB._sync_postgres_sequences method; Media's existing schema/postgres_sequence_maintenance.py adapter; new backends/postgres_sequence_maintenance.py; new tests/DB_Management/test_chacha_postgres_sequence_monotonicity.py; this plan. Dictionary schema/RLS/capture production and tests stay byte-identical to their independently reviewed freeze.

**Limits:** Three distinct causal experiments maximum; case300/outer600 seconds. Installed activated Python3.12, official restricted-role PG fixture, fresh exact retained-container readiness, documented credential inputs in memory, NO_DOCKER=1/POSTGRES_REQUIRED=1. No operator resets, homemade database/role, installs, runtime/browser/provider/model/private logs/data, parent task or Git changes. Root owns final live acceptance. Previously arbitrarily rewound hidden imported rows are not repaired or qualified by this forward-only correction.

## Stage 1: Causal public and allocation regressions

**Goal:** Establish the actual tenant-hidden bootstrap/sequence state and fail normal public Character creation and unsafe allocation reuse before production edits.
**Success Criteria:** Real cookie auth and restricted FORCE RLS setup pass; only actual creation/allocation contract assertions fail.
**Tests:** Authenticated public Character POST with no explicit ID, subsequent owners/reopen, rollback allocations, visible high-ID import with caller rollback, real concurrent nextval interleaving.
**Status:** Complete

**Red experiment1:** Four actual contract failures,236warnings,23.46s (28.893s outer), natural1. Cookie authentication, non-super/non-bypass role and FORCE owner RLS passed. The public POST returned409 instead of201 with real bootstrap state last_value1/is_calledFalse; rollback allocated1 then reused1; visible import701 allocated702 then caller rollback/maintenance reset1; real concurrent allocation reached21 and held maintenance reset next8. No setup failure or guessed retained constraint attribution. The API's existing generic unique mapping labels the collision as a name conflict; copy/error-mapping repair is outside this unit.

## Stage 2: Shared forward-only maintenance

**Goal:** Reuse Media's existing locking and forward-only algorithm without changing inventories or caller transactions.
**Success Criteria:** New causal cases pass; reviewed dictionary scope remains byte-identical; Media wrapper API and inventory remain intact.
**Tests:** New regressions plus genuinely affected existing Media monotonic and ChaCha sequence ownership modules, including unrelated Media keyword preservation.
**Status:** Complete

**Green experiment2:** Four new cases passed naturally,236warnings,18.99s (23.171s outer). The observed final-write hook accepts both original and catalog-qualified quoted sequence names; it still delegates every actual SQL/result/error and retains the full allocation assertions. Public creation/reopen succeeds without an explicit ID; rollback/import/concurrent allocation assertions remain intact.

**Final affected experiment3:**21passed,236warnings,17.17s (20.910s outer), natural0/no skips. This includes the strengthened public case with the exact retained system_init seed confirmed through the official fixture's separate read-only admin connection; HTTP still uses the asserted non-bypass role and forced policies. Fourteen existing Media monotonic cases, three ChaCha ownership cases (including unrelated Media keyword preservation) and three sequence adapter unit cases are also green. Total24distinct passing cases:4new plus20existing. No fourth causal experiment or unchanged broad controls replay.

## Stage 3: Static qualification and independent review

**Goal:** Qualify exact affected scope, compare lint/security against pre-unit bytes and freeze final source/tests for independent review.
**Success Criteria:** Parse/compile success, zero new Ruff/Bandit findings, exact red/green ledger and independent source review handoff.
**Tests:** Configured affected pytest; scoped AST/compile, Ruff/Bandit baseline/current; final connected source review.
**Status:** In Progress

**Static/source checks:** Four touched Python files parse/compile. Installed configured Ruff and Bandit baseline/current are0/0 (no scan errors); no new suppression needed. Exact AST comparison retains Media's inventory, all protocol classes and full forward-only algorithm except the explicit owned-inventory parameter. Exact source-byte comparison confirms every ChaCha byte outside the single released _sync method is unchanged, plus all eight other reviewed dictionary source/test hashes. Independent source review is pending; actual PostgreSQL browser/provider acceptance and arbitrary previously rewound hidden-row repair remain unqualified.
