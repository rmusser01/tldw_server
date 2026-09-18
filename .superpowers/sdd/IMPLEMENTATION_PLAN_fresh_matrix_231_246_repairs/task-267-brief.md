# UAT267 ingestion-sources PostgreSQL repair brief

## Stage 1: Prove the PostgreSQL cause

**Goal:** Reproduce schema initialization through the official guarded PostgreSQL fixture.

**Success criteria:** The retained causal red demonstrates PostgreSQL rejects the service-local SQLite `AUTOINCREMENT` schema before owner list logic runs.

**Tests:** New focused `test_db_pool` integration test under `tests/Ingestion_Sources/`, run by the mandated fixture runner.

**Status:** Complete

## Stage 2: Make the local service boundary portable

**Goal:** Add explicit SQLite/PostgreSQL schema statements and a service-local raw-connection query adapter.

**Success criteria:** All six tables initialize twice; PostgreSQL uses generated identities, converted positional placeholders, and `RETURNING id` for the four generated-ID creation paths, while SQLite retains its existing behavior.

**Tests:** Focused SQLite cold/idempotent, source/state, owner-isolation, item-upsert, snapshot/artifact/event tests and official PostgreSQL equivalents.

**Status:** Complete

## Stage 3: Verify preserved boundaries

**Goal:** Preserve guarded ownership, text JSON, integer flags, existing upsert semantics, and service-local scope.

**Success criteria:** Owner A cannot read owner B's source, updates/readback retain values, and static checks introduce no production finding.

**Tests:** Existing closest service/API controls plus scoped compile, Ruff, Bandit, and diff checks.

**Status:** Complete

## Scope and limits

This repair is limited to `tldw_Server_API/app/core/Ingestion_Sources/service.py` and focused maintained Ingestion Sources tests. It will not alter route identity/authentication, AuthNZ ownership guards, JSON storage, flag storage, RLS, global SQL conversion, pool/session configuration, or unrelated schemas. The official fixture runner supplies PostgreSQL; no custom database setup will be used.

## Attempt and reassessment record

1. The first PostgreSQL test invocation exposed only fixture discovery: the Ingestion Sources test directory does not inherit `test_db_pool`. Registering the existing AuthNZ fixture plugin corrected the test harness without product changes.
2. The retained causal PostgreSQL red then failed during guarded schema initialization, as expected from the diagnosed SQLite `AUTOINCREMENT` DDL. The explicit PostgreSQL schema branch passed its follow-up control.
3. The first owner-isolation test used an incompatible direct user fixture path under the guarded profile. It was reassessed before source work and changed to the maintained `UsersDB` creation path; no guard was weakened.
4. The resulting real query red was the expected raw-connection portability failure. The service-local adapter and four `RETURNING id` paths made the round trip green.
5. Expanded PostgreSQL state coverage exposed an untyped `CASE` parameter in asyncpg. A portable `CAST(? AS INTEGER)` preserves the existing predicate. Review then found asyncpg's update-status string lacked the existing active-job fence's `rowcount`; the same local adapter now exposes its parsed count. The final official PostgreSQL run passed both lifecycle tests.
