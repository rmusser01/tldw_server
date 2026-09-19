# UAT298 — PostgreSQL token-blacklist initialization

Task: TASK13260.235. Serialize the existing bootstrap DDL across requests/processes using the database transaction; preserve fail-closed revocation semantics.

## Stage 1: causal concurrency regression
**Goal**: Force overlapping real PostgreSQL initializations through the existing repository.
**Success Criteria**: New table/index creation controls reproduce the catalog race before repair, using the official isolated fixture.
**Tests**: Hold the first DDL transaction open until the second request is blocked; verify both initializations and durable token revocation.
**Status**: Complete

Evidence: two deterministic pre-fix PostgreSQL table/index catalog races; two cancellation controls pass. Initial setup attempt was rejected by the write guard and is retained separately from causal failures.

## Stage 2: repair and focused validation
**Goal**: Apply the existing transaction-scoped advisory-lock pattern before PostgreSQL blacklist DDL.
**Success Criteria**: Concurrency, rollback, SQLite and fail-closed controls pass; no new lint/security findings; review clear.
**Tests**: PostgreSQL repository concurrency/revocation, SQLite repository/service controls, failure handling, Ruff and Bandit.
**Status**: Complete

Evidence: 45 focused tests pass with zero skips, including real PostgreSQL overlap, cancellation, persisted revocation and SQLite/fail-closed controls. Existing legacy shared-fixture test now uses the mandated isolated fixture with unchanged behavior assertions. Ruff and Bandit have zero findings; independent review is clear.

## Stage 3: fresh native acceptance
**Goal**: Verify first login/authentication on a fresh PostgreSQL install.
**Success Criteria**: Native concurrent initialization completes without the token-blacklist catalog race; logout still revokes tokens. Source/evidence tracked and test processes closed.
**Tests**: Fresh PostgreSQL native login/logout and cluster/backend audit; record limits in tracker and Backlog.
**Status**: In Progress
