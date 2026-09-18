# Sources PostgreSQL portability review — UAT267

**CLEAR for the reviewed implementation; native catalogue acceptance remains pending.** Root independently reviewed the three changed files and executed nine SQLite/official PostgreSQL service tests and29 worker/API/cleanup tests. Both commands exited0, no skips. Frozen hashes match the author's report.

The original service emits SQLite DDL and expects cursor-returning execute calls. AuthNZ PostgreSQL transactions instead expose a guarded asyncpg connection. The local adapter preserves the guard and caller transaction, delegates positional conversion to the existing AuthNZ converter, supplies read-cursor and command-rowcount shapes, and uses RETURNING for four generated-ID writes. Public entry decoration is idempotent when service operations call each other. No raw connection unwrap or global driver change is introduced.

The explicit PostgreSQL schema preserves the six service tables, foreign keys, unique item path, integer flags and text JSON/timestamps. Its five generated IDs use identity columns. SQLite DDL and existing-column compatibility remain intact. Source ownership predicates, source identity immutability and active-job fence are preserved; the actual PG test verifies mismatch rejection. Worker/scheduler/cleanup callers obtain connections from the same AuthNZ transaction interface.

Tests cover repeated initialization, two owners, generated IDs, item upsert identity, updates, state completion, artifact/snapshot deletion and JSON/flag readback. Root adjacent checks exercise worker, endpoint and cleanup behavior. Bandit reports no production findings,55 test assertions and no parsing errors. Author compiler/diff checks pass; the sole Ruff SIM118 also exists in baseline.

The scope is compatibility of the existing service. No new migration, user permission, global schema or model configuration is claimed. The original authenticated Sources catalogue must return200 after a committed-source upgrade before267 or the client-route264 finding closes.
