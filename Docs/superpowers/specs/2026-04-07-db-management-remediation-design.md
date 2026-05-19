# DB_Management Remediation Design

- Date: 2026-04-07
- Project: tldw_server
- Topic: Remediate confirmed DB_Management review findings in one fail-closed branch
- Mode: Design for implementation planning

## 1. Objective

Implement one branch that addresses all confirmed DB_Management findings and the one approved probable-risk item without broadening into unrelated subsystem redesign.

The remediation must:

- remove silent partial-success behavior in security- and schema-critical flows
- make database bootstrap and migration failures truthful
- tighten shared backend and filesystem trust-boundary behavior
- normalize cross-backend search and read contracts where current semantics drift by backend or failure mode
- add direct regression coverage for every corrected contract

## 2. Scope

### In Scope

- PostgreSQL RLS policy installers:
  - `tldw_Server_API/app/core/DB_Management/backends/pg_rls_policies.py`
  - `tldw_Server_API/app/main.py` for startup auto-ensure caller alignment
- Migration loading, execution, verification, and CLI wiring:
  - `tldw_Server_API/app/core/DB_Management/db_migration.py`
  - `tldw_Server_API/app/core/DB_Management/migrate_db.py`
- Auth bootstrap and RBAC seeding paths still driven by `UserDatabase_v2`:
  - `tldw_Server_API/app/core/DB_Management/UserDatabase_v2.py`
  - direct consumers that still route through it:
    - `tldw_Server_API/app/core/AuthNZ/db_config.py`
    - `tldw_Server_API/app/core/AuthNZ/migrate_to_multiuser.py`
  - `tldw_Server_API/app/core/AuthNZ/repos/rbac_repo.py`
- Shared PostgreSQL content backend lifecycle:
  - `tldw_Server_API/app/core/DB_Management/content_backend.py`
  - `tldw_Server_API/app/core/DB_Management/media_db/runtime/defaults.py`
  - `tldw_Server_API/app/core/DB_Management/DB_Manager.py`
- Trusted SQLite path handling:
  - `tldw_Server_API/app/core/DB_Management/db_path_utils.py`
  - representative helper modules that currently bypass or weaken the shared trust boundary:
    - `tldw_Server_API/app/core/DB_Management/TopicMonitoring_DB.py`
    - `tldw_Server_API/app/core/DB_Management/watchlist_alert_rules_db.py`
    - `tldw_Server_API/app/core/DB_Management/Voice_Registry_DB.py`
- Media DB package-level read helpers and backend FTS semantics:
  - `tldw_Server_API/app/core/DB_Management/media_db/api.py`
  - `tldw_Server_API/app/core/DB_Management/backends/base.py`
  - `tldw_Server_API/app/core/DB_Management/backends/sqlite_backend.py`
  - `tldw_Server_API/app/core/DB_Management/backends/postgresql_backend.py`
  - `tldw_Server_API/app/core/DB_Management/backends/fts_translator.py`
- Callers and tests that currently encode the buggy behavior:
  - `tldw_Server_API/app/core/MCP_unified/modules/implementations/media_module.py`
  - `tldw_Server_API/app/core/StudyPacks/source_resolver.py`
  - `tldw_Server_API/app/core/RAG/rag_service/agentic_chunker.py`
  - `tldw_Server_API/app/api/v1/endpoints/media/navigation.py`
  - targeted files in `tldw_Server_API/tests/DB_Management/`
  - targeted caller tests outside `tests/DB_Management` when they directly depend on the changed contracts

### Out of Scope

- Broad AuthNZ redesign beyond the live `UserDatabase_v2` bootstrap and seed path
- Replacing `UserDatabase_v2` entirely in this branch
- Reworking every DB_Management helper module that uses SQLite paths; this pass covers the shared helper and the representative modules reviewed
- Unrelated migration-framework redesign
- Performance-only tuning unless required to make the corrected contracts safe

## 3. Findings To Fix

1. PostgreSQL RLS installers can partially fail and still report success.
2. The migration loader fails open on malformed inputs and upgrades do not require contiguous version advancement.
3. `UserDatabase_v2` can complete bootstrap with missing required columns or missing baseline RBAC state.
4. Shared PostgreSQL content backend resets can leak superseded pools.
5. Trusted SQLite path validation is lexical only and allows symlink escape.
6. Package-level `media_db.api` lookup helpers collapse backend failures into benign values.
7. Backend-level FTS behavior is not syntax-parity across SQLite and PostgreSQL.
8. `migrate_db.py --no-backup` is not wired to the migrator.

## 4. Locked Decisions

- This branch is fail-closed by default. Silent partial success is not preserved behind compatibility flags.
- The work stays in one branch, but implementation will still be staged in reviewable commits from highest risk to lowest.
- `UserDatabase_v2` is treated as a live production path because it is still imported by active AuthNZ code.
- `media_db.api` backend/query faults should surface through the existing DB exception family rather than a brand-new exception hierarchy.
- Backend FTS normalization becomes a backend responsibility at the shared abstraction boundary, not an optional caller convention.
- Trusted path enforcement must resolve symlinks before containment checks and must be reused by representative helper modules instead of ad hoc local validation.

## 5. Approaches Considered

### Recommended: Contract-Hardening Remediation

Fix the underlying DB-management contracts and make failure semantics explicit at the boundaries where the review found fail-open behavior.

Pros:

- Removes the underlying correctness and isolation defects instead of masking them.
- Keeps cross-backend behavior more coherent.
- Produces direct regression tests for the corrected contracts.

Cons:

- Can surface latent bad states during bootstrap or migration that were previously hidden.
- Requires updating tests and a few callers that implicitly depended on silent fallback.

### Alternative: Compatibility-Flag Remediation

Fix the underlying defects but preserve lenient behavior behind compatibility switches where possible.

Pros:

- Lower immediate rollout shock.

Cons:

- Adds branching and long-term complexity.
- Undercuts the goal of correcting fail-open DB contracts.

### Rejected: Narrow Local Patches

Patch each finding only at the most visible call site.

Reason rejected:

- Leaves the same contract drift in place and would likely miss related failure paths in the same module boundaries.

## 6. Approved Behavior Changes

### 6.1 PostgreSQL RLS Installation

- `ensure_prompt_studio_rls(...)` and `ensure_chacha_rls(...)` must return success only when every required statement in that policy set succeeds.
- For PostgreSQL backends, any required-statement failure must abort the installer, roll back the transaction, and raise `DatabaseError`.
- Returning `False` is reserved for explicit no-op cases where the supplied backend is not PostgreSQL or cannot be positively identified as PostgreSQL.
- Debug-only “best effort” application is no longer acceptable for tenant-isolation setup.
- The installers remain idempotent for already-correct databases.

### 6.2 Migration Loading And Execution

- Malformed `.json` or `.sql` migration artifacts must fail migration loading, not be logged and skipped.
- Duplicate version numbers across migration artifacts must fail migration loading.
- Upgrade execution must require a contiguous version path from current version to target version.
- A request to upgrade to latest available version must fail if an intermediate version is missing.
- Downgrade execution must also require a contiguous version path and a defined `down_sql` for every migration in the requested rollback range before any downgrade step runs.
- Verification should report missing migration files, checksum drift, and version gaps explicitly.

### 6.3 `UserDatabase_v2` Bootstrap

- Required `users` columns and required `registration_codes.role_id` normalization must be fatal if they cannot be established.
- Baseline role, permission, and required role-permission links must be created transactionally and verified after seeding.
- Missing required RBAC state after seeding is an initialization failure, not a debug log.
- Optional or compatibility-only seed data may remain best-effort only if it is explicitly classified as non-required in code and tests.

### 6.4 Shared PostgreSQL Backend Lifecycle

- Replacing a cached PostgreSQL content backend must close the displaced backend pool before losing the reference.
- Cache-clear helpers used by runtime reset paths must apply the same close-before-replace rule.
- Reset helpers must remain safe when no cached backend exists or when the cached backend has already been closed.

### 6.5 Trusted Path Enforcement

- `resolve_trusted_database_path(...)` must resolve candidate paths and trusted roots with symlink-aware containment checks.
- Representative SQLite helper modules must use the shared trust helper or an equivalent shared wrapper, not local partial validation.
- Legitimate project-root, configured user-db-root, and test-temp-root behavior remains supported.

### 6.6 `media_db.api` Read Contract

- Helper functions that currently swallow backend/query exceptions and return `False`, `None`, or `[]` must instead raise a typed DB/read error.
- Legitimate “not found” outcomes remain benign.
- The surfaced exception is the existing backend `DatabaseError` family, wrapping non-DB exceptions when needed for consistency.
- Callers that intentionally want fallback behavior must make that choice explicitly.

### 6.7 Backend FTS Parity

- Shared `FTSQuery` input must be normalized per backend at the backend abstraction boundary before SQL generation.
- SQLite keeps FTS5-compatible query execution.
- PostgreSQL receives normalized tsquery-compatible text through the shared backend API rather than assuming callers pre-normalized it.
- Caller code that already normalizes queries may continue to work; the backend layer must not require that normalization for correctness for the shared `fts_search(...)` abstraction.

### 6.8 Migration CLI Contract

- `migrate_db.py migrate --no-backup` must pass `create_backup=False` through to the actual migrator call.
- Default CLI behavior remains backup-on unless the flag is supplied.

## 7. Design

### 7.1 RLS Installer Hardening

`pg_rls_policies.py` should stop treating policy application as “successful if anything worked.” The installer must execute the full statement set inside one transaction and fail the whole operation on the first required-statement error.

Recommended shape:

- add a shared internal helper that executes a named policy set
- collect the statement index or SQL label for error context
- rollback on failure
- return `True` only after a full successful commit

Tests should exercise partial failure by injecting one broken statement between otherwise valid ones and asserting the installer reports failure rather than success.

Caller contract:

- `app/main.py` should treat installer exceptions as failed auto-ensure, not as a boolean-applied result.
- When `RAG_ENSURE_PG_RLS=true`, startup logging must never claim the policy set was applied if either installer raised.

### 7.2 Migration Contract Hardening

`db_migration.py` should distinguish between “no migrations exist” and “migration set is invalid.” An invalid migration set must raise `MigrationError`.

Required design details:

- `load_migrations()` raises on malformed file content, unreadable executable SQL with invalid version metadata, or duplicate versions
- migration planning validates contiguous versions for the requested upgrade range before creating backups or executing SQL
- downgrade planning validates contiguous versions and the presence of `down_sql` for the entire requested rollback range before creating backups or executing any rollback SQL
- verification explicitly reports version gaps in applied or available migrations
- failure messages should name the offending version or file so operators can repair the migration set without reproducing locally

This is intentionally stricter than current behavior because silent skipping is the defect.

### 7.3 `UserDatabase_v2` Fail-Closed Bootstrap

The current `_ensure_core_columns()` and `_seed_default_data()` paths need to be split conceptually into:

- required schema normalization
- required baseline RBAC seeding
- optional compatibility extras if any remain

Required schema and RBAC work must either succeed fully or raise `UserDatabaseError`.

Required postconditions:

- `users` contains `uuid`, `metadata`, `failed_login_attempts`, `locked_until`, and `is_superuser`
- `registration_codes` contains `role_id`
- roles `admin`, `user`, and `viewer` exist
- baseline permissions required by the module exist
- required role-permission mappings for those baseline roles exist

Implementation should prefer shared helpers that both perform the writes and verify the required postconditions rather than relying on best-effort inserts plus later assumptions.

### 7.4 Shared Backend Reset Hardening

The cache owner in `content_backend.py` should own close-before-replace behavior so reset callers do not each reimplement lifecycle handling.

Design rule:

- if a new backend is about to replace a cached backend with a different signature, close the old backend pool first
- if reset code clears the cache without immediate replacement, close the old backend first
- keep lock coverage around cache mutation and close-reference swaps so concurrent reloads do not race into duplicate-close or leaked-reference behavior

`media_db/runtime/defaults.py` and `DB_Manager.py` should call into the hardened cache-reset behavior rather than manually nulling cache globals.

### 7.5 Trusted Path Consolidation

`db_path_utils.py` should become the single trust-boundary authority for representative SQLite helper modules.

Design details:

- resolve trusted roots and candidate path symlink targets before final containment checks
- preserve support for non-existent target files by resolving the parent directory strictly enough to prevent escape while still allowing file creation
- keep test-context temporary directories trusted

Representative helper modules should be normalized as follows:

- `TopicMonitoringDB`: require `resolve_trusted_database_path(...)` instead of only rejecting bare relative filenames
- `watchlist_alert_rules_db`: validate incoming `db_path` through the shared trust helper at the entry points that open SQLite connections
- `VoiceRegistryDB`: move from custom base-dir containment logic to the shared trust helper so all three modules share the same rule

### 7.6 `media_db.api` Error Semantics

The package-level helpers in `media_db/api.py` are a read-contract surface, not a silent fallback layer.

Design details:

- add a small internal helper that converts unexpected backend/query exceptions into `DatabaseError` with operation context
- apply it to the chunk-existence, chunk-navigation, chunk-range, and section-lookup helpers identified in the review
- preserve current benign returns for valid miss cases such as “no row found”

Caller handling:

- modules like `MCP_unified/modules/implementations/media_module.py` may catch the typed error if they intentionally want degraded fallback behavior
- `StudyPacks/source_resolver.py`, `RAG/rag_service/agentic_chunker.py`, and `api/v1/endpoints/media/navigation.py` must be reviewed and either explicitly catch/log the typed error for intentional fallback behavior or allow it to surface
- if callers do catch and degrade, that decision should be explicit and logged at the caller boundary, not hidden inside `media_db.api`

### 7.7 Backend FTS Normalization

The shared backend `fts_search(...)` contract should normalize query text before SQL generation.

Recommended design:

- keep `FTSQuery.query` as the caller-facing input
- add a backend-local normalization step that uses `FTSQueryTranslator.normalize_query(...)` for the relevant backend family
- raise a clear `DatabaseError` for empty or invalid normalized PostgreSQL tsquery input where execution would otherwise misbehave

This keeps the shared abstraction honest: callers hand the backend one logical search string and the backend adapts it for its own SQL dialect.

Scope boundary:

- This remediation guarantees parity at the shared backend `fts_search(...)` abstraction.
- Existing direct raw-SQL search paths that already branch on backend and call `FTSQueryTranslator` themselves are not rewritten in this branch unless a touched regression test requires it.

### 7.8 CLI Wiring

`migrate_db.py` should thread the parsed `--no-backup` flag directly into `DatabaseMigrator.migrate_to_version(..., create_backup=...)`.

No wider CLI redesign is needed in this branch.

## 8. File Responsibilities

- `pg_rls_policies.py`
  - own all-or-nothing RLS installer behavior
- `app/main.py`
  - keep startup auto-ensure handling aligned with the hardened RLS exception contract
- `db_migration.py`
  - own invalid-migration detection, contiguous upgrade validation, and stronger verification output
- `migrate_db.py`
  - own truthful CLI-to-migrator flag wiring
- `UserDatabase_v2.py`
  - own required schema/RBAC bootstrap verification and fatal failure behavior
- `content_backend.py`
  - own close-before-replace backend cache semantics
- `media_db/runtime/defaults.py`
  - stop bypassing backend close behavior during resets
- `DB_Manager.py`
  - keep reset/shutdown wiring aligned with the hardened backend lifecycle
- `db_path_utils.py`
  - own symlink-safe trusted path validation
- `TopicMonitoring_DB.py`, `watchlist_alert_rules_db.py`, `Voice_Registry_DB.py`
  - consume the shared trusted-path rule consistently
- `media_db/api.py`
  - own truthful read-helper error semantics
- `sqlite_backend.py` and `postgresql_backend.py`
  - own backend-local FTS normalization at the abstraction boundary
- targeted tests
  - lock in the corrected contracts and prevent regression

## 9. Testing Strategy

This work should be implemented with TDD where practical.

Required regression coverage:

- RLS installer fails overall when any required statement fails
- RLS installer still succeeds idempotently on a fully valid policy set
- startup auto-ensure does not log a successful applied state after an installer exception
- malformed migration `.json` and malformed SQL migration metadata fail loading
- duplicate migration versions fail loading
- missing intermediate versions fail planned upgrade
- missing intermediate versions or missing `down_sql` fail planned rollback before any rollback step executes
- `verify_migrations()` reports version-gap issues
- `UserDatabase_v2` initialization fails when required column normalization fails
- `UserDatabase_v2` initialization fails when required RBAC seeding leaves missing roles, permissions, or mappings
- shared backend reset closes superseded backend pools
- symlink escape attempts are rejected by `resolve_trusted_database_path(...)`
- representative helper modules apply the shared trust boundary
- `media_db.api` helper methods raise typed DB errors on backend failure while still returning benign values for true miss cases
- backend FTS search normalizes equivalent logical queries for SQLite and PostgreSQL consistently enough for the abstraction contract
- direct callers of the changed `media_db.api` helpers either explicitly handle typed DB errors or fail transparently in tests
- `migrate_db.py --no-backup` disables backup creation in the migration call path

Verification should prefer focused unit and integration slices in `tldw_Server_API/tests/DB_Management/` plus only the direct caller tests needed to confirm changed error handling.

## 10. Risk Management

Primary risks:

- latent bad databases now fail fast during bootstrap or migration
- existing tests may encode the buggy fail-open behavior
- a few callers may currently rely on silent `media_db.api` fallback and need explicit error handling
- startup auto-ensure logging currently assumes boolean RLS installer results and must be kept aligned with the new exception contract
- path hardening may reject historically tolerated but unsafe filesystem layouts

Mitigations:

- keep file and public entrypoint churn narrow
- make error messages specific enough for operators and tests
- update direct callers and tests in the same branch
- stage commits from highest-risk contract fixes to lowest-risk tooling fixes for easier review and bisecting

## 11. Success Criteria

The remediation is successful when:

- PostgreSQL RLS setup cannot report success after partial failure
- invalid migration sets fail before applying later versions
- `UserDatabase_v2` cannot report successful bootstrap without required schema and baseline RBAC state
- shared PostgreSQL backend resets do not leak displaced pools
- trusted SQLite path checks reject symlink escape and representative helper modules use the same trust boundary
- `media_db.api` no longer hides backend/query faults behind benign values
- the shared backend `fts_search(...)` abstraction no longer depends on callers knowing different SQLite vs PostgreSQL query syntax rules
- the migration CLI accurately reflects whether backups are enabled
- each corrected behavior has direct regression coverage
