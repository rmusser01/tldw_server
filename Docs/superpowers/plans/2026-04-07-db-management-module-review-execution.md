# DB_Management Module Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the approved `DB_Management` audit and deliver one consolidated, evidence-backed review covering correctness, security and isolation, maintainability, and test gaps for `tldw_Server_API/app/core/DB_Management`.

**Architecture:** This is a read-first, risk-led audit plan. Execution starts from the approved seed files, records findings in stage reports under `Docs/superpowers/reviews/db-management/`, uses targeted tests to confirm backend-sensitive claims, and finishes with one ranked synthesis that distinguishes confirmed findings, probable risks, and improvements.

**Tech Stack:** Python 3, SQLite, PostgreSQL, pytest, ripgrep, git, Markdown

---

## Review File Map

**Create during execution:**
- `Docs/superpowers/reviews/db-management/README.md`
- `Docs/superpowers/reviews/db-management/2026-04-07-stage1-review-artifacts-and-inventory.md`
- `Docs/superpowers/reviews/db-management/2026-04-07-stage2-foundations-backends-factories.md`
- `Docs/superpowers/reviews/db-management/2026-04-07-stage3-paths-tenancy-migrations-backups.md`
- `Docs/superpowers/reviews/db-management/2026-04-07-stage4-media-db-and-representative-helpers.md`
- `Docs/superpowers/reviews/db-management/2026-04-07-stage5-test-gaps-and-synthesis.md`

**Primary source files to inspect during the review:**
- `tldw_Server_API/app/core/DB_Management/content_backend.py`
- `tldw_Server_API/app/core/DB_Management/db_path_utils.py`
- `tldw_Server_API/app/core/DB_Management/DB_Manager.py`
- `tldw_Server_API/app/core/DB_Management/scope_context.py`
- `tldw_Server_API/app/core/DB_Management/transaction_utils.py`
- `tldw_Server_API/app/core/DB_Management/async_db_wrapper.py`
- `tldw_Server_API/app/core/DB_Management/sqlite_policy.py`
- `tldw_Server_API/app/core/DB_Management/migration_tools.py`
- `tldw_Server_API/app/core/DB_Management/db_migration.py`
- `tldw_Server_API/app/core/DB_Management/migrate_db.py`
- `tldw_Server_API/app/core/DB_Management/migrations.py`
- `tldw_Server_API/app/core/DB_Management/content_migrate.py`
- `tldw_Server_API/app/core/DB_Management/DB_Backups.py`
- `tldw_Server_API/app/core/DB_Management/backends/base.py`
- `tldw_Server_API/app/core/DB_Management/backends/factory.py`
- `tldw_Server_API/app/core/DB_Management/backends/query_utils.py`
- `tldw_Server_API/app/core/DB_Management/backends/fts_translator.py`
- `tldw_Server_API/app/core/DB_Management/backends/sqlite_backend.py`
- `tldw_Server_API/app/core/DB_Management/backends/postgresql_backend.py`
- `tldw_Server_API/app/core/DB_Management/backends/pg_rls_policies.py`
- `tldw_Server_API/app/core/DB_Management/media_db/api.py`
- `tldw_Server_API/app/core/DB_Management/media_db/native_class.py`
- `tldw_Server_API/app/core/DB_Management/media_db/media_database.py`
- `tldw_Server_API/app/core/DB_Management/media_db/media_database_impl.py`
- `tldw_Server_API/app/core/DB_Management/media_db/schema/bootstrap.py`
- `tldw_Server_API/app/core/DB_Management/media_db/schema/migrations.py`
- `tldw_Server_API/app/core/DB_Management/media_db/runtime/connection_lifecycle.py`
- `tldw_Server_API/app/core/DB_Management/media_db/runtime/backend_resolution.py`
- `tldw_Server_API/app/core/DB_Management/media_db/runtime/backend_prepare_ops.py`
- `tldw_Server_API/app/core/DB_Management/media_db/runtime/scope_resolution_ops.py`
- `tldw_Server_API/app/core/DB_Management/media_db/runtime/session.py`
- `tldw_Server_API/app/core/DB_Management/media_db/runtime/execution.py`
- `tldw_Server_API/app/core/DB_Management/media_db/runtime/execution_ops.py`
- `tldw_Server_API/app/core/DB_Management/media_db/runtime/bootstrap_lifecycle_ops.py`
- `tldw_Server_API/app/core/DB_Management/media_db/runtime/sqlite_bootstrap.py`
- `tldw_Server_API/app/core/DB_Management/UserDatabase_v2.py`
- `tldw_Server_API/app/core/DB_Management/TopicMonitoring_DB.py`
- `tldw_Server_API/app/core/DB_Management/Voice_Registry_DB.py`
- `tldw_Server_API/app/core/DB_Management/Workflows_Scheduler_DB.py`
- `tldw_Server_API/app/core/DB_Management/watchlist_alert_rules_db.py`

**High-value existing tests to reuse during the review:**
- `tldw_Server_API/tests/DB_Management/test_backend_utils.py`
- `tldw_Server_API/tests/DB_Management/test_content_backend_cache.py`
- `tldw_Server_API/tests/DB_Management/test_database_backends.py`
- `tldw_Server_API/tests/DB_Management/test_db_manager_config_behavior.py`
- `tldw_Server_API/tests/DB_Management/test_db_manager_wrappers.py`
- `tldw_Server_API/tests/DB_Management/test_transaction_utils.py`
- `tldw_Server_API/tests/DB_Management/unit/test_postgres_placeholder_prepare.py`
- `tldw_Server_API/tests/DB_Management/unit/test_postgres_pool_fallback.py`
- `tldw_Server_API/tests/DB_Management/unit/test_postgres_transaction_manager.py`
- `tldw_Server_API/tests/DB_Management/unit/test_sqlite_pool_pruning.py`
- `tldw_Server_API/tests/DB_Management/test_db_path_utils.py`
- `tldw_Server_API/tests/DB_Management/test_db_path_utils_env.py`
- `tldw_Server_API/tests/DB_Management/test_db_paths_media_prompts_env.py`
- `tldw_Server_API/tests/DB_Management/test_research_db_paths.py`
- `tldw_Server_API/tests/DB_Management/test_output_storage_normalization.py`
- `tldw_Server_API/tests/DB_Management/test_sqlite_policy.py`
- `tldw_Server_API/tests/DB_Management/test_sqlite_policy_integrations.py`
- `tldw_Server_API/tests/DB_Management/test_db_migration_loader.py`
- `tldw_Server_API/tests/DB_Management/test_db_migration_path_validation.py`
- `tldw_Server_API/tests/DB_Management/test_migration_cli_integration.py`
- `tldw_Server_API/tests/DB_Management/test_migration_tools.py`
- `tldw_Server_API/tests/DB_Management/test_backup_restore_verification.py`
- `tldw_Server_API/tests/DB_Management/test_db_backup_integrity.py`
- `tldw_Server_API/tests/DB_Management/test_db_backup_name_validation.py`
- `tldw_Server_API/tests/DB_Management/test_media_db_api_imports.py`
- `tldw_Server_API/tests/DB_Management/test_media_db_bootstrap_lifecycle_ops.py`
- `tldw_Server_API/tests/DB_Management/test_media_db_connection_cleanup.py`
- `tldw_Server_API/tests/DB_Management/test_media_db_runtime_factory.py`
- `tldw_Server_API/tests/DB_Management/test_media_db_schema_bootstrap.py`
- `tldw_Server_API/tests/DB_Management/test_media_db_scope_resolution_ops.py`
- `tldw_Server_API/tests/DB_Management/test_media_db_request_scope_isolation.py`
- `tldw_Server_API/tests/DB_Management/test_media_db_core_repositories.py`
- `tldw_Server_API/tests/DB_Management/test_media_db_v2_regressions.py`
- `tldw_Server_API/tests/DB_Management/test_media_db_migration_missing_scripts_error.py`
- `tldw_Server_API/tests/DB_Management/test_media_db_postgres_rls_ops.py`
- `tldw_Server_API/tests/DB_Management/test_media_postgres_migrations.py`
- `tldw_Server_API/tests/DB_Management/test_media_postgres_support.py`
- `tldw_Server_API/tests/DB_Management/test_users_db_sqlite.py`
- `tldw_Server_API/tests/DB_Management/unit/test_users_db_update_backend_detection.py`
- `tldw_Server_API/tests/DB_Management/test_workflows_scheduler_db_paths.py`
- `tldw_Server_API/tests/AuthNZ/integration/test_authnz_backends_pytest.py`
- `tldw_Server_API/tests/Claims/test_claims_service_backend_selection.py`
- `tldw_Server_API/tests/Claims/test_ingestion_claims_sql.py`
- `tldw_Server_API/tests/Media/test_media_reprocess_endpoint.py`
- `tldw_Server_API/tests/test_watchlist_alert_rules.py`
- `tldw_Server_API/tests/test_utils.py`

## Stage Overview

## Stage 1: Review Artifact Setup and Inventory
**Goal:** Create stable review output files, capture the exact scoped source and test surface, and freeze the final review structure before deep reading begins.
**Success Criteria:** Review artifacts exist under `Docs/superpowers/reviews/db-management/`, the source and test inventories are recorded, and the final output structure is fixed so later stages do not drift.
**Tests:** None
**Status:** Not Started

## Stage 2: Shared Foundations, Backends, and Factory Routing
**Goal:** Inspect the shared abstractions that can affect many databases at once, with emphasis on backend feature parity, transaction behavior, query translation, and backend selection or caching.
**Success Criteria:** Shared foundations are reviewed with evidence, backend-sensitive claims are tested where feasible, and blast-radius findings are captured before narrower domain helpers are considered.
**Tests:** Backend, transaction, pool, content-backend, and DB-manager tests listed below.
**Status:** Not Started

## Stage 3: Paths, Tenancy, Policies, Migrations, and Backups
**Goal:** Validate filesystem safety, per-user database routing, scope propagation, policy or RLS assumptions, and schema or backup recovery logic.
**Success Criteria:** Path containment, tenancy boundaries, migration safety, and backup invariants are documented with confirmed findings, probable risks, or explicit no-finding notes.
**Tests:** Path, policy, migration, and backup tests listed below.
**Status:** Not Started

## Stage 4: Media DB Core Runtime and Representative Helper Review
**Goal:** Review `media_db` bootstrapping and runtime composition plus selected non-`media_db` helpers with custom transaction, schema, or filesystem logic.
**Success Criteria:** `media_db` execution flow, scope isolation, and bootstrap behavior are traced end to end, representative helper modules are reviewed from different risk profiles, and major hotspots are documented with evidence.
**Tests:** `media_db`, representative helper, and caller-level tests listed below.
**Status:** Not Started

## Stage 5: Test-Gap Pass and Final Synthesis
**Goal:** Compare the reviewed code paths against the available test surface, identify missing or weak invariants, and produce one final ranked review.
**Success Criteria:** Coverage gaps are prioritized by risk reduction value, duplicates from earlier stages are removed, and the final report matches the approved output model.
**Tests:** Reuse the earlier inventory and run only the narrowest additional validations needed to settle disputed claims.
**Status:** Not Started

### Task 1: Prepare Review Artifacts and Inventory

**Files:**
- Create: `Docs/superpowers/reviews/db-management/README.md`
- Create: `Docs/superpowers/reviews/db-management/2026-04-07-stage1-review-artifacts-and-inventory.md`
- Create: `Docs/superpowers/reviews/db-management/2026-04-07-stage2-foundations-backends-factories.md`
- Create: `Docs/superpowers/reviews/db-management/2026-04-07-stage3-paths-tenancy-migrations-backups.md`
- Create: `Docs/superpowers/reviews/db-management/2026-04-07-stage4-media-db-and-representative-helpers.md`
- Create: `Docs/superpowers/reviews/db-management/2026-04-07-stage5-test-gaps-and-synthesis.md`
- Test: none

- [ ] **Step 1: Create the review output directory**

Run:
```bash
mkdir -p Docs/superpowers/reviews/db-management
```

Expected: the `Docs/superpowers/reviews/db-management` directory exists and no source files change.

- [ ] **Step 2: Create one markdown file per stage with a fixed findings template**

Each stage file should contain:
```markdown
# Stage N Title

## Scope
## Code Paths Reviewed
## Tests Reviewed
## Validation Commands
## Findings
## Coverage Gaps
## Improvements
## Exit Note
```

- [ ] **Step 3: Write `Docs/superpowers/reviews/db-management/README.md`**

Document:
- the stage order `1 -> 2 -> 3 -> 4 -> 5`
- the path to each stage report
- the rule that findings must be written before remediation ideas
- the rule that uncertain items are labeled as assumptions or probable risks, not overstated as confirmed defects
- the rule that backend-sensitive claims require targeted verification or an explicit confidence downgrade

- [ ] **Step 4: Capture the scoped source and test inventory**

Run:
```bash
source .venv/bin/activate
rg --files tldw_Server_API/app/core/DB_Management tldw_Server_API/tests/DB_Management | sort
```

Expected: a stable list of `DB_Management` implementation files and direct test files, with no unrelated module trees mixed into the inventory.

- [ ] **Step 5: Capture the default recent-history baseline**

Run:
```bash
git log --oneline -n 20 -- tldw_Server_API/app/core/DB_Management
```

Expected: a stable first-pass churn window for `DB_Management`; only expand to older commits later if a hotspot needs more context.

- [ ] **Step 6: Freeze the final review output structure before deep reading**

Use this final response structure:
```markdown
## Findings

1. Severity: concise issue statement with file references and impact
2. Severity: concise issue statement with file references and impact

## Open Questions

- only if needed for unresolved assumptions
```

For each finding, record severity, confidence, why it matters, and the exact file references needed for action.

- [ ] **Step 7: Verify the workspace starts in a safe state**

Run:
```bash
git status --short
```

Expected: no source files under `tldw_Server_API/app/core/DB_Management` or `tldw_Server_API/tests/DB_Management` are modified as part of setup.

- [ ] **Step 8: Commit the review scaffold**

Run:
```bash
git add Docs/superpowers/reviews/db-management Docs/superpowers/plans/2026-04-07-db-management-module-review-execution.md
git commit -m "docs: scaffold DB_Management review artifacts"
```

Expected: one docs-only commit captures the review workspace before audit notes are added.

### Task 2: Execute Stage 2 Shared Foundations, Backends, and Factory Routing

**Files:**
- Modify: `Docs/superpowers/reviews/db-management/2026-04-07-stage2-foundations-backends-factories.md`
- Inspect: `tldw_Server_API/app/core/DB_Management/content_backend.py`
- Inspect: `tldw_Server_API/app/core/DB_Management/DB_Manager.py`
- Inspect: `tldw_Server_API/app/core/DB_Management/transaction_utils.py`
- Inspect: `tldw_Server_API/app/core/DB_Management/async_db_wrapper.py`
- Inspect: `tldw_Server_API/app/core/DB_Management/backends/base.py`
- Inspect: `tldw_Server_API/app/core/DB_Management/backends/factory.py`
- Inspect: `tldw_Server_API/app/core/DB_Management/backends/query_utils.py`
- Inspect: `tldw_Server_API/app/core/DB_Management/backends/fts_translator.py`
- Inspect: `tldw_Server_API/app/core/DB_Management/backends/sqlite_backend.py`
- Inspect: `tldw_Server_API/app/core/DB_Management/backends/postgresql_backend.py`
- Inspect: `tldw_Server_API/app/core/DB_Management/media_db/api.py`
- Test: `tldw_Server_API/tests/DB_Management/test_backend_utils.py`
- Test: `tldw_Server_API/tests/DB_Management/test_content_backend_cache.py`
- Test: `tldw_Server_API/tests/DB_Management/test_database_backends.py`
- Test: `tldw_Server_API/tests/DB_Management/test_db_manager_config_behavior.py`
- Test: `tldw_Server_API/tests/DB_Management/test_db_manager_wrappers.py`
- Test: `tldw_Server_API/tests/DB_Management/test_transaction_utils.py`
- Test: `tldw_Server_API/tests/DB_Management/unit/test_postgres_placeholder_prepare.py`
- Test: `tldw_Server_API/tests/DB_Management/unit/test_postgres_pool_fallback.py`
- Test: `tldw_Server_API/tests/DB_Management/unit/test_postgres_transaction_manager.py`
- Test: `tldw_Server_API/tests/DB_Management/unit/test_sqlite_pool_pruning.py`

- [ ] **Step 1: Map the shared entry points and backend abstraction surface**

Run:
```bash
rg -n "class (DatabaseBackend|SQLiteBackend|PostgreSQLBackend)|def (get_|create_|transaction|execute|executemany|prepare_|translate_)" \
  tldw_Server_API/app/core/DB_Management/content_backend.py \
  tldw_Server_API/app/core/DB_Management/DB_Manager.py \
  tldw_Server_API/app/core/DB_Management/transaction_utils.py \
  tldw_Server_API/app/core/DB_Management/async_db_wrapper.py \
  tldw_Server_API/app/core/DB_Management/backends/base.py \
  tldw_Server_API/app/core/DB_Management/backends/factory.py \
  tldw_Server_API/app/core/DB_Management/backends/query_utils.py \
  tldw_Server_API/app/core/DB_Management/backends/fts_translator.py \
  tldw_Server_API/app/core/DB_Management/backends/sqlite_backend.py \
  tldw_Server_API/app/core/DB_Management/backends/postgresql_backend.py \
  tldw_Server_API/app/core/DB_Management/media_db/api.py
```

Expected: a compact map of the public construction, transaction, query preparation, translation, and execution paths that later findings can reference.

- [ ] **Step 2: Trace backend feature parity and transaction semantics**

Read and record:
- placeholder and prepared-query differences between SQLite and PostgreSQL
- explicit versus implicit transaction boundaries
- connection-pool ownership and cleanup rules
- FTS or query translation assumptions that can diverge by backend

- [ ] **Step 3: Trace content backend selection, caching, and factory routing**

Confirm:
- where shared backends are cached
- when file-backed SQLite paths are chosen instead of shared backends
- how `DB_Manager.py` and `media_db/api.py` route callers into the right implementation
- whether any factory path can accidentally bypass expected safety checks

- [ ] **Step 4: Review the targeted tests and extract what they actually protect**

For each listed test file, record:
- the main invariant it asserts
- whether it checks SQLite, PostgreSQL, or both
- which backend-sensitive claim could be upgraded from probable risk to confirmed finding because of it

- [ ] **Step 5: Run the targeted shared-foundation tests**

Run:
```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/DB_Management/test_backend_utils.py \
  tldw_Server_API/tests/DB_Management/test_content_backend_cache.py \
  tldw_Server_API/tests/DB_Management/test_database_backends.py \
  tldw_Server_API/tests/DB_Management/test_db_manager_config_behavior.py \
  tldw_Server_API/tests/DB_Management/test_db_manager_wrappers.py \
  tldw_Server_API/tests/DB_Management/test_transaction_utils.py \
  tldw_Server_API/tests/DB_Management/unit/test_postgres_placeholder_prepare.py \
  tldw_Server_API/tests/DB_Management/unit/test_postgres_pool_fallback.py \
  tldw_Server_API/tests/DB_Management/unit/test_postgres_transaction_manager.py \
  tldw_Server_API/tests/DB_Management/unit/test_sqlite_pool_pruning.py -v
```

Expected: tests collect and mostly pass; any failure is either environment noise or a finding that must be explained in the stage report.

- [ ] **Step 6: Write the Stage 2 report**

Record:
- ranked findings with severity and confidence
- which issues are backend-agnostic versus backend-specific
- any open questions that block a stronger claim
- low-risk improvements that do not crowd out defects

- [ ] **Step 7: Commit the Stage 2 report**

Run:
```bash
git add Docs/superpowers/reviews/db-management/2026-04-07-stage2-foundations-backends-factories.md
git commit -m "docs: record DB_Management foundation review findings"
```

Expected: one docs-only commit contains the Stage 2 findings.

### Task 3: Execute Stage 3 Paths, Tenancy, Policies, Migrations, and Backups

**Files:**
- Modify: `Docs/superpowers/reviews/db-management/2026-04-07-stage3-paths-tenancy-migrations-backups.md`
- Inspect: `tldw_Server_API/app/core/DB_Management/db_path_utils.py`
- Inspect: `tldw_Server_API/app/core/DB_Management/scope_context.py`
- Inspect: `tldw_Server_API/app/core/DB_Management/sqlite_policy.py`
- Inspect: `tldw_Server_API/app/core/DB_Management/backends/pg_rls_policies.py`
- Inspect: `tldw_Server_API/app/core/DB_Management/migration_tools.py`
- Inspect: `tldw_Server_API/app/core/DB_Management/db_migration.py`
- Inspect: `tldw_Server_API/app/core/DB_Management/migrate_db.py`
- Inspect: `tldw_Server_API/app/core/DB_Management/migrations.py`
- Inspect: `tldw_Server_API/app/core/DB_Management/content_migrate.py`
- Inspect: `tldw_Server_API/app/core/DB_Management/DB_Backups.py`
- Test: `tldw_Server_API/tests/DB_Management/test_db_path_utils.py`
- Test: `tldw_Server_API/tests/DB_Management/test_db_path_utils_env.py`
- Test: `tldw_Server_API/tests/DB_Management/test_db_paths_media_prompts_env.py`
- Test: `tldw_Server_API/tests/DB_Management/test_research_db_paths.py`
- Test: `tldw_Server_API/tests/DB_Management/test_output_storage_normalization.py`
- Test: `tldw_Server_API/tests/DB_Management/test_sqlite_policy.py`
- Test: `tldw_Server_API/tests/DB_Management/test_sqlite_policy_integrations.py`
- Test: `tldw_Server_API/tests/DB_Management/test_db_migration_loader.py`
- Test: `tldw_Server_API/tests/DB_Management/test_db_migration_path_validation.py`
- Test: `tldw_Server_API/tests/DB_Management/test_migration_cli_integration.py`
- Test: `tldw_Server_API/tests/DB_Management/test_migration_tools.py`
- Test: `tldw_Server_API/tests/DB_Management/test_backup_restore_verification.py`
- Test: `tldw_Server_API/tests/DB_Management/test_db_backup_integrity.py`
- Test: `tldw_Server_API/tests/DB_Management/test_db_backup_name_validation.py`

- [ ] **Step 1: Map the path, scope, migration, and backup entry points**

Run:
```bash
rg -n "class |def (get_|resolve_|build_|validate_|apply_|run_|migrate|rollback|backup|restore)|PRAGMA|ALTER TABLE|CREATE POLICY|row level security|scope" \
  tldw_Server_API/app/core/DB_Management/db_path_utils.py \
  tldw_Server_API/app/core/DB_Management/scope_context.py \
  tldw_Server_API/app/core/DB_Management/sqlite_policy.py \
  tldw_Server_API/app/core/DB_Management/backends/pg_rls_policies.py \
  tldw_Server_API/app/core/DB_Management/migration_tools.py \
  tldw_Server_API/app/core/DB_Management/db_migration.py \
  tldw_Server_API/app/core/DB_Management/migrate_db.py \
  tldw_Server_API/app/core/DB_Management/migrations.py \
  tldw_Server_API/app/core/DB_Management/content_migrate.py \
  tldw_Server_API/app/core/DB_Management/DB_Backups.py
```

Expected: a concise list of the filesystem, scope, migration, and backup control points to trace in the stage report.

- [ ] **Step 2: Trace per-user path containment and scope propagation**

Confirm:
- how per-user roots are resolved and normalized
- where path traversal or cross-root writes are blocked
- how request scope is stored and propagated
- where SQLite policy emulation or PostgreSQL RLS assumptions can silently diverge

- [ ] **Step 3: Trace migration and backup safety invariants**

Confirm:
- how schema versions are discovered and advanced
- whether migration scripts are loaded and validated safely
- how partial migration failure is surfaced
- whether backup naming, backup containment, and restore verification can fail open

- [ ] **Step 4: Review the targeted tests and extract the protected invariants**

For each listed test file, record:
- which path, scope, migration, or backup invariant it actually checks
- whether the test covers only the happy path or also the unsafe or malformed path
- whether any important negative case is still untested

- [ ] **Step 5: Run the path and policy tests**

Run:
```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/DB_Management/test_db_path_utils.py \
  tldw_Server_API/tests/DB_Management/test_db_path_utils_env.py \
  tldw_Server_API/tests/DB_Management/test_db_paths_media_prompts_env.py \
  tldw_Server_API/tests/DB_Management/test_research_db_paths.py \
  tldw_Server_API/tests/DB_Management/test_output_storage_normalization.py \
  tldw_Server_API/tests/DB_Management/test_sqlite_policy.py \
  tldw_Server_API/tests/DB_Management/test_sqlite_policy_integrations.py -v
```

Expected: tests pass or fail in ways that directly sharpen the path and policy findings.

- [ ] **Step 6: Run the migration and backup tests**

Run:
```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/DB_Management/test_db_migration_loader.py \
  tldw_Server_API/tests/DB_Management/test_db_migration_path_validation.py \
  tldw_Server_API/tests/DB_Management/test_migration_cli_integration.py \
  tldw_Server_API/tests/DB_Management/test_migration_tools.py \
  tldw_Server_API/tests/DB_Management/test_backup_restore_verification.py \
  tldw_Server_API/tests/DB_Management/test_db_backup_integrity.py \
  tldw_Server_API/tests/DB_Management/test_db_backup_name_validation.py -v
```

Expected: tests collect and pass; any failure either confirms a migration or backup issue or must be explained as environment noise.

- [ ] **Step 7: Write the Stage 3 report**

Record:
- ranked findings with file references
- whether the issue is primarily correctness, isolation, or maintainability
- test coverage gaps that materially weaken confidence
- the exit note for what Stage 4 must verify inside `media_db`

- [ ] **Step 8: Commit the Stage 3 report**

Run:
```bash
git add Docs/superpowers/reviews/db-management/2026-04-07-stage3-paths-tenancy-migrations-backups.md
git commit -m "docs: record DB_Management path and migration findings"
```

Expected: one docs-only commit captures the Stage 3 report.

### Task 4: Execute Stage 4 Media DB Core Runtime and Representative Helper Review

**Files:**
- Modify: `Docs/superpowers/reviews/db-management/2026-04-07-stage4-media-db-and-representative-helpers.md`
- Inspect: `tldw_Server_API/app/core/DB_Management/media_db/api.py`
- Inspect: `tldw_Server_API/app/core/DB_Management/media_db/native_class.py`
- Inspect: `tldw_Server_API/app/core/DB_Management/media_db/media_database.py`
- Inspect: `tldw_Server_API/app/core/DB_Management/media_db/media_database_impl.py`
- Inspect: `tldw_Server_API/app/core/DB_Management/media_db/schema/bootstrap.py`
- Inspect: `tldw_Server_API/app/core/DB_Management/media_db/schema/migrations.py`
- Inspect: `tldw_Server_API/app/core/DB_Management/media_db/runtime/connection_lifecycle.py`
- Inspect: `tldw_Server_API/app/core/DB_Management/media_db/runtime/backend_resolution.py`
- Inspect: `tldw_Server_API/app/core/DB_Management/media_db/runtime/backend_prepare_ops.py`
- Inspect: `tldw_Server_API/app/core/DB_Management/media_db/runtime/scope_resolution_ops.py`
- Inspect: `tldw_Server_API/app/core/DB_Management/media_db/runtime/session.py`
- Inspect: `tldw_Server_API/app/core/DB_Management/media_db/runtime/execution.py`
- Inspect: `tldw_Server_API/app/core/DB_Management/media_db/runtime/execution_ops.py`
- Inspect: `tldw_Server_API/app/core/DB_Management/media_db/runtime/bootstrap_lifecycle_ops.py`
- Inspect: `tldw_Server_API/app/core/DB_Management/media_db/runtime/sqlite_bootstrap.py`
- Inspect: `tldw_Server_API/app/core/DB_Management/UserDatabase_v2.py`
- Inspect: `tldw_Server_API/app/core/DB_Management/TopicMonitoring_DB.py`
- Inspect: `tldw_Server_API/app/core/DB_Management/Voice_Registry_DB.py`
- Inspect: `tldw_Server_API/app/core/DB_Management/Workflows_Scheduler_DB.py`
- Inspect: `tldw_Server_API/app/core/DB_Management/watchlist_alert_rules_db.py`
- Test: `tldw_Server_API/tests/DB_Management/test_media_db_api_imports.py`
- Test: `tldw_Server_API/tests/DB_Management/test_media_db_bootstrap_lifecycle_ops.py`
- Test: `tldw_Server_API/tests/DB_Management/test_media_db_connection_cleanup.py`
- Test: `tldw_Server_API/tests/DB_Management/test_media_db_runtime_factory.py`
- Test: `tldw_Server_API/tests/DB_Management/test_media_db_schema_bootstrap.py`
- Test: `tldw_Server_API/tests/DB_Management/test_media_db_scope_resolution_ops.py`
- Test: `tldw_Server_API/tests/DB_Management/test_media_db_request_scope_isolation.py`
- Test: `tldw_Server_API/tests/DB_Management/test_media_db_core_repositories.py`
- Test: `tldw_Server_API/tests/DB_Management/test_media_db_v2_regressions.py`
- Test: `tldw_Server_API/tests/DB_Management/test_media_db_migration_missing_scripts_error.py`
- Test: `tldw_Server_API/tests/DB_Management/test_media_db_postgres_rls_ops.py`
- Test: `tldw_Server_API/tests/DB_Management/test_media_postgres_migrations.py`
- Test: `tldw_Server_API/tests/DB_Management/test_media_postgres_support.py`
- Test: `tldw_Server_API/tests/DB_Management/test_users_db_sqlite.py`
- Test: `tldw_Server_API/tests/DB_Management/unit/test_users_db_update_backend_detection.py`
- Test: `tldw_Server_API/tests/DB_Management/test_workflows_scheduler_db_paths.py`
- Test: `tldw_Server_API/tests/AuthNZ/integration/test_authnz_backends_pytest.py`
- Test: `tldw_Server_API/tests/test_watchlist_alert_rules.py`
- Test: `tldw_Server_API/tests/Claims/test_claims_service_backend_selection.py`
- Test: `tldw_Server_API/tests/Claims/test_ingestion_claims_sql.py`
- Test: `tldw_Server_API/tests/Media/test_media_reprocess_endpoint.py`
- Test: `tldw_Server_API/tests/test_utils.py`

- [ ] **Step 1: Map `media_db` bootstrap, session, scope, and execution entry points**

Run:
```bash
rg -n "def (create_|managed_|get_|bootstrap|initialize|resolve_|prepare_|execute|session|transaction|close|cleanup)|class " \
  tldw_Server_API/app/core/DB_Management/media_db/api.py \
  tldw_Server_API/app/core/DB_Management/media_db/native_class.py \
  tldw_Server_API/app/core/DB_Management/media_db/media_database.py \
  tldw_Server_API/app/core/DB_Management/media_db/media_database_impl.py \
  tldw_Server_API/app/core/DB_Management/media_db/schema/bootstrap.py \
  tldw_Server_API/app/core/DB_Management/media_db/schema/migrations.py \
  tldw_Server_API/app/core/DB_Management/media_db/runtime/connection_lifecycle.py \
  tldw_Server_API/app/core/DB_Management/media_db/runtime/backend_resolution.py \
  tldw_Server_API/app/core/DB_Management/media_db/runtime/backend_prepare_ops.py \
  tldw_Server_API/app/core/DB_Management/media_db/runtime/scope_resolution_ops.py \
  tldw_Server_API/app/core/DB_Management/media_db/runtime/session.py \
  tldw_Server_API/app/core/DB_Management/media_db/runtime/execution.py \
  tldw_Server_API/app/core/DB_Management/media_db/runtime/execution_ops.py \
  tldw_Server_API/app/core/DB_Management/media_db/runtime/bootstrap_lifecycle_ops.py \
  tldw_Server_API/app/core/DB_Management/media_db/runtime/sqlite_bootstrap.py
```

Expected: a clear map of the `media_db` control flow from factory entrypoint to runtime execution and cleanup.

- [ ] **Step 2: Trace `media_db` request isolation, backend resolution, and bootstrap behavior**

Confirm:
- how request scope reaches runtime execution
- where backend resolution can change semantics
- how schema bootstrap and migration paths are invoked
- whether connection cleanup and lifecycle rules can leak state or hide failure

- [ ] **Step 3: Inspect representative non-`media_db` helpers with custom schema or filesystem logic**

For `UserDatabase_v2.py`, `TopicMonitoring_DB.py`, `Voice_Registry_DB.py`, `Workflows_Scheduler_DB.py`, and `watchlist_alert_rules_db.py`, record:
- whether the module bypasses shared abstractions
- whether it manages transactions or schema changes manually
- whether it introduces path, isolation, or SQL safety risks not covered by shared layers

- [ ] **Step 4: Review the targeted tests and caller-level checks**

For each listed test file, record:
- the primary invariant it covers
- whether it validates direct DB behavior or only a caller assumption
- which representative helper or `media_db` path it meaningfully constrains

- [ ] **Step 5: Run the core `media_db` tests**

Run:
```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/DB_Management/test_media_db_api_imports.py \
  tldw_Server_API/tests/DB_Management/test_media_db_bootstrap_lifecycle_ops.py \
  tldw_Server_API/tests/DB_Management/test_media_db_connection_cleanup.py \
  tldw_Server_API/tests/DB_Management/test_media_db_runtime_factory.py \
  tldw_Server_API/tests/DB_Management/test_media_db_schema_bootstrap.py \
  tldw_Server_API/tests/DB_Management/test_media_db_scope_resolution_ops.py \
  tldw_Server_API/tests/DB_Management/test_media_db_request_scope_isolation.py \
  tldw_Server_API/tests/DB_Management/test_media_db_core_repositories.py \
  tldw_Server_API/tests/DB_Management/test_media_db_v2_regressions.py \
  tldw_Server_API/tests/DB_Management/test_media_db_migration_missing_scripts_error.py \
  tldw_Server_API/tests/DB_Management/test_media_db_postgres_rls_ops.py \
  tldw_Server_API/tests/DB_Management/test_media_postgres_migrations.py \
  tldw_Server_API/tests/DB_Management/test_media_postgres_support.py -v
```

Expected: tests collect and mostly pass; failures either validate an audit concern or must be explained as environment-specific.

- [ ] **Step 6: Run the representative helper and caller-level tests**

Run:
```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/DB_Management/test_users_db_sqlite.py \
  tldw_Server_API/tests/DB_Management/unit/test_users_db_update_backend_detection.py \
  tldw_Server_API/tests/DB_Management/test_workflows_scheduler_db_paths.py \
  tldw_Server_API/tests/AuthNZ/integration/test_authnz_backends_pytest.py \
  tldw_Server_API/tests/test_watchlist_alert_rules.py \
  tldw_Server_API/tests/Claims/test_claims_service_backend_selection.py \
  tldw_Server_API/tests/Claims/test_ingestion_claims_sql.py \
  tldw_Server_API/tests/Media/test_media_reprocess_endpoint.py \
  tldw_Server_API/tests/test_utils.py -v
```

Expected: tests sharpen whether the representative modules behave safely from caller-facing entry points.

- [ ] **Step 7: Write the Stage 4 report**

Record:
- ranked `media_db` and representative-helper findings
- caller-level evidence that confirms or weakens the claim
- coverage gaps worth carrying into the final synthesis
- the exit note for any remaining high-risk area that still needs a spot check in Stage 5

- [ ] **Step 8: Commit the Stage 4 report**

Run:
```bash
git add Docs/superpowers/reviews/db-management/2026-04-07-stage4-media-db-and-representative-helpers.md
git commit -m "docs: record DB_Management media and helper review findings"
```

Expected: one docs-only commit captures the Stage 4 report.

### Task 5: Execute Stage 5 Test-Gap Pass and Final Synthesis

**Files:**
- Modify: `Docs/superpowers/reviews/db-management/2026-04-07-stage5-test-gaps-and-synthesis.md`
- Inspect: `Docs/superpowers/reviews/db-management/2026-04-07-stage2-foundations-backends-factories.md`
- Inspect: `Docs/superpowers/reviews/db-management/2026-04-07-stage3-paths-tenancy-migrations-backups.md`
- Inspect: `Docs/superpowers/reviews/db-management/2026-04-07-stage4-media-db-and-representative-helpers.md`
- Inspect: `tldw_Server_API/tests/DB_Management`
- Test: re-run only the narrowest disputed commands from earlier stages if a finding still lacks enough evidence

- [ ] **Step 1: Build a coverage matrix from the full DB_Management test inventory**

Run:
```bash
source .venv/bin/activate
rg --files tldw_Server_API/tests/DB_Management | sort
```

Expected: a complete local inventory that can be compared against the code paths and tests already reviewed in Stages 2 through 4.

- [ ] **Step 2: Record the highest-value missing or weak invariants**

Capture:
- important shared abstractions with little or no direct test coverage
- backend-sensitive paths where tests exist but do not exercise divergence or failure behavior
- areas where tests are caller-level smoke checks rather than true invariant checks

- [ ] **Step 3: Re-run only the narrowest disputed validations if needed**

Use the smallest earlier command or single-test selector that can settle a contested claim. If no disputed claims remain, explicitly record `No additional validation required` in the stage report.

- [ ] **Step 4: Write the final synthesis report**

Use this exact section order:
```markdown
# DB_Management Review Synthesis

## Findings
1. Severity: concise issue statement with file references and impact
2. Severity: concise issue statement with file references and impact

## Open Questions
- only unresolved assumptions that materially affect confidence

## Test Gaps
- highest-value missing invariants first
```

Each finding must state:
- severity
- confidence
- whether it is a confirmed finding, probable risk, or improvement
- why it matters across correctness, isolation, or maintainability
- the exact source references that support it

- [ ] **Step 5: Deduplicate and rank findings before the final response**

Remove overlaps across the stage reports so the final answer:
- leads with the most serious issues first
- does not repeat the same root cause under different filenames
- keeps improvements clearly secondary to defects and probable risks

- [ ] **Step 6: Commit the final synthesis report**

Run:
```bash
git add Docs/superpowers/reviews/db-management/2026-04-07-stage5-test-gaps-and-synthesis.md
git commit -m "docs: record DB_Management review synthesis"
```

Expected: one docs-only commit captures the final stage artifact.

- [ ] **Step 7: Deliver the in-session review response**

Use the final synthesis report to produce the user-facing response:
- findings first, ordered by severity
- file references on every substantive item
- brief open questions only if they materially affect confidence
- no remediation plan unless the user asks for it afterward
