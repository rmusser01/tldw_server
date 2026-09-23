# DB, Migrations, And Data Durability Domain Review

## Scope

- Baseline: `origin/dev` at `669092178b0ba0fa1e840a37250b0deb55acd5a3`
- Report owner: DB, Migrations, and Data Durability
- In scope: SQLite/Postgres behavior, migrations, path resolution, transaction patterns, soft-delete/versioning assumptions, sync logs, and DB-focused tests.
- Out of scope: remediation implementation and schema feature additions.

## Findings Table

| ID | Evidence Tier | Evidence Strength | Severity | Confidence | Category | Title | Status | Validation Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CANDIDATE-db-migrations-data-durability-001 | confirmed_issue | runtime_reproduced | high | high | data_durability | SQLite Media DB upgrades before v22 cannot reach current schema from packaged migrations | open | validated |
| CANDIDATE-db-migrations-data-durability-002 | confirmed_issue | runtime_reproduced | medium | high | data_durability | Generic SQLite migrations can leave partial DDL after a failed multi-statement script | open | validated |

## Index Mapping

| Candidate ID | Proposed Index ID | source_report | owner_domain |
| --- | --- | --- | --- |
| CANDIDATE-db-migrations-data-durability-001 | AUDIT-2026-06-27-DB-001 | `Docs/superpowers/reviews/2026-06-27-repo-audit/domains/db-migrations-data-durability.md` | DB, Migrations, and Data Durability |
| CANDIDATE-db-migrations-data-durability-002 | AUDIT-2026-06-27-DB-002 | `Docs/superpowers/reviews/2026-06-27-repo-audit/domains/db-migrations-data-durability.md` | DB, Migrations, and Data Durability |

## Confirmed Issues

### CANDIDATE-db-migrations-data-durability-001 - SQLite Media DB upgrades before v22 cannot reach current schema from packaged migrations

- Evidence tier: confirmed_issue
- Evidence strength: runtime_reproduced
- Severity: high
- Confidence: high
- Category: data_durability
- Status: open
- Validation status: validated
- Affected paths:
  - `tldw_Server_API/app/core/DB_Management/media_db/schema/backends/sqlite_helpers.py:209`
  - `tldw_Server_API/app/core/DB_Management/media_db/schema/backends/sqlite_helpers.py:236`
  - `tldw_Server_API/app/core/DB_Management/db_migration.py:113`
  - `tldw_Server_API/app/core/DB_Management/db_migration.py:120`
  - `tldw_Server_API/app/core/DB_Management/db_migration.py:561`
  - `tldw_Server_API/app/core/DB_Management/db_migration.py:596`
  - `tldw_Server_API/app/core/DB_Management/migrations/001_prompt_studio_schema.sql:1`
  - `tldw_Server_API/app/core/DB_Management/migrations/006_prompt_studio_structured_prompts.sql:1`
  - `tldw_Server_API/app/core/DB_Management/migrations/023_transcript_run_history.sql:1`
- Evidence:
  - `bootstrap_sqlite_schema()` sends existing file-backed SQLite Media DBs below the current schema version through `DatabaseMigrator`.
  - `DatabaseMigrator` defaults to the package-level `app/core/DB_Management/migrations` directory when no directory is provided.
  - That directory currently contains versions 1-4 and 6 for Prompt Studio, version 5 for `ChunkingTemplates`, and version 23 for Media DB transcript run history. It does not contain a contiguous Media DB chain for versions 9-22.
  - Runtime reproduction in `Docs/superpowers/reviews/2026-06-27-repo-audit/evidence/db-migrations-data-durability-reproductions.txt` created a file-backed DB with `schema_version = 8` and called `migrate_to_version(23, create_backup=False)`. The migrator raised `MigrationError Missing migration versions: [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]`.
  - Existing coverage includes `test_on_disk_sqlite_migration_to_v23_backfills_transcript_run_history`, which validates v22-to-v23 only; it does not cover representative older Media DB versions.
- Impact:
  - Users with older SQLite Media DBs cannot be automatically upgraded to the current schema. Startup or first DB access can fail before the application reaches normal media operations.
  - The failure is data-preserving in the reproduced case, but it leaves older user data unavailable without manual intervention or a custom recovery path.
- Recommendation:
  - Split package migrations by database/domain, or otherwise scope `DatabaseMigrator` so Media DB upgrades do not consume Prompt Studio versions.
  - Provide a contiguous supported Media DB migration chain, or explicitly reject unsupported legacy versions with a documented backup/export/rebuild workflow.
  - Add real package-directory upgrade tests for representative legacy versions, including at least one version older than v22.

### CANDIDATE-db-migrations-data-durability-002 - Generic SQLite migrations can leave partial DDL after a failed multi-statement script

- Evidence tier: confirmed_issue
- Evidence strength: runtime_reproduced
- Severity: medium
- Confidence: high
- Category: data_durability
- Status: open
- Validation status: validated
- Affected paths:
  - `tldw_Server_API/app/core/DB_Management/db_migration.py:432`
  - `tldw_Server_API/app/core/DB_Management/db_migration.py:458`
  - `tldw_Server_API/app/core/DB_Management/db_migration.py:465`
  - `tldw_Server_API/app/core/DB_Management/db_migration.py:471`
  - `tldw_Server_API/app/core/DB_Management/db_migration.py:488`
  - `tldw_Server_API/app/core/DB_Management/db_migration.py:493`
  - `tldw_Server_API/app/core/DB_Management/db_migration.py:510`
- Evidence:
  - `execute_migration()` deletes any previous failed row and commits before applying the migration.
  - For multi-statement SQL, it calls `conn.executescript(sql)`, then commits the migration SQL, then separately inserts `schema_migrations`, commits again, then updates `schema_version` and commits again.
  - Runtime reproduction in `Docs/superpowers/reviews/2026-06-27-repo-audit/evidence/db-migrations-data-durability-reproductions.txt` applied a two-statement migration where the first statement created `kept_after_failure` and the second statement inserted into a missing table. The migrator raised a failure and recorded the migration as failed, but `kept_after_failure` remained in `sqlite_master`.
  - The focused migration tests pass, but they do not assert rollback of a failing multi-statement script after a successful first DDL statement.
- Impact:
  - A failed migration can leave an altered schema while the migration ledger says the version failed. Retrying may then hit "already exists" errors or, worse, proceed against a partially mutated schema.
  - Successful migrations also have crash windows between SQL application, `schema_migrations` insertion, and `schema_version` update.
- Recommendation:
  - Execute each migration, success ledger insert/delete, and `schema_version` update inside one explicit transaction boundary, preferably `BEGIN IMMEDIATE`.
  - Normalize migration files so the migrator owns transaction control, or reject/handle files with embedded transaction control explicitly.
  - Add regression coverage for a failing multi-statement script that must leave no partial DDL behind.

## Likely Risks

No likely-risk candidate is promoted from this pass. Residual concerns that should be revisited during remediation:

- AuthNZ SQLite migration functions commonly call `conn.commit()` internally while `MigrationManager.migrate()` also wraps each migration in a `BEGIN`/ledger/`COMMIT` sequence (`tldw_Server_API/app/core/DB_Management/migrations.py:120`, `tldw_Server_API/app/core/AuthNZ/migrations.py:75`, `tldw_Server_API/app/core/AuthNZ/migrations.py:138`, `tldw_Server_API/app/core/AuthNZ/migrations.py:209`). I did not promote this as a confirmed candidate because I did not build a failure reproduction against a real AuthNZ migration, but it has the same durability shape as the generic migration issue.
- Scheduler migration CLI behavior was statically inspected only. It appears to use a schema-validation task enqueue rather than a traditional schema migration, so it should receive separate owner review before being treated as a data migration surface.

## Improvement Opportunities

No separate improvement-opportunity findings are recorded. The main improvement is covered by the recommendations for the two confirmed issues: domain-scoped migration registries, contiguous legacy upgrade coverage, and single-transaction migration application.

## Coverage And Evidence

### Files Inspected

- `Docs/superpowers/reviews/2026-06-27-repo-audit/inventory.md`
- `Docs/superpowers/reviews/2026-06-27-repo-audit/findings-index.json`
- `Docs/superpowers/reviews/2026-06-27-repo-audit/evidence/db-migration-inventory.txt`
- `Docs/superpowers/reviews/2026-06-27-repo-audit/evidence/backend-test-inventory.txt`
- `Docs/superpowers/reviews/2026-06-27-repo-audit/evidence/bandit-app-summary.txt`
- `tldw_Server_API/app/core/DB_Management/db_migration.py`
- `tldw_Server_API/app/core/DB_Management/migrations.py`
- `tldw_Server_API/app/core/DB_Management/DB_Backups.py`
- `tldw_Server_API/app/core/DB_Management/transaction_utils.py`
- `tldw_Server_API/app/core/DB_Management/sqlite_policy.py`
- `tldw_Server_API/app/core/DB_Management/backends/sqlite_backend.py`
- `tldw_Server_API/app/core/DB_Management/backends/postgresql_backend.py`
- `tldw_Server_API/app/core/DB_Management/media_db/schema/backends/sqlite_helpers.py`
- `tldw_Server_API/app/core/DB_Management/media_db/schema/backends/postgres_helpers.py`
- `tldw_Server_API/app/core/DB_Management/media_db/schema/migrations.py`
- `tldw_Server_API/app/core/DB_Management/media_db/runtime/connection_lifecycle.py`
- `tldw_Server_API/app/core/DB_Management/media_db/runtime/sqlite_bootstrap.py`
- `tldw_Server_API/app/core/DB_Management/migrations/*.sql`
- `tldw_Server_API/app/core/AuthNZ/migrations.py`
- `tldw_Server_API/app/core/AuthNZ/database.py`
- `tldw_Server_API/Databases/SQLite/Schema/sqlite_users.sql`
- `tldw_Server_API/Databases/Postgres/Schema/postgresql_users.sql`
- `tldw_Server_API/app/core/Scheduler/migrations/migrate.py`
- `tldw_Server_API/tests/DB_Management/test_db_migration_planning.py`
- `tldw_Server_API/tests/DB_Management/test_db_migration_verification.py`
- `tldw_Server_API/tests/DB_Management/test_db_backup_integrity.py`
- `tldw_Server_API/tests/DB_Management/test_media_db_schema_bootstrap.py`
- `tldw_Server_API/tests/MediaDB2/test_sqlite_db.py`

### Inspection Commands Run

- `git status -sb` and `git status --short`
- `find Docs/superpowers/reviews/2026-06-27-repo-audit -maxdepth 2 -type f | sort`
- `rg --files tldw_Server_API/app/core/DB_Management`
- `sed -n ...` on the assigned context files, report scaffold, DB management modules, AuthNZ schemas, scheduler migration entrypoint, and relevant tests.
- `nl -ba ... | sed -n ...` on `db_migration.py`, `sqlite_helpers.py`, package SQL migrations, Media DB bootstrap tests, `migrations.py`, and `AuthNZ/migrations.py` to capture exact line references.
- `git diff -- Docs/superpowers/reviews/2026-06-27-repo-audit/domains/db-migrations-data-durability.md Docs/superpowers/reviews/2026-06-27-repo-audit/evidence/db-migrations-data-durability-reproductions.txt`
- `git diff --check -- Docs/superpowers/reviews/2026-06-27-repo-audit/domains/db-migrations-data-durability.md`
- `git diff --no-index --check /dev/null Docs/superpowers/reviews/2026-06-27-repo-audit/evidence/db-migrations-data-durability-reproductions.txt`
- `git diff --no-index --check /dev/null "backlog/tasks/task-12055 - Conduct-DB-migrations-and-data-durability-domain-audit-report.md"`
- `wc -l Docs/superpowers/reviews/2026-06-27-repo-audit/domains/db-migrations-data-durability.md Docs/superpowers/reviews/2026-06-27-repo-audit/evidence/db-migrations-data-durability-reproductions.txt "backlog/tasks/task-12055 - Conduct-DB-migrations-and-data-durability-domain-audit-report.md"`

### Tests Or Scans Run

- `LOGURU_LEVEL=ERROR /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/DB_Management/test_db_migration_planning.py tldw_Server_API/tests/DB_Management/test_db_migration_verification.py tldw_Server_API/tests/DB_Management/test_db_backup_integrity.py -q`
  - Result: 25 passed, 57 warnings.
- `LOGURU_LEVEL=ERROR /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/DB_Management/test_media_db_schema_bootstrap.py::test_on_disk_sqlite_migration_to_v23_backfills_transcript_run_history -q`
  - Result: 1 passed, 9 warnings.
- Reproduction command for `CANDIDATE-db-migrations-data-durability-001` using a temporary SQLite DB with `schema_version = 8`.
  - Result: `MigrationError Missing migration versions: [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]`.
- Reproduction command for `CANDIDATE-db-migrations-data-durability-002` using a temporary SQLite DB and a failing two-statement migration.
  - Result: failed migration was recorded, but table `kept_after_failure` remained present.
- Reviewed coordinator Bandit summary at `Docs/superpowers/reviews/2026-06-27-repo-audit/evidence/bandit-app-summary.txt`.
- `/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m bandit -r Docs/superpowers/reviews/2026-06-27-repo-audit/domains/db-migrations-data-durability.md Docs/superpowers/reviews/2026-06-27-repo-audit/evidence/db-migrations-data-durability-reproductions.txt "backlog/tasks/task-12055 - Conduct-DB-migrations-and-data-durability-domain-audit-report.md" -f json -o /tmp/bandit_db_migrations_data_durability.json`
  - Result: 0 findings; Bandit reported AST parse errors because the touched files are markdown/text, not Python source.

### Blocked Or Unverified Areas

- No Docker, services, network access, or dependency installation were used per coordinator rules.
- PostgreSQL behavior was reviewed statically and through existing test inventory only; I did not start or connect to a PostgreSQL service.
- I did not run the full repository test suite. Verification was scoped to migration planning, migration verification, backup integrity, and the specific Media DB v22-to-v23 migration test.
- The worktree did not have its own `.venv`; commands used the parent repository virtual environment at `/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python`, matching the audit inventory note.
- Bandit was not re-run over production code because this domain agent did not edit production/source code. The coordinator-provided app Bandit summary was reviewed instead, and the docs-only touched-scope invocation is noted above.

### Evidence Notes

- Reproduction details and observed outputs are saved in `Docs/superpowers/reviews/2026-06-27-repo-audit/evidence/db-migrations-data-durability-reproductions.txt`.
- Backup/recovery code and tests showed meaningful safeguards: SQLite backup API usage, restore preflight checks, rollback handling, and coverage for missing sources, URI paths, quoted paths, busy targets, and WAL-backed backup/restore behavior. No backup/recovery candidate finding was identified in the inspected scope.
- SQLite and PostgreSQL backend transaction helpers use explicit transaction handling in the inspected runtime paths. No separate backend transaction candidate finding was identified outside the generic migration executor issue.
- Soft-delete/versioning coverage was sampled through Media DB schema, migration, and DB-focused tests. No distinct soft-delete/versioning candidate finding was identified in this pass.
