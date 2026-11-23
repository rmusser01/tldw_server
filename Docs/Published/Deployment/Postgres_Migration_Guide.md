
# PostgreSQL Content Migration Guide

This guide covers the PostgreSQL ChaChaNotes schema upgrade and the separate
offline SQLite-to-PostgreSQL copy for Media. The bulk importer
(`migration_tools.py`) also has analytics and workflow targets. It does not
support copying a normal ChaChaNotes database with cyclic foreign keys. Keep
the SQLite original intact; the in-place SQLite schema upgrade is not an
engine conversion.

## Prerequisites

- Back up all SQLite databases (`<USER_DB_BASE_DIR>/<user_id>/<content-db>.db`, `Databases/workflows.db`, `<USER_DB_BASE_DIR>/<user_id>/ChaChaNotes.db`, `Databases/Analytics.db`). If you have multiple users, repeat for each `<user_id>`.
- Install PostgreSQL and ensure the target database is accessible (local host or remote).
- Install the Python dependency `psycopg` (listed under pyproject extras, e.g., `.[multiplayer]`). For convenience use the binary extra:
  - pip install "psycopg[binary]"
- Prepare the target PostgreSQL schemas in maintenance before copying data and keep the API offline
  during the copy. Run the ChaChaNotes schema upgrade as its owner, not as the restricted runtime
  role, as described below.
- Prepare Media separately using the application's `DB_Manager.create_media_database` initializer
  with the PostgreSQL content configuration pointed at the target and the Media schema owner.
  Verify the Media tables and constraints exist before copying. The policy installer below does
  not create them, and the bulk copier expects them to exist.

`USER_DB_BASE_DIR` is defined in `tldw_Server_API.app.core.config` (defaults to `Databases/user_databases/` under the project root). Override via environment variable or `Config_Files/config.txt` as needed.
Replace `<content-db>.db` with your configured per-user content DB filename in the examples below.

### ChaChaNotes schema 92 with a separate runtime role

With the API offline, run the repository's PostgreSQL policy installer
(`python -m Helper_Scripts.DB-Related.pg_apply_rls_policies --apply`) using the
maintenance shell configuration for the intended content database. This initializes or
upgrades both Prompt Studio and ChaChaNotes before applying row policies.
Use an authorized schema owner for every affected table, including any
existing Media claims publication tables. Check the configured target first:
`DATABASE_URL` takes precedence over the individual database settings for this
command. Do not depend on per-user API warm-up to upgrade the schema. Confirm
`SELECT version FROM db_schema_version WHERE schema_name = 'rag_char_chat_schema';`
returns 92 in the ChaChaNotes schema before proceeding.

If the API connects as a separate role, the owner must grant that existing role
access to the new message insertion-order ledger and its owned sequence. Use the
schema that contains the ChaChaNotes tables and replace `content_runtime` below
with the deployment's existing runtime role:

```sql
GRANT SELECT (id, client_id) ON messages TO content_runtime;
GRANT SELECT ON message_insertion_order TO content_runtime;
GRANT INSERT (message_id) ON message_insertion_order TO content_runtime;
GRANT USAGE ON SEQUENCE message_insertion_order_sequence_seq TO content_runtime;
```

Grant only the `SELECT` permissions for a read-only role. Writers must also retain their existing
permission to select and lock their conversation rows. The ledger's forced
owner-scoped row policies remain enabled. Do not grant access to `PUBLIC` or
give the runtime role schema ownership. Check a Chat write as the runtime role
before restoring traffic.

## Step 1 - Prepare connection details

Collect the PostgreSQL host, port, database, and credentials authorized for the
offline copy. These map to the import CLI arguments. The importer deletes existing
rows and updates sequences; its permissions are different from the API runtime
grants above. It does not establish per-tenant ChaChaNotes session context. Use it
only for an offline target where the configured maintenance login is already
authorized for the copy. Do not disable row policies or broaden the API login to
make the import work.

```bash
export PGHOST=localhost
export PGPORT=5432
export PGDATABASE=tldw_content
export PGUSER=tldw_migration
export PGPASSWORD='<migration password>'
```

## Step 2 - Run the migration utility

For the Media-only copy, pass the source file as `--content-sqlite`. Do not
pass `--chacha-sqlite` for a standard installation: the generic importer
rejects cyclic table dependencies such as the references between
conversations and messages. If the importer rejects the Media source schema,
stop and retain the source. Do not remove tables or constraints to force it.

The example imports one source into a dedicated target. Do not repeat it for
multiple users against the same target: without a tenant-scoped import procedure,
the next copy can delete rows loaded by the previous one.

```bash
user_path="${USER_DB_BASE_DIR:?set the source DB directory}/${MIGRATION_USER_ID:?set the source user ID}"
content_db="${user_path}/${CONTENT_DB_NAME:?set the source content DB filename}"
if [ ! -r "$content_db" ]; then
  echo "Error: the source content database must be readable." >&2
  exit 1
fi

python -m tldw_Server_API.app.core.DB_Management.migration_tools \
  --content-sqlite "$content_db" \
  --pg-host "$PGHOST" \
  --pg-port "$PGPORT" \
  --pg-database "$PGDATABASE" \
  --pg-user "$PGUSER" \
  --pg-password "$PGPASSWORD" \
  --skip-table db_schema_version \
  --skip-table schema_version \
  --batch-size 500
```

The script performs the following actions for each supplied database:

1. Reads table definitions from SQLite, respecting foreign-key dependencies.
2. Deletes existing rows in PostgreSQL (in dependency order) so the migration is idempotent.
3. Copies data in batches using the backend abstraction layer.
4. Resyncs sequences so future inserts continue from the correct IDs.
5. Emits a row-count summary for each migrated table (including workflow definitions, runs, steps,
   events, and artifacts) so you can compare the SQLite and Postgres totals at a glance.

Use `--skip-table <table_name>` to omit auxiliary tables (for example, to skip large log tables).
Keep the schema-version tables excluded: source SQLite versions are not
PostgreSQL migration receipts. Keep the target's existing migration records
and verify the PostgreSQL versions again after the copy.

## Step 3 - Validate the migration

- Compare row counts between SQLite and PostgreSQL:

  ```sql
  -- Run in psql
  SELECT COUNT(*) FROM media;
  SELECT COUNT(*) FROM claims;
  SELECT COUNT(*) FROM workflow_runs;
  SELECT COUNT(*) FROM workflow_steps;
  ```

  ```bash
  for user_id in $(ls -1 "$USER_DB_BASE_DIR"); do
    [ -d "$USER_DB_BASE_DIR/$user_id" ] || continue
    echo "User $user_id:"
    sqlite3 "${USER_DB_BASE_DIR}/${user_id}/${CONTENT_DB_NAME}" 'SELECT COUNT(*) FROM media;'
  done
  sqlite3 Databases/workflows.db 'SELECT COUNT(*) FROM workflow_runs;'
  ```

- Before cutover, verify representative migrated records and their owners
  through the restricted runtime in the isolated target. The intended
  principal must see the records; a different tenant must not. Row counts
  alone cannot validate ownership or access.

- Run the dual-backend regression tests to ensure the application-level routines behave correctly (requires a Postgres instance with `POSTGRES_TEST_*` env vars):

  ```bash
  pytest tldw_Server_API/tests/DB_Management/test_media_postgres_support.py \
         tldw_Server_API/tests/DB_Management/test_media_postgres_migrations.py \
         tldw_Server_API/tests/RAG/test_analytics_backend.py \
         tldw_Server_API/tests/RAG/test_dual_backend_rag_flow.py \
         tldw_Server_API/tests/RAG/test_dual_backend_end_to_end.py \
         tldw_Server_API/tests/DB_Management/test_migration_tools.py \
         tldw_Server_API/tests/DB_Management/test_migration_cli_integration.py \
         tldw_Server_API/tests/Workflows/test_workflows_postgres_migrations.py \
         tldw_Server_API/tests/Workflows/test_dual_backend_workflows.py \
         tldw_Server_API/tests/Workflows/test_dual_backend_engine.py
  ```

- Optionally run the workload stress sweep after verifying parity:

  ```bash
  TLDW_WORKFLOW_STRESS=1 pytest -v tldw_Server_API/tests/Workflows/test_workflow_stress.py
  ```
  Latest run (2025-10-08) processed 250 concurrent runs with 1,000 step events per backend in under
  two minutes and finished with zero failures.


## Step 4 - Switch the application configuration

The backend switch is shared by the content stores; the Media-only copy does
not preserve Chats or Notes. Keep the SQLite deployment in place until all
required content and its owner-scoped access are verified on the new backend.

1. Configure the deployment's API runtime credentials in the service environment
   or `Config_Files/config.txt`. Do not launch the API from the import shell or
   inherit its `PGUSER`, `PGPASSWORD`, or migration DSN. Check any configured
   PostgreSQL connection strings as well as the individual fields. Replace or
   remove maintenance values in `TLDW_CONTENT_PG_DSN`, `TLDW_CONTENT_PG_*`, and
   `[Database] pg_connection_string`; these can override the fallback settings:

   ```bash
   export TLDW_CONTENT_DB_BACKEND=postgresql
   export TLDW_PG_HOST="${CONTENT_RUNTIME_PG_HOST:?set the API database host}"
   export TLDW_PG_PORT="${CONTENT_RUNTIME_PG_PORT:?set the API database port}"
   export TLDW_PG_DATABASE="${CONTENT_RUNTIME_PG_DATABASE:?set the API database name}"
   export TLDW_PG_USER="${CONTENT_RUNTIME_PG_USER:?set the API runtime login}"
   export TLDW_PG_PASSWORD="${CONTENT_RUNTIME_PG_PASSWORD:?set the API runtime password}"
   # Optional overrides for workflow connection pooling
    export TLDW_WORKFLOW_DB_POOL_SIZE=15
    export TLDW_WORKFLOW_DB_MAX_OVERFLOW=30
    export TLDW_WORKFLOW_DB_TIMEOUT=30
   ```

2. Retain the original SQLite files and backups until the cutover and rollback
   procedure have been verified.

3. Restart the API/frontend stack. On startup the service should log that it is using PostgreSQL for
   Media, ChaChaNotes, Analytics, and Workflows. Monitor `/api/v1/workflows/status` to ensure the
   runtime heartbeat updates and that queue depth drops as pending runs are processed.

## Troubleshooting

- **Foreign-key errors during copy** - verify the target tables and constraints
  were created during maintenance before copying.
- **Sequence mismatch** - rerun the migration with `--batch-size 1` for the affected database to
  recompute the sequence state, or run the generated `SELECT setval(...)` statements manually.
- **Large datasets** - increase `--batch-size` (default 500) to improve throughput, or run per
  database to isolate issues.
- **Workflow runtime lag** - confirm the connection pool overrides above are applied and check
  `pg_stat_activity` for sessions waiting on locks. Enable the stress suite with reduced batch size
  (set `TLDW_WORKFLOW_STRESS_BATCH=25`) to reproduce issues under controlled load.

## Next steps

- Update deployment scripts to point at PostgreSQL hosts.
- Configure backups for the new PostgreSQL databases.
- Remove legacy cron jobs or tooling that assumed SQLite file copies.
- Add the workflow stress sweep (`TLDW_WORKFLOW_STRESS=1`) to nightly CI or a staging cron job to
  detect regressions early.
