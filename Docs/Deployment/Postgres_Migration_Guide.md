
# PostgreSQL Content Migration Guide

This document explains how to migrate existing SQLite content databases to PostgreSQL using the
built-in migration tooling (`migration_tools.py`). The script copies Media, analytics,
ChaChaNotes, **workflows**, and related metadata while preserving UUIDs, timestamps, and
soft-delete flags so you can switch the backend without re-ingesting content or re-running
workflow jobs.

## Prerequisites

- Back up all SQLite databases (`<USER_DB_BASE_DIR>/<user_id>/<content-db>.db`, `Databases/workflows.db`, `<USER_DB_BASE_DIR>/<user_id>/ChaChaNotes.db`, `Databases/Analytics.db`). If you have multiple users, repeat for each `<user_id>`.
- Install PostgreSQL and ensure the target database is accessible (local host or remote).
- Install the Python dependency `psycopg` (listed under pyproject extras, e.g., `.[multiplayer]`). For convenience use the binary extra:
  - pip install "psycopg[binary]"
- Configure the server once against PostgreSQL so the schema exists. The easiest option is to set
  `TLDW_CONTENT_DB_BACKEND=postgresql` temporarily and start the API; it will create all tables and
  FTS artefacts, then shut the server down before migrating data.

`USER_DB_BASE_DIR` is defined in `tldw_Server_API.app.core.config` (defaults to `Databases/user_databases/` under the project root). Override via environment variable or `Config_Files/config.txt` as needed.
Replace `<content-db>.db` with your configured per-user content DB filename in the examples below.

### ChaChaNotes schema 92 with a separate runtime role

Run the schema upgrade as the content schema owner before restarting the API.
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

Collect the PostgreSQL host, port, database, username, and password. These map directly onto the
CLI arguments and `DatabaseConfig` fields inside the migration utility.

```bash
export PGHOST=localhost
export PGPORT=5432
export PGDATABASE=tldw_content
export PGUSER=tldw_user
export PGPASSWORD=super-secret
```

## Step 2 - Run the migration utility

Invoke the migration module with paths to your existing SQLite databases. Provide `--content-sqlite`
for the main media database and optionally pass `--chacha-sqlite`, `--analytics-sqlite`, and
`--workflows-sqlite` if you want to migrate user notes, analytics data, and workflow runs in the
same pass.

```bash
# Repeat per user_id (single-user installs typically use the fixed default user_id).
if [ -z "${USER_DB_BASE_DIR}" ]; then
  echo "Error: USER_DB_BASE_DIR is not set." >&2
  exit 1
fi
if [ ! -d "${USER_DB_BASE_DIR}" ]; then
  echo "Error: USER_DB_BASE_DIR (${USER_DB_BASE_DIR}) is not a directory." >&2
  exit 1
fi

for user_path in "${USER_DB_BASE_DIR}"/*; do
  [ -d "${user_path}" ] || { echo "Skipping non-directory entry: ${user_path}" >&2; continue; }
  user_id="$(basename "${user_path}")"
  CONTENT_DB_NAME="${CONTENT_DB_NAME:?set CONTENT_DB_NAME to your configured per-user content DB filename}"
  content_db="${user_path}/${CONTENT_DB_NAME}"
  chacha_db="${user_path}/ChaChaNotes.db"

  if [ ! -r "${content_db}" ]; then
    echo "Warning: missing or unreadable content DB for user ${user_id} (${content_db}). Skipping." >&2
    continue
  fi

  migration_args=(
    --content-sqlite "${content_db}"
    --analytics-sqlite Databases/Analytics.db
    --workflows-sqlite Databases/workflows.db
    --pg-host "$PGHOST"
    --pg-port "$PGPORT"
    --pg-database "$PGDATABASE"
    --pg-user "$PGUSER"
    --pg-password "$PGPASSWORD"
    --batch-size 500
  )
  if [ -r "${chacha_db}" ]; then
    migration_args+=(--chacha-sqlite "${chacha_db}")
  else
    echo "Warning: missing or unreadable ChaChaNotes.db for user ${user_id} (${chacha_db}). Skipping." >&2
    continue
  fi

  python -m tldw_Server_API.app.core.DB_Management.migration_tools "${migration_args[@]}"
done
```

The script performs the following actions for each supplied database:

1. Reads table definitions from SQLite, respecting foreign-key dependencies.
2. Deletes existing rows in PostgreSQL (in dependency order) so the migration is idempotent.
3. Copies data in batches using the backend abstraction layer.
4. Resyncs sequences so future inserts continue from the correct IDs.
5. Emits a row-count summary for each migrated table (including workflow definitions, runs, steps,
   events, and artifacts) so you can compare the SQLite and Postgres totals at a glance.

Use `--skip-table <table_name>` to omit auxiliary tables (for example, to skip large log tables).

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

1. Update environment variables or `Config_Files/config.txt`:

   ```bash
   export TLDW_CONTENT_DB_BACKEND=postgresql
   export TLDW_PG_HOST=$PGHOST
   export TLDW_PG_PORT=$PGPORT
   export TLDW_PG_DATABASE=$PGDATABASE
   export TLDW_PG_USER=$PGUSER
   export TLDW_PG_PASSWORD=$PGPASSWORD
   # Optional overrides for workflow connection pooling
    export TLDW_WORKFLOW_DB_POOL_SIZE=15
    export TLDW_WORKFLOW_DB_MAX_OVERFLOW=30
    export TLDW_WORKFLOW_DB_TIMEOUT=30
   ```

2. Remove or archive the old SQLite files once you are confident the migration succeeded.

3. Restart the API/frontend stack. On startup the service should log that it is using PostgreSQL for
   Media, ChaChaNotes, Analytics, and Workflows. Monitor `/api/v1/workflows/status` to ensure the
   runtime heartbeat updates and that queue depth drops as pending runs are processed.

## Troubleshooting

- **Foreign-key errors during copy** - ensure the PostgreSQL schema was created by running the API
  once before migrating. The script expects tables and constraints to exist.
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
