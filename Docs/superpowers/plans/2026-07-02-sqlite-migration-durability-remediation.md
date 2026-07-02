# SQLite Migration Durability Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close `AUDIT-2026-06-27-DB-001` and `AUDIT-2026-06-27-DB-002` on current `origin/dev`.

**Architecture:** Keep automatic SQLite Media DB migration support limited to fresh databases and the known v22-to-v23 migration until real historical migrations are available. Route Media DB bootstrap through a Media-specific migration directory instead of the mixed package migration directory. Make `DatabaseMigrator.execute_migration()` record migration body, success ledger, and `schema_version` changes inside one owned SQLite transaction.

**Tech Stack:** SQLite, existing `DatabaseMigrator`, Media DB bootstrap helpers, pytest.

---

## Source Context

- Backlog task: `TASK-12092`
- Baseline: `origin/dev` at `30495536d3`
- Branch: `codex/audit-db-migrations-2026-07-02`
- Audit IDs: `AUDIT-2026-06-27-DB-001`, `AUDIT-2026-06-27-DB-002`
- Supported legacy decision: automatic Media DB migration supports version `22 -> 23`. Schema versions `1..21` are rejected with explicit backup/recovery guidance.

## File Map

- Modify: `tldw_Server_API/app/core/DB_Management/db_migration.py`
- Modify: `tldw_Server_API/app/core/DB_Management/media_db/schema/backends/sqlite_helpers.py`
- Create: `tldw_Server_API/app/core/DB_Management/media_db/schema/migrations/sqlite/023_transcript_run_history.sql`
- Modify: `tldw_Server_API/app/core/DB_Management/migrations/023_transcript_run_history.sql`
- Modify: `tldw_Server_API/tests/DB_Management/test_media_db_migration_missing_scripts_error.py`
- Modify: `tldw_Server_API/tests/DB_Management/test_media_db_schema_bootstrap.py`
- Create: `tldw_Server_API/tests/DB_Management/test_db_migration_atomicity.py`
- Modify: `tldw_Server_API/tests/DB_Management/test_db_migration_loader.py`

### Task 1: Reject Unsupported Pre-v22 Media DB Schemas Clearly

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/media_db/schema/backends/sqlite_helpers.py`
- Modify: `tldw_Server_API/tests/DB_Management/test_media_db_migration_missing_scripts_error.py`

- [ ] **Step 1: Write the failing unsupported-version test**

Replace the fake-migrator test with a real file-backed DB test:

```python
def test_media_db_upgrade_from_pre_v22_reports_unsupported_legacy_schema(tmp_path):
    db_path = tmp_path / "Media_DB_v2.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE schema_version (version INTEGER)")
        conn.execute("INSERT INTO schema_version (version) VALUES (8)")
        conn.commit()

    with pytest.raises(DatabaseError) as exc_info:
        MediaDatabase(db_path=str(db_path), client_id="legacy-version-test")

    msg = str(exc_info.value)
    assert "unsupported legacy Media DB schema version 8" in msg
    assert "supported automatic migration starts at version 22" in msg
    assert "backup" in msg.lower()
```

- [ ] **Step 2: Verify the test fails**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider tldw_Server_API/tests/DB_Management/test_media_db_migration_missing_scripts_error.py -q
```

Expected before implementation: failure because current code routes v8 through generic migration planning.

- [ ] **Step 3: Implement explicit pre-v22 rejection**

In `sqlite_helpers.py`, define:

```python
MIN_SUPPORTED_SQLITE_MEDIA_MIGRATION_VERSION = 22
```

Before constructing `DatabaseMigrator`, raise `SchemaError` when `0 < current_db_version < MIN_SUPPORTED_SQLITE_MEDIA_MIGRATION_VERSION`. The message must include current version, supported automatic migration start version, target version, and backup/recovery guidance.

- [ ] **Step 4: Verify the test passes**

Run the command from Step 2. Expected: pass.

### Task 2: Scope Media DB Migrations To Media-Owned SQL

**Files:**
- Create: `tldw_Server_API/app/core/DB_Management/media_db/schema/migrations/sqlite/023_transcript_run_history.sql`
- Modify: `tldw_Server_API/app/core/DB_Management/media_db/schema/backends/sqlite_helpers.py`
- Modify: `tldw_Server_API/tests/DB_Management/test_media_db_schema_bootstrap.py`
- Modify: `tldw_Server_API/tests/DB_Management/test_db_migration_loader.py`

- [ ] **Step 1: Write failing migration-directory test**

Add a test that monkeypatches `sqlite_helpers_module.DatabaseMigrator` and captures the `migrations_dir` argument for a v22 file-backed Media DB. Assert the path ends with:

```text
tldw_Server_API/app/core/DB_Management/media_db/schema/migrations/sqlite
```

Also assert the directory contains only Media DB migration files for this slice, starting with `023_transcript_run_history.sql`.

- [ ] **Step 2: Verify the test fails**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider \
  tldw_Server_API/tests/DB_Management/test_media_db_schema_bootstrap.py \
  tldw_Server_API/tests/DB_Management/test_db_migration_loader.py -q
```

Expected before implementation: failure because Media DB bootstrap still uses the mixed package migrations path for temp/test DBs and default package path otherwise.

- [ ] **Step 3: Add Media-specific migration directory**

Create the Media-specific migration directory and copy the v23 transcript run-history SQL into it. Remove explicit transaction control from the Media-specific copy:

```sql
PRAGMA foreign_keys = OFF;
-- migration body
PRAGMA foreign_keys = ON;
```

Keep trigger bodies intact; trigger `BEGIN ... END` blocks are not transaction control.

- [ ] **Step 4: Route Media DB bootstrap to the Media-specific directory**

In `sqlite_helpers.py`, compute the Media migration path relative to the file:

```python
MEDIA_SQLITE_MIGRATIONS_DIR = Path(__file__).resolve().parents[1] / "migrations" / "sqlite"
```

Pass `migrations_dir=str(MEDIA_SQLITE_MIGRATIONS_DIR)` for every non-memory Media DB migration from v22 to target. Remove the temp/test-only directory selection branch.

- [ ] **Step 5: Verify v22-to-v23 still passes**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider \
  tldw_Server_API/tests/DB_Management/test_media_db_schema_bootstrap.py::test_on_disk_sqlite_migration_to_v23_backfills_transcript_run_history -q
```

Expected: pass.

### Task 3: Make Migration Execution Atomic

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/db_migration.py`
- Modify: `tldw_Server_API/app/core/DB_Management/migrations/023_transcript_run_history.sql`
- Create: `tldw_Server_API/tests/DB_Management/test_db_migration_atomicity.py`

- [ ] **Step 1: Write failing atomicity test**

Create `test_db_migration_atomicity.py` with a migration file containing a successful `CREATE TABLE` followed by a failing statement:

```sql
-- version: 1
-- description: fail after creating a table
CREATE TABLE created_before_failure (id INTEGER PRIMARY KEY);
INSERT INTO missing_table(id) VALUES (1);
```

Run `migrator.migrate_to_version(1, create_backup=False)` and assert:

```python
with pytest.raises(MigrationError):
    migrator.migrate_to_version(1, create_backup=False)
with sqlite3.connect(db_path) as conn:
    assert conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='created_before_failure'"
    ).fetchone() is None
    assert conn.execute("SELECT COUNT(*) FROM schema_migrations WHERE success = 1").fetchone()[0] == 0
    assert conn.execute("SELECT version FROM schema_version").fetchone()[0] == 0
```

- [ ] **Step 2: Verify the atomicity test fails**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider tldw_Server_API/tests/DB_Management/test_db_migration_atomicity.py -q
```

Expected before implementation: the table remains or ledger/schema updates are not governed by one transaction.

- [ ] **Step 3: Implement transaction ownership**

In `DatabaseMigrator.execute_migration()`:

1. Delete prior failed rows before the owned migration transaction.
2. Reject SQL lines that start with transaction control:

```python
BEGIN TRANSACTION
BEGIN IMMEDIATE
BEGIN EXCLUSIVE
COMMIT
ROLLBACK
```

Do not reject trigger bodies that contain `BEGIN` after `CREATE TRIGGER`.

3. For multi-statement SQL, execute the script inside an explicit transaction by wrapping the migration body, success ledger write, and `schema_version` update in one transaction. Use rollback on every exception before recording any failed attempt.
4. Preserve idempotent `ALTER TABLE ADD COLUMN` duplicate-column skipping.

- [ ] **Step 4: Remove explicit transaction control from generic v23 migration**

Update `tldw_Server_API/app/core/DB_Management/migrations/023_transcript_run_history.sql` to remove `BEGIN TRANSACTION;` and `COMMIT;`. Keep `PRAGMA foreign_keys` statements and trigger bodies.

- [ ] **Step 5: Verify atomicity passes**

Run the command from Step 2. Expected: pass.

### Task 4: Final Focused Verification

**Files:**
- All files above
- Backlog task `TASK-12092`

- [ ] **Step 1: Run focused DB migration tests**

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider \
  tldw_Server_API/tests/DB_Management/test_media_db_migration_missing_scripts_error.py \
  tldw_Server_API/tests/DB_Management/test_media_db_schema_bootstrap.py \
  tldw_Server_API/tests/DB_Management/test_db_migration_loader.py \
  tldw_Server_API/tests/DB_Management/test_db_migration_atomicity.py -q
```

- [ ] **Step 2: Run Bandit on touched production paths**

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m bandit \
  tldw_Server_API/app/core/DB_Management/db_migration.py \
  tldw_Server_API/app/core/DB_Management/media_db/schema/backends/sqlite_helpers.py \
  -f json -o /tmp/bandit_db_migrations_12092.json
```

- [ ] **Step 3: Run whitespace check**

```bash
git diff --check
```

- [ ] **Step 4: Update `TASK-12092`**

Record verification results, closed findings, residual risk, and touched files in the Backlog task.
