# Email-Change Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A signed-in session can no longer change the account's email without the current password; every self-service change keeps the old address with its time and tells the old address.

**Architecture:** Two nullable `users` columns on both backends. The one self-service email writer (`DB_Management/user_profile_writes.update_user_email`) records the previous address in the same UPDATE. A new `core/AuthNZ/email_change.py` decides whether a change is allowed (password check) and sends the notice after the request transaction commits. The password check sits in `ProfileCommandService.apply`, which v1 and v2 `PATCH /users/me/profile` share, and in the legacy `PUT /users/me`. Chatbook account restore stops writing the email.

**Tech Stack:** FastAPI 0.142, Pydantic v2 (`SecretStr`), SQLite and PostgreSQL (asyncpg), the AuthNZ profile-version write gateway, jinja2 email templates, loguru, pytest.

**Spec:** `Docs/Design/2026-11-23-session-cap-alert-lock-design.md`, section "0. Email-change hardening (PR 0)" only.

## Global Constraints

- Precondition: the coordinator merges the design branch `docs/session-cap-alert-lock-design` (the spec, TASK-15632, this plan and the session-cap plan) to `dev` before execution starts. Branch from `origin/dev` after that merge.
- Spec values: the **current password** is required **in the same request**; the old address goes to `users.previous_email` with `users.email_changed_at`; a notice goes to the **old address**; the both-addresses window is **7 days**; **admin changes of another user's email are unchanged** (admin sessions are outside this threat model; "Decisions" and Deviation 5 give the exact behaviour of each admin path).
- Refusals are HTTP 403 with exactly these codes: `current_password_required`, `current_password_invalid`, `email_change_requires_admin`.
- Branch: `git -C /Users/macbook-dev/Documents/GitHub/tldw_server worktree add .claude/worktrees/email-change-hardening -b feat/task-15640-email-change-hardening origin/dev`. Work only in that worktree. One PR, base `dev`. Never `git stash`, never `--no-verify`, never arm auto-merge.
- Backlog first (Task 1 step 1) with the repo tool and explicit id `TASK-15640`. Edit the task only with `PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd "$PWD" task edit TASK-15640 ...`. Stage task files with `git add -A backlog/tasks`. The ADR assessment goes in the task notes.
- Python tests, always in this form (the guard deletes the basetemp afterwards):
  `zsh ~/.cache/claude-tools/disk_guard.zsh "$TMPDIR/ech-<n>" -- /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -p no:cacheprovider --basetemp "$TMPDIR/ech-<n>" <files> -q`
  Below, `PYTEST <files>` means this command with a fresh `<n>`.
- PostgreSQL tests use the existing `isolated_test_environment` (or `test_db_pool`) fixtures of `tldw_Server_API/tests/AuthNZ/conftest.py`; never create a database yourself. A Postgres test reported SKIPPED locally is recorded as skipped, not as passed.
- Frontend: no change. No WebUI, extension or admin UI caller sends `email` to these endpoints (checked: `apps/packages/ui/src`, `apps/tldw-frontend`, `apps/extension`, and top-level `admin-ui/`, which only reads `GET /users/me` and changes emails through admin `PUT /admin/users/{id}` at `admin-ui/lib/api-client.ts:708`). If that changes, frontend tests run as `cd apps/packages/ui && bunx vitest run <files> --maxWorkers=2`, one vitest process at a time.
- No raw SQL outside `core/DB_Management/` and `core/AuthNZ/` repo modules (tests are exempt). Loguru only. Never log passwords, tokens or email addresses; log the user id.
- Lint ratchets in `tldw_Server_API/tests/lint` must stay green. In particular: tests patch `core.AuthNZ.email_change` or other core/service modules, never new attributes on `api/*/endpoints` modules (endpoint monkeypatch ratchet). Add no new import statement from `services` to `api`, or from `DB_Management` to another core package. Exception tuples are named `*_NONCRITICAL_EXCEPTIONS` and never contain `BaseException` or `CancelledError`. No bare `except Exception`.
- This plan edits no mirrored doc. If you edit one (`Docs/ADR`, `Docs/Operations/Env_Vars.md`, guides), run `Helper_Scripts/refresh_docs_published.sh` and commit the `Docs/Published` diff in the same PR.
- No new dependencies, settings or environment variables.
- Every commit ends with the trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` (pass it as a second `-m`). The PR body ends with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
- Before merge: an independent review with `/superpowers:requesting-code-review` in a clean subagent.
- Change summary: owner-written or owner waiver recorded before merge.

## Review Focus

1. The same address sent again in another letter case is not a change: no password asked, no previous-address rotation, no notice. Task 4 `test_unchanged_address_in_other_case_needs_no_password`, Task 5 `test_no_notice_without_a_committed_change[unchanged]`.
2. A user fixes a typo minutes after changing the address: the change is accepted and the original address stays the previous one. Task 2 `test_second_change_within_seven_days_keeps_the_original_previous_address` (SQLite) and `test_email_change_keeps_the_original_previous_address_postgres`.
3. The mail provider is down: the change still succeeds and the warning names only the user id. Task 5 `test_failed_notice_does_not_fail_the_change`.
4. An account with no stored password is told to ask an administrator; it is not told its password is wrong. Task 4 `test_put_email_change_without_the_owners_password_is_refused[no-password-account]`.
5. An email change bundled with other profile keys and a wrong password applies none of them. Task 4 `test_refused_email_change_applies_none_of_the_other_updates`.

## Files

| File | Change |
| --- | --- |
| `tldw_Server_API/app/core/AuthNZ/migrations.py` | SQLite migration 108 adds the two columns |
| `tldw_Server_API/app/core/AuthNZ/pg_migrations_extra.py` | PostgreSQL owner migration adds the two columns |
| `tldw_Server_API/app/core/DB_Management/postgres_auth_catalog.py` | Non-owner PostgreSQL serving requires the columns |
| `tldw_Server_API/app/core/DB_Management/user_profile_writes.py` | `update_user_email` records the previous address |
| `tldw_Server_API/app/core/AuthNZ/profile_version.py` | The versioned gateway may write the two columns |
| `tldw_Server_API/app/core/UserProfiles/update_service.py` | Own `identity.email` edits go through `update_user_email` |
| `tldw_Server_API/app/core/Chatbooks/chatbook_service.py` | Account restore no longer writes `identity.email` |
| `tldw_Server_API/app/core/AuthNZ/email_change.py` (new) | Password check and the after-commit notice |
| `tldw_Server_API/app/core/AuthNZ/password_service.py` | Two-column read of email and password hash on the caller's connection |
| `tldw_Server_API/app/core/UserProfiles/contracts.py`, `command_service.py` | `current_password` on the command; the gate |
| `tldw_Server_API/app/api/v1/schemas/auth_schemas.py`, `user_profile_schemas.py` | `current_password` request field |
| `tldw_Server_API/app/api/v1/endpoints/users.py`, `app/api/v2/endpoints/user_profiles.py`, `app/services/admin_profiles_service.py` | Pass the password; send the notice |
| `tldw_Server_API/app/core/AuthNZ/email_service.py` | `email_changed` template and sender |
| `tldw_Server_API/tests/AuthNZ_SQLite/test_email_change_sqlite.py` (new) | SQLite storage and HTTP tests for this PR |

---

### Task 1: Email-change columns on both backends

**Files:**
- Modify: `tldw_Server_API/app/core/AuthNZ/migrations.py:6584` (registry) and after `:6596` (new function)
- Modify: `tldw_Server_API/app/core/AuthNZ/pg_migrations_extra.py` (new helper next to `ensure_authnz_core_tables_pg`, called after `ensure_postgres_session_last_activity(conn)` at `:3694`)
- Modify: `tldw_Server_API/app/core/DB_Management/postgres_auth_catalog.py:19-46` (`_USERS_COLUMNS`), `:600-618` (timestamp branch of `_column_type_allowed`)
- Modify: `tldw_Server_API/tests/AuthNZ/conftest.py:749` and `:1539` (both PostgreSQL `users` DDLs)
- Modify: `tldw_Server_API/tests/AuthNZ/unit/test_mcp_discovery_migration.py:41,97`
- Test: `tldw_Server_API/tests/AuthNZ_SQLite/test_email_change_sqlite.py` (create), `tldw_Server_API/tests/AuthNZ/unit/test_pg_migrations_authnz_core.py`, `tldw_Server_API/tests/AuthNZ/unit/test_postgres_runtime_checks.py`, `tldw_Server_API/tests/AuthNZ/integration/test_postgres_runtime_schema.py:430-489` (`test_nonowner_startup_refuses_required_drift_without_repair`)

**Interfaces:**
- Produces: `migration_108_add_users_email_change_columns(conn: sqlite3.Connection) -> None`; `pg_migrations_extra._ensure_users_email_change_columns(conn: Any) -> None`; columns `users.previous_email` (SQLite `TEXT`, PostgreSQL `VARCHAR(255)`, nullable) and `users.email_changed_at` (SQLite `TIMESTAMP`, PostgreSQL `TIMESTAMPTZ`, nullable).

- [ ] **Step 1: Create the worktree and the Backlog task**

```bash
git -C /Users/macbook-dev/Documents/GitHub/tldw_server fetch origin dev
git -C /Users/macbook-dev/Documents/GitHub/tldw_server worktree add .claude/worktrees/email-change-hardening -b feat/task-15640-email-change-hardening origin/dev
cd /Users/macbook-dev/Documents/GitHub/tldw_server/.claude/worktrees/email-change-hardening
PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd "$PWD" task create \
  "Email-change hardening: password on self-service change, previous address kept, notice to old address" \
  -s "In Progress" -l authnz,security \
  -d "Self-service email changes require the current password; the previous address and change time are stored and the previous address is notified. Prerequisite for the account lock (design: Docs/Design/2026-11-23-session-cap-alert-lock-design.md section 0)." \
  --ac "PUT /api/v1/users/me and PATCH /api/v1 and /api/v2 users/me/profile refuse a changed email without the correct current password (403 current_password_required or current_password_invalid) in multi-user mode" \
  --ac "An account with no stored password gets 403 email_change_requires_admin" \
  --ac "A self-service change stores users.previous_email and users.email_changed_at on SQLite and PostgreSQL; a second change within 7 days keeps the original previous address" \
  --ac "The previous address gets an email-changed notice after the change commits; a failed send does not fail the change" \
  --ac "Admin changes of another user's email are unchanged" \
  --ac "Chatbook account restore no longer changes the account email" \
  --notes "ADR assessment: no ADR in this PR. It applies the approved design (section 0); the ADR for the session cap and account lock rules is written with PR 1 as the design states." \
  --id TASK-15640
```
Expected: `backlog/tasks/task-15633 - ....md` exists.

- [ ] **Step 2: Write the failing tests**

Create `tldw_Server_API/tests/AuthNZ_SQLite/test_email_change_sqlite.py` with this header (later tasks append to it) and the migration test:

```python
"""TASK-15640: self-service email change needs the current password, keeps the previous address, notifies it."""

import sqlite3
from io import StringIO
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import check_rate_limit, get_auth_principal
from tldw_Server_API.app.api.v1.endpoints import users
from tldw_Server_API.app.api.v2.endpoints import user_profiles as v2_profiles
from tldw_Server_API.app.core.AuthNZ import migrations
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
from tldw_Server_API.app.core.AuthNZ.password_service import hash_password
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
from tldw_Server_API.app.core.DB_Management.user_profile_writes import update_user_email
from tldw_Server_API.app.core.UserProfiles.overrides_repo import UserProfileOverridesRepo
from tldw_Server_API.app.core.UserProfiles.update_service import UserProfileUpdateService
from tldw_Server_API.tests.AuthNZ_SQLite._user_fixtures import create_authnz_test_user

OWNER_EMAIL = "owner@example.test"
OWNER_PASSWORD = "Owner-Pass-2026!"


def test_migration_108_adds_email_change_columns_once(tmp_path) -> None:
    with sqlite3.connect(tmp_path / "users.db") as conn:
        conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, email TEXT NOT NULL)")
        conn.execute("INSERT INTO users (id, email) VALUES (1, 'owner@example.test')")
        migrations.migration_108_add_users_email_change_columns(conn)
        migrations.migration_108_add_users_email_change_columns(conn)
        columns = [row[1] for row in conn.execute("PRAGMA table_info(users)")]
        row = conn.execute("SELECT email, previous_email, email_changed_at FROM users").fetchone()
    assert (columns.count("previous_email"), columns.count("email_changed_at")) == (1, 1)
    assert row == ("owner@example.test", None, None)


def test_migration_108_skips_without_users_table(tmp_path) -> None:
    """Synthetic databases migrate to the latest version without a users table (cf. migrations 100 and 107)."""
    with sqlite3.connect(tmp_path / "synthetic.db") as conn:
        migrations.migration_108_add_users_email_change_columns(conn)
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert "users" not in tables
```

In `tldw_Server_API/tests/AuthNZ/unit/test_pg_migrations_authnz_core.py` add (the ALTER runs only for a missing column, so a normal start never takes the `ACCESS EXCLUSIVE` lock on `users`):

```python
_ADD_PREVIOUS = "ALTER TABLE public.users ADD COLUMN IF NOT EXISTS previous_email VARCHAR(255)"
_ADD_CHANGED_AT = "ALTER TABLE public.users ADD COLUMN IF NOT EXISTS email_changed_at TIMESTAMPTZ"


class _ColumnProbeConnection:
    def __init__(self, existing: set[str]) -> None:
        self.existing = existing
        self.executed: list[str] = []

    async def fetchval(self, query: str, column: str) -> bool:
        assert "information_schema.columns" in query
        return column in self.existing

    async def execute(self, query: str) -> None:
        self.executed.append(query)


@pytest.mark.parametrize(
    ("existing", "expected"),
    [
        (set(), [_ADD_PREVIOUS, _ADD_CHANGED_AT]),
        ({"previous_email"}, [_ADD_CHANGED_AT]),
        ({"previous_email", "email_changed_at"}, []),
    ],
    ids=["fresh", "half-migrated", "migrated"],
)
def test_users_email_change_columns_are_altered_only_when_missing(existing, expected) -> None:
    from tldw_Server_API.app.core.AuthNZ import pg_migrations_extra

    conn = _ColumnProbeConnection(existing)
    asyncio.run(pg_migrations_extra._ensure_users_email_change_columns(conn))
    assert conn.executed == expected
```

In `tldw_Server_API/tests/AuthNZ/unit/test_postgres_runtime_checks.py` add three rows to the parametrize list of `test_serving_types_preserve_legacy_families_without_truthy_text_flags`: `("email_changed_at", "timestamptz", True)`, `("email_changed_at", "text", False)`, `("previous_email", "varchar", True)`.

In `tldw_Server_API/tests/AuthNZ/integration/test_postgres_runtime_schema.py::test_nonowner_startup_refuses_required_drift_without_repair` add the drift `("ALTER TABLE public.users DROP COLUMN previous_email", "AUTHNZ_CORE_SCHEMA_NOT_READY")` with id `"user-previous-email"` (proves non-owner serving requires the column).

In `tldw_Server_API/tests/AuthNZ/unit/test_mcp_discovery_migration.py` change line 41 to `assert versions[-8:] == [101, 102, 103, 104, 105, 106, 107, 108]` and line 97 to `== 108` (keep the `# nosec B101` comments, updating "105-107" to "105-108").

- [ ] **Step 3: Run the tests to verify they fail**

Run: `PYTEST tldw_Server_API/tests/AuthNZ_SQLite/test_email_change_sqlite.py tldw_Server_API/tests/AuthNZ/unit/test_pg_migrations_authnz_core.py tldw_Server_API/tests/AuthNZ/unit/test_postgres_runtime_checks.py tldw_Server_API/tests/AuthNZ/unit/test_mcp_discovery_migration.py`
Expected: FAIL — `AttributeError` for `migration_108_add_users_email_change_columns` and `_ensure_users_email_change_columns`, `email_changed_at`/`timestamptz` not allowed, versions end at 107.

- [ ] **Step 4: Implement**

- `migrations.py`: `migration_108_add_users_email_change_columns(conn)` copies migration 107: first `if not _sqlite_table_exists(conn, "users"): logger.info("Migration 108: users table not present; skipping"); return` (synthetic test databases reach the latest version without `users`; see migration 100 at `:6669` and the migration 101 docstring), then read `PRAGMA table_info(users)`, `ALTER TABLE users ADD COLUMN previous_email TEXT` and `ALTER TABLE users ADD COLUMN email_changed_at TIMESTAMP` when missing, one `logger.info`. Register `Migration(108, "Add users.previous_email and users.email_changed_at", migration_108_add_users_email_change_columns)` after 107. If origin/dev already has a 108 at branch time, take the next free number and adjust the Step 2 version test.
- `pg_migrations_extra.py`: `_ensure_users_email_change_columns(conn)` walks a constant tuple of `(column, alter_statement)` pairs holding exactly the two Step 2 statements. For each column it runs `SELECT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema = 'public' AND table_name = 'users' AND column_name = $1)` with `fetchval` (the same probe style as the `registration_codes.uses` check at `:3695-3702`) and executes the ALTER only when the result is not `True`; `IF NOT EXISTS` stays for two owners starting at once. Call it in `ensure_authnz_core_tables_pg` right after `await ensure_postgres_session_last_activity(conn)`. The existing `_StubPostgresPool` tests keep passing: its `fetchval` answers `True`, so nothing is altered. Do not touch the bootstrap `CREATE TABLE public.users` in `Databases/Postgres/Schema/postgresql_users.sql`: the write guard accepts only its exact canonical column set (`profile_user_write_guard.py:48-77`).
- `postgres_auth_catalog.py`: add both names to `_USERS_COLUMNS`; add `"email_changed_at"` to the timestamp set in `_column_type_allowed`.
- `tests/AuthNZ/conftest.py`: in both `CREATE TABLE IF NOT EXISTS users` blocks, change `password_changed_at TIMESTAMP` to `password_changed_at TIMESTAMP,` and add `previous_email VARCHAR(255),` and `email_changed_at TIMESTAMPTZ`.

- [ ] **Step 5: Run the tests to verify they pass**

Run the Step 3 command. Expected: PASS.
Then the synthetic-database migration tests that run every migration on a database without `users`:
`PYTEST tldw_Server_API/tests/AuthNZ/unit/test_storage_quota_backfill_migration.py tldw_Server_API/tests/AuthNZ/unit/test_llm_usage_log_index_migration.py tldw_Server_API/tests/Usage/test_llm_quota_byok_rows.py`
Expected: PASS.
Then the real PostgreSQL owner-migration and non-owner serving path:
`PYTEST tldw_Server_API/tests/AuthNZ/integration/test_postgres_runtime_schema.py -k "serves_without_owner_authority or startup"`
Expected: PASS (the ALTERs pass the write guard, the restricted role finds the columns, and `[user-previous-email]` refuses startup without them).

- [ ] **Step 6: Commit**

```bash
git add -A backlog/tasks
git add tldw_Server_API/app/core/AuthNZ/migrations.py tldw_Server_API/app/core/AuthNZ/pg_migrations_extra.py \
  tldw_Server_API/app/core/DB_Management/postgres_auth_catalog.py tldw_Server_API/tests/AuthNZ/conftest.py \
  tldw_Server_API/tests/AuthNZ_SQLite/test_email_change_sqlite.py tldw_Server_API/tests/AuthNZ/unit/test_pg_migrations_authnz_core.py \
  tldw_Server_API/tests/AuthNZ/unit/test_postgres_runtime_checks.py tldw_Server_API/tests/AuthNZ/unit/test_mcp_discovery_migration.py \
  tldw_Server_API/tests/AuthNZ/integration/test_postgres_runtime_schema.py
git commit -m "feat(authnz): add users.previous_email and users.email_changed_at (TASK-15640)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 2: The self-service writer keeps the previous address

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/user_profile_writes.py:1-32`
- Modify: `tldw_Server_API/app/core/AuthNZ/profile_version.py:28-44` (`_WRITABLE_USER_FIELDS`)
- Modify: `tldw_Server_API/app/core/UserProfiles/update_service.py:319-334` (`identity.email` branch) and next to `_update_user_field` (`:819`)
- Test: `tldw_Server_API/tests/AuthNZ_SQLite/test_email_change_sqlite.py`, `tldw_Server_API/tests/UserProfile/test_user_profile_updates.py`, `tldw_Server_API/tests/UserProfile/test_user_profile_legacy_me_update.py:177-183`, `tldw_Server_API/tests/AuthNZ/integration/test_user_endpoints_integration.py`

**Interfaces:**
- Consumes: the Task 1 columns.
- Produces: `update_user_email(connection: Any, *, backend: str, user_id: int, email: str, gateway: VersionedUserWriteGateway | None = None, ownership: UserVersionOwnership = UserVersionOwnership.GATEWAY_OWNS_ANCHOR) -> UserWriteResult` (no-op result with empty `affected_user_ids` when the address is unchanged); `update_service._update_own_email(db_conn: Any, user_id: int, email: str, *, anchor: _CallerOwnedAnchor) -> None`.

- [ ] **Step 1: Write the failing tests**

Append to `test_email_change_sqlite.py`:

```python
@pytest.fixture
async def owner_db(tmp_path, monkeypatch, fresh_services):
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'users.db'}")
    monkeypatch.setenv("JWT_SECRET_KEY", "synthetic-signing-key-for-regression-tests-only")
    monkeypatch.setenv("REDIS_URL", "")
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "user_databases"))
    reset_settings()
    pool = await get_db_pool()
    user_id = await create_authnz_test_user(
        pool, username="owner", email=OWNER_EMAIL, password_hash=hash_password(OWNER_PASSWORD), is_verified=True
    )
    try:
        yield SimpleNamespace(pool=pool, user_id=user_id)
    finally:
        reset_settings()


async def _columns(db) -> tuple:
    row = dict(await db.pool.fetchone("SELECT email, previous_email, email_changed_at FROM users WHERE id = ?", db.user_id))
    return row["email"], row["previous_email"], row["email_changed_at"]


def _raw(db, statement: str) -> None:
    """Write outside the AuthNZ write guard, as an operator or the passage of time would."""
    with sqlite3.connect(db.pool.db_path) as conn:
        conn.execute(statement, (db.user_id,))


async def _change(db, email: str):
    async with db.pool.transaction() as conn:
        return await update_user_email(conn, backend="sqlite", user_id=db.user_id, email=email)


@pytest.mark.asyncio
async def test_email_change_keeps_the_previous_address_and_time(owner_db):
    await _change(owner_db, "second@example.test")
    email, previous, changed_at = await _columns(owner_db)
    assert (email, previous) == ("second@example.test", OWNER_EMAIL)
    assert changed_at is not None


@pytest.mark.asyncio
async def test_second_change_within_seven_days_keeps_the_original_previous_address(owner_db):
    await _change(owner_db, "typo@example.test")
    await _change(owner_db, "fixed@example.test")
    assert (await _columns(owner_db))[:2] == ("fixed@example.test", OWNER_EMAIL)


@pytest.mark.asyncio
async def test_change_after_seven_days_records_the_immediately_previous_address(owner_db):
    await _change(owner_db, "second@example.test")
    _raw(owner_db, "UPDATE users SET email_changed_at = datetime('now', '-8 days') WHERE id = ?")
    await _change(owner_db, "third@example.test")
    assert (await _columns(owner_db))[:2] == ("third@example.test", "second@example.test")


@pytest.mark.asyncio
async def test_same_address_in_any_case_changes_nothing(owner_db):
    result = await _change(owner_db, OWNER_EMAIL.upper())
    assert result.affected_user_ids == ()
    assert await _columns(owner_db) == (OWNER_EMAIL, None, None)


@pytest.mark.asyncio
async def test_own_profile_email_edit_records_the_previous_address(owner_db):
    async with owner_db.pool.transaction() as conn:
        result = await UserProfileUpdateService(owner_db.pool).apply_updates(
            user_id=owner_db.user_id, updates=(("identity.email", "Second@Example.test"),), roles={"user"},
            dry_run=False, db_conn=conn, updated_by=owner_db.user_id,
        )
    assert result.applied == ["identity.email"]
    assert (await _columns(owner_db))[:2] == ("second@example.test", OWNER_EMAIL)
```

In `tldw_Server_API/tests/UserProfile/test_user_profile_updates.py`, change `test_user_profile_update_accepts_valid_identity_email_with_pydantic_v2` to patch `update_service_module._update_own_email` (signature `(db_conn, user_id, email, *, anchor)`) instead of `_update_user_field`, asserting `updates == [(7, "restored.profile@example.com")]`. Add:

```python
@pytest.mark.asyncio
async def test_identity_email_edit_of_another_user_keeps_the_plain_field_write(monkeypatch: pytest.MonkeyPatch) -> None:
    writes: list[tuple[int, str, str]] = []

    async def _capture(_db_conn, user_id: int, field: str, value: str, **_kwargs) -> None:
        writes.append((user_id, field, value))

    async def _must_not_run(*_args, **_kwargs) -> None:
        raise AssertionError("another user's email must not record a previous address")

    monkeypatch.setattr(update_service_module, "_update_user_field", _capture)
    monkeypatch.setattr(update_service_module, "_update_own_email", _must_not_run)
    result = await update_service_module.UserProfileUpdateService(db_pool=object()).apply_updates(
        user_id=7, updates=(("identity.email", "Target@Example.com"),), roles={"user"},
        dry_run=False, db_conn=object(), updated_by=99,
    )
    assert result.applied == ["identity.email"]
    assert writes == [(7, "email", "target@example.com")]
```

In `tldw_Server_API/tests/UserProfile/test_user_profile_legacy_me_update.py::test_users_me_update_succeeds_when_row_is_updated`, the asserted guarded statement (`:177-183`) becomes the Step 3 PostgreSQL statement with the table qualified, and the arguments gain the repeated address:

```python
    assert fake_db.calls == [
        (
            "UPDATE public.users SET email = $1, previous_email = CASE WHEN previous_email IS NOT NULL"
            " AND email_changed_at > CURRENT_TIMESTAMP - INTERVAL '7 days' THEN previous_email ELSE email END,"
            " email_changed_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP WHERE id = $2 AND LOWER(email) <> LOWER($3)",
            "updated@example.com",
            1,
            "updated@example.com",
        )
    ]
```

In `tldw_Server_API/tests/AuthNZ/integration/test_user_endpoints_integration.py` add module helpers (imports `asyncpg` and `from tldw_Server_API.tests.helpers.pg_env import get_pg_env`; connect with `get_pg_env()` host/port/user/password and `database=db_name`, as `tests/AuthNZ/integration/test_admin_budgets.py:37-44` does):
`_email_columns(db_name, username) -> tuple[str, str | None, bool]` returning `(email, previous_email, email_changed_at IS NOT NULL)`, and `_age_email_change(db_name, username, days)` running `UPDATE users SET email_changed_at = CURRENT_TIMESTAMP - make_interval(days => $2) WHERE username = $1`. Then add to `TestUserEndpointsIntegration`:

```python
    @pytest.mark.asyncio
    async def test_email_change_keeps_the_original_previous_address_postgres(self, isolated_test_environment):
        client, db_name = isolated_test_environment
        password = "Change@Pass#2024!"
        client.post("/api/v1/auth/register", json={"username": "pgchange", "email": "pgchange@example.com", "password": password})
        token = client.post("/api/v1/auth/login", data={"username": "pgchange", "password": password}).json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        def _put(email: str) -> None:
            response = client.put("/api/v1/users/me", headers=headers, json={"email": email, "current_password": password})
            assert response.status_code == 200, response.text

        _put("typo@example.com")
        _put("fixed@example.com")
        assert await _email_columns(db_name, "pgchange") == ("fixed@example.com", "pgchange@example.com", True)
        await _age_email_change(db_name, "pgchange", days=8)
        _put("final@example.com")
        assert await _email_columns(db_name, "pgchange") == ("final@example.com", "fixed@example.com", True)
```
(`current_password` is ignored until Task 4 adds the field, so this test needs no edit later.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTEST tldw_Server_API/tests/AuthNZ_SQLite/test_email_change_sqlite.py tldw_Server_API/tests/UserProfile/test_user_profile_updates.py tldw_Server_API/tests/UserProfile/test_user_profile_legacy_me_update.py tldw_Server_API/tests/AuthNZ/integration/test_user_endpoints_integration.py`
Expected: FAIL — `previous_email` is `None` after a change; `_update_own_email` does not exist; the legacy PUT test still sees the old statement.

- [ ] **Step 3: Implement**

- `profile_version.py`: add `"previous_email"` and `"email_changed_at"` to `_WRITABLE_USER_FIELDS`. Leave `profile_user_fields.RAW_SAFE_USER_UPDATE_FIELDS` alone: these columns are written only together with `email` through the gateway, and a standalone write stays rejected.
- `user_profile_writes.py`: extend the existing `profile_version` import with `UserVersionOwnership` (one statement). New signature as in Interfaces; `gateway or VersionedUserWriteGateway(backend)` executes with `profile_visible_fields=("email",)`, the given `ownership`, and parameters `(email, user_id, email)`. The statements are fixed text (SET right-hand sides read the old row on both backends):

```sql
-- sqlite
UPDATE users SET email = ?, previous_email = CASE WHEN previous_email IS NOT NULL AND email_changed_at > datetime('now', '-7 days') THEN previous_email ELSE email END, email_changed_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP WHERE id = ? AND LOWER(email) <> LOWER(?)
-- postgres
UPDATE users SET email = $1, previous_email = CASE WHEN previous_email IS NOT NULL AND email_changed_at > CURRENT_TIMESTAMP - INTERVAL '7 days' THEN previous_email ELSE email END, email_changed_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP WHERE id = $2 AND LOWER(email) <> LOWER($3)
```
  Docstring: an address equal in any letter case is no change; within 7 days of the last change the original previous address is kept, so repeated changes cannot push the owner's address out (PR 3 sends lock mail to it).
- `update_service.py`: import `update_user_email`; add `_update_own_email` next to `_update_user_field`: `await anchor.capture()`, call `update_user_email(db_conn, backend=anchor.gateway.backend, user_id=user_id, email=email, gateway=anchor.gateway, ownership=UserVersionOwnership.CALLER_OWNS_ANCHOR)`, `anchor.mark_changed()` when `affected_user_ids` is non-empty. No `try`/`except`: errors propagate to the request transaction as they would anyway. In the `identity.email` branch call `_update_own_email` when `updated_by == user_id`, else the existing `_update_user_field(..., "email", ...)` (admin edits of another user are unchanged).

- [ ] **Step 4: Run the tests to verify they pass**

Run the Step 2 command. Expected: PASS (the Postgres test may report SKIPPED locally; record it).

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/DB_Management/user_profile_writes.py tldw_Server_API/app/core/AuthNZ/profile_version.py \
  tldw_Server_API/app/core/UserProfiles/update_service.py tldw_Server_API/tests/AuthNZ_SQLite/test_email_change_sqlite.py \
  tldw_Server_API/tests/UserProfile/test_user_profile_updates.py tldw_Server_API/tests/UserProfile/test_user_profile_legacy_me_update.py \
  tldw_Server_API/tests/AuthNZ/integration/test_user_endpoints_integration.py
git commit -m "feat(authnz): keep the previous address on a self-service email change (TASK-15640)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 3: Chatbook account restore never changes the email

A chatbook import needs only a session, and today it sends `identity.email` from the archive to `ProfileCommandService` (`chatbook_service.py:3635-3639`): a password-less email change. After Task 4 it would also fail every multi-user restore whose archive email differs.

**Files:**
- Modify: `tldw_Server_API/app/core/Chatbooks/chatbook_service.py:3630-3640`
- Modify: `Helper_Scripts/Testing-related/chatbooks_full_account_uat_fixture.py:509-510`
- Modify: `Helper_Scripts/Testing-related/chatbooks_full_account_browser_uat.py:436-438` (`_assert_destination_matches_expected`)
- Test: `tldw_Server_API/tests/Chatbooks/test_chatbooks_full_account_import_restore.py:652-725`, `tldw_Server_API/tests/UserProfile/test_stage2_caller_characterization.py:404-438`, `tldw_Server_API/tests/Chatbooks/test_chatbooks_full_account_uat_fixture.py:184`, `tldw_Server_API/tests/Chatbooks/test_chatbooks_full_account_browser_uat.py:544,570`

**Interfaces:**
- Produces: `ChatbookService._restore_account_state_payloads` never puts `identity.email` in a profile command; the export is unchanged.

- [ ] **Step 1: Update the tests to the new behaviour**

- `test_account_restore_delegates_ordered_profile_updates_to_command_service`: rename to `test_account_restore_delegates_settings_and_never_restores_the_email`; keep the payload (it still carries `identity.email: restored@example.com`); expect `command.updates == (("preferences.audio.voice", "alloy"), ("preferences.ui.theme", "paper"))` and drop `identity.email` from the stub's `applied`; `result` stays `{"account_profile": 1, "account_settings": 1}`.
- Stage-2 characterization: rename case `chatbooks_restore_preserves_adapter_update_order` to `chatbooks_restore_never_sends_identity_email` with `applied_keys=("preferences.ui.theme",)` and `command_result.applied=("preferences.ui.theme",)` (same `updates` and body). In `chatbooks_restore_rejection_is_generic` use `updates=(("preferences.ui.theme", 7),)` and `skipped=({"key": "preferences.ui.theme", "message": "type_mismatch"},)`, other fields unchanged.
- `test_fixture_round_trip_verifies_full_inventory_and_exact_embeddings`: `assert verified["profile"] == {"identity.email": fixture.DESTINATION_EMAIL}`.
- Browser UAT tests: `_FakeRuntime.verify_destination` (`:544`) returns `"profile": {"identity.email": "chatbooks-backup-destination@example.com"}` (what the real fixture `verify()` now reports); delete the `("profile", "identity.email", "preexisting@example.com")` row from the `test_destination_comparison_rejects_any_restored_account_mismatch` parametrize list (`:570`), since the destination profile is no longer compared with the source.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTEST tldw_Server_API/tests/Chatbooks/test_chatbooks_full_account_import_restore.py tldw_Server_API/tests/UserProfile/test_stage2_caller_characterization.py tldw_Server_API/tests/Chatbooks/test_chatbooks_full_account_uat_fixture.py tldw_Server_API/tests/Chatbooks/test_chatbooks_full_account_browser_uat.py`
Expected: FAIL — `identity.email` still sent; the round-trip destination email is the source email; the browser UAT rejects the destination profile.

- [ ] **Step 3: Implement**

- `chatbook_service.py`: delete the `identity.email` comparison and append (keep the destination-exists check and the `profile` dict validation). One comment line: the sign-in address is never restored from an archive; changing it needs the account password (TASK-15640).
- UAT fixture `verify()`: compare the destination profile with `{"identity.email": DESTINATION_EMAIL}` instead of `expected["profile"]`. Keep `expected["profile"]` as is: the browser UAT checks the archive (`chatbooks_full_account_browser_uat.py:360`), which still holds the source email.
- Browser UAT `_assert_destination_matches_expected`: compare only `settings` with the source (`for section in ("settings",)`). The real runtime's `verify_destination` runs the fixture's `verify()` (`:856-857`), which already requires the destination's own email.

- [ ] **Step 4: Run the tests to verify they pass**

Run the Step 2 command. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/Chatbooks/chatbook_service.py Helper_Scripts/Testing-related/chatbooks_full_account_uat_fixture.py \
  Helper_Scripts/Testing-related/chatbooks_full_account_browser_uat.py \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_full_account_import_restore.py tldw_Server_API/tests/UserProfile/test_stage2_caller_characterization.py \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_full_account_uat_fixture.py tldw_Server_API/tests/Chatbooks/test_chatbooks_full_account_browser_uat.py
git commit -m "fix(chatbooks): account restore never changes the sign-in email (TASK-15640)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 4: The current password is required

**Files:**
- Create: `tldw_Server_API/app/core/AuthNZ/email_change.py`
- Modify: `tldw_Server_API/app/core/AuthNZ/password_service.py:506-522` (sibling of `fetch_password_hash`)
- Modify: `tldw_Server_API/app/api/v1/schemas/auth_schemas.py:94-112` (`UpdateProfileRequest`), `tldw_Server_API/app/api/v1/schemas/user_profile_schemas.py:173-181` (`UserProfileUpdateRequest`)
- Modify: `tldw_Server_API/app/core/UserProfiles/contracts.py:49-59`, `tldw_Server_API/app/core/UserProfiles/command_service.py:126`
- Modify: `tldw_Server_API/app/api/v1/endpoints/users.py:481-534` (PATCH), `:584-665` (PUT); `tldw_Server_API/app/api/v2/endpoints/user_profiles.py:58-66`; `tldw_Server_API/app/services/admin_profiles_service.py:809-819`
- Modify: `apps/packages/api-types/openapi.fingerprint.json` (regenerated)
- Test: `tldw_Server_API/tests/AuthNZ_SQLite/test_email_change_sqlite.py`, `tldw_Server_API/tests/UserProfile/test_profile_command_service.py`, `tldw_Server_API/tests/UserProfile/test_profile_contracts.py`, `tldw_Server_API/tests/UserProfile/test_admin_profiles_service_update.py`, `tldw_Server_API/tests/AuthNZ/integration/test_user_endpoints_integration.py`, `tldw_Server_API/tests/AuthNZ/unit/test_user_endpoints.py`, `tldw_Server_API/tests/AuthNZ/property/test_user_endpoints_property.py`, `tldw_Server_API/tests/AuthNZ_Postgres/test_profile_update_pool_capacity_pg.py`

**Interfaces:**
- Consumes: Task 2 writer (unchanged call sites); `update_service._is_postgres_backend_for_pool(db_pool: Any) -> bool`.
- Produces:
  - `email_change.EmailChangeRefused(Exception)` with `__init__(self, code: str, detail: str)` and attributes `code`, `detail`.
  - `async email_change.authorize_own_email_change(*, db_conn: Any, is_postgres: bool, user_id: int, new_emails: Sequence[str], current_password: str | None) -> None` — raises `EmailChangeRefused`; returns when allowed. It reads only on `db_conn` (profile updates must fit in their existing PostgreSQL transaction slot). `new_emails` holds every `identity.email` value in the request.
  - `async password_service.fetch_email_and_password_hash(db_connection: Any, user_id: int, *, is_postgres: bool) -> tuple[str | None, str | None]` — `(email, password_hash)`, `(None, None)` when the row is missing.
  - `ProfileUpdateCommand.current_password: str | None = field(default=None, repr=False, compare=False)`.
  - `current_password: SecretStr | None = Field(None, description=...)` on `UpdateProfileRequest` and `UserProfileUpdateRequest`.

- [ ] **Step 1: Write the failing tests**

Append to `test_email_change_sqlite.py`:

```python
class _FixedStorage:
    async def user_quota_status(self, user_id: int) -> dict[str, float]:
        return {"quota_mb": 5120, "used_mb": 0.0}


@pytest.fixture
async def email_api(owner_db, monkeypatch):
    async def _principal():
        return AuthPrincipal(kind="user", user_id=owner_db.user_id, username="owner", email=OWNER_EMAIL, roles=["user"])

    async def _storage():
        return _FixedStorage()

    async def _no_rate_limit():
        return None

    monkeypatch.setattr("tldw_Server_API.app.services.storage_quota_service.get_storage_service", _storage)
    app = FastAPI()
    app.include_router(users.router, prefix="/api/v1")
    app.include_router(v2_profiles.router, prefix="/api/v2")
    app.dependency_overrides[get_auth_principal] = _principal
    app.dependency_overrides[check_rate_limit] = _no_rate_limit
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        yield client


EMAIL_UPDATE = {"updates": [{"key": "identity.email", "value": "new@example.test"}]}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("blank_hash", "password", "code"),
    [
        (False, None, "current_password_required"),
        (False, "Wrong-Pass-2026!", "current_password_invalid"),
        (True, OWNER_PASSWORD, "email_change_requires_admin"),
    ],
    ids=["missing", "wrong", "no-password-account"],
)
async def test_put_email_change_without_the_owners_password_is_refused(owner_db, email_api, blank_hash, password, code):
    if blank_hash:
        _raw(owner_db, "UPDATE users SET password_hash = '' WHERE id = ?")
    body = {"email": "new@example.test", **({"current_password": password} if password else {})}
    response = await email_api.put("/api/v1/users/me", json=body)
    assert (response.status_code, response.json()["error_code"]) == (403, code)
    assert (await _columns(owner_db))[:2] == (OWNER_EMAIL, None)


@pytest.mark.asyncio
async def test_put_email_change_with_the_owners_password_succeeds(owner_db, email_api):
    response = await email_api.put("/api/v1/users/me", json={"email": "new@example.test", "current_password": OWNER_PASSWORD})
    assert response.status_code == 200, response.text
    assert response.json()["email"] == "new@example.test"
    assert (await _columns(owner_db))[:2] == ("new@example.test", OWNER_EMAIL)


@pytest.mark.asyncio
async def test_patch_v1_email_change_requires_the_owners_password(owner_db, email_api):
    refused = await email_api.patch("/api/v1/users/me/profile", json=EMAIL_UPDATE)
    assert (refused.status_code, refused.json()["error_code"]) == (403, "current_password_required")
    accepted = await email_api.patch("/api/v1/users/me/profile", json={**EMAIL_UPDATE, "current_password": OWNER_PASSWORD})
    assert accepted.status_code == 200, accepted.text
    assert "identity.email" in accepted.json()["applied"]
    assert (await _columns(owner_db))[:2] == ("new@example.test", OWNER_EMAIL)


@pytest.mark.asyncio
async def test_patch_v2_email_change_requires_the_owners_password(owner_db, email_api):
    response = await email_api.patch("/api/v2/users/me/profile", json={**EMAIL_UPDATE, "current_password": "Wrong-Pass-2026!"})
    assert response.status_code == 403
    assert response.json()["detail"]["error_code"] == "current_password_invalid"
    assert (await _columns(owner_db))[0] == OWNER_EMAIL


@pytest.mark.asyncio
async def test_refused_email_change_applies_none_of_the_other_updates(owner_db, email_api):
    body = {
        "updates": [{"key": "preferences.ui.theme", "value": "paper"}, *EMAIL_UPDATE["updates"]],
        "current_password": "Wrong-Pass-2026!",
    }
    response = await email_api.patch("/api/v1/users/me/profile", json=body)
    assert response.status_code == 403
    overrides = await UserProfileOverridesRepo(owner_db.pool).list_overrides_for_user(owner_db.user_id)
    assert all(row["key"] != "preferences.ui.theme" for row in overrides)


@pytest.mark.asyncio
async def test_unchanged_address_in_other_case_needs_no_password(owner_db, email_api):
    body = {"updates": [{"key": "identity.email", "value": OWNER_EMAIL.upper()}]}
    response = await email_api.patch("/api/v1/users/me/profile", json=body)
    assert response.status_code == 200, response.text
    assert await _columns(owner_db) == (OWNER_EMAIL, None, None)


@pytest.mark.asyncio
async def test_any_changed_identity_email_in_the_request_needs_the_password(owner_db, email_api):
    body = {
        "updates": [
            {"key": "identity.email", "value": "new@example.test"},
            {"key": "identity.email", "value": OWNER_EMAIL},
        ]
    }
    response = await email_api.patch("/api/v1/users/me/profile", json=body)
    assert (response.status_code, response.json()["error_code"]) == (403, "current_password_required")
    assert await _columns(owner_db) == (OWNER_EMAIL, None, None)


@pytest.mark.asyncio
async def test_dry_run_email_change_needs_no_password(owner_db, email_api):
    response = await email_api.patch("/api/v1/users/me/profile", json={**EMAIL_UPDATE, "dry_run": True})
    assert response.status_code == 200, response.text
    assert (await _columns(owner_db))[0] == OWNER_EMAIL


@pytest.mark.asyncio
async def test_single_user_mode_needs_no_password(monkeypatch):
    from tldw_Server_API.app.core.AuthNZ import email_change

    monkeypatch.setenv("AUTH_MODE", "single_user")
    reset_settings()
    try:
        # db_conn=None: single-user mode must decide without reading anything.
        await email_change.authorize_own_email_change(
            db_conn=None, is_postgres=False, user_id=1, new_emails=("new@example.test",), current_password=None
        )
    finally:
        reset_settings()
```

In `tldw_Server_API/tests/AuthNZ_Postgres/test_profile_update_pool_capacity_pg.py`, extract the seed block of the existing test into `async def _seed_user(test_db_pool, username: str, password_hash: str) -> int` (email `f"{username}@example.test"`; the existing test calls it with `"profile-capacity", "unused-test-hash"`), then add:

```python
@pytest.mark.asyncio
async def test_own_email_change_completes_when_other_pool_slots_are_busy(test_db_pool) -> None:
    """The password check reads on the request's own connection, never a second pool slot."""
    user_id = await _seed_user(test_db_pool, "email-capacity", hash_password("Capacity-Pass-2026!"))
    async with AsyncExitStack() as slots:
        connection = await slots.enter_async_context(test_db_pool.acquire())
        for _ in range(test_db_pool.pool.get_max_size() - 1):
            await slots.enter_async_context(test_db_pool.acquire())
        async with connection.transaction():
            result = await asyncio.wait_for(
                ProfileCommandService(db_pool=test_db_pool).apply(
                    ProfileUpdateCommand(
                        actor_user_id=user_id,
                        target_user_id=user_id,
                        updates=(("identity.email", "email-capacity-new@example.test"),),
                        roles=frozenset({"user"}),
                        dry_run=False,
                        current_password="Capacity-Pass-2026!",
                    ),
                    db_conn=connection,
                    scope=None,
                ),
                timeout=5,
            )
            assert result.applied == ("identity.email",)
            row = await connection.fetchrow("SELECT email, previous_email FROM users WHERE id = $1", user_id)
            assert (row["email"], row["previous_email"]) == (
                "email-capacity-new@example.test",
                "email-capacity@example.test",
            )
```
(`hash_password` from `tldw_Server_API.app.core.AuthNZ.password_service`; `test_db_pool` runs in multi-user mode, so the check really reads.)

In `tldw_Server_API/tests/UserProfile/test_profile_command_service.py` (uses the file's `_ProfileService`, `_Planner`, `_Executor`):

```python
@pytest.mark.asyncio
async def test_own_email_change_refusal_stops_before_the_executor(monkeypatch) -> None:
    from tldw_Server_API.app.core.AuthNZ import email_change

    async def _refuse(**_kwargs):
        raise email_change.EmailChangeRefused("current_password_invalid", "Current password is incorrect. ...")

    monkeypatch.setattr(email_change, "authorize_own_email_change", _refuse)
    executor = _Executor()
    command = ProfileUpdateCommand(
        actor_user_id=7, target_user_id=7, updates=(("identity.email", "new@example.test"),),
        roles=frozenset({"user"}), dry_run=False, current_password="wrong",
    )
    result = await ProfileCommandService(
        db_pool=object(), profile_service=_ProfileService(), planner=_Planner(), executor=executor
    ).apply(command, db_conn=object(), scope=None)
    assert (result.status_code, result.error_code) == (403, "current_password_invalid")
    assert result.skipped == ({"key": "identity.email", "message": "current_password_invalid"},)
    assert executor.called is False


@pytest.mark.asyncio
async def test_email_change_for_another_user_skips_the_password_gate(monkeypatch) -> None:
    from tldw_Server_API.app.core.AuthNZ import email_change

    async def _must_not_run(**_kwargs):
        raise AssertionError("an admin edit of another user needs no password")

    monkeypatch.setattr(email_change, "authorize_own_email_change", _must_not_run)
    executor = _Executor()
    command = ProfileUpdateCommand(
        actor_user_id=99, target_user_id=7, updates=(("identity.email", "new@example.test"),),
        roles=frozenset({"platform_admin", "user"}), dry_run=False,
    )
    result = await ProfileCommandService(
        db_pool=object(), profile_service=_ProfileService(), planner=_Planner(), executor=executor
    ).apply(command, db_conn=object(), scope=None)
    assert result.status_code == 200
    assert executor.called is True
```

In `tldw_Server_API/tests/UserProfile/test_profile_contracts.py`:

```python
def test_update_command_never_shows_the_password() -> None:
    command = ProfileUpdateCommand(
        actor_user_id=7, target_user_id=7, updates=(), roles=frozenset({"user"}), dry_run=False,
        current_password="s3cret-value",
    )
    assert "s3cret-value" not in repr(command)
```

In `test_admin_profiles_service_update.py::test_admin_profile_update_delegates_to_command_service` add `current_password="admin-own-pass"` to the payload and `assert command.current_password == "admin-own-pass"`.

Existing multi-user tests that change email now send the password:
- `tests/AuthNZ/integration/test_user_endpoints_integration.py::TestUserEndpointsIntegration::test_update_user_profile`: add `"current_password": "Update@Pass#2024!"`. Add `test_update_email_without_password_is_refused` to the same class: register and log in `nopass` / `nopass@example.com`, `PUT /api/v1/users/me` with only `email` → 403, `error_code == "current_password_required"`, and `GET /api/v1/users/me` still returns `nopass@example.com`.
- `tests/AuthNZ/unit/test_user_endpoints.py::TestUserEndpoints::test_update_user_profile`: add `"current_password": test_user["password"]`.
- `tests/AuthNZ/property/test_user_endpoints_property.py::TestUserEndpointsProperty::test_update_profile_with_various_emails`: add `"current_password": test_user["password"]`.
- `tests/UserProfile/*` run in single-user mode (`tests/UserProfile/conftest.py:36-43`) and need no change.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTEST tldw_Server_API/tests/AuthNZ_SQLite/test_email_change_sqlite.py tldw_Server_API/tests/UserProfile/test_profile_command_service.py tldw_Server_API/tests/UserProfile/test_profile_contracts.py tldw_Server_API/tests/UserProfile/test_admin_profiles_service_update.py tldw_Server_API/tests/AuthNZ/integration/test_user_endpoints_integration.py`
Expected: FAIL — changes succeed without a password; `email_change` and `current_password` do not exist.

- [ ] **Step 3: Implement `core/AuthNZ/email_change.py`**

First add `fetch_email_and_password_hash` to `password_service.py` beside `fetch_password_hash`, same shape with two columns: PostgreSQL `fetchrow("SELECT email, password_hash FROM users WHERE id = $1", user_id)`, SQLite `execute("SELECT email, password_hash FROM users WHERE id = ?", (user_id,))` then `fetchone()`, accepting dict or tuple rows like the existing helper.

`authorize_own_email_change`, in this order:
1. `settings.is_single_user_mode()` → return without reading anything (the API key is the only credential there and no lock exists).
2. `email, password_hash = await fetch_email_and_password_hash(db_conn, user_id, is_postgres=is_postgres)`: one read on the caller's connection, never `AuthnzUsersRepo.from_pool()`, which would need a second pool slot inside `ProfileCommandService.apply`. A missing row gives `(None, None)` and fails closed below.
3. Every value in `new_emails`, stripped and lower-cased, equals `(email or "").strip().lower()` → return (no change, no password). One differing value is enough to require the password, because the executor applies every update in order.
4. Empty or missing `password_hash` → `EmailChangeRefused("email_change_requires_admin", "This account has no password. Ask an administrator to change its email address.")`.
5. No `current_password` → `EmailChangeRefused("current_password_required", "Enter your current password to change your email address.")`.
6. `password_service.verify_password(current_password, password_hash)[0]` is false → `EmailChangeRefused("current_password_invalid", "Current password is incorrect. If you sign in with SSO or an email link, use Forgot password to set one first.")`.
Every refusal logs `logger.info("Self-service email change refused for user {}: {}", user_id, code)`. Compare the password as sent; do not strip it.

- [ ] **Step 4: Wire the request fields, the command and the gate**

- `UpdateProfileRequest` and `UserProfileUpdateRequest`: `current_password: Optional[SecretStr] = Field(None, description="Your current password. Required when the request changes your own email address (multi-user mode).")`. Add `"current_password"` to the `UpdateProfileRequest` example.
- `contracts.ProfileUpdateCommand`: the field from Interfaces, last.
- `command_service.py`: `from tldw_Server_API.app.core.AuthNZ import email_change as _email_change` (module attribute, so tests can patch it). Extend the existing `update_service` import statement with `_is_postgres_backend_for_pool`. Module-level `async def _own_email_change_refusal(command: ProfileUpdateCommand, *, db_conn: Any, is_postgres: bool) -> LegacyProfileCommandResult | None`: when `command.actor_user_id == command.target_user_id` and the command holds at least one `identity.email` update, await `_email_change.authorize_own_email_change(db_conn=db_conn, is_postgres=is_postgres, user_id=command.target_user_id, new_emails=tuple(str(value) for key, value in command.updates if key == "identity.email"), current_password=command.current_password)`; on `EmailChangeRefused` return `LegacyProfileCommandResult(status_code=403, error_code=exc.code, detail=exc.detail, skipped=({"key": "identity.email", "message": exc.code},))`. In `apply`, call it with `db_conn=db_conn, is_postgres=_is_postgres_backend_for_pool(self._db_pool)` immediately before `executor_kwargs` (after the dry-run return and the version lock) and return the refusal if any. Dry runs therefore need no password.
- `users.py` PATCH `/me/profile` and v2 `PATCH /users/me/profile`: pass `current_password=payload.current_password.get_secret_value() if payload.current_password else None` into `ProfileUpdateCommand`.
- `admin_profiles_service.update_user_profile`: pass the same expression into its `ProfileUpdateCommand` (an admin changing their own email here is a self-service change, Deviation 5; others' emails never reach the gate). Inline `get_secret_value()`; do not import `admin_schemas.unwrap_optional_secret`, which strips whitespace and would add a services-to-api import.
- `users.py` PUT `/me`: `import tldw_Server_API.app.core.AuthNZ.email_change as _email_change` beside the other module aliases. Make the existing same-address check case-insensitive like the writer: `request.email.lower() != str(user_context.get("email") or "").lower()` (same address → 400 "No updates provided", as today). Compute `is_pg = await _database.is_postgres_backend()` before the check instead of inside it. Inside the existing `try`, before `update_user_email`, await `_email_change.authorize_own_email_change(db_conn=db, is_postgres=is_pg, user_id=user_id, new_emails=(request.email,), current_password=<same expression on request.current_password>)`; on `EmailChangeRefused` return `_profile_error_response(status_code=403, error_code=exc.code, detail=exc.detail, errors=[UserProfileErrorDetail(key="email", message=exc.code)])`. `EmailChangeRefused` subclasses `Exception` directly so `_USERS_ENDPOINT_EXCEPTIONS` never turns it into a 500.

- [ ] **Step 5: Regenerate the OpenAPI fingerprint**

```bash
PY=/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python
$PY Helper_Scripts/export_openapi_schema.py --fingerprint apps/packages/api-types/openapi.fingerprint.json
$PY Helper_Scripts/export_openapi_schema.py --check apps/packages/api-types/openapi.fingerprint.json --check-paths apps/packages/api-types/paths.d.ts
git status --short apps/packages/api-types
```
Expected: the check exits 0; only `openapi.fingerprint.json` changed (no new paths, so `paths.d.ts` is unchanged).

- [ ] **Step 6: Run the tests to verify they pass**

Run the Step 2 command, then `PYTEST tldw_Server_API/tests/AuthNZ_Postgres/test_profile_update_pool_capacity_pg.py tldw_Server_API/tests/AuthNZ/unit/test_user_endpoints.py tldw_Server_API/tests/AuthNZ/property/test_user_endpoints_property.py tldw_Server_API/tests/UserProfile/test_user_profile_legacy_me_update.py tldw_Server_API/tests/UserProfile/test_stage2_caller_characterization.py tldw_Server_API/tests/UserProfile/test_user_profile_updates.py tldw_Server_API/tests/UserProfile/test_user_profile_v2_contract.py tldw_Server_API/tests/UserProfile/test_user_profile_legacy_contract_characterization.py`
Expected: PASS (Postgres-backed ones may SKIP locally; record which).

- [ ] **Step 7: Commit**

```bash
git add tldw_Server_API/app/core/AuthNZ/email_change.py tldw_Server_API/app/core/AuthNZ/password_service.py \
  tldw_Server_API/tests/AuthNZ_Postgres/test_profile_update_pool_capacity_pg.py tldw_Server_API/app/api/v1/schemas/auth_schemas.py \
  tldw_Server_API/app/api/v1/schemas/user_profile_schemas.py tldw_Server_API/app/core/UserProfiles/contracts.py \
  tldw_Server_API/app/core/UserProfiles/command_service.py tldw_Server_API/app/api/v1/endpoints/users.py \
  tldw_Server_API/app/api/v2/endpoints/user_profiles.py tldw_Server_API/app/services/admin_profiles_service.py \
  apps/packages/api-types/openapi.fingerprint.json tldw_Server_API/tests/AuthNZ_SQLite/test_email_change_sqlite.py \
  tldw_Server_API/tests/UserProfile/test_profile_command_service.py tldw_Server_API/tests/UserProfile/test_profile_contracts.py \
  tldw_Server_API/tests/UserProfile/test_admin_profiles_service_update.py tldw_Server_API/tests/AuthNZ/integration/test_user_endpoints_integration.py \
  tldw_Server_API/tests/AuthNZ/unit/test_user_endpoints.py tldw_Server_API/tests/AuthNZ/property/test_user_endpoints_property.py
git commit -m "feat(authnz): require the current password for self-service email changes (TASK-15640)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 5: Notice to the previous address

**Files:**
- Modify: `tldw_Server_API/app/core/AuthNZ/email_service.py:402` (template dict) and after `:1095` (sender)
- Modify: `tldw_Server_API/app/core/AuthNZ/email_change.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/users.py` (helpers; PUT and PATCH call sites), `tldw_Server_API/app/api/v2/endpoints/user_profiles.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_email_service.py`, `tldw_Server_API/tests/AuthNZ_SQLite/test_email_change_sqlite.py`

**Interfaces:**
- Consumes: Task 4 endpoints and `user_context` (holds the old address until the handler overwrites it).
- Produces:
  - `EmailService.send_email_changed_email(self, *, to_email: str, username: str, new_email: str, ip_address: str = "Unknown") -> bool` (template `email_changed`; masks `new_email` itself).
  - `async email_change.notify_previous_address(*, user_id: int, old_email: str, new_email: str, username: str, ip_address: str) -> None`.
  - `users._requested_email(payload: UserProfileUpdateRequest) -> str | None` (last `identity.email` value) and `async users._notify_email_change(http_request: Request | None, user_context: dict[str, Any], new_email: str | None) -> None` (imported by v2).

- [ ] **Step 0: Recheck the email_service.py anchors**

Open PR #3453 ("Clarify password reset and sign-in emails") rewrites `EMAIL_TEMPLATES` in `email_service.py`. At branch time, confirm where the template dict ends and where `send_mfa_enabled_email` sits (`grep -n "EMAIL_TEMPLATES = {\|def send_mfa_enabled_email\|^}" tldw_Server_API/app/core/AuthNZ/email_service.py`) and use those lines instead of `:402` and `:1095` if they moved. The existing recipient logging in `email_service.py` (`:642` mock "To:", `:834` "Email sent successfully to", `:838` SMTP error text) is pre-existing and out of scope for this PR; do not copy that pattern.

- [ ] **Step 1: Write the failing tests**

In `test_email_service.py`: add `assert "email_changed" in EMAIL_TEMPLATES` to a new `TestEmailTemplates.test_email_changed_template_exists`, and in `TestTemplateEmailSending`:

```python
    @pytest.mark.asyncio
    async def test_send_email_changed_email_masks_the_new_address(self, email_service, monkeypatch):
        sent: dict[str, object] = {}

        async def _capture(to_email, subject, html_body, text_body=None, **kwargs):
            sent.update(to=to_email, subject=subject, html=html_body, text=text_body, template=kwargs.get("_template"))
            return True

        monkeypatch.setattr(email_service, "send_email", _capture)
        assert await email_service.send_email_changed_email(
            to_email="old@example.org", username="owner", new_email="new.owner@example.org", ip_address="203.0.113.9"
        ) is True
        assert (sent["to"], sent["template"]) == ("old@example.org", "email_changed")
        assert "email address was changed" in str(sent["subject"])
        for body in (sent["html"], sent["text"]):
            assert "n***@example.org" in body and "203.0.113.9" in body
            assert "new.owner@example.org" not in body
```

Append to `test_email_change_sqlite.py`:

```python
@pytest.fixture
def sent_notices(owner_db, monkeypatch):
    from tldw_Server_API.app.core.AuthNZ import email_change

    notices: list[dict] = []

    class _Mail:
        async def send_email_changed_email(self, **kwargs):
            with sqlite3.connect(owner_db.pool.db_path) as conn:
                committed = conn.execute("SELECT email FROM users WHERE id = ?", (owner_db.user_id,)).fetchone()[0]
            notices.append({**kwargs, "committed_email": committed})
            return True

    monkeypatch.setattr(email_change, "get_email_service", lambda: _Mail())
    return notices


@pytest.mark.asyncio
async def test_email_change_notifies_the_previous_address_after_commit(owner_db, email_api, sent_notices):
    response = await email_api.put("/api/v1/users/me", json={"email": "new@example.test", "current_password": OWNER_PASSWORD})
    assert response.status_code == 200, response.text
    assert [(n["to_email"], n["new_email"], n["committed_email"]) for n in sent_notices] == [
        (OWNER_EMAIL, "new@example.test", "new@example.test")
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/api/v1/users/me/profile", "/api/v2/users/me/profile"])
async def test_profile_email_change_notifies_the_previous_address(owner_db, email_api, sent_notices, path):
    response = await email_api.patch(path, json={**EMAIL_UPDATE, "current_password": OWNER_PASSWORD})
    assert response.status_code == 200, response.text
    assert [n["to_email"] for n in sent_notices] == [OWNER_EMAIL]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [EMAIL_UPDATE, {**EMAIL_UPDATE, "dry_run": True}, {"updates": [{"key": "identity.email", "value": OWNER_EMAIL.upper()}]}],
    ids=["refused", "dry-run", "unchanged"],
)
async def test_no_notice_without_a_committed_change(owner_db, email_api, sent_notices, body):
    await email_api.patch("/api/v1/users/me/profile", json=body)
    assert sent_notices == []


@pytest.mark.asyncio
async def test_failed_notice_does_not_fail_the_change(owner_db, email_api, monkeypatch):
    from tldw_Server_API.app.core.AuthNZ import email_change

    class _BrokenMail:
        async def send_email_changed_email(self, **_kwargs):
            raise OSError(f"smtp unreachable for {OWNER_EMAIL}")

    monkeypatch.setattr(email_change, "get_email_service", lambda: _BrokenMail())
    buffer = StringIO()
    sink = logger.add(buffer, level="DEBUG")
    try:
        response = await email_api.put("/api/v1/users/me", json={"email": "new@example.test", "current_password": OWNER_PASSWORD})
    finally:
        logger.remove(sink)
    assert response.status_code == 200, response.text
    assert (await _columns(owner_db))[0] == "new@example.test"
    warnings = [line for line in buffer.getvalue().splitlines() if "Email-change notice" in line]
    assert len(warnings) == 1
    assert OWNER_EMAIL not in warnings[0] and "new@example.test" not in warnings[0]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTEST tldw_Server_API/tests/AuthNZ/unit/test_email_service.py tldw_Server_API/tests/AuthNZ_SQLite/test_email_change_sqlite.py`
Expected: FAIL — no `email_changed` template, no sender, no notices.

- [ ] **Step 3: Implement**

- `EMAIL_TEMPLATES["email_changed"]`, in the style of `mfa_enabled`. Subject: `Your {{ app_name }} email address was changed`. Text (the HTML carries the same sentences):

```text
Hello {{ username }},

The email address on your {{ app_name }} account was changed to {{ new_email_masked }} at {{ timestamp }} from IP address {{ ip_address }}.

If you made this change, you do not need to do anything.

If you did not, sign in with your password, change your password, and change your email address back. If you cannot sign in, contact your administrator.
```
  `send_email_changed_email` renders with `new_email_masked = f"{local[:1] or '*'}***@{domain}"` (from `new_email.partition("@")`) and a UTC `timestamp`, then `send_email(..., _template="email_changed")`.
- `email_change.notify_previous_address`: an inner `async def _send()` calls `get_email_service().send_email_changed_email(to_email=old_email, username=username, new_email=new_email, ip_address=ip_address)`; the call (template rendering included) sits inside the `try`, which catches `_NOTICE_NONCRITICAL_EXCEPTIONS = (OSError, RuntimeError, ValueError, TypeError, KeyError, jinja2.TemplateError)` (jinja2 is already a dependency of `email_service.py`) and, on an exception or a `False` result, logs `logger.warning("Email-change notice to the previous address was not sent (user {})", user_id)`. Then `if not defer_until_after_commit(_send): await _send()` (`core/AuthNZ/transaction_hooks.py`, as `services/admin_audit_service.py:52` does). The request transaction from `get_db_transaction` runs the hook only after a successful commit and drops it on rollback.
- `users.py`: `_requested_email` and `_notify_email_change` per Interfaces. `_notify_email_change` returns unless `new_email` differs (stripped, case-insensitive) from `user_context["email"]`; otherwise it awaits `_email_change.notify_previous_address(user_id=int(user_context["id"]), old_email=..., new_email=new_email.strip().lower(), username=str(user_context["username"]), ip_address=resolve_client_ip(http_request) or "unknown")` (`from tldw_Server_API.app.core.AuthNZ.ip_allowlist import resolve_client_ip`).
  - PUT `/me`: add `http_request: Request = None` (the handler's existing `response: Response = None` pattern) and call `_notify_email_change(http_request, user_context, request.email)` after `update_user_email` succeeds and before `user_context["email"]` is overwritten.
  - v1 PATCH `/me/profile`: after the `JSONResponse` early return, when `not payload.dry_run and "identity.email" in response.applied`, call `_notify_email_change(http_request, user_context, _requested_email(payload))`.
  - v2: add `_requested_email` and `_notify_email_change` to the existing `from tldw_Server_API.app.api.v1.endpoints.users import (...)` statement (`user_profiles.py:18-22`). Do not add a second import statement: `layering_baseline.txt` allows `api.v2.endpoints.user_profiles -> api.v1.endpoints.users: 1`. Then the same call with `result.applied`, before building the response.

- [ ] **Step 4: Run the tests to verify they pass**

Run the Step 2 command, then the Task 4 Step 6 command. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/AuthNZ/email_service.py tldw_Server_API/app/core/AuthNZ/email_change.py \
  tldw_Server_API/app/api/v1/endpoints/users.py tldw_Server_API/app/api/v2/endpoints/user_profiles.py \
  tldw_Server_API/tests/AuthNZ/unit/test_email_service.py tldw_Server_API/tests/AuthNZ_SQLite/test_email_change_sqlite.py
git commit -m "feat(authnz): notify the previous address after an email change (TASK-15640)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 6: Branch verification and PR

**Files:** Backlog only: edit TASK-15640, create TASK-15641.

- [ ] **Step 1: Run every test that touches the changed surfaces**

```bash
git grep -l -E "identity\.email|/api/v1/users/me['\"]|/users/me/profile|update_user_email|_restore_account_state_payloads|_CREATE_AUTHNZ_CORE_TABLES|ensure_authnz_core_tables_pg|_USERS_COLUMNS|get_authnz_migrations|apply_authnz_migrations|ensure_authnz_tables|ProfileCommandService|fetch_password_hash|EMAIL_TEMPLATES" -- tldw_Server_API/tests
```
Run `PYTEST` over every file listed. Expected: PASS; list any SKIPPED Postgres tests in the task notes. A failure is compared against `origin/dev` before it is called pre-existing.

- [ ] **Step 2: Lint ratchets, ruff, OpenAPI drift**

```bash
PY=/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python
PYTEST tldw_Server_API/tests/lint
for f in $(git diff --name-only --diff-filter=A origin/dev...HEAD -- '*.py'); do $PY -m ruff check "$f"; done
for f in $(git diff --name-only --diff-filter=M origin/dev...HEAD -- '*.py'); do
  diff <(git show "origin/dev:$f" | $PY -m ruff check --statistics --stdin-filename "$f" -) <($PY -m ruff check --statistics "$f") >/dev/null || echo "NEW RUFF FINDINGS: $f"
done
$PY Helper_Scripts/export_openapi_schema.py --check apps/packages/api-types/openapi.fingerprint.json --check-paths apps/packages/api-types/paths.d.ts
```
Expected: lint ratchets pass; no new ruff findings (new files report none; no `NEW RUFF FINDINGS` line; explicit paths bypass the config's `tests` exclude and several touched files already have findings on dev, so only new ones count); the OpenAPI check exits 0. Bandit is not installed in the venv: record "bandit: environment skip" in the DoD note unless it is available.

- [ ] **Step 3: File the rate-limit follow-up**

```bash
PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd "$PWD" task create \
  "Per-user rate limits on password-checked endpoints and self-service email changes" -l authnz,security \
  -d "A wrong current password is answered without any per-user limit on PUT /api/v1/users/me, PATCH /api/v1/users/me/profile (email change, TASK-15640) and POST /api/v1/users/change-password, so a signed-in session can guess the password. Successful self-service email changes are not limited either, and each one mails the previous address: an account can set its email to someone else's address and change it away again, repeatedly, to flood that inbox. Add a per-user limit on failed password checks shared by these endpoints, and a per-user cap on successful self-service email changes." \
  --ac "Repeated wrong current passwords on these endpoints are refused after a per-user limit, on SQLite and PostgreSQL" \
  --ac "The limit does not lock the account or affect sign-in" \
  --ac "A per-user cap on successful self-service email changes (e.g. 5 per 24 h) on PUT /users/me and PATCH v1/v2 users/me/profile" \
  --id TASK-15641
git add -A backlog/tasks
```

- [ ] **Step 4: Close out the Backlog task**

`task edit TASK-15640 -s Done --check-ac 1 ... --check-ac 6 --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6 --append-notes "<tests run, Postgres skips, bandit note, known limits>" --final-summary "<what changed>"`, then `git add -A backlog/tasks` and commit with the trailer. The six DoD items come from `backlog/config.yml:15`; #6 is "Known skips or blockers documented". `Done` is set in the PR branch, as recent tasks do (TASK-15582 closed in its own PR), so it lands with the merge.

- [ ] **Step 5: Open the PR**

```bash
git push -u origin feat/task-15640-email-change-hardening
gh pr create --base dev --title "feat(authnz): require the current password for self-service email changes (TASK-15640)" --body-file <body.md>
```
The body lists:
- the four self-service paths and the three admin paths, and how each behaves (exactly as in "Decisions" below);
- the new columns and migrations, with this upgrade note: non-owner PostgreSQL deployments refuse to start with `AUTHNZ_CORE_SCHEMA_NOT_READY` (`services/startup_auth.py:122-164`) until the owner migration (`python -m tldw_Server_API.app.core.AuthNZ.initialize` or an owner-role start) has run;
- the notice, the chatbook restore change, the deviations and the known limits below, and the follow-up TASK-15641;
- the pre-existing recipient logging in `email_service.py`, left out of scope;
- the Claude Code line at the end.

Run the clean-subagent review and wait for green CI. No auto-merge.

---

## Deviations from spec

1. **Four self-service paths, not two.** A user's own session changes the email through `PUT /api/v1/users/me`, `PATCH /api/v1/users/me/profile`, `PATCH /api/v2/users/me/profile` (same `ProfileCommandService` as v1), and the account restore of a chatbook import (`ChatbookService._restore_account_state_payloads`, `chatbook_service.py:3635-3639`, no password at all). The gate sits in `ProfileCommandService.apply` (v1, v2) plus the PUT handler; chatbook restore stops writing the email (Task 3). Three admin paths also change emails: `PUT /api/v1/admin/users/{id}`, `PATCH /api/v1/admin/users/{id}/profile` and `POST /api/v1/admin/users/profile/bulk`; their behaviour is listed under Decisions.
2. **"No password (SSO only)" is not what the code stores.** OIDC JIT provisioning (`auth.py:895`) and first magic-link sign-in (`auth.py:3143`) create accounts with a random generated password the user never sees. Only an empty `password_hash` (the single-user bootstrap row, `initialize.py:913`; `admin_rbac.py:666,682`) can be recognised. Those accounts get `email_change_requires_admin`. JIT and magic-link accounts get `current_password_invalid` and can set a password with "Forgot password" first, which mails the current address.
3. **PostgreSQL columns come from the owner migration, not the users DDL.** The write guard accepts only the canonical `CREATE TABLE public.users` column set (`profile_user_write_guard.py:48-77,879-886`). Both columns are therefore added by `_ensure_users_email_change_columns`, called from `ensure_authnz_core_tables_pg`, which probes `information_schema.columns` and alters only a missing column. They are required by the non-owner serving check, so a non-owner PostgreSQL deployment must run the owner migration (initialize) before the new version serves. That is the repo's existing rule for schema changes.
4. **The 7-day both-addresses rule is not consumed in PR 0.** No lock or unlock mail exists yet. PR 0 stores the data; PR 3 sends to `previous_email` as well when `email_changed_at` is less than 7 days old. That applies to the `account_locked` mail from `POST /api/v1/auth/account/lock` and to the locked-account branches of the magic-link request and forgot-password.
5. **One admin path asks for a password.** The spec says admin email changes are unchanged. `PATCH /api/v1/admin/users/{id}/profile` with the admin's own id goes through `ProfileCommandService` with actor == target, so the shared gate applies: it needs `current_password` (passed through by `admin_profiles_service.update_user_profile`), records the previous address, and sends no notice. Every other admin change, of any user, is unchanged.

## Decisions the spec left open

- Refusal status 403 (not 401, so clients do not treat it as a lost session), with the three codes in Global Constraints.
- Single-user mode is exempt from the password check: the API key is the only credential and lock is not offered there. The previous address and the notice still apply.
- A second change within 7 days keeps the original previous address; otherwise two quick changes would push the owner's address out before a lock.
- An unchanged address (any letter case) and dry runs need no password and send no notice.
- Admin sessions are outside this threat model. Exact behaviour of the admin paths:
  - `PUT /api/v1/admin/users/{id}` (`admin_users_service.py:395-402`), any target including the admin's own account: no password, no previous address, no notice. Code unchanged.
  - `POST /api/v1/admin/users/profile/bulk` (`admin_profiles_service.py:934-975`, calls `UserProfileUpdateService.apply_updates` directly, never the command service): no password for any target. For another user: no previous address, no notice. For the admin's own row (`updated_by == user_id`) the previous address and change time **are** recorded, but there is no password and no notice. This is the simplest result of the Task 2 rule and is stated in the PR body.
  - `PATCH /api/v1/admin/users/{id}/profile` for another user: no password, no previous address, no notice. For the admin's own id, see Deviation 5.
  - On both profile paths `identity.email` is only editable when the admin principal also holds the `user` role (catalog `editable_by: [user]`); otherwise the key is refused as `forbidden`, as today.
- The notice goes to the old address only, after the transaction commits, best effort. It shows the masked new address, UTC time and IP, and what to do. It does not mention the 7-day rule until PR 3 makes it true.
- `is_verified` is left as is on change. Resetting it would lock the user out of every endpoint guarded by `_require_active_verified_user`.
- Chatbook export still includes the email; restore ignores it.
- No per-user limit on wrong passwords or on successful email changes in this PR: a signed-in session can keep guessing through `PUT /api/v1/users/me` and `PATCH /api/v1/users/me/profile`, as it already can through `POST /api/v1/users/change-password`. Follow-up TASK-15641 (filed in Task 6) adds a per-user limit for all password-checked endpoints and a cap on successful self-service email changes.
- `email_service.py` already logs recipient addresses (`:642`, `:834`, `:838`). That is pre-existing and out of scope; the new code logs user ids only.

## Known limits

- With a stolen password, a change made inside the 7-day window after the owner's own change keeps the older previous address. Example: the owner changes Z to A, then the intruder changes A to B within 7 days. The result is `email = B`, `previous_email = Z`, so A (the owner's current mailbox) gets only the change notice and no later lock mail. One `previous_email` column cannot fix this; PR 3 or a later change would need a short history of addresses.
- The old-address notice can be used to flood someone's inbox. The new address is not verified, so any account owner can set their email to another person's (unused) address and change it away again; each change away mails that person, with no limit (`PUT /api/v1/users/me`, `PATCH /api/v1/users/me/profile` and v2). TASK-15641 adds a per-user cap on successful self-service email changes.
- Sending lock mail to both addresses for 7 days lets the owner learn of a takeover, but does not by itself let the owner recover when the intruder has also changed the password. PR 3 must decide whether a link sent to `previous_email` within 7 days may restore that address, or whether forgot-password within 7 days also mails `previous_email`.
