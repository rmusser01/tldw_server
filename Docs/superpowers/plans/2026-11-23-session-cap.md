# Session Cap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bound every user to `MAX_SESSIONS_PER_USER` (default 4) live AuthNZ sessions in both auth modes and on SQLite and PostgreSQL. A new sign-in replaces the oldest session from the same browser install first, then the user's oldest session. Sign-in is never refused because of the cap.

**Architecture:** One repo method counts, evicts and inserts inside a single per-user-serialized transaction (a PostgreSQL two-key advisory lock, or SQLite `BEGIN IMMEDIATE`). Every `SessionManager.create_session` call goes through it. One "live" predicate, `is_active AND NOT is_revoked AND COALESCE(refresh_expires_at, expires_at) > now`, drives the cap, the session listing and cleanup. A random per-install id, sent by the WebUI and the extension in the `X-TLDW-Device-Id` header, is stored in `sessions.device_id` and used only to group a browser's own sessions for eviction. A request on a cap-evicted session gets 401 with code `session_replaced`, and the WebUI sign-in page explains it.

**Tech Stack:** FastAPI, asyncpg / aiosqlite through `AuthNZ/database.py` `DatabasePool`, pydantic-settings, loguru, pytest. On the frontend: TypeScript, Next.js pages router, vitest + Testing Library.

**Spec:** `Docs/Design/2026-11-23-session-cap-alert-lock-design.md`, section "1. Session cap (PR 1)" and the ADR that the spec says is written with PR 1. Sections 0, 2, 3 and 4 of the spec are out of scope.

## Global Constraints

- **Preconditions:** both of these must be merged into `dev` first. Then create branch `feat/session-cap-15628` from a fresh `origin/dev` in a worktree.
  - PR #3578 (TASK-15627, branch `fix/runtime-signin-retry-storm`). It adds the TASK-15628 file and rewrites `bootstrapCookieSession` in `runtime-bootstrap.ts`.
  - The design branch `docs/session-cap-alert-lock-design`: the spec, TASK-15632 and both plans. The coordinator merges it.
- **Function-size ratchet** (`tldw_Server_API/tests/lint/function_size_baseline.json`, threshold 300):
  - `auth:login` is at its ceiling of 454 and must not grow; Task 3 removes more lines than it adds.
  - `User_DB_Handling:verify_jwt_and_fetch_user` is at its ceiling of 436, so Task 4 keeps it net-zero.
  - `auth_principal_resolver:get_auth_principal` is at 301 of its 314 ceiling.
  - New functions stay at or under 300 lines.
- Open one PR with base `dev` (`gh pr create --base dev`). It closes Backlog task TASK-15628.
- Edit the backlog only with `PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd <worktree> task edit TASK-15628 ...`. Stage the result with `git add -A backlog/tasks`.
- `run_py <id> <files…>` in this plan means the following command. The guard deletes the basetemp when the run ends.
  `zsh ~/.cache/claude-tools/disk_guard.zsh "$TMPDIR/sc-<id>" -- /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -p no:cacheprovider --basetemp "$TMPDIR/sc-<id>" -q <files…>`
- AuthNZ PostgreSQL tests use the `isolated_test_environment` fixture from `tldw_Server_API/tests/AuthNZ/conftest.py`. Never build a database by hand. A PostgreSQL step passes only if the output shows the tests **passed**. If they were skipped, say so in the PR and rely on the CI Postgres job.
- Run frontend tests with `cd apps/packages/ui && bunx vitest run <files> --maxWorkers=2` or `cd apps/tldw-frontend && bunx vitest run <files> --maxWorkers=2`. Run one vitest process at a time.
- App code has no raw SQL outside `AuthnzSessionsRepo` and the other DB_Management abstractions. Tests may backdate or bulk-seed rows with raw SQL, as the existing AuthNZ repo tests do.
- Use loguru. Never log tokens, cookie values or install ids. Honour `PII_REDACT_LOGS` the way the neighbouring log lines do.
- No schema change and no migration. `sessions.device_id VARCHAR(255)`, `sessions.revoke_reason` and `idx_sessions_user_id` already exist on both backends. The sessions DDL in `tests/AuthNZ/conftest.py` stays as it is.
- Use these spec values verbatim:
  - cap default `4`, minimum `1`
  - revoke reason `session_cap`
  - 401 code `session_replaced`
  - single-user device tag `single-user-cookie:v1:<id>`
  - eviction order: oldest live session with the same `device_id`, then the user's oldest live session by `created_at` (ties broken by `id`)
  - cap-evicted rows purged after `max(1 hour, ACCESS_TOKEN_EXPIRE_MINUTES)`
  - UI copy "You were signed out because your account was signed in on another device."
- Mirrored docs (`Docs/ADR/*`, `Docs/Operations/Env_Vars.md`): run `Helper_Scripts/refresh_docs_published.sh` and commit the `Docs/Published` diff in the same commit as the source edit.
- Every commit message ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Never use `--no-verify` or `git stash`. Never arm or mention auto-merge.
- No new dependencies, backend or frontend.
- Change summary: owner-written or owner waiver recorded before merge.
- Before merge, run a clean-subagent `/superpowers:requesting-code-review` and compare the non-required Frontend E2E and admin-ui real-backend results with `dev`.

## Review Focus

1. **Backlog of storm rows.** The TASK-15627 storm made about 37 rows a minute, each live for 30 days, so an install can hold tens of thousands of live `single-user-cookie:v1` rows. That install signs in once. Expected:
   - one creation evicts down to the cap with one SELECT and one UPDATE, with no per-row loop or per-row transaction;
   - no statement binds one variable per victim (SQLite fails above 32,766 variables, asyncpg at 32,767);
   - the sign-in succeeds.

   Pinned by Task 2 `test_cap_backlog_is_evicted_in_one_creation` and its `_postgres` twin, which seed 33,000 rows.
2. **`MAX_SESSIONS_PER_USER=1`.** Each sign-in replaces the previous session and is never refused. Pinned by Task 2 `test_cap_of_one_replaces_the_only_session`.
3. **Malformed or hostile `X-TLDW-Device-Id`.** Examples: 65 characters, spaces, a colon, the literal `single-user-cookie:v1`. Expected: treated as absent, never stored, never forwarded by the Next proxy. Pinned by Task 3 `test_request_device_id_accepts_only_safe_ids` and Task 5 "drops a malformed install id".
4. **Browser storage or `crypto.randomUUID` unavailable.** Storage can be blocked (private window, storage disabled), and `randomUUID` is missing on plain-HTTP origins other than localhost. Expected:
   - sign-in never fails because of the id;
   - with `randomUUID` missing, the id falls back to `crypto.getRandomValues` hex;
   - with no storage and no `crypto`, no header is sent. Multi-user then evicts by the default rule. A single-user mint joins the bare `single-user-cookie:v1` group (Deviation 9).

   Pinned by Task 5 "omits the header when storage is unavailable" and "falls back to getRandomValues".
5. **MFA login started before the deploy and completed after it.** The cached payload still carries `session_id`. Expected: completion creates its own row and ignores the stale id; the old placeholder expires with the MFA TTL. Pinned by Task 3's update to `test_mfa_login_completes_tokens[True]`.

---

### Task 1: One live-session predicate for listing and cleanup

**Files:**
- Modify: `tldw_Server_API/app/core/AuthNZ/repos/sessions_repo.py`
  - add module constants after the imports (lines 1-11)
  - `get_active_sessions_for_user` (lines 629-699)
  - `cleanup_expired_sessions` (lines 701-759)
- Modify: `tldw_Server_API/app/core/AuthNZ/session_manager.py:1700-1723` (`cleanup_expired_sessions`)
- Modify: `tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_sessions_repo_sqlite.py:94-128`, in `test_authnz_sessions_repo_basic_crud_sqlite`:
  - The seeded old row's `refresh_expires_at` changes from `now - timedelta(days=1)` to `now - timedelta(days=2)`. Under the new predicate, cleanup compares `datetime(...)` at one-second resolution against exactly `now - 1 day`. The row is therefore usually not deleted, and `deleted >= 1` would flake.
  - The call becomes `repo.cleanup_expired_sessions(session_cap_retention=timedelta(hours=1))`.
- Test (create): `tldw_Server_API/tests/AuthNZ_SQLite/test_session_cap_sqlite.py`
- Test (create): `tldw_Server_API/tests/AuthNZ/integration/test_session_cap_postgres.py`

**Interfaces:**
- Produces, in `sessions_repo.py`:
  - `SESSION_CAP_REVOKE_REASON = "session_cap"`
  - the live-predicate SQL fragment, one module constant per backend
  - `AuthnzSessionsRepo.get_active_sessions_for_user(user_id: int) -> list[dict[str, Any]]`: returns only live rows, with the same keys as today. `expires_at` now holds the session end, `COALESCE(refresh_expires_at, expires_at)`.
  - `AuthnzSessionsRepo.cleanup_expired_sessions(*, session_cap_retention: timedelta) -> int`. The keyword is required.
- Produces, in the SQLite test module, the fixture and helper that later tasks reuse:
  - fixture `authnz_sqlite` → `(pool, user_id, other_user_id)`. Build it like `test_authnz_sessions_repo_sqlite.py`: `pytest_plugins = ("tldw_Server_API.tests._plugins.authnz_full_fixtures",)`, `fresh_services`, a `tmp_path` DB, `ensure_authnz_tables`, `UsersDB.create_user`. Also `monkeypatch.setenv("SESSION_ENCRYPTION_KEY", "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=")`, as `test_single_user_cookie_session.py` does, so `SessionManager` never writes a key file.
  - helper `async def _row(pool, user_id, tag, *, access_in, refresh_in, device_id=None) -> int`. It inserts through the uncapped `create_session_record` with `token_hash=f"h-{tag}"` and `refresh_token_hash=f"r-{tag}"`; the refresh hash is UNIQUE.
  - helper `async def _revoke(pool, session_id, reason, *, minutes_ago=0)`. It calls `revoke_session_record` and then backdates `revoked_at`.
  - helpers `async def _fetch(pool, session_id) -> dict` (returns `is_active`, `revoke_reason`, `device_id`) and `async def live_ids(pool, user_id) -> set[int]` (wraps `get_active_sessions_for_user`). The test snippets below call them without `await` for brevity.

- [ ] **Step 1: Write the failing SQLite tests** in `test_session_cap_sqlite.py`. Each test seeds the same four rows:

| Row | Access token expires | Refresh token expires | State |
| --- | --- | --- | --- |
| `live` | in 1 h | in 7 d | live |
| `idle` | 2 d ago | in 5 d | live; today's cleanup deletes it because the access token expired more than 1 d ago |
| `dead` | 3 d ago | 2 d ago | expired |
| `revoked` | in 1 h | in 7 d | revoked with "User logout" |

```python
async def test_listing_uses_live_predicate(authnz_sqlite):
    ids = {r["id"] for r in await AuthnzSessionsRepo(pool).get_active_sessions_for_user(uid)}
    assert ids == {live, idle}

async def test_listing_reports_session_end_as_expires_at(authnz_sqlite):
    row = next(r for r in rows if r["id"] == idle)
    assert abs((_aware(row["expires_at"]) - idle_refresh_exp).total_seconds()) < 2

async def test_cleanup_keeps_idle_refreshable_and_deletes_dead_after_grace(authnz_sqlite):
    await SessionManager(db_pool=pool).cleanup_expired_sessions()
    assert remaining_ids >= {live, idle} and dead not in remaining_ids

@pytest.mark.parametrize("access_minutes,minutes_ago,kept", [(30, 61, False), (30, 30, True), (120, 90, True)])
async def test_cleanup_purges_cap_evicted_rows_after_retention(authnz_sqlite, monkeypatch, access_minutes, minutes_ago, kept):
    # ACCESS_TOKEN_EXPIRE_MINUTES via monkeypatch.setenv + reset_settings(); fresh SessionManager(db_pool=pool)
    # cap row: _revoke(..., "session_cap", minutes_ago=minutes_ago); control row: _revoke(..., "User logout", minutes_ago=61)
    assert (cap_row in remaining_ids) is kept
    assert logout_row in remaining_ids           # ordinary revocations still keep 7 days
```

- [ ] **Step 2: Write the failing caller tests** in the same module. These cover every caller of `get_active_sessions` / `get_user_sessions` outside the HTTP aliases; the aliases are covered in Task 3. Each test monkeypatches the caller module's `get_session_manager` to return `SessionManager(db_pool=pool)`. "Dead-only" means only the `dead` and `revoked` rows exist; "idle" adds the `idle` row.

```python
@pytest.mark.parametrize("with_idle,expect_run", [(False, False), (True, True)])
async def test_workflows_require_online_uses_live_predicate(authnz_sqlite, monkeypatch, with_idle, expect_run):
    # harness of tests/Workflows/test_workflows_scheduler.py::test_run_schedule_submits_with_resolved_owner_user_id,
    # schedule.require_online=True, owner=uid; _StubDB.set_history records kwargs
    await svc._run_schedule("s1", user_id=uid)
    assert ("payload" in captured) is expect_run
    assert (history[-1].get("last_status") == "skipped_offline") is (not expect_run)

@pytest.mark.parametrize("with_idle,expect_run", [(False, False), (True, True)])
async def test_reading_digest_require_online_uses_live_predicate(authnz_sqlite, monkeypatch, with_idle, expect_run):
    # reading_digest_scheduler.JobManager patched to a stub recording create_job; stub db returns an enabled
    # schedule (cron "0 * * * *", require_online=True, user_id=uid) and try_claim_reading_digest_run -> True
    await _ReadingDigestScheduler()._run_schedule("d1", user_id=uid)
    assert bool(jobs.created) is expect_run

async def test_admin_session_listing_uses_live_predicate(authnz_sqlite, monkeypatch):
    # admin_scope_service.enforce_admin_user_scope patched to an async no-op
    out = await admin_sessions_mfa_service.list_user_sessions(principal, uid, SessionManager(db_pool=pool))
    assert {s.id for s in out} == {live, idle}

async def test_profile_security_counts_live_sessions(authnz_sqlite, monkeypatch):
    # service.get_mfa_service patched (enabled False); api_key_manager stub list_user_keys -> []
    sec = await UserProfileService(pool).build_security(user_id=uid, session_manager=SessionManager(db_pool=pool), api_key_manager=stub)
    assert sec["active_sessions"] == 2
```

- [ ] **Step 3: Write the failing PostgreSQL test** in `test_session_cap_postgres.py`. It uses `isolated_test_environment`, `get_db_pool()` and a `_create_user` helper like `test_authnz_sessions_repo_postgres.py`, and seeds the same four rows (`idle`: access expired 2 d ago, refresh valid 5 d).

```python
async def test_live_predicate_and_cleanup_postgres(isolated_test_environment):
    assert {r["id"] for r in await repo.get_active_sessions_for_user(uid)} == {live, idle}
    # add cap rows revoked 61 and 30 minutes ago
    await repo.cleanup_expired_sessions(session_cap_retention=timedelta(hours=1))
    assert cap_61 not in remaining and cap_30 in remaining
    assert dead not in remaining and idle in remaining
```

- [ ] **Step 4: Run the tests and confirm they fail**

Run: `run_py t1-red tldw_Server_API/tests/AuthNZ_SQLite/test_session_cap_sqlite.py tldw_Server_API/tests/AuthNZ/integration/test_session_cap_postgres.py`
Expected: FAIL, for these reasons:
- The listing also returns `dead`, because today's query filters on `is_active` only.
- The listed `expires_at` is the access expiry.
- Today's cleanup deletes `idle`, because its access token expired more than 1 day ago.
- The cap row revoked 61 minutes ago survives under the 7-day rule.
- The PostgreSQL test raises `TypeError` on `session_cap_retention`.

- [ ] **Step 5: Implement**

1. Add the live predicate to `sessions_repo.py`, one constant per backend:
   - SQL text: `is_active = TRUE AND COALESCE(is_revoked, FALSE) = FALSE AND COALESCE(refresh_expires_at, expires_at) > CURRENT_TIMESTAMP`
   - SQLite form: `datetime(COALESCE(refresh_expires_at, expires_at)) > datetime('now')`
2. Use the predicate in both branches of `get_active_sessions_for_user`, including the SQLite fallback for a missing `last_activity` column. Select `COALESCE(refresh_expires_at, expires_at) AS expires_at`.
3. Change `cleanup_expired_sessions` to delete three kinds of rows:
   - `COALESCE(refresh_expires_at, expires_at)` older than 1 day (same grace as today)
   - inactive rows revoked more than 7 days ago (as today)
   - rows with `revoke_reason = 'session_cap'` revoked longer ago than `session_cap_retention`
4. In `SessionManager.cleanup_expired_sessions`, pass `session_cap_retention=timedelta(minutes=max(60, self.settings.ACCESS_TOKEN_EXPIRE_MINUTES))`.

- [ ] **Step 6: Run the tests and confirm they pass**

Run: the Step 4 command, plus `tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_sessions_repo_sqlite.py tldw_Server_API/tests/AuthNZ_Postgres/test_session_cleanup_pg.py tldw_Server_API/tests/Workflows/test_workflows_scheduler.py tldw_Server_API/tests/Admin/test_admin_sessions_mfa_service_sanitizers.py`
Expected: PASS. The PostgreSQL tests run and are not skipped.

- [ ] **Step 7: Commit**

```bash
git add tldw_Server_API/app/core/AuthNZ/repos/sessions_repo.py tldw_Server_API/app/core/AuthNZ/session_manager.py \
  tldw_Server_API/tests/AuthNZ_SQLite/test_session_cap_sqlite.py tldw_Server_API/tests/AuthNZ/integration/test_session_cap_postgres.py \
  tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_sessions_repo_sqlite.py
git commit -m "fix(authnz): count a session live until its refresh token expires (TASK-15628)"
```

---

### Task 2: Capped session insert and `MAX_SESSIONS_PER_USER`

**Files:**
- Modify: `tldw_Server_API/app/core/AuthNZ/settings.py:209-212`. Add the new field after `REFRESH_TOKEN_EXPIRE_DAYS`.
- Modify: `tldw_Server_API/app/core/AuthNZ/repos/sessions_repo.py:59-146`
  - extract `_insert_session_row`
  - add `create_session_record_with_cap`
- Modify: `tldw_Server_API/app/core/AuthNZ/session_manager.py:1106-1122` (`create_session`)
- Modify: `Docs/Operations/Env_Vars.md:415`, plus the `Docs/Published/Env_Vars.md` mirror
- Test (create): `tldw_Server_API/tests/AuthNZ/unit/test_session_cap_settings.py`
- Test: `tldw_Server_API/tests/AuthNZ_SQLite/test_session_cap_sqlite.py`, `tldw_Server_API/tests/AuthNZ/integration/test_session_cap_postgres.py`

**Interfaces:**
- Consumes: the live predicate and `SESSION_CAP_REVOKE_REASON` (Task 1).
- Produces:
  - `Settings.MAX_SESSIONS_PER_USER: int = Field(default=4, ge=1, description="Live signed-in sessions per user; creating one more revokes the oldest")`
  - The capped insert, which returns the new id and the evicted rows. Each evicted row is a dict with `id, user_id, access_jti, refresh_jti, expires_at, refresh_expires_at`, normalised with `_normalize_session_details`.
    ```python
    AuthnzSessionsRepo.create_session_record_with_cap(
        *, user_id: int, token_hash: str, refresh_token_hash: str,
        encrypted_token: str, encrypted_refresh: str,
        expires_at: datetime, refresh_expires_at: datetime | None,
        ip_address: str | None, user_agent: str | None, device_id: str | None,
        access_jti: str | None, refresh_jti: str | None,
        max_live_sessions: int,
    ) -> tuple[int, list[dict[str, Any]]]
    ```
  - No placeholder for the account lock. PR 3 adds its own re-read inside this transaction, right after the lock.
  - `create_session_record` keeps its signature as the uncapped low-level insert that the repo tests use. Its docstring states that `SessionManager` uses the capped method.
  - `SessionManager.create_session(...)`: signature and return dict unchanged. It now always enforces the cap.

- [ ] **Step 1: Write the failing settings tests** in `test_session_cap_settings.py`, following the `make_settings` pattern from `test_settings_single_user_session_cookie.py`:

```python
def test_max_sessions_per_user_defaults_to_four(monkeypatch):
    monkeypatch.delenv("MAX_SESSIONS_PER_USER", raising=False)
    assert make_settings().MAX_SESSIONS_PER_USER == 4

@pytest.mark.parametrize("value", [0, -1])
def test_max_sessions_per_user_below_one_is_rejected(value):
    with pytest.raises(ValidationError):
        make_settings(MAX_SESSIONS_PER_USER=value)
```

- [ ] **Step 2: Write the failing SQLite cap tests** in `test_session_cap_sqlite.py`.
  - `manager` is `SessionManager(db_pool=pool)`.
  - `_create(manager, uid, device=None)` calls `manager.create_session` with fresh `temp_access_*` / `temp_refresh_*` tokens.
  - `live_ids(uid)` is the set of ids from `get_active_sessions_for_user`.

```python
async def test_fifth_session_evicts_the_oldest(authnz_sqlite):
    other = await _create(manager, other_uid)
    ids = [await _create(manager, uid) for _ in range(5)]
    assert live_ids(uid) == set(ids[1:])
    assert row(ids[0]).revoke_reason == "session_cap" and not row(ids[0]).is_active
    assert other in live_ids(other_uid)

async def test_same_device_id_replaces_only_its_own_row(authnz_sqlite):
    a, b, c = [await _create(manager, uid, d) for d in ("dev-aaaaaaaa", "dev-bbbbbbbb", "dev-cccccccc")]
    for _ in range(6):
        await _create(manager, uid, "dev-dddddddd")
    rows = await repo.get_active_sessions_for_user(uid)
    assert Counter(r["device_id"] for r in rows) == {"dev-aaaaaaaa": 1, "dev-bbbbbbbb": 1, "dev-cccccccc": 1, "dev-dddddddd": 1}
    assert {a, b, c} <= live_ids(uid)

async def test_expired_refresh_rows_do_not_count_toward_the_cap(authnz_sqlite):
    # 4 dead rows (refresh expired) + 3 live rows via _row
    await _create(manager, uid)
    assert len(live_ids(uid)) == 4 and not rows_with_reason("session_cap")

async def test_concurrent_creates_at_the_cap_never_exceed_it(authnz_sqlite):
    for _ in range(4):
        await _create(manager, uid)
    results = await asyncio.gather(*[_create(manager, uid) for _ in range(6)])
    assert len(results) == 6 and len(live_ids(uid)) == 4

async def test_cap_backlog_is_evicted_in_one_creation(authnz_sqlite):
    # 33,000 live rows (more than SQLite's 32,766-variable limit), device_id "single-user-cookie:v1".
    # Seed with one INSERT ... SELECT over a recursive CTE counter, with unique token_hash/refresh_token_hash.
    new_id = await _create(manager, uid)
    assert len(live_ids(uid)) == 4 and new_id in live_ids(uid)

async def test_cap_of_one_replaces_the_only_session(authnz_sqlite, monkeypatch):
    # MAX_SESSIONS_PER_USER=1 + reset_settings(); fresh manager
    first, second = await _create(manager, uid), await _create(manager, uid)
    assert live_ids(uid) == {second}

async def test_evicted_tokens_are_blacklisted_after_commit(authnz_sqlite, monkeypatch):
    # record manager._blacklist_session_tokens calls
    ids = [await _create(manager, uid) for _ in range(5)]
    assert [s["id"] for s in calls[-1].args[0]] == [ids[0]]
    assert calls[-1].kwargs["reason"] == "session_cap"
```

- [ ] **Step 3: Write the failing PostgreSQL cap tests** in `test_session_cap_postgres.py`. They call the repo directly with `max_live_sessions=4`:
  - `test_fifth_session_evicts_the_oldest_postgres`: the same assertions as the SQLite test.
  - `test_concurrent_creates_at_the_cap_never_exceed_it_postgres`: seed 4, run `asyncio.gather` over 6 `create_session_record_with_cap` calls, then `len(live) == 4`.
  - `test_expired_refresh_rows_do_not_count_toward_the_cap_postgres`.
  - `test_cap_backlog_is_evicted_in_one_creation_postgres`: seed 33,000 live rows (more than asyncpg's 32,767-argument limit) with one `INSERT … SELECT … FROM generate_series(1, 33000)`, then one capped insert, then `len(live) == 4`.

- [ ] **Step 4: Run the tests and confirm they fail**

Run: `run_py t2-red tldw_Server_API/tests/AuthNZ/unit/test_session_cap_settings.py tldw_Server_API/tests/AuthNZ_SQLite/test_session_cap_sqlite.py tldw_Server_API/tests/AuthNZ/integration/test_session_cap_postgres.py`
Expected: FAIL. `MAX_SESSIONS_PER_USER` does not exist, `create_session_record_with_cap` does not exist, and the live count reaches 5.

- [ ] **Step 5: Implement**

1. Add the setting.
2. Extract the backend-specific INSERT from `create_session_record` into `_insert_session_row(conn, **fields) -> int`.
3. Implement `create_session_record_with_cap`. It runs one `self.db_pool.transaction()`, which is already `BEGIN IMMEDIATE` on SQLite.
   - Every statement runs on that transaction's `conn`, never on `self.db_pool.fetch*`, which would use another connection outside the lock.
   - The PostgreSQL advisory lock serializes correctly only under READ COMMITTED. That is the server default, and `DatabasePool.transaction()` uses it (`conn.transaction()` with no isolation argument). Do not raise the isolation level.

   Inside the transaction:
   1. On PostgreSQL only, the first statement is `await acquire_schema_lock(conn.execute, "authnz_sessions_cap", str(user_id))` from `tldw_Server_API/app/core/DB_Management/backends/postgresql_locks.py`. It takes the two-key `pg_advisory_xact_lock(hashtext($1), hashtext($2))`; `AuthNZ/repos/token_blacklist_repo.py:11` already imports it.
   2. Count the user's live rows.
   3. If `excess = live - (max_live_sessions - 1)` is greater than 0:
      - The victim order is `ORDER BY CASE WHEN device_id = ? THEN 0 ELSE 1 END, created_at, id LIMIT ?`. There is no same-device preference when `device_id` is None.
      - Read the victims' details (`id, user_id, access_jti, refresh_jti, expires_at, refresh_expires_at`) with one SELECT: the live rows of `user_id` in that order and limit.
      - Revoke them with one UPDATE that binds no per-victim values: `UPDATE sessions SET is_active = <false>, is_revoked = <true>, revoked_at = <now>, revoked_by = NULL, revoke_reason = 'session_cap' WHERE id IN (SELECT id FROM sessions WHERE user_id = ? AND <live> ORDER BY <same order> LIMIT ?)`. It runs after the detail SELECT in the same transaction, so both statements see the same rows. On PostgreSQL, `WHERE id = ANY($1::int[])` with the selected ids as one array parameter is also acceptable.
      - Do not call `revoke_session_record`.
   4. Insert the new row with `_insert_session_row`.
   5. Return `(new_id, victim_details)`.
4. In `SessionManager.create_session`, call the capped method with `max_live_sessions=self.settings.MAX_SESSIONS_PER_USER`. After the transaction returns, if `evicted`:
   - `await self._blacklist_session_tokens(evicted, reason=SESSION_CAP_REVOKE_REASON, revoked_by=None)`
   - log the evicted count. The log carries no ids when `PII_REDACT_LOGS` is set.
   - Do not clear the Redis cache. `validate_session` always re-reads the row by id with `is_active`, and `_clear_user_sessions_cache` scans every user's `session:*` keys.
5. Add to `Docs/Operations/Env_Vars.md` after the `REFRESH_TOKEN_EXPIRE_DAYS` line: ``- `MAX_SESSIONS_PER_USER`: Live signed-in sessions per user (default 4, minimum 1). One more sign-in revokes the oldest session of the same browser install, else the user's oldest; sign-in is never refused. API keys are not sessions.``
6. Run `Helper_Scripts/refresh_docs_published.sh`.

- [ ] **Step 6: Run the tests and confirm they pass**

Run: the Step 4 command, plus `tldw_Server_API/tests/lint/test_env_vars_docs_ratchet.py tldw_Server_API/tests/Docs tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_sessions_repo_sqlite.py tldw_Server_API/tests/AuthNZ/integration/test_authnz_sessions_repo_postgres.py tldw_Server_API/tests/AuthNZ/unit/test_session_revocation_blacklist.py tldw_Server_API/tests/AuthNZ/unit/test_session_refresh_cache.py`
Expected: PASS. The PostgreSQL tests are not skipped.

- [ ] **Step 7: Commit**

```bash
git add tldw_Server_API/app/core/AuthNZ/settings.py tldw_Server_API/app/core/AuthNZ/repos/sessions_repo.py \
  tldw_Server_API/app/core/AuthNZ/session_manager.py tldw_Server_API/tests/AuthNZ/unit/test_session_cap_settings.py \
  tldw_Server_API/tests/AuthNZ_SQLite/test_session_cap_sqlite.py tldw_Server_API/tests/AuthNZ/integration/test_session_cap_postgres.py \
  Docs/Operations/Env_Vars.md Docs/Published
git commit -m "feat(authnz): cap live sessions per user and evict same-install first (TASK-15628)"
```

---

### Task 3: Per-install id and every session creator

**Files:**
- Modify: `tldw_Server_API/app/core/AuthNZ/session_manager.py`. Add the module-level header constant and helper near the other module constants (lines 65-96).
- Modify: `tldw_Server_API/app/core/AuthNZ/single_user_session.py:21,51-74,96`
- Modify: `tldw_Server_API/app/api/v1/endpoints/auth.py`
  - `_issue_multi_user_tokens` create call (lines 410-416)
  - single-user `/login` row (lines 1859-1866)
  - MFA branch and multi-user create (lines 1868-1920)
  - `verify_magic_link` create (lines 3200-3206)
  - `mfa_login` (lines 3525-3649)
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_single_user_session.py`, `tldw_Server_API/tests/AuthNZ/unit/test_auth_endpoints_extended.py` (lines 936-1192), `tldw_Server_API/tests/AuthNZ_SQLite/test_mfa_login_sqlite.py`, `tldw_Server_API/tests/AuthNZ/integration/test_single_user_login_api_key.py`, `tldw_Server_API/tests/AuthNZ/integration/test_single_user_cookie_session.py`
- Test (create): `tldw_Server_API/tests/AuthNZ_SQLite/test_session_cap_http_sqlite.py`

**Interfaces:**
- Consumes: the capped `SessionManager.create_session` (Task 2).
- Produces:
  - `DEVICE_ID_HEADER = "X-TLDW-Device-Id"`
  - `request_device_id(request: Any) -> Optional[str]`: returns the header value only if it matches `^[A-Za-z0-9_-]{8,64}$`, else None. Both live in `session_manager.py`.
  - `is_single_user_cookie_device(value: object) -> bool` in `single_user_session.py`: `value == SESSION_DEVICE_ID or str(value).startswith(SESSION_DEVICE_ID + ":")`.
  - `mint_single_user_session` stores `f"{SESSION_DEVICE_ID}:{id}"` when the request carries a valid id, and the bare `SESSION_DEVICE_ID` otherwise.
  - The HTTP module fixture `multi_user_client` → `(client, user_id, password)`, built like `test_mfa_login_sqlite.py`: multi-user SQLite, `CSRF_ENABLED=false`, no `REDIS_URL`, and a user with `is_verified=True`. The helper `_login(client, password, device=None) -> dict` posts to `/api/v1/auth/login` with the optional header.

- [ ] **Step 1: Write the failing unit tests**

1. In `test_single_user_session.py`, extend `_request` with `device_id: str | None = None`, which adds the header. Then add:

```python
async def test_mint_tags_session_with_install_id(session_settings):
    await mint_single_user_session(_request(device_id="install-12345678"), manager)
    assert manager.create_session.call_args.kwargs["device_id"] == "single-user-cookie:v1:install-12345678"

async def test_mint_without_install_id_uses_bare_tag(session_settings):
    assert kwargs["device_id"] == "single-user-cookie:v1"

async def test_validation_accepts_tagged_install_rows(session_settings):
    # validate_session returns row with device_id "single-user-cookie:v1:install-12345678"
    assert (await validate_single_user_session(_request(cookie="opaque"), manager)).session_id == 4

@pytest.mark.parametrize("device", ["single-user-cookie:v1x", "single-user-cookie:v2:install-12345678", None])
async def test_validation_rejects_lookalike_device_tags(session_settings, device):
    assert await validate_single_user_session(_request(cookie="opaque"), manager) is None

@pytest.mark.parametrize("raw,expected", [("install-12345678", "install-12345678"), ("short", None), ("a" * 65, None),
                                          ("has space here", None), ("single-user-cookie:v1", None), (None, None)])
def test_request_device_id_accepts_only_safe_ids(raw, expected):
    assert request_device_id(_request(device_id=raw)) == expected
```

2. In `test_auth_endpoints_extended.py`, update the MFA tests:
   - `test_login_returns_mfa_challenge_when_enabled`: the stub's `create_session` raises `AssertionError("no session row before MFA completes")`. Replace the cached-payload assertion with `cached == {"user_id": 5, "login_lockout_key": build_login_client_lockout_key("203.0.113.9", "mfa_user")}`. Delete the `expires_at_override` and delta assertions.
   - `test_mfa_login_completes_tokens` (both parametrizations):
     - The stub's `create_session(**kw)` records `kw` and returns `{"session_id": 56}`.
     - The stub gains `consume_ephemeral_value(key)`, which pops from `self.ephemeral` and records the key in `self.consumed`.
     - Assert `session_manager.created["user_id"] == 5`, `session_manager.updated["session_id"] == 56` and `cache_key in session_manager.consumed`. The last one replaces the `deleted` assertion.
     - The legacy payload still carries `"session_id": 55`; it is ignored.
   - New test, built on the same stubs:

```python
async def test_mfa_login_is_single_use(monkeypatch):
    # get_ephemeral_value always returns the payload (two submits that both read before either consumed);
    # consume_ephemeral_value returns it once, then None
    first = await auth.mfa_login(data=..., request=..., ...)
    assert first.access_token == "ACCESS"
    with pytest.raises(auth.HTTPException) as exc:
        await auth.mfa_login(data=..., request=..., ...)
    assert exc.value.status_code == 400
    assert len(session_manager.created_calls) == 1
```

- [ ] **Step 2: Write the failing integration tests**

1. In `test_mfa_login_sqlite.py`, add assertions to `test_mfa_login_http_sqlite`:
   - after the 202 response: `await AuthnzSessionsRepo(pool).get_active_sessions_for_user(uid) == []`
   - after the 200 response: there is exactly one live row, and its id equals `jose.jwt.get_unverified_claims(tokens["access_token"])["session_id"]`.
2. In `test_single_user_login_api_key.py`:

```python
def test_single_user_login_inserts_no_session_row(single_user_client):
    client, password, api_key = single_user_client
    assert client.post("/api/v1/auth/login", data={"username": "single_user", "password": password}).status_code == 200
    assert client.get("/api/v1/auth/sessions", headers={"Authorization": f"Bearer {api_key}"}).json() == []
```

3. In `test_single_user_cookie_session.py`, add the TASK-15628 regression. `_mint` gains a `device` parameter that sets `X-TLDW-Device-Id`.

The test is sync like the rest of the file, so it asserts only over HTTP: four live rows plus every older install still authenticating means install d holds exactly one row.

```python
def test_install_that_drops_its_cookie_replaces_only_its_own_session(single_user_cookie_client):
    cookies = {}
    for device in ("install-aaaaaaaa", "install-bbbbbbbb", "install-cccccccc"):
        client.cookies.clear()
        assert _mint(client, api_key, device).status_code == 200
        cookies[device] = client.cookies["tldw_single_user_session"]
    for _ in range(10):
        client.cookies.clear()
        assert _mint(client, api_key, "install-dddddddd").status_code == 200
    client.cookies.clear()
    assert len(client.get("/api/v1/auth/sessions", headers={"X-API-KEY": api_key}).json()) == 4
    for cookie in cookies.values():
        client.cookies.clear()
        client.cookies.set("tldw_single_user_session", cookie)
        assert client.get("/api/v1/auth/sessions").status_code == 200
```

4. In the new `test_session_cap_http_sqlite.py`. Its tests are `async def`, like `test_mfa_login_sqlite.py`, so they can call the repo between HTTP calls.

```python
async def test_same_install_login_replaces_only_its_own_session(multi_user_client):
    a, b, c = (_login(client, pw, d) for d in ("dev-aaaaaaaa", "dev-bbbbbbbb", "dev-cccccccc"))
    for _ in range(5):
        _login(client, pw, "dev-dddddddd")
    rows = await AuthnzSessionsRepo(pool).get_active_sessions_for_user(uid)
    assert Counter(r["device_id"] for r in rows) == {"dev-aaaaaaaa": 1, "dev-bbbbbbbb": 1, "dev-cccccccc": 1, "dev-dddddddd": 1}
    assert client.get("/api/v1/auth/sessions", headers=bearer(a)).status_code == 200

async def test_users_sessions_alias_lists_and_revokes_idle_refreshable_session(multi_user_client):
    tok = _login(client, pw)
    idle = insert_row(uid, access_in=-2d, refresh_in=+5d)  # via the repo's uncapped insert
    dead = insert_row(uid, access_in=-3d, refresh_in=-1h)
    for path in ("/api/v1/users/sessions", "/api/v1/auth/sessions"):
        listed = {s["id"] for s in client.get(path, headers=bearer(tok)).json()}
        assert idle in listed and dead not in listed
    assert client.delete(f"/api/v1/users/sessions/{idle}", headers=bearer(tok)).status_code == 200
    assert idle not in {s["id"] for s in client.get("/api/v1/users/sessions", headers=bearer(tok)).json()}
```

- [ ] **Step 3: Run the tests and confirm they fail**

Run: `run_py t3-red tldw_Server_API/tests/AuthNZ/unit/test_single_user_session.py tldw_Server_API/tests/AuthNZ/unit/test_auth_endpoints_extended.py tldw_Server_API/tests/AuthNZ_SQLite/test_mfa_login_sqlite.py tldw_Server_API/tests/AuthNZ/integration/test_single_user_login_api_key.py tldw_Server_API/tests/AuthNZ/integration/test_single_user_cookie_session.py tldw_Server_API/tests/AuthNZ_SQLite/test_session_cap_http_sqlite.py`
Expected: FAIL, for these reasons:
- `request_device_id` cannot be imported.
- The device tag is still the bare constant.
- The MFA placeholder row exists.
- The second MFA completion also returns tokens.
- Single-user `/login` lists a row.

- [ ] **Step 4: Implement**

1. Add `DEVICE_ID_HEADER` and `request_device_id` to `session_manager.py`, and `is_single_user_cookie_device` to `single_user_session.py`.
2. In `single_user_session.py`, the mint composes the device tag and the validator calls `is_single_user_cookie_device`.
3. In `auth.py`:
   - Pass `device_id=request_device_id(request)` to the `create_session` calls in `_issue_multi_user_tokens`, in the multi-user non-MFA login path and in `verify_magic_link`.
   - Delete the single-user `/login` `create_session` call.
   - When `mfa_required`, `login` creates no row and caches `{"user_id", "login_lockout_key"}`. The `mfa_expires_at` and `mfa_ttl_seconds` overrides go away with the placeholder.
   - `mfa_login`:
     - Require only `user_id` from the payload and ignore a legacy `session_id`.
     - Right after `token_ok`, call `consume_ephemeral_value(cache_key)` (`session_manager.py:1915`, an atomic get-and-delete). A miss raises 400 "MFA session expired or invalid". A wrong code therefore leaves the challenge usable, and a second concurrent completion cannot create a second session.
     - Then return through `_issue_multi_user_tokens(request=request, db=db, user=user, jwt_service=jwt_service, session_manager=session_manager, settings=settings)`. That helper creates the row with this request's IP, user agent and install id. It replaces the duplicated org-membership / claims / token / `update_session_tokens` / last-login / audit block at lines 3592-3649.
     - Keep the rate-limiter reset and the counters after it. Drop the trailing `delete_ephemeral_value`, because the consume already deleted the value.
   - `login` must end no longer than today; see the function-size constraint.

- [ ] **Step 5: Run the tests and confirm they pass**

Run: the Step 3 command, plus `tldw_Server_API/tests/AuthNZ_SQLite/test_magic_link_redemption.py tldw_Server_API/tests/AuthNZ/unit/test_websocket_session_auth.py tldw_Server_API/tests/AuthNZ/unit/test_csrf_binding.py tldw_Server_API/tests/lint/test_function_size_ratchet.py`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tldw_Server_API/app/core/AuthNZ/session_manager.py tldw_Server_API/app/core/AuthNZ/single_user_session.py \
  tldw_Server_API/app/api/v1/endpoints/auth.py tldw_Server_API/tests/AuthNZ tldw_Server_API/tests/AuthNZ_SQLite
git commit -m "feat(authnz): group sessions by install id, no row before MFA or for single-user /login (TASK-15628)"
```

---

### Task 4: 401 `session_replaced` for cap-evicted sessions

**Files:**
- Modify: `tldw_Server_API/app/core/AuthNZ/exceptions.py:62-69`. Add `SessionReplacedError` after `InvalidTokenError`.
- Modify: `tldw_Server_API/app/core/AuthNZ/repos/sessions_repo.py`. Add the new lookup after `has_revoked_session_for_token_hash_candidates` (lines 453-537).
- Modify: `tldw_Server_API/app/core/AuthNZ/session_manager.py`. Add `session_replaced_http_exception` at module level, and `get_revoke_reason` and `revoked_token_exception` next to `is_token_blacklisted` (lines 1598-1667).
- Modify: `tldw_Server_API/app/core/AuthNZ/single_user_session.py`
- Modify: `tldw_Server_API/app/core/AuthNZ/User_DB_Handling.py:585-595`
- Modify: `tldw_Server_API/app/core/AuthNZ/auth_principal_resolver.py:349-375`
- Modify: `tldw_Server_API/app/api/v1/endpoints/auth.py:2372-2382` and `:2535-2549` (refresh)
- Test: `tldw_Server_API/tests/AuthNZ_SQLite/test_session_cap_http_sqlite.py`, `tldw_Server_API/tests/AuthNZ/integration/test_single_user_cookie_session.py`

**Interfaces:**
- Consumes: `SESSION_CAP_REVOKE_REASON` (Task 1), the cap (Task 2), the HTTP fixtures (Task 3).
- Produces:
  - `class SessionReplacedError(InvalidTokenError)`
  - `AuthnzSessionsRepo.fetch_revoke_reason_for_token_hash_candidates(token_hash_candidates: list[str]) -> str | None`: the `revoke_reason` of a revoked row whose `token_hash` or `refresh_token_hash` matches.
  - `session_replaced_http_exception() -> HTTPException` in `session_manager.py`. It returns `HTTPException(401, detail={"code": "session_replaced", "message": "You were signed out because your account was signed in on another device."}, headers={"WWW-Authenticate": "Bearer"})`.
  - `SessionManager.get_revoke_reason(self, token: str) -> Optional[str]`: returns None on any lookup error.
  - `SessionManager.revoked_token_exception(self, token: str, default: HTTPException) -> HTTPException`: returns `session_replaced_http_exception()` when the reason is `session_cap`, else `default`.
  - `async def cookie_session_was_replaced(request: Any, manager: Any | None = None) -> bool` in `single_user_session.py`

- [ ] **Step 1: Write the failing tests**

1. In `test_session_cap_http_sqlite.py`:

```python
async def test_fifth_login_evicts_oldest_and_its_token_gets_session_replaced(multi_user_client):
    first, *_rest, fifth = [_login(client, pw) for _ in range(5)]
    r = client.get("/api/v1/auth/sessions", headers=bearer(first))
    assert r.status_code == 401 and r.json()["detail"]["code"] == "session_replaced"
    assert len(client.get("/api/v1/auth/sessions", headers=bearer(fifth)).json()) == 4

async def test_evicted_refresh_token_gets_session_replaced(multi_user_client):
    first = _login(client, pw)
    for _ in range(4):
        _login(client, pw)
    r = client.post("/api/v1/auth/refresh", json={"refresh_token": first["refresh_token"]})
    assert r.status_code == 401 and r.json()["detail"]["code"] == "session_replaced"

async def test_logged_out_token_keeps_generic_401(multi_user_client):
    tok = _login(client, pw)
    client.post("/api/v1/auth/logout", headers=bearer(tok))
    r = client.get("/api/v1/auth/sessions", headers=bearer(tok))
    assert r.status_code == 401 and r.json()["detail"] == "Could not validate credentials"
```

2. In `test_single_user_cookie_session.py`:

```python
def test_cap_evicted_cookie_gets_session_replaced(single_user_cookie_client):
    # mint install-aaaaaaaa and keep its cookie; mint four more installs, clearing cookies each time
    client.cookies.clear()
    client.cookies.set("tldw_single_user_session", a_cookie)
    r = client.get("/api/v1/auth/sessions")
    assert r.status_code == 401 and r.json()["detail"]["code"] == "session_replaced"
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `run_py t4-red tldw_Server_API/tests/AuthNZ_SQLite/test_session_cap_http_sqlite.py tldw_Server_API/tests/AuthNZ/integration/test_single_user_cookie_session.py`
Expected: FAIL. The detail is the string "Could not validate credentials", "Invalid or expired refresh token", or "Not authenticated…".

- [ ] **Step 3: Implement**

1. The new repo lookup and `get_revoke_reason`, which uses `_token_hash_candidates(token)`.
2. `cookie_session_was_replaced(request, manager)` reads the cookie, returns False when there is none, and compares `get_revoke_reason(cookie)` with `SESSION_CAP_REVOKE_REASON`.
3. In `verify_jwt_and_fetch_user`, replace only the `raise credentials_exception` inside the blacklist branch (line 590) with `raise await session_manager.revoked_token_exception(token, credentials_exception)`. This keeps the function net-zero against its 436-line ceiling.
4. In `get_auth_principal`, the no-credentials branch: before the generic 401, `if await cookie_session_was_replaced(request): raise session_replaced_http_exception()`. That is two lines, within its 314 ceiling.
5. In `/auth/refresh`, in the blacklist branch: if the reason is `session_cap`, raise `SessionReplacedError()`. In the `except (TokenExpiredError, InvalidTokenError)` handler, `if isinstance(e, SessionReplacedError): raise session_replaced_http_exception() from e` comes before the generic 401.
6. The reason is looked up only on these failure paths, so the success path gets no new queries.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: the Step 2 command, plus `tldw_Server_API/tests/AuthNZ/unit/test_auth_endpoints_extended.py tldw_Server_API/tests/AuthNZ/unit/test_user_db_handling_jwt_membership.py tldw_Server_API/tests/AuthNZ/unit/test_auth_principal_service_and_single_user_tokens.py tldw_Server_API/tests/AuthNZ_Unit/test_auth_refresh_sqlite_connection_scope.py tldw_Server_API/tests/lint/test_function_size_ratchet.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/AuthNZ tldw_Server_API/app/api/v1/endpoints/auth.py \
  tldw_Server_API/tests/AuthNZ_SQLite/test_session_cap_http_sqlite.py tldw_Server_API/tests/AuthNZ/integration/test_single_user_cookie_session.py
git commit -m "feat(authnz): answer a cap-evicted session with 401 session_replaced (TASK-15628)"
```

---

### Task 5: WebUI and extension — install id and the sign-out notice

**Files:**
- Create: `apps/packages/ui/src/services/tldw/session-identity.ts`
- Modify: `apps/packages/ui/src/services/tldw/TldwAuth.ts`
  - `login` headers (line 176)
  - `verifyMagicLink` headers (line 216)
  - `publishLogin` (lines 227-252)
  - `performTokenRefresh` 401 branch (lines 359-370)
- Modify: `apps/packages/ui/src/services/tldw/auth-refresh-error.ts:4-17`
- Modify: `apps/packages/ui/src/services/background-proxy.ts:755-758`, the 401 branch of `refreshAuthDirect`
- Modify: `apps/tldw-frontend/extension/shims/runtime-bootstrap.ts`, the POST in `bootstrapCookieSession` (after #3578)
- Modify: `apps/tldw-frontend/pages/api/_tldw-webui/session.ts:98-104`
- Modify: `apps/tldw-frontend/routes/login.tsx`, `SelfHostLogin` (lines 30-123)
- Test (create): `apps/packages/ui/src/services/tldw/__tests__/session-identity.test.ts`, `apps/packages/ui/src/services/tldw/__tests__/TldwAuth.session-identity.test.ts`. Copy the `vi.mock` preamble of `TldwAuth.refresh.test.ts` and add `invalidateRefreshSession` to the `tldwClient` mock.
- Test: `apps/packages/ui/src/services/__tests__/background-proxy.web-refresh.test.ts`, `apps/tldw-frontend/__tests__/pages/api/runtime-session.test.ts`, `apps/tldw-frontend/__tests__/extension/runtime-bootstrap.test.ts`, `apps/tldw-frontend/__tests__/pages/login-page.test.tsx`

**Interfaces:**
- Consumes: the backend header name `X-TLDW-Device-Id` and id format `^[A-Za-z0-9_-]{8,64}$` (Task 3), and the 401 body `{"detail": {"code": "session_replaced", ...}}` (Task 4).
- Produces (`session-identity.ts`; storage keys `tldw:install-id` and `tldw:session-replaced`; every storage access wrapped in try/catch):
  - `INSTALL_ID_HEADER = "X-TLDW-Device-Id"`
  - `getInstallIdHeaders(): Record<string, string>`. The whole body sits in one try/catch, because sign-in must never fail because of the id.
    - It creates the id once with `crypto.randomUUID()`.
    - When `randomUUID` is absent (plain-HTTP origins other than localhost), it falls back to 16 bytes of `crypto.getRandomValues` as hex, following `generateSessionId` in `apps/tldw-frontend/lib/session.ts:126-145`.
    - It returns `{}` when neither is available or storage throws.
  - `noteSessionReplaced(error: unknown): void`: records the notice only when `status === 401` and `(error.details?.detail ?? error.detail)?.code === "session_replaced"`.
  - `hasSessionReplacedNotice(): boolean`
  - `clearSessionReplacedNotice(): void`
  - `SESSION_REPLACED_MESSAGE = "You were signed out because your account was signed in on another device."`
- `createTokenRefreshError` additionally carries `details: response.data`.

- [ ] **Step 1: Write the failing tests**

```ts
// session-identity.test.ts — beforeEach: localStorage.clear(); vi.unstubAllGlobals(); vi.restoreAllMocks()
it("creates one install id and reuses it", () => {
  const first = getInstallIdHeaders()["X-TLDW-Device-Id"]
  expect(first).toMatch(/^[A-Za-z0-9_-]{8,64}$/)
  expect(getInstallIdHeaders()["X-TLDW-Device-Id"]).toBe(first)
  expect(localStorage.getItem("tldw:install-id")).toBe(first)
})
it("omits the header when storage is unavailable", () => {
  vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => { throw new Error("denied") })
  expect(getInstallIdHeaders()).toEqual({})
})
it("falls back to getRandomValues when randomUUID is missing", () => {
  vi.stubGlobal("crypto", { getRandomValues: (a: Uint8Array) => a.fill(171) })
  expect(getInstallIdHeaders()["X-TLDW-Device-Id"]).toBe("ab".repeat(16))
})
it("omits the header when no crypto is available", () => {
  vi.stubGlobal("crypto", undefined)
  expect(getInstallIdHeaders()).toEqual({})
})
it("records the notice only for a 401 session_replaced", () => {
  noteSessionReplaced({ status: 401, details: { detail: "Invalid or expired refresh token" } })
  noteSessionReplaced({ status: 500, details: { detail: { code: "session_replaced" } } })
  expect(hasSessionReplacedNotice()).toBe(false)
  noteSessionReplaced({ status: 401, details: { detail: { code: "session_replaced" } } })
  expect(hasSessionReplacedNotice()).toBe(true)
  clearSessionReplacedNotice()
  expect(hasSessionReplacedNotice()).toBe(false)
})

// TldwAuth.session-identity.test.ts
it("sends the install id on password login", async () => {
  await tldwAuth.login({ username: "u", password: "p" })
  expect(mocks.bgRequest.mock.calls[0][0].headers["X-TLDW-Device-Id"]).toBe(localStorage.getItem("tldw:install-id"))
})
it("sends the install id on magic-link verify", async () => { /* same assertion for verifyMagicLink("t") */ })
it("clears the replaced-session notice after a successful login", async () => {
  localStorage.setItem("tldw:session-replaced", "1")
  await tldwAuth.login({ username: "u", password: "p" })
  expect(hasSessionReplacedNotice()).toBe(false)
})
it("records session_replaced when refresh is refused with it", async () => {
  mocks.bgRequest.mockRejectedValue({ status: 401, details: { detail: { code: "session_replaced" } } })
  await expect(tldwAuth.refreshToken()).rejects.toBeTruthy()
  expect(hasSessionReplacedNotice()).toBe(true)
})

// background-proxy.web-refresh.test.ts
it("records session_replaced when the web refresh returns it", async () => {
  // /api/v1/auth/refresh fetch -> 401 JSON {detail:{code:"session_replaced"}}; drive a 401 request as the file's existing cases do
  expect(hasSessionReplacedNotice()).toBe(true)
})

// runtime-session.test.ts
it("forwards a well-formed install id to the backend", async () => {
  // makeRequest({ headers: { "x-tldw-device-id": "install-12345678" } })
  expect(mockFetch.mock.calls[0][1].headers["X-TLDW-Device-Id"]).toBe("install-12345678")
})
it("drops a malformed install id", async () => {
  // header "bad id!"
  expect(mockFetch.mock.calls[0][1].headers).not.toHaveProperty("X-TLDW-Device-Id")
})

// runtime-bootstrap.test.ts
it("sends the per-install id on the cookie mint", async () => {
  const [, init] = fetchMock.mock.calls.find(([url]) => url === "/api/_tldw-webui/session")!
  expect(init.headers["X-TLDW-Device-Id"]).toBe(localStorage.getItem("tldw:install-id"))
})

// login-page.test.tsx
it("explains a sign-out caused by a newer sign-in", async () => {
  localStorage.setItem("tldw:session-replaced", "1")
  render(<LoginPage />)
  expect(await screen.findByText("You were signed out because your account was signed in on another device.")).toBeInTheDocument()
})
it("shows no sign-out notice by default", async () => {
  render(<LoginPage />)
  await screen.findByRole("heading", { name: "Sign in to tldw" })
  expect(screen.queryByText(/signed in on another device/)).toBeNull()
})
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run, one at a time:
- `cd apps/packages/ui && bunx vitest run src/services/tldw/__tests__/session-identity.test.ts src/services/tldw/__tests__/TldwAuth.session-identity.test.ts src/services/__tests__/background-proxy.web-refresh.test.ts --maxWorkers=2`
- `cd apps/tldw-frontend && bunx vitest run __tests__/pages/api/runtime-session.test.ts __tests__/extension/runtime-bootstrap.test.ts __tests__/pages/login-page.test.tsx --maxWorkers=2`

Expected: FAIL. The `session-identity` module is missing, and no header or notice is produced.

- [ ] **Step 3: Implement**

1. `session-identity.ts` as specified in Interfaces, including the try/catch and the `getRandomValues` fallback.
2. `TldwAuth.login` and `verifyMagicLink`: spread `getInstallIdHeaders()` into the request headers. `publishLogin` calls `clearSessionReplacedNotice()`. The 401 branch of `performTokenRefresh` calls `noteSessionReplaced(error)` before it invalidates the session.
3. The 401 branch of `refreshAuthDirect` calls `noteSessionReplaced(error)`.
4. In `runtime-bootstrap.ts`, the mint POST gets `headers: getInstallIdHeaders()`.
5. The Next proxy copies `req.headers['x-tldw-device-id']` to `X-TLDW-Device-Id` only when it is a string that matches `/^[A-Za-z0-9_-]{8,64}$/`.
6. `SelfHostLogin` reads `hasSessionReplacedNotice()` in a `useState` initializer. It must be a read, not a consume, so React StrictMode cannot lose it. When true, it renders `SESSION_REPLACED_MESSAGE` in a `role="status"` paragraph above the form. The notice is cleared by the next successful login.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: the two commands from Step 2, plus the existing suites one at a time:
- `cd apps/packages/ui && bunx vitest run src/services/tldw/__tests__ src/services/__tests__/background-proxy.test.ts src/services/__tests__/tldw-auth.refresh-rotation.test.ts --maxWorkers=2`
- `cd apps/tldw-frontend && bunx vitest run __tests__/extension __tests__/pages __tests__/navigation/login-page.test.tsx --maxWorkers=2`
- `bun run --cwd apps/packages/ui tsc --noEmit`

Expected: PASS, with no type errors.

- [ ] **Step 5: Commit**

```bash
git add apps/packages/ui/src/services apps/tldw-frontend/extension/shims/runtime-bootstrap.ts \
  apps/tldw-frontend/pages/api/_tldw-webui/session.ts apps/tldw-frontend/routes/login.tsx apps/tldw-frontend/__tests__
git commit -m "feat(webui): send a per-install id on sign-in and explain a replaced session (TASK-15628)"
```

---

### Task 6: ADR, backlog, full verification, PR

**Files:**
- Create: `Docs/ADR/098-per-user-session-cap.md`. The number is confirmed in Step 1.
- Modify: `Docs/ADR/README.md`. Append one index row after the last row.
- Modify: `Docs/Published/ADR/*` via the refresh script
- Modify: `backlog/tasks/task-15628 - …md`, through the backlog tool only

**Interfaces:**
- Consumes: everything above. Produces documentation only.

- [ ] **Step 1: Confirm the ADR number**

Run:
```bash
git ls-tree --name-only origin/dev Docs/ADR/ | grep -oE '/[0-9]{3}-' | sort -u | tail -1
gh pr list --state open --limit 300 --json number,files \
  -q '.[] | .number as $n | .files[].path | select(test("^Docs/ADR/[0-9]{3}-")) | "\($n) \(.)"' | sort -k2 | tail -5
```
Expected: the highest claimed number is `097`, held by #3437 and #3071 as of 2026-11-23. Use the next free number; the file names in this task assume 098.

- [ ] **Step 2: Write the ADR** from `Docs/ADR/000-template.md`.
  - Header: `Status: Accepted`, `Backfilled from: not backfilled`, decision owner "repository owner (design approval 2026-11-23)", related tasks TASK-15628 and TASK-15632, related spec/plan paths.
  - **Decision (one sentence):** every AuthNZ session row is created through one per-user-serialized, capped insert that keeps at most `MAX_SESSIONS_PER_USER` (default 4) live sessions. It revokes (does not delete) the oldest live session of the same client install first, then the user's oldest. Sign-in is never refused.
  - **Context:** the 2026-11-23 storm (a new 30-day row about every 1.7 s), no bound anywhere in AuthNZ, and single-user mode where every browser looked identical.
  - **Alternatives table:**
    - refuse sign-in at the cap: rejected because it can lock the owner out;
    - short-lived mint, extended on first use: TASK-15628 option (b), which does not bound live rows from a fast client;
    - group by IP or user agent: spoofable, and merges machines behind one NAT;
    - delete instead of revoke: the evicted client could not be told why;
    - a single-key advisory lock: the key space is shared with other code;
    - per-row `revoke_session` calls: each opens its own transaction, so the cap would not be atomic.
  - **Consequences:**
    - The live predicate `COALESCE(refresh_expires_at, expires_at) > now` is shared by the cap, the listing, `require_online` and cleanup.
    - An idle session now stays refreshable for the full `REFRESH_TOKEN_EXPIRE_DAYS`. Before this change, cleanup deleted the row about 1 day after the access token expired, which ended the session early.
    - Evicted clients get 401 `session_replaced`.
    - Cap rows are purged after `max(1 h, access-token lifetime)`.
    - The install id is a grouping hint only.
    - The MFA row is created at completion.
    - Single-user `/login` creates no row.
    - API keys are not sessions.
    - Test suites that sign one user in from more than four parallel workers must raise `MAX_SESSIONS_PER_USER` in that suite's backend env.
  - Add the README index row `| [ADR-098](098-per-user-session-cap.md) | Accepted | Cap live sessions per user; evict same-install first; never refuse sign-in. |`.
  - The account-lock rule from spec section 3 gets its own ADR with PR 3.

- [ ] **Step 3: Mirror the docs and check them**

Run: `Helper_Scripts/refresh_docs_published.sh && run_py t6-docs tldw_Server_API/tests/Docs`
Expected: PASS, including `test_adr_number_identifies_one_document_in_each_catalog`.

- [ ] **Step 4: Update the backlog task**

Run: `PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd "$PWD" task edit TASK-15628 --status Done --check-ac 1 --check-ac 2 --check-ac 3 --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 5 --check-dod 6 --final-summary "<what shipped: cap 4 via MAX_SESSIONS_PER_USER, same-install eviction, live predicate, session_replaced, MFA row at completion, single-user /login no row; tests SQLite+PG; ADR-098>" --append-notes "Design: Docs/Design/2026-11-23-session-cap-alert-lock-design.md section 1; plan: Docs/superpowers/plans/2026-11-23-session-cap.md"`
Then: `git add -A backlog/tasks`
Expected: the file shows status Done with every AC checked. DoD #4 (Bandit) is checked in Step 6.

- [ ] **Step 5: Commit**

```bash
git add Docs/ADR Docs/Published && git add -A backlog/tasks
git commit -m "docs(adr): ADR-098 per-user session cap; close TASK-15628"
```

- [ ] **Step 6: Run the full verification**

1. Run every test file that touches session creation, listing, cleanup or the single-user cookie, in batches of about 20 files per `run_py` call. Get the list with:
   `git grep -l -e 'auth/login' -e '\.create_session(' -e 'single-user/session' -e 'magic-link/verify' -e 'mfa/login' -e 'auth/refresh' -e 'get_active_sessions' -e 'get_user_sessions' -e 'cleanup_expired_sessions' -e 'single-user-cookie' -- 'tldw_Server_API/tests/**/test_*.py' | grep -v -e /TTS/ -e /Sandbox/ -e /Agent_Client_Protocol/ -e /Persona/`
   Also run `tldw_Server_API/tests/AuthNZ_SQLite tldw_Server_API/tests/AuthNZ_Unit tldw_Server_API/tests/lint tldw_Server_API/tests/Collections/test_reading_digests.py`. The lint directory includes `test_function_size_ratchet.py` and `test_env_vars_docs_ratchet.py`.
   - Expected: PASS.
   - For any failure, re-run the same test id in one reusable trial worktree on `origin/dev` before calling it pre-existing.
   - A test that signs one user in more than four times and then reuses the first token is a real behaviour change: fix the test to use a fresh token, or set `MAX_SESSIONS_PER_USER` in that test.
2. Bandit: `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m bandit -r tldw_Server_API/app/core/AuthNZ/session_manager.py tldw_Server_API/app/core/AuthNZ/repos/sessions_repo.py tldw_Server_API/app/core/AuthNZ/single_user_session.py tldw_Server_API/app/core/AuthNZ/User_DB_Handling.py tldw_Server_API/app/core/AuthNZ/auth_principal_resolver.py tldw_Server_API/app/api/v1/endpoints/auth.py -f json -o "$TMPDIR/bandit_task15628.json"`. Expected: no new findings in the changed lines. Then run `task edit TASK-15628 --check-dod 4` with the backlog tool and commit the backlog change.
3. Lint: `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/ruff check <touched .py files>`. Expected: clean.

- [ ] **Step 7: Decide the E2E session budget before pushing**

Read the two Playwright suites that sign one user in repeatedly. For each, count the most live sessions of one user it needs at once, and check whether it ever reuses a cookie or token after four or more later sign-ins.
- **`apps/tldw-frontend/e2e/single-user-cookie-lifecycle.spec.ts`.** The suite is `test.describe.serial`, with one worker. Its backend env is the spec's own `backendEnvironment` (line 55), not `webServerEnv` in `apps/tldw-frontend/playwright.config.ts`, which configures the Next server.
  - It relaunches persistent profiles, which keep one install id.
  - It also mints directly with the API key and no header, which puts those rows in the bare-tag group.
- **admin-ui real-backend.** `admin-ui/playwright.config.ts` sets no `workers`, so Playwright uses its default of half the cores. The seeded users (`_ADMIN_USERNAME` and the others in `admin_e2e_support_service.py`) get a session from `bootstrap-jwt-session` in global setup and sign in through `/api/auth/login` in each test. The backend env is `buildBackendEnv` in `admin-ui/tests/e2e/real-backend/helpers/backend-lifecycle.ts:55`.

Expected: a written decision for each suite, recorded in the PR body. Either "fits within 4", or the suite's own backend env sets `MAX_SESSIONS_PER_USER` to the counted need, committed in this PR. Never change the default cap.

- [ ] **Step 8: Push and open the PR**

1. `git push -u origin feat/session-cap-15628`, then `gh pr create --base dev`.
   - Title: `feat(authnz): cap live sessions per user, same-install eviction (TASK-15628)`.
   - Body: summary, test evidence (including whether the PostgreSQL tests ran), the E2E decision from Step 7, the "Deviations from spec" list below, and the attribution line from the session reminder.
2. Run the clean-subagent code review.
3. After CI, diff the Frontend E2E journeys and the admin-ui real-backend runs against a recent `dev` run.
4. Merge directly when CLEAN. Never use auto-merge.

---

## Self-review

- **Spec coverage:**
  - limit and setting: Task 2
  - live predicate and its callers: Task 1 (schedulers, admin listing, profile security) and Task 3 (`/auth/sessions` and `/users/sessions` list and revoke)
  - what counts as a session (single-user `/login` and MFA): Task 3
  - per-install id: Tasks 3 and 5
  - eviction at the cap, the eviction rule and atomicity: Task 2
  - `session_replaced` and the WebUI message: Tasks 4 and 5
  - bounded storage: Task 1
  - single-user mode (TASK-15628): Tasks 2 and 3
  - ADR: Task 6
  - lock re-read: not planned. PR 3 adds it inside the cap transaction, right after the lock.
- **Required tests:**
  - 5th sign-in evicts the oldest: `test_fifth_session_evicts_the_oldest` (+`_postgres`, + HTTP)
  - same device replaces only its own row: `test_same_device_id_replaces_only_its_own_row`, `test_same_install_login_replaces_only_its_own_session`, cookie `test_install_that_drops_its_cookie_replaces_only_its_own_session`
  - concurrent sign-ins at the cap: `test_concurrent_creates_at_the_cap_never_exceed_it` (+`_postgres`)
  - expired-refresh rows do not count: `test_expired_refresh_rows_do_not_count_toward_the_cap` (+`_postgres`)
  - evicted session gets 401 `session_replaced`: JWT, refresh and cookie tests in Task 4
  - single-user `/login` inserts no row: `test_single_user_login_inserts_no_session_row`
  - purge after `max(1 h, access lifetime)`: `test_cleanup_purges_cap_evicted_rows_after_retention`
  - a value below 1 is rejected: `test_max_sessions_per_user_below_one_is_rejected`
- **Type consistency:** these names are used identically everywhere: `SESSION_CAP_REVOKE_REASON`, `create_session_record_with_cap`, `request_device_id`, `DEVICE_ID_HEADER`, `is_single_user_cookie_device`, `get_revoke_reason`, `revoked_token_exception`, `session_replaced_http_exception`, `cookie_session_was_replaced`, `getInstallIdHeaders`, `noteSessionReplaced`, `hasSessionReplacedNotice`, `clearSessionReplacedNotice`.

## Decisions the spec left open

1. **Transport:** the header `X-TLDW-Device-Id`, with format `^[A-Za-z0-9_-]{8,64}$`. The colon is excluded so a client cannot claim the single-user tag. A malformed value is treated as absent.
2. **Listing `expires_at`:** `get_active_sessions_for_user` reports `expires_at` as the session end, `COALESCE(refresh_expires_at, expires_at)`. Otherwise live idle sessions would list a past expiry. This matches `SessionResponse.expires_at` ("Session expiration time").
3. **Cleanup grace:** cleanup keeps today's 1-day grace, now measured on the same session-end expression.
4. **Purge retention:** computed as `max(60, ACCESS_TOKEN_EXPIRE_MINUTES)` minutes, from the JWT access lifetime also for cookie sessions. The cookie's 30-day life would defeat the bound.
5. **Eviction details:** `revoked_by` is NULL for cap evictions. Ties break by `id`. The Redis session cache is not cleared on eviction, because validation always re-reads the row.
6. **Uncapped insert:** `create_session_record` stays as the uncapped insert that the repo tests use. App code reaches only `create_session_record_with_cap` through `SessionManager.create_session`.
7. **`mfa_login`:** reuses `_issue_multi_user_tokens` instead of keeping a duplicated token-issuing block. The row takes the IP, user agent and install id of the `/mfa/login` request; the cached payload must not carry client data (an existing test asserts this).
8. **The notice:**
   - stored in localStorage (`tldw:session-replaced`), shown on the sign-in page, cleared by the next successful sign-in
   - recorded from both web refresh paths (`TldwAuth.performTokenRefresh`, `refreshAuthDirect`)
   - not recorded by the extension's background worker refresh, which has no sign-in page
9. **E2E suites:** a suite that signs one seeded user in from more than four parallel workers raises `MAX_SESSIONS_PER_USER` in its own backend env.

## Deviations from spec

1. **TASK-15628 is not on `dev` yet.** It exists only on the open PR #3578 branch, which also rewrites `runtime-bootstrap.ts`. This plan's branch must start after #3578 merges.
2. **ADR number.** The spec says "the next free ADR number". On `origin/dev` that is 097, but open PRs #3437 and #3071 already add `Docs/ADR/097-*.md`, and `test_adr_number_identifies_one_document_in_each_catalog` forbids duplicates. Use 098 (re-checked in Task 6 Step 1).
3. **ADR scope.** The spec envisions one ADR covering both the cap and the locked-account state. `Docs/ADR/README.md` rule 5 says "one decision per ADR", so the PR 1 ADR records the cap only, and the lock gets its own ADR with PR 3.
4. **More listing callers than the spec names.** The spec lists three (`require_online` in the workflows scheduler, the admin listing, the `/users/sessions` aliases). The code also has:
   - `/auth/sessions` list and revoke-one (`auth.py:2175`, `:2216`)
   - `require_online` in `services/reading_digest_scheduler.py:284`
   - `UserProfileService.build_security` (`core/UserProfiles/service.py:822`)

   All of them get tests.
5. **The listing had no expiry predicate.** Today's `get_active_sessions_for_user` filters on `is_active` only. It still lists expired rows until cleanup deletes them, so the new predicate adds an expiry check rather than changing one. One effect: `require_online` now treats any session with a valid refresh token (up to 7 days idle) as online.
6. **Revoked sessions never raise `SessionRevokedException`.** `validate_session` never raises it for a revoked row (`session_manager.py:1245`): `fetch_session_for_validation_*` filters `is_active`, so revoked rows come back as None. `session_replaced` therefore needs an explicit `revoke_reason` lookup on the failure path, for both the JWT and the cookie paths.
7. **`session_replaced` on refresh too.** The spec says "a request". The plan also returns it from `POST /auth/refresh`, because that is the call through which the WebUI learns its session is gone. WebSocket cookie auth (`websocket_session_auth.py`) keeps its generic close.
8. **MFA on SQLite.** The spec's constraint says MFA exists only on PostgreSQL, but `tests/AuthNZ_SQLite/test_mfa_login_sqlite.py` runs MFA on SQLite, so the MFA change is tested there. Removing the placeholder is safe. Only `mfa_login` reads its id (through the cached payload); no failure path revokes it; three unit tests in `test_auth_endpoints_extended.py` assert the old shape and are updated. The WebUI has no MFA UI.
9. **Single-user mint without an install id.** The row is stored and grouped under the bare `single-user-cookie:v1` tag. Header-less mints therefore replace each other before they touch browsers that sent an id, instead of the spec's "no id → user's oldest". This is simpler (no extra parameter) and stricter for TASK-15628. The bundled WebUI always sends the header.
10. **OIDC callback.** It is a browser redirect, so it carries no install id and evicts by the default rule.
11. **Known limit: single-user cookie mode never shows the notice.** The spec says the WebUI shows it on the sign-in page. But in single-user mode, `/login` redirects to settings, and the notice is recorded only from the multi-user refresh paths. A cap-evicted browser re-mints its cookie with the API key on its next bootstrap. The backend still answers that browser's requests with 401 `session_replaced` until then.
