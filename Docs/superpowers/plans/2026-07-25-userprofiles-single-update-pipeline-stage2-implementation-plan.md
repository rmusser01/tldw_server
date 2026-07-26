# UserProfiles Stage 2 Single-Update Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace every UserProfiles single-update caller with one typed, transaction-owning pipeline that preserves caller contracts while making versioning, membership writes, required effects, rollback, and concurrency behavior explicit.

**Architecture:** AuthNZ owns the durable transaction, profile-version anchor, and shared membership-write primitives. UserProfiles builds immutable plans, executes typed mutations on the supplied transaction, fences the required evaluations effect, commits once, and maps a transport-neutral result through caller-specific adapters. Delivery is split into five sequential review packages; each package must have its own Backlog child task, verification record, commit series, and review checkpoint.

**Tech Stack:** Python 3.11+, FastAPI adapters, frozen dataclasses, asyncio, aiosqlite, asyncpg, SQLite/PostgreSQL migrations, pytest/pytest-asyncio, Hypothesis for pure policy, Loguru, Bandit.

**Source design:** `Docs/superpowers/specs/2026-07-20-userprofiles-single-update-pipeline-stage2-design.md`

**Planning baseline:** Reconciled against `origin/dev` at `2e0d3f1a2cfcad9798008f5bd249d91bbac43f07`. Before implementation, start a fresh `codex/` worktree from the then-current `origin/dev` and bring this spec and plan onto it. Re-run the inventory tests before editing because the runtime writer set may have grown.

---

## File Responsibility Map

### Storage and transaction foundations

- Create `tldw_Server_API/app/core/AuthNZ/profile_version.py`: strict UTC timestamp normalization, one-statement candidate reads, final-touch calculation, anchor persistence, and connection-aware versioned user writes.
- Modify `tldw_Server_API/app/core/AuthNZ/database.py`: generic rollback pass-through, typed PostgreSQL concurrency pass-through, bounded transaction acquisition, cancellation propagation, and sanitized translation.
- Modify `tldw_Server_API/app/core/AuthNZ/exceptions.py`: `RollbackSignal` and `DatabaseConcurrencyConflict` without backend text.
- Modify `tldw_Server_API/app/core/AuthNZ/settings.py`: resolve `AUTHNZ_DB_POOL_ACQUIRE_TIMEOUT_SECONDS` with a five-second default.
- Create `tldw_Server_API/app/core/AuthNZ/transaction_policy.py`: one sanitized policy object for SQLite entry retry/backoff, busy retry metadata, and PostgreSQL acquisition timeout; UserProfiles, membership wrappers, and FastAPI dependencies all consume it.
- Modify `tldw_Server_API/Databases/SQLite/Schema/sqlite_users.sql`, `tldw_Server_API/Databases/Postgres/Schema/postgresql_users.sql`, `tldw_Server_API/app/core/AuthNZ/migrations.py`, and `tldw_Server_API/app/core/AuthNZ/pg_migrations_extra.py`: add, backfill, and verify `users.profile_version`.
- Create `tldw_Server_API/app/core/UserProfiles/version_gateway.py`: pool-owning stale reads and supplied-connection transaction reads over the AuthNZ profile-version store.
- Create `tldw_Server_API/app/core/UserProfiles/transaction_gateway.py`: preserve SQLite entry retry/backoff and map transaction failures to stable domain failures.
- Modify `tldw_Server_API/app/core/UserProfiles/overrides_repo.py`: connection-required version candidate queries that never reacquire or hide failures.

### Membership writer protocol

- Create `tldw_Server_API/app/core/AuthNZ/membership_writer.py`: closed actor/trusted-system context, complete lock-set derivation, total-order locking, volatile rechecks, request-order mutation, and explicit anchor ownership.
- Modify `tldw_Server_API/app/core/AuthNZ/repos/orgs_teams_repo.py`: make every membership DML path delegate to the shared writer.
- Modify `tldw_Server_API/app/core/AuthNZ/orgs_teams.py`: preserve public return/exception contracts while passing actor context and bounded transaction settings.
- Modify `tldw_Server_API/app/services/registration_service.py` and current invite/provisioning entry points found by the inventory test: use the audited `registration` trusted reason.

### Typed pipeline and effects

- Rewrite `tldw_Server_API/app/core/UserProfiles/contracts.py`: discriminated immutable commands, plans, mutations, effects, rejections, and command results.
- Create `tldw_Server_API/app/core/UserProfiles/update_policy.py`: pure catalog, role, scalar, email, and membership-payload policy.
- Rewrite `tldw_Server_API/app/core/UserProfiles/planner.py`: read-only planning with injected operation time, lockout configuration, and membership gateway.
- Create `tldw_Server_API/app/core/UserProfiles/executor.py`: validated typed mutation registry and storage-bound executors.
- Rewrite `tldw_Server_API/app/core/UserProfiles/effects.py`: validated typed effect registry, required pre-commit dispatch, and best-effort post-commit dispatch.
- Rewrite `tldw_Server_API/app/core/UserProfiles/command_service.py`: transaction ownership, two version checks, one final touch, rollback signaling, commit confirmation, and post-commit effects.
- Create `tldw_Server_API/app/core/UserProfiles/composition.py`: the only production construction root and startup readiness hook.
- Create `tldw_Server_API/app/core/Evaluations/rate_limit_config_repo.py`: generation-aware configuration repository used by every configuration writer and cache reader.
- Create `tldw_Server_API/app/core/Evaluations/rate_limit_config_models.py`: cycle-free `UserTier`, `RateLimitConfig`, ordered limit changes, reservation, and applied-write contracts.
- Create `tldw_Server_API/app/core/Evaluations/rate_limit_config_process.py`: killable subprocess adapter with bounded terminate, kill, and reap phases.
- Modify `tldw_Server_API/app/core/Evaluations/user_rate_limiter.py`: delegate default creation, expiry reset, direct tier update, and cache validation to the repository.

### Adapters and removal

- Rewrite `tldw_Server_API/app/core/UserProfiles/response_mappers.py`: pure v1, v2, admin, Chatbooks, and deprecated-email mappings.
- Modify `tldw_Server_API/app/core/UserProfiles/error_mapping.py`: domain-only taxonomy with no FastAPI import.
- Modify `tldw_Server_API/app/api/v1/endpoints/users.py`, `tldw_Server_API/app/api/v2/endpoints/user_profiles.py`, `tldw_Server_API/app/services/admin_profiles_service.py`, and `tldw_Server_API/app/core/Chatbooks/chatbook_service.py`: call only `apply(command)`.
- Modify `tldw_Server_API/app/core/UserProfiles/bulk_command_service.py` and the admin bulk adapter: share pure policy and perform one caller-owned anchor touch without adopting Stage 2 all-or-nothing semantics.
- Remove `tldw_Server_API/app/core/UserProfiles/update_service.py` only after structural tests prove no production or test import remains.
- Update `tldw_Server_API/app/core/UserProfiles/README.md`: architecture, cross-system limitation, writer boundaries, and operational failure semantics.

## Delivery Rules

1. Create one child of `TASK-13001` for each work package below before package edits begin. Use the package title verbatim and attach this plan and the design spec.
2. Keep at most one package in progress. Complete its focused tests, PostgreSQL gate where required, Bandit, review, and Backlog finalization before starting the next.
3. Infrastructure may be unused until its adapter package, but no runtime flag or second production single-update path may be introduced.
4. Do not retry a transaction after any required external effect could have completed.
5. Use barriers/events for concurrency tests. Do not use sleeps as correctness synchronization.
6. Treat a skipped local PostgreSQL fixture as incomplete verification; PostgreSQL CI is a merge gate.

## Work Package 1: Storage and Transaction Foundations

### Task 1: Freeze caller contracts and writer inventories

**Files:**
- Create: `tldw_Server_API/tests/UserProfile/test_stage2_caller_characterization.py`
- Create: `tldw_Server_API/tests/UserProfile/test_profile_write_boundaries.py`
- Modify: `tldw_Server_API/tests/UserProfile/test_user_profile_legacy_contract_characterization.py`
- Modify: `tldw_Server_API/tests/UserProfile/test_user_profile_v2_contract.py`
- Modify: `tldw_Server_API/tests/UserProfile/test_admin_profiles_service_update.py`
- Modify: `tldw_Server_API/tests/Chatbooks/test_chatbooks_full_account_import_restore.py`
- Modify: `tldw_Server_API/tests/UserProfile/test_user_profile_legacy_me_update.py`

- [ ] **Step 1: Add parameterized caller characterization before changing internals**

Create a table whose rows contain `caller`, `dry_run`, `updates`, expected status, exact detail shape, applied-key order, headers, and audit count. Include duplicate keys, mixed accepted/rejected input, stale version, runtime rollback, empty updates, and the complete deprecated-email matrix.

```python
@pytest.mark.parametrize("case", CALLER_CASES, ids=lambda case: case.name)
async def test_stage2_caller_contract_is_characterized(case, caller_harness):
    observed = await caller_harness.invoke(case)
    assert observed.status_code == case.status_code
    assert observed.body == case.body
    assert observed.headers == case.headers
    assert observed.audit_count == case.audit_count
```

- [ ] **Step 2: Add AST inventories for profile-visible users writes and membership DML**

Parse Python string constants and call sites rather than grepping source text. Classify AuthNZ `users` INSERT/UPDATE statements separately from content-database tables. Make the expected file/operation inventory explicit so a new writer fails the test with its path and line.

```python
PROFILE_VISIBLE_COLUMNS = frozenset({
    "uuid", "username", "email", "role", "is_superuser", "is_active",
    "is_verified", "two_factor_enabled", "last_login",
    "storage_quota_mb", "storage_used_mb",
})
MEMBERSHIP_TABLES = frozenset({"org_members", "team_members"})
```

- [ ] **Step 3: Run the characterization and inventory tests**

Run:
```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/UserProfile/test_stage2_caller_characterization.py tldw_Server_API/tests/UserProfile/test_profile_write_boundaries.py -q
```
Expected: characterization passes against current behavior; boundary tests fail with the complete current direct-writer inventory printed for migration.

- [ ] **Step 4: Record the inventory in the Work Package 1 Backlog task**

Record exact production files, writer functions, and parent-delete paths. Do not add exemptions for runtime writers; only offline migrations and unrelated content databases may be classified as exclusions.

- [ ] **Step 5: Commit the characterization checkpoint**

```bash
git add tldw_Server_API/tests/UserProfile tldw_Server_API/tests/Chatbooks backlog/tasks
git commit -m "test: characterize UserProfiles stage 2 boundaries"
```

### Task 2: Add rollback, conflict, and bounded transaction primitives

**Files:**
- Modify: `tldw_Server_API/app/core/AuthNZ/exceptions.py`
- Modify: `tldw_Server_API/app/core/AuthNZ/database.py:605`
- Modify: `tldw_Server_API/app/core/AuthNZ/settings.py`
- Create: `tldw_Server_API/app/core/AuthNZ/transaction_policy.py`
- Modify: `tldw_Server_API/app/api/v1/API_Deps/auth_deps.py:147`
- Create: `tldw_Server_API/tests/AuthNZ/unit/test_database_transaction_signals.py`
- Create: `tldw_Server_API/tests/AuthNZ/unit/test_database_transaction_acquire_timeout.py`
- Modify: `tldw_Server_API/tests/AuthNZ/unit/test_sqlite_transaction_modes.py`

- [ ] **Step 1: Write failing transaction-boundary tests**

Cover unchanged `RollbackSignal` identity, no log emission for that signal, SQLSTATE `40P01` and `40001` from statements and commit, pool acquisition timeout, explicit exhaustion, cancellation, SQLite rollback, and absence of raw exception text/path in logs and translated exceptions.

```python
class TestRollback(RollbackSignal):
    pass

async def test_transaction_rethrows_rollback_signal_unchanged(pool):
    signal = TestRollback()
    with pytest.raises(TestRollback) as raised:
        async with pool.transaction(acquire_timeout_seconds=5.0):
            raise signal
    assert raised.value is signal
```

- [ ] **Step 2: Run tests and confirm the missing contracts fail**

Run:
```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/AuthNZ/unit/test_database_transaction_signals.py tldw_Server_API/tests/AuthNZ/unit/test_database_transaction_acquire_timeout.py tldw_Server_API/tests/AuthNZ/unit/test_sqlite_transaction_modes.py -q
```
Expected: FAIL because the signal types and bounded transaction argument do not exist.

- [ ] **Step 3: Implement the generic exception and transaction contracts**

Use this public shape; neither exception stores raw backend text. The policy parser is the only place that reads SQLite retry/backoff, busy retry metadata, and PostgreSQL acquisition timeout settings.

```python
class RollbackSignal(Exception):
    """Trusted control-flow signal that requires transaction rollback."""


class DatabaseConcurrencyConflict(DatabaseError):
    """Sanitized deadlock or serialization conflict."""


@asynccontextmanager
async def transaction(self, *, acquire_timeout_seconds: float | None = None):
    if not self._initialized:
        await self.initialize()
    async with self._transaction_context(acquire_timeout_seconds) as conn:
        yield conn
```

Implement `_transaction_context()` as the existing backend-specific body, not as a second public transaction API. Pass the timeout to asyncpg acquisition, catch `RollbackSignal` and `DatabaseConcurrencyConflict` before broad translation, classify SQLSTATE from statement and commit failures, re-raise cancellation, sanitize ordinary translation with `from None`, and preserve SQLite `BEGIN IMMEDIATE` plus rollback. Resolve `AUTHNZ_DB_POOL_ACQUIRE_TIMEOUT_SECONDS=5` in `AuthnzTransactionPolicy`; do not import FastAPI. Make `get_db_transaction()` consume the same policy while preserving its exact HTTP mapping at the dependency boundary.

- [ ] **Step 4: Run the transaction tests**

Run the command from Step 2.
Expected: PASS; captured logs contain exception class and stable backend code only.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/AuthNZ/exceptions.py tldw_Server_API/app/core/AuthNZ/database.py tldw_Server_API/app/core/AuthNZ/settings.py tldw_Server_API/app/core/AuthNZ/transaction_policy.py tldw_Server_API/app/api/v1/API_Deps/auth_deps.py tldw_Server_API/tests/AuthNZ/unit
git commit -m "refactor(authnz): add safe transaction rollback contracts"
```

### Task 3: Add and verify the durable profile-version anchor

**Files:**
- Modify: `tldw_Server_API/Databases/SQLite/Schema/sqlite_users.sql:1`
- Modify: `tldw_Server_API/Databases/Postgres/Schema/postgresql_users.sql:1`
- Modify: `tldw_Server_API/app/core/AuthNZ/migrations.py`
- Modify: `tldw_Server_API/app/core/AuthNZ/pg_migrations_extra.py`
- Modify: `tldw_Server_API/tests/AuthNZ/conftest.py`
- Create: `tldw_Server_API/tests/AuthNZ/unit/test_profile_version_migration.py`
- Create: `tldw_Server_API/tests/AuthNZ_Postgres/test_profile_version_migration_pg.py`

- [ ] **Step 1: Write migration tests first**

Test fresh schema type/default, upgrade backfill from naive/aware `updated_at`, canonical SQLite `YYYY-MM-DDTHH:MM:SS.ffffffZ`, PostgreSQL `TIMESTAMPTZ`, idempotence, null/unparsable rejection, startup failure, and unchanged external version at the migration boundary.

```python
def test_sqlite_profile_version_backfill_is_canonical(tmp_path):
    db_path = create_legacy_users_db(tmp_path, updated_at="2026-01-02 03:04:05.123456")
    ensure_authnz_tables(db_path)
    value = fetch_scalar(db_path, "SELECT profile_version FROM users")
    assert value == "2026-01-02T03:04:05.123456Z"
```

- [ ] **Step 2: Run migration tests and verify red**

Run:
```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/AuthNZ/unit/test_profile_version_migration.py -q
python -m pytest tldw_Server_API/tests/AuthNZ_Postgres/test_profile_version_migration_pg.py -q
```
Expected: SQLite FAIL because the column is absent; PostgreSQL FAIL or explicit local skip, with CI required to pass.

- [ ] **Step 3: Implement schema and upgrade migration**

Fresh SQLite uses `profile_version TEXT NOT NULL DEFAULT (STRFTIME('%Y-%m-%dT%H:%M:%f000Z', 'now'))` so the staged migration remains deployable before every creator is converted; converted creators still supply an explicit canonical value. PostgreSQL uses `profile_version TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP`. The upgrade adds nullable, backfills after strict normalization, verifies every row, then enforces readiness. Keep `update_users_timestamp` limited to `updated_at`; it must not write `profile_version`.

- [ ] **Step 4: Run both migration suites and existing schema fail-fast tests**

Run:
```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/AuthNZ/unit/test_profile_version_migration.py tldw_Server_API/tests/AuthNZ/unit/test_database_sqlite_schema_fail_fast.py tldw_Server_API/tests/AuthNZ/unit/test_pg_migrations_user_timestamps.py -q
python -m pytest tldw_Server_API/tests/AuthNZ_Postgres/test_profile_version_migration_pg.py -q
```
Expected: all available tests PASS; PostgreSQL CI remains a required check when local Docker is unavailable.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/Databases tldw_Server_API/app/core/AuthNZ/migrations.py tldw_Server_API/app/core/AuthNZ/pg_migrations_extra.py tldw_Server_API/tests/AuthNZ tldw_Server_API/tests/AuthNZ_Postgres
git commit -m "feat(authnz): add durable user profile version anchor"
```

### Task 4: Build the fail-closed profile-version and transaction gateways

**Files:**
- Create: `tldw_Server_API/app/core/AuthNZ/profile_version.py`
- Create: `tldw_Server_API/app/core/UserProfiles/version_gateway.py`
- Create: `tldw_Server_API/app/core/UserProfiles/transaction_gateway.py`
- Modify: `tldw_Server_API/app/core/UserProfiles/overrides_repo.py`
- Modify: `tldw_Server_API/app/core/UserProfiles/service.py:280`
- Create: `tldw_Server_API/tests/UserProfile/test_profile_version_gateway.py`
- Create: `tldw_Server_API/tests/UserProfile/test_profile_transaction_gateway.py`
- Modify: `tldw_Server_API/tests/AuthNZ_Postgres/test_user_profile_version_locking_pg.py`

- [ ] **Step 1: Write failing version and transaction gateway tests**

Assert one backend-specific statement returns the complete candidate set, all transaction reads use the supplied connection, missing users and component failures fail closed, normalization rejects invalid timestamps, old/new snapshots never hybridize, lock reads include the user row, and `touch_value` is exactly `max(clock_now_utc, version_floor + timedelta(microseconds=1))`.

```python
def test_compute_touch_value_exceeds_future_floor():
    now = utc("2026-01-01T00:00:00.000000Z")
    floor = utc("2026-01-02T00:00:00.999999Z")
    assert compute_touch_value(now, floor) == utc("2026-01-02T00:00:01.000000Z")
```

Also assert SQLite retry count/backoff, exact `database_busy` retry metadata, PostgreSQL deadline, conflict mapping, cancellation, and no automatic conflict retry.

- [ ] **Step 2: Run tests and verify red**

Run:
```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/UserProfile/test_profile_version_gateway.py tldw_Server_API/tests/UserProfile/test_profile_transaction_gateway.py tldw_Server_API/tests/AuthNZ_Postgres/test_user_profile_version_locking_pg.py -q
```
Expected: FAIL because the gateway modules do not exist.

- [ ] **Step 3: Implement the narrow contracts**

Use these interfaces and return immutable candidates rather than a permissive timestamp fallback.

```python
@dataclass(frozen=True)
class ProfileVersionCandidates:
    user_exists: bool
    values: tuple[datetime, ...]

    @property
    def maximum(self) -> datetime:
        if not self.user_exists or not self.values:
            raise ProfileVersionNotFound()
        return max(self.values)


class ProfileVersionGatewayProtocol(Protocol):
    async def read(self, user_id: int) -> datetime:
        raise NotImplementedError

    async def read_in_transaction(
        self, conn: Any, user_id: int, *, lock_user: bool
    ) -> datetime:
        raise NotImplementedError

    async def touch(self, conn: Any, user_id: int, value: datetime) -> None:
        raise NotImplementedError
```

The SQL statement must include `users.profile_version`, personal override timestamps, current membership IDs, and inherited org/team override timestamps. The PostgreSQL locked-user CTE performs `SELECT ... FOR UPDATE` before candidate aggregation. Define sanitized `ProfileVersionNotFound`, `ProfileVersionInvalid`, and `ProfileVersionReadFailed` types in `AuthNZ/profile_version.py`. `UserProfileService.get_profile_version()` delegates to the gateway; `build_profile()` no longer catches component failure or returns `datetime.now()` for the version.

- [ ] **Step 4: Run focused and existing profile read/version tests**

Run:
```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/UserProfile/test_profile_version_gateway.py tldw_Server_API/tests/UserProfile/test_profile_transaction_gateway.py tldw_Server_API/tests/UserProfile/test_user_profile_effective_layers.py tldw_Server_API/tests/UserProfile/test_profile_query_service.py -q
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/AuthNZ/profile_version.py tldw_Server_API/app/core/UserProfiles tldw_Server_API/tests/UserProfile tldw_Server_API/tests/AuthNZ_Postgres/test_user_profile_version_locking_pg.py
git commit -m "feat(userprofiles): add fail-closed version transaction gateways"
```

### Task 5: Route profile-visible AuthNZ users writes through one gateway

**Files:**
- Modify: `tldw_Server_API/app/core/AuthNZ/profile_version.py`
- Modify: `tldw_Server_API/app/core/AuthNZ/repos/users_repo.py`
- Modify: `tldw_Server_API/app/core/AuthNZ/repos/mfa_repo.py`
- Modify: `tldw_Server_API/app/services/auth_service.py`
- Modify: `tldw_Server_API/app/services/registration_service.py`
- Modify: `tldw_Server_API/app/services/admin_users_service.py`
- Modify: `tldw_Server_API/app/services/storage_quota_service.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/admin/admin_tenant_provisioning.py`
- Modify: `tldw_Server_API/app/core/UserProfiles/update_service.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/users.py`
- Modify: `tldw_Server_API/app/services/admin_profiles_service.py`
- Modify: every additional inventoried production writer from Task 1
- Create: `tldw_Server_API/tests/AuthNZ/unit/test_versioned_user_write_gateway.py`
- Modify: `tldw_Server_API/tests/UserProfile/test_profile_write_boundaries.py`

- [ ] **Step 1: Add gateway behavior tests**

Test user-row lock, pre/post candidate capture, explicit one-touch write, insert initialization, caller-owned mode, strict advance under a same-clock update, future inherited override floor, secret-only exclusion, and rollback on failure.

```python
class UserVersionOwnership(str, Enum):
    GATEWAY_OWNS_ANCHOR = "gateway_owns_anchor"
    CALLER_OWNS_ANCHOR = "caller_owns_anchor"


@dataclass(frozen=True)
class UserWriteResult:
    affected_user_ids: tuple[int, ...]
    version_floor: datetime
```

- [ ] **Step 2: Run behavior and boundary tests to identify remaining direct writes**

Run:
```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/AuthNZ/unit/test_versioned_user_write_gateway.py tldw_Server_API/tests/UserProfile/test_profile_write_boundaries.py -q
```
Expected: gateway tests FAIL initially; the boundary test reports every unmigrated creator/writer.

- [ ] **Step 3: Implement and migrate the complete inventory**

`VersionedUserWriteGateway` accepts a supplied connection, a closed whitelist of columns, operation time, and ownership mode. It initializes `profile_version` on every user insert. For each inventoried update it locks, captures pre/post candidates, performs the write, and touches once unless the caller owns the anchor. Secret-only password/TOTP/backup-code changes do not touch unless the same statement changes an inventoried field. During this foundation package, route the still-transitional `UserProfileUpdateService`, deprecated email endpoint, and admin bulk path through the gateway without changing their public contracts; Work Package 4 later replaces those adapters with the command service. The boundary test permits no direct SQL exception for them.

- [ ] **Step 4: Run affected AuthNZ and structural suites**

Run:
```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/AuthNZ/unit/test_versioned_user_write_gateway.py tldw_Server_API/tests/AuthNZ/unit/test_authnz_mfa_repo_backend_selection.py tldw_Server_API/tests/AuthNZ/unit/test_registration_service_backend_selection.py tldw_Server_API/tests/AuthNZ/unit/test_storage_quota_service_backend_selection.py tldw_Server_API/tests/Admin/test_admin_users_role_consistency.py tldw_Server_API/tests/UserProfile/test_profile_write_boundaries.py -q
```
Expected: PASS and no runtime profile-visible users writer outside approved gateway/caller-owned paths.

- [ ] **Step 5: Run Work Package 1 security and review gates**

Run:
```bash
source .venv/bin/activate
python -m compileall -q tldw_Server_API/app/core/AuthNZ tldw_Server_API/app/core/UserProfiles
python -m bandit -r tldw_Server_API/app/core/AuthNZ tldw_Server_API/app/core/UserProfiles -f json -o /tmp/bandit_userprofiles_stage2_wp1.json
git diff --check
```
Expected: zero compile failures, no new Bandit findings, clean diff check. Request code review, resolve valid findings, complete the Work Package 1 Backlog child, then commit any review fixes.

## Work Package 2: Membership Writer Protocol

### Task 6: Implement deterministic membership write contracts and lock planning

**Files:**
- Create: `tldw_Server_API/app/core/AuthNZ/membership_writer.py`
- Create: `tldw_Server_API/tests/AuthNZ/unit/test_membership_writer_lock_plan.py`
- Create: `tldw_Server_API/tests/AuthNZ/unit/test_membership_writer_context.py`

- [ ] **Step 1: Write pure lock-plan and context tests**

Cover actor versus closed trusted reasons, rejection of missing context, complete unique sets, ascending user/org/team order, `(scope_type, scope_id, user_id)` membership order, owner rows last, team parent-org inclusion, opposite request orders producing identical locks, and original mutation-order preservation.

```python
class AnchorOwnership(str, Enum):
    CALLER_OWNS_ANCHOR = "caller_owns_anchor"
    WRITER_OWNS_ANCHOR = "writer_owns_anchor"


@dataclass(frozen=True)
class MembershipWriteContext:
    actor_user_id: int | None = None
    trusted_reason: TrustedMembershipReason | None = None


@dataclass(frozen=True)
class MembershipLockSet:
    user_ids: tuple[int, ...]
    org_ids: tuple[int, ...]
    team_ids: tuple[int, ...]
    membership_rows: tuple[MembershipRowLock, ...]
    owner_user_ids: tuple[int, ...]
```

- [ ] **Step 2: Run tests and verify red**

Run:
```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/AuthNZ/unit/test_membership_writer_lock_plan.py tldw_Server_API/tests/AuthNZ/unit/test_membership_writer_context.py -q
```
Expected: FAIL because the writer module does not exist.

- [ ] **Step 3: Implement immutable planning and validation**

Trusted reasons are a closed enum containing `registration`, `bootstrap`, and `offline_migration`; runtime wrappers reject `offline_migration` while serving. Lock planning performs no writes. PostgreSQL lock SQL is generated in the documented total order; SQLite relies on the caller's `BEGIN IMMEDIATE` and still computes the same set for rechecks and tests.

- [ ] **Step 4: Run tests and commit**

Run the command from Step 2; expected PASS.

```bash
git add tldw_Server_API/app/core/AuthNZ/membership_writer.py tldw_Server_API/tests/AuthNZ/unit/test_membership_writer_lock_plan.py tldw_Server_API/tests/AuthNZ/unit/test_membership_writer_context.py
git commit -m "feat(authnz): define ordered membership writer protocol"
```

### Task 7: Migrate direct membership APIs to the shared writer

**Files:**
- Modify: `tldw_Server_API/app/core/AuthNZ/membership_writer.py`
- Modify: `tldw_Server_API/app/core/AuthNZ/repos/orgs_teams_repo.py:846`
- Modify: `tldw_Server_API/app/core/AuthNZ/orgs_teams.py:127`
- Modify: `tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_orgs_teams_repo_sqlite.py`
- Modify: `tldw_Server_API/tests/AuthNZ/integration/test_authnz_orgs_teams_repo_postgres.py`
- Create: `tldw_Server_API/tests/AuthNZ_Postgres/test_membership_writer_concurrency_pg.py`

- [ ] **Step 1: Write connection, authorization, and concurrency tests**

For add/remove/role changes on orgs and teams, assert one supplied connection, parent locks before membership locks, actor authorization re-read after locking, membership existence and last-owner recheck, request-order mutation, exact legacy return shape, and one strict affected-user anchor advance in writer-owned mode.

- [ ] **Step 2: Run focused suites and verify current bypasses fail**

Run:
```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_orgs_teams_repo_sqlite.py tldw_Server_API/tests/AuthNZ/integration/test_authnz_orgs_teams_repo_postgres.py tldw_Server_API/tests/AuthNZ_Postgres/test_membership_writer_concurrency_pg.py -q
```
Expected: new delegation/locking assertions FAIL against direct repository DML.

- [ ] **Step 3: Implement the transaction-aware core and wrappers**

Expose a single write entry point:

```python
async def apply_membership_mutations(
    self,
    *,
    conn: Any,
    context: MembershipWriteContext,
    mutations: tuple[MembershipMutation, ...],
    anchor_ownership: AnchorOwnership,
    operation_time: datetime,
) -> MembershipWriteResult:
    raise NotImplementedError
```

Public direct functions open the bounded AuthNZ transaction and select `WRITER_OWNS_ANCHOR`. Stage 2 and bulk will supply their transaction and select `CALLER_OWNS_ANCHOR`. Preserve existing public exceptions and result dictionaries at wrapper boundaries.

- [ ] **Step 4: Run SQLite/PostgreSQL suites and commit**

Run the command from Step 2; expected all available tests PASS and PostgreSQL CI required.

```bash
git add tldw_Server_API/app/core/AuthNZ/membership_writer.py tldw_Server_API/app/core/AuthNZ/repos/orgs_teams_repo.py tldw_Server_API/app/core/AuthNZ/orgs_teams.py tldw_Server_API/tests/AuthNZ_SQLite tldw_Server_API/tests/AuthNZ/integration tldw_Server_API/tests/AuthNZ_Postgres/test_membership_writer_concurrency_pg.py
git commit -m "refactor(authnz): route membership APIs through ordered writer"
```

### Task 8: Migrate ownership, provisioning, default-team, and scope deletion paths

**Files:**
- Modify: `tldw_Server_API/app/core/AuthNZ/repos/orgs_teams_repo.py:469`
- Modify: `tldw_Server_API/app/services/registration_service.py`
- Modify: current invite/bootstrap provisioning files reported by Task 1
- Modify: `tldw_Server_API/tests/AuthNZ/unit/test_registration_default_role_membership.py`
- Modify: `tldw_Server_API/tests/AuthNZ/integration/test_registration_role_membership_postgres.py`
- Create: `tldw_Server_API/tests/AuthNZ_Postgres/test_membership_scope_delete_concurrency_pg.py`
- Modify: `tldw_Server_API/tests/UserProfile/test_profile_write_boundaries.py`

- [ ] **Step 1: Write failing all-writer and scope-delete race tests**

Cover ownership transfer, registration/invite, default-team helpers, org/team deletion, affected-user one-touch behavior, removal of future inherited overrides, and complete-set changes after preflight including a new membership and an empty child team. Assert changed discovery aborts and restarts from a fresh transaction within a bounded retry count.

- [ ] **Step 2: Run tests and verify boundary failures**

Run:
```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/AuthNZ/unit/test_registration_default_role_membership.py tldw_Server_API/tests/AuthNZ/integration/test_registration_role_membership_postgres.py tldw_Server_API/tests/AuthNZ_Postgres/test_membership_scope_delete_concurrency_pg.py tldw_Server_API/tests/UserProfile/test_profile_write_boundaries.py -q
```
Expected: new concurrency and boundary assertions FAIL until every writer delegates.

- [ ] **Step 3: Migrate all paths without exemptions**

Registration/invite use trusted reason `registration`; bootstrap uses `bootstrap`. Parent deletion locks the preflight set, locks parent scopes in total order, recomputes the complete set, and raises a private retry signal on any difference. The outer wrapper starts a fresh bounded transaction; it never acquires a newly discovered lower-order lock in place.

- [ ] **Step 4: Run package verification**

Run:
```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/AuthNZ_SQLite/test_orgs_teams_sqlite.py tldw_Server_API/tests/AuthNZ_Postgres/test_orgs_teams_pg.py tldw_Server_API/tests/AuthNZ tldw_Server_API/tests/UserProfile/test_profile_write_boundaries.py -q
python -m bandit -r tldw_Server_API/app/core/AuthNZ/membership_writer.py tldw_Server_API/app/core/AuthNZ/repos/orgs_teams_repo.py tldw_Server_API/app/core/AuthNZ/orgs_teams.py tldw_Server_API/app/services/registration_service.py -f json -o /tmp/bandit_userprofiles_stage2_wp2.json
git diff --check
```
Expected: focused suites PASS, no new Bandit findings, and structural inventory contains no runtime membership DML bypass.

- [ ] **Step 5: Review and commit**

Request code review with special attention to lock order and owner invariants. Resolve valid findings, complete the Work Package 2 Backlog child, then commit:

```bash
git add tldw_Server_API/app/core/AuthNZ tldw_Server_API/app/services tldw_Server_API/tests/AuthNZ tldw_Server_API/tests/AuthNZ_Postgres tldw_Server_API/tests/UserProfile/test_profile_write_boundaries.py backlog/tasks
git commit -m "refactor(authnz): complete membership writer migration"
```

## Work Package 3: Typed Pipeline and Effects

### Task 9: Replace open payload contracts with frozen discriminated types

**Files:**
- Rewrite: `tldw_Server_API/app/core/UserProfiles/contracts.py`
- Modify: `tldw_Server_API/app/core/UserProfiles/error_mapping.py`
- Modify: `tldw_Server_API/tests/UserProfile/test_profile_contracts.py`

- [ ] **Step 1: Write exhaustive immutability and discrimination tests**

Assert recursive immutability, submitted values hidden from repr, ordered duplicates retained, no command contract mode/connection, distinct accepted/applied keys, exactly one touch marker in non-empty apply plans, and stable domain outcomes. Define and test `LockoutConfiguration`, the minimum-data `MembershipSnapshot`, immutable execution identifiers, `ExecutionState`, `ProfileOutcome`, and `ProfileFailureCode` here so later tasks do not invent incompatible shapes.

```python
Mutation = (
    UserFieldMutation | TouchUserVersionMutation | AccountLockStateMutation |
    OverrideUpsertMutation | OverrideDeleteMutation | OrgRoleMutation |
    TeamRoleMutation | TeamMembershipMutation
)
Effect = SetEvaluationLimits | InvalidateStorageQuotaCache
PlanDecision = PlanAccepted | PlanRejected
```

- [ ] **Step 2: Run tests and verify red**

Run:
```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/UserProfile/test_profile_contracts.py -q
```
Expected: FAIL because existing contracts use open mappings, string operations, and `ProfileContractMode`.

- [ ] **Step 3: Implement complete frozen contracts**

Define explicit fields for every variant. Use `repr=False` for normalized email and other submitted values. `ProfileCommandResult` contains outcome, profile version, accepted keys, applied keys, ordered rejections, stable error code, and optional bounded retry seconds; it contains no HTTP status or caller text. Move taxonomy precedence and stable domain codes into `error_mapping.py` without importing FastAPI; numeric status and caller detail text remain adapter concerns.

- [ ] **Step 4: Run and commit**

Run the command from Step 2; expected PASS.

```bash
git add tldw_Server_API/app/core/UserProfiles/contracts.py tldw_Server_API/app/core/UserProfiles/error_mapping.py tldw_Server_API/tests/UserProfile/test_profile_contracts.py
git commit -m "refactor(userprofiles): define typed stage 2 contracts"
```

### Task 10: Extract pure policy and preserve bulk parity

**Files:**
- Create: `tldw_Server_API/app/core/UserProfiles/update_policy.py`
- Modify: `tldw_Server_API/app/core/UserProfiles/bulk_command_service.py`
- Create: `tldw_Server_API/tests/UserProfile/test_profile_update_policy.py`
- Modify: `tldw_Server_API/tests/UserProfile/test_profile_bulk_command_service.py`

- [ ] **Step 1: Write table and property tests**

Cover every catalog type/boundary, admin implied roles, editable-by rules, email normalization, null preference delete, membership payload parsing, duplicate/order preservation, deterministic rejection precedence, numeric limits, and equality with characterized bulk policy results. Keep Hypothesis tests database-free.

```python
@given(st.lists(profile_update_pairs(), max_size=25))
def test_policy_preserves_input_order_and_duplicates(updates):
    decision = policy.evaluate(updates, roles=frozenset({"user"}))
    assert [item.request_index for item in decision.items] == list(range(len(updates)))
```

- [ ] **Step 2: Run tests and verify red**

Run:
```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/UserProfile/test_profile_update_policy.py tldw_Server_API/tests/UserProfile/test_profile_bulk_command_service.py -q
```
Expected: FAIL because pure policy does not exist.

- [ ] **Step 3: Implement pure functions only**

No database, cache, limiter, logger, clock, FastAPI, or service imports are allowed in `update_policy.py`. The bulk facade consumes the policy decisions but preserves current target ordering, partial success, result shape, dry-run behavior, and transaction ownership.

- [ ] **Step 4: Run and commit**

Run the command from Step 2; expected PASS.

```bash
git add tldw_Server_API/app/core/UserProfiles/update_policy.py tldw_Server_API/app/core/UserProfiles/bulk_command_service.py tldw_Server_API/tests/UserProfile/test_profile_update_policy.py tldw_Server_API/tests/UserProfile/test_profile_bulk_command_service.py
git commit -m "refactor(userprofiles): extract shared pure update policy"
```

### Task 11: Build the independent planner and typed executor registry

**Files:**
- Rewrite: `tldw_Server_API/app/core/UserProfiles/planner.py`
- Create: `tldw_Server_API/app/core/UserProfiles/executor.py`
- Modify: `tldw_Server_API/app/core/UserProfiles/overrides_repo.py`
- Modify: `tldw_Server_API/tests/UserProfile/test_profile_update_planner.py`
- Create: `tldw_Server_API/tests/UserProfile/test_profile_mutation_executors.py`
- Create: `tldw_Server_API/tests/UserProfile/test_profile_account_lock_executor.py`

- [ ] **Step 1: Write planner isolation and executor tests**

Prove the planner performs no write/effect call, loads minimal membership context only for membership keys, fails closed on authorization gateway failure, validates each payload once, emits ordered typed variants, binds exact lock expiry from operation time/config, preserves duplicates, and rejects all-or-nothing. Prove registry duplicate/missing handlers fail before transaction entry.

- [ ] **Step 2: Add executor transaction tests**

Assert whitelisted user writes, override upsert/delete, `CALLER_OWNS_ANCHOR` membership delegation, volatile rechecks, account lock exact set semantics on `failed_attempts` and `account_lockouts`, repeated lock idempotence, unlock reset/removal, no rate-limiter calls, one supplied connection, and typed sanitized unique-email/execution failures.

- [ ] **Step 3: Run tests and verify red**

Run:
```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/UserProfile/test_profile_update_planner.py tldw_Server_API/tests/UserProfile/test_profile_mutation_executors.py tldw_Server_API/tests/UserProfile/test_profile_account_lock_executor.py -q
```
Expected: FAIL because planner still delegates to `UserProfileUpdateService` and executors do not exist.

- [ ] **Step 4: Implement planner and executor interfaces**

```python
class ProfileUpdatePlanner:
    async def plan(
        self,
        command: ProfileUpdateCommand,
        *,
        operation_time: datetime,
        lockout: LockoutConfiguration,
    ) -> PlanDecision:
        raise NotImplementedError


class MutationExecutorRegistry:
    def validate_plan(self, plan: PlanAccepted) -> None:
        raise NotImplementedError

    async def execute_non_touch(
        self, conn: Any, plan: PlanAccepted
    ) -> ExecutionState:
        raise NotImplementedError

    async def execute_touch(
        self, conn: Any, mutation: TouchUserVersionMutation, value: datetime
    ) -> None:
        raise NotImplementedError
```

Connection-aware repositories do not call `ensure_tables()`, acquire another connection, catch-and-omit failures, or log raw exceptions.

- [ ] **Step 5: Run focused tests and commit**

Run the command from Step 3; expected PASS.

```bash
git add tldw_Server_API/app/core/UserProfiles/planner.py tldw_Server_API/app/core/UserProfiles/executor.py tldw_Server_API/app/core/UserProfiles/overrides_repo.py tldw_Server_API/tests/UserProfile
git commit -m "feat(userprofiles): add independent planner and mutation executors"
```

### Task 12: Fence every evaluations configuration writer

**Files:**
- Create: `tldw_Server_API/app/core/Evaluations/rate_limit_config_repo.py`
- Create: `tldw_Server_API/app/core/Evaluations/rate_limit_config_models.py`
- Create: `tldw_Server_API/app/core/Evaluations/rate_limit_config_process.py`
- Modify: `tldw_Server_API/app/core/Evaluations/user_rate_limiter.py:201`
- Create: `tldw_Server_API/tests/Evaluations/unit/test_rate_limit_config_generations.py`
- Create: `tldw_Server_API/tests/Evaluations/unit/test_rate_limit_config_process.py`
- Create: `tldw_Server_API/tests/Evaluations/unit/test_rate_limit_config_boundaries.py`

- [ ] **Step 1: Write generation, cache, and repository-boundary tests**

Test migration/backfill of `reserved_generation` and `applied_generation`, monotonic reservation, one reservation for ordered changes, conditional complete-state write, atomic applied advance, stale-token no-op, default creation, expiry reset, tier upgrade, independent cache instances, reload between reserve/apply, read-failure fail closed, and configuration DML boundary enforcement.

```python
@dataclass(frozen=True)
class ReservedConfigWrite:
    user_id: str
    generation: int
    changes: tuple[EvaluationLimitChange, ...]


@dataclass(frozen=True)
class AppliedConfigWrite:
    generation: int
    config: RateLimitConfig
```

- [ ] **Step 2: Write subprocess lifecycle tests**

Use a child that succeeds, times out, ignores graceful termination, exits with failure, and cannot be confirmed reaped. Assert finite SQLite busy timeout, event-loop responsiveness, soft terminate then hard kill/reap, no late commit, cancellation propagation, sanitized errors, and process-fatal readiness/exit callback on unreaped child.

- [ ] **Step 3: Run tests and verify red**

Run:
```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Evaluations/unit/test_rate_limit_config_generations.py tldw_Server_API/tests/Evaluations/unit/test_rate_limit_config_process.py tldw_Server_API/tests/Evaluations/unit/test_rate_limit_config_boundaries.py -q
```
Expected: FAIL because generation columns, repository, and subprocess adapter are absent.

- [ ] **Step 4: Implement the repository and killable adapter**

Move `UserTier` and `RateLimitConfig` into `rate_limit_config_models.py` so the repository, subprocess worker, and limiter import one cycle-free contract module. The parent first runs a bounded, killable reservation child that commits the next token. It then runs a separate bounded, killable apply child that reads/merges/writes complete config only while the reservation matches and atomically advances `applied_generation`. A failed reservation or apply consumes at most a token and never reports success. The parent reports success only after child reap and local cache eviction. Cached config is served only after a nonblocking, finite-busy applied-generation read matches the entry; cancellation of that read cannot leave a writer behind. The 60-second TTL is retention only.

- [ ] **Step 5: Route every configuration writer and run tests**

Default row creation, automatic expiry reset, `upgrade_user_tier()`, and Stage 2 use the repository. Tracking and usage tables remain outside this protocol.

Run:
```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Evaluations/unit/test_rate_limit_config_generations.py tldw_Server_API/tests/Evaluations/unit/test_rate_limit_config_process.py tldw_Server_API/tests/Evaluations/unit/test_rate_limit_config_boundaries.py tldw_Server_API/tests/Evaluations/unit/test_user_rate_limiter_minute_exact_and_reset.py -q
```
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tldw_Server_API/app/core/Evaluations/rate_limit_config_models.py tldw_Server_API/app/core/Evaluations/rate_limit_config_repo.py tldw_Server_API/app/core/Evaluations/rate_limit_config_process.py tldw_Server_API/app/core/Evaluations/user_rate_limiter.py tldw_Server_API/tests/Evaluations
git commit -m "feat(evaluations): fence rate limit configuration writes"
```

### Task 13: Implement effect dispatch and transaction-owning command orchestration

**Files:**
- Rewrite: `tldw_Server_API/app/core/UserProfiles/effects.py`
- Rewrite: `tldw_Server_API/app/core/UserProfiles/command_service.py`
- Create: `tldw_Server_API/app/core/UserProfiles/composition.py`
- Modify: `tldw_Server_API/app/services/startup_auth.py`
- Rewrite: `tldw_Server_API/tests/UserProfile/test_profile_command_service.py`
- Create: `tldw_Server_API/tests/UserProfile/test_profile_effects.py`
- Create: `tldw_Server_API/tests/UserProfile/test_profile_command_concurrency.py`

- [ ] **Step 1: Write dispatcher registry and timing tests**

Assert missing/duplicate handlers fail before transaction, required effects run in plan order before commit under timeout, all evaluation changes share one set operation/token, best-effort storage invalidation runs only after confirmed commit, ordinary best-effort failure is sanitized and non-fatal, and cancellation/BaseException is never swallowed.

- [ ] **Step 2: Write command state-machine tests with barriers**

Cover stale-first expected mismatch, deterministic plan rejection without write transaction, dry-run accepted/no applied keys, target lock, inner version mismatch, request-order non-touch execution, exact version floor and one final touch, required-effect rollback, commit failure, concurrency conflict, database busy, no post-commit effect after rollback, empty plan preserving version, and cancellation at every boundary. Assert counters for planner rejection, version conflict, concurrency conflict, required-effect failure, rollback code, database busy, commit failure, and best-effort effect failure use only bounded labels.

```python
class _ProfileRollback(RollbackSignal):
    def __init__(self, failure: ProfileCommandResult) -> None:
        super().__init__(failure.error_code)
        self.failure = failure
```

- [ ] **Step 3: Run tests and verify red**

Run:
```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/UserProfile/test_profile_effects.py tldw_Server_API/tests/UserProfile/test_profile_command_service.py tldw_Server_API/tests/UserProfile/test_profile_command_concurrency.py -q
```
Expected: FAIL because the current service requires `db_conn`, returns HTTP-oriented data, and never dispatches real effects.

- [ ] **Step 4: Implement `apply(command)` with rollback outside-context mapping**

The service samples operation time once, performs optional stale read, plans, validates registries/lock set, opens its transaction, locks and unconditionally reads pre-version, rechecks expected version, executes non-touch mutations, computes post candidates and final touch, runs required effects, reads result version, exits/commits, constructs applied keys, then runs best-effort effects. Every known in-transaction failure raises `_ProfileRollback`; the catch that returns its domain failure is outside `async with`.

- [ ] **Step 5: Assemble production dependencies and readiness**

`composition.py` constructs gateways, registries, clock, metrics sink, and configuration exactly once. Startup verifies migrations, override tables, and handler coverage. Domain modules do not import FastAPI, Chatbooks, endpoint schemas, or audit emitters.

- [ ] **Step 6: Run package verification and commit**

Run:
```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/UserProfile/test_profile_contracts.py tldw_Server_API/tests/UserProfile/test_profile_update_policy.py tldw_Server_API/tests/UserProfile/test_profile_update_planner.py tldw_Server_API/tests/UserProfile/test_profile_mutation_executors.py tldw_Server_API/tests/UserProfile/test_profile_effects.py tldw_Server_API/tests/UserProfile/test_profile_command_service.py tldw_Server_API/tests/UserProfile/test_profile_command_concurrency.py tldw_Server_API/tests/Evaluations/unit -q
python -m bandit -r tldw_Server_API/app/core/UserProfiles tldw_Server_API/app/core/Evaluations/rate_limit_config_repo.py tldw_Server_API/app/core/Evaluations/rate_limit_config_process.py -f json -o /tmp/bandit_userprofiles_stage2_wp3.json
git diff --check
```
Expected: PASS and no new Bandit findings. Request code review, resolve valid findings, complete Work Package 3, then commit:

```bash
git add tldw_Server_API/app/core/UserProfiles tldw_Server_API/app/core/Evaluations tldw_Server_API/app/services/startup_auth.py tldw_Server_API/tests/UserProfile tldw_Server_API/tests/Evaluations backlog/tasks
git commit -m "feat(userprofiles): add transaction owning update pipeline"
```

## Work Package 4: Adapter Migration

### Task 14: Implement pure caller mappers and migrate v1/v2 self updates

**Files:**
- Rewrite: `tldw_Server_API/app/core/UserProfiles/response_mappers.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/users.py:450`
- Modify: `tldw_Server_API/app/api/v2/endpoints/user_profiles.py:35`
- Modify: `tldw_Server_API/tests/UserProfile/test_profile_error_mapping.py`
- Modify: `tldw_Server_API/tests/UserProfile/test_user_profile_updates.py`
- Modify: `tldw_Server_API/tests/UserProfile/test_user_profile_v2_contract.py`
- Modify: `tldw_Server_API/tests/UserProfile/test_user_profile_user_audit.py`

- [ ] **Step 1: Write pure mapper matrix tests**

For every domain outcome, assert exact v1 JSON response, v2 nested detail, status, error ordering, retry header metadata, dry-run legacy applied mapping, and empty applied keys for rollback. Assert mappers have no FastAPI imports and return adapter-facing dataclasses.

- [ ] **Step 2: Run tests and verify red**

Run:
```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/UserProfile/test_profile_error_mapping.py tldw_Server_API/tests/UserProfile/test_user_profile_updates.py tldw_Server_API/tests/UserProfile/test_user_profile_v2_contract.py tldw_Server_API/tests/UserProfile/test_user_profile_user_audit.py -q
```
Expected: new mapper assertions FAIL against `LegacyProfileCommandResult` and endpoint-owned mappings.

- [ ] **Step 3: Implement separate public mapper entry points**

```python
def map_v1_self_update(result: ProfileCommandResult) -> LegacyUpdateDecision:
    return _map_legacy_update(result, caller=LegacyCaller.V1_SELF)


def map_v2_self_update(result: ProfileCommandResult) -> V2UpdateDecision:
    return _map_v2_update(result)


def map_admin_single_update(result: ProfileCommandResult) -> LegacyUpdateDecision:
    return _map_legacy_update(result, caller=LegacyCaller.ADMIN_SINGLE)


def map_chatbooks_restore(result: ProfileCommandResult) -> ChatbooksRestoreDecision:
    return ChatbooksRestoreDecision(accepted=result.outcome is ProfileOutcome.SUCCESS)


def map_deprecated_email_update(result: ProfileCommandResult) -> DeprecatedEmailDecision:
    return _map_deprecated_email(result)
```

Move HTTP numeric constants and response text to adapters. Keep v1 and v2 success audit after successful non-dry-run mapping only.

- [ ] **Step 4: Remove request transaction dependencies from v1/v2 update routes**

Build `ProfileUpdateCommand` without contract mode and call `await command_service.apply(command)`. Leave auth/active/verified dependencies and request-level empty-update checks unchanged.

- [ ] **Step 5: Run tests and commit**

Run the command from Step 2; expected PASS.

```bash
git add tldw_Server_API/app/core/UserProfiles/response_mappers.py tldw_Server_API/app/api/v1/endpoints/users.py tldw_Server_API/app/api/v2/endpoints/user_profiles.py tldw_Server_API/tests/UserProfile
git commit -m "refactor(userprofiles): migrate self update adapters"
```

### Task 15: Migrate admin single and give bulk one caller-owned touch

**Files:**
- Modify: `tldw_Server_API/app/services/admin_profiles_service.py:764`
- Modify: `tldw_Server_API/app/api/v1/endpoints/admin/admin_profiles.py`
- Modify: `tldw_Server_API/app/core/UserProfiles/bulk_command_service.py`
- Modify: `tldw_Server_API/tests/UserProfile/test_admin_profiles_service_update.py`
- Modify: `tldw_Server_API/tests/UserProfile/test_user_profile_bulk.py`
- Modify: `tldw_Server_API/tests/UserProfile/test_user_profile_admin_audit.py`
- Modify: `tldw_Server_API/tests/Admin/test_admin_profiles_endpoint_sanitizers.py`

- [ ] **Step 1: Add admin compatibility and bulk anchor tests**

Assert admin scope enforcement precedes command construction, admin single calls only `apply(command)`, successful/dry-run audit metadata remains exact, rollback emits no success audit, and bulk preserves partial-success/order/shape while touching exactly once only when at least one profile mutation commits.

- [ ] **Step 2: Run tests and verify red**

Run:
```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/UserProfile/test_admin_profiles_service_update.py tldw_Server_API/tests/UserProfile/test_user_profile_bulk.py tldw_Server_API/tests/UserProfile/test_user_profile_admin_audit.py tldw_Server_API/tests/Admin/test_admin_profiles_endpoint_sanitizers.py -q
```
Expected: FAIL because admin passes a request transaction and bulk does not own the dedicated anchor.

- [ ] **Step 3: Migrate admin single and bulk**

Admin single maps the transport-neutral result and returns its existing response/audit tuple. Bulk remains on its characterized facade, uses shared pure policy, opens one transaction per target, routes membership through caller-owned mode, computes one version floor, and performs one final explicit touch after any committed profile change.

- [ ] **Step 4: Run tests and commit**

Run the command from Step 2; expected PASS.

```bash
git add tldw_Server_API/app/services/admin_profiles_service.py tldw_Server_API/app/api/v1/endpoints/admin/admin_profiles.py tldw_Server_API/app/core/UserProfiles/bulk_command_service.py tldw_Server_API/tests/UserProfile tldw_Server_API/tests/Admin/test_admin_profiles_endpoint_sanitizers.py
git commit -m "refactor(userprofiles): migrate admin and bulk update paths"
```

### Task 16: Migrate Chatbooks and the deprecated email endpoint

**Files:**
- Modify: `tldw_Server_API/app/core/Chatbooks/chatbook_service.py:3485`
- Modify: `tldw_Server_API/app/api/v1/endpoints/users.py:549`
- Modify: `tldw_Server_API/tests/Chatbooks/test_chatbooks_full_account_import_restore.py`
- Modify: `tldw_Server_API/tests/e2e/test_chatbooks_full_account_media_roundtrip.py`
- Modify: `tldw_Server_API/tests/UserProfile/test_user_profile_legacy_me_update.py`
- Modify: `tldw_Server_API/tests/UserProfile/test_user_profile_deprecations.py`

- [ ] **Step 1: Write exact Chatbooks and deprecated adapter tests**

Chatbooks must preserve email-first then sorted-overrides order, remove its outer AuthNZ transaction, count successful restored payloads, and convert every failure to the existing generic `ValidationError` without values. Deprecated email must cover disabled 410 without deprecation headers, request 422, omitted/unchanged 400, normalized success 200 with headers, disappearing target 404, duplicate/concurrency/commit 500, and busy 503 with `Retry-After` only.

- [ ] **Step 2: Run tests and verify red**

Run:
```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Chatbooks/test_chatbooks_full_account_import_restore.py tldw_Server_API/tests/UserProfile/test_user_profile_legacy_me_update.py tldw_Server_API/tests/UserProfile/test_user_profile_deprecations.py -q
```
Expected: new assertions FAIL because Chatbooks owns an outer transaction and deprecated email writes SQL directly.

- [ ] **Step 3: Migrate both adapters**

Chatbooks calls `ProfileCommandService.apply(command)` directly and maps failure after the service transaction exits. Deprecated email performs the existing no-change check, submits only `identity.email`, has no `Depends(get_db_transaction)`, never issues users SQL, and adds deprecation headers only to a successful response.

- [ ] **Step 4: Run adapter and end-to-end tests**

Run:
```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Chatbooks/test_chatbooks_full_account_import_restore.py tldw_Server_API/tests/e2e/test_chatbooks_full_account_media_roundtrip.py tldw_Server_API/tests/UserProfile/test_user_profile_legacy_me_update.py tldw_Server_API/tests/UserProfile/test_user_profile_deprecations.py -q
```
Expected: PASS.

- [ ] **Step 5: Run Work Package 4 regression/review gate and commit**

Run:
```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/UserProfile tldw_Server_API/tests/Chatbooks/test_chatbooks_full_account_import_restore.py tldw_Server_API/tests/Admin/test_admin_profiles_endpoint_sanitizers.py -q
python -m bandit -r tldw_Server_API/app/api/v1/endpoints/users.py tldw_Server_API/app/api/v2/endpoints/user_profiles.py tldw_Server_API/app/services/admin_profiles_service.py tldw_Server_API/app/core/Chatbooks/chatbook_service.py -f json -o /tmp/bandit_userprofiles_stage2_wp4.json
git diff --check
```
Expected: PASS and no new Bandit findings. Request code review, resolve valid findings, complete Work Package 4, then commit:

```bash
git add tldw_Server_API/app/api tldw_Server_API/app/services/admin_profiles_service.py tldw_Server_API/app/core/Chatbooks/chatbook_service.py tldw_Server_API/tests backlog/tasks
git commit -m "refactor(userprofiles): complete stage 2 adapter migration"
```

## Work Package 5: Removal and Gates

### Task 17: Remove the transitional path and enforce architecture boundaries

**Files:**
- Delete: `tldw_Server_API/app/core/UserProfiles/update_service.py`
- Modify: `tldw_Server_API/app/core/UserProfiles/__init__.py`
- Modify: `tldw_Server_API/app/core/UserProfiles/README.md`
- Modify: `tldw_Server_API/tests/UserProfile/test_profile_write_boundaries.py`
- Create: `tldw_Server_API/tests/UserProfile/test_stage2_import_boundaries.py`
- Remove or rewrite: tests whose only purpose is direct `UserProfileUpdateService` behavior

- [ ] **Step 1: Write structural gates before deletion**

Use AST/import inspection to reject FastAPI in UserProfiles domain modules, `UserProfileUpdateService`, `apply_with_connection`, prepared-result contracts, caller direct profile SQL, unapproved users/membership/evaluations config DML, pool acquisition from connection-aware methods, and unbounded/high-cardinality metric labels.

```python
FORBIDDEN_SINGLE_UPDATE_NAMES = frozenset({
    "UserProfileUpdateService", "apply_with_connection", "PreparedProfileUpdate",
})
DOMAIN_FORBIDDEN_IMPORT_PREFIXES = ("fastapi", "tldw_Server_API.app.api")
```

- [ ] **Step 2: Run structural tests and enumerate every remaining dependency**

Run:
```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/UserProfile/test_stage2_import_boundaries.py tldw_Server_API/tests/UserProfile/test_profile_write_boundaries.py -q
```
Expected: FAIL with exact remaining transitional imports/calls, then PASS only after all are migrated or deleted.

- [ ] **Step 3: Delete transitional implementation and update module documentation**

Remove `update_service.py` and direct-behavior tests superseded by pure policy, planner, executor, and adapter tests. Document transaction order, rollback rule, profile-version algorithm, membership writer ownership, evaluations fencing, best-effort cache limitation, metrics, and five public adapter boundaries.

- [ ] **Step 4: Run imports and compile checks**

Run:
```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/UserProfile/test_stage2_import_boundaries.py tldw_Server_API/tests/UserProfile/test_profile_write_boundaries.py -q
python -m compileall -q tldw_Server_API/app/core/UserProfiles tldw_Server_API/app/core/AuthNZ tldw_Server_API/app/core/Evaluations
```
Expected: PASS with no transitional name or import remaining.

- [ ] **Step 5: Commit**

```bash
git add -A tldw_Server_API/app/core/UserProfiles tldw_Server_API/tests/UserProfile
git commit -m "refactor(userprofiles): remove transitional update pipeline"
```

### Task 18: Run cross-backend, privacy, OpenAPI, and security release gates

**Files:**
- Create: `tldw_Server_API/tests/UserProfile/test_stage2_privacy.py`
- Modify: `tldw_Server_API/tests/UserProfile/test_stage2_caller_characterization.py`
- Modify: relevant OpenAPI snapshot/contract tests discovered by `rg -n "openapi" tldw_Server_API/tests`
- Modify: Work Package 5 Backlog child and `TASK-13001`

- [ ] **Step 1: Add end-to-end privacy and metric-label tests**

Inject a unique email, username, synthetic secret, and absolute database path through unique conflict, database busy, membership failure, effect timeout, commit failure, and unexpected gateway failure. Capture lower-layer through adapter logs/results and assert none of those values appears. Assert metrics contain stable bounded labels only.

- [ ] **Step 2: Run all focused SQLite suites**

Run:
```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/UserProfile tldw_Server_API/tests/AuthNZ/unit tldw_Server_API/tests/AuthNZ_SQLite tldw_Server_API/tests/Evaluations/unit tldw_Server_API/tests/Chatbooks/test_chatbooks_full_account_import_restore.py tldw_Server_API/tests/Admin/test_admin_profiles_endpoint_sanitizers.py -q
```
Expected: PASS.

- [ ] **Step 3: Run PostgreSQL integration gates**

Run:
```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/AuthNZ_Postgres/test_user_profile_version_locking_pg.py tldw_Server_API/tests/AuthNZ_Postgres/test_membership_writer_concurrency_pg.py tldw_Server_API/tests/AuthNZ_Postgres/test_membership_scope_delete_concurrency_pg.py tldw_Server_API/tests/AuthNZ_Postgres/test_orgs_teams_pg.py tldw_Server_API/tests/AuthNZ/integration/test_authnz_orgs_teams_repo_postgres.py tldw_Server_API/tests/AuthNZ/integration/test_registration_role_membership_postgres.py -q
```
Expected: PASS against PostgreSQL. A local environment skip is recorded but does not satisfy the merge gate; required CI checks must pass.

- [ ] **Step 4: Compare public OpenAPI and exact caller fixtures**

Regenerate the application OpenAPI through the existing project test helper and compare the v1/v2/admin/deprecated operations and response schemas. Run the full characterization table and verify only the approved non-200 rollback correction differs from legacy runtime behavior.

Run:
```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/UserProfile/test_stage2_caller_characterization.py tldw_Server_API/tests/UserProfile/test_user_profile_v2_contract.py tldw_Server_API/tests/UserProfile/test_user_profile_deprecations.py -q
```
Expected: PASS with exact status/body/header/audit fixtures.

- [ ] **Step 5: Run final static and security gates**

Run:
```bash
source .venv/bin/activate
python -m compileall -q tldw_Server_API/app
python -m bandit -r tldw_Server_API/app/core/UserProfiles tldw_Server_API/app/core/AuthNZ/profile_version.py tldw_Server_API/app/core/AuthNZ/membership_writer.py tldw_Server_API/app/core/AuthNZ/database.py tldw_Server_API/app/core/Evaluations/rate_limit_config_repo.py tldw_Server_API/app/core/Evaluations/rate_limit_config_process.py tldw_Server_API/app/api/v1/endpoints/users.py tldw_Server_API/app/api/v2/endpoints/user_profiles.py tldw_Server_API/app/services/admin_profiles_service.py -f json -o /tmp/bandit_userprofiles_stage2_final.json
git diff --check
```
Expected: zero compile failures, no new Bandit findings, and clean diff check.

- [ ] **Step 6: Review, finalize tracking, and commit**

Request a final independent review focused on rollback/commit ordering, total lock order, generation fencing, privacy, contract drift, and missing writer paths. Resolve every valid issue and repeat affected gates. Complete the Work Package 5 Backlog child and `TASK-13001` with verification evidence, CI links, known cross-system limitation, touched files, and final summary.

```bash
git add tldw_Server_API Docs/superpowers backlog/tasks
git commit -m "test(userprofiles): enforce stage 2 release gates"
```

## Final Acceptance Checklist

- [ ] All five callers use only `ProfileCommandService.apply(command)`.
- [ ] Chatbooks owns no outer AuthNZ transaction and no connection bridge exists.
- [ ] No single-update runtime module imports or invokes `UserProfileUpdateService`.
- [ ] Commands/plans are frozen, typed, ordered, and transport-neutral.
- [ ] Every non-empty successful apply has exactly one final user anchor touch and strict version advance.
- [ ] Every profile-visible AuthNZ users creator/writer initializes or advances the anchor in the same transaction.
- [ ] Every production org/team membership writer follows the total-order protocol and explicit anchor ownership.
- [ ] Account lock/unlock is exact, idempotent, connection-bound set mutation.
- [ ] Required evaluations state is generation-fenced, bounded, killable, and applied-generation cache aware.
- [ ] Rollback, busy, conflict, required-effect, commit, and cancellation paths cannot report success or emit success audit.
- [ ] Public status/body/header/OpenAPI contracts are unchanged except the approved non-200 rollback correction.
- [ ] Privacy tests prove no submitted value, identity string, secret, raw exception, or absolute path leaks below or above the command service.
- [ ] SQLite and PostgreSQL gates, structural boundaries, compile checks, Bandit, and final independent review pass.
