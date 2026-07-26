# OpenAI Credential Lock Pool Capacity Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent distinct-user OpenAI credential mutations from exhausting the main AuthNZ PostgreSQL pool while preserving cross-worker serialization and fail-closed behavior on every lock backend.

**Architecture:** `DatabasePool` owns a second asyncpg pool with `min_size=0` and fixed `max_size=4`, exposes one explicit connection-acquisition context manager, and closes the dedicated pool before the main pool. `byok_runtime` exposes one public OpenAI credential-mutation lock and keeps the existing OAuth refresh helper as a compatibility delegate. SQLite continues using native `FileLock`; Redis remains an explicitly configured ownership lease and the documented scale path.

**Tech Stack:** Python 3.11, asyncio, asyncpg, native `fcntl`/`msvcrt` file locks, redis-py asyncio, Pydantic Settings, pytest.

## Global Constraints

- Work is tracked by TASK-12963 and the approved design at `Docs/superpowers/specs/2026-07-14-openai-oauth-lock-pool-capacity-hardening-design.md`.
- Do not commit; the root agent owns final integration.
- Do not fall back to the main AuthNZ pool when the dedicated PostgreSQL lock pool is unavailable.
- Canonicalize the provider before deriving a lock key; OpenAI aliases contend together and non-OpenAI identities are rejected.
- Require positive PostgreSQL advisory-unlock confirmation without masking an exception already raised by the protected body.
- Keep `OPENAI_OAUTH_REFRESH_LOCK_BACKEND` for compatibility; missing or invalid values resolve to `db`.
- PostgreSQL DB locking requires a direct or session-pooled connection. PgBouncer transaction-pooling deployments must use Redis.
- Redis selected without `REDIS_URL` fails closed with `credential_store_unavailable`.
- No endpoint, adapter, credential-snapshot, RAG, or embeddings files are in this slice.

---

### Task 1: Dedicated DatabasePool lifecycle

**Files:**
- Modify: `tldw_Server_API/app/core/AuthNZ/database.py`
- Create: `tldw_Server_API/tests/AuthNZ/unit/test_database_openai_credential_lock_pool.py`

**Interfaces:**
- Produces: `DatabasePool.acquire_openai_credential_lock_connection(*, timeout: float | None = None)` as an async context manager yielding one dedicated asyncpg connection.
- Produces: `OPENAI_CREDENTIAL_LOCK_POOL_MAX_SIZE = 4`.
- Owns: creation and shutdown of `DatabasePool._openai_credential_lock_pool`.

- [x] **Step 1: Write failing creation, acquisition, and lifecycle tests**

Create fakes for the main and lock pools and patch `asyncpg.create_pool`. Assert that PostgreSQL initialization makes two pool calls, the second uses `min_size=0` and `max_size=4`, the explicit acquire method yields only the lock-pool connection, and `close()` closes the lock pool before the main pool. Add a failure case proving main-pool cleanup when dedicated-pool creation fails.

```python
assert create_calls[1]["min_size"] == 0
assert create_calls[1]["max_size"] == 4
async with pool.acquire_openai_credential_lock_connection(timeout=1) as conn:
    assert conn is lock_connection
await pool.close()
assert close_order == ["lock", "main"]
```

- [x] **Step 2: Run the new test file and verify RED**

Run:

```bash
source .venv/bin/activate
python -m pytest -q --randomly-seed=12963 \
  tldw_Server_API/tests/AuthNZ/unit/test_database_openai_credential_lock_pool.py
```

Expected: failures because the dedicated pool attribute, explicit acquire method, and bounded second pool do not exist.

- [x] **Step 3: Implement the minimal DatabasePool-owned pool**

Add the fixed bound and nullable attribute. In the PostgreSQL initialization path, create the second pool with zero minimum connections and otherwise matching DSN/lifetime options. Add an async context manager that fails if the pool is missing, bounds acquire/release, and uses the existing cancellation-safe connection-release helper.

```python
OPENAI_CREDENTIAL_LOCK_POOL_MAX_SIZE = 4

@asynccontextmanager
async def acquire_openai_credential_lock_connection(self, *, timeout=None):
    if not self._initialized:
        await self.initialize()
    pool = self._openai_credential_lock_pool
    if pool is None:
        raise DatabaseError("OpenAI credential lock pool unavailable")
    conn = await pool.acquire(timeout=timeout)
    try:
        yield conn
    finally:
        await _await_connection_release(pool.release(conn, timeout=timeout))
```

Close the dedicated pool before the main pool and clean up a partially initialized main pool if dedicated-pool creation fails.

- [x] **Step 4: Run the new test file and verify GREEN**

Run the Step 2 command. Expected: all tests pass.

---

### Task 2: Shared mutation lock and main-pool capacity regression

**Files:**
- Modify: `tldw_Server_API/app/core/AuthNZ/byok_runtime.py`
- Modify: `tldw_Server_API/tests/AuthNZ_Unit/test_byok_runtime.py`
- Create: `tldw_Server_API/tests/AuthNZ_Unit/test_openai_credential_mutation_lock.py`

**Interfaces:**
- Consumes: `DatabasePool.acquire_openai_credential_lock_connection(timeout=...)`.
- Produces: `openai_credential_mutation_lock(*, user_id: int, provider: str = "openai")`, yielding the connection-bound user-secret repository for PostgreSQL and `None` for SQLite, Redis, and memory.
- Produces: `openai_oauth_credential_generation(payload)`, an opaque access-token generation digest for mutation coalescing.
- Preserves: `_openai_oauth_refresh_lock(...)` as a delegating compatibility context manager.

- [x] **Step 1: Write the distinct-user capacity RED test**

Use a fake DatabasePool with independent semaphores for its main and dedicated pools. Start four different-user mutation-lock holders and block their bodies. While all four hold advisory locks, acquire the main pool under a short timeout and assert success.

```python
holders = [asyncio.create_task(_hold(user_id)) for user_id in range(4)]
await all_holders_entered.wait()
async with asyncio.timeout(0.1):
    async with pool.acquire():
        pass
```

Current code must fail because it uses `pool.acquire()` for the advisory lock.

- [x] **Step 2: Write the public seam and fail-closed tests**

Assert the public context manager yields the connection-bound repository, the private refresh helper delegates to it, and a fake PostgreSQL pool without the explicit dedicated acquisition method fails with `credential_store_unavailable` rather than borrowing `acquire()`. Exercise revoke and CAS through the yielded repository so `fetchone` and `execute` are proven to use the lock-owning connection; assert the CAS query has one active-row/blob predicate. Assert `OAI` and `OpenAI` contend on one canonical lock and a non-OpenAI identity is rejected. Assert false/`None` advisory unlock fails closed, unlock cleanup completes before cancellation, and cleanup failure does not mask an existing protected-body exception. Assert the public generation seam is opaque, stable for an unchanged access token, changes for a changed access token, and ignores refresh-token-only or unrelated metadata changes. Keep `_openai_oauth_generation` as a compatibility wrapper.

- [x] **Step 3: Run the new capacity/seam tests and verify RED**

Run:

```bash
source .venv/bin/activate
python -m pytest -q --randomly-seed=12963 \
  tldw_Server_API/tests/AuthNZ_Unit/test_openai_credential_mutation_lock.py \
  -k 'distinct_user or public or dedicated'
```

Expected: failures because PostgreSQL still uses the main acquire method and the public mutation context does not exist.

- [x] **Step 4: Implement the shared mutation context**

Refactor the existing backend switch into `openai_credential_mutation_lock`. Canonicalize and validate the OpenAI provider identity before deriving the lock key. Make the PostgreSQL branch call only `acquire_openai_credential_lock_connection(timeout=remaining)`. Preserve the connection-bound `AuthnzUserProviderSecretsRepo`, add its exact `execute` mutation adapter alongside `fetchone`, and retain the lock timeout, SQLite FileLock path, Redis lease, and memory lock. Require a truthy advisory-unlock result after a successful body, preserve an already-propagating body exception if cleanup fails, and make `_openai_oauth_refresh_lock` delegate without nesting.

Rename the generation implementation to
`openai_oauth_credential_generation`, retain the private name as a thin
compatibility wrapper, and use the public name internally.

```python
@contextlib.asynccontextmanager
async def _openai_oauth_refresh_lock(*, user_id: int, provider: str):
    async with openai_credential_mutation_lock(
        user_id=user_id,
        provider=provider,
    ) as locked_user_repo:
        yield locked_user_repo
```

Update only the existing PostgreSQL lock fakes in `test_byok_runtime.py` to implement the new production interface.

- [x] **Step 5: Run the capacity/seam tests and existing OAuth lock baseline**

Run:

```bash
source .venv/bin/activate
python -m pytest -q --randomly-seed=12963 \
  tldw_Server_API/tests/AuthNZ_Unit/test_openai_credential_mutation_lock.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_byok_runtime.py \
  -k 'credential_mutation_lock or oauth_refresh_lock or postgres_oauth_refresh'
```

Expected: all selected tests pass.

---

### Task 3: SQLite, settings, and Redis regressions

**Files:**
- Modify: `tldw_Server_API/tests/AuthNZ_Unit/test_openai_credential_mutation_lock.py`
- Create: `tldw_Server_API/tests/AuthNZ/unit/test_openai_credential_lock_settings.py`

**Interfaces:**
- Verifies existing `FileLock`, backend normalization, and Redis configuration behavior through the public mutation-lock boundary.

- [x] **Step 1: Add real FileLock concurrency regressions**

Use one temporary lock directory and a fake SQLite backend. Run two callers in separate threads with independent `asyncio.run()` event loops and assert the protected-body maximum is one. Add async tests proving timeout, owner-cancellation release, and that cancelling a waiter cannot release the owner's native lock.

```python
with ThreadPoolExecutor(max_workers=2) as executor:
    futures = [executor.submit(lambda: asyncio.run(_worker())) for _ in range(2)]
assert max_active == 1
```

- [x] **Step 2: Add settings and Redis regressions**

Construct `Settings(_env_file=None, ...)` with the backend omitted and invalid and assert `db`. Select Redis while returning settings with `REDIS_URL=None`; assert the public lock raises `ByokResolutionError` with code `credential_store_unavailable` and never calls memory/DB lock paths.

- [x] **Step 3: Run the new regressions**

Run:

```bash
source .venv/bin/activate
python -m pytest -q --randomly-seed=12963 \
  tldw_Server_API/tests/AuthNZ_Unit/test_openai_credential_mutation_lock.py \
  tldw_Server_API/tests/AuthNZ/unit/test_openai_credential_lock_settings.py
```

Expected: all characterization tests pass after Task 2; any failure indicates the documented existing backend contract is not actually preserved and must be fixed at the shared boundary.

---

### Task 4: Canonical operator documentation

**Files:**
- Modify: `tldw_Server_API/Config_Files/.env.example`
- Modify: `Docs/Operations/Env_Vars.md`
- Modify: `Docs/Deployment/horizontal-scaling.md`

**Interfaces:**
- Documents: `OPENAI_OAUTH_REFRESH_LOCK_BACKEND=db|redis|memory`, `OPENAI_OAUTH_REFRESH_LOCK_DIR`, and `REDIS_URL` dependency.

- [x] **Step 1: Add canonical environment examples**

State that `db` is the safe default, `memory` is single-process only, and explicit Redis requires `REDIS_URL` with no fallback.

- [x] **Step 2: Add deployment guidance**

Document the dedicated DB pool's zero-idle/four-session per-process bound. Require Redis for high credential-mutation concurrency, multi-process coordination where DB session locks are unsuitable, and every PgBouncer transaction-pooling deployment. State that direct PostgreSQL and PgBouncer session pooling are compatible with DB advisory locks.

- [x] **Step 3: Verify canonical wording**

Run:

```bash
rg -n "OPENAI_OAUTH_REFRESH_LOCK_BACKEND|PgBouncer|credential mutation|four" \
  tldw_Server_API/Config_Files/.env.example \
  Docs/Operations/Env_Vars.md \
  Docs/Deployment/horizontal-scaling.md
```

Expected: all three canonical documents describe backend selection and the deployment guide contains the PgBouncer restriction and Redis scale guidance.

---

### Task 5: Final verification and task record

**Files:**
- Modify through official Backlog workflow: TASK-12963 implementation notes/documentation links

- [x] **Step 1: Run focused and adjacent suites**

```bash
source .venv/bin/activate
python -m pytest -q --randomly-seed=12963 \
  tldw_Server_API/tests/AuthNZ/unit/test_database_openai_credential_lock_pool.py \
  tldw_Server_API/tests/AuthNZ/unit/test_openai_credential_lock_settings.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_openai_credential_mutation_lock.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_byok_runtime.py \
  tldw_Server_API/tests/Services/test_startup_auth.py \
  tldw_Server_API/tests/Services/test_shutdown_auth_db_pool.py
```

- [x] **Step 2: Run static and security gates**

```bash
python -m ruff check \
  tldw_Server_API/app/core/AuthNZ/database.py \
  tldw_Server_API/app/core/AuthNZ/byok_runtime.py \
  tldw_Server_API/tests/AuthNZ/unit/test_database_openai_credential_lock_pool.py \
  tldw_Server_API/tests/AuthNZ/unit/test_openai_credential_lock_settings.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_openai_credential_mutation_lock.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_byok_runtime.py
python -m py_compile \
  tldw_Server_API/app/core/AuthNZ/database.py \
  tldw_Server_API/app/core/AuthNZ/byok_runtime.py \
  tldw_Server_API/tests/AuthNZ/unit/test_database_openai_credential_lock_pool.py \
  tldw_Server_API/tests/AuthNZ/unit/test_openai_credential_lock_settings.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_openai_credential_mutation_lock.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_byok_runtime.py
python -m bandit -r \
  tldw_Server_API/app/core/AuthNZ/database.py \
  tldw_Server_API/app/core/AuthNZ/byok_runtime.py \
  -f json -o /tmp/bandit_TASK-12963_openai_lock_pool.json
git diff --check -- \
  tldw_Server_API/app/core/AuthNZ/database.py \
  tldw_Server_API/app/core/AuthNZ/byok_runtime.py \
  tldw_Server_API/tests/AuthNZ/unit/test_database_openai_credential_lock_pool.py \
  tldw_Server_API/tests/AuthNZ/unit/test_openai_credential_lock_settings.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_openai_credential_mutation_lock.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_byok_runtime.py \
  tldw_Server_API/Config_Files/.env.example \
  Docs/Operations/Env_Vars.md \
  Docs/Deployment/horizontal-scaling.md
```

Expected: focused tests, Ruff, compilation, and diff checks pass; Bandit reports no new findings.

- [x] **Step 3: Self-review production invariants**

Confirm no PostgreSQL lock path calls the main `DatabasePool.acquire()`, no explicit Redis path falls back, every acquired native/distributed lock has cancellation-safe release, the dedicated pool closes before the main pool, and endpoint callers can use the public mutation seam without nesting forced resolution.

- [x] **Step 4: Update TASK-12963 through Backlog MCP**

Append the design/plan links, files, test counts, Bandit result, documented PgBouncer restriction, and any residual blocker. Do not commit.
