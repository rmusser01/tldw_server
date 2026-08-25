# AuthNZ Request-Time UsersDB Schema DDL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent ordinary authenticated repository operations from running users-table schema DDL while preserving explicit `UsersDB` schema initialization.

**Architecture:** `DatabasePool` and AuthNZ startup remain the schema owners. `UsersDB.initialize()` gains a keyword-only, default-on schema assurance switch and tracks pool readiness separately from schema assurance; `AuthnzUsersRepo` explicitly opts out because it receives the shared ready pool. SQLite legacy repair moves out of request writes into migration 91, while the canonical base schema matches the `UsersDB` write contract.

**Tech Stack:** Python 3.11, asyncio, FastAPI AuthNZ repositories, SQLite, PostgreSQL/asyncpg, pytest

---

### Task 1: Write Every Regression Before Production Code

**Files:**
- Create: `tldw_Server_API/tests/AuthNZ/unit/test_users_repo_request_path_initialization.py`
- Modify: `tldw_Server_API/tests/AuthNZ/integration/test_authnz_users_repo_postgres.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_users_repo_request_path_initialization.py`
- Test: `tldw_Server_API/tests/AuthNZ/integration/test_authnz_users_repo_postgres.py`

- [x] **Step 1: Write the failing repository regression test**

Create a ready pool stub whose `fetchone()` returns a valid user and whose
`transaction()` raises if request-time initialization attempts schema DDL:

```python
class ReadyPoolStub:
    pool = object()

    async def fetchone(self, query: str, *args: object) -> dict[str, object]:
        return {
            "id": 7,
            "uuid": "00000000-0000-0000-0000-000000000007",
            "username": "reader",
            "email": "reader@example.test",
            "password_hash": "hash",
            "is_active": True,
        }

    def transaction(self) -> None:
        raise AssertionError("repository lookup attempted schema DDL")


@pytest.mark.asyncio
async def test_user_lookup_does_not_run_schema_ddl() -> None:
    repo = AuthnzUsersRepo(db_pool=ReadyPoolStub())

    user = await repo.get_user_by_id(7)

    assert user is not None
    assert user["id"] == 7
```

- [x] **Step 2: Add complete initialization-contract tests**

Add both API guarantees explicitly:

```python
@pytest.mark.asyncio
async def test_users_db_initialize_ensures_schema_by_default(monkeypatch) -> None:
    db = UsersDB(db_pool=object())
    calls = 0

    async def record_create_tables() -> None:
        nonlocal calls
        calls += 1

    monkeypatch.setattr(db, "_create_tables", record_create_tables)

    await db.initialize()

    assert calls == 1


@pytest.mark.asyncio
async def test_schema_opt_out_still_acquires_global_pool(monkeypatch) -> None:
    ready_pool = object()
    db = UsersDB()

    async def get_ready_pool() -> object:
        return ready_pool

    async def reject_create_tables() -> None:
        raise AssertionError("schema opt-out attempted DDL")

    monkeypatch.setattr(users_db_module, "get_db_pool", get_ready_pool)
    monkeypatch.setattr(db, "_create_tables", reject_create_tables)

    await db.initialize(ensure_schema=False)

    assert db.db_pool is ready_pool
```

The default test passes before implementation and protects backward
compatibility. The opt-out test fails with the expected unexpected-keyword
error before implementation.

- [x] **Step 3: Add the PostgreSQL concurrent no-DDL regression**

In the existing PostgreSQL integration test, wrap `UsersDB._create_tables` to
count invocations, then perform 24 concurrent lookups through the real asyncpg
pool:

```python
ddl_calls = 0

async def count_create_tables(self: UsersDB) -> None:
    nonlocal ddl_calls
    ddl_calls += 1

monkeypatch.setattr(UsersDB, "_create_tables", count_create_tables)
rows = await asyncio.gather(
    *(repo.get_user_by_id(user_id) for _ in range(24))
)

assert all(row is not None and row["id"] == user_id for row in rows)
assert ddl_calls == 0
```

Before the fix this fails because request lookups invoke the spy. The spy must
not call the real DDL implementation: doing so can recreate the deadlock the
test is intended to detect and make the regression flaky. After the fix the
lookups perform only concurrent user queries.

- [x] **Step 4: Run the new tests and verify red**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/AuthNZ/unit/test_users_repo_request_path_initialization.py -v
TLDW_TEST_POSTGRES_REQUIRED=1 python -m pytest \
  tldw_Server_API/tests/AuthNZ/integration/test_authnz_users_repo_postgres.py \
  -v
```

Expected: the repository and opt-out unit tests fail for the missing opt-out;
the PostgreSQL test fails because DDL is observed or deadlocks. Confirm the
default schema-assurance test passes independently.

### Task 2: Add the Explicit Schema-Assurance Switch

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/Users_DB.py:103-116`
- Modify: `tldw_Server_API/app/core/AuthNZ/repos/users_repo.py:42-45`
- Modify: `tldw_Server_API/app/core/AuthNZ/migrations.py`
- Modify: `tldw_Server_API/Databases/SQLite/Schema/sqlite_users.sql`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_users_repo_request_path_initialization.py`
- Test: `tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_users_repo_sqlite.py`

- [x] **Step 1: Implement the minimal `UsersDB` API change**

Change initialization to keep schema assurance on by default:

```python
async def initialize(self, *, ensure_schema: bool = True) -> None:
    """Initialize database access and optionally ensure users tables exist."""
    if self._initialized and (not ensure_schema or self._schema_ensured):
        return

    if not self.db_pool:
        self.db_pool = await get_db_pool()

    if ensure_schema and not self._schema_ensured:
        await self._create_tables()
        self._schema_ensured = True

    self._initialized = True
    logger.info("UsersDB initialized")
```

- [x] **Step 2: Opt the repository out of schema DDL**

Use the existing shared pool explicitly:

```python
async def _users_db(self) -> UsersDB:
    db = UsersDB(self.db_pool)
    await db.initialize(ensure_schema=False)
    return db
```

- [x] **Step 3: Run the focused tests and verify green**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/AuthNZ/unit/test_users_repo_request_path_initialization.py \
  tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_users_repo_sqlite.py \
  -v
TLDW_TEST_POSTGRES_REQUIRED=1 python -m pytest \
  tldw_Server_API/tests/AuthNZ/integration/test_authnz_users_repo_postgres.py \
  -v
```

Expected: PASS. The repository regression performs only the user query; the
existing SQLite path still bootstraps through explicit `UsersDB.initialize()`,
and the PostgreSQL test cannot silently skip.

- [x] **Step 4: Move legacy SQLite repair to startup migration ownership**

Add idempotent migration 91 to ensure the columns required by `UsersDB` writes,
backfill missing UUIDs, and create the unique UUID index. Align
`sqlite_users.sql` for fresh databases. Remove schema inspection and mutation
from `UsersDB.create_user()`, and prove `AuthnzUsersRepo.create_user()` performs
no `PRAGMA`, `ALTER`, or `CREATE` statements.

- [x] **Step 5: Run static checks on the touched modules**

Run:

```bash
source .venv/bin/activate
python -m compileall -q \
  tldw_Server_API/app/core/DB_Management/Users_DB.py \
  tldw_Server_API/app/core/AuthNZ/repos/users_repo.py \
  tldw_Server_API/app/core/AuthNZ/migrations.py \
  tldw_Server_API/tests/AuthNZ/unit/test_users_repo_request_path_initialization.py \
  tldw_Server_API/tests/AuthNZ/integration/test_authnz_users_repo_postgres.py \
  tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_users_repo_sqlite.py
python -m ruff check \
  tldw_Server_API/app/core/DB_Management/Users_DB.py \
  tldw_Server_API/app/core/AuthNZ/repos/users_repo.py \
  tldw_Server_API/app/core/AuthNZ/migrations.py \
  tldw_Server_API/tests/AuthNZ/unit/test_users_repo_request_path_initialization.py \
  tldw_Server_API/tests/AuthNZ/integration/test_authnz_users_repo_postgres.py \
  tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_users_repo_sqlite.py
```

Expected: no errors.

### Task 3: Run the AuthNZ Regression Set

**Files:**
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_users_repo_request_path_initialization.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_user_db_handling_jwt_membership.py`
- Test: `tldw_Server_API/tests/AuthNZ_Unit/test_auth_jwt_membership_validation.py`
- Test: `tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_users_repo_sqlite.py`
- Test: `tldw_Server_API/tests/AuthNZ/integration/test_authnz_users_repo_postgres.py`

- [x] **Step 1: Run the nearby AuthNZ regression set**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/AuthNZ/unit/test_users_repo_request_path_initialization.py \
  tldw_Server_API/tests/AuthNZ/unit/test_user_db_handling_jwt_membership.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_auth_jwt_membership_validation.py \
  tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_users_repo_sqlite.py \
  tldw_Server_API/tests/AuthNZ/unit/test_rbac_seed_helper.py \
  -v
TLDW_TEST_POSTGRES_REQUIRED=1 python -m pytest \
  tldw_Server_API/tests/AuthNZ/integration/test_authnz_users_repo_postgres.py \
  -v
```

Expected: PASS with PostgreSQL executed, not skipped.

### Task 4: Live Acceptance, Security, and Closeout

**Files:**
- Modify: `backlog/tasks/task-12020.43 - Stop-request-time-UsersDB-schema-DDL-during-authenticated-repository-access.md`
- Modify temporary evidence runner: `/private/tmp/task-12020-39/shared-with-me-cdp.mjs`

- [x] **Step 1: Make live HTTP diagnostics gating**

Add a targeted notes-search response collection and require the request to occur
and return 2xx. Before assigning `targeted-flow-passed`, reject every HTTP error
except one non-shared local context 404 that is proven transient by a later 2xx
response from the exact same generated workspace path. The local workspace UUID
is generated during each clean-browser run and cannot be hardcoded safely:

```javascript
const WORKSPACE_CONTEXT_PATH = /^\/api\/v1\/workspaces\/([^/]+)\/context$/

const notesResponsePromise = page.waitForResponse(
  (response) => new URL(response.url()).pathname === "/api/v1/notes/search/",
  { timeout: FLOW_TIMEOUT_MS }
)

// Start the promise before opening the shared workspace. Await it after the
// shared banner renders so the initialization request is mandatory.
const notesResponse = await notesResponsePromise
evidence.targetedResponses.push({
  path: "/api/v1/notes/search/",
  status: notesResponse.status()
})

// If exactly one non-shared context path returned 404, require that same path
// to return 2xx before excluding the transient miss from unexpected errors.
if (notesResponse.status() < 200 || notesResponse.status() >= 300) {
  throw new Error(`Notes search returned HTTP ${notesResponse.status()}`)
}
if (unexpectedHttpErrors.length > 0) {
  throw new Error(`Live flow recorded ${unexpectedHttpErrors.length} unexpected HTTP error(s)`)
}
evidence.assertions.notesSearchReturned2xx = true
evidence.assertions.noUnexpectedHttpErrors = true
```

This prevents the prior false-positive state where the evidence file said
`targeted-flow-passed` while containing the notes 500, without masking a
persistent or unrelated workspace-context failure.

- [x] **Step 2: Restart the isolated task backend exactly**

The task39 PostgreSQL fixture is persistent and already contains the owner,
recipient, org membership, workspace, and share. Do not reset or reseed it in
this blocker task. Start the existing container and patched API from the
worktree root:

```bash
docker start task1202039-postgres
source .venv/bin/activate
set -a
source /private/tmp/task-12020-39/runtime.env
set +a
export AUTH_MODE=multi_user
export DATABASE_URL="postgresql://tldw_uat:${TASK_12020_39_PG_PASSWORD}@127.0.0.1:55439/task1202039"
export JWT_SECRET_KEY="${TASK_12020_39_JWT_SECRET}"
export SESSION_ENCRYPTION_KEY="${TASK_12020_39_SESSION_KEY}"
export MCP_JWT_SECRET="${TASK_12020_39_MCP_JWT_SECRET}"
export MCP_API_KEY_SALT="${TASK_12020_39_MCP_API_KEY_SALT}"
export TLDW_ADMIN_E2E_SUPPORT_KEY="${TASK_12020_39_SUPPORT_KEY}"
export TLDW_CONFIG_FILE=/private/tmp/task-12020-39/config.txt
export USER_DB_BASE_DIR=/private/tmp/task-12020-39/user_databases
export JOBS_DB_PATH=/private/tmp/task-12020-39/jobs-task43.db
export CIRCUIT_BREAKER_REGISTRY_DB_PATH=/private/tmp/task-12020-39/circuit-breakers-task43.db
export WATCHLIST_TEMPLATE_DIR=/private/tmp/task-12020-39/watchlist-templates
export SCHEDULER_BASE_PATH=/private/tmp/task-12020-39/scheduler
export HOME=/private/tmp/task-12020-39/home
python -m uvicorn tldw_Server_API.app.main:app \
  --host 127.0.0.1 --port 18242
```

In a second shell, require a healthy response:

```bash
curl -sf http://127.0.0.1:18242/api/v1/health \
  > /private/tmp/task-12020-39/health-task43.json
```

- [x] **Step 3: Start the WebUI and isolated Google Chrome CDP target**

Start the WebUI from `apps/tldw-frontend`:

```bash
NEXT_TELEMETRY_DISABLED=1 \
NEXT_PUBLIC_API_URL=http://127.0.0.1:18242 \
bun run dev:webpack -- -H 127.0.0.1 -p 18240
```

Require the page server to respond:

```bash
curl -sf http://127.0.0.1:18240/shared > /dev/null
```

Start a real isolated Google Chrome process. All browser interaction after
startup is through CDP; do not use computer control:

```bash
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --headless=new \
  --remote-debugging-address=127.0.0.1 \
  --remote-debugging-port=18241 \
  --user-data-dir=/private/tmp/task-12020-39/chrome-profile \
  --no-first-run \
  --no-default-browser-check \
  about:blank
```

Require the CDP endpoint to expose a websocket debugger URL:

```bash
curl -sf http://127.0.0.1:18241/json/version \
  | jq -e '.webSocketDebuggerUrl | type == "string" and length > 0'
```

- [x] **Step 4: Refresh login and verify the inherited fixture**

Use the real OAuth form endpoint, then prove the persisted fixture is present
before opening Chrome:

```bash
set -a
source /private/tmp/task-12020-39/runtime.env
set +a
curl -sf -X POST http://127.0.0.1:18242/api/v1/auth/login \
  -H 'Content-Type: application/x-www-form-urlencoded' \
  --data-urlencode 'username=member' \
  --data-urlencode "password=${TASK_12020_39_MEMBER_PASSWORD}" \
  > /private/tmp/task-12020-39/member-login-patched.json
jq -e '.access_token | type == "string" and length > 0' \
  /private/tmp/task-12020-39/member-login-patched.json
TOKEN="$(jq -r '.access_token' /private/tmp/task-12020-39/member-login-patched.json)"
curl -sf http://127.0.0.1:18242/api/v1/sharing/shared-with-me \
  -H "Authorization: Bearer ${TOKEN}" \
  | jq -e '.items[] | select(.workspace_id == "task-12020-39-owner-workspace")'
unset TOKEN
```

The final `jq` is the fixture gate. If it fails, stop and restore the inherited
TASK-12020.39 fixture through its supported API setup rather than silently
running against an empty list.

- [x] **Step 5: Rerun the real Google Chrome CDP acceptance flow**

Refresh the recipient login artifact through the real auth API, then run:

```bash
node /private/tmp/task-12020-39/shared-with-me-cdp.mjs
```

Expected:

- canonical shared-with-me envelope and populated row assertions pass;
- exact `/research-workspace?shared=1` handoff passes;
- shared read-only metadata banner renders;
- `notesSearchReturned2xx` and `noUnexpectedHttpErrors` are true;
- the known initial local workspace context 404 remains assigned to
  TASK-12020.40 and is not misreported as fixed here.

- [x] **Step 6: Run Bandit and final diff checks**

Run:

```bash
source .venv/bin/activate
python -m bandit -r \
  tldw_Server_API/app/core/DB_Management/Users_DB.py \
  tldw_Server_API/app/core/AuthNZ/repos/users_repo.py \
  tldw_Server_API/app/core/AuthNZ/migrations.py \
  -f json -o /tmp/bandit_TASK_12020_43.json
git diff --check
```

Expected: no new Bandit findings and no whitespace errors.

- [x] **Step 7: Request independent code review**

Review the final diff for correctness, regression risk, and missing tests. Fix
all actionable findings and rerun affected checks.

The independent review identified three actionable issues: SQLite write-time
schema repair, conflated pool/schema initialization state, and a PostgreSQL spy
that called real DDL. All three were corrected and covered by focused tests.
Two attempted follow-up reviewer agents did not return before shutdown; their
silence was not treated as approval.

- [x] **Step 8: Complete the backlog record and stage the task-only change set**

Record exact tests, live evidence, Bandit results, and known residuals in
TASK-12020.43. Complete the task only after every acceptance criterion is
satisfied.

Stage only TASK-12020.43 implementation files and commit:

```bash
git add \
  tldw_Server_API/Databases/SQLite/Schema/sqlite_users.sql \
  tldw_Server_API/app/core/AuthNZ/migrations.py \
  tldw_Server_API/app/core/DB_Management/Users_DB.py \
  tldw_Server_API/app/core/AuthNZ/repos/users_repo.py \
  tldw_Server_API/tests/AuthNZ/unit/test_users_repo_request_path_initialization.py \
  tldw_Server_API/tests/AuthNZ/integration/test_authnz_users_repo_postgres.py \
  tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_users_repo_sqlite.py \
  tldw_Server_API/tests/AuthNZ/unit/test_rbac_seed_helper.py \
  Docs/Plans/IMPLEMENTATION_PLAN_authnz_request_time_schema_DDL_TASK_12020_43.md \
  "backlog/tasks/task-12020.43 - Stop-request-time-UsersDB-schema-DDL-during-authenticated-repository-access.md"
git commit -m "fix(authnz): remove request-time users schema DDL"
```
