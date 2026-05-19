# SQLite Backend Log Noise Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce repeated SQLite backend success-log noise by demoting low-value factory `INFO` logs to `DEBUG`, consolidating high-signal owner logs on infrequent initialization paths, and preventing hot per-request wrappers from becoming the new noise source.

**Architecture:** Keep backend lifetime behavior unchanged and treat this as a logging-contract cleanup. The shared factory becomes SQLite-diagnostic-only, while meaningful `INFO` logs stay on cache-miss, startup, and long-lived owner boundaries such as media DB factory initialization, auth DB initialization, and scheduler DB startup. Hot constructors like collections/watchlists `for_user(...)` remain `DEBUG`-or-silent on successful construction.

**Tech Stack:** Python, FastAPI dependency helpers, Loguru, pytest, monkeypatch, SQLite temp databases

---

## Stage Overview

## Stage 1: Prepare Isolated Execution Context
**Goal**: Start implementation from a dedicated worktree/branch and confirm the logging-related test surface is local and reproducible.
**Success Criteria**: The implementer is working in an isolated `codex/` branch, the virtualenv is active, and the current targeted tests run from a clean baseline.
**Tests**: `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/DB_Management/test_database_backends.py::TestDatabaseBackends::test_backend_factory_sqlite -q`
**Status**: Complete

## Stage 2: Lock The Logging Contract With Red Tests
**Goal**: Add focused failing tests for the SQLite factory contract, media DB cache-miss logging, long-lived owner logs, and hot-path silence.
**Success Criteria**: New tests fail against the current code for the expected reasons: SQLite factory success still logs at `INFO`, media cache-miss logs are too chatty, and scheduler logging duplicates its success line.
**Tests**: `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/DB_Management/test_sqlite_factory_logging.py tldw_Server_API/tests/DB_Management/test_media_db_request_scope_isolation.py tldw_Server_API/tests/DB_Management/test_sqlite_owner_logging.py -q`
**Status**: Complete

## Stage 3: Implement Factory And Owner Logging Cleanup
**Goal**: Update the factory and owner modules to satisfy the approved logging contract without changing backend reuse semantics.
**Success Criteria**: SQLite success logging in the factory is `DEBUG`-only, media cache-miss logs collapse to one meaningful `INFO` line, `UserDatabase` and `WorkflowsSchedulerDB` each emit one contextual `INFO` line, and hot constructors stay silent on success.
**Tests**: The Stage 2 pytest command plus targeted existing regressions for scheduler DB paths.
**Status**: Complete

## Stage 4: Verify, Security-Check, And Record Results
**Goal**: Re-run the targeted regression set, run Bandit on the touched scope, and update this plan with the final execution status.
**Success Criteria**: Targeted pytest commands pass, Bandit reports no new findings on touched files, and the plan file reflects actual completion state.
**Tests**: `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/DB_Management/test_sqlite_factory_logging.py tldw_Server_API/tests/DB_Management/test_media_db_request_scope_isolation.py tldw_Server_API/tests/DB_Management/test_sqlite_owner_logging.py tldw_Server_API/tests/DB_Management/test_workflows_scheduler_db_paths.py -q`; `source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/DB_Management/backends/factory.py tldw_Server_API/app/api/v1/API_Deps/DB_Deps.py tldw_Server_API/app/core/DB_Management/UserDatabase_v2.py tldw_Server_API/app/core/DB_Management/Workflows_Scheduler_DB.py -f json -o /tmp/bandit_sqlite_backend_log_noise.json`
**Status**: Complete

## File Map

- `tldw_Server_API/app/core/DB_Management/backends/factory.py`
  Responsibility: keep SQLite success logs out of `INFO`, preserve non-SQLite behavior, and centralize any safe SQLite target formatting helper needed by the factory.
- `tldw_Server_API/app/api/v1/API_Deps/DB_Deps.py`
  Responsibility: collapse the current multi-line cache-miss `MediaDbFactory` success logging into one contextual `INFO` line while keeping cache hits and concurrent reuse at `DEBUG`.
- `tldw_Server_API/app/core/DB_Management/UserDatabase_v2.py`
  Responsibility: make the long-lived auth DB success log contextual enough to stand on its own once the factory stops emitting SQLite `INFO`.
- `tldw_Server_API/app/core/DB_Management/Workflows_Scheduler_DB.py`
  Responsibility: keep one meaningful scheduler SQLite init log and remove the current duplicate success log emitted during schema setup.
- `tldw_Server_API/tests/DB_Management/test_sqlite_factory_logging.py`
  Responsibility: focused low-level regression tests for SQLite-vs-PostgreSQL factory success logging.
- `tldw_Server_API/tests/DB_Management/test_media_db_request_scope_isolation.py`
  Responsibility: assert media DB factory cache-miss logging is high-signal and does not repeat on cache hits.
- `tldw_Server_API/tests/DB_Management/test_sqlite_owner_logging.py`
  Responsibility: focused owner-level logging tests for `UserDatabase`, `WorkflowsSchedulerDB`, and hot-path silence for `CollectionsDatabase.for_user(...)`.
- `Docs/superpowers/plans/2026-04-07-sqlite-backend-log-noise-implementation-plan.md`
  Responsibility: execution checklist and status tracking; update only to reflect real progress/results.

### Task 1: Prepare The Execution Branch And Baseline

**Files:**
- Modify: `Docs/superpowers/plans/2026-04-07-sqlite-backend-log-noise-implementation-plan.md`

- [x] **Step 1: Create or switch to an isolated worktree**

```bash
git worktree add ../tldw_server2-sqlite-log-noise -b codex/sqlite-backend-log-noise
```

Expected: a new worktree exists on branch `codex/sqlite-backend-log-noise`.

- [x] **Step 2: Activate the project virtualenv in the worktree**

```bash
source .venv/bin/activate
```

Expected: subsequent `python -m pytest` and `python -m bandit` commands use the repo venv required by `AGENTS.md`.

- [x] **Step 3: Run one existing backend smoke test from the clean baseline**

Run: `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/DB_Management/test_database_backends.py::TestDatabaseBackends::test_backend_factory_sqlite -q`

Expected: PASS. This confirms the backend abstraction test surface is runnable before adding any new assertions.

### Task 2: Add And Land Factory Logging Regression Coverage

**Files:**
- Create: `tldw_Server_API/tests/DB_Management/test_sqlite_factory_logging.py`
- Modify: `tldw_Server_API/app/core/DB_Management/backends/factory.py`

- [x] **Step 1: Write the failing factory logging tests**

```python
class _RecordingLogger:
    def __init__(self) -> None:
        self.info_calls: list[str] = []
        self.debug_calls: list[str] = []

    def info(self, message, *args, **kwargs) -> None:
        self.info_calls.append(message.format(*args))

    def debug(self, message, *args, **kwargs) -> None:
        self.debug_calls.append(message.format(*args))


def test_create_backend_logs_sqlite_success_at_debug_not_info(tmp_path, monkeypatch):
    recorder = _RecordingLogger()
    monkeypatch.setattr(factory_mod, "logger", recorder, raising=True)
    cfg = DatabaseConfig(backend_type=BackendType.SQLITE, sqlite_path=str(tmp_path / "db.sqlite"))

    backend = factory_mod.DatabaseBackendFactory.create_backend(cfg)

    assert backend.backend_type == BackendType.SQLITE
    assert recorder.info_calls == []
    assert any("sqlite" in msg for msg in recorder.debug_calls)
```

Also add a PostgreSQL parity test by monkeypatching `_BACKEND_REGISTRY[BackendType.POSTGRESQL]` to a fake backend class and asserting non-SQLite success logging still uses `INFO`.

- [x] **Step 2: Run the new factory test file and confirm it fails**

Run: `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/DB_Management/test_sqlite_factory_logging.py -q`

Expected: FAIL because [factory.py](/Users/macbook-dev/Documents/GitHub/tldw_server2/tldw_Server_API/app/core/DB_Management/backends/factory.py) still routes SQLite success through `logger.info(...)`.

- [x] **Step 3: Implement the minimal factory logging change**

```python
from pathlib import Path


def _describe_sqlite_target(config: DatabaseConfig) -> str:
    raw_path = (config.sqlite_path or "").strip()
    if raw_path == ":memory:":
        return ":memory:"
    if raw_path.lower().startswith("file:"):
        return raw_path
    return str(Path(raw_path).resolve()) if raw_path else "<default>"


if backend_type == BackendType.SQLITE:
    logger.debug("Creating sqlite backend for {}", _describe_sqlite_target(config))
else:
    logger.info("Creating {} backend", backend_type.value)
```

Keep unsupported backend errors and registry behavior unchanged.

- [x] **Step 4: Re-run the factory tests and confirm they pass**

Run: `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/DB_Management/test_sqlite_factory_logging.py -q`

Expected: PASS. SQLite success is now `DEBUG`-only and non-SQLite success remains visible at `INFO`.

- [x] **Step 5: Commit the factory slice**

```bash
git add tldw_Server_API/tests/DB_Management/test_sqlite_factory_logging.py \
        tldw_Server_API/app/core/DB_Management/backends/factory.py
git commit -m "refactor: quiet sqlite backend factory logs"
```

### Task 3: Add And Land Media Cache-Miss Logging Tests

**Files:**
- Modify: `tldw_Server_API/tests/DB_Management/test_media_db_request_scope_isolation.py`
- Modify: `tldw_Server_API/app/api/v1/API_Deps/DB_Deps.py`

- [x] **Step 1: Write the failing media cache-miss logging test**

```python
def test_get_or_create_media_db_factory_logs_one_info_on_cache_miss(monkeypatch) -> None:
    recorder = _RecordingLogger()
    monkeypatch.setattr(deps, "logger", recorder, raising=True)
    monkeypatch.setattr(deps, "_media_db_factories", {}, raising=False)
    monkeypatch.setattr(deps, "_get_db_path_for_user", lambda user_id: Path(f"/tmp/{user_id}.db"), raising=True)
    monkeypatch.setattr(deps, "get_content_backend_instance", lambda: None, raising=True)

    class FakeFactory:
        pass

    monkeypatch.setattr(
        deps.MediaDbFactory,
        "for_sqlite_path",
        classmethod(lambda cls, db_path, client_id: FakeFactory()),
        raising=True,
    )

    first = deps._get_or_create_media_db_factory(_make_user())
    second = deps._get_or_create_media_db_factory(_make_user())

    assert first is second
    assert len(recorder.info_calls) == 1
    assert "MediaDbFactory" in recorder.info_calls[0]
    assert "/tmp/1.db" in recorder.info_calls[0]
```

This test should intentionally reject the current three-line cache-miss `INFO` pattern.

- [x] **Step 2: Run the targeted media-factory test and confirm it fails**

Run: `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/DB_Management/test_media_db_request_scope_isolation.py::test_get_or_create_media_db_factory_logs_one_info_on_cache_miss -q`

Expected: FAIL because `DB_Deps.py` currently emits multiple `INFO` lines for one cache miss.

- [x] **Step 3: Implement the minimal cache-miss log consolidation**

```python
if factory:
    logger.debug("Using cached MediaDbFactory for user_id: {}", user_id)
    return factory

# after resolving db_path / backend mode
logger.info(
    "Initializing MediaDbFactory user_id={} backend={} target={}",
    user_id,
    "sqlite" if not use_shared_backend else "postgresql",
    db_path,
)
```

Remove the redundant preamble/success `INFO` lines, keep cache hits and concurrent reuse at `DEBUG`, and preserve test-mode warning logs as-is.

- [x] **Step 4: Re-run the targeted media-factory test and the existing cache test**

Run:
- `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/DB_Management/test_media_db_request_scope_isolation.py::test_get_or_create_media_db_factory_logs_one_info_on_cache_miss -q`
- `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/DB_Management/test_media_db_request_scope_isolation.py::test_get_or_create_media_db_factory_caches_factory_per_user -q`

Expected: PASS. Cache miss emits one contextual `INFO` line; cache hit behavior still reuses one factory.

- [x] **Step 5: Commit the media cache-miss slice**

```bash
git add tldw_Server_API/tests/DB_Management/test_media_db_request_scope_isolation.py \
        tldw_Server_API/app/api/v1/API_Deps/DB_Deps.py
git commit -m "refactor: consolidate media db init logs"
```

### Task 4: Add And Land Owner Logging Contract Tests

**Files:**
- Create: `tldw_Server_API/tests/DB_Management/test_sqlite_owner_logging.py`
- Modify: `tldw_Server_API/app/core/DB_Management/UserDatabase_v2.py`
- Modify: `tldw_Server_API/app/core/DB_Management/Workflows_Scheduler_DB.py`

- [x] **Step 1: Write the failing owner logging tests**

```python
def test_user_database_logs_one_contextual_sqlite_init_line(tmp_path, monkeypatch):
    recorder = _RecordingLogger()
    monkeypatch.setattr(user_db_mod, "logger", recorder, raising=True)
    monkeypatch.setattr(user_db_mod.UserDatabase, "_initialize_schema", lambda self: None, raising=True)

    cfg = DatabaseConfig(backend_type=BackendType.SQLITE, sqlite_path=str(tmp_path / "users.db"))
    user_db_mod.UserDatabase(config=cfg, client_id="test-suite")

    assert len(recorder.info_calls) == 1
    assert "UserDatabase" in recorder.info_calls[0]
    assert "users.db" in recorder.info_calls[0]


def test_collections_for_user_success_path_emits_no_info(monkeypatch):
    recorder = _RecordingLogger()
    monkeypatch.setattr(collections_mod, "logger", recorder, raising=True)
    monkeypatch.setattr(collections_mod.CollectionsDatabase, "_resolve_backend", lambda self: object(), raising=True)
    monkeypatch.setattr(collections_mod.CollectionsDatabase, "ensure_schema", lambda self: None, raising=True)
    monkeypatch.setattr(collections_mod.CollectionsDatabase, "_seed_watchlists_output_templates", lambda self: None, raising=True)

    collections_mod.CollectionsDatabase.for_user(123)

    assert recorder.info_calls == []
```

Also add a real `WorkflowsSchedulerDB` constructor test using a temp per-user base dir so the duplicate-path behavior is exercised end-to-end:

```python
def test_workflows_scheduler_logs_one_sqlite_init_line(monkeypatch, tmp_path):
    recorder = _RecordingLogger()
    monkeypatch.setattr(workflows_mod, "logger", recorder, raising=True)
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "user_databases"))
    monkeypatch.delenv("WORKFLOWS_SCHEDULER_DATABASE_URL", raising=False)
    monkeypatch.delenv("WORKFLOWS_SCHEDULER_SQLITE_PATH", raising=False)

    db = workflows_mod.WorkflowsSchedulerDB(user_id=1)
    try:
        assert len(recorder.info_calls) == 1
        assert "WorkflowsSchedulerDB" in recorder.info_calls[0]
    finally:
        db.backend.get_pool().close_all()
```

- [x] **Step 2: Run the owner logging test file and confirm it fails**

Run: `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/DB_Management/test_sqlite_owner_logging.py -q`

Expected: FAIL because `UserDatabase` lacks target context and `WorkflowsSchedulerDB` currently emits duplicate SQLite-path `INFO` logs.

- [x] **Step 3: Implement the minimal owner logging cleanup**

```python
def _describe_backend_target(self) -> str:
    cfg = getattr(self.backend, "config", None)
    if self.backend.backend_type == BackendType.SQLITE:
        return getattr(cfg, "sqlite_path", "") or "<sqlite-default>"
    return getattr(cfg, "connection_string", None) or getattr(cfg, "pg_database", None) or "<postgres>"


logger.info(
    "UserDatabase initialized backend={} target={} client_id={}",
    self.backend.backend_type.value,
    self._describe_backend_target(),
    client_id,
)
```

For `WorkflowsSchedulerDB`, keep one contextual constructor-time `INFO` line for the effective SQLite path and remove the duplicate `INFO` emission from `_ensure_schema()`. Do not add any new success `INFO` log to `CollectionsDatabase`.

- [x] **Step 4: Re-run owner logging tests and the existing scheduler path regression**

Run:
- `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/DB_Management/test_sqlite_owner_logging.py -q`
- `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/DB_Management/test_workflows_scheduler_db_paths.py -q`

Expected: PASS. `UserDatabase` and `WorkflowsSchedulerDB` each emit one contextual success line; `CollectionsDatabase.for_user(...)` stays silent on success.

- [x] **Step 5: Commit the owner logging slice**

```bash
git add tldw_Server_API/tests/DB_Management/test_sqlite_owner_logging.py \
        tldw_Server_API/app/core/DB_Management/UserDatabase_v2.py \
        tldw_Server_API/app/core/DB_Management/Workflows_Scheduler_DB.py
git commit -m "refactor: tighten db owner initialization logs"
```

### Task 5: Run Full Verification, Security Check, And Update This Plan

**Files:**
- Modify: `Docs/superpowers/plans/2026-04-07-sqlite-backend-log-noise-implementation-plan.md`

- [x] **Step 1: Run the focused regression suite for the touched contract**

Run:
- `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/DB_Management/test_sqlite_factory_logging.py -q`
- `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/DB_Management/test_media_db_request_scope_isolation.py -q`
- `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/DB_Management/test_sqlite_owner_logging.py -q`
- `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/DB_Management/test_workflows_scheduler_db_paths.py -q`

Expected: PASS for all four commands.

- [x] **Step 2: Run Bandit on the touched implementation files**

Run: `source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/DB_Management/backends/factory.py tldw_Server_API/app/api/v1/API_Deps/DB_Deps.py tldw_Server_API/app/core/DB_Management/UserDatabase_v2.py tldw_Server_API/app/core/DB_Management/Workflows_Scheduler_DB.py -f json -o /tmp/bandit_sqlite_backend_log_noise.json`

Expected: Bandit completes successfully and reports no new findings in the touched scope.

- [x] **Step 3: Update stage statuses and execution notes in this plan**

```markdown
**Status**: Complete
**Execution Notes**: Added focused logging contract tests, demoted SQLite factory success logs to DEBUG, consolidated media cache-miss info logs, and removed duplicate scheduler success lines.
```

Expected: the plan file reflects real completion state rather than remaining as all unchecked defaults.

## Execution Notes

**Status**: Complete
**Worktree**: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/sqlite-backend-log-noise`
**Branch**: `codex/sqlite-backend-log-noise`

**Commits**:
- `dceb86acb` `refactor: quiet sqlite backend factory logs`
- `2365b8366` `refactor: consolidate media db init logs`
- `f6153eccb` `fix: clarify media db cache-miss postgres target log`
- `fef27f57d` `refactor: tighten db owner initialization logs`
- `2979d3f70` `fix: tighten db owner log targets`

**Verification**:
- `python -m pytest -q tldw_Server_API/tests/DB_Management/test_sqlite_factory_logging.py -q` -> `2 passed`
- `python -m pytest -q tldw_Server_API/tests/DB_Management/test_media_db_request_scope_isolation.py -q` -> `16 passed`
- `python -m pytest -q tldw_Server_API/tests/DB_Management/test_sqlite_owner_logging.py -q` -> `6 passed`
- `python -m pytest -q tldw_Server_API/tests/DB_Management/test_workflows_scheduler_db_paths.py -q` -> `1 passed`
- `python -m bandit -r tldw_Server_API/app/core/DB_Management/backends/factory.py tldw_Server_API/app/api/v1/API_Deps/DB_Deps.py tldw_Server_API/app/core/DB_Management/UserDatabase_v2.py tldw_Server_API/app/core/DB_Management/Workflows_Scheduler_DB.py -f json -o /tmp/bandit_sqlite_backend_log_noise.json` -> `results 0`, `errors 0`

**Review Notes**:
- Task 3 required a fix-and-rereview loop so shared PostgreSQL media-factory logging no longer reported `target=:memory:` and the regression test pinned the exact log line.
- Task 4 required a follow-up fix after spec review so owner logs do not expose PostgreSQL credentials and scheduler success logging only emits after successful initialization.
