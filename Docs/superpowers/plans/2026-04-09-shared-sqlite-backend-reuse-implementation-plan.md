# Shared SQLite Backend Reuse Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce canonical shared SQLite backend reuse for safe file-backed paths, centralize shutdown/reset semantics, preserve wrapper-local cleanup, and eliminate duplicate backend construction without breaking request, test, or helper isolation.

**Architecture:** Keep PostgreSQL behavior unchanged and treat the factory as the single owner of managed shared SQLite pools. The first implementation slice focuses on filesystem-backed SQLite targets plus the media request path and proven-safe helper wrappers; raw `:memory:` stays unshared, and wrappers with per-instance connection-state assumptions must pass compatibility tests before joining shared reuse. Reset behavior is explicit: hard reset for tests/shutdown, graceful eviction for runtime reconfiguration.

**Tech Stack:** Python, FastAPI dependency helpers, sqlite3, Loguru, pytest, monkeypatch, dataclasses, threading

---

## Stage Overview

## Stage 1: Prepare Isolated Execution Context
**Goal**: Start from an isolated worktree/branch and confirm the shared-backend test surface is reproducible.
**Success Criteria**: Work happens in a dedicated `codex/` branch, the repo virtualenv is active, and the baseline DB backend tests run before any new assertions are added.
**Tests**: `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/DB_Management/test_database_backends.py::TestDatabaseBackends::test_backend_factory_sqlite -q`
**Status**: Complete

## Stage 2: Lock The Factory Reuse Contract With Red Tests
**Goal**: Add focused failing tests for canonical SQLite identity, config snapshotting, named-cache integration, raw-memory exclusion, and hard/graceful reset behavior.
**Success Criteria**: New tests fail against the current code because `create_backend()` still returns fresh SQLite backends, named and direct caches are independent, mutable configs are held by reference, and no reset API exists for managed shared SQLite pools.
**Tests**: `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/DB_Management/test_sqlite_shared_backend_registry.py -q`
**Status**: Complete

## Stage 3: Implement Factory Registry And Reset Modes
**Goal**: Add the shared SQLite registry in the factory, make named caches converge on canonical shared backends, and wire hard/graceful reset entry points.
**Success Criteria**: File-backed SQLite callers get one canonical backend per normalized target/policy, raw `:memory:` remains unshared, named `get_backend(...)` references point at canonical shared backends, and reset helpers support both hard and graceful semantics.
**Tests**: Stage 2 test command plus targeted existing backend tests.
**Status**: Complete

## Stage 4: Migrate First-Wave Wrapper Ownership Safely
**Goal**: Update media runtime and direct helper cleanup paths so they preserve local cleanup while stopping direct shutdown of factory-managed shared SQLite pools.
**Success Criteria**: `MediaDbSession` and `MediaDbFactory` preserve request-local cleanup, `CollectionsDatabase.close()` no longer kills a shared pool, and `CharactersRAGDB` preserves rollback/checkpoint/thread-local cleanup while either proving shared-backend compatibility or explicitly staying on isolated SQLite backends for now.
**Tests**: Media/runtime, collections, and Chacha targeted pytest commands all pass.
**Status**: Complete

## Stage 5: Verify, Security-Check, And Record Results
**Goal**: Re-run the targeted regression suite, run Bandit on the touched scope, and update this plan with real execution status.
**Success Criteria**: Targeted pytest commands pass, Bandit reports no new findings in touched files, and the plan file reflects actual completion state.
**Tests**: Combined pytest command plus Bandit command on touched files.
**Status**: Complete

## Execution Notes

- Worktree: `.worktrees/shared-sqlite-backend-reuse`
- Branch: `codex/shared-sqlite-backend-reuse`
- Task 2 commits: `0df434930` (`fix: preserve sqlite uri options and synchronize pool init`) on top of `cbffc7399`
- Task 3 commit: `ce169c9b2`
- Task 4 commits: `bc3663212`, `d2f580b54`, `c0924e826`
- Final follow-up commit: `9d640fbd6` (`fix(db): target media sqlite reset eviction scope`)
- Final targeted regression verification:
  - `source ../../.venv/bin/activate && python -m pytest -q tldw_Server_API/tests/DB_Management/test_sqlite_shared_backend_registry.py tldw_Server_API/tests/DB_Management/test_sqlite_factory_logging.py tldw_Server_API/tests/DB_Management/test_database_backends.py::TestDatabaseBackends::test_backend_factory_sqlite tldw_Server_API/tests/DB_Management/test_database_backends.py::TestDatabaseBackends::test_sqlite_backend_get_pool_is_singleton_under_concurrent_first_use tldw_Server_API/tests/DB_Management/test_media_db_request_scope_isolation.py tldw_Server_API/tests/DB_Management/test_media_db_connection_cleanup.py tldw_Server_API/tests/DB_Management/test_db_manager_config_behavior.py tldw_Server_API/tests/DB_Management/test_content_backend_cache.py::test_reset_media_runtime_defaults_blocks_stale_backend_reads_during_clear tldw_Server_API/tests/Collections/test_collections_close.py tldw_Server_API/tests/DB_Management/test_chacha_shared_sqlite_cleanup.py tldw_Server_API/tests/DB_Management/test_chacha_postgres_session_scope.py tldw_Server_API/tests/RAG/test_analytics_backend.py -q`
  - Result: `69 passed, 5 warnings`
- Final Bandit verification:
  - `source ../../.venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/DB_Management/backends/factory.py tldw_Server_API/app/core/DB_Management/backends/sqlite_backend.py tldw_Server_API/app/api/v1/API_Deps/DB_Deps.py tldw_Server_API/app/core/DB_Management/media_db/runtime/session.py tldw_Server_API/app/core/DB_Management/media_db/runtime/defaults.py tldw_Server_API/app/core/DB_Management/DB_Manager.py tldw_Server_API/app/core/DB_Management/Collections_DB.py tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py -f json -o /tmp/bandit_shared_sqlite_backend_reuse_final_v2.json`
  - Result: exit `0`, JSON `results: []`
- Known pre-existing failure checked during Task 4 review:
  - `tldw_Server_API/tests/ChaChaNotesDB/test_chachanotes_db.py::TestConversationsAndMessages::test_soft_delete_conversation_and_messages`
  - Reproduced unchanged on detached pre-Task-4 commit `ce169c9b2`, so it was treated as baseline noise rather than a Task 4 regression.

## File Map

- `tldw_Server_API/app/core/DB_Management/backends/factory.py`
  Responsibility: canonical SQLite signature generation, managed shared-backend registry, config snapshotting, named-cache convergence, and hard/graceful reset helpers.
- `tldw_Server_API/app/api/v1/API_Deps/DB_Deps.py`
  Responsibility: route media DB cache reset through the factory hard-reset path and keep higher-level factory caching behavior coherent.
- `tldw_Server_API/app/core/DB_Management/media_db/runtime/session.py`
  Responsibility: preserve request-local cleanup while removing direct shutdown of factory-managed shared SQLite pools from `MediaDbSession` and `MediaDbFactory`.
- `tldw_Server_API/app/core/DB_Management/Collections_DB.py`
  Responsibility: stop direct `pool.close_all()` on factory-managed shared SQLite backends while keeping non-shared ownership behavior explicit.
- `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
  Responsibility: preserve connection-local rollback/checkpoint/thread-local cleanup and avoid direct shutdown of factory-managed shared SQLite pools; if same-thread compatibility fails, keep this wrapper on isolated SQLite backends for the first slice.
- `tldw_Server_API/app/core/DB_Management/DB_Manager.py`
  Responsibility: expose or forward the graceful runtime reset entry point for managed shared SQLite backends.
- `tldw_Server_API/tests/DB_Management/test_sqlite_shared_backend_registry.py`
  Responsibility: low-level registry contract tests for identity, config snapshotting, named-cache integration, raw-memory exclusion, and reset semantics.
- `tldw_Server_API/tests/DB_Management/test_media_db_request_scope_isolation.py`
  Responsibility: media request-path reuse and cache-reset regressions.
- `tldw_Server_API/tests/DB_Management/test_db_manager_config_behavior.py`
  Responsibility: graceful reset wiring regression for `DB_Manager.reset_content_backend(...)`.
- `tldw_Server_API/tests/Collections/test_collections_close.py`
  Responsibility: owned-vs-shared SQLite close behavior for `CollectionsDatabase`.
- `tldw_Server_API/tests/DB_Management/test_chacha_shared_sqlite_cleanup.py`
  Responsibility: `CharactersRAGDB` cleanup behavior and same-thread compatibility gate for shared SQLite backends.
- `Docs/superpowers/plans/2026-04-09-shared-sqlite-backend-reuse-implementation-plan.md`
  Responsibility: execution checklist and status tracking; update only to reflect actual progress/results.

### Task 1: Prepare The Worktree And Baseline

**Files:**
- Modify: `Docs/superpowers/plans/2026-04-09-shared-sqlite-backend-reuse-implementation-plan.md`

- [ ] **Step 1: Create or switch to an isolated worktree**

```bash
git worktree add ../tldw_server2-shared-sqlite-reuse -b codex/shared-sqlite-backend-reuse
```

Expected: a new worktree exists on branch `codex/shared-sqlite-backend-reuse`.

- [ ] **Step 2: Activate the project virtualenv**

```bash
source .venv/bin/activate
```

Expected: subsequent `python -m pytest` and `python -m bandit` commands use the repo venv required by `AGENTS.md`.

- [ ] **Step 3: Run one existing backend smoke test from the clean baseline**

Run: `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/DB_Management/test_database_backends.py::TestDatabaseBackends::test_backend_factory_sqlite -q`

Expected: PASS. This confirms the baseline DB backend test surface is runnable before new assertions are added.

### Task 2: Add And Land Factory Registry Contract Tests

**Files:**
- Create: `tldw_Server_API/tests/DB_Management/test_sqlite_shared_backend_registry.py`
- Modify: `tldw_Server_API/app/core/DB_Management/backends/factory.py`

- [ ] **Step 1: Write the failing shared-registry tests**

```python
from __future__ import annotations

from dataclasses import replace
import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, DatabaseConfig
from tldw_Server_API.app.core.DB_Management.backends import factory as factory_mod


@pytest.fixture(autouse=True)
def _reset_backend_caches():
    factory_mod.close_all_backends()
    yield
    factory_mod.close_all_backends()


def test_create_backend_reuses_file_backed_sqlite_by_normalized_path(tmp_path):
    db_path = tmp_path / "shared.db"
    cfg_a = DatabaseConfig(backend_type=BackendType.SQLITE, sqlite_path=str(db_path))
    cfg_b = DatabaseConfig(backend_type=BackendType.SQLITE, sqlite_path=str(db_path.resolve()))

    first = factory_mod.DatabaseBackendFactory.create_backend(cfg_a)
    second = factory_mod.DatabaseBackendFactory.create_backend(cfg_b)

    assert first is second


def test_create_backend_does_not_share_raw_memory_sqlite():
    cfg = DatabaseConfig(backend_type=BackendType.SQLITE, sqlite_path=":memory:")

    first = factory_mod.DatabaseBackendFactory.create_backend(cfg)
    second = factory_mod.DatabaseBackendFactory.create_backend(cfg)

    assert first is not second


def test_create_backend_does_not_share_anonymous_memory_uri():
    cfg = DatabaseConfig(
        backend_type=BackendType.SQLITE,
        sqlite_path="file::memory:?cache=shared",
    )

    first = factory_mod.DatabaseBackendFactory.create_backend(cfg)
    second = factory_mod.DatabaseBackendFactory.create_backend(cfg)

    assert first is not second


def test_create_backend_reuses_equivalent_file_uri_and_path(tmp_path):
    db_path = (tmp_path / "uri.db").resolve()
    path_cfg = DatabaseConfig(backend_type=BackendType.SQLITE, sqlite_path=str(db_path))
    uri_cfg = DatabaseConfig(backend_type=BackendType.SQLITE, sqlite_path=f"file:{db_path}")

    first = factory_mod.DatabaseBackendFactory.create_backend(path_cfg)
    second = factory_mod.DatabaseBackendFactory.create_backend(uri_cfg)

    assert first is second


def test_create_backend_reuses_named_shared_cache_memory_uri_by_exact_identity():
    uri = "file:sharedmem?mode=memory&cache=shared"
    cfg_a = DatabaseConfig(backend_type=BackendType.SQLITE, sqlite_path=uri)
    cfg_b = DatabaseConfig(backend_type=BackendType.SQLITE, sqlite_path=uri)

    first = factory_mod.DatabaseBackendFactory.create_backend(cfg_a)
    second = factory_mod.DatabaseBackendFactory.create_backend(cfg_b)

    assert first is second


def test_named_cache_and_direct_create_share_same_sqlite_backend(tmp_path):
    cfg = DatabaseConfig(backend_type=BackendType.SQLITE, sqlite_path=str(tmp_path / "named.db"))

    direct = factory_mod.DatabaseBackendFactory.create_backend(cfg)
    named = factory_mod.get_backend("shared-test", replace(cfg))

    assert named is direct


def test_shared_backend_snapshots_effective_sqlite_config(tmp_path):
    cfg = DatabaseConfig(backend_type=BackendType.SQLITE, sqlite_path=str(tmp_path / "snap.db"))
    backend = factory_mod.DatabaseBackendFactory.create_backend(cfg)
    cfg.sqlite_path = str(tmp_path / "mutated.db")

    assert backend.config.sqlite_path == str((tmp_path / "snap.db"))


def test_failed_sqlite_backend_creation_does_not_poison_registry(tmp_path, monkeypatch):
    cfg = DatabaseConfig(backend_type=BackendType.SQLITE, sqlite_path=str(tmp_path / "broken.db"))
    calls = {"count": 0}

    class _FlakySQLiteBackend:
        def __init__(self, config):
            calls["count"] += 1
            if calls["count"] == 1:
                raise RuntimeError("boom")
            self.config = config
            self.backend_type = BackendType.SQLITE

    monkeypatch.setitem(factory_mod._BACKEND_REGISTRY, BackendType.SQLITE, _FlakySQLiteBackend)

    with pytest.raises(RuntimeError, match="boom"):
        factory_mod.DatabaseBackendFactory.create_backend(cfg)

    backend = factory_mod.DatabaseBackendFactory.create_backend(cfg)

    assert backend.backend_type == BackendType.SQLITE
    assert calls["count"] == 2
```

Also add reset tests that assert:
- hard reset closes the canonical shared backend exactly once and clears named references
- graceful reset evicts registry references first and defers pool close
- `get_backend(name, ...)` and direct `create_backend(...)` still converge on the same canonical backend after reset
- `close_all_backends()` routes through the hard-reset path so named-cache shutdown and canonical-registry shutdown stay in sync

- [ ] **Step 2: Run the new factory registry test file and verify it fails**

Run: `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/DB_Management/test_sqlite_shared_backend_registry.py -q`

Expected: FAIL because the current factory still returns fresh SQLite backends, holds caller-owned config by reference, and exposes no shared-registry reset contract.

- [ ] **Step 3: Implement the minimal shared-registry behavior in the factory**

```python
from dataclasses import replace
from threading import RLock, Thread
import time

_sqlite_backend_registry: dict[tuple, DatabaseBackend] = {}
_sqlite_registry_lock = RLock()
_SQLITE_EVICTION_GRACE_SECONDS = 5.0


def _snapshot_sqlite_config(config: DatabaseConfig) -> DatabaseConfig:
    return replace(config)


def _sqlite_signature(config: DatabaseConfig) -> tuple | None:
    raw_path = (config.sqlite_path or "").strip()
    if raw_path == ":memory:":
        return None
    return (
        BackendType.SQLITE,
        _normalize_sqlite_target(raw_path),
        bool(config.sqlite_wal_mode),
        bool(config.sqlite_foreign_keys),
    )


def _get_or_create_shared_sqlite_backend(config: DatabaseConfig) -> DatabaseBackend:
    signature = _sqlite_signature(config)
    backend_class = _BACKEND_REGISTRY[BackendType.SQLITE]
    if signature is None:
        return backend_class(_snapshot_sqlite_config(config))
    with _sqlite_registry_lock:
        existing = _sqlite_backend_registry.get(signature)
        if existing is not None:
            return existing
        backend = backend_class(_snapshot_sqlite_config(config))
        _sqlite_backend_registry[signature] = backend
        return backend
```

Also update `get_backend(...)` so named cache entries point at canonical shared SQLite backends and add a factory reset helper with `mode="hard"` and `mode="graceful"` semantics.

- [ ] **Step 4: Re-run the new registry tests and the existing backend smoke tests**

Run:
- `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/DB_Management/test_sqlite_shared_backend_registry.py -q`
- `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/DB_Management/test_database_backends.py::TestDatabaseBackends::test_backend_factory_sqlite -q`
- `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/DB_Management/test_sqlite_factory_logging.py -q`

Expected: PASS. File-backed SQLite backends are reused canonically, raw `:memory:` stays isolated, named and direct caches converge, and existing factory logging behavior only fires on actual creation.

- [ ] **Step 5: Commit the factory-registry slice**

```bash
git add tldw_Server_API/tests/DB_Management/test_sqlite_shared_backend_registry.py \
        tldw_Server_API/app/core/DB_Management/backends/factory.py
git commit -m "refactor: add shared sqlite backend registry"
```

### Task 3: Wire Reset Entry Points And Media Runtime Cleanup

**Files:**
- Modify: `tldw_Server_API/app/api/v1/API_Deps/DB_Deps.py`
- Modify: `tldw_Server_API/app/core/DB_Management/media_db/runtime/session.py`
- Modify: `tldw_Server_API/app/core/DB_Management/DB_Manager.py`
- Modify: `tldw_Server_API/tests/DB_Management/test_media_db_request_scope_isolation.py`
- Modify: `tldw_Server_API/tests/DB_Management/test_db_manager_config_behavior.py`

- [ ] **Step 1: Write the failing media/runtime cleanup and reset tests**

```python
def test_media_db_factory_close_does_not_close_factory_managed_shared_pool(monkeypatch):
    closed = []

    class _Pool:
        def close_all(self):
            closed.append("closed")

    class _Backend:
        backend_type = BackendType.SQLITE
        def get_pool(self):
            return _Pool()

    factory = media_db_session.MediaDbFactory(
        db_path="/tmp/shared.db",
        client_id="1",
        backend=_Backend(),
    )

    monkeypatch.setattr(media_db_session, "release_managed_backend", lambda backend: None, raising=False)

    factory.close()

    assert closed == []


def test_reset_media_db_cache_hard_resets_factory_managed_sqlite_registry(monkeypatch):
    calls = []
    monkeypatch.setattr(deps, "reset_sqlite_backend_registry", lambda mode="hard": calls.append(mode), raising=False)
    monkeypatch.setattr(deps, "_media_db_factories", {}, raising=True)
    monkeypatch.setattr(deps, "_user_db_instances", {}, raising=True)

    deps.reset_media_db_cache()

    assert calls == ["hard"]
```

Also add a test that `MediaDbSession.release_context_connection()` still calls wrapper-local `release_context_connection()` or `close_connection()` while avoiding direct shared-pool shutdown.
Also add a `test_db_manager_config_behavior.py` regression that `DB_Manager.reset_content_backend(...)` routes SQLite runtime reconfiguration through the graceful reset mode instead of collapsing into a hard reset path.
Also add a PostgreSQL regression proving `MediaDbFactory.close()` does not attempt SQLite release behavior when the cached backend is PostgreSQL-backed.

- [ ] **Step 2: Run the targeted media/runtime tests and confirm they fail**

Run:
- `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/DB_Management/test_media_db_request_scope_isolation.py -q`
- `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/DB_Management/test_db_manager_config_behavior.py -q`

Expected: FAIL because `MediaDbFactory.close()` still closes the backend pool directly, `MediaDbSession.release_context_connection()` still mixes local cleanup with pool shutdown, `reset_media_db_cache()` does not call a factory reset helper, and `DB_Manager.reset_content_backend(...)` does not yet route runtime SQLite reconfiguration through a graceful reset path.

- [ ] **Step 3: Implement the minimal reset wiring and media cleanup split**

```python
def reset_media_db_cache() -> None:
    with _user_db_lock:
        ...
        reset_sqlite_backend_registry(mode="hard")


class MediaDbSession:
    def release_context_connection(self) -> None:
        release = getattr(self.database, "release_context_connection", None)
        if callable(release):
            release()
        close_connection = getattr(self.database, "close_connection", None)
        if callable(close_connection):
            close_connection()


class MediaDbFactory:
    def close(self) -> None:
        backend = self.backend
        if backend is not None and is_factory_managed_backend(backend):
            release_managed_backend(backend)
            return
        if backend is not None:
            backend.get_pool().close_all()
```

In `DB_Manager.py`, expose the graceful runtime reset helper and document that runtime reconfiguration should use `mode="graceful"` while test/shutdown paths stay `mode="hard"`.

- [ ] **Step 4: Re-run the targeted media/runtime tests and current cache regressions**

Run:
- `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/DB_Management/test_media_db_request_scope_isolation.py -q`
- `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/DB_Management/test_media_db_connection_cleanup.py -q`
- `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/DB_Management/test_db_manager_config_behavior.py -q`

Expected: PASS. Media runtime cleanup still performs wrapper-local teardown, hard reset clears factory-managed SQLite state for tests, graceful reset remains available for runtime reconfiguration, and existing request-scope behavior remains intact.

- [ ] **Step 5: Commit the reset/media cleanup slice**

```bash
git add tldw_Server_API/app/api/v1/API_Deps/DB_Deps.py \
        tldw_Server_API/app/core/DB_Management/media_db/runtime/session.py \
        tldw_Server_API/app/core/DB_Management/DB_Manager.py \
        tldw_Server_API/tests/DB_Management/test_media_db_request_scope_isolation.py \
        tldw_Server_API/tests/DB_Management/test_db_manager_config_behavior.py
git commit -m "refactor: split sqlite backend ownership from media cleanup"
```

### Task 4: Migrate Direct Helper Ownership With A Compatibility Gate

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/Collections_DB.py`
- Modify: `tldw_Server_API/tests/Collections/test_collections_close.py`
- Create: `tldw_Server_API/tests/DB_Management/test_chacha_shared_sqlite_cleanup.py`
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`

- [ ] **Step 1: Write the failing helper ownership and compatibility tests**

```python
def test_collections_for_user_reuses_canonical_backend_and_close_keeps_shared_pool_usable(monkeypatch, tmp_path):
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path))
    db_path = tmp_path / "7" / "Media_DB_v2.db"
    canonical = factory_mod.DatabaseBackendFactory.create_backend(
        DatabaseConfig(backend_type=BackendType.SQLITE, sqlite_path=str(db_path))
    )
    db = CollectionsDatabase.for_user(user_id=7)
    pool = canonical.get_pool()
    pool.get_connection()

    assert db.backend is canonical

    db.close()
    assert pool.get_connection() is not None


def test_chacha_close_all_connections_keeps_shared_pool_usable_for_canonical_backend(tmp_path):
    shared_backend = factory_mod.DatabaseBackendFactory.create_backend(
        DatabaseConfig(backend_type=BackendType.SQLITE, sqlite_path=str(tmp_path / "chacha.db"))
    )
    db = CharactersRAGDB(db_path=str(tmp_path / "chacha.db"), client_id="7")
    pool = shared_backend.get_pool()
    pool.get_connection()

    assert db.backend is shared_backend

    db.close_all_connections()

    assert getattr(db._local, "conn", None) is None
    assert pool.get_connection() is not None
```

Also add:
- a same-thread compatibility test for `CharactersRAGDB` that proves shared reuse is safe, or explicitly fails and drives a first-slice opt-out for its backend creation path
- a cross-wrapper regression where a media path and a direct helper path point at the same SQLite file, one wrapper closes, and the other remains usable
- a short review checklist entry in the plan execution notes that remaining direct SQLite factory callers were reviewed and left unchanged only if they do not close or invalidate shared pools

- [ ] **Step 2: Run the targeted helper tests and confirm they fail**

Run:
- `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Collections/test_collections_close.py -q`
- `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/DB_Management/test_chacha_shared_sqlite_cleanup.py -q`

Expected: FAIL because `CollectionsDatabase.close()` currently closes the pool it resolved, and `CharactersRAGDB` still uses direct pool shutdown paths without distinguishing shared ownership.

- [ ] **Step 3: Implement the minimal safe helper ownership behavior**

```python
class CollectionsDatabase:
    def close(self) -> None:
        if not self._owns_backend:
            return
        if is_factory_managed_backend(self.backend):
            release_managed_backend(self.backend)
            return
        self.backend.get_pool().close_all()
```

```python
class CharactersRAGDB:
    def close_all_connections(self) -> None:
        self.close_connection()  # local/thread cleanup first
        if self.backend_type == BackendType.SQLITE and is_factory_managed_backend(self.backend):
            release_managed_backend(self.backend)
            return
        pool = self.backend.get_pool()
        pool.close_all()
```

If the same-thread compatibility test shows `CharactersRAGDB` cannot safely share a canonical SQLite backend yet, implement the minimal safe scope by keeping its cleanup compatible with factory-managed backends while leaving its backend creation path isolated for the first slice. Do not broaden reuse past the tested-safe boundary.

- [ ] **Step 4: Re-run the helper tests and one existing Chacha regression**

Run:
- `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Collections/test_collections_close.py -q`
- `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/DB_Management/test_chacha_shared_sqlite_cleanup.py -q`
- `source .venv/bin/activate && python -m pytest -q tldw_Server_API/tests/ChaChaNotesDB/test_chachanotes_db.py -q`

Expected: PASS. Collections no longer kills a shared pool, Chacha local cleanup remains intact, and any unsupported shared-backend case is intentionally kept isolated instead of silently broken.

- [ ] **Step 5: Commit the helper-ownership slice**

```bash
git add tldw_Server_API/app/core/DB_Management/Collections_DB.py \
        tldw_Server_API/tests/Collections/test_collections_close.py \
        tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py \
        tldw_Server_API/tests/DB_Management/test_chacha_shared_sqlite_cleanup.py
git commit -m "refactor: harden sqlite helper ownership cleanup"
```

### Task 5: Verify, Security-Check, And Record Results

**Files:**
- Modify: `Docs/superpowers/plans/2026-04-09-shared-sqlite-backend-reuse-implementation-plan.md`

- [ ] **Step 1: Run the targeted regression suite**

Run:

```bash
source .venv/bin/activate && python -m pytest -q \
  tldw_Server_API/tests/DB_Management/test_sqlite_shared_backend_registry.py \
  tldw_Server_API/tests/DB_Management/test_sqlite_factory_logging.py \
  tldw_Server_API/tests/DB_Management/test_database_backends.py::TestDatabaseBackends::test_backend_factory_sqlite \
  tldw_Server_API/tests/DB_Management/test_media_db_request_scope_isolation.py \
  tldw_Server_API/tests/DB_Management/test_media_db_connection_cleanup.py \
  tldw_Server_API/tests/DB_Management/test_db_manager_config_behavior.py \
  tldw_Server_API/tests/Collections/test_collections_close.py \
  tldw_Server_API/tests/DB_Management/test_chacha_shared_sqlite_cleanup.py -q
```

Expected: PASS.

- [ ] **Step 2: Run Bandit on the touched scope**

Run:

```bash
source .venv/bin/activate && python -m bandit -r \
  tldw_Server_API/app/core/DB_Management/backends/factory.py \
  tldw_Server_API/app/api/v1/API_Deps/DB_Deps.py \
  tldw_Server_API/app/core/DB_Management/media_db/runtime/session.py \
  tldw_Server_API/app/core/DB_Management/Collections_DB.py \
  tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py \
  tldw_Server_API/app/core/DB_Management/DB_Manager.py \
  -f json -o /tmp/bandit_shared_sqlite_backend_reuse.json
```

Expected: report generated with no new findings in touched code.

- [ ] **Step 3: Update this plan with actual execution status**

```markdown
## Stage 2: Lock The Factory Reuse Contract With Red Tests
**Status**: Complete
```

Expected: every completed stage/task in this file reflects reality; do not mark incomplete work as done.

- [ ] **Step 4: Commit the verification/status update**

```bash
git add Docs/superpowers/plans/2026-04-09-shared-sqlite-backend-reuse-implementation-plan.md
git commit -m "docs: record shared sqlite reuse verification"
```
