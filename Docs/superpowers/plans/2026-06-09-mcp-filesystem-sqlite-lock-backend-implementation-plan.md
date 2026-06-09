# MCP Filesystem SQLite Lock Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a package-level SQLite filesystem lock backend for MCP while keeping `tldw_server` integration to a compatibility shim, import swap, and config passthrough.

**Architecture:** Extract the existing lock lease model, protocol, exceptions, memory backend, and factory into `mcp_unified.filesystem_locks`. Add a lazy optional SQLAlchemy Core SQLite backend behind the same synchronous protocol. Keep `FilesystemModule` behavior unchanged by default and prove standalone package artifacts include the new subpackage.

**Tech Stack:** Python 3.10+, SQLAlchemy Core SQLite dialect, pytest/pytest-asyncio, Bandit, existing MCP Unified filesystem module and standalone package artifact gates.

---

## Command Note

This plan assumes execution from the dedicated worktree at `.worktrees/mcp-fs-sqlite-lock-backend`. The shared project virtualenv is at `../../.venv` from that location. If you run the plan from a different checkout, activate that checkout's equivalent project virtualenv first.

## File Map

- Create `mcp_unified/filesystem_locks/__init__.py`: lightweight public exports and lazy SQLite export handling.
- Create `mcp_unified/filesystem_locks/models.py`: `FilesystemLockLease`, `FilesystemLockManager`, `FilesystemLockConflict`, `FilesystemLockMissing`.
- Create `mcp_unified/filesystem_locks/memory.py`: moved `InMemoryFilesystemLockManager` and `create_filesystem_lock_manager()` factory.
- Create `mcp_unified/filesystem_locks/sqlite.py`: optional `SQLiteFilesystemLockManager`.
- Modify `tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_locks.py`: compatibility re-export only.
- Modify `tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_module.py`: import package-level lock API and make lock tool descriptions backend-neutral.
- Create `tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_lock_managers.py`: package-level memory/SQLite backend contract tests.
- Modify `tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py`: config and descriptor integration tests.
- Modify `tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py`: lazy import and artifact inclusion assertions.
- Modify `mcp_unified/pyproject.toml`: add explicit `mcp_unified.filesystem_locks` package and package-dir mapping.
- Modify `mcp_unified/README.md`: short feature/config note.
- Modify `mcp_unified/USER_GUIDE.md`: operator guidance for memory vs SQLite lock backend.
- Modify `backlog/tasks/task-2345 - Implement-MCP-filesystem-SQLite-lock-backend.md`: keep task status, plan, touched files, and validation current.

## Task 1: Extract Package-Level Lock Models And Memory Backend

**Files:**
- Create: `mcp_unified/filesystem_locks/__init__.py`
- Create: `mcp_unified/filesystem_locks/models.py`
- Create: `mcp_unified/filesystem_locks/memory.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_locks.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_lock_managers.py`

- [ ] **Step 1: Write failing import and memory contract tests**

Add tests that import from `mcp_unified.filesystem_locks`, import the old compatibility path, and exercise memory acquire/conflict/renew/release behavior.

```python
def test_package_and_compatibility_imports_expose_same_lock_types() -> None:
    from mcp_unified.filesystem_locks import InMemoryFilesystemLockManager
    from tldw_Server_API.app.core.MCP_unified.modules.implementations.filesystem_locks import (
        InMemoryFilesystemLockManager as CompatManager,
    )

    assert CompatManager is InMemoryFilesystemLockManager


def test_memory_lock_manager_acquire_conflict_renew_release() -> None:
    from mcp_unified.filesystem_locks import (
        FilesystemLockConflict,
        InMemoryFilesystemLockManager,
    )

    manager = InMemoryFilesystemLockManager()
    lease, renewed = manager.acquire(
        workspace_key="ws",
        path="docs/story.txt",
        owner="agent-a",
        ttl_seconds=60,
    )

    assert renewed is False
    with pytest.raises(FilesystemLockConflict):
        manager.acquire(workspace_key="ws", path="docs/story.txt", owner="agent-b", ttl_seconds=60)

    renewed_lease, renewed = manager.acquire(
        workspace_key="ws",
        path="docs/story.txt",
        owner="agent-a",
        ttl_seconds=60,
        lease_id=lease.lease_id,
    )
    assert renewed is True
    assert renewed_lease.lease_id == lease.lease_id

    released = manager.release(workspace_key="ws", path="docs/story.txt", lease_id=lease.lease_id)
    assert released is not None
```

- [ ] **Step 2: Run the failing tests**

Run:

```bash
source ../../.venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_lock_managers.py -q
```

Expected: FAIL because `mcp_unified.filesystem_locks` does not exist.

- [ ] **Step 3: Move neutral code into package files**

Move the dataclass, protocol, and exceptions into `models.py`. Move `InMemoryFilesystemLockManager` and the factory into `memory.py`. Keep the code behavior the same except factory backend parsing:

```python
raw_backend = (settings or {}).get("lock_manager_backend")
if raw_backend is None:
    return InMemoryFilesystemLockManager()
backend = str(raw_backend).strip().lower()
if backend in {"memory", "in_memory"}:
    return InMemoryFilesystemLockManager()
if backend == "":
    raise ValueError(f"unsupported filesystem lock_manager_backend: {raw_backend!r}")
...
```

Do not import SQLAlchemy anywhere in `models.py`, `memory.py`, or ordinary `__init__.py` imports.

- [ ] **Step 4: Implement explicit compatibility shim**

Replace the host `filesystem_locks.py` body with explicit imports:

```python
"""Compatibility exports for MCP filesystem lock leases."""

from __future__ import annotations

from mcp_unified.filesystem_locks import (
    FilesystemLockConflict,
    FilesystemLockLease,
    FilesystemLockManager,
    FilesystemLockMissing,
    InMemoryFilesystemLockManager,
    create_filesystem_lock_manager,
)

__all__ = [
    "FilesystemLockConflict",
    "FilesystemLockLease",
    "FilesystemLockManager",
    "FilesystemLockMissing",
    "InMemoryFilesystemLockManager",
    "create_filesystem_lock_manager",
]
```

- [ ] **Step 5: Run tests until green**

Run:

```bash
source ../../.venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_lock_managers.py -q
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py::test_filesystem_lock_manager_injection_shares_leases_between_modules -q
```

Expected: PASS.

- [ ] **Step 6: Commit extraction**

```bash
git add mcp_unified/filesystem_locks tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_locks.py tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_lock_managers.py
git commit -m "Extract MCP filesystem lock managers"
```

## Task 2: Add SQLite Lock Manager With Matching Semantics

**Files:**
- Create/Modify: `mcp_unified/filesystem_locks/sqlite.py`
- Modify: `mcp_unified/filesystem_locks/__init__.py`
- Modify: `mcp_unified/filesystem_locks/memory.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_lock_managers.py`

- [ ] **Step 1: Add failing SQLite contract tests**

Add file-backed SQLite tests. Use `tmp_path / "locks.db"`, not `:memory:`, for shared-instance behavior.

```python
def test_sqlite_lock_manager_coordinates_two_instances(tmp_path: Path) -> None:
    from mcp_unified.filesystem_locks import FilesystemLockConflict, SQLiteFilesystemLockManager

    db_path = tmp_path / "locks.db"
    first = SQLiteFilesystemLockManager(db_path)
    second = SQLiteFilesystemLockManager(db_path)
    try:
        lease, renewed = first.acquire(workspace_key="ws", path="docs/story.txt", owner="agent-a", ttl_seconds=60)
        assert renewed is False

        with pytest.raises(FilesystemLockConflict) as exc:
            second.acquire(workspace_key="ws", path="docs/story.txt", owner="agent-b", ttl_seconds=60)

        assert exc.value.lease.lease_id == lease.lease_id
    finally:
        first.close()
        second.close()
```

Also add:

- expired-token renew returns `FilesystemLockMissing`
- expired row does not block new acquire
- wrong-token release raises `FilesystemLockConflict`
- missing/expired release returns `None`
- validate returns matching lease and classifies wrong/missing tokens correctly
- factory with `{"lock_manager_backend": "sqlite", "lock_manager_sqlite_path": str(db_path)}` returns `SQLiteFilesystemLockManager`
- factory with `sqlite` and no path raises `ValueError`

- [ ] **Step 2: Run the failing SQLite tests**

Run:

```bash
source ../../.venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_lock_managers.py -q
```

Expected: FAIL because `SQLiteFilesystemLockManager` does not exist.

- [ ] **Step 3: Implement SQLAlchemy Core schema and helpers**

In `sqlite.py`, build an isolated SQLAlchemy Core table:

```python
class SQLiteFilesystemLockManager:
    def __init__(
        self,
        path: str | Path,
        *,
        timeout_seconds: float = 30.0,
        token_bytes: int = 24,
        cleanup_interval: int = 64,
        cleanup_limit: int = 512,
    ) -> None:
        db_path = Path(path).expanduser()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.path = str(db_path)
        self._metadata = MetaData()
        self._table = Table(
            "mcp_filesystem_lock_leases",
            self._metadata,
            Column("workspace_key", String, primary_key=True),
            Column("path", String, primary_key=True),
            Column("lease_id", String, nullable=False),
            Column("owner", String, nullable=False),
            Column("expires_at_epoch_us", Integer, nullable=False),
            Column("ttl_seconds", Integer, nullable=False),
            Column("workspace_id", String),
            Column("session_id", String),
            Column("updated_at_epoch_us", Integer, nullable=False),
        )
        Index("idx_mcp_fs_lock_expires_at", self._table.c.expires_at_epoch_us)
        self._engine = create_engine(
            URL.create("sqlite", database=self.path),
            connect_args={"timeout": timeout_seconds, "check_same_thread": False},
            future=True,
        )
        self._metadata.create_all(self._engine)
```

Add helpers:

```python
def _now_us() -> int:
    return int(time.time() * 1_000_000)

def _lease_from_row(row: Mapping[str, Any]) -> FilesystemLockLease:
    return FilesystemLockLease(
        workspace_key=str(row["workspace_key"]),
        path=str(row["path"]),
        lease_id=str(row["lease_id"]),
        owner=str(row["owner"]),
        expires_at=int(row["expires_at_epoch_us"]) / 1_000_000,
        ttl_seconds=int(row["ttl_seconds"]),
        workspace_id=row["workspace_id"],
        session_id=row["session_id"],
    )
```

- [ ] **Step 4: Implement atomic acquire paths**

Use SQLite dialect insert for first acquire:

```python
statement = sqlite_insert(table).values(**values)
upsert = statement.on_conflict_do_update(
    index_elements=[table.c.workspace_key, table.c.path],
    set_={
        "lease_id": statement.excluded.lease_id,
        "owner": statement.excluded.owner,
        "expires_at_epoch_us": statement.excluded.expires_at_epoch_us,
        "ttl_seconds": statement.excluded.ttl_seconds,
        "workspace_id": statement.excluded.workspace_id,
        "session_id": statement.excluded.session_id,
        "updated_at_epoch_us": statement.excluded.updated_at_epoch_us,
    },
    where=table.c.expires_at_epoch_us <= now_us,
)
```

For renewal, use an `update()` with predicates on key, token, and active expiry. If no row updates, select the current row and raise `FilesystemLockConflict` for a different active token or `FilesystemLockMissing` for absent/expired rows.

- [ ] **Step 5: Implement release, validate, cleanup, and close**

Release and validate can select the row inside `engine.begin()` / `engine.connect()` and classify:

- no row: missing/`None`
- expired row: opportunistically delete and missing/`None`
- different token: conflict
- matching token: return/delete as appropriate

Add:

```python
def close(self) -> None:
    self._engine.dispose()
```

- [ ] **Step 6: Wire lazy SQLite exports and factory**

In `mcp_unified/filesystem_locks/__init__.py`, expose `SQLiteFilesystemLockManager` lazily through `__getattr__`, matching the pattern in `mcp_unified/storage/__init__.py`.

In the factory, import SQLite only inside the `backend == "sqlite"` branch and parse:

- `lock_manager_sqlite_path`
- `lock_manager_sqlite_timeout_seconds`
- `lock_manager_cleanup_interval`
- `lock_manager_cleanup_limit`

- [ ] **Step 7: Run tests until green**

Run:

```bash
source ../../.venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_lock_managers.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit SQLite backend**

```bash
git add mcp_unified/filesystem_locks tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_lock_managers.py
git commit -m "Add SQLite filesystem lock backend"
```

## Task 3: Integrate FilesystemModule Without Host Abstraction Growth

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_module.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py`

- [ ] **Step 1: Add failing module integration tests**

Add tests for backend-neutral descriptions and config creation:

```python
@pytest.mark.asyncio
async def test_filesystem_lock_tool_descriptions_are_backend_neutral(tmp_path: Path) -> None:
    mod = FilesystemModule(ModuleConfig(name="filesystem"))
    tools = {tool["name"]: tool for tool in await mod.get_tools()}

    assert "process-local" not in tools["fs.lock_acquire"]["description"]
    assert "process-local" not in tools["fs.lock_release"]["description"]


def test_filesystem_lock_manager_sqlite_backend_config_creates_manager(tmp_path: Path) -> None:
    mod = FilesystemModule(
        ModuleConfig(
            name="filesystem",
            settings={
                "lock_manager_backend": "sqlite",
                "lock_manager_sqlite_path": str(tmp_path / "locks.db"),
            },
        )
    )

    assert mod._lock_leases.__class__.__name__ == "SQLiteFilesystemLockManager"
```

Extend the unsupported backend parametrization to include `""` and add a separate `sqlite` missing path test.

- [ ] **Step 2: Run failing module tests**

Run:

```bash
source ../../.venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py::test_filesystem_lock_tool_descriptions_are_backend_neutral tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py::test_filesystem_lock_manager_sqlite_backend_config_creates_manager -q
```

Expected: FAIL until descriptions/import behavior are updated.

- [ ] **Step 3: Swap imports and descriptions**

Change the import block in `filesystem_module.py` from the local `.filesystem_locks` import to:

```python
from mcp_unified.filesystem_locks import (
    FilesystemLockConflict,
    FilesystemLockManager,
    FilesystemLockMissing,
    create_filesystem_lock_manager,
)
```

Update tool descriptions:

- `fs.lock_acquire`: "Acquire or renew an advisory lock lease for one workspace path."
- `fs.lock_release`: "Release an advisory lock lease for one workspace path."

Do not add any new host persistence layer or DB adapter.

- [ ] **Step 4: Run focused module tests**

Run:

```bash
source ../../.venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit host integration**

```bash
git add tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_module.py tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py
git commit -m "Wire filesystem module to package lock backend"
```

## Task 4: Update Standalone Packaging, Docs, And Artifact Gates

**Files:**
- Modify: `mcp_unified/pyproject.toml`
- Modify: `mcp_unified/README.md`
- Modify: `mcp_unified/USER_GUIDE.md`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py`

- [ ] **Step 1: Add failing package boundary tests**

In `test_runtime_package_boundary.py`, update the import-boundary test that checks SQLAlchemy is not imported by core package imports to include `mcp_unified.filesystem_locks`.

Add artifact assertions in `test_mcp_unified_standalone_sdist_contains_only_package_boundary()` and/or wheel member checks:

```python
assert any(member.endswith("/filesystem_locks/__init__.py") for member in members)
assert any(member.endswith("/filesystem_locks/models.py") for member in members)
assert any(member.endswith("/filesystem_locks/memory.py") for member in members)
assert any(member.endswith("/filesystem_locks/sqlite.py") for member in members)
```

If adding wheel checks, assert:

```python
assert "mcp_unified/filesystem_locks/__init__.py" in wheel_members
```

- [ ] **Step 2: Run failing package boundary tests**

Run:

```bash
source ../../.venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py -q
```

Expected: FAIL until standalone package metadata is updated.

- [ ] **Step 3: Update standalone package metadata**

In `mcp_unified/pyproject.toml`, add:

```toml
packages = [
  "mcp_unified",
  "mcp_unified.federation",
  "mcp_unified.filesystem_locks",
  "mcp_unified.gateway",
  "mcp_unified.interfaces",
  "mcp_unified.profiles",
  "mcp_unified.storage",
]

[tool.setuptools.package-dir]
mcp_unified = "."
"mcp_unified.federation" = "federation"
"mcp_unified.filesystem_locks" = "filesystem_locks"
...
```

Do not add SQLAlchemy to core dependencies. `mcp_unified/package_metadata.py` should remain unchanged unless tests prove a metadata mismatch.

- [ ] **Step 4: Update package-local docs**

In `mcp_unified/README.md`, add a short "Filesystem Lock Backends" section after "Minimal Gateway Config" or before "Tool-Use Reporting":

```markdown
## Filesystem Lock Backends

The package includes advisory filesystem lock manager primitives for hosts and
gateway integrations. Memory locks are process-local. The optional SQLite
backend coordinates cooperating processes that point at the same local database
file; it is not a distributed lock service.
```

In `mcp_unified/USER_GUIDE.md`, add an operator subsection after "Choose A Store":

```markdown
## 2.1 Configure Filesystem Lock Storage

Filesystem lock storage is separate from the gateway profile store. Use memory
for single-process development and SQLite when multiple local gateway/server
processes must coordinate advisory `fs.lock_acquire` leases.

The SQLite path is operator configuration, not a path an agent may choose
through filesystem tools.
```

Include sample settings names:

```json
{
  "lock_manager_backend": "sqlite",
  "lock_manager_sqlite_path": "./mcp-filesystem-locks.db"
}
```

- [ ] **Step 5: Run package and artifact tests**

Run:

```bash
source ../../.venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py -q
python -m pytest -c mcp_unified/pytest-artifact-gate.ini .github/tests/test_mcp_unified_artifact_gate.py -q
```

Expected: PASS. If artifact build dependencies are unavailable, record the exact failure and reason in `TASK-2345`.

- [ ] **Step 6: Commit packaging and docs**

```bash
git add mcp_unified/pyproject.toml mcp_unified/README.md mcp_unified/USER_GUIDE.md tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py
git commit -m "Package filesystem lock backend"
```

## Task 5: Full Focused Validation And Backlog Closeout

**Files:**
- Modify: `backlog/tasks/task-2345 - Implement-MCP-filesystem-SQLite-lock-backend.md`

- [ ] **Step 1: Run focused behavior tests**

Run:

```bash
source ../../.venv/bin/activate
python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_lock_managers.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py \
  -q
```

Expected: PASS.

- [ ] **Step 2: Run package boundary tests**

Run:

```bash
source ../../.venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py -q
python -m pytest -c mcp_unified/pytest-artifact-gate.ini .github/tests/test_mcp_unified_artifact_gate.py -q
```

Expected: PASS or documented environment skip for artifact build dependencies.

- [ ] **Step 3: Run compile check**

Run:

```bash
source ../../.venv/bin/activate
python -m py_compile \
  mcp_unified/filesystem_locks/__init__.py \
  mcp_unified/filesystem_locks/models.py \
  mcp_unified/filesystem_locks/memory.py \
  mcp_unified/filesystem_locks/sqlite.py \
  tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_locks.py \
  tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_module.py
```

Expected: exit 0.

- [ ] **Step 4: Run Bandit on touched implementation scope**

Run:

```bash
source ../../.venv/bin/activate
python -m bandit -r \
  mcp_unified/filesystem_locks \
  tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_locks.py \
  tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_module.py \
  -f json -o /tmp/bandit_mcp_fs_sqlite_locks.json
```

Expected: no new findings in touched code. Fix new findings before continuing.

- [ ] **Step 5: Run diff hygiene**

Run:

```bash
git diff --check
```

Expected: no output.

- [ ] **Step 6: Update Backlog task**

Use the Backlog MCP task edit tool for `TASK-2345`:

- set status to `Done`
- add touched files
- add verification commands and outcomes
- add final summary
- record any known skips/blockers

- [ ] **Step 7: Final commit**

```bash
git add backlog/tasks/task-2345\ -\ Implement-MCP-filesystem-SQLite-lock-backend.md
git commit -m "Close MCP filesystem SQLite lock backend task"
```

If the Backlog update is already included in the final implementation commit, skip this separate closeout commit and note it in the final response.
