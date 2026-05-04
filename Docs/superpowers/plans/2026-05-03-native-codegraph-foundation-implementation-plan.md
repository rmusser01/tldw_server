# Native CodeGraph Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first native CodeGraph foundation slice for Unified MCP: dependency health, trusted workspace resolution, tldw-managed SQLite index storage, language registry, bounded foreground index/sync, and the initial MCP tool shell.

**Architecture:** Keep graph behavior in `tldw_Server_API/app/core/CodeGraph/` and keep `CodeGraphModule` as a thin MCP adapter. Stage 1 indexes source-file inventory and durable run metadata only; it deliberately does not extract symbols, calls, import graphs, or task context. Deep Python and JS/TS extractors land in later slices against the same repository and registry contracts.

**Tech Stack:** Python 3.10+, Unified MCP `BaseModule`, SQLite/FTS5, `McpHubWorkspaceRootResolver`, pytest/pytest-asyncio, Loguru, optional Tree-sitter dependency probe behind `.[codegraph]`.

---

## Source Inputs

- Spec: `Docs/superpowers/specs/2026-05-03-native-codegraph-mcp-module-design.md`
- Tracking task: `TASK-16`
- MCP patterns:
  - `tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_module.py`
  - `tldw_Server_API/app/core/MCP_unified/modules/implementations/template_module.py`
  - `tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py`
- SQLite helper:
  - `tldw_Server_API/app/core/DB_Management/sqlite_policy.py`

## Scope Boundaries

Included in this slice:

- Core `CodeGraph` package skeleton.
- Settings parsing and optional dependency health checks.
- Stable workspace key generation and workspace-bound path resolution.
- SQLite schema and repository methods for file inventory, index runs, and empty graph counts.
- Language registry with first-slice file inventory support for Python, JavaScript, and TypeScript extensions, plus planned metadata for C, C++, C#, Java, and Kotlin.
- Bounded foreground `index` and `sync` runs that discover files, hash them, persist file records, and record run counters.
- MCP tools: `codegraph.status`, `codegraph.index`, `codegraph.sync`, and `codegraph.files`.
- Disabled `codegraph` module entry in `tldw_Server_API/Config_Files/mcp_modules.yaml`.

Not included in this slice:

- Symbol extraction.
- Call graph extraction or resolution.
- `codegraph.search`, `codegraph.node`, `codegraph.callers`, `codegraph.callees`, `codegraph.impact`, or `codegraph.context`.
- Jobs worker integration.
- File watcher.
- Real extractors for C, C++, C#, Java, or Kotlin.

Important behavior decision:

- Do not advertise deep graph query tools before they work. `codegraph.status` can report `planned_tools` for later slices, but `get_tools()` should expose only the four Stage 1 tools above.

## Planned File Structure

Create:

- `tldw_Server_API/app/core/CodeGraph/__init__.py`
  - Package exports and no optional parser imports.
- `tldw_Server_API/app/core/CodeGraph/config.py`
  - `CodeGraphSettings`, settings coercion, default excludes, foreground limits.
- `tldw_Server_API/app/core/CodeGraph/models.py`
  - Dataclasses/enums for workspace, language, file records, run results, status summaries.
- `tldw_Server_API/app/core/CodeGraph/workspace.py`
  - Trusted workspace resolution wrapper and stable workspace key generation.
- `tldw_Server_API/app/core/CodeGraph/language_registry.py`
  - Extension mapping, planned languages, dependency probe summary.
- `tldw_Server_API/app/core/CodeGraph/dependencies.py`
  - Optional dependency probe using `importlib.util.find_spec`; no hard Tree-sitter imports at module import time.
- `tldw_Server_API/app/core/DB_Management/codegraph/repository.py`
  - SQLite connection setup, schema migration, file/run CRUD, counts, stale cleanup for file rows.
- `tldw_Server_API/app/core/DB_Management/codegraph/schema.sql`
  - Stage 1 schema with future graph tables present but not populated.
- `tldw_Server_API/app/core/CodeGraph/indexer.py`
  - File discovery, exclude matching, symlink safety, hashing, foreground bounds, index/sync orchestration.
- `tldw_Server_API/app/core/MCP_unified/modules/implementations/codegraph_module.py`
  - Thin MCP adapter for status/index/sync/files.
- `tldw_Server_API/tests/CodeGraph/test_codegraph_config.py`
- `tldw_Server_API/tests/CodeGraph/test_codegraph_language_registry.py`
- `tldw_Server_API/tests/CodeGraph/test_codegraph_workspace.py`
- `tldw_Server_API/tests/CodeGraph/test_codegraph_repository.py`
- `tldw_Server_API/tests/CodeGraph/test_codegraph_indexer.py`
- `tldw_Server_API/app/core/MCP_unified/tests/test_codegraph_module.py`

Modify:

- `pyproject.toml`
  - Add `codegraph` optional dependency only after a parser matrix smoke has been verified.
- `tldw_Server_API/Config_Files/mcp_modules.yaml`
  - Add disabled CodeGraph module config.
- `backlog/tasks/task-16 - Implement-native-CodeGraph-foundation-slice.md`
  - Keep plan, notes, verification, and final summary current through Backlog MCP.

## Data Model Notes

Stage 1 schema should include future graph tables now so Stage 2 can add extractors without a storage rewrite:

- `schema_versions`
- `files`
- `nodes`
- `edges`
- `unresolved_refs`
- `index_runs`
- `project_metadata`
- `nodes_fts`

Stage 1 repository methods only need to write:

- `schema_versions`
- `files`
- `index_runs`
- `project_metadata` as needed

`nodes`, `edges`, and `unresolved_refs` should be created and counted, but remain empty until extractor slices land.

Stable IDs:

- Add pure helpers in `models.py`:
  - `stable_hash_id(prefix: str, identity: str) -> str`
  - `make_node_id(workspace_key, language, file_path, kind, qualified_name, start_line) -> str`
  - `make_edge_id(source_node_id, edge_kind, target_or_ref, file_path, line, column) -> str`
- Test those helpers now even though Stage 1 does not populate nodes or edges. This locks the stale-edge cleanup contract from the design.

## Task 1: Plan And Backlog Setup

**Files:**

- Create: `Docs/superpowers/plans/2026-05-03-native-codegraph-foundation-implementation-plan.md`
- Modify through MCP: `TASK-16`

- [ ] **Step 1: Confirm task is In Progress and points at the design**

Use Backlog MCP:

```text
task_view TASK-16
```

Expected: status is `In Progress`, documentation includes `Docs/superpowers/specs/2026-05-03-native-codegraph-mcp-module-design.md`.

- [ ] **Step 2: Record this plan path in TASK-16**

Use `task_edit` with `planSet` that includes:

```text
Plan: Docs/superpowers/plans/2026-05-03-native-codegraph-foundation-implementation-plan.md
Scope: Stage 1 foundation only. No symbol extraction, deep query tools, Jobs integration, or broad language extraction in this task.
```

- [ ] **Step 3: Wait for explicit approval before production code**

Expected: no files under `tldw_Server_API/app/core/CodeGraph/` or `codegraph_module.py` exist until the user approves executing this plan.

## Task 2: Settings And Optional Dependency Health

**Files:**

- Create: `tldw_Server_API/app/core/CodeGraph/__init__.py`
- Create: `tldw_Server_API/app/core/CodeGraph/config.py`
- Create: `tldw_Server_API/app/core/CodeGraph/dependencies.py`
- Test: `tldw_Server_API/tests/CodeGraph/test_codegraph_config.py`

- [ ] **Step 1: Write failing settings tests**

Create tests for:

```python
from tldw_Server_API.app.core.CodeGraph.config import CodeGraphSettings


def test_settings_use_safe_defaults() -> None:
    settings = CodeGraphSettings.from_mapping({})

    assert str(settings.index_base_dir).endswith("Databases/codegraph")
    assert settings.max_file_size_bytes == 1_048_576
    assert settings.foreground_max_files == 500
    assert ".git" in settings.exclude_dirs
    assert "node_modules" in settings.exclude_dirs


def test_settings_coerce_positive_integer_limits() -> None:
    settings = CodeGraphSettings.from_mapping(
        {"max_file_size_bytes": "2048", "foreground_max_files": "12"}
    )

    assert settings.max_file_size_bytes == 2048
    assert settings.foreground_max_files == 12
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/CodeGraph/test_codegraph_config.py -q
```

Expected: FAIL because `tldw_Server_API.app.core.CodeGraph` does not exist.

- [ ] **Step 3: Implement settings minimally**

Implement:

```python
@dataclass(frozen=True)
class CodeGraphSettings:
    index_base_dir: Path = Path("Databases/codegraph")
    max_file_size_bytes: int = 1_048_576
    foreground_max_files: int = 500
    foreground_max_bytes: int = 50_000_000
    max_index_seconds: float = 20.0
    exclude_dirs: tuple[str, ...] = DEFAULT_EXCLUDE_DIRS

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "CodeGraphSettings":
        ...
```

Use explicit positive integer coercion. Invalid values should fall back to defaults unless the value is unsafe, such as zero or negative, in which case clamp to at least `1`.

- [ ] **Step 4: Write failing dependency probe tests**

Create tests that monkeypatch `importlib.util.find_spec`:

```python
from tldw_Server_API.app.core.CodeGraph.dependencies import probe_codegraph_dependencies


def test_dependency_probe_reports_missing_without_importing_tree_sitter(monkeypatch) -> None:
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)

    health = probe_codegraph_dependencies()

    assert health.available is False
    assert "tree_sitter" in health.missing
```

- [ ] **Step 5: Verify RED, then implement dependency probe**

Run the same focused config test file.

Expected RED: import or attribute missing.

Implement `dependencies.py` using only `importlib.util.find_spec`. Probe names should be:

- `tree_sitter`
- `tree_sitter_python`
- `tree_sitter_javascript`
- `tree_sitter_typescript`

- [ ] **Step 6: Verify GREEN**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/CodeGraph/test_codegraph_config.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit checkpoint**

```bash
git add tldw_Server_API/app/core/CodeGraph/__init__.py \
  tldw_Server_API/app/core/CodeGraph/config.py \
  tldw_Server_API/app/core/CodeGraph/dependencies.py \
  tldw_Server_API/tests/CodeGraph/test_codegraph_config.py
git commit -m "feat: add codegraph settings and dependency health"
```

## Task 3: Models, Stable IDs, And Language Registry

**Files:**

- Create: `tldw_Server_API/app/core/CodeGraph/models.py`
- Create: `tldw_Server_API/app/core/CodeGraph/language_registry.py`
- Test: `tldw_Server_API/tests/CodeGraph/test_codegraph_language_registry.py`

- [ ] **Step 1: Write failing stable-ID tests**

```python
from tldw_Server_API.app.core.CodeGraph.models import make_edge_id, make_node_id


def test_node_id_is_deterministic_for_same_identity() -> None:
    first = make_node_id("ws", "python", "app/main.py", "function", "main", 10)
    second = make_node_id("ws", "python", "app/main.py", "function", "main", 10)

    assert first == second
    assert first.startswith("node_")


def test_edge_id_changes_when_target_changes() -> None:
    first = make_edge_id("node_a", "calls", "node_b", "app/main.py", 12, 4)
    second = make_edge_id("node_a", "calls", "node_c", "app/main.py", 12, 4)

    assert first != second
```

- [ ] **Step 2: Run and verify RED**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/CodeGraph/test_codegraph_language_registry.py -q
```

Expected: FAIL because models do not exist.

- [ ] **Step 3: Implement models minimally**

Add dataclasses:

- `DependencyHealth`
- `LanguageInfo`
- `WorkspaceResolution`
- `IndexedFile`
- `IndexRunSummary`
- `CodeGraphStatus`

Add `stable_hash_id` using `hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]`. Keep raw identity strings out of logs.

- [ ] **Step 4: Write failing language registry tests**

```python
from tldw_Server_API.app.core.CodeGraph.dependencies import DependencyHealth
from tldw_Server_API.app.core.CodeGraph.language_registry import CodeGraphLanguageRegistry


def test_registry_reports_foundation_languages_and_planned_languages() -> None:
    registry = CodeGraphLanguageRegistry(
        dependency_health=DependencyHealth(available=False, missing=("tree_sitter",), present=())
    )

    by_id = {language.language_id: language for language in registry.list_languages()}

    assert by_id["python"].stage == "foundation"
    assert by_id["javascript"].stage == "foundation"
    assert by_id["typescript"].stage == "foundation"
    assert by_id["java"].stage == "planned"
    assert by_id["kotlin"].stage == "planned"


def test_registry_maps_extensions_without_claiming_symbol_extraction() -> None:
    registry = CodeGraphLanguageRegistry()

    assert registry.language_for_path("api/server.py").language_id == "python"
    assert registry.language_for_path("apps/ui/page.tsx").language_id == "typescript"
    assert registry.language_for_path("src/main.cc").stage == "planned"
```

Planned-language mapping is metadata only. Stage 1 indexer tests must verify
that planned languages are not persisted as indexed files until real extractors
exist.

- [ ] **Step 5: Verify RED, implement, verify GREEN**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/CodeGraph/test_codegraph_language_registry.py -q
```

Expected: PASS after implementation.

- [ ] **Step 6: Commit checkpoint**

```bash
git add tldw_Server_API/app/core/CodeGraph/models.py \
  tldw_Server_API/app/core/CodeGraph/language_registry.py \
  tldw_Server_API/tests/CodeGraph/test_codegraph_language_registry.py
git commit -m "feat: add codegraph language registry"
```

## Task 4: Trusted Workspace Resolution

**Files:**

- Create: `tldw_Server_API/app/core/CodeGraph/workspace.py`
- Test: `tldw_Server_API/tests/CodeGraph/test_codegraph_workspace.py`

- [ ] **Step 1: Write failing workspace tests**

Mirror filesystem module behavior:

```python
from pathlib import Path

import pytest

from tldw_Server_API.app.core.CodeGraph.config import CodeGraphSettings
from tldw_Server_API.app.core.CodeGraph.workspace import CodeGraphWorkspaceResolver
from tldw_Server_API.app.core.MCP_unified.protocol import RequestContext


class FakeRootResolver:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.calls = []

    async def resolve_for_context(self, **kwargs):
        self.calls.append(kwargs)
        return {"workspace_root": str(self.root), "workspace_id": "ws-1", "source": "test"}


@pytest.mark.asyncio
async def test_workspace_resolver_rejects_session_only_without_user(tmp_path: Path) -> None:
    resolver = CodeGraphWorkspaceResolver(FakeRootResolver(tmp_path), CodeGraphSettings.from_mapping({}))
    context = RequestContext(request_id="req", session_id="sess-1", user_id=None, metadata={})

    with pytest.raises(PermissionError, match="workspace_root_unavailable"):
        await resolver.resolve(context)


@pytest.mark.asyncio
async def test_workspace_key_is_stable_and_index_path_is_not_inside_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    index_base = tmp_path / "indexes"
    resolver = CodeGraphWorkspaceResolver(
        FakeRootResolver(workspace),
        CodeGraphSettings.from_mapping({"index_base_dir": str(index_base)}),
    )
    context = RequestContext(request_id="req", session_id="sess-1", user_id="7", metadata={"workspace_id": "ws-1"})

    resolved = await resolver.resolve(context)

    assert resolved.workspace_root == workspace.resolve()
    assert resolved.index_db_path == index_base / resolved.workspace_key / "codegraph.db"
    assert workspace.resolve() not in resolved.index_db_path.resolve(strict=False).parents
```

- [ ] **Step 2: Run and verify RED**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/CodeGraph/test_codegraph_workspace.py -q
```

Expected: FAIL because workspace resolver does not exist.

- [ ] **Step 3: Implement workspace resolver**

Use the filesystem module's context extraction pattern:

- Extract `session_id`, `user_id`, `workspace_id`, `workspace_trust_source`, owner scope fields from `RequestContext.metadata`.
- Reject session-only contexts unless `workspace_trust_source == "shared_registry"`.
- Call `McpHubWorkspaceRootResolver.resolve_for_context`.
- Resolve workspace root with `Path(...).expanduser().resolve(strict=False)`.
- Build `workspace_key` from stable normalized parts:
  - user id when present
  - workspace id when present
  - trust source when present
  - resolved root path
- Hash with SHA-256 and prefix with `ws_`.
- Build index DB path from `settings.index_base_dir / workspace_key / "codegraph.db"`.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/CodeGraph/test_codegraph_workspace.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit checkpoint**

```bash
git add tldw_Server_API/app/core/CodeGraph/workspace.py \
  tldw_Server_API/tests/CodeGraph/test_codegraph_workspace.py
git commit -m "feat: add codegraph workspace resolver"
```

## Task 5: SQLite Schema And Repository

**Files:**

- Create: `tldw_Server_API/app/core/DB_Management/codegraph/schema.sql`
- Create: `tldw_Server_API/app/core/DB_Management/codegraph/repository.py`
- Test: `tldw_Server_API/tests/CodeGraph/test_codegraph_repository.py`

- [ ] **Step 1: Write failing repository initialization test**

```python
from pathlib import Path

from tldw_Server_API.app.core.DB_Management.codegraph.repository import CodeGraphRepository


def test_repository_initializes_schema_and_counts_empty_graph(tmp_path: Path) -> None:
    repo = CodeGraphRepository(tmp_path / "codegraph.db")
    repo.initialize()

    counts = repo.counts()

    assert counts["files"] == 0
    assert counts["nodes"] == 0
    assert counts["edges"] == 0
```

- [ ] **Step 2: Run and verify RED**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/CodeGraph/test_codegraph_repository.py -q
```

Expected: FAIL because repository does not exist.

- [ ] **Step 3: Implement schema and initialize**

Schema requirements:

- Use `TEXT PRIMARY KEY` IDs for deterministic graph rows.
- Use `path TEXT UNIQUE NOT NULL` in `files`.
- Store file errors as JSON text.
- Add FTS5 table `nodes_fts` with triggers if feasible. If triggers add too much complexity for Stage 1, create the FTS table and document triggers for Stage 2.
- Enable foreign keys.

Repository requirements:

- Ensure parent directory exists.
- Use `sqlite3.connect`.
- Call `configure_sqlite_connection` from `tldw_Server_API.app.core.DB_Management.sqlite_policy`.
- Load `schema.sql` with `Path(__file__).with_name("schema.sql").read_text(encoding="utf-8")`.
- Provide `initialize()`, `counts()`, `record_index_run_start()`, `finish_index_run()`, `upsert_file()`, `list_files()`, and `delete_missing_files()`.

- [ ] **Step 4: Write failing file/run CRUD tests**

```python
def test_repository_upserts_files_and_records_runs(tmp_path: Path) -> None:
    repo = CodeGraphRepository(tmp_path / "codegraph.db")
    repo.initialize()
    run_id = repo.record_index_run_start(workspace_key="ws_test", mode="foreground_index")

    repo.upsert_file(
        path="app/main.py",
        language="python",
        size=12,
        content_hash="abc",
        modified_at=1.5,
        status="indexed",
        errors=[],
    )
    repo.finish_index_run(run_id, status="complete", counters={"files_indexed": 1}, error_summary=[])

    assert repo.counts()["files"] == 1
    assert repo.list_files(limit=10)[0].path == "app/main.py"
    assert repo.last_index_run().status == "complete"
```

- [ ] **Step 5: Write failing stale graph cleanup test**

Manually seed future graph rows to make cleanup behavior testable before real
extractors exist:

```python
def test_repository_replacing_file_removes_owned_graph_rows_and_dangling_edges(tmp_path: Path) -> None:
    repo = CodeGraphRepository(tmp_path / "codegraph.db")
    repo.initialize()
    repo.upsert_file(
        path="app/main.py",
        language="python",
        size=12,
        content_hash="old",
        modified_at=1.0,
        status="indexed",
        errors=[],
    )
    repo.seed_graph_rows_for_test(
        nodes=[
            {"id": "node_old", "identity_key": "old", "kind": "function", "name": "old", "file_path": "app/main.py"},
            {"id": "node_other", "identity_key": "other", "kind": "function", "name": "other", "file_path": "app/other.py"},
        ],
        edges=[
            {"id": "edge_owned", "source": "node_old", "target": "node_other", "kind": "calls", "file_path": "app/main.py"},
            {"id": "edge_dangling", "source": "node_other", "target": "node_old", "kind": "calls", "file_path": "app/other.py"},
        ],
        unresolved_refs=[
            {"from_node_id": "node_old", "reference_name": "missing", "reference_kind": "call", "file_path": "app/main.py"},
        ],
    )

    repo.prepare_file_replacement("app/main.py")

    assert repo.counts()["nodes"] == 1
    assert repo.counts()["edges"] == 0
    assert repo.counts()["unresolved_refs"] == 0
```

Implementation requirements:

- Add a production cleanup method such as `prepare_file_replacement(path: str)`.
- Delete nodes and unresolved references owned by the file path.
- Delete edges owned by the file path.
- Delete edges whose `source` or `target` no longer exists after node cleanup.
- Keep the test-only seeding helper private or clearly named for tests.

- [ ] **Step 6: Verify RED, implement, verify GREEN**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/CodeGraph/test_codegraph_repository.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit checkpoint**

```bash
git add tldw_Server_API/app/core/DB_Management/codegraph/schema.sql \
  tldw_Server_API/app/core/DB_Management/codegraph/repository.py \
  tldw_Server_API/tests/CodeGraph/test_codegraph_repository.py
git commit -m "feat: add codegraph sqlite repository"
```

## Task 6: Bounded Foreground Index And Sync Skeleton

**Files:**

- Create: `tldw_Server_API/app/core/CodeGraph/indexer.py`
- Test: `tldw_Server_API/tests/CodeGraph/test_codegraph_indexer.py`

- [ ] **Step 1: Write failing file discovery and exclude tests**

```python
from pathlib import Path

from tldw_Server_API.app.core.CodeGraph.config import CodeGraphSettings
from tldw_Server_API.app.core.CodeGraph.indexer import CodeGraphIndexer
from tldw_Server_API.app.core.CodeGraph.language_registry import CodeGraphLanguageRegistry
from tldw_Server_API.app.core.DB_Management.codegraph.repository import CodeGraphRepository


def test_indexer_indexes_supported_file_inventory_and_skips_excluded_dirs(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "app.py").write_text("print('hi')\n", encoding="utf-8")
    node_modules = workspace / "node_modules"
    node_modules.mkdir()
    (node_modules / "ignored.ts").write_text("export const ignored = true\n", encoding="utf-8")

    repo = CodeGraphRepository(tmp_path / "index" / "codegraph.db")
    indexer = CodeGraphIndexer(
        settings=CodeGraphSettings.from_mapping({"index_base_dir": str(tmp_path / "index")}),
        registry=CodeGraphLanguageRegistry(),
    )

    result = indexer.index_workspace(
        workspace_root=workspace,
        workspace_key="ws_test",
        repository=repo,
        force=True,
        languages=None,
        max_files=10,
    )

    assert result.status == "complete"
    assert result.counters["files_indexed"] == 1
    assert repo.list_files(limit=10)[0].path == "app.py"
```

- [ ] **Step 2: Run and verify RED**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/CodeGraph/test_codegraph_indexer.py -q
```

Expected: FAIL because indexer does not exist.

- [ ] **Step 3: Implement minimal indexer**

Behavior:

- Walk with `os.scandir` or `Path.iterdir` recursively.
- Reject symlink targets escaping the workspace.
- Skip default excluded directories.
- Match files by language registry extension.
- Persist only languages whose registry stage is `foundation`; skip planned
  languages with a clear `planned_language_skipped` counter.
- Skip files over `max_file_size_bytes`.
- Enforce total candidate bytes with `foreground_max_bytes`.
- Enforce wall-clock bounds with `max_index_seconds`; inject a monotonic clock
  into `CodeGraphIndexer` tests so timeout behavior is deterministic and does
  not sleep.
- Skip files containing NUL bytes in the first small read.
- Compute SHA-256 content hash from bytes.
- Persist `files` rows only.
- Create index run rows with counters:
  - `files_seen`
  - `files_indexed`
  - `files_skipped`
  - `files_too_large`
  - `planned_language_skipped`
  - `unsupported_language`
  - `errors`
- If candidate file count or total bytes exceeds bounded limits, return status
  `index_too_large_for_foreground` and do not partially index.
- If the monotonic clock exceeds `max_index_seconds`, stop before starting the
  next file and return status `index_timed_out_for_foreground` with counters
  showing any completed file work.

- [ ] **Step 4: Write failing bounds and sync tests**

```python
def test_indexer_rejects_over_limit_foreground_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    for index in range(3):
        (workspace / f"file_{index}.py").write_text("x = 1\n", encoding="utf-8")

    repo = CodeGraphRepository(tmp_path / "codegraph.db")
    indexer = CodeGraphIndexer(settings=CodeGraphSettings.from_mapping({}), registry=CodeGraphLanguageRegistry())

    result = indexer.index_workspace(
        workspace_root=workspace,
        workspace_key="ws_test",
        repository=repo,
        force=True,
        languages=None,
        max_files=2,
    )

    assert result.status == "index_too_large_for_foreground"
    assert repo.counts()["files"] == 0


def test_indexer_rejects_over_total_byte_budget_without_partial_index(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "small.py").write_text("x = 1\n", encoding="utf-8")
    (workspace / "large.py").write_text("x = '" + ("a" * 64) + "'\n", encoding="utf-8")
    repo = CodeGraphRepository(tmp_path / "codegraph.db")
    indexer = CodeGraphIndexer(
        settings=CodeGraphSettings.from_mapping({"foreground_max_bytes": 16}),
        registry=CodeGraphLanguageRegistry(),
    )

    result = indexer.index_workspace(
        workspace_root=workspace,
        workspace_key="ws_test",
        repository=repo,
        force=True,
        languages=None,
        max_files=10,
    )

    assert result.status == "index_too_large_for_foreground"
    assert repo.counts()["files"] == 0


def test_indexer_stops_when_foreground_time_budget_expires(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "a.py").write_text("x = 1\n", encoding="utf-8")
    (workspace / "b.py").write_text("y = 2\n", encoding="utf-8")
    repo = CodeGraphRepository(tmp_path / "codegraph.db")
    ticks = iter([0.0, 0.0, 10.0])
    indexer = CodeGraphIndexer(
        settings=CodeGraphSettings.from_mapping({"max_index_seconds": 1}),
        registry=CodeGraphLanguageRegistry(),
        monotonic=lambda: next(ticks),
    )

    result = indexer.index_workspace(
        workspace_root=workspace,
        workspace_key="ws_test",
        repository=repo,
        force=True,
        languages=None,
        max_files=10,
    )

    assert result.status == "index_timed_out_for_foreground"
    assert result.counters["files_indexed"] == 1


def test_indexer_skips_planned_language_files_until_extractors_exist(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "main.cc").write_text("int main() { return 0; }\n", encoding="utf-8")
    repo = CodeGraphRepository(tmp_path / "codegraph.db")
    indexer = CodeGraphIndexer(settings=CodeGraphSettings.from_mapping({}), registry=CodeGraphLanguageRegistry())

    result = indexer.index_workspace(
        workspace_root=workspace,
        workspace_key="ws_test",
        repository=repo,
        force=True,
        languages=None,
        max_files=10,
    )

    assert result.status == "complete"
    assert result.counters["planned_language_skipped"] == 1
    assert repo.counts()["files"] == 0


def test_sync_removes_deleted_files(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = workspace / "app.py"
    source.write_text("x = 1\n", encoding="utf-8")
    repo = CodeGraphRepository(tmp_path / "codegraph.db")
    indexer = CodeGraphIndexer(settings=CodeGraphSettings.from_mapping({}), registry=CodeGraphLanguageRegistry())

    indexer.index_workspace(workspace, "ws_test", repo, force=True, languages=None, max_files=10)
    source.unlink()
    result = indexer.sync_workspace(workspace, "ws_test", repo, languages=None, max_files=10)

    assert result.status == "complete"
    assert repo.counts()["files"] == 0
```

- [ ] **Step 5: Verify RED, implement, verify GREEN**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/CodeGraph/test_codegraph_indexer.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit checkpoint**

```bash
git add tldw_Server_API/app/core/CodeGraph/indexer.py \
  tldw_Server_API/tests/CodeGraph/test_codegraph_indexer.py
git commit -m "feat: add bounded codegraph indexer"
```

## Task 7: MCP Module Shell

**Files:**

- Create: `tldw_Server_API/app/core/MCP_unified/modules/implementations/codegraph_module.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_codegraph_module.py`

- [ ] **Step 1: Write failing MCP tool metadata tests**

```python
import pytest

from tldw_Server_API.app.core.MCP_unified.modules.base import ModuleConfig
from tldw_Server_API.app.core.MCP_unified.modules.implementations.codegraph_module import CodeGraphModule


@pytest.mark.asyncio
async def test_codegraph_exposes_stage1_tools_only(tmp_path) -> None:
    module = CodeGraphModule(ModuleConfig(name="CodeGraph", settings={"index_base_dir": str(tmp_path)}))

    tools = await module.get_tools()
    by_name = {tool["name"]: tool for tool in tools}

    assert set(by_name) == {"codegraph.status", "codegraph.index", "codegraph.sync", "codegraph.files"}
    assert by_name["codegraph.status"]["metadata"]["readOnlyHint"] is True
    assert by_name["codegraph.index"]["metadata"]["category"] == "management"
    assert by_name["codegraph.sync"]["metadata"]["category"] == "management"
```

- [ ] **Step 2: Run and verify RED**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_codegraph_module.py -q
```

Expected: FAIL because module does not exist.

- [ ] **Step 3: Implement module definitions and argument validation**

Follow `FilesystemModule` patterns:

- Constructor accepts `ModuleConfig` and optional workspace root resolver for tests.
- `check_health()` returns dependency and repository readiness checks without requiring optional Tree-sitter dependencies.
- `execute_tool()` must offload blocking filesystem and SQLite work through
  `asyncio.to_thread` for `codegraph.index`, `codegraph.sync`, and
  `codegraph.files`. If `codegraph.status` inspects an existing DB, perform the
  DB read through `asyncio.to_thread` as well.
- `get_tools()` returns four tool definitions with:
  - `uses_filesystem: True`
  - `path_boundable: True`
  - `path_argument_hints: []`
  - read-only hint for `status` and `files`
  - management category for `index` and `sync`
- `validate_tool_arguments()` rejects unknown parameters.
- `mode` accepts only `"foreground"` in Stage 1.
- `max_files` must be a positive integer when provided.
- `languages` must be a list of known language IDs when provided.

- [ ] **Step 4: Write failing status/index/files behavior tests**

Use fake workspace resolver as in filesystem tests:

```python
@pytest.mark.asyncio
async def test_codegraph_status_is_read_only_when_index_is_absent(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    index_base = tmp_path / "indexes"
    resolver = FakeWorkspaceRootResolver({"workspace_root": str(workspace), "workspace_id": "ws-1"})
    module = CodeGraphModule(
        ModuleConfig(name="CodeGraph", settings={"index_base_dir": str(index_base)}),
        workspace_root_resolver=resolver,
    )
    context = RequestContext(request_id="req", session_id="sess-1", user_id="7", metadata={"workspace_id": "ws-1"})

    status = await module.execute_tool("codegraph.status", {}, context=context)

    assert status["workspace_key"].startswith("ws_")
    assert status["index_present"] is False
    assert status["counts"]["files"] == 0
    assert status["last_index_run"] is None
    assert str(workspace) not in status["index_db_path"]
    assert index_base.exists() is False


@pytest.mark.asyncio
async def test_codegraph_index_and_files_roundtrip(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "app.py").write_text("x = 1\n", encoding="utf-8")
    resolver = FakeWorkspaceRootResolver({"workspace_root": str(workspace), "workspace_id": "ws-1"})
    module = CodeGraphModule(
        ModuleConfig(name="CodeGraph", settings={"index_base_dir": str(tmp_path / "indexes")}),
        workspace_root_resolver=resolver,
    )
    context = RequestContext(request_id="req", session_id="sess-1", user_id="7", metadata={"workspace_id": "ws-1"})

    result = await module.execute_tool("codegraph.index", {"mode": "foreground", "max_files": 10}, context=context)
    files = await module.execute_tool("codegraph.files", {"limit": 10}, context=context)

    assert result["status"] == "complete"
    assert files["files"][0]["path"] == "app.py"


@pytest.mark.asyncio
async def test_codegraph_index_sync_and_files_offload_blocking_work(tmp_path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "app.py").write_text("x = 1\n", encoding="utf-8")
    resolver = FakeWorkspaceRootResolver({"workspace_root": str(workspace), "workspace_id": "ws-1"})
    module = CodeGraphModule(
        ModuleConfig(name="CodeGraph", settings={"index_base_dir": str(tmp_path / "indexes")}),
        workspace_root_resolver=resolver,
    )
    context = RequestContext(request_id="req", session_id="sess-1", user_id="7", metadata={"workspace_id": "ws-1"})
    offloaded = []

    async def fake_to_thread(func, *args, **kwargs):
        offloaded.append(getattr(func, "__name__", repr(func)))
        return func(*args, **kwargs)

    monkeypatch.setattr("asyncio.to_thread", fake_to_thread)

    await module.execute_tool("codegraph.index", {"mode": "foreground", "max_files": 10}, context=context)
    await module.execute_tool("codegraph.files", {"limit": 10}, context=context)
    await module.execute_tool("codegraph.sync", {"mode": "foreground", "max_files": 10}, context=context)

    assert len(offloaded) >= 3
```

- [ ] **Step 5: Verify RED, implement, verify GREEN**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_codegraph_module.py -q
```

Expected: PASS.

- [ ] **Step 6: Run protocol validation regression**

Add a test through `MCPProtocol` similar to `test_protocol_rejects_unknown_fs_read_text_arguments`, verifying unknown arguments to `codegraph.index` are rejected as invalid params.

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_codegraph_module.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit checkpoint**

```bash
git add tldw_Server_API/app/core/MCP_unified/modules/implementations/codegraph_module.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_codegraph_module.py
git commit -m "feat: add codegraph mcp module shell"
```

## Task 8: Module Config And Optional Dependency Matrix

**Files:**

- Modify: `tldw_Server_API/Config_Files/mcp_modules.yaml`
- Modify: `pyproject.toml`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_dynamic_module_catalog.py` or `test_codegraph_module.py`

- [ ] **Step 1: Write failing disabled-config registration test**

Add or extend a test that loads a temp module YAML containing the CodeGraph config and asserts disabled modules are skipped. If existing loader coverage is enough, add a narrower `mcp_modules.yaml` parse assertion instead.

Expected behavior:

- Default `mcp_modules.yaml` includes CodeGraph.
- CodeGraph entry is `enabled: false`.
- Server does not register it unless enabled by config/env.

- [ ] **Step 2: Add disabled YAML entry**

Add:

```yaml
  - id: codegraph
    class: tldw_Server_API.app.core.MCP_unified.modules.implementations.codegraph_module:CodeGraphModule
    enabled: false
    name: CodeGraph
    version: "0.1.0"
    department: code
    max_concurrent: 4
    settings:
      index_base_dir: Databases/codegraph
      max_file_size_bytes: 1048576
      foreground_max_files: 500
      foreground_max_bytes: 50000000
      max_context_chars: 35000
      max_search_results: 100
```

- [ ] **Step 3: Verify parser dependency matrix before editing `pyproject.toml`**

Use a disposable environment or the project venv only after approval for dependency installation if network is needed.

Candidate packages from the design:

```toml
"tree-sitter>=0.25,<0.26"
"tree-sitter-python>=0.25,<0.26"
"tree-sitter-javascript>=0.25,<0.26"
"tree-sitter-typescript>=0.23,<0.24"
```

Smoke expectations:

- `import tree_sitter`
- `import tree_sitter_python`
- `import tree_sitter_javascript`
- `import tree_sitter_typescript`
- Python parse succeeds on `def f(): return 1`
- JavaScript parse succeeds on `export function f() { return 1 }`
- TypeScript parse succeeds on `export type T = { x: number }`
- TSX parse succeeds on `export const C = () => <div />`

If the install or smoke cannot be verified because the sandbox has no network, do not add unverified dependency pins. Record the blocker in TASK-16 and leave `pyproject.toml` unchanged until the matrix is tested.

- [ ] **Step 4: Add verified `codegraph` optional extra**

Only after Step 3 passes, modify `pyproject.toml`:

```toml
codegraph = [
  "tree-sitter>=0.25,<0.26",
  "tree-sitter-python>=0.25,<0.26",
  "tree-sitter-javascript>=0.25,<0.26",
  "tree-sitter-typescript>=0.23,<0.24",
]
```

Adjust ranges to the actually verified compatible set.

- [ ] **Step 5: Commit checkpoint**

```bash
git add tldw_Server_API/Config_Files/mcp_modules.yaml pyproject.toml \
  tldw_Server_API/app/core/MCP_unified/tests/test_dynamic_module_catalog.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_codegraph_module.py
git commit -m "chore: register optional codegraph mcp module"
```

## Task 9: Focused Verification And Security Check

**Files:**

- All files touched by Tasks 2-8.
- Backlog `TASK-16`.

- [ ] **Step 1: Run focused CodeGraph tests**

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/CodeGraph \
  tldw_Server_API/app/core/MCP_unified/tests/test_codegraph_module.py \
  -q
```

Expected: PASS.

- [ ] **Step 2: Run MCP regression tests around touched surfaces**

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_dynamic_module_catalog.py \
  -q
```

Expected: PASS.

- [ ] **Step 3: Run Bandit on touched code**

```bash
source .venv/bin/activate && python -m bandit -r \
  tldw_Server_API/app/core/CodeGraph \
  tldw_Server_API/app/core/MCP_unified/modules/implementations/codegraph_module.py \
  -f json -o /tmp/bandit_codegraph_foundation.json
```

Expected: no new actionable findings in touched code. If Bandit is not installed, record the environment blocker in TASK-16.

- [ ] **Step 4: Check whitespace**

```bash
git diff --check
```

Expected: no whitespace errors.

- [ ] **Step 5: Update TASK-16**

Use Backlog MCP:

- Check acceptance criteria as they are met.
- Append verification command results.
- Add any dependency-matrix or Bandit blockers.
- Add final summary only after implementation is complete.

- [ ] **Step 6: Final commit**

If any verification or task-record updates remain uncommitted:

```bash
git status --short
git add <touched files and TASK-16>
git commit -m "test: verify codegraph foundation"
```

## Review Checkpoints

Checkpoint after Task 1:

- User approves plan before implementation starts.

Checkpoint after Task 4:

- Workspace resolution and identity model are locked before storage/indexing code grows.

Checkpoint after Task 6:

- File inventory indexing works before MCP surface is exposed.

Checkpoint after Task 8:

- Optional dependency pins are based on an actual parser smoke, not resolver guesses.

## Risks And Mitigations

- **Risk: Stage 1 overclaims graph capability.**
  - Mitigation: expose only `status`, `index`, `sync`, and `files`. Report later tools as planned, not callable.
- **Risk: optional Tree-sitter dependencies cannot be tested in the sandbox.**
  - Mitigation: dependency probe can land without optional deps; `pyproject.toml` extra waits for verified matrix.
- **Risk: foreground indexing blocks MCP.**
  - Mitigation: enforce file, byte, and wall-clock bounds; return `index_too_large_for_foreground` or `sync_too_large_for_foreground`.
- **Risk: index files leak into user workspace.**
  - Mitigation: tests assert index DB path lives under configured `index_base_dir`, outside the workspace root.
- **Risk: stale graph cleanup is deferred too far.**
  - Mitigation: create graph tables and deterministic ID helpers now; Stage 1 sync deletes file rows and empty dependent graph rows by file path even before nodes exist.

## Plan Review Note

The writing-plans skill recommends a plan-review subagent. This session cannot spawn subagents unless the user explicitly asks for parallel/subagent work, so the review for this plan is local and checklist-based:

- The plan directly addresses the four review findings from the design review.
- Jobs are deferred and v1 tools accept only bounded foreground mode.
- JS/TS alias work is explicitly out of Stage 1 and reserved for the JS/TS extractor slice.
- Stable IDs are specified and tested in Stage 1.
- `.[codegraph]` dependency pins are gated on a verified parser matrix.
