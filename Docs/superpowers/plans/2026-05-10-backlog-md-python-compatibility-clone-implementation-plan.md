# Backlog.md Python Compatibility Clone Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python compatibility clone of Backlog.md in staged, testable slices without cutting over this repository until CLI/MCP parity gates pass.

**Architecture:** Create an isolated Python package under `tools/backlog-py` with a shared domain library, CLI adapter, MCP adapter, and later browser/TUI adapters. Upstream Backlog.md is used only by oracle fixture-generation jobs; normal runtime and normal CI must not require Node/Bun. The first implementation slice proves inventory, pinned fixture strategy, project discovery, config loading, Markdown parsing, and read-only CLI/MCP behavior before any live mutation or PATH cutover.

**Tech Stack:** Python 3.10+, `setuptools`, `pytest`, `click`, `PyYAML`, standard-library `dataclasses`/`pathlib`/`tempfile`, optional future extras for MCP/browser/search after supply-chain review.

---

## Source Spec

- `Docs/superpowers/specs/2026-05-10-backlog-md-python-compatibility-clone-design.md`
- Tracking task: `TASK-244.1`

## Scope Check

The approved design covers a full compatibility clone: CLI, MCP, browser, TUI, config, search, mutation, packaging, and cutover. That is too broad for a single implementation PR. This plan is a master sequence of reviewable PR-sized tasks.

The first executable PR should stop at read-only compatibility plus oracle/inventory infrastructure. Do not implement live mutation, browser parity, or PATH cutover in the first PR.

## File Structure

- Create: `tools/backlog-py/pyproject.toml`
  - Isolated Python package metadata and console script definition. Keep dependencies minimal.
- Create: `tools/backlog-py/README.md`
  - Local development commands, cutover warnings, and oracle fixture policy.
- Create: `tools/backlog-py/src/backlog_py/__init__.py`
  - Version export only.
- Create: `tools/backlog-py/src/backlog_py/__main__.py`
  - Module entrypoint for `python -m backlog_py`.
- Create: `tools/backlog-py/src/backlog_py/cli/__init__.py`
- Create: `tools/backlog-py/src/backlog_py/cli/main.py`
  - Minimal Click entrypoint in Task 1, expanded in Task 5.
- Create: `tools/backlog-py/src/backlog_py/compat/__init__.py`
- Create: `tools/backlog-py/src/backlog_py/compat/inventory.py`
  - Typed upstream compatibility inventory model and validation helpers.
- Create: `tools/backlog-py/src/backlog_py/oracle/__init__.py`
- Create: `tools/backlog-py/src/backlog_py/oracle/manifest.py`
  - Pinned oracle manifest loader and validation.
- Create: `tools/backlog-py/src/backlog_py/storage/__init__.py`
- Create: `tools/backlog-py/src/backlog_py/storage/project.py`
  - Project root, config path, backlog directory, `BACKLOG_CWD`, and `--cwd` discovery.
- Create: `tools/backlog-py/src/backlog_py/storage/config.py`
  - YAML config loading, defaults, and camelCase/snake_case compatibility mapping.
- Create: `tools/backlog-py/src/backlog_py/markdown/__init__.py`
- Create: `tools/backlog-py/src/backlog_py/markdown/task_parser.py`
  - Loss-conscious task Markdown parser and renderer.
- Create: `tools/backlog-py/src/backlog_py/core/__init__.py`
- Create: `tools/backlog-py/src/backlog_py/core/models.py`
  - Task, checklist item, config, and repository model dataclasses.
- Create: `tools/backlog-py/src/backlog_py/core/repository.py`
  - Read-only repository operations: list/view/search/board data.
- Create: `tools/backlog-py/src/backlog_py/search/__init__.py`
- Create: `tools/backlog-py/src/backlog_py/search/simple.py`
  - Dependency-free deterministic search used before optional fuzzy search.
- Create: `tools/backlog-py/src/backlog_py/mcp/__init__.py`
- Create: `tools/backlog-py/src/backlog_py/mcp/resources.py`
  - Workflow resource content and resource alias support.
- Create: `tools/backlog-py/src/backlog_py/mcp/tools.py`
  - Tool registry functions backed by the core repository.
- Create: `tools/backlog-py/src/backlog_py/mcp/server.py`
  - Stdio MCP server adapter, introduced only after tool registry tests pass.
- Create: `tools/backlog-py/src/backlog_py/security/__init__.py`
- Create: `tools/backlog-py/src/backlog_py/security/paths.py`
  - Path traversal checks and backlog-root containment helpers.
- Create: `tools/backlog-py/tests/fixtures/oracle/manifest.yml`
  - Pinned upstream version/source metadata and command matrix.
- Create: `tools/backlog-py/tests/fixtures/repos/basic/backlog/config.yml`
  - Minimal local fixture repository.
- Create: `tools/backlog-py/tests/fixtures/repos/basic/backlog/tasks/task-1 - Example-task.md`
  - Representative Backlog.md task with frontmatter, sections, AC, DoD, and unknown metadata.
- Create: `tools/backlog-py/tests/test_inventory.py`
- Create: `tools/backlog-py/tests/test_oracle_manifest.py`
- Create: `tools/backlog-py/tests/test_project_discovery.py`
- Create: `tools/backlog-py/tests/test_task_parser.py`
- Create: `tools/backlog-py/tests/test_readonly_repository.py`
- Create: `tools/backlog-py/tests/test_cli_readonly.py`
- Create: `tools/backlog-py/tests/test_mcp_resources.py`
- Create: `tools/backlog-py/tests/test_security_paths.py`
- Modify: `backlog/tasks/task-244.1 - Write-Backlog.md-Python-compatibility-clone-implementation-plan.md`
  - Track plan path, verification, review, and final summary.

## Agent-Critical Parity Checklist

These operations block any agent workflow cutover:

- CLI: `backlog --help`
- CLI: `backlog task list --plain`
- CLI: `backlog task <id> --plain`
- CLI: `backlog search <query> --plain`
- CLI: `backlog board`
- CLI: `backlog config list`
- MCP resource: `backlog://workflow/overview`
- MCP resource alias: `backlog://docs/task-workflow`
- MCP tool: task search
- MCP tool: task view
- MCP tool: task create
- MCP tool: task edit
- MCP tool: document list/search/view/create/update
- MCP tool: milestone list/add/rename/remove/archive
- MCP tool: Definition of Done defaults get/upsert

Browser, interactive TUI, rich colored terminal output, shell completion installation, `onStatusChange`, remote git operations, auto-commit, and hook bypass must not block the first read-only agent cutover candidate. They must be explicitly marked as deferred or covered in later milestones.

## Task 1: Package Skeleton And Upstream Inventory

**Files:**
- Create: `tools/backlog-py/pyproject.toml`
- Create: `tools/backlog-py/README.md`
- Create: `tools/backlog-py/src/backlog_py/__init__.py`
- Create: `tools/backlog-py/src/backlog_py/__main__.py`
- Create: `tools/backlog-py/src/backlog_py/cli/__init__.py`
- Create: `tools/backlog-py/src/backlog_py/cli/main.py`
- Create: `tools/backlog-py/src/backlog_py/compat/__init__.py`
- Create: `tools/backlog-py/src/backlog_py/compat/inventory.py`
- Create: `tools/backlog-py/tests/test_inventory.py`
- Modify: `backlog/tasks/task-244.1 - Write-Backlog.md-Python-compatibility-clone-implementation-plan.md`

- [x] **Step 1: Write failing inventory tests**

Create `tools/backlog-py/tests/test_inventory.py`:

```python
from backlog_py.compat.inventory import load_builtin_inventory


def test_inventory_starts_with_agent_critical_commands():
    inventory = load_builtin_inventory()
    names = {item.name for item in inventory.items}

    assert "cli:task-list-plain" in names
    assert "cli:task-view-plain" in names
    assert "cli:search-plain" in names
    assert "mcp:workflow-overview" in names
    assert "mcp:task-search" in names


def test_inventory_classifies_browser_and_interactive_deferrals():
    inventory = load_builtin_inventory()
    by_name = {item.name: item for item in inventory.items}

    assert by_name["browser:kanban-drag-drop"].classification == "browser-deferred"
    assert by_name["cli:interactive-board"].classification == "interactive-deferred"
```

- [x] **Step 2: Run test to verify it fails**

Run:

```bash
source .venv/bin/activate
python -m pytest tools/backlog-py/tests/test_inventory.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'backlog_py'`.

- [x] **Step 3: Create package skeleton**

Create `tools/backlog-py/pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "backlog-py"
version = "0.1.0"
description = "Python compatibility clone of Backlog.md"
requires-python = ">=3.10"
dependencies = [
  "click>=8.0.0",
  "PyYAML>=6.0.0",
]

[project.scripts]
backlog-py = "backlog_py.cli.main:main"

[tool.setuptools.packages.find]
where = ["src"]
```

Create `tools/backlog-py/src/backlog_py/__init__.py`:

```python
"""Python compatibility clone of Backlog.md."""

__version__ = "0.1.0"
```

Create `tools/backlog-py/src/backlog_py/__main__.py`:

```python
from backlog_py.cli.main import main


if __name__ == "__main__":
    main()
```

Create `tools/backlog-py/src/backlog_py/cli/__init__.py` and
`tools/backlog-py/src/backlog_py/compat/__init__.py` as empty package markers.

Create a minimal `tools/backlog-py/src/backlog_py/cli/main.py` so the console
entry point is importable before the read-only CLI is implemented:

```python
from __future__ import annotations

import click


@click.group()
def main() -> None:
    """Python compatibility clone of Backlog.md."""
```

Create `tools/backlog-py/README.md`:

```markdown
# backlog-py

Python compatibility clone of Backlog.md.

This package is experimental. Do not put it on PATH as `backlog` and do not use
it to mutate the live repository until the cutover gates in the design spec pass.
```

- [x] **Step 4: Implement inventory model**

Create `tools/backlog-py/src/backlog_py/compat/inventory.py`:

```python
from __future__ import annotations

from dataclasses import dataclass


Classification = str


@dataclass(frozen=True)
class CompatibilityItem:
    name: str
    classification: Classification
    upstream_reference: str


@dataclass(frozen=True)
class CompatibilityInventory:
    items: list[CompatibilityItem]


def load_builtin_inventory() -> CompatibilityInventory:
    items = (
        CompatibilityItem("cli:task-list-plain", "golden-required", "CLI-INSTRUCTIONS.md"),
        CompatibilityItem("cli:task-view-plain", "golden-required", "CLI-INSTRUCTIONS.md"),
        CompatibilityItem("cli:search-plain", "golden-required", "CLI-INSTRUCTIONS.md"),
        CompatibilityItem("cli:config-list", "golden-required", "ADVANCED-CONFIG.md"),
        CompatibilityItem("mcp:workflow-overview", "golden-required", "agent-nudge.md"),
        CompatibilityItem("mcp:task-search", "golden-required", "MCP tools"),
        CompatibilityItem("browser:kanban-drag-drop", "browser-deferred", "README.md"),
        CompatibilityItem("cli:interactive-board", "interactive-deferred", "CLI-INSTRUCTIONS.md"),
    )
    return CompatibilityInventory(items=items)
```

- [x] **Step 5: Run focused tests**

Run:

```bash
source .venv/bin/activate
python -m pip install -e tools/backlog-py
python -m pytest tools/backlog-py/tests/test_inventory.py -v
```

Expected: PASS.

- [x] **Step 6: Commit Task 1**

```bash
git add tools/backlog-py "backlog/tasks/task-244.1 - Write-Backlog.md-Python-compatibility-clone-implementation-plan.md"
git commit -m "Add Backlog.md Python clone inventory scaffold"
```

## Task 2: Pinned Oracle Fixture Manifest

**Files:**
- Create: `tools/backlog-py/src/backlog_py/oracle/manifest.py`
- Create: `tools/backlog-py/tests/fixtures/oracle/manifest.yml`
- Create: `tools/backlog-py/tests/test_oracle_manifest.py`
- Modify: `tools/backlog-py/README.md`
- Modify: `backlog/tasks/task-244.1 - Write-Backlog.md-Python-compatibility-clone-implementation-plan.md`

- [x] **Step 1: Write failing manifest tests**

Create `tools/backlog-py/tests/test_oracle_manifest.py`:

```python
from pathlib import Path

from backlog_py.oracle.manifest import load_oracle_manifest


def test_manifest_pins_upstream_version_and_source():
    manifest = load_oracle_manifest(Path("tools/backlog-py/tests/fixtures/oracle/manifest.yml"))

    assert manifest.upstream_version == "1.44.0"
    assert manifest.source_kind in {"npm-release", "github-release", "source-commit"}
    assert manifest.source_ref
    assert manifest.package_metadata_sha256


def test_manifest_marks_agent_critical_fixtures():
    manifest = load_oracle_manifest(Path("tools/backlog-py/tests/fixtures/oracle/manifest.yml"))
    names = {fixture.name for fixture in manifest.fixtures if fixture.agent_critical}

    assert "cli:task-list-plain" in names
    assert "mcp:workflow-overview" in names
```

- [x] **Step 2: Run test to verify it fails**

Run:

```bash
source .venv/bin/activate
python -m pytest tools/backlog-py/tests/test_oracle_manifest.py -v
```

Expected: FAIL with missing `backlog_py.oracle`.

- [x] **Step 3: Add pinned manifest fixture**

Create `tools/backlog-py/tests/fixtures/oracle/manifest.yml`:

```yaml
upstream_version: "1.44.0"
source_kind: "npm-release"
source_ref: "backlog.md@1.44.0"
package_metadata_sha256: "b890dde4a33480361ff34195192e1c0a23d6c7dc1c47b095933a29c7ccb4eee6"
generated_at: "2026-05-10"
fixtures:
  - name: "cli:task-list-plain"
    command: ["backlog", "task", "list", "--plain"]
    agent_critical: true
  - name: "cli:task-view-plain"
    command: ["backlog", "task", "1", "--plain"]
    agent_critical: true
  - name: "cli:search-plain"
    command: ["backlog", "search", "Backlog.md", "--plain"]
    agent_critical: true
  - name: "cli:config-list"
    command: ["backlog", "config", "list"]
    agent_critical: true
  - name: "mcp:workflow-overview"
    resource: "backlog://workflow/overview"
    agent_critical: true
  - name: "mcp:task-search"
    tool: "task_search"
    agent_critical: true
  - name: "browser:kanban-drag-drop"
    command: ["backlog", "browser"]
    agent_critical: false
    classification: "browser-deferred"
```

- [x] **Step 4: Implement manifest loader**

Create `tools/backlog-py/src/backlog_py/oracle/manifest.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class OracleFixture:
    name: str
    agent_critical: bool
    command: list[str] | None = None
    resource: str | None = None
    tool: str | None = None
    classification: str = "golden-required"


@dataclass(frozen=True)
class OracleManifest:
    upstream_version: str
    source_kind: str
    source_ref: str
    package_metadata_sha256: str
    fixtures: list[OracleFixture]


def load_oracle_manifest(path: Path) -> OracleManifest:
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    fixtures = [
        OracleFixture(
            name=item["name"],
            agent_critical=bool(item["agent_critical"]),
            command=item.get("command"),
            resource=item.get("resource"),
            tool=item.get("tool"),
            classification=item.get("classification", "golden-required"),
        )
        for item in raw["fixtures"]
    ]
    return OracleManifest(
        upstream_version=raw["upstream_version"],
        source_kind=raw["source_kind"],
        source_ref=raw["source_ref"],
        package_metadata_sha256=raw["package_metadata_sha256"],
        fixtures=fixtures,
    )
```

- [x] **Step 5: Run focused tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tools/backlog-py/tests/test_oracle_manifest.py -v
```

Expected: PASS.

- [x] **Step 6: Commit Task 2**

```bash
git add tools/backlog-py "backlog/tasks/task-244.1 - Write-Backlog.md-Python-compatibility-clone-implementation-plan.md"
git commit -m "Add pinned Backlog.md oracle manifest"
```

## Task 3: Project Discovery And Config Loading

**Files:**
- Create: `tools/backlog-py/src/backlog_py/core/models.py`
- Create: `tools/backlog-py/src/backlog_py/storage/config.py`
- Create: `tools/backlog-py/src/backlog_py/storage/project.py`
- Create: `tools/backlog-py/tests/test_project_discovery.py`
- Modify: `backlog/tasks/task-244.1 - Write-Backlog.md-Python-compatibility-clone-implementation-plan.md`

- [x] **Step 1: Write failing discovery tests**

Create `tools/backlog-py/tests/test_project_discovery.py`:

```python
from pathlib import Path

from backlog_py.storage.project import discover_project


def test_discovers_folder_local_config(tmp_path):
    (tmp_path / "backlog").mkdir()
    (tmp_path / "backlog" / "config.yml").write_text("project_name: demo\nremote_operations: false\n", encoding="utf-8")

    project = discover_project(tmp_path)

    assert project.root == tmp_path
    assert project.backlog_dir == tmp_path / "backlog"
    assert project.config.remote_operations is False


def test_backlog_cwd_overrides_process_cwd(tmp_path, monkeypatch):
    project_root = tmp_path / "project"
    (project_root / "backlog").mkdir(parents=True)
    (project_root / "backlog" / "config.yml").write_text("project_name: env-demo\n", encoding="utf-8")
    monkeypatch.setenv("BACKLOG_CWD", str(project_root))

    project = discover_project(tmp_path)

    assert project.root == project_root
    assert project.config.project_name == "env-demo"
```

- [x] **Step 2: Run test to verify it fails**

Run:

```bash
source .venv/bin/activate
python -m pytest tools/backlog-py/tests/test_project_discovery.py -v
```

Expected: FAIL with missing `backlog_py.storage`.

- [x] **Step 3: Implement config and project models**

Create `tools/backlog-py/src/backlog_py/core/models.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BacklogConfig:
    project_name: str
    statuses: list[str] | None = None
    default_status: str = "To Do"
    remote_operations: bool = True
    auto_commit: bool = False
    bypass_git_hooks: bool = False
    check_active_branches: bool = True
    active_branch_days: int = 30
    definition_of_done: list[str] | None = None


@dataclass(frozen=True)
class BacklogProject:
    root: Path
    backlog_dir: Path
    config_path: Path
    config: BacklogConfig
```

Create `tools/backlog-py/src/backlog_py/storage/config.py` and `project.py` with pure functions:

`load_config(path: Path) -> BacklogConfig` should read YAML safely, map both
snake_case and camelCase keys to `BacklogConfig`, and apply built-in defaults
for omitted values.

`discover_project(cwd: Path, explicit_cwd: Path | None = None) -> BacklogProject`
should resolve the effective working directory, locate the supported config
shape, and return the resolved backlog root plus loaded config.

Support `backlog.config.yml`, `backlog/config.yml`, `.backlog/config.yml`, `BACKLOG_CWD`, and explicit `--cwd` precedence.

- [x] **Step 4: Add root and `.backlog` tests**

Add tests for:

- root `backlog.config.yml`
- `.backlog/config.yml`
- snake_case YAML keys rendering into dataclass attributes
- camelCase keys accepted when present in generated config output
- `--no-git` style config with `remote_operations=false`, `auto_commit=false`, `check_active_branches=false`

- [x] **Step 5: Run focused tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tools/backlog-py/tests/test_project_discovery.py -v
```

Expected: PASS.

- [x] **Step 6: Commit Task 3**

```bash
git add tools/backlog-py "backlog/tasks/task-244.1 - Write-Backlog.md-Python-compatibility-clone-implementation-plan.md"
git commit -m "Add Backlog.md project discovery"
```

## Task 4: Loss-Conscious Task Markdown Parser

**Files:**
- Create: `tools/backlog-py/src/backlog_py/markdown/task_parser.py`
- Create: `tools/backlog-py/tests/fixtures/repos/basic/backlog/tasks/task-1 - Example-task.md`
- Create: `tools/backlog-py/tests/test_task_parser.py`
- Modify: `tools/backlog-py/src/backlog_py/core/models.py`
- Modify: `backlog/tasks/task-244.1 - Write-Backlog.md-Python-compatibility-clone-implementation-plan.md`

- [x] **Step 1: Write fixture task**

Create `tools/backlog-py/tests/fixtures/repos/basic/backlog/tasks/task-1 - Example-task.md` with frontmatter, unknown metadata, Description, Acceptance Criteria, Implementation Notes, Final Summary, and Definition of Done section markers.

- [x] **Step 2: Write failing parser tests**

Create `tools/backlog-py/tests/test_task_parser.py`:

```python
from pathlib import Path

from backlog_py.markdown.task_parser import parse_task_markdown, render_task_markdown


FIXTURE = Path("tools/backlog-py/tests/fixtures/repos/basic/backlog/tasks/task-1 - Example-task.md")


def test_parse_preserves_unknown_frontmatter_and_sections():
    source = FIXTURE.read_text(encoding="utf-8")
    parsed = parse_task_markdown(source)

    assert parsed.frontmatter["id"] == "TASK-1"
    assert parsed.frontmatter["custom_field"] == "preserve-me"
    assert parsed.sections["DESCRIPTION"].strip()


def test_round_trip_without_mutation_is_exact():
    source = FIXTURE.read_text(encoding="utf-8")
    parsed = parse_task_markdown(source)

    assert render_task_markdown(parsed) == source
```

- [x] **Step 3: Run test to verify it fails**

Run:

```bash
source .venv/bin/activate
python -m pytest tools/backlog-py/tests/test_task_parser.py -v
```

Expected: FAIL with missing parser.

- [x] **Step 4: Implement parser**

Implement a deliberately conservative parser:

- split YAML frontmatter only when the file starts with `---`
- keep original raw frontmatter order for no-op rendering
- detect `<!-- SECTION:<NAME>:BEGIN -->` and `<!-- SECTION:<NAME>:END -->`
- parse checklist lines into structured data without discarding raw lines
- preserve all unrecognized body content
- raise a structured error on unterminated owned sections

- [x] **Step 5: Run parser tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tools/backlog-py/tests/test_task_parser.py -v
```

Expected: PASS.

- [x] **Step 6: Commit Task 4**

```bash
git add tools/backlog-py "backlog/tasks/task-244.1 - Write-Backlog.md-Python-compatibility-clone-implementation-plan.md"
git commit -m "Add Backlog.md task parser"
```

## Task 5: Read-Only Repository Operations And CLI

**Files:**
- Create: `tools/backlog-py/src/backlog_py/core/repository.py`
- Create: `tools/backlog-py/src/backlog_py/search/simple.py`
- Create: `tools/backlog-py/src/backlog_py/cli/main.py`
- Create: `tools/backlog-py/tests/test_readonly_repository.py`
- Create: `tools/backlog-py/tests/test_cli_readonly.py`
- Modify: `backlog/tasks/task-244.1 - Write-Backlog.md-Python-compatibility-clone-implementation-plan.md`

- [x] **Step 1: Write failing read-only repository tests**

Test:

- list tasks from fixture repo
- view `TASK-1`
- search by title/body
- board grouping by status
- no file changes after read-only operations

- [x] **Step 2: Write failing CLI tests**

Use `click.testing.CliRunner`:

```python
from click.testing import CliRunner

from backlog_py.cli.main import main


def test_task_list_plain_outputs_task_id(fixture_repo):
    result = CliRunner().invoke(main, ["--cwd", str(fixture_repo), "task", "list", "--plain"])

    assert result.exit_code == 0
    assert "TASK-1" in result.output


def test_config_list_outputs_safe_defaults(fixture_repo):
    result = CliRunner().invoke(main, ["--cwd", str(fixture_repo), "config", "list"])

    assert result.exit_code == 0
    assert "autoCommit: false" in result.output
```

- [x] **Step 3: Run tests to verify failure**

Run:

```bash
source .venv/bin/activate
python -m pytest tools/backlog-py/tests/test_readonly_repository.py tools/backlog-py/tests/test_cli_readonly.py -v
```

Expected: FAIL with missing repository/CLI implementation.

- [x] **Step 4: Implement read-only repository and CLI**

Implement:

- `task list --plain`
- `task <id> --plain`
- `search <query> --plain`
- `board`
- `config list`
- top-level `--cwd`
- top-level `--help`

Keep console script as `backlog-py`, not `backlog`, until cutover gates pass.

- [x] **Step 5: Run focused tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tools/backlog-py/tests/test_readonly_repository.py tools/backlog-py/tests/test_cli_readonly.py -v
```

Expected: PASS.

- [x] **Step 6: Run live read-only smoke against this repo**

Run:

```bash
source .venv/bin/activate
git status --short -- backlog > /tmp/backlog_status_before.txt
backlog-py --cwd /Users/macbook-dev/Documents/GitHub/tldw_server2 task list --plain
backlog-py --cwd /Users/macbook-dev/Documents/GitHub/tldw_server2 task TASK-1 --plain
backlog-py --cwd /Users/macbook-dev/Documents/GitHub/tldw_server2 search "Backlog.md" --plain
backlog-py --cwd /Users/macbook-dev/Documents/GitHub/tldw_server2 config list
git status --short -- backlog > /tmp/backlog_status_after.txt
diff -u /tmp/backlog_status_before.txt /tmp/backlog_status_after.txt
```

Expected: commands exit 0 and the before/after backlog status files match
exactly. This avoids false failures from unrelated pre-existing dirty backlog
task files while still proving the Python read-only commands did not write.

- [x] **Step 7: Commit Task 5**

```bash
git add tools/backlog-py "backlog/tasks/task-244.1 - Write-Backlog.md-Python-compatibility-clone-implementation-plan.md"
git commit -m "Add Backlog.md read-only CLI"
```

## Task 6: Read-Only MCP Resource And Tool Registry

**Files:**
- Create: `tools/backlog-py/src/backlog_py/mcp/resources.py`
- Create: `tools/backlog-py/src/backlog_py/mcp/tools.py`
- Create: `tools/backlog-py/src/backlog_py/mcp/server.py`
- Create: `tools/backlog-py/tests/test_mcp_resources.py`
- Modify: `tools/backlog-py/pyproject.toml` only after dependency verification
- Modify: `backlog/tasks/task-244.1 - Write-Backlog.md-Python-compatibility-clone-implementation-plan.md`

- [x] **Step 1: Verify MCP dependency availability before editing package deps**

Run:

```bash
source .venv/bin/activate
python - <<'PY'
import importlib.util
print(importlib.util.find_spec("mcp") is not None)
PY
```

Expected: prints `True` if MCP SDK is already available. If `False`, do not add an unverified dependency yet; implement and test `resources.py` and `tools.py` as pure functions first, then ask before installing network dependencies.

- [x] **Step 2: Write failing pure registry tests**

Test:

- `backlog://workflow/overview` returns workflow instructions
- `backlog://docs/task-workflow` alias returns compatible instructions
- task search tool returns fixture task
- task view tool returns fixture task
- unsupported mutation tools return explicit not-implemented errors until Task 7

- [x] **Step 3: Implement resource and tool registry**

Implement pure functions:

Implement these functions with concrete return values backed by the read-only
repository layer:

- `read_resource(uri: str) -> str`
- `task_search(project: BacklogProject, query: str, limit: int = 10) -> list[dict]`
- `task_view(project: BacklogProject, task_id: str) -> dict`

Do not expose generic shell execution.

- [x] **Step 4: Add stdio server adapter only when SDK is available**

If `mcp` SDK is installed, implement `server.py` as a thin adapter over the pure registry. If not installed, leave `server.py` with a clear import-time message and keep tests focused on pure functions.

- [x] **Step 5: Run MCP tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tools/backlog-py/tests/test_mcp_resources.py -v
```

Expected: PASS.

- [x] **Step 6: Commit Task 6**

```bash
git add tools/backlog-py "backlog/tasks/task-244.1 - Write-Backlog.md-Python-compatibility-clone-implementation-plan.md"
git commit -m "Add Backlog.md read-only MCP registry"
```

## Task 7: Safe Mutation Core

**Files:**
- Create: `tools/backlog-py/src/backlog_py/security/paths.py`
- Create: `tools/backlog-py/tests/test_security_paths.py`
- Create: `tools/backlog-py/tests/test_task_mutations.py`
- Modify: `tools/backlog-py/src/backlog_py/core/repository.py`
- Modify: `tools/backlog-py/src/backlog_py/cli/main.py`
- Modify: `tools/backlog-py/src/backlog_py/mcp/tools.py`
- Modify: `backlog/tasks/task-244.1 - Write-Backlog.md-Python-compatibility-clone-implementation-plan.md`

- [x] **Step 1: Write failing mutation/security tests**

Test:

- create task in temp repo
- edit description/notes without rewriting unowned sections
- check/uncheck AC and DoD by valid indexes
- reject invalid checklist indexes
- reject path traversal
- reject duplicate IDs
- reject circular dependencies
- write atomically without partial file on validation error
- `onStatusChange` disabled by default

- [x] **Step 2: Run tests to verify failure**

Run:

```bash
source .venv/bin/activate
python -m pytest tools/backlog-py/tests/test_security_paths.py tools/backlog-py/tests/test_task_mutations.py -v
```

Expected: FAIL with missing mutation/security functions.

- [x] **Step 3: Implement safe writes**

Implement:

- path containment helper
- temporary file in same directory
- `os.replace` atomic commit
- no writes before validation succeeds
- section-scoped mutations
- disabled-by-default `onStatusChange` path with explicit not-implemented/disabled behavior

- [x] **Step 4: Add CLI/MCP mutation adapters**

Implement only after core tests pass:

- CLI `task create`
- CLI `task edit --append-notes`, `--check-ac`, `--check-dod`, `--final-summary`
- MCP task create/edit equivalents

- [x] **Step 5: Run mutation and regression tests**

Worker implementation note: focused mutation/security tests and the full
`tools/backlog-py/tests` suite passed on 2026-05-10. Bandit, diff checks, and
two-stage review remain controller-owned gates for TASK-244.8.

Run:

```bash
source .venv/bin/activate
python -m pytest tools/backlog-py/tests -v
python -m bandit -r tools/backlog-py/src -f json -o /tmp/bandit_backlog_py.json
git diff --check
```

Expected: pytest PASS, Bandit has no new findings, diff check clean.

- [x] **Step 6: Commit Task 7**

```bash
git add tools/backlog-py "backlog/tasks/task-244.1 - Write-Backlog.md-Python-compatibility-clone-implementation-plan.md"
git commit -m "Add Backlog.md safe task mutations"
```

## Task 8: Documents, Decisions, Milestones, And Definition Of Done

**Files:**
- Create: `tools/backlog-py/src/backlog_py/core/documents.py`
- Create: `tools/backlog-py/src/backlog_py/core/milestones.py`
- Create: `tools/backlog-py/tests/test_documents.py`
- Create: `tools/backlog-py/tests/test_milestones.py`
- Create: `tools/backlog-py/tests/test_definition_of_done.py`
- Modify: `tools/backlog-py/src/backlog_py/core/repository.py`
- Modify: `tools/backlog-py/src/backlog_py/storage/config.py`
- Modify: `tools/backlog-py/src/backlog_py/cli/main.py`
- Modify: `tools/backlog-py/src/backlog_py/mcp/tools.py`
- Modify: `Docs/superpowers/plans/2026-05-10-backlog-md-python-compatibility-clone-implementation-plan.md`
- Modify: `backlog/tasks/task-244.9 - Implement-Backlog.md-Python-clone-documents-milestones-and-Definition-of-Done-parity.md`

- [x] **Step 1: Write failing document tests**

Test:

- nested document create under `backlog/docs/guides`
- document list searches nested paths
- document view by ID
- document update preserves omitted metadata
- absolute paths and `..` traversal are rejected

- [x] **Step 2: Write failing milestone tests**

Test:

- milestone add creates the expected milestone file
- milestone list includes active milestone files
- milestone rename updates file names and task references only when requested
- milestone remove can clear task references
- milestone archive moves the milestone to the archive path

- [x] **Step 3: Write failing Definition of Done tests**

Test:

- project defaults load from `definition_of_done`
- defaults upsert replaces the config value
- new task creation inherits defaults unless disabled
- task-specific DoD append does not mutate project defaults

- [x] **Step 4: Run tests to verify failure**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tools/backlog-py/tests/test_documents.py \
  tools/backlog-py/tests/test_milestones.py \
  tools/backlog-py/tests/test_definition_of_done.py \
  -v
```

Expected: FAIL with missing document/milestone/DoD implementations.

- [x] **Step 5: Implement document and milestone services**

Implement:

- document ID/path lookup scoped globally under `backlog/docs`
- docs-relative path validation
- decision/document frontmatter preservation
- milestone file operations
- task milestone reference updates through the task parser
- DoD defaults get/upsert through config writes

- [x] **Step 6: Add CLI/MCP adapters**

Implement:

- CLI `doc list/view/create/update`
- CLI `milestone list/add/rename/remove/archive`
- CLI config or helper path for Definition of Done defaults if upstream exposes it through CLI
- MCP document tools
- MCP milestone tools
- MCP Definition of Done defaults get/upsert tools

- [x] **Step 7: Run focused tests**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tools/backlog-py/tests/test_documents.py \
  tools/backlog-py/tests/test_milestones.py \
  tools/backlog-py/tests/test_definition_of_done.py \
  -v
```

Expected: PASS.

- [x] **Step 8: Run security regression checks**

Run:

```bash
source .venv/bin/activate
python -m pytest tools/backlog-py/tests/test_security_paths.py -v
python -m bandit -r tools/backlog-py/src -f json -o /tmp/bandit_backlog_py_docs_milestones.json
git diff --check
```

Expected: tests PASS, Bandit has no new findings, diff check clean.

- [x] **Step 9: Commit Task 8**

```bash
git add tools/backlog-py \
  Docs/superpowers/plans/2026-05-10-backlog-md-python-compatibility-clone-implementation-plan.md \
  "backlog/tasks/task-244.9 - Implement-Backlog.md-Python-clone-documents-milestones-and-Definition-of-Done-parity.md"
git commit -m "Add Backlog.md document and milestone parity"
```

## Task 9: Agent Cutover Candidate Validation

**Files:**
- Create: `tools/backlog-py/docs/agent-critical-parity.md`
- Create: `tools/backlog-py/tests/test_agent_critical_matrix.py`
- Modify: `tools/backlog-py/src/backlog_py/compat/inventory.py`
- Modify: `tools/backlog-py/tests/fixtures/oracle/manifest.yml`
- Modify: `tools/backlog-py/README.md`
- Modify: `backlog/tasks/task-244.1 - Write-Backlog.md-Python-compatibility-clone-implementation-plan.md`

- [ ] **Step 1: Write parity matrix doc**

Document every agent-critical CLI/MCP operation, expected command/tool/resource, status, and fixture coverage.

- [ ] **Step 2: Expand inventory and manifest coverage**

Update `tools/backlog-py/src/backlog_py/compat/inventory.py` and
`tools/backlog-py/tests/fixtures/oracle/manifest.yml` so every operation in the
Agent-Critical Parity Checklist is represented either as `golden-required` with
a fixture or as an explicit later-milestone blocker. The matrix must not silently
ignore task create/edit, document operations, milestone operations, or Definition
of Done tools.

- [ ] **Step 3: Write matrix test**

The test should fail if any agent-critical item lacks fixture coverage or implementation status:

```python
from pathlib import Path

from backlog_py.compat.inventory import load_builtin_inventory
from backlog_py.oracle.manifest import load_oracle_manifest


def test_agent_critical_inventory_has_fixture_coverage():
    inventory = load_builtin_inventory()
    manifest = load_oracle_manifest(Path("tools/backlog-py/tests/fixtures/oracle/manifest.yml"))
    fixture_names = {fixture.name for fixture in manifest.fixtures}

    missing = [
        item.name
        for item in inventory.items
        if item.classification == "golden-required" and item.name not in fixture_names
    ]
    assert missing == []
```

- [ ] **Step 4: Run full local validation**

Run:

```bash
source .venv/bin/activate
python -m pytest tools/backlog-py/tests -v
python -m bandit -r tools/backlog-py/src -f json -o /tmp/bandit_backlog_py.json
git diff --check
```

Expected: all checks pass.

- [ ] **Step 5: Run copied-repo mutation smoke**

Run against a temporary copy only:

```bash
source .venv/bin/activate
tmpdir="$(mktemp -d)"
cp -R backlog "$tmpdir/backlog"
backlog-py --cwd "$tmpdir" task create "Temporary smoke task" --status "To Do" --plain
backlog-py --cwd "$tmpdir" task list --plain
```

Expected: commands exit 0 and no live repository files change.

- [ ] **Step 6: Commit Task 9**

```bash
git add tools/backlog-py "backlog/tasks/task-244.1 - Write-Backlog.md-Python-compatibility-clone-implementation-plan.md"
git commit -m "Document Backlog.md agent parity gate"
```

## Task 10: Browser And Interactive Deferral Decision

**Files:**
- Create: `tools/backlog-py/docs/browser-parity.md`
- Create: `tools/backlog-py/docs/interactive-deferrals.md`
- Modify: `tools/backlog-py/README.md`
- Modify: `backlog/tasks/task-244.1 - Write-Backlog.md-Python-compatibility-clone-implementation-plan.md`

- [ ] **Step 1: Document browser parity requirements**

Cover:

- responsive Kanban board
- drag-and-drop
- task create/edit forms
- acceptance criteria editor
- Definition of Done settings
- real-time updates
- archive confirmations
- rich Markdown editing
- mermaid rendering
- service mode
- mobile behavior

- [ ] **Step 2: Classify each browser item**

Classify each as:

- required for full clone
- not required for agent cutover
- intentionally deferred
- rejected with reason

- [ ] **Step 3: Document interactive CLI/TUI deferrals**

Cover:

- colored output exactness
- interactive board
- overview TUI
- editor launch
- shell completions
- `onStatusChange`
- auto-commit and hook bypass
- remote operations

- [ ] **Step 4: Run docs verification**

Run:

```bash
rg -n "drag-and-drop|onStatusChange|service mode|auto-commit|hook bypass" tools/backlog-py/docs
git diff --check
```

Expected: search finds documented decisions and diff check passes.

- [ ] **Step 5: Commit Task 10**

```bash
git add tools/backlog-py "backlog/tasks/task-244.1 - Write-Backlog.md-Python-compatibility-clone-implementation-plan.md"
git commit -m "Document Backlog.md browser parity deferrals"
```

## Final Verification Before Any Cutover

Do not symlink or alias `backlog` to the Python implementation until all of these pass:

```bash
source .venv/bin/activate
python -m pytest tools/backlog-py/tests -v
python -m bandit -r tools/backlog-py/src -f json -o /tmp/bandit_backlog_py.json
backlog-py --cwd /Users/macbook-dev/Documents/GitHub/tldw_server2 task list --plain
backlog-py --cwd /Users/macbook-dev/Documents/GitHub/tldw_server2 task TASK-1 --plain
backlog-py --cwd /Users/macbook-dev/Documents/GitHub/tldw_server2 search "Backlog.md" --plain
backlog-py --cwd /Users/macbook-dev/Documents/GitHub/tldw_server2 config list
git diff --check
```

Expected:

- pytest passes
- Bandit returns zero new findings in touched Python code
- read-only live-repo commands exit 0
- no live repository backlog files change except intentional task tracking updates
- existing `backlog` upstream command remains available as rollback until user explicitly approves cutover

## Execution Notes

- Keep `backlog-py` as the console command until the user explicitly approves PATH cutover.
- Never run mutation commands against `/Users/macbook-dev/Documents/GitHub/tldw_server2/backlog` until copied-fixture mutation tests pass and the user approves a live smoke.
- If network is needed to install an MCP/browser/search dependency, stop and request approval before adding the dependency.
- If upstream behavior disagrees with the design, update the inventory and ask whether to preserve upstream behavior or keep the Python security constraint.
- Commit after each task. Include the relevant Backlog task update in each commit.
