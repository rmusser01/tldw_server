# Web Scraping Phase 0 Import Inventory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Phase 0 import inventory, compatibility map, and dependency guardrails needed before moving Web_Scraping runtime behavior into new packages.

**Architecture:** Add a repository-local inventory helper that scans Python imports from `tldw_Server_API/app` and `tldw_Server_API/tests`, writes stable JSON and Markdown artifacts, and powers tests that keep the inventory current. Add guardrail tests that prevent future internal Web_Scraping packages from depending on legacy wrapper files such as `Article_Extractor_Lib.py`, `enhanced_web_scraping.py`, and `WebSearch_APIs.py`.

**Tech Stack:** Python stdlib `ast`, `dataclasses`, `json`, `pathlib`; pytest; existing Backlog.md task tracking.

---

## Scope

This plan implements only Phase 0 from `Docs/superpowers/specs/2026-07-03-web-scraping-refactor-design.md`.

It must not move runtime behavior, rename modules, change API responses, or alter scraping/search execution. It prepares later phases by making the current import surface explicit and adding guardrails.

## File Map

- Create `Helper_Scripts/web_scraping_refactor_inventory.py`: AST-based scanner and artifact writer.
- Create `Docs/Design/WebScraping_Refactor_Import_Inventory.md`: generated human-readable import inventory and compatibility map.
- Create `Docs/Design/web_scraping_refactor_import_inventory.json`: generated machine-readable import inventory.
- Create `tldw_Server_API/tests/WebScraping/test_refactor_import_inventory.py`: scanner unit tests, inventory freshness test, and dependency guardrail tests.
- Modify `Docs/Design/WebScraping.md`: link to the Phase 0 inventory artifact.
- Modify `backlog/tasks/task-12026 - Plan-Web-Scraping-refactor-Phase-0-import-inventory-and-guardrails.md`: record execution notes and verification.

## Target Modules To Inventory

Use these exact module targets in the helper:

```python
TARGET_MODULES = {
    "Article_Extractor_Lib": "tldw_Server_API.app.core.Web_Scraping.Article_Extractor_Lib",
    "enhanced_web_scraping": "tldw_Server_API.app.core.Web_Scraping.enhanced_web_scraping",
    "WebSearch_APIs": "tldw_Server_API.app.core.Web_Scraping.WebSearch_APIs",
    "scraper_analyzers": "tldw_Server_API.app.core.Web_Scraping.scraper_analyzers",
    "url_utils": "tldw_Server_API.app.core.Web_Scraping.url_utils",
    "ua_profiles": "tldw_Server_API.app.core.Web_Scraping.ua_profiles",
    "handlers": "tldw_Server_API.app.core.Web_Scraping.handlers",
    "scraper_router": "tldw_Server_API.app.core.Web_Scraping.scraper_router",
    "scoring": "tldw_Server_API.app.core.Web_Scraping.scoring",
}
```

Use these exact legacy wrappers for the guardrail:

```python
LEGACY_WRAPPER_MODULES = {
    "tldw_Server_API.app.core.Web_Scraping.Article_Extractor_Lib",
    "tldw_Server_API.app.core.Web_Scraping.enhanced_web_scraping",
    "tldw_Server_API.app.core.Web_Scraping.WebSearch_APIs",
}
```

Use these exact future package names for the guardrail:

```python
NEW_INTERNAL_PACKAGES = {
    "contracts",
    "config",
    "policy",
    "runtime",
    "routing",
    "content",
    "sources",
    "preflight",
    "extraction",
    "crawl",
    "cookies",
    "search",
    "search_providers",
    "orchestration",
    "jobs",
}
```

## Task 1: Inventory Scanner Helper

**Files:**
- Create: `Helper_Scripts/web_scraping_refactor_inventory.py`
- Create: `tldw_Server_API/tests/WebScraping/test_refactor_import_inventory.py`

- [ ] **Step 1: Write failing scanner unit tests**

Create `tldw_Server_API/tests/WebScraping/test_refactor_import_inventory.py` with this initial content:

```python
from __future__ import annotations

import json
from pathlib import Path

import pytest

from Helper_Scripts.web_scraping_refactor_inventory import (
    LEGACY_WRAPPER_MODULES,
    NEW_INTERNAL_PACKAGES,
    TARGET_MODULES,
    ImportRecord,
    inventory_for_roots,
    scan_file,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
INVENTORY_JSON = REPO_ROOT / "Docs/Design/web_scraping_refactor_import_inventory.json"
NEW_PACKAGE_ROOT = REPO_ROOT / "tldw_Server_API/app/core/Web_Scraping"


@pytest.mark.unit
def test_scan_file_finds_direct_imports(tmp_path: Path) -> None:
    source = tmp_path / "caller.py"
    source.write_text(
        "\n".join(
            [
                "from tldw_Server_API.app.core.Web_Scraping.Article_Extractor_Lib import scrape_article",
                "from tldw_Server_API.app.core.Web_Scraping import WebSearch_APIs as ws",
                "import tldw_Server_API.app.core.Web_Scraping.enhanced_web_scraping as enhanced",
            ]
        ),
        encoding="utf-8",
    )

    records = scan_file(source, project_root=tmp_path, targets=TARGET_MODULES)

    assert [
        (record.target_name, record.module, record.imported_name, record.line)
        for record in records
    ] == [
        (
            "Article_Extractor_Lib",
            "tldw_Server_API.app.core.Web_Scraping.Article_Extractor_Lib",
            "scrape_article",
            1,
        ),
        (
            "WebSearch_APIs",
            "tldw_Server_API.app.core.Web_Scraping",
            "WebSearch_APIs",
            2,
        ),
        (
            "enhanced_web_scraping",
            "tldw_Server_API.app.core.Web_Scraping.enhanced_web_scraping",
            None,
            3,
        ),
    ]


@pytest.mark.unit
def test_inventory_for_roots_groups_records_by_target(tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    tests_dir = tmp_path / "tests"
    app_dir.mkdir()
    tests_dir.mkdir()
    (app_dir / "service.py").write_text(
        "from tldw_Server_API.app.core.Web_Scraping.WebSearch_APIs import perform_websearch\n",
        encoding="utf-8",
    )
    (tests_dir / "test_service.py").write_text(
        "from tldw_Server_API.app.core.Web_Scraping.Article_Extractor_Lib import scrape_article\n",
        encoding="utf-8",
    )

    inventory = inventory_for_roots(tmp_path, [app_dir, tests_dir], targets=TARGET_MODULES)

    assert inventory["WebSearch_APIs"][0].path == "app/service.py"
    assert inventory["Article_Extractor_Lib"][0].path == "tests/test_service.py"


@pytest.mark.unit
def test_new_internal_packages_constant_excludes_legacy_wrappers() -> None:
    assert "Article_Extractor_Lib" not in NEW_INTERNAL_PACKAGES
    assert "enhanced_web_scraping" not in NEW_INTERNAL_PACKAGES
    assert "WebSearch_APIs" not in NEW_INTERNAL_PACKAGES
    assert "tldw_Server_API.app.core.Web_Scraping.Article_Extractor_Lib" in LEGACY_WRAPPER_MODULES
```

- [ ] **Step 2: Run scanner tests to verify they fail**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/WebScraping/test_refactor_import_inventory.py -q
```

Expected: fails with `ModuleNotFoundError: No module named 'Helper_Scripts.web_scraping_refactor_inventory'`.

- [ ] **Step 3: Implement the inventory helper**

Create `Helper_Scripts/web_scraping_refactor_inventory.py` with:

```python
from __future__ import annotations

import argparse
import ast
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


TARGET_MODULES: dict[str, str] = {
    "Article_Extractor_Lib": "tldw_Server_API.app.core.Web_Scraping.Article_Extractor_Lib",
    "enhanced_web_scraping": "tldw_Server_API.app.core.Web_Scraping.enhanced_web_scraping",
    "WebSearch_APIs": "tldw_Server_API.app.core.Web_Scraping.WebSearch_APIs",
    "scraper_analyzers": "tldw_Server_API.app.core.Web_Scraping.scraper_analyzers",
    "url_utils": "tldw_Server_API.app.core.Web_Scraping.url_utils",
    "ua_profiles": "tldw_Server_API.app.core.Web_Scraping.ua_profiles",
    "handlers": "tldw_Server_API.app.core.Web_Scraping.handlers",
    "scraper_router": "tldw_Server_API.app.core.Web_Scraping.scraper_router",
    "scoring": "tldw_Server_API.app.core.Web_Scraping.scoring",
}

LEGACY_WRAPPER_MODULES: set[str] = {
    "tldw_Server_API.app.core.Web_Scraping.Article_Extractor_Lib",
    "tldw_Server_API.app.core.Web_Scraping.enhanced_web_scraping",
    "tldw_Server_API.app.core.Web_Scraping.WebSearch_APIs",
}

NEW_INTERNAL_PACKAGES: set[str] = {
    "contracts",
    "config",
    "policy",
    "runtime",
    "routing",
    "content",
    "sources",
    "preflight",
    "extraction",
    "crawl",
    "cookies",
    "search",
    "search_providers",
    "orchestration",
    "jobs",
}

SKIP_DIR_NAMES: set[str] = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "node_modules",
}


@dataclass(frozen=True, order=True)
class ImportRecord:
    target_name: str
    module: str
    imported_name: str | None
    path: str
    line: int
    import_kind: str


def _relpath(path: Path, project_root: Path) -> str:
    return path.resolve().relative_to(project_root.resolve()).as_posix()


def _module_matches(module_name: str, target_module: str) -> bool:
    return module_name == target_module or module_name.startswith(f"{target_module}.")


def _from_import_matches(module_name: str, imported_name: str, target_module: str) -> bool:
    imported_module = f"{module_name}.{imported_name}" if module_name else imported_name
    return _module_matches(module_name, target_module) or _module_matches(imported_module, target_module)


def discover_python_files(roots: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if root.is_file() and root.suffix == ".py":
            files.append(root)
            continue
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if any(part in SKIP_DIR_NAMES for part in path.parts):
                continue
            files.append(path)
    return sorted(files)


def scan_file(path: Path, *, project_root: Path, targets: dict[str, str]) -> list[ImportRecord]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError:
        return []

    records: list[ImportRecord] = []
    relative_path = _relpath(path, project_root)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for target_name, target_module in targets.items():
                    if _module_matches(alias.name, target_module):
                        records.append(
                            ImportRecord(
                                target_name=target_name,
                                module=alias.name,
                                imported_name=None,
                                path=relative_path,
                                line=node.lineno,
                                import_kind="import",
                            )
                        )
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            for alias in node.names:
                for target_name, target_module in targets.items():
                    if _from_import_matches(node.module, alias.name, target_module):
                        records.append(
                            ImportRecord(
                                target_name=target_name,
                                module=node.module,
                                imported_name=alias.name,
                                path=relative_path,
                                line=node.lineno,
                                import_kind="from",
                            )
                        )
    return sorted(records)


def inventory_for_roots(project_root: Path, roots: Iterable[Path], *, targets: dict[str, str]) -> dict[str, list[ImportRecord]]:
    inventory: dict[str, list[ImportRecord]] = {target_name: [] for target_name in targets}
    for path in discover_python_files(roots):
        for record in scan_file(path, project_root=project_root, targets=targets):
            inventory[record.target_name].append(record)
    return {target: sorted(records) for target, records in inventory.items()}


def _records_for_json(inventory: dict[str, list[ImportRecord]]) -> dict[str, list[dict[str, object]]]:
    return {target: [asdict(record) for record in records] for target, records in sorted(inventory.items())}


def write_json_inventory(path: Path, inventory: dict[str, list[ImportRecord]], *, roots: list[str]) -> None:
    payload = {
        "schema_version": 1,
        "description": "Current imports of Web_Scraping legacy modules and shared helpers.",
        "scan_roots": roots,
        "targets": TARGET_MODULES,
        "records": _records_for_json(inventory),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown_inventory(path: Path, inventory: dict[str, list[ImportRecord]], *, json_path: str) -> None:
    lines: list[str] = [
        "# Web_Scraping Refactor Import Inventory",
        "",
        "This file is generated by `Helper_Scripts/web_scraping_refactor_inventory.py`.",
        "It records current imports that must remain compatible during the modular refactor.",
        "",
        f"Machine-readable inventory: `{json_path}`",
        "",
        "## Compatibility Targets",
        "",
    ]
    for target_name, target_module in sorted(TARGET_MODULES.items()):
        lines.append(f"- `{target_name}`: `{target_module}`")
    lines.extend(["", "## Import Records", ""])
    for target_name, records in sorted(inventory.items()):
        lines.append(f"### {target_name}")
        lines.append("")
        if not records:
            lines.append("No current imports found.")
            lines.append("")
            continue
        lines.append("| Path | Line | Import |")
        lines.append("| --- | ---: | --- |")
        for record in records:
            imported = record.module
            if record.imported_name:
                imported = f"{record.module}.{record.imported_name}"
            lines.append(f"| `{record.path}` | {record.line} | `{imported}` |")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def default_scan_roots(project_root: Path) -> list[Path]:
    return [
        project_root / "tldw_Server_API/app",
        project_root / "tldw_Server_API/tests",
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Web_Scraping refactor import inventory artifacts.")
    parser.add_argument("--root", default=".", help="Repository root path.")
    parser.add_argument("--json", required=True, help="Output JSON inventory path.")
    parser.add_argument("--markdown", required=True, help="Output Markdown inventory path.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = Path(args.root).resolve()
    roots = default_scan_roots(project_root)
    inventory = inventory_for_roots(project_root, roots, targets=TARGET_MODULES)
    write_json_inventory(Path(args.json), inventory, roots=[_relpath(root, project_root) for root in roots])
    write_markdown_inventory(Path(args.markdown), inventory, json_path=_relpath(Path(args.json), project_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run scanner unit tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/WebScraping/test_refactor_import_inventory.py -q
```

Expected: `3 passed`.

- [ ] **Step 5: Commit scanner helper**

Run:

```bash
git add Helper_Scripts/web_scraping_refactor_inventory.py tldw_Server_API/tests/WebScraping/test_refactor_import_inventory.py
git commit -m "test: add web scraping refactor import scanner"
```

## Task 2: Generated Inventory Artifacts

**Files:**
- Modify: `tldw_Server_API/tests/WebScraping/test_refactor_import_inventory.py`
- Create: `Docs/Design/web_scraping_refactor_import_inventory.json`
- Create: `Docs/Design/WebScraping_Refactor_Import_Inventory.md`

- [ ] **Step 1: Add the failing inventory freshness test**

Append this test to `tldw_Server_API/tests/WebScraping/test_refactor_import_inventory.py`:

```python

@pytest.mark.unit
def test_import_inventory_artifact_matches_current_import_surface() -> None:
    assert INVENTORY_JSON.exists(), "Run Helper_Scripts/web_scraping_refactor_inventory.py to create the inventory artifact"

    expected = json.loads(INVENTORY_JSON.read_text(encoding="utf-8"))
    current = inventory_for_roots(
        REPO_ROOT,
        [
            REPO_ROOT / "tldw_Server_API/app",
            REPO_ROOT / "tldw_Server_API/tests",
        ],
        targets=TARGET_MODULES,
    )
    current_records = {
        target: [
            {
                "import_kind": record.import_kind,
                "imported_name": record.imported_name,
                "line": record.line,
                "module": record.module,
                "path": record.path,
                "target_name": record.target_name,
            }
            for record in records
        ]
        for target, records in sorted(current.items())
    }

    assert expected["records"] == current_records
```

- [ ] **Step 2: Run the freshness test to verify it fails**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/WebScraping/test_refactor_import_inventory.py::test_import_inventory_artifact_matches_current_import_surface -q
```

Expected: fails with `AssertionError: Run Helper_Scripts/web_scraping_refactor_inventory.py to create the inventory artifact`.

- [ ] **Step 3: Generate inventory artifacts**

Run:

```bash
source .venv/bin/activate && python Helper_Scripts/web_scraping_refactor_inventory.py --root . --json Docs/Design/web_scraping_refactor_import_inventory.json --markdown Docs/Design/WebScraping_Refactor_Import_Inventory.md
```

Expected: command exits 0 and writes both inventory files.

- [ ] **Step 4: Inspect inventory for obvious scan errors**

Run:

```bash
python -m json.tool Docs/Design/web_scraping_refactor_import_inventory.json >/tmp/web_scraping_refactor_import_inventory.pretty.json
```

Expected: command exits 0. Then read the Markdown summary and confirm it includes imports from RAG, MCP, Workflows, Watchlists, Collections, services, and WebScraping/WebSearch tests where present.

- [ ] **Step 5: Run inventory tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/WebScraping/test_refactor_import_inventory.py -q
```

Expected: `4 passed`.

- [ ] **Step 6: Commit generated inventory artifacts**

Run:

```bash
git add Helper_Scripts/web_scraping_refactor_inventory.py tldw_Server_API/tests/WebScraping/test_refactor_import_inventory.py Docs/Design/web_scraping_refactor_import_inventory.json Docs/Design/WebScraping_Refactor_Import_Inventory.md
git commit -m "docs: inventory web scraping refactor imports"
```

## Task 3: Legacy Wrapper Dependency Guardrail

**Files:**
- Modify: `tldw_Server_API/tests/WebScraping/test_refactor_import_inventory.py`

- [ ] **Step 1: Add guardrail tests**

Append these tests to `tldw_Server_API/tests/WebScraping/test_refactor_import_inventory.py`:

```python

def _new_internal_package_files() -> list[Path]:
    files: list[Path] = []
    for package_name in sorted(NEW_INTERNAL_PACKAGES):
        package_root = NEW_PACKAGE_ROOT / package_name
        if not package_root.exists():
            continue
        files.extend(sorted(package_root.rglob("*.py")))
    return files


@pytest.mark.unit
def test_guardrail_detects_legacy_wrapper_imports_in_new_internal_package(tmp_path: Path) -> None:
    package_root = tmp_path / "tldw_Server_API/app/core/Web_Scraping/runtime"
    package_root.mkdir(parents=True)
    module = package_root / "fetch.py"
    module.write_text(
        "from tldw_Server_API.app.core.Web_Scraping.Article_Extractor_Lib import scrape_article\n",
        encoding="utf-8",
    )

    records = scan_file(module, project_root=tmp_path, targets={name: name for name in LEGACY_WRAPPER_MODULES})

    assert len(records) == 1
    assert records[0].target_name == "tldw_Server_API.app.core.Web_Scraping.Article_Extractor_Lib"


@pytest.mark.unit
def test_new_internal_web_scraping_packages_do_not_import_legacy_wrappers() -> None:
    violations: list[str] = []
    targets = {name: name for name in LEGACY_WRAPPER_MODULES}
    for path in _new_internal_package_files():
        for record in scan_file(path, project_root=REPO_ROOT, targets=targets):
            violations.append(f"{record.path}:{record.line} imports {record.module}")

    assert violations == []
```

- [ ] **Step 2: Run guardrail tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/WebScraping/test_refactor_import_inventory.py::test_guardrail_detects_legacy_wrapper_imports_in_new_internal_package tldw_Server_API/tests/WebScraping/test_refactor_import_inventory.py::test_new_internal_web_scraping_packages_do_not_import_legacy_wrappers -q
```

Expected: `2 passed`. The real-package guardrail passes before new packages exist and will fail later if future internal packages import the legacy wrappers.

- [ ] **Step 3: Run full inventory test file**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/WebScraping/test_refactor_import_inventory.py -q
```

Expected: `6 passed`.

- [ ] **Step 4: Commit guardrail tests**

Run:

```bash
git add tldw_Server_API/tests/WebScraping/test_refactor_import_inventory.py
git commit -m "test: guard web scraping refactor dependencies"
```

## Task 4: Link Inventory From Design Docs

**Files:**
- Modify: `Docs/Design/WebScraping.md`
- Modify: `Docs/superpowers/specs/2026-07-03-web-scraping-refactor-design.md`

- [ ] **Step 1: Update design docs**

In `Docs/Design/WebScraping.md`, add this paragraph under `## References` before the external links:

```markdown
- `Docs/Design/WebScraping_Refactor_Import_Inventory.md` records the current compatibility import surface for the modular refactor. Update it with `Helper_Scripts/web_scraping_refactor_inventory.py` whenever Web_Scraping compatibility imports are added, removed, or migrated.
```

In `Docs/superpowers/specs/2026-07-03-web-scraping-refactor-design.md`, add this paragraph at the end of `### Phase 0: Import Inventory And Guardrails`:

```markdown
The Phase 0 implementation should produce `Docs/Design/WebScraping_Refactor_Import_Inventory.md` and `Docs/Design/web_scraping_refactor_import_inventory.json`, with tests that fail when the inventory no longer matches current imports.
```

- [ ] **Step 2: Run documentation checks**

Run:

```bash
rg -n "WebScraping_Refactor_Import_Inventory|web_scraping_refactor_import_inventory" Docs/Design/WebScraping.md Docs/superpowers/specs/2026-07-03-web-scraping-refactor-design.md
```

Expected: both docs reference the inventory artifacts.

- [ ] **Step 3: Run scoped whitespace check**

Run:

```bash
git diff --check -- Docs/Design/WebScraping.md Docs/superpowers/specs/2026-07-03-web-scraping-refactor-design.md
```

Expected: no output and exit 0.

- [ ] **Step 4: Commit documentation links**

Run:

```bash
git add Docs/Design/WebScraping.md Docs/superpowers/specs/2026-07-03-web-scraping-refactor-design.md
git commit -m "docs: link web scraping refactor inventory"
```

## Task 5: Final Verification And Tracking

**Files:**
- Modify: `backlog/tasks/task-12026 - Plan-Web-Scraping-refactor-Phase-0-import-inventory-and-guardrails.md`

- [ ] **Step 1: Run focused tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/WebScraping/test_refactor_import_inventory.py -q
```

Expected: `6 passed`.

- [ ] **Step 2: Run compile check for the helper**

Run:

```bash
source .venv/bin/activate && python -m py_compile Helper_Scripts/web_scraping_refactor_inventory.py
```

Expected: command exits 0.

- [ ] **Step 3: Run Bandit for touched executable code**

Run:

```bash
source .venv/bin/activate && python -m bandit Helper_Scripts/web_scraping_refactor_inventory.py -f json -o /tmp/bandit_web_scraping_phase0_inventory.json
```

Expected: no high or medium findings. Review `/tmp/bandit_web_scraping_phase0_inventory.json` before closing.

- [ ] **Step 4: Run scoped whitespace check**

Run:

```bash
git diff --check -- Helper_Scripts/web_scraping_refactor_inventory.py tldw_Server_API/tests/WebScraping/test_refactor_import_inventory.py Docs/Design/WebScraping_Refactor_Import_Inventory.md Docs/Design/web_scraping_refactor_import_inventory.json Docs/Design/WebScraping.md Docs/superpowers/specs/2026-07-03-web-scraping-refactor-design.md "backlog/tasks/task-12026 - Plan-Web-Scraping-refactor-Phase-0-import-inventory-and-guardrails.md"
```

Expected: no output and exit 0.

- [ ] **Step 5: Update Backlog task**

Use the Backlog MCP task edit tool to:

- Check all acceptance criteria.
- Check all Definition of Done items.
- Add verification commands and results.
- Add final summary:

```markdown
Implemented Phase 0 import inventory and guardrails for the Web_Scraping modular refactor. Added an AST inventory helper, generated JSON/Markdown import artifacts, guardrail tests for future internal package dependencies, docs links, and focused verification.
```

- [ ] **Step 6: Commit tracking update**

Run:

```bash
git add "backlog/tasks/task-12026 - Plan-Web-Scraping-refactor-Phase-0-import-inventory-and-guardrails.md"
git commit -m "chore: close web scraping phase 0 inventory task"
```

## Plan Self-Review

- Spec coverage: This plan implements Phase 0 from the approved refactor spec: import inventory, compatibility mapping, dependency guardrails, docs links, focused tests, Bandit, and Backlog tracking. It deliberately does not move runtime behavior, preserving preflight analyzer functionality and existing dict-shaped contracts.
- Placeholder scan: No unfinished-work markers, placeholder marker, or unspecified implementation step remains. Generated inventory data is produced by a concrete helper command rather than handwritten into the plan.
- Type consistency: `TARGET_MODULES`, `LEGACY_WRAPPER_MODULES`, `NEW_INTERNAL_PACKAGES`, `ImportRecord`, `scan_file`, and `inventory_for_roots` are defined before tests use them, and all paths match the File Map.
