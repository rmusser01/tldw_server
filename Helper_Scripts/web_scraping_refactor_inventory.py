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
