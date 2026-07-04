from __future__ import annotations

import argparse
import ast
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

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


@dataclass(frozen=True)
class ImportRecord:
    target_name: str
    module: str
    imported_name: str | None
    path: str
    line: int
    import_kind: str


def _relpath(path: Path, project_root: Path) -> str:
    return path.resolve().relative_to(project_root.resolve()).as_posix()


def _record_sort_key(record: ImportRecord) -> tuple[str, int, str, str, str, str]:
    return (
        record.path,
        record.line,
        record.import_kind,
        record.target_name,
        record.module,
        record.imported_name or "",
    )


def _module_name_for_path(path: Path, project_root: Path) -> str:
    module_path = path.resolve().relative_to(project_root.resolve()).with_suffix("")
    parts = list(module_path.parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _resolve_from_import_module(path: Path, project_root: Path, node: ast.ImportFrom) -> str | None:
    if node.level == 0:
        return node.module

    current_module = _module_name_for_path(path, project_root)
    if not current_module:
        return node.module

    current_parts = current_module.split(".")
    package_parts = current_parts if path.name == "__init__.py" else current_parts[:-1]
    if node.level > 1:
        levels_up = node.level - 1
        if levels_up > len(package_parts):
            return node.module
        package_parts = package_parts[:-levels_up]

    module_parts = package_parts
    if node.module:
        module_parts = [*module_parts, *node.module.split(".")]
    return ".".join(module_parts)


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
            module_name = _resolve_from_import_module(path, project_root, node)
            if module_name is None:
                continue
            for alias in node.names:
                for target_name, target_module in targets.items():
                    if _from_import_matches(module_name, alias.name, target_module):
                        records.append(
                            ImportRecord(
                                target_name=target_name,
                                module=module_name,
                                imported_name=alias.name,
                                path=relative_path,
                                line=node.lineno,
                                import_kind="from",
                            )
                        )
    return sorted(records, key=_record_sort_key)


def inventory_for_roots(project_root: Path, roots: Iterable[Path], *, targets: dict[str, str]) -> dict[str, list[ImportRecord]]:
    inventory: dict[str, list[ImportRecord]] = {target_name: [] for target_name in targets}
    for path in discover_python_files(roots):
        for record in scan_file(path, project_root=project_root, targets=targets):
            inventory[record.target_name].append(record)
    return {target: sorted(records, key=_record_sort_key) for target, records in inventory.items()}


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


def _resolve_output_path(raw_path: str, project_root: Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = project_root / path
    resolved_path = path.resolve()
    try:
        resolved_path.relative_to(project_root)
    except ValueError as exc:
        raise ValueError(f"Output path must be under project root: {resolved_path}") from exc
    return resolved_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Web_Scraping refactor import inventory artifacts.")
    parser.add_argument("--root", default=".", help="Repository root path.")
    parser.add_argument("--json", required=True, help="Output JSON inventory path.")
    parser.add_argument("--markdown", required=True, help="Output Markdown inventory path.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = Path(args.root).resolve()
    json_path = _resolve_output_path(args.json, project_root)
    markdown_path = _resolve_output_path(args.markdown, project_root)
    roots = default_scan_roots(project_root)
    inventory = inventory_for_roots(project_root, roots, targets=TARGET_MODULES)
    write_json_inventory(json_path, inventory, roots=[_relpath(root, project_root) for root in roots])
    write_markdown_inventory(markdown_path, inventory, json_path=_relpath(json_path, project_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
