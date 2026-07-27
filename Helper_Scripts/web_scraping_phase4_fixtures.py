#!/usr/bin/env python3
"""Generate immutable Phase 4 predecessor behavior fixtures."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import subprocess  # nosec B404
import sys
import tempfile
import uuid
import warnings
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[1]
CASE_NAMES = (
    "article_orchestration_fakes",
    "content",
    "extraction",
    "metadata",
    "selectors",
)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Helper_Scripts.web_scraping_phase4.content_metadata import (  # noqa: E402
    build_content_cases,
    build_metadata_cases,
)
from Helper_Scripts.web_scraping_phase4.extraction import (  # noqa: E402
    build_extraction_cases,
)
from Helper_Scripts.web_scraping_phase4.orchestration import (  # noqa: E402
    build_article_cases,
)
from Helper_Scripts.web_scraping_phase4.selectors import (  # noqa: E402
    build_selector_cases,
)
from Helper_Scripts.web_scraping_phase4.shared import FIXED_ENV  # noqa: E402


def build_manifest(predecessor_commit: str, case_files: dict[str, str]) -> dict[str, object]:
    if re.fullmatch(r"[0-9a-f]{40}", predecessor_commit) is None:
        raise ValueError("predecessor_commit must be a full 40-character lowercase commit id")
    return {
        "schema_version": SCHEMA_VERSION,
        "predecessor_commit": predecessor_commit,
        "cases": dict(sorted(case_files.items())),
    }


def _run_git(source_root: Path, *args: str) -> str:
    git_executable = shutil.which("git")
    if git_executable is None:
        raise OSError("git executable not found")
    completed = subprocess.run(  # nosec B603
        [git_executable, *args],
        cwd=source_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _validate_source_root(predecessor_commit: str, source_root: Path) -> Path:
    build_manifest(predecessor_commit, {})
    resolved_root = source_root.resolve(strict=True)
    top_level = Path(_run_git(resolved_root, "rev-parse", "--show-toplevel")).resolve()
    if top_level != resolved_root:
        raise ValueError(f"source-root must be the git worktree root: {resolved_root}")

    head = _run_git(resolved_root, "rev-parse", "HEAD")
    if predecessor_commit != head:
        raise ValueError(f"predecessor_commit {predecessor_commit} does not match source-root HEAD {head}")

    status = _run_git(
        resolved_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--ignored=no",
    )
    if status:
        raise ValueError(f"source-root is not clean: {resolved_root}\n{status}")
    return resolved_root


def _remove_loaded_production_modules() -> dict[str, Any]:
    removed: dict[str, Any] = {}
    for name in tuple(sys.modules):
        if name == "tldw_Server_API" or name.startswith("tldw_Server_API."):
            removed[name] = sys.modules.pop(name)
    return removed


def _assert_production_modules_under(source_root: Path) -> None:
    for name, module in tuple(sys.modules.items()):
        if name != "tldw_Server_API" and not name.startswith("tldw_Server_API."):
            continue
        module_file = getattr(module, "__file__", None)
        if module_file is not None:
            resolved_locations = (Path(module_file).resolve(),)
        else:
            module_path = getattr(module, "__path__", None)
            if module_path is None:
                raise RuntimeError(f"Imported predecessor module {name} has no resolvable path")
            resolved_locations = tuple(Path(location).resolve() for location in module_path)
            if not resolved_locations:
                raise RuntimeError(f"Imported predecessor module {name} has an empty package path")
        outside_root = [location for location in resolved_locations if not location.is_relative_to(source_root)]
        if outside_root:
            raise RuntimeError(
                f"Imported predecessor module {name} resolved outside source-root: "
                f"{', '.join(map(str, outside_root))}"
            )


@contextmanager
def _predecessor_modules(source_root: Path) -> Iterator[tuple[Any, Any, Any, Any]]:
    previous_modules = _remove_loaded_production_modules()
    previous_path = list(sys.path)
    sys.path.insert(0, str(source_root))
    try:
        from tldw_Server_API.app.core.Watchlists import fetchers
        from tldw_Server_API.app.core.Web_Scraping import Article_Extractor_Lib as article
        from tldw_Server_API.app.core.Web_Scraping.runtime import FetchResponse, PolicyDecision

        _assert_production_modules_under(source_root)
        yield article, fetchers, FetchResponse, PolicyDecision
        _assert_production_modules_under(source_root)
    finally:
        _remove_loaded_production_modules()
        sys.modules.update(previous_modules)
        sys.path[:] = previous_path


def build_case_payloads(source_root: Path) -> dict[str, dict[str, Any]]:
    with _predecessor_modules(source_root) as modules:
        article, fetchers, FetchResponse, PolicyDecision = modules
        try:
            with patch.dict(os.environ, FIXED_ENV, clear=False):
                fetchers.reload_selector_guardrails_from_env()
                payloads = {
                    "article_orchestration_fakes": {
                        "category": "article_orchestration_fakes",
                        "cases": asyncio.run(build_article_cases(article, FetchResponse, PolicyDecision)),
                    },
                    "content": {
                        "category": "content",
                        "cases": build_content_cases(article),
                    },
                    "extraction": {
                        "category": "extraction",
                        "cases": build_extraction_cases(article),
                    },
                    "metadata": {
                        "category": "metadata",
                        "cases": build_metadata_cases(article),
                    },
                    "selectors": {
                        "category": "selectors",
                        "cases": build_selector_cases(fetchers),
                    },
                }
        finally:
            fetchers.reload_selector_guardrails_from_env()

    if set(payloads) != set(CASE_NAMES):
        raise RuntimeError("Fixture category set is incomplete")
    return payloads


def _write_json(path: Path, payload: object) -> None:
    encoded = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    path.write_text(encoded, encoding="utf-8", newline="\n")


def _write_fixture_set(
    output: Path,
    predecessor_commit: str,
    payloads: Mapping[str, object],
) -> None:
    case_files: dict[str, str] = {}
    for category in sorted(payloads):
        filename = f"{category}.json"
        _write_json(output / filename, payloads[category])
        case_files[category] = filename

    _write_json(output / "manifest.json", build_manifest(predecessor_commit, case_files))


def _validate_fixture_set(output: Path, predecessor_commit: str) -> None:
    case_files = {category: f"{category}.json" for category in CASE_NAMES}
    expected_names = {"manifest.json", *case_files.values()}
    actual_names = {path.name for path in output.iterdir() if path.is_file()}
    if actual_names != expected_names:
        raise RuntimeError(
            f"Generated fixture set is incomplete: expected {sorted(expected_names)}, " f"found {sorted(actual_names)}"
        )

    decoded: dict[str, object] = {}
    for path in sorted(output.iterdir()):
        if not path.is_file():
            raise RuntimeError(f"Generated fixture set contains a non-file entry: {path.name}")
        raw = path.read_bytes()
        payload = json.loads(raw.decode("ascii"))
        canonical = (json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode("ascii")
        if raw != canonical:
            raise RuntimeError(f"Generated fixture is not canonical JSON: {path.name}")
        decoded[path.name] = payload

    expected_manifest = build_manifest(predecessor_commit, case_files)
    if decoded["manifest.json"] != expected_manifest:
        raise RuntimeError("Generated fixture manifest does not match the requested provenance")
    for category, filename in case_files.items():
        payload = decoded[filename]
        if not isinstance(payload, dict) or payload.get("category") != category:
            raise RuntimeError(f"Generated fixture category is invalid: {filename}")
        cases = payload.get("cases")
        if not isinstance(cases, list) or not cases:
            raise RuntimeError(f"Generated fixture has no cases: {filename}")


def _replace_output_directory(staging: Path, output: Path) -> None:
    backup: Path | None = None
    if output.exists():
        if not output.is_dir():
            raise ValueError(f"output must be a directory: {output}")
        backup = output.with_name(f".{output.name}.backup-{uuid.uuid4().hex}")
        output.replace(backup)

    try:
        staging.replace(output)
    except BaseException:
        if backup is not None and backup.exists() and not output.exists():
            backup.replace(output)
        raise
    else:
        if backup is not None:
            try:
                shutil.rmtree(backup)
            except OSError:
                warnings.warn(
                    "Fixture output committed; backup cleanup failed. "
                    f"Backup retained at {backup.name!r} for manual cleanup.",
                    RuntimeWarning,
                    stacklevel=2,
                )


def generate_fixtures(
    predecessor_commit: str,
    output: Path,
    *,
    source_root: Path | None = None,
) -> None:
    resolved_source_root = _validate_source_root(
        predecessor_commit,
        source_root or REPO_ROOT,
    )
    try:
        payloads = build_case_payloads(resolved_source_root)
    finally:
        _validate_source_root(predecessor_commit, resolved_source_root)

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.staging-",
            dir=output.parent,
        )
    )
    try:
        _write_fixture_set(staging, predecessor_commit, payloads)
        _validate_fixture_set(staging, predecessor_commit)
        _replace_output_directory(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predecessor-commit", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-root", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        generate_fixtures(
            args.predecessor_commit,
            args.output,
            source_root=args.source_root,
        )
    except (OSError, RuntimeError, subprocess.CalledProcessError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
