from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from Helper_Scripts import web_scraping_phase4_fixtures as fixture_generator

from tldw_Server_API.app.core.Web_Scraping import Article_Extractor_Lib as legacy
from tldw_Server_API.app.core.Web_Scraping import content
from tldw_Server_API.app.core.Web_Scraping.content import formatting as formatting_module
from tldw_Server_API.app.core.Web_Scraping.content import metadata as metadata_module

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "phase4"
CONTENT_ROOT = Path(__file__).resolve().parents[2] / "app" / "core" / "Web_Scraping" / "content"
REPO_ROOT = Path(__file__).resolve().parents[3]

_CONTENT_CASE_NAMES = (
    "paragraph_and_inline_formatting",
    "mixed_block_and_paragraph_formatting",
)
_METADATA_CASE_NAMES = (
    "canonical_formatted_envelope",
    "canonical_envelope_inspection",
    "malformed_envelope_passes_through",
    "nesting_boundary_is_accepted",
    "nesting_over_boundary_is_rejected",
    "metadata_only_changes_do_not_change_body_hash",
    "body_changes_are_detected",
)
_HASH_PATTERN = re.compile(r"[0-9a-f]{64}")
_CONTENT_IMPORT_ALLOWLIST = {
    "__init__.py": {".formatting", ".metadata"},
    "formatting.py": {"bs4", "loguru"},
    "metadata.py": {"datetime", "hashlib", "json", "typing"},
}
_ISOLATION_BLOCKED_MODULES = (
    "tldw_Server_API.app.core.http_client",
    "tldw_Server_API.app.core.Metrics",
    "tldw_Server_API.app.core.Security",
)


def _assert_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    location: str,
) -> None:
    assert set(value) == expected, location


def _assert_exact_type(value: object, expected: type, *, location: str) -> None:
    assert type(value) is expected, location


def _validate_content_case(case: Mapping[str, Any], *, index: int) -> None:
    location = f"content.cases[{index}]"
    _assert_exact_keys(
        case,
        {"expected", "html", "name", "operation"},
        location=location,
    )
    for key in ("expected", "html", "name", "operation"):
        _assert_exact_type(case[key], str, location=f"{location}.{key}")
    assert case["operation"] == "convert_html_to_markdown", location


def _validate_metadata_case(case: Mapping[str, Any], *, index: int) -> None:
    location = f"metadata.cases[{index}]"
    operation = case.get("operation")
    _assert_exact_type(operation, str, location=f"{location}.operation")

    if operation == "format":
        _assert_exact_keys(
            case,
            {
                "additional_metadata",
                "content",
                "expected",
                "name",
                "operation",
                "pipeline",
                "url",
            },
            location=location,
        )
        for key in ("content", "expected", "name", "operation", "pipeline", "url"):
            _assert_exact_type(case[key], str, location=f"{location}.{key}")
        _assert_exact_type(
            case["additional_metadata"],
            dict,
            location=f"{location}.additional_metadata",
        )
        assert all(
            type(key) is str and type(value) is str for key, value in case["additional_metadata"].items()
        ), location
        assert case["expected"].count("<TIMESTAMP>") == 1, location
        return

    if operation == "inspect":
        _assert_exact_keys(
            case,
            {"content", "expected", "name", "operation"},
            location=location,
        )
        for key in ("content", "name", "operation"):
            _assert_exact_type(case[key], str, location=f"{location}.{key}")
        _assert_exact_type(case["expected"], dict, location=f"{location}.expected")
        expected = case["expected"]
        _assert_exact_keys(
            expected,
            {"clean_content", "content_hash", "has_metadata", "metadata", "stripped"},
            location=f"{location}.expected",
        )
        for key in ("clean_content", "content_hash", "stripped"):
            _assert_exact_type(expected[key], str, location=f"{location}.expected.{key}")
        _assert_exact_type(
            expected["has_metadata"],
            bool,
            location=f"{location}.expected.has_metadata",
        )
        _assert_exact_type(
            expected["metadata"],
            dict,
            location=f"{location}.expected.metadata",
        )
        assert _HASH_PATTERN.fullmatch(expected["content_hash"]), location
        return

    if operation == "content_changed":
        _assert_exact_keys(
            case,
            {"expected", "name", "new_content", "old_content", "operation"},
            location=location,
        )
        for key in ("name", "new_content", "old_content", "operation"):
            _assert_exact_type(case[key], str, location=f"{location}.{key}")
        _assert_exact_type(case["expected"], bool, location=f"{location}.expected")
        return

    raise AssertionError(f"Unknown metadata fixture operation at {location}: {operation!r}")


def _load_cases(category: str) -> list[dict[str, Any]]:
    with fixture_generator.fixture_publication_reader(
        FIXTURE_ROOT,
        source_root=REPO_ROOT,
    ) as locked_root:
        fixture_path = locked_root / f"{category}.json"
        payload = json.loads(fixture_path.read_text(encoding="ascii"))

    _assert_exact_type(payload, dict, location=fixture_path.name)
    _assert_exact_keys(payload, {"cases", "category"}, location=fixture_path.name)
    assert payload["category"] == category, fixture_path.name
    _assert_exact_type(payload["cases"], list, location=f"{fixture_path.name}.cases")
    assert payload["cases"], fixture_path.name

    validator = _validate_content_case if category == "content" else _validate_metadata_case
    for index, case in enumerate(payload["cases"]):
        _assert_exact_type(case, dict, location=f"{category}.cases[{index}]")
        validator(case, index=index)

    names = tuple(case["name"] for case in payload["cases"])
    expected_names = _CONTENT_CASE_NAMES if category == "content" else _METADATA_CASE_NAMES
    assert names == expected_names, fixture_path.name
    return payload["cases"]


def _case_by_name(cases: list[dict[str, Any]], name: str) -> dict[str, Any]:
    matching = [case for case in cases if case["name"] == name]
    assert len(matching) == 1, name
    return matching[0]


def _inspect(content_value: str) -> dict[str, Any]:
    metadata, clean_content = content.ContentMetadataHandler.extract_metadata(content_value)
    return {
        "clean_content": clean_content,
        "content_hash": content.ContentMetadataHandler.get_content_hash(content_value),
        "has_metadata": content.ContentMetadataHandler.has_metadata(content_value),
        "metadata": metadata,
        "stripped": content.ContentMetadataHandler.strip_metadata(content_value),
    }


def _max_json_nesting(envelope: str) -> int:
    metadata_text = envelope.split("[METADATA]", 1)[1].split("[/METADATA]", 1)[0]
    depth = 0
    maximum = 0
    in_string = False
    escaped = False
    for char in metadata_text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "[{":
            depth += 1
            maximum = max(maximum, depth)
        elif char in "]}":
            depth -= 1
    return maximum


def _import_targets(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    targets: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            targets.add(f"{'.' * node.level}{node.module or ''}")
    return targets


def test_phase4_content_fixtures_have_strict_contracts() -> None:
    assert len(_load_cases("content")) == len(_CONTENT_CASE_NAMES)
    assert len(_load_cases("metadata")) == len(_METADATA_CASE_NAMES)


def test_paragraph_formatting_matches_fixtures() -> None:
    for case in _load_cases("content"):
        assert content.convert_html_to_markdown(case["html"]) == case["expected"]


def test_paragraph_formatting_preserves_logging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages: list[str] = []
    monkeypatch.setattr(formatting_module, "logging", SimpleNamespace(info=messages.append))

    content.convert_html_to_markdown("<p>Body</p>")

    assert messages == ["Converting HTML to Markdown"]


def test_canonical_metadata_envelope_matches_fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FrozenDateTime:
        @classmethod
        def now(cls) -> datetime:
            return datetime(2026, 7, 27, 12, 34, 56)

    case = _case_by_name(_load_cases("metadata"), "canonical_formatted_envelope")
    monkeypatch.setattr(metadata_module, "datetime", _FrozenDateTime)

    actual = content.ContentMetadataHandler.format_content_with_metadata(
        case["url"],
        case["content"],
        pipeline=case["pipeline"],
        additional_metadata=case["additional_metadata"],
    )

    assert actual == case["expected"].replace("<TIMESTAMP>", "2026-07-27 12:34:56")


@pytest.mark.parametrize(
    "case_name",
    [
        "canonical_envelope_inspection",
        "malformed_envelope_passes_through",
        "nesting_boundary_is_accepted",
        "nesting_over_boundary_is_rejected",
    ],
)
def test_metadata_inspection_matches_fixtures(case_name: str) -> None:
    case = _case_by_name(_load_cases("metadata"), case_name)

    assert _inspect(case["content"]) == case["expected"]


def test_metadata_nesting_fixtures_pin_64_level_guard() -> None:
    cases = _load_cases("metadata")
    accepted = _case_by_name(cases, "nesting_boundary_is_accepted")
    rejected = _case_by_name(cases, "nesting_over_boundary_is_rejected")

    assert _max_json_nesting(accepted["content"]) == 64
    assert _max_json_nesting(rejected["content"]) == 65
    assert content.ContentMetadataHandler.has_metadata(accepted["content"]) is True
    assert content.ContentMetadataHandler.has_metadata(rejected["content"]) is False


def test_non_string_metadata_input_passes_through() -> None:
    sentinel = object()

    assert content.ContentMetadataHandler.extract_metadata(sentinel) == ({}, sentinel)
    assert content.ContentMetadataHandler.has_metadata(sentinel) is False
    assert content.ContentMetadataHandler.strip_metadata(sentinel) is sentinel


@pytest.mark.parametrize(
    "case_name",
    [
        "metadata_only_changes_do_not_change_body_hash",
        "body_changes_are_detected",
    ],
)
def test_body_only_content_change_semantics_match_fixtures(case_name: str) -> None:
    case = _case_by_name(_load_cases("metadata"), case_name)

    assert (
        content.ContentMetadataHandler.content_changed(
            case["old_content"],
            case["new_content"],
        )
        is case["expected"]
    )


def test_legacy_content_exports_are_canonical() -> None:
    assert content.__all__ == ["ContentMetadataHandler", "convert_html_to_markdown"]
    assert legacy.convert_html_to_markdown is content.convert_html_to_markdown
    assert legacy.ContentMetadataHandler is content.ContentMetadataHandler


def test_content_implementations_have_single_canonical_owner() -> None:
    legacy_tree = ast.parse(Path(legacy.__file__).read_text(encoding="utf-8"))
    legacy_definitions = {
        node.name
        for node in legacy_tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert "convert_html_to_markdown" not in legacy_definitions
    assert "ContentMetadataHandler" not in legacy_definitions


def test_content_modules_use_only_allowlisted_leaf_dependencies() -> None:
    source_files = {path.name for path in CONTENT_ROOT.glob("*.py")}
    allowlisted_files = set(_CONTENT_IMPORT_ALLOWLIST)
    inventory_errors = []
    if unlisted_files := sorted(source_files - allowlisted_files):
        inventory_errors.append(f"add explicit import allowlists for: {unlisted_files}")
    if missing_files := sorted(allowlisted_files - source_files):
        inventory_errors.append(f"remove stale import allowlists for: {missing_files}")

    violations = []
    for filename, allowed_targets in sorted(_CONTENT_IMPORT_ALLOWLIST.items()):
        path = CONTENT_ROOT / filename
        if not path.is_file():
            continue
        unexpected_targets = sorted(_import_targets(path) - allowed_targets)
        if unexpected_targets:
            violations.append(
                f"{filename}: disallowed imports {unexpected_targets}; "
                f"allowed imports are {sorted(allowed_targets)}"
            )

    errors = inventory_errors + violations
    assert not errors, "Content import allowlist violations:\n" + "\n".join(errors)


def test_importing_content_formatting_does_not_load_project_infrastructure() -> None:
    script = f"""
import importlib
import json
import sys

importlib.import_module("tldw_Server_API.app.core.Web_Scraping.content.formatting")
blocked = {list(_ISOLATION_BLOCKED_MODULES)!r}
loaded = sorted(
    name
    for name in sys.modules
    if any(name == prefix or name.startswith(prefix + ".") for prefix in blocked)
)
print(json.dumps(loaded))
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == [], (
        "content.formatting imported project infrastructure: " + result.stdout.strip()
    )
