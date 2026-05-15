"""Validate redaction-safe deterministic Persona Chat quality fixture records."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from tldw_Server_API.tests.Persona.persona_chat_quality_cases import all_cases, case_by_id

REPO_ROOT = Path(__file__).resolve().parents[3]
TAXONOMY_PATH = REPO_ROOT / "Docs/Reviews/PERSONA_CHAT_TRACE_ERROR_TAXONOMY_2026_05_10.md"
TAXONOMY_LABEL_PATTERN = re.compile(r"PC-[A-Z]+-\d{3}")
FIRST_PASS_LABELS = {
    "PC-ID-001",
    "PC-ID-002",
    "PC-EX-002",
    "PC-EX-003",
    "PC-EX-004",
    "PC-EX-005",
    "PC-PREV-001",
    "PC-MEM-001",
    "PC-MEM-002",
    "PC-TRACE-001",
}
FORBIDDEN_PRIVATE_PATTERNS = [
    re.compile(r"/Users/[^\s\"']+"),
    re.compile(r"/home/[^\s\"']+"),
    re.compile(r"/root(?:/|$)"),
    re.compile(r"\b[A-Za-z]:\\(?:Users|Documents and Settings)\\", re.IGNORECASE),
    re.compile(r"sk-[A-Za-z0-9]"),
    re.compile(r"api[_-]?key", re.IGNORECASE),
    re.compile(r"secret", re.IGNORECASE),
    re.compile(r"password", re.IGNORECASE),
    re.compile(r"raw private", re.IGNORECASE),
]


def _load_cases() -> list[dict[str, Any]]:
    """Load fixture cases as a mutable list of independent records."""
    return list(all_cases())


def _extract_taxonomy_labels(markdown: str) -> set[str]:
    """Extract labels from the taxonomy Failure Labels markdown table only."""
    parts = markdown.split("## Failure Labels", maxsplit=1)
    if len(parts) != 2:
        return set()
    failure_labels_section = parts[1]
    next_heading = re.search(r"\n##\s+", failure_labels_section)
    if next_heading is not None:
        failure_labels_section = failure_labels_section[: next_heading.start()]

    labels: set[str] = set()
    for raw_line in failure_labels_section.splitlines():
        line = raw_line.strip()
        if not line.startswith("|"):
            continue
        cells = line.strip().strip("|").split("|")
        if not cells:
            continue
        label_cell = cells[0].strip().replace("`", "").replace("*", "").strip()
        if TAXONOMY_LABEL_PATTERN.fullmatch(label_cell):
            labels.add(label_cell)
    return labels


def _taxonomy_labels() -> set[str]:
    """Load the current taxonomy labels from the repo-root review artifact."""
    text = TAXONOMY_PATH.read_text(encoding="utf-8")
    return _extract_taxonomy_labels(text)


def _walk_strings(value: Any) -> list[str]:
    """Return all string leaves from a nested fixture record."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        strings: list[str] = []
        for item in value:
            strings.extend(_walk_strings(item))
        return strings
    if isinstance(value, dict):
        strings = []
        for item in value.values():
            strings.extend(_walk_strings(item))
        return strings
    return []


def _assert_redaction_safe(text: str) -> None:
    """Assert a fixture string does not contain private local paths or secrets."""
    for pattern in FORBIDDEN_PRIVATE_PATTERNS:
        assert not pattern.search(text), text  # nosec B101


def test_taxonomy_label_parser_accepts_markdown_cell_variants() -> None:
    labels = _extract_taxonomy_labels(
        """
## Representative Case Set

| Case | Labels |
| --- | --- |
| PC-CASE-001 | `PC-ID-999` |

## Failure Labels

| Label | Name |
| --- | --- |
| PC-ID-001 | Plain spacing |
|`PC-MEM-001`| Backtick wrapped |
  | **PC-EX-003** | Bold wrapped with indentation |
| not-a-label | Ignored |

## Later Section

| PC-CASE-002 | Not a failure label |
| PC-ID-998 | Outside the Failure Labels table |
"""
    )

    assert labels == {"PC-ID-001", "PC-MEM-001", "PC-EX-003"}  # nosec B101


@pytest.mark.parametrize(
    "private_value",
    [
        "/Users/alice/project/private-note.txt",
        "/home/alice/project/private-note.txt",
        "/root/private-note.txt",
        r"C:\Users\Alice\private-note.txt",
    ],
)
def test_redaction_guard_rejects_user_owned_path_variants(private_value: str) -> None:
    with pytest.raises(AssertionError):
        _assert_redaction_safe(private_value)


def test_case_loader_returns_independent_case_copies() -> None:
    mutable_case = case_by_id("PC-CASE-001")
    mutable_case["labels"].append("PC-CASE-999")

    fresh_case = case_by_id("PC-CASE-001")
    assert "PC-CASE-999" not in fresh_case["labels"]  # nosec B101

    mutable_cases = all_cases()
    mutable_cases[0]["labels"].append("PC-CASE-998")

    fresh_cases = all_cases()
    assert "PC-CASE-998" not in fresh_cases[0]["labels"]  # nosec B101


def test_persona_chat_quality_cases_are_redaction_safe_and_schema_valid() -> None:
    cases = _load_cases()

    assert len(cases) == 20  # nosec B101
    assert [case["case_id"] for case in cases] == [  # nosec B101
        f"PC-CASE-{index:03d}" for index in range(1, 21)
    ]

    taxonomy_labels = _taxonomy_labels()
    represented_labels: set[str] = set()
    for case in cases:
        assert case["assistant_kind"] == "persona"  # nosec B101
        assert case["assistant_id"]  # nosec B101
        assert case["persona_memory_mode"] in {"read_only", "read_write"}  # nosec B101
        assert case["input"]  # nosec B101
        assert isinstance(case["expected_context"], dict)  # nosec B101
        assert isinstance(case["expected_context"].get("trace_refs"), list)  # nosec B101
        assert case["expected_context"]["trace_refs"]  # nosec B101
        assert isinstance(case["response_observation"], dict)  # nosec B101
        assert isinstance(case["labels"], list) and case["labels"]  # nosec B101
        assert isinstance(case["expected_evidence"], list) and case["expected_evidence"]  # nosec B101
        assert not any(label.startswith("PC-CASE-") for label in case["labels"])  # nosec B101
        assert set(case["labels"]).issubset(taxonomy_labels)  # nosec B101
        represented_labels.update(case["labels"])

        for text in _walk_strings(case):
            _assert_redaction_safe(text)

    assert FIRST_PASS_LABELS.issubset(represented_labels)  # nosec B101
    trace_case = next(case for case in cases if case["case_id"] == "PC-CASE-020")
    assert trace_case["labels"] == ["PC-TRACE-001"]  # nosec B101
    assert {  # nosec B101
        "case_id",
        "assistant_identity",
        "selected_exemplar_ids",
        "memory_mode",
    }.issubset(trace_case["expected_context"]["trace_refs"])
