"""Characterization tests for hierarchical paragraph-span detection."""

from __future__ import annotations

import builtins
from typing import Any

import pytest
from loguru import logger

from tldw_Server_API.app.core.Chunking import regex_safety
from tldw_Server_API.app.core.Chunking.hierarchical.spans import compute_paragraph_spans

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", []),
        ("\n\n", [(0, 1, "blank"), (1, 2, "blank")]),
        ("alpha\nbeta", [(0, 10, "paragraph")]),
        ("# Title\n", [(0, 8, "header_atx")]),
        ("---\n", [(0, 4, "hr")]),
        ("- item\n", [(0, 7, "list_unordered")]),
        ("1. item\n", [(0, 8, "list_ordered")]),
        ("| a |\n", [(0, 6, "table_md")]),
    ],
)
def test_builtin_paragraph_span_kinds(
    text: str,
    expected: list[tuple[int, int, str]],
) -> None:
    assert compute_paragraph_spans(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("```py\ncode\n```\n", [(0, 15, "code_fence")]),
        ("```py\ncode", [(0, 10, "code_fence")]),
        ("~~~\nx\n~~~\n", [(0, 10, "code_fence")]),
        ("~~~\nx", [(0, 5, "code_fence")]),
    ],
)
def test_closed_and_unclosed_fence_spans(
    text: str,
    expected: list[tuple[int, int, str]],
) -> None:
    assert compute_paragraph_spans(text) == expected


def test_template_boundary_accepts_custom_kind_and_flags() -> None:
    template = {"boundaries": [{"kind": "custom_heading", "pattern": "^custom", "flags": "i"}]}

    assert compute_paragraph_spans("CUSTOM\n", template) == [(0, 7, "custom_heading")]


def test_template_boundary_uses_only_first_twenty_rules() -> None:
    boundaries = [{"kind": f"ignored_{index}", "pattern": "^does-not-match$"} for index in range(20)]
    boundaries.append({"kind": "too_late", "pattern": "^target"})

    assert compute_paragraph_spans("target\n", {"boundaries": boundaries}) == [(0, 7, "paragraph")]


@pytest.mark.parametrize(
    "rule",
    [
        {"kind": "overlong", "pattern": "a" * 257},
        {"kind": "unsafe", "pattern": "(a+)+"},
        object(),
    ],
)
def test_invalid_template_rules_are_skipped(rule: object) -> None:
    assert compute_paragraph_spans("ordinary\n", {"boundaries": [rule]}) == [(0, 9, "paragraph")]


def test_invalid_template_flags_fall_back_to_no_flags() -> None:
    template = {"boundaries": [{"kind": "case_sensitive", "pattern": "^CUSTOM", "flags": "invalid"}]}

    assert compute_paragraph_spans("custom\n", template) == [(0, 7, "paragraph")]


@pytest.mark.parametrize("boundaries", [1, {"private_template": "secret-template-value"}])
def test_template_boundary_setup_failure_warns_and_preserves_builtin_spans(boundaries: object) -> None:
    text = "# Title\nsecret-document-value\n\n- item\n"
    expected = [(0, 8, "header_atx"), (8, 30, "paragraph"), (30, 31, "blank"), (31, 38, "list_unordered")]
    records: list[dict[str, Any]] = []
    sink_id = logger.add(lambda message: records.append(message.record), level="WARNING")
    try:
        spans = compute_paragraph_spans(text, {"boundaries": boundaries})
    finally:
        logger.remove(sink_id)

    assert spans == expected
    assert [record["message"] for record in records] == [
        "Template boundary setup failed; using built-in paragraph detection"
    ]
    assert records[0]["level"].name == "WARNING"
    assert records[0]["exception"] is None
    assert "secret-" not in repr(records[0]["extra"])


def test_template_boundary_setup_warning_omits_exception_details() -> None:
    class FailingTemplate(dict[str, Any]):
        """Model a template provider that fails before individual rules are read."""

        def get(self, key: str, default: Any = None) -> Any:
            raise RuntimeError("secret-exception-value")

    records: list[dict[str, Any]] = []
    sink_id = logger.add(lambda message: records.append(message.record), level="WARNING")
    try:
        spans = compute_paragraph_spans("ordinary\n", FailingTemplate(boundaries=["secret-template-value"]))
    finally:
        logger.remove(sink_id)

    assert spans == [(0, 9, "paragraph")]
    assert [record["message"] for record in records] == [
        "Template boundary setup failed; using built-in paragraph detection"
    ]
    assert records[0]["exception"] is None
    assert "secret-" not in repr(records[0]["extra"])


@pytest.mark.parametrize(
    ("text", "template", "expected"),
    [
        ("ordinary\n", None, [(0, 9, "paragraph")]),
        ("ordinary\n", {"boundaries": []}, [(0, 9, "paragraph")]),
        ("CUSTOM\n", {"boundaries": [{"kind": "custom", "pattern": "^CUSTOM"}]}, [(0, 7, "custom")]),
        ("", {"boundaries": 1}, []),
    ],
)
def test_successful_or_empty_template_setup_does_not_warn(
    text: str,
    template: dict[str, Any] | None,
    expected: list[tuple[int, int, str]],
) -> None:
    records: list[dict[str, Any]] = []
    sink_id = logger.add(lambda message: records.append(message.record), level="WARNING")
    try:
        spans = compute_paragraph_spans(text, template)
    finally:
        logger.remove(sink_id)

    assert spans == expected
    assert records == []


def test_invalid_template_rule_warnings_have_stable_level_and_text() -> None:
    records: list[dict[str, Any]] = []
    sink_id = logger.add(lambda message: records.append(message.record), level="WARNING")
    try:
        compute_paragraph_spans(
            "ordinary\n",
            {
                "boundaries": [
                    {"kind": "overlong", "pattern": "a" * 257},
                    {"kind": "unsafe", "pattern": "(a+)+"},
                    object(),
                ]
            },
        )
    finally:
        logger.remove(sink_id)

    warning_messages = [record["message"] for record in records if record["level"].name == "WARNING"]
    assert any("Skipping overlong boundary pattern (>256 chars)" in message for message in warning_messages)
    assert any("Skipping boundary pattern due to safety check:" in message for message in warning_messages)
    assert any("Ignoring invalid boundary rule:" in message for message in warning_messages)


def test_safe_search_is_looked_up_at_call_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []

    def replacement(pattern: Any, text: str) -> bool:
        calls.append((pattern.pattern, text))
        return True

    monkeypatch.setattr(regex_safety, "safe_search", replacement)

    spans = compute_paragraph_spans(
        "anything\n",
        {"boundaries": [{"kind": "runtime", "pattern": "^never$"}]},
    )

    assert spans == [(0, 9, "runtime")]
    assert calls == [("^never$", "anything\n")]


def test_safe_search_lookup_failure_uses_direct_pattern_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__
    guarded_interceptions: list[str] = []

    def guarded_import(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        package = (globals or {}).get("__package__")
        chunking_regex_import = name == "tldw_Server_API.app.core.Chunking.regex_safety" or (
            name == "regex_safety" and level == 2 and package == "tldw_Server_API.app.core.Chunking.hierarchical"
        )
        if chunking_regex_import and "safe_search" in fromlist:
            guarded_interceptions.append("Chunking.regex_safety.safe_search")
            raise ImportError("safe_search unavailable")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    assert compute_paragraph_spans(
        "CUSTOM\n",
        {"boundaries": [{"kind": "direct", "pattern": "^CUSTOM"}]},
    ) == [(0, 7, "direct")]
    assert guarded_interceptions == ["Chunking.regex_safety.safe_search"]


def test_safe_search_invocation_failure_skips_pattern_without_direct_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def failing_safe_search(_pattern: Any, text: str) -> bool:
        calls.append(text)
        raise RuntimeError("search failure")

    monkeypatch.setattr(regex_safety, "safe_search", failing_safe_search)

    assert compute_paragraph_spans(
        "CUSTOM\n",
        {"boundaries": [{"kind": "not_emitted", "pattern": "^CUSTOM"}]},
    ) == [(0, 7, "paragraph")]
    assert calls == ["CUSTOM\n"]
