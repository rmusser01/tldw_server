"""Characterization contracts for the public hierarchical chunking API."""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any

import pytest
from loguru import logger

from tldw_Server_API.app.core.Chunking import Chunker


class BrokenBool:
    """Option value whose truthiness cannot be determined."""

    def __bool__(self) -> bool:
        raise RuntimeError("cannot decide")


class EventBool:
    """Option value that records when its truthiness is evaluated."""

    def __init__(self, events: list[str]) -> None:
        self.events = events

    def __bool__(self) -> bool:
        self.events.append("truthiness")
        return True


def _patch_single_block(
    monkeypatch: pytest.MonkeyPatch,
    chunker: Chunker,
    *,
    resolved_method: str,
) -> None:
    monkeypatch.setattr(chunker, "_sanitize_input", lambda *_args, **_kwargs: "raw")
    monkeypatch.setattr(
        chunker,
        "_compute_paragraph_spans",
        lambda *_args, **_kwargs: [(0, 3, "paragraph")],
    )
    monkeypatch.setattr(
        chunker,
        "_resolve_method",
        lambda *_args, **_kwargs: resolved_method,
    )


def _capture_log_records(call: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    sink_id = logger.add(lambda message: records.append(message.record), level="DEBUG")
    try:
        call()
    finally:
        logger.remove(sink_id)
    return records


def test_public_hierarchy_signatures_are_stable() -> None:
    assert str(inspect.signature(Chunker.chunk_text_hierarchical_tree)) == (
        "(self, text: str, method: str | None = None, max_size: int | None = None, "
        "overlap: int | None = None, language: str | None = None, "
        "template: dict[str, Any] | None = None, "
        "method_options: dict[str, Any] | None = None) -> dict[str, typing.Any]"
    )
    assert str(inspect.signature(Chunker.flatten_hierarchical)) == (
        "(self, tree: dict[str, typing.Any]) -> list[dict[str, typing.Any]]"
    )
    assert str(inspect.signature(Chunker.chunk_text_hierarchical_flat)) == (
        "(self, text: str, method: str | None = None, max_size: int | None = None, "
        "overlap: int | None = None, language: str | None = None, "
        "template: dict[str, Any] | None = None, "
        "method_options: dict[str, Any] | None = None) -> list[dict[str, typing.Any]]"
    )


def test_public_flat_method_composes_overridable_public_methods(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunker = Chunker()
    calls: list[tuple[str, Any]] = []
    sentinel_tree = {"root": {"kind": "root", "children": []}}
    sentinel_rows = [{"text": "flat", "metadata": {}}]
    template = {"boundaries": []}
    method_options = {"nested": {"value": 1}}

    def fake_tree(**kwargs: Any) -> dict[str, Any]:
        calls.append(("tree", kwargs))
        return sentinel_tree

    def fake_flatten(tree: dict[str, Any]) -> list[dict[str, Any]]:
        calls.append(("flatten", tree))
        return sentinel_rows

    monkeypatch.setattr(chunker, "chunk_text_hierarchical_tree", fake_tree)
    monkeypatch.setattr(chunker, "flatten_hierarchical", fake_flatten)

    result = chunker.chunk_text_hierarchical_flat(
        "source",
        method="words",
        max_size=10,
        overlap=2,
        language="en",
        template=template,
        method_options=method_options,
    )

    assert result is sentinel_rows
    assert calls[0] == (
        "tree",
        {
            "text": "source",
            "method": "words",
            "max_size": 10,
            "overlap": 2,
            "language": "en",
            "template": template,
            "method_options": method_options,
        },
    )
    assert calls[0][1]["template"] is template
    assert calls[0][1]["method_options"] is method_options
    assert calls[1][0] == "flatten"
    assert calls[1][1] is sentinel_tree


@pytest.mark.parametrize(
    ("method_options", "expected_text"),
    [
        ({}, "SAN"),
        ({"sanitize_output": False}, "raw"),
        ({"sanitize_output": BrokenBool()}, "SAN"),
    ],
)
def test_hierarchy_options_are_shallow_copied_and_sanitization_is_resolved_first(
    monkeypatch: pytest.MonkeyPatch,
    method_options: dict[str, Any],
    expected_text: str,
) -> None:
    chunker = Chunker()
    nested = {"value": []}
    original_options = {**method_options, "nested": nested}
    resolved_options: list[dict[str, Any]] = []

    monkeypatch.setattr(
        chunker,
        "_sanitize_input",
        lambda text, **_kwargs: "SAN" if text == "raw" else text,
    )
    monkeypatch.setattr(
        chunker,
        "_resolve_method",
        lambda _method, _language, options: resolved_options.append(options) or "structure_aware",
    )

    tree = chunker.chunk_text_hierarchical_tree(
        "raw",
        method="structure_aware",
        max_size=10,
        overlap=0,
        method_options=original_options,
    )

    emitted = tree["root"]["children"][0]["children"][0]["chunks"]
    assert emitted[0]["text"] == expected_text
    assert resolved_options[0] is not original_options
    assert resolved_options[0]["nested"] is nested
    assert "sanitize_output" not in resolved_options[0]
    assert original_options.get("sanitize_output") is method_options.get("sanitize_output")


def test_sanitize_output_truthiness_is_evaluated_before_method_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunker = Chunker()
    events: list[str] = []

    monkeypatch.setattr(chunker, "_sanitize_input", lambda *_args, **_kwargs: "SAN")

    def fake_resolve_method(
        _method: str,
        _language: str,
        _options: dict[str, Any],
    ) -> str:
        events.append("resolve")
        return "structure_aware"

    monkeypatch.setattr(chunker, "_resolve_method", fake_resolve_method)

    chunker.chunk_text_hierarchical_tree(
        "raw",
        method="structure_aware",
        max_size=10,
        overlap=0,
        method_options={"sanitize_output": EventBool(events)},
    )

    assert events == ["truthiness", "resolve"]


@pytest.mark.parametrize("method", ["words", "sentences", "tokens"])
def test_metadata_leaf_methods_call_metadata_once_on_success(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
) -> None:
    chunker = Chunker()
    trace: list[str] = []
    _patch_single_block(monkeypatch, chunker, resolved_method=method)

    def fake_metadata(*_args: Any, **_kwargs: Any) -> list[SimpleNamespace]:
        trace.append("metadata")
        return [
            SimpleNamespace(
                metadata=SimpleNamespace(start_char=0, end_char=3),
            )
        ]

    def fake_plain(*_args: Any, **_kwargs: Any) -> list[str]:
        trace.append("plain")
        return ["raw"]

    monkeypatch.setattr(chunker, "chunk_text_with_metadata", fake_metadata)
    monkeypatch.setattr(chunker, "chunk_text", fake_plain)

    tree = chunker.chunk_text_hierarchical_tree("raw", method=method, max_size=10)

    assert trace == ["metadata"]
    assert tree["root"]["children"][0]["children"][0]["chunks"][0]["text"] == "raw"


@pytest.mark.parametrize(
    "metadata",
    [SimpleNamespace(), SimpleNamespace(start_char="0", end_char=3)],
)
def test_invalid_metadata_offsets_emit_nothing_without_plain_fallback(
    monkeypatch: pytest.MonkeyPatch,
    metadata: SimpleNamespace,
) -> None:
    chunker = Chunker()
    trace: list[str] = []
    _patch_single_block(monkeypatch, chunker, resolved_method="words")

    def fake_metadata(*_args: Any, **_kwargs: Any) -> list[SimpleNamespace]:
        trace.append("metadata")
        return [SimpleNamespace(metadata=metadata)]

    def fake_plain(*_args: Any, **_kwargs: Any) -> list[str]:
        trace.append("plain")
        return ["raw"]

    monkeypatch.setattr(chunker, "chunk_text_with_metadata", fake_metadata)
    monkeypatch.setattr(chunker, "chunk_text", fake_plain)

    tree = chunker.chunk_text_hierarchical_tree("raw", method="words", max_size=10)

    assert trace == ["metadata"]
    assert tree["root"]["children"][0]["children"][0]["chunks"] == []


def test_metadata_failure_calls_plain_fallback_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunker = Chunker()
    trace: list[str] = []
    _patch_single_block(monkeypatch, chunker, resolved_method="words")

    def fake_metadata(*_args: Any, **_kwargs: Any) -> list[SimpleNamespace]:
        trace.append("metadata")
        raise RuntimeError("metadata failure")

    def fake_plain(*_args: Any, **_kwargs: Any) -> list[str]:
        trace.append("plain")
        return ["raw"]

    monkeypatch.setattr(chunker, "chunk_text_with_metadata", fake_metadata)
    monkeypatch.setattr(chunker, "chunk_text", fake_plain)

    chunker.chunk_text_hierarchical_tree("raw", method="words", max_size=10)

    assert trace == ["metadata", "plain"]


def test_outer_fallback_retries_plain_call_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunker = Chunker()
    trace: list[str] = []
    plain_calls = 0
    _patch_single_block(monkeypatch, chunker, resolved_method="words")

    def fake_metadata(*_args: Any, **_kwargs: Any) -> list[SimpleNamespace]:
        trace.append("metadata")
        raise RuntimeError("metadata failure")

    def fake_plain(*_args: Any, **_kwargs: Any) -> list[str]:
        nonlocal plain_calls
        trace.append("plain")
        plain_calls += 1
        if plain_calls == 1:
            raise RuntimeError("first plain failure")
        return ["raw"]

    monkeypatch.setattr(chunker, "chunk_text_with_metadata", fake_metadata)
    monkeypatch.setattr(chunker, "chunk_text", fake_plain)

    tree = chunker.chunk_text_hierarchical_tree("raw", method="words", max_size=10)
    chunks = tree["root"]["children"][0]["children"][0]["chunks"]

    assert trace == ["metadata", "plain", "plain"]
    assert len(chunks) == 1


@pytest.mark.parametrize(
    ("method", "expected_trace"),
    [
        ("semantic", ["plain"]),
        ("fixed", ["plain"]),
        ("structure_aware", []),
    ],
)
def test_non_metadata_leaf_call_traces(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    expected_trace: list[str],
) -> None:
    chunker = Chunker()
    trace: list[str] = []
    _patch_single_block(monkeypatch, chunker, resolved_method=method)

    def fake_metadata(*_args: Any, **_kwargs: Any) -> list[SimpleNamespace]:
        trace.append("metadata")
        return []

    def fake_plain(*_args: Any, **_kwargs: Any) -> list[str]:
        trace.append("plain")
        return ["raw"]

    monkeypatch.setattr(chunker, "chunk_text_with_metadata", fake_metadata)
    monkeypatch.setattr(chunker, "chunk_text", fake_plain)

    chunker.chunk_text_hierarchical_tree("raw", method=method, max_size=10)

    assert trace == expected_trace


@pytest.mark.parametrize("method", ["words", "fixed", "semantic"])
def test_sanitize_output_is_not_forwarded_to_leaf_calls(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
) -> None:
    chunker = Chunker()
    leaf_options: list[dict[str, Any]] = []
    nested: list[str] = []
    _patch_single_block(monkeypatch, chunker, resolved_method=method)

    def fake_metadata(*_args: Any, **kwargs: Any) -> list[SimpleNamespace]:
        leaf_options.append(kwargs)
        return [
            SimpleNamespace(
                metadata=SimpleNamespace(start_char=0, end_char=3),
            )
        ]

    def fake_plain(*_args: Any, **kwargs: Any) -> list[str]:
        leaf_options.append(kwargs)
        return ["raw"]

    monkeypatch.setattr(chunker, "chunk_text_with_metadata", fake_metadata)
    monkeypatch.setattr(chunker, "chunk_text", fake_plain)

    chunker.chunk_text_hierarchical_tree(
        "raw",
        method=method,
        max_size=10,
        method_options={"sanitize_output": False, "nested": nested},
    )

    assert len(leaf_options) == 1
    assert "sanitize_output" not in leaf_options[0]
    assert leaf_options[0]["nested"] is nested


@pytest.mark.parametrize("method", ["words", "sentences"])
def test_word_and_sentence_metadata_failures_have_stable_debug_log(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
) -> None:
    chunker = Chunker()
    _patch_single_block(monkeypatch, chunker, resolved_method=method)
    monkeypatch.setattr(
        chunker,
        "chunk_text_with_metadata",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("metadata failure")),
    )
    monkeypatch.setattr(chunker, "chunk_text", lambda *_args, **_kwargs: ["raw"])

    records = _capture_log_records(lambda: chunker.chunk_text_hierarchical_tree("raw", method=method, max_size=10))

    assert any(
        record["level"].name == "DEBUG" and f"{method} metadata mapping failed, using fallback:" in record["message"]
        for record in records
    )


def test_token_metadata_failure_has_stable_debug_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunker = Chunker()
    _patch_single_block(monkeypatch, chunker, resolved_method="tokens")
    monkeypatch.setattr(
        chunker,
        "chunk_text_with_metadata",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("metadata failure")),
    )
    monkeypatch.setattr(chunker, "chunk_text", lambda *_args, **_kwargs: ["raw"])

    records = _capture_log_records(lambda: chunker.chunk_text_hierarchical_tree("raw", method="tokens", max_size=10))

    assert any(
        record["level"].name == "DEBUG" and "Token metadata mapping failed, using fallback:" in record["message"]
        for record in records
    )


def test_outer_offset_failure_has_stable_warning_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunker = Chunker()
    plain_calls = 0
    _patch_single_block(monkeypatch, chunker, resolved_method="words")
    monkeypatch.setattr(
        chunker,
        "chunk_text_with_metadata",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("metadata failure")),
    )

    def fake_plain(*_args: Any, **_kwargs: Any) -> list[str]:
        nonlocal plain_calls
        plain_calls += 1
        if plain_calls == 1:
            raise RuntimeError("first plain failure")
        return ["raw"]

    monkeypatch.setattr(chunker, "chunk_text", fake_plain)

    records = _capture_log_records(lambda: chunker.chunk_text_hierarchical_tree("raw", method="words", max_size=10))

    assert any(
        record["level"].name == "WARNING"
        and "Offset mapping failed for method=words:" in record["message"]
        and "using naive offsets" in record["message"]
        for record in records
    )
