"""Characterization contracts for the public hierarchical chunking API."""

from __future__ import annotations

import ast
import importlib.util
import inspect
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NamedTuple

import pytest
from loguru import logger

from tldw_Server_API.app.core.Chunking import Chunker
from tldw_Server_API.app.core.Chunking.hierarchical import builder, service
from tldw_Server_API.app.core.Chunking.hierarchical.models import (
    HierarchyContext,
    HierarchyTextViews,
    LeafChunkingContext,
    ResolvedHierarchyOptions,
)
from tldw_Server_API.app.core.Chunking.process_text.models import ProcessTextContext

_CHUNKING_PACKAGE_NAME = "tldw_Server_API.app.core.Chunking"
_HIERARCHICAL_PACKAGE_NAME = f"{_CHUNKING_PACKAGE_NAME}.hierarchical"
_HIERARCHICAL_PACKAGE = Path(__file__).parents[2] / "app" / "core" / "Chunking" / "hierarchical"
_FORBIDDEN_HIERARCHICAL_IMPORTS = {
    f"{_CHUNKING_PACKAGE_NAME}.chunker",
    f"{_CHUNKING_PACKAGE_NAME}.process_text",
}
_FORBIDDEN_LOWER_LAYER_IMPORTS = {
    f"{_HIERARCHICAL_PACKAGE_NAME}.builder",
    f"{_HIERARCHICAL_PACKAGE_NAME}.flatten",
    f"{_HIERARCHICAL_PACKAGE_NAME}.service",
}


class _ResolvedImport(NamedTuple):
    module: str
    imported_name: str | None = None


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
        service,
        "compute_paragraph_spans",
        lambda *_args, **_kwargs: [(0, 3, "paragraph")],
    )
    monkeypatch.setattr(
        chunker,
        "_resolve_method",
        lambda *_args, **_kwargs: resolved_method,
    )


def _resolved_imports(module_path: Path) -> list[_ResolvedImport]:
    tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
    imports: list[_ResolvedImport] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(_ResolvedImport(alias.name) for alias in node.names)
            continue
        if not isinstance(node, ast.ImportFrom):
            continue

        if node.level:
            relative_name = f"{'.' * node.level}{node.module or ''}"
            resolved_module = importlib.util.resolve_name(relative_name, _HIERARCHICAL_PACKAGE_NAME)
        else:
            resolved_module = node.module or ""
        if resolved_module:
            imports.extend(_ResolvedImport(resolved_module, alias.name) for alias in node.names)
    return imports


def _is_exact_or_descendant(module: str, blocked: str) -> bool:
    return module == blocked or module.startswith(f"{blocked}.")


def _is_forbidden_import(resolved: _ResolvedImport, forbidden: set[str]) -> bool:
    if resolved.module == _CHUNKING_PACKAGE_NAME and resolved.imported_name in {"Chunker", "*"}:
        return True

    candidates = [resolved.module]
    if resolved.imported_name not in {None, "*"}:
        candidates.append(f"{resolved.module}.{resolved.imported_name}")
    return any(_is_exact_or_descendant(candidate, blocked) for candidate in candidates for blocked in forbidden)


def _assert_no_forbidden_resolved_imports(module_path: Path, forbidden: set[str]) -> None:
    offenders = [resolved for resolved in _resolved_imports(module_path) if _is_forbidden_import(resolved, forbidden)]
    assert offenders == [], f"{module_path.name} imports forbidden modules: {offenders}"


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


def test_hierarchy_models_are_passive_and_frozen() -> None:
    nested: list[str] = []
    options = ResolvedHierarchyOptions("words", "10", 0, None, {"nested": nested}, True)
    views = HierarchyTextViews(original="a", sanitized="bb", output="ccc")

    assert options.method_options["nested"] is nested
    assert (views.original, views.sanitized, views.output) == ("a", "bb", "ccc")
    with pytest.raises(FrozenInstanceError):
        views.output = "changed"  # type: ignore[misc]


def test_hierarchy_protocols_expose_required_members() -> None:
    leaf_members = {"chunk_text", "chunk_text_with_metadata"}
    hierarchy_members = {
        "config",
        "_enforce_text_size",
        "_normalize_method_argument",
        "_resolve_method",
        "_sanitize_input",
        "normalize_chunk_type",
        *leaf_members,
    }
    available_hierarchy_members = (
        set(HierarchyContext.__dict__) | set(HierarchyContext.__annotations__) | set(LeafChunkingContext.__dict__)
    )

    assert leaf_members.issubset(LeafChunkingContext.__dict__)
    assert hierarchy_members.issubset(available_hierarchy_members)
    assert LeafChunkingContext in HierarchyContext.__mro__
    assert HierarchyContext.__annotations__["config"] == "ChunkerConfig"

    for method_name in leaf_members:
        signature = inspect.signature(getattr(LeafChunkingContext, method_name))
        assert list(signature.parameters) == [
            "self",
            "text",
            "method",
            "max_size",
            "overlap",
            "language",
            "options",
        ]
        assert all(signature.parameters[name].default is None for name in ("method", "max_size", "overlap", "language"))
        assert signature.parameters["options"].kind is inspect.Parameter.VAR_KEYWORD

    enforce_signature = inspect.signature(HierarchyContext._enforce_text_size)
    sanitize_signature = inspect.signature(HierarchyContext._sanitize_input)
    assert enforce_signature.parameters["source"].kind is inspect.Parameter.KEYWORD_ONLY
    assert sanitize_signature.parameters["suppress_security_log"].kind is inspect.Parameter.KEYWORD_ONLY
    assert sanitize_signature.parameters["suppress_security_log"].default is False


def test_process_text_context_no_longer_exposes_private_paragraph_spans() -> None:
    assert "_compute_paragraph_spans" not in ProcessTextContext.__dict__


def test_chunker_no_longer_exposes_private_hierarchy_helpers() -> None:
    assert not hasattr(Chunker, "_compute_paragraph_spans")
    assert not hasattr(Chunker, "_extract_header_title")


def test_builder_does_not_import_coordination_or_flattening() -> None:
    _assert_no_forbidden_resolved_imports(
        _HIERARCHICAL_PACKAGE / "builder.py",
        _FORBIDDEN_HIERARCHICAL_IMPORTS
        | {f"{_HIERARCHICAL_PACKAGE_NAME}.{name}" for name in ("service", "flatten", "grouping")},
    )


def test_hierarchical_modules_do_not_import_outer_owners() -> None:
    for module_path in sorted(_HIERARCHICAL_PACKAGE.glob("*.py")):
        _assert_no_forbidden_resolved_imports(module_path, _FORBIDDEN_HIERARCHICAL_IMPORTS)


def test_hierarchical_lower_layers_do_not_import_upper_layers() -> None:
    for module_name in ("leaves.py", "grouping.py"):
        module_path = _HIERARCHICAL_PACKAGE / module_name
        if module_path.exists():
            _assert_no_forbidden_resolved_imports(module_path, _FORBIDDEN_LOWER_LAYER_IMPORTS)


@pytest.mark.parametrize(
    "source",
    [
        "import tldw_Server_API.app.core.Chunking.chunker\n",
        "import tldw_Server_API.app.core.Chunking.chunker.internal\n",
        "from .. import chunker\n",
        "from ..process_text import dispatch\n",
        "from ..process_text.dispatch import dispatch_chunks\n",
        "from .. import Chunker\n",
        "from .. import Chunker as PublicChunker\n",
        "from tldw_Server_API.app.core.Chunking import Chunker\n",
        "from tldw_Server_API.app.core.Chunking import Chunker as PublicChunker\n",
        "from .. import *\n",
    ],
)
def test_hierarchical_import_boundary_resolves_absolute_and_relative_imports(
    tmp_path: Path,
    source: str,
) -> None:
    module_path = tmp_path / "candidate.py"
    module_path.write_text(source, encoding="utf-8")

    with pytest.raises(AssertionError):
        _assert_no_forbidden_resolved_imports(module_path, _FORBIDDEN_HIERARCHICAL_IMPORTS)


@pytest.mark.parametrize(
    "source",
    [
        "from . import service\n",
        "from .builder import build_hierarchy_tree\n",
        "import tldw_Server_API.app.core.Chunking.hierarchical.flatten.helpers\n",
    ],
)
def test_hierarchical_lower_layer_boundary_rejects_upper_layers(
    tmp_path: Path,
    source: str,
) -> None:
    module_path = tmp_path / "candidate.py"
    module_path.write_text(source, encoding="utf-8")

    with pytest.raises(AssertionError):
        _assert_no_forbidden_resolved_imports(module_path, _FORBIDDEN_LOWER_LAYER_IMPORTS)


@pytest.mark.parametrize(
    "source",
    [
        "import tldw_Server_API.app.core.Chunking.chunker_tools\n",
        "from .. import chunker_tools\n",
        "from tldw_Server_API.app.core.Chunking import chunker_tools as tools\n",
        "import tldw_Server_API.app.core.Chunking.process_text_helpers\n",
        "from .. import process_text_helpers\n",
        "from tldw_Server_API.app.core.Chunking import process_text_helpers as helpers\n",
        "from ..base import ChunkerConfig\n",
    ],
)
def test_hierarchical_import_boundary_allows_similarly_named_modules(
    tmp_path: Path,
    source: str,
) -> None:
    module_path = tmp_path / "candidate.py"
    module_path.write_text(source, encoding="utf-8")

    _assert_no_forbidden_resolved_imports(module_path, _FORBIDDEN_HIERARCHICAL_IMPORTS)


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
    assert len(calls) == 2
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


def test_hierarchy_tree_uses_call_time_leaf_builder_with_resolved_call_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunker = Chunker()
    nested = {"items": []}
    sentinel_block = {"kind": "sentinel", "children": []}
    calls: list[tuple[Any, Any, Any, Any]] = []
    _patch_single_block(monkeypatch, chunker, resolved_method="fixed_size")

    def fake_build_leaf_block(
        context: Any,
        texts: Any,
        span: Any,
        options: Any,
    ) -> dict[str, Any]:
        calls.append((context, texts, span, options))
        return sentinel_block

    monkeypatch.setattr(builder, "build_leaf_block", fake_build_leaf_block)

    tree = chunker.chunk_text_hierarchical_tree(
        "raw",
        method="fixed_size",
        max_size=7,
        overlap=2,
        language="de",
        method_options={"sanitize_output": False, "nested": nested},
    )

    assert len(calls) == 1
    context, texts, span, options = calls[0]
    assert context is chunker
    assert texts == HierarchyTextViews(original="raw", sanitized="raw", output="raw")
    assert span == (0, 3, "paragraph")
    assert options == ResolvedHierarchyOptions(
        method="fixed_size",
        max_size=7,
        overlap=2,
        language="de",
        method_options={"nested": nested},
        sanitize_output=False,
    )
    assert options.method_options["nested"] is nested
    assert tree["root"]["children"][0]["children"] == [sentinel_block]
    assert tree["root"]["children"][0]["children"][0] is sentinel_block


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


@pytest.mark.parametrize("method", ["words", "sentences", "tokens"])
@pytest.mark.parametrize(
    "metadata",
    [
        SimpleNamespace(end_char=3),
        SimpleNamespace(start_char="0", end_char=3),
        SimpleNamespace(start_char=0),
        SimpleNamespace(start_char=0, end_char="3"),
    ],
    ids=["missing-start", "invalid-start", "missing-end", "invalid-end"],
)
def test_invalid_metadata_offsets_emit_nothing_without_plain_fallback(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    metadata: SimpleNamespace,
) -> None:
    chunker = Chunker()
    trace: list[str] = []
    _patch_single_block(monkeypatch, chunker, resolved_method=method)

    def fake_metadata(*_args: Any, **_kwargs: Any) -> list[SimpleNamespace]:
        trace.append("metadata")
        return [SimpleNamespace(metadata=metadata)]

    def fake_plain(*_args: Any, **_kwargs: Any) -> list[str]:
        trace.append("plain")
        return ["raw"]

    monkeypatch.setattr(chunker, "chunk_text_with_metadata", fake_metadata)
    monkeypatch.setattr(chunker, "chunk_text", fake_plain)

    tree = chunker.chunk_text_hierarchical_tree("raw", method=method, max_size=10)

    assert trace == ["metadata"]
    assert tree["root"]["children"][0]["children"][0]["chunks"] == []


@pytest.mark.parametrize("method", ["words", "sentences", "tokens"])
def test_metadata_failure_calls_plain_fallback_once(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
) -> None:
    chunker = Chunker()
    trace: list[str] = []
    _patch_single_block(monkeypatch, chunker, resolved_method=method)

    def fake_metadata(*_args: Any, **_kwargs: Any) -> list[SimpleNamespace]:
        trace.append("metadata")
        raise RuntimeError("metadata failure")

    def fake_plain(*_args: Any, **_kwargs: Any) -> list[str]:
        trace.append("plain")
        return ["raw"]

    monkeypatch.setattr(chunker, "chunk_text_with_metadata", fake_metadata)
    monkeypatch.setattr(chunker, "chunk_text", fake_plain)

    chunker.chunk_text_hierarchical_tree("raw", method=method, max_size=10)

    assert trace == ["metadata", "plain"]


@pytest.mark.parametrize("method", ["words", "sentences", "tokens"])
def test_outer_fallback_retries_plain_call_once(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
) -> None:
    chunker = Chunker()
    trace: list[str] = []
    plain_calls = 0
    _patch_single_block(monkeypatch, chunker, resolved_method=method)

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

    tree = chunker.chunk_text_hierarchical_tree("raw", method=method, max_size=10)
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

    error_text = f"{method} metadata sentinel"

    def fail_metadata(*_args: Any, **_kwargs: Any) -> list[SimpleNamespace]:
        raise RuntimeError(error_text)

    monkeypatch.setattr(
        chunker,
        "chunk_text_with_metadata",
        fail_metadata,
    )
    monkeypatch.setattr(chunker, "chunk_text", lambda *_args, **_kwargs: ["raw"])

    records = _capture_log_records(lambda: chunker.chunk_text_hierarchical_tree("raw", method=method, max_size=10))

    expected_message = f"{method} metadata mapping failed, using fallback: {error_text}"
    matching_records = [
        record for record in records if record["level"].name == "DEBUG" and record["message"] == expected_message
    ]
    assert len(matching_records) == 1


def test_token_metadata_failure_has_stable_debug_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunker = Chunker()
    _patch_single_block(monkeypatch, chunker, resolved_method="tokens")

    error_text = "token metadata sentinel"

    def fail_metadata(*_args: Any, **_kwargs: Any) -> list[SimpleNamespace]:
        raise RuntimeError(error_text)

    monkeypatch.setattr(
        chunker,
        "chunk_text_with_metadata",
        fail_metadata,
    )
    monkeypatch.setattr(chunker, "chunk_text", lambda *_args, **_kwargs: ["raw"])

    records = _capture_log_records(lambda: chunker.chunk_text_hierarchical_tree("raw", method="tokens", max_size=10))

    expected_message = f"Token metadata mapping failed, using fallback: {error_text}"
    matching_records = [
        record for record in records if record["level"].name == "DEBUG" and record["message"] == expected_message
    ]
    assert len(matching_records) == 1


def test_outer_offset_failure_has_stable_warning_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunker = Chunker()
    plain_calls = 0
    _patch_single_block(monkeypatch, chunker, resolved_method="words")

    def fail_metadata(*_args: Any, **_kwargs: Any) -> list[SimpleNamespace]:
        raise RuntimeError("outer metadata sentinel")

    monkeypatch.setattr(
        chunker,
        "chunk_text_with_metadata",
        fail_metadata,
    )

    def fake_plain(*_args: Any, **_kwargs: Any) -> list[str]:
        nonlocal plain_calls
        plain_calls += 1
        if plain_calls == 1:
            raise RuntimeError("outer plain sentinel")
        return ["raw"]

    monkeypatch.setattr(chunker, "chunk_text", fake_plain)

    records = _capture_log_records(lambda: chunker.chunk_text_hierarchical_tree("raw", method="words", max_size=10))

    expected_message = "Offset mapping failed for method=words: outer plain sentinel; " "using naive offsets"
    matching_records = [
        record for record in records if record["level"].name == "WARNING" and record["message"] == expected_message
    ]
    assert len(matching_records) == 1
