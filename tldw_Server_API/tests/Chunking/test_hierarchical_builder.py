"""Direct tree mutation and per-call coordination contracts."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.core.Chunking.exceptions import InvalidInputError
from tldw_Server_API.app.core.Chunking.hierarchical import builder, service
from tldw_Server_API.app.core.Chunking.hierarchical.models import (
    HierarchyTextViews,
    ResolvedHierarchyOptions,
)


class FakeContext:
    """Mutable minimal hierarchy context with observable calls."""

    def __init__(self) -> None:
        self.config = SimpleNamespace(
            default_method=SimpleNamespace(value="structure_aware"),
            default_max_size=10,
            default_overlap=0,
            language="en",
        )
        self.calls: list[Any] = []

    def _enforce_text_size(self, text: str, *, source: str) -> None:
        self.calls.append(("enforce", text, source))

    def _normalize_method_argument(self, method: Any) -> Any:
        self.calls.append(("normalize", method))
        return method

    def _resolve_method(self, method: Any, language: Any, options: dict[str, Any]) -> Any:
        self.calls.append(("resolve", method, language, options))
        return method

    def _sanitize_input(self, text: str, *, suppress_security_log: bool = False) -> str:
        self.calls.append(("sanitize", text, suppress_security_log))
        return text

    def chunk_text(self, *_args: Any, **_kwargs: Any) -> list[Any]:
        pytest.fail("structure-aware builder must not chunk through the context")

    def chunk_text_with_metadata(self, *_args: Any, **_kwargs: Any) -> list[Any]:
        pytest.fail("structure-aware builder must not request metadata")

    def normalize_chunk_type(self, value: Any) -> Any:
        return value


@pytest.fixture
def options() -> ResolvedHierarchyOptions:
    return ResolvedHierarchyOptions("structure_aware", 10, 0, "en", {"nested": []}, True)


@pytest.mark.parametrize(
    ("text", "spans", "expected"),
    [
        (
            "abc",
            [],
            {"kind": "root", "level": 0, "title": None, "start_offset": 0, "end_offset": 3, "children": []},
        ),
        (
            "abc",
            [(0, 3, "paragraph")],
            {
                "kind": "root",
                "level": 0,
                "title": None,
                "start_offset": 0,
                "end_offset": 3,
                "children": [
                    {
                        "kind": "section",
                        "level": 1,
                        "title": None,
                        "start_offset": 0,
                        "end_offset": 3,
                        "children": [{"span": (0, 3, "paragraph"), "children": []}],
                    }
                ],
            },
        ),
        (
            "abc# H",
            [(0, 3, "paragraph"), (3, 6, "header_atx")],
            {
                "kind": "root",
                "level": 0,
                "title": None,
                "start_offset": 0,
                "end_offset": 6,
                "children": [
                    {
                        "kind": "section",
                        "level": 1,
                        "title": None,
                        "start_offset": 0,
                        "end_offset": 3,
                        "children": [{"span": (0, 3, "paragraph"), "children": []}],
                    },
                    {
                        "kind": "section",
                        "level": 1,
                        "title": "H",
                        "start_offset": 3,
                        "end_offset": 6,
                        "source_kind": "header_atx",
                        "children": [{"span": (3, 6, "header_atx"), "children": []}],
                    },
                ],
            },
        ),
        (
            "# A## B## C# D",
            [(0, 3, "header_atx"), (3, 7, "header_atx"), (7, 11, "header_atx"), (11, 14, "header_atx")],
            {
                "kind": "root",
                "level": 0,
                "title": None,
                "start_offset": 0,
                "end_offset": 14,
                "children": [
                    {
                        "kind": "section",
                        "level": 1,
                        "title": "A",
                        "start_offset": 0,
                        "end_offset": 11,
                        "source_kind": "header_atx",
                        "children": [
                            {"span": (0, 3, "header_atx"), "children": []},
                            {
                                "kind": "section",
                                "level": 2,
                                "title": "B",
                                "start_offset": 3,
                                "end_offset": 7,
                                "source_kind": "header_atx",
                                "children": [{"span": (3, 7, "header_atx"), "children": []}],
                            },
                            {
                                "kind": "section",
                                "level": 2,
                                "title": "C",
                                "start_offset": 7,
                                "end_offset": 11,
                                "source_kind": "header_atx",
                                "children": [{"span": (7, 11, "header_atx"), "children": []}],
                            },
                        ],
                    },
                    {
                        "kind": "section",
                        "level": 1,
                        "title": "D",
                        "start_offset": 11,
                        "end_offset": 14,
                        "source_kind": "header_atx",
                        "children": [{"span": (11, 14, "header_atx"), "children": []}],
                    },
                ],
            },
        ),
        (
            "# A## B**C****C**",
            [(0, 3, "header_atx"), (3, 7, "header_atx"), (7, 12, "bold_subsection"), (12, 17, "bold_subsection")],
            {
                "kind": "root",
                "level": 0,
                "title": None,
                "start_offset": 0,
                "end_offset": 17,
                "children": [
                    {
                        "kind": "section",
                        "level": 1,
                        "title": "A",
                        "start_offset": 0,
                        "end_offset": 17,
                        "source_kind": "header_atx",
                        "children": [
                            {"span": (0, 3, "header_atx"), "children": []},
                            {
                                "kind": "section",
                                "level": 2,
                                "title": "B",
                                "start_offset": 3,
                                "end_offset": 17,
                                "source_kind": "header_atx",
                                "children": [
                                    {"span": (3, 7, "header_atx"), "children": []},
                                    {
                                        "kind": "section",
                                        "level": 3,
                                        "title": "**C**",
                                        "start_offset": 7,
                                        "end_offset": 12,
                                        "source_kind": "bold_subsection",
                                        "children": [{"span": (7, 12, "bold_subsection"), "children": []}],
                                    },
                                    {
                                        "kind": "section",
                                        "level": 3,
                                        "title": "**C**",
                                        "start_offset": 12,
                                        "end_offset": 17,
                                        "source_kind": "bold_subsection",
                                        "children": [{"span": (12, 17, "bold_subsection"), "children": []}],
                                    },
                                ],
                            },
                        ],
                    }
                ],
            },
        ),
        (
            "**A**x",
            [(0, 5, "bold_subsection"), (5, 6, "paragraph")],
            {
                "kind": "root",
                "level": 0,
                "title": None,
                "start_offset": 0,
                "end_offset": 6,
                "children": [
                    {
                        "kind": "section",
                        "level": 1,
                        "title": None,
                        "start_offset": 0,
                        "end_offset": 6,
                        "children": [
                            {
                                "kind": "section",
                                "level": 2,
                                "title": "**A**",
                                "start_offset": 0,
                                "end_offset": 6,
                                "source_kind": "bold_subsection",
                                "children": [
                                    {"span": (0, 5, "bold_subsection"), "children": []},
                                    {"span": (5, 6, "paragraph"), "children": []},
                                ],
                            }
                        ],
                    }
                ],
            },
        ),
        (
            "# H",
            [(0, 3, "header_atx")],
            {
                "kind": "root",
                "level": 0,
                "title": None,
                "start_offset": 0,
                "end_offset": 3,
                "children": [
                    {
                        "kind": "section",
                        "level": 1,
                        "title": "H",
                        "start_offset": 0,
                        "end_offset": 3,
                        "source_kind": "header_atx",
                        "children": [{"span": (0, 3, "header_atx"), "children": []}],
                    }
                ],
            },
        ),
        (
            " \n",
            [(0, 2, "blank")],
            {"kind": "root", "level": 0, "title": None, "start_offset": 0, "end_offset": 2, "children": []},
        ),
    ],
    ids=[
        "no-spans",
        "preface-tail",
        "preface-before-header",
        "atx-nesting-and-siblings",
        "nearest-non-bold-and-repeated-siblings",
        "bold-preface",
        "header-only",
        "blank",
    ],
)
def test_builder_tree_shapes(
    monkeypatch: pytest.MonkeyPatch,
    options: ResolvedHierarchyOptions,
    text: str,
    spans: list[tuple[int, int, str]],
    expected: dict[str, Any],
) -> None:
    context = FakeContext()
    texts = HierarchyTextViews(text, text, text)
    original = deepcopy((texts, spans, options))
    returned: list[dict[str, Any]] = []

    def fake_leaf(ctx: Any, views: Any, span: Any, opts: Any) -> dict[str, Any]:
        assert ctx is context
        assert views is texts
        assert opts is options
        block = {"span": span, "children": []}
        returned.append(block)
        return block

    monkeypatch.setattr(builder, "build_leaf_block", fake_leaf)
    root = builder.build_hierarchy_tree(context, texts, spans, options)
    assert root == expected
    assert (texts, spans, options) == original

    appended = []

    def collect(node: dict[str, Any]) -> None:
        for child in node["children"]:
            if "span" in child:
                appended.append(child)
            collect(child)

    collect(root)
    assert len(appended) == len(returned)
    assert all(actual is fresh for actual, fresh in zip(appended, returned))
    assert len({id(block) for block in returned}) == len(returned)


def test_builder_does_not_append_none(monkeypatch: pytest.MonkeyPatch, options: ResolvedHierarchyOptions) -> None:
    monkeypatch.setattr(builder, "build_leaf_block", lambda *_args: None)
    assert builder.build_hierarchy_tree(
        FakeContext(), HierarchyTextViews("x", "x", "x"), [(0, 1, "paragraph")], options
    ) == {
        "kind": "root",
        "level": 0,
        "title": None,
        "start_offset": 0,
        "end_offset": 1,
        "children": [
            {"kind": "section", "level": 1, "title": None, "start_offset": 0, "end_offset": 1, "children": []}
        ],
    }


def test_real_builder_and_leaf_keep_header_and_preface(options: ResolvedHierarchyOptions) -> None:
    context = FakeContext()
    texts = HierarchyTextViews("raw# H", "RAW# H", "RAW# H")
    assert builder.build_hierarchy_tree(context, texts, [(0, 3, "paragraph"), (3, 6, "header_atx")], options) == {
        "kind": "root",
        "level": 0,
        "title": None,
        "start_offset": 0,
        "end_offset": 6,
        "children": [
            {
                "kind": "section",
                "level": 1,
                "title": None,
                "start_offset": 0,
                "end_offset": 3,
                "children": [
                    {
                        "kind": "paragraph",
                        "start_offset": 0,
                        "end_offset": 3,
                        "children": [],
                        "chunks": [
                            {
                                "type": "text",
                                "text": "RAW",
                                "metadata": {
                                    "method": "structure_aware",
                                    "start_offset": 0,
                                    "end_offset": 3,
                                    "language": "en",
                                    "paragraph_kind": "paragraph",
                                },
                            },
                        ],
                    }
                ],
            },
            {
                "kind": "section",
                "level": 1,
                "title": "H",
                "start_offset": 3,
                "end_offset": 6,
                "source_kind": "header_atx",
                "children": [
                    {
                        "kind": "header_atx",
                        "start_offset": 3,
                        "end_offset": 6,
                        "children": [],
                        "chunks": [
                            {
                                "type": "text",
                                "text": "# H",
                                "metadata": {
                                    "method": "structure_aware",
                                    "start_offset": 3,
                                    "end_offset": 6,
                                    "language": "en",
                                    "paragraph_kind": "header_atx",
                                },
                            },
                        ],
                    }
                ],
            },
        ],
    }
    assert context.calls == []


def test_service_empty_input_has_no_work(monkeypatch: pytest.MonkeyPatch) -> None:
    context = FakeContext()
    monkeypatch.setattr(service, "compute_paragraph_spans", lambda *_args: pytest.fail("span computation"))
    monkeypatch.setattr(service, "build_hierarchy_tree", lambda *_args: pytest.fail("tree construction"))
    assert service.HierarchyService(context).build_tree("") == {
        "type": "hierarchical",
        "schema_version": 1,
        "root": {"kind": "root", "children": []},
    }
    assert context.calls == []


def test_service_rejects_non_string_before_other_work() -> None:
    context = FakeContext()
    with pytest.raises(InvalidInputError, match="Expected string input, got int"):
        service.HierarchyService(context).build_tree(3)
    assert context.calls == []


def test_service_observes_mutated_defaults_and_replaced_method(monkeypatch: pytest.MonkeyPatch) -> None:
    context = FakeContext()
    coordinator = service.HierarchyService(context)
    monkeypatch.setattr(service, "compute_paragraph_spans", lambda *_args: [])
    root = {"kind": "root", "children": []}
    received: list[Any] = []

    def fake_tree(ctx: Any, texts: Any, spans: Any, opts: Any) -> dict[str, Any]:
        received.append((ctx, texts, spans, opts))
        return root

    monkeypatch.setattr(service, "build_hierarchy_tree", fake_tree)
    assert coordinator.build_tree("raw") == {
        "type": "hierarchical",
        "schema_version": 1,
        "method": "structure_aware",
        "language": "en",
        "max_size": 10,
        "overlap": 0,
        "root": root,
    }
    context.config.default_method.value = "fixed"
    context.config.default_max_size = "20"
    context.config.default_overlap = 2
    context.config.language = "de"
    context._sanitize_input = lambda *_args, **_kwargs: "NEW"
    assert coordinator.build_tree("raw") == {
        "type": "hierarchical",
        "schema_version": 1,
        "method": "fixed",
        "language": "de",
        "max_size": "20",
        "overlap": 2,
        "root": root,
    }
    assert received[0] == (
        context,
        HierarchyTextViews("raw", "raw", "raw"),
        [],
        ResolvedHierarchyOptions("structure_aware", 10, 0, "en", {}, True),
    )
    assert received[1] == (
        context,
        HierarchyTextViews("raw", "NEW", "NEW"),
        [],
        ResolvedHierarchyOptions("fixed", "20", 2, "de", {}, True),
    )


@pytest.mark.parametrize("grouping", [{"by_kind": True, "element_weights": {"paragraph": 2}}, {}, None])
def test_service_ignores_present_template_grouping(monkeypatch: pytest.MonkeyPatch, grouping: Any) -> None:
    template = {"hierarchy": {"grouping": grouping}}

    def fixed_spans(text: str, supplied_template: Any) -> list[tuple[int, int, str]]:
        assert text == "raw"
        assert supplied_template is template
        return [(0, 3, "paragraph")]

    monkeypatch.setattr(service, "compute_paragraph_spans", fixed_spans)
    result = service.HierarchyService(FakeContext()).build_tree("raw", template=template)
    assert set(result) == {"type", "schema_version", "method", "language", "max_size", "overlap", "root"}
    assert result == {
        "type": "hierarchical",
        "schema_version": 1,
        "method": "structure_aware",
        "language": "en",
        "max_size": 10,
        "overlap": 0,
        "root": {
            "kind": "root",
            "level": 0,
            "title": None,
            "start_offset": 0,
            "end_offset": 3,
            "children": [
                {
                    "kind": "section",
                    "level": 1,
                    "title": None,
                    "start_offset": 0,
                    "end_offset": 3,
                    "children": [
                        {
                            "kind": "paragraph",
                            "start_offset": 0,
                            "end_offset": 3,
                            "children": [],
                            "chunks": [
                                {
                                    "type": "text",
                                    "text": "raw",
                                    "metadata": {
                                        "method": "structure_aware",
                                        "start_offset": 0,
                                        "end_offset": 3,
                                        "language": "en",
                                        "paragraph_kind": "paragraph",
                                    },
                                }
                            ],
                        }
                    ],
                }
            ],
        },
    }


def test_service_does_not_inspect_malformed_template_hierarchy(monkeypatch: pytest.MonkeyPatch) -> None:
    class UnusedHierarchy:
        def __bool__(self) -> bool:
            pytest.fail("tree coordination must not evaluate template hierarchy")

        def get(self, *_args: Any) -> Any:
            pytest.fail("tree coordination must not inspect template hierarchy")

    template = {"hierarchy": UnusedHierarchy()}

    def fixed_spans(text: str, supplied_template: Any) -> list[tuple[int, int, str]]:
        assert text == "raw"
        assert supplied_template is template
        return [(0, 3, "paragraph")]

    monkeypatch.setattr(service, "compute_paragraph_spans", fixed_spans)
    result = service.HierarchyService(FakeContext()).build_tree("raw", template=template)
    assert set(result) == {"type", "schema_version", "method", "language", "max_size", "overlap", "root"}
    assert result["root"] == {
        "kind": "root",
        "level": 0,
        "title": None,
        "start_offset": 0,
        "end_offset": 3,
        "children": [
            {
                "kind": "section",
                "level": 1,
                "title": None,
                "start_offset": 0,
                "end_offset": 3,
                "children": [
                    {
                        "kind": "paragraph",
                        "start_offset": 0,
                        "end_offset": 3,
                        "children": [],
                        "chunks": [
                            {
                                "type": "text",
                                "text": "raw",
                                "metadata": {
                                    "method": "structure_aware",
                                    "start_offset": 0,
                                    "end_offset": 3,
                                    "language": "en",
                                    "paragraph_kind": "paragraph",
                                },
                            }
                        ],
                    }
                ],
            }
        ],
    }
