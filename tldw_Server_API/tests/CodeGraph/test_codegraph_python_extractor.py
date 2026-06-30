from __future__ import annotations

from tldw_Server_API.app.core.CodeGraph.extractors.python_extractor import PythonAstExtractor


def test_python_extractor_captures_symbols_imports_and_same_file_calls() -> None:
    source = b"""
import os
from app.tools import external_call


class Greeter:
    def greet(self, name):
        return helper(name)


def helper(value):
    external_call(value)
    return value.upper()
"""
    extractor = PythonAstExtractor()

    result = extractor.extract(workspace_key="ws_test", file_path="pkg/sample.py", source=source)

    by_qualified = {node.qualified_name: node for node in result.nodes}

    assert by_qualified["pkg.sample"].kind == "module"
    assert by_qualified["os"].kind == "import"
    assert by_qualified["app.tools.external_call"].kind == "import"
    assert by_qualified["Greeter"].kind == "class"
    assert by_qualified["Greeter.greet"].kind == "method"
    assert by_qualified["helper"].kind == "function"

    greet = by_qualified["Greeter.greet"]
    helper = by_qualified["helper"]

    assert any(
        edge.kind == "calls" and edge.source == greet.id and edge.target == helper.id
        for edge in result.edges
    )
    assert any(
        ref.from_node_id == helper.id
        and ref.reference_name == "external_call"
        and ref.reference_kind == "call"
        for ref in result.unresolved_refs
    )


def test_python_extractor_uses_deterministic_node_ids() -> None:
    source = b"def helper():\n    return 1\n"
    extractor = PythonAstExtractor()

    first = extractor.extract(workspace_key="ws_test", file_path="pkg/sample.py", source=source)
    second = extractor.extract(workspace_key="ws_test", file_path="pkg/sample.py", source=source)

    assert [node.id for node in first.nodes] == [node.id for node in second.nodes]
    assert [edge.id for edge in first.edges] == [edge.id for edge in second.edges]


def test_python_extractor_does_not_link_external_attribute_calls_to_same_file_symbols() -> None:
    source = b"""
def save():
    return True


def persist(external_client):
    external_client.save()
"""
    extractor = PythonAstExtractor()

    result = extractor.extract(workspace_key="ws_test", file_path="pkg/sample.py", source=source)

    by_qualified = {node.qualified_name: node for node in result.nodes}
    save = by_qualified["save"]
    persist = by_qualified["persist"]

    assert not any(
        edge.kind == "calls" and edge.source == persist.id and edge.target == save.id
        for edge in result.edges
    )
    assert any(
        ref.from_node_id == persist.id
        and ref.reference_name == "external_client.save"
        and ref.reference_kind == "call"
        for ref in result.unresolved_refs
    )


def test_python_extractor_reports_value_error_parse_failures() -> None:
    extractor = PythonAstExtractor()

    result = extractor.extract(workspace_key="ws_test", file_path="pkg/broken.py", source=b"def broken():\n\x00\n")

    assert result.nodes == ()
    assert result.errors
