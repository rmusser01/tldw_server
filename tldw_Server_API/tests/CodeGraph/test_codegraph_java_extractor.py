from __future__ import annotations

import pytest

from tldw_Server_API.app.core.CodeGraph.extractors.java_extractor import JavaTreeSitterExtractor
from tldw_Server_API.app.core.CodeGraph.extractors.tree_sitter_loader import load_parser
from tldw_Server_API.app.core.CodeGraph.models import ExtractionResult

pytestmark = pytest.mark.skipif(
    not load_parser("java").available,
    reason="tree-sitter-java parser is not available",
)

JAVA_FIXTURE = b"""
package com.example.app;

import java.util.List;
import static java.util.Collections.emptyList;
import java.util.*;
import com.example.tools.Helper;

public class Greeter {
    public Greeter() {
        setup();
    }

    public String greet(String name) {
        return helper(name);
    }

    private String helper(String value) {
        return value.toUpperCase();
    }
}
"""


def test_java_extractor_records_package_imports_types_methods_and_calls() -> None:
    """Extract conservative Java symbols and same-file method calls."""
    result = JavaTreeSitterExtractor().extract(
        workspace_key="ws",
        file_path="src/main/java/com/example/app/Greeter.java",
        source=JAVA_FIXTURE,
    )

    nodes_by_kind_name = {(node.kind, node.name): node for node in result.nodes}

    assert ("module", "Greeter") in nodes_by_kind_name
    assert ("package", "com.example.app") in nodes_by_kind_name
    assert ("import", "java.util.List") in nodes_by_kind_name
    assert ("import", "static java.util.Collections.emptyList") in nodes_by_kind_name
    assert ("import", "java.util.*") in nodes_by_kind_name
    assert ("import", "com.example.tools.Helper") in nodes_by_kind_name
    assert ("class", "Greeter") in nodes_by_kind_name
    assert ("constructor", "Greeter") in nodes_by_kind_name
    assert ("method", "greet") in nodes_by_kind_name
    assert ("method", "helper") in nodes_by_kind_name

    greeter = nodes_by_kind_name[("class", "Greeter")]
    constructor = nodes_by_kind_name[("constructor", "Greeter")]
    greet = nodes_by_kind_name[("method", "greet")]
    helper = nodes_by_kind_name[("method", "helper")]

    assert greeter.qualified_name == "com.example.app.Greeter"
    assert greeter.visibility == "public"
    assert constructor.qualified_name == "com.example.app.Greeter.Greeter"
    assert greet.qualified_name == "com.example.app.Greeter.greet"
    assert helper.qualified_name == "com.example.app.Greeter.helper"
    assert (greet.id, helper.id) in {(edge.source, edge.target) for edge in result.edges}
    assert any(
        ref.reference_kind == "call" and ref.reference_name == "setup" and ref.from_node_id == constructor.id
        for ref in result.unresolved_refs
    )
    assert any(
        ref.reference_kind == "import" and ref.reference_name == "java.util.List"
        for ref in result.unresolved_refs
    )
    assert any(
        ref.reference_kind == "import" and ref.reference_name == "static java.util.Collections.emptyList"
        for ref in result.unresolved_refs
    )
    assert any(
        ref.reference_kind == "import" and ref.reference_name == "java.util.*"
        for ref in result.unresolved_refs
    )
    assert result.errors == ()


def test_java_extractor_marks_parse_errors() -> None:
    """Return extraction errors for invalid Java syntax."""
    result = JavaTreeSitterExtractor().extract(
        workspace_key="ws",
        file_path="src/Broken.java",
        source=b"class Broken {",
    )

    assert result == ExtractionResult(errors=("Java parse error",))


def test_java_extractor_uses_deterministic_node_ids() -> None:
    """Use stable node IDs across repeated Java extraction."""
    first = JavaTreeSitterExtractor().extract(
        workspace_key="ws",
        file_path="src/Greeter.java",
        source=JAVA_FIXTURE,
    )
    second = JavaTreeSitterExtractor().extract(
        workspace_key="ws",
        file_path="src/Greeter.java",
        source=JAVA_FIXTURE,
    )

    assert [node.id for node in first.nodes] == [node.id for node in second.nodes]
