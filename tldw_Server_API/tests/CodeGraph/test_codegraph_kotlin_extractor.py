from __future__ import annotations

import pytest

from tldw_Server_API.app.core.CodeGraph.extractors.kotlin_extractor import KotlinTreeSitterExtractor
from tldw_Server_API.app.core.CodeGraph.extractors.tree_sitter_loader import load_parser
from tldw_Server_API.app.core.CodeGraph.models import ExtractionResult

pytestmark = pytest.mark.skipif(
    not load_parser("kotlin").available,
    reason="tree-sitter-kotlin parser is not available",
)

KOTLIN_FIXTURE = b"""
package com.example.app

import com.example.tools.Helper
import com.example.tools.Helper as ToolHelper
import kotlin.collections.*
import kotlin.collections.List

internal class Greeter {
    fun greet(name: String): String {
        return helper(name)
    }

    private fun helper(value: String): String {
        return value.uppercase()
    }
}

object Registry {
    fun create(): Greeter {
        com.example.factories.GreeterFactory.create()
        return Greeter()
    }
}

@Deprecated("use NewMarker")
internal interface Marker {
    fun mark()
}
"""


def test_kotlin_extractor_records_package_imports_types_functions_and_calls() -> None:
    """Extract conservative Kotlin symbols and same-file function calls."""
    result = KotlinTreeSitterExtractor().extract(
        workspace_key="ws",
        file_path="src/main/kotlin/com/example/app/Greeter.kt",
        source=KOTLIN_FIXTURE,
    )

    nodes_by_kind_name = {(node.kind, node.name): node for node in result.nodes}

    assert ("module", "Greeter") in nodes_by_kind_name
    assert ("package", "com.example.app") in nodes_by_kind_name
    assert ("import", "com.example.tools.Helper") in nodes_by_kind_name
    assert ("import", "com.example.tools.Helper as ToolHelper") in nodes_by_kind_name
    assert ("import", "kotlin.collections.*") in nodes_by_kind_name
    assert ("import", "kotlin.collections.List") in nodes_by_kind_name
    assert ("class", "Greeter") in nodes_by_kind_name
    assert ("function", "greet") in nodes_by_kind_name
    assert ("function", "helper") in nodes_by_kind_name
    assert ("object", "Registry") in nodes_by_kind_name
    assert ("function", "create") in nodes_by_kind_name
    assert ("interface", "Marker") in nodes_by_kind_name

    greeter = nodes_by_kind_name[("class", "Greeter")]
    marker = nodes_by_kind_name[("interface", "Marker")]
    registry = nodes_by_kind_name[("object", "Registry")]
    greet = nodes_by_kind_name[("function", "greet")]
    helper = nodes_by_kind_name[("function", "helper")]
    create = nodes_by_kind_name[("function", "create")]

    assert greeter.qualified_name == "com.example.app.Greeter"
    assert greeter.visibility == "internal"
    assert marker.qualified_name == "com.example.app.Marker"
    assert marker.visibility == "internal"
    assert registry.qualified_name == "com.example.app.Registry"
    assert greet.qualified_name == "com.example.app.Greeter.greet"
    assert helper.qualified_name == "com.example.app.Greeter.helper"
    assert create.qualified_name == "com.example.app.Registry.create"
    assert (greet.id, helper.id) in {(edge.source, edge.target) for edge in result.edges}
    assert any(
        ref.reference_kind == "call" and ref.reference_name == "Greeter" and ref.from_node_id == create.id
        for ref in result.unresolved_refs
    )
    assert any(
        ref.reference_kind == "import" and ref.reference_name == "com.example.tools.Helper"
        for ref in result.unresolved_refs
    )
    assert any(
        ref.reference_kind == "import" and ref.reference_name == "com.example.tools.Helper as ToolHelper"
        for ref in result.unresolved_refs
    )
    assert any(
        ref.reference_kind == "import" and ref.reference_name == "kotlin.collections.*"
        for ref in result.unresolved_refs
    )
    assert any(
        ref.reference_kind == "call" and ref.reference_name == "com.example.factories.GreeterFactory.create"
        for ref in result.unresolved_refs
    )
    assert result.errors == ()


def test_kotlin_extractor_marks_parse_errors() -> None:
    """Return extraction errors for invalid Kotlin syntax."""
    result = KotlinTreeSitterExtractor().extract(
        workspace_key="ws",
        file_path="src/Broken.kt",
        source=b"class Broken {",
    )

    assert result == ExtractionResult(errors=("Kotlin parse error",))


def test_kotlin_extractor_uses_deterministic_node_ids() -> None:
    """Use stable node IDs across repeated Kotlin extraction."""
    first = KotlinTreeSitterExtractor().extract(
        workspace_key="ws",
        file_path="src/Greeter.kt",
        source=KOTLIN_FIXTURE,
    )
    second = KotlinTreeSitterExtractor().extract(
        workspace_key="ws",
        file_path="src/Greeter.kt",
        source=KOTLIN_FIXTURE,
    )

    assert [node.id for node in first.nodes] == [node.id for node in second.nodes]
