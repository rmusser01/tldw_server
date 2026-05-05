from __future__ import annotations

from tldw_Server_API.app.core.CodeGraph.extractors.csharp_extractor import CSharpTreeSitterExtractor
from tldw_Server_API.app.core.CodeGraph.models import ExtractionResult

CSHARP_FIXTURE = b"""
using System;
using Collections = System.Collections.Generic;

namespace Example.App;

public class Greeter {
    public Greeter() {
        Setup();
    }

    public string Name { get; set; }

    public string Greet(string name) {
        return Helper(name);
    }

    private string Helper(string value) {
        return value.ToUpperInvariant();
    }
}

internal interface IMarker {
    void Mark();
}

public record Person(string Name);

public struct Point {
    public int X { get; set; }
}

public enum Mode {
    Basic,
    Advanced
}
"""


def test_csharp_extractor_records_namespaces_imports_types_members_and_calls() -> None:
    """Extract conservative C# symbols and same-file method calls."""
    result = CSharpTreeSitterExtractor().extract(
        workspace_key="ws",
        file_path="src/Example/App/Greeter.cs",
        source=CSHARP_FIXTURE,
    )

    nodes_by_kind_name = {(node.kind, node.name): node for node in result.nodes}

    assert ("module", "Greeter") in nodes_by_kind_name
    assert ("namespace", "Example.App") in nodes_by_kind_name
    assert ("import", "System") in nodes_by_kind_name
    assert ("import", "Collections = System.Collections.Generic") in nodes_by_kind_name
    assert ("class", "Greeter") in nodes_by_kind_name
    assert ("constructor", "Greeter") in nodes_by_kind_name
    assert ("property", "Name") in nodes_by_kind_name
    assert ("method", "Greet") in nodes_by_kind_name
    assert ("method", "Helper") in nodes_by_kind_name
    assert ("interface", "IMarker") in nodes_by_kind_name
    assert ("method", "Mark") in nodes_by_kind_name
    assert ("record", "Person") in nodes_by_kind_name
    assert ("struct", "Point") in nodes_by_kind_name
    assert ("property", "X") in nodes_by_kind_name
    assert ("enum", "Mode") in nodes_by_kind_name

    greeter = nodes_by_kind_name[("class", "Greeter")]
    constructor = nodes_by_kind_name[("constructor", "Greeter")]
    name_property = nodes_by_kind_name[("property", "Name")]
    greet = nodes_by_kind_name[("method", "Greet")]
    helper = nodes_by_kind_name[("method", "Helper")]
    marker = nodes_by_kind_name[("interface", "IMarker")]
    mark = nodes_by_kind_name[("method", "Mark")]
    point_x = nodes_by_kind_name[("property", "X")]

    assert greeter.qualified_name == "Example.App.Greeter"
    assert greeter.visibility == "public"
    assert constructor.qualified_name == "Example.App.Greeter.Greeter"
    assert name_property.qualified_name == "Example.App.Greeter.Name"
    assert greet.qualified_name == "Example.App.Greeter.Greet"
    assert helper.qualified_name == "Example.App.Greeter.Helper"
    assert helper.visibility == "private"
    assert marker.qualified_name == "Example.App.IMarker"
    assert marker.visibility == "internal"
    assert mark.qualified_name == "Example.App.IMarker.Mark"
    assert point_x.qualified_name == "Example.App.Point.X"
    assert (greet.id, helper.id) in {(edge.source, edge.target) for edge in result.edges}
    assert any(
        ref.reference_kind == "call" and ref.reference_name == "Setup" and ref.from_node_id == constructor.id
        for ref in result.unresolved_refs
    )
    assert any(
        ref.reference_kind == "call" and ref.reference_name == "value.ToUpperInvariant"
        for ref in result.unresolved_refs
    )
    assert any(
        ref.reference_kind == "import" and ref.reference_name == "Collections = System.Collections.Generic"
        for ref in result.unresolved_refs
    )
    assert result.errors == ()


def test_csharp_extractor_marks_parse_errors() -> None:
    """Return extraction errors for invalid C# syntax."""
    result = CSharpTreeSitterExtractor().extract(
        workspace_key="ws",
        file_path="src/Broken.cs",
        source=b"class Broken {",
    )

    assert result == ExtractionResult(errors=("C# parse error",))


def test_csharp_extractor_uses_deterministic_node_ids() -> None:
    """Use stable node IDs across repeated C# extraction."""
    first = CSharpTreeSitterExtractor().extract(
        workspace_key="ws",
        file_path="src/Greeter.cs",
        source=CSHARP_FIXTURE,
    )
    second = CSharpTreeSitterExtractor().extract(
        workspace_key="ws",
        file_path="src/Greeter.cs",
        source=CSHARP_FIXTURE,
    )

    assert [node.id for node in first.nodes] == [node.id for node in second.nodes]
