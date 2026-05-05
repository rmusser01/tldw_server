from __future__ import annotations

import pytest

from tldw_Server_API.app.core.CodeGraph.extractors.c_family_extractor import CppTreeSitterExtractor
from tldw_Server_API.app.core.CodeGraph.extractors.tree_sitter_loader import load_parser
from tldw_Server_API.app.core.CodeGraph.models import ExtractionResult

pytestmark = pytest.mark.skipif(
    not load_parser("cpp").available,
    reason="tree-sitter-cpp parser is not available",
)

CPP_FIXTURE = b"""
#include <string>

namespace demo {
class Greeter {
public:
    std::string greet(std::string name) {
        return helper(name);
    }

private:
    std::string helper(std::string value) {
        return value;
    }
};

struct Point {
    int x;
};

enum Mode {
    Basic,
    Advanced
};
}
"""


def test_cpp_extractor_records_includes_namespaces_types_methods_and_calls() -> None:
    """Extract conservative C++ symbols and same-file method calls."""
    result = CppTreeSitterExtractor().extract(
        workspace_key="ws",
        file_path="src/demo/greeter.cpp",
        source=CPP_FIXTURE,
    )

    nodes_by_kind_name = {(node.kind, node.name): node for node in result.nodes}

    assert ("module", "greeter") in nodes_by_kind_name
    assert ("import", "<string>") in nodes_by_kind_name
    assert ("namespace", "demo") in nodes_by_kind_name
    assert ("class", "Greeter") in nodes_by_kind_name
    assert ("method", "greet") in nodes_by_kind_name
    assert ("method", "helper") in nodes_by_kind_name
    assert ("struct", "Point") in nodes_by_kind_name
    assert ("enum", "Mode") in nodes_by_kind_name

    namespace = nodes_by_kind_name[("namespace", "demo")]
    greeter = nodes_by_kind_name[("class", "Greeter")]
    greet = nodes_by_kind_name[("method", "greet")]
    helper = nodes_by_kind_name[("method", "helper")]
    include = nodes_by_kind_name[("import", "<string>")]
    point = nodes_by_kind_name[("struct", "Point")]

    assert namespace.qualified_name == "demo"
    assert greeter.qualified_name == "demo.Greeter"
    assert greet.qualified_name == "demo.Greeter.greet"
    assert helper.qualified_name == "demo.Greeter.helper"
    assert point.qualified_name == "demo.Point"
    assert (greet.id, helper.id) in {(edge.source, edge.target) for edge in result.edges}
    assert any(
        ref.reference_kind == "import" and ref.reference_name == "<string>" and ref.from_node_id == include.id
        for ref in result.unresolved_refs
    )
    assert result.errors == ()


def test_cpp_extractor_marks_parse_errors() -> None:
    """Return extraction errors for invalid C++ syntax."""
    result = CppTreeSitterExtractor().extract(
        workspace_key="ws",
        file_path="src/broken.cpp",
        source=b"class Broken {",
    )

    assert result == ExtractionResult(errors=("C++ parse error",))


def test_cpp_extractor_uses_deterministic_node_ids() -> None:
    """Use stable node IDs across repeated C++ extraction."""
    first = CppTreeSitterExtractor().extract(
        workspace_key="ws",
        file_path="src/greeter.cpp",
        source=CPP_FIXTURE,
    )
    second = CppTreeSitterExtractor().extract(
        workspace_key="ws",
        file_path="src/greeter.cpp",
        source=CPP_FIXTURE,
    )

    assert [node.id for node in first.nodes] == [node.id for node in second.nodes]
