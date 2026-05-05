from __future__ import annotations

import pytest

from tldw_Server_API.app.core.CodeGraph.extractors.c_family_extractor import CTreeSitterExtractor
from tldw_Server_API.app.core.CodeGraph.extractors.tree_sitter_loader import load_parser
from tldw_Server_API.app.core.CodeGraph.models import ExtractionResult

pytestmark = pytest.mark.skipif(
    not load_parser("c").available,
    reason="tree-sitter-c parser is not available",
)

C_FIXTURE = b"""
#include <stdio.h>

struct Greeter {
    int value;
};

enum Mode {
    MODE_BASIC,
    MODE_ADVANCED
};

static int helper(int value) {
    return value + 1;
}

int greet(int name) {
    return helper(name);
}
"""


def test_c_extractor_records_includes_types_functions_and_calls() -> None:
    """Extract conservative C symbols and same-file function calls."""
    result = CTreeSitterExtractor().extract(
        workspace_key="ws",
        file_path="src/greeter.c",
        source=C_FIXTURE,
    )

    nodes_by_kind_name = {(node.kind, node.name): node for node in result.nodes}

    assert ("module", "greeter") in nodes_by_kind_name
    assert ("import", "<stdio.h>") in nodes_by_kind_name
    assert ("struct", "Greeter") in nodes_by_kind_name
    assert ("enum", "Mode") in nodes_by_kind_name
    assert ("function", "helper") in nodes_by_kind_name
    assert ("function", "greet") in nodes_by_kind_name

    helper = nodes_by_kind_name[("function", "helper")]
    greet = nodes_by_kind_name[("function", "greet")]
    include = nodes_by_kind_name[("import", "<stdio.h>")]

    assert helper.qualified_name == "helper"
    assert greet.qualified_name == "greet"
    assert (greet.id, helper.id) in {(edge.source, edge.target) for edge in result.edges}
    assert any(
        ref.reference_kind == "import" and ref.reference_name == "<stdio.h>" and ref.from_node_id == include.id
        for ref in result.unresolved_refs
    )
    assert result.errors == ()


def test_c_extractor_marks_parse_errors() -> None:
    """Return extraction errors for invalid C syntax."""
    result = CTreeSitterExtractor().extract(
        workspace_key="ws",
        file_path="src/broken.c",
        source=b"int broken(",
    )

    assert result == ExtractionResult(errors=("C parse error",))


def test_c_extractor_uses_deterministic_node_ids() -> None:
    """Use stable node IDs across repeated C extraction."""
    first = CTreeSitterExtractor().extract(
        workspace_key="ws",
        file_path="src/greeter.c",
        source=C_FIXTURE,
    )
    second = CTreeSitterExtractor().extract(
        workspace_key="ws",
        file_path="src/greeter.c",
        source=C_FIXTURE,
    )

    assert [node.id for node in first.nodes] == [node.id for node in second.nodes]
