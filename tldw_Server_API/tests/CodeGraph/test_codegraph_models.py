"""Tests for shared CodeGraph model serialization helpers."""

from __future__ import annotations

from tldw_Server_API.app.core.CodeGraph.models import CodeGraphNode, codegraph_node_to_dict


def test_codegraph_node_to_dict_serializes_public_fields() -> None:
    """Serialize CodeGraph nodes into stable public MCP payload fields."""
    node = CodeGraphNode(
        id="node_helper",
        identity_key="helper-key",
        kind="function",
        name="helper",
        qualified_name="pkg.helper",
        file_path="pkg/sample.py",
        language="python",
        start_line=4,
        end_line=8,
        start_column=1,
        end_column=12,
        signature="helper(value)",
        docstring="Help.",
        visibility="public",
        flags=("async",),
        metadata={"decorators": ["cached"]},
    )

    assert codegraph_node_to_dict(node) == {
        "id": "node_helper",
        "kind": "function",
        "name": "helper",
        "qualified_name": "pkg.helper",
        "file_path": "pkg/sample.py",
        "language": "python",
        "start_line": 4,
        "end_line": 8,
        "start_column": 1,
        "end_column": 12,
        "signature": "helper(value)",
        "docstring": "Help.",
        "visibility": "public",
        "flags": ["async"],
        "metadata": {"decorators": ["cached"]},
    }
