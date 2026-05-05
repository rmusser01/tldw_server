"""Tests for bounded workspace-safe CodeGraph context assembly."""

from __future__ import annotations

from pathlib import Path

from tldw_Server_API.app.core.CodeGraph.context import CodeGraphContextBuilder, rank_context_nodes
from tldw_Server_API.app.core.CodeGraph.models import CodeGraphNode


def test_context_builder_reads_bounded_source_snippet(tmp_path: Path) -> None:
    """Read a source snippet around the indexed node with surrounding context."""
    source = tmp_path / "pkg" / "sample.py"
    source.parent.mkdir()
    source.write_text(
        "\n".join(
            [
                "import os",
                "",
                "def entry():",
                "    helper()",
                "",
                "def helper(value):",
                "    return value + 1",
                "",
                "def leaf():",
                "    return helper(1)",
            ]
        ),
        encoding="utf-8",
    )
    builder = CodeGraphContextBuilder(
        workspace_root=tmp_path,
        max_context_chars=500,
        max_file_size_bytes=10_000,
    )

    result = builder.build(
        task="update helper",
        nodes=(_node("node_helper", "pkg/sample.py", start_line=6, end_line=7),),
        relationships=(),
        max_files=3,
        include_code=True,
    )

    assert result["files"][0]["path"] == "pkg/sample.py"
    snippet = result["files"][0]["snippets"][0]
    assert snippet["start_line"] == 3
    assert snippet["end_line"] == 10
    assert "def helper" in snippet["text"]
    assert result["truncation"]["truncated"] is False


def test_rank_context_nodes_prefers_task_token_matches() -> None:
    """Rank direct task-token matches ahead of weaker search-order candidates."""
    weak = _node("node_weak", "pkg/config.py", name="parse_config", qualified_name="parse_config")
    strong = _node("node_strong", "pkg/helper.py", name="helper", qualified_name="services.helper")

    ranked = rank_context_nodes("fix helper behavior", (weak, strong), relationships=())

    assert [node.id for node in ranked] == ["node_strong", "node_weak"]


def test_rank_context_nodes_boosts_related_nodes_on_ties() -> None:
    """Prefer task-matching nodes connected to the selected relationship neighborhood."""
    unrelated = _node(
        "node_unrelated",
        "pkg/helpers_misc.py",
        name="helper_misc",
        qualified_name="helpers.helper_misc",
    )
    related = _node(
        "node_related",
        "pkg/helpers_related.py",
        name="helper_related",
        qualified_name="helpers.helper_related",
    )
    entry = _node("node_entry", "pkg/entry.py", name="entry", qualified_name="entry")
    relationship = {
        "id": "edge_entry_related",
        "source": {"id": "node_entry"},
        "target": {"id": "node_related"},
    }

    ranked = rank_context_nodes("helper", (unrelated, related, entry), relationships=(relationship,))

    assert [node.id for node in ranked[:2]] == ["node_related", "node_unrelated"]


def test_rank_context_nodes_ignores_relationships_to_non_candidate_nodes() -> None:
    """Avoid ranking high-degree external hubs above local candidate relationships."""
    hub = _node("node_hub", "pkg/helper_hub.py", name="helper_hub", qualified_name="helpers.helper_hub")
    related = _node(
        "node_related",
        "pkg/helper_related.py",
        name="helper_related",
        qualified_name="helpers.helper_related",
    )
    peer = _node("node_peer", "pkg/helper_peer.py", name="helper_peer", qualified_name="helpers.helper_peer")
    relationships = (
        {"id": "edge_hub_external_a", "source": {"id": "node_hub"}, "target": {"id": "node_external_a"}},
        {"id": "edge_hub_external_b", "source": {"id": "node_external_b"}, "target": {"id": "node_hub"}},
        {"id": "edge_peer_related", "source": {"id": "node_peer"}, "target": {"id": "node_related"}},
    )

    ranked = rank_context_nodes("helper", (hub, related, peer), relationships=relationships)

    assert [node.id for node in ranked[:2]] == ["node_related", "node_peer"]


def test_rank_context_nodes_boosts_filename_stem_matches() -> None:
    """Rank filename stem matches above weaker path substring matches."""
    weak = _node("node_weak", "pkg/app_config.py", name="handler", qualified_name="handler")
    strong = _node("node_strong", "pkg/app.py", name="handler", qualified_name="handler")

    ranked = rank_context_nodes("update app", (weak, strong), relationships=())

    assert [node.id for node in ranked] == ["node_strong", "node_weak"]


def test_context_builder_groups_duplicate_file_snippets(tmp_path: Path) -> None:
    """Group multiple node snippets from the same file under one file entry."""
    source = tmp_path / "pkg" / "sample.py"
    source.parent.mkdir()
    source.write_text(
        "\n".join(
            [
                "def entry():",
                "    helper()",
                "",
                "def helper():",
                "    return leaf()",
                "",
                "def leaf():",
                "    return 1",
            ]
        ),
        encoding="utf-8",
    )
    builder = CodeGraphContextBuilder(
        workspace_root=tmp_path,
        max_context_chars=1_000,
        max_file_size_bytes=10_000,
    )

    result = builder.build(
        task="inspect helper",
        nodes=(
            _node("node_helper", "pkg/sample.py", start_line=4, end_line=5),
            _node("node_leaf", "pkg/sample.py", start_line=7, end_line=8),
        ),
        relationships=(),
        max_files=3,
        include_code=True,
    )

    assert [file_context["path"] for file_context in result["files"]] == ["pkg/sample.py"]
    assert len(result["files"][0]["snippets"]) == 2


def test_context_builder_respects_context_character_budget(tmp_path: Path) -> None:
    """Truncate snippet text when the context character budget is exhausted."""
    source = tmp_path / "pkg" / "large.py"
    source.parent.mkdir()
    source.write_text(
        "\n".join(
            [
                "def before():",
                "    return 'before'",
                "",
                "def helper():",
                "    return 'this snippet is intentionally too long for the budget'",
                "",
                "def after():",
                "    return 'after'",
            ]
        ),
        encoding="utf-8",
    )
    builder = CodeGraphContextBuilder(
        workspace_root=tmp_path,
        max_context_chars=48,
        max_file_size_bytes=10_000,
    )

    result = builder.build(
        task="short budget",
        nodes=(_node("node_helper", "pkg/large.py", start_line=4, end_line=5),),
        relationships=(),
        max_files=3,
        include_code=True,
    )

    snippet = result["files"][0]["snippets"][0]
    assert len(snippet["text"]) <= 48
    assert snippet["truncated"] is True
    assert result["truncation"]["used_chars"] <= 48
    assert result["truncation"]["truncated"] is True


def test_context_builder_skips_unsafe_paths(tmp_path: Path) -> None:
    """Skip absolute and parent-traversal file paths instead of reading them."""
    builder = CodeGraphContextBuilder(
        workspace_root=tmp_path,
        max_context_chars=500,
        max_file_size_bytes=10_000,
    )

    result = builder.build(
        task="path safety",
        nodes=(
            _node("node_parent", "../outside.py"),
            _node("node_absolute", str(tmp_path / "absolute.py")),
        ),
        relationships=(),
        max_files=3,
        include_code=True,
    )

    assert result["files"] == []
    assert result["truncation"]["skipped_files"] == 2


def test_context_builder_reports_missing_files_without_raising(tmp_path: Path) -> None:
    """Represent missing source files as file-context errors without raising."""
    builder = CodeGraphContextBuilder(
        workspace_root=tmp_path,
        max_context_chars=500,
        max_file_size_bytes=10_000,
    )

    result = builder.build(
        task="missing source",
        nodes=(_node("node_missing", "pkg/missing.py"),),
        relationships=(),
        max_files=3,
        include_code=True,
    )

    assert result["files"] == [
        {
            "path": "pkg/missing.py",
            "language": "python",
            "exists": False,
            "snippets": [],
            "errors": ["source file not found"],
        }
    ]


def test_context_builder_can_exclude_source_text(tmp_path: Path) -> None:
    """Return metadata-only file context when source snippets are disabled."""
    source = tmp_path / "pkg" / "sample.py"
    source.parent.mkdir()
    source.write_text("def helper():\n    return 1\n", encoding="utf-8")
    builder = CodeGraphContextBuilder(
        workspace_root=tmp_path,
        max_context_chars=500,
        max_file_size_bytes=10_000,
    )

    result = builder.build(
        task="metadata only",
        nodes=(_node("node_helper", "pkg/sample.py", start_line=1, end_line=2),),
        relationships=(),
        max_files=3,
        include_code=False,
    )

    assert result["files"] == [
        {
            "path": "pkg/sample.py",
            "language": "python",
            "exists": True,
            "snippets": [],
            "errors": [],
        }
    ]
    assert result["truncation"]["used_chars"] == 0


def _node(
    node_id: str,
    file_path: str,
    *,
    name: str | None = None,
    qualified_name: str | None = None,
    start_line: int | None = 1,
    end_line: int | None = 1,
) -> CodeGraphNode:
    """Build a minimal CodeGraph node for context builder tests."""
    node_name = name or node_id.removeprefix("node_")
    return CodeGraphNode(
        id=node_id,
        identity_key=node_id,
        kind="function",
        name=node_name,
        qualified_name=qualified_name or node_name,
        file_path=file_path,
        language="python",
        start_line=start_line,
        end_line=end_line,
    )
