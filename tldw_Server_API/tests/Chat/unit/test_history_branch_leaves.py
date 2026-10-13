"""Pure D7 P1 leaf-check contract: which live messages make a selected tip stale."""

import pytest

from tldw_Server_API.app.core.Chat.history_selection import (
    HistoryBranchChangedError,
    HistorySelectionError,
    history_branch_leaves,
)

GRAPH = {"kind": "parent_graph_v1"}


def node(mid, parent):
    return {"id": mid, "parent_id": parent}


def test_tip_without_live_child_has_no_leaves():
    nodes = [node("u1", None), node("a1", "u1")]
    assert history_branch_leaves(nodes, GRAPH, "a1") == ()


def test_child_of_tip_makes_it_stale_and_reports_its_leaf():
    nodes = [node("u1", None), node("a1", "u1"), node("u2", "a1")]
    assert history_branch_leaves(nodes, GRAPH, "a1") == ("u2",)


def test_leaves_are_deepest_descendants_in_manifest_order():
    # Two assistant variants under the newer user turn are both current leaves.
    nodes = [
        node("u1", None), node("a1", "u1"), node("u2", "a1"),
        node("a2", "u2"), node("u3", "a2"), node("a2b", "u2"),
    ]
    assert history_branch_leaves(nodes, GRAPH, "a1") == ("u3", "a2b")


def test_assistant_variant_siblings_of_the_tip_do_not_count():
    # Continuing from an older variant is an explicit pager choice, not staleness.
    nodes = [node("u1", None), node("a1", "u1"), node("a1b", "u1")]
    assert history_branch_leaves(nodes, GRAPH, "a1") == ()
    assert history_branch_leaves(nodes, GRAPH, "a1b") == ()


def test_unrelated_branches_are_not_reported():
    nodes = [node("u1", None), node("a1", "u1"), node("x", "u1"), node("u2", "a1")]
    assert history_branch_leaves(nodes, GRAPH, "a1") == ("u2",)


def test_empty_selection_treats_existing_roots_as_children():
    assert history_branch_leaves([], GRAPH, None) == ()
    nodes = [node("u1", None), node("a1", "u1")]
    assert history_branch_leaves(nodes, GRAPH, None) == ("a1",)


def test_legacy_projection_uses_reviewed_order_as_parent_edges():
    status = {"kind": "legacy_linear_v1", "projection_id": "p", "ordered_path_ids": ("two", "one")}
    nodes = [node("one", None), node("two", None)]
    assert history_branch_leaves(nodes, status, "one") == ()
    assert history_branch_leaves(nodes, status, "two") == ("one",)
    nodes.append(node("native", "one"))
    assert history_branch_leaves(nodes, status, "one") == ("native",)
    assert history_branch_leaves(nodes, status, "two") == ("native",)


def test_branch_changed_error_carries_refresh_details():
    error = HistoryBranchChangedError(
        conversation_id="c1", parent_message_id="a1", leaf_ids=("u2",), history_version="7",
    )
    assert isinstance(error, HistorySelectionError)
    assert error.code == "history_branch_changed"
    assert error.details == {
        "conversation_id": "c1",
        "parent_message_id": "a1",
        "leaf_ids": ["u2"],
        "history_version": 7,
    }


def test_plain_selection_errors_have_no_details():
    assert HistorySelectionError("stale_selection").details == {}


@pytest.mark.parametrize("parent", ["missing", "u1"])
def test_unknown_or_childless_tip_is_never_stale(parent):
    assert history_branch_leaves([node("u1", None)], GRAPH, parent) == ()
