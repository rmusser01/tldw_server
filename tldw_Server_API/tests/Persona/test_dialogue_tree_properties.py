import pytest
from hypothesis import given, settings, strategies as st

from tldw_Server_API.app.core.Persona.dialogue_tree import (
    DialogueTreeBudget,
    DialogueTreeEngine,
    TreeCandidate,
)


pytestmark = pytest.mark.unit


def _generator(node):
    return [
        TreeCandidate(action_type="assistant", text=f"{node.node_id}-d"),
        TreeCandidate(action_type="assistant", text=f"{node.node_id}-b"),
        TreeCandidate(action_type="assistant", text=f"{node.node_id}-a"),
        TreeCandidate(action_type="assistant", text=f"{node.node_id}-c"),
    ]


@settings(max_examples=40, deadline=None)
@given(
    max_depth=st.integers(min_value=0, max_value=4),
    max_branching=st.integers(min_value=0, max_value=4),
    max_candidates=st.integers(min_value=0, max_value=20),
)
def test_tree_expansion_respects_budget_caps(
    max_depth: int, max_branching: int, max_candidates: int
) -> None:
    engine = DialogueTreeEngine(
        budget=DialogueTreeBudget(
            max_depth=max_depth,
            max_branching=max_branching,
            max_candidates=max_candidates,
            max_provider_calls=1,
        ),
        generators=[_generator],
    )
    result = engine.expand(root_payload={"scenario": "property"})
    non_root_nodes = [node for node in result.nodes if node.parent_node_id is not None]

    assert result.max_depth_seen <= max_depth
    assert len(non_root_nodes) <= max_candidates
    assert all(len(children) <= max_branching for children in result.children_by_parent.values())


@settings(max_examples=40, deadline=None)
@given(
    max_depth=st.integers(min_value=1, max_value=4),
    max_branching=st.integers(min_value=1, max_value=4),
    max_candidates=st.integers(min_value=1, max_value=20),
)
def test_tree_expansion_parent_links_are_acyclic(
    max_depth: int, max_branching: int, max_candidates: int
) -> None:
    engine = DialogueTreeEngine(
        budget=DialogueTreeBudget(
            max_depth=max_depth,
            max_branching=max_branching,
            max_candidates=max_candidates,
            max_provider_calls=1,
        ),
        generators=[_generator],
    )
    result = engine.expand(root_payload={"scenario": "property"})
    node_by_id = {node.node_id: node for node in result.nodes}

    for node in result.nodes:
        seen: set[str] = set()
        current = node
        while current.parent_node_id is not None:
            assert current.node_id not in seen
            seen.add(current.node_id)
            assert current.parent_node_id in node_by_id
            parent = node_by_id[current.parent_node_id]
            assert parent.depth < current.depth
            current = parent
