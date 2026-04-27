import pytest


pytestmark = pytest.mark.unit


def test_tree_expansion_respects_depth_branching_and_order() -> None:
    from tldw_Server_API.app.core.Persona.dialogue_tree import (
        DialogueTreeBudget,
        DialogueTreeEngine,
        TreeCandidate,
    )

    def generator(node):
        return [
            TreeCandidate(action_type="assistant", text=f"{node.node_id}-b"),
            TreeCandidate(action_type="assistant", text=f"{node.node_id}-a"),
            TreeCandidate(action_type="assistant", text=f"{node.node_id}-c"),
        ]

    engine = DialogueTreeEngine(
        budget=DialogueTreeBudget(max_depth=2, max_branching=2, max_candidates=10),
        generators=[generator],
    )
    result = engine.expand(root_payload={"scenario": "benign"})

    assert result.max_depth_seen == 2
    assert all(
        len(result.children_by_parent[parent_id]) <= 2
        for parent_id in result.children_by_parent
    )
    assert [node.candidate.text for node in result.nodes[1:3]] == ["root-a", "root-b"]


def test_tree_expansion_respects_total_provider_call_budget() -> None:
    from tldw_Server_API.app.core.Persona.dialogue_tree import (
        DialogueTreeBudget,
        DialogueTreeEngine,
        TreeCandidate,
    )

    call_count = 0

    def generator(node):
        nonlocal call_count
        call_count += 1
        return [
            TreeCandidate(action_type="assistant", text=f"{node.node_id}-a"),
            TreeCandidate(action_type="assistant", text=f"{node.node_id}-b"),
        ]

    engine = DialogueTreeEngine(
        budget=DialogueTreeBudget(
            max_depth=3,
            max_branching=2,
            max_candidates=20,
            max_provider_calls=1,
        ),
        generators=[generator],
    )

    result = engine.expand(root_payload={"scenario": "provider-budget"})

    assert call_count == 1
    assert result.max_depth_seen == 1
    assert [node.node_id for node in result.nodes] == ["root", "root.1", "root.2"]


def test_tree_result_exposes_immutable_node_and_child_containers() -> None:
    from tldw_Server_API.app.core.Persona.dialogue_tree import (
        DialogueTreeBudget,
        DialogueTreeEngine,
        TreeCandidate,
    )

    engine = DialogueTreeEngine(
        budget=DialogueTreeBudget(max_depth=1, max_branching=1),
        generators=[lambda _node: [TreeCandidate(action_type="assistant", text="safe")]],
    )

    result = engine.expand()

    assert isinstance(result.nodes, tuple)
    assert isinstance(result.children_by_parent["root"], tuple)
    with pytest.raises(AttributeError):
        result.nodes.append(result.nodes[0])  # type: ignore[attr-defined]
    with pytest.raises(TypeError):
        result.children_by_parent["root"] += ("root.99",)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_depth": -1},
        {"max_branching": -1},
        {"max_candidates": -1},
        {"max_provider_calls": -1},
    ],
)
def test_tree_budget_rejects_negative_values(kwargs) -> None:
    from tldw_Server_API.app.core.Persona.dialogue_tree import DialogueTreeBudget

    with pytest.raises(ValueError, match="must be >= 0"):
        DialogueTreeBudget(**kwargs)
