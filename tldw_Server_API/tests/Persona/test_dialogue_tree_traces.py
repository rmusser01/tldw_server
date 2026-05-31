import pytest

from tldw_Server_API.app.core.Persona.dialogue_tree import (
    DialogueTreeNode,
    DialogueTreeResult,
    TreeCandidate,
)


pytestmark = pytest.mark.unit


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_trace_serialization_uses_stable_node_order_and_includes_diagnostics() -> None:
    from tldw_Server_API.app.core.Persona.dialogue_tree_pruners import (
        PruneDecision,
        PruneReason,
        PruneSeverity,
    )
    from tldw_Server_API.app.core.Persona.dialogue_tree_scorers import (
        ScoreResult,
        ScoreSeverity,
    )
    from tldw_Server_API.app.core.Persona.dialogue_tree_traces import (
        serialize_dialogue_tree_trace,
    )

    tree_result = DialogueTreeResult(
        nodes=[
            DialogueTreeNode(node_id="root.10", parent_node_id="root", depth=1, candidate=None),
            DialogueTreeNode(
                node_id="root.2",
                parent_node_id="root",
                depth=1,
                candidate=TreeCandidate(action_type="assistant", text="c"),
            ),
            DialogueTreeNode(node_id="root", parent_node_id=None, depth=0, candidate=None),
            DialogueTreeNode(
                node_id="root.1",
                parent_node_id="root",
                depth=1,
                candidate=TreeCandidate(action_type="assistant", text="a"),
            ),
        ],
        children_by_parent={"root": ["root.10", "root.2", "root.1"]},
        max_depth_seen=1,
    )
    prune_diagnostics = {
        "root.1": [
            PruneDecision(
                pruned=True,
                severity=PruneSeverity.SOFT,
                reason=PruneReason.DUPLICATE_LOW_DIVERSITY,
                message="duplicate",
            )
        ]
    }
    score_diagnostics = {
        "root.1": [ScoreResult(scorer="policy", score=1.0, severity=ScoreSeverity.PASS)]
    }

    trace = serialize_dialogue_tree_trace(
        tree_result,
        prune_diagnostics=prune_diagnostics,
        score_diagnostics=score_diagnostics,
        selected_node_id="root.1",
        fallback_node_id="root",
        decision_label="selected",
        metadata={"run_id": "run-1"},
    )

    _check(
        [node["node_id"] for node in trace["nodes"]] == ["root", "root.1", "root.2", "root.10"],
        "trace node ordering mismatch",
    )
    _check(
        trace["children_by_parent"]["root"] == ["root.1", "root.2", "root.10"],
        "children ordering mismatch",
    )
    _check(trace["decision"]["selected_node_id"] == "root.1", "selected node missing")
    _check(trace["decision"]["fallback_node_id"] == "root", "fallback node missing")
    _check(
        trace["nodes"][1]["prune_diagnostics"][0]["reason"] == "duplicate_low_diversity",
        "prune diagnostic missing",
    )
    _check(
        trace["nodes"][1]["score_diagnostics"][0]["scorer"] == "policy",
        "score diagnostic missing",
    )


def test_trace_serialization_redacts_secrets_and_raw_like_payloads() -> None:
    from tldw_Server_API.app.core.Persona.dialogue_tree_traces import (
        serialize_dialogue_tree_trace,
    )

    secret_token = "Bearer " + "-".join(("top", "secret", "token"))
    raw_response = "external raw response body"
    tree_result = DialogueTreeResult(
        nodes=[
            DialogueTreeNode(
                node_id="root",
                parent_node_id=None,
                depth=0,
                candidate=None,
                payload={"Authorization": secret_token},
            ),
            DialogueTreeNode(
                node_id="root.1",
                parent_node_id="root",
                depth=1,
                candidate=TreeCandidate(
                    action_type="assistant",
                    text=f"Authorization: {secret_token}",
                    tool_plan={"action": "search", "api_key": "sk-test-value"},
                    metadata={"raw": raw_response},
                ),
                payload={"raw_response": raw_response},
            ),
        ],
        children_by_parent={"root": ["root.1"]},
        max_depth_seen=1,
    )

    trace = serialize_dialogue_tree_trace(
        tree_result,
        metadata={"token": "sk-" + "-".join(("test", "other"))},
    )
    serialized = repr(trace)

    _check("top-secret-token" not in serialized, "trace leaked bearer token")
    _check("sk-test-value" not in serialized, "trace leaked provider key")
    _check("external raw response body" not in serialized, "trace leaked raw response")
    _check(
        trace["nodes"][1]["payload"]["raw_response"] == "[REDACTED]",
        "payload raw response not redacted",
    )
    _check(
        trace["nodes"][1]["candidate"]["metadata"]["raw"] == "[REDACTED]",
        "candidate raw metadata not redacted",
    )


def test_trace_serialization_redacts_common_raw_tool_output_fields() -> None:
    from tldw_Server_API.app.core.Persona.dialogue_tree_traces import (
        serialize_dialogue_tree_trace,
    )

    output_value = "private output bytes"
    result_value = "private result bytes"
    response_value = "private response bytes"
    content_value = "private content bytes"
    tree_result = DialogueTreeResult(
        nodes=[
            DialogueTreeNode(node_id="root", parent_node_id=None, depth=0, candidate=None),
            DialogueTreeNode(
                node_id="root.1",
                parent_node_id="root",
                depth=1,
                candidate=TreeCandidate(
                    action_type="tool",
                    tool_plan={
                        "action": "search",
                        "output": output_value,
                        "result": result_value,
                        "response": response_value,
                    },
                    metadata={"content": content_value},
                ),
                payload={"content": content_value, "response": response_value},
            ),
        ],
        children_by_parent={"root": ["root.1"]},
        max_depth_seen=1,
    )

    trace = serialize_dialogue_tree_trace(tree_result)
    serialized = repr(trace)

    _check(output_value not in serialized, "trace leaked tool output field")
    _check(result_value not in serialized, "trace leaked tool result field")
    _check(response_value not in serialized, "trace leaked tool response field")
    _check(content_value not in serialized, "trace leaked tool content field")
    _check(
        trace["nodes"][1]["candidate"]["tool_plan"]["output"] == "[REDACTED]",
        "candidate tool output not redacted",
    )
    _check(
        trace["nodes"][1]["candidate"]["metadata"]["content"] == "[REDACTED]",
        "candidate metadata content not redacted",
    )


def test_trace_serialization_normalizes_metadata_and_mixed_sort_keys() -> None:
    from tldw_Server_API.app.core.Persona.dialogue_tree_traces import (
        serialize_dialogue_tree_trace,
    )

    tree_result = DialogueTreeResult(
        nodes=[
            DialogueTreeNode(node_id="root.a", parent_node_id="root", depth=1, candidate=None),
            DialogueTreeNode(node_id="root.2", parent_node_id="root", depth=1, candidate=None),
            DialogueTreeNode(node_id="root", parent_node_id=None, depth=0, candidate=None),
        ],
        children_by_parent={"root": ["root.a", "root.2"]},
        max_depth_seen=1,
    )

    trace = serialize_dialogue_tree_trace(
        tree_result,
        metadata={"mixed_set": {2, "10", 1}, "nested": {"values": {"b", "a"}}},
    )

    _check(
        [node["node_id"] for node in trace["nodes"]] == ["root", "root.2", "root.a"],
        "mixed node ids were not sorted stably",
    )
    _check(trace["metadata"]["mixed_set"] == [1, 2, "10"], "mixed metadata set was not portable")
    _check(trace["metadata"]["nested"]["values"] == ["a", "b"], "nested metadata set was not portable")


def test_trace_serialization_includes_required_root_edges_and_trajectory_scores() -> None:
    from tldw_Server_API.app.core.Persona.dialogue_tree_scorers import (
        ScoreResult,
        ScoreSeverity,
    )
    from tldw_Server_API.app.core.Persona.dialogue_tree_traces import (
        serialize_dialogue_tree_trace,
    )

    root_marker = "sk-" + "root"
    tree_result = DialogueTreeResult(
        nodes=[
            DialogueTreeNode(node_id="root", parent_node_id=None, depth=0, candidate=None),
            DialogueTreeNode(
                node_id="root.1",
                parent_node_id="root",
                depth=1,
                candidate=TreeCandidate(
                    action_type="assistant",
                    text="safe answer",
                    tool_plan={"action": "search"},
                ),
            ),
        ],
        children_by_parent={"root": ["root.1"]},
        max_depth_seen=1,
    )

    trace = serialize_dialogue_tree_trace(
        tree_result,
        root={"persona_id": "p1", "api_key": root_marker},
        trajectory_scores=[ScoreResult(scorer="policy", score=1.0, severity=ScoreSeverity.PASS)],
    )
    serialized = repr(trace)

    _check(trace["root"]["persona_id"] == "p1", "trace root persona id missing")
    _check("sk-root" not in serialized, "trace root secret leaked")
    _check(trace["edges"][0]["parent_node_id"] == "root", "edge parent mismatch")
    _check(trace["edges"][0]["node_id"] == "root.1", "edge node mismatch")
    _check(trace["edges"][0]["action_type"] == "assistant", "edge action type mismatch")
    _check(trace["trajectory_scores"][0]["scorer"] == "policy", "trajectory score missing")
