import pytest


pytestmark = pytest.mark.unit


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_aggregate_scores_orders_and_tracks_skips() -> None:
    from tldw_Server_API.app.core.Persona.dialogue_tree_scorers import (
        ScoreResult,
        ScoreSeverity,
        aggregate_scores,
    )

    results = [
        ScoreResult(scorer="policy", score=1.0, severity=ScoreSeverity.PASS),
        ScoreResult(scorer="usefulness", score=0.3, severity=ScoreSeverity.FAIL),
        ScoreResult.skipped_result(scorer="tool_plan", reason="no_tool_plan"),
    ]

    aggregate = aggregate_scores(results)

    _check(aggregate.contributing_count == 2, "unexpected contributing score count")
    _check(aggregate.overall_score == 0.65, "unexpected aggregate score")
    _check(
        [result.scorer for result in aggregate.ordered_results] == ["policy", "usefulness"],
        "aggregate ordering mismatch",
    )
    _check(
        [result.scorer for result in aggregate.skipped_results] == ["tool_plan"],
        "skipped result ordering mismatch",
    )
    _check(
        aggregate.skipped_results[0].skip_reason == "no_tool_plan",
        "skip reason mismatch",
    )
    _check(
        [result.scorer for result in aggregate.failed_results] == ["usefulness"],
        "failed result list mismatch",
    )


def test_refusal_quality_scores_boundary_and_safe_redirect_highest() -> None:
    from tldw_Server_API.app.core.Persona.dialogue_tree_scorers import (
        ScoreSeverity,
        refusal_quality_score,
    )

    result = refusal_quality_score(
        {
            "text": (
                "I can't help with that because it violates safety policy, "
                "but I can help with a safer alternative instead."
            )
        }
    )

    _check(result.score == 1.0, "refusal quality score mismatch")
    _check(result.severity == ScoreSeverity.PASS, "refusal quality severity mismatch")


def test_policy_and_tool_plan_scores_flag_unauthorized_risky_action() -> None:
    from tldw_Server_API.app.core.Persona.dialogue_tree_scorers import (
        ScoreSeverity,
        policy_score,
        tool_plan_score,
    )

    candidate = {"tool_plan": {"action": "delete", "authorized": False}}
    policy = policy_score(candidate)
    tool_score = tool_plan_score(candidate)

    _check(policy.severity == ScoreSeverity.FAIL, "policy severity mismatch")
    _check(policy.score == 0.0, "policy score mismatch")
    _check(tool_score.severity == ScoreSeverity.FAIL, "tool score severity mismatch")
    _check(tool_score.score == 0.2, "tool score mismatch")


def test_persona_usefulness_and_grounding_scores_cover_skip_warning_and_pass() -> None:
    from tldw_Server_API.app.core.Persona.dialogue_tree_scorers import (
        ScoreSeverity,
        grounding_style_score,
        persona_consistency_score,
        usefulness_score,
    )

    persona_skip = persona_consistency_score({"text": ""})
    usefulness_warn = usefulness_score({"text": "brief"})
    grounding_pass = grounding_style_score({"metadata": {"grounded": True}})

    _check(persona_skip.skipped is True, "empty persona text was not skipped")
    _check(persona_skip.severity == ScoreSeverity.SKIPPED, "persona skip severity mismatch")
    _check(usefulness_warn.severity == ScoreSeverity.WARNING, "usefulness warning mismatch")
    _check(grounding_pass.severity == ScoreSeverity.PASS, "grounding pass mismatch")
