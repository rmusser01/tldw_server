import pytest

from tldw_Server_API.app.core.Persona.robustness_eval import (
    PersonaRobustnessCase,
    PersonaRobustnessEval,
    build_default_smoke_suite,
)


pytestmark = pytest.mark.unit


def test_smoke_suite_has_required_case_ids() -> None:
    suite = build_default_smoke_suite()
    case_ids = [case.case_id for case in suite]
    assert "benign_basic" in case_ids, "missing benign_basic in smoke suite"
    assert "persona_drift_boundary" in case_ids, "missing persona_drift_boundary in smoke suite"
    assert "prompt_injection_policy_override" in case_ids, "missing prompt_injection_policy_override in smoke suite"
    assert "unsafe_tool_plan" in case_ids, "missing unsafe_tool_plan in smoke suite"
    assert PersonaRobustnessEval is not None, "expected evaluator class import"


def test_run_suite_returns_report_with_summary_and_trace_artifacts() -> None:
    suite = build_default_smoke_suite()
    evaluator = PersonaRobustnessEval()

    report = evaluator.run_suite(
        persona={"id": "persona-1", "policy_snapshot": {"mode": "strict"}},
        character={"id": "char-1"},
        suite=suite,
    )
    payload = report.model_dump(mode="json")
    case_ids = [case["case_id"] for case in payload["cases"]]

    assert report.summary["total_cases"] == len(suite), "summary did not support dict-style access"
    assert payload["summary"]["total_cases"] == len(suite), "summary total cases mismatch"
    assert payload["summary"]["trace_artifact_count"] == len(suite), "trace artifact count mismatch"
    assert payload["summary"]["hard_prune_count"] > 0, "expected hard prune count > 0"
    assert payload["summary"]["skipped_scorer_count"] > 0, "expected skipped scorer count > 0"
    assert payload["summary"]["selected_trajectory_count"] == len(suite), (
        "expected selected trajectory count to match cases"
    )
    assert len(payload["trace_artifacts"]) == len(suite), "trace artifacts length mismatch"
    assert "benign_basic" in case_ids, "missing benign_basic report"
    assert "persona_drift_boundary" in case_ids, "missing persona_drift_boundary report"
    assert "prompt_injection_policy_override" in case_ids, "missing prompt_injection_policy_override report"
    assert "unsafe_tool_plan" in case_ids, "missing unsafe_tool_plan report"


def test_run_suite_does_not_call_persistence_callbacks_or_mutate_input_lists() -> None:
    calls = {"memory": 0, "exemplar": 0, "state": 0, "history": 0}

    def _count_memory_write(*_args, **_kwargs):
        calls["memory"] += 1

    def _count_exemplar_write(*_args, **_kwargs):
        calls["exemplar"] += 1

    def _count_state_write(*_args, **_kwargs):
        calls["state"] += 1

    def _count_history_write(*_args, **_kwargs):
        calls["history"] += 1

    memory_entries = [{"id": "m1", "content": "keep"}]
    exemplar_entries = [("style", "steady", 1.0)]
    state_docs = [{"id": "s1", "content": "state"}]
    chat_history = [{"role": "user", "content": "hello"}]

    persona = {
        "id": "persona-no-persist",
        "memory_write_callback": _count_memory_write,
        "exemplar_write_callback": _count_exemplar_write,
        "state_write_callback": _count_state_write,
        "chat_history_write_callback": _count_history_write,
        "memory_entries": memory_entries,
        "state_docs": state_docs,
        "chat_history": chat_history,
    }
    character = {
        "id": "character-no-persist",
        "exemplar_sections": exemplar_entries,
        "persist_memory": _count_memory_write,
        "persist_exemplar": _count_exemplar_write,
        "persist_state": _count_state_write,
        "persist_chat_history": _count_history_write,
    }
    suite = [
        PersonaRobustnessCase(
            case_id="no_persist_probe",
            prompt="simple prompt",
            candidates=[{"action_type": "assistant", "text": "safe deterministic response"}],
        )
    ]

    evaluator = PersonaRobustnessEval()
    report = evaluator.run_suite(persona=persona, character=character, suite=suite)

    assert report.summary.total_cases == 1, "expected single-case report"
    assert calls["memory"] == 0, "memory persistence callback was called"
    assert calls["exemplar"] == 0, "exemplar persistence callback was called"
    assert calls["state"] == 0, "state persistence callback was called"
    assert calls["history"] == 0, "chat history persistence callback was called"
    assert memory_entries == [{"id": "m1", "content": "keep"}], "memory entries were mutated"
    assert state_docs == [{"id": "s1", "content": "state"}], "state docs were mutated"
    assert chat_history == [{"role": "user", "content": "hello"}], "chat history was mutated"
    assert exemplar_entries == [("style", "steady", 1.0)], "exemplar entries were mutated"


def test_run_suite_redacts_selected_candidate_report_payload() -> None:
    provider_marker = "sk-" + "report"
    raw_output = "private selected output"
    suite = [
        PersonaRobustnessCase(
            case_id="selected_redaction",
            prompt="choose safe candidate",
            candidates=[
                {
                    "action_type": "assistant",
                    "text": "Based on the provided context, safe useful answer.",
                    "tool_plan": {
                        "action": "search",
                        "api_key": provider_marker,
                        "output": raw_output,
                    },
                    "metadata": {"grounded": True, "response": raw_output},
                }
            ],
        )
    ]

    report = PersonaRobustnessEval().run_suite(
        persona={"id": "persona-redact"},
        character=None,
        suite=suite,
    )
    serialized = repr(report.model_dump(mode="json"))

    assert "sk-report" not in serialized, "selected candidate report leaked provider marker"
    assert raw_output not in serialized, "selected candidate report leaked raw output"


def test_run_suite_does_not_score_soft_pruned_duplicate_candidates() -> None:
    suite = [
        PersonaRobustnessCase(
            case_id="duplicate_soft_prune",
            prompt="choose one candidate",
            candidates=[
                {"action_type": "assistant", "text": "same answer"},
                {"action_type": "assistant", "text": "same answer"},
            ],
        )
    ]

    report = PersonaRobustnessEval().run_suite(
        persona={"id": "persona-duplicate"},
        character=None,
        suite=suite,
    )
    payload = report.model_dump(mode="json")
    trace_nodes = payload["trace_artifacts"][0]["trace"]["nodes"]
    duplicate_node = next(node for node in trace_nodes if node["node_id"] == "root.2")

    assert payload["cases"][0]["soft_prune_count"] == 1
    assert duplicate_node["prune_diagnostics"][0]["reason"] == "none"
    assert any(
        diagnostic["reason"] == "duplicate_low_diversity"
        for diagnostic in duplicate_node["prune_diagnostics"]
    )
    assert duplicate_node["score_diagnostics"] == []
