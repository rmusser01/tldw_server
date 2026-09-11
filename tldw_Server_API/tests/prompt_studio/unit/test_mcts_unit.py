import os
import pytest

import tldw_Server_API.app.core.Prompt_Management.prompt_studio.mcts_optimizer as mcts_module
from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import PromptStudioDatabase
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.mcts_optimizer import MCTSOptimizer
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.prompt_quality import PromptQualityScorer
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.optimization_engine import MetricType
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.test_runner import TestRunner


@pytest.fixture
def temp_ps_db(tmp_path) -> PromptStudioDatabase:
    os.environ.setdefault("TEST_MODE", "true")
    return PromptStudioDatabase(str(tmp_path / "ps_mcts_unit.db"), client_id="mcts-unit")


def _seed_prompt_and_tests(db: PromptStudioDatabase, n_tests: int = 2):
    proj = db.create_project("MCTS-Unit", "desc")
    pid = int(proj["id"]) if isinstance(proj, dict) else int(proj)
    p = db.create_prompt(
        project_id=pid,
        name="Base-MCTS",
        system_prompt="Base system",
        user_prompt="Answer {q}",
        version_number=1,
    )
    t_ids = []
    for i in range(n_tests):
        tc = db.create_test_case(
            project_id=pid,
            name=f"TC-{i}",
            inputs={"q": f"q-{i}"},
            expected_outputs={"response": f"a-{i}"},
            is_golden=False,
        )
        t_ids.append(int(tc["id"]))
    return int(p["id"]), t_ids


def test_uct_selection_behavior():


    parent = MCTSOptimizer._Node.__new__(MCTSOptimizer._Node)  # bypass __init__ signature
    parent.parent = None
    parent.segment_index = 0
    parent.system_text = ""
    parent.q_sum = 0.0
    parent.n_visits = 10
    parent.children = []
    parent.children_by_bin = {}
    parent.score_bin = -1

    # Two children with different q/n - ensure UCT favors higher exploitation when visits equal
    def _mk(q_sum, n_vis):
        ch = MCTSOptimizer._Node.__new__(MCTSOptimizer._Node)
        ch.parent = parent
        ch.segment_index = 1
        ch.system_text = "x"
        ch.q_sum = q_sum
        ch.n_visits = n_vis
        ch.children = []
        ch.children_by_bin = {}
        ch.score_bin = 0
        return ch

    a = _mk(9.0, 3)   # avg = 3.0
    b = _mk(4.0, 2)   # avg = 2.0
    parent.children = [a, b]

    # With same exploration_c, child 'a' should have higher UCT
    ua = a.uct(exploration_c=1.4)
    ub = b.uct(exploration_c=1.4)
    assert ua > ub


@pytest.mark.asyncio
async def test_node_dedup_by_score_bins(monkeypatch, temp_ps_db):
    prompt_id, test_ids = _seed_prompt_and_tests(temp_ps_db)
    mcts = MCTSOptimizer(temp_ps_db, TestRunner(temp_ps_db))
    mcts._counters = {}

    # Force candidate proposals to two items mapped into same score bin
    async def fake_props(system_so_far, segment_text, k):
        return [system_so_far + " A", system_so_far + " B"]

    monkeypatch.setattr(mcts, "_propose_candidates", fake_props)

    # Map scores so both fall in same bin for bin_size=0.5
    async def fake_score_prompt_async(*, system_text: str, user_text: str) -> float:
        return 0.74 if system_text.endswith("A") else 0.76

    monkeypatch.setattr(PromptQualityScorer, "score_prompt_async", fake_score_prompt_async, raising=False)

    node = MCTSOptimizer._Node(parent=None, segment_index=0, system_text="S0")
    child = await mcts._expand_node(
        node,
        segment="seg",
        base_user="user",
        k_candidates=2,
        score_bin_size=0.5,
        min_quality=0.0,
    )

    # Only one child created; dedup counter incremented
    assert isinstance(child, MCTSOptimizer._Node)
    assert len(node.children) == 1
    assert mcts._counters.get("prune_dedup", 0) >= 1


@pytest.mark.asyncio
async def test_token_budget_cutoff(monkeypatch, temp_ps_db):
    prompt_id, test_ids = _seed_prompt_and_tests(temp_ps_db)
    mcts_low = MCTSOptimizer(temp_ps_db, TestRunner(temp_ps_db))
    mcts_high = MCTSOptimizer(temp_ps_db, TestRunner(temp_ps_db))

    # Enable token callback in scorer by configuring a scorer_model and monkeypatching executor
    class _Exec:
        async def _call_llm(self, **kwargs):
            return {"content": "5", "tokens": 50}

    mcts_low.scorer.set_model("gpt-3.5-turbo")
    mcts_low.scorer.set_executor(_Exec())
    mcts_high.scorer.set_model("gpt-3.5-turbo")
    mcts_high.scorer.set_executor(_Exec())

    # Decomposer: single segment to allow one expansion
    monkeypatch.setattr(mcts_low.decomposer, "decompose_text", lambda _: ["seg"])  # type: ignore[attr-defined]
    monkeypatch.setattr(mcts_high.decomposer, "decompose_text", lambda _: ["seg"])  # type: ignore[attr-defined]
    # _propose_candidates: one child
    monkeypatch.setattr(mcts_low, "_propose_candidates", lambda sys, seg, k: [sys + " X"])  # type: ignore[misc]
    monkeypatch.setattr(mcts_high, "_propose_candidates", lambda sys, seg, k: [sys + " X"])  # type: ignore[misc]
    # Evaluator: constant score
    async def fake_eval(*args, **kwargs):
        return {"success": True, "scores": {"aggregate_score": 0.1}}
    monkeypatch.setattr(TestRunner, "run_single_test", fake_eval)

    low_budget = 40
    high_budget = 1_000

    res_low = await mcts_low.optimize(
        initial_prompt_id=prompt_id,
        test_case_ids=test_ids,
        model_config={"model": "dummy"},
        max_iterations=50,
        target_metric=MetricType.ACCURACY,
        strategy_params={
            "mcts_simulations": 50,
            "token_budget": low_budget,  # smaller than fake tokens per scorer call
            "mcts_max_depth": 2,
        },
    )
    res_high = await mcts_high.optimize(
        initial_prompt_id=prompt_id,
        test_case_ids=test_ids,
        model_config={"model": "dummy"},
        max_iterations=50,
        target_metric=MetricType.ACCURACY,
        strategy_params={
            "mcts_simulations": 50,
            "token_budget": high_budget,
            "mcts_max_depth": 2,
        },
    )
    # Budget materially affects loop length.
    assert res_low["iterations"] < res_high["iterations"]
    assert res_low["iterations"] < 50
    assert res_low["final_metrics"]["applied_params"]["token_budget"] == low_budget
    assert res_high["final_metrics"]["applied_params"]["token_budget"] == high_budget


@pytest.mark.asyncio
async def test_early_stop_no_improve(monkeypatch, temp_ps_db):
    prompt_id, test_ids = _seed_prompt_and_tests(temp_ps_db)
    mcts = MCTSOptimizer(temp_ps_db, TestRunner(temp_ps_db))

    # Single segment, constant candidates and constant evaluation equals baseline
    monkeypatch.setattr(mcts.decomposer, "decompose_text", lambda _: ["seg1"])  # type: ignore[attr-defined]
    monkeypatch.setattr(mcts, "_propose_candidates", lambda sys, seg, k: [sys + " X"])  # type: ignore[misc]

    async def fake_eval_prompt(self, prompt_id: int, test_case_ids, model_config, target_metric):
        return 0.5  # constant
    monkeypatch.setattr(MCTSOptimizer, "_evaluate_prompt", fake_eval_prompt)

    res = await mcts.optimize(
        initial_prompt_id=prompt_id,
        test_case_ids=test_ids,
        model_config={"model": "dummy"},
        max_iterations=100,
        target_metric=MetricType.ACCURACY,
        strategy_params={"mcts_simulations": 100, "early_stop_no_improve": 2, "mcts_max_depth": 2},
    )
    # Expect iterations equal to early_stop_no_improve before break
    assert res["iterations"] == 2


@pytest.mark.asyncio
async def test_cancellation_state_read_failure_stops_before_next_provider_dispatch(
    monkeypatch,
    temp_ps_db,
):
    prompt_id, test_ids = _seed_prompt_and_tests(temp_ps_db, n_tests=1)
    mcts = MCTSOptimizer(temp_ps_db, TestRunner(temp_ps_db))
    provider_dispatches = 0

    async def fake_evaluate_prompt(*_args, **_kwargs):
        nonlocal provider_dispatches
        provider_dispatches += 1
        return 0.5

    def fail_state_read(_optimization_id: int):
        raise RuntimeError("optimization state unavailable")

    monkeypatch.setattr(mcts, "_evaluate_prompt", fake_evaluate_prompt, raising=True)
    monkeypatch.setattr(mcts.decomposer, "decompose_text", lambda _text: [], raising=True)
    monkeypatch.setattr(temp_ps_db, "get_optimization", fail_state_read, raising=True)

    with pytest.raises(RuntimeError, match="optimization state unavailable"):
        await mcts.optimize(
            initial_prompt_id=prompt_id,
            optimization_id=999,
            test_case_ids=test_ids,
            model_config={"model": "dummy"},
            max_iterations=3,
            target_metric=MetricType.ACCURACY,
            strategy_params={
                "mcts_simulations": 3,
                "feedback_enabled": False,
            },
        )

    assert provider_dispatches == 1


@pytest.mark.asyncio
async def test_scorer_model_parameter_applies_to_runtime(monkeypatch, temp_ps_db):
    prompt_id, test_ids = _seed_prompt_and_tests(temp_ps_db)
    mcts = MCTSOptimizer(temp_ps_db, TestRunner(temp_ps_db))
    seen: dict[str, str] = {}

    original_set_model = mcts.scorer.set_model

    def capture_set_model(model_name: str):
        seen["model"] = str(model_name)
        return original_set_model(model_name)

    monkeypatch.setattr(mcts.scorer, "set_model", capture_set_model, raising=True)
    monkeypatch.setattr(mcts.decomposer, "decompose_text", lambda _: ["seg"])  # type: ignore[attr-defined]
    monkeypatch.setattr(mcts, "_propose_candidates", lambda sys, seg, k: [sys + " X"])  # type: ignore[misc]

    async def fake_eval_prompt(self, prompt_id: int, test_case_ids, model_config, target_metric):
        return 0.4

    monkeypatch.setattr(MCTSOptimizer, "_evaluate_prompt", fake_eval_prompt, raising=True)

    res = await mcts.optimize(
        initial_prompt_id=prompt_id,
        test_case_ids=test_ids,
        model_config={"model": "dummy"},
        max_iterations=2,
        target_metric=MetricType.ACCURACY,
        strategy_params={
            "mcts_simulations": 2,
            "mcts_max_depth": 2,
            "scorer_model": "gpt-4o-mini",
            "feedback_enabled": False,
        },
    )

    assert seen.get("model") == "gpt-4o-mini"
    assert res["final_metrics"]["applied_params"]["scorer_model"] == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_feedback_knobs_change_refinement_behavior(monkeypatch, temp_ps_db):
    prompt_id, test_ids = _seed_prompt_and_tests(temp_ps_db)

    class _FakeRefiner:
        calls = 0

        def __init__(self, db, test_runner):
            self.db = db
            self.test_runner = test_runner

        async def optimize(self, *, prompt_id, test_case_ids, model_config, max_iterations=1, optimization_id=None):
            _FakeRefiner.calls += 1
            return {"optimized_prompt_id": int(prompt_id) + 100_000}

    async def fake_eval_prompt(self, prompt_id: int, test_case_ids, model_config, target_metric):
        return 0.95 if int(prompt_id) >= 100_000 else 0.2

    monkeypatch.setattr(MCTSOptimizer, "_evaluate_prompt", fake_eval_prompt, raising=True)

    async def _run_once(*, feedback_enabled: bool, feedback_max_retries: int):
        mcts = MCTSOptimizer(temp_ps_db, TestRunner(temp_ps_db))
        mcts._refiner_cls = _FakeRefiner
        monkeypatch.setattr(mcts.decomposer, "decompose_text", lambda _: ["seg"])  # type: ignore[attr-defined]
        monkeypatch.setattr(mcts, "_propose_candidates", lambda sys, seg, k: [sys + " X"])  # type: ignore[misc]
        _FakeRefiner.calls = 0
        result = await mcts.optimize(
            initial_prompt_id=prompt_id,
            test_case_ids=test_ids,
            model_config={"model": "dummy"},
            max_iterations=1,
            target_metric=MetricType.ACCURACY,
            strategy_params={
                "mcts_simulations": 1,
                "mcts_max_depth": 1,
                "feedback_enabled": feedback_enabled,
                "feedback_threshold": 9.0,
                "feedback_max_retries": feedback_max_retries,
            },
        )
        return result, _FakeRefiner.calls

    disabled_result, disabled_calls = await _run_once(feedback_enabled=False, feedback_max_retries=2)
    zero_retry_result, zero_retry_calls = await _run_once(feedback_enabled=True, feedback_max_retries=0)
    enabled_result, enabled_calls = await _run_once(feedback_enabled=True, feedback_max_retries=2)

    assert disabled_calls == 0
    assert zero_retry_calls == 0
    assert enabled_calls >= 1
    assert enabled_result["final_score"] > disabled_result["final_score"]
    assert enabled_result["final_score"] > zero_retry_result["final_score"]
    assert enabled_result["final_metrics"]["applied_params"]["feedback_enabled"] is True
    assert enabled_result["final_metrics"]["applied_params"]["feedback_max_retries"] == 2


@pytest.mark.asyncio
async def test_mcts_error_metrics_emit_all_required_labels(monkeypatch, temp_ps_db):
    prompt_id, test_ids = _seed_prompt_and_tests(temp_ps_db, n_tests=1)
    prompt = temp_ps_db.get_prompt(prompt_id) or {}
    optimization = temp_ps_db.create_optimization(
        project_id=int(prompt["project_id"]),
        name="mcts-errors",
        initial_prompt_id=prompt_id,
        optimizer_type="mcts",
        optimization_config={
            "optimizer_type": "mcts",
            "target_metric": "accuracy",
            "strategy_params": {
                "mcts_simulations": 1,
                "mcts_max_depth": 1,
            },
        },
        max_iterations=1,
        status="running",
    )

    mcts = MCTSOptimizer(temp_ps_db, TestRunner(temp_ps_db))
    monkeypatch.setattr(mcts.decomposer, "decompose_text", lambda _: ["seg"])  # type: ignore[attr-defined]

    async def _fake_props(system_so_far, segment_text, k):
        return [
            system_so_far + " FAILSCORE",
            system_so_far + " LOWQ",
            system_so_far + " GOOD_A",
            system_so_far + " GOOD_B",
        ]

    async def _fake_score_prompt_async(*, system_text: str, user_text: str) -> float:
        if system_text.endswith("FAILSCORE"):
            raise RuntimeError("scorer failure")
        if system_text.endswith("LOWQ"):
            return 0.05
        if system_text.endswith("GOOD_A"):
            return 0.82
        return 0.84

    async def _timeout_run_single_test(self, *, prompt_id: int, test_case_id: int, model_config, metrics=None):
        raise TimeoutError("evaluation timed out")

    emitted: list[tuple[str, int]] = []

    def _capture_mcts_error(*, error: str, count: int = 1, strategy: str = "mcts") -> None:
        emitted.append((str(error), int(count)))

    monkeypatch.setattr(mcts, "_propose_candidates", _fake_props, raising=True)
    monkeypatch.setattr(mcts.scorer, "score_prompt_async", _fake_score_prompt_async, raising=True)
    monkeypatch.setattr(TestRunner, "run_single_test", _timeout_run_single_test, raising=True)
    monkeypatch.setattr(mcts_module.prompt_studio_metrics, "record_mcts_error", _capture_mcts_error, raising=True)

    result = await mcts.optimize(
        initial_prompt_id=prompt_id,
        optimization_id=int(optimization["id"]),
        test_case_ids=test_ids,
        model_config={"model": "dummy"},
        max_iterations=1,
        target_metric=MetricType.ACCURACY,
        strategy_params={
            "mcts_simulations": 1,
            "mcts_max_depth": 1,
            "prompt_candidates_per_node": 4,
            "score_dedup_bin": 0.1,
            "min_quality": 0.2,
            "feedback_enabled": False,
        },
    )

    labels = {label for label, _count in emitted}
    assert {
        "prune_low_quality",
        "prune_dedup",
        "scorer_failure",
        "evaluator_timeout",
    }.issubset(labels)

    errors = result["final_metrics"]["errors"]
    assert int(errors["prune_low_quality"]) >= 1
    assert int(errors["prune_dedup"]) >= 1
    assert int(errors["scorer_failures"]) >= 1
    assert int(errors["evaluator_timeouts"]) >= 1
