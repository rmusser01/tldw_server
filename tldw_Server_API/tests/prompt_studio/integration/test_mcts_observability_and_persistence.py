import asyncio
import json

import pytest

from tldw_Server_API.app.core.Prompt_Management.prompt_studio.event_broadcaster import EventType
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.mcts_optimizer import MCTSOptimizer
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.optimization_engine import OptimizationEngine
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.test_runner import TestRunner


pytestmark = pytest.mark.integration


def _seed_minimal_prompt_and_case(db):
    project = db.create_project(name="MCTSProj", description="", user_id="tester")
    prompt = db.create_prompt(
        project_id=project["id"],
        name="MCTSPrompt",
        system_prompt="You are precise.",
        user_prompt="Echo: {text}",
    )
    case = db.create_test_case(
        project_id=project["id"],
        name="Case",
        inputs={"text": "hello"},
        expected_outputs={"response": "hello"},
    )
    return project, prompt, case


def _expected_throttled_iterations(n_sims: int, throttle_every: int) -> list[int]:
    expected = {1, n_sims}
    expected.update(range(throttle_every, n_sims + 1, throttle_every))
    return sorted(expected)


def _as_dict(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    raise AssertionError(f"Expected dict-like value, got: {type(value)}")


@pytest.mark.parametrize("n_sims,throttle_every", [(3, 5), (20, 4)])
def test_mcts_ws_schema_and_persistence_match_throttle_cadence(
    prompt_studio_dual_backend_db,
    monkeypatch,
    n_sims,
    throttle_every,
):
    monkeypatch.setenv("PROMPT_STUDIO_ENABLE_MCTS", "true")

    _label, db = prompt_studio_dual_backend_db
    project, prompt, case = _seed_minimal_prompt_and_case(db)

    opt = db.create_optimization(
        project_id=project["id"],
        name=f"WS-Throttle-{n_sims}",
        initial_prompt_id=prompt["id"],
        optimizer_type="mcts",
        optimization_config={
            "optimizer_type": "mcts",
            "target_metric": "accuracy",
            "strategy_params": {
                "mcts_simulations": n_sims,
                "mcts_max_depth": 2,
                "prompt_candidates_per_node": 1,
                "ws_throttle_every": throttle_every,
                "token_budget": 1000,
                "early_stop_no_improve": n_sims + 1,
                "feedback_enabled": False,
            },
        },
        max_iterations=n_sims,
        status="pending",
    )
    db.update_optimization(opt["id"], {"test_case_ids": [case["id"]]})

    async def _fake_props(self, system_so_far, segment_text, k):
        return [f"{system_so_far}\n\nstable-{segment_text}"]

    async def _fake_run_single_test(self, *, prompt_id, test_case_id, model_config, metrics=None):
        return {"success": True, "scores": {"aggregate_score": 0.33}}

    monkeypatch.setattr(MCTSOptimizer, "_propose_candidates", _fake_props, raising=True)
    monkeypatch.setattr(TestRunner, "run_single_test", _fake_run_single_test, raising=True)

    lifecycle_events = []
    iteration_payloads = []

    async def _capture_event(self, event_type, data, client_ids=None, project_id=None):
        et = event_type.value if hasattr(event_type, "value") else str(event_type)
        if et == EventType.OPTIMIZATION_ITERATION.value:
            iteration_payloads.append(dict(data))
        elif et in {
            EventType.OPTIMIZATION_STARTED.value,
            EventType.OPTIMIZATION_COMPLETED.value,
        }:
            lifecycle_events.append(et)
        return None

    from tldw_Server_API.app.core.Prompt_Management.prompt_studio import event_broadcaster as eb_mod

    monkeypatch.setattr(eb_mod.EventBroadcaster, "broadcast_event", _capture_event, raising=True)

    engine = OptimizationEngine(db)
    asyncio.run(engine.optimize(opt["id"]))

    assert lifecycle_events.count(EventType.OPTIMIZATION_STARTED.value) == 1
    assert lifecycle_events.count(EventType.OPTIMIZATION_COMPLETED.value) == 1

    expected_iterations = _expected_throttled_iterations(n_sims, throttle_every)
    seen_iterations = [int(p["iteration"]) for p in iteration_payloads]
    assert seen_iterations == expected_iterations

    required_keys = {
        "optimization_id",
        "iteration",
        "max_iterations",
        "current_metric",
        "best_metric",
        "progress",
        "strategy",
        "sim_index",
        "depth",
        "reward",
        "best_reward",
        "token_spend_so_far",
        "trace_summary",
    }
    for payload in iteration_payloads:
        assert required_keys.issubset(payload.keys())
        assert int(payload["sim_index"]) == int(payload["iteration"])
        expected_progress = (float(payload["iteration"]) / float(payload["max_iterations"])) * 100.0
        assert payload["progress"] == pytest.approx(expected_progress)
        trace_summary = _as_dict(payload["trace_summary"])
        assert {"prompt_id", "system_hash"}.issubset(trace_summary.keys())

    records = db.list_optimization_iterations(opt["id"], page=1, per_page=500)["iterations"]
    persisted_iterations = [int(r["iteration_number"]) for r in records]
    assert persisted_iterations == expected_iterations
    assert len(records) == len(iteration_payloads)

    for record in records:
        prompt_variant = _as_dict(record.get("prompt_variant"))
        metrics = _as_dict(record.get("metrics"))
        assert record.get("note") == "mcts-iteration"
        assert {"prompt_id", "system_hash", "system_preview"}.issubset(prompt_variant.keys())
        assert {"score", "best_metric"}.issubset(metrics.keys())


@pytest.mark.asyncio
async def test_mcts_cancellation_mid_run(prompt_studio_dual_backend_db, monkeypatch):
    monkeypatch.setenv("PROMPT_STUDIO_ENABLE_MCTS", "true")

    _label, db = prompt_studio_dual_backend_db
    project, prompt, case = _seed_minimal_prompt_and_case(db)
    n_sims = 40
    opt = db.create_optimization(
        project_id=project["id"],
        name="Cancel-Run",
        initial_prompt_id=prompt["id"],
        optimizer_type="mcts",
        optimization_config={
            "optimizer_type": "mcts",
            "target_metric": "accuracy",
            "strategy_params": {
                "mcts_simulations": n_sims,
                "mcts_max_depth": 2,
                "prompt_candidates_per_node": 1,
                "ws_throttle_every": 1,
                "token_budget": 2000,
                "feedback_enabled": False,
            },
        },
        max_iterations=n_sims,
        status="pending",
    )
    db.update_optimization(opt["id"], {"test_case_ids": [case["id"]]})

    async def _fake_props(self, system_so_far, segment_text, k):
        return [f"{system_so_far}\n\nstable-{segment_text}"]

    async def _slow_run_single_test(self, *, prompt_id, test_case_id, model_config, metrics=None):
        await asyncio.sleep(0.01)
        return {"success": True, "scores": {"aggregate_score": 0.2}}

    monkeypatch.setattr(MCTSOptimizer, "_propose_candidates", _fake_props, raising=True)
    monkeypatch.setattr(TestRunner, "run_single_test", _slow_run_single_test, raising=True)

    engine = OptimizationEngine(db)

    async def _cancel_soon():
        await asyncio.sleep(0.05)
        db.set_optimization_status(opt["id"], "cancelled", mark_completed=True)

    task_opt = asyncio.create_task(engine.optimize(opt["id"]))
    task_cancel = asyncio.create_task(_cancel_soon())
    results = await task_opt
    await task_cancel

    assert results["iterations"] < n_sims


def test_mcts_final_trace_persisted(prompt_studio_dual_backend_db, monkeypatch):
    monkeypatch.setenv("PROMPT_STUDIO_ENABLE_MCTS", "true")
    monkeypatch.setenv("PROMPT_STUDIO_MCTS_DEBUG_DECISIONS", "true")

    _label, db = prompt_studio_dual_backend_db
    project, prompt, case = _seed_minimal_prompt_and_case(db)
    n_sims = 3
    opt = db.create_optimization(
        project_id=project["id"],
        name="Trace-Persist",
        initial_prompt_id=prompt["id"],
        optimizer_type="mcts",
        optimization_config={
            "optimizer_type": "mcts",
            "target_metric": "accuracy",
            "strategy_params": {
                "mcts_simulations": n_sims,
                "mcts_max_depth": 2,
                "prompt_candidates_per_node": 1,
                "trace_top_k": 2,
                "token_budget": 1000,
                "feedback_enabled": False,
            },
        },
        max_iterations=n_sims,
        status="pending",
    )
    db.update_optimization(opt["id"], {"test_case_ids": [case["id"]]})

    async def _fake_props(self, system_so_far, segment_text, k):
        return [f"{system_so_far}\n\nstable-{segment_text}"]

    async def _fake_run_single_test(self, *, prompt_id, test_case_id, model_config, metrics=None):
        return {"success": True, "scores": {"aggregate_score": 0.4}}

    monkeypatch.setattr(MCTSOptimizer, "_propose_candidates", _fake_props, raising=True)
    monkeypatch.setattr(TestRunner, "run_single_test", _fake_run_single_test, raising=True)

    engine = OptimizationEngine(db)
    asyncio.run(engine.optimize(opt["id"]))

    row = db.get_optimization(opt["id"]) or {}
    fm = row.get("final_metrics") or {}
    assert isinstance(fm, dict)
    assert "trace" in fm and isinstance(fm["trace"], dict)
    trace = fm["trace"]
    assert "best_path" in trace and "top_candidates" in trace
    assert "debug_top_scores_by_depth" in trace
