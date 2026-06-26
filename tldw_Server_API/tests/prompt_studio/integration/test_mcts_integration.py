import os
import pytest

from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import PromptStudioDatabase
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.mcts_optimizer import MCTSOptimizer
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.optimization_engine import MetricType
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.test_runner import TestRunner
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.event_broadcaster import EventBroadcaster, EventType


@pytest.fixture
def temp_ps_db(tmp_path) -> PromptStudioDatabase:
    os.environ.setdefault("TEST_MODE", "true")
    return PromptStudioDatabase(str(tmp_path / "ps_mcts_int.db"), client_id="mcts-int")


def _seed_prompt_and_tests(db: PromptStudioDatabase, n_tests: int = 2):
    proj = db.create_project("MCTS-Int", "desc")
    pid = int(proj["id"]) if isinstance(proj, dict) else int(proj)
    p = db.create_prompt(
        project_id=pid,
        name="Base-MCTS",
        system_prompt="Base system",
        user_prompt="Respond {q}",
        version_number=1,
    )
    t_ids = []
    for i in range(n_tests):
        tc = db.create_test_case(
            project_id=pid,
            name=f"TC-{i}",
            inputs={"q": f"q-{i}"},
            expected_outputs={"response": f"ok {i}"},
            is_golden=False,
        )
        t_ids.append(int(tc["id"]))
    return int(p["id"]), t_ids


def _create_mcts_optimization(db: PromptStudioDatabase, *, prompt_id: int, test_case_ids: list[int], n_sims: int) -> int:
    prompt = db.get_prompt(prompt_id) or {}
    project_id = int(prompt["project_id"])
    optimization = db.create_optimization(
        project_id=project_id,
        name="mcts-int-opt",
        initial_prompt_id=prompt_id,
        optimizer_type="mcts",
        optimization_config={
            "optimizer_type": "mcts",
            "target_metric": "accuracy",
            "strategy_params": {
                "mcts_simulations": n_sims,
                "mcts_max_depth": 2,
                "prompt_candidates_per_node": 1,
                "feedback_enabled": False,
            },
        },
        max_iterations=n_sims,
        status="running",
    )
    db.update_optimization(int(optimization["id"]), {"test_case_ids": list(test_case_ids)})
    return int(optimization["id"])


@pytest.mark.asyncio
async def test_end_to_end_improves_and_ws_events(monkeypatch, temp_ps_db):
    prompt_id, test_ids = _seed_prompt_and_tests(temp_ps_db)
    mcts = MCTSOptimizer(temp_ps_db, TestRunner(temp_ps_db))

    # Deterministic decompose and proposals: include typical suffix that should improve
    monkeypatch.setattr(mcts.decomposer, "decompose_text", lambda _: ["segA", "segB"])  # type: ignore[attr-defined]
    async def fake_props(sys, seg, k):
        # Include the typical "Ensure outputs strictly follow ..." suffix used by engine
        return [sys + "\n\nEnsure outputs strictly follow the required format and constraints."]
    monkeypatch.setattr(mcts, "_propose_candidates", fake_props)

    # Score by content: if system contains "Ensure outputs" give higher score
    async def fake_run_single_test(self, *, prompt_id: int, test_case_id: int, model_config, metrics=None):
        p = self.db.get_prompt(prompt_id)
        sys_text = (p or {}).get("system_prompt", "")
        base = 0.2
        if "Ensure outputs" in sys_text:
            base = 0.9
        return {"success": True, "scores": {"aggregate_score": base}}
    monkeypatch.setattr(TestRunner, "run_single_test", fake_run_single_test)

    # Capture WS events via EventBroadcaster monkeypatch
    events = []
    async def fake_broadcast_event(self, event_type, data, client_ids=None, project_id=None):
        events.append((event_type, data))
    async def fake_broadcast_iter(self, *args, **kwargs):
        events.append((EventType.OPTIMIZATION_ITERATION, kwargs))
    monkeypatch.setattr(EventBroadcaster, "broadcast_event", fake_broadcast_event)
    monkeypatch.setattr(EventBroadcaster, "broadcast_optimization_iteration", fake_broadcast_iter)

    optimization_id = _create_mcts_optimization(
        temp_ps_db,
        prompt_id=prompt_id,
        test_case_ids=test_ids,
        n_sims=5,
    )

    res = await mcts.optimize(
        initial_prompt_id=prompt_id,
        optimization_id=optimization_id,
        test_case_ids=test_ids,
        model_config={"model": "dummy"},
        max_iterations=10,
        target_metric=MetricType.ACCURACY,
        strategy_params={"mcts_simulations": 5, "mcts_max_depth": 2},
    )

    assert res["final_score"] >= res["initial_score"]
    assert res["optimized_prompt_id"] != res["initial_prompt_id"]
    # WS events include start and completion and at least one iteration
    types = [t[0] for t in events]
    assert EventType.OPTIMIZATION_STARTED in types
    assert EventType.OPTIMIZATION_COMPLETED in types
    assert any(t == EventType.OPTIMIZATION_ITERATION for t in types)


@pytest.mark.asyncio
@pytest.mark.property
async def test_simulations_monotonic_vs_baseline(monkeypatch, temp_ps_db):
    prompt_id, test_ids = _seed_prompt_and_tests(temp_ps_db)
    mcts = MCTSOptimizer(temp_ps_db, TestRunner(temp_ps_db))
    # Deterministic scoring: improvement only when suffix present
    monkeypatch.setattr(mcts.decomposer, "decompose_text", lambda _: ["seg"])  # type: ignore[attr-defined]
    async def fake_props(sys, seg, k):
        return [sys, sys + "\n\nEnsure outputs strictly follow the required format and constraints."]
    monkeypatch.setattr(mcts, "_propose_candidates", fake_props)
    async def fake_run_single_test(self, *, prompt_id: int, test_case_id: int, model_config, metrics=None):
        p = self.db.get_prompt(prompt_id)
        sys_text = (p or {}).get("system_prompt", "")
        base = 0.2
        if "Ensure outputs" in sys_text:
            base = 0.8
        return {"success": True, "scores": {"aggregate_score": base}}
    monkeypatch.setattr(TestRunner, "run_single_test", fake_run_single_test)

    res1 = await mcts.optimize(
        initial_prompt_id=prompt_id,
        test_case_ids=test_ids,
        model_config={"model": "dummy"},
        max_iterations=2,
        target_metric=MetricType.ACCURACY,
        strategy_params={"mcts_simulations": 2, "mcts_max_depth": 2},
    )
    res2 = await mcts.optimize(
        initial_prompt_id=prompt_id,
        test_case_ids=test_ids,
        model_config={"model": "dummy"},
        max_iterations=6,
        target_metric=MetricType.ACCURACY,
        strategy_params={"mcts_simulations": 6, "mcts_max_depth": 2},
    )
    assert res2["final_score"] >= res1["final_score"]


@pytest.mark.asyncio
async def test_cancellation_responsive(monkeypatch, temp_ps_db):
    prompt_id, test_ids = _seed_prompt_and_tests(temp_ps_db)
    mcts = MCTSOptimizer(temp_ps_db, TestRunner(temp_ps_db))
    # Deterministic proposals and eval
    monkeypatch.setattr(mcts.decomposer, "decompose_text", lambda _: ["seg"])  # type: ignore[attr-defined]
    monkeypatch.setattr(mcts, "_propose_candidates", lambda sys, seg, k: [sys + "X"])  # type: ignore[misc]
    async def fake_run_single_test(self, *, prompt_id: int, test_case_id: int, model_config, metrics=None):
        return {"success": True, "scores": {"aggregate_score": 0.1}}
    monkeypatch.setattr(TestRunner, "run_single_test", fake_run_single_test)
    optimization_id = _create_mcts_optimization(
        temp_ps_db,
        prompt_id=prompt_id,
        test_case_ids=test_ids,
        n_sims=50,
    )

    # Make db.get_optimization flip to cancelled after first check
    calls = {"n": 0}
    def fake_get_opt(oid):
        calls["n"] += 1
        status = "cancelled" if calls["n"] >= 2 else "running"
        return {"id": oid, "status": status}
    monkeypatch.setattr(temp_ps_db, "get_optimization", fake_get_opt)

    res = await mcts.optimize(
        initial_prompt_id=prompt_id,
        optimization_id=optimization_id,
        test_case_ids=test_ids,
        model_config={"model": "dummy"},
        max_iterations=50,
        target_metric=MetricType.ACCURACY,
        strategy_params={"mcts_simulations": 50, "mcts_max_depth": 2},
    )
    assert res["iterations"] < 50


@pytest.mark.asyncio
async def test_runner_python_scores_via_evaluator_toggle(monkeypatch, temp_ps_db):
    # Seed one test case marked to use runner="python"
    proj = temp_ps_db.create_project("PE", "desc")
    pid = int(proj["id"]) if isinstance(proj, dict) else int(proj)
    p = temp_ps_db.create_prompt(project_id=pid, name="P", system_prompt="S", user_prompt="U", version_number=1)
    prompt_id = int(p["id"])
    tc = temp_ps_db.create_test_case(
        project_id=pid,
        name="T",
        inputs={"runner": "python"},
        expected_outputs={"runner": "python", "objective": "maximize", "metric_var": "val"},
        is_golden=False,
    )
    tc_id = int(tc["id"])

    tr = TestRunner(temp_ps_db)

    # Monkeypatch run_test_case to return code output
    async def fake_run_test_case(
        self,
        prompt_id: int,
        test_case_id: int,
        model: str = "",
        temperature: float = 0.0,
        max_tokens: int = 0,
        **kwargs,
    ):
        return {
            "success": True,
            "test_case_id": test_case_id,
            "prompt_id": prompt_id,
            "inputs": {},
            "expected": {"runner": "python"},
            "actual": {"response": "```python\nval = 2.0\nprint('ok')\n```"},
            "model": model,
        }
    monkeypatch.setattr(TestRunner, "run_test_case", fake_run_test_case)

    # Enabled → sandbox path
    monkeypatch.setenv("PROMPT_STUDIO_ENABLE_CODE_EVAL", "true")
    monkeypatch.setenv("PROMPT_STUDIO_ALLOW_UNSAFE_CODE_EVAL", "true")
    res_on = await tr.run_single_test(prompt_id=prompt_id, test_case_id=tc_id, model_config={"model": "dummy"})
    score_on = float(res_on["scores"]["aggregate_score"])
    # Disabled → heuristic fallback
    monkeypatch.setenv("PROMPT_STUDIO_ENABLE_CODE_EVAL", "false")
    res_off = await tr.run_single_test(prompt_id=prompt_id, test_case_id=tc_id, model_config={"model": "dummy"})
    score_off = float(res_off["scores"]["aggregate_score"])
    assert score_on > score_off
