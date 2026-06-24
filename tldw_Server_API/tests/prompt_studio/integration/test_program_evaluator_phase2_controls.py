import os

import pytest

from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import PromptStudioDatabase
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.test_runner import TestRunner


pytestmark = pytest.mark.integration


@pytest.fixture
def temp_ps_db(tmp_path) -> PromptStudioDatabase:
    os.environ.setdefault("TEST_MODE", "true")
    return PromptStudioDatabase(str(tmp_path / "ps_program_eval_int.db"), client_id="ps-prog-int")


def _seed_python_runner_case(db: PromptStudioDatabase, *, project_metadata=None):
    proj = db.create_project(
        "ProgramEval-Int",
        "desc",
        metadata=project_metadata,
    )
    project_id = int(proj["id"]) if isinstance(proj, dict) else int(proj)
    prompt = db.create_prompt(
        project_id=project_id,
        name="PE-Prompt",
        system_prompt="You are a coder.",
        user_prompt="Return Python only.",
        version_number=1,
    )
    test_case = db.create_test_case(
        project_id=project_id,
        name="PE-Test",
        inputs={"runner": "python"},
        expected_outputs={"runner": "python", "objective": "maximize", "metric_var": "val"},
        is_golden=False,
    )
    return int(prompt["id"]), int(test_case["id"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "global_enabled,project_enabled,expected_mode",
    [
        (False, None, "heuristic"),
        (True, None, "sandbox"),
        (False, True, "sandbox"),
        (True, False, "heuristic"),
    ],
)
async def test_program_evaluator_flag_precedence_via_test_runner(
    temp_ps_db,
    monkeypatch,
    global_enabled,
    project_enabled,
    expected_mode,
):
    monkeypatch.setenv("PROMPT_STUDIO_ENABLE_CODE_EVAL", "true" if global_enabled else "false")
    monkeypatch.setenv("PROMPT_STUDIO_ALLOW_UNSAFE_CODE_EVAL", "true")
    if project_enabled is True:
        monkeypatch.setenv("PROMPT_STUDIO_ALLOW_PROJECT_CODE_EVAL", "true")
    else:
        monkeypatch.delenv("PROMPT_STUDIO_ALLOW_PROJECT_CODE_EVAL", raising=False)
    metadata = {} if project_enabled is None else {"enable_code_eval": bool(project_enabled)}
    prompt_id, test_case_id = _seed_python_runner_case(temp_ps_db, project_metadata=metadata)

    async def _fake_run_test_case(
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
            "inputs": {"runner": "python"},
            "expected": {"runner": "python", "objective": "maximize", "metric_var": "val"},
            "actual": {"response": "```python\nimport math\nval = 2.0\nprint('ok')\n```"},
            "model": model or "dummy",
            "execution_time_ms": 1,
            "tokens_used": 1,
        }

    monkeypatch.setattr(TestRunner, "run_test_case", _fake_run_test_case, raising=True)

    runner = TestRunner(temp_ps_db)
    result = await runner.run_single_test(
        prompt_id=prompt_id,
        test_case_id=test_case_id,
        model_config={"model": "dummy"},
    )

    program_eval = result.get("program_eval") or {}
    metrics = program_eval.get("metrics") or {}
    reward = float(result["scores"]["reward"])
    score = float(result["scores"]["aggregate_score"])

    assert metrics.get("mode") == expected_mode
    assert score == pytest.approx(max(0.0, min(1.0, reward / 10.0)))
    assert 0.0 <= score <= 1.0


@pytest.mark.asyncio
async def test_program_evaluator_timeout_maps_to_zero_score(temp_ps_db, monkeypatch):
    monkeypatch.setenv("PROMPT_STUDIO_ENABLE_CODE_EVAL", "true")
    monkeypatch.setenv("PROMPT_STUDIO_ALLOW_UNSAFE_CODE_EVAL", "true")
    monkeypatch.setenv("PROMPT_STUDIO_CODE_EVAL_TIMEOUT_MS", "100")
    prompt_id, test_case_id = _seed_python_runner_case(
        temp_ps_db,
        project_metadata={"enable_code_eval": True},
    )

    async def _fake_run_test_case(
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
            "inputs": {"runner": "python"},
            "expected": {"runner": "python", "objective": "maximize", "metric_var": "val"},
            "actual": {"response": "```python\nwhile True:\n    pass\n```"},
            "model": model or "dummy",
            "execution_time_ms": 1,
            "tokens_used": 1,
        }

    monkeypatch.setattr(TestRunner, "run_test_case", _fake_run_test_case, raising=True)

    runner = TestRunner(temp_ps_db)
    result = await runner.run_single_test(
        prompt_id=prompt_id,
        test_case_id=test_case_id,
        model_config={"model": "dummy"},
    )

    assert result["success"] is False
    assert float(result["scores"]["reward"]) == -1.0
    assert float(result["scores"]["aggregate_score"]) == 0.0
    assert result["program_eval"]["return_code"] == 124
    assert (result["program_eval"]["metrics"] or {}).get("failure_kind") == "timeout"
