"""Smoke tests for Bootstrap and Genetic optimizers with mocked LLM and runner.

These tests validate that prompt variants are created using the correct schema
fields (system_prompt, user_prompt, version_number) and do not depend on
external providers.
"""

import asyncio
import os
import tempfile
from typing import Dict, Any, List

import pytest

from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import PromptStudioDatabase
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.optimization_engine import (
    BootstrapOptimizer,
    MIPROOptimizer,
    TestRunner,
)
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.optimization_strategies import (
    GeneticOptimizer,
)


@pytest.fixture
def temp_ps_db(tmp_path) -> PromptStudioDatabase:
     # Ensure test mode to relax auth paths inside some helpers
    os.environ.setdefault("TEST_MODE", "true")
    db_path = tmp_path / "ps_smoke.db"
    db = PromptStudioDatabase(str(db_path), client_id="smoke-test")
    return db


def _seed_project_prompt(db: PromptStudioDatabase) -> Dict[str, int]:
    proj = db.create_project("PS-Smoke", "desc")
    pid = int(proj["id"]) if isinstance(proj, dict) else int(proj)
    p = db.create_prompt(
        project_id=pid,
        name="Base",
        system_prompt="Be helpful",
        user_prompt="Answer for {q}",
        version_number=1,
    )
    return {"project_id": pid, "prompt_id": int(p["id"]) }


def _seed_test_cases(db: PromptStudioDatabase, project_id: int, n: int = 5) -> List[int]:
    ids: List[int] = []
    for i in range(n):
        tc = db.create_test_case(
            project_id=project_id,
            name=f"TC-{i}",
            inputs={"q": f"question-{i}"},
            expected_outputs={"response": f"answer-{i}"},
            is_golden=bool(i % 2 == 0),
        )
        ids.append(int(tc["id"]))
    return ids


@pytest.mark.asyncio
async def test_bootstrap_optimizer_smoke(temp_ps_db, monkeypatch):
    ids = _seed_project_prompt(temp_ps_db)
    project_id = ids["project_id"]
    prompt_id = ids["prompt_id"]
    test_case_ids = _seed_test_cases(temp_ps_db, project_id, n=6)

    # Patch TestRunner.run_single_test to return successful runs with scores
    async def _fake_run_single_test(self, prompt_id: int, test_case_id: int, model_config: Dict[str, Any]):
        return {
            "success": True,
            "test_case_id": test_case_id,
            "inputs": {"q": f"question-{test_case_id}"},
            "expected_outputs": {"response": f"answer-{test_case_id}"},
            "actual_output": {"response": f"answer-{test_case_id}"},
            "scores": {"aggregate_score": 0.91},
        }

    monkeypatch.setattr(TestRunner, "run_single_test", _fake_run_single_test)

    # Make asyncio.sleep instant for speed (used in some code paths)
    async def _fast_sleep(_):
        return None

    monkeypatch.setattr(asyncio, "sleep", _fast_sleep)

    runner = TestRunner(temp_ps_db)
    bo = BootstrapOptimizer(temp_ps_db, runner)
    result = await bo.optimize(
        prompt_id=prompt_id,
        test_case_ids=test_case_ids,
        model_config={"model": "dummy"},
        num_examples=2,
        selection_strategy="best",
    )

    assert result["optimized_prompt_id"]
    new_id = int(result["optimized_prompt_id"])
    new_prompt = temp_ps_db.get_prompt(new_id)
    assert new_prompt is not None
    # Ensure examples were prepended to user_prompt
    assert new_prompt.get("user_prompt", "").startswith("Here are some examples:")
    # Schema fields present
    assert new_prompt.get("system_prompt") is not None
    assert isinstance(new_prompt.get("version_number"), int)
    # Improvement metrics present (may be zero with mocked constant scores)
    assert "initial_score" in result and "final_score" in result and "improvement" in result
    assert isinstance(result["improvement"], (int, float))


@pytest.mark.asyncio
async def test_optimizer_variants_allocate_unique_versions(temp_ps_db):
    ids = _seed_project_prompt(temp_ps_db)
    prompt_id = ids["prompt_id"]
    runner = TestRunner(temp_ps_db)

    mipro = MIPROOptimizer(temp_ps_db, runner)
    first_mipro_id = await mipro._create_prompt_variant(prompt_id, "Instruction A")
    second_mipro_id = await mipro._create_prompt_variant(prompt_id, "Instruction B")

    first_mipro = temp_ps_db.get_prompt(first_mipro_id)
    second_mipro = temp_ps_db.get_prompt(second_mipro_id)
    assert first_mipro["name"] == "Base (Optimized)"
    assert second_mipro["name"] == "Base (Optimized)"
    assert {first_mipro["version_number"], second_mipro["version_number"]} == {2, 3}

    bootstrap = BootstrapOptimizer(temp_ps_db, runner)
    examples = [{"inputs": {"q": "x"}, "actual_output": {"response": "y"}}]
    first_bootstrap_id = await bootstrap._create_prompt_with_examples(prompt_id, examples)
    second_bootstrap_id = await bootstrap._create_prompt_with_examples(prompt_id, examples)

    first_bootstrap = temp_ps_db.get_prompt(first_bootstrap_id)
    second_bootstrap = temp_ps_db.get_prompt(second_bootstrap_id)
    assert first_bootstrap["name"] == "Base (Bootstrap)"
    assert second_bootstrap["name"] == "Base (Bootstrap)"
    assert {first_bootstrap["version_number"], second_bootstrap["version_number"]} == {2, 3}


@pytest.mark.asyncio
async def test_genetic_optimizer_smoke(temp_ps_db, monkeypatch):
    ids = _seed_project_prompt(temp_ps_db)
    project_id = ids["project_id"]
    prompt_id = ids["prompt_id"]
    test_case_ids = _seed_test_cases(temp_ps_db, project_id, n=4)

    # Patch TestRunner.run_single_test to provide aggregate scores
    async def _fake_run_single_test(self, prompt_id: int, test_case_id: int, model_config: Dict[str, Any]):
        return {
            "success": True,
            "test_case_id": test_case_id,
            "inputs": {"q": f"question-{test_case_id}"},
            "expected_outputs": {"response": f"answer-{test_case_id}"},
            "actual_output": {"response": f"answer-{test_case_id}"},
            "scores": {"aggregate_score": 0.85},
        }

    monkeypatch.setattr(TestRunner, "run_single_test", _fake_run_single_test)

    # Patch PromptExecutor._call_llm to return a quick content string
    from tldw_Server_API.app.core.Prompt_Management.prompt_studio import prompt_executor as _pe

    async def _fake_call_llm(*args, **kwargs) -> Dict[str, Any]:
        return {"content": "mutated user prompt"}

    monkeypatch.setattr(_pe.PromptExecutor, "_call_llm", staticmethod(_fake_call_llm))

    # Speed up any sleeps
    async def _fast_sleep(_):
        return None

    monkeypatch.setattr(asyncio, "sleep", _fast_sleep)

    runner = TestRunner(temp_ps_db)
    go = GeneticOptimizer(temp_ps_db, runner)
    result = await go.optimize(
        prompt_id=prompt_id,
        test_case_ids=test_case_ids,
        model_config={"model": "dummy"},
        population_size=4,
        generations=2,
        mutation_rate=0.5,
    )

    assert result["optimized_prompt_id"]
    new_id = int(result["optimized_prompt_id"])
    new_prompt = temp_ps_db.get_prompt(new_id)
    assert new_prompt is not None
    # user_prompt should be present and non-empty
    assert isinstance(new_prompt.get("user_prompt"), str) and len(new_prompt["user_prompt"]) > 0
    # The schema fields exist
    assert new_prompt.get("system_prompt") is not None
    # Generation history present and non-empty
    assert "generation_history" in result and isinstance(result["generation_history"], list)
    assert len(result["generation_history"]) >= 1
    # Best score numeric and generations count > 0
    assert isinstance(result.get("best_score", 0.0), (int, float))
    assert int(result.get("generations", 0)) >= 1
