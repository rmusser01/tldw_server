"""A real TestRunner result must reach the refinement prompt without data loss."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import PromptStudioDatabase
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.optimization_strategies import (
    IterativeRefinementOptimizer,
)
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.test_runner import TestRunner


@pytest.mark.asyncio
async def test_real_runner_expected_and_actual_reach_refinement(tmp_path, monkeypatch):
    db = PromptStudioDatabase(str(tmp_path / "refinement-context.db"), client_id="refinement-context")
    try:
        project = db.create_project(name="Refinement context")
        prompt = db.create_prompt(
            project_id=project["id"],
            name="Director",
            system_prompt="Use the source.",
            user_prompt="Who directs {source}?",
            version_number=1,
        )
        case = db.create_test_case(
            project_id=project["id"],
            name="Rowan",
            inputs={"source": "Rowan Observatory"},
            expected_outputs={"response": "Dr. Mira Vale"},
        )
        runner = TestRunner(db)
        monkeypatch.setattr(
            runner, "_call_adapter", lambda **_: {"choices": [{"message": {"content": "Dr. Tomas Reed"}}]}
        )
        config = {
            "provider": "llama.cpp",
            "model": "configured-model",
            "parameters": {"temperature": 0, "max_tokens": 128},
        }
        result = await runner.run_single_test(prompt_id=prompt["id"], test_case_id=case["id"], model_config=config)
        optimizer = IterativeRefinementOptimizer(db, runner)
        call = AsyncMock(return_value={"content": "Preserve the source name.", "tokens": 5})
        monkeypatch.setattr(optimizer.executor, "_call_llm", call)
        await optimizer._generate_refinement(prompt["id"], optimizer._analyze_errors([result]), model_config=config)
        sent = call.call_args.kwargs["prompt"]
        assert "Expected: {'response': 'Dr. Mira Vale'}" in sent  # nosec B101 - Verify retained refinement examples.
        assert "Got: {'response': 'Dr. Tomas Reed'}" in sent  # nosec B101 - Verify retained refinement examples.
        assert "Input: {'source': 'Rowan Observatory'}" in sent  # nosec B101 - Verify retained refinement examples.
    finally:
        db.close()


@pytest.mark.parametrize("expected,actual", [({}, {}), ({"response": "old expected"}, {"response": "old actual"})])
def test_legacy_runner_shape_remains_supported(expected, actual):
    optimizer = IterativeRefinementOptimizer(SimpleNamespace(client_id="legacy-context"), None)
    errors = optimizer._analyze_errors([{"success": False, "expected_outputs": expected, "actual_output": actual}])
    assert errors[0]["expected"] == expected  # nosec B101 - Verify retained refinement examples.
    assert errors[0]["actual"] == actual  # nosec B101 - Verify retained refinement examples.
