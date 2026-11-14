"""Evaluation labels must survive the synchronous and asynchronous creation paths."""

from types import SimpleNamespace

import pytest
from fastapi import BackgroundTasks
from starlette.requests import Request

from tldw_Server_API.app.api.v1.endpoints.prompt_studio import prompt_studio_evaluations as endpoint
from tldw_Server_API.app.api.v1.schemas.prompt_studio_schemas import EvaluationCreate
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.evaluation_manager import EvaluationManager


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("run_async", [False, True])
@pytest.mark.parametrize("named", [False, True])
async def test_evaluation_metadata_survives_canonical_reload(isolated_db, monkeypatch, run_async, named):
    project = isolated_db.create_project(name="C02 evaluation labels", user_id="1")
    prompt = isolated_db.create_prompt(project_id=project["id"], name="C02 label prompt", user_prompt="Say hello.")
    case = isolated_db.create_test_case(
        project_id=project["id"], name="C02 label case", inputs={}, expected_outputs={"response": "hello"}
    )
    credential = SimpleNamespace(api_key=None, app_config={}, credentials_resolved=True)
    closed = []

    class Runtime:
        async def resolve(self, _provider, *, model=None):
            return credential

        async def mark_used(self, _selected):
            return True

        async def close(self):
            closed.append(True)

    monkeypatch.setattr(endpoint, "ProviderCredentialRuntime", lambda **_kwargs: Runtime())
    monkeypatch.setattr(EvaluationManager, "_call_adapter_text", staticmethod(lambda **_kwargs: "hello"))
    payload = EvaluationCreate(
        project_id=project["id"],
        prompt_id=prompt["id"],
        test_case_ids=[case["id"]],
        config={"provider": "llama.cpp", "model_name": "c02-test-model"},
        run_async=run_async,
        **({"name": "C02 named evaluation", "description": "Keep my evaluation description."} if named else {}),
    )
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/v1/prompt-studio/evaluations",
            "headers": [],
            "query_string": b"",
        }
    )
    user = {"user_id": "1", "is_admin": False, "client_id": "c02-metadata-test"}
    response = await endpoint.create_evaluation(
        evaluation=payload, background_tasks=BackgroundTasks(), request=request, db=isolated_db, user_context=user
    )
    public = await endpoint.get_evaluation(response.id, request=request, db=isolated_db, user_context=user)
    stored = isolated_db.get_evaluation(response.id)
    expected = {"name": payload.name or "Evaluation", "description": payload.description or ""}
    assert {key: stored[key] for key in expected} == expected  # nosec B101 - Pytest persistence regression, not runtime validation.
    assert {key: public[key] for key in expected} == expected  # nosec B101 - Pytest persistence regression, not runtime validation.
    assert public["project_id"] == project["id"]  # nosec B101 - Pytest persistence regression, not runtime validation.
    assert public["prompt_id"] == prompt["id"]  # nosec B101 - Pytest persistence regression, not runtime validation.
    assert len(public["test_run_ids"]) == 1  # nosec B101 - Pytest persistence regression, not runtime validation.
    assert public["status"] == "completed"  # nosec B101 - Pytest persistence regression, not runtime validation.
    assert closed == [True]  # nosec B101 - Pytest persistence regression, not runtime validation.
