"""Required blank inputs stop before model dispatch and keep input diagnostics."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from tldw_Server_API.app.api.v1.endpoints.prompt_studio import prompt_studio_test_cases as endpoint
from tldw_Server_API.app.api.v1.schemas.prompt_studio_test import RunTestCasesSimpleRequest
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.prompt_executor import PromptExecutor
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.test_runner import TestRunner as StudioTestRunner
from tldw_Server_API.app.core.Prompt_Management.structured_prompts import (
    StructuredPromptAssemblyError,
    assemble_prompt_definition,
)

pytestmark = pytest.mark.integration
CONFIG = {"provider": "ollama", "model": "local-test", "parameters": {}}


def definition(*, required=True, default=None):
    return {
        "schema_version": 1,
        "format": "structured",
        "variables": [{"name": "source", "required": required, "default_value": default}],
        "blocks": [
            {
                "id": "source",
                "name": "Source",
                "role": "user",
                "order": 10,
                "content": "Source {{source}}",
                "is_template": True,
            }
        ],
    }


@pytest.fixture
def owned_prompt(isolated_db, request):
    project = isolated_db.create_project(name="Required empty input", user_id="1")
    kwargs = (
        {"prompt_format": "structured", "prompt_schema_version": 1, "prompt_definition": definition()}
        if request.param == "structured"
        else {"user_prompt": "Source {{source}}"}
    )
    return isolated_db.create_prompt(project_id=project["id"], name="Required source", **kwargs)


@pytest.fixture
def provider_call(monkeypatch):
    call = AsyncMock(return_value={"content": "Misleading response"})
    monkeypatch.setattr(PromptExecutor, "_call_llm", call)
    return call


@pytest.mark.asyncio
@pytest.mark.parametrize("owned_prompt", ["legacy", "structured"], indirect=True)
@pytest.mark.parametrize("value", [None, "", " \t\n"])
async def test_empty_required_execution_input_never_dispatches(
    isolated_db,
    owned_prompt,
    provider_call,
    value,
):
    result = await PromptExecutor(isolated_db).execute_prompt(owned_prompt["id"], {"source": value}, CONFIG)
    provider_call.assert_not_awaited()
    assert result["success"] is False  # nosec B101 - pytest required-input behavior assertion
    assert result["error_code"] == "missing_required_variable"  # nosec B101 - pytest required-input behavior assertion


@pytest.mark.asyncio
@pytest.mark.parametrize("owned_prompt", ["legacy", "structured"], indirect=True)
@pytest.mark.parametrize("value, rendered", [(0, "0"), (False, "False")])
async def test_zero_and_false_are_valid_supplied_required_values(
    isolated_db,
    owned_prompt,
    provider_call,
    value,
    rendered,
):
    result = await PromptExecutor(isolated_db).execute_prompt(owned_prompt["id"], {"source": value}, CONFIG)
    provider_call.assert_awaited_once()
    kwargs = provider_call.call_args.kwargs
    content = (
        kwargs["messages"][0]["content"] if owned_prompt.get("prompt_format") == "structured" else kwargs["prompt"]
    )
    assert content == "Source " + rendered  # nosec B101 - pytest literal supplied-value assertion
    assert result["success"] is True  # nosec B101 - pytest supplied-value success assertion


@pytest.mark.parametrize("value", [None, "", " \t\n"])
def test_canonical_structured_required_value_rejects_explicit_blank(value):
    with pytest.raises(StructuredPromptAssemblyError) as failure:
        assemble_prompt_definition(definition(), {"source": value})
    assert failure.value.code == "missing_required_variable"  # nosec B101 - pytest typed validation assertion
    assert failure.value.variable_name == "source"  # nosec B101 - pytest required-field identity assertion


@pytest.mark.parametrize("value, rendered", [(None, ""), ("", ""), (" \t", " \t"), (0, "0"), (False, "False")])
def test_canonical_optional_values_keep_literal_rendering(value, rendered):
    result = assemble_prompt_definition(definition(required=False), {"source": value})
    assert result.messages == [{"role": "user", "content": "Source " + rendered}]  # nosec B101 - pytest optional-value compatibility assertion


def test_absent_required_value_keeps_declared_default():
    result = assemble_prompt_definition(definition(default="DEFAULT"), {})
    assert result.messages == [{"role": "user", "content": "Source DEFAULT"}]  # nosec B101 - pytest default compatibility assertion


@pytest.mark.asyncio
@pytest.mark.parametrize("owned_prompt", ["legacy", "structured"], indirect=True)
async def test_non_strict_runner_preserves_required_input_error_code(
    isolated_db,
    owned_prompt,
    monkeypatch,
):
    case = isolated_db.create_test_case(
        project_id=owned_prompt["project_id"], name="Required empty source", inputs={"source": ""}
    )
    adapter = Mock(return_value="Misleading response")
    monkeypatch.setattr(StudioTestRunner, "_call_adapter", adapter)
    result = await StudioTestRunner(isolated_db).run_test_case(owned_prompt["id"], case["id"])
    adapter.assert_not_called()
    assert result["actual"] == {
        "error": "Required prompt inputs are missing or empty.",
        "error_code": "missing_required_variable",
    }  # nosec B101 - pytest input-error classification assertion
    saved = isolated_db.get_test_run(result["id"])
    assert saved["outputs"] == result["actual"]  # nosec B101 - pytest failed-result persistence assertion


@pytest.fixture
def credential_runtime(monkeypatch):
    runtime = SimpleNamespace(
        resolve=AsyncMock(return_value=SimpleNamespace(api_key=None, app_config={}, credentials_resolved=True)),
        mark_used=AsyncMock(),
        close=AsyncMock(),
    )
    factory = Mock(return_value=runtime)
    monkeypatch.setattr(endpoint, "ProviderCredentialRuntime", factory)
    return factory, runtime


def owned_request(user_id=1):
    request = Request({"type": "http", "method": "POST", "path": "/api/v1/prompt-studio/test-cases/run", "headers": []})
    request.state.auth = AuthContext(principal=AuthPrincipal(kind="user", user_id=user_id))
    return request


@pytest.mark.asyncio
@pytest.mark.parametrize("owned_prompt", ["legacy", "structured"], indirect=True)
@pytest.mark.parametrize("inputs", [{}, {"source": None}, {"source": ""}, {"source": " \t\n"}])
async def test_owned_run_endpoint_returns_input_400_before_dispatch_or_success_history(
    isolated_db,
    owned_prompt,
    monkeypatch,
    credential_runtime,
    inputs,
):
    case = isolated_db.create_test_case(
        project_id=owned_prompt["project_id"], name="Required empty source", inputs=inputs
    )
    before = isolated_db.list_test_runs(owned_prompt["project_id"])
    adapter = Mock(return_value="Misleading response")
    monkeypatch.setattr(StudioTestRunner, "_call_adapter", adapter)
    with pytest.raises(HTTPException) as failure:
        await endpoint.run_test_cases_simple(
            RunTestCasesSimpleRequest(
                prompt_id=owned_prompt["id"], test_case_ids=[case["id"]], provider="ollama", model="local-test"
            ),
            owned_request(),
            db=isolated_db,
            user_context={"user_id": "1", "is_admin": False},
        )
    assert failure.value.status_code == 400  # nosec B101 - pytest owned input-error status assertion
    assert failure.value.detail == {
        "error_code": "missing_required_variable",
        "message": "Required prompt inputs are missing or empty.",
    }  # nosec B101 - pytest bounded correction-guidance assertion
    adapter.assert_not_called()
    credential_runtime[1].mark_used.assert_not_awaited()
    credential_runtime[1].close.assert_awaited_once()
    assert isolated_db.list_test_runs(owned_prompt["project_id"]) == before  # nosec B101 - pytest no successful execution-history assertion


@pytest.mark.asyncio
@pytest.mark.parametrize("owned_prompt", ["legacy", "structured"], indirect=True)
async def test_foreign_owner_denied_before_required_input_details(
    isolated_db,
    owned_prompt,
    credential_runtime,
):
    case = isolated_db.create_test_case(
        project_id=owned_prompt["project_id"], name="Required empty source", inputs={"source": ""}
    )
    with pytest.raises(HTTPException) as failure:
        await endpoint.run_test_cases_simple(
            RunTestCasesSimpleRequest(
                prompt_id=owned_prompt["id"], test_case_ids=[case["id"]], provider="ollama", model="local-test"
            ),
            owned_request(2),
            db=isolated_db,
            user_context={"user_id": "2", "is_admin": False},
        )
    assert failure.value.status_code == 403  # nosec B101 - pytest retained foreign-owner boundary assertion
    credential_runtime[0].assert_not_called()
