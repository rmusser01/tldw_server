"""Required execution inputs stop at the shared template boundary."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from tldw_Server_API.app.api.v1.endpoints.prompt_studio import prompt_studio_prompts as endpoint
from tldw_Server_API.app.api.v1.schemas.prompt_studio_schemas import ExecutePromptSimpleRequest
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.prompt_executor import PromptExecutor
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.test_runner import TestRunner as StudioTestRunner

pytestmark = pytest.mark.integration
TEMPLATE = "Source:\n{{source}}\n\nWho directs Rowan Observatory? Give full name only."
SOURCE = "Rowan Observatory is directed by Dr. Mira Vale."
MODEL_CONFIG = {"provider": "ollama", "model": "local-test", "parameters": {}}


@pytest.fixture
def legacy_prompt(isolated_db):
    project = isolated_db.create_project(name="Required-variable owner", user_id="1")
    return isolated_db.create_prompt(
        project_id=project["id"], name="Required source", user_prompt=TEMPLATE,
        system_prompt="Keep literal {{system_marker}} instructions.",
    )


@pytest.fixture
def provider_call(monkeypatch):
    """Observe the external seam after real DB lookup and prompt construction."""
    call = AsyncMock(return_value={"content": "Dr. Mira Vale", "tokens": 7})
    monkeypatch.setattr(PromptExecutor, "_call_llm", call)
    return call


@pytest.fixture
def credential_runtime(monkeypatch):
    """Isolate external credential resolution; keep canonical owner/scope logic."""
    runtime = SimpleNamespace(
        resolve=AsyncMock(return_value=SimpleNamespace(
            api_key=None, app_config={}, credentials_resolved=True,
        )),
        close=AsyncMock(),
    )
    factory = Mock(return_value=runtime)
    monkeypatch.setattr(endpoint, "ProviderCredentialRuntime", factory)
    return factory, runtime


def request():
    return Request({"type": "http", "method": "POST", "path": "/api/v1/prompt-studio/prompts/execute", "headers": []})


def structured_definition(*, required=True, default=None, literal=False):
    return {
        "schema_version": 1, "format": "structured",
        "variables": [] if literal else [{"name": "source", "required": required, "default_value": default}],
        "blocks": [{"id": "source", "name": "Source", "role": "user", "order": 10,
                    "content": "Source {{source}}", "is_template": not literal}],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("token", ["{{source}}", "{source}", "$source", "<source>"])
async def test_missing_legacy_execution_variable_never_dispatches_provider(isolated_db, legacy_prompt, provider_call, token):
    prompt = isolated_db.create_prompt(
        project_id=legacy_prompt["project_id"], name="Missing " + token, user_prompt="Source " + token,
    )
    result = await PromptExecutor(isolated_db).execute(prompt["id"], inputs={}, provider="ollama", model="local-test")
    provider_call.assert_not_awaited()
    assert result["success"] is False  # nosec B101 - pytest behavior assertion
    assert result["error_code"] == "missing_required_variable"  # nosec B101 - pytest behavior assertion
    assert result["error"] == "Missing required variable: source"  # nosec B101 - pytest behavior assertion


@pytest.mark.asyncio
async def test_unrelated_input_cannot_satisfy_required_variable(isolated_db, legacy_prompt, provider_call):
    result = await PromptExecutor(isolated_db).execute_prompt(legacy_prompt["id"], {"source_extra": SOURCE}, MODEL_CONFIG)
    provider_call.assert_not_awaited()
    assert result["error"] == "Missing required variable: source"  # nosec B101 - pytest behavior assertion


@pytest.mark.asyncio
@pytest.mark.parametrize("version", [1, 2])
async def test_valid_source_renders_for_each_actual_prompt_version(isolated_db, legacy_prompt, provider_call, version):
    prompt = legacy_prompt if version == 1 else isolated_db.create_prompt(
        project_id=legacy_prompt["project_id"], name="Required source", version_number=2,
        parent_version_id=legacy_prompt["id"], user_prompt=TEMPLATE,
        system_prompt="Keep literal {{system_marker}} instructions.",
    )
    result = await PromptExecutor(isolated_db).execute_prompt(prompt["id"], {"source": SOURCE}, MODEL_CONFIG)
    provider_call.assert_awaited_once()
    assert provider_call.call_args.kwargs["prompt"] == "Source:\n" + SOURCE + "\n\nWho directs Rowan Observatory? Give full name only."  # nosec B101 - pytest behavior assertion
    assert provider_call.call_args.kwargs["system_prompt"] == "Keep literal {{system_marker}} instructions."  # nosec B101 - pytest behavior assertion
    assert result["raw_output"] == "Dr. Mira Vale"  # nosec B101 - pytest behavior assertion
    assert isolated_db.get_prompt(legacy_prompt["id"])["version_number"] == 1  # nosec B101 - pytest behavior assertion


@pytest.mark.asyncio
@pytest.mark.parametrize("value, expected", [(0, "0"), (False, "False")])
async def test_required_presence_keeps_zero_and_false_value_semantics(isolated_db, legacy_prompt, provider_call, value, expected):
    result = await PromptExecutor(isolated_db).execute_prompt(legacy_prompt["id"], {"source": value}, MODEL_CONFIG)
    provider_call.assert_awaited_once()
    assert provider_call.call_args.kwargs["prompt"].startswith("Source:\n" + expected + "\n\n")  # nosec B101 - pytest behavior assertion
    assert result["success"] is True  # nosec B101 - pytest behavior assertion


@pytest.mark.asyncio
async def test_replacement_text_and_invalid_token_syntax_remain_literal(isolated_db, legacy_prompt, provider_call):
    prompt = isolated_db.create_prompt(
        project_id=legacy_prompt["project_id"], name="Literal replacement", user_prompt="{{source}} {not-a-variable}",
    )
    result = await PromptExecutor(isolated_db).execute_prompt(prompt["id"], {"source": "literal {{unbound}}"}, MODEL_CONFIG)
    assert provider_call.call_args.kwargs["prompt"] == "literal {{unbound}} {not-a-variable}"  # nosec B101 - pytest behavior assertion
    assert result["success"] is True  # nosec B101 - pytest behavior assertion


@pytest.mark.asyncio
@pytest.mark.parametrize("mode, expected", [("default", "Source DEFAULT"), ("optional", "Source "), ("literal", "Source {{source}}")])
async def test_structured_default_optional_and_literal_contracts_stay_valid(isolated_db, legacy_prompt, provider_call, mode, expected):
    definition = structured_definition(required=mode != "optional", default="DEFAULT" if mode == "default" else None, literal=mode == "literal")
    prompt = isolated_db.create_prompt(
        project_id=legacy_prompt["project_id"], name="Structured " + mode, prompt_format="structured",
        prompt_schema_version=1, prompt_definition=definition,
    )
    result = await PromptExecutor(isolated_db).execute_prompt(prompt["id"], {}, MODEL_CONFIG)
    provider_call.assert_awaited_once()
    assert provider_call.call_args.kwargs["messages"] == [{"role": "user", "content": expected}]  # nosec B101 - pytest behavior assertion
    assert result["success"] is True  # nosec B101 - pytest behavior assertion


@pytest.mark.asyncio
@pytest.mark.parametrize("prompt_format", ["legacy", "structured"])
async def test_endpoint_missing_input_returns_bounded_400_after_real_owner_gate(isolated_db, legacy_prompt, provider_call, credential_runtime, prompt_format):
    prompt = legacy_prompt if prompt_format == "legacy" else isolated_db.create_prompt(
        project_id=legacy_prompt["project_id"], name="Structured required", prompt_format="structured",
        prompt_schema_version=1, prompt_definition=structured_definition(),
    )
    with pytest.raises(HTTPException) as failure:
        await endpoint.execute_prompt_simple(
            ExecutePromptSimpleRequest(prompt_id=prompt["id"], inputs={}, provider="ollama", model="local-test"),
            request(), db=isolated_db, user_context={"user_id": "1", "is_admin": False},
        )
    provider_call.assert_not_awaited()
    assert failure.value.status_code == 400  # nosec B101 - pytest behavior assertion
    assert failure.value.detail == {"error_code": "missing_required_variable", "message": "Required prompt inputs are missing."}  # nosec B101 - pytest behavior assertion
    credential_runtime[1].close.assert_awaited_once()


@pytest.mark.asyncio
async def test_foreign_owner_is_denied_before_input_diagnostics_or_runtime(isolated_db, legacy_prompt, provider_call, credential_runtime):
    with pytest.raises(HTTPException) as failure:
        await endpoint.execute_prompt_simple(
            ExecutePromptSimpleRequest(prompt_id=legacy_prompt["id"], inputs={}, provider="ollama", model="local-test"),
            request(), db=isolated_db, user_context={"user_id": "2", "is_admin": False},
        )
    assert failure.value.status_code == 403  # nosec B101 - pytest behavior assertion
    credential_runtime[0].assert_not_called()
    provider_call.assert_not_awaited()


@pytest.mark.asyncio
async def test_test_runner_shared_execution_seam_refuses_missing_inputs(isolated_db, legacy_prompt, monkeypatch):
    case = isolated_db.create_test_case(project_id=legacy_prompt["project_id"], name="Missing source", inputs={}, expected_outputs={})
    adapter = Mock(return_value="Not specified.")
    monkeypatch.setattr(StudioTestRunner, "_call_adapter", adapter)
    result = await StudioTestRunner(isolated_db).run_test_case(
        legacy_prompt["id"], case["id"], provider="ollama", model="local-test", persist_run=False,
    )
    adapter.assert_not_called()
    assert "error" in result["actual"]  # nosec B101 - pytest behavior assertion


def test_low_level_partial_legacy_renderer_keeps_exact_existing_behavior():
    result = PromptExecutor(SimpleNamespace(client_id="partial-renderer"))._build_prompt(
        {"user_prompt": "$id $idea {id} {{id}} <id> {i-d}"}, None, {"id": "42"},
    )
    assert result == "42 $idea 42 42 42 {i-d}"  # nosec B101 - pytest behavior assertion
