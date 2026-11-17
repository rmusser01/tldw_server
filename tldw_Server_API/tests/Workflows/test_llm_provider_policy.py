"""A workflow must refuse denied provider calls before external dispatch."""

import json

import pytest

from tldw_Server_API.app.core.AuthNZ import llm_provider_overrides as overrides
from tldw_Server_API.app.core.AuthNZ.byok_runtime import ByokResolutionError
from tldw_Server_API.app.core.Chat import chat_service
from tldw_Server_API.app.core.Workflows.adapters import run_llm_adapter

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("healthy_no_override_tts_credential_snapshot")]


@pytest.fixture
def provider_calls(monkeypatch):
    for key in ("TEST_MODE", "TLDW_TEST_MODE"):
        monkeypatch.delenv(key, raising=False)
    calls = []

    async def stream():
        yield "data: " + json.dumps({"choices": [{"delta": {"content": "allowed"}}]})
        yield "data: [DONE]"

    async def dispatch(**kwargs):
        calls.append(kwargs)
        return stream() if kwargs.get("stream") else {"choices": [{"message": {"content": "allowed"}}]}

    monkeypatch.setattr(chat_service, "perform_chat_api_call_async", dispatch)
    return calls


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("policy", ["disabled", "model", "unhealthy"])
async def test_denied_workflow_provider_never_dispatches(provider_calls, policy, stream):
    overrides.set_llm_provider_overrides_cache_for_tests(
        {
            "llama.cpp": overrides.LLMProviderOverride(
                provider="llama.cpp", is_enabled=policy != "disabled", allowed_models=["allowed-model"]
            )
        },
        healthy=policy != "unhealthy",
    )
    with pytest.raises(ByokResolutionError):
        await run_llm_adapter(
            {"provider": "llama-cpp", "model": "forbidden-model", "prompt": "hello", "stream": stream},
            {"user_id": "1"},
        )
    assert provider_calls == []  # nosec B101 -- regression: denial precedes external dispatch.


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_enabled_allowed_workflow_provider_dispatches(provider_calls, stream):
    overrides.set_llm_provider_overrides_cache_for_tests(
        {
            "llama.cpp": overrides.LLMProviderOverride(
                provider="llama.cpp", is_enabled=True, allowed_models=["allowed-model"]
            )
        }
    )
    result = await run_llm_adapter(
        {"provider": "llama-cpp", "model": "allowed-model", "prompt": "hello", "stream": stream},
        {"user_id": "1"},
    )
    assert result["text"] == "allowed" and len(provider_calls) == 1  # nosec B101 -- allowed call remains functional.


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("configured_model", ["allowed-model", "forbidden-model", None])
async def test_workflow_default_model_is_checked_and_bound(provider_calls, monkeypatch, stream, configured_model):
    from tldw_Server_API.app.core.LLM_Calls import adapter_utils

    monkeypatch.setattr(
        adapter_utils, "ensure_app_config", lambda config=None: {"llama_api": {"model": configured_model}}
    )
    overrides.set_llm_provider_overrides_cache_for_tests(
        {"llama.cpp": overrides.LLMProviderOverride(provider="llama.cpp", allowed_models=["allowed-model"])}
    )
    config = {"provider": "llama.cpp", "prompt": "hello", "stream": stream}
    if configured_model != "allowed-model":
        with pytest.raises(ByokResolutionError):
            await run_llm_adapter(config, {"user_id": "1"})
        assert provider_calls == []  # nosec B101 -- an unspecified or denied default cannot bypass restrictions.
    else:
        await run_llm_adapter(config, {"user_id": "1"})
        assert provider_calls[0]["model"] == configured_model  # nosec B101 -- policy and dispatch use one model.


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_workflow_without_override_retains_optional_model(provider_calls, stream):
    result = await run_llm_adapter({"provider": "llama.cpp", "prompt": "hello", "stream": stream}, {"user_id": "1"})
    assert result["text"] == "allowed" and len(provider_calls) == 1  # nosec B101 -- unrestricted local defaults remain supported.


@pytest.mark.asyncio
async def test_provider_denial_fails_persisted_prerequisite(provider_calls, tmp_path):
    from tldw_Server_API.app.core.DB_Management.Workflows_DB import WorkflowsDatabase
    from tldw_Server_API.app.core.Workflows.engine import WorkflowEngine

    overrides.set_llm_provider_overrides_cache_for_tests(
        {"llama.cpp": overrides.LLMProviderOverride(provider="llama.cpp", is_enabled=False)}
    )
    db = WorkflowsDatabase(str(tmp_path / "provider-policy.db"))
    try:
        db.create_run(
            run_id="provider-policy",
            tenant_id="default",
            user_id="1",
            inputs={},
            workflow_id=None,
            definition_version=1,
            definition_snapshot={
                "name": "denied",
                "version": 1,
                "steps": [
                    {
                        "id": "provider",
                        "type": "llm",
                        "retry": 0,
                        "config": {"provider": "llama.cpp", "model": "allowed-model", "prompt": "hello"},
                    },
                    {"id": "dependent", "type": "template_render", "retry": 0, "config": {"template": "must not run"}},
                ],
            },
        )
        await WorkflowEngine(db).start_run("provider-policy")
        rows = db.list_step_runs(run_id="provider-policy")
        assert (db.get_run("provider-policy").status, [(r["step_id"], r["status"]) for r in rows], provider_calls) == (
            "failed",
            [("provider", "failed")],
            [],
        )  # nosec B101 -- denied prerequisite never admits dependent work.
    finally:
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_workflow_alias_default_model_is_bound(provider_calls, monkeypatch, stream):
    from tldw_Server_API.app.core.LLM_Calls import adapter_utils

    monkeypatch.setattr(
        adapter_utils, "ensure_app_config", lambda config=None: {"llama_api": {"model": "allowed-model"}}
    )
    overrides.set_llm_provider_overrides_cache_for_tests(
        {"llama.cpp": overrides.LLMProviderOverride(provider="llama.cpp", allowed_models=["allowed-model"])}
    )
    await run_llm_adapter({"provider": "llama-cpp", "prompt": "hello", "stream": stream}, {"user_id": "1"})
    assert provider_calls[0]["model"] == "allowed-model"  # nosec B101 -- alias lookup uses canonical config.


def test_strict_workflow_policy_keeps_disabled_denial_precedence():
    overrides.set_llm_provider_overrides_cache_for_tests(
        {
            "llama.cpp": overrides.LLMProviderOverride(
                provider="llama.cpp", is_enabled=False, allowed_models=["allowed-model"]
            )
        }
    )
    with pytest.raises(overrides.ProviderOverridePolicyError) as error:
        overrides.capture_provider_override_call_snapshot("llama.cpp").enforce(None, require_model=True)
    assert error.value.policy_code == "provider_disabled"  # nosec B101 -- final-model guard preserves existing denial.
