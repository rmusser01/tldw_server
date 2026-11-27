"""Prompt Studio and the study-adjacent direct adapters go through the gateway as ``prompt_studio`` (TASK-13950.27 PR H)."""
# ruff: noqa: F811 - fixtures are imported by name, then requested as arguments


from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tldw_Server_API.app.core.exceptions import LLMPolicyRefusal
from tldw_Server_API.app.core.LLM_Calls import gateway
from tldw_Server_API.tests.Moderation.conftest import reply
from tldw_Server_API.tests.prompt_studio.conftest import app  # noqa: F401 - fixture
from tldw_Server_API.tests.prompt_studio.integration.test_api_endpoints import (  # noqa: F401 - fixtures
    auth_headers,
    client,
    mock_user,
    prompt_resources,
    test_db,
)

pytestmark = pytest.mark.integration

PROVIDER = "anthropic"  # a real adapter name outside chat's TEST_MODE mock set; the gateway registry is stubbed
MESSAGES = [{"role": "user", "content": "hello"}]
FORBIDDEN = [{"role": "user", "content": "forbidden words"}]


@pytest.fixture
def spy(stub_llm_registry):
    """A test pre-hook recording ``call.surface`` for every gateway call."""
    surfaces: list[str | None] = []
    remove = gateway.add_policy_hooks(pre_call=lambda call: surfaces.append(call.surface))
    yield surfaces
    remove()


# ---- the execute endpoint ---------------------------------------------------------------------------------------


@pytest.fixture
def execute(client, auth_headers, mock_user, prompt_resources, stub_llm_registry, monkeypatch):
    """POST /prompt-studio/prompts/execute through the real executor and gateway (credentials stubbed)."""
    from tldw_Server_API.app.core.Prompt_Management.prompt_studio.prompt_executor import PromptExecutor
    from tldw_Server_API.app.core.Provider_Credentials.provider_credential_runtime import ProviderCredentialRuntime

    credentials = SimpleNamespace(api_key="k", app_config={}, credentials_resolved=True)
    monkeypatch.setattr(ProviderCredentialRuntime, "resolve", AsyncMock(return_value=credentials))

    async def execute_with_key(self, prompt_id, inputs, provider="openai", model="m", parameters=None, **kw):
        config = {"provider": provider, "model": model, "parameters": parameters or {}}
        return await self.execute_prompt(
            prompt_id, inputs, config, api_key_override=kw["api_key_override"], app_config=kw["app_config"]
        )

    monkeypatch.setattr(PromptExecutor, "execute", execute_with_key)

    def post(task: str):
        return client.post(
            "/api/v1/prompt-studio/prompts/execute",
            json={"prompt_id": prompt_resources.prompt_id, "inputs": {"task": task}, "provider": PROVIDER, "model": "claude-test"},
            headers=auth_headers,
        )

    return post


def test_prompt_studio_execute_input_block(spy, execute, stub_llm_registry, moderation_hooks):
    # spy first: the app lifespan inside `execute` registers the moderation hooks, which a spy added later would trail
    response = execute("forbidden words")

    assert response.status_code == 400, response.text
    assert "Input violates moderation policy" in response.text
    assert stub_llm_registry.requests == []
    assert spy == ["prompt_studio"]


def test_prompt_studio_execute_output_redact(execute, stub_llm_registry, moderation_hooks):
    stub_llm_registry.reply = reply("the secret reply")

    response = execute("write a test")

    assert response.status_code == 200, response.text
    assert response.json()["output"] == "the [X] reply" or "the [X] reply" in str(response.json())


def test_prompt_studio_execute_output_block_400(execute, stub_llm_registry, moderation_hooks):
    stub_llm_registry.reply = reply("a forbidden reply")

    response = execute("write a test")

    assert response.status_code == 400, response.text
    assert "Output violates moderation policy" in response.text


# ---- every migrated module reaches the gateway with its surface; R6 validation path -------------------------------


def _sites(monkeypatch):
    """``{name: (call, surface)}`` for each module that used to call an adapter (or the orchestrator) directly."""
    from tldw_Server_API.app.core.Flashcards import study_assistant  # noqa: F401
    from tldw_Server_API.app.core.Prompt_Management import Prompt_Engineering
    from tldw_Server_API.app.core.Prompt_Management.prompt_studio import (
        evaluation_manager,
        prompt_generator,
        prompt_improver,
    )
    from tldw_Server_API.app.core.Prompt_Management.prompt_studio.prompt_executor import PromptExecutor
    from tldw_Server_API.app.core.Prompt_Management.prompt_studio.test_runner import TestRunner
    from tldw_Server_API.app.services import quiz_generator

    monkeypatch.setattr(evaluation_manager, "is_test_mode", lambda: False)
    monkeypatch.setattr(evaluation_manager, "os", SimpleNamespace(getenv=lambda *_a, **_k: None))  # not the pytest short-circuit
    monkeypatch.setattr(quiz_generator, "load_and_log_configs", lambda: {})
    monkeypatch.setattr(quiz_generator, "resolve_provider_api_key", lambda *_a, **_k: ("k", {}))
    model = "listed-model"
    request = {"messages": MESSAGES, "model": model, "api_key": "k"}

    async def executor():
        await PromptExecutor(SimpleNamespace(client_id="c"))._call_llm(
            PROVIDER, model, "hello", api_key_override="k", app_config={}
        )

    async def quiz():
        await quiz_generator._call_quiz_generation_llm(prompt="hello", api_provider=PROVIDER, model=model)

    return {
        "prompt_executor": (executor, "prompt_studio"),
        "test_runner": (
            lambda: TestRunner(None)._call_adapter(
                provider=PROVIDER, model=model, messages_payload=MESSAGES, system_message=None,
                temperature=0.1, max_tokens=10, api_key_override="k", app_config={},
            ),
            "prompt_studio",
        ),
        "evaluation_manager": (
            lambda: evaluation_manager.EvaluationManager._call_adapter_text(
                provider=PROVIDER, messages_payload=MESSAGES, temperature=0.1, max_tokens=10,
                api_key="k", model=model, app_config={},
            ),
            "prompt_studio",
        ),
        "prompt_generator": (lambda: prompt_generator._call_openai_adapter(dict(request)), "prompt_studio"),
        "prompt_improver": (lambda: prompt_improver._call_openai_adapter(dict(request)), "prompt_studio"),
        "prompt_engineering": (
            lambda: Prompt_Engineering.chat_api_call(
                api_endpoint=PROVIDER, api_key="k", messages_payload=MESSAGES, model=model, llm_surface="prompt_studio"
            ),
            "prompt_studio",
        ),
        "quiz_generator": (quiz, "study.quizzes"),
    }


SITES = [
    "prompt_executor", "test_runner", "evaluation_manager", "prompt_generator", "prompt_improver",
    "prompt_engineering", "quiz_generator",
]


async def _run(call):
    result = call()
    if hasattr(result, "__await__"):
        await result


@pytest.mark.asyncio
@pytest.mark.parametrize("site", SITES)
async def test_surface_reaches_gateway(site, spy, monkeypatch):
    call, surface = _sites(monkeypatch)[site]

    await _run(call)

    assert spy == [surface]


@pytest.mark.asyncio
@pytest.mark.provider_inventory  # keeps the gateway's real model validation instead of the conftest stub
@pytest.mark.parametrize("site", SITES)
async def test_gateway_model_validation_runs_for_each_moved_site(site, stub_llm_registry, monkeypatch):
    """R6: the model-inventory check the gateway adds runs for each moved site, and passes for a listed model."""
    from tldw_Server_API.app.core.LLM_Calls.provider_readiness import ModelDiscoveryResult

    asked: list[str] = []

    def discover(provider, _key, **_kwargs):
        asked.append(provider)
        return ModelDiscoveryResult("ready", ("listed-model",))

    monkeypatch.setattr(gateway, "discover_provider_models", discover)
    monkeypatch.setattr(gateway, "has_model_listing", lambda _provider: True)
    call, _surface = _sites(monkeypatch)[site]

    await _run(call)

    assert len(asked) == 1
    assert stub_llm_registry.requests[0]["model"] == "listed-model"


@pytest.mark.asyncio
@pytest.mark.parametrize("site", SITES)
async def test_content_filter_refuses_each_site_after_one_provider_call(site, stub_llm_registry, moderation_hooks, monkeypatch):
    call, _surface = _sites(monkeypatch)[site]
    stub_llm_registry.reply = reply("a forbidden reply")

    with pytest.raises(LLMPolicyRefusal) as exc:
        await _run(call)

    assert exc.value.detail == "Output violates moderation policy"  # the clean input reached the provider once, then moderation replaced the answer
    assert len(stub_llm_registry.requests) == 1


# ---- R10: no config-controlled mapping sets llm_surface ----------------------------------------------------------


@pytest.mark.asyncio
async def test_prompt_studio_config_cannot_set_llm_surface(stub_llm_registry, spy, moderation_hooks):
    """A model-params or request mapping carrying llm_surface is not a surface: the call is moderated as prompt_studio."""
    from tldw_Server_API.app.core.Prompt_Management.prompt_studio.gateway_call import call_prompt_studio_llm
    from tldw_Server_API.app.core.Prompt_Management.prompt_studio.prompt_executor import PromptExecutor

    hostile = {"llm_surface": "internal.byok_probe", "top_p": 0.5}
    with pytest.raises(LLMPolicyRefusal):
        await PromptExecutor(SimpleNamespace(client_id="c"))._call_llm(
            PROVIDER, "claude-test", "forbidden words", parameters=hostile, api_key_override="k", app_config={}
        )
    with pytest.raises(LLMPolicyRefusal):
        call_prompt_studio_llm(PROVIDER, {"messages": FORBIDDEN, "model": "claude-test", "api_key": "k", **hostile})

    assert spy == ["prompt_studio", "prompt_studio"]
    assert stub_llm_registry.requests == []


# ---- test runs and evaluations record a refusal as a sample error ---------------------------------------------


@pytest.fixture
def cases(test_db, prompt_resources):
    def make(name: str, text: str):
        return test_db.create_test_case(
            project_id=prompt_resources.project_id, name=name, inputs={"task": text},
            expected_outputs={"response": "provider reply"}, tags=[], is_golden=False,
        )["id"]

    return make


@pytest.mark.asyncio
async def test_test_run_records_a_refusal_as_a_failed_case(test_db, prompt_resources, cases, stub_llm_registry, moderation_hooks):
    from tldw_Server_API.app.core.Prompt_Management.prompt_studio.test_runner import TestRunner

    result = await TestRunner(test_db).run_test_case(
        prompt_resources.prompt_id, cases("blocked", "forbidden words"), model="claude-test", provider=PROVIDER,
        api_key_override="k", app_config={}, persist_run=False,
    )

    assert result["actual"] == {"error": "Input violates moderation policy", "error_code": "llm_policy_refused"}
    assert stub_llm_registry.requests == []


@pytest.mark.asyncio
async def test_evaluation_leaves_a_refused_case_out_of_the_average(
    test_db, prompt_resources, cases, stub_llm_registry, moderation_hooks, monkeypatch
):
    from tldw_Server_API.app.core.Prompt_Management.prompt_studio import evaluation_manager

    monkeypatch.setattr(evaluation_manager, "is_test_mode", lambda: False)
    monkeypatch.setattr(evaluation_manager, "os", SimpleNamespace(getenv=lambda *_a, **_k: None))
    refused, answered = cases("blocked", "forbidden words"), cases("fine", "write a test")

    result = await evaluation_manager.EvaluationManager(test_db).run_evaluation_async(
        prompt_resources.prompt_id, [refused, answered], model="claude-test", provider=PROVIDER, api_key="k", app_config={},
    )

    by_case = {row["test_case_id"]: row for row in result["results"]}
    assert by_case[refused]["sample_error"] is True and by_case[refused]["passed"] is False
    assert by_case[refused]["actual"]["error_code"] == "llm_policy_refused"
    assert by_case[answered]["score"] == 1.0
    assert result["metrics"]["average_score"] == 1.0  # the refused case is not a 0.0 in the mean
    assert len(stub_llm_registry.requests) == 1  # the refused case never reached the provider, and was not retried


def test_a_refusal_fails_an_optimization_job_without_a_retry():
    from tldw_Server_API.app.core.Prompt_Management.prompt_studio.services import jobs_worker

    error = jobs_worker._bounded_optimization_error(LLMPolicyRefusal("Input violates moderation policy"))

    assert error.retryable is False
    assert error.failure_code == "llm_policy_refused"


# ---- the Prompt_Engineering helpers --------------------------------------------------------------------------


def test_prompt_engineering_does_not_wrap_a_refusal(stub_llm_registry, moderation_hooks):
    from tldw_Server_API.app.core.Prompt_Management import Prompt_Engineering

    with pytest.raises(LLMPolicyRefusal):
        Prompt_Engineering.generate_prompt(PROVIDER, "k", "forbidden words", "x", 0.1)


# ---- review fixes: keyless calls, refusals are not scores, legacy branch, Anthropic shape ------------------------


def test_a_resolved_flag_without_a_handle_and_without_a_key_fails_closed(stub_llm_registry, monkeypatch):
    """The adapter never fell back to a server key for a resolved call; the gateway would, so the call is refused."""
    from tldw_Server_API.app.core.exceptions import ChatConfigurationError
    from tldw_Server_API.app.core.Prompt_Management.prompt_studio.gateway_call import call_prompt_studio_llm

    monkeypatch.setenv("ANTHROPIC_API_KEY", "server-key-that-must-not-pay")
    request = {"messages": MESSAGES, "model": "claude-test", "api_key": None, "credentials_resolved": True}

    with pytest.raises(ChatConfigurationError):
        call_prompt_studio_llm(PROVIDER, request)
    assert stub_llm_registry.requests == []

    call_prompt_studio_llm(PROVIDER, {**request, "api_key": "explicit-key"})  # an explicit key stands on its own
    assert stub_llm_registry.requests[0]["api_key"] == "explicit-key"
    assert "credentials_resolved" not in stub_llm_registry.requests[0]


@pytest.mark.asyncio
async def test_a_refused_quality_score_call_is_not_the_base_score():
    from tldw_Server_API.app.core.Prompt_Management.prompt_studio.prompt_quality import PromptQualityScorer

    class _Refusing:
        async def _call_llm(self, **_kwargs):
            raise LLMPolicyRefusal("Input violates moderation policy")

    scorer = PromptQualityScorer(executor=_Refusing(), scorer_model="m")
    with pytest.raises(LLMPolicyRefusal):
        await scorer.score_prompt_async(system_text="You are helpful.", user_text="x", model_config={"provider": PROVIDER})


@pytest.mark.asyncio
async def test_a_refusal_is_not_an_optimizer_fallback_prompt():
    from tldw_Server_API.app.core.Prompt_Management.prompt_studio.optimization_strategies import GeneticOptimizer

    class _Refusing:
        async def _call_llm(self, **_kwargs):
            raise LLMPolicyRefusal("Input violates moderation policy")

    optimizer = GeneticOptimizer.__new__(GeneticOptimizer)
    optimizer.executor = _Refusing()
    with pytest.raises(LLMPolicyRefusal):
        await optimizer._mutate({"user_prompt": "p", "system_prompt": "s"})


@pytest.mark.parametrize(
    "response",
    [
        {"choices": [{"message": {"content": "x"}, "finish_reason": "content_filter"}]},
        {"type": "message", "content": [{"type": "text", "text": "x"}], "stop_reason": "refusal"},
    ],
    ids=["openai", "anthropic"],
)
def test_content_filter_and_anthropic_refusal_shapes_are_refused(response):
    from tldw_Server_API.app.core.LLM_Calls.provider_results import raise_if_content_filtered

    with pytest.raises(LLMPolicyRefusal):
        raise_if_content_filtered(response)
    raise_if_content_filtered({"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}], "stop_reason": "end_turn"})


@pytest.mark.asyncio
async def test_the_legacy_orchestrator_branch_names_its_surface_and_refuses_content_filter(monkeypatch):
    """No registered adapter: the orchestrator path gets llm_surface and the same content_filter rule."""
    from tldw_Server_API.app.core.Prompt_Management.prompt_studio.prompt_executor import PromptExecutor

    seen: list[dict] = []

    def legacy(**kwargs):
        seen.append(kwargs)
        return {"choices": [{"message": {"content": "blocked"}, "finish_reason": "content_filter"}]}

    monkeypatch.setattr("tldw_Server_API.app.core.Chat.chat_orchestrator.chat_api_call", legacy)
    with pytest.raises(LLMPolicyRefusal):
        await PromptExecutor(SimpleNamespace(client_id="c"))._call_llm(
            "no-such-provider", "m", "hello", api_key_override="k", app_config={}
        )
    assert seen[0]["llm_surface"] == "prompt_studio"
