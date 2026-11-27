"""Workflow LLM steps name their surface and a refusal fails the step (TASK-13950.27 PR G)."""

from __future__ import annotations

import pytest

from tldw_Server_API.app.core.exceptions import LLMPolicyRefusal
from tldw_Server_API.app.core.LLM_Calls import gateway
from tldw_Server_API.app.core.Workflows.failures import is_retriable_reason
from tldw_Server_API.tests.Moderation.conftest import PROVIDER, reply

pytestmark = pytest.mark.integration

CONTEXT = {"user_id": "1", "workflow_id": "w", "step_run_id": "s"}
STEP = {"provider": PROVIDER, "model": "m"}


@pytest.fixture
def spy(stub_llm_registry):
    """A test pre-hook recording ``call.surface`` for every gateway call."""
    surfaces: list[str | None] = []
    remove = gateway.add_policy_hooks(pre_call=lambda call: surfaces.append(call.surface))
    yield surfaces
    remove()


async def _llm(**extra):
    from tldw_Server_API.app.core.Workflows.adapters.llm.llm import run_llm_adapter

    return await run_llm_adapter({**STEP, "prompt": "hello", **extra}, CONTEXT)


async def test_workflow_llm_step_input_block_fails_step(moderation_hooks, stub_llm_registry):
    with pytest.raises(LLMPolicyRefusal) as exc:
        await _llm(prompt="forbidden words")
    assert exc.value.detail == "Input violates moderation policy"
    assert stub_llm_registry.requests == []


async def test_workflow_llm_step_output_redact(moderation_hooks, stub_llm_registry):
    stub_llm_registry.reply = reply("the secret reply")
    assert (await _llm())["text"] == "the [X] reply"


async def test_workflow_step_config_cannot_set_llm_surface(stub_llm_registry, spy, moderation_hooks):
    """R10: a step config carrying llm_surface is not a surface; the step is still moderated."""
    with pytest.raises(LLMPolicyRefusal):
        await _llm(prompt="forbidden words", llm_surface="internal.byok_probe", call_args={"llm_surface": "internal.byok_probe"})
    assert spy == ["workflows.step"]
    assert stub_llm_registry.requests == []


@pytest.mark.parametrize("adapter", ["flashcards", "keywords", "image_describe", "translate"])
async def test_refusal_is_not_swallowed_by_a_broad_except(moderation_hooks, stub_llm_registry, adapter):
    """These adapters end in ``except Exception`` and used to turn any failure into an error dict or fallback."""
    from tldw_Server_API.app.core.Workflows.adapters.content import run_flashcard_generate_adapter
    from tldw_Server_API.app.core.Workflows.adapters.content.image import run_image_describe_adapter
    from tldw_Server_API.app.core.Workflows.adapters.llm.translate import run_translate_adapter
    from tldw_Server_API.app.core.Workflows.adapters.text.nlp import run_keyword_extract_adapter

    text = "forbidden " + "words " * 20
    calls = {
        "flashcards": lambda: run_flashcard_generate_adapter({**STEP, "text": text}, CONTEXT),
        "keywords": lambda: run_keyword_extract_adapter({**STEP, "text": text, "method": "llm"}, CONTEXT),
        "image_describe": lambda: run_image_describe_adapter({**STEP, "image_url": "data:image/png;base64,AAAA", "prompt": text}, CONTEXT),
        "translate": lambda: run_translate_adapter({"input": text, "target_lang": "fr"}, CONTEXT),
    }
    with pytest.raises(LLMPolicyRefusal):
        await calls[adapter]()


def test_a_refusal_is_not_retried_by_the_engine():
    assert is_retriable_reason("400") is False  # str(LLMPolicyRefusal) starts "400:"


async def test_translate_reaches_gateway(spy):
    from tldw_Server_API.app.core.Workflows.adapters.llm.translate import run_translate_adapter

    out = await run_translate_adapter({"input": "hello", "target_lang": "fr"}, CONTEXT)
    assert spy == ["workflows.step"]
    assert out["text"] == "provider reply" and out["provider"] == "openai"


@pytest.mark.provider_inventory  # keeps the gateway's real model validation instead of the conftest stub
async def test_translate_runs_the_gateway_model_validation(stub_llm_registry, monkeypatch):
    """R6: the model-inventory check the gateway adds now runs for translate, and passes for a listed model."""
    from tldw_Server_API.app.core.LLM_Calls import adapter_utils
    from tldw_Server_API.app.core.LLM_Calls.provider_readiness import ModelDiscoveryResult
    from tldw_Server_API.app.core.Workflows.adapters.llm.translate import run_translate_adapter

    asked: list[str] = []

    def discover(provider, _key, **_kwargs):
        asked.append(provider)
        return ModelDiscoveryResult("ready", ("listed-model",))

    monkeypatch.setattr(gateway, "discover_provider_models", discover)
    monkeypatch.setattr(adapter_utils, "resolve_provider_model", lambda *_a: "listed-model")

    out = await run_translate_adapter({"input": "hello", "target_lang": "fr"}, CONTEXT)

    assert asked == ["openai"]
    assert stub_llm_registry.requests[0]["model"] == "listed-model"
    assert "fallback" not in out
