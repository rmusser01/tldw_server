"""Final-review fixes for TASK-13950.27 (PR K): refusals stay refusals, worded by origin, on every path."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.exceptions import LLMPolicyRefusal
from tldw_Server_API.app.core.LLM_Calls.provider_results import (
    OUTPUT_REFUSAL_TEXT,
    PROVIDER_FILTER_REFUSAL_TEXT,
    is_content_filtered,
    raise_if_content_filtered,
)
from tldw_Server_API.tests.Moderation.conftest import DONE, PROVIDER, finish_frame, openai_frame, reply

pytestmark = pytest.mark.integration

CONTEXT = {"user_id": "1", "workflow_id": "w", "step_run_id": "s"}
OURS = {"choices": [{"message": {"content": OUTPUT_REFUSAL_TEXT}, "finish_reason": "content_filter"}]}
THEIRS = {"choices": [{"message": {"content": ""}, "finish_reason": "content_filter"}]}
OURS_ANTHROPIC = {"type": "message", "content": [{"type": "text", "text": OUTPUT_REFUSAL_TEXT}], "stop_reason": "refusal"}
THEIRS_ANTHROPIC = {"type": "message", "content": [{"type": "text", "text": "x"}], "stop_reason": "refusal"}


# ---- I6 / M9: one helper, wording by origin ---------------------------------------------------------------------


def test_the_hook_text_the_helper_compares_is_the_hooks_own():
    from tldw_Server_API.app.core.Moderation.gateway_hooks import OUTPUT_REFUSAL

    assert OUTPUT_REFUSAL_TEXT == OUTPUT_REFUSAL


@pytest.mark.parametrize(
    ("response", "wording"),
    [
        (OURS, OUTPUT_REFUSAL_TEXT),
        (OURS_ANTHROPIC, OUTPUT_REFUSAL_TEXT),
        (THEIRS, PROVIDER_FILTER_REFUSAL_TEXT),
        (THEIRS_ANTHROPIC, PROVIDER_FILTER_REFUSAL_TEXT),
    ],
    ids=["ours-openai", "ours-anthropic", "provider-openai", "provider-anthropic"],
)
def test_a_filtered_reply_is_a_refusal_worded_by_origin(response, wording):
    assert is_content_filtered(response) is True
    with pytest.raises(LLMPolicyRefusal) as raised:
        raise_if_content_filtered(response)
    assert raised.value.detail == wording


def test_an_unfiltered_reply_passes():
    assert is_content_filtered(reply("fine")) is False
    raise_if_content_filtered(reply("fine"))


# ---- I3: the streaming workflow llm step ------------------------------------------------------------------------


async def _llm_stream():
    from tldw_Server_API.app.core.Workflows.adapters.llm.llm import run_llm_adapter

    return await run_llm_adapter({"provider": PROVIDER, "model": "m", "prompt": "hello", "stream": True}, CONTEXT)


async def test_streaming_llm_step_output_block_fails_the_step(moderation_hooks, stub_llm_registry):
    stub_llm_registry.items = [openai_frame("a forbidden answer"), finish_frame(), DONE]
    with pytest.raises(LLMPolicyRefusal) as raised:
        await _llm_stream()
    assert raised.value.detail == OUTPUT_REFUSAL_TEXT


async def test_streaming_llm_step_providers_own_filter_is_worded_as_theirs(make_service, stub_llm_registry):
    make_service(enabled=False)
    stub_llm_registry.items = [openai_frame("partial"), finish_frame("content_filter"), DONE]
    with pytest.raises(LLMPolicyRefusal) as raised:
        await _llm_stream()
    assert raised.value.detail == PROVIDER_FILTER_REFUSAL_TEXT


# ---- I2: dialogue workflows -------------------------------------------------------------------------------------


def test_dialogue_input_block_is_a_refusal_not_a_provider_error(moderation_hooks, stub_llm_registry):
    from tldw_Server_API.app.core.Chat_Workflows.dialogue_orchestrator import ChatWorkflowDialogueOrchestrator

    orchestrator = ChatWorkflowDialogueOrchestrator()
    with pytest.raises(LLMPolicyRefusal) as raised:
        asyncio.run(
            orchestrator._call_text_model(
                selection={"provider": PROVIDER, "model": "m"}, system_prompt="s", prompt="forbidden", user_identifier="u"
            )
        )
    assert raised.value.status_code == 400 and raised.value.detail == "Input violates moderation policy"
    assert stub_llm_registry.requests == []


def test_dialogue_moderator_output_block_is_a_refusal_not_invalid_json():
    from tldw_Server_API.app.core.Chat_Workflows.dialogue_orchestrator import ChatWorkflowDialogueOrchestrator

    async def blocked(**_kwargs):
        return OURS

    orchestrator = ChatWorkflowDialogueOrchestrator(llm_caller=blocked)
    with pytest.raises(LLMPolicyRefusal):
        asyncio.run(
            orchestrator._call_json_model(
                selection={"provider": PROVIDER, "model": "m"}, system_prompt="s", prompt="p", user_identifier="u"
            )
        )


def test_chat_api_call_sync_reraises_a_refusal_unchanged(monkeypatch):
    from tldw_Server_API.app.core.Chat import chat_orchestrator

    refusal = LLMPolicyRefusal("Input violates moderation policy")

    def boom(**_kwargs):
        raise refusal

    monkeypatch.setattr(chat_orchestrator, "perform_chat_api_call", boom, raising=False)
    with pytest.raises(LLMPolicyRefusal) as raised:
        chat_orchestrator.chat_api_call(api_endpoint="openai", messages_payload=[{"role": "user", "content": "x"}])
    assert raised.value is refusal


# ---- I5: the first-chat check is a moderated surface ------------------------------------------------------------


def test_first_chat_is_moderated_and_the_old_exempt_id_is_gone():
    from tldw_Server_API.app.core.Moderation.surfaces import EXEMPT_SURFACES, MODERATED_SURFACES

    assert "setup.first_chat" in MODERATED_SURFACES
    assert "internal.first_chat_verifier" not in EXEMPT_SURFACES


async def test_first_chat_refusal_is_raised_not_reported_as_ready(monkeypatch):
    from tldw_Server_API.app.core.Setup import first_chat_verifier

    async def refused(**_kwargs):
        return OURS

    monkeypatch.setattr(first_chat_verifier, "perform_chat_api_call_async", refused)
    with pytest.raises(LLMPolicyRefusal):
        await first_chat_verifier.verify_first_chat(provider="openai", model="m", prompt="anything")


async def test_first_chat_input_block_is_raised(moderation_hooks, stub_llm_registry):
    from tldw_Server_API.app.core.Setup import first_chat_verifier

    with pytest.raises(LLMPolicyRefusal) as raised:
        await first_chat_verifier.verify_first_chat(provider=PROVIDER, model="m", prompt="forbidden words")
    assert raised.value.status_code == 400
    assert stub_llm_registry.requests == []
