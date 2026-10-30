"""Actual synthesis-provider request and completion-boundary regressions."""

from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.Research.providers.synthesis import SynthesisProvider

pytestmark = pytest.mark.unit


async def invoke(config, response, calls):
    async def chat(**kwargs):
        calls.append(kwargs)
        return response

    return await SynthesisProvider(chat_fn=chat).summarize(
        plan=SimpleNamespace(query="Controlled question", focus_areas=["background"]),
        source_registry=[],
        evidence_notes=[],
        collection_summary={},
        config=config,
    )


@pytest.mark.parametrize(("raw", "expected"), [("", None), ("false", False), ("yes", True)])
def test_operator_thinking_policy_is_resolved_and_not_overridden(monkeypatch, raw, expected):
    from tldw_Server_API.app.core.Research.providers import config

    monkeypatch.setattr(
        config,
        "get_config_value",
        lambda section, key, default=None: raw if key == "deep_research_llamacpp_enable_thinking" else default,
    )
    resolved = config.resolve_provider_config({"synthesis": {"llamacpp_enable_thinking": not expected}})
    assert resolved["synthesis"]["llamacpp_enable_thinking"] is expected  # nosec B101 # Controlled provider-boundary behavior assertion.


@pytest.mark.parametrize("raw", ["sometimes", "2"])
def test_invalid_operator_thinking_policy_fails_explicitly(monkeypatch, raw):
    from tldw_Server_API.app.core.Research.providers import config

    monkeypatch.setattr(
        config,
        "get_config_value",
        lambda section, key, default=None: raw if key == "deep_research_llamacpp_enable_thinking" else default,
    )
    with pytest.raises(ValueError, match="deep_research_llamacpp_enable_thinking"):
        config.resolve_provider_config(None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "thinking", "temperature", "expected_temperature", "extra"),
    [
        ("llama.cpp", False, 0, 0, {"chat_template_kwargs": {"enable_thinking": False}}),
        ("llamacpp", True, None, 0.2, {"chat_template_kwargs": {"enable_thinking": True}}),
        ("openai", False, 0, 0, None),
    ],
)
async def test_synthesis_preserves_zero_and_scopes_template_control(
    provider, thinking, temperature, expected_temperature, extra
):
    calls = []
    await invoke(
        {"provider": provider, "model": "controlled", "temperature": temperature, "llamacpp_enable_thinking": thinking},
        {"choices": [{"finish_reason": "stop", "message": {"content": "{}"}}]},
        calls,
    )
    assert (calls[0]["temperature"], calls[0].get("extra_body"), calls[0]["max_tokens"]) == (  # nosec B101 # Controlled provider-boundary behavior assertion.
        expected_temperature,
        extra,
        2500,
    )


@pytest.mark.asyncio
async def test_synthesis_default_leaves_template_policy_unset():
    calls = []
    await invoke(
        {"provider": "llama.cpp", "model": "controlled"},
        {"choices": [{"finish_reason": "stop", "message": {"content": "{}"}}]},
        calls,
    )
    assert (calls[0]["temperature"], "extra_body" in calls[0]) == (0.2, False)  # nosec B101 # Controlled provider-boundary behavior assertion.


@pytest.mark.asyncio
async def test_malformed_direct_thinking_policy_refuses_before_chat():
    calls = []
    with pytest.raises(ValueError, match="llamacpp_enable_thinking"):
        await invoke({"provider": "llama.cpp", "model": "controlled", "llamacpp_enable_thinking": "false"}, {}, calls)
    assert calls == []  # nosec B101 # Controlled provider-boundary behavior assertion.


@pytest.mark.asyncio
@pytest.mark.parametrize("content", [None, '{"outline_sections": [], "claims": [], "report_sections": []}'])
async def test_truncated_completion_is_not_a_structured_synthesis(content):
    with pytest.raises(ValueError, match="synthesis_output_truncated"):
        await invoke(
            {"provider": "llama.cpp", "model": "controlled"},
            {
                "choices": [
                    {"finish_reason": "length", "message": {"content": content, "reasoning_content": "controlled"}}
                ]
            },
            [],
        )


@pytest.mark.asyncio
async def test_complete_structured_content_remains_unchanged():
    content = '{"outline_sections": [], "claims": [], "report_sections": []}'
    result = await invoke(
        {"provider": "openai", "model": "controlled"},
        {"choices": [{"finish_reason": "stop", "message": {"content": content}}]},
        [],
    )
    assert result == content  # nosec B101 # Controlled provider-boundary behavior assertion.


@pytest.mark.asyncio
async def test_provider_prompt_states_numeric_confidence_contract():
    import re

    calls = []
    await invoke(
        {"provider": "llama.cpp", "model": "controlled"},
        {"choices": [{"finish_reason": "stop", "message": {"content": "{}"}}]},
        calls,
    )
    assert re.search(r"confidence[^\n]*number[^\n]*0[^\n]*1", calls[0]["messages"][0]["content"], re.I)  # nosec B101 # Controlled outgoing contract assertion.


def test_qualitative_confidence_is_not_coerced_into_a_probability():
    from tldw_Server_API.app.core.Research.synthesizer import ResearchSynthesizer

    with pytest.raises(ValueError, match="could not convert string to float"):
        ResearchSynthesizer()._build_provider_result(
            payload={
                "outline_sections": [],
                "claims": [{"text": "Controlled", "confidence": "high"}],
                "report_sections": [],
            },
            plan=SimpleNamespace(focus_areas=[]),
            source_registry=[],
            evidence_notes=[],
        )
