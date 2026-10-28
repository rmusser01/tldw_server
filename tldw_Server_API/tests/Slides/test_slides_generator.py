import json

import pytest

from tldw_Server_API.app.core.Slides.slides_generator import (
    SlidesGenerator,
    SlidesGenerationInputError,
    SlidesGenerationOutputError,
    SlidesSourceTooLargeError,
    _positive_int_setting,
)


@pytest.fixture(autouse=True)
def empty_provider_policy(monkeypatch):
    # Unit LLM stubs need an explicit healthy policy dependency, not a real AuthNZ store.
    from tldw_Server_API.app.core.AuthNZ import llm_provider_overrides as policy

    monkeypatch.setattr(policy, "_get_healthy_override_snapshot", lambda _provider=None: {})


def _stub_llm_call(api_provider=None, messages=None, **kwargs):
    user_content = ""
    if messages:
        user_content = messages[-1].get("content", "")
    if "Summarize" in user_content:
        return {"choices": [{"message": {"content": "Summary chunk"}}]}
    payload = {
        "title": "Deck",
        "slides": [
            {"layout": "title", "title": "Deck", "content": "", "order": 0},
            {"layout": "content", "title": "Slide", "content": "- A\n- B", "order": 1},
        ],
    }
    return {"choices": [{"message": {"content": json.dumps(payload)}}]}


def _make_prompt_capture_llm(payload: dict[str, object], captured: dict[str, str]):
    def _llm_call(api_provider=None, messages=None, **kwargs):
        del api_provider, kwargs
        if messages:
            captured["system_prompt"] = str(messages[0].get("content", ""))
            captured["user_prompt"] = str(messages[-1].get("content", ""))
        return {"choices": [{"message": {"content": json.dumps(payload)}}]}

    return _llm_call


def test_generate_rejects_large_input_without_chunking():
    generator = SlidesGenerator(llm_call=_stub_llm_call)
    with pytest.raises(SlidesSourceTooLargeError):
        generator.generate_from_text(
            source_text="x" * 50,
            title_hint=None,
            provider="openai",
            model=None,
            api_key=None,
            temperature=None,
            max_tokens=None,
            max_source_tokens=None,
            max_source_chars=10,
            enable_chunking=False,
            chunk_size_tokens=None,
            summary_tokens=None,
        )


def test_generate_rejects_excessive_chunk_fanout(monkeypatch):
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Slides.slides_generator.is_test_mode",
        lambda: False,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Slides.slides_generator.MAX_SOURCE_CHUNKS",
        2,
    )
    generator = SlidesGenerator(llm_call=_stub_llm_call)

    with pytest.raises(SlidesSourceTooLargeError):
        generator.generate_from_text(
            source_text=" ".join(f"word{i}" for i in range(9)),
            title_hint=None,
            provider="openai",
            model=None,
            api_key=None,
            temperature=None,
            max_tokens=None,
            max_source_tokens=None,
            max_source_chars=None,
            enable_chunking=True,
            chunk_size_tokens=3,
            summary_tokens=None,
        )


def test_generate_rejects_non_positive_chunk_size_before_llm(monkeypatch):
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Slides.slides_generator.is_test_mode",
        lambda: False,
    )

    def _unexpected_llm_call(*args, **kwargs):
        raise AssertionError("LLM should not be called for invalid chunk size")

    generator = SlidesGenerator(llm_call=_unexpected_llm_call)

    with pytest.raises(SlidesGenerationInputError, match="chunk_size_invalid"):
        generator.generate_from_text(
            source_text=" ".join(f"word{i}" for i in range(20)),
            title_hint=None,
            provider="openai",
            model=None,
            api_key=None,
            temperature=None,
            max_tokens=None,
            max_source_tokens=None,
            max_source_chars=None,
            enable_chunking=True,
            chunk_size_tokens=-1,
            summary_tokens=None,
        )


def test_positive_int_setting_logs_invalid_config_and_reraises_unexpected_errors(monkeypatch):
    import tldw_Server_API.app.core.Slides.slides_generator as slides_generator

    logged: list[tuple[str, tuple[object, ...]]] = []
    monkeypatch.setattr(
        slides_generator,
        "settings",
        {"SLIDES_MAX_SOURCE_CHUNKS": "not-an-int"},
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Slides.slides_generator.logger.warning",
        lambda message, *args: logged.append((message, args)),
    )

    assert _positive_int_setting("SLIDES_MAX_SOURCE_CHUNKS", 20) == 20
    assert logged
    assert logged[0][1][0] == "SLIDES_MAX_SOURCE_CHUNKS"

    class BrokenSettings:
        def get(self, name, default=None):
            raise RuntimeError("settings backend failed")

    monkeypatch.setattr(slides_generator, "settings", BrokenSettings())

    with pytest.raises(RuntimeError, match="settings backend failed"):
        _positive_int_setting("SLIDES_MAX_SOURCE_CHUNKS", 20)


def test_chunk_char_approx_logs_invalid_divisor_setting(monkeypatch):
    import tldw_Server_API.app.core.Slides.slides_generator as slides_generator

    logged: list[tuple[str, tuple[object, ...]]] = []
    monkeypatch.setattr(
        slides_generator,
        "settings",
        {
            "TOKEN_ESTIMATOR_MODE": "char_approx",
            "TOKEN_CHAR_APPROX_DIVISOR": "bad",
            "SLIDES_MAX_SOURCE_CHUNKS": 10,
        },
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Slides.slides_generator.logger.warning",
        lambda message, *args: logged.append((message, args)),
    )
    generator = SlidesGenerator(llm_call=_stub_llm_call)

    summary = generator._chunk_and_summarize(
        source_text="abcdef",
        provider="openai",
        model=None,
        api_key=None,
        temperature=None,
        chunk_size_tokens=3,
        summary_tokens=None,
    )

    assert summary == "Summary chunk"
    assert logged
    assert logged[0][1][0] == "TOKEN_CHAR_APPROX_DIVISOR"


def test_generate_parses_json_response(monkeypatch):
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Slides.slides_generator.is_test_mode",
        lambda: False,
    )
    generator = SlidesGenerator(llm_call=_stub_llm_call)
    result = generator.generate_from_text(
        source_text="Content",
        title_hint="Hint",
        provider="openai",
        model=None,
        api_key=None,
        temperature=None,
        max_tokens=None,
        max_source_tokens=None,
        max_source_chars=None,
        enable_chunking=False,
        chunk_size_tokens=None,
        summary_tokens=None,
    )
    assert result["title"] == "Deck"
    assert len(result["slides"]) == 2


def test_generate_handles_invalid_json(monkeypatch):
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Slides.slides_generator.is_test_mode",
        lambda: False,
    )
    def bad_llm_call(api_provider=None, messages=None, **kwargs):
        return {"choices": [{"message": {"content": "not json"}}]}

    generator = SlidesGenerator(llm_call=bad_llm_call)
    with pytest.raises(SlidesGenerationOutputError):
        generator.generate_from_text(
            source_text="Content",
            title_hint=None,
            provider="openai",
            model=None,
            api_key=None,
            temperature=None,
            max_tokens=None,
            max_source_tokens=None,
            max_source_chars=None,
            enable_chunking=False,
            chunk_size_tokens=None,
            summary_tokens=None,
        )


def test_generate_rejects_placeholder_slide_content(monkeypatch):
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Slides.slides_generator.is_test_mode",
        lambda: False,
    )

    payload = {
        "title": "Deck",
        "slides": [
            {
                "layout": "content",
                "title": "Invalid",
                "content": "slides go here",
                "order": 0,
                "speaker_notes": "invalid",
            }
        ],
    }

    def placeholder_llm_call(**_kwargs):
        return {"choices": [{"message": {"content": json.dumps(payload)}}]}

    generator = SlidesGenerator(llm_call=placeholder_llm_call)
    with pytest.raises(SlidesGenerationOutputError, match="slide_content_missing"):
        generator.generate_from_text(
            source_text="Content",
            title_hint=None,
            provider="openai",
            model=None,
            api_key=None,
            temperature=None,
            max_tokens=None,
            max_source_tokens=None,
            max_source_chars=None,
            enable_chunking=False,
            chunk_size_tokens=None,
            summary_tokens=None,
        )


def test_timeline_style_generates_visual_block_and_text_fallback(monkeypatch):
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Slides.slides_generator.is_test_mode",
        lambda: False,
    )
    captured: dict[str, str] = {}

    def timeline_llm_call(
        api_provider: str | None = None,
        messages: list[dict[str, object]] | None = None,
        **_kwargs: object,
    ) -> dict[str, object]:
        del api_provider
        user_content = ""
        if messages:
            captured["system_prompt"] = str(messages[0].get("content", ""))
            user_content = str(messages[-1].get("content", ""))
        if "Summarize" in user_content:
            return {"choices": [{"message": {"content": "Summary chunk"}}]}
        payload = {
            "title": "History Deck",
            "slides": [
                {"layout": "title", "title": "History Deck", "content": "", "order": 0},
                {
                    "layout": "content",
                    "title": "Key Events",
                    "content": "",
                    "order": 1,
                    "metadata": {
                        "visual_blocks": [
                            {
                                "type": "timeline",
                                "items": [
                                    {
                                        "label": "1776",
                                        "title": "Declaration",
                                        "description": "American independence declared",
                                    },
                                    {
                                        "label": "1947",
                                        "title": "Independence",
                                        "description": "India becomes independent",
                                    },
                                ],
                            }
                        ]
                    },
                },
            ],
        }
        return {"choices": [{"message": {"content": json.dumps(payload)}}]}

    generator = SlidesGenerator(llm_call=timeline_llm_call)
    result = generator.generate_from_text(
        source_text="1776, 1947",
        title_hint="History",
        provider="openai",
        model=None,
        api_key=None,
        temperature=None,
        max_tokens=None,
        max_source_tokens=None,
        max_source_chars=None,
        enable_chunking=False,
        chunk_size_tokens=None,
        summary_tokens=None,
        visual_style_snapshot={
            "id": "timeline",
            "name": "Timeline",
            "scope": "builtin",
        },
    )

    slide = result["slides"][1]
    assert (
        "Style description: Chronology-first slides focused on sequence, causality, and milestones."
        in captured["system_prompt"]
    )
    assert "Favor chronology, causality, and milestone sequencing." in captured["system_prompt"]
    assert slide["metadata"]["visual_blocks"][0]["type"] == "timeline"
    assert "1776" in slide["content"]
    assert "1947" in slide["content"]


def test_builtin_blueprint_prompt_uses_profile_guidance_and_artifact_preferences(
    monkeypatch,
):
    captured: dict[str, str] = {}
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Slides.slides_generator.is_test_mode",
        lambda: False,
    )
    generator = SlidesGenerator(
        llm_call=_make_prompt_capture_llm(
            {
                "title": "Blueprint Deck",
                "slides": [
                    {"layout": "title", "title": "Blueprint Deck", "content": "", "order": 0},
                ],
            },
            captured,
        )
    )

    generator.generate_from_text(
        source_text="Blueprint source",
        title_hint="Blueprint",
        provider="openai",
        model=None,
        api_key=None,
        temperature=None,
        max_tokens=None,
        max_source_tokens=None,
        max_source_chars=None,
        enable_chunking=False,
        chunk_size_tokens=None,
        summary_tokens=None,
        visual_style_snapshot={
            "id": "notebooklm-blueprint",
            "scope": "builtin",
        },
    )

    system_prompt = captured["system_prompt"]
    assert "Visual style preset: Blueprint." in system_prompt
    assert "Cyan grid blueprint treatment with technical linework." in system_prompt
    assert "Prompt profile: Technical Precision." in system_prompt
    assert "prefer exact sequencing and component naming" in system_prompt
    assert "Preferred visual block types: process_flow, timeline" in system_prompt
    assert "Fallback instructions:" in system_prompt


def test_generation_prompt_forbids_unsupported_slide_claims(monkeypatch):
    captured: dict[str, str] = {}
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Slides.slides_generator.is_test_mode",
        lambda: False,
    )
    generator = SlidesGenerator(
        llm_call=_make_prompt_capture_llm(
            {
                "title": "Grounded Deck",
                "slides": [
                    {"layout": "title", "title": "Grounded Deck", "content": "", "order": 0},
                ],
            },
            captured,
        )
    )

    generator.generate_from_text(
        source_text="Trial started March 3. Treatment basil gained 18%. Control gained 7%.",
        title_hint="Grounded Deck",
        provider="openai",
        model=None,
        api_key=None,
        temperature=None,
        max_tokens=None,
        max_source_tokens=None,
        max_source_chars=None,
        enable_chunking=False,
        chunk_size_tokens=None,
        summary_tokens=None,
    )

    system_prompt = captured["system_prompt"]
    assert "Use only facts explicitly supported by the source material." in system_prompt
    assert "Do not invent recommendations, next steps, objectives, methods, trial status" in system_prompt
    assert "Speaker notes must stay source-grounded" in system_prompt
    assert "Short sources may produce 2-5 slides" in system_prompt
    assert "Omit conclusion or summary slides for short sources unless the source states conclusions" in system_prompt


def test_timeline_style_prompts_chronology_first_behavior(monkeypatch):
    captured: dict[str, str] = {}
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Slides.slides_generator.is_test_mode",
        lambda: False,
    )
    generator = SlidesGenerator(
        llm_call=_make_prompt_capture_llm(
            {
                "title": "Timeline Deck",
                "slides": [
                    {"layout": "title", "title": "Timeline Deck", "content": "", "order": 0},
                ],
            },
            captured,
        )
    )

    generator.generate_from_text(
        source_text="1776, 1947",
        title_hint="History",
        provider="openai",
        model=None,
        api_key=None,
        temperature=None,
        max_tokens=None,
        max_source_tokens=None,
        max_source_chars=None,
        enable_chunking=False,
        chunk_size_tokens=None,
        summary_tokens=None,
        visual_style_snapshot={
            "id": "timeline",
            "scope": "builtin",
        },
    )

    system_prompt = captured["system_prompt"]
    assert "Visual style preset: Timeline." in system_prompt
    assert "Chronology-first slides focused on sequence, causality, and milestones." in system_prompt
    assert "Favor chronology, causality, and milestone sequencing." in system_prompt
    assert '"chronology_bias": "high"' in system_prompt
    assert "Preferred visual block types: timeline, stat_group" in system_prompt


def test_user_style_snapshot_uses_raw_generation_rules_and_artifact_preferences(
    monkeypatch,
):
    captured: dict[str, str] = {}
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Slides.slides_generator.is_test_mode",
        lambda: False,
    )
    generator = SlidesGenerator(
        llm_call=_make_prompt_capture_llm(
            {
                "title": "Custom Deck",
                "slides": [
                    {"layout": "title", "title": "Custom Deck", "content": "", "order": 0},
                ],
            },
            captured,
        )
    )

    generator.generate_from_text(
        source_text="Custom source",
        title_hint="Custom",
        provider="openai",
        model=None,
        api_key=None,
        temperature=None,
        max_tokens=None,
        max_source_tokens=None,
        max_source_chars=None,
        enable_chunking=False,
        chunk_size_tokens=None,
        summary_tokens=None,
        visual_style_snapshot={
            "id": "notebooklm-blueprint",
            "scope": "user",
            "name": "User Blueprint",
            "description": "User supplied blueprint style",
            "generation_rules": {"custom_bias": "high"},
            "artifact_preferences": ["stat_group"],
            "fallback_policy": {"mode": "custom-outline", "preserve_key_stats": False},
        },
    )

    system_prompt = captured["system_prompt"]
    assert "User supplied blueprint style" in system_prompt
    assert '"custom_bias": "high"' in system_prompt
    assert "Preferred visual block types: stat_group" in system_prompt
    assert "custom-outline" in system_prompt
    assert "Cyan grid blueprint treatment with technical linework." not in system_prompt


@pytest.mark.parametrize(
    "override,model,expected",
    [
        ({"is_enabled": False}, "qwen", "provider_disabled"),
        ({"allowed_models": ["allowed"]}, "blocked", "model_not_allowed"),
        ({"allowed_models": ["allowed"]}, None, "model_not_allowed"),
    ],
)
def test_slides_provider_policy_refuses_before_llm(monkeypatch, override, model, expected):
    from unittest.mock import Mock

    from tldw_Server_API.app.core.AuthNZ import llm_provider_overrides as policy
    from tldw_Server_API.app.core.LLM_Calls import adapter_utils

    monkeypatch.setattr(adapter_utils, "ensure_app_config", lambda: {})
    monkeypatch.setattr(adapter_utils, "resolve_provider_model", lambda _provider, _config: None)
    snapshot = policy.ProviderOverrideCallSnapshot(
        provider="llama.cpp", _override=policy.LLMProviderOverride(provider="llama.cpp", **override)
    )
    monkeypatch.setattr(policy, "capture_provider_override_call_snapshot", lambda _provider: snapshot)
    llm = Mock(return_value=_stub_llm_call())
    generator = SlidesGenerator(llm_call=llm)
    with pytest.raises(SlidesGenerationInputError, match="^" + expected + "$"):
        generator._call_llm(
            provider="llamacpp",
            model=model,
            api_key="explicit-test-key",
            temperature=None,
            max_tokens=256,
            system_prompt="Source only",
            user_prompt="Rowan",
        )
    llm.assert_not_called()


def test_slides_override_store_unavailable_refuses_before_llm(monkeypatch):
    from unittest.mock import Mock

    from tldw_Server_API.app.core.AuthNZ import llm_provider_overrides as policy
    from tldw_Server_API.app.core.Slides import slides_generator

    def unavailable(_provider):
        raise policy.LLMProviderOverridesRefreshError()

    monkeypatch.setattr(policy, "capture_provider_override_call_snapshot", unavailable)
    llm = Mock(return_value=_stub_llm_call())
    with pytest.raises(slides_generator.SlidesGenerationError, match="^llm_call_failed$"):
        SlidesGenerator(llm_call=llm)._call_llm(
            provider="llama.cpp",
            model="qwen",
            api_key=None,
            temperature=None,
            max_tokens=256,
            system_prompt="Source only",
            user_prompt="Rowan",
        )
    llm.assert_not_called()


@pytest.mark.parametrize("explicit_model,configured_model", [("allowed", "other"), (None, "allowed")])
def test_slides_dispatches_the_same_model_checked_by_policy(monkeypatch, explicit_model, configured_model):
    from unittest.mock import Mock

    from tldw_Server_API.app.core.AuthNZ import llm_provider_overrides as policy
    from tldw_Server_API.app.core.LLM_Calls import adapter_utils

    snapshot = policy.ProviderOverrideCallSnapshot(
        provider="llama.cpp", _override=policy.LLMProviderOverride(provider="llama.cpp", allowed_models=["allowed"])
    )
    monkeypatch.setattr(policy, "capture_provider_override_call_snapshot", lambda _provider: snapshot)
    monkeypatch.setattr(adapter_utils, "ensure_app_config", lambda: {"llama_api": {"model": configured_model}})
    llm = Mock(return_value=_stub_llm_call())
    SlidesGenerator(llm_call=llm)._call_llm(
        provider="llamacpp",
        model=explicit_model,
        api_key=None,
        temperature=None,
        max_tokens=256,
        system_prompt="Source only",
        user_prompt="Rowan",
    )
    assert llm.call_args.kwargs["model"] == "allowed"  # nosec B101 # Pytest checks the exact policy-bound outbound model.


def test_slides_rechecks_policy_between_summary_chunks(monkeypatch):
    from unittest.mock import Mock

    from tldw_Server_API.app.core.AuthNZ import llm_provider_overrides as policy
    from tldw_Server_API.app.core.Slides import slides_generator

    monkeypatch.setattr(slides_generator, "is_test_mode", lambda: False)
    monkeypatch.setitem(slides_generator.settings, "TOKEN_ESTIMATOR_MODE", "whitespace")
    allowed = policy.ProviderOverrideCallSnapshot(provider="llama.cpp", _override=None)
    denied = policy.ProviderOverrideCallSnapshot(
        provider="llama.cpp", _override=policy.LLMProviderOverride(provider="llama.cpp", is_enabled=False)
    )
    monkeypatch.setattr(policy, "capture_provider_override_call_snapshot", Mock(side_effect=[allowed, denied]))
    llm = Mock(return_value={"choices": [{"message": {"content": "Controlled summary"}}]})
    with pytest.raises(SlidesGenerationInputError, match="^provider_disabled$"):
        SlidesGenerator(llm_call=llm).generate_from_text(
            source_text="one two three four",
            title_hint=None,
            provider="llamacpp",
            model="allowed",
            api_key=None,
            temperature=None,
            max_tokens=256,
            max_source_tokens=None,
            max_source_chars=None,
            enable_chunking=True,
            chunk_size_tokens=2,
            summary_tokens=32,
        )
    llm.assert_called_once()


@pytest.mark.parametrize("provider", ["llama.cpp", "llamacpp", "llama"])
def test_slides_disabled_provider_alias_uses_canonical_policy(monkeypatch, provider):
    from unittest.mock import Mock

    from tldw_Server_API.app.core.AuthNZ import llm_provider_overrides as policy

    def snapshot(canonical):
        assert canonical == "llama.cpp"  # nosec B101 # Pytest checks alias lookup uses the canonical policy key.
        return {canonical: policy.LLMProviderOverride(provider=canonical, is_enabled=False)}

    monkeypatch.setattr(policy, "_get_healthy_override_snapshot", snapshot)
    llm = Mock(return_value=_stub_llm_call())
    with pytest.raises(SlidesGenerationInputError, match="^provider_disabled$"):
        SlidesGenerator(llm_call=llm)._call_llm(
            provider=provider,
            model="qwen",
            api_key=None,
            temperature=None,
            max_tokens=256,
            system_prompt="Source only",
            user_prompt="Rowan",
        )
    llm.assert_not_called()


def test_slides_refuses_disallowed_configured_default(monkeypatch):
    from unittest.mock import Mock

    from tldw_Server_API.app.core.AuthNZ import llm_provider_overrides as policy
    from tldw_Server_API.app.core.LLM_Calls import adapter_utils

    snapshot = policy.ProviderOverrideCallSnapshot(
        provider="llama.cpp", _override=policy.LLMProviderOverride(provider="llama.cpp", allowed_models=["allowed"])
    )
    monkeypatch.setattr(policy, "capture_provider_override_call_snapshot", lambda _provider: snapshot)
    monkeypatch.setattr(adapter_utils, "ensure_app_config", lambda: {"llama_api": {"model": "blocked"}})
    llm = Mock(return_value=_stub_llm_call())
    with pytest.raises(SlidesGenerationInputError, match="^model_not_allowed$"):
        SlidesGenerator(llm_call=llm)._call_llm(
            provider="llamacpp",
            model=None,
            api_key=None,
            temperature=None,
            max_tokens=256,
            system_prompt="Source only",
            user_prompt="Rowan",
        )
    llm.assert_not_called()
