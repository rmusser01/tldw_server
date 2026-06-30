from collections.abc import Mapping
from typing import Any

import pytest

from tldw_Server_API.app.core.Chat.Chat_Deps import (
    ChatConfigurationError,
    ChatProviderError,
    ChatRateLimitError,
)
from tldw_Server_API.app.core.VN_Play import adapters as vn_play_adapters
from tldw_Server_API.app.core.VN_Play.adapters import (
    GenerationModerationAdapter,
    ScriptedVNGenerationAdapter,
    VNGenerationAdapterError,
    VNGenerationCallRequest,
)
from tldw_Server_API.app.core.VN_Play.generated_outputs import (
    VNGenerationOutputParseError,
    parse_vn_generation_output,
)


def test_narrative_dialogue_rejects_unknown_root_and_nested_fields() -> None:
    with pytest.raises(VNGenerationOutputParseError, match="invalid_generation_output"):
        parse_vn_generation_output(
            {
                "schema": "narrative_dialogue",
                "narrative": [{"text": "The door opens."}],
                "unexpected": True,
            },
            output_schema="narrative_dialogue",
        )

    with pytest.raises(VNGenerationOutputParseError, match="invalid_generation_output"):
        parse_vn_generation_output(
            {
                "schema": "narrative_dialogue",
                "narrative": [
                    {
                        "text": "The door opens.",
                        "metadata": {"model_owned": True},
                    }
                ],
            },
            output_schema="narrative_dialogue",
        )

    with pytest.raises(VNGenerationOutputParseError, match="invalid_generation_output"):
        parse_vn_generation_output(
            {
                "schema": "narrative_dialogue",
                "dialogue": [
                    {
                        "speaker": "Mira",
                        "text": "Someone was here.",
                        "next_label": "model-owned-control-flow",
                    }
                ],
            },
            output_schema="narrative_dialogue",
        )


def test_narrative_dialogue_requires_generated_text() -> None:
    with pytest.raises(VNGenerationOutputParseError, match="empty_narrative_dialogue"):
        parse_vn_generation_output(
            {"schema": "narrative_dialogue", "narrative": [], "dialogue": []},
            output_schema="narrative_dialogue",
        )


def test_choice_set_duplicate_choice_id_preserves_error_code() -> None:
    with pytest.raises(VNGenerationOutputParseError, match="duplicate_choice_id"):
        parse_vn_generation_output(
            {
                "schema": "choice_set",
                "choices": [
                    {"id": "ask_map", "text": "Ask about the map"},
                    {"id": "ask_map", "text": "Ask about it again"},
                ],
            },
            output_schema="choice_set",
        )


def test_attached_character_validation_hook_rejects_unknown_character_id() -> None:
    def is_attached(character_id: str) -> bool:
        return character_id == "character_mira"

    parsed = parse_vn_generation_output(
        {
            "schema": "narrative_dialogue",
            "dialogue": [
                {
                    "character_id": "character_mira",
                    "speaker": "Mira",
                    "text": "The hook accepts this attached character.",
                }
            ],
        },
        output_schema="narrative_dialogue",
        attached_character_validator=is_attached,
    )
    assert parsed.public_payload["dialogue"][0]["character_id"] == "character_mira"

    with pytest.raises(VNGenerationOutputParseError, match="character_not_attached"):
        parse_vn_generation_output(
            {
                "schema": "narrative_dialogue",
                "dialogue": [
                    {
                        "character_id": "character_unauthorized",
                        "text": "This should not attach.",
                    }
                ],
            },
            output_schema="narrative_dialogue",
            attached_character_validator=is_attached,
        )


def test_choice_set_validates_choice_ids_uniqueness_and_public_payload() -> None:
    parsed = parse_vn_generation_output(
        {
            "schema": "choice_set",
            "lead_in": "Mira watches your reaction.",
            "choices": [
                {"id": "ask_map", "text": "Ask about the map", "metadata": {"tone": "curious"}},
                {"id": "wait-quietly", "text": "Wait quietly"},
            ],
        },
        output_schema="choice_set",
    )

    assert parsed.public_payload == {
        "schema": "choice_set",
        "lead_in": "Mira watches your reaction.",
        "choices": [
            {"id": "ask_map", "text": "Ask about the map", "metadata": {"tone": "curious"}},
            {"id": "wait-quietly", "text": "Wait quietly"},
        ],
    }

    for payload in (
        {
            "schema": "choice_set",
            "choices": [{"id": "bad id", "text": "Invalid"}],
        },
        {
            "schema": "choice_set",
            "choices": [{"id": "allowed", "text": "No injected target", "source": "generated"}],
        },
        {
            "schema": "choice_set",
            "choices": [{"id": "allowed", "text": "No injected target", "target": "model_label"}],
        },
        {
            "schema": "choice_set",
            "choices": [{"id": "allowed", "text": "No injected target", "next_label": "model_label"}],
        },
    ):
        with pytest.raises(VNGenerationOutputParseError, match="invalid_generation_output"):
            parse_vn_generation_output(payload, output_schema="choice_set")


def test_metadata_and_visual_labels_are_capped() -> None:
    oversized = "x" * 4097
    with pytest.raises(VNGenerationOutputParseError, match="metadata_too_large"):
        parse_vn_generation_output(
            {
                "schema": "choice_set",
                "choices": [
                    {"id": "ask", "text": "Ask", "metadata": {"blob": oversized}},
                ],
            },
            output_schema="choice_set",
        )

    with pytest.raises(VNGenerationOutputParseError, match="visual_labels_too_large"):
        parse_vn_generation_output(
            {
                "schema": "scene_update",
                "narrative": [{"text": "The room changes."}],
                "visual_directives": [
                    {
                        "asset_type": "background",
                        "slot_key": "archive_night",
                        "labels": {"blob": oversized},
                    }
                ],
            },
            output_schema="scene_update",
        )


def test_scene_update_rejects_invalid_visual_directive_shapes() -> None:
    with pytest.raises(VNGenerationOutputParseError, match="invalid_generation_output"):
        parse_vn_generation_output(
            {
                "schema": "scene_update",
                "narrative": [{"text": "The room changes."}],
                "visual_directives": [
                    {
                        "asset_type": "background",
                        "directive_type": "set_location",
                        "labels": {"location": "archive"},
                    }
                ],
            },
            output_schema="scene_update",
        )


@pytest.mark.asyncio
async def test_generation_adapter_uses_pinned_snapshot_and_preserves_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    async def fake_chat_call(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {
            "choices": [{"message": {"content": '{"schema":"narrative_dialogue","narrative":[{"text":"A clue appears."}]}'}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
            "id": "chatcmpl-vn",
        }

    monkeypatch.setattr(vn_play_adapters, "perform_chat_api_call_async", fake_chat_call)

    adapter = ScriptedVNGenerationAdapter()
    result = await adapter.generate(
        VNGenerationCallRequest(
            profile_snapshot={
                "id": 44,
                "definition": {
                    "provider": "openai",
                    "model": "gpt-4.1-mini",
                    "max_output_tokens": 321,
                    "temperature": 0.2,
                },
            },
            messages=[{"role": "user", "content": "continue"}],
            output_schema="narrative_dialogue",
            usage_context={
                "vn_session_id": 10,
                "script_id": 20,
                "script_version_id": 30,
                "generation_id": 40,
                "generation_request_id": 50,
                "generation_revision_id": 60,
                "generation_profile_key": "default",
                "generation_profile_snapshot_id": 44,
                "generation_point_key": "intro:1:beat",
            },
        )
    )

    assert result.raw_content == '{"schema":"narrative_dialogue","narrative":[{"text":"A clue appears."}]}'
    assert result.usage_metadata == {
        "prompt_tokens": 11,
        "completion_tokens": 7,
        "total_tokens": 18,
    }
    assert calls[0]["provider"] == "openai"
    assert calls[0]["api_endpoint"] == "openai"
    assert calls[0]["model"] == "gpt-4.1-mini"
    assert calls[0]["max_tokens"] == 321
    assert calls[0]["temp"] == 0.2
    assert calls[0]["vn_session_id"] == 10
    assert calls[0]["generation_point_key"] == "intro:1:beat"


@pytest.mark.asyncio
async def test_generation_adapter_maps_provider_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    async def rate_limited(**kwargs: Any) -> None:
        raise ChatRateLimitError(provider="openai")

    monkeypatch.setattr(vn_play_adapters, "perform_chat_api_call_async", rate_limited)

    adapter = ScriptedVNGenerationAdapter()
    with pytest.raises(VNGenerationAdapterError) as exc_info:
        await adapter.generate(_request_for_adapter())
    assert exc_info.value.public_error_code == "provider_unavailable"
    assert exc_info.value.debug_metadata["provider"] == "openai"

    async def timed_out(**kwargs: Any) -> None:
        raise TimeoutError("provider timed out")

    monkeypatch.setattr(vn_play_adapters, "perform_chat_api_call_async", timed_out)
    with pytest.raises(VNGenerationAdapterError) as timeout_info:
        await adapter.generate(_request_for_adapter())
    assert timeout_info.value.public_error_code == "model_timeout"

    async def missing_provider(**kwargs: Any) -> None:
        raise ChatConfigurationError(provider="openai")

    monkeypatch.setattr(vn_play_adapters, "perform_chat_api_call_async", missing_provider)
    with pytest.raises(VNGenerationAdapterError) as unavailable_info:
        await adapter.generate(_request_for_adapter())
    assert unavailable_info.value.public_error_code == "provider_unavailable"

    async def malformed_response(**kwargs: Any) -> dict[str, Any]:
        return {"choices": [{"message": {}}]}

    monkeypatch.setattr(vn_play_adapters, "perform_chat_api_call_async", malformed_response)
    with pytest.raises(VNGenerationAdapterError) as model_error_info:
        await adapter.generate(_request_for_adapter())
    assert model_error_info.value.public_error_code == "model_error"

    async def gateway_timeout(**kwargs: Any) -> None:
        raise ChatProviderError(status_code=504)

    monkeypatch.setattr(vn_play_adapters, "perform_chat_api_call_async", gateway_timeout)
    with pytest.raises(VNGenerationAdapterError) as timeout_status_info:
        await adapter.generate(_request_for_adapter())
    assert timeout_status_info.value.public_error_code == "model_timeout"


@pytest.mark.asyncio
async def test_moderation_adapter_fails_closed_for_public_profiles() -> None:
    class BlockingModeration:
        async def moderate(self, text: str, *, context: Mapping[str, Any]) -> Mapping[str, Any]:
            return {"allowed": False, "public_error_code": "moderation_blocked", "reason": "unsafe"}

    adapter = GenerationModerationAdapter(moderation_service=BlockingModeration())
    result = await adapter.moderate_output(
        "unsafe text",
        profile_snapshot={"definition": {"hosting": "hosted", "moderation": {"required": True}}},
        context={"generation_request_id": 50},
    )

    assert result.allowed is False
    assert result.status == "blocked"
    assert result.public_error_code == "moderation_blocked"


@pytest.mark.asyncio
async def test_moderation_adapter_does_not_allow_public_profile_opt_out() -> None:
    adapter = GenerationModerationAdapter(moderation_service=None)
    result = await adapter.moderate_output(
        "public text",
        profile_snapshot={"definition": {"provider_class": "Hosted", "moderation": {"required": False}}},
        context={"generation_request_id": 50},
    )

    assert result.allowed is False
    assert result.status == "failed"
    assert result.public_error_code == "moderation_unavailable"

    explicit_required = await adapter.moderate_output(
        "public text",
        profile_snapshot={"definition": {"moderation_required": True}},
        context={"generation_request_id": 50},
    )
    assert explicit_required.allowed is False
    assert explicit_required.public_error_code == "moderation_unavailable"


@pytest.mark.asyncio
async def test_moderation_adapter_rejects_malformed_allow_decisions() -> None:
    class MalformedModeration:
        async def moderate(self, text: str, *, context: Mapping[str, Any]) -> Mapping[str, Any]:
            return {"allowed": "false"}

    adapter = GenerationModerationAdapter(moderation_service=MalformedModeration())
    result = await adapter.moderate_output(
        "text",
        profile_snapshot={"definition": {"moderation_required": True}},
        context={"generation_request_id": 50},
    )

    assert result.allowed is False
    assert result.status == "failed"
    assert result.public_error_code == "moderation_unavailable"


@pytest.mark.asyncio
async def test_moderation_adapter_records_local_policy_opt_out() -> None:
    class ExplodingModeration:
        async def moderate(self, text: str, *, context: Mapping[str, Any]) -> Mapping[str, Any]:
            raise AssertionError("local opt-out should not call moderation")

    adapter = GenerationModerationAdapter(moderation_service=ExplodingModeration())
    result = await adapter.moderate_output(
        "local text",
        profile_snapshot={"definition": {"hosting": "local", "moderation": {"required": False}}},
        context={"generation_request_id": 50},
    )

    assert result.allowed is True
    assert result.status == "skipped"
    assert result.audit_metadata == {"moderation_skipped_by_policy": True}


def _request_for_adapter() -> VNGenerationCallRequest:
    return VNGenerationCallRequest(
        profile_snapshot={
            "id": 44,
            "definition": {
                "provider": "openai",
                "model": "gpt-4.1-mini",
                "max_output_tokens": 321,
                "temperature": 0.2,
            },
        },
        messages=[{"role": "user", "content": "continue"}],
        output_schema="narrative_dialogue",
        usage_context={"generation_profile_snapshot_id": 44},
    )
