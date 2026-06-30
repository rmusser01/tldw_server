from __future__ import annotations

from tldw_Server_API.app.core.VN_Scripts.validator import (
    VNScriptValidationContext,
    validate_script_program,
)


def _context() -> VNScriptValidationContext:
    return VNScriptValidationContext(
        approved_slot_keys={
            "background.archive.default",
            "sprite.mira.neutral",
            "cg.opening",
        },
        audio_refs={
            "bgm.archive": {"mime_type": "audio/mpeg", "owner_user_id": 42},
            "voice.mira.hello": {"mime_type": "audio/ogg", "owner_user_id": 42},
            "image.not-audio": {"mime_type": "image/png", "owner_user_id": 42},
        },
        generation_profile={
            "profile_id": "story_default",
            "allowed_content_ratings": ["general", "teen"],
            "max_choices": 4,
            "max_model_expansion_scope": "scene",
            "supports_structured_output": True,
            "supported_output_schemas": ["narrative_dialogue", "scene_update"],
        },
        available_generation_profiles={
            "default": {
                "profile_id": "story_default",
                "allowed_content_ratings": ["general", "teen"],
                "max_choices": 4,
                "max_model_expansion_scope": "scene",
                "supports_structured_output": True,
                "supported_output_schemas": ["narrative_dialogue", "scene_update"],
            },
            "choice_writer": {
                "profile_id": "choice_profile",
                "allowed_content_ratings": ["general"],
                "max_choices": 3,
                "max_model_expansion_scope": "turn",
                "supports_structured_output": True,
                "supported_output_schemas": ["choice_set"],
            },
        },
        content_rating="general",
        owner_user_id=42,
    )


def _program() -> dict:
    return {
        "schema_version": "vn_script_program.v1",
        "title": "Archive Door",
        "primary_asset_pack_id": 7,
        "entry_label": "start",
        "variables": {
            "has_key": {"type": "boolean", "default": False, "public": True},
            "trust": {"type": "integer", "default": 0, "public": True},
            "name": {"type": "string", "default": "Mira", "public": False},
        },
        "generation_defaults": {"profile_id": "story_default", "persist_model_outputs": True},
        "labels": {
            "start": [
                {"op": "set_background", "slot_key": "background.archive.default"},
                {"op": "show_sprite", "slot_key": "sprite.mira.neutral"},
                {"op": "set", "var": "has_key", "value": True},
                {"op": "choice", "id": "door", "choices": [{"id": "open", "text": "Open it", "target": "open"}]},
            ],
            "open": [
                {"op": "play_bgm", "media_ref": "bgm.archive"},
                {"op": "generate", "scope": "scene", "max_choices": 2, "output_schema": "narrative_dialogue"},
                {"op": "end"},
            ],
        },
    }


def _error_codes(result) -> set[str]:
    return {entry["code"] for entry in result.errors}


def _warning_codes(result) -> set[str]:
    return {entry["code"] for entry in result.warnings}


def test_valid_script_program_passes() -> None:
    result = validate_script_program(_program(), _context())

    assert result.valid is True
    assert result.errors == []
    assert result.warnings == []


def test_canonical_schema_version_and_unknown_opcodes_are_errors() -> None:
    program = _program()
    program["schema_version"] = "vn_script_program.v2"
    program["labels"]["start"].append({"op": "teleport", "target": "open"})

    result = validate_script_program(program, _context())

    assert result.valid is False
    assert {"schema_version_invalid", "opcode_unknown"}.issubset(_error_codes(result))


def test_variable_declaration_shape_and_condition_ops_are_errors() -> None:
    program = _program()
    program["variables"]["has_key"]["type"] = "flag"
    program["variables"]["trust"]["default"] = "high"
    program["variables"]["name"]["public"] = "no"
    program["labels"]["start"].append({"op": "jump", "target": "open", "if": {"var": "trust", "op": "around", "value": 2}})

    result = validate_script_program(program, _context())

    assert result.valid is False
    assert {
        "variable_type_invalid",
        "variable_default_type_mismatch",
        "variable_public_invalid",
        "condition_operator_invalid",
    }.issubset(_error_codes(result))


def test_missing_entry_and_targets_are_errors() -> None:
    program = _program()
    program["entry_label"] = "missing_entry"
    program["labels"]["start"].append({"op": "jump", "target": "missing_target"})
    program["labels"]["start"][3]["choices"].append(
        {"id": "bad", "text": "Bad target", "target": "also_missing"}
    )

    result = validate_script_program(program, _context())

    assert result.valid is False
    assert {"entry_label_missing", "jump_target_missing", "choice_target_missing"}.issubset(_error_codes(result))


def test_typed_assignment_and_condition_operand_errors() -> None:
    program = _program()
    program["labels"]["start"].insert(2, {"op": "set", "var": "has_key", "value": "yes"})
    program["labels"]["start"].insert(3, {"op": "jump", "target": "open", "if": {"var": "trust", "op": "gte", "value": "high"}})

    result = validate_script_program(program, _context())

    assert result.valid is False
    assert {"variable_assignment_type_mismatch", "condition_operand_type_mismatch"}.issubset(_error_codes(result))


def test_visual_audio_and_generation_profile_restrictions_are_errors() -> None:
    program = _program()
    program["labels"]["start"][0]["slot_key"] = "background.missing"
    program["labels"]["open"][0]["media_ref"] = "image.not-audio"
    program["labels"]["open"][1]["scope"] = "session"
    program["labels"]["open"][1]["max_choices"] = 8

    result = validate_script_program(program, _context())

    assert result.valid is False
    assert {
        "visual_slot_key_missing",
        "audio_media_type_invalid",
        "generation_scope_not_allowed",
        "generation_max_choices_exceeded",
    }.issubset(_error_codes(result))


def test_generation_defaults_reject_raw_provider_routing() -> None:
    program = _program()
    program["generation_defaults"]["provider"] = "openai"
    program["generation_defaults"]["model"] = "gpt-5"

    result = validate_script_program(program, _context())

    assert result.valid is False
    assert "generation_raw_routing_not_allowed" in _error_codes(result)


def test_generate_opcode_rejects_raw_provider_routing() -> None:
    program = _program()
    program["labels"]["open"][1]["provider"] = "openai"
    program["labels"]["open"][1]["model"] = "gpt-5"

    result = validate_script_program(program, _context())

    assert result.valid is False
    assert "generation_raw_routing_not_allowed" in _error_codes(result)


def test_generate_opcode_rejects_unknown_or_invalid_profile_key() -> None:
    program = _program()
    program["labels"]["open"][1]["profile_key"] = "missing_profile"

    unknown_result = validate_script_program(program, _context())

    program["labels"]["open"][1]["profile_key"] = "Bad Key!"
    invalid_result = validate_script_program(program, _context())

    assert unknown_result.valid is False
    assert "generation_profile_key_unknown" in _error_codes(unknown_result)
    assert invalid_result.valid is False
    assert "generation_profile_key_invalid" in _error_codes(invalid_result)


def test_generate_opcode_rejects_unsupported_output_schema_for_profile() -> None:
    program = _program()
    program["labels"]["open"][1].update(
        {
            "profile_key": "choice_writer",
            "output_schema": "scene_update",
            "scope": "turn",
        }
    )

    result = validate_script_program(program, _context())

    assert result.valid is False
    assert "generation_output_schema_not_supported" in _error_codes(result)


def test_generate_opcode_rejects_non_string_output_schema() -> None:
    program = _program()
    program["labels"]["open"][1]["output_schema"] = ["choice_set"]

    result = validate_script_program(program, _context())

    assert result.valid is False
    assert "generation_output_schema_invalid" in _error_codes(result)


def test_generate_choice_set_requires_authored_generated_choice_target() -> None:
    program = _program()
    program["labels"]["generated_choice"] = [{"op": "narrate", "text": "Choice received."}, {"op": "end"}]
    program["labels"]["open"][1].update(
        {
            "profile_key": "choice_writer",
            "output_schema": "choice_set",
            "scope": "turn",
        }
    )

    result = validate_script_program(program, _context())

    assert result.valid is False
    assert "generation_on_generated_choice_missing" in _error_codes(result)

    program["labels"]["open"][1]["on_generated_choice"] = "missing"
    missing_result = validate_script_program(program, _context())

    assert missing_result.valid is False
    assert "generation_on_generated_choice_missing" in _error_codes(missing_result)


def test_generate_confirmation_and_cancel_shapes_are_validated() -> None:
    program = _program()
    program["labels"]["open"][1]["requires_user_confirm"] = "yes"
    program["labels"]["open"][1]["on_cancel"] = "missing"

    result = validate_script_program(program, _context())

    assert result.valid is False
    assert {
        "generation_requires_user_confirm_invalid",
        "generation_on_cancel_missing",
    }.issubset(_error_codes(result))


def test_literal_generation_remains_valid_without_output_schema() -> None:
    program = _program()
    program["labels"]["open"][1] = {"op": "generate", "narrative_text": "A literal line."}

    result = validate_script_program(program, _context())

    assert result.valid is True
    assert result.errors == []


def test_unreachable_label_is_warning() -> None:
    program = _program()
    program["labels"]["secret"] = [{"op": "narrate", "text": "Hidden."}]

    result = validate_script_program(program, _context())

    assert result.valid is True
    assert "label_unreachable" in _warning_codes(result)
