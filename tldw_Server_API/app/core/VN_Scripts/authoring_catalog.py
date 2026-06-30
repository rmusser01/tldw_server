"""Preview-safe VN script authoring catalog metadata."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from tldw_Server_API.app.core.VN_Scripts.validator import (
    forbidden_generation_routing_keys,
    known_script_ops,
    supported_generation_output_schemas,
)

_SCHEMA_VERSION = "vn_script_authoring_catalog.v1"
_PROGRAM_SCHEMA_VERSION = "vn_script_program.v1"

_CAPABILITY_TOKENS: tuple[str, ...] = (
    "script_authoring_catalog",
    "scripted_generation",
    "scripted_generation.output_schema.choice_set",
    "scripted_generation.output_schema.scene_update",
    "scripted_generation.user_confirmation",
)

_OPERATION_CATEGORIES: dict[str, tuple[str, ...]] = {
    "audio": ("play_bgm", "play_sfx", "stop_bgm", "voice_cue"),
    "branching": ("choice", "jump", "random", "return"),
    "generation": ("generate",),
    "state": ("increment", "set"),
    "story": ("end", "label", "narrate", "say"),
    "visuals": ("clear_visuals", "hide_sprite", "set_background", "show_cg", "show_sprite"),
}

_OP_METADATA: dict[str, dict[str, Any]] = {
    "choice": {"label": "Choice", "capability_tokens": ()},
    "clear_visuals": {"label": "Clear visuals", "capability_tokens": ()},
    "end": {"label": "End", "capability_tokens": ()},
    "generate": {"label": "Generate", "capability_tokens": ("scripted_generation",)},
    "hide_sprite": {"label": "Hide sprite", "capability_tokens": ()},
    "increment": {"label": "Increment variable", "capability_tokens": ()},
    "jump": {"label": "Jump", "capability_tokens": ()},
    "label": {"label": "Label", "capability_tokens": ()},
    "narrate": {"label": "Narrate", "capability_tokens": ()},
    "play_bgm": {"label": "Play BGM", "capability_tokens": ()},
    "play_sfx": {"label": "Play SFX", "capability_tokens": ()},
    "random": {"label": "Random branch", "capability_tokens": ()},
    "return": {"label": "Return", "capability_tokens": ()},
    "say": {"label": "Say", "capability_tokens": ()},
    "set": {"label": "Set variable", "capability_tokens": ()},
    "set_background": {"label": "Set background", "capability_tokens": ()},
    "show_cg": {"label": "Show CG", "capability_tokens": ()},
    "show_sprite": {"label": "Show sprite", "capability_tokens": ()},
    "stop_bgm": {"label": "Stop BGM", "capability_tokens": ()},
    "voice_cue": {"label": "Voice cue", "capability_tokens": ()},
}

_CONDITION_FIELD: dict[str, Any] = {
    "name": "if",
    "type": "condition",
    "required": False,
    "description": "Optional variable condition evaluated by the runtime.",
}

_OP_FIELDS: dict[str, tuple[dict[str, Any], ...]] = {
    "choice": (
        {
            "name": "id",
            "type": "string",
            "required": False,
            "description": "Optional stable choice block identifier.",
        },
        {
            "name": "choices",
            "type": "array",
            "required": True,
            "items": {
                "type": "object",
                "required": ["target"],
                "properties": {
                    "id": {"type": "string", "required": False},
                    "text": {"type": "string", "required": False},
                    "target": {"type": "label", "required": True},
                },
            },
        },
    ),
    "clear_visuals": (),
    "end": (),
    "generate": (
        {"name": "profile_key", "type": "string", "required": False, "default": "default"},
        {"name": "scope", "type": "enum", "required": False, "values": ["turn", "scene", "session"]},
        {
            "name": "output_schema",
            "type": "enum",
            "required": False,
            "values": list(supported_generation_output_schemas()),
        },
        {"name": "max_choices", "type": "integer", "required": False, "minimum": 1},
        {"name": "requires_user_confirm", "type": "boolean", "required": False},
        {"name": "on_generated_choice", "type": "label", "required": False},
        {"name": "on_cancel", "type": "label", "required": False},
        {"name": "narrative_text", "type": "string", "required": False, "multiline": True},
        {"name": "regeneration_text", "type": "string", "required": False, "multiline": True},
    ),
    "hide_sprite": ({"name": "slot_key", "type": "asset_slot", "required": False},),
    "increment": (
        {"name": "var", "type": "variable", "required": True},
        {"name": "amount", "type": "number", "required": False, "default": 1},
    ),
    "jump": ({"name": "target", "type": "label", "required": True},),
    "label": ({"name": "name", "type": "label", "required": True},),
    "narrate": ({"name": "text", "type": "string", "required": True, "multiline": True},),
    "play_bgm": ({"name": "media_ref", "type": "audio_ref", "required": True},),
    "play_sfx": ({"name": "media_ref", "type": "audio_ref", "required": True},),
    "random": (
        {"name": "id", "type": "string", "required": False},
        {"name": "var", "type": "variable", "required": False},
        {"name": "min", "type": "integer", "required": False},
        {"name": "max", "type": "integer", "required": False},
    ),
    "return": (),
    "say": (
        {"name": "speaker", "type": "string", "required": True},
        {"name": "text", "type": "string", "required": True, "multiline": True},
    ),
    "set": (
        {"name": "var", "type": "variable", "required": True},
        {"name": "value", "type": "json", "required": True},
    ),
    "set_background": ({"name": "slot_key", "type": "asset_slot", "required": True},),
    "show_cg": ({"name": "slot_key", "type": "asset_slot", "required": True},),
    "show_sprite": ({"name": "slot_key", "type": "asset_slot", "required": True},),
    "stop_bgm": (),
    "voice_cue": ({"name": "media_ref", "type": "audio_ref", "required": True},),
}

_OP_PREVIEWS: dict[str, dict[str, Any]] = {
    "choice": {"op": "choice", "choices": [{"text": "Open the door.", "target": "open_door"}]},
    "clear_visuals": {"op": "clear_visuals"},
    "end": {"op": "end"},
    "generate": {"op": "generate", "scope": "turn", "output_schema": "choice_set", "on_generated_choice": "generated_choice"},
    "hide_sprite": {"op": "hide_sprite", "slot_key": "sprite.character.neutral"},
    "increment": {"op": "increment", "var": "trust", "amount": 1},
    "jump": {"op": "jump", "target": "next_scene"},
    "label": {"op": "label", "name": "next_scene"},
    "narrate": {"op": "narrate", "text": "The scene opens."},
    "play_bgm": {"op": "play_bgm", "media_ref": "bgm_theme"},
    "play_sfx": {"op": "play_sfx", "media_ref": "door_knock"},
    "random": {"op": "random", "var": "roll", "min": 1, "max": 6},
    "return": {"op": "return"},
    "say": {"op": "say", "speaker": "Mira", "text": "Which way?"},
    "set": {"op": "set", "var": "has_key", "value": True},
    "set_background": {"op": "set_background", "slot_key": "background.archive.default"},
    "show_cg": {"op": "show_cg", "slot_key": "cg.reveal.default"},
    "show_sprite": {"op": "show_sprite", "slot_key": "sprite.mira.neutral"},
    "stop_bgm": {"op": "stop_bgm"},
    "voice_cue": {"op": "voice_cue", "media_ref": "voice_mira_line_001"},
}

_SUPPORTS_CONDITION = {
    "choice",
    "clear_visuals",
    "end",
    "generate",
    "hide_sprite",
    "increment",
    "jump",
    "narrate",
    "play_bgm",
    "play_sfx",
    "random",
    "return",
    "say",
    "set",
    "set_background",
    "show_cg",
    "show_sprite",
    "stop_bgm",
    "voice_cue",
}

_OP_NOTES: dict[str, tuple[str, ...]] = {
    "generate": (
        "Generation profiles, output-schema support, moderation, and limits are resolved by backend preview/apply/validate.",
    ),
    "set_background": ("slot_key must resolve to an approved visual asset pack slot.",),
    "show_cg": ("slot_key must resolve to an approved visual asset pack slot.",),
    "show_sprite": ("slot_key must resolve to an approved visual asset pack slot.",),
    "play_bgm": ("media_ref must resolve to accessible audio metadata.",),
    "play_sfx": ("media_ref must resolve to accessible audio metadata.",),
    "voice_cue": ("media_ref must resolve to accessible audio metadata.",),
}

_SNIPPETS: tuple[dict[str, Any], ...] = (
    {
        "id": "narration",
        "schema_version": _PROGRAM_SCHEMA_VERSION,
        "label": "Narration",
        "operation_sequence": ("narrate",),
        "parameters_schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["text"],
            "properties": {
                "text": {"type": "string", "minLength": 1},
            },
        },
        "preview": [{"op": "narrate", "text": "{text}"}],
    },
    {
        "id": "dialogue",
        "schema_version": _PROGRAM_SCHEMA_VERSION,
        "label": "Dialogue",
        "operation_sequence": ("say",),
        "parameters_schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["speaker", "text"],
            "properties": {
                "speaker": {"type": "string", "minLength": 1},
                "text": {"type": "string", "minLength": 1},
            },
        },
        "preview": [{"op": "say", "speaker": "{speaker}", "text": "{text}"}],
    },
    {
        "id": "authored_choice",
        "schema_version": _PROGRAM_SCHEMA_VERSION,
        "label": "Authored choice",
        "operation_sequence": ("choice",),
        "parameters_schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["choice_id", "choices"],
            "properties": {
                "choice_id": {"type": "string", "minLength": 1},
                "choices": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["id", "text", "target_label"],
                        "properties": {
                            "id": {"type": "string", "minLength": 1},
                            "text": {"type": "string", "minLength": 1},
                            "target_label": {"type": "string", "minLength": 1},
                        },
                    },
                },
            },
        },
        "preview": [{"op": "choice", "id": "{choice_id}", "choices": [{"target": "{target_label}"}]}],
    },
    {
        "id": "generated_choice_set",
        "schema_version": _PROGRAM_SCHEMA_VERSION,
        "label": "Generated choice set",
        "operation_sequence": ("generate",),
        "required_capability_tokens": (
            "scripted_generation",
            "scripted_generation.output_schema.choice_set",
        ),
        "parameters_schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["handler_label"],
            "properties": {
                "profile_key": {"type": "string", "minLength": 1, "default": "default"},
                "scope": {"type": "string", "enum": ["turn", "scene"]},
                "max_choices": {"type": "integer", "minimum": 1},
                "requires_user_confirm": {"type": "boolean"},
                "handler_label": {"type": "string", "minLength": 1},
                "on_cancel": {"type": "string", "minLength": 1},
            },
        },
        "default_parameters": {"handler_label": "generated_choice"},
        "preview": [
            {
                "op": "generate",
                "scope": "turn",
                "output_schema": "choice_set",
                "on_generated_choice": "{handler_label}",
            }
        ],
    },
    {
        "id": "scene_update_generation",
        "schema_version": _PROGRAM_SCHEMA_VERSION,
        "label": "Scene update generation",
        "operation_sequence": ("generate",),
        "required_capability_tokens": (
            "scripted_generation",
            "scripted_generation.output_schema.scene_update",
        ),
        "parameters_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "profile_key": {"type": "string", "minLength": 1, "default": "default"},
                "scope": {"type": "string", "enum": ["scene"]},
                "max_choices": {"type": "integer", "minimum": 1},
            },
        },
        "preview": [{"op": "generate", "scope": "scene", "output_schema": "scene_update"}],
    },
    {
        "id": "confirm_gated_generation",
        "schema_version": _PROGRAM_SCHEMA_VERSION,
        "label": "Confirm-gated generation",
        "operation_sequence": ("generate",),
        "required_capability_tokens": (
            "scripted_generation",
            "scripted_generation.user_confirmation",
        ),
        "parameters_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "profile_key": {"type": "string", "minLength": 1, "default": "default"},
                "scope": {"type": "string", "enum": ["turn", "scene"]},
                "max_choices": {"type": "integer", "minimum": 1},
                "on_cancel": {"type": "string", "minLength": 1},
            },
        },
        "preview": [{"op": "generate", "scope": "turn", "requires_user_confirm": True}],
    },
    {
        "id": "set_background",
        "schema_version": _PROGRAM_SCHEMA_VERSION,
        "label": "Set background",
        "operation_sequence": ("set_background",),
        "parameters_schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["slot_key"],
            "properties": {
                "slot_key": {"type": "string", "minLength": 1},
            },
        },
        "preview": [{"op": "set_background", "slot_key": "{slot_key}"}],
    },
    {
        "id": "show_sprite",
        "schema_version": _PROGRAM_SCHEMA_VERSION,
        "label": "Show sprite",
        "operation_sequence": ("show_sprite",),
        "parameters_schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["slot_key"],
            "properties": {
                "slot_key": {"type": "string", "minLength": 1},
            },
        },
        "preview": [{"op": "show_sprite", "slot_key": "{slot_key}"}],
    },
    {
        "id": "play_bgm",
        "schema_version": _PROGRAM_SCHEMA_VERSION,
        "label": "Play BGM",
        "operation_sequence": ("play_bgm",),
        "parameters_schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["media_ref"],
            "properties": {
                "media_ref": {"type": "string", "minLength": 1},
            },
        },
        "preview": [{"op": "play_bgm", "media_ref": "{media_ref}"}],
    },
    {
        "id": "set_variable",
        "schema_version": _PROGRAM_SCHEMA_VERSION,
        "label": "Set variable",
        "operation_sequence": ("set",),
        "parameters_schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["var", "value"],
            "properties": {
                "var": {"type": "string", "minLength": 1},
                "value": {
                    "anyOf": [
                        {"type": "boolean"},
                        {"type": "integer"},
                        {"type": "number"},
                        {"type": "string"},
                    ]
                },
            },
        },
        "preview": [{"op": "set", "var": "{var}", "value": "{value}"}],
    },
    {
        "id": "ending",
        "schema_version": _PROGRAM_SCHEMA_VERSION,
        "label": "Ending",
        "operation_sequence": ("end",),
        "parameters_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {},
        },
        "preview": [{"op": "end"}],
    },
)

_LIMITS: dict[str, int] = {
    "max_title_length": 200,
    "max_label_length": 64,
    "max_variable_name_length": 64,
    "max_choices_per_choice_op": 12,
}


def list_authoring_catalog() -> dict[str, Any]:
    """Return preview-safe metadata for VN script authoring clients."""
    return deepcopy(
        {
            "schema_version": _SCHEMA_VERSION,
            "program_schema_version": _PROGRAM_SCHEMA_VERSION,
            "capability_tokens": list(_CAPABILITY_TOKENS),
            "generation_output_schemas": list(supported_generation_output_schemas()),
            "operation_categories": _operation_categories_payload(),
            "operations": _operations_payload(),
            "snippets": list(_SNIPPETS),
            "limits": _LIMITS,
        }
    )


def _operation_categories_payload() -> dict[str, list[str]]:
    return {category: list(ops) for category, ops in sorted(_OPERATION_CATEGORIES.items())}


def _operations_payload() -> list[dict[str, Any]]:
    operations: list[dict[str, Any]] = []
    categories_by_op = {
        op: category
        for category, ops in _OPERATION_CATEGORIES.items()
        for op in ops
    }
    for op in known_script_ops():
        metadata = _OP_METADATA.get(op, {"label": op.replace("_", " ").title(), "capability_tokens": ()})
        operations.append(
            {
                "op": op,
                "label": metadata["label"],
                "category": categories_by_op.get(op, "other"),
                "description": metadata.get("description", f"Author {metadata['label'].lower()} operations."),
                "fields": _operation_fields(op),
                "capability_tokens": list(metadata.get("capability_tokens", ())),
                "forbidden_fields": _forbidden_generation_fields() if op == "generate" else [],
                "supports_condition": op in _SUPPORTS_CONDITION,
                "preview": deepcopy(_OP_PREVIEWS.get(op)),
                "output_compatibility": _operation_output_compatibility(op),
                "notes": list(_OP_NOTES.get(op, ("Backend diagnostics remain authoritative.",))),
            }
        )
    return operations


def _operation_fields(op: str) -> list[dict[str, Any]]:
    fields = [deepcopy(field) for field in _OP_FIELDS.get(op, ())]
    if op in _SUPPORTS_CONDITION:
        fields.append(deepcopy(_CONDITION_FIELD))
    return fields


def _forbidden_generation_fields() -> list[str]:
    return list(forbidden_generation_routing_keys())


def _operation_output_compatibility(op: str) -> dict[str, Any]:
    if op != "generate":
        return {}
    return {
        "supported_output_schemas": list(supported_generation_output_schemas()),
        "choice_set_requires": ["on_generated_choice"],
    }
