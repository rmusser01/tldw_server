"""Preview-safe VN script authoring catalog metadata."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from tldw_Server_API.app.core.VN_Scripts.validator import (
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
            "required": ["id", "choices"],
            "properties": {
                "id": {"type": "string", "minLength": 1},
                "choices": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["id", "text", "target"],
                        "properties": {
                            "id": {"type": "string", "minLength": 1},
                            "text": {"type": "string", "minLength": 1},
                            "target": {"type": "string", "minLength": 1},
                        },
                    },
                },
            },
        },
        "preview": [{"op": "choice", "id": "{id}", "choices": []}],
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
                "handler_label": "{handler_label}",
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
                "capability_tokens": list(metadata.get("capability_tokens", ())),
            }
        )
    return operations
