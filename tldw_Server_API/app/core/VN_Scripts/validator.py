"""Pure validator for canonical JSON VN script programs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import re
from typing import Any

from tldw_Server_API.app.core.VN_Scripts.models import VNScriptDiagnostic, VNScriptValidationResult

_VISUAL_OPS = {"set_background", "show_sprite", "show_cg"}
_AUDIO_OPS = {"play_bgm", "play_sfx", "voice_cue"}
_LINK_OPS = {"jump", "choice"}
_GENERATION_SCOPE_ORDER = {"none": 0, "turn": 1, "scene": 2, "session": 3}
_SCRIPT_SCHEMA_VERSION = "vn_script_program.v1"
_PROFILE_KEY_RE = re.compile(r"^[a-z0-9_.-]{1,64}$")
_SUPPORTED_OUTPUT_SCHEMAS = {"narrative_dialogue", "choice_set", "scene_update"}
_ALLOWED_VARIABLE_TYPES = {"boolean", "integer", "number", "string"}
_ALLOWED_CONDITION_OPERATORS = {"eq", "ne", "neq", "lt", "lte", "gt", "gte", "in", "not_in"}
_RAW_GENERATION_ROUTING_KEYS = {
    "api_base",
    "api_key",
    "api_provider",
    "base_url",
    "endpoint",
    "model",
    "provider",
    "provider_config",
}
_KNOWN_OPS = {
    "choice",
    "clear_visuals",
    "end",
    "generate",
    "hide_sprite",
    "increment",
    "jump",
    "label",
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


@dataclass(frozen=True)
class VNScriptValidationContext:
    """Resolved context used by the pure script validator."""

    approved_slot_keys: set[str] = field(default_factory=set)
    audio_refs: dict[str, dict[str, Any]] = field(default_factory=dict)
    generation_profile: dict[str, Any] = field(default_factory=dict)
    available_generation_profiles: dict[str, dict[str, Any]] = field(default_factory=dict)
    content_rating: str = "general"
    owner_user_id: int | None = None


def validate_script_program(
    program: Mapping[str, Any],
    context: VNScriptValidationContext,
) -> VNScriptValidationResult:
    """Validate a canonical VN script program without DB or network access."""
    errors: list[VNScriptDiagnostic] = []
    warnings: list[VNScriptDiagnostic] = []

    if program.get("schema_version") != _SCRIPT_SCHEMA_VERSION:
        errors.append(
            _diag(
                "schema_version_invalid",
                "Script program schema_version must be vn_script_program.v1.",
                "$.schema_version",
                {"schema_version": program.get("schema_version")},
            )
        )

    _validate_top_level_shape(program, errors)

    labels = program.get("labels")
    if not isinstance(labels, Mapping) or not labels:
        errors.append(_diag("labels_missing", "Script must define at least one label.", "$.labels"))
        return _result(errors, warnings)

    label_names = {str(label) for label in labels}
    entry_label = str(program.get("entry_label") or "")
    if entry_label not in label_names:
        errors.append(
            _diag(
                "entry_label_missing",
                "Entry label was not found.",
                "$.entry_label",
                {"entry_label": entry_label},
            )
        )

    variables = _variables(program.get("variables"), errors)
    reachable = _reachable_labels(entry_label, labels) if entry_label in label_names else set()

    for label, raw_ops in labels.items():
        path_prefix = f"$.labels.{label}"
        if not isinstance(raw_ops, list):
            errors.append(_diag("label_body_invalid", "Label body must be a list of opcodes.", path_prefix))
            continue
        for index, opcode in enumerate(raw_ops):
            op_path = f"{path_prefix}[{index}]"
            if not isinstance(opcode, Mapping):
                errors.append(_diag("opcode_invalid", "Opcode must be an object.", op_path))
                continue
            _validate_opcode(opcode, op_path, label_names, variables, context, errors)

    for label in sorted(label_names - reachable):
        if label != entry_label:
            warnings.append(
                _diag(
                    "label_unreachable",
                    "Label is never reached from the entry label.",
                    f"$.labels.{label}",
                    {"label": label},
                )
            )

    return _result(errors, warnings)


def _validate_opcode(
    opcode: Mapping[str, Any],
    path: str,
    label_names: set[str],
    variables: Mapping[str, dict[str, Any]],
    context: VNScriptValidationContext,
    errors: list[VNScriptDiagnostic],
) -> None:
    op = str(opcode.get("op") or "")
    if op not in _KNOWN_OPS:
        errors.append(_diag("opcode_unknown", "Opcode is not part of vn_script_program.v1.", f"{path}.op", {"op": op}))
    elif op == "jump":
        _validate_target(opcode.get("target"), f"{path}.target", "jump_target_missing", label_names, errors)
    elif op == "choice":
        _validate_choices(opcode, path, label_names, context, errors)
    elif op in _VISUAL_OPS:
        slot_key = opcode.get("slot_key")
        if not isinstance(slot_key, str) or slot_key not in context.approved_slot_keys:
            errors.append(
                _diag(
                    "visual_slot_key_missing",
                    "Visual slot key is not present in the approved manifest.",
                    f"{path}.slot_key",
                    {"slot_key": slot_key},
                )
            )
    elif op in _AUDIO_OPS:
        _validate_audio_ref(opcode.get("media_ref"), f"{path}.media_ref", context, errors)
    elif op == "set":
        _validate_assignment(opcode, path, variables, errors)
    elif op == "increment":
        _validate_increment(opcode, path, variables, errors)
    elif op == "generate":
        _validate_generation(opcode, path, label_names, context, errors)

    condition = opcode.get("if")
    if condition is not None:
        _validate_condition(condition, f"{path}.if", variables, errors)


def _validate_choices(
    opcode: Mapping[str, Any],
    path: str,
    label_names: set[str],
    context: VNScriptValidationContext,
    errors: list[VNScriptDiagnostic],
) -> None:
    choices = opcode.get("choices")
    if not isinstance(choices, list) or not choices:
        errors.append(_diag("choice_options_missing", "Choice opcode requires at least one option.", f"{path}.choices"))
        return
    max_choices = int(context.generation_profile.get("max_choices") or 0)
    if max_choices > 0 and len(choices) > max_choices:
        errors.append(
            _diag(
                "generation_max_choices_exceeded",
                "Choice count exceeds the selected generation profile.",
                f"{path}.choices",
                {"max_choices": max_choices, "actual": len(choices)},
            )
        )
    for index, choice in enumerate(choices):
        if isinstance(choice, Mapping):
            _validate_target(
                choice.get("target"),
                f"{path}.choices[{index}].target",
                "choice_target_missing",
                label_names,
                errors,
            )
        else:
            errors.append(_diag("choice_option_invalid", "Choice option must be an object.", f"{path}.choices[{index}]"))


def _validate_target(
    target: Any,
    path: str,
    code: str,
    label_names: set[str],
    errors: list[VNScriptDiagnostic],
) -> None:
    if not isinstance(target, str) or target not in label_names:
        errors.append(_diag(code, "Target label was not found.", path, {"target": target}))


def _validate_assignment(
    opcode: Mapping[str, Any],
    path: str,
    variables: Mapping[str, dict[str, Any]],
    errors: list[VNScriptDiagnostic],
) -> None:
    var_name = str(opcode.get("var") or "")
    definition = variables.get(var_name)
    if definition is None:
        errors.append(_diag("variable_unknown", "Variable is not declared.", f"{path}.var", {"var": var_name}))
        return
    if not _value_matches_type(opcode.get("value"), str(definition.get("type") or "")):
        errors.append(
            _diag(
                "variable_assignment_type_mismatch",
                "Assigned value does not match declared variable type.",
                f"{path}.value",
                {"var": var_name, "expected_type": definition.get("type")},
            )
        )


def _validate_increment(
    opcode: Mapping[str, Any],
    path: str,
    variables: Mapping[str, dict[str, Any]],
    errors: list[VNScriptDiagnostic],
) -> None:
    var_name = str(opcode.get("var") or "")
    definition = variables.get(var_name)
    if definition is None:
        errors.append(_diag("variable_unknown", "Variable is not declared.", f"{path}.var", {"var": var_name}))
        return
    if str(definition.get("type") or "") not in {"integer", "number"}:
        errors.append(
            _diag(
                "variable_increment_type_mismatch",
                "Only numeric variables can be incremented.",
                f"{path}.var",
                {"var": var_name, "expected_type": definition.get("type")},
            )
        )


def _validate_audio_ref(
    media_ref: Any,
    path: str,
    context: VNScriptValidationContext,
    errors: list[VNScriptDiagnostic],
) -> None:
    if not isinstance(media_ref, str) or media_ref not in context.audio_refs:
        errors.append(_diag("audio_media_ref_inaccessible", "Audio media reference is not accessible.", path, {"media_ref": media_ref}))
        return
    metadata = context.audio_refs[media_ref]
    mime_type = str(metadata.get("mime_type") or "")
    if not mime_type.startswith("audio/"):
        errors.append(
            _diag(
                "audio_media_type_invalid",
                "Audio media reference must point to an audio file.",
                path,
                {"media_ref": media_ref, "mime_type": mime_type},
            )
        )


def _validate_generation(
    opcode: Mapping[str, Any],
    path: str,
    label_names: set[str],
    context: VNScriptValidationContext,
    errors: list[VNScriptDiagnostic],
) -> None:
    raw_routing_keys = sorted(str(key) for key in opcode if str(key) in _RAW_GENERATION_ROUTING_KEYS)
    if raw_routing_keys:
        errors.append(
            _diag(
                "generation_raw_routing_not_allowed",
                "Scripts must reference generation profiles instead of raw provider or model routing.",
                path,
                {"keys": raw_routing_keys},
            )
        )
    profile_key = opcode.get("profile_key", "default")
    if not isinstance(profile_key, str) or not _PROFILE_KEY_RE.fullmatch(profile_key):
        errors.append(
            _diag(
                "generation_profile_key_invalid",
                "Generation profile_key must match ^[a-z0-9_.-]{1,64}$.",
                f"{path}.profile_key",
                {"profile_key": profile_key},
            )
        )
        profile_key = "default"
    available_profiles = context.available_generation_profiles or {"default": context.generation_profile}
    profile = available_profiles.get(profile_key)
    if profile is None:
        errors.append(
            _diag(
                "generation_profile_key_unknown",
                "Generation profile_key is not declared by script metadata.",
                f"{path}.profile_key",
                {"profile_key": profile_key},
            )
        )
        profile = context.generation_profile

    is_literal_generation = isinstance(opcode.get("narrative_text"), str) or isinstance(opcode.get("regeneration_text"), str)
    output_schema = opcode.get("output_schema")
    if output_schema is None and not is_literal_generation:
        output_schema = "narrative_dialogue"
    if output_schema is not None:
        if not isinstance(output_schema, str) or output_schema not in _SUPPORTED_OUTPUT_SCHEMAS:
            errors.append(
                _diag(
                    "generation_output_schema_invalid",
                    "Generation output_schema is not supported.",
                    f"{path}.output_schema",
                    {"output_schema": output_schema},
                )
            )
        elif not _profile_supports_output_schema(profile, str(output_schema)):
            errors.append(
                _diag(
                    "generation_output_schema_not_supported",
                    "Generation profile does not support the requested output_schema.",
                    f"{path}.output_schema",
                    {"profile_key": profile_key, "output_schema": output_schema},
                )
            )
    requires_user_confirm = opcode.get("requires_user_confirm")
    if requires_user_confirm is not None and not isinstance(requires_user_confirm, bool):
        errors.append(
            _diag(
                "generation_requires_user_confirm_invalid",
                "requires_user_confirm must be a boolean when provided.",
                f"{path}.requires_user_confirm",
                {"requires_user_confirm": requires_user_confirm},
            )
        )
    if "on_cancel" in opcode:
        _validate_target(opcode.get("on_cancel"), f"{path}.on_cancel", "generation_on_cancel_missing", label_names, errors)
    if output_schema == "choice_set":
        _validate_target(
            opcode.get("on_generated_choice"),
            f"{path}.on_generated_choice",
            "generation_on_generated_choice_missing",
            label_names,
            errors,
        )
    elif "on_generated_choice" in opcode:
        _validate_target(
            opcode.get("on_generated_choice"),
            f"{path}.on_generated_choice",
            "generation_on_generated_choice_missing",
            label_names,
            errors,
        )
    provider_class = str(profile.get("provider_class") or profile.get("deployment_class") or "")
    if provider_class in {"hosted", "public"} and not bool(profile.get("moderation_required")):
        errors.append(
            _diag(
                "generation_moderation_required",
                "Hosted or public generation profiles must require moderation.",
                f"{path}.profile_key",
                {"profile_key": profile_key, "provider_class": provider_class},
            )
        )
    batch_cap = profile.get("automatic_generation_batch_cap", profile.get("max_automatic_generation_batch"))
    if requires_user_confirm is not True and batch_cap is not None:
        try:
            batch_cap_int = int(batch_cap)
        except (TypeError, ValueError):
            batch_cap_int = 0
        if batch_cap_int < 1:
            errors.append(
                _diag(
                    "generation_batch_cap_invalid",
                    "Automatic generation profiles must allow at least one generation per batch.",
                    f"{path}.profile_key",
                    {"profile_key": profile_key, "batch_cap": batch_cap},
                )
            )
    allowed_ratings = {str(rating) for rating in profile.get("allowed_content_ratings", [])}
    if allowed_ratings and context.content_rating not in allowed_ratings:
        errors.append(
            _diag(
                "generation_content_rating_not_allowed",
                "Content rating is not allowed by the selected generation profile.",
                "$.content_rating",
                {"content_rating": context.content_rating},
            )
        )
    requested_scope = str(opcode.get("scope") or "turn")
    max_scope = str(profile.get("max_model_expansion_scope") or "none")
    if _GENERATION_SCOPE_ORDER.get(requested_scope, 99) > _GENERATION_SCOPE_ORDER.get(max_scope, 0):
        errors.append(
            _diag(
                "generation_scope_not_allowed",
                "Generation scope exceeds the selected generation profile.",
                f"{path}.scope",
                {"scope": requested_scope, "max_scope": max_scope},
            )
        )
    max_choices = int(profile.get("max_choices") or 0)
    requested_choices = opcode.get("max_choices")
    if isinstance(requested_choices, int) and max_choices > 0 and requested_choices > max_choices:
        errors.append(
            _diag(
                "generation_max_choices_exceeded",
                "Requested generated choice count exceeds the selected generation profile.",
                f"{path}.max_choices",
                {"max_choices": max_choices, "requested": requested_choices},
            )
        )


def _profile_supports_output_schema(profile: Mapping[str, Any], output_schema: str) -> bool:
    supported = profile.get("supported_output_schemas", profile.get("allowed_output_schemas"))
    if isinstance(supported, list):
        return output_schema in {str(schema) for schema in supported}
    if output_schema in {"choice_set", "scene_update"} and not bool(profile.get("supports_structured_output", False)):
        return False
    return True


def _validate_condition(
    condition: Any,
    path: str,
    variables: Mapping[str, dict[str, Any]],
    errors: list[VNScriptDiagnostic],
) -> None:
    if not isinstance(condition, Mapping):
        errors.append(_diag("condition_invalid", "Condition must be an object.", path))
        return
    for logical_key in ("all", "any"):
        if logical_key in condition:
            operands = condition[logical_key]
            if not isinstance(operands, list):
                errors.append(_diag("condition_operands_invalid", "Logical condition operands must be a list.", f"{path}.{logical_key}"))
                return
            for index, operand in enumerate(operands):
                _validate_condition(operand, f"{path}.{logical_key}[{index}]", variables, errors)
            return
    if "not" in condition:
        _validate_condition(condition["not"], f"{path}.not", variables, errors)
        return

    var_name = str(condition.get("var") or "")
    definition = variables.get(var_name)
    if definition is None:
        errors.append(_diag("condition_variable_unknown", "Condition variable is not declared.", f"{path}.var", {"var": var_name}))
        return
    operator = condition.get("op")
    if operator not in _ALLOWED_CONDITION_OPERATORS:
        errors.append(_diag("condition_operator_invalid", "Condition operator is not supported.", f"{path}.op", {"op": operator}))
        return
    if operator in {"in", "not_in"}:
        values = condition.get("value")
        if not isinstance(values, list):
            errors.append(_diag("condition_operand_type_mismatch", "Set membership condition value must be a list.", f"{path}.value"))
            return
        for index, value in enumerate(values):
            if not _value_matches_type(value, str(definition.get("type") or "")):
                errors.append(
                    _diag(
                        "condition_operand_type_mismatch",
                        "Condition operand does not match declared variable type.",
                        f"{path}.value[{index}]",
                        {"var": var_name, "expected_type": definition.get("type")},
                    )
                )
        return
    if not _value_matches_type(condition.get("value"), str(definition.get("type") or "")):
        errors.append(
            _diag(
                "condition_operand_type_mismatch",
                "Condition operand does not match declared variable type.",
                f"{path}.value",
                {"var": var_name, "expected_type": definition.get("type")},
            )
        )


def _validate_top_level_shape(program: Mapping[str, Any], errors: list[VNScriptDiagnostic]) -> None:
    entry_label = program.get("entry_label")
    if not isinstance(entry_label, str) or not entry_label:
        errors.append(_diag("entry_label_invalid", "Entry label must be a non-empty string.", "$.entry_label"))
    asset_pack_id = program.get("primary_asset_pack_id")
    if not isinstance(asset_pack_id, int) or isinstance(asset_pack_id, bool) or asset_pack_id < 1:
        errors.append(
            _diag(
                "primary_asset_pack_id_invalid",
                "primary_asset_pack_id must be a positive integer.",
                "$.primary_asset_pack_id",
                {"primary_asset_pack_id": asset_pack_id},
            )
        )
    generation_defaults = program.get("generation_defaults")
    if generation_defaults is not None:
        if not isinstance(generation_defaults, Mapping):
            errors.append(_diag("generation_defaults_invalid", "generation_defaults must be an object.", "$.generation_defaults"))
        else:
            raw_routing_keys = sorted(str(key) for key in generation_defaults if str(key) in _RAW_GENERATION_ROUTING_KEYS)
            if raw_routing_keys:
                errors.append(
                    _diag(
                        "generation_raw_routing_not_allowed",
                        "Scripts must reference generation profiles instead of raw provider or model routing.",
                        "$.generation_defaults",
                        {"keys": raw_routing_keys},
                    )
                )
            if "profile_id" in generation_defaults and not isinstance(generation_defaults.get("profile_id"), str):
                errors.append(
                    _diag(
                        "generation_profile_id_invalid",
                        "generation_defaults.profile_id must be a string.",
                        "$.generation_defaults.profile_id",
                        {"profile_id": generation_defaults.get("profile_id")},
                    )
                )


def _variables(raw_variables: Any, errors: list[VNScriptDiagnostic]) -> dict[str, dict[str, Any]]:
    if raw_variables is None:
        return {}
    if not isinstance(raw_variables, Mapping):
        errors.append(_diag("variables_invalid", "variables must be an object.", "$.variables"))
        return {}
    variables: dict[str, dict[str, Any]] = {}
    for name, definition in raw_variables.items():
        path = f"$.variables.{name}"
        if not isinstance(name, str) or not name:
            errors.append(_diag("variable_name_invalid", "Variable names must be non-empty strings.", path))
            continue
        if not isinstance(definition, Mapping):
            errors.append(_diag("variable_definition_invalid", "Variable definition must be an object.", path))
            continue
        definition_dict = dict(definition)
        var_type = str(definition_dict.get("type") or "")
        if var_type not in _ALLOWED_VARIABLE_TYPES:
            errors.append(
                _diag(
                    "variable_type_invalid",
                    "Variable type is not supported.",
                    f"{path}.type",
                    {"var": name, "type": definition_dict.get("type")},
                )
            )
        elif "default" in definition_dict and not _value_matches_type(definition_dict.get("default"), var_type):
            errors.append(
                _diag(
                    "variable_default_type_mismatch",
                    "Variable default does not match declared type.",
                    f"{path}.default",
                    {"var": name, "expected_type": var_type},
                )
            )
        if "public" in definition_dict and not isinstance(definition_dict.get("public"), bool):
            errors.append(
                _diag(
                    "variable_public_invalid",
                    "Variable public flag must be a boolean.",
                    f"{path}.public",
                    {"var": name, "public": definition_dict.get("public")},
                )
            )
        variables[name] = definition_dict
    return {
        str(name): dict(definition)
        for name, definition in variables.items()
    }


def _reachable_labels(entry_label: str, labels: Mapping[str, Any]) -> set[str]:
    reachable: set[str] = set()
    stack = [entry_label]
    while stack:
        label = stack.pop()
        if label in reachable:
            continue
        reachable.add(label)
        raw_ops = labels.get(label)
        if not isinstance(raw_ops, list):
            continue
        for opcode in raw_ops:
            if not isinstance(opcode, Mapping):
                continue
            if opcode.get("op") == "jump" and isinstance(opcode.get("target"), str):
                stack.append(str(opcode["target"]))
            elif opcode.get("op") == "generate":
                if isinstance(opcode.get("on_cancel"), str):
                    stack.append(str(opcode["on_cancel"]))
                if isinstance(opcode.get("on_generated_choice"), str):
                    stack.append(str(opcode["on_generated_choice"]))
            elif opcode.get("op") == "choice" and isinstance(opcode.get("choices"), list):
                for choice in opcode["choices"]:
                    if isinstance(choice, Mapping) and isinstance(choice.get("target"), str):
                        stack.append(str(choice["target"]))
    return reachable


def _value_matches_type(value: Any, expected_type: str) -> bool:
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return (isinstance(value, int | float) and not isinstance(value, bool))
    if expected_type == "string":
        return isinstance(value, str)
    return True


def _diag(
    code: str,
    message: str,
    path: str,
    details: Mapping[str, Any] | None = None,
) -> VNScriptDiagnostic:
    return VNScriptDiagnostic(code=code, message=message, path=path, details=dict(details or {}))


def _result(
    errors: list[VNScriptDiagnostic],
    warnings: list[VNScriptDiagnostic],
) -> VNScriptValidationResult:
    return VNScriptValidationResult(
        valid=not errors,
        errors=[error.to_dict() for error in errors],
        warnings=[warning.to_dict() for warning in warnings],
    )
