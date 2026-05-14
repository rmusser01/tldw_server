"""Pure VN script snippet patching helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import json
from typing import Any

from tldw_Server_API.app.core.VN_Scripts.authoring_errors import VNScriptAuthoringError
from tldw_Server_API.app.core.VN_Scripts.validator import forbidden_generation_routing_keys

MAX_SNIPPET_PARAMETER_DEPTH = 8
MAX_SNIPPET_PARAMETER_STRING_LENGTH = 8000
MAX_SNIPPET_PARAMETER_PAYLOAD_BYTES = 65536


@dataclass(frozen=True)
class SnippetPatchResult:
    """Result returned after applying a VN script snippet patch."""

    draft: dict[str, Any]
    patch_summary: dict[str, Any]


def _json_path_key(key: str) -> str:
    escaped = key.replace("\\", "\\\\").replace("'", "\\'")
    return f"['{escaped}']"


def apply_snippet_patch(
    draft: Mapping[str, Any],
    snippet_id: str,
    anchor: Mapping[str, Any],
    parameters: Mapping[str, Any],
) -> SnippetPatchResult:
    """Apply a preview-safe snippet to a parsed script draft."""
    _validate_parameters(parameters)
    _validate_snippet_parameter_shape(snippet_id, parameters)
    patched = deepcopy(dict(draft))
    labels = _labels(patched)
    label, insert_at = _resolve_anchor(labels, anchor)
    snippet = _render_snippet(snippet_id, parameters, labels)

    labels[label][insert_at:insert_at] = snippet["ops"]
    created_labels: list[str] = []
    for created_label, body in snippet["labels"].items():
        if created_label in labels:
            raise VNScriptAuthoringError(
                "snippet_label_conflict",
                "Snippet label already exists.",
                details={"label": created_label},
            )
        labels[created_label] = body
        created_labels.append(created_label)

    inserted_count = len(snippet["ops"])
    changed_paths = [
        f"$.labels{_json_path_key(label)}[{index}]"
        for index in range(insert_at, insert_at + inserted_count)
    ]
    changed_paths.extend(f"$.labels{_json_path_key(created_label)}" for created_label in created_labels)
    return SnippetPatchResult(
        draft=patched,
        patch_summary={
            "inserted_ops": inserted_count,
            "created_labels": created_labels,
            "changed_paths": changed_paths,
        },
    )


def _validate_parameters(parameters: Mapping[str, Any]) -> None:
    _validate_parameter_value(parameters, "$.parameters", 0, set(forbidden_generation_routing_keys()))
    try:
        payload = json.dumps(parameters, sort_keys=True, separators=(",", ":")).encode("utf-8")
    except (RecursionError, TypeError, ValueError) as exc:
        raise VNScriptAuthoringError(
            "snippet_parameter_invalid",
            "Snippet parameters must be JSON serializable.",
            details={"field_path": "$.parameters"},
        ) from exc
    if len(payload) > MAX_SNIPPET_PARAMETER_PAYLOAD_BYTES:
        raise VNScriptAuthoringError(
            "snippet_parameter_invalid",
            "Snippet parameters exceed the payload size limit.",
            details={"field_path": "$.parameters"},
        )


def _validate_parameter_value(value: Any, path: str, depth: int, forbidden_keys: set[str]) -> None:
    if depth > MAX_SNIPPET_PARAMETER_DEPTH:
        raise VNScriptAuthoringError(
            "snippet_parameter_invalid",
            "Snippet parameters exceed the nesting depth limit.",
            details={"field_path": path},
        )
    if isinstance(value, str):
        if len(value) > MAX_SNIPPET_PARAMETER_STRING_LENGTH:
            raise VNScriptAuthoringError(
                "snippet_parameter_invalid",
                "Snippet parameter string exceeds the length limit.",
                details={"field_path": path},
            )
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            child_path = f"{path}.{key_text}"
            if key_text in forbidden_keys:
                raise VNScriptAuthoringError(
                    "snippet_parameter_invalid",
                    "Snippet parameters must not include raw generation routing keys.",
                    details={"field_path": child_path, "key": key_text},
                )
            _validate_parameter_value(child, child_path, depth + 1, forbidden_keys)
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _validate_parameter_value(child, f"{path}[{index}]", depth + 1, forbidden_keys)


def _validate_snippet_parameter_shape(snippet_id: str, parameters: Mapping[str, Any]) -> None:
    allowed_fields: dict[str, Any] = {
        "narration": {"text": None},
        "dialogue": {"speaker": None, "text": None},
        "authored_choice": {"choice_id": None, "choices": {"id": None, "text": None, "target_label": None}},
        "generated_choice_set": {
            "profile_key": None,
            "scope": None,
            "max_choices": None,
            "requires_user_confirm": None,
            "handler_label": None,
            "on_cancel": None,
        },
        "scene_update_generation": {"profile_key": None, "scope": None, "max_choices": None},
        "confirm_gated_generation": {"profile_key": None, "scope": None, "max_choices": None, "on_cancel": None},
        "set_background": {"slot_key": None},
        "show_sprite": {"slot_key": None},
        "play_bgm": {"media_ref": None},
        "set_variable": {"var": None, "value": None},
        "ending": {},
    }
    if snippet_id not in allowed_fields:
        return
    _reject_unknown_fields(parameters, allowed_fields[snippet_id], "$.parameters")


def _reject_unknown_fields(value: Any, allowed_fields: Mapping[str, Any] | None, path: str) -> None:
    if not isinstance(value, Mapping) or allowed_fields is None:
        return
    for key, child in value.items():
        key_text = str(key)
        if key_text not in allowed_fields:
            raise VNScriptAuthoringError(
                "snippet_parameter_invalid",
                "Snippet parameter is not supported by this snippet.",
                details={"field_path": f"{path}.{key_text}"},
            )
        child_allowed = allowed_fields[key_text]
        if isinstance(child_allowed, Mapping) and isinstance(child, list):
            for index, item in enumerate(child):
                _reject_unknown_fields(item, child_allowed, f"{path}.{key_text}[{index}]")
        elif isinstance(child_allowed, Mapping):
            _reject_unknown_fields(child, child_allowed, f"{path}.{key_text}")


def _labels(draft: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    labels = draft.get("labels")
    if not isinstance(labels, dict):
        raise VNScriptAuthoringError(
            "snippet_anchor_invalid",
            "Draft labels must be an object.",
            details={"anchor": None},
        )
    return labels


def _resolve_anchor(labels: Mapping[str, Any], anchor: Mapping[str, Any]) -> tuple[str, int]:
    label = anchor.get("label")
    mode = anchor.get("mode", "append")
    if not isinstance(label, str) or not label or mode not in {"before", "after", "append"}:
        raise VNScriptAuthoringError(
            "snippet_anchor_invalid",
            "Snippet anchor is invalid.",
            details={"anchor": dict(anchor)},
        )
    if label not in labels:
        raise VNScriptAuthoringError(
            "snippet_anchor_not_found",
            "Snippet anchor label was not found.",
            details={"anchor": dict(anchor)},
        )
    body = labels[label]
    if not isinstance(body, list):
        raise VNScriptAuthoringError(
            "snippet_anchor_invalid",
            "Snippet anchor label body is invalid.",
            details={"anchor": dict(anchor)},
        )
    if mode == "append":
        return label, len(body)

    op_index = anchor.get("op_index")
    if isinstance(op_index, bool) or not isinstance(op_index, int):
        raise VNScriptAuthoringError(
            "snippet_anchor_invalid",
            "Snippet anchor operation index is invalid.",
            details={"anchor": dict(anchor)},
        )
    if op_index < 0 or op_index >= len(body):
        raise VNScriptAuthoringError(
            "snippet_anchor_not_found",
            "Snippet anchor operation was not found.",
            details={"anchor": dict(anchor)},
        )
    return label, op_index if mode == "before" else op_index + 1


def _render_snippet(
    snippet_id: str,
    parameters: Mapping[str, Any],
    existing_labels: Mapping[str, Any],
) -> dict[str, Any]:
    if snippet_id == "narration":
        return _snippet([{"op": "narrate", "text": _required_string(parameters, "text")}])
    if snippet_id == "dialogue":
        return _snippet(
            [
                {
                    "op": "say",
                    "speaker": _required_string(parameters, "speaker"),
                    "text": _required_string(parameters, "text"),
                }
            ]
        )
    if snippet_id == "authored_choice":
        return _snippet(
            [{"op": "choice", "id": _required_string(parameters, "choice_id"), "choices": _choices(parameters)}]
        )
    if snippet_id == "generated_choice_set":
        return _generated_choice_set(parameters, existing_labels)
    if snippet_id == "scene_update_generation":
        op = _generate_base(parameters, default_scope="scene", allowed_scopes={"scene"})
        op["output_schema"] = "scene_update"
        return _snippet([op])
    if snippet_id == "confirm_gated_generation":
        op = _generate_base(parameters, default_scope="turn", allowed_scopes={"turn", "scene"})
        op["requires_user_confirm"] = True
        _copy_optional_strings(op, parameters, ("on_cancel",))
        return _snippet([op])
    if snippet_id == "set_background":
        return _snippet([{"op": "set_background", "slot_key": _required_string(parameters, "slot_key")}])
    if snippet_id == "show_sprite":
        return _snippet([{"op": "show_sprite", "slot_key": _required_string(parameters, "slot_key")}])
    if snippet_id == "play_bgm":
        return _snippet([{"op": "play_bgm", "media_ref": _required_string(parameters, "media_ref")}])
    if snippet_id == "set_variable":
        if "value" not in parameters:
            _raise_missing("value")
        return _snippet([{"op": "set", "var": _required_string(parameters, "var"), "value": deepcopy(parameters["value"])}])
    if snippet_id == "ending":
        return _snippet([{"op": "end"}])
    raise VNScriptAuthoringError(
        "snippet_not_found",
        "Snippet was not found.",
        details={"snippet_id": snippet_id},
    )


def _snippet(ops: list[dict[str, Any]], labels: dict[str, list[dict[str, Any]]] | None = None) -> dict[str, Any]:
    return {"ops": ops, "labels": labels or {}}


def _generated_choice_set(parameters: Mapping[str, Any], existing_labels: Mapping[str, Any]) -> dict[str, Any]:
    handler_label = _required_string(parameters, "handler_label")
    if handler_label in existing_labels:
        raise VNScriptAuthoringError(
            "snippet_label_conflict",
            "Generated choice handler label already exists.",
            details={"label": handler_label},
        )
    op = _generate_base(parameters, default_scope="turn", allowed_scopes={"turn", "scene"})
    op["output_schema"] = "choice_set"
    op["on_generated_choice"] = handler_label
    _copy_optional_strings(op, parameters, ("on_cancel",))
    return _snippet(
        [op],
        {
            handler_label: [
                {"op": "narrate", "text": "Handle the selected generated choice here."},
                {"op": "end"},
            ]
        },
    )


def _generate_base(parameters: Mapping[str, Any], *, default_scope: str, allowed_scopes: set[str]) -> dict[str, Any]:
    scope = parameters.get("scope", default_scope)
    if not isinstance(scope, str) or scope not in allowed_scopes:
        raise VNScriptAuthoringError(
            "snippet_parameter_invalid",
            "scope must be one of the allowed values for this snippet.",
            details={"field_path": "$.parameters.scope"},
        )
    op: dict[str, Any] = {"op": "generate", "scope": scope}
    _copy_optional_strings(op, parameters, ("profile_key",))
    max_choices = parameters.get("max_choices")
    if max_choices is not None:
        if not isinstance(max_choices, int) or isinstance(max_choices, bool) or max_choices < 1:
            raise VNScriptAuthoringError(
                "snippet_parameter_invalid",
                "max_choices must be a positive integer.",
                details={"field_path": "$.parameters.max_choices"},
            )
        op["max_choices"] = max_choices
    requires_user_confirm = parameters.get("requires_user_confirm")
    if requires_user_confirm is not None:
        if not isinstance(requires_user_confirm, bool):
            raise VNScriptAuthoringError(
                "snippet_parameter_invalid",
                "requires_user_confirm must be a boolean.",
                details={"field_path": "$.parameters.requires_user_confirm"},
            )
        op["requires_user_confirm"] = requires_user_confirm
    return op


def _copy_optional_strings(target: dict[str, Any], parameters: Mapping[str, Any], keys: Sequence[str]) -> None:
    for key in keys:
        value = parameters.get(key)
        if value is not None:
            target[key] = _string_value(value, key)


def _choices(parameters: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_choices = parameters.get("choices")
    if not isinstance(raw_choices, list) or not raw_choices:
        raise VNScriptAuthoringError(
            "snippet_parameter_invalid",
            "choices must be a non-empty list.",
            details={"field_path": "$.parameters.choices"},
        )
    choices: list[dict[str, Any]] = []
    for index, raw_choice in enumerate(raw_choices):
        if not isinstance(raw_choice, Mapping):
            raise VNScriptAuthoringError(
                "snippet_parameter_invalid",
                "choice must be an object.",
                details={"field_path": f"$.parameters.choices[{index}]"},
            )
        choices.append(
            {
                "id": _required_string(raw_choice, "id", prefix=f"$.parameters.choices[{index}]"),
                "text": _required_string(raw_choice, "text", prefix=f"$.parameters.choices[{index}]"),
                "target": _required_string(raw_choice, "target_label", prefix=f"$.parameters.choices[{index}]"),
            }
        )
    return choices


def _required_string(parameters: Mapping[str, Any], key: str, *, prefix: str = "$.parameters") -> str:
    if key not in parameters:
        _raise_missing(key, prefix=prefix)
    return _string_value(parameters[key], key, prefix=prefix)


def _string_value(value: Any, key: str, *, prefix: str = "$.parameters") -> str:
    if not isinstance(value, str) or not value:
        raise VNScriptAuthoringError(
            "snippet_parameter_invalid",
            f"{key} must be a non-empty string.",
            details={"field_path": f"{prefix}.{key}"},
        )
    return value


def _raise_missing(key: str, *, prefix: str = "$.parameters") -> None:
    raise VNScriptAuthoringError(
        "snippet_parameter_invalid",
        f"{key} is required.",
        details={"field_path": f"{prefix}.{key}"},
    )
