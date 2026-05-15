"""Pure scripted-story interpreter helpers for VN Play and script preflight."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from tldw_Server_API.app.core.VN_Play.constants import ERROR_INVALID_CHOICE_ID
from tldw_Server_API.app.core.VN_Play.errors import VNPlayConflictError, VNPlayTurnError

MAX_SCRIPT_EXECUTION_STEPS = 500
SCRIPT_GENERATION_SOURCE_LITERAL = "script_literal"


def _initial_script_position(program: Any) -> dict[str, Any]:
    """Build the deterministic initial interpreter position for a script."""
    if not isinstance(program, Mapping):
        return {"label": "start", "index": 0, "ended": False, "variables": {}}
    entry_label = str(program.get("entry_label") or "start")
    return {
        "label": entry_label,
        "index": 0,
        "ended": False,
        "variables": _initial_script_variables(program.get("variables")),
    }


def _initial_script_variables(raw_variables: Any) -> dict[str, Any]:
    if not isinstance(raw_variables, Mapping):
        return {}
    variables: dict[str, Any] = {}
    for name, definition in raw_variables.items():
        if isinstance(name, str) and isinstance(definition, Mapping):
            variables[name] = definition.get("default")
    return variables


def _execute_script_program(
    program: Any,
    position: Mapping[str, Any],
    *,
    choice_id: str | None,
    seed: str | None,
    max_steps: int = MAX_SCRIPT_EXECUTION_STEPS,
) -> dict[str, Any]:
    """Run a deterministic script segment until the next visible boundary."""
    if not isinstance(program, Mapping):
        raise VNPlayTurnError("script_program_missing")
    labels = program.get("labels")
    if not isinstance(labels, Mapping):
        raise VNPlayTurnError("script_labels_missing")

    current_position = dict(position or _initial_script_position(program))
    variables = dict(current_position.get("variables") or {})
    selected_choice: dict[str, Any] | None = None
    if choice_id is not None:
        selected_choice = _script_selected_choice(current_position, choice_id)
        if selected_choice.get("source") == "generated":
            variables.update(_generated_choice_variables(selected_choice))
        current_position = {
            "label": selected_choice["target"],
            "index": 0,
            "ended": False,
            "variables": variables,
        }

    label = str(current_position.get("label") or program.get("entry_label") or "start")
    index = int(current_position.get("index") or 0)
    narrative_lines: list[str] = []
    dialogue: list[dict[str, Any]] = []
    visible_choices: list[dict[str, Any]] = []
    random_results: list[dict[str, Any]] = []
    generation_results: list[dict[str, Any]] = []
    ended = False

    for _ in range(max(1, int(max_steps))):
        ops = labels.get(label)
        if not isinstance(ops, list):
            raise VNPlayTurnError("script_label_missing")
        if index >= len(ops):
            ended = True
            break
        opcode = ops[index]
        index += 1
        if not isinstance(opcode, Mapping):
            continue
        if not _script_condition_matches(opcode.get("if"), variables):
            continue

        op = str(opcode.get("op") or "")
        if op == "narrate":
            text = str(opcode.get("text") or "")
            if text:
                narrative_lines.append(text)
                dialogue.append({"speaker": "Narrator", "text": text})
        elif op == "say":
            text = str(opcode.get("text") or "")
            speaker = str(opcode.get("speaker") or opcode.get("character") or "")
            if text:
                dialogue.append({"speaker": speaker or "Narrator", "text": text})
        elif op == "set":
            var_name = str(opcode.get("var") or "")
            if var_name:
                variables[var_name] = opcode.get("value")
        elif op == "increment":
            var_name = str(opcode.get("var") or "")
            amount = opcode.get("amount", 1)
            current = variables.get(var_name, 0)
            if isinstance(current, (int, float)) and isinstance(amount, (int, float)):
                variables[var_name] = current + amount
        elif op == "random":
            result = _script_random_result(opcode, seed=seed, label=label, index=index - 1)
            var_name = result.get("var")
            if isinstance(var_name, str) and var_name:
                variables[var_name] = result.get("value")
            random_results.append(result)
        elif op == "generate":
            result = _script_generation_result(opcode, seed=seed, label=label, index=index - 1)
            if result.get("pending") is True:
                current_position = {
                    "label": label,
                    "index": index,
                    "ended": False,
                    "variables": variables,
                }
                return {
                    **_script_execution_payload(
                        position=current_position,
                        variables=variables,
                        narrative_lines=narrative_lines,
                        dialogue=dialogue,
                        visible_choices=[],
                        selected_choice=selected_choice,
                        random_results=random_results,
                        generation_results=generation_results,
                    ),
                    "pending_generation": result,
                }
            generation_results.append(result)
            text = str(result.get("narrative_text") or "")
            if text:
                narrative_lines.append(text)
            dialogue.extend(_list_of_dicts(result.get("dialogue")))
            current_position = {
                "label": label,
                "index": index,
                "ended": False,
                "variables": variables,
                "last_generation": result,
            }
            return _script_execution_payload(
                position=current_position,
                variables=variables,
                narrative_lines=narrative_lines,
                dialogue=dialogue,
                visible_choices=[],
                selected_choice=selected_choice,
                random_results=random_results,
                generation_results=generation_results,
            )
        elif op == "jump":
            label = str(opcode.get("target") or "")
            index = 0
        elif op == "choice":
            visible_choices = _script_visible_choices(opcode)
            current_position = {
                "label": label,
                "index": index - 1,
                "ended": False,
                "variables": variables,
                "waiting_choice_id": str(opcode.get("id") or ""),
                "waiting_choices": visible_choices,
            }
            return _script_execution_payload(
                position=current_position,
                variables=variables,
                narrative_lines=narrative_lines,
                dialogue=dialogue,
                visible_choices=visible_choices,
                selected_choice=selected_choice,
                random_results=random_results,
                generation_results=generation_results,
            )
        elif op == "end":
            ended = True
            break
        elif op == "return":
            ended = True
            break

    current_position = {
        "label": label,
        "index": index,
        "ended": ended,
        "variables": variables,
    }
    return _script_execution_payload(
        position=current_position,
        variables=variables,
        narrative_lines=narrative_lines,
        dialogue=dialogue,
        visible_choices=[],
        selected_choice=selected_choice,
        random_results=random_results,
        generation_results=generation_results,
    )


def _script_selected_choice(position: Mapping[str, Any], choice_id: str) -> dict[str, Any]:
    choices = _list_of_dicts(position.get("waiting_choices"))
    for choice in choices:
        if str(choice.get("id")) == choice_id and choice.get("target"):
            return choice
    raise VNPlayTurnError(ERROR_INVALID_CHOICE_ID)


def _script_visible_choices(opcode: Mapping[str, Any]) -> list[dict[str, Any]]:
    choices = _list_of_dicts(opcode.get("choices"))
    return [
        {
            "id": str(choice.get("id") or ""),
            "text": str(choice.get("text") or choice.get("id") or ""),
            "target": str(choice.get("target") or ""),
        }
        for choice in choices
        if choice.get("id") and choice.get("target")
    ]


def _script_execution_payload(
    *,
    position: Mapping[str, Any],
    variables: Mapping[str, Any],
    narrative_lines: Sequence[str],
    dialogue: Sequence[Mapping[str, Any]],
    visible_choices: Sequence[Mapping[str, Any]],
    selected_choice: Mapping[str, Any] | None,
    random_results: Sequence[Mapping[str, Any]],
    generation_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    payload = {
        "position": dict(position),
        "variables": dict(variables),
        "narrative_text": "\n".join(narrative_lines),
        "dialogue": [dict(item) for item in dialogue],
        "visible_choices": [dict(choice) for choice in visible_choices],
        "random_results": [dict(result) for result in random_results],
        "generation_results": [dict(result) for result in generation_results],
    }
    if selected_choice is not None:
        payload["selected_choice"] = dict(selected_choice)
    return payload


def _script_random_result(
    opcode: Mapping[str, Any],
    *,
    seed: str | None,
    label: str,
    index: int,
) -> dict[str, Any]:
    random_id = str(opcode.get("id") or f"{label}:{index}")
    var_name = str(opcode.get("var") or "")
    digest = int(_payload_hash({"seed": seed or "", "id": random_id, "label": label, "index": index}), 16)
    choices = opcode.get("choices")
    if isinstance(choices, Sequence) and not isinstance(choices, (str, bytes)) and len(choices) > 0:
        value = choices[digest % len(choices)]
        result_type = "choice"
    else:
        minimum = opcode.get("min", 0)
        maximum = opcode.get("max", 1)
        if not isinstance(minimum, int) or isinstance(minimum, bool):
            minimum = 0
        if not isinstance(maximum, int) or isinstance(maximum, bool):
            maximum = 1
        if maximum < minimum:
            minimum, maximum = maximum, minimum
        value = minimum + (digest % (maximum - minimum + 1))
        result_type = "integer"

    return {"id": random_id, "var": var_name, "type": result_type, "value": value}


def _script_generation_result(
    opcode: Mapping[str, Any],
    *,
    seed: str | None,
    label: str,
    index: int,
) -> dict[str, Any]:
    generation_id = str(opcode.get("id") or f"{label}:{index}")
    prompt = str(opcode.get("prompt") or opcode.get("text") or generation_id)
    narrative_text = str(opcode.get("narrative_text") or opcode.get("text") or "")
    if not narrative_text:
        return {
            "id": generation_id,
            "label": label,
            "index": index,
            "prompt": prompt,
            "prompt_hash": _payload_hash({"seed": seed or "", "prompt": prompt})[:16],
            "profile_key": str(opcode.get("profile_key") or "default"),
            "output_schema": str(opcode.get("output_schema") or "narrative_dialogue"),
            "requires_user_confirm": bool(opcode.get("requires_user_confirm")),
            "opcode_snapshot": dict(opcode),
            "source": "model",
            "model_invoked": True,
            "pending": True,
        }
    speaker = str(opcode.get("speaker") or "Narrator")
    regeneration_text = str(opcode.get("regeneration_text") or opcode.get("regenerate_text") or "")
    return {
        "id": generation_id,
        "label": label,
        "index": index,
        "prompt_hash": _payload_hash({"seed": seed or "", "prompt": prompt})[:16],
        "source": SCRIPT_GENERATION_SOURCE_LITERAL,
        "model_invoked": False,
        "narrative_text": narrative_text,
        "dialogue": [{"speaker": speaker, "text": narrative_text}],
        "regeneration_text": regeneration_text or None,
        "regeneration_supported": bool(regeneration_text),
        "regenerated": False,
    }


def _script_regeneration_result(generation: Mapping[str, Any], *, idempotency_key: str) -> dict[str, Any]:
    regenerated_text = str(generation.get("regeneration_text") or "")
    if not regenerated_text:
        raise VNPlayConflictError("script_regenerate_unavailable")
    return {
        **dict(generation),
        "regenerated": True,
        "regeneration_key_hash": _payload_hash({"idempotency_key": idempotency_key})[:16],
        "narrative_text": regenerated_text,
        "dialogue": [{"speaker": "Narrator", "text": regenerated_text}],
    }


def _script_state_payload(*, session_id: int, scene_version: int, position: Mapping[str, Any]) -> dict[str, Any]:
    variables = dict(position.get("variables") or {})
    waiting_choices = _list_of_dicts(position.get("waiting_choices"))
    waiting_choice = None
    if waiting_choices:
        waiting_choice = {"id": position.get("waiting_choice_id"), "choices": waiting_choices}
    return {
        "session_id": session_id,
        "scene_version": scene_version,
        "position": dict(position),
        "variables": variables,
        "waiting_choice": waiting_choice,
        "ended": bool(position.get("ended")),
    }


def _script_public_state_payload(
    *,
    session_id: int,
    scene_version: int,
    position: Mapping[str, Any],
    program: Any,
) -> dict[str, Any]:
    waiting_choices = _script_public_choices(position.get("waiting_choices"))
    waiting_choice = None
    if waiting_choices:
        waiting_choice = {"id": position.get("waiting_choice_id"), "choices": waiting_choices}
    public_position = {"progress_token": _script_progress_token(position)}
    if position.get("waiting_reason"):
        public_position["waiting_reason"] = str(position["waiting_reason"])
    payload = {
        "session_id": session_id,
        "scene_version": scene_version,
        "position": public_position,
        "variables": _script_public_variables(program, position.get("variables")),
        "waiting_choice": waiting_choice,
        "ended": bool(position.get("ended")),
    }
    waiting_generation = _script_public_waiting_generation(position.get("waiting_generation_confirmation"))
    if waiting_generation is not None:
        payload["waiting_generation_confirmation"] = waiting_generation
    active_generation = _script_public_generation(position.get("last_generation"))
    if active_generation is not None:
        payload["active_generation"] = active_generation
    return payload


def _script_public_generation(generation: Any) -> dict[str, Any] | None:
    if not isinstance(generation, Mapping):
        return None
    generation_id = _optional_int(generation.get("generation_id"))
    revision_id = _optional_int(generation.get("revision_id"))
    if generation_id is None or revision_id is None:
        return None
    return {
        "generation_id": generation_id,
        "revision_id": revision_id,
        "generation_point_key": str(generation.get("generation_point_key") or ""),
        "output_schema": str(generation.get("output_schema") or ""),
        "public_output": _mapping_or_empty(generation.get("public_output")),
    }


def _script_public_waiting_generation(waiting_generation: Any) -> dict[str, Any] | None:
    if not isinstance(waiting_generation, Mapping):
        return None
    generation_id = _optional_int(waiting_generation.get("generation_id"))
    request_id = _optional_int(waiting_generation.get("generation_request_id"))
    if generation_id is None or request_id is None:
        return None
    return {
        "generation_id": generation_id,
        "generation_request_id": request_id,
        "generation_point_key": str(waiting_generation.get("generation_point_key") or ""),
        "profile_key": str(waiting_generation.get("profile_key") or ""),
        "output_schema": str(waiting_generation.get("output_schema") or ""),
    }


_SENSITIVE_DEBUG_KEY_FRAGMENTS = ("content", "message", "output", "prompt", "raw", "text")


def _safe_debug_metadata(metadata: Any) -> dict[str, Any]:
    """Return primitive debug metadata without provider prompt/output payloads."""
    if not isinstance(metadata, Mapping):
        return {}
    safe: dict[str, Any] = {}
    for key, value in metadata.items():
        key_text = str(key)
        if any(fragment in key_text.lower() for fragment in _SENSITIVE_DEBUG_KEY_FRAGMENTS):
            safe[key_text] = {"redacted": True}
            continue
        if value is None or isinstance(value, (bool, int, float, str)):
            safe[key_text] = value
        elif isinstance(value, list):
            safe[key_text] = [item for item in value if item is None or isinstance(item, (bool, int, float, str))]
        elif isinstance(value, Mapping):
            safe[key_text] = _safe_debug_metadata(value)
        else:
            safe[key_text] = str(type(value).__name__)
    return safe


def _script_progress_token(position: Mapping[str, Any]) -> str:
    return _payload_hash(
        {
            "label": position.get("label"),
            "index": position.get("index"),
            "waiting_choice_id": position.get("waiting_choice_id"),
            "ended": bool(position.get("ended")),
        }
    )[:16]


def _script_public_variables(program: Any, raw_variables: Any) -> dict[str, Any]:
    if not isinstance(program, Mapping) or not isinstance(raw_variables, Mapping):
        return {}
    definitions = program.get("variables")
    if not isinstance(definitions, Mapping):
        return {}
    public_variables: dict[str, Any] = {}
    for name, value in raw_variables.items():
        if not isinstance(name, str):
            continue
        definition = definitions.get(name)
        if isinstance(definition, Mapping) and definition.get("public") is True:
            public_variables[name] = value
    return public_variables


def _script_public_choices(raw_choices: Any) -> list[dict[str, Any]]:
    return [_script_public_choice(choice) for choice in _list_of_dicts(raw_choices) if choice.get("id")]


def _script_public_choice(choice: Mapping[str, Any]) -> dict[str, Any]:
    public_choice = {"id": str(choice.get("id") or ""), "text": str(choice.get("text") or choice.get("id") or "")}
    if choice.get("source") == "generated":
        public_choice["source"] = "generated"
        generation_id = _optional_int(choice.get("generation_id"))
        revision_id = _optional_int(choice.get("revision_id"))
        if generation_id is not None:
            public_choice["generation_id"] = generation_id
        if revision_id is not None:
            public_choice["revision_id"] = revision_id
    return public_choice


def _generated_choice_variables(choice: Mapping[str, Any]) -> dict[str, Any]:
    metadata = choice.get("metadata")
    return {
        "last_generated_choice.id": str(choice.get("id") or ""),
        "last_generated_choice.text": str(choice.get("text") or choice.get("id") or ""),
        "last_generated_choice.metadata": dict(metadata) if isinstance(metadata, Mapping) else {},
    }


def _script_condition_matches(condition: Any, variables: Mapping[str, Any]) -> bool:
    if condition is None:
        return True
    if not isinstance(condition, Mapping):
        return False
    if "all" in condition and isinstance(condition["all"], Sequence) and not isinstance(condition["all"], (str, bytes)):
        return all(_script_condition_matches(item, variables) for item in condition["all"])
    if "any" in condition and isinstance(condition["any"], Sequence) and not isinstance(condition["any"], (str, bytes)):
        return any(_script_condition_matches(item, variables) for item in condition["any"])
    if "not" in condition:
        return not _script_condition_matches(condition["not"], variables)
    value = variables.get(str(condition.get("var") or ""))
    expected = condition.get("value")
    operator = str(condition.get("op") or "eq")
    if operator == "eq":
        return value == expected
    if operator in {"ne", "neq"}:
        return value != expected
    if operator == "in" and isinstance(expected, Sequence) and not isinstance(expected, (str, bytes)):
        return value in expected
    if operator == "not_in" and isinstance(expected, Sequence) and not isinstance(expected, (str, bytes)):
        return value not in expected
    if operator in {"lt", "lte", "gt", "gte"}:
        return _compare_script_values(value, expected, operator)
    return False


def _compare_script_values(value: Any, expected: Any, operator: str) -> bool:
    if not isinstance(value, (int, float)) or not isinstance(expected, (int, float)):
        return False
    if operator == "lt":
        return value < expected
    if operator == "lte":
        return value <= expected
    if operator == "gt":
        return value > expected
    if operator == "gte":
        return value >= expected
    return False


def _mapping_or_empty(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _payload_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded, usedforsecurity=False).hexdigest()
