"""Pure VN script playtest traversal."""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from tldw_Server_API.app.core.VN_Play.errors import VNPlayTurnError
from tldw_Server_API.app.core.VN_Play.script_runtime import (
    MAX_SCRIPT_EXECUTION_STEPS,
    _execute_script_program,
    _initial_script_position,
    _payload_hash,
    _script_progress_token,
)

SCHEMA_VERSION = "vn_script_playtest.v1"
PLAYTEST_SEMANTICS_VERSION = "vn_script_playtest_paths.v1"
DEFAULT_MAX_PATHS = 100
MAX_MAX_PATHS = 1000
MAX_MAX_STEPS = 5000

PlaytestSource = Literal["stored_draft", "supplied_draft", "published_version"]
ValidationContextSource = Literal["current_draft_context", "published_version_snapshot"]


def build_script_playtest(
    program: Mapping[str, Any],
    *,
    source: PlaytestSource,
    script_id: int | None = None,
    base_revision: int | None = None,
    version_id: int | None = None,
    validation_diagnostics: Mapping[str, Any] | None = None,
    validation_context_source: ValidationContextSource = "current_draft_context",
    max_steps: int = MAX_SCRIPT_EXECUTION_STEPS,
    max_paths: int = DEFAULT_MAX_PATHS,
    seed: str | None = "playtest",
) -> dict[str, Any]:
    """Traverse deterministic script paths without creating runtime state or calling models."""
    bounded_max_steps = _bounded_int(max_steps, minimum=1, maximum=MAX_MAX_STEPS)
    bounded_max_paths = _bounded_int(max_paths, minimum=1, maximum=MAX_MAX_PATHS)
    validation = _validation_payload(validation_diagnostics)
    diagnostics = {"errors": [], "warnings": []}
    paths: list[dict[str, Any]] = []
    choice_boundaries: list[dict[str, Any]] = []
    generation_boundaries: list[dict[str, Any]] = []
    endings: list[dict[str, Any]] = []
    visited_labels: set[str] = set()
    seen_states: set[str] = set()
    queue: deque[tuple[dict[str, Any], list[dict[str, Any]]]] = deque()
    queue.append((_initial_script_position(program), []))
    truncated = False
    total_steps = 0

    while queue and len(paths) < bounded_max_paths:
        position, decisions = queue.popleft()
        path_id = f"path:{len(paths) + 1}"
        state_key = _state_key(position)
        if state_key in seen_states:
            diagnostics["warnings"].append(
                _diag(
                    "playtest_loop_detected",
                    "Traversal reached an already visited script position.",
                    "$.labels",
                    {"progress_token": _script_progress_token(position), "path_id": path_id},
                )
            )
            endings.append(_ending(path_id, position, "loop_detected"))
            paths.append(_path(path_id, decisions, position, status="loop_detected"))
            continue
        seen_states.add(state_key)

        try:
            execution = _execute_script_program(
                program,
                position,
                choice_id=None,
                seed=seed,
                max_steps=bounded_max_steps,
            )
        except VNPlayTurnError as exc:
            diagnostics["errors"].append(
                _diag(
                    "playtest_runtime_error",
                    str(exc) or "Script runtime traversal failed.",
                    "$.labels",
                    {"path_id": path_id, "progress_token": _script_progress_token(position)},
                )
            )
            paths.append(_path(path_id, decisions, position, status="runtime_error"))
            continue

        total_steps += 1
        next_position = _mapping_or_empty(execution.get("position"))
        label = str(next_position.get("label") or position.get("label") or "")
        if label:
            visited_labels.add(label)

        pending_generation = execution.get("pending_generation")
        if isinstance(pending_generation, Mapping):
            boundary = _generation_boundary(path_id, pending_generation, decisions, next_position)
            generation_boundaries.append(boundary)
            paths.append(_path(path_id, decisions, next_position, status="generation_boundary", boundary=boundary))
            continue

        choices = _list_of_dicts(execution.get("visible_choices"))
        if choices:
            boundary = _choice_boundary(path_id, choices, decisions, next_position)
            choice_boundaries.append(boundary)
            paths.append(_path(path_id, decisions, next_position, status="choice_boundary", boundary=boundary))
            for choice in choices:
                if len(paths) + len(queue) >= bounded_max_paths:
                    truncated = True
                    continue
                queue.append(
                    (
                        {
                            "label": str(choice.get("target") or ""),
                            "index": 0,
                            "ended": False,
                            "variables": dict(next_position.get("variables") or {}),
                        },
                        [*decisions, {"choice_id": str(choice["id"]), "text": str(choice.get("text") or "")}],
                    )
                )
            continue

        if bool(next_position.get("ended")):
            ending = _ending(path_id, next_position, "ended")
            endings.append(ending)
            paths.append(_path(path_id, decisions, next_position, status="ended", boundary=ending))
        else:
            truncated = True
            diagnostics["warnings"].append(
                _diag(
                    "playtest_truncated",
                    "Script segment reached the max step limit before a boundary.",
                    "$.labels",
                    {"path_id": path_id, "max_steps": bounded_max_steps},
                )
            )
            paths.append(_path(path_id, decisions, next_position, status="truncated"))

    if queue:
        truncated = True
        diagnostics["warnings"].append(
            _diag(
                "playtest_truncated",
                "Traversal reached the max path limit before all paths were explored.",
                "$.labels",
                {"max_paths": bounded_max_paths, "remaining_paths": len(queue)},
            )
        )

    all_labels = _program_labels(program)
    unvisited_labels = sorted(all_labels - visited_labels)
    valid = bool(validation.get("valid"))
    runtime_ready = valid and not diagnostics["errors"] and not truncated
    return {
        "schema_version": SCHEMA_VERSION,
        "playtest_semantics_version": PLAYTEST_SEMANTICS_VERSION,
        "program_schema_version": str(program.get("schema_version") or ""),
        "script_id": script_id,
        "source": source,
        "base_revision": base_revision,
        "version_id": version_id,
        "validation_context_source": validation_context_source,
        "valid": valid,
        "runtime_ready": runtime_ready,
        "truncated": truncated,
        "limits": {"max_steps": bounded_max_steps, "max_paths": bounded_max_paths},
        "summary": {
            "path_count": len(paths),
            "choice_boundary_count": len(choice_boundaries),
            "generation_boundary_count": len(generation_boundaries),
            "ending_count": len(endings),
            "visited_label_count": len(visited_labels),
            "unvisited_label_count": len(unvisited_labels),
            "execution_segments": total_steps,
        },
        "visited_labels": sorted(visited_labels),
        "unvisited_labels": unvisited_labels,
        "paths": paths,
        "choice_boundaries": choice_boundaries,
        "generation_boundaries": generation_boundaries,
        "endings": endings,
        "diagnostics": diagnostics,
        "validation_diagnostics": validation,
    }


def _choice_boundary(
    path_id: str,
    choices: Sequence[Mapping[str, Any]],
    decisions: Sequence[Mapping[str, Any]],
    position: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "path_id": path_id,
        "progress_token": _script_progress_token(position),
        "label": str(position.get("label") or ""),
        "op_index": int(position.get("index") or 0),
        "decisions": [dict(item) for item in decisions],
        "choice_id": str(position.get("waiting_choice_id") or ""),
        "choices": [dict(choice) for choice in choices],
    }


def _generation_boundary(
    path_id: str,
    generation: Mapping[str, Any],
    decisions: Sequence[Mapping[str, Any]],
    position: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "path_id": path_id,
        "progress_token": _script_progress_token(position),
        "label": str(generation.get("label") or position.get("label") or ""),
        "op_index": int(generation.get("index") or 0),
        "generation_id": str(generation.get("id") or ""),
        "profile_key": str(generation.get("profile_key") or "default"),
        "output_schema": str(generation.get("output_schema") or "narrative_dialogue"),
        "requires_user_confirm": bool(generation.get("requires_user_confirm")),
        "prompt_hash": str(generation.get("prompt_hash") or ""),
        "decisions": [dict(item) for item in decisions],
    }


def _ending(path_id: str, position: Mapping[str, Any], reason: str) -> dict[str, Any]:
    return {
        "path_id": path_id,
        "reason": reason,
        "progress_token": _script_progress_token(position),
        "label": str(position.get("label") or ""),
        "op_index": int(position.get("index") or 0),
    }


def _path(
    path_id: str,
    decisions: Sequence[Mapping[str, Any]],
    position: Mapping[str, Any],
    *,
    status: str,
    boundary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "id": path_id,
        "status": status,
        "decisions": [dict(item) for item in decisions],
        "progress_token": _script_progress_token(position),
        "label": str(position.get("label") or ""),
        "op_index": int(position.get("index") or 0),
    }
    if boundary is not None:
        payload["boundary"] = dict(boundary)
    return payload


def _state_key(position: Mapping[str, Any]) -> str:
    return _payload_hash(
        {
            "label": position.get("label"),
            "index": position.get("index"),
            "waiting_choice_id": position.get("waiting_choice_id"),
            "ended": bool(position.get("ended")),
            "variables": _json_safe(position.get("variables")),
        }
    )


def _program_labels(program: Mapping[str, Any]) -> set[str]:
    labels = program.get("labels")
    if not isinstance(labels, Mapping):
        return set()
    return {str(label) for label in labels if isinstance(label, str)}


def _validation_payload(validation: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(validation, Mapping):
        return {"valid": False, "errors": [{"code": "validation_missing"}], "warnings": []}
    return {
        "valid": bool(validation.get("valid")),
        "errors": _list_of_dicts(validation.get("errors")),
        "warnings": _list_of_dicts(validation.get("warnings")),
    }


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value, sort_keys=True)
        return value
    except TypeError:
        return str(value)


def _mapping_or_empty(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _bounded_int(value: int, *, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = minimum
    return max(minimum, min(parsed, maximum))


def _diag(code: str, message: str, path: str, details: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "code": code,
        "severity": "warning" if code in {"playtest_loop_detected", "playtest_truncated"} else "error",
        "message": message,
        "path": path,
        "details": dict(details or {}),
    }
