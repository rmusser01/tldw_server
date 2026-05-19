"""Persona dialogue-tree robustness recipe helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tldw_Server_API.app.api.v1.schemas.evaluation_recipe_schemas import RecipeManifest
from tldw_Server_API.app.core.Evaluations.recipes.base import RecipeDefinition
from tldw_Server_API.app.core.Persona.dialogue_tree_context import redact_sensitive_payload


_DIRECT_PERSONA_KEYS = ("persona", "persona_target")
_DIRECT_CHARACTER_KEYS = ("character", "character_target")
_PERSONA_LIST_KEYS = ("personas", "persona_targets")
_CHARACTER_LIST_KEYS = ("characters", "character_targets")
_TRUE_BOOL_STRINGS = {"1", "true", "yes", "y", "on"}
_FALSE_BOOL_STRINGS = {"0", "false", "no", "n", "off"}


class PersonaDialogueTreeRobustnessRecipe(RecipeDefinition):
    """Offline persona/character robustness recipe backed by Evaluations runs."""

    manifest = RecipeManifest(
        recipe_id="persona_dialogue_tree_robustness",
        recipe_version="1",
        name="Persona Dialogue-Tree Robustness",
        description=(
            "Run deterministic dialogue-tree robustness scenarios against persona and "
            "character targets using the existing Evaluations recipe-run pipeline."
        ),
        supported_modes=["labeled", "unlabeled"],
        tags=["persona", "character", "dialogue-tree", "robustness", "recipe-v1"],
        capabilities={
            "execution": "offline",
            "trace_artifacts": True,
            "hard_pruners": "deterministic_only",
        },
        default_run_config={
            "targets": [],
            "include_trace_artifacts": True,
        },
    )

    def normalize_run_config(self, run_config: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(run_config, dict):
            raise ValueError("run_config must be an object.")
        target_errors = _target_config_errors(run_config)
        if target_errors:
            raise ValueError("; ".join(target_errors))
        return {
            "targets": _redact_targets(_normalize_targets(run_config)),
            "include_trace_artifacts": _coerce_run_config_bool(
                run_config.get("include_trace_artifacts", True),
                field_name="include_trace_artifacts",
            ),
        }

    def validate_dataset(
        self,
        dataset: list[dict[str, Any]],
        *,
        run_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        errors: list[str] = []
        raw_samples = list(dataset or [])
        samples: list[dict[str, Any]] = []
        for index, sample in enumerate(raw_samples):
            if not isinstance(sample, Mapping):
                errors.append(f"Scenario {index} must be an object, got {type(sample).__name__}.")
                continue
            samples.append(dict(sample))
        errors.extend(_target_config_errors(run_config or {}))
        targets = _normalize_targets(run_config or {})
        if not targets:
            errors.append("Run config must include at least one persona or character target.")

        if not samples:
            errors.append("Dataset must include at least one scenario.")
            return {
                "valid": False,
                "errors": errors,
                "dataset_mode": None,
                "sample_count": len(raw_samples),
                "target_count": len(targets),
                "review_sample": {"required": False, "sample_size": 0, "sample_ids": []},
            }

        labeled_flags: list[bool] = []
        sample_ids: list[str] = []
        for index, sample in enumerate(samples):
            sample_id = _extract_case_id(sample, index)
            sample_ids.append(sample_id)
            if not _extract_prompt(sample):
                errors.append(f"Scenario {index} must include a non-empty prompt or input.")
            candidates = sample.get("candidates")
            if not isinstance(candidates, list) or not candidates:
                errors.append(f"Scenario {index} must include at least one candidate.")
            else:
                for candidate_index, candidate in enumerate(candidates):
                    if not isinstance(candidate, dict):
                        errors.append(
                            f"Scenario {index} candidate {candidate_index} must be an object."
                        )
            metadata = sample.get("metadata")
            if metadata is not None and not isinstance(metadata, Mapping):
                errors.append(f"Scenario {index} metadata must be an object.")
            labeled_flags.append(_has_label(sample))

        dataset_mode = _detect_dataset_mode(labeled_flags)
        if dataset_mode == "mixed":
            errors.append("Dataset must use a consistent labeling mode for robustness scenarios.")

        review_sample = (
            _reserve_review_sample(sample_ids)
            if dataset_mode == "unlabeled"
            else {"required": False, "sample_size": 0, "sample_ids": []}
        )
        return {
            "valid": not errors,
            "errors": errors,
            "dataset_mode": dataset_mode,
            "sample_count": len(raw_samples),
            "target_count": len(targets),
            "review_sample": review_sample,
        }

    def build_report(
        self,
        *,
        dataset_mode: str,
        review_sample: dict[str, Any],
        target_results: list[dict[str, Any]],
        trace_artifacts: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        normalized_targets = [
            dict(target_result)
            for target_result in (target_results or [])
            if isinstance(target_result, dict)
        ]
        summary = {
            "target_count": len(normalized_targets),
            "total_cases": 0,
            "hard_prune_count": 0,
            "soft_prune_count": 0,
            "selected_trajectory_count": 0,
            "skipped_scorer_count": 0,
            "trace_artifact_count": 0,
        }
        selected_trajectories: list[dict[str, Any]] = []
        trace_refs: list[dict[str, Any]] = []

        for target_result in normalized_targets:
            target_id = str(target_result.get("target_id") or "")
            target_summary = _as_mapping(target_result.get("summary"))
            summary["total_cases"] += int(target_summary.get("total_cases") or 0)
            summary["hard_prune_count"] += int(target_summary.get("hard_prune_count") or 0)
            summary["soft_prune_count"] += int(target_summary.get("soft_prune_count") or 0)
            summary["selected_trajectory_count"] += int(
                target_summary.get("selected_trajectory_count") or 0
            )
            summary["skipped_scorer_count"] += int(target_summary.get("skipped_scorer_count") or 0)

            for case in _as_mapping_list(target_result.get("cases")):
                selected_node_id = case.get("selected_node_id")
                if selected_node_id is None:
                    continue
                selected_trajectories.append(
                    {
                        "target_id": target_id,
                        "case_id": str(case.get("case_id") or ""),
                        "selected_node_id": str(selected_node_id),
                    }
                )

            for ref in _as_mapping_list(target_result.get("trace_artifact_refs")):
                trace_refs.append(
                    {
                        "target_id": target_id,
                        "artifact_id": str(ref.get("artifact_id") or ""),
                        "case_id": str(ref.get("case_id") or ""),
                    }
                )

        seen_trace_refs = {
            (ref["target_id"], ref["artifact_id"], ref["case_id"])
            for ref in trace_refs
        }
        existing_target_by_artifact_case = {
            (ref["artifact_id"], ref["case_id"]): ref["target_id"]
            for ref in trace_refs
        }
        for artifact in _as_mapping_list(trace_artifacts):
            artifact_id = str(artifact.get("artifact_id") or "")
            case_id = str(artifact.get("case_id") or "")
            target_id = str(artifact.get("target_id") or "")
            if not target_id:
                target_id = existing_target_by_artifact_case.get((artifact_id, case_id), "")
            synthesized_ref = {
                "target_id": target_id,
                "artifact_id": artifact_id,
                "case_id": case_id,
            }
            ref_key = (
                synthesized_ref["target_id"],
                synthesized_ref["artifact_id"],
                synthesized_ref["case_id"],
            )
            if ref_key not in seen_trace_refs:
                trace_refs.append(synthesized_ref)
                seen_trace_refs.add(ref_key)

        summary["trace_artifact_count"] = len(trace_refs)

        return {
            "dataset_mode": dataset_mode,
            "review_sample": review_sample,
            "summary": summary,
            "target_results": [
                {
                    "target_id": str(target_result.get("target_id") or ""),
                    "persona_id": target_result.get("persona_id"),
                    "character_id": target_result.get("character_id"),
                    "summary": _as_mapping(target_result.get("summary")),
                    "cases": _as_mapping_list(target_result.get("cases")),
                    "trace_artifact_refs": _as_mapping_list(
                        target_result.get("trace_artifact_refs")
                    ),
                }
                for target_result in normalized_targets
            ],
            "selected_trajectories": selected_trajectories,
            "trace_artifact_refs": trace_refs,
        }


def _normalize_targets(run_config: Mapping[str, Any]) -> list[dict[str, Any]]:
    explicit_targets = run_config.get("targets")
    if isinstance(explicit_targets, dict):
        normalized = _normalize_target(explicit_targets, index=0)
        return [normalized] if _target_has_payload(normalized) else []
    if isinstance(explicit_targets, list):
        return [
            normalized
            for index, target in enumerate(explicit_targets)
            if isinstance(target, Mapping)
            for normalized in [_normalize_target(target, index=index)]
            if _target_has_payload(normalized)
        ]

    persona = _first_mapping(run_config, _DIRECT_PERSONA_KEYS)
    character = _first_mapping(run_config, _DIRECT_CHARACTER_KEYS)
    if persona is not None or character is not None:
        return [_build_target(persona=persona, character=character, index=0)]

    personas = _first_mapping_list(run_config, _PERSONA_LIST_KEYS)
    characters = _first_mapping_list(run_config, _CHARACTER_LIST_KEYS)
    targets: list[dict[str, Any]] = []
    if personas and characters:
        for persona_index, persona_payload in enumerate(personas):
            for character_index, character_payload in enumerate(characters):
                targets.append(
                    _build_target(
                        persona=persona_payload,
                        character=character_payload,
                        index=(persona_index * len(characters)) + character_index,
                    )
                )
        return targets
    if personas:
        return [
            _build_target(persona=persona_payload, character=None, index=index)
            for index, persona_payload in enumerate(personas)
        ]
    if characters:
        return [
            _build_target(persona=None, character=character_payload, index=index)
            for index, character_payload in enumerate(characters)
        ]
    return []


def _coerce_run_config_bool(value: Any, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in _TRUE_BOOL_STRINGS:
            return True
        if normalized in _FALSE_BOOL_STRINGS:
            return False
    raise ValueError(f"{field_name} must be a boolean.")


def _target_config_errors(run_config: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if "targets" in run_config:
        explicit_targets = run_config.get("targets")
        if isinstance(explicit_targets, Mapping):
            errors.extend(_nested_target_payload_errors(explicit_targets, prefix="targets"))
            normalized = _normalize_target(explicit_targets, index=0)
            if not _target_has_payload(normalized):
                errors.append("targets must include a persona or character target.")
            return errors
        if isinstance(explicit_targets, list):
            for index, target in enumerate(explicit_targets):
                if not isinstance(target, Mapping):
                    errors.append(f"targets[{index}] must be an object.")
                    continue
                errors.extend(
                    _nested_target_payload_errors(target, prefix=f"targets[{index}]")
                )
                normalized = _normalize_target(target, index=index)
                if not _target_has_payload(normalized):
                    errors.append(
                        f"targets[{index}] must include a persona or character target."
                    )
            return errors
        errors.append("targets must be an object or list.")
        return errors

    for key in _DIRECT_PERSONA_KEYS + _DIRECT_CHARACTER_KEYS:
        if key in run_config and run_config.get(key) is not None and not isinstance(
            run_config.get(key),
            Mapping,
        ):
            errors.append(f"{key} must be an object.")
        elif key in run_config and isinstance(run_config.get(key), Mapping) and not run_config.get(key):
            errors.append(f"{key} must not be empty.")

    for key in _PERSONA_LIST_KEYS + _CHARACTER_LIST_KEYS:
        if key not in run_config:
            continue
        value = run_config.get(key)
        if not isinstance(value, list):
            errors.append(f"{key} must be a list.")
            continue
        for index, item in enumerate(value):
            if not isinstance(item, Mapping):
                errors.append(f"{key}[{index}] must be an object.")
            elif not item:
                errors.append(f"{key}[{index}] must include target fields.")
    return errors


def _nested_target_payload_errors(
    target: Mapping[str, Any],
    *,
    prefix: str,
) -> list[str]:
    errors: list[str] = []
    for key in (
        "persona",
        "persona_target",
        "character",
        "character_target",
    ):
        if key in target and target.get(key) is not None and not isinstance(
            target.get(key),
            Mapping,
        ):
            errors.append(f"{prefix}.{key} must be an object.")
        elif key in target and isinstance(target.get(key), Mapping) and not target.get(key):
            errors.append(f"{prefix}.{key} must not be empty.")
    return errors


def _normalize_target(target: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    persona = _optional_mapping(target.get("persona") or target.get("persona_target"))
    character = _optional_mapping(target.get("character") or target.get("character_target"))
    target_id = str(target.get("target_id") or "").strip()
    return _build_target(
        persona=persona,
        character=character,
        index=index,
        explicit_target_id=target_id or None,
    )


def _build_target(
    *,
    persona: dict[str, Any] | None,
    character: dict[str, Any] | None,
    index: int,
    explicit_target_id: str | None = None,
) -> dict[str, Any]:
    target_id = explicit_target_id or _derive_target_id(
        persona=persona,
        character=character,
        index=index,
    )
    return {
        "target_id": target_id,
        "persona": persona or {},
        "character": character or {},
    }


def _derive_target_id(
    *,
    persona: dict[str, Any] | None,
    character: dict[str, Any] | None,
    index: int,
) -> str:
    persona_id = str((persona or {}).get("id") or "").strip()
    character_id = str((character or {}).get("id") or "").strip()
    if persona_id and character_id:
        return f"{persona_id}:{character_id}"
    if persona_id:
        return persona_id
    if character_id:
        return character_id
    return f"target-{index + 1}"


def _target_has_payload(target: dict[str, Any]) -> bool:
    return bool(target.get("persona") or target.get("character"))


def _redact_targets(targets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    redacted_targets: list[dict[str, Any]] = []
    for target in targets:
        redacted = redact_sensitive_payload(target)
        if isinstance(redacted, dict):
            redacted_targets.append(redacted)
    return redacted_targets


def _first_mapping(
    payload: Mapping[str, Any],
    keys: tuple[str, ...],
) -> dict[str, Any] | None:
    for key in keys:
        value = _optional_mapping(payload.get(key))
        if value:
            return value
    return None


def _first_mapping_list(
    payload: Mapping[str, Any],
    keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    for key in keys:
        values = _as_mapping_list(payload.get(key))
        if values:
            return values
    return []


def _optional_mapping(value: Any) -> dict[str, Any] | None:
    if isinstance(value, Mapping):
        return {str(key): sub_value for key, sub_value in value.items()}
    return None


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(key): sub_value for key, sub_value in value.items()}
    return {}


def _as_mapping_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [_as_mapping(item) for item in value if isinstance(item, Mapping) and item]


def _extract_case_id(sample: Mapping[str, Any], index: int) -> str:
    for key in ("case_id", "scenario_id", "sample_id", "id"):
        value = sample.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return f"scenario-{index + 1}"


def _extract_prompt(sample: Mapping[str, Any]) -> str:
    for key in ("prompt", "input", "user_message"):
        value = sample.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _has_label(sample: Mapping[str, Any]) -> bool:
    return any(
        sample.get(key) is not None
        for key in ("expected", "expected_candidate_id", "expected_behavior", "reference")
    )


def _detect_dataset_mode(labeled_flags: list[bool]) -> str | None:
    if not labeled_flags:
        return None
    if all(labeled_flags):
        return "labeled"
    if not any(labeled_flags):
        return "unlabeled"
    return "mixed"


def _reserve_review_sample(sample_ids: list[str]) -> dict[str, Any]:
    if not sample_ids:
        return {"required": False, "sample_size": 0, "sample_ids": []}
    sample_size = min(len(sample_ids), min(25, max(1, len(sample_ids) // 5)))
    return {
        "required": True,
        "sample_size": sample_size,
        "sample_ids": sample_ids[:sample_size],
    }


__all__ = ["PersonaDialogueTreeRobustnessRecipe"]
