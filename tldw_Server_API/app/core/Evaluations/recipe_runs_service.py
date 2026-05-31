"""Minimal parent recipe-run service for the Task 2 service/API slice."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from typing import Any

from loguru import logger
from pydantic import ValidationError as PydanticValidationError

from tldw_Server_API.app.api.v1.schemas.evaluation_recipe_schemas import (
    ConfidenceSummary,
    RecommendationSlot,
    RecipeManifest,
    RecipeRunRecord,
)
from tldw_Server_API.app.api.v1.schemas.evaluation_schemas_unified import RunStatus
from tldw_Server_API.app.core.AuthNZ.settings import is_single_user_mode
from tldw_Server_API.app.core.DB_Management.Evaluations_DB import EvaluationsDatabase
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Evaluations.recipes.dataset_snapshot import (
    build_dataset_content_hash,
    build_dataset_snapshot_ref,
)
from tldw_Server_API.app.core.Evaluations.recipes.registry import (
    RecipeRegistry,
    get_builtin_recipe_registry,
)
from tldw_Server_API.app.core.Evaluations.recipes.reporting import RecipeRunReport

RECIPE_RUN_REUSE_ENTITY_TYPE = "recipe_run_reuse"
_SENSITIVE_RUN_CONFIG_KEYS = {
    "api_key",
    "api_keys",
    "candidate_api_keys",
    "token",
    "access_token",
    "secret",
    "client_secret",
    "password",
    "authorization",
    "auth_header",
}
REQUIRED_RECOMMENDATION_SLOTS: tuple[str, ...] = (
    "best_overall",
    "best_quality",
    "best_cheap",
    "best_local",
)


class RecipeDefinitionNotFoundError(LookupError):
    """Raised when a requested recipe manifest is not registered."""


class RecipeDefinitionNotLaunchableError(RuntimeError):
    """Raised when a requested recipe exists but is not launchable."""

    def __init__(self, recipe_id: str) -> None:
        self.recipe_id = recipe_id
        super().__init__(f"Recipe '{recipe_id}' is not launchable yet.")


class RecipeRunNotFoundError(LookupError):
    """Raised when a recipe run cannot be found."""


class RecipeRunsService:
    """Thin orchestration service around Task 1 recipe-run persistence."""

    def __init__(
        self,
        *,
        db: EvaluationsDatabase,
        user_id: str | None = None,
        recipe_registry: RecipeRegistry | None = None,
    ) -> None:
        self.db = db
        self.user_id = (user_id or "").strip()
        self.recipe_registry = recipe_registry or get_builtin_recipe_registry()

    def list_manifests(self) -> list[RecipeManifest]:
        """Return all manifests in stable recipe-id order."""
        manifests = self.recipe_registry.list_manifests().values()
        return sorted(manifests, key=lambda manifest: manifest.recipe_id)

    def get_manifest(self, recipe_id: str) -> RecipeManifest:
        """Return one manifest or raise a domain-specific lookup error."""
        try:
            return self.recipe_registry.get_manifest(recipe_id)
        except KeyError as exc:
            raise RecipeDefinitionNotFoundError(recipe_id) from exc

    def validate_dataset(
        self,
        recipe_id: str,
        *,
        dataset_id: str | None = None,
        dataset: list[dict[str, Any]] | None = None,
        run_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Lightweight dataset validation that is real enough to gate launch."""
        manifest = self.get_manifest(recipe_id)
        self._ensure_launchable(manifest)
        recipe = self.recipe_registry.get_recipe(recipe_id)
        errors: list[str] = []
        resolved = self._resolve_dataset(dataset_id=dataset_id, dataset=dataset, errors=errors)
        review_payload: dict[str, Any] = {}
        sample_count = len(resolved["samples"])
        normalized_run_config: dict[str, Any] | None = None

        if not errors:
            if run_config is not None:
                normalized_run_config = self._normalize_run_config(recipe_id, run_config)
            validator = getattr(recipe, "validate_dataset", None)
            if callable(validator):
                validator_signature = inspect.signature(validator)
                if "run_config" in validator_signature.parameters:
                    validation = dict(
                        validator(
                            resolved["samples"],
                            run_config=normalized_run_config or run_config,
                        )
                    )
                else:
                    validation = dict(validator(resolved["samples"]))
                errors.extend(validation.get("errors") or [])
                dataset_mode = validation.get("dataset_mode")
                sample_count = int(validation.get("sample_count") or sample_count)
                review_payload = {
                    key: value
                    for key, value in validation.items()
                    if key not in {"valid", "errors", "dataset_mode", "sample_count"}
                }
            else:
                dataset_mode = self._detect_dataset_mode(resolved["samples"], errors)
        else:
            dataset_mode = None

        if dataset_mode in {"labeled", "unlabeled"} and dataset_mode not in manifest.supported_modes:
            errors.append(
                f"Recipe '{recipe_id}' does not support dataset mode '{dataset_mode}'. "
                f"Supported modes: {', '.join(manifest.supported_modes)}."
            )
        elif not callable(getattr(recipe, "validate_dataset", None)):
            for index, sample in enumerate(resolved["samples"]):
                input_value = sample.get("input")
                if input_value is None:
                    errors.append(f"Dataset sample {index} must include an input value.")
                    continue
                if isinstance(input_value, str) and not input_value.strip():
                    errors.append(f"Dataset sample {index} input must not be empty.")

        result = {
            "valid": not errors,
            "errors": errors,
            "dataset_mode": dataset_mode,
            "sample_count": sample_count,
            "dataset_snapshot_ref": resolved["dataset_snapshot_ref"],
            "dataset_content_hash": resolved["dataset_content_hash"],
        }
        result.update(review_payload)
        return result

    def build_reuse_hash(
        self,
        recipe_id: str,
        *,
        dataset_id: str | None = None,
        dataset: list[dict[str, Any]] | None = None,
        run_config: dict[str, Any],
        normalized_run_config: dict[str, Any] | None = None,
        validation: dict[str, Any] | None = None,
    ) -> str:
        """Build the explicit reuse hash inputs required for recipe-run reuse."""
        manifest = self.get_manifest(recipe_id)
        self._ensure_launchable(manifest)
        normalized_run_config = normalized_run_config or self._normalize_run_config(recipe_id, run_config)
        validation = validation or self.validate_dataset(
            recipe_id,
            dataset_id=dataset_id,
            dataset=dataset,
            run_config=normalized_run_config,
        )
        if not validation["valid"]:
            joined = "; ".join(validation["errors"])
            raise ValueError(f"Dataset validation failed: {joined}")
        payload = {
            "recipe_id": manifest.recipe_id,
            "recipe_version": manifest.recipe_version,
            "dataset_snapshot_ref": validation["dataset_snapshot_ref"],
            "dataset_content_hash": validation["dataset_content_hash"],
            "run_config": normalized_run_config,
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def create_run(
        self,
        recipe_id: str,
        *,
        dataset_id: str | None = None,
        dataset: list[dict[str, Any]] | None = None,
        run_config: dict[str, Any],
        force_rerun: bool = False,
    ) -> RecipeRunRecord:
        """Create a pending parent run or reuse a completed run when eligible."""
        manifest = self.get_manifest(recipe_id)
        self._ensure_launchable(manifest)
        normalized_run_config = self._normalize_run_config(recipe_id, run_config)
        validation = self.validate_dataset(
            recipe_id,
            dataset_id=dataset_id,
            dataset=dataset,
            run_config=normalized_run_config,
        )
        if not validation["valid"]:
            joined = "; ".join(validation["errors"])
            raise ValueError(f"Dataset validation failed: {joined}")
        recipe = self.recipe_registry.get_recipe(recipe_id)
        launch_validator = getattr(recipe, "validate_run_config_for_launch", None)
        if callable(launch_validator):
            launch_errors = [
                str(error)
                for error in (launch_validator(normalized_run_config) or [])
                if str(error).strip()
            ]
            if launch_errors:
                joined = "; ".join(launch_errors)
                raise ValueError(f"Run config validation failed: {joined}")
        validation_metadata = {
            key: value
            for key, value in validation.items()
            if key
            not in {
                "valid",
                "errors",
                "dataset_mode",
                "sample_count",
                "dataset_snapshot_ref",
                "dataset_content_hash",
            }
        }
        reuse_hash = self.build_reuse_hash(
            recipe_id,
            dataset_id=dataset_id,
            dataset=dataset,
            run_config=normalized_run_config,
            normalized_run_config=normalized_run_config,
            validation=validation,
        )

        if not force_rerun:
            reusable = self._get_reusable_completed_run(reuse_hash)
            if reusable is not None:
                self._record_reuse_mapping(reusable.run_id, reuse_hash)
                return self.get_run(reusable.run_id)

        run_id = self.db.create_recipe_run(
            recipe_id=manifest.recipe_id,
            recipe_version=manifest.recipe_version,
            status=RunStatus.PENDING,
            dataset_snapshot_ref=validation["dataset_snapshot_ref"],
            dataset_content_hash=validation["dataset_content_hash"],
            metadata={
                "run_config": self._sanitize_run_config_for_metadata(normalized_run_config),
                "run_config_internal": normalized_run_config,
                "reuse_hash": reuse_hash,
                "dataset_mode": validation["dataset_mode"],
                "dataset_id": dataset_id,
                "inline_dataset": (
                    self._normalize_dataset_samples(dataset)
                    if dataset_id is None and dataset is not None
                    else None
                ),
                "owner_user_id": self.user_id,
                "recipe_validation": validation_metadata,
                "review_sample": validation_metadata.get("review_sample"),
            },
        )
        self._record_reuse_mapping(run_id, reuse_hash)
        return self.get_run(run_id)

    def _ensure_launchable(self, manifest: RecipeManifest) -> None:
        if manifest.launchable:
            return
        raise RecipeDefinitionNotLaunchableError(manifest.recipe_id)

    def get_run(self, run_id: str) -> RecipeRunRecord:
        """Fetch one persisted recipe run."""
        record = self.db.get_recipe_run(run_id)
        if record is None:
            raise RecipeRunNotFoundError(run_id)
        owner_user_id = str(record.metadata.get("owner_user_id") or "").strip()
        if owner_user_id:
            if owner_user_id != self.user_id:
                raise RecipeRunNotFoundError(run_id)
        elif self.user_id and not is_single_user_mode():
            raise RecipeRunNotFoundError(run_id)
        return self._public_record(record)

    def get_report(self, run_id: str) -> RecipeRunReport:
        """Fetch a normalized report shell for a recipe run."""
        record = self.get_run(run_id)
        recipe = self.recipe_registry.get_recipe(record.recipe_id)
        report_builder = getattr(recipe, "build_report", None)
        report_inputs = self._extract_recipe_report_inputs(record)
        if callable(report_builder) and report_inputs is not None:
            built_report = dict(report_builder(**report_inputs))
            report_metadata = dict(record.metadata)
            report_metadata["recipe_report"] = built_report
            report_record = record.model_copy(update={"metadata": report_metadata})
            confidence_summary = None
            confidence_payload = built_report.get("confidence_summary")
            if confidence_payload is not None:
                try:
                    confidence_summary = ConfidenceSummary.model_validate(confidence_payload)
                except PydanticValidationError as exc:
                    logger.warning(
                        "Ignoring invalid recipe confidence_summary for run {}: {}",
                        run_id,
                        exc,
                    )
            return RecipeRunReport(
                run=report_record,
                confidence_summary=confidence_summary,
                recommendation_slots=self._normalize_report_slots(
                    built_report.get("recommendation_slots") or {}
                ),
            )
        return RecipeRunReport(
            run=record,
            confidence_summary=record.confidence_summary,
            recommendation_slots=self._normalize_report_slots(record.recommendation_slots),
        )

    def _get_reusable_completed_run(self, reuse_hash: str) -> RecipeRunRecord | None:
        reusable_run_id = self.db.lookup_idempotency(
            RECIPE_RUN_REUSE_ENTITY_TYPE,
            reuse_hash,
            self.user_id,
        )
        if reusable_run_id:
            record = self.db.get_recipe_run(reusable_run_id)
            if record is not None and record.status is RunStatus.COMPLETED:
                return record

        record = self._find_latest_completed_run_by_reuse_hash(reuse_hash)
        if record is None:
            return None
        self._record_reuse_mapping(record.run_id, reuse_hash)
        return record

    def _record_reuse_mapping(self, run_id: str, reuse_hash: str) -> None:
        current_mapping = self.db.lookup_idempotency(
            RECIPE_RUN_REUSE_ENTITY_TYPE,
            reuse_hash,
            self.user_id,
        )
        if current_mapping == run_id:
            return
        if not current_mapping:
            self.db.record_idempotency(
                RECIPE_RUN_REUSE_ENTITY_TYPE,
                reuse_hash,
                run_id,
                self.user_id,
            )
            return

        uid = self.user_id or ""
        self.db.update_idempotency_entity(
            RECIPE_RUN_REUSE_ENTITY_TYPE,
            reuse_hash,
            run_id,
            uid,
        )

    def _find_latest_completed_run_by_reuse_hash(self, reuse_hash: str) -> RecipeRunRecord | None:
        return self.db.find_latest_recipe_run_by_reuse_hash(
            reuse_hash=reuse_hash,
            owner_user_id=self.user_id or None,
            allow_legacy_unowned=bool(self.user_id) and is_single_user_mode(),
        )

    def _resolve_dataset(
        self,
        *,
        dataset_id: str | None,
        dataset: list[dict[str, Any]] | None,
        errors: list[str],
    ) -> dict[str, Any]:
        if dataset_id and dataset is not None:
            errors.append("Provide either dataset_id or dataset, not both.")
            return {
                "samples": [],
                "dataset_snapshot_ref": None,
                "dataset_content_hash": None,
            }
        if dataset_id:
            dataset_row = self.db.get_dataset(dataset_id, created_by=self.user_id or None)
            if not dataset_row:
                errors.append(f"Dataset '{dataset_id}' was not found.")
                return {
                    "samples": [],
                    "dataset_snapshot_ref": None,
                    "dataset_content_hash": None,
                }
            samples = self._normalize_dataset_samples(dataset_row.get("samples"))
            created_value = dataset_row.get("created") or dataset_row.get("created_at") or "unknown"
            return {
                "samples": samples,
                "dataset_snapshot_ref": build_dataset_snapshot_ref(dataset_id, created_value),
                "dataset_content_hash": build_dataset_content_hash(samples),
            }
        if dataset is None:
            errors.append("A dataset_id or inline dataset is required.")
            return {
                "samples": [],
                "dataset_snapshot_ref": None,
                "dataset_content_hash": None,
            }
        samples = self._normalize_dataset_samples(dataset)
        return {
            "samples": samples,
            "dataset_snapshot_ref": None,
            "dataset_content_hash": build_dataset_content_hash(samples),
        }

    def _normalize_dataset_samples(self, samples: Any) -> list[dict[str, Any]]:
        if samples is None:
            return []
        normalized_samples = [dict(sample) for sample in list(samples)]
        return [
            sample for sample in normalized_samples
            if self._synthetic_sample_is_active(sample)
        ]

    def _synthetic_sample_is_active(self, sample: dict[str, Any]) -> bool:
        metadata = sample.get("metadata")
        synthetic_eval = sample.get("synthetic_eval")
        if synthetic_eval is None and isinstance(metadata, dict):
            synthetic_eval = metadata.get("synthetic_eval")
        if not isinstance(synthetic_eval, dict):
            return True
        review_state = str(synthetic_eval.get("review_state") or "").strip().lower()
        promotion_state = str(synthetic_eval.get("promotion_state") or "").strip().lower()
        if review_state != "approved":
            return False
        if promotion_state != "promoted":
            return False
        return True

    def _detect_dataset_mode(self, samples: list[dict[str, Any]], errors: list[str]) -> str | None:
        if not samples:
            errors.append("Dataset must contain at least one sample.")
            return None

        has_expected = [
            sample.get("expected") is not None
            for sample in samples
        ]
        if all(has_expected):
            return "labeled"
        if not any(has_expected):
            return "unlabeled"

        errors.append(
            "Dataset must use a consistent labeling mode; do not mix labeled and unlabeled samples."
        )
        return "mixed"

    def _normalize_run_config(self, recipe_id: str, run_config: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(run_config, dict):
            raise ValueError("run_config must be an object.")

        recipe = self.recipe_registry.get_recipe(recipe_id)
        normalizer = getattr(recipe, "normalize_run_config", None)
        if callable(normalizer):
            normalized = dict(normalizer(run_config))
            if recipe_id == "rag_answer_quality":
                normalized["candidates"] = self._normalize_rag_answer_quality_candidates(
                    run_config.get("candidates")
                )
            return normalized

        candidate_model_ids = run_config.get("candidate_model_ids") or []
        if not isinstance(candidate_model_ids, list) or not candidate_model_ids:
            raise ValueError("run_config.candidate_model_ids must contain at least one model id.")

        normalized_candidate_model_ids: list[str] = []
        seen_model_ids: set[str] = set()
        for model_id in candidate_model_ids:
            normalized_model_id = str(model_id).strip()
            if not normalized_model_id or normalized_model_id in seen_model_ids:
                continue
            seen_model_ids.add(normalized_model_id)
            normalized_candidate_model_ids.append(normalized_model_id)
        if not normalized_candidate_model_ids:
            raise ValueError("run_config.candidate_model_ids must contain at least one model id.")

        comparison_mode = str(run_config.get("comparison_mode") or "").strip()
        if not comparison_mode:
            raise ValueError("run_config.comparison_mode is required.")

        weights = run_config.get("weights") or {}
        if not isinstance(weights, dict):
            raise ValueError("run_config.weights must be an object.")
        normalized_weights: dict[str, float] = {}
        for key, value in weights.items():
            normalized_key = str(key).strip()
            try:
                normalized_weights[normalized_key] = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"run_config.weights.{normalized_key or '<empty>'} must be numeric."
                ) from exc

        return {
            "candidate_model_ids": normalized_candidate_model_ids,
            "judge_config": self._normalize_mapping(run_config.get("judge_config")),
            "prompts": {
                str(key): str(value)
                for key, value in self._normalize_mapping(run_config.get("prompts")).items()
            },
            "weights": normalized_weights,
            "comparison_mode": comparison_mode,
            "source_normalization": self._normalize_mapping(run_config.get("source_normalization")),
            "context_policy": self._normalize_mapping(run_config.get("context_policy")),
            "execution_policy": self._normalize_mapping(run_config.get("execution_policy")),
        }

    def _normalize_rag_answer_quality_candidates(self, value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        normalized_candidates: list[dict[str, Any]] = []
        for candidate in value:
            if not isinstance(candidate, dict):
                continue
            normalized_candidates.append(
                {
                    "candidate_id": candidate.get("candidate_id"),
                    "provider": candidate.get("provider"),
                    "model": candidate.get("model"),
                    "generation_model": candidate.get("generation_model"),
                    "prompt_variant": candidate.get("prompt_variant"),
                    "formatting_citation_mode": candidate.get("formatting_citation_mode"),
                    "is_local": candidate.get("is_local"),
                    "cost_usd": candidate.get("cost_usd"),
                    "generation_config": self._normalize_mapping(candidate.get("generation_config")),
                }
            )
        return normalized_candidates

    def _normalize_mapping(self, value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("Recipe run config sections must be objects when provided.")
        return dict(value)

    def _normalize_report_slots(
        self,
        recommendation_slots: dict[str, RecommendationSlot] | dict[str, dict[str, Any]],
    ) -> dict[str, RecommendationSlot]:
        normalized = {
            slot_name: RecommendationSlot.model_validate(slot_value)
            for slot_name, slot_value in recommendation_slots.items()
        }
        for slot_name in REQUIRED_RECOMMENDATION_SLOTS:
            normalized.setdefault(
                slot_name,
                RecommendationSlot(
                    candidate_run_id=None,
                    reason_code="not_available",
                    explanation=f"No recommendation has been recorded for '{slot_name}'.",
                ),
            )
        return normalized

    def _public_record(self, record: RecipeRunRecord) -> RecipeRunRecord:
        metadata = dict(record.metadata or {})
        metadata.pop("run_config_internal", None)
        return record.model_copy(update={"metadata": metadata})

    def _sanitize_run_config_for_metadata(self, value: Any) -> Any:
        if isinstance(value, dict):
            sanitized: dict[str, Any] = {}
            for key, item in value.items():
                key_str = str(key)
                lowered = key_str.lower()
                if lowered in _SENSITIVE_RUN_CONFIG_KEYS or lowered.endswith(
                    ("_api_key", "_token", "_secret", "_password")
                ):
                    sanitized[key_str] = self._redact_sensitive_value(item)
                else:
                    sanitized[key_str] = self._sanitize_run_config_for_metadata(item)
            return sanitized
        if isinstance(value, list):
            return [self._sanitize_run_config_for_metadata(item) for item in value]
        return value

    def _redact_sensitive_value(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): "[REDACTED]" for key in value.keys()}
        if isinstance(value, list):
            return ["[REDACTED]" for _ in value]
        return "[REDACTED]"

    def _extract_recipe_report_inputs(self, record: RecipeRunRecord) -> dict[str, Any] | None:
        explicit_inputs = record.metadata.get("recipe_report_inputs")
        if isinstance(explicit_inputs, dict):
            return dict(explicit_inputs)

        candidate_results = record.metadata.get("candidate_results")
        dataset_mode = record.metadata.get("dataset_mode")
        review_sample = (
            record.metadata.get("review_sample")
            or (record.metadata.get("recipe_validation") or {}).get("review_sample")
        )
        if candidate_results is None or not dataset_mode:
            return None
        return {
            "dataset_mode": dataset_mode,
            "review_sample": review_sample or {
                "required": False,
                "sample_size": 0,
                "sample_query_ids": [],
            },
            "candidate_results": candidate_results,
        }


def get_recipe_runs_service_for_user(user_id: str | int | None) -> RecipeRunsService:
    """Build a recipe-runs service bound to the appropriate evaluations database."""
    db_path = os.getenv("EVALUATIONS_TEST_DB_PATH")
    if not db_path:
        db_path = str(DatabasePaths.get_evaluations_db_path(user_id))
    return RecipeRunsService(
        db=EvaluationsDatabase(db_path),
        user_id=str(user_id) if user_id is not None else None,
    )
