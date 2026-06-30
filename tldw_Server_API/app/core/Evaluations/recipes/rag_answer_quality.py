"""V1 RAG answer-quality recipe helpers."""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Mapping
from typing import Any

from tldw_Server_API.app.api.v1.schemas.evaluation_recipe_schemas import (
    ConfidenceSummary,
    RecipeManifest,
    RecommendationSlot,
)
from tldw_Server_API.app.core.Evaluations.recipes.base import RecipeDefinition

_SUPPORTED_EVALUATION_MODES = {"fixed_context", "live_end_to_end"}
_SUPPORTED_SUPERVISION_MODES = {"rubric", "reference_answer", "pairwise", "mixed"}
_SUPPORTED_CANDIDATE_DIMENSIONS = {
    "generation_model",
    "prompt_variant",
    "formatting_citation_mode",
}
_SUPPORTED_WEIGHT_KEYS = (
    "grounding",
    "answer_relevance",
    "format_style",
    "abstention_behavior",
)
_DEFAULT_WEIGHTS = {
    "grounding": 0.4,
    "answer_relevance": 0.3,
    "format_style": 0.2,
    "abstention_behavior": 0.1,
}
_DEFAULT_GROUNDING_THRESHOLD = 0.65
_RECOMMENDATION_SLOTS = ("best_overall", "best_quality", "best_cheap", "best_local")
_FAILURE_LABEL_ORDER = ("hallucinated", "missed_answer", "bad_abstention", "format_failure")


class RAGAnswerQualityRecipe(RecipeDefinition):
    """Answer-quality recipe for fixed-context and live-end-to-end RAG runs."""

    manifest = RecipeManifest(
        recipe_id="rag_answer_quality",
        recipe_version="1",
        name="RAG Answer Quality",
        description=(
            "Compare answer-generation candidates with fixed-context and live end-to-end supervision."
        ),
        launchable=True,
        supported_modes=["labeled", "unlabeled"],
        tags=["rag", "answer-quality", "recipe-v1"],
        capabilities={
            "evaluation_modes": ["fixed_context", "live_end_to_end"],
            "supervision_modes": ["rubric", "reference_answer", "pairwise", "mixed"],
            "candidate_dimensions": [
                "generation_model",
                "prompt_variant",
                "formatting_citation_mode",
            ],
        },
        default_run_config={
            "evaluation_mode": "fixed_context",
            "supervision_mode": "rubric",
            "candidate_dimensions": [
                "generation_model",
                "prompt_variant",
                "formatting_citation_mode",
            ],
            "weights": dict(_DEFAULT_WEIGHTS),
            "grounding_threshold": _DEFAULT_GROUNDING_THRESHOLD,
        },
    )

    def validate_dataset(
        self,
        dataset: list[dict[str, Any]],
        *,
        run_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        errors: list[str] = []
        samples = list(dataset or [])
        if not samples:
            errors.append("Dataset must contain at least one sample.")
            return {
                "valid": False,
                "errors": errors,
                "dataset_mode": None,
                "sample_count": 0,
                "review_sample": {"required": False, "sample_size": 0, "sample_ids": []},
            }

        normalized_run_config = run_config if isinstance(run_config, dict) else {}
        if run_config is not None and not isinstance(run_config, dict):
            errors.append("run_config must be an object.")

        evaluation_mode = self._resolve_evaluation_mode(normalized_run_config)
        supervision_mode = self._resolve_supervision_mode(normalized_run_config)
        run_level_context_available = bool(self._normalize_contexts_for_storage(
            normalized_run_config.get("inline_contexts")
        ))
        labeled_flags: list[bool] = []
        sample_ids: list[str] = []
        for index, raw_sample in enumerate(samples):
            if not isinstance(raw_sample, Mapping):
                errors.append(f"Dataset sample {index} must be an object.")
                continue
            sample = dict(raw_sample)
            sample_ids.append(self._extract_sample_id(sample, index))

            prompt = self._extract_prompt(sample)
            if not prompt:
                errors.append(
                    f"Dataset sample {index} must include a non-empty input, question, or prompt."
                )

            expected_behavior = self._extract_expected_behavior(sample)
            has_reference_answer = self._sample_has_reference_answer(sample)
            if expected_behavior is None and not has_reference_answer:
                labeled_flags.append(False)
            else:
                labeled_flags.append(True)
                if expected_behavior is not None and expected_behavior not in {"answer", "hedge", "abstain"}:
                    errors.append(
                        f"Dataset sample {index} expected_behavior must be one of: answer, hedge, abstain."
                    )

            if evaluation_mode == "fixed_context" and not (
                self._sample_has_inline_context(sample) or run_level_context_available
            ):
                errors.append(
                    f"Dataset sample {index} needs actual context resolved inline or via run_config.inline_contexts for fixed_context runs."
                )

            if supervision_mode in {"reference_answer", "pairwise", "mixed"} and not self._sample_has_reference_answer(sample):
                errors.append(
                    f"Dataset sample {index} requires reference_answer for supervision_mode '{supervision_mode}'."
                )

        dataset_mode = self._detect_dataset_mode(labeled_flags)
        if dataset_mode == "mixed":
            errors.append(
                "Dataset must use a consistent labeling mode; do not mix labeled and unlabeled samples."
            )

        if evaluation_mode == "live_end_to_end" and not self._has_retrieval_baseline(normalized_run_config):
            errors.append(
                "live_end_to_end runs require run_config.retrieval_baseline_ref."
            )

        review_sample = (
            self._reserve_review_sample(sample_ids)
            if dataset_mode == "unlabeled"
            else {"required": False, "sample_size": 0, "sample_ids": []}
        )
        return {
            "valid": not errors,
            "errors": errors,
            "dataset_mode": dataset_mode,
            "sample_count": len(samples),
            "review_sample": review_sample,
        }

    def normalize_run_config(self, run_config: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(run_config, dict):
            raise ValueError("run_config must be an object.")

        evaluation_mode = self._normalize_mode(
            run_config.get("evaluation_mode"),
            allowed_modes=_SUPPORTED_EVALUATION_MODES,
            field_name="run_config.evaluation_mode",
            default="fixed_context",
        )
        supervision_mode = self._normalize_mode(
            run_config.get("supervision_mode"),
            allowed_modes=_SUPPORTED_SUPERVISION_MODES,
            field_name="run_config.supervision_mode",
            default="rubric",
        )
        normalized_candidates = self._normalize_candidates(run_config.get("candidates"))
        candidate_dimensions = self._normalize_candidate_dimensions(
            run_config.get("candidate_dimensions")
        )
        weights = self._normalize_weights(run_config.get("weights"))
        grounding_threshold = self._normalize_grounding_threshold(
            run_config.get("grounding_threshold")
        )

        normalized: dict[str, Any] = {
            "evaluation_mode": evaluation_mode,
            "supervision_mode": supervision_mode,
            "candidates": normalized_candidates,
            "candidate_dimensions": candidate_dimensions,
            "weights": weights,
            "grounding_threshold": grounding_threshold,
        }
        prompts = self._normalize_prompt_mapping(run_config.get("prompts"))
        judge_config = self._normalize_mapping(run_config.get("judge_config"), field_name="run_config.judge_config")
        candidate_api_keys = self._normalize_candidate_api_keys(run_config.get("candidate_api_keys"))
        execution_policy = self._normalize_mapping(
            run_config.get("execution_policy"),
            field_name="run_config.execution_policy",
        )
        if prompts:
            normalized["prompts"] = prompts
        if judge_config:
            normalized["judge_config"] = judge_config
        if candidate_api_keys:
            normalized["candidate_api_keys"] = candidate_api_keys
        if execution_policy:
            normalized["execution_policy"] = execution_policy

        if evaluation_mode == "fixed_context":
            context_snapshot_ref = self._normalize_ref_value(
                run_config.get("context_snapshot_ref"),
                field_name="run_config.context_snapshot_ref",
            )
            inline_contexts = self._normalize_contexts_for_storage(run_config.get("inline_contexts"))
            run_anchor_ref = self._normalize_ref_value(
                run_config.get("run_anchor_ref") or run_config.get("context_anchor_ref"),
                field_name="run_config.run_anchor_ref",
            )
            if context_snapshot_ref is not None:
                normalized["context_snapshot_ref"] = context_snapshot_ref
            if run_anchor_ref is not None:
                normalized["run_anchor_ref"] = run_anchor_ref
            if inline_contexts is not None:
                normalized["inline_contexts"] = inline_contexts
        else:
            retrieval_baseline_ref = self._normalize_ref_value(
                run_config.get("retrieval_baseline_ref"),
                field_name="run_config.retrieval_baseline_ref",
            )
            if not retrieval_baseline_ref:
                raise ValueError(
                    "live_end_to_end runs require run_config.retrieval_baseline_ref."
                )
            normalized["retrieval_baseline_ref"] = retrieval_baseline_ref
            live_retrieval_config = self._normalize_live_retrieval_config(run_config)
            normalized.update(live_retrieval_config)

        return normalized

    def build_report(
        self,
        *,
        dataset_mode: str,
        review_sample: dict[str, Any],
        candidate_results: list[dict[str, Any]],
        evaluation_mode: str | None = None,
        supervision_mode: str | None = None,
        context_snapshot_ref: str | None = None,
        run_anchor_ref: str | None = None,
        retrieval_baseline_ref: str | None = None,
        retrieval_preset_hash: str | None = None,
        weights: dict[str, float] | None = None,
        grounding_threshold: float | None = None,
    ) -> dict[str, Any]:
        normalized_weights = self._normalize_weights(weights)
        threshold = (
            self._normalize_grounding_threshold(grounding_threshold)
            if grounding_threshold is not None
            else _DEFAULT_GROUNDING_THRESHOLD
        )

        candidate_summaries = [
            self._summarize_candidate(candidate_result, normalized_weights, threshold)
            for candidate_result in candidate_results
        ]
        candidate_summaries = [summary for summary in candidate_summaries if summary is not None]
        candidate_summaries.sort(
            key=lambda summary: (
                0 if summary["metrics"]["grounding_gate_passed"] else 1,
                -float(summary["metrics"]["quality_score"]),
                float(summary["latency_ms"] or 0.0),
                str(summary["candidate_id"]),
            )
        )

        grounded_candidates = [
            summary for summary in candidate_summaries if summary["metrics"]["grounding_gate_passed"]
        ]
        best_overall = self._pick_best_overall(grounded_candidates)
        best_quality = self._pick_best_quality(grounded_candidates)
        best_cheap = self._pick_best_cheap(candidate_summaries)
        best_local = self._pick_best_local(candidate_summaries)

        quality_scores = [float(summary["metrics"]["quality_score"]) for summary in grounded_candidates]
        sample_count = min((int(summary["sample_count"]) for summary in candidate_summaries), default=0)
        spread = statistics.pstdev(quality_scores) if len(quality_scores) > 1 else 0.0
        winner_margin = self._winner_margin(grounded_candidates, best_overall)
        confidence_summary = ConfidenceSummary(
            kind="aggregate",
            confidence=self._confidence_score(
                sample_count=sample_count,
                spread=spread,
                winner_margin=winner_margin,
            ),
            sample_count=sample_count,
            spread=spread,
            margin=winner_margin,
            notes="Unlabeled run reserves review sample." if dataset_mode == "unlabeled" else None,
        )

        recommendation_slots = {
            "best_overall": self._build_slot(
                "best_overall",
                best_overall,
                reason_code="highest_grounded_quality" if best_overall else "grounding_gate_failed",
                confidence=confidence_summary.confidence,
                grounding_threshold=threshold,
            ),
            "best_quality": self._build_slot(
                "best_quality",
                best_quality,
                reason_code="highest_quality" if best_quality else "grounding_gate_failed",
                confidence=confidence_summary.confidence,
                grounding_threshold=threshold,
            ),
            "best_cheap": self._build_slot(
                "best_cheap",
                best_cheap,
                reason_code="lowest_cost",
                grounding_threshold=threshold,
            ),
            "best_local": self._build_slot(
                "best_local",
                best_local,
                reason_code="best_local_quality",
                grounding_threshold=threshold,
            ),
        }
        for slot_name in _RECOMMENDATION_SLOTS:
            recommendation_slots.setdefault(
                slot_name,
                self._build_slot(slot_name, None, reason_code="not_available", grounding_threshold=threshold),
            )
        failure_examples = self._collect_report_failure_examples(candidate_summaries)

        return {
            "dataset_mode": dataset_mode,
            "review_sample": review_sample,
            "evaluation_mode": evaluation_mode or "fixed_context",
            "supervision_mode": supervision_mode or "rubric",
            "context_snapshot_ref": context_snapshot_ref,
            "run_anchor_ref": run_anchor_ref,
            "retrieval_baseline_ref": retrieval_baseline_ref,
            "retrieval_preset_hash": retrieval_preset_hash,
            "weights": normalized_weights,
            "grounding_threshold": threshold,
            "candidates": candidate_summaries,
            "best_overall": best_overall,
            "best_quality": best_quality,
            "best_cheap": best_cheap,
            "best_local": best_local,
            "failure_examples": failure_examples,
            "recommendation_slots": {
                slot_name: slot.model_dump(mode="json")
                for slot_name, slot in recommendation_slots.items()
            },
            "confidence_summary": confidence_summary.model_dump(mode="json"),
            "confidence_inputs": {
                "sample_count": sample_count,
                "spread": spread,
                "winner_margin": winner_margin,
                "grounding_threshold": threshold,
            },
        }

    def _resolve_evaluation_mode(self, run_config: dict[str, Any]) -> str:
        return self._normalize_mode(
            run_config.get("evaluation_mode"),
            allowed_modes=_SUPPORTED_EVALUATION_MODES,
            field_name="run_config.evaluation_mode",
            default="fixed_context",
        )

    def _resolve_supervision_mode(self, run_config: Mapping[str, Any]) -> str:
        return self._normalize_mode(
            run_config.get("supervision_mode"),
            allowed_modes=_SUPPORTED_SUPERVISION_MODES,
            field_name="run_config.supervision_mode",
            default="rubric",
        )

    def _normalize_mode(
        self,
        value: Any,
        *,
        allowed_modes: set[str],
        field_name: str,
        default: str,
    ) -> str:
        mode = str(value or default).strip().lower()
        if mode not in allowed_modes:
            allowed = ", ".join(sorted(allowed_modes))
            raise ValueError(f"{field_name} must be one of: {allowed}.")
        return mode

    def _normalize_candidate_dimensions(self, value: Any) -> list[str]:
        canonical_dimensions = [
            "generation_model",
            "prompt_variant",
            "formatting_citation_mode",
        ]
        if value is None:
            return [dimension for dimension in canonical_dimensions if dimension in _SUPPORTED_CANDIDATE_DIMENSIONS]
        if not isinstance(value, list) or not value:
            raise ValueError(
                "run_config.candidate_dimensions must contain at least one supported dimension."
            )
        provided = {str(item).strip() for item in value if str(item).strip()}
        if not provided:
            raise ValueError(
                "run_config.candidate_dimensions must contain at least one supported dimension."
            )
        unknown = sorted(provided - _SUPPORTED_CANDIDATE_DIMENSIONS)
        if unknown:
            raise ValueError(
                "run_config.candidate_dimensions may only contain generation_model, prompt_variant, and formatting_citation_mode."
            )
        return [dimension for dimension in canonical_dimensions if dimension in provided]

    def _normalize_weights(self, weights: Any) -> dict[str, float]:
        if weights is None:
            normalized = dict(_DEFAULT_WEIGHTS)
        elif not isinstance(weights, Mapping):
            raise ValueError("run_config.weights must be an object when provided.")
        else:
            unknown_keys = sorted(set(weights) - set(_SUPPORTED_WEIGHT_KEYS))
            if unknown_keys:
                raise ValueError(
                    "run_config.weights may only contain grounding, answer_relevance, format_style, and abstention_behavior."
                )
            normalized = {
                key: float(weights.get(key, _DEFAULT_WEIGHTS[key]))
                for key in _SUPPORTED_WEIGHT_KEYS
            }
        total = sum(normalized.values())
        if total <= 0:
            raise ValueError("run_config.weights must sum to a positive value.")
        return {key: value / total for key, value in normalized.items()}

    def _normalize_grounding_threshold(self, value: Any) -> float:
        threshold = (
            float(value)
            if value is not None
            else _DEFAULT_GROUNDING_THRESHOLD
        )
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError("run_config.grounding_threshold must be between 0 and 1.")
        return threshold

    def _normalize_ref_value(self, value: Any, *, field_name: str) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            normalized = value.strip()
            if normalized:
                return normalized
            raise ValueError(f"{field_name} must not be empty.")
        normalized = str(value).strip()
        if not normalized:
            raise ValueError(f"{field_name} must not be empty.")
        return normalized

    def _normalize_contexts_for_storage(self, value: Any) -> list[dict[str, Any]] | None:
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            return [{"source": "inline", "text": stripped}]
        if isinstance(value, Mapping):
            value = [value]
        if not isinstance(value, list):
            raise ValueError("run_config.inline_contexts must be a list when provided.")
        normalized: list[dict[str, Any]] = []
        for item in value:
            if isinstance(item, Mapping):
                text = str(item.get("text") or item.get("content") or item.get("body") or "").strip()
                if not text:
                    continue
                normalized.append(
                    {
                        "source": str(item.get("source") or item.get("origin") or "inline").strip() or "inline",
                        "text": text,
                    }
                )
                continue
            if isinstance(item, str) and item.strip():
                normalized.append({"source": "inline", "text": item.strip()})
                continue
            raise ValueError("run_config.inline_contexts entries must be objects or non-empty strings.")
        return normalized

    def _normalize_candidates(self, value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list) or not value:
            raise ValueError("run_config.candidates must contain at least one candidate.")
        normalized: list[dict[str, Any]] = []
        seen_candidate_ids: set[str] = set()
        for index, raw_candidate in enumerate(value, start=1):
            if not isinstance(raw_candidate, Mapping):
                raise ValueError("run_config.candidates entries must be objects.")
            candidate = dict(raw_candidate)
            provider, model = self._resolve_provider_model(candidate)
            candidate_id = self._resolve_candidate_id(
                candidate,
                provider=provider,
                model=model,
                index=index,
            )
            if candidate_id in seen_candidate_ids:
                raise ValueError(
                    f"run_config.candidates candidate_id values must be unique; duplicate {candidate_id!r}."
                )
            seen_candidate_ids.add(candidate_id)
            normalized.append(
                {
                    "candidate_id": candidate_id,
                    "provider": provider,
                    "model": model,
                    "generation_model": candidate.get("generation_model") or candidate.get("model"),
                    "prompt_variant": str(candidate.get("prompt_variant") or "default").strip() or "default",
                    "formatting_citation_mode": str(
                        candidate.get("formatting_citation_mode") or "plain"
                    ).strip() or "plain",
                    "is_local": bool(candidate.get("is_local")),
                    "cost_usd": candidate.get("cost_usd"),
                    "generation_config": self._normalize_mapping(
                        candidate.get("generation_config"),
                        field_name="run_config.candidates[].generation_config",
                    ),
                }
            )
        if not normalized:
            raise ValueError("run_config.candidates must contain at least one candidate.")
        return normalized

    def _normalize_mapping(self, value: Any, *, field_name: str) -> dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise ValueError(f"{field_name} must be an object when provided.")
        return dict(value)

    def _normalize_prompt_mapping(self, value: Any) -> dict[str, str]:
        prompts = self._normalize_mapping(value, field_name="run_config.prompts")
        return {
            str(key): str(prompt)
            for key, prompt in prompts.items()
            if str(key).strip() and str(prompt).strip()
        }

    def _normalize_candidate_api_keys(self, value: Any) -> dict[str, str]:
        candidate_api_keys = self._normalize_mapping(
            value,
            field_name="run_config.candidate_api_keys",
        )
        return {
            str(provider).strip(): str(api_key).strip()
            for provider, api_key in candidate_api_keys.items()
            if str(provider).strip() and str(api_key).strip()
        }

    def _normalize_live_retrieval_config(self, run_config: Mapping[str, Any]) -> dict[str, Any]:
        normalized: dict[str, Any] = {}
        retrieval_sources = run_config.get("retrieval_sources")
        if retrieval_sources is not None:
            if not isinstance(retrieval_sources, list):
                raise ValueError("run_config.retrieval_sources must be a list when provided.")
            normalized["retrieval_sources"] = [
                str(source).strip() for source in retrieval_sources if str(source).strip()
            ]
        for field_name in ("search_mode", "fts_level", "reranking_strategy"):
            value = run_config.get(field_name)
            if value is not None and str(value).strip():
                normalized[field_name] = str(value).strip()
        for field_name in ("hybrid_alpha", "min_score"):
            value = run_config.get(field_name)
            if value is not None:
                normalized[field_name] = float(value)
        for field_name in ("top_k", "rerank_top_k"):
            value = run_config.get(field_name)
            if value is not None:
                normalized[field_name] = int(value)
        for field_name in ("enable_reranking", "enable_cache"):
            if field_name in run_config:
                normalized[field_name] = bool(run_config.get(field_name))
        return normalized

    def _has_run_anchor(self, run_config: Mapping[str, Any]) -> bool:
        for field_name in ("context_snapshot_ref", "run_anchor_ref", "context_anchor_ref"):
            try:
                if self._normalize_ref_value(
                    run_config.get(field_name),
                    field_name=f"run_config.{field_name}",
                ):
                    return True
            except ValueError:
                continue
        return False

    def _has_retrieval_baseline(self, run_config: Mapping[str, Any]) -> bool:
        try:
            return self._normalize_ref_value(
                run_config.get("retrieval_baseline_ref"),
                field_name="run_config.retrieval_baseline_ref",
            ) is not None
        except ValueError:
            return False

    def _sample_has_inline_context(self, sample: Mapping[str, Any]) -> bool:
        for field_name in ("inline_contexts", "contexts", "context", "source_contexts"):
            value = sample.get(field_name)
            if value is None:
                continue
            try:
                normalized = self._normalize_contexts_for_storage(value)
            except ValueError:
                continue
            if normalized:
                return True
        return False

    def _sample_has_reference_answer(self, sample: Mapping[str, Any]) -> bool:
        for field_name in ("reference_answer", "expected", "expected_answer"):
            value = sample.get(field_name)
            if isinstance(value, str) and value.strip():
                return True
        return False

    def _extract_prompt(self, sample: Mapping[str, Any]) -> str:
        for field_name in ("input", "question", "query", "prompt"):
            value = sample.get(field_name)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    def _extract_expected_behavior(self, sample: Mapping[str, Any]) -> str | None:
        value = sample.get("expected_behavior")
        if value is None:
            return None
        normalized = str(value).strip().lower()
        return normalized or None

    def _extract_sample_id(self, sample: Mapping[str, Any], index: int) -> str:
        for field_name in ("sample_id", "id"):
            value = sample.get(field_name)
            if isinstance(value, str) and value.strip():
                return value.strip()
            if value is not None and str(value).strip():
                return str(value).strip()
        metadata = sample.get("metadata")
        if isinstance(metadata, Mapping):
            sample_id = metadata.get("sample_id")
            if isinstance(sample_id, str) and sample_id.strip():
                return sample_id.strip()
            if sample_id is not None and str(sample_id).strip():
                return str(sample_id).strip()
        return f"sample-{index}"

    def _detect_dataset_mode(self, labeled_flags: list[bool]) -> str | None:
        if not labeled_flags:
            return None
        if all(labeled_flags):
            return "labeled"
        if not any(labeled_flags):
            return "unlabeled"
        return "mixed"

    def _reserve_review_sample(self, sample_ids: list[str]) -> dict[str, Any]:
        if not sample_ids:
            return {"required": False, "sample_size": 0, "sample_ids": []}
        sample_size = min(len(sample_ids), min(25, max(3, math.ceil(len(sample_ids) * 0.2))))
        return {
            "required": True,
            "sample_size": sample_size,
            "sample_ids": sample_ids[:sample_size],
        }

    def _summarize_candidate(
        self,
        candidate_result: dict[str, Any],
        weights: dict[str, float],
        grounding_threshold: float,
    ) -> dict[str, Any] | None:
        if not isinstance(candidate_result, Mapping):
            return None

        sample_results = [
            dict(sample_result)
            for sample_result in (candidate_result.get("sample_results") or candidate_result.get("results") or [])
            if isinstance(sample_result, Mapping)
        ]
        rubric_samples = [self._coerce_sample_metrics(sample_result) for sample_result in sample_results]
        rubric_samples = [sample for sample in rubric_samples if sample is not None]

        if rubric_samples:
            grounding = statistics.mean(sample["grounding"] for sample in rubric_samples)
            answer_relevance = statistics.mean(sample["answer_relevance"] for sample in rubric_samples)
            format_style = statistics.mean(sample["format_style"] for sample in rubric_samples)
            abstention_behavior = statistics.mean(
                sample["abstention_behavior"] for sample in rubric_samples
            )
            latency_values = [
                float(sample_result.get("latency_ms"))
                for sample_result in sample_results
                if sample_result.get("latency_ms") is not None
            ]
            latency_ms = statistics.mean(latency_values) if latency_values else None
            sample_count = len(rubric_samples)
        else:
            top_level_metrics = self._coerce_metrics(candidate_result.get("metrics") or {})
            if top_level_metrics is None:
                return None
            grounding = top_level_metrics["grounding"]
            answer_relevance = top_level_metrics["answer_relevance"]
            format_style = top_level_metrics["format_style"]
            abstention_behavior = top_level_metrics["abstention_behavior"]
            latency_ms = candidate_result.get("latency_ms")
            sample_count = int(candidate_result.get("sample_count") or 0)

        quality_score = (
            (weights.get("grounding", 0.0) * grounding)
            + (weights.get("answer_relevance", 0.0) * answer_relevance)
            + (weights.get("format_style", 0.0) * format_style)
            + (weights.get("abstention_behavior", 0.0) * abstention_behavior)
        )
        grounding_gate_passed = grounding >= grounding_threshold
        candidate_id = candidate_result.get("candidate_id")
        candidate_id_str = str(candidate_id or candidate_result.get("model") or "").strip()
        failure_label_counts, failure_examples = self._collect_failure_details(
            sample_results,
            candidate_id=candidate_id_str or None,
        )

        return {
            "candidate_id": candidate_id_str or None,
            "candidate_run_id": (
                str(candidate_result.get("candidate_run_id") or "").strip() or None
            ),
            "provider": candidate_result.get("provider"),
            "model": candidate_result.get("model"),
            "is_local": bool(candidate_result.get("is_local")),
            "cost_usd": candidate_result.get("cost_usd"),
            "sample_count": sample_count,
            "latency_ms": latency_ms,
            "failure_label_counts": failure_label_counts,
            "failure_examples": failure_examples,
            "metrics": {
                "grounding": grounding,
                "answer_relevance": answer_relevance,
                "format_style": format_style,
                "abstention_behavior": abstention_behavior,
                "quality_score": quality_score,
                "grounding_gate_passed": grounding_gate_passed,
            },
        }

    def _coerce_sample_metrics(self, sample_result: Mapping[str, Any]) -> dict[str, float] | None:
        metrics = self._coerce_metrics(sample_result.get("metrics") or {})
        if metrics is None:
            return None
        if not all(metric_name in metrics for metric_name in _SUPPORTED_WEIGHT_KEYS):
            return None
        return metrics

    def _collect_failure_details(
        self,
        sample_results: list[dict[str, Any]],
        *,
        candidate_id: str | None,
    ) -> tuple[dict[str, int], list[dict[str, Any]]]:
        failure_label_counts: dict[str, int] = {}
        failure_examples: list[dict[str, Any]] = []

        for sample_result in sample_results:
            labels = self._normalize_failure_labels(sample_result.get("failure_labels"))
            if not labels:
                continue
            for label in labels:
                failure_label_counts[label] = failure_label_counts.get(label, 0) + 1
            metrics = self._coerce_metrics(sample_result.get("metrics") or {}) or {}
            failure_examples.append(
                {
                    "candidate_id": candidate_id,
                    "sample_id": sample_result.get("sample_id"),
                    "query": sample_result.get("query"),
                    "expected_behavior": sample_result.get("expected_behavior"),
                    "failure_labels": labels,
                    "answer": sample_result.get("answer"),
                    "reference_answer": sample_result.get("reference_answer"),
                    "metrics": metrics,
                    "latency_ms": sample_result.get("latency_ms"),
                }
            )
        failure_examples.sort(
            key=lambda example: (
                -len(example["failure_labels"]),
                str(example.get("sample_id") or ""),
            )
        )
        return failure_label_counts, failure_examples[:3]

    def _normalize_failure_labels(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        labels = {str(item).strip() for item in value if str(item).strip()}
        return [label for label in _FAILURE_LABEL_ORDER if label in labels]

    def _coerce_metrics(self, metrics: Any) -> dict[str, float] | None:
        if metrics is None:
            return None
        if isinstance(metrics, str):
            try:
                metrics = json.loads(metrics)
            except (TypeError, ValueError):
                return None
        if not isinstance(metrics, Mapping):
            return None

        grounding = self._coerce_metric_value(metrics, "grounding")
        answer_relevance = self._coerce_metric_value(metrics, "answer_relevance")
        format_style = self._coerce_metric_value(
            metrics,
            "format_style",
            fallback_keys=("format_style_compliance", "citation_style", "style_compliance"),
        )
        abstention_behavior = self._coerce_metric_value(metrics, "abstention_behavior")
        if None in {grounding, answer_relevance, format_style, abstention_behavior}:
            return None
        return {
            "grounding": float(grounding),
            "answer_relevance": float(answer_relevance),
            "format_style": float(format_style),
            "abstention_behavior": float(abstention_behavior),
        }

    def _coerce_metric_value(
        self,
        metrics: Mapping[str, Any],
        metric_name: str,
        *,
        fallback_keys: tuple[str, ...] = (),
    ) -> float | None:
        value = metrics.get(metric_name)
        if value is None:
            for fallback_key in fallback_keys:
                value = metrics.get(fallback_key)
                if value is not None:
                    break
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _pick_best_overall(self, candidate_summaries: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not candidate_summaries:
            return None
        return max(
            candidate_summaries,
            key=lambda summary: (
                float(summary["metrics"]["quality_score"]),
                float(summary["metrics"]["grounding"]),
                -float(summary["latency_ms"] or 0.0),
                str(summary["candidate_id"]),
            ),
        )

    def _pick_best_quality(self, candidate_summaries: list[dict[str, Any]]) -> dict[str, Any] | None:
        return self._pick_best_overall(candidate_summaries)

    def _pick_best_cheap(self, candidate_summaries: list[dict[str, Any]]) -> dict[str, Any] | None:
        cheap_candidates = [
            summary for summary in candidate_summaries if summary.get("cost_usd") is not None
        ]
        if not cheap_candidates:
            return None
        return min(
            cheap_candidates,
            key=lambda summary: (
                float(summary["cost_usd"]),
                -float(summary["metrics"]["quality_score"]),
                float(summary["latency_ms"] or 0.0),
                str(summary["candidate_id"]),
            ),
        )

    def _pick_best_local(self, candidate_summaries: list[dict[str, Any]]) -> dict[str, Any] | None:
        local_candidates = [
            summary for summary in candidate_summaries if bool(summary.get("is_local"))
        ]
        if not local_candidates:
            return None
        return max(
            local_candidates,
            key=lambda summary: (
                float(summary["metrics"]["quality_score"]),
                float(summary["metrics"]["grounding"]),
                -float(summary["latency_ms"] or 0.0),
                str(summary["candidate_id"]),
            ),
        )

    def _winner_margin(
        self,
        candidate_summaries: list[dict[str, Any]],
        best_overall: dict[str, Any] | None,
    ) -> float:
        if not best_overall or len(candidate_summaries) < 2:
            return 0.0
        ordered = sorted(
            candidate_summaries,
            key=lambda summary: (
                float(summary["metrics"]["quality_score"]),
                float(summary["metrics"]["grounding"]),
                -float(summary["latency_ms"] or 0.0),
            ),
            reverse=True,
        )
        return max(0.0, float(ordered[0]["metrics"]["quality_score"]) - float(ordered[1]["metrics"]["quality_score"]))

    def _confidence_score(self, *, sample_count: int, spread: float, winner_margin: float) -> float:
        sample_factor = min(1.0, sample_count / 10.0)
        margin_factor = min(1.0, winner_margin / 0.25)
        spread_penalty = min(1.0, spread)
        confidence = 0.35 + (sample_factor * 0.35) + (margin_factor * 0.35) - (spread_penalty * 0.15)
        return max(0.0, min(1.0, confidence))

    def _collect_report_failure_examples(
        self,
        candidate_summaries: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        failure_examples: list[dict[str, Any]] = []
        for candidate_summary in candidate_summaries:
            examples = candidate_summary.get("failure_examples") or []
            if not isinstance(examples, list):
                continue
            for example in examples:
                if isinstance(example, Mapping):
                    failure_examples.append(dict(example))
        failure_examples.sort(
            key=lambda example: (
                -len(example.get("failure_labels") or []),
                str(example.get("candidate_id") or ""),
                str(example.get("sample_id") or ""),
            )
        )
        return failure_examples[:10]

    def _build_slot(
        self,
        slot_name: str,
        candidate_summary: dict[str, Any] | None,
        *,
        reason_code: str,
        confidence: float | None = None,
        grounding_threshold: float | None = None,
    ) -> RecommendationSlot:
        if candidate_summary is None:
            metadata: dict[str, Any] = {}
            if grounding_threshold is not None:
                metadata["grounding_threshold"] = grounding_threshold
            return RecommendationSlot(
                candidate_run_id=None,
                reason_code=reason_code,
                explanation=f"No candidate qualified for '{slot_name}'.",
                metadata=metadata,
            )

        metadata = {
            "candidate_id": candidate_summary.get("candidate_id"),
            "model": candidate_summary.get("model"),
            "provider": candidate_summary.get("provider"),
            "quality_score": candidate_summary.get("metrics", {}).get("quality_score"),
            "grounding": candidate_summary.get("metrics", {}).get("grounding"),
            "answer_relevance": candidate_summary.get("metrics", {}).get("answer_relevance"),
            "format_style": candidate_summary.get("metrics", {}).get("format_style"),
            "abstention_behavior": candidate_summary.get("metrics", {}).get("abstention_behavior"),
            "grounding_gate_passed": candidate_summary.get("metrics", {}).get(
                "grounding_gate_passed"
            ),
        }
        if grounding_threshold is not None:
            metadata["grounding_threshold"] = grounding_threshold
        return RecommendationSlot(
            candidate_run_id=candidate_summary.get("candidate_run_id"),
            reason_code=reason_code,
            explanation=(
                f"{candidate_summary.get('model') or candidate_summary.get('candidate_id')} "
                f"won '{slot_name}' with grounded quality {candidate_summary['metrics']['quality_score']:.3f}."
            ),
            confidence=confidence,
            metadata=metadata,
        )

    def _resolve_provider_model(self, candidate: Mapping[str, Any]) -> tuple[str, str]:
        provider = str(candidate.get("provider") or "").strip()
        model = str(candidate.get("model") or "").strip()
        generation_model = str(candidate.get("generation_model") or "").strip()
        if not provider or not model:
            if generation_model and ":" in generation_model:
                provider, model = generation_model.split(":", 1)
                provider = provider.strip()
                model = model.strip()
            elif model and ":" in model:
                provider, model = model.split(":", 1)
                provider = provider.strip()
                model = model.strip()
        if not provider or not model:
            raise ValueError("Each candidate must include provider/model or generation_model.")
        return provider, model

    def _resolve_candidate_id(
        self,
        candidate: Mapping[str, Any],
        *,
        provider: str,
        model: str,
        index: int,
    ) -> str:
        candidate_id = str(candidate.get("candidate_id") or "").strip()
        if candidate_id:
            return candidate_id
        prompt_variant = str(candidate.get("prompt_variant") or "default").strip() or "default"
        formatting_mode = str(candidate.get("formatting_citation_mode") or "plain").strip() or "plain"
        return f"{provider}:{model}::{prompt_variant}::{formatting_mode}::{index}"
