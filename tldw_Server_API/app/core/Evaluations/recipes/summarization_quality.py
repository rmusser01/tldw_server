"""V1 summarization quality recipe helpers."""

from __future__ import annotations

import math
import statistics
from typing import Any

from tldw_Server_API.app.api.v1.schemas.evaluation_recipe_schemas import (
    ConfidenceSummary,
    RecipeManifest,
    RecommendationSlot,
)
from tldw_Server_API.app.core.Evaluations.recipes.base import RecipeDefinition
from tldw_Server_API.app.core.Evaluations.scoring import normalize_geval_metric, normalize_likert

_RECOMMENDATION_SLOTS = ("best_overall", "best_cheap", "best_local")
_DEFAULT_WEIGHTS = {
    "grounding": 0.5,
    "coverage": 0.3,
    "usefulness": 0.2,
}


class SummarizationQualityRecipe(RecipeDefinition):
    """Weighted summarization recipe backed by source-grounded G-Eval metrics."""

    manifest = RecipeManifest(
        recipe_id="summarization_quality",
        recipe_version="1",
        name="Summarization Quality",
        description="Compare summarization candidates with a weighted grounding-focused rubric.",
        supported_modes=["labeled", "unlabeled"],
        tags=["summarization", "quality", "recipe-v1"],
    )

    def validate_dataset(self, dataset: list[dict[str, Any]]) -> dict[str, Any]:
        errors: list[str] = []
        samples = [dict(sample) for sample in (dataset or [])]
        if not samples:
            errors.append("Dataset must contain at least one sample.")
            return {
                "valid": False,
                "errors": errors,
                "dataset_mode": None,
                "sample_count": 0,
                "review_sample": {"required": False, "sample_size": 0, "sample_ids": []},
            }

        labeled_flags: list[bool] = []
        sample_ids: list[str] = []
        for index, sample in enumerate(samples):
            sample_ids.append(self._extract_sample_id(sample, index))
            if not self._extract_source_text(sample):
                errors.append(
                    f"Dataset sample {index} must include a non-empty input/source_text value."
                )
            labeled_flags.append(self._extract_reference_summary(sample) is not None)

        dataset_mode = self._detect_dataset_mode(labeled_flags)
        if dataset_mode == "mixed":
            errors.append(
                "Dataset must use a consistent labeling mode; do not mix labeled and unlabeled samples."
            )

        review_sample = self._reserve_review_sample(sample_ids) if dataset_mode == "unlabeled" else {
            "required": False,
            "sample_size": 0,
            "sample_ids": [],
        }
        return {
            "valid": not errors,
            "errors": errors,
            "dataset_mode": dataset_mode,
            "sample_count": len(samples),
            "review_sample": review_sample,
        }

    def build_report(
        self,
        *,
        dataset_mode: str,
        review_sample: dict[str, Any],
        candidate_results: list[dict[str, Any]],
        weights: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        normalized_weights = self._normalize_weights(weights)
        candidate_summaries = [
            self._summarize_candidate(candidate_result, normalized_weights)
            for candidate_result in candidate_results
        ]
        candidate_summaries = [summary for summary in candidate_summaries if summary is not None]
        candidate_summaries.sort(
            key=lambda summary: (
                -float(summary["metrics"]["quality_score"]),
                float(summary["latency_ms"] or 0.0),
                str(summary["candidate_id"]),
            )
        )

        best_overall = self._pick_best_overall(candidate_summaries)
        best_cheap = self._pick_best_cheap(candidate_summaries)
        best_local = self._pick_best_local(candidate_summaries)

        quality_scores = [float(summary["metrics"]["quality_score"]) for summary in candidate_summaries]
        sample_count = min((int(summary["sample_count"]) for summary in candidate_summaries), default=0)
        spread = statistics.pstdev(quality_scores) if len(quality_scores) > 1 else 0.0
        winner_margin = self._winner_margin(candidate_summaries)
        confidence_inputs = {
            "sample_count": sample_count,
            "spread": spread,
            "winner_margin": winner_margin,
        }
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
                reason_code="highest_quality_score",
                confidence=confidence_summary.confidence,
            ),
            "best_cheap": self._build_slot(
                "best_cheap",
                best_cheap,
                reason_code="lowest_cost",
            ),
            "best_local": self._build_slot(
                "best_local",
                best_local,
                reason_code="best_local_quality",
            ),
        }
        for slot_name in _RECOMMENDATION_SLOTS:
            recommendation_slots.setdefault(
                slot_name,
                self._build_slot(slot_name, None, reason_code="not_available"),
            )

        return {
            "dataset_mode": dataset_mode,
            "review_sample": review_sample,
            "weights": normalized_weights,
            "candidates": candidate_summaries,
            "best_overall": best_overall,
            "best_cheap": best_cheap,
            "best_local": best_local,
            "recommendation_slots": {
                slot_name: slot.model_dump(mode="json")
                for slot_name, slot in recommendation_slots.items()
            },
            "confidence_summary": confidence_summary.model_dump(mode="json"),
            "confidence_inputs": confidence_inputs,
        }

    def _normalize_weights(self, weights: dict[str, float] | None) -> dict[str, float]:
        if not isinstance(weights, dict):
            return dict(_DEFAULT_WEIGHTS)
        normalized = {
            key: float(weights[key])
            for key in _DEFAULT_WEIGHTS
            if weights.get(key) is not None
        }
        if not normalized:
            return dict(_DEFAULT_WEIGHTS)
        total = sum(normalized.values())
        if total <= 0:
            return dict(_DEFAULT_WEIGHTS)
        return {key: value / total for key, value in normalized.items()}

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
    ) -> dict[str, Any] | None:
        sample_results = [
            dict(sample_result)
            for sample_result in (candidate_result.get("sample_results") or [])
            if isinstance(sample_result, dict)
        ]
        rubric_samples = [
            self._coerce_sample_metrics(sample_result)
            for sample_result in sample_results
        ]
        rubric_samples = [sample for sample in rubric_samples if sample is not None]

        if rubric_samples:
            grounding = statistics.mean(sample["grounding"] for sample in rubric_samples)
            coverage = statistics.mean(sample["coverage"] for sample in rubric_samples)
            usefulness = statistics.mean(sample["usefulness"] for sample in rubric_samples)
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
            coverage = top_level_metrics["coverage"]
            usefulness = top_level_metrics["usefulness"]
            latency_ms = candidate_result.get("latency_ms")
            sample_count = int(candidate_result.get("sample_count") or 0)

        quality_score = (
            (weights.get("grounding", 0.0) * grounding)
            + (weights.get("coverage", 0.0) * coverage)
            + (weights.get("usefulness", 0.0) * usefulness)
        )
        return {
            "candidate_id": candidate_result.get("candidate_id"),
            "candidate_run_id": candidate_result.get("candidate_run_id"),
            "provider": candidate_result.get("provider"),
            "model": candidate_result.get("model"),
            "is_local": bool(candidate_result.get("is_local")),
            "cost_usd": candidate_result.get("cost_usd"),
            "sample_count": sample_count,
            "latency_ms": latency_ms,
            "metrics": {
                "grounding": grounding,
                "coverage": coverage,
                "usefulness": usefulness,
                "quality_score": quality_score,
            },
        }

    def _coerce_sample_metrics(self, sample_result: dict[str, Any]) -> dict[str, float] | None:
        return self._coerce_metrics(sample_result.get("metrics") or {})

    def _coerce_metrics(self, metrics: dict[str, Any]) -> dict[str, float] | None:
        if not isinstance(metrics, dict) or not metrics:
            return None
        if all(key in metrics for key in ("grounding", "coverage", "usefulness")):
            try:
                return {
                    "grounding": self._normalize_unit_score(metrics.get("grounding")),
                    "coverage": self._normalize_unit_score(metrics.get("coverage")),
                    "usefulness": self._normalize_unit_score(metrics.get("usefulness")),
                }
            except (TypeError, ValueError):
                return None

        try:
            grounding = self._normalize_geval_score("consistency", metrics.get("consistency"))
            coverage = self._normalize_geval_score("relevance", metrics.get("relevance"))
            coherence = self._normalize_geval_score("coherence", metrics.get("coherence"))
            fluency = self._normalize_geval_score("fluency", metrics.get("fluency"))
        except (TypeError, ValueError):
            return None
        usefulness = (coherence + fluency) / 2.0
        return {
            "grounding": grounding,
            "coverage": coverage,
            "usefulness": usefulness,
        }

    @staticmethod
    def _non_negative(value: Any) -> float:
        score = float(value)
        if score < 0:
            raise ValueError("Scores must be non-negative.")
        return score

    def _normalize_unit_score(self, value: Any) -> float:
        return normalize_likert(self._non_negative(value), scale_min=0.0, scale_max=1.0)

    def _normalize_geval_score(self, metric: str, value: Any) -> float:
        return normalize_geval_metric(metric, self._non_negative(value))

    def _pick_best_overall(self, candidate_summaries: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not candidate_summaries:
            return None
        return max(
            candidate_summaries,
            key=lambda summary: (
                float(summary["metrics"]["quality_score"]),
                -float(summary["latency_ms"] or 0.0),
            ),
        )

    def _pick_best_cheap(self, candidate_summaries: list[dict[str, Any]]) -> dict[str, Any] | None:
        priced = [summary for summary in candidate_summaries if summary.get("cost_usd") is not None]
        if not priced:
            return None
        return min(
            priced,
            key=lambda summary: (
                float(summary["cost_usd"]),
                -float(summary["metrics"]["quality_score"]),
                float(summary["latency_ms"] or 0.0),
            ),
        )

    def _pick_best_local(self, candidate_summaries: list[dict[str, Any]]) -> dict[str, Any] | None:
        local_candidates = [summary for summary in candidate_summaries if summary.get("is_local")]
        if not local_candidates:
            return None
        return self._pick_best_overall(local_candidates)

    def _winner_margin(self, candidate_summaries: list[dict[str, Any]]) -> float:
        scores = sorted(
            (float(summary["metrics"]["quality_score"]) for summary in candidate_summaries),
            reverse=True,
        )
        if len(scores) < 2:
            return 0.0
        return max(0.0, scores[0] - scores[1])

    def _confidence_score(
        self,
        *,
        sample_count: int,
        spread: float,
        winner_margin: float,
    ) -> float:
        sample_factor = min(1.0, sample_count / 10.0)
        spread_penalty = min(0.2, spread / 5.0)
        margin_bonus = min(0.2, winner_margin)
        return max(0.0, min(1.0, 0.45 + (0.35 * sample_factor) + margin_bonus - spread_penalty))

    def _build_slot(
        self,
        slot_name: str,
        candidate_summary: dict[str, Any] | None,
        *,
        reason_code: str,
        confidence: float | None = None,
    ) -> RecommendationSlot:
        if candidate_summary is None:
            return RecommendationSlot(
                candidate_run_id=None,
                reason_code="not_available",
                explanation=f"No recommendation has been recorded for '{slot_name}'.",
            )
        candidate_id = str(candidate_summary.get("candidate_id") or "")
        candidate_run_id = str(candidate_summary.get("candidate_run_id") or candidate_id)
        return RecommendationSlot(
            candidate_run_id=candidate_run_id,
            reason_code=reason_code,
            explanation=f"{slot_name.replace('_', ' ')} selected for candidate '{candidate_id}'.",
            confidence=confidence,
            metadata={
                "candidate_id": candidate_id,
                "provider": candidate_summary.get("provider"),
                "model": candidate_summary.get("model"),
                "quality_score": candidate_summary["metrics"]["quality_score"],
            },
        )

    def _extract_source_text(self, sample: dict[str, Any]) -> str:
        input_value = sample.get("input")
        if isinstance(input_value, dict):
            source_text = input_value.get("source_text")
            if isinstance(source_text, str):
                return source_text.strip()
        if isinstance(input_value, str):
            return input_value.strip()
        source_text = sample.get("source_text")
        if isinstance(source_text, str):
            return source_text.strip()
        return ""

    def _extract_reference_summary(self, sample: dict[str, Any]) -> str | None:
        for key in ("expected", "reference_summary", "summary"):
            value = sample.get(key)
            if isinstance(value, str):
                stripped = value.strip()
                if stripped:
                    return stripped
        return None

    def _extract_sample_id(self, sample: dict[str, Any], index: int) -> str:
        metadata = sample.get("metadata")
        if isinstance(metadata, dict):
            sample_id = metadata.get("sample_id")
            if isinstance(sample_id, str) and sample_id.strip():
                return sample_id.strip()
        top_level = sample.get("sample_id")
        if isinstance(top_level, str) and top_level.strip():
            return top_level.strip()
        return f"sample-{index}"
