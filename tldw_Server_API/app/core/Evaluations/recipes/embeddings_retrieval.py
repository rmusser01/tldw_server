"""V1 retrieval-only embeddings recipe helpers."""

from __future__ import annotations

import json
import math
import statistics
from typing import Any

from tldw_Server_API.app.api.v1.schemas.evaluation_recipe_schemas import (
    ConfidenceSummary,
    RecipeManifest,
    RecommendationSlot,
)
from tldw_Server_API.app.core.Evaluations.metrics_retrieval import (
    mrr,
    ndcg,
    recall_at_k,
)
from tldw_Server_API.app.core.Evaluations.recipes.base import RecipeDefinition

_SUPPORTED_COMPARISON_MODES = {"embedding_only", "retrieval_stack"}
_RECOMMENDATION_SLOTS = ("best_overall", "best_cheap", "best_local")


class EmbeddingsRetrievalRecipe(RecipeDefinition):
    """Small retrieval recipe focused on dataset validation and report assembly."""

    manifest = RecipeManifest(
        recipe_id="embeddings_model_selection",
        recipe_version="1",
        name="Embeddings Model Selection",
        description="Compare retrieval-focused embedding candidates with shared recommendation slots.",
        supported_modes=["labeled", "unlabeled"],
        tags=["embeddings", "retrieval", "recipe-v1"],
        capabilities={
            "guided_ui": True,
            "rag_embedding_selection": True,
            "media_scoped_execution": True,
            "source_labeling": {
                "supported": True,
                "source_id_contract": {"kind": "media_id", "type": "integer"},
                "advanced_manual_ids": True,
            },
            "candidate_discovery": {
                "supported": True,
                "endpoint": "/api/v1/evaluations/recipes/embeddings_model_selection/candidates",
                "runnable_statuses": [
                    "ready",
                    "missing_key",
                    "disallowed_provider",
                    "disallowed_model",
                    "quota_risk",
                    "unknown",
                ],
            },
            "apply_target": {
                "preview_supported": True,
                "live_apply_supported": False,
                "config_section": "Embeddings",
                "config_keys": ["embedding_provider", "embedding_model"],
            },
        },
        default_run_config={
            "comparison_mode": "embedding_only",
            "top_k": 10,
            "hybrid_alpha": 0.7,
            "guided_source_labeling": True,
            "source_id_contract": "media_id",
            "candidates": [],
        },
    )

    def validate_dataset(
        self,
        dataset: list[dict[str, Any]],
        *,
        run_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        errors: list[str] = []
        warnings: list[str] = []
        normalized_run_config = run_config if isinstance(run_config, dict) else {}
        if run_config is not None and not isinstance(run_config, dict):
            errors.append("run_config must be an object.")

        guided_source_labeling = bool(
            normalized_run_config.get(
                "guided_source_labeling",
                self.manifest.default_run_config["guided_source_labeling"],
            )
        )
        source_id_contract = str(
            normalized_run_config.get(
                "source_id_contract",
                self.manifest.default_run_config["source_id_contract"],
            )
            or ""
        )
        samples = [dict(sample) for sample in (dataset or [])]
        if not samples:
            errors.append("Dataset must contain at least one sample.")
            return {
                "valid": False,
                "errors": errors,
                "warnings": warnings,
                "dataset_mode": None,
                "sample_count": 0,
                "review_sample": {"required": False, "sample_size": 0, "sample_query_ids": []},
            }

        labeled_flags: list[bool] = []
        query_ids: list[str] = []
        for index, sample in enumerate(samples):
            query_id = str(sample.get("query_id") or "").strip()
            if not query_id:
                errors.append(f"Dataset sample {index} must include a non-empty query_id.")
            else:
                query_ids.append(query_id)

            input_value = str(sample.get("input") or "").strip()
            if not input_value:
                errors.append(f"Dataset sample {index} must include a non-empty input.")

            expected_ids = sample.get("expected_ids")
            if expected_ids is None:
                labeled_flags.append(False)
                continue

            if not isinstance(expected_ids, list) or not [str(value).strip() for value in expected_ids if str(value).strip()]:
                errors.append(f"Dataset sample {index} must include a non-empty expected_ids list for labeled retrieval.")
                labeled_flags.append(False)
                continue
            invalid_expected_ids = [
                str(value).strip()
                for value in expected_ids
                if str(value).strip() and not str(value).strip().isdigit()
            ]
            if source_id_contract == "media_id" and invalid_expected_ids:
                errors.append(
                    f"Dataset sample {index} expected_ids must contain integer media id values for the media_id source contract."
                )
                labeled_flags.append(False)
                continue

            labeled_flags.append(True)

        dataset_mode = self._detect_dataset_mode(labeled_flags)
        if dataset_mode == "mixed":
            errors.append("Dataset must use a consistent labeling mode for retrieval samples.")

        review_sample = self._reserve_review_sample(query_ids) if dataset_mode == "unlabeled" else {
            "required": False,
            "sample_size": 0,
            "sample_query_ids": [],
        }
        if guided_source_labeling and dataset_mode == "unlabeled":
            warnings.append(
                "Guided source labeling is enabled but no expected sources were provided; "
                "reviewers must add expected sources before labeled retrieval metrics are available."
            )
        return {
            "valid": not errors,
            "errors": errors,
            "warnings": warnings,
            "dataset_mode": dataset_mode,
            "sample_count": len(samples),
            "review_sample": review_sample,
        }

    def normalize_run_config(self, run_config: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(run_config, dict):
            raise ValueError("run_config must be an object.")
        comparison_mode = str(run_config.get("comparison_mode") or "").strip()
        if comparison_mode not in _SUPPORTED_COMPARISON_MODES:
            raise ValueError(
                "run_config.comparison_mode must be one of: embedding_only, retrieval_stack."
            )
        candidates = run_config.get("candidates") or []
        if not isinstance(candidates, list):
            raise ValueError("run_config.candidates must be a list when provided.")
        normalized_candidates = sorted(
            (dict(candidate) for candidate in candidates),
            key=lambda candidate: (
                str(candidate.get("candidate_id") or "").strip(),
                str(candidate.get("model") or "").strip(),
                str(candidate.get("provider") or "").strip(),
            ),
        )
        normalized: dict[str, Any] = {
            "comparison_mode": comparison_mode,
            "candidates": normalized_candidates,
        }
        media_ids = run_config.get("media_ids")
        if media_ids is not None:
            if not isinstance(media_ids, list):
                raise ValueError("run_config.media_ids must be a list when provided.")
            normalized["media_ids"] = [int(media_id) for media_id in media_ids]
        if run_config.get("top_k") is not None:
            normalized["top_k"] = int(run_config["top_k"])
        if run_config.get("hybrid_alpha") is not None:
            normalized["hybrid_alpha"] = float(run_config["hybrid_alpha"])
        if "guided_source_labeling" in run_config:
            normalized["guided_source_labeling"] = bool(run_config["guided_source_labeling"])
        if run_config.get("source_id_contract") is not None:
            normalized["source_id_contract"] = str(run_config["source_id_contract"]).strip()
        return normalized

    def build_report(
        self,
        *,
        dataset_mode: str,
        review_sample: dict[str, Any],
        candidate_results: list[dict[str, Any]],
        k: int = 10,
    ) -> dict[str, Any]:
        candidate_summaries = [
            self._summarize_candidate(candidate_result, k=k)
            for candidate_result in candidate_results
        ]
        candidate_summaries = [summary for summary in candidate_summaries if summary is not None]

        best_overall = self._pick_best_overall(candidate_summaries)
        best_cheap = self._pick_best_cheap(candidate_summaries)
        best_local = self._pick_best_local(candidate_summaries)

        winner_margin = self._winner_margin(candidate_summaries, best_overall)
        spreads = [summary["metrics"]["quality_score"] for summary in candidate_summaries]
        spread = statistics.pstdev(spreads) if len(spreads) > 1 else 0.0
        judge_agreements = [
            float(summary["judge_agreement"])
            for summary in candidate_summaries
            if summary.get("judge_agreement") is not None
        ]
        sample_count = 0
        if candidate_summaries:
            sample_count = min(int(summary["sample_count"]) for summary in candidate_summaries)
        confidence_inputs = {
            "sample_count": sample_count,
            "spread": spread,
            "winner_margin": winner_margin,
            "judge_agreement": (
                sum(judge_agreements) / len(judge_agreements)
                if judge_agreements
                else None
            ),
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
            judge_agreement=confidence_inputs["judge_agreement"],
            notes=(
                "Unlabeled run reserves review sample."
                if dataset_mode == "unlabeled"
                else None
            ),
        )

        slots = {
            "best_overall": best_overall,
            "best_cheap": best_cheap,
            "best_local": best_local,
        }
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
            slots.setdefault(slot_name, None)
            recommendation_slots.setdefault(
                slot_name,
                self._build_slot(slot_name, None, reason_code="not_available"),
            )

        return {
            "dataset_mode": dataset_mode,
            "review_sample": review_sample,
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

    def _detect_dataset_mode(self, labeled_flags: list[bool]) -> str | None:
        if not labeled_flags:
            return None
        if all(labeled_flags):
            return "labeled"
        if not any(labeled_flags):
            return "unlabeled"
        return "mixed"

    def _reserve_review_sample(self, query_ids: list[str]) -> dict[str, Any]:
        if not query_ids:
            return {"required": False, "sample_size": 0, "sample_query_ids": []}
        sample_size = min(len(query_ids), min(25, max(3, math.ceil(len(query_ids) * 0.2))))
        return {
            "required": True,
            "sample_size": sample_size,
            "sample_query_ids": query_ids[:sample_size],
        }

    def _summarize_candidate(self, candidate_result: dict[str, Any], *, k: int) -> dict[str, Any] | None:
        query_results = self._coerce_query_results(candidate_result)
        aggregated_metrics = self._coerce_metrics_payload(candidate_result.get("metrics"))
        if not query_results and not aggregated_metrics:
            return None

        recall_scores: list[float] = []
        mrr_scores: list[float] = []
        ndcg_scores: list[float] = []
        latencies: list[float] = []
        for query_result in query_results:
            ranked_ids = [str(value) for value in (query_result.get("ranked_ids") or [])]
            expected_ids = [str(value) for value in (query_result.get("expected_ids") or [])]
            metrics = self._coerce_metrics_payload(
                query_result.get("metrics") or query_result.get("metrics_json")
            )
            recall_scores.append(
                self._resolve_metric_value(
                    direct_value=query_result.get("recall_at_k"),
                    metrics=metrics,
                    metric_name="recall_at_k",
                    ranked_ids=ranked_ids,
                    expected_ids=expected_ids,
                    k=k,
                    calculator=recall_at_k,
                )
            )
            mrr_scores.append(
                self._resolve_metric_value(
                    direct_value=query_result.get("mrr"),
                    metrics=metrics,
                    metric_name="mrr",
                    ranked_ids=ranked_ids,
                    expected_ids=expected_ids,
                    k=k,
                    calculator=mrr,
                )
            )
            ndcg_scores.append(
                self._resolve_metric_value(
                    direct_value=query_result.get("ndcg"),
                    metrics=metrics,
                    metric_name="ndcg",
                    ranked_ids=ranked_ids,
                    expected_ids=expected_ids,
                    k=k,
                    calculator=ndcg,
                )
            )
            latencies.append(float(query_result.get("latency_ms") or 0.0))

        if not recall_scores and aggregated_metrics:
            recall_value = float(aggregated_metrics.get("recall_at_k") or 0.0)
            mrr_value = float(aggregated_metrics.get("mrr") or 0.0)
            ndcg_value = float(aggregated_metrics.get("ndcg") or 0.0)
            latency_value = float(
                aggregated_metrics.get("latency_ms_mean")
                or aggregated_metrics.get("latency_ms")
                or 0.0
            )
            sample_count = int(candidate_result.get("sample_count") or 0)
        else:
            recall_value = sum(recall_scores) / len(recall_scores)
            mrr_value = sum(mrr_scores) / len(mrr_scores)
            ndcg_value = sum(ndcg_scores) / len(ndcg_scores)
            latency_value = sum(latencies) / len(latencies)
            sample_count = len(query_results)

        cost_value = candidate_result.get("cost_usd")
        cost_usd = float(cost_value) if cost_value is not None else None
        quality_score = (recall_value * 0.4) + (mrr_value * 0.25) + (ndcg_value * 0.35)
        is_local = candidate_result.get("is_local")
        candidate_run_id = candidate_result.get("candidate_run_id")

        return {
            "candidate_id": str(candidate_result.get("candidate_id") or candidate_result.get("model") or ""),
            "candidate_run_id": str(candidate_run_id).strip() if candidate_run_id is not None else None,
            "model": str(candidate_result.get("model") or ""),
            "provider": str(candidate_result.get("provider") or ""),
            "is_local": bool(is_local) if is_local is not None else None,
            "cost_usd": cost_usd,
            "latency_ms": latency_value,
            "metrics": {
                "recall_at_k": recall_value,
                "mrr": mrr_value,
                "ndcg": ndcg_value,
                "quality_score": quality_score,
            },
            "recall_at_k": recall_value,
            "mrr": mrr_value,
            "ndcg": ndcg_value,
            "quality_score": quality_score,
            "sample_count": sample_count,
            "judge_agreement": candidate_result.get("judge_agreement"),
        }

    def _pick_best_overall(self, candidate_summaries: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not candidate_summaries:
            return None
        return max(
            candidate_summaries,
            key=lambda summary: (
                summary["metrics"]["quality_score"],
                (
                    -summary["cost_usd"]
                    if summary.get("cost_usd") is not None
                    else float("-inf")
                ),
                -summary["latency_ms"],
            ),
        )

    def _pick_best_cheap(self, candidate_summaries: list[dict[str, Any]]) -> dict[str, Any] | None:
        affordable_candidates = [
            summary for summary in candidate_summaries if summary.get("cost_usd") is not None
        ]
        if not affordable_candidates:
            return None
        return min(
            affordable_candidates,
            key=lambda summary: (
                summary["cost_usd"],
                -summary["metrics"]["quality_score"],
                summary["latency_ms"],
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
                summary["metrics"]["quality_score"],
                -summary["latency_ms"],
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
            key=lambda summary: summary["metrics"]["quality_score"],
            reverse=True,
        )
        return max(
            0.0,
            float(ordered[0]["metrics"]["quality_score"]) - float(ordered[1]["metrics"]["quality_score"]),
        )

    def _confidence_score(self, *, sample_count: int, spread: float, winner_margin: float) -> float:
        sample_factor = min(1.0, sample_count / 10.0)
        margin_factor = min(1.0, winner_margin / 0.25)
        spread_penalty = min(1.0, spread)
        confidence = 0.35 + (sample_factor * 0.35) + (margin_factor * 0.35) - (spread_penalty * 0.15)
        return max(0.0, min(1.0, confidence))

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
                explanation=f"No candidate qualified for '{slot_name}'.",
            )

        candidate_id = str(candidate_summary.get("candidate_id") or "").strip() or None
        candidate_run_id = (
            str(candidate_summary.get("candidate_run_id") or "").strip() or None
        )
        provider, model = self._provider_and_model_for_metadata(candidate_summary)
        apply_warnings = self._apply_warnings(candidate_summary)
        return RecommendationSlot(
            candidate_run_id=candidate_run_id,
            reason_code=reason_code,
            explanation=(
                f"{candidate_summary.get('model') or candidate_summary.get('candidate_id')} "
                f"won '{slot_name}' for this retrieval run."
            ),
            confidence=confidence,
            metadata={
                "candidate_id": candidate_id,
                "candidate_run_id": candidate_run_id,
                "provider": provider,
                "model": model,
                "is_local": candidate_summary.get("is_local"),
                "apply_eligible": not apply_warnings,
                "apply_warnings": apply_warnings,
                "confidence": confidence,
                "quality_score": candidate_summary["metrics"]["quality_score"],
                "cost_usd": candidate_summary.get("cost_usd"),
                "latency_ms": candidate_summary.get("latency_ms"),
            },
        )

    def _provider_and_model_for_metadata(
        self,
        candidate_summary: dict[str, Any],
    ) -> tuple[str | None, str | None]:
        provider = str(candidate_summary.get("provider") or "").strip() or None
        raw_model = str(candidate_summary.get("model") or "").strip()
        if ":" in raw_model:
            model_provider, model_name = raw_model.split(":", 1)
            provider = provider or model_provider.strip() or None
            raw_model = model_name.strip()
        return provider, raw_model or None

    def _apply_warnings(self, candidate_summary: dict[str, Any]) -> list[str]:
        warnings: list[str] = []
        provider, model = self._provider_and_model_for_metadata(candidate_summary)
        if not str(candidate_summary.get("candidate_run_id") or "").strip():
            warnings.append("missing_candidate_run_id")
        if not provider:
            warnings.append("missing_provider")
        if not model:
            warnings.append("missing_model")
        return warnings

    def _coerce_query_results(self, candidate_result: dict[str, Any]) -> list[dict[str, Any]]:
        raw_query_results = candidate_result.get("query_results")
        if raw_query_results is None:
            raw_query_results = candidate_result.get("results")
        if not isinstance(raw_query_results, list):
            return []
        return [dict(query_result) for query_result in raw_query_results]

    def _coerce_metrics_payload(self, metrics: Any) -> dict[str, Any]:
        if metrics is None:
            return {}
        if isinstance(metrics, dict):
            return dict(metrics)
        if isinstance(metrics, str):
            try:
                parsed = json.loads(metrics)
            except (TypeError, ValueError):
                return {}
            if isinstance(parsed, dict):
                return dict(parsed)
        return {}

    def _resolve_metric_value(
        self,
        *,
        direct_value: Any,
        metrics: dict[str, Any],
        metric_name: str,
        ranked_ids: list[str],
        expected_ids: list[str],
        k: int,
        calculator: Any,
    ) -> float:
        if direct_value is not None:
            return float(direct_value)
        if metrics.get(metric_name) is not None:
            return float(metrics[metric_name])
        if expected_ids:
            return float(calculator(ranked_ids, expected_ids, k))
        return 0.0
