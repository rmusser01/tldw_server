"""V1 RAG retrieval tuning recipe helpers."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping
from typing import Any

from tldw_Server_API.app.core.Evaluations.evaluation_recipe_schemas import (
    ConfidenceSummary,
    RecipeManifest,
    RecommendationSlot,
)
from tldw_Server_API.app.core.Evaluations.recipes.base import RecipeDefinition

from .rag_retrieval_tuning_candidates import (
    build_auto_sweep,
    normalize_candidate_config,
)
from .rag_retrieval_tuning_execution import (
    CandidateIndexPlan,
    build_unified_rag_request as build_unified_rag_request_helper,
    plan_candidate_indexes as plan_candidate_indexes_helper,
    summarize_candidate_metrics as summarize_candidate_metrics_helper,
)

_SUPPORTED_CORPUS_SOURCES = {"media_db", "notes"}
_SUPPORTED_CANDIDATE_CREATION_MODES = {"auto_sweep", "manual"}
_DEFAULT_WEAK_SUPERVISION_BUDGET = {
    "review_sample_fraction": 0.2,
    "max_review_samples": 25,
    "min_review_samples": 3,
    "synthetic_query_limit": 20,
}
_TARGET_KEYS = {
    "relevant_media_ids",
    "relevant_note_ids",
    "relevant_chunk_ids",
    "relevant_spans",
}
_ALLOWED_SPAN_SOURCES = {"media_db", "notes"}
_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}
_GRADE_MIN = 0
_GRADE_MAX = 3
_RECOMMENDATION_SLOTS = ("best_overall", "best_cheap", "best_local")


class RAGRetrievalTuningRecipe(RecipeDefinition):
    """Retrieval-only recipe for corpus-scoped RAG tuning."""

    manifest = RecipeManifest(
        recipe_id="rag_retrieval_tuning",
        recipe_version="1",
        name="RAG Retrieval Tuning",
        description="Tune retrieval candidates against run-level media_db and notes corpora.",
        launchable=True,
        supported_modes=["labeled", "unlabeled"],
        tags=["rag", "retrieval", "tuning", "recipe-v1"],
        capabilities={
            "corpus_sources": ["media_db", "notes"],
            "candidate_creation_modes": ["auto_sweep", "manual"],
            "graded_relevance_scale": {"min": 0, "max": 3},
        },
        default_run_config={
            "candidate_creation_mode": "auto_sweep",
            "corpus_scope": {"sources": ["media_db", "notes"]},
            "weak_supervision_budget": dict(_DEFAULT_WEAK_SUPERVISION_BUDGET),
        },
    )

    def validate_dataset(
        self,
        dataset: list[dict[str, Any]],
        *,
        run_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Validate dataset shape, corpus scope, and supervision consistency."""
        raw_samples = list(dataset or [])
        sample_count = len(raw_samples)
        errors: list[str] = []
        normalized_scope = self._validate_corpus_scope_for_dataset(run_config, errors)
        weak_supervision_budget = self._resolve_weak_supervision_budget(run_config, errors=errors)
        if sample_count == 0:
            errors.append("Dataset must contain at least one sample.")
            return self._build_validation_result(
                valid=False,
                errors=errors,
                dataset_mode=None,
                sample_count=0,
                review_sample={"required": False, "sample_size": 0, "sample_ids": []},
                corpus_scope=normalized_scope,
                weak_supervision_budget=weak_supervision_budget,
            )

        labeled_flags: list[bool] = []
        sample_ids: list[str] = []
        for index, raw_sample in enumerate(raw_samples):
            if not isinstance(raw_sample, Mapping):
                errors.append(f"Dataset sample {index} must be an object.")
                continue
            sample = dict(raw_sample)
            sample_ids.append(self._extract_sample_id(sample, index))
            query = self._extract_query(sample)
            if not query:
                errors.append(f"Dataset sample {index} must include a non-empty query value.")

            targets = sample.get("targets")
            labeled_flags.append(
                self._validate_targets(
                    targets,
                    normalized_scope,
                    errors,
                    index=index,
                    run_config=run_config or {},
                )
            )

        dataset_mode = self._detect_dataset_mode(labeled_flags)
        if dataset_mode == "mixed":
            errors.append(
                "Dataset must use a consistent labeling mode; do not mix labeled and unlabeled samples."
            )

        review_sample = (
            self._reserve_review_sample(sample_ids, weak_supervision_budget)
            if dataset_mode == "unlabeled"
            else {"required": False, "sample_size": 0, "sample_ids": []}
        )
        return self._build_validation_result(
            valid=not errors,
            errors=errors,
            dataset_mode=dataset_mode,
            sample_count=sample_count,
            review_sample=review_sample,
            corpus_scope=normalized_scope,
            weak_supervision_budget=weak_supervision_budget,
        )

    def normalize_run_config(self, run_config: dict[str, Any]) -> dict[str, Any]:
        """Normalize the bounded run config for the V1 recipe."""
        if not isinstance(run_config, dict):
            raise ValueError("run_config must be an object.")

        candidate_creation_mode = str(
            run_config.get("candidate_creation_mode") or "auto_sweep"
        ).strip().lower()
        if candidate_creation_mode not in _SUPPORTED_CANDIDATE_CREATION_MODES:
            raise ValueError(
                "run_config.candidate_creation_mode must be one of: auto_sweep, manual."
            )

        normalized_scope = self._normalize_corpus_scope(run_config, [], required=True)
        weak_supervision_budget = self._resolve_weak_supervision_budget(run_config)

        if candidate_creation_mode == "manual":
            candidates = run_config.get("candidates")
            if not isinstance(candidates, list) or not candidates:
                raise ValueError("run_config.candidates must contain at least one candidate.")
            normalized_candidates = [
                normalize_candidate_config(candidate, default_candidate_id=f"manual-{index + 1}")
                for index, candidate in enumerate(candidates)
            ]
        else:
            normalized_candidates = build_auto_sweep(self._extract_auto_sweep_base_config(run_config))

        return {
            "candidate_creation_mode": candidate_creation_mode,
            "corpus_scope": normalized_scope,
            "weak_supervision_budget": weak_supervision_budget,
            "candidates": normalized_candidates,
        }

    def plan_candidate_indexes(
        self,
        *,
        corpus_scope: Mapping[str, Any],
        candidates: list[dict[str, Any]],
        dataset_content_hash: str,
        owner_user_id: str,
    ) -> dict[str, CandidateIndexPlan]:
        """Plan isolated index keys for execution without mutating the live index."""
        return plan_candidate_indexes_helper(
            corpus_scope=corpus_scope,
            candidates=candidates,
            dataset_content_hash=dataset_content_hash,
            owner_user_id=owner_user_id,
        )

    def build_unified_rag_request(
        self,
        *,
        query: str,
        corpus_scope: Mapping[str, Any],
        candidate: Mapping[str, Any],
        index_key: str | None = None,
    ) -> Any:
        """Construct a unified RAG request for a candidate execution."""
        return build_unified_rag_request_helper(
            query=query,
            corpus_scope=corpus_scope,
            candidate=candidate,
            index_key=index_key,
        )

    def summarize_candidate_metrics(
        self,
        *,
        first_pass_hits: list[Mapping[str, Any]],
        reranked_hits: list[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Summarize first-pass and post-rerank metrics for a candidate."""
        return summarize_candidate_metrics_helper(
            first_pass_hits=first_pass_hits,
            reranked_hits=reranked_hits,
        )

    def build_report(
        self,
        *,
        dataset_mode: str,
        review_sample: dict[str, Any],
        corpus_scope: dict[str, Any],
        candidate_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        candidate_summaries = [
            self._summarize_candidate(candidate_result)
            for candidate_result in candidate_results
        ]
        candidate_summaries = [summary for summary in candidate_summaries if summary is not None]
        candidate_summaries.sort(
            key=lambda summary: (
                -float(summary["metrics"]["retrieval_quality_score"]),
                float(summary["latency_ms"] or 0.0),
                str(summary["candidate_id"]),
            )
        )

        best_overall = self._pick_best_overall(candidate_summaries)
        best_cheap = self._pick_best_cheap(candidate_summaries)
        best_local = self._pick_best_local(candidate_summaries)

        quality_scores = [
            float(summary["metrics"]["retrieval_quality_score"])
            for summary in candidate_summaries
        ]
        sample_count = min(
            (int(summary["sample_count"]) for summary in candidate_summaries),
            default=0,
        )
        spread = statistics.pstdev(quality_scores) if len(quality_scores) > 1 else 0.0
        winner_margin = self._winner_margin(candidate_summaries)
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
            notes=(
                "Unlabeled run reserves review sample."
                if dataset_mode == "unlabeled"
                else None
            ),
        )

        recommendation_slots = {
            "best_overall": self._build_slot(
                "best_overall",
                best_overall,
                reason_code="highest_retrieval_quality",
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
                reason_code="best_local_retrieval_quality",
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
            "corpus_scope": corpus_scope,
            "candidates": candidate_summaries,
            "best_overall": best_overall,
            "best_cheap": best_cheap,
            "best_local": best_local,
            "recommendation_slots": {
                slot_name: slot.model_dump(mode="json")
                for slot_name, slot in recommendation_slots.items()
            },
            "confidence_summary": confidence_summary.model_dump(mode="json"),
        }

    def _build_validation_result(
        self,
        *,
        valid: bool,
        errors: list[str],
        dataset_mode: str | None,
        sample_count: int,
        review_sample: dict[str, Any],
        corpus_scope: dict[str, Any] | None,
        weak_supervision_budget: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "valid": valid,
            "errors": errors,
            "dataset_mode": dataset_mode,
            "sample_count": sample_count,
            "review_sample": review_sample,
            "corpus_scope": corpus_scope,
            "weak_supervision_budget": weak_supervision_budget,
            "graded_relevance": {"min": _GRADE_MIN, "max": _GRADE_MAX},
        }

    def _summarize_candidate(self, candidate_result: dict[str, Any]) -> dict[str, Any] | None:
        metrics = dict(candidate_result.get("metrics") or {})
        query_results = [
            dict(query_result)
            for query_result in (candidate_result.get("query_results") or [])
            if isinstance(query_result, dict)
        ]
        pre_rerank = metrics.get("first_pass_recall_score", metrics.get("pre_rerank_recall_at_k"))
        post_rerank = metrics.get(
            "post_rerank_quality_score",
            metrics.get("post_rerank_ndcg_at_k"),
        )

        if pre_rerank is None or post_rerank is None:
            pre_values = []
            post_values = []
            latency_values = []
            for query_result in query_results:
                query_metrics = dict(query_result.get("metrics") or {})
                if query_metrics.get("first_pass_recall_score") is not None:
                    pre_values.append(float(query_metrics["first_pass_recall_score"]))
                elif query_metrics.get("pre_rerank_recall_at_k") is not None:
                    pre_values.append(float(query_metrics["pre_rerank_recall_at_k"]))
                if query_metrics.get("post_rerank_quality_score") is not None:
                    post_values.append(float(query_metrics["post_rerank_quality_score"]))
                elif query_metrics.get("post_rerank_ndcg_at_k") is not None:
                    post_values.append(float(query_metrics["post_rerank_ndcg_at_k"]))
                if query_result.get("latency_ms") is not None:
                    latency_values.append(float(query_result["latency_ms"]))
            if pre_rerank is None:
                pre_rerank = statistics.mean(pre_values) if pre_values else 0.0
            if post_rerank is None:
                post_rerank = statistics.mean(post_values) if post_values else 0.0
            latency_ms = (
                float(candidate_result["latency_ms"])
                if candidate_result.get("latency_ms") is not None
                else (statistics.mean(latency_values) if latency_values else None)
            )
        else:
            latency_ms = (
                float(candidate_result["latency_ms"])
                if candidate_result.get("latency_ms") is not None
                else None
            )

        pre_rerank = float(pre_rerank)
        post_rerank = float(post_rerank)
        retrieval_quality_score = (0.6 * pre_rerank) + (0.4 * post_rerank)
        sample_count = len(query_results) or int(candidate_result.get("sample_count") or 0)

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
                "pre_rerank_recall_at_k": pre_rerank,
                "post_rerank_ndcg_at_k": post_rerank,
                "retrieval_quality_score": retrieval_quality_score,
            },
        }

    def _pick_best_overall(self, candidate_summaries: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not candidate_summaries:
            return None
        return max(
            candidate_summaries,
            key=lambda summary: (
                float(summary["metrics"]["retrieval_quality_score"]),
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
                -float(summary["metrics"]["retrieval_quality_score"]),
                float(summary["latency_ms"] or 0.0),
            ),
        )

    def _pick_best_local(self, candidate_summaries: list[dict[str, Any]]) -> dict[str, Any] | None:
        local_candidates = [summary for summary in candidate_summaries if summary.get("is_local")]
        if not local_candidates:
            return None
        return max(
            local_candidates,
            key=lambda summary: (
                float(summary["metrics"]["retrieval_quality_score"]),
                -float(summary["latency_ms"] or 0.0),
            ),
        )

    def _winner_margin(self, candidate_summaries: list[dict[str, Any]]) -> float:
        if len(candidate_summaries) < 2:
            return 1.0 if candidate_summaries else 0.0
        ordered = sorted(
            float(summary["metrics"]["retrieval_quality_score"])
            for summary in candidate_summaries
        )
        return ordered[-1] - ordered[-2]

    def _confidence_score(self, *, sample_count: int, spread: float, winner_margin: float) -> float:
        sample_factor = min(sample_count / 25.0, 1.0)
        margin_factor = min(max(winner_margin, 0.0) / 0.25, 1.0)
        spread_penalty = min(max(spread, 0.0), 0.5)
        return round(max(0.0, min(1.0, (0.5 * sample_factor) + (0.5 * margin_factor) - (0.25 * spread_penalty))), 3)

    def _build_slot(
        self,
        slot_name: str,
        summary: dict[str, Any] | None,
        *,
        reason_code: str,
        confidence: float | None = None,
    ) -> RecommendationSlot:
        if summary is None:
            return RecommendationSlot(
                candidate_run_id=None,
                reason_code="not_available",
                explanation=f"No recommendation is available for '{slot_name}'.",
            )
        return RecommendationSlot(
            candidate_run_id=summary.get("candidate_run_id"),
            reason_code=reason_code,
            explanation=(
                f"Selected '{summary.get('candidate_id')}' for {slot_name} "
                f"with retrieval quality {summary['metrics']['retrieval_quality_score']:.3f}."
            ),
            confidence=confidence,
        )

    def _normalize_corpus_scope(
        self,
        run_config: dict[str, Any] | None,
        errors: list[str],
        *,
        required: bool,
    ) -> dict[str, Any] | None:
        if not run_config or run_config.get("corpus_scope") is None:
            if required:
                raise ValueError("run_config.corpus_scope is required.")
            return None

        raw_scope = run_config.get("corpus_scope")
        if not isinstance(raw_scope, dict):
            raise ValueError("run_config.corpus_scope must be an object.")

        sources = raw_scope.get("sources")
        if not isinstance(sources, list) or not sources:
            raise ValueError("run_config.corpus_scope.sources must contain at least one source.")

        normalized_sources = []
        for source in sources:
            normalized_source = str(source).strip()
            if normalized_source not in _SUPPORTED_CORPUS_SOURCES:
                errors.append(
                    f"run_config.corpus_scope.sources may only contain media_db and notes; got {normalized_source!r}."
                )
                continue
            if normalized_source not in normalized_sources:
                normalized_sources.append(normalized_source)

        if required and (errors or not normalized_sources):
            raise ValueError(
                "run_config.corpus_scope.sources may only contain media_db and notes."
            )

        normalized_scope: dict[str, Any] = {"sources": normalized_sources}
        for field_name in ("media_ids", "note_ids"):
            if raw_scope.get(field_name) is None:
                continue
            if not isinstance(raw_scope[field_name], list):
                raise ValueError(f"run_config.corpus_scope.{field_name} must be a list when provided.")
            normalized_scope[field_name] = [
                self._normalize_reference_id(item)
                for item in raw_scope[field_name]
                if str(item).strip()
            ]
        if raw_scope.get("indexing_fixed") is not None:
            normalized_scope["indexing_fixed"] = self._parse_bool_value(
                raw_scope["indexing_fixed"],
                field_name="run_config.corpus_scope.indexing_fixed",
            )
        return normalized_scope

    def _resolve_weak_supervision_budget(
        self,
        run_config: dict[str, Any] | None,
        *,
        errors: list[str] | None = None,
    ) -> dict[str, Any]:
        raw_budget = run_config.get("weak_supervision_budget") if run_config else None
        if not isinstance(raw_budget, dict):
            return dict(_DEFAULT_WEAK_SUPERVISION_BUDGET)

        normalized = dict(_DEFAULT_WEAK_SUPERVISION_BUDGET)
        for key in normalized:
            if raw_budget.get(key) is None:
                continue
            try:
                if key in {"review_sample_fraction"}:
                    normalized[key] = float(raw_budget[key])
                else:
                    normalized[key] = int(raw_budget[key])
            except (TypeError, ValueError):
                if errors is not None:
                    errors.append(
                        f"run_config.weak_supervision_budget.{key} must be a numeric value."
                    )
                    continue
                raise
        normalized["review_sample_fraction"] = min(
            1.0,
            max(0.0, float(normalized["review_sample_fraction"])),
        )
        normalized["max_review_samples"] = max(1, int(normalized["max_review_samples"]))
        normalized["min_review_samples"] = max(1, int(normalized["min_review_samples"]))
        normalized["synthetic_query_limit"] = max(0, int(normalized["synthetic_query_limit"]))
        return normalized

    def _validate_corpus_scope_for_dataset(
        self,
        run_config: dict[str, Any] | None,
        errors: list[str],
    ) -> dict[str, Any] | None:
        try:
            return self._normalize_corpus_scope(run_config, errors, required=True)
        except ValueError as exc:
            errors.append(str(exc))
            return None

    def _validate_targets(
        self,
        targets: Any,
        normalized_scope: dict[str, Any] | None,
        errors: list[str],
        *,
        index: int,
        run_config: dict[str, Any],
    ) -> bool:
        if targets is None:
            return False
        if not isinstance(targets, dict):
            errors.append(f"Dataset sample {index} targets must be an object when provided.")
            return False
        if not targets:
            errors.append(
                f"Dataset sample {index} targets must include at least one supported target field."
            )
            return False

        unknown_target_keys = sorted(set(targets) - _TARGET_KEYS)
        if unknown_target_keys:
            errors.append(f"Dataset sample {index} contains unsupported target fields.")

        labeled = False
        if "relevant_media_ids" in targets:
            labeled = True
            self._validate_target_list(
                targets["relevant_media_ids"],
                index=index,
                target_name="relevant_media_ids",
                errors=errors,
                normalized_scope=normalized_scope,
                scope_field="media_ids",
            )
            self._require_scope_source(normalized_scope, "media_db", errors, index=index)
        if "relevant_note_ids" in targets:
            labeled = True
            self._validate_target_list(
                targets["relevant_note_ids"],
                index=index,
                target_name="relevant_note_ids",
                errors=errors,
                normalized_scope=normalized_scope,
                scope_field="note_ids",
            )
            self._require_scope_source(normalized_scope, "notes", errors, index=index)
        if "relevant_spans" in targets:
            labeled = True
            self._validate_spans(
                targets["relevant_spans"],
                normalized_scope,
                errors,
                index=index,
            )
        if "relevant_chunk_ids" in targets:
            labeled = True
            if not self._is_fixed_indexing(
                run_config,
                normalized_scope,
                errors=errors,
                index=index,
            ):
                errors.append(
                    f"Dataset sample {index} chunk-level targets require stable spans or fixed indexing."
                )
            self._validate_target_list(
                targets["relevant_chunk_ids"],
                index=index,
                target_name="relevant_chunk_ids",
                errors=errors,
            )
        return labeled

    def _validate_target_list(
        self,
        items: Any,
        *,
        index: int,
        target_name: str,
        errors: list[str],
        normalized_scope: dict[str, Any] | None = None,
        scope_field: str | None = None,
    ) -> None:
        if not isinstance(items, list) or not items:
            errors.append(f"Dataset sample {index} must include a non-empty {target_name} list.")
            return
        for item in items:
            grade = 1
            target_id = item
            if isinstance(item, dict):
                target_id = item.get("id") or item.get("record_id")
                if item.get("grade") is not None:
                    parsed_grade = self._parse_int_field(
                        item["grade"],
                        errors=errors,
                        index=index,
                        field_name=f"{target_name} integer grade",
                    )
                    if parsed_grade is not None:
                        grade = parsed_grade
            if not str(target_id).strip():
                errors.append(f"Dataset sample {index} has an empty target id in {target_name}.")
            elif scope_field:
                self._validate_reference_membership(
                    normalized_scope,
                    scope_field=scope_field,
                    candidate_id=target_id,
                    errors=errors,
                    index=index,
                )
            if grade < _GRADE_MIN or grade > _GRADE_MAX:
                errors.append(
                    f"Dataset sample {index} target grades must be between {_GRADE_MIN} and {_GRADE_MAX}."
                )

    def _validate_spans(
        self,
        spans: Any,
        normalized_scope: dict[str, Any] | None,
        errors: list[str],
        *,
        index: int,
    ) -> None:
        if not isinstance(spans, list) or not spans:
            errors.append(f"Dataset sample {index} must include a non-empty relevant_spans list.")
            return
        for span in spans:
            if not isinstance(span, dict):
                errors.append(f"Dataset sample {index} relevant_spans entries must be objects.")
                continue
            source = str(span.get("source") or "").strip()
            if source not in _ALLOWED_SPAN_SOURCES:
                errors.append(f"Dataset sample {index} relevant_spans entries must use media_db or notes.")
            elif normalized_scope and source not in normalized_scope.get("sources", []):
                errors.append(
                    f"Dataset sample {index} relevant_spans entries must stay within the run corpus_scope."
                )
            record_id = str(span.get("record_id") or "").strip()
            if not record_id:
                errors.append(f"Dataset sample {index} relevant_spans entries must include record_id.")
            else:
                self._validate_reference_membership(
                    normalized_scope,
                    scope_field="media_ids" if source == "media_db" else "note_ids",
                    candidate_id=record_id,
                    errors=errors,
                    index=index,
                )
            start = span.get("start")
            end = span.get("end")
            if start is None or end is None:
                errors.append(f"Dataset sample {index} relevant_spans entries must include start and end.")
            else:
                parsed_start = self._parse_int_field(
                    start,
                    errors=errors,
                    index=index,
                    field_name="relevant_spans integer start and end offsets",
                )
                parsed_end = self._parse_int_field(
                    end,
                    errors=errors,
                    index=index,
                    field_name="relevant_spans integer start and end offsets",
                )
                if (
                    parsed_start is not None
                    and parsed_end is not None
                    and parsed_end <= parsed_start
                ):
                    errors.append(
                        f"Dataset sample {index} relevant_spans entries must have end greater than start."
                    )
            grade_value = span.get("grade")
            grade = 1
            if grade_value is not None:
                parsed_grade = self._parse_int_field(
                    grade_value,
                    errors=errors,
                    index=index,
                    field_name="relevant_spans integer grade",
                )
                if parsed_grade is not None:
                    grade = parsed_grade
            if grade < _GRADE_MIN or grade > _GRADE_MAX:
                errors.append(
                    f"Dataset sample {index} target grades must be between {_GRADE_MIN} and {_GRADE_MAX}."
                )

    def _require_scope_source(
        self,
        normalized_scope: dict[str, Any] | None,
        source: str,
        errors: list[str],
        *,
        index: int,
    ) -> None:
        if normalized_scope is None:
            return
        if source not in normalized_scope.get("sources", []):
            errors.append(
                f"Dataset sample {index} targets require the run corpus_scope to include {source}."
            )

    def _validate_reference_membership(
        self,
        normalized_scope: dict[str, Any] | None,
        *,
        scope_field: str,
        candidate_id: Any,
        errors: list[str],
        index: int,
    ) -> None:
        if normalized_scope is None or scope_field not in normalized_scope:
            return
        allowed_ids = {
            self._normalize_reference_id(item)
            for item in normalized_scope.get(scope_field, [])
        }
        normalized_candidate = self._normalize_reference_id(candidate_id)
        if normalized_candidate not in allowed_ids:
            errors.append(
                f"Dataset sample {index} target {normalized_candidate!r} falls outside the declared corpus_scope.{scope_field}."
            )

    def _parse_int_field(
        self,
        value: Any,
        *,
        errors: list[str],
        index: int,
        field_name: str,
    ) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            errors.append(f"Dataset sample {index} must provide a valid {field_name}.")
            return None

    def _parse_bool_flag(
        self,
        value: Any,
        *,
        errors: list[str] | None = None,
        index: int | None = None,
        field_name: str = "boolean flag",
    ) -> bool:
        if value is None:
            return False
        try:
            return self._parse_bool_value(value, field_name=field_name)
        except ValueError as exc:
            if errors is not None:
                prefix = f"Dataset sample {index} " if index is not None else ""
                errors.append(f"{prefix}{exc}")
                return False
            raise

    def _parse_bool_value(self, value: Any, *, field_name: str) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, int) and value in {0, 1}:
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in _TRUE_VALUES:
                return True
            if normalized in _FALSE_VALUES:
                return False
        raise ValueError(f"{field_name} must be a boolean value.")

    def _is_fixed_indexing(
        self,
        run_config: dict[str, Any],
        normalized_scope: dict[str, Any] | None,
        *,
        errors: list[str] | None = None,
        index: int | None = None,
    ) -> bool:
        if self._parse_bool_flag(
            run_config.get("indexing_fixed"),
            errors=errors,
            index=index,
            field_name="run_config.indexing_fixed",
        ):
            return True
        if self._parse_bool_flag(
            run_config.get("chunking_fixed"),
            errors=errors,
            index=index,
            field_name="run_config.chunking_fixed",
        ):
            return True
        if isinstance(run_config.get("indexing_config"), dict):
            if str(run_config["indexing_config"].get("chunking_preset") or "").strip().lower() == "fixed_index":
                return True
        if normalized_scope and self._parse_bool_flag(
            normalized_scope.get("indexing_fixed"),
            errors=errors,
            index=index,
            field_name="run_config.corpus_scope.indexing_fixed",
        ):
            return True
        return False

    def _extract_auto_sweep_base_config(self, run_config: dict[str, Any]) -> dict[str, Any]:
        base_config: dict[str, Any] = {}
        if isinstance(run_config.get("retrieval_config"), dict):
            base_config["retrieval_config"] = dict(run_config["retrieval_config"])
        if isinstance(run_config.get("indexing_config"), dict):
            base_config["indexing_config"] = dict(run_config["indexing_config"])
        for key in ("search_mode", "top_k", "hybrid_alpha", "enable_reranking", "reranking_strategy", "rerank_top_k", "chunking_preset"):
            if run_config.get(key) is not None:
                base_config[key] = run_config[key]
        return base_config

    def _detect_dataset_mode(self, labeled_flags: list[bool]) -> str | None:
        if not labeled_flags:
            return None
        if all(labeled_flags):
            return "labeled"
        if not any(labeled_flags):
            return "unlabeled"
        return "mixed"

    def _reserve_review_sample(
        self,
        sample_ids: list[str],
        weak_supervision_budget: dict[str, Any],
    ) -> dict[str, Any]:
        if not sample_ids:
            return {"required": False, "sample_size": 0, "sample_ids": []}
        desired = math.ceil(
            len(sample_ids) * float(weak_supervision_budget["review_sample_fraction"])
        )
        sample_size = min(
            len(sample_ids),
            min(
                int(weak_supervision_budget["max_review_samples"]),
                max(int(weak_supervision_budget["min_review_samples"]), desired),
            ),
        )
        return {
            "required": True,
            "sample_size": sample_size,
            "sample_ids": sample_ids[:sample_size],
        }

    def _extract_sample_id(self, sample: dict[str, Any], index: int) -> str:
        for key in ("sample_id", "query_id", "id"):
            value = sample.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        return f"sample-{index + 1}"

    def _extract_query(self, sample: dict[str, Any]) -> str:
        for key in ("query", "input"):
            value = sample.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        return ""

    def _normalize_reference_id(self, value: Any) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("reference ids must not be empty.")
        return normalized
