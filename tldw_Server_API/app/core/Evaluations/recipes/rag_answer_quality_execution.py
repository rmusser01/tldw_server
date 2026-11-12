"""Fixed-context execution helpers for the RAG answer-quality recipe."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import statistics
import time
from collections.abc import Mapping
from typing import Any

from tldw_Server_API.app.core.DB_Management.db_path_utils import (
    get_user_chacha_db_path,
    get_user_media_db_path,
)
from tldw_Server_API.app.core.Evaluations.ms_g_eval import run_geval
from tldw_Server_API.app.core.Evaluations.scoring import normalize_geval_metric
from tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib import analyze
from tldw_Server_API.app.core.RAG.rag_service.unified_pipeline import unified_rag_pipeline

_DEFAULT_WEIGHTS = {
    "grounding": 0.4,
    "answer_relevance": 0.3,
    "format_style": 0.2,
    "abstention_behavior": 0.1,
}
_FAILURE_LABEL_ORDER = (
    "hallucinated",
    "missed_answer",
    "bad_abstention",
    "format_failure",
)
_SUPPORTED_SUPERVISION_MODES = {"rubric", "reference_answer", "pairwise", "mixed"}
_ABSTENTION_KEYWORDS = (
    "i cannot",
    "cannot answer",
    "can not answer",
    "not enough context",
    "insufficient context",
    "cannot determine",
    "unclear",
    "unsure",
    "cannot be determined",
    "abstain",
    "withhold",
    "unknown",
)
_HEDGE_KEYWORDS = (
    "likely",
    "appears",
    "seems",
    "probably",
    "possibly",
    "might",
    "may",
    "uncertain",
)
_LIVE_RETRIEVAL_FROZEN_FLAGS = {
    "adaptive_cache": False,
    "adaptive_hybrid_weights": False,
    "adaptive_rerun_bypass_cache": False,
    "adaptive_rerun_include_generation": False,
    "adaptive_rerun_on_low_confidence": False,
    "enable_dynamic_granularity": False,
    "enable_document_grading": False,
    "enable_evidence_accumulation": False,
    "enable_gap_analysis": False,
    "enable_hyde": False,
    "enable_intent_routing": False,
    "enable_post_verification": False,
    "enable_prf": False,
    "enable_query_decomposition": False,
    "enable_query_rewriting_loop": False,
    "enable_web_fallback": False,
    "expand_query": False,
    "spell_check": False,
}


def _run_config_for_execution(record: Any) -> dict[str, Any]:
    metadata = getattr(record, "metadata", {}) or {}
    return dict(metadata.get("run_config_internal") or metadata.get("run_config") or {})


def execute_rag_answer_quality_recipe_run(
    *,
    record: Any,
    db: Any,
    user_id: str | None,
    service: Any,
) -> dict[str, Any]:
    """Dispatch rag_answer_quality execution to fixed or live mode."""
    run_config = _run_config_for_execution(record)
    evaluation_mode = str(run_config.get("evaluation_mode") or "fixed_context").strip().lower()
    if evaluation_mode == "live_end_to_end":
        return _execute_live_end_to_end_rag_answer_quality_recipe_run(
            record=record,
            db=db,
            user_id=user_id,
            service=service,
        )
    return execute_fixed_context_rag_answer_quality_recipe_run(
        record=record,
        db=db,
        user_id=user_id,
        service=service,
    )


def execute_fixed_context_rag_answer_quality_recipe_run(
    *,
    record: Any,
    db: Any,
    user_id: str | None,
    service: Any,
) -> dict[str, Any]:
    """Execute a fixed-context answer-quality run and return persisted artifacts."""
    del service

    run_config = _run_config_for_execution(record)
    evaluation_mode = str(run_config.get("evaluation_mode") or "fixed_context").strip().lower()
    if evaluation_mode != "fixed_context":
        raise NotImplementedError("rag_answer_quality fixed-context execution only handles fixed_context runs.")

    return _execute_rag_answer_quality_recipe_run(
        record=record,
        db=db,
        user_id=user_id,
        run_config=run_config,
        evaluation_mode=evaluation_mode,
    )


def _execute_live_end_to_end_rag_answer_quality_recipe_run(
    *,
    record: Any,
    db: Any,
    user_id: str | None,
    service: Any,
) -> dict[str, Any]:
    """Execute a live end-to-end answer-quality run with frozen retrieval behavior."""
    run_config = _run_config_for_execution(record)
    evaluation_mode = str(run_config.get("evaluation_mode") or "live_end_to_end").strip().lower()
    if evaluation_mode != "live_end_to_end":
        raise NotImplementedError("rag_answer_quality live execution only handles live_end_to_end runs.")

    return _execute_rag_answer_quality_recipe_run(
        record=record,
        db=db,
        user_id=user_id,
        run_config=run_config,
        evaluation_mode=evaluation_mode,
        service=service,
    )


def _execute_rag_answer_quality_recipe_run(
    *,
    record: Any,
    db: Any,
    user_id: str | None,
    run_config: dict[str, Any],
    evaluation_mode: str,
    service: Any = None,
) -> dict[str, Any]:
    dataset = _resolve_inline_or_persisted_dataset(record, db=db, user_id=user_id)
    candidates = _normalize_candidates(run_config.get("candidates"))
    if not candidates:
        raise ValueError(
            "rag_answer_quality runs require run_config.candidates to be populated."
        )

    supervision_mode = str(run_config.get("supervision_mode") or "rubric").strip().lower()
    if supervision_mode not in _SUPPORTED_SUPERVISION_MODES:
        raise ValueError(
            "run_config.supervision_mode must be one of: rubric, reference_answer, pairwise, mixed."
        )

    grounding_threshold = _normalize_grounding_threshold(run_config.get("grounding_threshold"))
    normalized_weights = _normalize_weights(run_config.get("weights"))
    candidate_results: list[dict[str, Any]] = []
    live_retrieval_request = None
    retrieval_preset_hash = None
    if evaluation_mode == "live_end_to_end":
        live_retrieval_request = _resolve_live_retrieval_request(
            run_config=run_config,
            db=db,
            service=service,
        )
        retrieval_preset_hash = _hash_payload(
            {key: value for key, value in live_retrieval_request.items() if key != "query"}
        )

    for candidate in candidates:
        candidate_results.append(
            _execute_candidate(
                candidate=candidate,
                dataset=dataset,
                run_config=run_config,
                evaluation_mode=evaluation_mode,
                supervision_mode=supervision_mode,
                grounding_threshold=grounding_threshold,
                weights=normalized_weights,
                user_id=user_id,
                retrieval_baseline_ref=run_config.get("retrieval_baseline_ref"),
                live_retrieval_request=live_retrieval_request,
                run_retrieval_preset_hash=retrieval_preset_hash,
            )
        )

    recipe_report_inputs = {
        "dataset_mode": record.metadata.get("dataset_mode"),
        "review_sample": _resolve_review_sample(record),
        "evaluation_mode": evaluation_mode,
        "supervision_mode": supervision_mode,
        "context_snapshot_ref": run_config.get("context_snapshot_ref"),
        "run_anchor_ref": run_config.get("run_anchor_ref") or run_config.get("context_anchor_ref"),
        "retrieval_baseline_ref": run_config.get("retrieval_baseline_ref"),
        "retrieval_preset_hash": retrieval_preset_hash,
        "grounding_threshold": grounding_threshold,
        "weights": normalized_weights,
        "candidate_results": candidate_results,
    }
    return {
        "child_run_ids": [],
        "metadata": {
            "candidate_results": candidate_results,
            "recipe_report_inputs": recipe_report_inputs,
        },
    }


def _execute_candidate(
    *,
    candidate: dict[str, Any],
    dataset: list[dict[str, Any]],
    run_config: dict[str, Any],
    evaluation_mode: str,
    supervision_mode: str,
    grounding_threshold: float,
    weights: dict[str, float],
    user_id: str | None,
    retrieval_baseline_ref: str | None,
    live_retrieval_request: dict[str, Any] | None,
    run_retrieval_preset_hash: str | None,
) -> dict[str, Any]:
    provider, model = _resolve_provider_model(candidate)
    candidate_id = _resolve_candidate_id(candidate, provider=provider, model=model)
    sample_results: list[dict[str, Any]] = []
    latency_values: list[float] = []

    for index, sample in enumerate(dataset):
        query = _extract_query(sample)
        if not query:
            raise ValueError(f"Dataset sample {index} is missing a query.")
        retrieval_preset_hash = None
        if evaluation_mode == "live_end_to_end":
            contexts, retrieval_preset_hash = _resolve_live_contexts(
                query=query,
                user_id=user_id,
                retrieval_request=live_retrieval_request,
                retrieval_preset_hash=run_retrieval_preset_hash,
            )
        else:
            contexts = _resolve_fixed_contexts(sample=sample, run_config=run_config)
            if not contexts:
                raise ValueError(
                    f"Dataset sample {index} must include resolved inline contexts for fixed-context execution."
                )
        reference_answer = _extract_reference_answer(sample)
        expected_behavior = _extract_expected_behavior(sample)
        sample_id = _extract_sample_id(sample, index)

        started = time.perf_counter()
        answer = _generate_answer_for_candidate(
            provider=provider,
            model=model,
            query=query,
            contexts=contexts,
            run_config=run_config,
            candidate=candidate,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        latency_values.append(latency_ms)

        score_bundle = _score_sample(
            query=query,
            contexts=contexts,
            answer=answer,
            reference_answer=reference_answer,
            expected_behavior=expected_behavior,
            candidate=candidate,
            run_config=run_config,
            supervision_mode=supervision_mode,
            grounding_threshold=grounding_threshold,
            weights=weights,
        )
        sample_results.append(
            {
                "sample_id": sample_id,
                "query": query,
                "expected_behavior": expected_behavior,
                "context_snapshot_ref": sample.get("context_snapshot_ref")
                or run_config.get("context_snapshot_ref"),
                "retrieved_contexts": contexts,
                "retrieval_baseline_ref": retrieval_baseline_ref,
                "retrieval_preset_hash": retrieval_preset_hash,
                "reference_answer": reference_answer,
                "answer": answer,
                "latency_ms": latency_ms,
                "metrics": score_bundle["metrics"],
                "rubric": score_bundle["rubric"],
                "reference_comparison": score_bundle["reference_comparison"],
                "pairwise": score_bundle["pairwise"],
                "failure_labels": score_bundle["failure_labels"],
            }
        )

    aggregate_metrics = _aggregate_candidate_metrics(sample_results, weights, grounding_threshold)
    return {
        "candidate_id": candidate_id,
        "candidate_run_id": candidate_id,
        "provider": provider,
        "model": model,
        "generation_model": candidate.get("generation_model") or candidate.get("model"),
        "prompt_variant": candidate.get("prompt_variant"),
        "formatting_citation_mode": candidate.get("formatting_citation_mode"),
        "is_local": bool(candidate.get("is_local")),
        "cost_usd": candidate.get("cost_usd"),
        "retrieval_baseline_ref": retrieval_baseline_ref,
        "retrieval_preset_hash": (
            sample_results[0].get("retrieval_preset_hash") if sample_results else None
        ),
        "sample_results": sample_results,
        "metrics": aggregate_metrics,
        "sample_count": len(sample_results),
        "latency_ms": (
            statistics.mean(latency_values) if latency_values else None
        ),
    }


def _generate_answer_for_candidate(
    *,
    provider: str,
    model: str,
    query: str,
    contexts: list[dict[str, Any]],
    run_config: dict[str, Any],
    candidate: Mapping[str, Any],
) -> str:
    prompts = run_config.get("prompts") or {}
    prompt_variant = str(candidate.get("prompt_variant") or "default").strip() or "default"
    formatting_mode = str(candidate.get("formatting_citation_mode") or "plain").strip() or "plain"
    context_lines = []
    for index, context in enumerate(contexts, start=1):
        source = str(context.get("source") or f"context-{index}").strip()
        text = str(context.get("text") or context.get("content") or "").strip()
        if not text:
            continue
        context_lines.append(f"[{index}] {source}: {text}")

    system_message = prompts.get("system") or _build_system_message(
        prompt_variant=prompt_variant,
        formatting_mode=formatting_mode,
    )
    user_prompt = prompts.get("user") or _build_user_prompt(
        prompt_variant=prompt_variant,
        formatting_mode=formatting_mode,
    )
    prompt = "\n\n".join(
        [
            f"Question: {query}",
            "Context:",
            "\n".join(context_lines),
            f"Formatting mode: {formatting_mode}",
        ]
    ).strip()
    summary = analyze(
        provider,
        prompt,
        user_prompt,
        api_key=(run_config.get("candidate_api_keys") or {}).get(provider),
        system_message=system_message,
        temp=(run_config.get("execution_policy") or {}).get("temperature"),
        model_override=model,
    )
    if not isinstance(summary, str) or not summary.strip() or summary.startswith("Error:"):
        raise ValueError(
            f"RAG answer-quality candidate '{provider}:{model}' failed to generate a valid answer."
        )
    return summary.strip()


def _resolve_live_contexts(
    *,
    query: str,
    user_id: str | None,
    retrieval_request: dict[str, Any] | None,
    retrieval_preset_hash: str | None,
) -> tuple[list[dict[str, Any]], str]:
    if not retrieval_request:
        raise ValueError("rag_answer_quality live_end_to_end runs require a resolved retrieval request.")
    request_payload = dict(retrieval_request)
    request_payload["query"] = query
    response = asyncio.run(
        _run_live_retrieval_request(
            request_payload=request_payload,
            user_id=user_id,
        )
    )
    contexts = _extract_live_contexts(response)
    if not contexts:
        raise ValueError("rag_answer_quality live_end_to_end retrieval returned no contexts.")
    return contexts, str(retrieval_preset_hash or "")


async def _run_live_retrieval_request(
    *,
    request_payload: dict[str, Any],
    user_id: str | None,
) -> dict[str, Any]:
    response = await unified_rag_pipeline(
        media_db_path=get_user_media_db_path(user_id) if user_id else None,
        notes_db_path=get_user_chacha_db_path(user_id) if user_id else None,
        **request_payload,
    )
    if hasattr(response, "model_dump"):
        return dict(response.model_dump(mode="json"))
    return dict(response)


def _build_frozen_live_retrieval_request(
    *,
    query: str,
    run_config: Mapping[str, Any],
) -> dict[str, Any]:
    request_payload: dict[str, Any] = {
        "query": query,
        "sources": list(run_config.get("retrieval_sources") or ["media_db"]),
        "search_mode": str(run_config.get("search_mode") or "hybrid"),
        "fts_level": str(run_config.get("fts_level") or "media"),
        "hybrid_alpha": float(run_config.get("hybrid_alpha") or 0.7),
        "top_k": int(run_config.get("top_k") or 10),
        "min_score": float(run_config.get("min_score") or 0.0),
        "enable_reranking": bool(run_config.get("enable_reranking", True)),
        "reranking_strategy": str(run_config.get("reranking_strategy") or "flashrank"),
        "rerank_top_k": run_config.get("rerank_top_k"),
        "enable_cache": bool(run_config.get("enable_cache", True)),
        "enable_intent_routing": False,
        "expand_query": False,
        "spell_check": False,
        "adaptive_cache": False,
        "enable_prf": False,
        "enable_hyde": False,
        "enable_gap_analysis": False,
        "enable_post_verification": False,
        "adaptive_rerun_on_low_confidence": False,
        "adaptive_rerun_include_generation": False,
        "adaptive_rerun_bypass_cache": False,
        "adaptive_hybrid_weights": False,
        "enable_query_decomposition": False,
        "enable_web_fallback": False,
        "enable_document_grading": False,
        "enable_dynamic_granularity": False,
        "enable_evidence_accumulation": False,
        "enable_query_rewriting_loop": False,
        "enable_generation": False,
        "enable_streaming": False,
        "track_cost": False,
    }
    include_media_ids = run_config.get("include_media_ids")
    if include_media_ids is not None:
        request_payload["include_media_ids"] = [int(item) for item in list(include_media_ids)]
    include_note_ids = run_config.get("include_note_ids")
    if include_note_ids is not None:
        request_payload["include_note_ids"] = [
            str(item).strip() for item in list(include_note_ids) if str(item).strip()
        ]
    index_namespace = str(run_config.get("index_namespace") or "").strip()
    if index_namespace:
        request_payload["index_namespace"] = index_namespace
    for key, value in _LIVE_RETRIEVAL_FROZEN_FLAGS.items():
        request_payload[key] = value
    return request_payload


def _resolve_live_retrieval_request(
    *,
    run_config: Mapping[str, Any],
    db: Any,
    service: Any = None,
) -> dict[str, Any]:
    baseline_ref = str(run_config.get("retrieval_baseline_ref") or "").strip()
    if not baseline_ref:
        raise ValueError("run_config.retrieval_baseline_ref is required for live_end_to_end runs.")

    baseline_record = None
    if service is not None:
        try:
            baseline_record = service.get_run(baseline_ref)
        except Exception:
            baseline_record = None
    elif db is not None:
        baseline_record = db.get_recipe_run(baseline_ref)
    if baseline_record is None:
        raise ValueError(
            f"run_config.retrieval_baseline_ref '{baseline_ref}' could not be resolved to a recipe run."
        )
    if getattr(baseline_record, "recipe_id", None) != "rag_retrieval_tuning":
        raise ValueError(
            f"run_config.retrieval_baseline_ref '{baseline_ref}' must reference a rag_retrieval_tuning run."
        )

    baseline_run_config = dict((baseline_record.metadata or {}).get("run_config") or {})
    baseline_candidate_id = _resolve_live_baseline_candidate_id(baseline_record)
    baseline_candidate = _find_retrieval_baseline_candidate(
        baseline_run_config.get("candidates"),
        baseline_candidate_id,
    )
    if baseline_candidate is None:
        raise ValueError(
            f"run_config.retrieval_baseline_ref '{baseline_ref}' could not resolve a baseline candidate."
        )

    candidate_retrieval_config = dict(baseline_candidate.get("retrieval_config") or {})
    merged_run_config = {
        **candidate_retrieval_config,
        **{
            key: value
            for key, value in dict(run_config).items()
            if key
            in {
                "retrieval_sources",
                "search_mode",
                "fts_level",
                "hybrid_alpha",
                "top_k",
                "min_score",
                "enable_reranking",
                "reranking_strategy",
                "rerank_top_k",
                "enable_cache",
            }
        },
    }
    corpus_scope = dict(baseline_run_config.get("corpus_scope") or {})
    if "retrieval_sources" not in merged_run_config and corpus_scope.get("sources"):
        merged_run_config["retrieval_sources"] = list(corpus_scope.get("sources") or [])
    if "include_media_ids" not in merged_run_config and corpus_scope.get("media_ids"):
        merged_run_config["include_media_ids"] = list(corpus_scope.get("media_ids") or [])
    if "include_note_ids" not in merged_run_config and corpus_scope.get("note_ids"):
        merged_run_config["include_note_ids"] = list(corpus_scope.get("note_ids") or [])
    if "index_namespace" not in merged_run_config:
        index_namespace = (
            baseline_candidate.get("index_namespace")
            or corpus_scope.get("index_namespace")
        )
        if index_namespace is not None and str(index_namespace).strip():
            merged_run_config["index_namespace"] = str(index_namespace).strip()

    return _build_frozen_live_retrieval_request(query="", run_config=merged_run_config)


def _extract_live_contexts(response: Mapping[str, Any]) -> list[dict[str, Any]]:
    documents = response.get("documents") or []
    contexts = _documents_to_contexts(documents)
    if contexts:
        return contexts
    metadata = response.get("metadata") or {}
    for field_name in ("reranked_documents", "pre_rerank_documents", "first_pass_documents"):
        contexts = _documents_to_contexts(metadata.get(field_name) or [])
        if contexts:
            return contexts
    return []


def _documents_to_contexts(documents: Any) -> list[dict[str, Any]]:
    contexts: list[dict[str, Any]] = []
    if not isinstance(documents, list):
        return contexts
    for index, item in enumerate(documents, start=1):
        if not isinstance(item, Mapping):
            continue
        text = str(item.get("text") or item.get("content") or item.get("body") or "").strip()
        if not text:
            continue
        context = {
            "source": str(item.get("source") or f"context-{index}").strip() or f"context-{index}",
            "text": text,
        }
        for field_name in ("media_id", "note_id", "chunk_id", "record_id", "start", "end"):
            value = item.get(field_name)
            if value is not None:
                context[field_name] = value
        metadata = item.get("metadata")
        if isinstance(metadata, Mapping):
            for field_name in ("media_id", "note_id", "chunk_id", "record_id", "start", "end"):
                if field_name in context:
                    continue
                value = metadata.get(field_name)
                if value is not None:
                    context[field_name] = value
        contexts.append(context)
    return contexts


def _hash_payload(payload: Mapping[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _score_sample(
    *,
    query: str,
    contexts: list[dict[str, Any]],
    answer: str,
    reference_answer: str | None,
    expected_behavior: str | None,
    candidate: Mapping[str, Any],
    run_config: dict[str, Any],
    supervision_mode: str,
    grounding_threshold: float,
    weights: dict[str, float],
) -> dict[str, Any]:
    context_text = " ".join(
        str(context.get("text") or context.get("content") or "").strip()
        for context in contexts
    ).strip()
    rubric = _build_rubric_artifact(
        query=query,
        contexts=contexts,
        answer=answer,
        expected_behavior=expected_behavior,
        candidate=candidate,
        grounding_threshold=grounding_threshold,
    )
    reference_comparison: dict[str, Any] | None = None
    if supervision_mode in {"reference_answer", "mixed"} and reference_answer:
        reference_comparison = _build_reference_comparison(
            query=query,
            contexts=context_text,
            answer=answer,
            reference_answer=reference_answer,
            run_config=run_config,
        )
        rubric = _blend_with_reference_artifact(rubric, reference_comparison)

    pairwise = None
    if supervision_mode in {"pairwise", "mixed"}:
        pairwise = _build_pairwise_artifact(
            answer=answer,
            reference_answer=reference_answer,
            expected_behavior=expected_behavior,
        )

    metrics = {
        "grounding": rubric["grounding"],
        "answer_relevance": rubric["answer_relevance"],
        "format_style": rubric["format_style"],
        "abstention_behavior": rubric["abstention_behavior"],
    }
    metrics["quality_score"] = (
        (float(weights["grounding"]) * metrics["grounding"])
        + (float(weights["answer_relevance"]) * metrics["answer_relevance"])
        + (float(weights["format_style"]) * metrics["format_style"])
        + (float(weights["abstention_behavior"]) * metrics["abstention_behavior"])
    )
    metrics["grounding_gate_passed"] = metrics["grounding"] >= grounding_threshold
    failure_labels = _derive_failure_labels(
        metrics=metrics,
        expected_behavior=expected_behavior,
        answer=answer,
        grounding_threshold=grounding_threshold,
    )

    return {
        "metrics": metrics,
        "rubric": rubric,
        "reference_comparison": reference_comparison,
        "pairwise": pairwise,
        "failure_labels": failure_labels,
    }


def _build_rubric_artifact(
    *,
    query: str,
    contexts: list[dict[str, Any]],
    answer: str,
    expected_behavior: str | None,
    candidate: Mapping[str, Any],
    grounding_threshold: float,
) -> dict[str, Any]:
    query_tokens = _tokenize(query)
    answer_tokens = _tokenize(answer)
    context_tokens = _tokenize(
        " ".join(
            str(context.get("text") or context.get("content") or "").strip()
            for context in contexts
        )
    )
    answer_overlap = _overlap_ratio(answer_tokens, query_tokens)
    grounding_overlap = _overlap_ratio(answer_tokens, context_tokens)
    citations_expected = str(candidate.get("formatting_citation_mode") or "").strip().lower()
    format_style = _format_style_score(answer, citations_expected)
    abstention_behavior = _abstention_score(answer, expected_behavior)

    return {
        "grounding": _clamp01(grounding_overlap),
        "answer_relevance": _clamp01((answer_overlap * 0.7) + (grounding_overlap * 0.3)),
        "format_style": _clamp01(format_style),
        "abstention_behavior": _clamp01(abstention_behavior),
        "expected_behavior": expected_behavior,
        "grounding_threshold": grounding_threshold,
        "cited": _contains_citation(answer),
    }


def _build_reference_comparison(
    *,
    query: str,
    contexts: str,
    answer: str,
    reference_answer: str,
    run_config: dict[str, Any],
) -> dict[str, Any] | None:
    judge_config = run_config.get("judge_config") or {}
    try:
        result = run_geval(
            transcript="\n\n".join(
                [
                    f"Question: {query}",
                    f"Context: {contexts}",
                ]
            ).strip(),
            summary=answer,
            reference_summary=reference_answer,
            api_key=judge_config.get("api_key"),
            api_name=judge_config.get("provider") or "openai",
            model=judge_config.get("model"),
            save=False,
        )
    except Exception as exc:
        return {"status": "error", "error": str(exc)}
    return dict(result)


def _blend_with_reference_artifact(
    rubric: dict[str, Any],
    reference_comparison: dict[str, Any] | None,
) -> dict[str, Any]:
    if not reference_comparison or reference_comparison.get("status") == "error":
        return rubric
    metrics = reference_comparison.get("metrics") or {}
    scores = [
        _coerce_geval_score(key, metrics.get(key))
        for key in ("consistency", "relevance", "coherence", "fluency")
    ]
    reference_score = statistics.mean(score for score in scores if score is not None) if any(
        score is not None for score in scores
    ) else None
    if reference_score is None:
        return rubric
    blended = dict(rubric)
    blended["grounding"] = _clamp01((rubric["grounding"] * 0.6) + (reference_score * 0.4))
    blended["answer_relevance"] = _clamp01(
        (rubric["answer_relevance"] * 0.5) + (reference_score * 0.5)
    )
    return blended


def _aggregate_candidate_metrics(
    sample_results: list[dict[str, Any]],
    weights: dict[str, float],
    grounding_threshold: float,
) -> dict[str, Any]:
    if not sample_results:
        return {
            "grounding": 0.0,
            "answer_relevance": 0.0,
            "format_style": 0.0,
            "abstention_behavior": 0.0,
            "quality_score": 0.0,
            "grounding_gate_passed": False,
        }
    grounding = statistics.mean(float(sample["metrics"]["grounding"]) for sample in sample_results)
    answer_relevance = statistics.mean(
        float(sample["metrics"]["answer_relevance"]) for sample in sample_results
    )
    format_style = statistics.mean(float(sample["metrics"]["format_style"]) for sample in sample_results)
    abstention_behavior = statistics.mean(
        float(sample["metrics"]["abstention_behavior"]) for sample in sample_results
    )
    quality_score = (
        (weights["grounding"] * grounding)
        + (weights["answer_relevance"] * answer_relevance)
        + (weights["format_style"] * format_style)
        + (weights["abstention_behavior"] * abstention_behavior)
    )
    return {
        "grounding": grounding,
        "answer_relevance": answer_relevance,
        "format_style": format_style,
        "abstention_behavior": abstention_behavior,
        "quality_score": quality_score,
        "grounding_gate_passed": grounding >= grounding_threshold,
    }


def _resolve_inline_or_persisted_dataset(
    record: Any,
    *,
    db: Any,
    user_id: str | None,
) -> list[dict[str, Any]]:
    inline_dataset = record.metadata.get("inline_dataset")
    if isinstance(inline_dataset, list):
        return [dict(sample) for sample in inline_dataset]

    dataset = record.metadata.get("dataset")
    if isinstance(dataset, list):
        return [dict(sample) for sample in dataset]

    dataset_id = record.metadata.get("dataset_id")
    if isinstance(dataset_id, str) and dataset_id.strip():
        dataset_row = db.get_dataset(dataset_id.strip(), created_by=user_id or None)
        if not dataset_row:
            raise ValueError(
                f"rag_answer_quality fixed-context dataset '{dataset_id}' was not found."
            )
        samples = dataset_row.get("samples") or []
        return [dict(sample) for sample in samples]
    raise ValueError("rag_answer_quality fixed-context runs require inline or persisted datasets.")


def _normalize_candidates(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    normalized: list[dict[str, Any]] = []
    for index, candidate in enumerate(value, start=1):
        if not isinstance(candidate, Mapping):
            continue
        provider, model = _resolve_provider_model(candidate)
        candidate_id = _resolve_candidate_id(candidate, provider=provider, model=model, index=index)
        normalized.append(
            {
                "candidate_id": candidate_id,
                "provider": provider,
                "model": model,
                "generation_model": candidate.get("generation_model") or candidate.get("model"),
                "prompt_variant": _normalize_text(candidate.get("prompt_variant"), default="default"),
                "formatting_citation_mode": _normalize_text(
                    candidate.get("formatting_citation_mode"),
                    default="plain",
                ),
                "is_local": bool(candidate.get("is_local")),
                "cost_usd": candidate.get("cost_usd"),
                "generation_config": dict(candidate.get("generation_config") or {}),
            }
        )
    return normalized


def _resolve_provider_model(candidate: Mapping[str, Any]) -> tuple[str, str]:
    provider = _normalize_text(candidate.get("provider"))
    model = _normalize_text(candidate.get("model"))
    generation_model = _normalize_text(candidate.get("generation_model"))
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
    candidate: Mapping[str, Any],
    *,
    provider: str,
    model: str,
    index: int | None = None,
) -> str:
    candidate_id = _normalize_text(candidate.get("candidate_id"))
    if candidate_id:
        return candidate_id
    prompt_variant = _normalize_text(candidate.get("prompt_variant"), default="default")
    formatting_mode = _normalize_text(
        candidate.get("formatting_citation_mode"),
        default="plain",
    )
    base = f"{provider}:{model}"
    if index is not None:
        return f"{base}::{prompt_variant}::{formatting_mode}::{index}"
    return f"{base}::{prompt_variant}::{formatting_mode}"


def _extract_query(sample: Mapping[str, Any]) -> str:
    for field_name in ("query", "input", "question", "prompt"):
        value = sample.get(field_name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _extract_contexts(sample: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_contexts = None
    for field_name in ("retrieved_contexts", "inline_contexts", "contexts", "source_contexts"):
        value = sample.get(field_name)
        if value:
            raw_contexts = value
            break
    if raw_contexts is None and isinstance(sample.get("context"), Mapping):
        raw_contexts = [sample.get("context")]
    if raw_contexts is None:
        return []
    if isinstance(raw_contexts, Mapping):
        raw_contexts = [raw_contexts]
    elif isinstance(raw_contexts, str):
        raw_contexts = [raw_contexts]
    elif not isinstance(raw_contexts, list):
        return []
    contexts: list[dict[str, Any]] = []
    for item in raw_contexts:
        if isinstance(item, Mapping):
            contexts.append(
                {
                    "source": str(item.get("source") or item.get("origin") or "").strip(),
                    "text": str(item.get("text") or item.get("content") or item.get("body") or "").strip(),
                }
            )
        elif isinstance(item, str) and item.strip():
            contexts.append({"source": "inline", "text": item.strip()})
    return [context for context in contexts if context["text"]]


def _resolve_fixed_contexts(
    *,
    sample: Mapping[str, Any],
    run_config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    sample_contexts = _extract_contexts(sample)
    if sample_contexts:
        return sample_contexts
    run_level_contexts = run_config.get("inline_contexts")
    if run_level_contexts:
        return _extract_contexts({"inline_contexts": run_level_contexts})
    return []


def _extract_reference_answer(sample: Mapping[str, Any]) -> str | None:
    for field_name in ("reference_answer", "expected", "expected_answer"):
        value = sample.get(field_name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _extract_expected_behavior(sample: Mapping[str, Any]) -> str | None:
    value = sample.get("expected_behavior")
    if value is None:
        return None
    normalized = str(value).strip().lower()
    return normalized or None


def _extract_sample_id(sample: Mapping[str, Any], index: int) -> str:
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


def _resolve_review_sample(record: Any) -> dict[str, Any]:
    review_sample = record.metadata.get("review_sample")
    if isinstance(review_sample, dict):
        return dict(review_sample)
    recipe_validation = record.metadata.get("recipe_validation")
    if isinstance(recipe_validation, dict):
        nested_review_sample = recipe_validation.get("review_sample")
        if isinstance(nested_review_sample, dict):
            return dict(nested_review_sample)
    return {"required": False, "sample_size": 0, "sample_ids": []}


def _resolve_live_baseline_candidate_id(record: Any) -> str:
    recommendation_slots = getattr(record, "recommendation_slots", None) or {}
    best_overall = recommendation_slots.get("best_overall")
    if best_overall is not None:
        candidate_run_id = getattr(best_overall, "candidate_run_id", None)
        if candidate_run_id:
            return str(candidate_run_id).strip()
        if isinstance(best_overall, Mapping):
            candidate_run_id = best_overall.get("candidate_run_id")
            if candidate_run_id:
                return str(candidate_run_id).strip()
    recipe_report = (getattr(record, "metadata", None) or {}).get("recipe_report") or {}
    best_overall_report = recipe_report.get("best_overall") or {}
    for field_name in ("candidate_run_id", "candidate_id"):
        value = best_overall_report.get(field_name)
        if value is not None and str(value).strip():
            return str(value).strip()
    raise ValueError("run_config.retrieval_baseline_ref did not resolve a best_overall retrieval candidate.")


def _find_retrieval_baseline_candidate(
    candidates: Any,
    candidate_id: str,
) -> dict[str, Any] | None:
    if not isinstance(candidates, list):
        return None
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        if str(candidate.get("candidate_id") or "").strip() == candidate_id:
            return dict(candidate)
    return None


def _build_system_message(*, prompt_variant: str, formatting_mode: str) -> str:
    return (
        "You are a careful RAG answer generator. "
        f"Prompt variant: {prompt_variant}. "
        f"Formatting mode: {formatting_mode}."
    )


def _build_user_prompt(*, prompt_variant: str, formatting_mode: str) -> str:
    return (
        "Answer the question using only the provided context. "
        "If the context is insufficient, hedge or abstain clearly. "
        f"Prompt variant: {prompt_variant}. "
        f"Formatting mode: {formatting_mode}."
    )


def _normalize_text(value: Any, *, default: str | None = None) -> str | None:
    if value is None:
        return default
    normalized = str(value).strip()
    return normalized or default


def _normalize_grounding_threshold(value: Any) -> float:
    if value is None:
        return 0.65
    threshold = float(value)
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError("run_config.grounding_threshold must be between 0 and 1.")
    return threshold


def _normalize_weights(value: Any) -> dict[str, float]:
    if value is None:
        normalized = dict(_DEFAULT_WEIGHTS)
    elif not isinstance(value, Mapping):
        raise ValueError("run_config.weights must be an object when provided.")
    else:
        normalized = {}
        for key, default_value in _DEFAULT_WEIGHTS.items():
            normalized[key] = float(value.get(key, default_value))
    total = sum(normalized.values())
    if total <= 0:
        raise ValueError("run_config.weights must sum to a positive value.")
    return {key: score / total for key, score in normalized.items()}


def _build_pairwise_artifact(
    *,
    answer: str,
    reference_answer: str | None,
    expected_behavior: str | None,
) -> dict[str, Any] | None:
    if not reference_answer:
        return None
    answer_score = _overlap_ratio(_tokenize(answer), _tokenize(reference_answer))
    return {
        "status": "available",
        "winner": "answer" if answer_score >= 0.5 else "reference",
        "score": answer_score,
        "expected_behavior": expected_behavior,
    }


def _coerce_geval_score(metric: str, value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return normalize_geval_metric(metric, numeric)


def _contains_citation(answer: str) -> bool:
    return bool(re.search(r"\[[0-9]+\]|\([^)]+source[^)]*\)", answer, re.IGNORECASE))


def _contains_abstention(answer: str) -> bool:
    lowered = answer.lower()
    return any(keyword in lowered for keyword in _ABSTENTION_KEYWORDS)


def _format_style_score(answer: str, citations_expected: str) -> float:
    has_citation = _contains_citation(answer)
    if citations_expected in {"citations", "citation", "source", "sources"}:
        return 1.0 if has_citation else 0.45
    if citations_expected in {"plain", "none"}:
        return 0.85 if not has_citation else 0.55
    return 0.75 if answer.strip() else 0.0


def _abstention_score(answer: str, expected_behavior: str | None) -> float:
    lowered = answer.lower()
    abstains = _contains_abstention(answer)
    hedges = any(keyword in lowered for keyword in _HEDGE_KEYWORDS)
    if expected_behavior == "abstain":
        return 1.0 if abstains else 0.1
    if expected_behavior == "hedge":
        return 0.9 if hedges or abstains else 0.35
    if expected_behavior == "answer":
        return 0.9 if not abstains else 0.2
    return 0.5 if not abstains else 0.3


def _tokenize(value: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", value.lower()) if token}


def _overlap_ratio(answer_tokens: set[str], other_tokens: set[str]) -> float:
    if not answer_tokens:
        return 0.0
    return len(answer_tokens & other_tokens) / max(1, len(answer_tokens))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _derive_failure_labels(
    *,
    metrics: Mapping[str, Any],
    expected_behavior: str | None,
    answer: str,
    grounding_threshold: float,
) -> list[str]:
    labels: list[str] = []
    grounding = float(metrics.get("grounding", 0.0) or 0.0)
    answer_relevance = float(metrics.get("answer_relevance", 0.0) or 0.0)
    format_style = float(metrics.get("format_style", 0.0) or 0.0)
    abstention_behavior = float(metrics.get("abstention_behavior", 0.0) or 0.0)
    abstains = _contains_abstention(answer)

    if grounding < grounding_threshold and not abstains:
        labels.append("hallucinated")
    if expected_behavior == "answer" and answer_relevance < 0.6:
        labels.append("missed_answer")
    if expected_behavior == "answer" and abstains:
        labels.append("bad_abstention")
    elif expected_behavior == "abstain" and abstention_behavior < 0.5:
        labels.append("bad_abstention")
    elif expected_behavior == "hedge" and abstention_behavior < 0.5:
        labels.append("bad_abstention")
    if format_style < 0.6:
        labels.append("format_failure")

    ordered = [label for label in _FAILURE_LABEL_ORDER if label in labels]
    return ordered
