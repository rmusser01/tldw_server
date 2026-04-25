"""Worker entrypoints for queued evaluation recipe runs."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
from typing import Any

from loguru import logger

from tldw_Server_API.app.api.v1.schemas.evaluation_schemas_unified import RunStatus
from tldw_Server_API.app.api.v1.schemas.embeddings_abtest_schemas import (
    ABTestArm,
    ABTestChunking,
    ABTestQuery,
    ABTestRetrieval,
    EmbeddingsABTestConfig,
)
from tldw_Server_API.app.core.DB_Management.DB_Manager import get_content_backend_instance
from tldw_Server_API.app.core.DB_Management.Evaluations_DB import EvaluationsDatabase
from tldw_Server_API.app.core.DB_Management.db_path_utils import (
    DatabasePaths,
    get_user_chacha_db_path,
    get_user_media_db_path,
)
from tldw_Server_API.app.core.DB_Management.media_db.api import managed_media_database
from tldw_Server_API.app.core.Evaluations.embeddings_abtest_service import run_abtest_full
from tldw_Server_API.app.core.Evaluations.ms_g_eval import run_geval
from tldw_Server_API.app.core.Evaluations.recipes.rag_retrieval_tuning_execution import (
    serialize_index_plan,
)
from tldw_Server_API.app.core.Evaluations.recipes.rag_answer_quality_execution import (
    execute_rag_answer_quality_recipe_run,
)
from tldw_Server_API.app.core.Evaluations.recipe_runs_jobs import (
    RECIPE_RUN_JOB_DOMAIN,
    parse_recipe_run_job_payload,
    recipe_run_queue,
)
from tldw_Server_API.app.core.Evaluations.recipe_runs_service import (
    RecipeRunsService,
    get_recipe_runs_service_for_user,
)
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.worker_sdk import WorkerConfig, WorkerSDK
from tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib import analyze
from tldw_Server_API.app.core.RAG.rag_service.unified_pipeline import unified_rag_pipeline
from tldw_Server_API.app.core.testing import env_flag_enabled

_LOCAL_SUMMARIZATION_PROVIDERS = {
    "ollama",
    "llama.cpp",
    "llamacpp",
    "kobold.cpp",
    "koboldcpp",
    "oobabooga",
    "tabbyapi",
    "vllm",
    "aphrodite",
    "local",
}


class RecipeRunJobError(RuntimeError):
    """Worker error wrapper with explicit retry semantics."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool,
        backoff_seconds: int = 10,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.backoff_seconds = int(backoff_seconds)


def _get_db(*, user_id: str | None) -> EvaluationsDatabase:
    db_path = os.getenv("EVALUATIONS_TEST_DB_PATH")
    if not db_path:
        db_path = str(DatabasePaths.get_evaluations_db_path(user_id))
    return EvaluationsDatabase(db_path)


def _get_service(*, user_id: str | None, db: EvaluationsDatabase | None = None) -> RecipeRunsService:
    if db is not None:
        return RecipeRunsService(db=db, user_id=user_id)
    return get_recipe_runs_service_for_user(user_id)


def _parse_json_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, list):
            return list(parsed)
    return []


def _run_config_for_execution(record: Any) -> dict[str, Any]:
    metadata = getattr(record, "metadata", {}) or {}
    return dict(metadata.get("run_config_internal") or metadata.get("run_config") or {})


def _resolve_embeddings_dataset(record: Any, db: EvaluationsDatabase, user_id: str | None) -> list[dict[str, Any]]:
    inline_dataset = record.metadata.get("inline_dataset")
    if isinstance(inline_dataset, list):
        return [dict(sample) for sample in inline_dataset]

    dataset_id = record.metadata.get("dataset_id")
    if not dataset_id:
        raise ValueError("Embeddings recipe run requires inline_dataset or dataset_id metadata.")
    dataset_row = db.get_dataset(str(dataset_id), created_by=user_id or None)
    if not dataset_row:
        raise ValueError(f"Dataset '{dataset_id}' was not found for embeddings recipe execution.")
    samples = dataset_row.get("samples") or []
    return [dict(sample) for sample in samples]


def _resolve_inline_or_persisted_dataset(
    record: Any,
    db: EvaluationsDatabase,
    user_id: str | None,
) -> list[dict[str, Any]]:
    inline_dataset = record.metadata.get("inline_dataset")
    if isinstance(inline_dataset, list):
        return [dict(sample) for sample in inline_dataset]

    dataset_id = record.metadata.get("dataset_id")
    if not dataset_id:
        raise ValueError("Recipe run requires inline_dataset or dataset_id metadata.")
    dataset_row = db.get_dataset(str(dataset_id), created_by=user_id or None)
    if not dataset_row:
        raise ValueError(f"Dataset '{dataset_id}' was not found for recipe execution.")
    samples = dataset_row.get("samples") or []
    return [dict(sample) for sample in samples]


def _resolve_candidate_provider_model(candidate: dict[str, Any]) -> tuple[str, str]:
    provider = str(candidate.get("provider") or "").strip()
    model = str(candidate.get("model") or "").strip()
    if provider and model:
        return provider, model
    if ":" in model:
        inferred_provider, inferred_model = model.split(":", 1)
        return inferred_provider.strip(), inferred_model.strip()
    raise ValueError("Embeddings recipe candidates must include provider and model.")


def _split_candidate_model_id(candidate_model_id: str) -> tuple[str, str]:
    normalized = str(candidate_model_id or "").strip()
    if ":" not in normalized:
        raise ValueError("Summarization recipe candidate_model_ids must use 'provider:model' format.")
    provider, model = normalized.split(":", 1)
    provider = provider.strip()
    model = model.strip()
    if not provider or not model:
        raise ValueError("Summarization recipe candidates must include both provider and model.")
    return provider, model


def _is_local_summarization_provider(provider: str) -> bool:
    return provider.strip().lower() in _LOCAL_SUMMARIZATION_PROVIDERS


def _coerce_media_ids(run_config: dict[str, Any], dataset: list[dict[str, Any]]) -> list[int]:
    explicit_media_ids = run_config.get("media_ids")
    if isinstance(explicit_media_ids, list) and explicit_media_ids:
        return [int(media_id) for media_id in explicit_media_ids]

    derived_media_ids: set[int] = set()
    for sample in dataset:
        for expected_id in sample.get("expected_ids") or []:
            try:
                derived_media_ids.add(int(expected_id))
            except (TypeError, ValueError):
                logger.debug("Skipping non-integer embeddings expected_id while deriving media_ids: {}", expected_id)
                continue
    if derived_media_ids:
        return sorted(derived_media_ids)
    raise ValueError(
        "Embeddings recipe execution requires run_config.media_ids or labeled expected_ids to derive the corpus."
    )


def _classify_recipe_run_job_error(exc: Exception) -> RecipeRunJobError:
    if isinstance(exc, RecipeRunJobError):
        return exc
    if hasattr(exc, "retryable"):
        return RecipeRunJobError(
            str(exc),
            retryable=bool(getattr(exc, "retryable", False)),
            backoff_seconds=int(getattr(exc, "backoff_seconds", 10)),
        )
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return RecipeRunJobError(str(exc), retryable=True, backoff_seconds=15)
    return RecipeRunJobError(str(exc), retryable=False, backoff_seconds=0)


def _extract_summarization_source_text(sample: dict[str, Any]) -> str:
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


def _extract_summarization_reference(sample: dict[str, Any]) -> str | None:
    for key in ("expected", "reference_summary", "summary"):
        value = sample.get(key)
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                return stripped
    return None


def _extract_summarization_sample_id(sample: dict[str, Any], index: int) -> str:
    metadata = sample.get("metadata")
    if isinstance(metadata, dict):
        sample_id = metadata.get("sample_id")
        if isinstance(sample_id, str) and sample_id.strip():
            return sample_id.strip()
    sample_id = sample.get("sample_id")
    if isinstance(sample_id, str) and sample_id.strip():
        return sample_id.strip()
    return f"sample-{index}"


def _extract_rag_sample_id(sample: dict[str, Any], index: int) -> str:
    for key in ("sample_id", "query_id", "id"):
        value = sample.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return f"sample-{index + 1}"


def _extract_rag_query(sample: dict[str, Any]) -> str:
    for key in ("query", "input"):
        value = sample.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    raise ValueError("RAG retrieval tuning samples must include a non-empty query.")


def _normalize_rag_targets(sample: dict[str, Any]) -> dict[str, Any]:
    targets = sample.get("targets") or {}
    normalized = {
        "media": {},
        "notes": {},
        "chunks": {},
        "spans": [],
    }

    for item in targets.get("relevant_media_ids") or []:
        try:
            normalized["media"][str(item.get("id")).strip()] = min(3, max(0, int(item.get("grade", 0))))
        except (AttributeError, TypeError, ValueError):
            continue
    for item in targets.get("relevant_note_ids") or []:
        try:
            normalized["notes"][str(item.get("id")).strip()] = min(3, max(0, int(item.get("grade", 0))))
        except (AttributeError, TypeError, ValueError):
            continue
    for item in targets.get("relevant_chunk_ids") or []:
        try:
            normalized["chunks"][str(item.get("id")).strip()] = min(3, max(0, int(item.get("grade", 0))))
        except (AttributeError, TypeError, ValueError):
            continue
    for item in targets.get("relevant_spans") or []:
        if not isinstance(item, dict):
            continue
        try:
            normalized["spans"].append(
                {
                    "source": str(item.get("source") or "").strip(),
                    "record_id": str(item.get("record_id") or "").strip(),
                    "start": int(item.get("start")),
                    "end": int(item.get("end")),
                    "grade": min(3, max(0, int(item.get("grade", 0)))),
                }
            )
        except (TypeError, ValueError):
            continue
    return normalized


async def _run_unified_rag_request(
    *,
    request: Any,
    user_id: str | None,
) -> dict[str, Any]:
    payload = request.model_dump(exclude_none=True) if hasattr(request, "model_dump") else dict(request)
    response = await unified_rag_pipeline(
        **payload,
        media_db_path=get_user_media_db_path(user_id) if user_id else None,
        notes_db_path=get_user_chacha_db_path(user_id) if user_id else None,
        enable_generation=False,
        enable_streaming=False,
        enable_intent_routing=False,
        expand_query=False,
        enable_prf=False,
        enable_hyde=False,
        enable_gap_analysis=False,
        enable_post_verification=False,
        adaptive_hybrid_weights=False,
        track_cost=False,
    )
    if hasattr(response, "model_dump"):
        return dict(response.model_dump(mode="json"))
    return dict(response)


def _extract_ranked_documents(response_payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    documents = [
        dict(doc)
        for doc in (response_payload.get("documents") or [])
        if isinstance(doc, dict)
    ]
    metadata = response_payload.get("metadata") or {}
    first_pass_docs = metadata.get("pre_rerank_documents") or metadata.get("first_pass_documents")
    reranked_docs = metadata.get("reranked_documents")
    if not isinstance(first_pass_docs, list):
        first_pass_docs = documents
    if not isinstance(reranked_docs, list):
        reranked_docs = documents
    return (
        [dict(doc) for doc in first_pass_docs if isinstance(doc, dict)],
        [dict(doc) for doc in reranked_docs if isinstance(doc, dict)],
    )


def _extract_doc_identity(doc: dict[str, Any]) -> dict[str, Any]:
    metadata = doc.get("metadata") or {}
    source = str(
        doc.get("source")
        or metadata.get("source")
        or doc.get("sql_target_id")
        or metadata.get("sql_target_id")
        or ""
    ).strip()
    media_id = doc.get("media_id")
    if media_id is None:
        media_id = metadata.get("media_id")
    note_id = doc.get("note_id")
    if note_id is None:
        note_id = metadata.get("note_id")
    record_id = doc.get("record_id")
    if record_id is None:
        record_id = metadata.get("record_id")
    chunk_id = doc.get("chunk_id")
    if chunk_id is None:
        chunk_id = metadata.get("chunk_id")
    if chunk_id is None:
        chunk_id = doc.get("id")
    if record_id is None:
        record_id = media_id if media_id is not None else note_id
    start = doc.get("start")
    if start is None:
        start = metadata.get("start")
    if start is None:
        start = metadata.get("chunk_start")
    end = doc.get("end")
    if end is None:
        end = metadata.get("end")
    if end is None:
        end = metadata.get("chunk_end")
    return {
        "source": source,
        "media_id": str(media_id).strip() if media_id is not None else None,
        "note_id": str(note_id).strip() if note_id is not None else None,
        "chunk_id": str(chunk_id).strip() if chunk_id is not None else None,
        "record_id": str(record_id).strip() if record_id is not None else None,
        "start": int(start) if start is not None else None,
        "end": int(end) if end is not None else None,
    }


def _grade_rag_document(doc: dict[str, Any], normalized_targets: dict[str, Any]) -> int:
    identity = _extract_doc_identity(doc)
    grade = 0
    media_id = identity.get("media_id")
    note_id = identity.get("note_id")
    if media_id is not None:
        grade = max(grade, int(normalized_targets["media"].get(media_id, 0)))
    if note_id is not None:
        grade = max(grade, int(normalized_targets["notes"].get(note_id, 0)))
    chunk_id = identity.get("chunk_id")
    if chunk_id is not None:
        grade = max(grade, int(normalized_targets["chunks"].get(chunk_id, 0)))
    record_id = identity.get("record_id")
    start = identity.get("start")
    end = identity.get("end")
    if record_id and start is not None and end is not None:
        for span in normalized_targets["spans"]:
            if span["record_id"] != record_id:
                continue
            if span["source"] and identity.get("source") and span["source"] != identity["source"]:
                continue
            overlaps = not (end <= span["start"] or start >= span["end"])
            if overlaps:
                grade = max(grade, int(span["grade"]))
    return min(3, max(0, grade))


def _aggregate_rag_candidate_metrics(query_results: list[dict[str, Any]]) -> dict[str, Any]:
    if not query_results:
        zero_score = 0.0
        return {
            "pre_rerank_recall_at_k": zero_score,
            "post_rerank_ndcg_at_k": zero_score,
            "first_pass_recall_score": zero_score,
            "post_rerank_quality_score": zero_score,
        }
    pre_values = []
    post_values = []
    for query_result in query_results:
        metrics = dict(query_result.get("metrics") or {})
        pre_values.append(
            float(
                metrics.get(
                    "first_pass_recall_score",
                    metrics.get("pre_rerank_recall_at_k", 0.0),
                )
            )
        )
        post_values.append(
            float(
                metrics.get(
                    "post_rerank_quality_score",
                    metrics.get("post_rerank_ndcg_at_k", 0.0),
                )
            )
        )
    return {
        "pre_rerank_recall_at_k": sum(pre_values) / len(pre_values),
        "post_rerank_ndcg_at_k": sum(post_values) / len(post_values),
        "first_pass_recall_score": sum(pre_values) / len(pre_values),
        "post_rerank_quality_score": sum(post_values) / len(post_values),
    }


def _generate_summary_for_candidate(
    *,
    provider: str,
    model: str,
    source_text: str,
    run_config: dict[str, Any],
) -> str:
    prompts = run_config.get("prompts") or {}
    execution_policy = run_config.get("execution_policy") or {}
    summary = analyze(
        provider,
        source_text,
        prompts.get("user"),
        api_key=(run_config.get("candidate_api_keys") or {}).get(provider),
        system_message=prompts.get("system"),
        temp=execution_policy.get("temperature"),
        model_override=model,
    )
    if not isinstance(summary, str) or not summary.strip() or summary.startswith("Error:"):
        raise ValueError(
            f"Summarization candidate '{provider}:{model}' failed to generate a valid summary."
        )
    return summary.strip()


def _score_summary_with_geval(
    *,
    source_text: str,
    summary: str,
    run_config: dict[str, Any],
    reference_summary: str | None,
) -> dict[str, Any]:
    del reference_summary
    judge_config = run_config.get("judge_config") or {}
    result = run_geval(
        transcript=source_text,
        summary=summary,
        api_key=judge_config.get("api_key"),
        api_name=judge_config.get("provider") or "openai",
        model=judge_config.get("model"),
        save=False,
    )
    return dict(result)


def _build_embeddings_abtest_config(record: Any, dataset: list[dict[str, Any]]) -> EmbeddingsABTestConfig:
    run_config = _run_config_for_execution(record)
    candidates = run_config.get("candidates") or []
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Embeddings recipe run_config.candidates must be populated before execution.")

    comparison_mode = str(run_config.get("comparison_mode") or "embedding_only").strip()
    search_mode = "hybrid" if comparison_mode == "retrieval_stack" else "vector"
    top_k = int(run_config.get("top_k") or 10)
    hybrid_alpha = run_config.get("hybrid_alpha")
    media_ids = _coerce_media_ids(run_config, dataset)

    arms = []
    for candidate in candidates:
        provider, model = _resolve_candidate_provider_model(dict(candidate))
        arms.append(
            ABTestArm(
                provider=provider,
                model=model,
                dimensions=candidate.get("dimensions"),
            )
        )

    queries = []
    for sample in dataset:
        queries.append(
            ABTestQuery(
                text=str(sample.get("input") or ""),
                expected_ids=[
                    int(expected_id)
                    for expected_id in (sample.get("expected_ids") or [])
                ] or None,
                metadata={
                    "query_id": str(sample.get("query_id") or ""),
                },
            )
        )

    return EmbeddingsABTestConfig(
        arms=arms,
        media_ids=media_ids,
        chunking=ABTestChunking(method="sentences", size=200, overlap=20, language=None),
        retrieval=ABTestRetrieval(
            k=top_k,
            search_mode=search_mode,
            hybrid_alpha=float(hybrid_alpha) if hybrid_alpha is not None else None,
        ),
        queries=queries,
        metric_level="media",
        reuse_existing=True,
    )


def _collect_embeddings_candidate_results(
    *,
    db: EvaluationsDatabase,
    test_id: str,
    user_id: str | None,
    record: Any,
) -> list[dict[str, Any]]:
    arms = db.get_abtest_arms(test_id, created_by=user_id or None)
    query_rows = db.get_abtest_queries(test_id, created_by=user_id or None)
    results_rows, _ = db.list_abtest_results(test_id, limit=1000, offset=0, created_by=user_id or None)
    abtest_row = db.get_abtest(test_id, created_by=user_id or None) or {}

    stats_payload = abtest_row.get("stats_json")
    if isinstance(stats_payload, str):
        try:
            stats_payload = json.loads(stats_payload)
        except json.JSONDecodeError:
            stats_payload = {}
    if not isinstance(stats_payload, dict):
        stats_payload = {}
    aggregates = stats_payload.get("aggregates") or {}

    query_lookup: dict[str, dict[str, Any]] = {}
    for row in query_rows:
        metadata_payload = row.get("metadata_json")
        if isinstance(metadata_payload, str):
            try:
                metadata_payload = json.loads(metadata_payload)
            except json.JSONDecodeError:
                metadata_payload = {}
        if not isinstance(metadata_payload, dict):
            metadata_payload = {}
        query_lookup[str(row.get("query_id") or "")] = {
            "expected_ids": [str(value) for value in _parse_json_list(row.get("ground_truth_ids"))],
            "query_id": str(metadata_payload.get("query_id") or row.get("query_id") or ""),
        }

    candidate_configs = list(_run_config_for_execution(record).get("candidates") or [])
    results_by_arm: dict[str, list[dict[str, Any]]] = {}
    for row in results_rows:
        results_by_arm.setdefault(str(row.get("arm_id") or ""), []).append(dict(row))

    candidate_results: list[dict[str, Any]] = []
    for index, arm in enumerate(arms):
        arm_id = str(arm.get("arm_id") or "")
        candidate_cfg = dict(candidate_configs[index]) if index < len(candidate_configs) else {}
        query_results = []
        for row in results_by_arm.get(arm_id, []):
            metrics_payload = row.get("metrics_json")
            if isinstance(metrics_payload, str):
                try:
                    metrics_payload = json.loads(metrics_payload)
                except json.JSONDecodeError:
                    metrics_payload = {}
            query_id = str(row.get("query_id") or "")
            query_results.append(
                {
                    "query_id": query_lookup.get(query_id, {}).get("query_id") or query_id,
                    "ranked_ids": [str(value) for value in _parse_json_list(row.get("ranked_ids"))],
                    "expected_ids": query_lookup.get(query_id, {}).get("expected_ids") or [],
                    "metrics": metrics_payload or {},
                    "latency_ms": row.get("latency_ms"),
                }
            )
        candidate_results.append(
            {
                "candidate_id": arm_id,
                "candidate_run_id": arm_id,
                "model": arm.get("model_id"),
                "provider": arm.get("provider"),
                "is_local": candidate_cfg.get("is_local"),
                "cost_usd": candidate_cfg.get("cost_usd"),
                "metrics": aggregates.get(arm_id) or {},
                "query_results": query_results,
            }
        )
    return candidate_results


def _execute_embeddings_recipe_run(
    *,
    record: Any,
    db: EvaluationsDatabase,
    user_id: str | None,
    service: RecipeRunsService | Any,
) -> dict[str, Any]:
    del service
    resolved_user_id = str(user_id or "")
    dataset = _resolve_embeddings_dataset(record, db, user_id)
    config = _build_embeddings_abtest_config(record, dataset)
    test_id = db.create_abtest(
        name=f"recipe-{record.run_id}",
        config=config.model_dump(),
        created_by=resolved_user_id or None,
    )
    for idx, arm in enumerate(config.arms):
        db.upsert_abtest_arm(
            test_id=test_id,
            arm_index=idx,
            provider=arm.provider,
            model_id=arm.model,
            dimensions=arm.dimensions,
            status="pending",
        )
    db.insert_abtest_queries(test_id, [query.model_dump() for query in config.queries])

    backend = get_content_backend_instance()
    db_path = get_user_media_db_path(resolved_user_id)
    with managed_media_database(
        client_id=f"recipe_runs_jobs_worker:{resolved_user_id}",
        db_path=db_path,
        backend=backend,
    ) as media_db:
        asyncio.run(run_abtest_full(db, config, test_id, resolved_user_id, media_db))
    candidate_results = _collect_embeddings_candidate_results(
        db=db,
        test_id=test_id,
        user_id=resolved_user_id or None,
        record=record,
    )
    child_run_ids = [
        str(candidate_result.get("candidate_run_id") or candidate_result.get("candidate_id") or "")
        for candidate_result in candidate_results
        if str(candidate_result.get("candidate_run_id") or candidate_result.get("candidate_id") or "").strip()
    ]
    recipe_report_inputs = {
        "dataset_mode": record.metadata.get("dataset_mode"),
        "review_sample": record.metadata.get("review_sample") or {
            "required": False,
            "sample_size": 0,
            "sample_query_ids": [],
        },
        "candidate_results": candidate_results,
    }
    return {
        "child_run_ids": child_run_ids,
        "metadata": {
            "abtest": {"test_id": test_id},
            "candidate_results": candidate_results,
            "recipe_report_inputs": recipe_report_inputs,
        },
    }


def _execute_summarization_recipe_run(
    *,
    record: Any,
    db: EvaluationsDatabase,
    user_id: str | None,
    service: RecipeRunsService | Any,
) -> dict[str, Any]:
    del service
    dataset = _resolve_inline_or_persisted_dataset(record, db, user_id)
    run_config = _run_config_for_execution(record)
    candidate_model_ids = list(run_config.get("candidate_model_ids") or [])
    if not candidate_model_ids:
        raise ValueError(
            "Summarization recipe run_config.candidate_model_ids must be populated before execution."
        )

    candidate_results: list[dict[str, Any]] = []
    for candidate_model_id in candidate_model_ids:
        provider, model = _split_candidate_model_id(candidate_model_id)
        sample_results: list[dict[str, Any]] = []
        for index, sample in enumerate(dataset):
            source_text = _extract_summarization_source_text(sample)
            if not source_text:
                raise ValueError(
                    f"Summarization dataset sample {index} is missing input/source_text."
                )
            reference_summary = _extract_summarization_reference(sample)
            sample_id = _extract_summarization_sample_id(sample, index)
            started = time.perf_counter()
            summary = _generate_summary_for_candidate(
                provider=provider,
                model=model,
                source_text=source_text,
                run_config=run_config,
            )
            evaluation = _score_summary_with_geval(
                source_text=source_text,
                summary=summary,
                run_config=run_config,
                reference_summary=reference_summary,
            )
            latency_ms = (time.perf_counter() - started) * 1000.0
            sample_results.append(
                {
                    "sample_id": sample_id,
                    "summary": summary,
                    "reference_summary": reference_summary,
                    "metrics": dict(evaluation.get("metrics") or {}),
                    "assessment": evaluation.get("assessment"),
                    "average_score": evaluation.get("average_score"),
                    "latency_ms": latency_ms,
                }
            )

        candidate_results.append(
            {
                "candidate_id": candidate_model_id,
                "candidate_run_id": candidate_model_id,
                "provider": provider,
                "model": model,
                "is_local": _is_local_summarization_provider(provider),
                "sample_results": sample_results,
            }
        )

    recipe_report_inputs = {
        "dataset_mode": record.metadata.get("dataset_mode"),
        "review_sample": record.metadata.get("review_sample") or {
            "required": False,
            "sample_size": 0,
            "sample_ids": [],
        },
        "weights": run_config.get("weights"),
        "candidate_results": candidate_results,
    }
    return {
        "child_run_ids": [],
        "metadata": {
            "candidate_results": candidate_results,
            "recipe_report_inputs": recipe_report_inputs,
        },
    }


def _execute_rag_retrieval_tuning_recipe_run(
    *,
    record: Any,
    db: EvaluationsDatabase,
    user_id: str | None,
    service: RecipeRunsService | Any,
) -> dict[str, Any]:
    dataset = _resolve_inline_or_persisted_dataset(record, db, user_id)
    run_config = _run_config_for_execution(record)
    recipe = service.recipe_registry.get_recipe(record.recipe_id)
    corpus_scope = dict(run_config.get("corpus_scope") or {})
    candidates = list(run_config.get("candidates") or [])
    if not candidates:
        raise ValueError("RAG retrieval tuning run_config.candidates must be populated before execution.")

    dataset_content_hash = str(record.dataset_content_hash or record.metadata.get("dataset_content_hash") or "").strip()
    owner_user_id = str(user_id or record.metadata.get("owner_user_id") or "").strip()
    index_plans = recipe.plan_candidate_indexes(
        corpus_scope=corpus_scope,
        candidates=candidates,
        dataset_content_hash=dataset_content_hash,
        owner_user_id=owner_user_id,
    )

    candidate_results: list[dict[str, Any]] = []
    serialized_index_plans = {
        candidate_id: serialize_index_plan(plan)
        for candidate_id, plan in index_plans.items()
    }
    for candidate in candidates:
        candidate_id = str(candidate.get("candidate_id") or "").strip()
        plan = index_plans[candidate_id]
        query_results: list[dict[str, Any]] = []
        candidate_latency_values: list[float] = []
        for index, sample in enumerate(dataset):
            query = _extract_rag_query(sample)
            sample_id = _extract_rag_sample_id(sample, index)
            request = recipe.build_unified_rag_request(
                query=query,
                corpus_scope=corpus_scope,
                candidate=candidate,
                index_key=plan.index_key,
            )
            started = time.perf_counter()
            response_payload = asyncio.run(
                _run_unified_rag_request(
                    request=request,
                    user_id=owner_user_id or None,
                )
            )
            latency_ms = (time.perf_counter() - started) * 1000.0
            candidate_latency_values.append(latency_ms)
            first_pass_docs, reranked_docs = _extract_ranked_documents(response_payload)
            normalized_targets = _normalize_rag_targets(sample)
            first_pass_hits = [
                {"grade": _grade_rag_document(doc, normalized_targets)}
                for doc in first_pass_docs
            ]
            reranked_hits = [
                {"grade": _grade_rag_document(doc, normalized_targets)}
                for doc in reranked_docs
            ]
            query_results.append(
                {
                    "query_id": sample_id,
                    "query": query,
                    "latency_ms": latency_ms,
                    "metrics": recipe.summarize_candidate_metrics(
                        first_pass_hits=first_pass_hits,
                        reranked_hits=reranked_hits,
                    )["metrics"],
                    "first_pass_hits": first_pass_hits,
                    "reranked_hits": reranked_hits,
                }
            )

        candidate_results.append(
            {
                "candidate_id": candidate_id,
                "candidate_run_id": candidate_id,
                "is_local": bool(candidate.get("is_local")),
                "cost_usd": candidate.get("cost_usd"),
                "latency_ms": (
                    sum(candidate_latency_values) / len(candidate_latency_values)
                    if candidate_latency_values
                    else None
                ),
                "metrics": _aggregate_rag_candidate_metrics(query_results),
                "query_results": query_results,
                "index_plan": serialized_index_plans[candidate_id],
            }
        )

    recipe_report_inputs = {
        "dataset_mode": record.metadata.get("dataset_mode"),
        "review_sample": record.metadata.get("review_sample") or {
            "required": False,
            "sample_size": 0,
            "sample_ids": [],
        },
        "corpus_scope": corpus_scope,
        "candidate_results": candidate_results,
    }
    return {
        "child_run_ids": [],
        "metadata": {
            "candidate_results": candidate_results,
            "candidate_index_plans": serialized_index_plans,
            "recipe_report_inputs": recipe_report_inputs,
        },
    }


def _execute_recipe_run(
    *,
    record: Any,
    db: EvaluationsDatabase,
    user_id: str | None,
    service: RecipeRunsService | Any,
) -> dict[str, Any] | None:
    if record.recipe_id == "embeddings_model_selection" and not record.metadata.get("candidate_results"):
        return _execute_embeddings_recipe_run(
            record=record,
            db=db,
            user_id=user_id,
            service=service,
        )
    if record.recipe_id == "summarization_quality" and not record.metadata.get("candidate_results"):
        return _execute_summarization_recipe_run(
            record=record,
            db=db,
            user_id=user_id,
            service=service,
        )
    if record.recipe_id == "rag_retrieval_tuning" and not record.metadata.get("candidate_results"):
        return _execute_rag_retrieval_tuning_recipe_run(
            record=record,
            db=db,
            user_id=user_id,
            service=service,
        )
    if record.recipe_id == "rag_answer_quality" and not record.metadata.get("candidate_results"):
        return execute_rag_answer_quality_recipe_run(
            record=record,
            db=db,
            user_id=user_id,
            service=service,
        )
    return None


def handle_recipe_run_job(
    job: dict[str, Any],
    *,
    db: EvaluationsDatabase | None = None,
    user_id: str | None = None,
    service: RecipeRunsService | Any | None = None,
) -> dict[str, Any]:
    """Execute one queued recipe run and persist a normalized report shell."""
    payload = parse_recipe_run_job_payload(job.get("payload") or {})
    resolved_user_id = user_id or payload.get("owner_user_id")
    db = db or _get_db(user_id=resolved_user_id)
    service = service or _get_service(user_id=resolved_user_id, db=db)
    run_id = payload["run_id"]
    job_id = str(job.get("id")) if job.get("id") is not None else None

    service.get_run(run_id)
    record = db.get_recipe_run(run_id)
    if record is None:
        raise ValueError(f"Recipe run '{run_id}' was not found.")
    if record.status is RunStatus.COMPLETED:
        return {
            "status": "completed",
            "run_id": run_id,
            "job_id": job_id,
            "recipe_id": record.recipe_id,
            "reused": True,
        }

    running_metadata = dict(record.metadata)
    running_metadata["jobs"] = {
        "job_id": job_id,
        "worker_state": "running",
    }
    db.update_recipe_run(
        run_id,
        status=RunStatus.RUNNING,
        metadata=running_metadata,
    )

    try:
        execution_artifacts = _execute_recipe_run(
            record=record,
            db=db,
            user_id=resolved_user_id,
            service=service,
        )
        if execution_artifacts:
            refreshed_record = db.get_recipe_run(run_id)
            if refreshed_record is None:
                raise ValueError(f"Recipe run '{run_id}' disappeared during execution.")
            merged_metadata = dict(refreshed_record.metadata)
            merged_metadata.update(execution_artifacts.get("metadata") or {})
            db.update_recipe_run(
                run_id,
                metadata=merged_metadata,
            )
            child_run_ids = execution_artifacts.get("child_run_ids") or []
            if child_run_ids:
                db.set_recipe_run_children(run_id, list(child_run_ids))
        report = service.get_report(run_id)
        refreshed_record = db.get_recipe_run(run_id)
        if refreshed_record is None:
            raise ValueError(f"Recipe run '{run_id}' disappeared during report persistence.")
        completed_metadata = dict(refreshed_record.metadata)
        recipe_report = report.run.metadata.get("recipe_report")
        if recipe_report is not None:
            completed_metadata["recipe_report"] = recipe_report
        completed_metadata["jobs"] = {
            "job_id": job_id,
            "worker_state": "completed",
        }
        db.update_recipe_run(
            run_id,
            status=RunStatus.COMPLETED,
            confidence_summary=report.confidence_summary,
            recommendation_slots=report.recommendation_slots,
            metadata=completed_metadata,
        )
    except Exception as exc:
        job_error = _classify_recipe_run_job_error(exc)
        logger.exception("Recipe run job failed: run_id={} job_id={}", run_id, job_id)
        failed_record = db.get_recipe_run(run_id)
        failed_metadata = dict(failed_record.metadata) if failed_record is not None else {}
        failed_metadata["jobs"] = {
            "job_id": job_id,
            "worker_state": "retrying" if job_error.retryable else "failed",
            "error": "recipe_run_retrying" if job_error.retryable else "recipe_run_failed",
            "error_type": type(exc).__name__,
            "error_message": (
                "Recipe run execution will be retried."
                if job_error.retryable
                else "Recipe run execution failed."
            ),
            "retryable": job_error.retryable,
        }
        db.update_recipe_run(
            run_id,
            status=RunStatus.PENDING if job_error.retryable else RunStatus.FAILED,
            metadata=failed_metadata,
        )
        raise job_error from exc

    logger.info("Recipe run job completed: run_id={} job_id={}", run_id, job_id)
    return {
        "status": "completed",
        "run_id": run_id,
        "job_id": job_id,
        "recipe_id": payload["recipe_id"],
        "reused": False,
    }


async def handle_recipe_run_job_async(job: dict[str, Any]) -> dict[str, Any]:
    """Async WorkerSDK adapter for the synchronous recipe run handler."""
    return await asyncio.to_thread(handle_recipe_run_job, job)


async def run_recipe_run_jobs_worker(stop_event: asyncio.Event | None = None) -> None:
    """Run the WorkerSDK loop for recipe-run Jobs."""
    worker_id = (os.getenv("EVALUATIONS_RECIPE_RUN_JOBS_WORKER_ID") or f"recipe-run-{os.getpid()}").strip()
    cfg = WorkerConfig(
        domain=RECIPE_RUN_JOB_DOMAIN,
        queue=recipe_run_queue(),
        worker_id=worker_id,
    )
    jm = JobManager()
    sdk = WorkerSDK(jm, cfg)
    stop_task: asyncio.Task[None] | None = None
    if stop_event is not None:
        async def _watch_stop() -> None:
            await stop_event.wait()
            sdk.stop()

        stop_task = asyncio.create_task(
            _watch_stop(),
            name="recipe_run_jobs_worker_stop_watch",
        )
    logger.info("Recipe run Jobs worker starting: queue={} worker_id={}", cfg.queue, worker_id)
    try:
        await sdk.run(handler=handle_recipe_run_job_async)
    finally:
        if stop_task is not None:
            stop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stop_task


def recipe_run_jobs_worker_enabled() -> bool:
    """Return True when the recipe-run Jobs worker is explicitly enabled."""
    return env_flag_enabled("EVALUATIONS_RECIPE_RUN_JOBS_WORKER_ENABLED") or env_flag_enabled(
        "EVALS_RECIPE_RUN_JOBS_WORKER_ENABLED"
    )


async def start_recipe_run_jobs_worker(
    stop_event: asyncio.Event | None = None,
) -> asyncio.Task[None] | None:
    """Start the recipe-run worker as a background task."""
    if not recipe_run_jobs_worker_enabled():
        return None
    return asyncio.create_task(
        run_recipe_run_jobs_worker(stop_event),
        name="recipe_run_jobs_worker",
    )


__all__ = [
    "RecipeRunJobError",
    "handle_recipe_run_job",
    "handle_recipe_run_job_async",
    "recipe_run_jobs_worker_enabled",
    "run_recipe_run_jobs_worker",
    "start_recipe_run_jobs_worker",
]
