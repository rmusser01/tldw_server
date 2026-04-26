"""
Benchmark endpoints extracted from the legacy benchmark router and mounted under
the unified evaluations namespace.
"""

import asyncio
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger
from pydantic import BaseModel, Field

from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.AuthNZ.permissions import EVALS_MANAGE, EVALS_READ
from tldw_Server_API.app.core.Evaluations.benchmark_loaders import load_benchmark_dataset
from tldw_Server_API.app.core.Evaluations.benchmark_registry import get_registry
from tldw_Server_API.app.core.Evaluations.evaluation_manager import EvaluationManager
from tldw_Server_API.app.core.Evaluations.identity import EvaluationIdentity

from .evaluations_auth import (
    check_evaluation_rate_limit,
    get_evaluation_identity,
    get_eval_request_user,
    require_eval_permissions,
    sanitize_error_message,
    verify_api_key,
)

benchmarks_router = APIRouter()


def _get_evaluation_manager_for_user(identity: EvaluationIdentity) -> EvaluationManager:
    return EvaluationManager(user_id=identity.user_scope)


class BenchmarkRunRequest(BaseModel):
    limit: Optional[int] = Field(None, ge=1, description="Limit number of samples")
    api_name: str = Field("openai", description="Provider to use for evaluation")
    parallel: int = Field(4, ge=1, le=16, description="Number of parallel workers")
    save_results: bool = Field(True, description="Persist summary into evaluations DB")
    filter_categories: Optional[list[str]] = Field(
        None,
        description="Optional category/topic/domain filter",
    )


class BenchmarkRunResponse(BaseModel):
    benchmark: str
    total_samples: int
    results_summary: dict[str, Any]
    evaluation_id: Optional[str] = None


@benchmarks_router.get(
    "/benchmarks",
    dependencies=[Depends(require_eval_permissions(EVALS_READ))],
)
async def list_benchmarks(
    user_id: str = Depends(verify_api_key),
    current_user: User = Depends(get_eval_request_user),
):
    _ = user_id, current_user
    registry = get_registry()
    data: list[dict[str, Any]] = []
    for name in registry.list_benchmarks():
        info = registry.get_benchmark_info(name)
        if info:
            data.append(info)
    return {"object": "list", "data": data, "total": len(data)}


@benchmarks_router.get(
    "/benchmarks/{benchmark_name}",
    dependencies=[Depends(require_eval_permissions(EVALS_READ))],
)
async def get_benchmark_info(
    benchmark_name: str,
    user_id: str = Depends(verify_api_key),
    current_user: User = Depends(get_eval_request_user),
):
    _ = user_id, current_user
    registry = get_registry()
    config = registry.get(benchmark_name)
    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Benchmark '{benchmark_name}' not found",
        )
    return registry.get_benchmark_info(benchmark_name)


@benchmarks_router.post(
    "/benchmarks/{benchmark_name}/run",
    response_model=BenchmarkRunResponse,
    dependencies=[
        Depends(require_eval_permissions(EVALS_MANAGE)),
        Depends(check_evaluation_rate_limit),
    ],
)
async def run_benchmark(
    benchmark_name: str,
    request: BenchmarkRunRequest,
    user_id: str = Depends(verify_api_key),
    current_user: User = Depends(get_eval_request_user),
):
    try:
        identity = get_evaluation_identity(current_user)
        evaluation_manager = _get_evaluation_manager_for_user(identity)
        stable_user_id = identity.created_by
        registry = get_registry()
        config = registry.get(benchmark_name)
        if not config:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Benchmark '{benchmark_name}' not found",
            )

        evaluator = registry.create_evaluator(benchmark_name)
        if not evaluator:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail=f"Evaluator not implemented for benchmark type: {config.evaluation_type}",
            )

        # Do not force config.dataset_source globally; benchmark-specific loaders
        # decide whether to use packaged defaults or an explicit source.
        dataset = load_benchmark_dataset(benchmark_name, limit=request.limit)
        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Failed to load dataset for benchmark '{benchmark_name}'",
            )

        if request.filter_categories:
            allowed = {str(item) for item in request.filter_categories}
            dataset = [
                item
                for item in dataset
                if (
                    str(item.get("category")) in allowed
                    or str(item.get("topic")) in allowed
                    or str(item.get("domain")) in allowed
                )
            ]

        results = []
        batch_size = request.parallel

        for i in range(0, len(dataset), batch_size):
            batch = dataset[i : i + batch_size]
            tasks = []
            for item in batch:
                eval_data = evaluator.format_for_custom_metric(item)
                task = evaluation_manager.evaluate_custom_metric(
                    metric_name=eval_data["name"],
                    description=eval_data["description"],
                    evaluation_prompt=eval_data["evaluation_prompt"],
                    input_data=eval_data["input_data"],
                    scoring_criteria=eval_data["scoring_criteria"],
                    api_name=request.api_name,
                )
                tasks.append(task)

            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            for item, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    logger.error("Benchmark evaluation failed for item_id={}: {}", item.get("id"), result)
                    results.append({"item": item, "score": 0.0, "error": str(result)})
                else:
                    results.append(
                        {
                            "item": item,
                            "score": result.get("score", 0.0),
                            "explanation": result.get("explanation", ""),
                        }
                    )

        successful_results = [r for r in results if "error" not in r]
        scores = [float(r.get("score", 0.0)) for r in successful_results]

        summary: dict[str, Any] = {
            "total_evaluated": len(results),
            "successful": len(successful_results),
            "failed": len(results) - len(successful_results),
            "average_score": (sum(scores) / len(scores)) if scores else 0.0,
            "min_score": min(scores) if scores else 0.0,
            "max_score": max(scores) if scores else 0.0,
        }

        grouped_scores: dict[str, list[float]] = {}
        for record in successful_results:
            item = record.get("item", {})
            key = item.get("category") or item.get("topic") or item.get("domain") or "unknown"
            grouped_scores.setdefault(str(key), []).append(float(record.get("score", 0.0)))
        if grouped_scores:
            summary["by_category"] = {
                key: {"count": len(values), "average": sum(values) / len(values)}
                for key, values in grouped_scores.items()
            }

        evaluation_id = None
        if request.save_results:
            evaluation_id = await evaluation_manager.store_evaluation(
                evaluation_type=benchmark_name,
                input_data={"dataset": [r["item"] for r in results[:10]]},
                results={"summary": summary, "scores": scores},
                metadata={
                    "api_name": request.api_name,
                    "user_id": stable_user_id,
                    "current_user_id": identity.user_scope,
                    "total_samples": len(dataset),
                },
            )

        return BenchmarkRunResponse(
            benchmark=benchmark_name,
            total_samples=len(results),
            results_summary=summary,
            evaluation_id=evaluation_id,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to run benchmark: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to run benchmark: {sanitize_error_message(e, 'benchmark run')}",
        ) from e
