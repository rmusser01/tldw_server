"""
Benchmark API endpoints for the evaluation system.

Provides endpoints for running and managing benchmarks.
"""

import asyncio
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from loguru import logger
from pydantic import BaseModel, Field

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_rate_limiter_dep
from tldw_Server_API.app.core.Evaluations.benchmark_loaders import load_benchmark_dataset
from tldw_Server_API.app.core.Evaluations.benchmark_registry import get_registry
from tldw_Server_API.app.core.Evaluations.evaluation_manager import EvaluationManager

# Create router
router = APIRouter(prefix="/benchmarks", tags=["benchmarks"])

# Initialize components
evaluation_manager = EvaluationManager()


# Request/Response schemas
class BenchmarkListResponse(BaseModel):
    """Response for benchmark list."""
    benchmarks: list[dict[str, Any]] = Field(..., description="List of available benchmarks")
    total: int = Field(..., description="Total number of benchmarks")


class BenchmarkInfoResponse(BaseModel):
    """Response for benchmark information."""
    name: str = Field(..., description="Benchmark name")
    description: str = Field(..., description="Benchmark description")
    evaluation_type: str = Field(..., description="Type of evaluation")
    dataset_source: str = Field(..., description="Dataset source")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class BenchmarkRunRequest(BaseModel):
    """Request to run a benchmark."""
    limit: Optional[int] = Field(None, description="Limit number of samples")
    api_name: str = Field("openai", description="API to use for evaluation")
    api_key: Optional[str] = Field(None, description="API key if not in config")
    parallel: int = Field(4, ge=1, le=16, description="Number of parallel workers")
    save_results: bool = Field(True, description="Save results to database")
    filter_categories: Optional[list[str]] = Field(None, description="Filter by categories")


class BenchmarkRunResponse(BaseModel):
    """Response from benchmark run."""
    benchmark: str = Field(..., description="Benchmark name")
    total_samples: int = Field(..., description="Total samples evaluated")
    results_summary: dict[str, Any] = Field(..., description="Summary of results")
    evaluation_id: Optional[str] = Field(None, description="Evaluation ID if saved")


class BenchmarkSampleResponse(BaseModel):
    """Response with benchmark samples."""
    benchmark: str = Field(..., description="Benchmark name")
    samples: list[dict[str, Any]] = Field(..., description="Sample questions/items")
    total_available: int = Field(..., description="Total samples in dataset")


@router.get("/list", response_model=BenchmarkListResponse)
async def list_benchmarks():
    """
    List all available benchmarks.

    Returns information about registered benchmarks including
    their types, descriptions, and metadata.
    """
    try:
        registry = get_registry()
        benchmark_names = registry.list_benchmarks()

        benchmarks = []
        for name in benchmark_names:
            info = registry.get_benchmark_info(name)
            if info:
                benchmarks.append(info)

        return BenchmarkListResponse(
            benchmarks=benchmarks,
            total=len(benchmarks)
        )

    except Exception as e:
        logger.error("Failed to list benchmarks")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list benchmarks"
        ) from e


@router.get("/{benchmark_name}/info", response_model=BenchmarkInfoResponse)
async def get_benchmark_info(benchmark_name: str):
    """
    Get detailed information about a specific benchmark.

    Args:
        benchmark_name: Name of the benchmark

    Returns:
        Detailed benchmark information including configuration and metadata.
    """
    try:
        registry = get_registry()
        config = registry.get(benchmark_name)

        if not config:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Benchmark '{benchmark_name}' not found"
            )

        info = registry.get_benchmark_info(benchmark_name)
        return BenchmarkInfoResponse(**info)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get benchmark info")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get benchmark info"
        ) from e


@router.get("/{benchmark_name}/samples", response_model=BenchmarkSampleResponse)
async def get_benchmark_samples(
    benchmark_name: str,
    limit: int = Query(5, ge=1, le=100, description="Number of samples to return")
):
    """
    Get sample items from a benchmark dataset.

    Useful for previewing benchmark questions before running full evaluation.

    Args:
        benchmark_name: Name of the benchmark
        limit: Number of samples to return (max 100)

    Returns:
        Sample items from the benchmark dataset.
    """
    try:
        registry = get_registry()
        config = registry.get(benchmark_name)

        if not config:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Benchmark '{benchmark_name}' not found"
            )

        # Load limited samples
        dataset = load_benchmark_dataset(benchmark_name, limit=limit)

        # Get total count (load 1 more to check if there are more)
        full_dataset = load_benchmark_dataset(benchmark_name, limit=limit + 1)
        total_available = len(full_dataset) if len(full_dataset) <= limit else f"{limit}+"

        return BenchmarkSampleResponse(
            benchmark=benchmark_name,
            samples=dataset,
            total_available=total_available if isinstance(total_available, int) else limit
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get benchmark samples")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get samples"
        ) from e


@router.post("/{benchmark_name}/run", response_model=BenchmarkRunResponse)
async def run_benchmark(
    benchmark_name: str,
    request: BenchmarkRunRequest,
    user_id: str = Depends(get_rate_limiter_dep)
):
    """
    Run a benchmark evaluation.

    This endpoint loads the benchmark dataset, evaluates each item,
    and returns aggregated results.

    Args:
        benchmark_name: Name of the benchmark to run
        request: Run configuration
        user_id: User ID from auth

    Returns:
        Summary of benchmark results.
    """
    try:
        registry = get_registry()
        config = registry.get(benchmark_name)

        if not config:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Benchmark '{benchmark_name}' not found"
            )

        # Create evaluator
        evaluator = registry.create_evaluator(benchmark_name)
        if not evaluator:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail=f"Evaluator not implemented for benchmark type: {config.evaluation_type}"
            )

        # Load dataset
        dataset = load_benchmark_dataset(benchmark_name, limit=request.limit)
        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Failed to load dataset for benchmark '{benchmark_name}'"
            )

        # Filter by categories if specified
        if request.filter_categories:
            dataset = [
                item for item in dataset
                if item.get("category") in request.filter_categories
                or item.get("topic") in request.filter_categories
            ]

        # Process items in batches
        results = []
        batch_size = request.parallel

        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]

            # Create evaluation tasks
            tasks = []
            for item in batch:
                # Format for evaluation
                eval_data = evaluator.format_for_custom_metric(item)

                # Create evaluation task
                task = evaluation_manager.evaluate_custom_metric(
                    metric_name=eval_data['name'],
                    description=eval_data['description'],
                    evaluation_prompt=eval_data['evaluation_prompt'],
                    input_data=eval_data['input_data'],
                    scoring_criteria=eval_data['scoring_criteria'],
                    api_name=request.api_name,
                    api_key=request.api_key,
                )
                tasks.append(task)

            # Run batch
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for item, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    logger.error("Benchmark item evaluation failed")
                    results.append({
                        "item": item,
                        "score": 0.0,
                        "error": str(result)
                    })
                else:
                    results.append({
                        "item": item,
                        "score": result.get("score", 0.0),
                        "explanation": result.get("explanation", "")
                    })

        # Calculate summary statistics
        successful_results = [r for r in results if "error" not in r]
        scores = [r["score"] for r in successful_results]

        summary = {
            "total_evaluated": len(results),
            "successful": len(successful_results),
            "failed": len(results) - len(successful_results),
            "average_score": sum(scores) / len(scores) if scores else 0.0,
            "min_score": min(scores) if scores else 0.0,
            "max_score": max(scores) if scores else 0.0
        }

        # Add category breakdown if available
        categories = {}
        for r in successful_results:
            cat = r["item"].get("category") or r["item"].get("topic", "unknown")
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(r["score"])

        if categories:
            summary["by_category"] = {
                cat: {
                    "count": len(scores),
                    "average": sum(scores) / len(scores)
                }
                for cat, scores in categories.items()
            }

        # Save results if requested
        eval_id = None
        if request.save_results:
            eval_id = await evaluation_manager.store_evaluation(
                evaluation_type=benchmark_name,
                input_data={"dataset": [r["item"] for r in results[:10]]},  # Sample for storage
                results={
                    "summary": summary,
                    "scores": [r["score"] for r in successful_results]
                },
                metadata={
                    "api_name": request.api_name,
                    "user_id": user_id,
                    "total_samples": len(dataset)
                }
            )

        return BenchmarkRunResponse(
            benchmark=benchmark_name,
            total_samples=len(results),
            results_summary=summary,
            evaluation_id=eval_id
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to run benchmark")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to run benchmark"
        ) from e


# Special endpoint for SimpleQA with its grading system
@router.post("/simpleqa/evaluate")
async def evaluate_simpleqa(
    question: str = Query(..., description="The question to ask"),
    api_name: str = Query("openai", description="API to use"),
    strict_grading: bool = Query(True, description="Use strict grading"),
    user_id: str = Depends(get_rate_limiter_dep)
):
    """
    Evaluate a single SimpleQA-style question.

    Returns the three-grade classification: correct, incorrect, or not_attempted.
    """
    try:
        from tldw_Server_API.app.core.Evaluations.simpleqa_eval import SimpleQAEvaluation

        SimpleQAEvaluation(
            grading_model=api_name,
            strict_grading=strict_grading
        )

        # This would need to be integrated with the actual model calling
        # For now, return a placeholder
        return {
            "question": question,
            "grade": "not_implemented",
            "explanation": "SimpleQA evaluation endpoint placeholder",
            "note": "Full implementation requires model integration"
        }

    except Exception as e:
        logger.error("SimpleQA evaluation failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Evaluation failed"
        ) from e
