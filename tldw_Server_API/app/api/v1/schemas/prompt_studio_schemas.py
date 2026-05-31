# prompt_studio_schemas.py
# Comprehensive schemas for Prompt Studio feature

from datetime import datetime
from enum import Enum
from typing import Any, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tldw_Server_API.app.api.v1.schemas.pagination import (
    OffsetPaginationMeta,
    default_offset_pagination_aliases,
)
from .prompt_studio_base import EvaluationStatus, JobStatus, JobType, TimestampMixin, UUIDMixin

########################################################################################################################
# Evaluation Schemas

class EvaluationMetrics(BaseModel):
    """Metrics for evaluation"""
    accuracy: Optional[float] = Field(None, ge=0, le=1)
    precision: Optional[float] = Field(None, ge=0, le=1)
    recall: Optional[float] = Field(None, ge=0, le=1)
    f1_score: Optional[float] = Field(None, ge=0, le=1)
    bleu_score: Optional[float] = Field(None, ge=0, le=1)
    rouge_score: Optional[dict[str, float]] = None
    perplexity: Optional[float] = Field(None, ge=0)
    latency_ms: Optional[float] = Field(None, ge=0)
    tokens_used: Optional[int] = Field(None, ge=0)
    cost: Optional[float] = Field(None, ge=0)
    custom_metrics: Optional[dict[str, Any]] = None

class EvaluationConfig(BaseModel):
    """Configuration for evaluation"""
    model_name: Optional[str] = Field(None, max_length=100)
    temperature: Optional[float] = Field(None, ge=0, le=2)
    max_tokens: Optional[int] = Field(None, ge=1, le=100000)
    top_p: Optional[float] = Field(None, ge=0, le=1)
    frequency_penalty: Optional[float] = Field(None, ge=-2, le=2)
    presence_penalty: Optional[float] = Field(None, ge=-2, le=2)
    api_endpoint: Optional[str] = Field(None, max_length=500)
    api_key_name: Optional[str] = Field(None, max_length=100)
    timeout_seconds: Optional[int] = Field(None, ge=1, le=3600)
    retry_count: Optional[int] = Field(None, ge=0, le=10)
    parallel_requests: Optional[int] = Field(None, ge=1, le=100)

class EvaluationCreate(BaseModel):
    """Create evaluation request"""
    project_id: int = Field(..., description="Project ID")
    prompt_id: int = Field(..., description="Prompt ID to evaluate")
    test_run_id: Optional[Union[int, str]] = Field(None, description="Test run ID if part of a test run")
    name: Optional[str] = Field(None, max_length=200, description="Evaluation name")
    description: Optional[str] = Field(None, max_length=1000, description="Evaluation description")
    metrics: Optional[EvaluationMetrics] = Field(None, description="Metrics to track")
    config: Optional[EvaluationConfig] = Field(None, description="Evaluation configuration")
    run_async: bool = Field(default=False, description="Run evaluation asynchronously")
    test_case_ids: Optional[list[int]] = Field(None, description="Specific test cases to run")
    tags: Optional[list[str]] = Field(None, max_length=20, description="Tags for categorization")

class EvaluationUpdate(BaseModel):
    """Update evaluation request"""
    metrics: Optional[EvaluationMetrics] = Field(None, description="Updated metrics")
    status: Optional[EvaluationStatus] = Field(None, description="Updated status")
    error_message: Optional[str] = Field(None, max_length=1000, description="Error message if failed")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")

class EvaluationResponse(TimestampMixin, UUIDMixin):
    """Evaluation response"""
    id: int = Field(..., description="Evaluation ID")
    project_id: int = Field(..., description="Project ID")
    prompt_id: int = Field(..., description="Prompt ID")
    test_run_id: Optional[int] = Field(None, description="Test run ID")
    name: Optional[str] = Field(None, description="Evaluation name")
    description: Optional[str] = Field(None, description="Evaluation description")
    metrics: dict[str, Any] = Field(default_factory=dict, description="Evaluation metrics")
    config: dict[str, Any] = Field(default_factory=dict, description="Evaluation config")
    status: EvaluationStatus = Field(..., description="Current status")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    tags: list[str] = Field(default_factory=list, description="Tags")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")

    model_config = ConfigDict(from_attributes=True)

class EvaluationList(BaseModel):
    """List of evaluations response"""
    evaluations: list[EvaluationResponse]
    total: int = Field(..., ge=0, description="Total count")
    limit: int = Field(..., ge=1, description="Page limit")
    offset: int = Field(..., ge=0, description="Page offset")
    has_more: bool | None = Field(default=None, description="Alias for pagination.has_more")
    next_offset: int | None = Field(default=None, ge=0, description="Alias for pagination.next_offset")
    pagination: OffsetPaginationMeta

    @model_validator(mode="after")
    def _default_pagination_aliases(self) -> "EvaluationList":
        return default_offset_pagination_aliases(self)


# Simple execute request for prompts (compat endpoint)
class ExecutePromptSimpleRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    prompt_id: int
    inputs: dict[str, Any] = Field(default_factory=dict)
    provider: Optional[str] = Field(default="openai")
    model: Optional[str] = Field(default="gpt-3.5-turbo")

########################################################################################################################
# Optimization Schemas

class OptimizationStrategy(str, Enum):
    """Optimization strategies"""
    PROMPT_TUNING = "prompt_tuning"
    FEW_SHOT = "few_shot"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    INSTRUCTION_TUNING = "instruction_tuning"
    REINFORCEMENT = "reinforcement"
    GENETIC = "genetic"
    BAYESIAN = "bayesian"
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"

class OptimizationConfig(BaseModel):
    """Configuration for optimization"""
    strategy: OptimizationStrategy = Field(..., description="Optimization strategy")
    max_iterations: int = Field(default=10, ge=1, le=1000, description="Max iterations")
    target_metric: str = Field(..., max_length=100, description="Metric to optimize")
    target_value: Optional[float] = Field(None, description="Target value for metric")
    constraints: Optional[dict[str, Any]] = Field(None, description="Optimization constraints")
    hyperparameters: Optional[dict[str, Any]] = Field(None, description="Strategy hyperparameters")
    early_stopping: bool = Field(default=True, description="Enable early stopping")
    patience: int = Field(default=3, ge=1, le=100, description="Early stopping patience")

class OptimizationCreate(BaseModel):
    """Create optimization request"""
    project_id: int = Field(..., description="Project ID")
    prompt_id: int = Field(..., description="Prompt ID to optimize")
    name: Optional[str] = Field(None, max_length=200, description="Optimization name")
    description: Optional[str] = Field(None, max_length=1000, description="Optimization description")
    config: OptimizationConfig = Field(..., description="Optimization configuration")
    test_case_ids: Optional[list[int]] = Field(None, description="Test cases to use")
    baseline_evaluation_id: Optional[int] = Field(None, description="Baseline evaluation for comparison")
    run_async: bool = Field(default=True, description="Run optimization asynchronously")

class OptimizationUpdate(BaseModel):
    """Update optimization request"""
    status: Optional[JobStatus] = Field(None, description="Updated status")
    current_iteration: Optional[int] = Field(None, ge=0, description="Current iteration")
    best_prompt_id: Optional[int] = Field(None, description="Best prompt found so far")
    best_score: Optional[float] = Field(None, description="Best score achieved")
    error_message: Optional[str] = Field(None, max_length=1000, description="Error message")
    results: Optional[dict[str, Any]] = Field(None, description="Optimization results")

class OptimizationResponse(TimestampMixin, UUIDMixin):
    """Optimization response"""
    id: int = Field(..., description="Optimization ID")
    project_id: int = Field(..., description="Project ID")
    prompt_id: int = Field(..., description="Original prompt ID")
    name: Optional[str] = Field(None, description="Optimization name")
    description: Optional[str] = Field(None, description="Optimization description")
    config: dict[str, Any] = Field(..., description="Optimization config")
    status: JobStatus = Field(..., description="Current status")
    current_iteration: int = Field(default=0, description="Current iteration")
    max_iterations: int = Field(..., description="Maximum iterations")
    best_prompt_id: Optional[int] = Field(None, description="Best prompt ID")
    best_score: Optional[float] = Field(None, description="Best score")
    results: dict[str, Any] = Field(default_factory=dict, description="Results")
    error_message: Optional[str] = Field(None, description="Error message")

    model_config = ConfigDict(from_attributes=True)

class OptimizationList(BaseModel):
    """List of optimizations response"""
    optimizations: list[OptimizationResponse]
    total: int = Field(..., ge=0, description="Total count")
    limit: int = Field(..., ge=1, description="Page limit")
    offset: int = Field(..., ge=0, description="Page offset")

########################################################################################################################
# WebSocket Schemas

class WebSocketMessage(BaseModel):
    """WebSocket message"""
    type: str = Field(..., max_length=50, description="Message type")
    data: dict[str, Any] = Field(..., description="Message data")
    timestamp: datetime = Field(default_factory=datetime.now, description="Message timestamp")
    request_id: Optional[str] = Field(None, max_length=100, description="Request ID for correlation")

class WebSocketRequest(BaseModel):
    """WebSocket request"""
    action: str = Field(..., max_length=50, description="Action to perform")
    data: dict[str, Any] = Field(default_factory=dict, description="Request data")
    request_id: Optional[str] = Field(None, max_length=100, description="Request ID")

class WebSocketResponse(BaseModel):
    """WebSocket response"""
    type: str = Field(..., max_length=50, description="Response type")
    success: bool = Field(..., description="Success status")
    data: Optional[dict[str, Any]] = Field(None, description="Response data")
    error: Optional[str] = Field(None, description="Error message")
    request_id: Optional[str] = Field(None, description="Request ID")

########################################################################################################################
# Job Queue Schemas

class JobCreate(BaseModel):
    """Create job request"""
    project_id: int = Field(..., description="Project ID")
    job_type: JobType = Field(..., description="Job type")
    entity_id: int = Field(..., description="Entity ID (prompt, evaluation, etc)")
    config: dict[str, Any] = Field(default_factory=dict, description="Job configuration")
    priority: int = Field(default=5, ge=0, le=10, description="Job priority")
    scheduled_at: Optional[datetime] = Field(None, description="Schedule time")

class JobUpdate(BaseModel):
    """Update job request"""
    status: Optional[JobStatus] = Field(None, description="Updated status")
    progress: Optional[float] = Field(None, ge=0, le=1, description="Progress (0-1)")
    result: Optional[dict[str, Any]] = Field(None, description="Job result")
    error: Optional[str] = Field(None, description="Error message")
    completed_at: Optional[datetime] = Field(None, description="Completion time")

class JobResponse(TimestampMixin, UUIDMixin):
    """Job response"""
    id: int = Field(..., description="Job ID")
    project_id: int = Field(..., description="Project ID")
    job_type: JobType = Field(..., description="Job type")
    entity_id: int = Field(..., description="Entity ID")
    status: JobStatus = Field(..., description="Job status")
    priority: int = Field(..., description="Priority")
    progress: float = Field(default=0, description="Progress")
    config: dict[str, Any] = Field(default_factory=dict, description="Configuration")
    result: Optional[dict[str, Any]] = Field(None, description="Result")
    error: Optional[str] = Field(None, description="Error message")
    scheduled_at: Optional[datetime] = Field(None, description="Scheduled time")
    started_at: Optional[datetime] = Field(None, description="Start time")
    completed_at: Optional[datetime] = Field(None, description="Completion time")

    model_config = ConfigDict(from_attributes=True)

class JobList(BaseModel):
    """List of jobs response"""
    jobs: list[JobResponse]
    total: int = Field(..., ge=0, description="Total count")
    limit: int = Field(..., ge=1, description="Page limit")
    offset: int = Field(..., ge=0, description="Page offset")

########################################################################################################################
# Analytics Schemas

class PromptAnalytics(BaseModel):
    """Analytics for a prompt"""
    prompt_id: int = Field(..., description="Prompt ID")
    total_evaluations: int = Field(default=0, description="Total evaluations")
    average_accuracy: Optional[float] = Field(None, description="Average accuracy")
    average_latency: Optional[float] = Field(None, description="Average latency (ms)")
    total_tokens: int = Field(default=0, description="Total tokens used")
    total_cost: float = Field(default=0, description="Total cost")
    success_rate: Optional[float] = Field(None, description="Success rate")
    last_evaluated: Optional[datetime] = Field(None, description="Last evaluation time")

class ProjectAnalytics(BaseModel):
    """Analytics for a project"""
    project_id: int = Field(..., description="Project ID")
    total_prompts: int = Field(default=0, description="Total prompts")
    total_evaluations: int = Field(default=0, description="Total evaluations")
    total_optimizations: int = Field(default=0, description="Total optimizations")
    total_test_cases: int = Field(default=0, description="Total test cases")
    active_jobs: int = Field(default=0, description="Active jobs")
    average_accuracy: Optional[float] = Field(None, description="Average accuracy across prompts")
    total_tokens: int = Field(default=0, description="Total tokens used")
    total_cost: float = Field(default=0, description="Total cost")
    last_activity: Optional[datetime] = Field(None, description="Last activity time")

########################################################################################################################
# Export/Import Schemas

class ExportRequest(BaseModel):
    """Export request"""
    project_id: int = Field(..., description="Project ID to export")
    include_prompts: bool = Field(default=True, description="Include prompts")
    include_test_cases: bool = Field(default=True, description="Include test cases")
    include_evaluations: bool = Field(default=False, description="Include evaluations")
    include_optimizations: bool = Field(default=False, description="Include optimizations")
    format: str = Field(default="json", pattern="^(json|yaml|csv)$", description="Export format")

class ExportResponse(BaseModel):
    """Export response"""
    project: dict[str, Any] = Field(..., description="Project data")
    prompts: Optional[list[dict[str, Any]]] = Field(None, description="Prompts")
    test_cases: Optional[list[dict[str, Any]]] = Field(None, description="Test cases")
    evaluations: Optional[list[dict[str, Any]]] = Field(None, description="Evaluations")
    optimizations: Optional[list[dict[str, Any]]] = Field(None, description="Optimizations")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Export metadata")

class ImportRequest(BaseModel):
    """Import request"""
    data: dict[str, Any] = Field(..., description="Data to import")
    create_new_project: bool = Field(default=True, description="Create new project")
    project_id: Optional[int] = Field(None, description="Existing project ID if not creating new")
    merge_strategy: str = Field(default="skip", pattern="^(skip|overwrite|version)$", description="Merge strategy")
