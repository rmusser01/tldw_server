# openai_eval_schemas.py - OpenAI-compatible evaluation schemas
"""
Pydantic schemas for OpenAI-compatible evaluation API.

Follows OpenAI's API conventions:
- Snake case field names
- "object" field for resource identification
- Unix timestamps for dates
- Consistent list response format
"""

from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ============= Base Models =============

class OpenAIBaseModel(BaseModel):
    """Base model with OpenAI conventions"""
    model_config = ConfigDict(
        populate_by_name=True,
        # Keep identity alias generator for explicitness
        alias_generator=lambda field_name: field_name,
    )


# ============= Evaluation Models =============

class EvaluationSpec(OpenAIBaseModel):
    """Specification for how to run an evaluation"""
    evaluator_model: Optional[str] = Field(
        default="gpt-4",
        description="Model to use for evaluation"
    )
    metrics: list[str] = Field(
        default_factory=list,
        description="Metrics to evaluate"
    )
    scoring_prompt: Optional[str] = Field(
        default=None,
        description="Custom prompt for scoring"
    )
    threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Pass/fail threshold"
    )
    sub_type: Optional[str] = Field(
        default=None,
        description="Sub-type for model_graded evals (e.g., 'summarization', 'rag')"
    )


class EvaluationMetadata(OpenAIBaseModel):
    """Metadata for an evaluation"""
    author: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    version: Optional[str] = None
    custom_fields: dict[str, Any] = Field(default_factory=dict)


class CreateEvaluationRequest(OpenAIBaseModel):
    """Request to create a new evaluation"""
    name: str = Field(..., description="Evaluation name")
    description: Optional[str] = Field(None, description="Evaluation description")
    eval_type: Literal["model_graded", "exact_match", "includes", "fuzzy_match", "custom"] = Field(
        ...,
        description="Type of evaluation"
    )
    eval_spec: EvaluationSpec = Field(..., description="Evaluation specification")
    dataset_id: Optional[str] = Field(None, description="Reference to existing dataset")
    dataset: Optional[list[dict[str, Any]]] = Field(
        None,
        description="Inline dataset (if not using dataset_id)"
    )
    metadata: Optional[EvaluationMetadata] = None

    @model_validator(mode='after')
    def validate_dataset(self):
        """Ensure either dataset_id or dataset is provided"""
        if not self.dataset_id and not self.dataset:
            raise ValueError("Either dataset_id or dataset must be provided")
        if self.dataset_id and self.dataset:
            raise ValueError("Only one of dataset_id or dataset should be provided")
        return self


class UpdateEvaluationRequest(OpenAIBaseModel):
    """Request to update an evaluation"""
    name: Optional[str] = None
    description: Optional[str] = None
    eval_spec: Optional[EvaluationSpec] = None
    dataset_id: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None  # Accept any dict for flexible updates


class EvaluationResponse(OpenAIBaseModel):
    """Response for a single evaluation"""
    id: str = Field(..., description="Evaluation ID")
    object: Literal["evaluation"] = "evaluation"
    created: int = Field(..., description="Unix timestamp of creation")
    name: str
    description: Optional[str] = None
    eval_type: str
    eval_spec: dict[str, Any]
    dataset_id: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ============= Run Models =============

class RunConfig(OpenAIBaseModel):
    """Configuration for an evaluation run"""
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_workers: int = Field(default=4, ge=1, le=16)
    timeout_seconds: int = Field(default=300, ge=1)
    batch_size: int = Field(default=10, ge=1, le=100)


class DatasetOverride(OpenAIBaseModel):
    """Override dataset for a specific run"""
    samples: list[dict[str, Any]] = Field(..., description="Evaluation samples")


class CreateRunRequest(OpenAIBaseModel):
    """Request to create and start an evaluation run"""
    target_model: Optional[str] = Field(
        None,
        description="Model being evaluated"
    )
    dataset_override: Optional[DatasetOverride] = Field(
        None,
        description="Override evaluation's dataset"
    )
    config: RunConfig = Field(
        default_factory=RunConfig,
        description="Run configuration"
    )
    webhook_url: Optional[str] = Field(
        None,
        description="URL for completion webhook"
    )


class RunProgress(OpenAIBaseModel):
    """Progress information for a running evaluation"""
    total_samples: int = 0
    completed_samples: int = 0
    failed_samples: int = 0
    current_batch: int = 0
    percent_complete: float = 0.0

    @model_validator(mode='after')
    def calculate_percent(self):
        """Calculate completion percentage"""
        if self.total_samples > 0:
            self.percent_complete = (self.completed_samples / self.total_samples) * 100
        return self


class RunResponse(OpenAIBaseModel):
    """Response for an evaluation run"""
    id: str = Field(..., description="Run ID")
    object: Literal["evaluation.run"] = "evaluation.run"
    created: int = Field(..., description="Unix timestamp")
    eval_id: str = Field(..., description="Parent evaluation ID")
    status: Literal["pending", "running", "completed", "failed", "cancelled"]
    target_model: Optional[str] = None
    progress: Optional[RunProgress] = None
    estimated_completion: Optional[int] = None
    error_message: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MetricStats(OpenAIBaseModel):
    """Statistics for a single metric"""
    mean: float
    std: float
    min: float
    max: float
    median: Optional[float] = None


class AggregateResults(OpenAIBaseModel):
    """Aggregate evaluation results"""
    mean_score: Optional[float] = Field(None, description="None when no sample was scored")
    std_dev: Optional[float] = None
    min_score: Optional[float] = None
    max_score: Optional[float] = None
    pass_rate: Optional[float] = Field(None, description="Share of scored samples passing threshold (None when none scored)")
    total_samples: int
    failed_samples: int


class SampleResult(OpenAIBaseModel):
    """Result for a single evaluation sample"""
    sample_id: str
    scores: dict[str, float]
    passed: bool
    error: Optional[str] = None


class UsageStats(OpenAIBaseModel):
    """Token usage statistics"""
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    cost_estimate: Optional[float] = None


class RunResultsResponse(OpenAIBaseModel):
    """Complete results for an evaluation run"""
    id: str
    object: Literal["evaluation.run.result"] = "evaluation.run.result"
    eval_id: str
    status: Literal["completed", "failed"]
    started_at: int
    completed_at: int
    results: dict[str, Any] = Field(
        ...,
        description="Contains aggregate, by_metric, and sample_results"
    )
    usage: Optional[UsageStats] = None
    duration_seconds: float


# ============= Dataset Models =============

class CreateDatasetRequest(OpenAIBaseModel):
    """Request to create a dataset"""
    name: str = Field(..., description="Dataset name")
    description: Optional[str] = None
    samples: list[dict[str, Any]] = Field(
        ...,
        description="Dataset samples"
    )
    metadata: Optional[dict[str, Any]] = None


class DatasetResponse(OpenAIBaseModel):
    """Response for a dataset"""
    id: str
    object: Literal["dataset"] = "dataset"
    created: int
    name: str
    description: Optional[str] = None
    sample_count: int
    samples: Optional[list[dict[str, Any]]] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ============= List Response Models =============

class ListResponse(OpenAIBaseModel):
    """Generic list response following OpenAI conventions"""
    object: Literal["list"] = "list"
    data: list[Union[EvaluationResponse, RunResponse, DatasetResponse]]
    has_more: bool = False
    first_id: Optional[str] = None
    last_id: Optional[str] = None


class EvaluationListResponse(ListResponse):
    """List of evaluations"""
    data: list[EvaluationResponse]


class RunListResponse(ListResponse):
    """List of runs"""
    data: list[RunResponse]


class DatasetListResponse(ListResponse):
    """List of datasets"""
    data: list[DatasetResponse]


# ============= Error Models =============

class ErrorDetail(OpenAIBaseModel):
    """Error detail following OpenAI format"""
    message: str
    type: Literal[
        "invalid_request_error",
        "authentication_error",
        "permission_error",
        "not_found_error",
        "rate_limit_error",
        "server_error"
    ]
    param: Optional[str] = None
    code: Optional[str] = None


class ErrorResponse(OpenAIBaseModel):
    """Error response following OpenAI format"""
    error: ErrorDetail


# ============= Query Parameters =============

class ListQueryParams(OpenAIBaseModel):
    """Common query parameters for list endpoints"""
    limit: int = Field(default=20, ge=1, le=100, description="Results per page")
    after: Optional[str] = Field(None, description="Cursor for pagination")
    order: Literal["asc", "desc"] = Field(default="desc", description="Sort order")


class RunListQueryParams(ListQueryParams):
    """Query parameters for listing runs"""
    status: Optional[Literal["pending", "running", "completed", "failed", "cancelled"]] = None


# ============= Webhook Models =============

class WebhookPayload(OpenAIBaseModel):
    """Webhook payload for run completion"""
    event: Literal["run.completed", "run.failed", "run.cancelled"]
    run_id: str
    eval_id: str
    status: str
    completed_at: int
    results_url: str
    summary: Optional[dict[str, Any]] = None
