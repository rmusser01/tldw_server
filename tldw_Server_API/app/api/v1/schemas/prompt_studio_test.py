# prompt_studio_test.py
# Test case and evaluation schemas for Prompt Studio

from datetime import datetime
from typing import Any, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from tldw_Server_API.app.core.LLM_Calls.provider_identity import (
    canonical_provider_name,
)

from .prompt_studio_base import EvaluationStatus, TimestampMixin, UUIDMixin

########################################################################################################################
# Test Case Schemas

class TestCaseBase(BaseModel):
    """Base test case model"""
    name: Optional[str] = Field(None, max_length=255, description="Test case name")
    description: Optional[str] = Field(None, max_length=1000, description="Test case description")
    inputs: dict[str, Any] = Field(..., description="Input data for test")
    expected_outputs: Optional[dict[str, Any]] = Field(None, description="Expected output data")
    tags: Optional[list[str]] = Field(None, description="Tags for categorization")
    is_golden: bool = Field(default=False, description="Golden test case flag")

class TestCaseCreate(TestCaseBase):
    """Test case creation request"""
    project_id: int = Field(..., description="Parent project ID")
    signature_id: Optional[int] = Field(None, description="Associated signature ID")

class TestCaseUpdate(BaseModel):
    """Test case update request"""
    name: Optional[str] = Field(None, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    inputs: Optional[dict[str, Any]] = None
    expected_outputs: Optional[dict[str, Any]] = None
    tags: Optional[list[str]] = None
    is_golden: Optional[bool] = None

class TestCaseResponse(TestCaseBase, TimestampMixin, UUIDMixin):
    """Test case response model"""
    id: int
    project_id: int
    signature_id: Optional[int]
    actual_outputs: Optional[dict[str, Any]] = None
    is_generated: bool = False

    model_config = ConfigDict(from_attributes=True)

class TestCaseBulkCreate(BaseModel):
    """Bulk test case creation request"""
    project_id: int
    signature_id: Optional[int] = None
    test_cases: list[TestCaseBase]

class TestCaseImportRequest(BaseModel):
    """Test case import request"""
    project_id: int
    format: str = Field(..., pattern="^(csv|json)$", description="Import format")
    data: str = Field(..., description="Base64 encoded file data or JSON string")
    signature_id: Optional[int] = None
    auto_generate_names: bool = Field(default=True, description="Auto-generate names if missing")

class TestCaseExportRequest(BaseModel):
    """Test case export request
    Note: project_id is optional when provided via path param.
    """
    project_id: Optional[int] = None
    format: str = Field(default="json", pattern="^(csv|json)$", description="Export format")
    include_golden_only: bool = Field(default=False, description="Export only golden test cases")
    tag_filter: Optional[list[str]] = Field(None, description="Filter by tags")

class TestCaseGenerateRequest(BaseModel):
    """Request to auto-generate test cases"""
    project_id: int
    prompt_id: Optional[int] = None
    signature_id: Optional[int] = None
    num_cases: int = Field(default=5, ge=1, le=50, description="Number of test cases to generate")
    generation_strategy: str = Field(default="diverse", description="Generation strategy")
    base_on_description: Optional[str] = Field(None, max_length=2000, description="Description to base generation on")


class RunTestCasesSimpleRequest(BaseModel):
    """Simple run request used by compatibility endpoint."""
    model_config = ConfigDict(extra='forbid')

    prompt_id: int
    test_case_ids: list[Union[int, str]] = Field(default_factory=list)
    provider: str = "openai"
    model: Optional[str] = None
    # Back-compat: allow ignored project_id
    project_id: Optional[int] = None

    @field_validator('test_case_ids', mode='before')
    @classmethod
    def _coerce_ids(cls, v):
        if v is None:
            return []
        if isinstance(v, list):
            out = []
            for t in v:
                try:
                    out.append(int(t))
                except (ValueError, TypeError):
                    raise ValueError(f"test_case_ids must contain only integers, got: {t}") from None
            return out
        return v

    @field_validator("provider")
    @classmethod
    def _normalize_provider(cls, value: str) -> str:
        provider = canonical_provider_name(value)
        if not provider:
            raise ValueError("provider must be a non-empty string")
        return provider

########################################################################################################################
# Test Run Schemas

class TestRunBase(BaseModel):
    """Base test run model"""
    model_name: str = Field(..., description="Model used for test")
    model_params: Optional[dict[str, Any]] = Field(None, description="Model parameters")
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    expected_outputs: Optional[dict[str, Any]] = None
    scores: Optional[dict[str, float]] = None
    execution_time_ms: Optional[int] = None
    tokens_used: Optional[int] = None
    cost_estimate: Optional[float] = None
    error_message: Optional[str] = None

class TestRunCreate(BaseModel):
    """Test run execution request"""
    project_id: int
    prompt_id: int
    test_case_id: int
    model_name: str
    model_params: Optional[dict[str, Any]] = None

class TestRunResponse(TestRunBase, UUIDMixin):
    """Test run response model"""
    id: int
    project_id: int
    prompt_id: int
    test_case_id: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

class BatchTestRequest(BaseModel):
    """Batch test execution request"""
    project_id: int
    prompt_id: int
    test_case_ids: list[int] = Field(..., min_length=1, description="Test cases to run")
    model_name: str
    model_params: Optional[dict[str, Any]] = None
    parallel_execution: bool = Field(default=True, description="Execute tests in parallel")
    stop_on_error: bool = Field(default=False, description="Stop batch on first error")

class BatchTestResponse(BaseModel):
    """Batch test execution response"""
    total: int
    successful: int
    failed: int
    test_runs: list[TestRunResponse]
    aggregate_metrics: dict[str, Any]

########################################################################################################################
# Evaluation Schemas

class EvaluationMetric(BaseModel):
    """Evaluation metric configuration"""
    name: str = Field(..., description="Metric name")
    type: str = Field(..., description="Metric type (accuracy, f1, custom, etc.)")
    config: Optional[dict[str, Any]] = Field(None, description="Metric-specific configuration")
    weight: float = Field(default=1.0, ge=0.0, le=1.0, description="Metric weight")

class EvaluationCreate(BaseModel):
    """Evaluation creation request"""
    project_id: int
    prompt_id: int
    name: Optional[str] = Field(None, max_length=255, description="Evaluation name")
    description: Optional[str] = Field(None, max_length=1000, description="Evaluation description")
    test_case_ids: list[int] = Field(..., min_length=1, description="Test cases to evaluate")
    model_configs: list[dict[str, Any]] = Field(..., min_length=1, description="Model configurations")
    metrics: Optional[list[EvaluationMetric]] = Field(None, description="Metrics to calculate")

class EvaluationResponse(TimestampMixin, UUIDMixin):
    """Evaluation response model"""
    id: int
    project_id: int
    prompt_id: int
    name: Optional[str]
    description: Optional[str]
    test_case_ids: list[int]
    test_run_ids: Optional[list[int]]
    aggregate_metrics: Optional[dict[str, Any]]
    model_configs: list[dict[str, Any]]
    total_tokens: Optional[int]
    total_cost: Optional[float]
    status: EvaluationStatus
    error_message: Optional[str]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]

    model_config = ConfigDict(from_attributes=True)

class EvaluationStatusResponse(BaseModel):
    """Evaluation status response"""
    evaluation_id: int
    status: EvaluationStatus
    progress: Optional[float] = Field(None, ge=0.0, le=1.0, description="Progress percentage")
    current_step: Optional[str] = None
    error_message: Optional[str] = None

class EvaluationCompareRequest(BaseModel):
    """Request to compare multiple evaluations"""
    evaluation_ids: list[int] = Field(..., min_length=2, max_length=10, description="Evaluations to compare")
    metrics_focus: Optional[list[str]] = Field(None, description="Specific metrics to focus on")

class EvaluationCompareResponse(BaseModel):
    """Evaluation comparison response"""
    evaluations: list[EvaluationResponse]
    comparison_matrix: dict[str, Any]
    best_performing: dict[str, int]  # metric -> evaluation_id
    statistical_significance: Optional[dict[str, Any]] = None

########################################################################################################################
# Score Schemas

class ScoreConfig(BaseModel):
    """Configuration for scoring"""
    scoring_method: str = Field(default="exact_match", description="Scoring method")
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Score threshold")
    custom_scorer: Optional[str] = Field(None, description="Custom scorer function name")

class ScoreResult(BaseModel):
    """Score result for a test run"""
    metric_name: str
    score: float
    passed: bool
    details: Optional[dict[str, Any]] = None
