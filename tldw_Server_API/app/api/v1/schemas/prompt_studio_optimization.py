# prompt_studio_optimization.py
# Optimization and job queue schemas for Prompt Studio

from datetime import datetime
from typing import Any, Optional

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from tldw_Server_API.app.core.Prompt_Management.optimization_model_config import (
    ACCEPTED_OPTIMIZATION_STRATEGIES,
    normalize_durable_optimization_config,
    normalize_optimization_strategy,
)

from .prompt_studio_base import (
    DEFAULT_MAX_TEST_CASES,
    JobStatus,
    JobType,
    TimestampMixin,
    UUIDMixin,
)

########################################################################################################################
# Optimization Schemas

DURABLE_OPTIMIZATION_STRATEGIES = ACCEPTED_OPTIMIZATION_STRATEGIES


class OptimizationTechnique(str):
    """Accepted optimization techniques."""

    MIPRO = "mipro"
    BOOTSTRAP = "bootstrap"
    ITERATIVE = "iterative"
    MCTS = "mcts"
    # Retain historical constants for import compatibility. Public request
    # schemas reject these unsupported strategies.
    HILL_CLIMBING = "hill_climbing"
    RANDOM_SEARCH = "random_search"
    GRID_SEARCH = "grid_search"
    BAYESIAN = "bayesian"
    BEAM_SEARCH = "beam_search"
    GREEDY = "greedy"
    SIMULATED_ANNEALING = "simulated_annealing"
    GENETIC = "genetic"
    HYPERPARAMETER = "hyperparameter"


class OptimizationConfig(BaseModel):
    """Configuration for optimization run"""
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    optimizer_type: str = Field(..., description="Type of optimizer to use")
    max_iterations: int = Field(default=50, ge=1, le=500, description="Maximum iterations")
    target_metric: str = Field(..., description="Metric to optimize")
    target_value: Optional[float] = Field(None, description="Target metric value to achieve")
    early_stopping: bool = Field(default=True, description="Enable early stopping")
    early_stopping_patience: int = Field(default=5, ge=1, description="Iterations without improvement before stopping")
    temperature_range: list[float] = Field(default=[0.0, 1.0], description="Temperature range to explore")
    techniques_to_try: list[str] = Field(default=["cot", "few_shot"], description="Prompt techniques to try")
    models_to_test: Optional[list[str]] = Field(None, description="Models to test during optimization")
    budget_limit: Optional[float] = Field(None, ge=0.0, description="Maximum budget in dollars")
    llm_model_config: dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("model_config", "model_configuration"),
        serialization_alias="model_config",
    )
    strategy_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional parameters for the selected strategy",
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_durable_config(cls, value: Any) -> dict[str, Any]:
        return normalize_durable_optimization_config(
            value,
            reject_sensitive=True,
        )

    @field_validator("optimizer_type")
    @classmethod
    def validate_optimizer_type(cls, value: str) -> str:
        """Normalize accepted strategies and reject unknown values."""

        return normalize_optimization_strategy(value)

    @field_validator('temperature_range')
    @classmethod
    def validate_temperature_range(cls, v):
        if len(v) != 2 or v[0] >= v[1]:
            raise ValueError("temperature_range must be [min, max] where min < max")
        if v[0] < 0.0 or v[1] > 2.0:
            raise ValueError("temperature values must be between 0.0 and 2.0")
        return v

class BootstrapConfig(BaseModel):
    """Configuration for bootstrapping examples"""
    num_samples: int = Field(default=50, ge=10, le=1000, description="Number of samples to bootstrap")
    selection_method: str = Field(default="diverse", description="Selection method for examples")
    quality_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Quality threshold for examples")
    max_examples_per_prompt: int = Field(default=5, ge=1, le=20, description="Max examples to include")

class OptimizationCreate(BaseModel):
    """Optimization creation request"""
    project_id: int
    initial_prompt_id: int
    optimization_config: OptimizationConfig
    bootstrap_config: Optional[BootstrapConfig] = None
    test_case_ids: Optional[list[int]] = Field(
        None,
        min_length=1,
        max_length=DEFAULT_MAX_TEST_CASES,
        description="Specific test cases to optimize against",
    )
    name: Optional[str] = Field(None, max_length=255, description="Optimization run name")
    description: Optional[str] = Field(None, max_length=1000, description="Optimization run description")

    @field_validator("test_case_ids")
    @classmethod
    def validate_unique_test_case_ids(
        cls,
        value: Optional[list[int]],
    ) -> Optional[list[int]]:
        if value is not None and len(value) != len(set(value)):
            raise ValueError("test_case_ids must be unique")
        return value


class OptimizationResponse(TimestampMixin, UUIDMixin):
    """Optimization response model"""
    id: int
    project_id: int
    initial_prompt_id: int
    optimized_prompt_id: Optional[int]
    optimizer_type: str
    optimization_config: dict[str, Any]
    initial_metrics: Optional[dict[str, Any]]
    final_metrics: Optional[dict[str, Any]]
    improvement_percentage: Optional[float]
    iterations_completed: Optional[int]
    max_iterations: int
    bootstrap_samples: Optional[int]
    status: str
    error_message: Optional[str]
    total_tokens: Optional[int]
    total_cost: Optional[float]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]

    model_config = ConfigDict(from_attributes=True)

class OptimizationStatusResponse(BaseModel):
    """Optimization status response"""
    optimization_id: int
    status: str
    progress: float = Field(ge=0.0, le=1.0, description="Progress percentage")
    current_iteration: int
    max_iterations: int
    current_best_metric: Optional[float] = None
    estimated_time_remaining: Optional[int] = Field(None, description="Estimated seconds remaining")
    current_step: Optional[str] = None

class OptimizationIteration(BaseModel):
    """Single iteration of optimization"""
    iteration_number: int
    prompt_variant: dict[str, Any]
    metrics: dict[str, float]
    tokens_used: int
    cost: float
    timestamp: datetime

class OptimizationHistory(BaseModel):
    """Optimization history"""
    optimization_id: int
    iterations: list[OptimizationIteration]
    best_iteration: int
    convergence_data: dict[str, Any]

########################################################################################################################
# Job Queue Schemas

class JobCreate(BaseModel):
    """Job creation request"""
    job_type: JobType
    entity_id: int = Field(..., description="ID of entity (evaluation, optimization, etc.)")
    priority: int = Field(default=5, ge=1, le=10, description="Job priority (1=lowest, 10=highest)")
    payload: dict[str, Any] = Field(..., description="Job-specific payload")
    max_retries: int = Field(default=3, ge=0, le=10, description="Maximum retry attempts")

class JobResponse(TimestampMixin, UUIDMixin):
    """Job response model"""
    id: int
    job_type: JobType
    entity_id: int
    priority: int
    status: JobStatus
    payload: dict[str, Any]
    result: Optional[dict[str, Any]]
    error_message: Optional[str]
    retry_count: int
    max_retries: int
    started_at: Optional[datetime]
    completed_at: Optional[datetime]

    model_config = ConfigDict(from_attributes=True)

class JobStatusResponse(BaseModel):
    """Job status response"""
    job_id: int
    status: JobStatus
    progress: Optional[float] = Field(None, ge=0.0, le=1.0)
    current_step: Optional[str] = None
    error_message: Optional[str] = None
    retry_count: int

class JobListResponse(BaseModel):
    """Job list response"""
    jobs: list[JobResponse]
    queued_count: int
    processing_count: int
    completed_count: int
    failed_count: int

class JobCancelRequest(BaseModel):
    """Job cancellation request"""
    reason: Optional[str] = Field(None, max_length=500, description="Cancellation reason")

########################################################################################################################
# Module Configuration Schemas

from enum import Enum


class ModuleType(str, Enum):
    """Available prompt modules"""
    CHAIN_OF_THOUGHT = "chain_of_thought"
    REACT = "react"
    PROGRAM_OF_THOUGHT = "program_of_thought"
    MULTI_CHAIN = "multi_chain"
    ROLE_PLAY = "role_play"
    STRUCTURED_OUTPUT = "structured_output"

class ModuleConfig(BaseModel):
    """Module configuration"""
    module_type: ModuleType
    enabled: bool = True
    parameters: dict[str, Any] = Field(default_factory=dict)
    priority: int = Field(default=5, ge=1, le=10)

class ModuleLibrary(BaseModel):
    """Available modules library"""
    modules: list[ModuleConfig]
    presets: dict[str, list[ModuleConfig]]

########################################################################################################################
# Cost Analysis Schemas

class CostEstimate(BaseModel):
    """Cost estimate for an operation"""
    estimated_tokens: int
    estimated_cost: float
    cost_breakdown: dict[str, float]
    model_pricing: dict[str, float]

class CostAnalysisRequest(BaseModel):
    """Request for cost analysis"""
    project_id: int
    prompt_id: Optional[int] = None
    test_case_ids: Optional[list[int]] = None
    model_name: str
    include_optimization: bool = Field(default=False)
    optimization_iterations: int = Field(default=50)

class CostAnalysisResponse(BaseModel):
    """Cost analysis response"""
    total_estimated_cost: float
    cost_per_test_case: float
    cost_per_optimization_iteration: Optional[float] = None
    recommendations: list[str]
    alternative_models: dict[str, float]
