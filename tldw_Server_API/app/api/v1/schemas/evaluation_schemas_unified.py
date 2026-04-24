# evaluation_schemas_unified.py - Unified Pydantic schemas for evaluation endpoints
"""
Unified evaluation schemas combining OpenAI-compatible and tldw-specific schemas.

This module provides all request/response models for the unified evaluation API.
"""

import html
import re
from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional, Union

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator

from tldw_Server_API.app.core.Evaluations.run_state import normalize_run_status

try:
    import bleach
except Exception:
    bleach = None


# ============= Utility Functions =============

def sanitize_html_text(value: Optional[str]) -> Optional[str]:
    """Sanitize user-provided text to prevent injection attacks"""
    if value is None:
        return None

    v = value
    v = v.replace("\r", "\n")

    if bleach is not None:
        # Use bleach for proper HTML sanitization
        v = bleach.clean(
            v,
            tags=[],
            attributes={},
            protocols=[],
            strip=True,
            strip_comments=True,
        )
    else:
        # Fallback: More robust sanitization without bleach
        # First, decode any HTML entities to catch encoded attacks
        try:
            import html as html_module
            # Decode HTML entities multiple times to handle nested encoding
            for _ in range(3):
                decoded = html_module.unescape(v)
                if decoded == v:
                    break
                v = decoded
        except Exception as decode_error:
            logger.debug("Evaluation HTML sanitization decode fallback failed", exc_info=decode_error)

        # Remove all HTML tags and dangerous patterns more thoroughly
        # Remove script tags and their content (case insensitive, handles broken tags)
        v = re.sub(r'<\s*script[^>]*>.*?<\s*/\s*script\b[^>]*>', '', v, flags=re.IGNORECASE | re.DOTALL)
        v = re.sub(r'<\s*script[^>]*>', '', v, flags=re.IGNORECASE)

        # Remove style tags and their content
        v = re.sub(r'<\s*style[^>]*>.*?<\s*/\s*style\s*>', '', v, flags=re.IGNORECASE | re.DOTALL)
        v = re.sub(r'<\s*style[^>]*>', '', v, flags=re.IGNORECASE)

        # Remove all event handlers (onclick, onload, etc.)
        v = re.sub(r'\s*on\w+\s*=\s*["\']?[^"\'>\s]*["\']?', '', v, flags=re.IGNORECASE)

        # Remove javascript: protocol
        v = re.sub(r'javascript\s*:', '', v, flags=re.IGNORECASE)
        v = re.sub(r'data\s*:', '', v, flags=re.IGNORECASE)
        v = re.sub(r'vbscript\s*:', '', v, flags=re.IGNORECASE)

        # Remove all remaining HTML tags
        v = re.sub(r'<[^>]+>', '', v)

        # Finally, escape any remaining HTML special characters
        v = html.escape(v)

    # Remove null bytes and control characters
    v = v.replace('\x00', '')
    v = ''.join(ch for ch in v if ord(ch) >= 32 or ch in ('\n', '\t'))

    # Normalize whitespace
    v = re.sub(r'\n{3,}', '\n\n', v)
    v = re.sub(r' {2,}', ' ', v)

    return v.strip() if v is not None else None


# ============= Enums =============

class EvaluationType(str, Enum):
    """Supported evaluation types"""
    MODEL_GRADED = "model_graded"
    EXACT_MATCH = "exact_match"
    INCLUDES = "includes"
    FUZZY_MATCH = "fuzzy_match"
    GEVAL = "geval"
    RAG = "rag"
    RESPONSE_QUALITY = "response_quality"
    PROPOSITION_EXTRACTION = "proposition_extraction"
    OCR = "ocr"
    LABEL_CHOICE = "label_choice"  # Multi-class single-label selection
    NLI_FACTCHECK = "nli_factcheck"  # Claim verification via NLI-style labels
    CUSTOM = "custom"


class RunStatus(str, Enum):
    """Run status values"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WebhookEventType(str, Enum):
    """Webhook event types"""
    EVALUATION_STARTED = "evaluation.started"
    EVALUATION_PROGRESS = "evaluation.progress"
    EVALUATION_COMPLETED = "evaluation.completed"
    EVALUATION_FAILED = "evaluation.failed"
    EVALUATION_CANCELLED = "evaluation.cancelled"
    BATCH_STARTED = "batch.started"
    BATCH_COMPLETED = "batch.completed"
    BATCH_FAILED = "batch.failed"


# ============= Base Models =============

class EvaluationMetric(BaseModel):
    """Base evaluation metric"""
    name: str = Field(..., description="Metric name")
    score: float = Field(..., ge=0.0, le=1.0, description="Normalized score (0-1)")
    raw_score: Optional[float] = Field(None, description="Raw score before normalization")
    explanation: Optional[str] = Field(None, description="Explanation of the score")
    metadata: Optional[dict[str, Any]] = Field(default_factory=dict)


class EvaluationSpec(BaseModel):
    """Evaluation specification"""
    # Optional sub_type for model_graded evaluations
    sub_type: Optional[Literal['summarization', 'rag', 'response_quality', 'rag_pipeline']] = Field(
        default=None,
        description="Optional subtype for model_graded evaluations (e.g., rag_pipeline)"
    )
    metrics: Optional[list[str]] = Field(
        default=None,
        description="Metrics to compute"
    )
    threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Pass/fail threshold (legacy single-value form)"
    )
    thresholds: Optional[dict[str, float]] = Field(
        default=None,
        description="Pass/fail thresholds"
    )
    model: Optional[str] = Field(
        default="gpt-3.5-turbo",
        description="Model to use for evaluation"
    )
    temperature: Optional[float] = Field(
        default=0.3,
        ge=0.0,
        le=2.0,
        description="Temperature for model-based evaluation"
    )
    custom_prompts: Optional[dict[str, str]] = Field(
        default=None,
        description="Custom evaluation prompts"
    )

    # Optional nested spec for rag_pipeline evaluations
    rag_pipeline: Optional["RAGPipelineEvalSpec"] = Field(
        default=None,
        description="RAG pipeline evaluation configuration (when sub_type=rag_pipeline)"
    )

    # Optional fields for classification/label-choice and NLI fact-checking
    allowed_labels: Optional[list[str]] = Field(
        default=None,
        description="Allowed label set for label_choice/NLI evaluations"
    )
    label_mapping: Optional[dict[str, str]] = Field(
        default=None,
        description="Mapping of synonyms/aliases to canonical labels"
    )
    structured_output: Optional[bool] = Field(
        default=False,
        description="If true, instruct model to return strict JSON"
    )
    generate_predictions: Optional[bool] = Field(
        default=True,
        description="If true, call model to generate predictions; otherwise expect predictions in samples"
    )
    prompt_template: Optional[str] = Field(
        default=None,
        description="Override default prompt for label_choice/NLI"
    )
    nli_model: Optional[str] = Field(
        default=None,
        description="Optional NLI model identifier/provider name"
    )

    @model_validator(mode="after")
    def _validate_rag_pipeline(self) -> "EvaluationSpec":  # type: ignore[name-defined]
        """Ensure nested rag_pipeline spec exists when subtype requires it."""
        if self.sub_type == 'rag_pipeline' and self.rag_pipeline is None:
            raise ValueError("rag_pipeline subtype requires eval_spec.rag_pipeline configuration")
        return self


class RunConfig(BaseModel):
    """Run configuration"""
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    max_workers: Optional[int] = Field(4, ge=1, le=100)
    timeout_seconds: Optional[int] = Field(300, ge=1, le=3600)
    retry_attempts: Optional[int] = Field(3, ge=0, le=10)
    batch_size: Optional[int] = Field(10, ge=1, le=100)


class DatasetSample(BaseModel):
    """Dataset sample"""
    input: Any = Field(..., description="Input data")
    expected: Optional[Any] = Field(None, description="Expected output")
    metadata: Optional[dict[str, Any]] = Field(default_factory=dict)


class DatasetOverride(BaseModel):
    """Override dataset payload for a specific run."""
    samples: list[DatasetSample] = Field(..., min_length=1, description="Evaluation samples")
    name: Optional[str] = Field(None, description="Optional dataset name")
    description: Optional[str] = Field(None, description="Optional dataset description")
    metadata: Optional[dict[str, Any]] = Field(default_factory=dict)


class EvaluationMetadata(BaseModel):
    """Evaluation metadata"""
    project: Optional[str] = Field(None, description="Project name")
    version: Optional[str] = Field(None, description="Version")
    tags: Optional[list[str]] = Field(default_factory=list)
    custom: Optional[dict[str, Any]] = Field(default_factory=dict)


# ============= OpenAI-Compatible Schemas =============

class CreateEvaluationRequest(BaseModel):
    """Create evaluation request (OpenAI-compatible)"""
    name: str = Field(..., description="Unique evaluation name", min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    eval_type: EvaluationType = Field(..., description="Type of evaluation")
    eval_spec: EvaluationSpec = Field(..., description="Evaluation specification")
    dataset_id: Optional[str] = Field(None, description="Reference to existing dataset")
    dataset: Optional[list[DatasetSample]] = Field(None, description="Inline dataset")
    metadata: Optional[EvaluationMetadata] = Field(None)

    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError("Name must contain only alphanumeric characters, hyphens, and underscores")
        return v


class UpdateEvaluationRequest(BaseModel):
    """Update evaluation request"""
    description: Optional[str] = Field(None, max_length=1000)
    eval_spec: Optional[EvaluationSpec] = Field(None)
    metadata: Optional[EvaluationMetadata] = Field(None)


class EvaluationResponse(BaseModel):
    """Evaluation response (OpenAI-compatible)"""
    id: str = Field(..., description="Evaluation ID")
    object: str = Field(default="evaluation")
    name: str
    description: Optional[str] = None
    eval_type: str
    eval_spec: dict[str, Any]
    dataset_id: Optional[str] = None
    created: int = Field(..., description="Creation timestamp (OpenAI-compatible)")
    created_at: Optional[int] = Field(None, description="Creation timestamp (tldw-compatible)")
    created_by: str
    updated: Optional[int] = Field(None, description="Update timestamp (OpenAI-compatible)")
    updated_at: Optional[int] = Field(None, description="Update timestamp (tldw-compatible)")
    metadata: Optional[dict[str, Any]] = None

    model_config = ConfigDict(populate_by_name=True)


class CreateRunRequest(BaseModel):
    """Create run request"""
    target_model: str = Field(..., description="Model to evaluate")
    dataset_override: Optional[DatasetOverride] = Field(None)
    config: Optional[RunConfig] = Field(default_factory=RunConfig)
    webhook_url: Optional[HttpUrl] = Field(None)


class RunProgress(BaseModel):
    """Run progress information"""
    completed_samples: int = Field(0, ge=0)
    total_samples: int = Field(0, ge=0)
    percent_complete: float = Field(0.0, ge=0.0, le=100.0)
    current_sample: Optional[int] = None
    estimated_completion: Optional[int] = None


class RunResponse(BaseModel):
    """Run response"""
    id: str = Field(..., description="Run ID")
    object: str = Field(default="run")
    eval_id: str
    status: RunStatus
    target_model: str
    created: int = Field(..., description="Creation timestamp (OpenAI-compatible)")
    created_at: Optional[int] = Field(None, description="Creation timestamp (tldw-compatible)")
    started_at: Optional[int] = None
    completed_at: Optional[int] = None
    progress: Optional[RunProgress] = None
    error_message: Optional[str] = None
    results: Optional[dict[str, Any]] = None
    usage: Optional[dict[str, int]] = None

    model_config = ConfigDict(populate_by_name=True)

    @field_validator("status", mode="before")
    @classmethod
    def _normalize_status(cls, value: Any) -> RunStatus:
        return RunStatus(normalize_run_status(value))


class RunResultsResponse(BaseModel):
    """Run results response"""
    id: str
    eval_id: str
    status: RunStatus
    started_at: int
    completed_at: int
    results: dict[str, Any]
    usage: Optional[dict[str, int]] = None
    duration_seconds: float

    @field_validator("status", mode="before")
    @classmethod
    def _normalize_status(cls, value: Any) -> RunStatus:
        return RunStatus(normalize_run_status(value))


class CreateDatasetRequest(BaseModel):
    """Create dataset request"""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    samples: list[DatasetSample] = Field(..., min_length=1)
    metadata: Optional[dict[str, Any]] = None


class DatasetResponse(BaseModel):
    """Dataset response"""
    id: str
    object: str = Field(default="dataset")
    name: str
    description: Optional[str] = None
    sample_count: int
    samples: Optional[list[DatasetSample]] = Field(None, description="Dataset samples")
    created: int = Field(..., description="Creation timestamp (OpenAI-compatible)")
    created_at: Optional[int] = Field(None, description="Creation timestamp (tldw-compatible)")
    created_by: str
    metadata: Optional[dict[str, Any]] = None

    model_config = ConfigDict(populate_by_name=True)


# ============= OCR Evaluation Schemas =============

class OCREvaluationItem(BaseModel):
    id: str = Field(..., description="Document identifier")
    extracted_text: Optional[str] = Field(None, description="OCR-extracted text (if already computed)")
    ground_truth_text: str = Field(..., description="Ground-truth text for comparison")
    ground_truth_pages: Optional[list[str]] = Field(default=None, description="Per-page ground-truth texts (aligned to PDF pages)")
    metadata: Optional[dict[str, Any]] = Field(default_factory=dict)


class OCREvaluationRequest(BaseModel):
    items: list[OCREvaluationItem]
    metrics: Optional[list[Literal["cer", "wer", "coverage", "page_coverage"]]] = Field(
        default_factory=lambda: ["cer", "wer", "coverage", "page_coverage"],
        description="Metrics to compute"
    )
    ocr_options: Optional[dict[str, Any]] = Field(default=None, description="OCR configuration options")
    thresholds: Optional[dict[str, float]] = Field(default=None, description="Pass/fail thresholds: max_cer, max_wer, min_coverage, min_page_coverage")


class OCREvaluationResponse(BaseModel):
    evaluation_id: str
    results: dict[str, Any]
    evaluation_time: float


# ============= List Responses =============

class ListResponse(BaseModel):
    """Base list response"""
    object: str = Field(default="list")
    has_more: bool = Field(False)
    first_id: Optional[str] = None
    last_id: Optional[str] = None
    total: Optional[int] = Field(default=None, description="Total items available (if known)")


class EvaluationListResponse(ListResponse):
    """Evaluation list response"""
    data: list[EvaluationResponse]


class RunListResponse(ListResponse):
    """Run list response"""
    data: list[RunResponse]


class DatasetListResponse(ListResponse):
    """Dataset list response"""
    data: list[DatasetResponse]


# ============= tldw-Specific Evaluation Schemas =============

class GEvalRequest(BaseModel):
    """G-Eval summarization request"""
    source_text: str = Field(
        ...,
        description="Original source text",
        min_length=10,
        max_length=100000
    )
    summary: str = Field(
        ...,
        description="Summary to evaluate",
        min_length=10,
        max_length=50000
    )
    metrics: Optional[list[Literal["fluency", "consistency", "relevance", "coherence"]]] = Field(
        default=["fluency", "consistency", "relevance", "coherence"]
    )
    api_name: Optional[str] = Field("openai", description="LLM API to use")
    api_key: Optional[str] = Field(None, description="API key if not in config")
    save_results: bool = Field(False, description="Save results to database")

    @field_validator('source_text', 'summary')
    @classmethod
    def sanitize_text(cls, v: str) -> str:
        sanitized = sanitize_html_text(v)
        if not sanitized or len(sanitized.strip()) < 10:
            raise ValueError("Text too short after sanitization")
        return sanitized


class GEvalResponse(BaseModel):
    """G-Eval response"""
    metrics: dict[str, EvaluationMetric]
    average_score: float = Field(..., ge=0.0, le=1.0)
    summary_assessment: str
    evaluation_time: float
    metadata: Optional[dict[str, Any]] = None


class RAGEvaluationRequest(BaseModel):
    """RAG evaluation request"""
    query: str = Field(..., min_length=1, max_length=10000)
    retrieved_contexts: list[str] = Field(..., min_length=1, max_length=20)
    generated_response: str = Field(..., min_length=1, max_length=50000)
    ground_truth: Optional[str] = Field(None, max_length=50000)
    metrics: Optional[list[str]] = Field(
        default=["relevance", "faithfulness", "answer_similarity", "context_precision"]
    )
    api_name: Optional[str] = Field("openai")

    @field_validator('query', 'generated_response', 'ground_truth')
    @classmethod
    def sanitize_text_fields(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        return sanitize_html_text(v)


# ============= Proposition Extraction Schemas =============

class PropositionEvaluationRequest(BaseModel):
    """Evaluate proposition extraction quality"""
    extracted: list[str] = Field(..., min_length=1, description="Extracted propositions/claims")
    reference: list[str] = Field(..., min_length=1, description="Reference propositions/claims")
    method: Optional[Literal['semantic', 'jaccard']] = Field('semantic', description="Matching method")
    threshold: Optional[float] = Field(0.7, ge=0.0, le=1.0, description="Match threshold")

    @field_validator('extracted', 'reference')
    @classmethod
    def strip_items(cls, v: list[str]) -> list[str]:
        return [sanitize_html_text(x) or '' for x in v]


class PropositionEvaluationResponse(BaseModel):
    precision: float = Field(..., ge=0.0, le=1.0)
    recall: float = Field(..., ge=0.0, le=1.0)
    f1: float = Field(..., ge=0.0, le=1.0)
    matched: int
    total_extracted: int
    total_reference: int
    claim_density_per_100_tokens: float
    avg_prop_len_tokens: float
    dedup_rate: float
    details: Optional[dict[str, Any]] = None
    metadata: Optional[dict[str, Any]] = None


class RAGEvaluationResponse(BaseModel):
    """RAG evaluation response"""
    metrics: dict[str, EvaluationMetric]
    overall_score: float = Field(..., ge=0.0, le=1.0)
    retrieval_quality: float = Field(..., ge=0.0, le=1.0)
    generation_quality: float = Field(..., ge=0.0, le=1.0)
    suggestions: Optional[list[str]] = None
    metadata: Optional[dict[str, Any]] = None


class ResponseQualityRequest(BaseModel):
    """Response quality request"""
    prompt: str = Field(..., min_length=1, max_length=50000)
    response: str = Field(..., min_length=1, max_length=50000)
    expected_format: Optional[str] = Field(None, max_length=1000)
    evaluation_criteria: Optional[dict[str, str]] = None
    api_name: Optional[str] = Field("openai")

    @field_validator('prompt', 'response', 'expected_format')
    @classmethod
    def sanitize_text_fields(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        return sanitize_html_text(v)


class ResponseQualityResponse(BaseModel):
    """Response quality response"""
    metrics: dict[str, EvaluationMetric]
    overall_quality: float = Field(..., ge=0.0, le=1.0)
    format_compliance: Optional[dict[str, bool]] = None
    issues: Optional[list[str]] = None
    improvements: Optional[list[str]] = None

# ============= RAG Pipeline Evaluation Schemas =============

def _ensure_list(v):
    if v is None:
        return None
    if isinstance(v, list):
        return v
    return [v]


class ChunkingSweepConfig(BaseModel):
    """Retrieval-time chunking parameters to sweep via unified_rag_pipeline flags."""
    method: Optional[Union[str, list[str]]] = Field(
        default=None,
        description="Chunking method: structure_aware, sentences, markdown, code, xml"
    )
    chunk_size: Optional[Union[int, list[int]]] = Field(
        default=None, description="Chunk size(s)"
    )
    overlap: Optional[Union[int, list[int]]] = Field(
        default=None, description="Chunk overlap(s)"
    )
    structure_aware: Optional[Union[bool, list[bool]]] = Field(default=None)
    parent_expansion: Optional[Union[bool, list[bool]]] = Field(default=None)
    include_siblings: Optional[Union[bool, list[bool]]] = Field(default=None)
    chunk_type_filters: Optional[list[str]] = Field(
        default=None, description="Optional chunk type filters (e.g., ['code','text'])"
    )

    @field_validator('method', 'chunk_size', 'overlap', 'structure_aware', 'parent_expansion', 'include_siblings', mode='before')
    @classmethod
    def normalize_to_list(cls, v):
        return _ensure_list(v)


class RetrieverSweepConfig(BaseModel):
    """Retrieval parameters to sweep."""
    search_mode: Optional[Union[Literal["fts", "vector", "hybrid"], list[Literal["fts", "vector", "hybrid"]]]] = Field(default=None)
    # Remove ge/le constraints on Union[list] fields; validate ranges in pipeline executor if needed
    hybrid_alpha: Optional[Union[float, list[float]]] = Field(default=None)
    top_k: Optional[Union[int, list[int]]] = Field(default=None)
    min_score: Optional[Union[float, list[float]]] = Field(default=None)
    keyword_filter: Optional[list[str]] = Field(default=None)

    @field_validator('search_mode', 'hybrid_alpha', 'top_k', 'min_score', mode='before')
    @classmethod
    def normalize_to_list(cls, v):
        return _ensure_list(v)


class RerankerSweepConfig(BaseModel):
    """Reranker parameters to sweep."""
    strategy: Optional[Union[Literal["flashrank", "cross_encoder", "hybrid", "llama_cpp", "none"], list[Literal["flashrank", "cross_encoder", "hybrid", "llama_cpp", "none"]]]] = Field(default=None)
    top_k: Optional[Union[int, list[int]]] = Field(default=None)
    model: Optional[Union[str, list[str]]] = Field(default=None, description="Optional reranker model")

    @field_validator('strategy', 'top_k', 'model', mode='before')
    @classmethod
    def normalize_to_list(cls, v):
        return _ensure_list(v)


class GenerationSweepConfig(BaseModel):
    """Generation parameters to sweep for RAG answers."""
    model: Optional[Union[str, list[str]]] = Field(default=None)
    prompt_template: Optional[Union[str, list[str]]] = Field(default=None)
    temperature: Optional[Union[float, list[float]]] = Field(default=None)
    max_tokens: Optional[Union[int, list[int]]] = Field(default=None)

    @field_validator('model', 'prompt_template', 'temperature', 'max_tokens', mode='before')
    @classmethod
    def normalize_to_list(cls, v):
        return _ensure_list(v)


class PipelineMetricsSelection(BaseModel):
    """Select which metrics to compute at each step."""
    chunking_metrics: Optional[list[str]] = Field(default=None)
    retrieval_metrics: Optional[list[str]] = Field(default=None)
    generation_metrics: Optional[list[str]] = Field(default=None)


class RAGPipelineCaching(BaseModel):
    """Caching toggles for the pipeline evaluation."""
    cache_chunksets: Optional[bool] = Field(default=False)
    cache_embeddings: Optional[bool] = Field(default=True)
    cache_retrievals: Optional[bool] = Field(default=True)


class RAGPipelineEvalSpec(BaseModel):
    """Spec for evaluating RAG configurations via the unified RAG pipeline."""
    # Data
    dataset_id: Optional[str] = Field(default=None, description="Existing dataset id")
    dataset: Optional[list[DatasetSample]] = Field(default=None, description="Inline dataset samples")

    # Sweeps
    chunking: Optional[ChunkingSweepConfig] = None
    retrievers: Optional[list[RetrieverSweepConfig]] = None
    rerankers: Optional[list[RerankerSweepConfig]] = None
    rag: Optional[GenerationSweepConfig] = None

    # Execution
    search_strategy: Optional[Literal['grid', 'random']] = Field(default='grid')
    max_trials: Optional[int] = Field(default=None, ge=1, le=1000)
    metrics: Optional[PipelineMetricsSelection] = None
    caching: Optional[RAGPipelineCaching] = Field(default_factory=RAGPipelineCaching)
    concurrency: Optional[int] = Field(default=4, ge=1, le=64)
    timeout_seconds: Optional[int] = Field(default=300, ge=30, le=7200)
    index_namespace: Optional[str] = Field(default=None, description="Optional target index namespace (future use)")
    cleanup_collections: Optional[bool] = Field(default=False, description="Delete ephemeral collections after run")
    ephemeral_ttl_seconds: Optional[int] = Field(default=86400, ge=60, le=604800, description="TTL for ephemeral collections (seconds)")

    # Aggregation weights for leaderboard scoring
    aggregation_weights: Optional[dict[str, float]] = Field(
        default=None,
        description="Weights for combining metrics into config score (e.g., {'rag_overall':1.0,'retrieval_diversity':0.1,'chunk_cohesion':0.1})"
    )

    @model_validator(mode="after")
    def _validate_dataset(self) -> "RAGPipelineEvalSpec":
        if not self.dataset_id and not (self.dataset and len(self.dataset) > 0):
            raise ValueError("rag_pipeline requires either dataset_id or inline dataset samples")
        return self


# ============= Pipeline Presets & Cleanup Schemas =============

class PipelinePresetCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=128, pattern=r'^[a-zA-Z0-9_\-]+$')
    config: dict[str, Any] = Field(..., description="RAG pipeline config blocks: {chunking,retriever,reranker,rag}")


class PipelinePresetResponse(BaseModel):
    name: str
    config: dict[str, Any]
    created_at: Optional[int] = None
    updated_at: Optional[int] = None


class PipelinePresetListResponse(BaseModel):
    items: list[PipelinePresetResponse]
    total: int


class PipelineCleanupResponse(BaseModel):
    expired_count: int
    deleted_count: int
    errors: Optional[list[str]] = None

# ============= QA3 (Tri-Label) Evaluation Schemas =============

class QA3Item(BaseModel):
    id: Optional[str] = None
    question: str = Field(..., min_length=1)
    label: Optional[str] = Field(None, description="Gold label (e.g., SUPPORTED, REFUTED, NEI)")
    prediction: Optional[str] = Field(None, description="Optional precomputed prediction; if provided and generation disabled, used for scoring")
    context: Optional[str] = Field(None, description="Optional context to include in the prompt")

    @field_validator('question', 'context')
    @classmethod
    def sanitize_text_fields(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        return sanitize_html_text(v)


class QA3Request(BaseModel):
    items: list[QA3Item] = Field(..., min_length=1)
    allowed_labels: Optional[list[str]] = Field(default_factory=lambda: ["SUPPORTED","REFUTED","NEI"])
    label_mapping: Optional[dict[str, str]] = Field(None, description="Normalize gold labels, e.g., {'true':'SUPPORTED','false':'REFUTED'}")
    generate_predictions: Optional[bool] = Field(False, description="If true, call LLM to predict; else expect item.prediction and score-only")
    api_name: Optional[str] = Field("openai")
    temperature: Optional[float] = Field(0.0, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(3, ge=1, le=16)


class QA3PerLabel(BaseModel):
    precision: float
    recall: float
    f1: float
    support: int


class QA3Response(BaseModel):
    accuracy: float
    macro_f1: float
    per_label: dict[str, QA3PerLabel]
    confusion_matrix: dict[str, dict[str, int]]
    results: list[dict[str, Any]]
    metadata: Optional[dict[str, Any]] = None


class BatchEvaluationRequest(BaseModel):
    """Batch evaluation request"""
    evaluation_type: Literal["geval", "rag", "response_quality", "ocr", "propositions"]
    items: list[dict[str, Any]] = Field(..., min_length=1, max_length=1000)
    parallel_workers: int = Field(4, ge=1, le=20)
    continue_on_error: bool = Field(True)


class BatchEvaluationResponse(BaseModel):
    """Batch evaluation response"""
    total_items: int
    successful: int
    failed: int
    results: list[dict[str, Any]]
    aggregate_metrics: Optional[dict[str, float]] = None
    processing_time: float


class CustomMetricRequest(BaseModel):
    """Custom metric request"""
    name: str = Field(..., min_length=1, max_length=100)
    description: str = Field(..., max_length=1000)
    evaluation_prompt: str = Field(..., max_length=10000)
    input_data: dict[str, Any]
    scoring_criteria: dict[str, Any]
    api_name: Optional[str] = Field("openai")


class CustomMetricResponse(BaseModel):
    """Custom metric response"""
    metric_name: str
    score: float = Field(..., ge=0.0, le=1.0)
    explanation: str
    details: Optional[dict[str, Any]] = None


class EvaluationComparisonRequest(BaseModel):
    """Evaluation comparison request"""
    evaluation_ids: list[str] = Field(..., min_length=2, max_length=10)
    metrics_to_compare: Optional[list[str]] = None


class EvaluationComparisonResponse(BaseModel):
    """Evaluation comparison response"""
    evaluations: list[dict[str, Any]]
    metric_comparisons: dict[str, list[float]]
    improvements: Optional[dict[str, float]] = None
    regressions: Optional[dict[str, float]] = None
    summary: str


class EvaluationHistoryRequest(BaseModel):
    """Evaluation history request"""
    user_id: Optional[str] = None
    evaluation_type: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    limit: int = Field(20, ge=1, le=100)
    offset: int = Field(0, ge=0)


class EvaluationHistoryResponse(BaseModel):
    """Evaluation history response"""
    total_count: int
    items: list[dict[str, Any]]
    aggregations: Optional[dict[str, Any]] = None


# ============= Webhook Schemas =============

class WebhookRegistrationRequest(BaseModel):
    """Webhook registration request"""
    url: HttpUrl = Field(..., description="Webhook endpoint URL")
    events: list[WebhookEventType] = Field(..., min_length=1)
    secret: Optional[str] = Field(None, min_length=32)
    retry_count: Optional[int] = Field(
        3,
        ge=0,
        le=10,
        description="Number of retry attempts for failed webhook deliveries",
    )
    timeout_seconds: Optional[int] = Field(
        30,
        ge=1,
        le=300,
        description="Timeout in seconds for webhook delivery requests",
    )


class WebhookRegistrationResponse(BaseModel):
    """Webhook registration response"""
    webhook_id: int
    url: str
    events: list[str]
    secret: str
    created_at: datetime
    status: str = "active"
    retry_count: int = Field(3, ge=0, le=10)
    timeout_seconds: int = Field(30, ge=1, le=300)


class WebhookUpdateRequest(BaseModel):
    """Webhook update request"""
    url: Optional[HttpUrl] = None
    events: Optional[list[WebhookEventType]] = None
    status: Optional[Literal["active", "paused"]] = None


class WebhookStatusResponse(BaseModel):
    """Webhook status response"""
    webhook_id: int
    url: str
    events: list[str]
    status: str
    retry_count: Optional[int] = None
    timeout_seconds: Optional[int] = None
    created_at: datetime
    last_triggered: Optional[datetime] = None
    failure_count: int = 0


class WebhookTestRequest(BaseModel):
    """Webhook test request"""
    url: HttpUrl


class WebhookTestResponse(BaseModel):
    """Webhook test response"""
    success: bool
    status_code: Optional[int] = None
    response_time_ms: Optional[float] = None
    error: Optional[str] = None


# ============= Rate Limiting Schemas =============

class RateLimitStatusResponse(BaseModel):
    """Rate limit status response"""
    tier: str
    limits: dict[str, int]
    usage: dict[str, int]
    remaining: dict[str, int]
    reset_at: datetime


# ============= Error Response =============

class ErrorDetail(BaseModel):
    """Error detail"""
    message: str
    type: str = "error"
    param: Optional[str] = None
    code: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error response"""
    error: ErrorDetail


# ============= Health Check =============

class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: Literal["healthy", "degraded", "unhealthy"]
    version: str
    uptime: float
    database: str
    circuit_breaker: Optional[str] = None
    rate_limit: Optional[dict[str, Any]] = None
    checks: Optional[dict[str, bool]] = None
