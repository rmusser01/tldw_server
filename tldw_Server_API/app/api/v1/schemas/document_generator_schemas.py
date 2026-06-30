# document_generator_schemas.py
# Description: Pydantic schemas for Document Generator API endpoints
#
"""
Pydantic schemas for Document Generator functionality.

These schemas define the request and response models for the document generation
API endpoints, ensuring proper validation and serialization.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from tldw_Server_API.app.api.v1.schemas.pagination import OffsetPaginationMeta


class DocumentType(str, Enum):
    """Enumeration of supported document types."""
    TIMELINE = "timeline"
    STUDY_GUIDE = "study_guide"
    BRIEFING = "briefing"
    SUMMARY = "summary"
    QA = "q_and_a"
    MEETING_NOTES = "meeting_notes"


class GenerationStatus(str, Enum):
    """Status of document generation job."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PromptConfig(BaseModel):
    """Configuration for document generation prompts."""
    system_prompt: str = Field(..., min_length=1, max_length=5000, description="System prompt for LLM")
    user_prompt: str = Field(..., min_length=1, max_length=5000, description="User prompt template")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Generation temperature")
    max_tokens: int = Field(2000, ge=100, le=10000, description="Maximum tokens to generate")


class GenerateDocumentRequest(BaseModel):
    """Request schema for generating a document."""
    conversation_id: str = Field(..., min_length=1, description="ID of the conversation")
    document_type: DocumentType = Field(..., description="Type of document to generate")
    provider: str = Field(..., min_length=1, description="LLM provider name")
    model: str = Field(..., min_length=1, description="Model name")
    api_key: Optional[str] = Field(None, description="Explicit API key override for the provider")
    specific_message: Optional[str] = Field(None, max_length=10000, description="Specific message to focus on")
    custom_prompt: Optional[str] = Field(None, max_length=5000, description="Custom prompt override")
    stream: bool = Field(False, description="Whether to stream the response")
    async_generation: bool = Field(False, description="Whether to generate asynchronously")

    @field_validator("conversation_id", mode="before")
    @classmethod
    def normalize_conversation_id(cls, value: Any) -> str:
        """Accept UUID strings while continuing to tolerate historical integer IDs."""
        if isinstance(value, (int, float)):
            if int(value) <= 0:
                raise ValueError("conversation_id must be positive.")
            return str(int(value))
        if isinstance(value, str):
            cleaned = value.strip()
            if not cleaned:
                raise ValueError("conversation_id cannot be empty.")
            return cleaned
        raise ValueError("conversation_id must be a string or positive integer.")


class GenerateDocumentResponse(BaseModel):
    """Response schema for synchronous document generation."""
    document_id: int = Field(..., description="ID of the generated document")
    conversation_id: str = Field(..., description="Conversation ID")
    document_type: DocumentType = Field(..., description="Type of document generated")
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Generated document content")
    provider: str = Field(..., description="LLM provider used")
    model: str = Field(..., description="Model used")
    generation_time_ms: int = Field(..., description="Generation time in milliseconds")
    created_at: datetime = Field(..., description="Creation timestamp")


class AsyncGenerationResponse(BaseModel):
    """Response schema for async document generation."""
    job_id: str = Field(..., description="Job ID for tracking")
    status: GenerationStatus = Field(..., description="Current job status")
    conversation_id: str = Field(..., description="Conversation ID")
    document_type: DocumentType = Field(..., description="Type of document being generated")
    created_at: datetime = Field(..., description="Job creation timestamp")
    message: str = Field(..., description="Status message")


class JobStatusResponse(BaseModel):
    """Response schema for job status check."""
    job_id: str = Field(..., description="Job ID")
    conversation_id: str = Field(..., description="Conversation ID")
    document_type: DocumentType = Field(..., description="Type of document")
    status: GenerationStatus = Field(..., description="Current status")
    provider: str = Field(..., description="LLM provider")
    model: str = Field(..., description="Model name")
    result_content: Optional[str] = Field(None, description="Generated content (if completed)")
    error_message: Optional[str] = Field(None, description="Error message (if failed)")
    created_at: datetime = Field(..., description="Job creation time")
    started_at: Optional[datetime] = Field(None, description="Job start time")
    completed_at: Optional[datetime] = Field(None, description="Job completion time")
    progress_percentage: Optional[int] = Field(None, ge=0, le=100, description="Progress percentage")


class GeneratedDocument(BaseModel):
    """Schema for a generated document."""
    id: int = Field(..., description="Document ID")
    conversation_id: str = Field(..., description="Conversation ID")
    document_type: DocumentType = Field(..., description="Document type")
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Document content")
    provider: str = Field(..., description="LLM provider used")
    model: str = Field(..., description="Model used")
    generation_time_ms: int = Field(..., description="Generation time in milliseconds")
    token_count: Optional[int] = Field(None, description="Token count")
    created_at: datetime = Field(..., description="Creation timestamp")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    model_config = ConfigDict(from_attributes=True)


class DocumentListResponse(BaseModel):
    """Response schema for listing generated documents."""
    documents: list[GeneratedDocument] = Field(..., description="List of generated documents")
    total: int = Field(..., description="Total number of documents")
    pagination: OffsetPaginationMeta = Field(..., description="Canonical offset pagination metadata")
    conversation_id: Optional[str] = Field(None, description="Conversation ID if filtered")
    document_type: Optional[DocumentType] = Field(None, description="Document type if filtered")


class SavePromptConfigRequest(BaseModel):
    """Request schema for saving custom prompt configuration."""
    document_type: DocumentType = Field(..., description="Document type")
    system_prompt: str = Field(..., min_length=1, max_length=5000, description="System prompt")
    user_prompt: str = Field(..., min_length=1, max_length=5000, description="User prompt template")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Generation temperature")
    max_tokens: int = Field(2000, ge=100, le=10000, description="Maximum tokens")


class PromptConfigResponse(BaseModel):
    """Response schema for prompt configuration."""
    document_type: DocumentType = Field(..., description="Document type")
    system_prompt: str = Field(..., description="System prompt")
    user_prompt: str = Field(..., description="User prompt template")
    temperature: float = Field(..., description="Generation temperature")
    max_tokens: int = Field(..., description="Maximum tokens")
    is_custom: bool = Field(..., description="Whether this is a custom config")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp for custom configs")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")


class BulkGenerateRequest(BaseModel):
    """Request schema for bulk document generation."""
    conversation_ids: list[str] = Field(..., min_length=1, max_length=50, description="List of conversation IDs")
    document_types: list[DocumentType] = Field(..., min_length=1, description="Types of documents to generate")
    provider: str = Field(..., min_length=1, description="LLM provider name")
    model: str = Field(..., min_length=1, description="Model name")
    api_key: str = Field(..., min_length=1, description="API key for the provider")
    async_generation: bool = Field(True, description="Generate asynchronously (recommended)")

    @field_validator("conversation_ids", mode="before")
    @classmethod
    def normalize_conversation_ids(cls, values: Any) -> list[str]:
        """Normalize all conversation IDs to strings for downstream consistency."""
        if not isinstance(values, (list, tuple)):
            raise ValueError("conversation_ids must be a list of identifiers.")
        normalized: list[str] = []
        for item in values:
            if isinstance(item, (int, float)):
                if int(item) <= 0:
                    raise ValueError("conversation_ids must contain positive identifiers.")
                normalized.append(str(int(item)))
            elif isinstance(item, str):
                cleaned = item.strip()
                if not cleaned:
                    raise ValueError("conversation_ids cannot contain empty strings.")
                normalized.append(cleaned)
            else:
                raise ValueError("conversation_ids must only contain strings or integers.")
        return normalized


class BulkGenerateResponse(BaseModel):
    """Response schema for bulk generation."""
    total_jobs: int = Field(..., description="Total number of generation jobs created")
    job_ids: list[str] = Field(..., description="List of job IDs for tracking")
    estimated_time_seconds: Optional[int] = Field(None, description="Estimated completion time")
    message: str = Field(..., description="Status message")


class RegenerateDocumentRequest(BaseModel):
    """Request schema for regenerating a document."""
    document_id: int = Field(..., gt=0, description="ID of document to regenerate")
    provider: Optional[str] = Field(None, description="Override provider")
    model: Optional[str] = Field(None, description="Override model")
    api_key: Optional[str] = Field(None, description="API key if provider changed")
    custom_prompt: Optional[str] = Field(None, max_length=5000, description="Custom prompt override")


class DocumentTemplateBase(BaseModel):
    """Base schema for document templates."""
    name: str = Field(..., min_length=1, max_length=100, description="Template name")
    document_type: DocumentType = Field(..., description="Document type")
    description: Optional[str] = Field(None, max_length=500, description="Template description")
    system_prompt: str = Field(..., min_length=1, max_length=5000, description="System prompt")
    user_prompt: str = Field(..., min_length=1, max_length=5000, description="User prompt template")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Generation temperature")
    max_tokens: int = Field(2000, ge=100, le=10000, description="Maximum tokens")
    is_public: bool = Field(False, description="Whether template is public")


class DocumentTemplateCreate(DocumentTemplateBase):
    """Schema for creating a document template."""
    pass


class DocumentTemplateResponse(DocumentTemplateBase):
    """Response schema for document template."""
    id: int = Field(..., description="Template ID")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    usage_count: int = Field(0, description="Number of times used")

    model_config = ConfigDict(from_attributes=True)


class TemplateListResponse(BaseModel):
    """Response schema for listing templates."""
    templates: list[DocumentTemplateResponse] = Field(..., description="List of templates")
    total: int = Field(..., description="Total number of templates")
    document_type: Optional[DocumentType] = Field(None, description="Document type if filtered")


class GenerationStatistics(BaseModel):
    """Statistics for document generation."""
    total_documents: int = Field(..., description="Total documents generated")
    by_type: dict[str, int] = Field(..., description="Count by document type")
    by_provider: dict[str, int] = Field(..., description="Count by provider")
    average_generation_time_ms: float = Field(..., description="Average generation time")
    total_tokens_used: Optional[int] = Field(None, description="Total tokens used")
    last_generated: Optional[datetime] = Field(None, description="Last generation timestamp")
    most_used_model: Optional[str] = Field(None, description="Most frequently used model")


class ExportDocumentRequest(BaseModel):
    """Request schema for exporting a document."""
    document_id: int = Field(..., gt=0, description="Document ID to export")
    format: str = Field("markdown", pattern="^(markdown|html|pdf|docx|txt)$", description="Export format")
    include_metadata: bool = Field(True, description="Include metadata in export")


class ExportDocumentResponse(BaseModel):
    """Response schema for document export."""
    document_id: int = Field(..., description="Document ID")
    format: str = Field(..., description="Export format")
    content: Optional[str] = Field(None, description="Exported content (for text formats)")
    download_url: Optional[str] = Field(None, description="Download URL (for binary formats)")
    file_size_bytes: Optional[int] = Field(None, description="File size in bytes")
    expires_at: Optional[datetime] = Field(None, description="Download URL expiration")


# Error response schemas
class DocumentGeneratorError(BaseModel):
    """Error response for document generator operations."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[dict[str, Any]] = Field(None, description="Additional error details")


class QuotaExceededError(DocumentGeneratorError):
    """Quota exceeded error response."""
    error: str = Field("quota_exceeded", description="Error type")
    quota_limit: int = Field(..., description="Quota limit")
    quota_used: int = Field(..., description="Quota used")
    reset_at: Optional[datetime] = Field(None, description="When quota resets")


class ProviderError(DocumentGeneratorError):
    """Provider error response."""
    error: str = Field("provider_error", description="Error type")
    provider: str = Field(..., description="Provider name")
    status_code: Optional[int] = Field(None, description="HTTP status code from provider")
