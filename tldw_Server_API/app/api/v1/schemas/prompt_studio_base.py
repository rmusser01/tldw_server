# prompt_studio_base.py
# Base schemas for Prompt Studio feature

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator
from tldw_Server_API.app.api.v1.schemas.pagination import PagePaginationMeta

########################################################################################################################
# Enums

class ProjectStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    ARCHIVED = "archived"

class JobType(str, Enum):
    EVALUATION = "evaluation"
    OPTIMIZATION = "optimization"
    GENERATION = "generation"

class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class EvaluationStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

########################################################################################################################
# Base Models

class TimestampMixin(BaseModel):
    """Mixin for timestamp fields"""
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)

class SoftDeleteMixin(BaseModel):
    """Mixin for soft delete fields"""
    deleted: Optional[bool] = Field(default=False)
    deleted_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)

class UUIDMixin(BaseModel):
    """Mixin for UUID field"""
    uuid: Optional[UUID] = None

    model_config = ConfigDict(from_attributes=True)

########################################################################################################################
# Field Definitions for Signatures

class FieldDefinition(BaseModel):
    """Definition of an input or output field in a signature"""
    name: str = Field(..., min_length=1, max_length=100, description="Field name")
    type: str = Field(..., description="Field type (string, integer, array, etc.)")
    description: Optional[str] = Field(None, max_length=500, description="Field description")
    required: bool = Field(default=True, description="Whether field is required")
    default: Optional[Any] = Field(None, description="Default value if not required")
    constraints: Optional[dict[str, Any]] = Field(None, description="Additional constraints")

class ConstraintDefinition(BaseModel):
    """Constraint definition for signatures"""
    type: str = Field(..., description="Constraint type")
    field: str = Field(..., description="Field to apply constraint to")
    value: Any = Field(..., description="Constraint value")
    message: Optional[str] = Field(None, description="Error message if constraint violated")

########################################################################################################################
# Standard Response Models

class StandardResponse(BaseModel):
    """Standard API response wrapper"""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    error_code: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None

class PaginationMetadata(BaseModel):
    """Pagination metadata for list responses"""
    page: int = Field(ge=1)
    per_page: int = Field(ge=1, le=100)
    total: int = Field(ge=0)
    total_pages: int = Field(ge=0)

class ListResponse(StandardResponse):
    """Standard list response with pagination"""
    data: list[Any]
    metadata: PaginationMetadata


class PageListResponse(ListResponse):
    """Prompt Studio list response with canonical page pagination metadata."""

    pagination: PagePaginationMeta


class PageStandardResponse(StandardResponse):
    """Standard Prompt Studio response with canonical page pagination metadata."""

    pagination: PagePaginationMeta

########################################################################################################################
# Security Shim

class SecurityConfig(BaseModel):
    """Security configuration for prompt operations"""
    max_prompt_length: int = Field(default=50000, description="Maximum prompt length in characters")
    max_test_cases: int = Field(default=1000, description="Maximum test cases per project")
    max_concurrent_jobs: int = Field(default=10, description="Maximum concurrent jobs per user")
    enable_prompt_validation: bool = Field(default=True, description="Enable prompt validation")
    enable_rate_limiting: bool = Field(default=True, description="Enable rate limiting")

    @field_validator('max_prompt_length')
    @classmethod
    def validate_prompt_length(cls, v):
        if v < 100 or v > 1000000:
            raise ValueError("max_prompt_length must be between 100 and 1000000")
        return v

    @field_validator('max_test_cases')
    @classmethod
    def validate_test_cases(cls, v):
        if v < 1 or v > 10000:
            raise ValueError("max_test_cases must be between 1 and 10000")
        return v

########################################################################################################################
# Common Query Parameters

class ListQueryParams(BaseModel):
    """Common query parameters for list endpoints"""
    page: int = Field(default=1, ge=1, description="Page number")
    per_page: int = Field(default=20, ge=1, le=100, description="Items per page")
    include_deleted: bool = Field(default=False, description="Include soft-deleted items")
    sort_by: str = Field(default="updated_at", description="Field to sort by")
    sort_order: str = Field(default="desc", pattern="^(asc|desc)$", description="Sort order")
    search: Optional[str] = Field(None, max_length=200, description="Search query")

########################################################################################################################
# Base Error Models

class ValidationError(BaseModel):
    """Validation error detail"""
    field: str
    message: str
    code: Optional[str] = None

class ErrorResponse(BaseModel):
    """Error response model"""
    success: bool = False
    error: str
    error_code: Optional[str] = None
    details: Optional[list[ValidationError]] = None
    request_id: Optional[str] = None
