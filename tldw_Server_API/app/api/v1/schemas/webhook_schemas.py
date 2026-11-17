"""
Webhook schemas for the Evaluations API.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class WebhookEventType(str, Enum):
    """Supported webhook event types."""
    EVALUATION_STARTED = "evaluation.started"
    EVALUATION_PROGRESS = "evaluation.progress"
    EVALUATION_COMPLETED = "evaluation.completed"
    EVALUATION_FAILED = "evaluation.failed"
    EVALUATION_CANCELLED = "evaluation.cancelled"
    BATCH_STARTED = "batch.started"
    BATCH_COMPLETED = "batch.completed"
    BATCH_FAILED = "batch.failed"


class WebhookRegistrationRequest(BaseModel):
    """Request to register a webhook."""
    url: HttpUrl = Field(..., description="Webhook endpoint URL")
    events: list[WebhookEventType] = Field(
        ...,
        description="List of events to subscribe to",
        min_length=1
    )
    secret: Optional[str] = Field(
        None,
        description="Optional secret for HMAC signature (generated if not provided)",
        min_length=32
    )
    retry_count: Optional[int] = Field(
        3,
        ge=0,
        le=10,
        description="Number of times to retry delivery when the receiver returns an error",
    )
    timeout_seconds: Optional[int] = Field(
        30,
        ge=1,
        le=300,
        description="HTTP timeout for webhook delivery attempts",
    )

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "url": "https://example.com/webhook",
            "events": ["evaluation.completed", "evaluation.failed"],
            "secret": None,
            "retry_count": 3,
            "timeout_seconds": 30,
        }
    })


class WebhookRegistrationResponse(BaseModel):
    """Response for webhook registration."""
    webhook_id: int = Field(..., description="Unique webhook identifier")
    url: str = Field(..., description="Webhook endpoint URL")
    events: list[str] = Field(..., description="Subscribed events")
    secret: str = Field(..., description="Webhook secret (shown once)")
    active: bool = Field(..., description="Whether webhook is active")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    retry_count: int = Field(3, ge=0, le=10, description="Configured retry attempts")
    timeout_seconds: int = Field(30, ge=1, le=300, description="Delivery timeout in seconds")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "webhook_id": 1,
            "url": "https://example.com/webhook",
            "events": ["evaluation.completed", "evaluation.failed"],
            "secret": "wh_secret_abc123...",
            "active": True,
            "created_at": "2024-01-18T12:00:00Z",
            "retry_count": 3,
            "timeout_seconds": 30,
        }
    })


class WebhookUpdateRequest(BaseModel):
    """Request to update a webhook."""
    events: Optional[list[WebhookEventType]] = Field(
        None,
        description="Updated list of events to subscribe to"
    )
    active: Optional[bool] = Field(
        None,
        description="Enable or disable webhook"
    )


class WebhookStatusResponse(BaseModel):
    """Webhook status information."""
    webhook_id: int = Field(..., description="Webhook identifier")
    url: str = Field(..., description="Webhook URL")
    events: list[str] = Field(..., description="Subscribed events")
    active: bool = Field(..., description="Whether webhook is active")
    retry_count: Optional[int] = Field(None, description="Configured retry attempts")
    timeout_seconds: Optional[int] = Field(None, description="Delivery timeout in seconds")
    statistics: dict[str, Any] = Field(
        ...,
        description="Delivery statistics"
    )
    last_delivery_at: Optional[datetime] = Field(
        None,
        description="Last delivery timestamp"
    )
    last_error: Optional[str] = Field(
        None,
        description="Last error message"
    )
    created_at: datetime = Field(..., description="Creation timestamp")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "webhook_id": 1,
            "url": "https://example.com/webhook",
            "events": ["evaluation.completed"],
            "active": True,
            "retry_count": 3,
            "timeout_seconds": 30,
            "statistics": {
                "total_deliveries": 100,
                "successful_deliveries": 98,
                "failed_deliveries": 2,
                "success_rate": 0.98
            },
            "last_delivery_at": "2024-01-18T11:30:00Z",
            "last_error": None,
            "created_at": "2024-01-01T00:00:00Z"
        }
    })


class WebhookTestRequest(BaseModel):
    """Request to test a webhook."""
    url: HttpUrl = Field(..., description="Webhook URL to test")


class WebhookTestResponse(BaseModel):
    """Response for webhook test."""
    success: bool = Field(..., description="Whether test was successful")
    status_code: Optional[int] = Field(None, description="HTTP status code")
    response_time_ms: Optional[int] = Field(None, description="Response time in milliseconds")
    response_body: Optional[str] = Field(None, description="Response body (truncated)")
    error: Optional[str] = Field(None, description="Error message if failed")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "success": True,
            "status_code": 200,
            "response_time_ms": 150,
            "response_body": '{"status": "ok"}',
            "error": None
        }
    })


class WebhookPayloadSchema(BaseModel):
    """Schema for webhook payload sent to endpoints."""
    event: str = Field(..., description="Event type")
    evaluation_id: str = Field(..., description="Evaluation identifier")
    timestamp: str = Field(..., description="ISO timestamp")
    data: dict[str, Any] = Field(..., description="Event data")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "event": "evaluation.completed",
            "evaluation_id": "eval_abc123",
            "timestamp": "2024-01-18T12:00:00Z",
            "data": {
                "score": 0.95,
                "model": "gpt-4",
                "evaluation_type": "geval",
                "processing_time": 2.5
            }
        }
    })


class RateLimitStatusResponse(BaseModel):
    """Rate limit status for a user."""
    user_id: str = Field(..., description="User identifier")
    tier: str = Field(..., description="User tier")
    limits: dict[str, Any] = Field(..., description="Current limits")
    usage: dict[str, Any] = Field(..., description="Current usage")
    remaining: dict[str, Any] = Field(..., description="Remaining allowance")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "user_id": "user_123",
            "tier": "premium",
            "limits": {
                "per_minute": {
                    "evaluations": 100,
                    "batch_evaluations": 20,
                    "burst_size": 25
                },
                "daily": {
                    "evaluations": 10000,
                    "tokens": 10000000
                }
            },
            "usage": {
                "today": {
                    "evaluations": 150,
                    "tokens": 50000
                }
            },
            "remaining": {
                "daily_evaluations": 9850,
                "daily_tokens": 9950000
            }
        }
    })
