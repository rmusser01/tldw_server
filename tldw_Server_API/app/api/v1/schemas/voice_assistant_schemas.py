# voice_assistant_schemas.py
# Pydantic schemas for Voice Assistant API endpoints
#
#######################################################################################################################
from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from tldw_Server_API.app.api.v1.schemas.pagination import OffsetPaginationMeta


# WebSocket Message Types
class WSMessageType(str, Enum):
    """WebSocket message types for voice assistant protocol."""
    # Client -> Server
    AUTH = "auth"
    CONFIG = "config"
    AUDIO = "audio"
    COMMIT = "commit"
    CANCEL = "cancel"
    TEXT = "text"
    WORKFLOW_SUBSCRIBE = "workflow_subscribe"
    WORKFLOW_CANCEL = "workflow_cancel"

    # Server -> Client
    AUTH_OK = "auth_ok"
    AUTH_ERROR = "auth_error"
    CONFIG_ACK = "config_ack"
    TRANSCRIPTION = "transcription"
    INTENT = "intent"
    ACTION_START = "action_start"
    ACTION_RESULT = "action_result"
    TTS_CHUNK = "tts_chunk"
    TTS_END = "tts_end"
    ERROR = "error"
    STATE_CHANGE = "state_change"
    WORKFLOW_PROGRESS = "workflow_progress"
    WORKFLOW_COMPLETE = "workflow_complete"


class VoiceAssistantState(str, Enum):
    """Voice assistant session states."""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    ERROR = "error"


class VoiceActionType(str, Enum):
    """Types of voice command actions."""
    MCP_TOOL = "mcp_tool"
    WORKFLOW = "workflow"
    CUSTOM = "custom"
    LLM_CHAT = "llm_chat"


# WebSocket Messages - Client to Server

class WSAuthMessage(BaseModel):
    """Authentication message from client."""
    type: Literal[WSMessageType.AUTH] = WSMessageType.AUTH
    token: str = Field(..., description="JWT token or API key")


class WSConfigMessage(BaseModel):
    """Configuration message from client."""
    type: Literal[WSMessageType.CONFIG] = WSMessageType.CONFIG
    stt_model: Optional[str] = Field(
        default=None,
        description="STT model to use (e.g., 'parakeet', 'canary')"
    )
    stt_language: Optional[str] = Field(
        default=None,
        description="Language code for STT (e.g., 'en')"
    )
    tts_provider: Optional[str] = Field(
        default=None,
        description="TTS provider (e.g., 'kokoro', 'openai')"
    )
    tts_model: Optional[str] = Field(
        default=None,
        description="TTS model identifier (e.g., 'kokoro', 'tts-1')"
    )
    tts_voice: Optional[str] = Field(
        default=None,
        description="TTS voice identifier"
    )
    tts_format: Literal["mp3", "opus", "wav", "pcm"] = Field(
        default="mp3",
        description="Audio format for TTS output"
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Existing session ID to resume"
    )
    sample_rate: int = Field(
        default=16000,
        description="Audio sample rate in Hz"
    )


class WSAudioMessage(BaseModel):
    """Audio data message from client."""
    type: Literal[WSMessageType.AUDIO] = WSMessageType.AUDIO
    data: str = Field(..., description="Base64-encoded PCM audio data")
    sequence: int = Field(default=0, description="Frame sequence number")


class WSCommitMessage(BaseModel):
    """Commit message indicating end of utterance."""
    type: Literal[WSMessageType.COMMIT] = WSMessageType.COMMIT


class WSCancelMessage(BaseModel):
    """Cancel current operation."""
    type: Literal[WSMessageType.CANCEL] = WSMessageType.CANCEL


class WSTextMessage(BaseModel):
    """Text input instead of audio (for testing/accessibility)."""
    type: Literal[WSMessageType.TEXT] = WSMessageType.TEXT
    text: str = Field(..., description="Text to process as if spoken")


# WebSocket Messages - Server to Client

class WSAuthOKMessage(BaseModel):
    """Authentication success response."""
    type: Literal[WSMessageType.AUTH_OK] = WSMessageType.AUTH_OK
    user_id: int = Field(..., description="Authenticated user ID")
    session_id: str = Field(..., description="Voice session ID")


class WSAuthErrorMessage(BaseModel):
    """Authentication error response."""
    type: Literal[WSMessageType.AUTH_ERROR] = WSMessageType.AUTH_ERROR
    error: str = Field(..., description="Error message")


class WSConfigAckMessage(BaseModel):
    """Configuration acknowledgment."""
    type: Literal[WSMessageType.CONFIG_ACK] = WSMessageType.CONFIG_ACK
    session_id: str = Field(..., description="Session ID")
    stt_model: str = Field(..., description="Active STT model")
    tts_provider: str = Field(..., description="Active TTS provider")
    tts_model: str = Field(..., description="Active TTS model")


class WSTranscriptionMessage(BaseModel):
    """Transcription result."""
    type: Literal[WSMessageType.TRANSCRIPTION] = WSMessageType.TRANSCRIPTION
    text: str = Field(..., description="Transcribed text")
    is_final: bool = Field(default=False, description="Whether this is the final transcription")
    confidence: Optional[float] = Field(default=None, description="Transcription confidence")


class WSIntentMessage(BaseModel):
    """Parsed intent result."""
    type: Literal[WSMessageType.INTENT] = WSMessageType.INTENT
    action_type: VoiceActionType = Field(..., description="Detected action type")
    command_name: Optional[str] = Field(default=None, description="Matched command name")
    entities: dict[str, Any] = Field(default_factory=dict, description="Extracted entities")
    confidence: float = Field(..., description="Intent confidence score")
    requires_confirmation: bool = Field(default=False, description="Whether confirmation is needed")


class WSActionStartMessage(BaseModel):
    """Action execution started."""
    type: Literal[WSMessageType.ACTION_START] = WSMessageType.ACTION_START
    action_type: VoiceActionType = Field(..., description="Action type being executed")
    action_name: Optional[str] = Field(default=None, description="Action/tool name")


class WSActionResultMessage(BaseModel):
    """Action execution result."""
    type: Literal[WSMessageType.ACTION_RESULT] = WSMessageType.ACTION_RESULT
    success: bool = Field(..., description="Whether action succeeded")
    action_type: VoiceActionType = Field(..., description="Action type")
    result_data: Optional[dict[str, Any]] = Field(default=None, description="Action result data")
    response_text: str = Field(..., description="Response text for TTS")
    execution_time_ms: float = Field(default=0.0, description="Execution time in milliseconds")


class WSTTSChunkMessage(BaseModel):
    """TTS audio chunk."""
    type: Literal[WSMessageType.TTS_CHUNK] = WSMessageType.TTS_CHUNK
    data: str = Field(..., description="Base64-encoded audio chunk")
    sequence: int = Field(..., description="Chunk sequence number")
    format: str = Field(default="mp3", description="Audio format")


class WSTTSEndMessage(BaseModel):
    """TTS stream ended."""
    type: Literal[WSMessageType.TTS_END] = WSMessageType.TTS_END
    total_chunks: int = Field(..., description="Total chunks sent")
    total_bytes: int = Field(..., description="Total audio bytes")
    duration_ms: Optional[float] = Field(default=None, description="Audio duration in ms")


class WSErrorMessage(BaseModel):
    """Error message."""
    type: Literal[WSMessageType.ERROR] = WSMessageType.ERROR
    error: str = Field(..., description="Error message")
    code: Optional[str] = Field(default=None, description="Error code")
    recoverable: bool = Field(default=True, description="Whether the session can continue")


class WSStateChangeMessage(BaseModel):
    """Session state change notification."""
    type: Literal[WSMessageType.STATE_CHANGE] = WSMessageType.STATE_CHANGE
    state: VoiceAssistantState = Field(..., description="New state")
    previous_state: Optional[VoiceAssistantState] = Field(default=None, description="Previous state")


# Workflow-related WebSocket Messages

class WSWorkflowSubscribeMessage(BaseModel):
    """Subscribe to workflow progress updates."""
    type: Literal[WSMessageType.WORKFLOW_SUBSCRIBE] = WSMessageType.WORKFLOW_SUBSCRIBE
    run_id: str = Field(..., description="Workflow run ID to subscribe to")


class WSWorkflowCancelMessage(BaseModel):
    """Cancel a running workflow."""
    type: Literal[WSMessageType.WORKFLOW_CANCEL] = WSMessageType.WORKFLOW_CANCEL
    run_id: str = Field(..., description="Workflow run ID to cancel")


class WSWorkflowProgressMessage(BaseModel):
    """Workflow progress update."""
    type: Literal[WSMessageType.WORKFLOW_PROGRESS] = WSMessageType.WORKFLOW_PROGRESS
    run_id: str = Field(..., description="Workflow run ID")
    event_type: str = Field(..., description="Type of progress event")
    message: Optional[str] = Field(default=None, description="Human-readable progress message")
    data: dict[str, Any] = Field(default_factory=dict, description="Event data")
    timestamp: float = Field(..., description="Event timestamp")


class WSWorkflowCompleteMessage(BaseModel):
    """Workflow completion notification."""
    type: Literal[WSMessageType.WORKFLOW_COMPLETE] = WSMessageType.WORKFLOW_COMPLETE
    run_id: str = Field(..., description="Workflow run ID")
    status: str = Field(..., description="Final status (succeeded, failed, cancelled)")
    outputs: Optional[dict[str, Any]] = Field(default=None, description="Workflow outputs")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    duration_ms: Optional[int] = Field(default=None, description="Total execution time")
    response_text: str = Field(..., description="TTS-friendly response text")


# REST API Schemas

class VoiceCommandRequest(BaseModel):
    """Request for processing a voice command via REST."""
    text: str = Field(..., description="Transcribed text to process")
    persona_id: Optional[str] = Field(default=None, description="Optional persona identifier for scoped commands")
    session_id: Optional[str] = Field(default=None, description="Session ID for context")
    include_tts: bool = Field(default=True, description="Whether to generate TTS audio")
    tts_provider: Optional[str] = Field(default=None, description="TTS provider override")
    tts_model: Optional[str] = Field(default=None, description="TTS model override")
    tts_voice: Optional[str] = Field(default=None, description="TTS voice override")
    tts_format: Literal["mp3", "opus", "wav", "pcm"] = Field(default="mp3")


class VoiceCommandResponse(BaseModel):
    """Response from processing a voice command."""
    session_id: str = Field(..., description="Session ID")
    success: bool = Field(..., description="Whether command was processed successfully")
    transcription: str = Field(..., description="Input text (echo)")
    intent: WSIntentMessage = Field(..., description="Parsed intent")
    action_result: WSActionResultMessage = Field(..., description="Action result")
    output_audio: Optional[str] = Field(default=None, description="Base64-encoded TTS audio")
    output_audio_format: Optional[str] = Field(default=None, description="Audio format")
    processing_time_ms: float = Field(..., description="Total processing time")


class VoiceCommandDefinition(BaseModel):
    """Definition for creating/updating a voice command."""
    persona_id: Optional[str] = Field(default=None, description="Persona owner ID")
    connection_id: Optional[str] = Field(default=None, description="Reusable connection reference")
    name: str = Field(..., description="Human-readable command name")
    phrases: list[str] = Field(..., min_length=1, description="Trigger phrases")
    action_type: VoiceActionType = Field(..., description="Action type")
    action_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Action configuration (tool name, workflow ID, etc.)"
    )
    priority: int = Field(default=0, ge=0, le=100, description="Command priority")
    enabled: bool = Field(default=True, description="Whether command is active")
    requires_confirmation: bool = Field(default=False, description="Require confirmation")
    description: Optional[str] = Field(default=None, description="Command description")


class VoiceCommandInfo(BaseModel):
    """Information about a registered voice command."""
    id: str = Field(..., description="Command ID")
    user_id: int = Field(..., description="Owner user ID (0 for system)")
    persona_id: Optional[str] = Field(default=None, description="Persona owner ID")
    connection_id: Optional[str] = Field(default=None, description="Reusable connection reference")
    connection_status: Optional[Literal["ok", "missing"]] = Field(
        default=None,
        description="Whether the referenced reusable connection currently resolves"
    )
    connection_name: Optional[str] = Field(
        default=None,
        description="Resolved connection name when available"
    )
    name: str = Field(..., description="Command name")
    phrases: list[str] = Field(..., description="Trigger phrases")
    action_type: VoiceActionType = Field(..., description="Action type")
    action_config: dict[str, Any] = Field(..., description="Action configuration")
    priority: int = Field(..., description="Priority")
    enabled: bool = Field(..., description="Whether enabled")
    requires_confirmation: bool = Field(..., description="Requires confirmation")
    description: Optional[str] = Field(default=None, description="Description")
    created_at: Optional[datetime] = Field(default=None, description="Creation time")


class VoiceCommandListResponse(BaseModel):
    """Response listing voice commands."""
    commands: list[VoiceCommandInfo] = Field(..., description="List of commands")
    total: int = Field(..., description="Total count")


class VoiceSessionInfo(BaseModel):
    """Information about a voice session."""
    session_id: str = Field(..., description="Session ID")
    user_id: int = Field(..., description="User ID")
    state: VoiceAssistantState = Field(..., description="Current state")
    created_at: datetime = Field(..., description="Creation time")
    last_activity: datetime = Field(..., description="Last activity time")
    turn_count: int = Field(..., description="Number of conversation turns")


class VoiceSessionListResponse(BaseModel):
    """Response listing voice sessions."""
    sessions: list[VoiceSessionInfo] = Field(..., description="Active sessions")
    total: int = Field(..., description="Total count")
    pagination: OffsetPaginationMeta = Field(..., description="Canonical offset pagination metadata")


# Analytics Schemas

class VoiceCommandToggleRequest(BaseModel):
    """Request to toggle a voice command enabled state."""
    enabled: bool = Field(..., description="Whether the command is enabled")


class VoiceCommandUsage(BaseModel):
    """Usage stats for a voice command."""
    command_id: str = Field(..., description="Command ID")
    command_name: Optional[str] = Field(default=None, description="Command name")
    total_invocations: int = Field(..., description="Total invocations")
    success_count: int = Field(..., description="Successful invocations")
    error_count: int = Field(..., description="Failed invocations")
    avg_response_time_ms: float = Field(..., description="Average response time (ms)")
    last_used: Optional[datetime] = Field(default=None, description="Last usage time")


class VoiceAnalyticsTopCommand(BaseModel):
    """Top command summary for analytics."""
    command_id: str = Field(..., description="Command ID")
    command_name: Optional[str] = Field(default=None, description="Command name")
    count: int = Field(..., description="Invocation count")


class VoiceAnalytics(BaseModel):
    """Daily voice analytics summary."""
    date: str = Field(..., description="Date (YYYY-MM-DD)")
    total_commands: int = Field(..., description="Total commands processed")
    unique_users: int = Field(..., description="Unique users")
    success_rate: float = Field(..., description="Success rate for the day")
    avg_response_time_ms: float = Field(..., description="Average response time (ms)")
    top_commands: list[VoiceAnalyticsTopCommand] = Field(
        default_factory=list,
        description="Top commands for the day",
    )


class VoiceAnalyticsSummary(BaseModel):
    """Aggregate voice analytics summary."""
    total_commands_processed: int = Field(..., description="Total commands processed")
    active_sessions: int = Field(..., description="Active sessions count")
    total_voice_commands: int = Field(..., description="Total voice commands")
    enabled_commands: int = Field(..., description="Enabled voice commands")
    success_rate: float = Field(..., description="Overall success rate")
    avg_response_time_ms: float = Field(..., description="Average response time (ms)")
    top_commands: list[VoiceCommandUsage] = Field(..., description="Top commands")
    usage_by_day: list[VoiceAnalytics] = Field(..., description="Usage metrics by day")


# Workflow REST Schemas

class WorkflowStatusResponse(BaseModel):
    """Response for workflow status query."""
    run_id: str = Field(..., description="Workflow run ID")
    status: str = Field(..., description="Current status")
    status_reason: Optional[str] = Field(default=None, description="Status explanation")
    started_at: Optional[datetime] = Field(default=None, description="Start time")
    ended_at: Optional[datetime] = Field(default=None, description="End time")
    duration_ms: Optional[int] = Field(default=None, description="Execution time")
    outputs: Optional[dict[str, Any]] = Field(default=None, description="Workflow outputs")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class WorkflowCancelResponse(BaseModel):
    """Response for workflow cancel request."""
    run_id: str = Field(..., description="Workflow run ID")
    cancelled: bool = Field(..., description="Whether cancellation succeeded")
    message: str = Field(..., description="Status message")


class VoiceWorkflowTemplateInfo(BaseModel):
    """Information about a voice workflow template."""
    template_id: str = Field(..., description="Template identifier")
    name: str = Field(..., description="Template name")
    description: Optional[str] = Field(default=None, description="Template description")
    voice_trigger: bool = Field(default=True, description="Whether voice-triggerable")
    steps_count: int = Field(..., description="Number of steps")


class VoiceWorkflowTemplateListResponse(BaseModel):
    """Response listing available voice workflow templates."""
    templates: list[VoiceWorkflowTemplateInfo] = Field(..., description="Available templates")
    total: int = Field(..., description="Total count")


# Dry-Run Validation Schemas

class VoiceCommandValidationStep(BaseModel):
    """Result for a single validation step in a dry-run check."""
    name: str = Field(..., description="Step name (e.g. 'config_schema', 'action_target')")
    passed: bool = Field(..., description="Whether this validation step passed")
    message: str = Field(..., description="Human-readable result description")
    details: Optional[dict[str, Any]] = Field(
        default=None,
        description="Extra details (e.g. available tools, missing fields)",
    )


class VoiceCommandValidationResponse(BaseModel):
    """Aggregate dry-run validation report for a voice command."""
    command_id: str = Field(..., description="Validated command ID")
    command_name: str = Field(..., description="Command name")
    action_type: VoiceActionType = Field(..., description="Configured action type")
    valid: bool = Field(..., description="True when every step passed")
    steps: list[VoiceCommandValidationStep] = Field(
        ..., description="Per-step validation results"
    )


#
# End of voice_assistant_schemas.py
#######################################################################################################################
