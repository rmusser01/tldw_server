from __future__ import annotations

from enum import Enum
from typing import Any, ClassVar, Literal, Union

from pydantic import BaseModel, Field, model_validator
from tldw_Server_API.app.api.v1.schemas.pagination import OffsetPaginationMeta


def _default_offset_pagination_aliases(response):
    if response.has_more is None:
        response.has_more = response.pagination.has_more
    if response.next_offset is None:
        response.next_offset = response.pagination.next_offset
    return response


# -----------------------------------------------------------------------------
# Agent Types
# -----------------------------------------------------------------------------


# Agent type identifiers are user-configurable strings.
ACPAgentType = str
ACP_COMPATIBILITY_DOCS_URL = "/docs-static/Development/ACP_Compatibility_Matrix.md"
ACPSupportState = Literal[
    "supported",
    "supported_with_caveats",
    "experimental",
    "documented_unverified",
    "unsupported",
]
ACPVerificationLevel = Literal[
    "documented_only",
    "stub_smoke_tested",
    "live_e2e_tested",
    "sandbox_tested",
    "production_supported",
]
ACPEntryPointStrategy = Literal[
    "native_acp",
    "adapter_acp",
    "documented_candidate",
    "custom_template",
]
ACPProbeState = Literal["ready_to_probe", "blocked", "custom_template", "documented_only"]


class ACPAgentEntrypointStatus(BaseModel):
    """ACP stdio entrypoint readiness metadata for one downstream agent."""
    profile_key: str = Field(..., description="Registry profile key this metadata describes")
    entrypoint_strategy: ACPEntryPointStrategy = Field(default="documented_candidate")
    probe_state: ACPProbeState = Field(default="documented_only")
    acp_command: str = Field(default="")
    acp_args: list[str] = Field(default_factory=list)
    primary_blocker: str | None = Field(default=None)
    blockers: list[str] = Field(default_factory=list)
    status_message: str = Field(default="")
    docs_url: str | None = Field(default=ACP_COMPATIBILITY_DOCS_URL)


class ACPAgentInfo(BaseModel):
    """Information about an available agent."""
    type: ACPAgentType = Field(..., description="Agent type identifier")
    name: str = Field(..., description="Human-readable agent name")
    description: str = Field(default="", description="Agent description")
    is_configured: bool = Field(
        default=False, description="Whether the agent is properly configured"
    )
    requires_api_key: str | None = Field(
        default=None,
        description="Name of required API key if not configured (e.g., 'ANTHROPIC_API_KEY')",
    )
    support_state: ACPSupportState = Field(
        default="documented_unverified",
        description="Compatibility support state from the ACP compatibility matrix.",
    )
    verification_level: ACPVerificationLevel = Field(
        default="documented_only",
        description="Evidence level behind the compatibility support state.",
    )
    compatibility_notes: str = Field(
        default="Configured locally; live-agent ACP compatibility has not been certified.",
        description="Human-readable compatibility caveat or certification note.",
    )
    compatibility_docs_url: str | None = Field(
        default=ACP_COMPATIBILITY_DOCS_URL,
        description="Documentation path or URL for compatibility details.",
    )
    entrypoint: ACPAgentEntrypointStatus = Field(
        ...,
        description="ACP stdio entrypoint readiness metadata.",
    )


class ACPAgentListResponse(BaseModel):
    """Response for listing available agents."""
    agents: list[ACPAgentInfo] = Field(default_factory=list)
    default_agent: ACPAgentType = Field(default="custom")


class ACPAgentCompatibilityStatus(BaseModel):
    """Compatibility status for an ACP downstream agent."""
    support_state: ACPSupportState = Field(
        default="documented_unverified",
        description="Compatibility support state from the ACP compatibility matrix.",
    )
    verification_level: ACPVerificationLevel = Field(
        default="documented_only",
        description="Evidence level behind the compatibility support state.",
    )
    notes: str = Field(
        default="Configured locally; live-agent ACP compatibility has not been certified.",
        description="Human-readable compatibility caveat or certification note.",
    )
    docs_url: str = Field(
        default=ACP_COMPATIBILITY_DOCS_URL,
        description="Served documentation URL for compatibility details.",
    )


class ACPSetupGuideComponent(BaseModel):
    """Setup guide entry for a non-agent ACP component."""
    component: str = Field(..., description="Component identifier")
    status: str = Field(default="unknown", description="Runtime setup status")
    steps: list[str] = Field(default_factory=list, description="Actionable setup steps")


class ACPSetupGuideAgent(BaseModel):
    """Setup guide entry for one ACP downstream agent."""
    agent_type: ACPAgentType = Field(..., description="Agent type identifier")
    name: str = Field(..., description="Human-readable agent name")
    status: str = Field(default="unknown", description="Runtime setup status")
    compatibility: ACPAgentCompatibilityStatus = Field(
        default_factory=ACPAgentCompatibilityStatus,
        description="Compatibility support status and evidence level.",
    )
    steps: list[str] = Field(default_factory=list, description="Actionable setup or certification steps")
    docs_url: str | None = Field(default=None, description="Agent documentation URL")
    entrypoint: ACPAgentEntrypointStatus = Field(
        ...,
        description="ACP stdio entrypoint readiness metadata.",
    )


class ACPSetupGuideResponse(BaseModel):
    """Response for ACP setup-guide diagnostics."""
    timestamp: str = Field(..., description="ISO 8601 timestamp")
    runner: ACPSetupGuideComponent = Field(..., description="Runner setup guidance")
    guides: list[ACPSetupGuideAgent] = Field(default_factory=list, description="Agent setup guidance")


class ACPAgentRegisterRequest(BaseModel):
    """Request to register a new agent type."""
    agent_type: str = Field(..., description="Unique agent type identifier")
    name: str = Field(..., description="Human-readable agent name")
    description: str = Field(default="", description="Agent description")
    command: str = Field(default="", description="Agent CLI command")
    args: list[str] = Field(default_factory=list, description="Command arguments")
    env: dict[str, str] = Field(default_factory=dict, description="Environment variables")
    requires_api_key: str | None = Field(default=None, description="Required API key env var")
    install_instructions: list[str] = Field(default_factory=list, description="Installation steps")
    docs_url: str | None = Field(default=None, description="Documentation URL")
    entrypoint_strategy: ACPEntryPointStrategy = Field(default="documented_candidate")
    acp_command: str = Field(default="")
    acp_args: list[str] = Field(default_factory=list)
    adapter_source: str | None = Field(default=None)
    adapter_docs_url: str | None = Field(default=None)
    certification_blocker: str | None = Field(default=None)
    mcp_orchestration: Literal["agent_driven", "llm_driven"] = Field(
        default="agent_driven",
        description="MCP orchestration mode when protocol='mcp'",
    )
    mcp_entry_tool: str = Field(default="execute", description="Entry tool for agent-driven MCP agents")
    mcp_structured_response: bool = Field(
        default=False,
        description="Whether the entry tool returns structured JSON steps",
    )
    mcp_llm_provider: str | None = Field(default=None, description="Provider for llm_driven MCP orchestration")
    mcp_llm_model: str | None = Field(default=None, description="Model for llm_driven MCP orchestration")
    mcp_max_iterations: int = Field(default=20, description="Maximum llm_driven MCP iterations")
    mcp_refresh_tools: bool = Field(
        default=False,
        description="Refresh MCP tool inventory before each prompt",
    )


class ACPAgentUpdateRequest(BaseModel):
    """Request to update an existing agent."""
    NON_NULLABLE_FIELDS: ClassVar[frozenset[str]] = frozenset({
        "name",
        "description",
        "command",
        "entrypoint_strategy",
        "acp_command",
        "mcp_orchestration",
        "mcp_entry_tool",
        "mcp_structured_response",
        "mcp_max_iterations",
        "mcp_refresh_tools",
    })

    name: str | None = None
    description: str | None = None
    command: str | None = None
    args: list[str] | None = None
    env: dict[str, str] | None = None
    requires_api_key: str | None = None
    install_instructions: list[str] | None = None
    docs_url: str | None = None
    entrypoint_strategy: ACPEntryPointStrategy | None = None
    acp_command: str | None = None
    acp_args: list[str] | None = None
    adapter_source: str | None = None
    adapter_docs_url: str | None = None
    certification_blocker: str | None = None
    mcp_orchestration: Literal["agent_driven", "llm_driven"] | None = None
    mcp_entry_tool: str | None = None
    mcp_structured_response: bool | None = None
    mcp_llm_provider: str | None = None
    mcp_llm_model: str | None = None
    mcp_max_iterations: int | None = None
    mcp_refresh_tools: bool | None = None

    @model_validator(mode="before")
    @classmethod
    def reject_null_for_non_nullable_fields(cls, data: Any) -> Any:
        """Reject explicit null for scalar fields that are non-null at runtime."""
        if not isinstance(data, dict):
            return data
        null_fields = sorted(
            field for field in cls.NON_NULLABLE_FIELDS
            if field in data and data[field] is None
        )
        if null_fields:
            joined = ", ".join(null_fields)
            raise ValueError(f"Explicit null is not allowed for: {joined}")
        return data


# -----------------------------------------------------------------------------
# Agent Health
# -----------------------------------------------------------------------------


class ACPAgentRegistrationResponse(BaseModel):
    """Response for agent registration/update/deregistration."""
    status: str = Field(..., description="Operation result: registered, updated, deregistered")
    agent_type: str = Field(..., description="Agent type identifier")
    name: str | None = Field(default=None, description="Agent name (if applicable)")


class ACPAgentHealthEntry(BaseModel):
    """Health status for a single agent."""
    agent_type: str = Field(..., description="Agent type identifier")
    health: str = Field(..., description="Health state: healthy, degraded, unavailable, unknown")
    consecutive_failures: int = Field(default=0, description="Number of consecutive check failures")
    last_check: str | None = Field(default=None, description="ISO timestamp of last health check")
    last_healthy: str | None = Field(default=None, description="ISO timestamp of last healthy check")
    details: dict[str, Any] = Field(default_factory=dict, description="Raw availability check details")


class ACPAgentHealthResponse(BaseModel):
    """Response for agent health status."""
    agents: list[ACPAgentHealthEntry] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# MCP Server Configuration
# -----------------------------------------------------------------------------


class ACPMCPServerType(str, Enum):
    """MCP server connection types."""
    WEBSOCKET = "websocket"
    STDIO = "stdio"


class ACPMCPServerConfig(BaseModel):
    """Configuration for an MCP server."""
    name: str = Field(..., description="Server identifier/name")
    type: ACPMCPServerType = Field(..., description="Connection type")
    url: str | None = Field(default=None, description="WebSocket URL (for websocket type)")
    command: str | None = Field(default=None, description="Command to execute (for stdio type)")
    args: list[str] | None = Field(default=None, description="Command arguments (for stdio type)")
    env: dict[str, str] | None = Field(default=None, description="Environment variables")


# -----------------------------------------------------------------------------
# Permission Tiers
# -----------------------------------------------------------------------------


class ACPPermissionTier(str, Enum):
    """Permission tier for tool execution approval.

    - auto: Automatically approved (read-only operations)
    - batch: Can be approved in batches (write operations)
    - individual: Requires individual approval (destructive/execute operations)
    """
    AUTO = "auto"
    BATCH = "batch"
    INDIVIDUAL = "individual"


# -----------------------------------------------------------------------------
# WebSocket Message Types (Server → Client)
# -----------------------------------------------------------------------------


class ACPWSConnectedMessage(BaseModel):
    """Sent when WebSocket connection is established."""
    type: Literal["connected"] = "connected"
    session_id: str
    agent_capabilities: dict[str, Any] | None = None


class ACPWSUpdateMessage(BaseModel):
    """Streamed update from the agent session."""
    type: Literal["update"] = "update"
    session_id: str
    update_type: str = Field(..., description="Type of update (e.g., 'text', 'tool_call', 'tool_result')")
    data: dict[str, Any] = Field(default_factory=dict)


class ACPWSPermissionRequestMessage(BaseModel):
    """Request for permission to execute a tool."""
    type: Literal["permission_request"] = "permission_request"
    request_id: str = Field(..., description="Unique ID for this permission request")
    session_id: str
    tool_name: str
    tool_arguments: dict[str, Any] = Field(default_factory=dict)
    tier: ACPPermissionTier = ACPPermissionTier.INDIVIDUAL
    approval_requirement: str | None = Field(
        default=None,
        description="Resolved runtime approval requirement for this tool request",
    )
    governance_reason: str | None = Field(
        default=None,
        description="Human-readable reason the runtime policy required approval",
    )
    deny_reason: str | None = Field(
        default=None,
        description="Reason the runtime policy denied the request",
    )
    provenance_summary: dict[str, Any] | None = Field(
        default=None,
        description="Compact provenance summary for the runtime policy decision",
    )
    runtime_narrowing_reason: str | None = Field(
        default=None,
        description="Reason local runtime safety narrowed the effective policy",
    )
    policy_snapshot_fingerprint: str | None = Field(
        default=None,
        description="Fingerprint of the ACP runtime policy snapshot used for the decision",
    )
    timeout_seconds: int = Field(default=300, description="Seconds until auto-cancel if no response")


class ACPWSErrorMessage(BaseModel):
    """Error message from the server."""
    type: Literal["error"] = "error"
    code: str
    message: str
    session_id: str | None = None
    data: dict[str, Any] | None = None


class ACPWSPromptCompleteMessage(BaseModel):
    """Sent when a prompt execution completes."""
    type: Literal["prompt_complete"] = "prompt_complete"
    session_id: str
    stop_reason: str | None = None
    raw_result: dict[str, Any] = Field(default_factory=dict)


# -----------------------------------------------------------------------------
# WebSocket Message Types (Client → Server)
# -----------------------------------------------------------------------------


class ACPWSPermissionResponseMessage(BaseModel):
    """Response to a permission request."""
    type: Literal["permission_response"] = "permission_response"
    request_id: str
    approved: bool
    batch_approve_tier: ACPPermissionTier | None = Field(
        default=None,
        description="If set, auto-approve all future requests of this tier"
    )


class ACPWSCancelMessage(BaseModel):
    """Request to cancel the current operation."""
    type: Literal["cancel"] = "cancel"
    session_id: str


class ACPWSPromptMessage(BaseModel):
    """Send a prompt to the session via WebSocket."""
    type: Literal["prompt"] = "prompt"
    session_id: str
    prompt: list[dict[str, Any]]


# -----------------------------------------------------------------------------
# Union types for message handling
# -----------------------------------------------------------------------------


ACPWSServerMessage = Union[
    ACPWSConnectedMessage,
    ACPWSUpdateMessage,
    ACPWSPermissionRequestMessage,
    ACPWSErrorMessage,
    ACPWSPromptCompleteMessage,
]

ACPWSClientMessage = Union[
    ACPWSPermissionResponseMessage,
    ACPWSCancelMessage,
    ACPWSPromptMessage,
]


# -----------------------------------------------------------------------------
# REST API Schemas
# -----------------------------------------------------------------------------


class ACPSessionNewRequest(BaseModel):
    """Request to create a new ACP session."""
    cwd: str = Field(..., description="Absolute working directory for the ACP session")
    name: str | None = Field(
        default=None,
        description="Optional session name. Auto-generated from cwd if not provided.",
    )
    agent_type: ACPAgentType | None = Field(
        default=None, description="Type of agent to use"
    )
    tags: list[str] | None = Field(
        default=None, description="Optional tags for organizing sessions"
    )
    mcp_servers: list[ACPMCPServerConfig] | None = Field(
        default=None, description="Optional MCP server configurations"
    )
    persona_id: str | None = Field(
        default=None,
        description="Optional persona identifier bound to sandbox tenancy metadata",
    )
    workspace_id: str | None = Field(
        default=None,
        description="Optional workspace identifier bound to sandbox tenancy metadata",
    )
    workspace_group_id: str | None = Field(
        default=None,
        description="Optional workspace group identifier bound to sandbox tenancy metadata",
    )
    scope_snapshot_id: str | None = Field(
        default=None,
        description="Optional materialized scope snapshot identifier for sandbox tenancy metadata",
    )


class ACPSessionNewResponse(BaseModel):
    """Response when a new ACP session is created."""
    session_id: str
    name: str = Field(..., description="Session name (user-provided or auto-generated)")
    agent_type: ACPAgentType = Field(..., description="Type of agent used")
    agent_capabilities: dict[str, Any] | None = None
    sandbox_session_id: str | None = Field(default=None, description="Sandbox session backing this ACP session")
    sandbox_run_id: str | None = Field(default=None, description="Sandbox run backing this ACP session")
    ssh_ws_url: str | None = Field(default=None, description="WebSocket URL for browser-based SSH")
    ssh_user: str | None = Field(default=None, description="SSH username for this session")
    persona_id: str | None = Field(default=None, description="Persona identifier bound to this ACP session")
    workspace_id: str | None = Field(default=None, description="Workspace identifier bound to this ACP session")
    workspace_group_id: str | None = Field(
        default=None, description="Workspace group identifier bound to this ACP session"
    )
    scope_snapshot_id: str | None = Field(
        default=None, description="Scope snapshot identifier bound to this ACP session"
    )
    policy_snapshot_version: str | None = Field(
        default=None, description="Version identifier for the current ACP policy snapshot"
    )
    policy_snapshot_fingerprint: str | None = Field(
        default=None, description="Fingerprint of the current ACP policy snapshot"
    )
    policy_snapshot_refreshed_at: str | None = Field(
        default=None, description="ISO 8601 timestamp when the ACP policy snapshot last refreshed"
    )
    policy_summary: dict[str, Any] | None = Field(
        default=None, description="Compact summary of the ACP runtime policy snapshot"
    )
    policy_provenance_summary: dict[str, Any] | None = Field(
        default=None, description="Compact provenance summary for the ACP runtime policy snapshot"
    )
    policy_refresh_error: str | None = Field(
        default=None, description="Last ACP policy snapshot refresh error, if any"
    )


class ACPSessionPromptRequest(BaseModel):
    session_id: str
    prompt: list[dict[str, Any]]


class ACPSessionPromptResponse(BaseModel):
    stop_reason: str | None = None
    raw_result: dict[str, Any]
    usage: ACPTokenUsage | None = Field(default=None, description="Token usage for this prompt turn")


class ACPSessionCancelRequest(BaseModel):
    session_id: str


class ACPSessionCloseRequest(BaseModel):
    session_id: str


class ACPSessionUpdatesResponse(BaseModel):
    updates: list[dict[str, Any]]


# -----------------------------------------------------------------------------
# Token Usage Tracking
# -----------------------------------------------------------------------------


class ACPTokenUsage(BaseModel):
    """Token usage counts for an ACP session."""
    prompt_tokens: int = Field(default=0, description="Total prompt/input tokens consumed")
    completion_tokens: int = Field(default=0, description="Total completion/output tokens consumed")
    total_tokens: int = Field(default=0, description="Total tokens consumed (prompt + completion)")


# -----------------------------------------------------------------------------
# Session Listing & Detail
# -----------------------------------------------------------------------------


class ACPSessionStatus(str, Enum):
    """Status of an ACP session."""
    ACTIVE = "active"
    CLOSED = "closed"
    ERROR = "error"


class ACPSessionInfo(BaseModel):
    """Summary information about an ACP session."""
    session_id: str = Field(..., description="Unique session identifier")
    user_id: int = Field(..., description="ID of the user who owns this session")
    agent_type: str = Field(default="custom", description="Type of agent used")
    name: str = Field(default="", description="Session name")
    status: ACPSessionStatus = Field(default=ACPSessionStatus.ACTIVE, description="Current session status")
    created_at: str = Field(..., description="ISO 8601 creation timestamp")
    last_activity_at: str | None = Field(default=None, description="ISO 8601 timestamp of last activity")
    message_count: int = Field(default=0, description="Number of messages exchanged")
    usage: ACPTokenUsage = Field(default_factory=ACPTokenUsage, description="Token usage for this session")
    tags: list[str] = Field(default_factory=list, description="Tags for organizing sessions")
    has_websocket: bool = Field(default=False, description="Whether a WebSocket is connected")
    persona_id: str | None = Field(default=None, description="Persona identifier bound to this ACP session")
    workspace_id: str | None = Field(default=None, description="Workspace identifier bound to this ACP session")
    workspace_group_id: str | None = Field(
        default=None, description="Workspace group identifier bound to this ACP session"
    )
    scope_snapshot_id: str | None = Field(
        default=None, description="Scope snapshot identifier bound to this ACP session"
    )
    policy_snapshot_version: str | None = Field(
        default=None, description="Version identifier for the current ACP policy snapshot"
    )
    policy_snapshot_fingerprint: str | None = Field(
        default=None, description="Fingerprint of the current ACP policy snapshot"
    )
    policy_snapshot_refreshed_at: str | None = Field(
        default=None, description="ISO 8601 timestamp when the ACP policy snapshot last refreshed"
    )
    policy_summary: dict[str, Any] | None = Field(
        default=None, description="Compact summary of the ACP runtime policy snapshot"
    )
    policy_provenance_summary: dict[str, Any] | None = Field(
        default=None, description="Compact provenance summary for the ACP runtime policy snapshot"
    )
    policy_refresh_error: str | None = Field(
        default=None, description="Last ACP policy snapshot refresh error, if any"
    )
    forked_from: str | None = Field(default=None, description="Source session ID when this session was forked")
    model: str | None = Field(default=None, description="LLM model used in this session (for cost estimation)")
    estimated_cost_usd: float | None = Field(default=None, description="Estimated cost in USD based on token usage and model pricing")
    token_budget: int | None = Field(default=None, description="Maximum token count for this session (NULL = no limit)")
    auto_terminate_at_budget: bool = Field(default=False, description="Whether the session auto-terminates when budget is exceeded")
    budget_exhausted: bool = Field(default=False, description="Whether the session was terminated due to budget exhaustion")
    budget_remaining: int | None = Field(default=None, description="Tokens remaining before budget is hit (NULL if no budget set)")


class ACPSessionListResponse(BaseModel):
    """Response for listing ACP sessions."""
    sessions: list[ACPSessionInfo] = Field(default_factory=list)
    total: int = Field(default=0, description="Total number of sessions matching filters")
    pagination: OffsetPaginationMeta
    has_more: bool | None = Field(default=None, description="Alias for pagination.has_more")
    next_offset: int | None = Field(default=None, ge=0, description="Alias for pagination.next_offset")

    @model_validator(mode="after")
    def _default_pagination_aliases(self):
        return _default_offset_pagination_aliases(self)


class ACPSessionDetailResponse(ACPSessionInfo):
    """Detailed information about an ACP session, including message history."""
    messages: list[dict[str, Any]] = Field(default_factory=list, description="Message history (if available)")
    cwd: str | None = Field(default=None, description="Working directory for this session")
    fork_lineage: list[str] = Field(default_factory=list, description="Ancestor session IDs from oldest to most recent parent")


class ACPSessionForkRequest(BaseModel):
    """Request to fork an ACP session from a specific message index."""
    message_index: int = Field(
        ...,
        ge=0,
        description="Index of the last message to include in the forked session (0-based)",
    )
    name: str | None = Field(default=None, description="Optional name for the forked session")


class ACPSessionForkResponse(BaseModel):
    """Response when a session is forked."""
    session_id: str = Field(..., description="New forked session ID")
    name: str = Field(..., description="Name of the forked session")
    forked_from: str = Field(..., description="ID of the source session")
    message_count: int = Field(default=0, description="Number of messages copied to the fork")


class ACPSessionUsageResponse(BaseModel):
    """Usage details for a specific ACP session."""
    session_id: str
    user_id: int
    agent_type: str = "custom"
    usage: ACPTokenUsage = Field(default_factory=ACPTokenUsage)
    message_count: int = 0
    created_at: str = ""
    last_activity_at: str | None = None
    model: str | None = Field(default=None, description="LLM model used in this session")
    estimated_cost_usd: float | None = Field(default=None, description="Estimated cost in USD")


# -----------------------------------------------------------------------------
# Agent Metrics (Admin aggregation)
# -----------------------------------------------------------------------------


class ACPAgentMetrics(BaseModel):
    """Aggregated runtime metrics for a single agent type."""
    agent_type: str = Field(..., description="Agent type identifier")
    session_count: int = Field(default=0, description="Total number of sessions")
    active_sessions: int = Field(default=0, description="Currently active sessions")
    total_prompt_tokens: int = Field(default=0, description="Sum of prompt tokens across all sessions")
    total_completion_tokens: int = Field(default=0, description="Sum of completion tokens across all sessions")
    total_tokens: int = Field(default=0, description="Sum of total tokens across all sessions")
    total_messages: int = Field(default=0, description="Sum of messages across all sessions")
    last_used_at: str | None = Field(default=None, description="ISO 8601 timestamp of most recent activity")
    total_estimated_cost_usd: float | None = Field(default=None, description="Total estimated cost in USD across all sessions for this agent")


class ACPAgentMetricsListResponse(BaseModel):
    """Response for the agent metrics aggregation endpoint."""
    items: list[ACPAgentMetrics] = Field(default_factory=list)


class ACPExecutionHealthSessionSummary(BaseModel):
    """Aggregated ACP session status counts for admin execution-health reporting."""
    total: int = Field(default=0, description="Total ACP sessions considered in this summary")
    by_status: dict[str, int] = Field(default_factory=dict, description="Session count by persisted status")


class ACPExecutionHealthFailureBuckets(BaseModel):
    """Normalized ACP execution-health failure buckets for admin reporting."""
    setup_blockers: int = Field(default=0, description="Agents or sessions blocked by setup/certification gaps")
    runner_session_failures: int = Field(default=0, description="Sessions or events that indicate runner/session failure")
    reviewer_rejections: int = Field(default=0, description="Sessions with reviewer rejection outcomes")
    reviewer_failures: int = Field(default=0, description="Sessions with reviewer execution or decision failures")
    governance_denials: int = Field(default=0, description="Sessions with governance or permission-denial outcomes")


class ACPExecutionHealthAgentSummary(BaseModel):
    """Compatibility and setup posture for one configured ACP agent."""
    agent_type: str = Field(..., description="Agent type identifier")
    name: str = Field(default="", description="Human-readable agent name")
    is_configured: bool = Field(default=False, description="Whether local prerequisites appear configured")
    support_state: ACPSupportState = Field(default="documented_unverified")
    verification_level: ACPVerificationLevel = Field(default="documented_only")
    setup_blocked: bool = Field(default=False, description="Whether this agent contributes a setup blocker")
    primary_blocker: str | None = Field(default=None, description="Primary setup or entrypoint blocker, if known")


class ACPExecutionHealthCompatibilitySummary(BaseModel):
    """Downstream-agent compatibility evidence summary for ACP admin reporting."""
    by_support_state: dict[str, int] = Field(default_factory=dict)
    documented_unverified_agents: list[str] = Field(default_factory=list)
    live_certification_required: bool = Field(default=False)
    docs_url: str = Field(default=ACP_COMPATIBILITY_DOCS_URL)


class ACPExecutionHealthRetentionSummary(BaseModel):
    """Configured ACP retention posture for admin execution-health reporting."""
    session_retention_days: int = Field(default=30)
    audit_retention_days: int = Field(default=30)
    policy: str = Field(default="closed_error_sessions_and_audit_events_purged_after_retention")


class ACPExecutionHealthRedactionSummary(BaseModel):
    """Redaction posture for support-safe ACP drill-through surfaces."""
    detail_events_artifacts_redacted_views: bool = Field(default=True)
    diagnostics_sanitized: bool = Field(default=True)
    audit_metadata_sanitized: bool = Field(default=True)


class ACPExecutionHealthSummaryResponse(BaseModel):
    """Admin ACP execution-health rollup across sessions, agents, and support posture."""
    timestamp: str = Field(..., description="ISO 8601 timestamp for this summary")
    range_days: int = Field(..., description="Lookback window used for session aggregation")
    sessions: ACPExecutionHealthSessionSummary = Field(default_factory=ACPExecutionHealthSessionSummary)
    failure_buckets: ACPExecutionHealthFailureBuckets = Field(default_factory=ACPExecutionHealthFailureBuckets)
    agents: list[ACPExecutionHealthAgentSummary] = Field(default_factory=list)
    compatibility: ACPExecutionHealthCompatibilitySummary = Field(default_factory=ACPExecutionHealthCompatibilitySummary)
    retention: ACPExecutionHealthRetentionSummary = Field(default_factory=ACPExecutionHealthRetentionSummary)
    redaction: ACPExecutionHealthRedactionSummary = Field(default_factory=ACPExecutionHealthRedactionSummary)


class ACPAgentUsageItem(BaseModel):
    """Aggregated usage statistics for a single agent type."""
    agent_type: str
    invocation_count: int = 0
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error_count: int = 0
    estimated_cost_usd: float = 0.0
    avg_tokens_per_session: float = 0.0


class ACPAgentUsageResponse(BaseModel):
    """Response for aggregated per-agent token usage."""
    agents: list[ACPAgentUsageItem]
    range_days: int


# -----------------------------------------------------------------------------
# Agent Configuration (Admin-managed)
# -----------------------------------------------------------------------------


class ACPAgentConfigCreate(BaseModel):
    """Request to create a custom agent configuration."""
    type: str = Field(..., description="Unique agent type identifier")
    name: str = Field(..., description="Human-readable name")
    description: str = Field(default="", description="Agent description")
    system_prompt: str | None = Field(default=None, description="System prompt for the agent")
    allowed_tools: list[str] | None = Field(default=None, description="Tools this agent is allowed to use (null = all)")
    denied_tools: list[str] | None = Field(default=None, description="Tools this agent is denied from using")
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Agent parameters (temperature, topP, model, max_tokens, etc.)",
    )
    requires_api_key: str | None = Field(default=None, description="Env var name for required API key")
    org_id: int | None = Field(default=None, description="Restrict to specific organization")
    team_id: int | None = Field(default=None, description="Restrict to specific team")
    enabled: bool = Field(default=True, description="Whether the agent is enabled")
    default_token_budget: int | None = Field(
        default=None,
        description="Default token budget for new sessions using this agent (NULL = no limit)",
    )
    default_auto_terminate_at_budget: bool = Field(
        default=True,
        description="Whether new sessions using this agent auto-terminate when budget is exceeded",
    )
    max_token_budget: int | None = Field(default=None, description="Maximum total tokens per session (null = unlimited)")


class ACPAgentConfigResponse(ACPAgentConfigCreate):
    """Agent configuration as stored."""
    id: int = Field(..., description="Config ID")
    created_at: str = Field(..., description="ISO 8601 creation timestamp")
    updated_at: str | None = Field(default=None, description="ISO 8601 last update timestamp")
    is_configured: bool = Field(default=True, description="Whether required keys are present")


class ACPAgentConfigListResponse(BaseModel):
    """Response for listing agent configurations."""
    agents: list[ACPAgentConfigResponse] = Field(default_factory=list)
    total: int = Field(default=0)


# -----------------------------------------------------------------------------
# Permission Policy (Admin-managed)
# -----------------------------------------------------------------------------


class ACPPermissionPolicyRule(BaseModel):
    """A single permission policy rule."""
    tool_pattern: str = Field(..., description="Tool name or glob pattern (e.g., 'file_*', 'bash')")
    tier: ACPPermissionTier = Field(..., description="Permission tier to assign")


class ACPPermissionPolicyCreate(BaseModel):
    """Request to create/update a permission policy."""
    name: str = Field(..., description="Policy name")
    description: str = Field(default="", description="Policy description")
    rules: list[ACPPermissionPolicyRule] = Field(default_factory=list)
    org_id: int | None = Field(default=None, description="Restrict to specific organization")
    team_id: int | None = Field(default=None, description="Restrict to specific team")
    priority: int = Field(default=0, description="Higher priority policies take precedence")


class ACPPermissionPolicyResponse(ACPPermissionPolicyCreate):
    """Permission policy as stored."""
    id: int = Field(..., description="Policy ID")
    created_at: str = Field(..., description="ISO 8601 creation timestamp")
    updated_at: str | None = Field(default=None)


class ACPPermissionPolicyListResponse(BaseModel):
    """Response for listing permission policies."""
    policies: list[ACPPermissionPolicyResponse] = Field(default_factory=list)
    total: int = Field(default=0)


# -----------------------------------------------------------------------------
# Health Check Response
# -----------------------------------------------------------------------------


class ACPSessionBudgetRequest(BaseModel):
    """Request to set or update the token budget for an ACP session."""
    token_budget: int | None = Field(
        default=None,
        description="Maximum token count for this session. NULL removes the budget.",
    )
    auto_terminate_at_budget: bool = Field(
        default=True,
        description="Whether to auto-terminate the session when the budget is exceeded.",
    )


class ACPSessionBudgetResponse(BaseModel):
    """Response after updating a session's token budget."""
    session_id: str = Field(..., description="Session identifier")
    token_budget: int | None = Field(default=None, description="Current token budget")
    auto_terminate_at_budget: bool = Field(default=False, description="Auto-terminate enabled")
    budget_exhausted: bool = Field(default=False, description="Whether the budget has been exhausted")
    total_tokens: int = Field(default=0, description="Current total token usage")
    budget_remaining: int | None = Field(default=None, description="Tokens remaining in budget")


class ACPHealthRouteStatus(BaseModel):
    """Route-gating status for ACP endpoints."""
    stable_only: bool = Field(..., description="Whether the server is in stable-only route mode")
    acp_enabled: bool = Field(..., description="Whether ACP routes are currently enabled")
    note: str | None = Field(
        default=None,
        description="Helpful note when ACP might be hidden by route gating",
    )


class ACPHealthResponse(BaseModel):
    """ACP dependency chain health check response."""
    timestamp: str = Field(..., description="ISO 8601 timestamp")
    runner: dict[str, Any] = Field(default_factory=dict, description="Runner binary status")
    agents: list[dict[str, Any]] = Field(default_factory=list, description="Downstream agent statuses")
    runner_probe: dict[str, Any] = Field(default_factory=dict, description="Runner process probe result")
    routes: ACPHealthRouteStatus | None = Field(default=None, description="Route-gating status")
    overall: str = Field(default="unknown", description="ok | degraded | unavailable")
    message: str | None = Field(default=None, description="Human-readable status message")


# -----------------------------------------------------------------------------
# Structured Error Responses
# -----------------------------------------------------------------------------


class ACPErrorSuggestion(BaseModel):
    """A suggestion for resolving an error."""
    action: str = Field(..., description="Suggested action to take")
    description: str | None = Field(default=None, description="More details about the action")


class ACPErrorResponse(BaseModel):
    """Structured error response with suggestions."""
    code: str = Field(..., description="Error code for programmatic handling")
    message: str = Field(..., description="Human-readable error message")
    suggestions: list[ACPErrorSuggestion] = Field(
        default_factory=list, description="Suggestions for resolving the error"
    )
    data: dict[str, Any] | None = Field(
        default=None, description="Additional error context"
    )


# -----------------------------------------------------------------------------
# Async Fire-and-Forget API
# -----------------------------------------------------------------------------


class ACPAsyncPromptRequest(BaseModel):
    """Request body for async prompt submission."""
    prompt: str | list[dict[str, Any]] = Field(
        ..., description="Prompt text or message list"
    )
    cwd: str = Field(default=".", description="Working directory for the agent")
    agent_type: str | None = Field(
        default=None, description="Agent type identifier"
    )
    model: str | None = Field(default=None, description="Model override")
    token_budget: int | None = Field(
        default=None, description="Maximum token budget"
    )
    persona_id: str | None = Field(
        default=None, description="Persona context for session creation"
    )
    workspace_id: str | None = Field(
        default=None, description="Workspace context for session creation"
    )
    sandbox_enabled: bool = Field(
        default=False, description="Whether to enable sandbox mode"
    )


class ACPAsyncPromptResponse(BaseModel):
    """Response for async prompt submission."""
    task_id: str = Field(..., description="Unique identifier for the background task")
    poll_url: str = Field(
        ..., description="Relative URL to poll for task status"
    )
    status: str = Field(default="queued", description="Initial task status")


class ACPTaskStatusResponse(BaseModel):
    """Response for polling task status."""
    task_id: str = Field(..., description="Task identifier")
    status: str = Field(
        ...,
        description="Task status: queued, running, completed, failed",
    )
    result: Any | None = Field(default=None, description="Task result when completed")
    usage: dict[str, Any] = Field(
        default_factory=dict, description="Token usage information"
    )
    error: str | None = Field(default=None, description="Error message if failed")
    duration_ms: int | None = Field(
        default=None, description="Execution duration in milliseconds"
    )


class ACPRollbackRequest(BaseModel):
    """Request body for rolling back sandbox state to a checkpoint."""
    to_sequence: int | None = Field(
        default=None,
        description="Target event sequence number; the nearest checkpoint at or before this sequence is used.",
    )
    to_snapshot_id: str | None = Field(
        default=None,
        description="Direct snapshot ID to restore. Takes precedence over to_sequence if both provided.",
    )


class ACPRollbackResponse(BaseModel):
    """Response from a rollback operation."""
    restored: bool = Field(..., description="Whether the rollback succeeded")
    snapshot_id: str = Field(..., description="The snapshot ID that was restored")
    sequence: int | None = Field(
        default=None, description="The event sequence the snapshot corresponds to, if resolved from to_sequence"
    )


class ACPCheckpointListResponse(BaseModel):
    """Response listing available checkpoints for a session."""
    session_id: str = Field(..., description="The session these checkpoints belong to")
    checkpoints: dict[int, str] = Field(
        default_factory=dict,
        description="Mapping of event sequence numbers to snapshot IDs",
    )
