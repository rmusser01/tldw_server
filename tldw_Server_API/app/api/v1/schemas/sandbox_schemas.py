from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field
try:
    from pydantic import model_validator
except ImportError:  # pragma: no cover - pydantic v1 fallback
    from pydantic import root_validator as model_validator  # type: ignore

RuntimeType = Literal["docker", "firecracker", "lima", "vz_linux", "vz_macos", "seatbelt"]
TrustLevelType = Literal["trusted", "standard", "untrusted"]


class SandboxRuntimeInfo(BaseModel):
    name: RuntimeType
    available: bool = Field(description="Whether this runtime is detected/usable on host")
    reasons: list[str] | None = Field(default=None, description="Preflight reasons when the runtime is unavailable or constrained")
    supported_trust_levels: list[TrustLevelType] | None = Field(default=None, description="Trust levels supported by this runtime under current host policy")
    default_images: list[str] = Field(default_factory=list)
    max_cpu: float | None = Field(default=None, description="Max CPU (cores) per run")
    max_mem_mb: int | None = Field(default=None, description="Max memory (MB) per run")
    max_upload_mb: int | None = Field(default=None, description="Max inline/session upload size (MB)")
    max_log_bytes: int | None = Field(default=None, description="Max bytes streamed to logs per run")
    queue_max_length: int | None = Field(default=None, description="Max queued runs before 429 is returned")
    queue_ttl_sec: int | None = Field(default=None, description="Maximum time a run may remain queued before being dropped")
    workspace_cap_mb: int | None = Field(default=None, description="Default workspace size cap (MB)")
    artifact_ttl_hours: int | None = Field(default=None, description="Default artifact retention (hours)")
    supported_spec_versions: list[str] = Field(default_factory=lambda: ["1.0"], description="Supported spec versions (e.g., ['1.0','1.1'] when 1.1 features are enabled)")
    interactive_supported: bool | None = Field(default=None, description="Whether stdin-over-WS interactive runs are supported")
    egress_allowlist_supported: bool | None = Field(default=None, description="Whether egress allowlisting is supported by the runtime")
    strict_deny_all_supported: bool | None = Field(default=None, description="Whether strict deny-all network enforcement is supported")
    strict_allowlist_supported: bool | None = Field(default=None, description="Whether strict allowlist network enforcement is supported")
    enforcement_ready: dict[str, bool] | None = Field(default=None, description="Runtime enforcement readiness by network policy mode")
    host: dict[str, str | bool] | None = Field(default=None, description="Runtime host capability facts for troubleshooting")
    store_mode: str | None = Field(default=None, description="Current store backend mode (memory|sqlite|cluster)")
    notes: str | None = None


class SandboxRuntimesResponse(BaseModel):
    runtimes: list[SandboxRuntimeInfo]


class SandboxSessionCreateRequest(BaseModel):
    spec_version: str = Field(default="1.0")
    runtime: RuntimeType | None = Field(default=None, description="Preferred runtime; if omitted, policy decides")
    base_image: str | None = Field(default=None, description="Default base image for runs in this session")
    cpu_limit: float | None = Field(default=None, ge=0, description="vCPUs or CPU shares as supported by runtime")
    memory_mb: int | None = Field(default=None, ge=64, description="Memory limit in MB")
    timeout_sec: int | None = Field(default=300, ge=1, le=3600)
    network_policy: Literal["deny_all", "allowlist"] | None = Field(default="deny_all")
    env: dict[str, str] | None = Field(default=None, description="Non-secret environment variables")
    labels: dict[str, str] | None = Field(default=None)
    trust_level: TrustLevelType | None = Field(
        default="standard",
        description="Trust level for risk-based isolation: trusted (relaxed), standard (default), untrusted (strict)"
    )
    persona_id: str | None = Field(default=None, description="Optional persona identifier bound to this sandbox session")
    workspace_id: str | None = Field(default=None, description="Optional workspace identifier bound to this sandbox session")
    workspace_group_id: str | None = Field(default=None, description="Optional workspace-group identifier bound to this sandbox session")
    scope_snapshot_id: str | None = Field(default=None, description="Optional scope snapshot identifier bound to this sandbox session")


class SandboxSession(BaseModel):
    id: str
    runtime: RuntimeType
    base_image: str | None = None
    cpu_limit: float | None = None
    memory_mb: int | None = None
    timeout_sec: int | None = None
    network_policy: Literal["deny_all", "allowlist"] | None = None
    env: dict[str, str] | None = None
    labels: dict[str, str] | None = None
    trust_level: TrustLevelType | None = None
    expires_at: datetime | None = None
    policy_hash: str | None = None
    persona_id: str | None = None
    workspace_id: str | None = None
    workspace_group_id: str | None = None
    scope_snapshot_id: str | None = None


class SandboxFileUploadResponse(BaseModel):
    session_id: str
    bytes_received: int
    file_count: int


class RunResources(BaseModel):
    cpu: float | None = Field(default=None, ge=0)
    memory_mb: int | None = Field(default=None, ge=64)


class RunFile(BaseModel):
    path: str
    content_b64: str


class SandboxRunCreateRequest(BaseModel):
    spec_version: str = Field(default="1.0")
    session_id: str | None = None
    runtime: RuntimeType | None = None
    base_image: str | None = None
    command: list[str]
    env: dict[str, str] | None = None
    startup_timeout_sec: int | None = Field(default=None, ge=1, le=600, description="Provisioning timeout (image pull/start). Separate from execution timeout.")
    timeout_sec: int | None = Field(default=300, ge=1, le=3600)
    resources: RunResources | None = None
    network_policy: Literal["deny_all", "allowlist"] | None = Field(default=None)
    files: list[RunFile] | None = Field(default=None, description="Inline small files to write before run")
    capture_patterns: list[str] | None = Field(default=None, description="Glob patterns for artifact capture")
    # Spec 1.1: interactive stdin over WS (backward compatible; ignored when runtime does not support it)
    interactive: bool | None = Field(default=None, description="Enable interactive mode with stdin over WS (spec 1.1)")
    stdin_max_bytes: int | None = Field(default=None, ge=0, description="Max total stdin bytes across connection(s)")
    stdin_max_frame_bytes: int | None = Field(default=None, ge=0, description="Max bytes per stdin frame")
    stdin_bps: int | None = Field(default=None, ge=0, description="Approximate stdin bytes-per-second rate limit")
    stdin_idle_timeout_sec: int | None = Field(default=None, ge=0, description="Close WS after this many seconds of stdin inactivity")
    # Spec 1.1: Optional resume hint for clients; WS also supports a 'from_seq' query parameter on /runs/{id}/stream
    resume_from_seq: int | None = Field(default=None, ge=0, description="Suggest resuming WS from this sequence number (spec 1.1)")
    # Trust level for risk-based isolation profiles
    trust_level: TrustLevelType | None = Field(
        default="standard",
        description="Trust level for risk-based isolation: trusted (relaxed), standard (default), untrusted (strict)"
    )
    persona_id: str | None = Field(default=None, description="Optional persona identifier bound to this run")
    workspace_id: str | None = Field(default=None, description="Optional workspace identifier bound to this run")
    workspace_group_id: str | None = Field(default=None, description="Optional workspace-group identifier bound to this run")
    scope_snapshot_id: str | None = Field(default=None, description="Optional scope snapshot identifier bound to this run")

    @model_validator(mode="after")
    def validate_session_or_base_image(self) -> SandboxRunCreateRequest:
        has_session = isinstance(self.session_id, str) and bool(self.session_id.strip())
        has_image = isinstance(self.base_image, str) and bool(self.base_image.strip())
        if has_session == has_image:
            raise ValueError("Provide exactly one of session_id or base_image")
        return self


class SandboxRun(BaseModel):
    id: str
    session_id: str | None = None
    runtime: RuntimeType
    base_image: str | None = None
    command: list[str]
    created_at: datetime


class SandboxRunStatus(BaseModel):
    id: str
    spec_version: str | None = None
    runtime: RuntimeType | None = None
    runtime_version: str | None = None
    base_image: str | None = None
    image_digest: str | None = None
    policy_hash: str | None = None
    phase: Literal[
        "queued",
        "starting",
        "running",
        "completed",
        "failed",
        "killed",
        "timed_out",
    ]
    exit_code: int | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    message: str | None = None
    resource_usage: dict[str, int] | None = Field(default=None, description="Resource usage summary when available")
    estimated_start_time: datetime | None = None
    log_stream_url: str | None = Field(default=None, description="Optional WS URL (signed or unsigned) to stream logs; may include from_seq query (spec 1.1)")
    session_id: str | None = None
    persona_id: str | None = None
    workspace_id: str | None = None
    workspace_group_id: str | None = None
    scope_snapshot_id: str | None = None


class ArtifactInfo(BaseModel):
    path: str
    size: int
    download_url: str | None = None


class ArtifactListResponse(BaseModel):
    items: list[ArtifactInfo]


class CancelResponse(BaseModel):
    id: str
    cancelled: bool
    message: str | None = None


# Admin API Schemas
class SandboxAdminRunSummary(BaseModel):
    id: str
    user_id: str | None = None
    spec_version: str | None = None
    runtime: RuntimeType | None = None
    runtime_version: str | None = None
    base_image: str | None = None
    image_digest: str | None = None
    policy_hash: str | None = None
    phase: Literal[
        "queued",
        "starting",
        "running",
        "completed",
        "failed",
        "killed",
        "timed_out",
    ]
    exit_code: int | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    message: str | None = None
    session_id: str | None = None
    persona_id: str | None = None
    workspace_id: str | None = None
    workspace_group_id: str | None = None
    scope_snapshot_id: str | None = None


class SandboxAdminRunListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    has_more: bool
    items: list[SandboxAdminRunSummary]


class SandboxAdminRunDetails(SandboxAdminRunSummary):
    resource_usage: dict[str, int] | None = None


# Admin: Idempotency listing
class SandboxAdminIdempotencyItem(BaseModel):
    endpoint: str
    user_id: str | None = None
    key: str
    fingerprint: str | None = None
    object_id: str
    created_at: str | None = None


class SandboxAdminIdempotencyListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    has_more: bool
    items: list[SandboxAdminIdempotencyItem]


# Admin: Usage aggregates
class SandboxAdminUsageItem(BaseModel):
    user_id: str
    runs_count: int
    log_bytes: int
    artifact_bytes: int


class SandboxAdminUsageResponse(BaseModel):
    total: int
    limit: int
    offset: int
    has_more: bool
    items: list[SandboxAdminUsageItem]


class SandboxAdminMacOSHostDiagnostics(BaseModel):
    """Admin-facing host facts for macOS sandbox readiness checks."""

    os: str
    arch: str
    apple_silicon: bool
    macos_version: str | None = None
    supported: bool
    reasons: list[str] = Field(default_factory=list)


class SandboxAdminMacOSHelperDiagnostics(BaseModel):
    """Admin-facing helper readiness and optional helper metadata."""

    configured: bool
    path: str | None = None
    exists: bool
    executable: bool
    ready: bool
    transport: str | None = None
    protocol_version: str | None = None
    helper_version: str | None = None
    reasons: list[str] = Field(default_factory=list)


class SandboxAdminMacOSTemplateDiagnostics(BaseModel):
    """Admin-facing template readiness for a single VZ runtime family."""

    configured: bool
    ready: bool
    source: str | None = None
    reasons: list[str] = Field(default_factory=list)


class SandboxAdminMacOSRuntimeDiagnostics(BaseModel):
    """Admin-facing runtime posture derived from shared runtime preflight checks."""

    available: bool
    supported_trust_levels: list[TrustLevelType] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    execution_mode: Literal["fake", "real", "none"]
    remediation: str | None = None


class SandboxAdminMacOSDiagnosticsResponse(BaseModel):
    """Structured admin response for macOS sandbox diagnostics."""

    host: SandboxAdminMacOSHostDiagnostics
    helper: SandboxAdminMacOSHelperDiagnostics
    templates: dict[str, SandboxAdminMacOSTemplateDiagnostics] = Field(default_factory=dict)
    runtimes: dict[str, SandboxAdminMacOSRuntimeDiagnostics] = Field(default_factory=dict)


# Snapshot/Clone Schemas
class SnapshotCreateResponse(BaseModel):
    """Response when creating a session snapshot."""
    snapshot_id: str = Field(description="Unique identifier for the snapshot")
    created_at: str = Field(description="ISO 8601 timestamp of snapshot creation")
    size_bytes: int = Field(description="Size of the snapshot in bytes")


class SnapshotInfo(BaseModel):
    """Information about a session snapshot."""
    snapshot_id: str = Field(description="Unique identifier for the snapshot")
    session_id: str = Field(description="Session ID this snapshot belongs to")
    created_at: str = Field(description="ISO 8601 timestamp of snapshot creation")
    size_bytes: int = Field(description="Size of the snapshot in bytes")


class SnapshotListResponse(BaseModel):
    """Response listing available snapshots for a session."""
    items: list[SnapshotInfo] = Field(default_factory=list)


class SnapshotRestoreRequest(BaseModel):
    """Request to restore a session from a snapshot."""
    snapshot_id: str = Field(description="ID of the snapshot to restore")


class SnapshotRestoreResponse(BaseModel):
    """Response after restoring a session from a snapshot."""
    restored: bool = Field(description="Whether restoration was successful")
    snapshot_id: str = Field(description="ID of the restored snapshot")


class SessionCloneRequest(BaseModel):
    """Request to clone a session."""
    new_session_name: str | None = Field(
        default=None,
        description="Optional name for the new session"
    )


class SessionCloneResponse(BaseModel):
    """Response after cloning a session."""
    session_id: str = Field(description="ID of the newly created session")
    cloned_from: str = Field(description="ID of the original session")
