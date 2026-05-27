from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

try:
    from pydantic import model_validator
except ImportError:  # pragma: no cover - pydantic v1 fallback
    from pydantic import root_validator as model_validator  # type: ignore

from tldw_Server_API.app.api.v1.schemas.pagination import OffsetPaginationMeta
from tldw_Server_API.app.core.Sandbox.run_status_taxonomy import (
    RunStatusOperatorAction,
    RunStatusReasonCategory,
    RunStatusReasonCode,
    RunStatusReasonSeverity,
)
from tldw_Server_API.app.core.Sandbox.runtime_capabilities import (
    RuntimeBoundaryClass,
    RuntimeImplementationState,
    RuntimeIsolationWarningCode,
    RuntimeNetworkPolicyReadinessSource,
    RuntimeNetworkPolicySupportState,
    RuntimeReasonCategory,
    RuntimeReasonCode,
    RuntimeReasonOperatorAction,
    RuntimeReasonSeverity,
    RuntimeSessionReuseModel,
)

RuntimeType = Literal[
    "docker",
    "firecracker",
    "lima",
    "vz_linux",
    "vz_macos",
    "seatbelt",
    "worktree",
]
TrustLevelType = Literal["trusted", "standard", "untrusted"]


class SandboxRuntimeNetworkPolicyModeInfo(BaseModel):
    """Static network policy posture for one runtime policy mode."""

    support_state: RuntimeNetworkPolicySupportState = Field(
        ...,
        description=(
            "Static support state for this runtime/network-policy pair: "
            "supported, unsupported, scaffold, host_gated, or not_applicable"
        ),
    )
    strict_enforcement: bool = Field(
        ...,
        description=(
            "Whether the runtime can provide strict enforcement for this policy "
            "when its support and readiness requirements are satisfied"
        ),
    )
    readiness_source: RuntimeNetworkPolicyReadinessSource = Field(
        ...,
        description=(
            "Where current readiness should be read from: runtime_preflight, "
            "config, or not_applicable"
        ),
    )


class SandboxRuntimeNetworkPolicyContract(BaseModel):
    """Static deny-all and allowlist network policy posture for a runtime."""

    deny_all: SandboxRuntimeNetworkPolicyModeInfo
    allowlist: SandboxRuntimeNetworkPolicyModeInfo


class SandboxRuntimeSessionContract(BaseModel):
    """Static session semantics for a runtime."""

    support_state: RuntimeImplementationState = Field(
        ...,
        description=(
            "Static support state for session participation: supported, "
            "unsupported, scaffold, host_gated, or not_applicable"
        ),
    )
    reuse_model: RuntimeSessionReuseModel = Field(
        ...,
        description=(
            "Runtime reuse model for session-backed runs: none, workspace_only, "
            "warm_vm, or scaffold"
        ),
    )
    requires_live_health_check: bool = Field(
        ...,
        description=(
            "Whether safe session reuse depends on checking live runtime state "
            "before reusing the session runtime"
        ),
    )
    recovery_state: RuntimeImplementationState = Field(
        ...,
        description="Static support state for cross-restart session recovery semantics",
    )
    repair_state: RuntimeImplementationState = Field(
        ...,
        description="Static support state for explicit operator/admin session repair semantics",
    )


class SandboxRuntimeReasonDetails(BaseModel):
    """Structured metadata for a normalized sandbox runtime discovery reason."""

    model_config = ConfigDict(from_attributes=True)

    code: RuntimeReasonCode
    category: RuntimeReasonCategory
    severity: RuntimeReasonSeverity
    availability_blocking: bool
    operator_action: RuntimeReasonOperatorAction
    user_message_key: str


def _default_offset_pagination_aliases(response):
    if response.next_offset is None:
        response.next_offset = response.pagination.next_offset
    return response


class SandboxRuntimeInfo(BaseModel):
    name: RuntimeType
    available: bool = Field(description="Whether this runtime is detected/usable on host")
    implementation_state: RuntimeImplementationState | None = Field(
        default=None,
        description=(
            "Roadmap maturity label for this runtime independent of current host availability: "
            "supported, unsupported, scaffold, host_gated, or not_applicable"
        ),
    )
    reasons: list[str] | None = Field(default=None, description="Preflight reasons when the runtime is unavailable or constrained")
    normalized_reasons: list[RuntimeReasonCode] | None = Field(
        default=None,
        description=(
            "Stable, client-facing reason codes derived from raw runtime preflight reasons; "
            "raw reasons are preserved for operator diagnostics"
        ),
    )
    normalized_reason_details: list[SandboxRuntimeReasonDetails] = Field(
        default_factory=list,
        description=(
            "Structured metadata derived from normalized_reasons for client and "
            "operator presentation. Existing raw reasons remain authoritative."
        ),
    )
    supported_trust_levels: list[TrustLevelType] | None = Field(default=None, description="Trust levels supported by this runtime under current host policy")
    default_images: list[str] = Field(default_factory=list)
    max_cpu: float | None = Field(default=None, description="Max CPU (cores) per run")
    max_mem_mb: int | None = Field(default=None, description="Max memory (MB) per run")
    max_upload_mb: int | None = Field(default=None, description="Max inline/session upload size (MB)")
    max_log_bytes: int | None = Field(default=None, description="Max bytes streamed to logs per run")
    max_artifact_file_bytes: int | None = Field(default=None, description="Max bytes captured for a single artifact file")
    max_artifact_total_bytes: int | None = Field(default=None, description="Max total artifact bytes captured per run")
    queue_max_length: int | None = Field(default=None, description="Max queued runs before 429 is returned")
    queue_ttl_sec: int | None = Field(default=None, description="Maximum time a run may remain queued before being dropped")
    workspace_cap_mb: int | None = Field(default=None, description="Default workspace size cap (MB)")
    artifact_ttl_hours: int | None = Field(default=None, description="Default artifact retention (hours)")
    boundary_class: RuntimeBoundaryClass = Field(
        ...,
        description=(
            "Machine-readable runtime boundary category: container, host_local, "
            "vm_grade, or vm_grade_scaffold"
        ),
    )
    vm_grade_isolation: bool = Field(
        ...,
        description=(
            "Whether the runtime boundary is VM-grade for isolation claims; "
            "independent of current host availability"
        ),
    )
    untrusted_eligible: bool = Field(
        ...,
        description=(
            "Whether policy may admit this runtime for untrusted workloads when "
            "preflight and host readiness also pass"
        ),
    )
    isolation_warnings: list[RuntimeIsolationWarningCode] = Field(
        default_factory=list,
        description=(
            "Advisory static isolation warning codes for client UX; these do "
            "not replace policy admission, runtime preflight, or diagnostics"
        ),
    )
    network_policy_contract: SandboxRuntimeNetworkPolicyContract = Field(
        ...,
        description=(
            "Static runtime network policy posture; current host readiness remains "
            "in enforcement_ready and runtime preflight reasons"
        ),
    )
    session_contract: SandboxRuntimeSessionContract = Field(
        ...,
        description=(
            "Static runtime session semantics; current host readiness remains "
            "in available, reasons, and admin diagnostics"
        ),
    )
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


class SandboxRunStatusReasonDetails(BaseModel):
    """Structured metadata for a normalized sandbox run status reason."""

    model_config = ConfigDict(from_attributes=True)

    code: RunStatusReasonCode
    category: RunStatusReasonCategory
    severity: RunStatusReasonSeverity
    terminal: bool
    retryable: bool
    operator_action: RunStatusOperatorAction
    user_message_key: str


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
    status_reason_code: RunStatusReasonCode | None = Field(
        default=None,
        description=(
            "Stable client-facing reason code derived from phase, message, exit_code, "
            "and aggregate limit counters. Existing phase/message fields remain raw."
        ),
    )
    status_reason_details: SandboxRunStatusReasonDetails | None = Field(
        default=None,
        description=(
            "Structured metadata derived from status_reason_code for client and "
            "operator presentation. Existing raw status fields remain authoritative."
        ),
    )
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


class SandboxWorkspaceDiagnosticState(BaseModel):
    state: Literal[
        "available",
        "not_configured",
        "unavailable",
        "blocked",
        "unknown",
    ]
    reason_code: str | None = None
    message: str
    management_surface: str | None = None


class SandboxWorkspaceDiagnosticsRunSummary(BaseModel):
    id: str
    runtime: RuntimeType | None = None
    runtime_version: str | None = None
    base_image: str | None = None
    phase: Literal[
        "queued",
        "starting",
        "running",
        "completed",
        "failed",
        "killed",
        "timed_out",
    ]
    status_reason_code: RunStatusReasonCode | None = None
    status_reason_details: SandboxRunStatusReasonDetails | None = None
    exit_code: int | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    message: str | None = None
    session_id: str | None = None
    persona_id: str | None = None
    workspace_id: str | None = None
    workspace_group_id: str | None = None
    scope_snapshot_id: str | None = None


class SandboxWorkspaceDiagnosticsRunList(BaseModel):
    total: int
    limit: int
    has_more: bool
    items: list[SandboxWorkspaceDiagnosticsRunSummary]


class SandboxWorkspaceDiagnosticsLinks(BaseModel):
    runtime_config: str | None = None
    admin_runs: str | None = None


class SandboxWorkspaceDiagnosticsResponse(BaseModel):
    workspace_id: str
    source_label: Literal["research_workspace"]
    runtime: SandboxWorkspaceDiagnosticState
    admission: SandboxWorkspaceDiagnosticState
    runs: SandboxWorkspaceDiagnosticsRunList
    links: SandboxWorkspaceDiagnosticsLinks


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
    status_reason_code: RunStatusReasonCode | None = Field(
        default=None,
        description="Stable client-facing reason code derived from the run status summary",
    )
    status_reason_details: SandboxRunStatusReasonDetails | None = Field(
        default=None,
        description=(
            "Structured metadata derived from status_reason_code for admin/operator "
            "presentation."
        ),
    )
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
    next_offset: int | None = Field(default=None, ge=0, description="Alias for pagination.next_offset")
    pagination: OffsetPaginationMeta
    items: list[SandboxAdminRunSummary]

    @model_validator(mode="after")
    def default_pagination_aliases(self) -> SandboxAdminRunListResponse:
        return _default_offset_pagination_aliases(self)


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
    next_offset: int | None = Field(default=None, ge=0, description="Alias for pagination.next_offset")
    pagination: OffsetPaginationMeta
    items: list[SandboxAdminIdempotencyItem]

    @model_validator(mode="after")
    def default_pagination_aliases(self) -> SandboxAdminIdempotencyListResponse:
        return _default_offset_pagination_aliases(self)


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
    next_offset: int | None = Field(default=None, ge=0, description="Alias for pagination.next_offset")
    pagination: OffsetPaginationMeta
    items: list[SandboxAdminUsageItem]

    @model_validator(mode="after")
    def default_pagination_aliases(self) -> SandboxAdminUsageResponse:
        return _default_offset_pagination_aliases(self)


RuntimeDiagnosticsReadiness = Literal[
    "ready",
    "unavailable",
    "host_gated",
    "scaffold",
    "unsupported",
    "not_applicable",
]
RuntimeDiagnosticsAction = Literal[
    "none",
    "check_helper",
    "configure_template",
    "prepare_host",
    "adjust_request_policy",
    "use_different_runtime",
    "inspect_reasons",
]


class SandboxAdminRuntimeDiagnosticsSummary(BaseModel):
    """Aggregate operator posture for all sandbox runtimes."""

    total: int
    ready: int
    unavailable: int
    host_gated: int
    scaffold: int
    host_local_warning_runtimes: list[RuntimeType] = Field(default_factory=list)
    repair_supported_runtimes: list[RuntimeType] = Field(default_factory=list)


class SandboxAdminRuntimeDiagnosticsItem(BaseModel):
    """Read-only operator projection for one sandbox runtime."""

    name: RuntimeType
    available: bool
    implementation_state: RuntimeImplementationState | None = None
    readiness: RuntimeDiagnosticsReadiness
    reasons: list[str] = Field(default_factory=list)
    normalized_reasons: list[RuntimeReasonCode] = Field(default_factory=list)
    normalized_reason_details: list[SandboxRuntimeReasonDetails] = Field(default_factory=list)
    boundary_class: RuntimeBoundaryClass | None = None
    vm_grade_isolation: bool = False
    untrusted_eligible: bool = False
    isolation_warnings: list[RuntimeIsolationWarningCode] = Field(default_factory=list)
    strict_deny_all_supported: bool = False
    strict_allowlist_supported: bool = False
    session_reuse_model: RuntimeSessionReuseModel | None = None
    requires_live_health_check: bool = False
    repair_supported: bool = False
    recommended_action: RuntimeDiagnosticsAction


class SandboxAdminStartupWarningSummary(BaseModel):
    """Compact startup warning summary projected into sandbox diagnostics."""

    present: bool
    blocking: bool
    codes: list[str] = Field(default_factory=list)


class SandboxAdminRuntimeDiagnosticsResponse(BaseModel):
    """Admin-facing cross-runtime diagnostics derived from runtime discovery."""

    source: Literal["feature_discovery"]
    summary: SandboxAdminRuntimeDiagnosticsSummary
    runtimes: list[SandboxAdminRuntimeDiagnosticsItem]
    startup_warning_summary: SandboxAdminStartupWarningSummary | None = None


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


class SandboxAdminMacOSReconciliationItem(BaseModel):
    status: str
    session_id: str | None = None
    vm_id: str | None = None
    state: str | None = None
    healthy: bool | None = None
    reason: str | None = None
    termination_eligible: bool | None = None
    run_id: str | None = None
    helper_session_id: str | None = None
    template_id: str | None = None
    persisted_template_id: str | None = None
    helper_template_id: str | None = None
    template_id_matches_persisted: bool | None = None
    planning_source: str | None = None
    run_manifest_path: str | None = None
    run_manifest_present: bool | None = None


class SandboxAdminMacOSReconciliationDiagnostics(BaseModel):
    """Admin-facing comparison between persisted VZ session state and live helper VMs."""

    computed: bool
    persisted_sessions: int
    live_vms: int
    healthy_session_ids: list[str] = Field(default_factory=list)
    stale_session_ids: list[str] = Field(default_factory=list)
    unhealthy_session_ids: list[str] = Field(default_factory=list)
    skipped_active_session_ids: list[str] = Field(default_factory=list)
    orphaned_vm_ids: list[str] = Field(default_factory=list)
    owned_orphaned_vm_ids: list[str] = Field(default_factory=list)
    unknown_orphaned_vm_ids: list[str] = Field(default_factory=list)
    foreign_orphaned_vm_ids: list[str] = Field(default_factory=list)
    items: list[SandboxAdminMacOSReconciliationItem] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class SandboxAdminMacOSImageStoreItem(BaseModel):
    run_id: str
    template_id: str | None = None
    run_manifest_path: str | None = None
    run_manifest_present: bool | None = None
    gc_reason: str | None = None
    gc_path: str | None = None
    matched_vm_id: str | None = None
    matched_reconciliation_status: str | None = None
    matched_reconciliation_reason: str | None = None


class SandboxAdminMacOSImageStoreTemplate(BaseModel):
    template_id: str
    runtime: str
    template_name: str
    artifact_format: str
    source_path: str | None = None
    artifact_count: int = 0
    artifact_size_bytes: int = 0
    oci_image_ref: str | None = None
    oci_platform: str | None = None
    oci_manifest_digest: str | None = None
    oci_config_digest: str | None = None
    oci_layer_digests: list[str] = Field(default_factory=list)
    registry: str | None = None
    imported_at: str | None = None
    provenance: dict[str, object] = Field(default_factory=dict)


class SandboxAdminMacOSImageStoreDiagnostics(BaseModel):
    configured: bool
    root_path: str | None = None
    registered_templates: int = 0
    run_manifests: int = 0
    gc_candidates: int = 0
    templates: list[SandboxAdminMacOSImageStoreTemplate] = Field(default_factory=list)
    items: list[SandboxAdminMacOSImageStoreItem] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class SandboxAdminMacOSLogPointer(BaseModel):
    """Read-only pointer to a host log file without exposing file contents."""

    path: str | None = None
    exists: bool = False
    size_bytes: int | None = None


class SandboxAdminMacOSHelperLogPointers(BaseModel):
    """Pointers to the managed VZ helper stdout and stderr logs."""

    stdout: SandboxAdminMacOSLogPointer
    stderr: SandboxAdminMacOSLogPointer


class SandboxAdminMacOSGuestObservability(BaseModel):
    """Guest-agent readiness metadata reported by the helper for a live VM."""

    version: str | None = None
    workspace_root: str | None = None
    capabilities_known: bool | None = None
    capabilities: list[str] = Field(default_factory=list)
    compatibility: Literal["compatible", "unknown", "mismatch"] = "unknown"
    reasons: list[str] = Field(default_factory=list)
    expected_workspace_root: str | None = None
    required_capabilities: list[str] = Field(default_factory=list)
    missing_required_capabilities: list[str] = Field(default_factory=list)


class SandboxAdminMacOSVMObservability(BaseModel):
    """Per-VM boot-log, guest, and resource diagnostics for VZ Linux."""

    vm_id: str
    state: str | None = None
    healthy: bool
    run_id: str | None = None
    session_id: str | None = None
    session_mode: bool = False
    serial_log: SandboxAdminMacOSLogPointer
    guest: SandboxAdminMacOSGuestObservability
    resource_snapshot: dict[str, int] = Field(default_factory=dict)


class SandboxAdminMacOSObservabilityDiagnostics(BaseModel):
    """Aggregated read-only VZ Linux observability block for admin diagnostics."""

    configured: bool
    serial_log_dir: str | None = None
    helper_log_dir: str | None = None
    helper_log_dir_source: str | None = None
    helper_logs: SandboxAdminMacOSHelperLogPointers
    live_vms: int = 0
    vms: list[SandboxAdminMacOSVMObservability] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class SandboxAdminMacOSRecoverySummary(BaseModel):
    """Operator-facing recovery posture derived from macOS diagnostics blocks."""

    status: Literal["healthy", "action_recommended", "unavailable"]
    severity: Literal["ok", "warning", "error"]
    codes: list[str] = Field(default_factory=list)
    counts: dict[str, int] = Field(default_factory=dict)
    recommended_action: str | None = None
    repair_endpoint: str | None = None
    cleanup_plan_endpoint: str | None = None
    notes: list[str] = Field(default_factory=list)


class SandboxAdminMacOSDiagnosticsResponse(BaseModel):
    """Structured admin response for macOS sandbox diagnostics."""

    host: SandboxAdminMacOSHostDiagnostics
    helper: SandboxAdminMacOSHelperDiagnostics
    templates: dict[str, SandboxAdminMacOSTemplateDiagnostics] = Field(default_factory=dict)
    runtimes: dict[str, SandboxAdminMacOSRuntimeDiagnostics] = Field(default_factory=dict)
    reconciliation: SandboxAdminMacOSReconciliationDiagnostics | None = None
    image_store: SandboxAdminMacOSImageStoreDiagnostics | None = None
    observability: SandboxAdminMacOSObservabilityDiagnostics | None = None
    recovery_summary: SandboxAdminMacOSRecoverySummary | None = None
    startup_warning_summary: SandboxAdminStartupWarningSummary | None = None


class SandboxAdminMacOSReconciliationRepairRequest(BaseModel):
    delete_stale_session_controls: bool = True
    delete_unhealthy_session_controls: bool = True
    terminate_orphaned_vms: bool = False
    dry_run: bool = True


class SandboxAdminMacOSReconciliationRepairAction(BaseModel):
    type: str
    session_id: str | None = None
    vm_id: str | None = None
    status: str
    reason: str | None = None
    termination_eligible: bool | None = None
    run_id: str | None = None
    template_id: str | None = None
    planning_source: str | None = None
    run_manifest_path: str | None = None
    run_manifest_present: bool | None = None
    persisted_template_id: str | None = None
    helper_template_id: str | None = None
    template_id_matches_persisted: bool | None = None


class SandboxAdminMacOSReconciliationRepairSummary(BaseModel):
    stale_session_controls: int = 0
    unhealthy_session_controls: int = 0
    deleted_session_controls: int = 0
    skipped_active_sessions: int = 0
    orphaned_vms: int = 0
    terminated_orphaned_vms: int = 0


class SandboxAdminMacOSReconciliationRepairResponse(BaseModel):
    dry_run: bool
    helper: dict[str, object] = Field(default_factory=dict)
    summary: SandboxAdminMacOSReconciliationRepairSummary
    actions: list[SandboxAdminMacOSReconciliationRepairAction] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class SandboxAdminMacOSImageStoreCleanupAction(BaseModel):
    type: str
    run_id: str
    status: str
    error: str | None = None
    template_id: str | None = None
    run_manifest_path: str | None = None
    run_manifest_present: bool | None = None
    gc_reason: str | None = None
    gc_path: str | None = None
    matched_vm_id: str | None = None
    matched_reconciliation_status: str | None = None
    matched_reconciliation_reason: str | None = None


class SandboxAdminMacOSImageStoreCleanupPlanSummary(BaseModel):
    total_candidates: int = 0
    planned_actions: int = 0
    blocked_live_matches: int = 0
    planning_only_run_manifests: int = 0
    inactive_runs: int = 0
    legacy_run_directories: int = 0


class SandboxAdminMacOSImageStoreCleanupPlanResponse(BaseModel):
    dry_run: bool
    image_store: SandboxAdminMacOSImageStoreDiagnostics
    summary: SandboxAdminMacOSImageStoreCleanupPlanSummary
    actions: list[SandboxAdminMacOSImageStoreCleanupAction] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class SandboxAdminMacOSImageStoreCleanupRequest(BaseModel):
    dry_run: bool = True
    confirm_all: bool = False
    action_types: list[str] | None = None
    run_ids: list[str] | None = None


class SandboxAdminMacOSImageStoreCleanupSummary(BaseModel):
    total_candidates: int = 0
    planned_actions: int = 0
    deleted_actions: int = 0
    blocked_live_matches: int = 0
    planning_only_run_manifests: int = 0
    inactive_runs: int = 0
    legacy_run_directories: int = 0


class SandboxAdminMacOSImageStoreCleanupResponse(BaseModel):
    dry_run: bool
    image_store: SandboxAdminMacOSImageStoreDiagnostics
    summary: SandboxAdminMacOSImageStoreCleanupSummary
    actions: list[SandboxAdminMacOSImageStoreCleanupAction] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


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
