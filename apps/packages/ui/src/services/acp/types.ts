/**
 * ACP (Agent Client Protocol) TypeScript Types
 *
 * These types mirror the backend Pydantic schemas for ACP WebSocket communication.
 */

// -----------------------------------------------------------------------------
// Agent Types
// -----------------------------------------------------------------------------

export type ACPAgentType = string
export type ACPSupportState =
  | "supported"
  | "supported_with_caveats"
  | "experimental"
  | "documented_unverified"
  | "unsupported"

export type ACPVerificationLevel =
  | "documented_only"
  | "stub_smoke_tested"
  | "live_e2e_tested"
  | "sandbox_tested"
  | "production_supported"

export type ACPSetupHealthStatus = "ok" | "degraded" | "blocked" | "unknown"

export interface ACPAgentInfo {
  type: ACPAgentType
  name: string
  description: string
  is_configured: boolean
  requires_api_key?: string | null
  support_state?: ACPSupportState
  verification_level?: ACPVerificationLevel
  compatibility_notes?: string
  compatibility_docs_url?: string | null
}

export interface ACPAgentListResponse {
  agents: ACPAgentInfo[]
  default_agent: ACPAgentType
}

export interface ACPExecutionHealthSessionSummary {
  total: number
  by_status: Record<string, number>
}

export interface ACPExecutionHealthFailureBuckets {
  setup_blockers: number
  runner_session_failures: number
  reviewer_rejections: number
  reviewer_failures: number
  governance_denials: number
  structured_completion_failures: number
  sandbox_runtime_errors: number
  retention_redaction_actions: number
}

export interface ACPExecutionHealthSetupDimension {
  status: ACPSetupHealthStatus
  blockers: string[]
  evidence_count: number
}

export interface ACPExecutionHealthSetupSummary {
  agent: ACPExecutionHealthSetupDimension
  workspace: ACPExecutionHealthSetupDimension
  sandbox_runtime: ACPExecutionHealthSetupDimension
  mcp_injection: ACPExecutionHealthSetupDimension
  scheduler_trigger_path: ACPExecutionHealthSetupDimension
}

export interface ACPExecutionHealthAgentSummary {
  agent_type: string
  name: string
  is_configured: boolean
  support_state: ACPSupportState
  verification_level: ACPVerificationLevel
  setup_blocked: boolean
  primary_blocker?: string | null
}

export interface ACPExecutionHealthCompatibilitySummary {
  by_support_state: Record<string, number>
  documented_unverified_agents: string[]
  live_certification_required: boolean
  docs_url: string
}

export interface ACPExecutionHealthRetentionSummary {
  session_retention_days: number
  audit_retention_days: number
  policy: string
}

export interface ACPExecutionHealthRedactionSummary {
  detail_events_artifacts_redacted_views: boolean
  diagnostics_sanitized: boolean
  audit_metadata_sanitized: boolean
}

export interface ACPExecutionHealthSummaryResponse {
  timestamp: string
  range_days: number
  sessions: ACPExecutionHealthSessionSummary
  failure_buckets: ACPExecutionHealthFailureBuckets
  setup_health: ACPExecutionHealthSetupSummary
  agents: ACPExecutionHealthAgentSummary[]
  compatibility: ACPExecutionHealthCompatibilitySummary
  retention: ACPExecutionHealthRetentionSummary
  redaction: ACPExecutionHealthRedactionSummary
}

// -----------------------------------------------------------------------------
// MCP Server Configuration
// -----------------------------------------------------------------------------

export type ACPMCPServerType = "websocket" | "stdio"

export interface ACPMCPServerConfig {
  name: string
  type: ACPMCPServerType
  url?: string
  command?: string
  args?: string[]
  env?: Record<string, string>
}

// -----------------------------------------------------------------------------
// Permission Tiers
// -----------------------------------------------------------------------------

export type ACPPermissionTier = "auto" | "batch" | "individual"

// -----------------------------------------------------------------------------
// Session State
// -----------------------------------------------------------------------------

export type ACPSessionState =
  | "disconnected"
  | "connecting"
  | "connected"
  | "running"
  | "waiting_permission"
  | "error"

// -----------------------------------------------------------------------------
// WebSocket Message Types (Server → Client)
// -----------------------------------------------------------------------------

export interface ACPWSConnectedMessage {
  type: "connected"
  session_id: string
  agent_capabilities?: Record<string, unknown>
}

export interface ACPWSUpdateMessage {
  type: "update"
  session_id: string
  update_type: string
  data: Record<string, unknown>
}

export interface ACPWSPermissionRequestMessage {
  type: "permission_request"
  request_id: string
  session_id: string
  tool_name: string
  tool_arguments: Record<string, unknown>
  tier: ACPPermissionTier
  approval_requirement?: string | null
  governance_reason?: string | null
  deny_reason?: string | null
  provenance_summary?: Record<string, unknown> | null
  runtime_narrowing_reason?: string | null
  policy_snapshot_fingerprint?: string | null
  timeout_seconds: number
}

export interface ACPWSErrorMessage {
  type: "error"
  code: string
  message: string
  session_id?: string
  data?: Record<string, unknown>
}

export interface ACPWSPromptCompleteMessage {
  type: "prompt_complete"
  session_id: string
  stop_reason?: string
  raw_result: Record<string, unknown>
  usage?: ACPTokenUsage | null
}

export interface ACPWSDoneMessage {
  type: "done"
}

export type ACPWSServerMessage =
  | ACPWSConnectedMessage
  | ACPWSUpdateMessage
  | ACPWSPermissionRequestMessage
  | ACPWSErrorMessage
  | ACPWSPromptCompleteMessage
  | ACPWSDoneMessage

// -----------------------------------------------------------------------------
// WebSocket Message Types (Client → Server)
// -----------------------------------------------------------------------------

export interface ACPWSPermissionResponseMessage {
  type: "permission_response"
  request_id: string
  approved: boolean
  batch_approve_tier?: ACPPermissionTier
}

export interface ACPWSCancelMessage {
  type: "cancel"
  session_id: string
}

export interface ACPWSPromptMessage {
  type: "prompt"
  session_id: string
  prompt: Array<{
    role: "system" | "user" | "assistant"
    content: string
  }>
}

export type ACPWSClientMessage =
  | ACPWSPermissionResponseMessage
  | ACPWSCancelMessage
  | ACPWSPromptMessage

// -----------------------------------------------------------------------------
// REST API Types
// -----------------------------------------------------------------------------

export interface ACPSessionNewRequest {
  cwd: string
  name?: string
  agent_type?: ACPAgentType
  tags?: string[]
  mcp_servers?: ACPMCPServerConfig[]
  persona_id?: string | null
  workspace_id?: string | null
  workspace_group_id?: string | null
  scope_snapshot_id?: string | null
}

export interface ACPSessionNewResponse {
  session_id: string
  name: string
  agent_type: ACPAgentType
  agent_capabilities?: Record<string, unknown>
  sandbox_session_id?: string | null
  sandbox_run_id?: string | null
  ssh_ws_url?: string | null
  ssh_user?: string | null
  persona_id?: string | null
  workspace_id?: string | null
  workspace_group_id?: string | null
  scope_snapshot_id?: string | null
  policy_snapshot_version?: string | null
  policy_snapshot_fingerprint?: string | null
  policy_snapshot_refreshed_at?: string | null
  policy_summary?: Record<string, unknown> | null
  policy_provenance_summary?: Record<string, unknown> | null
  policy_refresh_error?: string | null
}

export interface ACPSessionPromptRequest {
  session_id: string
  prompt: Array<{
    role: "system" | "user" | "assistant"
    content: string
  }>
}

export interface ACPSessionPromptResponse {
  stop_reason?: string
  raw_result: Record<string, unknown>
  usage?: ACPTokenUsage | null
}

export interface ACPSessionCancelRequest {
  session_id: string
}

export interface ACPSessionCloseRequest {
  session_id: string
}

export interface ACPSessionUpdatesResponse {
  updates: Array<Record<string, unknown>>
}

// -----------------------------------------------------------------------------
// Session and Update Types
// -----------------------------------------------------------------------------

export interface ACPTokenUsage {
  prompt_tokens: number
  completion_tokens: number
  total_tokens: number
}

export type ACPBackendSessionStatus = "active" | "closed" | "error"

export interface ACPSessionListItem {
  session_id: string
  user_id: number
  agent_type: ACPAgentType
  name: string
  status: ACPBackendSessionStatus
  created_at: string
  last_activity_at?: string | null
  message_count: number
  usage: ACPTokenUsage
  tags: string[]
  has_websocket: boolean
  persona_id?: string | null
  workspace_id?: string | null
  workspace_group_id?: string | null
  scope_snapshot_id?: string | null
  policy_snapshot_version?: string | null
  policy_snapshot_fingerprint?: string | null
  policy_snapshot_refreshed_at?: string | null
  policy_summary?: Record<string, unknown> | null
  policy_provenance_summary?: Record<string, unknown> | null
  policy_refresh_error?: string | null
  forked_from?: string | null
}

export interface ACPSessionListResponse {
  sessions: ACPSessionListItem[]
  total: number
}

export interface ACPSessionDetailResponse extends ACPSessionListItem {
  messages: Array<Record<string, unknown>>
  cwd?: string | null
  fork_lineage: string[]
}

export interface ACPSessionUsageResponse {
  session_id: string
  user_id: number
  agent_type: ACPAgentType
  usage: ACPTokenUsage
  message_count: number
  created_at: string
  last_activity_at?: string | null
}

export interface ACPSessionForkRequest {
  message_index: number
  name?: string
}

export interface ACPSessionForkResponse {
  session_id: string
  name: string
  forked_from: string
  message_count: number
}

export interface ACPSession {
  id: string
  cwd: string
  name?: string
  forkParentSessionId?: string | null
  agentType?: ACPAgentType
  tags?: string[]
  mcpServers?: ACPMCPServerConfig[]
  personaId?: string | null
  workspaceId?: string | null
  workspaceGroupId?: string | null
  scopeSnapshotId?: string | null
  policySnapshotVersion?: string | null
  policySnapshotFingerprint?: string | null
  policySnapshotRefreshedAt?: Date | null
  policySummary?: Record<string, unknown> | null
  policyProvenanceSummary?: Record<string, unknown> | null
  policyRefreshError?: string | null
  state: ACPSessionState
  capabilities?: Record<string, unknown>
  sandboxSessionId?: string | null
  sandboxRunId?: string | null
  sshWsUrl?: string | null
  sshUser?: string | null
  backendStatus?: ACPBackendSessionStatus | null
  messageCount?: number
  usage?: ACPTokenUsage | null
  lastActivityAt?: Date | null
  updates: ACPUpdate[]
  pendingPermissions: ACPPendingPermission[]
  createdAt: Date
  updatedAt: Date
}

export interface ACPUpdate {
  timestamp: Date
  type: string
  data: Record<string, unknown>
}

/**
 * Discriminated union for parsed ACP update messages.
 * Provides type-safe access to message fields based on the `role` discriminant.
 */
export type ParsedUpdateMessage =
  | { role: "user"; content: string }
  | { role: "assistant"; content: string }
  | { role: "tool"; toolName: string; toolArgs: Record<string, unknown> }
  | { role: "tool_result"; toolName: string; result: unknown }
  | { role: "error"; content: string }
  | { role: "system"; content: string }

export interface ACPPendingPermission {
  request_id: string
  tool_name: string
  tool_arguments: Record<string, unknown>
  tier: ACPPermissionTier
  approval_requirement?: string | null
  governance_reason?: string | null
  deny_reason?: string | null
  provenance_summary?: Record<string, unknown> | null
  runtime_narrowing_reason?: string | null
  policy_snapshot_fingerprint?: string | null
  timeout_seconds: number
  requestedAt: Date
}

// -----------------------------------------------------------------------------
// Agent Capabilities
// -----------------------------------------------------------------------------

export interface ACPAgentCapabilities {
  fs?: {
    readTextFile?: boolean
    writeTextFile?: boolean
  }
  terminal?: boolean
  tools?: string[]
}

// -----------------------------------------------------------------------------
// Callbacks for useACPSession hook
// -----------------------------------------------------------------------------

export interface ACPSessionCallbacks {
  onStateChange?: (state: ACPSessionState) => void
  onUpdate?: (update: ACPWSUpdateMessage) => void
  onPermissionRequest?: (request: ACPWSPermissionRequestMessage) => void
  onPromptComplete?: (result: ACPWSPromptCompleteMessage) => void
  onError?: (error: ACPWSErrorMessage) => void
  onConnected?: (message: ACPWSConnectedMessage) => void
  onDisconnected?: () => void
}

// -----------------------------------------------------------------------------
// Structured Error Types
// -----------------------------------------------------------------------------

export interface ACPErrorSuggestion {
  action: string
  description?: string
}

export interface ACPStructuredError {
  code: string
  message: string
  suggestions: ACPErrorSuggestion[]
  data?: Record<string, unknown>
}

// -----------------------------------------------------------------------------
// ACP Health & Agent Registry
// -----------------------------------------------------------------------------

export interface ACPHealthResponse {
  runner: "ok" | "missing" | "error"
  agent: "ok" | "missing" | "error"
  api_keys: "ok" | "missing"
  details?: string
}

export interface ACPAgentRegistryEntry {
  type: string
  name: string
  description: string
  status: "available" | "unavailable" | "requires_setup"
  reason?: string
  is_default?: boolean
}

// -----------------------------------------------------------------------------
// Orchestration Types
// -----------------------------------------------------------------------------

export type OrchestrationTaskStatus = "todo" | "inprogress" | "review" | "complete" | "triage"
export type OrchestrationRunStatus = "running" | "completed" | "failed"

export interface OrchestrationProject {
  id: number
  name: string
  description?: string
  user_id: number
  created_at: string
  task_summary?: {
    total_tasks: number
    status_counts: Record<string, number>
  }
}

export interface OrchestrationTask {
  id: number
  project_id: number
  title: string
  description?: string
  status: OrchestrationTaskStatus
  agent_type?: string
  dependency_id?: number | null
  review_count: number
  max_review_attempts: number
  created_at: string
  updated_at: string
}

export interface OrchestrationRun {
  id: number
  task_id: number
  session_id?: string
  agent_type?: string
  status: OrchestrationRunStatus
  result_summary?: string
  error?: string
  started_at: string
  completed_at?: string
}

// -----------------------------------------------------------------------------
// Session Creation Progress
// -----------------------------------------------------------------------------

export type ACPSessionCreationStep =
  | "idle"
  | "creating"
  | "starting_agent"
  | "connecting"
  | "ready"
  | "error"

export interface ACPSessionCreationState {
  step: ACPSessionCreationStep
  error?: ACPStructuredError
}
