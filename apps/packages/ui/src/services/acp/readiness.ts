import type {
  ACPExecutionHealthAgentSummary,
  ACPExecutionHealthCompatibilitySummary,
  ACPExecutionHealthFailureBuckets,
  ACPExecutionHealthRedactionSummary,
  ACPExecutionHealthRetentionSummary,
  ACPExecutionHealthSessionSummary,
  ACPExecutionHealthSetupDimension,
  ACPExecutionHealthSetupSummary,
  ACPExecutionHealthSummaryResponse,
  ACPSetupHealthStatus,
  ACPSupportState,
  ACPVerificationLevel
} from "./types"

export type ACPHealthStatus = {
  runner: string
  agent: string
  api_keys: string
  details?: string
}

export type ACPSetupIssue = {
  code: string
  title: string
  description: string
}

const OK_STATUSES = new Set(["ok", "available", "healthy", "degraded"])

const ACP_SETUP_HEALTH_STATUSES: ReadonlySet<ACPSetupHealthStatus> = new Set([
  "ok",
  "degraded",
  "blocked",
  "unknown"
])

const ACP_SUPPORT_STATES: ReadonlySet<ACPSupportState> = new Set([
  "supported",
  "supported_with_caveats",
  "experimental",
  "documented_unverified",
  "unsupported"
])

const ACP_VERIFICATION_LEVELS: ReadonlySet<ACPVerificationLevel> = new Set([
  "documented_only",
  "stub_smoke_tested",
  "live_e2e_tested",
  "sandbox_tested",
  "production_supported"
])

const ACP_COMPATIBILITY_DOCS_URL = "/docs-static/Development/ACP_Compatibility_Matrix.md"

const FAILURE_BUCKET_KEYS: Array<keyof ACPExecutionHealthFailureBuckets> = [
  "setup_blockers",
  "runner_session_failures",
  "reviewer_rejections",
  "reviewer_failures",
  "governance_denials",
  "structured_completion_failures",
  "sandbox_runtime_errors",
  "retention_redaction_actions"
]

const SETUP_DIMENSION_KEYS: Array<keyof ACPExecutionHealthSetupSummary> = [
  "agent",
  "workspace",
  "sandbox_runtime",
  "mcp_injection",
  "scheduler_trigger_path"
]

const isRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === "object" && !Array.isArray(value)

const numberOrDefault = (value: unknown, fallback = 0): number =>
  typeof value === "number" && Number.isFinite(value) ? Math.max(0, Math.trunc(value)) : fallback

const booleanOrDefault = (value: unknown, fallback = false): boolean =>
  typeof value === "boolean" ? value : fallback

const stringOrDefault = (value: unknown, fallback = ""): string =>
  typeof value === "string" ? value : fallback

const stringListOrDefault = (value: unknown): string[] =>
  Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : []

const numberRecordOrDefault = (
  value: unknown,
  allowedKeys?: ReadonlySet<string>
): Record<string, number> => {
  if (!isRecord(value)) {
    return {}
  }

  const counts: Record<string, number> = {}
  for (const [key, count] of Object.entries(value)) {
    if (allowedKeys && !allowedKeys.has(key)) {
      continue
    }
    if (typeof count === "number" && Number.isFinite(count)) {
      counts[key] = Math.max(0, Math.trunc(count))
    }
  }
  return counts
}

const normalizeSetupHealthStatus = (value: unknown): ACPSetupHealthStatus =>
  typeof value === "string" && ACP_SETUP_HEALTH_STATUSES.has(value as ACPSetupHealthStatus)
    ? (value as ACPSetupHealthStatus)
    : "unknown"

const normalizeSupportState = (value: unknown): ACPSupportState =>
  typeof value === "string" && ACP_SUPPORT_STATES.has(value as ACPSupportState)
    ? (value as ACPSupportState)
    : "documented_unverified"

const normalizeVerificationLevel = (value: unknown): ACPVerificationLevel =>
  typeof value === "string" && ACP_VERIFICATION_LEVELS.has(value as ACPVerificationLevel)
    ? (value as ACPVerificationLevel)
    : "documented_only"

const formatHealthDetails = (
  message: unknown,
  runner: Record<string, unknown> | null,
  availableAgents: number,
  totalAgents: number
): string | undefined => {
  const parts: string[] = []
  if (typeof message === "string" && message.trim().length > 0) {
    parts.push(message.trim())
  }
  if (runner) {
    const source = typeof runner.source === "string" ? runner.source : null
    const path = typeof runner.path === "string" ? runner.path : null
    const runnerParts = [
      "Runner",
      source ? `source ${source}` : null,
      path ? `path ${path}` : null
    ].filter((part): part is string => Boolean(part))
    if (runnerParts.length > 1) {
      parts.push(runnerParts.join(" "))
    }
  }
  if (totalAgents > 0) {
    parts.push(`${availableAgents}/${totalAgents} agents available`)
  }
  return parts.length > 0 ? parts.join(" • ") : undefined
}

export const normalizeACPHealthStatus = (payload: unknown): ACPHealthStatus | null => {
  if (!payload || typeof payload !== "object") {
    return null
  }

  const record = payload as Record<string, unknown>
  if (
    typeof record.runner === "string" &&
    typeof record.agent === "string" &&
    typeof record.api_keys === "string"
  ) {
    return {
      runner: record.runner,
      agent: record.agent,
      api_keys: record.api_keys,
      details: typeof record.details === "string" ? record.details : undefined
    }
  }

  const runner =
    record.runner && typeof record.runner === "object" && !Array.isArray(record.runner)
      ? (record.runner as Record<string, unknown>)
      : null
  const agents = Array.isArray(record.agents)
    ? record.agents.filter(
        (agent): agent is Record<string, unknown> =>
          Boolean(agent) && typeof agent === "object" && !Array.isArray(agent)
      )
    : []
  const availableAgents = agents.filter((agent) => agent.status === "available").length
  const missingApiKeys = agents.some((agent) => agent.api_key_set === false)

  return {
    runner: typeof runner?.status === "string" ? runner.status : "unknown",
    agent:
      agents.length === 0
        ? "unavailable"
        : availableAgents === 0
          ? "unavailable"
          : availableAgents === agents.length
            ? "available"
            : "degraded",
    api_keys: missingApiKeys ? "missing" : "ok",
    details: formatHealthDetails(record.message, runner, availableAgents, agents.length)
  }
}

const normalizeExecutionHealthSessions = (
  value: unknown
): ACPExecutionHealthSessionSummary => {
  const record = isRecord(value) ? value : {}
  return {
    total: numberOrDefault(record.total),
    by_status: numberRecordOrDefault(record.by_status)
  }
}

const normalizeExecutionHealthFailureBuckets = (
  value: unknown
): ACPExecutionHealthFailureBuckets => {
  const record = isRecord(value) ? value : {}
  return Object.fromEntries(
    FAILURE_BUCKET_KEYS.map((key) => [key, numberOrDefault(record[key])])
  ) as unknown as ACPExecutionHealthFailureBuckets
}

const normalizeExecutionHealthSetupDimension = (
  value: unknown
): ACPExecutionHealthSetupDimension => {
  const record = isRecord(value) ? value : {}
  return {
    status: normalizeSetupHealthStatus(record.status),
    blockers: stringListOrDefault(record.blockers),
    evidence_count: numberOrDefault(record.evidence_count)
  }
}

const normalizeExecutionHealthSetupSummary = (
  value: unknown
): ACPExecutionHealthSetupSummary => {
  const record = isRecord(value) ? value : {}
  return Object.fromEntries(
    SETUP_DIMENSION_KEYS.map((key) => [
      key,
      normalizeExecutionHealthSetupDimension(record[key])
    ])
  ) as unknown as ACPExecutionHealthSetupSummary
}

const normalizeExecutionHealthAgent = (
  value: unknown
): ACPExecutionHealthAgentSummary | null => {
  if (!isRecord(value)) {
    return null
  }

  const agentType = stringOrDefault(value.agent_type)
  if (!agentType) {
    return null
  }

  return {
    agent_type: agentType,
    name: stringOrDefault(value.name),
    is_configured: booleanOrDefault(value.is_configured),
    support_state: normalizeSupportState(value.support_state),
    verification_level: normalizeVerificationLevel(value.verification_level),
    setup_blocked: booleanOrDefault(value.setup_blocked),
    primary_blocker:
      typeof value.primary_blocker === "string" ? value.primary_blocker : null
  }
}

const normalizeExecutionHealthAgents = (
  value: unknown
): ACPExecutionHealthAgentSummary[] =>
  Array.isArray(value)
    ? value
        .map(normalizeExecutionHealthAgent)
        .filter((agent): agent is ACPExecutionHealthAgentSummary => Boolean(agent))
    : []

const normalizeExecutionHealthCompatibility = (
  value: unknown
): ACPExecutionHealthCompatibilitySummary => {
  const record = isRecord(value) ? value : {}
  return {
    by_support_state: numberRecordOrDefault(record.by_support_state, ACP_SUPPORT_STATES),
    documented_unverified_agents: stringListOrDefault(record.documented_unverified_agents),
    live_certification_required: booleanOrDefault(record.live_certification_required),
    docs_url: stringOrDefault(record.docs_url, ACP_COMPATIBILITY_DOCS_URL)
  }
}

const normalizeExecutionHealthRetention = (
  value: unknown
): ACPExecutionHealthRetentionSummary => {
  const record = isRecord(value) ? value : {}
  return {
    session_retention_days: numberOrDefault(record.session_retention_days, 30),
    audit_retention_days: numberOrDefault(record.audit_retention_days, 30),
    policy: stringOrDefault(
      record.policy,
      "closed_error_sessions_and_audit_events_purged_after_retention"
    )
  }
}

const normalizeExecutionHealthRedaction = (
  value: unknown
): ACPExecutionHealthRedactionSummary => {
  const record = isRecord(value) ? value : {}
  return {
    detail_events_artifacts_redacted_views: booleanOrDefault(
      record.detail_events_artifacts_redacted_views
    ),
    diagnostics_sanitized: booleanOrDefault(record.diagnostics_sanitized),
    audit_metadata_sanitized: booleanOrDefault(record.audit_metadata_sanitized)
  }
}

export const normalizeACPExecutionHealthSummary = (
  payload: unknown
): ACPExecutionHealthSummaryResponse | null => {
  if (!isRecord(payload)) {
    return null
  }
  if (
    typeof payload.timestamp !== "string" ||
    payload.timestamp.trim().length === 0 ||
    typeof payload.range_days !== "number" ||
    !Number.isFinite(payload.range_days) ||
    payload.range_days < 0
  ) {
    return null
  }

  return {
    timestamp: stringOrDefault(payload.timestamp),
    range_days: numberOrDefault(payload.range_days),
    sessions: normalizeExecutionHealthSessions(payload.sessions),
    failure_buckets: normalizeExecutionHealthFailureBuckets(payload.failure_buckets),
    setup_health: normalizeExecutionHealthSetupSummary(payload.setup_health),
    agents: normalizeExecutionHealthAgents(payload.agents),
    compatibility: normalizeExecutionHealthCompatibility(payload.compatibility),
    retention: normalizeExecutionHealthRetention(payload.retention),
    redaction: normalizeExecutionHealthRedaction(payload.redaction)
  }
}

export const buildACPSetupIssues = (
  health: ACPHealthStatus | null,
  healthError?: string
): ACPSetupIssue[] => {
  if (!health) {
    return [
      {
        code: "acp_health_unavailable",
        title: "ACP health check unavailable",
        description: healthError || "The server did not return ACP health details."
      }
    ]
  }

  const issues: ACPSetupIssue[] = []
  if (!OK_STATUSES.has(String(health.runner || "").toLowerCase())) {
    issues.push({
      code: "runner_unavailable",
      title: health.runner === "missing" ? "Runner is missing" : "Runner is unavailable",
      description: health.details || "Install or configure the ACP runner binary before dispatching tasks."
    })
  }
  if (!OK_STATUSES.has(String(health.agent || "").toLowerCase())) {
    issues.push({
      code: "agent_unavailable",
      title: "Agent configuration is incomplete",
      description: "No configured ACP agent is ready to accept a task run."
    })
  }
  if (String(health.api_keys || "").toLowerCase() === "missing") {
    issues.push({
      code: "api_keys_missing",
      title: "API keys are missing",
      description: "Configure the required provider keys for the selected ACP agent."
    })
  }
  return issues
}
