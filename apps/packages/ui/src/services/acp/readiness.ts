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
