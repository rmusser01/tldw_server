import type {
  McpHubExternalServer,
  McpHubToolRegistryEntry
} from "@/services/tldw/mcp-hub"

export type McpReadinessAction =
  | "add_server"
  | "edit_config"
  | "open_credentials"
  | "refresh_discovery"
  | "validate"
  | "view_details"
  | "open_tool_catalog"
  | "open_audit"

export type McpDisplayState =
  | "needs_setup"
  | "checking"
  | "ready"
  | "needs_attention"
  | "no_tools"
  | "stale"

export type McpReasonCode =
  | "not_configured"
  | "preflight_failed"
  | "discovery_not_run"
  | "auth_missing"
  | "runtime_unavailable"
  | "unreachable"
  | "discovery_failed"
  | "no_tools_returned"
  | "config_changed"
  | "catalog_expired"
  | "partial_capability"

export type McpCredentialState =
  | "not_required"
  | "required_missing"
  | "configured"
  | "legacy_fallback"
  | "unknown"

export type McpCurrentOperationHint =
  | string
  | {
      operation?: string
      label?: string
      startedAt?: string | null
      serverId?: string | null
    }

export type McpServerReadinessHint = {
  currentOperation?: McpCurrentOperationHint | null
  preflightFailed?: boolean
  discoverySucceededWithNoTools?: boolean
  unreachable?: boolean
  discoveryFailed?: boolean
  configChanged?: boolean
  catalogExpired?: boolean
  partialCapability?: boolean
}

export type McpServerReadiness = {
  serverId: string
  serverName: string
  displayState: McpDisplayState
  credentialState: McpCredentialState
  toolCount: number
  reasonCodes: McpReasonCode[]
  primaryReasonCode: McpReasonCode | undefined
  allowedActions: McpReadinessAction[]
  message: string
  currentOperation?: McpCurrentOperationHint
}

export type McpHubReadiness = {
  displayState: McpDisplayState
  reasonCodes: McpReasonCode[]
  primaryReasonCode: McpReasonCode | undefined
  allowedActions: McpReadinessAction[]
  message: string
  servers: McpServerReadiness[]
  totalServers: number
  readyServerCount: number
  checkingServerCount: number
  attentionServerCount: number
  noToolServerCount: number
  staleServerCount: number
}

type ReadinessHintsByServerId =
  | Record<string, McpServerReadinessHint | undefined>
  | Map<string, McpServerReadinessHint>

const REASON_PRIORITY: Record<McpReasonCode, number> = {
  not_configured: 0,
  auth_missing: 10,
  runtime_unavailable: 20,
  preflight_failed: 30,
  unreachable: 40,
  discovery_failed: 50,
  config_changed: 60,
  discovery_not_run: 70,
  no_tools_returned: 80,
  catalog_expired: 90,
  partial_capability: 100
}

const REASON_ACTIONS: Record<McpReasonCode, McpReadinessAction[]> = {
  not_configured: ["add_server"],
  preflight_failed: ["edit_config", "validate", "view_details"],
  discovery_not_run: ["refresh_discovery", "edit_config"],
  auth_missing: ["open_credentials", "view_details"],
  runtime_unavailable: ["edit_config", "view_details"],
  unreachable: ["edit_config", "refresh_discovery", "view_details"],
  discovery_failed: ["refresh_discovery", "view_details"],
  no_tools_returned: ["refresh_discovery", "view_details"],
  config_changed: ["refresh_discovery", "edit_config"],
  catalog_expired: ["refresh_discovery", "view_details"],
  partial_capability: ["open_tool_catalog", "view_details"]
}

const SERVER_READY_ACTIONS: McpReadinessAction[] = ["open_tool_catalog", "view_details"]
const HUB_READY_ACTIONS: McpReadinessAction[] = ["open_tool_catalog"]
const CHECKING_ACTIONS: McpReadinessAction[] = ["view_details"]

const sortReasonCodes = (reasonCodes: McpReasonCode[]): McpReasonCode[] =>
  [...new Set(reasonCodes)].sort(
    (left, right) =>
      REASON_PRIORITY[left] - REASON_PRIORITY[right] || left.localeCompare(right)
  )

const uniqueActions = (actions: McpReadinessAction[]): McpReadinessAction[] => {
  const seenActions = new Set<McpReadinessAction>()

  return actions.filter((action) => {
    if (seenActions.has(action)) {
      return false
    }

    seenActions.add(action)
    return true
  })
}

const getActionsForReasons = (
  reasonCodes: McpReasonCode[],
  fallbackActions: McpReadinessAction[] = []
): McpReadinessAction[] => {
  if (reasonCodes.length === 0) {
    return uniqueActions(fallbackActions)
  }

  return uniqueActions(reasonCodes.flatMap((reasonCode) => REASON_ACTIONS[reasonCode]))
}

const getDisplayStateForReason = (
  primaryReasonCode: McpReasonCode | undefined
): McpDisplayState => {
  switch (primaryReasonCode) {
    case undefined:
    case "partial_capability":
      return "ready"
    case "not_configured":
      return "needs_setup"
    case "config_changed":
    case "catalog_expired":
      return "stale"
    case "no_tools_returned":
      return "no_tools"
    case "discovery_not_run":
    case "preflight_failed":
    case "auth_missing":
    case "runtime_unavailable":
    case "unreachable":
    case "discovery_failed":
      return "needs_attention"
  }
}

const getOperationLabel = (currentOperation: McpCurrentOperationHint): string => {
  if (typeof currentOperation === "string") {
    return currentOperation
  }

  return currentOperation.label ?? currentOperation.operation ?? "Operation"
}

const getCredentialMessage = (credentialState: McpCredentialState): string => {
  switch (credentialState) {
    case "not_required":
      return " No credentials required."
    case "required_missing":
      return " Required credentials are missing."
    case "configured":
      return " Credentials are configured."
    case "legacy_fallback":
      return " Legacy server-level secret fallback is configured."
    case "unknown":
      return ""
  }
}

const getMessageForServer = ({
  displayState,
  primaryReasonCode,
  credentialState,
  toolCount,
  currentOperation
}: {
  displayState: McpDisplayState
  primaryReasonCode: McpReasonCode | undefined
  credentialState: McpCredentialState
  toolCount: number
  currentOperation?: McpCurrentOperationHint
}): string => {
  if (displayState === "checking" && currentOperation) {
    return `${getOperationLabel(currentOperation)} is in progress.${getCredentialMessage(
      credentialState
    )}`
  }

  const credentialMessage = getCredentialMessage(credentialState)

  switch (primaryReasonCode) {
    case undefined:
      return `Server is ready with ${toolCount} available tool${toolCount === 1 ? "" : "s"}.${credentialMessage}`
    case "not_configured":
      return "Add an MCP server to begin setup."
    case "preflight_failed":
      return `Preflight validation failed.${credentialMessage}`
    case "auth_missing":
      return `Credentials are required before this server can be used.${credentialMessage}`
    case "runtime_unavailable":
      return `The configured runtime is not available.${credentialMessage}`
    case "unreachable":
      return `The server is unreachable.${credentialMessage}`
    case "discovery_failed":
      return `Tool discovery failed.${credentialMessage}`
    case "config_changed":
      return `Configuration changed after the last discovery run.${credentialMessage}`
    case "catalog_expired":
      return `The tool catalog is expired.${credentialMessage}`
    case "discovery_not_run":
      return `Run discovery to populate this server's tool catalog.${credentialMessage}`
    case "no_tools_returned":
      return `Discovery completed, but the server returned no tools.${credentialMessage}`
    case "partial_capability":
      return `Server is ready, but some capabilities need review.${credentialMessage}`
  }
}

const getMessageForHub = ({
  displayState,
  primaryReasonCode,
  totalServers,
  readyServerCount,
  currentOperation
}: {
  displayState: McpDisplayState
  primaryReasonCode: McpReasonCode | undefined
  totalServers: number
  readyServerCount: number
  currentOperation?: McpCurrentOperationHint
}): string => {
  if (displayState === "checking" && currentOperation) {
    return `${getOperationLabel(currentOperation)} is in progress.`
  }

  switch (primaryReasonCode) {
    case undefined:
    case "partial_capability":
      return `${readyServerCount} of ${totalServers} MCP server${
        totalServers === 1 ? "" : "s"
      } ready.`
    case "not_configured":
      return "Add an MCP server to begin setup."
    case "preflight_failed":
      return "One or more MCP servers failed preflight validation."
    case "auth_missing":
      return "One or more MCP servers are missing required credentials."
    case "runtime_unavailable":
      return "One or more MCP server runtimes are unavailable."
    case "unreachable":
      return "One or more MCP servers are unreachable."
    case "discovery_failed":
      return "One or more MCP servers failed tool discovery."
    case "config_changed":
      return "One or more MCP server configurations changed after discovery."
    case "catalog_expired":
      return "One or more MCP tool catalogs are expired."
    case "discovery_not_run":
      return "Run discovery to populate MCP tool catalogs."
    case "no_tools_returned":
      return "Discovery completed, but one or more MCP servers returned no tools."
  }
}

const countMatchingTools = (
  server: McpHubExternalServer,
  registryEntries: McpHubToolRegistryEntry[]
): number => {
  const externalModule = `external.${server.id}`
  const externalToolPrefix = `ext.${server.id}.`

  return registryEntries.filter((entry) => {
    const hasExternalToolName = entry.tool_name.startsWith(externalToolPrefix)

    // Built-in modules can share ids with external servers; require an external
    // namespace signal before counting a registry row for server readiness.
    return hasExternalToolName || entry.module === externalModule
  }).length
}

const getReadinessHint = (
  readinessHintsByServerId: ReadinessHintsByServerId | undefined,
  serverId: string
): McpServerReadinessHint | undefined => {
  if (!readinessHintsByServerId) {
    return undefined
  }

  if (readinessHintsByServerId instanceof Map) {
    return readinessHintsByServerId.get(serverId)
  }

  return readinessHintsByServerId[serverId]
}

const getAuthTemplateBlockedReason = (
  server: McpHubExternalServer
): string | undefined => {
  const blockedReason = server.auth_template_blocked_reason?.trim()

  return blockedReason || undefined
}

const allowsNoAuthTemplate = (server: McpHubExternalServer): boolean => {
  const blockedReason = getAuthTemplateBlockedReason(server)

  return !blockedReason || blockedReason === "no_auth_template"
}

const hasBlockingAuthTemplateIssue = (server: McpHubExternalServer): boolean => {
  return (
    (server.auth_template_present === true && server.auth_template_valid === false) ||
    !allowsNoAuthTemplate(server)
  )
}

const isOperationalManagedServer = (server: McpHubExternalServer): boolean =>
  server.enabled !== false &&
  server.server_source !== "legacy" &&
  !server.superseded_by_server_id

export const getMcpCredentialState = (
  server: McpHubExternalServer
): McpCredentialState => {
  const credentialSlots = server.credential_slots ?? []
  const hasAuthTemplate = server.auth_template_present === true

  if (credentialSlots.some((slot) => slot.is_required && !slot.secret_configured)) {
    return "required_missing"
  }

  if (
    credentialSlots.some((slot) => slot.secret_configured) ||
    server.auth_template_valid === true
  ) {
    return "configured"
  }

  if (server.secret_configured && !hasAuthTemplate && credentialSlots.length === 0) {
    return "legacy_fallback"
  }

  if (
    server.transport === "stdio" &&
    !hasAuthTemplate &&
    credentialSlots.length === 0 &&
    allowsNoAuthTemplate(server)
  ) {
    return "not_required"
  }

  return "unknown"
}

export const getMcpServerReadiness = ({
  server,
  registryEntries,
  readinessHint
}: {
  server: McpHubExternalServer
  registryEntries: McpHubToolRegistryEntry[]
  readinessHint?: McpServerReadinessHint
}): McpServerReadiness => {
  const credentialState = getMcpCredentialState(server)
  const toolCount = countMatchingTools(server, registryEntries)
  const unsortedReasonCodes: McpReasonCode[] = []

  if (readinessHint?.preflightFailed || hasBlockingAuthTemplateIssue(server)) {
    unsortedReasonCodes.push("preflight_failed")
  }

  if (credentialState === "required_missing") {
    unsortedReasonCodes.push("auth_missing")
  }

  if (server.enabled === false || server.runtime_executable === false) {
    unsortedReasonCodes.push("runtime_unavailable")
  }

  if (readinessHint?.unreachable) {
    unsortedReasonCodes.push("unreachable")
  }

  if (readinessHint?.discoveryFailed) {
    unsortedReasonCodes.push("discovery_failed")
  }

  if (readinessHint?.configChanged) {
    unsortedReasonCodes.push("config_changed")
  }

  if (readinessHint?.catalogExpired) {
    unsortedReasonCodes.push("catalog_expired")
  }

  if (toolCount === 0) {
    unsortedReasonCodes.push(
      readinessHint?.discoverySucceededWithNoTools
        ? "no_tools_returned"
        : "discovery_not_run"
    )
  }

  if (readinessHint?.partialCapability && toolCount > 0) {
    unsortedReasonCodes.push("partial_capability")
  }

  const reasonCodes = sortReasonCodes(unsortedReasonCodes)
  const primaryReasonCode = reasonCodes[0]
  const currentOperation = readinessHint?.currentOperation ?? undefined
  const displayState = currentOperation
    ? "checking"
    : getDisplayStateForReason(primaryReasonCode)
  const allowedActions = currentOperation
    ? CHECKING_ACTIONS
    : getActionsForReasons(reasonCodes, SERVER_READY_ACTIONS)

  return {
    serverId: server.id,
    serverName: server.name,
    displayState,
    credentialState,
    toolCount,
    reasonCodes,
    primaryReasonCode,
    allowedActions,
    message: getMessageForServer({
      displayState,
      primaryReasonCode,
      credentialState,
      toolCount,
      currentOperation
    }),
    ...(currentOperation ? { currentOperation } : {})
  }
}

export const getMcpHubReadiness = ({
  servers,
  registryEntries,
  readinessHintsByServerId
}: {
  servers: McpHubExternalServer[]
  registryEntries: McpHubToolRegistryEntry[]
  readinessHintsByServerId?: ReadinessHintsByServerId
}): McpHubReadiness => {
  const operationalServers = servers.filter(isOperationalManagedServer)

  if (operationalServers.length === 0) {
    const reasonCodes = sortReasonCodes(["not_configured"])
    const primaryReasonCode = reasonCodes[0]

    return {
      displayState: "needs_setup",
      reasonCodes,
      primaryReasonCode,
      allowedActions: getActionsForReasons(reasonCodes),
      message: getMessageForHub({
        displayState: "needs_setup",
        primaryReasonCode,
        totalServers: 0,
        readyServerCount: 0
      }),
      servers: [],
      totalServers: 0,
      readyServerCount: 0,
      checkingServerCount: 0,
      attentionServerCount: 0,
      noToolServerCount: 0,
      staleServerCount: 0
    }
  }

  const serverReadiness = operationalServers.map((server) =>
    getMcpServerReadiness({
      server,
      registryEntries,
      readinessHint: getReadinessHint(readinessHintsByServerId, server.id)
    })
  )
  const reasonCodes = sortReasonCodes(
    serverReadiness.flatMap((readiness) => readiness.reasonCodes)
  )
  const primaryReasonCode = reasonCodes[0]
  const currentOperation = serverReadiness.find(
    (readiness) => readiness.currentOperation
  )?.currentOperation
  const displayState = currentOperation
    ? "checking"
    : getDisplayStateForReason(primaryReasonCode)
  const readyServerCount = serverReadiness.filter(
    (readiness) => readiness.displayState === "ready"
  ).length

  return {
    displayState,
    reasonCodes,
    primaryReasonCode,
    allowedActions: currentOperation
      ? CHECKING_ACTIONS
      : getActionsForReasons(reasonCodes, HUB_READY_ACTIONS),
    message: getMessageForHub({
      displayState,
      primaryReasonCode,
      totalServers: operationalServers.length,
      readyServerCount,
      currentOperation
    }),
    servers: serverReadiness,
    totalServers: operationalServers.length,
    readyServerCount,
    checkingServerCount: serverReadiness.filter(
      (readiness) => readiness.displayState === "checking"
    ).length,
    attentionServerCount: serverReadiness.filter(
      (readiness) => readiness.displayState === "needs_attention"
    ).length,
    noToolServerCount: serverReadiness.filter(
      (readiness) => readiness.displayState === "no_tools"
    ).length,
    staleServerCount: serverReadiness.filter(
      (readiness) => readiness.displayState === "stale"
    ).length
  }
}
