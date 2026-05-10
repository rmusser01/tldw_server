export type ChatToolRecord = Record<string, unknown>

export type ResolvedMcpTool = {
  tool: ChatToolRecord
  rawName: string
  chatName: string
  displayName: string
  description?: string
  groupLabel: string
  canExecute: boolean
  disabled: boolean
  colliding: boolean
}

export type ChatToolFilterCounts = {
  discovered: number
  executable: number
  disabled: number
  colliding: number
  chatEnabled: number
}

export type ChatToolFilterState = {
  discoveredTools: ResolvedMcpTool[]
  availableTools: ResolvedMcpTool[]
  chatTools: ResolvedMcpTool[]
  collisionToolNames: string[]
  counts: ChatToolFilterCounts
}

export type ChatToolRequestChoice = "auto" | "none" | "required"
export type EffectiveChatToolRequestChoice = "auto" | "required"

export type ChatToolOmissionReason =
  | "tool_choice_none"
  | "model_lacks_tool_capability"
  | "mcp_absent"
  | "mcp_unavailable"
  | "mcp_unhealthy"
  | "no_enabled_executable_tools"
  | "no_normalized_request_tools"

export type ResolvedChatToolRequest = {
  tools?: ChatToolRecord[]
  toolChoice?: EffectiveChatToolRequestChoice
  omittedReason?: ChatToolOmissionReason
  counts: ChatToolFilterCounts
}

const TOOL_NAME_PATTERN = /^[a-zA-Z0-9_-]{1,64}$/

const isRecord = (value: unknown): value is ChatToolRecord =>
  typeof value === "object" && value !== null && !Array.isArray(value)

const unwrapToolCandidate = (tool: unknown): unknown => {
  if (!isRecord(tool)) return tool
  if (
    isRecord(tool.tool) &&
    typeof tool.rawName === "string" &&
    typeof tool.chatName === "string"
  ) {
    return tool.tool
  }
  return tool
}

export const normalizeChatToolName = (name: unknown): string | null => {
  if (typeof name !== "string") return null
  const trimmed = name.trim()
  if (!trimmed) return null
  if (TOOL_NAME_PATTERN.test(trimmed)) return trimmed

  let sanitized = trimmed.replace(/[^a-zA-Z0-9_-]+/g, "_")
  sanitized = sanitized.replace(/^_+|_+$/g, "")
  if (!sanitized) return null
  if (sanitized.length > 64) sanitized = sanitized.slice(0, 64)
  return TOOL_NAME_PATTERN.test(sanitized) ? sanitized : null
}

const readFunctionRecord = (tool: ChatToolRecord): ChatToolRecord | undefined =>
  isRecord(tool.function) ? tool.function : undefined

const readRawToolName = (tool: ChatToolRecord): string | null => {
  if (typeof tool.name === "string" && tool.name.trim()) {
    return tool.name.trim()
  }
  const functionRecord = readFunctionRecord(tool)
  if (typeof functionRecord?.name === "string" && functionRecord.name.trim()) {
    return functionRecord.name.trim()
  }
  return null
}

const readDescription = (tool: ChatToolRecord): string | undefined => {
  if (typeof tool.description === "string" && tool.description.trim()) {
    return tool.description
  }
  const functionRecord = readFunctionRecord(tool)
  if (
    typeof functionRecord?.description === "string" &&
    functionRecord.description.trim()
  ) {
    return functionRecord.description
  }
  return undefined
}

export const getMcpToolGroupLabel = (tool: ChatToolRecord): string => {
  const metadata = isRecord(tool.metadata) ? tool.metadata : undefined
  const serverName =
    typeof metadata?.server_name === "string" ? metadata.server_name.trim() : ""
  if (serverName) return serverName

  const serverId =
    typeof metadata?.server_id === "string" ? metadata.server_id.trim() : ""
  if (serverId) return serverId

  const rawName = readRawToolName(tool)
  if (rawName?.startsWith("ext.")) {
    const [, extServerId] = rawName.split(".")
    if (extServerId?.trim()) return extServerId.trim()
  }

  const moduleId = typeof tool.module === "string" ? tool.module.trim() : ""
  return moduleId || "MCP"
}

export const resolveMcpToolIdentity = (
  tool: unknown
): Omit<ResolvedMcpTool, "disabled" | "colliding"> | null => {
  if (!isRecord(tool)) return null
  const rawName = readRawToolName(tool)
  const chatName = normalizeChatToolName(rawName)
  if (!rawName || !chatName) return null
  return {
    tool,
    rawName,
    chatName,
    displayName: rawName,
    description: readDescription(tool),
    groupLabel: getMcpToolGroupLabel(tool),
    canExecute: tool.canExecute !== false
  }
}

const normalizeDisabledSet = (disabledToolNames: Iterable<string>): Set<string> => {
  const result = new Set<string>()
  for (const name of disabledToolNames) {
    const normalized = normalizeChatToolName(name)
    if (normalized) result.add(normalized)
  }
  return result
}

export const buildChatToolFilterState = ({
  tools,
  disabledToolNames = []
}: {
  tools?: unknown[]
  disabledToolNames?: string[]
}): ChatToolFilterState => {
  const disabledSet = normalizeDisabledSet(disabledToolNames)
  const discoveredBase = (Array.isArray(tools) ? tools : [])
    .map(unwrapToolCandidate)
    .map(resolveMcpToolIdentity)
    .filter(Boolean) as Array<Omit<ResolvedMcpTool, "disabled" | "colliding">>

  const nameCounts = new Map<string, number>()
  for (const tool of discoveredBase) {
    nameCounts.set(tool.chatName, (nameCounts.get(tool.chatName) ?? 0) + 1)
  }

  const collisionToolNames = [...nameCounts.entries()]
    .filter(([, count]) => count > 1)
    .map(([name]) => name)
    .sort((left, right) => left.localeCompare(right))
  const collisionSet = new Set(collisionToolNames)

  const discoveredTools: ResolvedMcpTool[] = discoveredBase.map((tool) => ({
    ...tool,
    disabled: disabledSet.has(tool.chatName),
    colliding: collisionSet.has(tool.chatName)
  }))
  const availableTools = discoveredTools.filter((tool) => tool.canExecute)
  const chatTools = availableTools.filter(
    (tool) => !tool.disabled && !tool.colliding
  )

  return {
    discoveredTools,
    availableTools,
    chatTools,
    collisionToolNames,
    counts: {
      discovered: discoveredTools.length,
      executable: availableTools.length,
      disabled: availableTools.filter((tool) => tool.disabled).length,
      colliding: availableTools.filter((tool) => tool.colliding).length,
      chatEnabled: chatTools.length
    }
  }
}

const readParameters = (tool: ChatToolRecord): ChatToolRecord => {
  const functionRecord = readFunctionRecord(tool)
  const candidates = [
    functionRecord?.parameters,
    tool.parameters,
    tool.input_schema,
    tool.inputSchema,
    tool.json_schema
  ]
  const parameters = candidates.find(isRecord)
  return parameters ?? { type: "object", properties: {} }
}

const normalizeRequestToolsFromState = (
  state: ChatToolFilterState
): ChatToolRecord[] =>
  state.chatTools.map((tool) => ({
    type: "function",
    function: {
      name: tool.chatName,
      description: tool.description,
      parameters: readParameters(tool.tool)
    }
  }))

export const normalizeChatToolsForRequest = (
  tools?: unknown[]
): ChatToolRecord[] | undefined => {
  const state = buildChatToolFilterState({ tools })
  const requestTools = normalizeRequestToolsFromState(state)
  return requestTools.length > 0 ? requestTools : undefined
}

export const resolveChatToolRequest = ({
  tools,
  toolChoice,
  modelSupportsTools = true,
  mcpHealthState = "healthy",
  hasMcp = true,
  disabledToolNames = [],
  counts: countsOverride
}: {
  tools?: unknown[]
  toolChoice?: ChatToolRequestChoice | string | null
  modelSupportsTools?: boolean
  mcpHealthState?: string | null
  hasMcp?: boolean
  disabledToolNames?: string[]
  counts?: ChatToolFilterCounts
}): ResolvedChatToolRequest => {
  const state = buildChatToolFilterState({ tools, disabledToolNames })
  const counts = countsOverride ?? state.counts
  const omit = (
    omittedReason: ChatToolOmissionReason
  ): ResolvedChatToolRequest => ({
    omittedReason,
    counts
  })

  if (toolChoice === "none") return omit("tool_choice_none")
  if (!modelSupportsTools) return omit("model_lacks_tool_capability")
  if (!hasMcp) return omit("mcp_absent")
  if (mcpHealthState === "unavailable") return omit("mcp_unavailable")
  if (mcpHealthState === "unhealthy") return omit("mcp_unhealthy")

  if (state.chatTools.length === 0 || counts.chatEnabled === 0) {
    const rawToolCount = Array.isArray(tools) ? tools.length : 0
    return omit(
      counts.discovered === 0 && rawToolCount > 0
        ? "no_normalized_request_tools"
        : "no_enabled_executable_tools"
    )
  }

  const requestTools = normalizeRequestToolsFromState(state)
  if (requestTools.length === 0) return omit("no_normalized_request_tools")

  const effectiveToolChoice =
    toolChoice === "auto" || toolChoice === "required" ? toolChoice : undefined

  return {
    tools: requestTools,
    toolChoice: effectiveToolChoice,
    counts
  }
}
