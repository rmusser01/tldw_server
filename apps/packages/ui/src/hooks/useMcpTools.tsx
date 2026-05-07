import React from "react"
import { useQuery } from "@tanstack/react-query"
import { apiSend } from "@/services/api-send"
import { useCanonicalConnectionConfig } from "@/hooks/useCanonicalConnectionConfig"
import { useServerCapabilities } from "@/hooks/useServerCapabilities"
import { useSetting } from "@/hooks/useSetting"
import {
  fetchMcpToolCatalogs,
  fetchMcpTools,
  fetchMcpToolCatalogsViaDiscovery,
  fetchMcpModulesViaDiscovery,
  fetchMcpToolsViaDiscovery,
  type McpToolCatalog,
  type McpToolDefinition
} from "@/services/tldw/mcp"
import {
  MCP_DISABLED_TOOLS_SETTING,
  MCP_TOOL_CATALOG_SETTING,
  MCP_TOOL_CATALOG_ID_SETTING,
  MCP_TOOL_CATALOG_STRICT_SETTING,
  MCP_TOOL_MODULE_SETTING,
  type McpDisabledToolPreferences
} from "@/services/settings/ui-settings"
import { useMcpToolsStore, type McpHealthState } from "@/store/mcp-tools"
import {
  buildChatToolFilterState,
  normalizeChatToolName,
  type ChatToolFilterCounts,
  type ResolvedMcpTool
} from "@/utils/chat-tools"
import type { TldwConfig } from "@/services/tldw/TldwApiClient"

type McpToolsStatus = {
  hasMcp: boolean
  healthState: McpHealthState
  healthLoading: boolean
  tools: McpToolDefinition[]
  discoveredTools: ResolvedMcpTool[]
  availableTools: ResolvedMcpTool[]
  chatTools: ResolvedMcpTool[]
  toolsLoading: boolean
  toolsAvailable: boolean | null
  disabledToolPreferences: McpDisabledToolPreferences
  activeToolPreferenceScope: string
  disabledToolNames: string[]
  collisionToolNames: string[]
  toolCounts: ChatToolFilterCounts
  catalogs: McpToolCatalog[]
  catalogsLoading: boolean
  toolCatalog: string
  toolCatalogId: number | null
  toolModules: string[]
  moduleOptions: string[]
  moduleOptionsLoading: boolean
  toolCatalogStrict: boolean
  setToolCatalog: (catalog: string) => void
  setToolCatalogId: (catalogId: number | null) => void
  setToolModules: (moduleIds: string[]) => void
  setToolCatalogStrict: (strict: boolean) => void
  setToolEnabled: (toolName: string, enabled: boolean) => void
  setToolsEnabled: (toolNames: string[], enabled: boolean) => void
  resetToolFilter: () => void
}

const normalizeModuleList = (modules: string[] | null | undefined): string[] => {
  const seen = new Set<string>()
  const result: string[] = []
  for (const moduleId of modules ?? []) {
    if (typeof moduleId !== "string") continue
    const trimmed = moduleId.trim()
    if (!trimmed || seen.has(trimmed)) continue
    seen.add(trimmed)
    result.push(trimmed)
  }
  return result
}

const areModuleListsEqual = (left: string[], right: string[]): boolean => {
  if (left.length !== right.length) return false
  return left.every((value, index) => value === right[index])
}

const DEFAULT_TOOL_PREFERENCE_SCOPE =
  "server:unknown|auth:single-user|org:none|principal:anonymous"

const normalizePreferenceScopeServerUrl = (
  serverUrl: string | null | undefined
): string => {
  const trimmed = typeof serverUrl === "string" ? serverUrl.trim() : ""
  if (!trimmed) return "unknown"
  try {
    const parsed = new URL(trimmed)
    const pathname =
      parsed.pathname && parsed.pathname !== "/"
        ? parsed.pathname.replace(/\/+$/g, "")
        : ""
    return `${parsed.protocol}//${parsed.host}${pathname}`
  } catch {
    return trimmed.replace(/\/+$/g, "")
  }
}

const decodeJwtPayload = (token: string | null | undefined): unknown => {
  if (typeof token !== "string" || !token.trim()) return null
  const [, payload] = token.split(".")
  if (!payload) return null
  try {
    const base64 = payload.replace(/-/g, "+").replace(/_/g, "/")
    const padded = base64.padEnd(Math.ceil(base64.length / 4) * 4, "=")
    const decoded = globalThis.atob(padded)
    return JSON.parse(decoded)
  } catch {
    return null
  }
}

const resolvePreferenceScopePrincipal = (
  config: TldwConfig | null | undefined
): string => {
  const payload = decodeJwtPayload(config?.accessToken)
  if (payload && typeof payload === "object") {
    const data = payload as Record<string, unknown>
    for (const key of ["sub", "user_id", "username", "email"]) {
      const value = data[key]
      if (typeof value === "string" && value.trim()) {
        return value.trim()
      }
      if (typeof value === "number" && Number.isFinite(value)) {
        return String(value)
      }
    }
  }
  return "anonymous"
}

const deriveMcpToolPreferenceScope = (
  config: TldwConfig | null | undefined
): string => {
  if (!config) return DEFAULT_TOOL_PREFERENCE_SCOPE
  const server = normalizePreferenceScopeServerUrl(config.serverUrl)
  const auth =
    config.authMode === "multi-user" || config.authMode === "single-user"
      ? config.authMode
      : "single-user"
  const org =
    typeof config.orgId === "number" && Number.isFinite(config.orgId)
      ? String(config.orgId)
      : "none"
  const principal = resolvePreferenceScopePrincipal(config)
  return `server:${server}|auth:${auth}|org:${org}|principal:${principal}`
}

const normalizeDisabledToolNames = (toolNames: string[]): string[] => {
  const seen = new Set<string>()
  for (const toolName of toolNames) {
    const normalized = normalizeChatToolName(toolName)
    if (normalized) seen.add(normalized)
  }
  return [...seen].sort((left, right) => left.localeCompare(right))
}

const normalizeDisabledToolPreferences = (
  preferences: McpDisabledToolPreferences | null | undefined
): McpDisabledToolPreferences => {
  if (
    preferences &&
    typeof preferences === "object" &&
    preferences.version === 1 &&
    preferences.scopes &&
    typeof preferences.scopes === "object"
  ) {
    return preferences
  }
  return {
    version: 1,
    scopes: {}
  }
}

type UseMcpToolsOptions = {
  enabled?: boolean
}

export const useMcpTools = (
  options: UseMcpToolsOptions = {}
): McpToolsStatus => {
  const { capabilities, loading } = useServerCapabilities()
  const { config: connectionConfig } = useCanonicalConnectionConfig()
  const hasMcp = Boolean(capabilities?.hasMcp) && !loading
  const probeEnabled = options.enabled ?? true
  const setToolFilterState = useMcpToolsStore((state) => state.setToolFilterState)
  const setHealthState = useMcpToolsStore((state) => state.setHealthState)
  const setToolsLoading = useMcpToolsStore((state) => state.setToolsLoading)
  const toolCatalog = useMcpToolsStore((state) => state.toolCatalog)
  const toolCatalogId = useMcpToolsStore((state) => state.toolCatalogId)
  const toolModules = useMcpToolsStore((state) => state.toolModules)
  const toolCatalogStrict = useMcpToolsStore((state) => state.toolCatalogStrict)
  const setToolCatalog = useMcpToolsStore((state) => state.setToolCatalog)
  const setToolCatalogId = useMcpToolsStore((state) => state.setToolCatalogId)
  const setToolModules = useMcpToolsStore((state) => state.setToolModules)
  const setToolCatalogStrict = useMcpToolsStore((state) => state.setToolCatalogStrict)

  const [storedCatalog, persistCatalog] = useSetting(MCP_TOOL_CATALOG_SETTING)
  const [storedCatalogId, persistCatalogId] = useSetting(MCP_TOOL_CATALOG_ID_SETTING)
  const [storedModule, persistModule] = useSetting(MCP_TOOL_MODULE_SETTING)
  const [storedStrict, persistStrict] = useSetting(MCP_TOOL_CATALOG_STRICT_SETTING)
  const [storedDisabledToolPreferences, persistStoredDisabledToolPreferences] =
    useSetting(MCP_DISABLED_TOOLS_SETTING)
  const normalizedStoredDisabledToolPreferences = React.useMemo(
    () => normalizeDisabledToolPreferences(storedDisabledToolPreferences),
    [storedDisabledToolPreferences]
  )
  const [disabledToolPreferences, setDisabledToolPreferences] =
    React.useState<McpDisabledToolPreferences>(
      normalizedStoredDisabledToolPreferences
    )
  React.useEffect(() => {
    setDisabledToolPreferences(normalizedStoredDisabledToolPreferences)
  }, [normalizedStoredDisabledToolPreferences])
  const activeToolPreferenceScope = React.useMemo(
    () => deriveMcpToolPreferenceScope(connectionConfig),
    [connectionConfig]
  )
  const disabledToolNames = React.useMemo(
    () =>
      normalizeDisabledToolNames(
        disabledToolPreferences.scopes[activeToolPreferenceScope]
          ?.disabledToolNames ?? []
      ),
    [activeToolPreferenceScope, disabledToolPreferences]
  )
  const normalizedToolModules = React.useMemo(
    () => normalizeModuleList(toolModules),
    [toolModules]
  )
  const normalizedStoredModules = React.useMemo(
    () => normalizeModuleList(storedModule),
    [storedModule]
  )
  const toolStateRef = React.useRef({
    toolCatalog,
    toolCatalogId,
    toolModules: normalizedToolModules,
    toolCatalogStrict
  })
  const healthQuery = useQuery({
    queryKey: ["mcp-health", activeToolPreferenceScope],
    queryFn: async () => apiSend({ path: "/api/v1/mcp/health", method: "GET" }),
    enabled: hasMcp && probeEnabled,
    staleTime: 60_000,
    refetchInterval: 60_000,
    refetchOnWindowFocus: false
  })

  let healthState: McpHealthState = "unknown"
  if (!hasMcp) {
    healthState = loading ? "unknown" : "unavailable"
  } else if (!probeEnabled) {
    healthState = "unknown"
  } else if (healthQuery.isLoading) {
    healthState = "unknown"
  } else if (healthQuery.data?.ok) {
    healthState = "healthy"
  } else if (healthQuery.data?.status === 404) {
    healthState = "unknown"
  } else {
    healthState = "unhealthy"
  }

  const toolsQuery = useQuery({
    queryKey: [
      "mcp-tools",
      activeToolPreferenceScope,
      toolCatalog,
      toolCatalogId,
      normalizedToolModules,
      toolCatalogStrict
    ],
    queryFn: async () => {
      let discoveryError: unknown
      try {
        const tools = await fetchMcpToolsViaDiscovery({
          catalog: toolCatalog,
          catalogId: toolCatalogId,
          module:
            normalizedToolModules.length > 0 ? normalizedToolModules : undefined,
          catalogStrict: toolCatalogStrict
        })
        return tools
      } catch (err) {
        discoveryError = err
        return await fetchMcpTools({
          catalog: toolCatalog,
          catalogId: toolCatalogId,
          module:
            normalizedToolModules.length > 0 ? normalizedToolModules : undefined,
          catalogStrict: toolCatalogStrict
        }).catch((fallbackErr) => {
          console.warn("MCP tools fetch failed", {
            discoveryError,
            fallbackErr
          })
          throw fallbackErr
        })
      }
    },
    enabled: hasMcp && probeEnabled,
    staleTime: 60_000,
    refetchInterval: 60_000,
    refetchOnWindowFocus: false
  })

  const catalogsQuery = useQuery({
    queryKey: ["mcp-tool-catalogs", activeToolPreferenceScope],
    queryFn: async () => {
      try {
        return await fetchMcpToolCatalogsViaDiscovery("all")
      } catch {
        return await fetchMcpToolCatalogs()
      }
    },
    enabled: hasMcp && probeEnabled,
    staleTime: 60_000,
    refetchInterval: 60_000,
    refetchOnWindowFocus: false
  })

  const moduleOptionsQuery = useQuery({
    queryKey: ["mcp-tool-modules", activeToolPreferenceScope],
    queryFn: async () => {
      try {
        return await fetchMcpModulesViaDiscovery()
      } catch {
        const tools = await fetchMcpTools()
        const seen = new Set<string>()
        const modules: string[] = []
        for (const tool of tools) {
          const moduleId =
            typeof tool?.module === "string" ? tool.module.trim() : ""
          if (!moduleId || seen.has(moduleId)) continue
          seen.add(moduleId)
          modules.push(moduleId)
        }
        return modules
      }
    },
    enabled: hasMcp && probeEnabled,
    staleTime: 60_000,
    refetchInterval: 60_000,
    refetchOnWindowFocus: false
  })

  const toolFilterState = React.useMemo(
    () =>
      buildChatToolFilterState({
        tools: toolsQuery.data ?? [],
        disabledToolNames
      }),
    [disabledToolNames, toolsQuery.data]
  )
  const {
    discoveredTools,
    availableTools,
    chatTools,
    collisionToolNames,
    counts: toolCounts
  } = toolFilterState
  const tools = React.useMemo(
    () => availableTools.map((tool) => tool.tool as McpToolDefinition),
    [availableTools]
  )
  const toolsAvailable =
    !probeEnabled || toolsQuery.isLoading ? null : availableTools.length > 0
  const catalogs = catalogsQuery.data ?? []
  const moduleOptionsSource =
    moduleOptionsQuery.data && moduleOptionsQuery.data.length > 0
      ? moduleOptionsQuery.data
      : toolsQuery.data ?? []
  const moduleOptions = React.useMemo(() => {
    const seen = new Set<string>()
    const result: string[] = []
    for (const entry of moduleOptionsSource ?? []) {
      const moduleId =
        typeof entry === "string"
          ? entry.trim()
          : typeof (entry as McpToolDefinition)?.module === "string"
            ? String((entry as McpToolDefinition).module).trim()
            : ""
      if (!moduleId || seen.has(moduleId)) continue
      seen.add(moduleId)
      result.push(moduleId)
    }
    return result.sort((a, b) => a.localeCompare(b))
  }, [moduleOptionsSource])
  const moduleOptionsLoading =
    moduleOptionsQuery.isLoading && moduleOptionsSource.length === 0

  React.useEffect(() => {
    setHealthState(healthState)
  }, [healthState, setHealthState])

  React.useEffect(() => {
    toolStateRef.current = {
      toolCatalog,
      toolCatalogId,
      toolModules: normalizedToolModules,
      toolCatalogStrict
    }
  }, [toolCatalog, toolCatalogId, normalizedToolModules, toolCatalogStrict])

  React.useEffect(() => {
    const current = toolStateRef.current
    const normalizedCatalogId = storedCatalogId ?? null
    if (storedCatalog !== current.toolCatalog) {
      setToolCatalog(storedCatalog)
    }
    if (normalizedCatalogId !== current.toolCatalogId) {
      setToolCatalogId(normalizedCatalogId)
    }
    if (!areModuleListsEqual(normalizedStoredModules, current.toolModules)) {
      setToolModules(normalizedStoredModules)
    }
    if (storedStrict !== current.toolCatalogStrict) {
      setToolCatalogStrict(storedStrict)
    }
  }, [
    normalizedStoredModules,
    setToolCatalog,
    setToolCatalogId,
    setToolCatalogStrict,
    setToolModules,
    storedCatalog,
    storedCatalogId,
    storedModule,
    storedStrict
  ])

  React.useEffect(() => {
    if (!hasMcp && !loading) {
      setToolFilterState({
        discoveredTools: [],
        availableTools: [],
        chatTools: [],
        disabledToolPreferences,
        activeToolPreferenceScope,
        disabledToolNames,
        collisionToolNames: [],
        toolCounts: {
          discovered: 0,
          executable: 0,
          disabled: 0,
          colliding: 0,
          chatEnabled: 0
        }
      })
      setToolsLoading(false)
      return
    }
    if (!probeEnabled) {
      setToolsLoading(false)
      return
    }
    setToolsLoading(toolsQuery.isLoading)
    if (!toolsQuery.isLoading) {
      setToolFilterState({
        discoveredTools,
        availableTools,
        chatTools,
        disabledToolPreferences,
        activeToolPreferenceScope,
        disabledToolNames,
        collisionToolNames,
        toolCounts
      })
    }
  }, [
    activeToolPreferenceScope,
    availableTools,
    chatTools,
    collisionToolNames,
    disabledToolNames,
    disabledToolPreferences,
    discoveredTools,
    hasMcp,
    loading,
    probeEnabled,
    setToolFilterState,
    setToolsLoading,
    toolCounts,
    toolsQuery.isLoading
  ])

  const persistToolCatalog = React.useCallback(
    (catalog: string) => {
      setToolCatalog(catalog)
      void persistCatalog(catalog)
    },
    [persistCatalog, setToolCatalog]
  )

  const persistToolCatalogId = React.useCallback(
    (catalogId: number | null) => {
      setToolCatalogId(catalogId)
      void persistCatalogId(catalogId)
    },
    [persistCatalogId, setToolCatalogId]
  )

  const persistToolModule = React.useCallback(
    (moduleIds: string[]) => {
      const normalized = normalizeModuleList(moduleIds)
      if (areModuleListsEqual(normalized, normalizedToolModules)) return
      setToolModules(normalized)
      void persistModule(normalized)
    },
    [normalizedToolModules, persistModule, setToolModules]
  )

  const persistToolCatalogStrict = React.useCallback(
    (strict: boolean) => {
      setToolCatalogStrict(strict)
      void persistStrict(strict)
    },
    [persistStrict, setToolCatalogStrict]
  )

  const updateDisabledToolPreferences = React.useCallback(
    (
      updater: (
        current: McpDisabledToolPreferences
      ) => McpDisabledToolPreferences
    ) => {
      const next = updater(disabledToolPreferences)
      setDisabledToolPreferences(next)
      void persistStoredDisabledToolPreferences(next)
    },
    [disabledToolPreferences, persistStoredDisabledToolPreferences]
  )

  const setToolsEnabled = React.useCallback(
    (toolNames: string[], enabled: boolean) => {
      const normalizedNames = normalizeDisabledToolNames(toolNames)
      if (normalizedNames.length === 0) return
      updateDisabledToolPreferences((current) => {
        const currentScopePreference =
          current.scopes[activeToolPreferenceScope]
        const disabledSet = new Set(
          normalizeDisabledToolNames(
            currentScopePreference?.disabledToolNames ?? []
          )
        )
        for (const toolName of normalizedNames) {
          if (enabled) {
            disabledSet.delete(toolName)
          } else {
            disabledSet.add(toolName)
          }
        }
        const disabledToolNames = [...disabledSet].sort((left, right) =>
          left.localeCompare(right)
        )
        const scopes = { ...current.scopes }
        if (disabledToolNames.length > 0) {
          scopes[activeToolPreferenceScope] = {
            disabledToolNames,
            updatedAt: new Date().toISOString()
          }
        } else {
          delete scopes[activeToolPreferenceScope]
        }
        return {
          version: 1,
          scopes
        }
      })
    },
    [activeToolPreferenceScope, updateDisabledToolPreferences]
  )

  const setToolEnabled = React.useCallback(
    (toolName: string, enabled: boolean) => {
      setToolsEnabled([toolName], enabled)
    },
    [setToolsEnabled]
  )

  const resetToolFilter = React.useCallback(() => {
    updateDisabledToolPreferences((current) => {
      if (!current.scopes[activeToolPreferenceScope]) return current
      const scopes = { ...current.scopes }
      delete scopes[activeToolPreferenceScope]
      return {
        version: 1,
        scopes
      }
    })
  }, [activeToolPreferenceScope, updateDisabledToolPreferences])

  return {
    hasMcp,
    healthState,
    healthLoading: probeEnabled ? healthQuery.isLoading : false,
    tools,
    discoveredTools,
    availableTools,
    chatTools,
    toolsLoading: probeEnabled ? toolsQuery.isLoading : false,
    toolsAvailable,
    disabledToolPreferences,
    activeToolPreferenceScope,
    disabledToolNames,
    collisionToolNames,
    toolCounts,
    catalogs,
    catalogsLoading: probeEnabled ? catalogsQuery.isLoading : false,
    toolCatalog,
    toolCatalogId,
    toolModules: normalizedToolModules,
    moduleOptions,
    moduleOptionsLoading: probeEnabled ? moduleOptionsLoading : false,
    toolCatalogStrict,
    setToolCatalog: persistToolCatalog,
    setToolCatalogId: persistToolCatalogId,
    setToolModules: persistToolModule,
    setToolCatalogStrict: persistToolCatalogStrict,
    setToolEnabled,
    setToolsEnabled,
    resetToolFilter
  }
}
