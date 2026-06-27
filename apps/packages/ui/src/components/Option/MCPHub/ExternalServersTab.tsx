import { useEffect, useMemo, useRef, useState } from "react"
import { Button, Card, Checkbox, Empty, List, Modal, Space, Tag, Tooltip, Typography } from "antd"
import { QuestionCircleOutlined } from "@ant-design/icons"
import { getDesignSystemState, type DesignSystemStateKey } from "@/design-system"
import { StatePanel } from "@/components/ui/state"

import {
  clearExternalServerSlotSecret,
  createExternalServer,
  createExternalServerCredentialSlot,
  deleteExternalServer,
  deleteExternalServerCredentialSlot,
  getExternalServerAuthTemplate,
  getMcpHubReadiness as fetchMcpHubReadiness,
  getToolRegistrySummary,
  importExternalServer,
  listExternalServers,
  refreshExternalServerDiscovery,
  setExternalServerSecret,
  setExternalServerSlotSecret,
  type McpHubDrillTarget,
  updateExternalServer,
  updateExternalServerAuthTemplate,
  updateExternalServerCredentialSlot,
  validateExternalServer,
  type McpHubExternalServer,
  type McpHubExternalServerAuthTemplateMapping,
  type McpHubExternalServerCreateInput,
  type McpHubExternalServerCredentialSlot,
  type McpHubReadiness,
  type McpHubReadinessAction,
  type McpHubServerReadiness,
  type McpHubToolRegistryEntry
} from "@/services/tldw/mcp-hub"

import {
  getExternalAuthTemplateBlockedReasonLabel,
  getManagedExternalServers,
  getManagedExternalServerSlots
} from "./policyHelpers"
import {
  formatMcpDiagnosticValue,
  getMcpServerReadiness,
  type McpReadinessAction,
  type McpServerReadiness
} from "./mcpHubReadiness"

const DEFAULT_SLOT_SECRET_KIND = "bearer_token"
const DEFAULT_SLOT_PRIVILEGE_CLASS = "read"
const DIAGNOSTIC_UNAVAILABLE = "Not available in this client"
type SetupMode = "choice" | "stdio" | "http" | "import" | "advanced"

type SetupResult = {
  serverId: string
  serverName: string
  discovered: boolean
  readiness: McpHubServerReadiness | null
}

type ManagedServerDraft = McpHubExternalServerCreateInput & {
  owner_scope_type: "global" | "org" | "team" | "user"
  enabled: boolean
}

const AUTH_TEMPLATE_TARGET_BY_TRANSPORT: Record<string, "header" | "env"> = {
  websocket: "header",
  stdio: "env"
}

const normalizeAuthTemplateMapping = (
  mapping: Partial<McpHubExternalServerAuthTemplateMapping>,
  fallbackTargetType: "header" | "env"
): McpHubExternalServerAuthTemplateMapping => ({
  slot_name: String(mapping.slot_name || "").trim(),
  target_type: mapping.target_type === "env" ? "env" : mapping.target_type === "header" ? "header" : fallbackTargetType,
  target_name: String(mapping.target_name || ""),
  prefix: String(mapping.prefix || ""),
  suffix: String(mapping.suffix || ""),
  required: mapping.required !== false
})

const getErrorMessage = (err: unknown): string =>
  err instanceof Error ? err.message : "Unknown error"

const parseArgs = (value: string): string[] =>
  value
    .split(/\s+/)
    .map((arg) => arg.trim())
    .filter(Boolean)

const parseEnvVars = (value: string): Record<string, string> => {
  const env: Record<string, string> = {}
  for (const rawLine of value.split(/\r?\n/)) {
    const line = rawLine.trim()
    if (!line) {
      continue
    }
    const separatorIndex = line.indexOf("=")
    if (separatorIndex <= 0) {
      throw new Error("Env vars must use KEY=value lines.")
    }
    env[line.slice(0, separatorIndex).trim()] = line.slice(separatorIndex + 1).trim()
  }
  return env
}

const getImportedManagedServerDraft = (value: unknown): ManagedServerDraft => {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("Import config JSON must decode to an object.")
  }
  const record = value as Record<string, unknown>
  const serverId = String(record.server_id ?? record.id ?? "").trim()
  const name = String(record.name ?? "").trim()
  const transport = String(record.transport ?? "").trim()
  const config = record.config ?? {}
  if (!serverId || !name || !transport) {
    throw new Error("Import config JSON must include server_id, name, and transport.")
  }
  if (!config || typeof config !== "object" || Array.isArray(config)) {
    throw new Error("Import config JSON config must be an object.")
  }
  return {
    server_id: serverId,
    name,
    transport,
    config: config as Record<string, unknown>,
    owner_scope_type:
      record.owner_scope_type === "org" ||
      record.owner_scope_type === "team" ||
      record.owner_scope_type === "user"
        ? record.owner_scope_type
        : "global",
    enabled: record.enabled !== false
  }
}

const getImportedManagedServerDraftFromText = (value: string): ManagedServerDraft => {
  try {
    return getImportedManagedServerDraft(JSON.parse(value))
  } catch (err) {
    if (err instanceof SyntaxError) {
      throw new Error("Import config JSON must be valid JSON.")
    }
    throw err
  }
}

const getClientDiagnosticValue = (value: unknown): string => {
  const normalized = typeof value === "string" ? value.trim() : ""
  return normalized || DIAGNOSTIC_UNAVAILABLE
}

const getMcpHubEnvironmentDiagnostics = () => {
  const deploymentMode = getClientDiagnosticValue(
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE
  )
  const apiOrigin = getClientDiagnosticValue(process.env.NEXT_PUBLIC_API_URL)

  return {
    deploymentMode,
    apiOrigin,
    healthEndpoint:
      apiOrigin === DIAGNOSTIC_UNAVAILABLE
        ? DIAGNOSTIC_UNAVAILABLE
        : `${apiOrigin.replace(/\/+$/, "")}/api/v1/health`,
    latestHealthResult: DIAGNOSTIC_UNAVAILABLE
  }
}

type ExternalServersTabProps = {
  drillTarget?: McpHubDrillTarget | null
  onDrillHandled?: (requestId: number) => void
  onOpenToolCatalog?: () => void
}

const READINESS_DISPLAY_LABELS: Record<McpServerReadiness["displayState"], string> = {
  needs_setup: "Needs setup",
  checking: "Checking",
  ready: getDesignSystemState("ready").label,
  needs_attention: "Needs attention",
  no_tools: "No tools",
  stale: "Stale"
}

const READINESS_DISPLAY_COLORS: Record<McpServerReadiness["displayState"], string> = {
  needs_setup: "blue",
  checking: "processing",
  ready: "green",
  needs_attention: "orange",
  no_tools: "gold",
  stale: "gold"
}

const CREDENTIAL_TAGS: Record<
  McpServerReadiness["credentialState"],
  { color?: string; label: string }
> = {
  not_required: { color: "green", label: "No credentials required" },
  required_missing: { color: "orange", label: "Credentials required" },
  configured: { color: "green", label: "credentials configured" },
  legacy_fallback: { color: "orange", label: "Legacy Secret Fallback" },
  unknown: { label: "credential status unknown" }
}

const DESIGN_STATE_BY_READINESS_DISPLAY: Record<
  McpServerReadiness["displayState"],
  DesignSystemStateKey
> = {
  needs_setup: "setup_required",
  checking: "loading",
  ready: "ready",
  needs_attention: "degraded",
  no_tools: "degraded",
  stale: "degraded"
}

const toMapperReadiness = (
  server: McpHubExternalServer,
  backendReadiness: McpHubServerReadiness | undefined,
  registryEntries: McpHubToolRegistryEntry[]
): McpServerReadiness => {
  if (!backendReadiness) {
    return getMcpServerReadiness({ server, registryEntries })
  }

  return {
    serverId: backendReadiness.server_id,
    serverName: backendReadiness.server_name,
    displayState: backendReadiness.display_state,
    credentialState: backendReadiness.credential_state,
    toolCount: backendReadiness.tool_count,
    reasonCodes: backendReadiness.reason_codes,
    primaryReasonCode: backendReadiness.primary_reason_code ?? undefined,
    allowedActions: backendReadiness.allowed_actions,
    message: backendReadiness.message,
    ...(backendReadiness.current_operation
      ? {
          currentOperation: {
            operation: backendReadiness.current_operation.operation_type,
            label:
              backendReadiness.current_operation.message ||
              backendReadiness.current_operation.operation_type,
            startedAt: backendReadiness.current_operation.started_at ?? null,
            serverId: backendReadiness.server_id
          }
        }
      : {})
  }
}

const formatDiagnosticTimestamp = (value?: string | null): string =>
  value?.trim() || "Not available"

const formatDiagnosticCurrentOperation = (
  operation?: McpHubServerReadiness["current_operation"]
): string => {
  if (!operation) {
    return "none"
  }

  const operationType = operation.operation_type || "operation"
  const startedAt = operation.started_at ? ` since ${operation.started_at}` : ""
  const message = operation.message ? `, ${operation.message}` : ""

  return `${operationType}${startedAt}${message}`
}

const formatDiagnosticNullable = (value?: string | null): string =>
  value?.trim() || "none"

export const ExternalServersTab = ({
  drillTarget = null,
  onDrillHandled,
  onOpenToolCatalog
}: ExternalServersTabProps) => {
  const handledDrillRequestRef = useRef<number | null>(null)
  const [servers, setServers] = useState<McpHubExternalServer[]>([])
  const [registryEntries, setRegistryEntries] = useState<McpHubToolRegistryEntry[]>([])
  const [hubReadiness, setHubReadiness] = useState<McpHubReadiness | null>(null)
  const [serversLoaded, setServersLoaded] = useState(false)
  const [loading, setLoading] = useState(false)
  const [rowActionLoadingKey, setRowActionLoadingKey] = useState<string | null>(null)
  const [detailsServerId, setDetailsServerId] = useState<string | null>(null)
  const [activeServerId, setActiveServerId] = useState<string>("")
  const [secretValue, setSecretValue] = useState("")
  const [saving, setSaving] = useState(false)
  const [importingServerId, setImportingServerId] = useState<string | null>(null)
  const [successMessage, setSuccessMessage] = useState<string | null>(null)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [readinessWarningMessage, setReadinessWarningMessage] = useState<string | null>(null)
  const [serverFormOpen, setServerFormOpen] = useState(false)
  const [setupMode, setSetupMode] = useState<SetupMode>("choice")
  const [setupResult, setSetupResult] = useState<SetupResult | null>(null)
  const [editingServerId, setEditingServerId] = useState<string | null>(null)
  const [serverIdValue, setServerIdValue] = useState("")
  const [serverNameValue, setServerNameValue] = useState("")
  const [transportValue, setTransportValue] = useState("stdio")
  const [ownerScopeType, setOwnerScopeType] = useState<"global" | "org" | "team" | "user">("global")
  const [enabledValue, setEnabledValue] = useState(true)
  const [configText, setConfigText] = useState("{}")
  const [stdioCommandValue, setStdioCommandValue] = useState("")
  const [stdioArgsValue, setStdioArgsValue] = useState("")
  const [stdioEnvValue, setStdioEnvValue] = useState("")
  const [stdioCwdValue, setStdioCwdValue] = useState("")
  const [httpUrlValue, setHttpUrlValue] = useState("")
  const [httpHeadersText, setHttpHeadersText] = useState("{}")
  const [importConfigText, setImportConfigText] = useState("")
  const [serverSaving, setServerSaving] = useState(false)
  const [slotFormOpen, setSlotFormOpen] = useState(false)
  const [editingSlotName, setEditingSlotName] = useState<string | null>(null)
  const [slotNameValue, setSlotNameValue] = useState("")
  const [slotDisplayNameValue, setSlotDisplayNameValue] = useState("")
  const [slotSecretKindValue, setSlotSecretKindValue] = useState(DEFAULT_SLOT_SECRET_KIND)
  const [slotPrivilegeClassValue, setSlotPrivilegeClassValue] = useState(DEFAULT_SLOT_PRIVILEGE_CLASS)
  const [slotIsRequiredValue, setSlotIsRequiredValue] = useState(true)
  const [slotSaving, setSlotSaving] = useState(false)
  const [slotDeletingKey, setSlotDeletingKey] = useState<string | null>(null)
  const [activeSlotName, setActiveSlotName] = useState("")
  const [slotSecretValue, setSlotSecretValue] = useState("")
  const [slotSecretSaving, setSlotSecretSaving] = useState(false)
  const [slotSecretClearing, setSlotSecretClearing] = useState(false)
  const [focusedServerId, setFocusedServerId] = useState<string | null>(null)
  const [authTemplateMappings, setAuthTemplateMappings] = useState<McpHubExternalServerAuthTemplateMapping[]>([])
  const [authTemplateLoading, setAuthTemplateLoading] = useState(false)
  const [authTemplateSaving, setAuthTemplateSaving] = useState(false)
  const environmentDiagnostics = useMemo(() => getMcpHubEnvironmentDiagnostics(), [])
  const managedServers = useMemo(() => getManagedExternalServers(servers), [servers])
  const activeManagedServer = useMemo(
    () => managedServers.find((server) => server.id === activeServerId) || null,
    [activeServerId, managedServers]
  )
  const activeSlots = useMemo(
    () => getManagedExternalServerSlots(activeManagedServer),
    [activeManagedServer]
  )
  const activeAuthTemplateTarget = useMemo(
    () =>
      activeManagedServer
        ? AUTH_TEMPLATE_TARGET_BY_TRANSPORT[String(activeManagedServer.transport || "").trim().toLowerCase()] || null
        : null,
    [activeManagedServer]
  )
  const activeAuthTemplateBlockedReason = getExternalAuthTemplateBlockedReasonLabel(
    activeManagedServer?.auth_template_blocked_reason
  )
  const backendReadinessByServerId = useMemo(() => {
    const nextReadiness = new Map<string, McpHubServerReadiness>()
    for (const readiness of hubReadiness?.servers ?? []) {
      nextReadiness.set(readiness.server_id, readiness)
    }
    return nextReadiness
  }, [hubReadiness])
  const rowReadinessByServerId = useMemo(() => {
    const nextReadiness = new Map<string, McpServerReadiness>()
    for (const server of servers) {
      if (server.server_source === "legacy") {
        continue
      }
      nextReadiness.set(
        server.id,
        toMapperReadiness(server, backendReadinessByServerId.get(server.id), registryEntries)
      )
    }
    return nextReadiness
  }, [backendReadinessByServerId, registryEntries, servers])
  const detailsServer = useMemo(
    () => servers.find((server) => server.id === detailsServerId) || null,
    [detailsServerId, servers]
  )
  const detailsReadiness = detailsServerId
    ? rowReadinessByServerId.get(detailsServerId)
    : undefined
  const detailsBackendReadiness = detailsServerId
    ? backendReadinessByServerId.get(detailsServerId)
    : undefined
  const detailsDiagnosticConfig = detailsServer
    ? formatMcpDiagnosticValue("config", detailsServer.config || {})
    : "{}"
  const activeManagedServerReadiness = activeManagedServer
    ? rowReadinessByServerId.get(activeManagedServer.id)
    : undefined
  const activeCredentialState = activeManagedServerReadiness?.credentialState
  const importPreview = useMemo(() => {
    if (setupMode !== "import" || !importConfigText.trim()) {
      return null
    }
    try {
      return {
        draft: getImportedManagedServerDraftFromText(importConfigText),
        error: null
      }
    } catch (err) {
      return {
        draft: null,
        error: getErrorMessage(err)
      }
    }
  }, [importConfigText, setupMode])

  const canSave = useMemo(
    () => activeServerId.trim().length > 0 && secretValue.trim().length > 0 && !saving,
    [activeServerId, secretValue, saving]
  )
  const canSaveSlotSecret = useMemo(
    () => activeServerId.trim().length > 0 && activeSlotName.trim().length > 0 && slotSecretValue.trim().length > 0 && !slotSecretSaving,
    [activeServerId, activeSlotName, slotSecretValue, slotSecretSaving]
  )
  const canSaveAuthTemplate = useMemo(
    () =>
      Boolean(activeManagedServer) &&
      Boolean(activeAuthTemplateTarget) &&
      authTemplateMappings.length > 0 &&
      authTemplateMappings.every(
        (mapping) => mapping.slot_name.trim().length > 0 && mapping.target_name.trim().length > 0
      ) &&
      !authTemplateSaving,
    [activeAuthTemplateTarget, activeManagedServer, authTemplateMappings, authTemplateSaving]
  )

  const loadServers = async () => {
    setLoading(true)
    setErrorMessage(null)
    setReadinessWarningMessage(null)
    try {
      const [rowsResult, registrySummaryResult, readinessResult] = await Promise.allSettled([
        listExternalServers(),
        getToolRegistrySummary(),
        fetchMcpHubReadiness()
      ])
      if (rowsResult.status === "rejected") {
        throw rowsResult.reason
      }

      const warningDetails: string[] = []
      const registrySummary =
        registrySummaryResult.status === "fulfilled" ? registrySummaryResult.value : null
      if (registrySummaryResult.status === "rejected") {
        warningDetails.push(
          `tool registry metadata could not be loaded (${getErrorMessage(registrySummaryResult.reason)})`
        )
      }

      const readiness =
        readinessResult.status === "fulfilled" ? readinessResult.value : null
      if (readinessResult.status === "rejected") {
        warningDetails.push(
          `readiness details could not be loaded (${getErrorMessage(readinessResult.reason)})`
        )
      }

      const nextServers = Array.isArray(rowsResult.value) ? rowsResult.value : []
      setServers(nextServers)
      setRegistryEntries(
        registrySummary && Array.isArray(registrySummary.entries) ? registrySummary.entries : []
      )
      setHubReadiness(readiness)
      if (warningDetails.length > 0) {
        setReadinessWarningMessage(`${warningDetails.join("; ")}.`)
      }
      const managedRows = getManagedExternalServers(nextServers)
      if (managedRows.some((server) => server.id === activeServerId)) {
        return
      }
      setActiveServerId(managedRows[0]?.id || "")
    } catch (err) {
      setServers([])
      setRegistryEntries([])
      setHubReadiness(null)
      setReadinessWarningMessage(null)
      setActiveServerId("")
      setErrorMessage(`Failed to load external servers: ${getErrorMessage(err)}`)
    } finally {
      setLoading(false)
      setServersLoaded(true)
    }
  }

  useEffect(() => {
    void loadServers()
  }, [])

  useEffect(() => {
    if (
      !drillTarget ||
      drillTarget.tab !== "credentials" ||
      drillTarget.object_kind !== "external_server"
    ) {
      return
    }
    if (
      handledDrillRequestRef.current === drillTarget.request_id ||
      loading ||
      !serversLoaded
    ) {
      return
    }
    const server = servers.find((row) => String(row.id) === String(drillTarget.object_id))
    if (server) {
      handledDrillRequestRef.current = drillTarget.request_id
      if (server.server_source === "legacy") {
        setFocusedServerId(server.id)
      } else {
        setFocusedServerId(server.id)
        setActiveServerId(server.id)
        if (drillTarget.action === "edit") {
          openEditForm(server)
        }
      }
      onDrillHandled?.(drillTarget.request_id)
    }
  }, [drillTarget, loading, onDrillHandled, servers, serversLoaded])

  useEffect(() => {
    if (activeSlots.length === 0) {
      setActiveSlotName("")
      return
    }
    if (!activeSlots.some((slot) => slot.slot_name === activeSlotName)) {
      setActiveSlotName(activeSlots[0]?.slot_name || "")
    }
  }, [activeSlotName, activeSlots])

  useEffect(() => {
    let cancelled = false

    const loadAuthTemplate = async () => {
      if (!activeManagedServer || !activeAuthTemplateTarget) {
        setAuthTemplateMappings([])
        setAuthTemplateLoading(false)
        return
      }

      setAuthTemplateLoading(true)
      try {
        const template = await getExternalServerAuthTemplate(activeManagedServer.id)
        if (cancelled) return
        const nextMappings = Array.isArray(template.mappings)
          ? template.mappings.map((mapping) =>
              normalizeAuthTemplateMapping(mapping, activeAuthTemplateTarget)
            )
          : []
        setAuthTemplateMappings(nextMappings)
      } catch {
        if (cancelled) return
        setAuthTemplateMappings([])
        setErrorMessage("Failed to load external server auth template.")
      } finally {
        if (!cancelled) {
          setAuthTemplateLoading(false)
        }
      }
    }

    void loadAuthTemplate()

    return () => {
      cancelled = true
    }
  }, [activeManagedServer?.id, activeAuthTemplateTarget])

  const resetSlotForm = () => {
    setSlotFormOpen(false)
    setEditingSlotName(null)
    setSlotNameValue("")
    setSlotDisplayNameValue("")
    setSlotSecretKindValue(DEFAULT_SLOT_SECRET_KIND)
    setSlotPrivilegeClassValue(DEFAULT_SLOT_PRIVILEGE_CLASS)
    setSlotIsRequiredValue(true)
    setSlotSaving(false)
  }

  const handleSaveSecret = async () => {
    if (!canSave) return
    setSaving(true)
    setErrorMessage(null)
    setSuccessMessage(null)
    try {
      await setExternalServerSecret(activeServerId, secretValue)
      setSecretValue("")
      setSuccessMessage("Secret configured")
      await loadServers()
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Unknown error"
      setErrorMessage(`Failed to save external server secret: ${msg}`)
    } finally {
      setSaving(false)
    }
  }

  const handleSaveSlotSecret = async () => {
    if (!canSaveSlotSecret) return
    setSlotSecretSaving(true)
    setErrorMessage(null)
    setSuccessMessage(null)
    try {
      await setExternalServerSlotSecret(activeServerId, activeSlotName, slotSecretValue)
      setSlotSecretValue("")
      setSuccessMessage("Slot secret configured")
      await loadServers()
    } catch {
      setErrorMessage("Failed to save slot secret.")
    } finally {
      setSlotSecretSaving(false)
    }
  }

  const handleClearSlotSecret = async () => {
    if (!activeServerId || !activeSlotName) return
    setSlotSecretClearing(true)
    setErrorMessage(null)
    setSuccessMessage(null)
    try {
      await clearExternalServerSlotSecret(activeServerId, activeSlotName)
      setSuccessMessage("Slot secret cleared")
      await loadServers()
    } catch {
      setErrorMessage("Failed to clear slot secret.")
    } finally {
      setSlotSecretClearing(false)
    }
  }

  const handleImport = async (serverId: string) => {
    setImportingServerId(serverId)
    setErrorMessage(null)
    setSuccessMessage(null)
    try {
      const imported = await importExternalServer(serverId)
      await loadServers()
      setActiveServerId(imported.id)
      setSuccessMessage("Legacy server imported")
    } catch {
      setErrorMessage("Failed to import legacy external server.")
    } finally {
      setImportingServerId(null)
    }
  }

  const resetServerForm = ({ keepResult = false }: { keepResult?: boolean } = {}) => {
    setServerFormOpen(false)
    setSetupMode("choice")
    setEditingServerId(null)
    setServerIdValue("")
    setServerNameValue("")
    setTransportValue("stdio")
    setOwnerScopeType("global")
    setEnabledValue(true)
    setConfigText("{}")
    setStdioCommandValue("")
    setStdioArgsValue("")
    setStdioEnvValue("")
    setStdioCwdValue("")
    setHttpUrlValue("")
    setHttpHeadersText("{}")
    setImportConfigText("")
    if (!keepResult) {
      setSetupResult(null)
    }
    setServerSaving(false)
  }

  const openCreateForm = () => {
    resetServerForm()
    setServerFormOpen(true)
  }

  const openEditForm = (server: McpHubExternalServer) => {
    setServerFormOpen(true)
    setSetupMode("advanced")
    setSetupResult(null)
    setEditingServerId(server.id)
    setServerIdValue(server.id)
    setServerNameValue(server.name)
    setTransportValue(server.transport || "stdio")
    setOwnerScopeType(server.owner_scope_type)
    setEnabledValue(Boolean(server.enabled))
    setConfigText(JSON.stringify(server.config || {}, null, 2))
  }

  const getAdvancedServerDraft = (): ManagedServerDraft | null => {
    if (!serverNameValue.trim() || !transportValue.trim()) {
      setErrorMessage("Server name and transport are required.")
      return null
    }
    if (!editingServerId && !serverIdValue.trim()) {
      setErrorMessage("Server id is required.")
      return null
    }
    let parsedConfig: Record<string, unknown> = {}
    try {
      parsedConfig = JSON.parse(configText || "{}") as Record<string, unknown>
      if (!parsedConfig || typeof parsedConfig !== "object" || Array.isArray(parsedConfig)) {
        throw new Error("config")
      }
    } catch {
      setErrorMessage("Server config JSON must decode to an object.")
      return null
    }
    return {
      server_id: serverIdValue.trim(),
      name: serverNameValue.trim(),
      transport: transportValue,
      config: parsedConfig,
      owner_scope_type: ownerScopeType,
      enabled: enabledValue
    }
  }

  const getStdioServerDraft = (): ManagedServerDraft | null => {
    if (!serverIdValue.trim() || !serverNameValue.trim()) {
      setErrorMessage("Server id and name are required.")
      return null
    }
    if (!stdioCommandValue.trim()) {
      setErrorMessage("Command is required for local stdio servers.")
      return null
    }
    let env: Record<string, string> = {}
    try {
      env = parseEnvVars(stdioEnvValue)
    } catch (err) {
      setErrorMessage(getErrorMessage(err))
      return null
    }
    return {
      server_id: serverIdValue.trim(),
      name: serverNameValue.trim(),
      transport: "stdio",
      config: {
        command: stdioCommandValue.trim(),
        args: parseArgs(stdioArgsValue),
        ...(stdioCwdValue.trim() ? { cwd: stdioCwdValue.trim() } : {}),
        ...(Object.keys(env).length > 0 ? { env } : {})
      },
      owner_scope_type: ownerScopeType,
      enabled: enabledValue
    }
  }

  const getHttpServerDraft = (): ManagedServerDraft | null => {
    if (!serverIdValue.trim() || !serverNameValue.trim()) {
      setErrorMessage("Server id and name are required.")
      return null
    }
    const urlValue = httpUrlValue.trim()
    try {
      new URL(urlValue)
    } catch {
      setErrorMessage("HTTP/SSE URL must be a valid URL.")
      return null
    }
    let headers: Record<string, unknown> = {}
    try {
      headers = JSON.parse(httpHeadersText || "{}") as Record<string, unknown>
      if (!headers || typeof headers !== "object" || Array.isArray(headers)) {
        throw new Error("headers")
      }
    } catch {
      setErrorMessage("Headers JSON must decode to an object.")
      return null
    }
    return {
      server_id: serverIdValue.trim(),
      name: serverNameValue.trim(),
      transport: "sse",
      config: {
        url: urlValue,
        ...(Object.keys(headers).length > 0 ? { headers } : {})
      },
      owner_scope_type: ownerScopeType,
      enabled: enabledValue
    }
  }

  const getImportServerDraft = (): ManagedServerDraft | null => {
    if (!importConfigText.trim()) {
      setErrorMessage("Import config JSON is required.")
      return null
    }
    try {
      return getImportedManagedServerDraftFromText(importConfigText)
    } catch {
      return null
    }
  }

  const getCreateServerDraft = (): ManagedServerDraft | null => {
    switch (setupMode) {
      case "stdio":
        return getStdioServerDraft()
      case "http":
        return getHttpServerDraft()
      case "import":
        return getImportServerDraft()
      case "advanced":
      case "choice":
        return getAdvancedServerDraft()
    }
  }

  const handleSaveServer = async ({ discover = false }: { discover?: boolean } = {}) => {
    setErrorMessage(null)
    const draft = getCreateServerDraft()
    if (!draft) {
      return
    }
    setServerSaving(true)
    setErrorMessage(null)
    setSuccessMessage(null)
    try {
      const payload = {
        name: draft.name,
        transport: draft.transport,
        config: draft.config || {},
        owner_scope_type: draft.owner_scope_type,
        enabled: draft.enabled
      }
      let savedServer: McpHubExternalServer
      if (editingServerId) {
        savedServer = await updateExternalServer(editingServerId, payload)
      } else {
        savedServer = await createExternalServer({
          server_id: draft.server_id,
          ...payload
        })
      }
      let readinessResult: McpHubServerReadiness | null = null
      if (discover) {
        readinessResult = await refreshExternalServerDiscovery(savedServer.id)
      }
      const result = editingServerId
        ? null
        : {
            serverId: savedServer.id,
            serverName: savedServer.name,
            discovered: discover,
            readiness: readinessResult
          }
      resetServerForm({ keepResult: Boolean(result) })
      await loadServers()
      setActiveServerId(savedServer.id)
      if (result) {
        setSetupResult(result)
      }
      setSuccessMessage(editingServerId ? "Server updated" : result ? null : "Server created")
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Unknown error"
      setErrorMessage(editingServerId ? `Failed to update external server: ${msg}` : `Failed to create external server: ${msg}`)
    } finally {
      setServerSaving(false)
    }
  }

  const handleDeleteServer = (server: McpHubExternalServer) => {
    Modal.confirm({
      title: "Delete External Server",
      content: `Are you sure you want to delete the server "${server.name}"? This cannot be undone.`,
      okText: "Delete",
      okType: "danger",
      cancelText: "Cancel",
      onOk: async () => {
        setErrorMessage(null)
        setSuccessMessage(null)
        try {
          await deleteExternalServer(server.id)
          await loadServers()
          setSuccessMessage("Server deleted")
        } catch (err) {
          const msg = err instanceof Error ? err.message : "Unknown error"
          setErrorMessage(`Failed to delete external server: ${msg}`)
        }
      }
    })
  }

  const handleValidateServer = async (server: McpHubExternalServer) => {
    const loadingKey = `${server.id}:validate`
    setRowActionLoadingKey(loadingKey)
    setErrorMessage(null)
    setSuccessMessage(null)
    try {
      await validateExternalServer(server.id)
      await loadServers()
      setSuccessMessage("Server validated")
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Unknown error"
      setErrorMessage(`Failed to validate external server: ${msg}`)
    } finally {
      setRowActionLoadingKey(null)
    }
  }

  const handleRefreshServerDiscovery = async (server: McpHubExternalServer) => {
    const loadingKey = `${server.id}:refresh_discovery`
    setRowActionLoadingKey(loadingKey)
    setErrorMessage(null)
    setSuccessMessage(null)
    try {
      await refreshExternalServerDiscovery(server.id)
      await loadServers()
      setSuccessMessage("Tool discovery refreshed")
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Unknown error"
      setErrorMessage(`Failed to refresh tool discovery: ${msg}`)
    } finally {
      setRowActionLoadingKey(null)
    }
  }

  const handleRefreshSetupResult = async () => {
    if (!setupResult) {
      return
    }
    const loadingKey = `${setupResult.serverId}:setup_refresh`
    setRowActionLoadingKey(loadingKey)
    setErrorMessage(null)
    try {
      const readinessResult = await refreshExternalServerDiscovery(setupResult.serverId)
      await loadServers()
      setSetupResult({
        ...setupResult,
        discovered: true,
        readiness: readinessResult
      })
      setSuccessMessage("Tool discovery refreshed")
    } catch (err) {
      setErrorMessage(`Failed to refresh tool discovery: ${getErrorMessage(err)}`)
    } finally {
      setRowActionLoadingKey(null)
    }
  }

  const handleOpenCredentials = (server: McpHubExternalServer) => {
    setFocusedServerId(server.id)
    setActiveServerId(server.id)
  }

  const renderReadinessAction = (
    action: McpReadinessAction | McpHubReadinessAction,
    server: McpHubExternalServer
  ) => {
    switch (action) {
      case "validate":
        return (
          <Button
            key={action}
            size="small"
            loading={rowActionLoadingKey === `${server.id}:validate`}
            onClick={() => void handleValidateServer(server)}
          >
            Validate
          </Button>
        )
      case "refresh_discovery":
        return (
          <Button
            key={action}
            size="small"
            loading={rowActionLoadingKey === `${server.id}:refresh_discovery`}
            onClick={() => void handleRefreshServerDiscovery(server)}
          >
            Refresh tools
          </Button>
        )
      case "edit_config":
        return (
          <Button key={action} size="small" onClick={() => openEditForm(server)}>
            Edit config
          </Button>
        )
      case "open_credentials":
        return (
          <Button key={action} size="small" onClick={() => handleOpenCredentials(server)}>
            Credentials
          </Button>
        )
      case "view_details":
        return (
          <Button key={action} size="small" onClick={() => setDetailsServerId(server.id)}>
            Details
          </Button>
        )
      case "open_tool_catalog":
        if (!onOpenToolCatalog) {
          return null
        }
        return (
          <Button key={action} size="small" onClick={onOpenToolCatalog}>
            Tool Catalog
          </Button>
        )
      case "add_server":
      case "open_audit":
        return null
    }
  }

  const openCreateSlotForm = () => {
    resetSlotForm()
    setSlotFormOpen(true)
  }

  const openEditSlotForm = (slot: McpHubExternalServerCredentialSlot) => {
    setSlotFormOpen(true)
    setEditingSlotName(slot.slot_name)
    setSlotNameValue(slot.slot_name)
    setSlotDisplayNameValue(slot.display_name)
    setSlotSecretKindValue(slot.secret_kind)
    setSlotPrivilegeClassValue(slot.privilege_class)
    setSlotIsRequiredValue(slot.is_required)
  }

  const handleSaveSlot = async () => {
    if (!activeManagedServer) return
    if (!slotDisplayNameValue.trim() || !slotSecretKindValue.trim() || !slotPrivilegeClassValue.trim()) {
      setErrorMessage("Slot display name, secret kind, and privilege class are required.")
      return
    }
    if (!editingSlotName && !slotNameValue.trim()) {
      setErrorMessage("Slot name is required.")
      return
    }
    setSlotSaving(true)
    setErrorMessage(null)
    setSuccessMessage(null)
    try {
      if (editingSlotName) {
        await updateExternalServerCredentialSlot(activeManagedServer.id, editingSlotName, {
          display_name: slotDisplayNameValue.trim(),
          secret_kind: slotSecretKindValue.trim(),
          privilege_class: slotPrivilegeClassValue.trim(),
          is_required: slotIsRequiredValue
        })
      } else {
        await createExternalServerCredentialSlot(activeManagedServer.id, {
          slot_name: slotNameValue.trim(),
          display_name: slotDisplayNameValue.trim(),
          secret_kind: slotSecretKindValue.trim(),
          privilege_class: slotPrivilegeClassValue.trim(),
          is_required: slotIsRequiredValue
        })
      }
      const nextActiveSlot = editingSlotName || slotNameValue.trim()
      resetSlotForm()
      await loadServers()
      setActiveSlotName(nextActiveSlot)
      setSuccessMessage(editingSlotName ? "Credential slot updated" : "Credential slot created")
    } catch {
      setErrorMessage(editingSlotName ? "Failed to update credential slot." : "Failed to create credential slot.")
    } finally {
      setSlotSaving(false)
    }
  }

  const handleDeleteSlot = (slot: McpHubExternalServerCredentialSlot) => {
    if (!activeManagedServer) return
    Modal.confirm({
      title: "Delete Credential Slot",
      content: `Are you sure you want to delete the credential slot "${slot.display_name}"? This cannot be undone.`,
      okText: "Delete",
      okType: "danger",
      cancelText: "Cancel",
      onOk: async () => {
        const slotKey = `${activeManagedServer.id}:${slot.slot_name}`
        setSlotDeletingKey(slotKey)
        setErrorMessage(null)
        setSuccessMessage(null)
        try {
          await deleteExternalServerCredentialSlot(activeManagedServer.id, slot.slot_name)
          await loadServers()
          setSuccessMessage("Credential slot deleted")
        } catch (err) {
          const msg = err instanceof Error ? err.message : "Unknown error"
          setErrorMessage(`Failed to delete credential slot: ${msg}`)
        } finally {
          setSlotDeletingKey(null)
        }
      }
    })
  }

  const handleAddAuthTemplateMapping = () => {
    if (!activeAuthTemplateTarget || activeSlots.length === 0) return
    setAuthTemplateMappings((current) => [
      ...current,
      {
        slot_name: activeSlots[0]?.slot_name || "",
        target_type: activeAuthTemplateTarget,
        target_name: "",
        prefix: "",
        suffix: "",
        required: true
      }
    ])
  }

  const handleAuthTemplateMappingChange = (
    index: number,
    field: keyof McpHubExternalServerAuthTemplateMapping,
    value: string | boolean
  ) => {
    setAuthTemplateMappings((current) =>
      current.map((mapping, currentIndex) =>
        currentIndex === index
          ? {
              ...mapping,
              [field]: value
            }
          : mapping
      )
    )
  }

  const handleRemoveAuthTemplateMapping = (index: number) => {
    setAuthTemplateMappings((current) => current.filter((_, currentIndex) => currentIndex !== index))
  }

  const handleSaveAuthTemplate = async () => {
    if (!activeManagedServer || !activeAuthTemplateTarget) return
    if (!authTemplateMappings.length) {
      setErrorMessage("Auth template requires at least one mapping.")
      return
    }
    if (
      authTemplateMappings.some(
        (mapping) => !mapping.slot_name.trim() || !mapping.target_name.trim()
      )
    ) {
      setErrorMessage("Each auth template mapping requires a slot and target name.")
      return
    }

    const serverId = activeManagedServer.id
    setAuthTemplateSaving(true)
    setErrorMessage(null)
    setSuccessMessage(null)
    try {
      const template = await updateExternalServerAuthTemplate(serverId, {
        mode: "template",
        mappings: authTemplateMappings.map((mapping) =>
          normalizeAuthTemplateMapping(mapping, activeAuthTemplateTarget)
        )
      })
      setAuthTemplateMappings(
        (template.mappings || []).map((mapping) =>
          normalizeAuthTemplateMapping(mapping, activeAuthTemplateTarget)
        )
      )
      await loadServers()
      setSuccessMessage("Auth template updated")
    } catch {
      setErrorMessage("Failed to update external server auth template.")
    } finally {
      setAuthTemplateSaving(false)
    }
  }

  return (
    <Space orientation="vertical" size="middle" style={{ width: "100%" }}>
      <Typography.Text type="secondary">
        Managed external MCP servers are executable here. Legacy file or environment servers remain
        visible as read-only inventory until they are imported into MCP Hub.
      </Typography.Text>
      {errorMessage ? (
        <StatePanel state="error" title={errorMessage} role="alert" aria-live="assertive" />
      ) : null}
      {successMessage ? (
        <StatePanel state="ready" title={successMessage} aria-live="polite" />
      ) : null}
      {readinessWarningMessage ? (
        <StatePanel
          state="degraded"
          title="Readiness details are limited"
          message={readinessWarningMessage}
          role="status"
          aria-live="polite"
        />
      ) : null}

      <Button type="primary" onClick={openCreateForm}>
        New Managed Server
      </Button>

      {setupResult ? (
        <StatePanel
          state={setupResult.discovered ? "ready" : "setup_required"}
          title={`${setupResult.serverName} saved`}
          message={
            setupResult.discovered
              ? setupResult.readiness?.message || "Tool discovery ran for this server."
              : "Tool discovery has not run for this server yet."
          }
          primaryAction={
            onOpenToolCatalog
              ? {
                  label: "Tool Catalog",
                  onClick: onOpenToolCatalog
                }
              : undefined
          }
          secondaryActions={
            setupResult.discovered
              ? []
              : [
                  {
                    label: "Refresh discovery",
                    loading: rowActionLoadingKey === `${setupResult.serverId}:setup_refresh`,
                    onClick: () => void handleRefreshSetupResult()
                  }
                ]
          }
        />
      ) : null}

      {serverFormOpen ? (
        <Card title={editingServerId ? "Edit Managed Server" : "Create Managed Server"}>
          <Space orientation="vertical" size="middle" style={{ width: "100%" }}>
            {!editingServerId && setupMode === "choice" ? (
              <>
                <Typography.Text type="secondary">
                  Choose the fastest setup path. You can still switch to manual JSON if needed.
                </Typography.Text>
                <Space wrap>
                  <Button onClick={() => setSetupMode("stdio")}>Local stdio</Button>
                  <Button onClick={() => setSetupMode("http")}>HTTP/SSE</Button>
                  <Button onClick={() => setSetupMode("import")}>Import config</Button>
                  <Button onClick={() => setSetupMode("advanced")}>Advanced/manual</Button>
                </Space>
              </>
            ) : null}

            {!editingServerId && setupMode !== "choice" ? (
              <Button onClick={() => setSetupMode("choice")}>Change setup type</Button>
            ) : null}

            {!editingServerId && (setupMode === "stdio" || setupMode === "http") ? (
              <>
                <Space orientation="vertical" style={{ width: "100%" }}>
                  <label htmlFor="mcp-external-server-id">Server ID</label>
                  <input
                    id="mcp-external-server-id"
                    aria-label="Server ID"
                    value={serverIdValue}
                    onChange={(event) => setServerIdValue(event.target.value)}
                    placeholder="docs-managed"
                  />
                </Space>

                <Space orientation="vertical" style={{ width: "100%" }}>
                  <label htmlFor="mcp-external-server-name">Name</label>
                  <input
                    id="mcp-external-server-name"
                    aria-label="Name"
                    value={serverNameValue}
                    onChange={(event) => setServerNameValue(event.target.value)}
                    placeholder="Docs Managed"
                  />
                </Space>
              </>
            ) : null}

            {!editingServerId && setupMode === "stdio" ? (
              <>
                <Space orientation="vertical" style={{ width: "100%" }}>
                  <label htmlFor="mcp-external-server-command">Command</label>
                  <input
                    id="mcp-external-server-command"
                    aria-label="Command"
                    value={stdioCommandValue}
                    onChange={(event) => setStdioCommandValue(event.target.value)}
                    placeholder="uvx"
                  />
                </Space>

                <Space orientation="vertical" style={{ width: "100%" }}>
                  <label htmlFor="mcp-external-server-args">Args</label>
                  <input
                    id="mcp-external-server-args"
                    aria-label="Args"
                    value={stdioArgsValue}
                    onChange={(event) => setStdioArgsValue(event.target.value)}
                    placeholder="mcp-server-docs --stdio"
                  />
                </Space>

                <Space orientation="vertical" style={{ width: "100%" }}>
                  <label htmlFor="mcp-external-server-cwd">Working Directory</label>
                  <input
                    id="mcp-external-server-cwd"
                    aria-label="Working Directory"
                    value={stdioCwdValue}
                    onChange={(event) => setStdioCwdValue(event.target.value)}
                    placeholder="/path/to/project"
                  />
                </Space>

                <Space orientation="vertical" style={{ width: "100%" }}>
                  <label htmlFor="mcp-external-server-env">Env vars</label>
                  <textarea
                    id="mcp-external-server-env"
                    aria-label="Env vars"
                    value={stdioEnvValue}
                    onChange={(event) => setStdioEnvValue(event.target.value)}
                    rows={3}
                    placeholder="TOKEN=..."
                  />
                </Space>
              </>
            ) : null}

            {!editingServerId && setupMode === "http" ? (
              <>
                <Space orientation="vertical" style={{ width: "100%" }}>
                  <label htmlFor="mcp-external-server-url">URL</label>
                  <input
                    id="mcp-external-server-url"
                    aria-label="URL"
                    value={httpUrlValue}
                    onChange={(event) => setHttpUrlValue(event.target.value)}
                    placeholder="https://example.test/mcp"
                  />
                </Space>

                <Space orientation="vertical" style={{ width: "100%" }}>
                  <label htmlFor="mcp-external-server-headers">Headers JSON</label>
                  <textarea
                    id="mcp-external-server-headers"
                    aria-label="Headers JSON"
                    value={httpHeadersText}
                    onChange={(event) => setHttpHeadersText(event.target.value)}
                    rows={4}
                  />
                </Space>
              </>
            ) : null}

            {!editingServerId && setupMode === "import" ? (
              <>
                <Space orientation="vertical" style={{ width: "100%" }}>
                  <label htmlFor="mcp-external-server-import-config">Managed Config JSON</label>
                  <textarea
                    id="mcp-external-server-import-config"
                    aria-label="Managed Config JSON"
                    value={importConfigText}
                    onChange={(event) => setImportConfigText(event.target.value)}
                    rows={8}
                    placeholder='{"server_id":"docs-http","name":"Docs HTTP","transport":"sse","config":{"url":"https://example.test/mcp"}}'
                  />
                </Space>
                {importPreview?.draft ? (
                  <StatePanel
                    state="ready"
                    title={`Preview: ${importPreview.draft.name}`}
                    message={`Server ID: ${importPreview.draft.server_id}; Transport: ${importPreview.draft.transport}`}
                  />
                ) : importPreview?.error ? (
                  <StatePanel state="error" title={importPreview.error} role="alert" />
                ) : null}
              </>
            ) : null}

            {(editingServerId || setupMode === "advanced") ? (
              <>
                {!editingServerId ? (
                  <Space orientation="vertical" style={{ width: "100%" }}>
                    <label htmlFor="mcp-external-server-id">Server ID</label>
                    <input
                      id="mcp-external-server-id"
                      aria-label="Server ID"
                      value={serverIdValue}
                      onChange={(event) => setServerIdValue(event.target.value)}
                      placeholder="docs-managed"
                    />
                  </Space>
                ) : null}

                <Space orientation="vertical" style={{ width: "100%" }}>
                  <label htmlFor="mcp-external-server-name">Name</label>
                  <input
                    id="mcp-external-server-name"
                    aria-label="Name"
                    value={serverNameValue}
                    onChange={(event) => setServerNameValue(event.target.value)}
                    placeholder="Docs Managed"
                  />
                </Space>

                <Space>
                  <Space orientation="vertical">
                    <span className="flex items-center gap-1">
                      <label htmlFor="mcp-external-server-transport">Transport</label>
                      <Tooltip title="How to communicate with the server. Use 'stdio' for local processes, 'websocket' for remote servers.">
                        <button
                          type="button"
                          aria-label="Connection mode help"
                          style={{ border: 0, background: "transparent", padding: 0, cursor: "help", lineHeight: 1 }}
                        >
                          <Typography.Text type="secondary">
                            <QuestionCircleOutlined />
                          </Typography.Text>
                        </button>
                      </Tooltip>
                    </span>
                    <select
                      id="mcp-external-server-transport"
                      aria-label="Transport"
                      value={transportValue}
                      onChange={(event) => setTransportValue(event.target.value)}
                    >
                      <option value="stdio">stdio</option>
                      <option value="websocket">websocket</option>
                    </select>
                  </Space>
                  <Space orientation="vertical">
                    <label htmlFor="mcp-external-server-scope">Owner Scope</label>
                    <select
                      id="mcp-external-server-scope"
                      aria-label="Owner Scope"
                      value={ownerScopeType}
                      onChange={(event) =>
                        setOwnerScopeType(event.target.value as typeof ownerScopeType)
                      }
                    >
                      <option value="global">Global</option>
                      <option value="org">Org</option>
                      <option value="team">Team</option>
                      <option value="user">User</option>
                    </select>
                  </Space>
                </Space>

                <Space orientation="vertical" style={{ width: "100%" }}>
                  <label htmlFor="mcp-external-server-config">Config JSON</label>
                  <textarea
                    id="mcp-external-server-config"
                    aria-label="Config JSON"
                    value={configText}
                    onChange={(event) => setConfigText(event.target.value)}
                    rows={6}
                  />
                </Space>
              </>
            ) : null}

            {!editingServerId && (setupMode === "stdio" || setupMode === "http") ? (
              <Space orientation="vertical">
                <label htmlFor="mcp-external-server-scope">Owner Scope</label>
                <select
                  id="mcp-external-server-scope"
                  aria-label="Owner Scope"
                  value={ownerScopeType}
                  onChange={(event) =>
                    setOwnerScopeType(event.target.value as typeof ownerScopeType)
                  }
                >
                  <option value="global">Global</option>
                  <option value="org">Org</option>
                  <option value="team">Team</option>
                  <option value="user">User</option>
                </select>
              </Space>
            ) : null}

            {(setupMode !== "choice" && setupMode !== "import") || editingServerId ? (
              <Checkbox checked={enabledValue} onChange={(event) => setEnabledValue(event.target.checked)}>
                Enabled
              </Checkbox>
            ) : null}

            <Space>
              {editingServerId ? (
                <Button
                  type="primary"
                  onClick={() => void handleSaveServer()}
                  loading={serverSaving}
                >
                  Update Server
                </Button>
              ) : setupMode !== "choice" ? (
                <>
                  <Button
                    type="primary"
                    onClick={() => void handleSaveServer({ discover: true })}
                    loading={serverSaving}
                  >
                    Save and discover tools
                  </Button>
                  <Button
                    onClick={() => void handleSaveServer({ discover: false })}
                    loading={serverSaving}
                  >
                    Save without discovery
                  </Button>
                </>
              ) : null}
              <Button onClick={() => resetServerForm()}>Cancel</Button>
            </Space>
          </Space>
        </Card>
      ) : null}

      {managedServers.length > 0 ? (
        <Space>
          <label htmlFor="mcp-external-server">Server</label>
          <select
            id="mcp-external-server"
            aria-label="Server"
            value={activeServerId}
            onChange={(event) => setActiveServerId(event.target.value)}
          >
            {managedServers.map((server) => (
              <option key={server.id} value={server.id}>
                {server.name}
              </option>
            ))}
          </select>
        </Space>
      ) : (
        <StatePanel
          state="setup_required"
          title="No external servers connected"
          message="External MCP servers extend your AI assistant with tools like web search, code execution, and more. Click 'New Managed Server' above to add one, or import a legacy server from the list below."
        />
      )}

      {activeManagedServer ? (
        <>
          <Card size="small" title="Credential Slots" extra={<Button onClick={openCreateSlotForm}>Add Slot</Button>}>
            <Space orientation="vertical" size="middle" style={{ width: "100%" }}>
              {slotFormOpen ? (
                <Card size="small" title={editingSlotName ? "Edit Credential Slot" : "Create Credential Slot"}>
                  <Space orientation="vertical" size="middle" style={{ width: "100%" }}>
                    {!editingSlotName ? (
                      <Space orientation="vertical" style={{ width: "100%" }}>
                        <label htmlFor="mcp-external-slot-name">Slot Name</label>
                        <input
                          id="mcp-external-slot-name"
                          aria-label="Slot Name"
                          value={slotNameValue}
                          onChange={(event) => setSlotNameValue(event.target.value)}
                          placeholder="token_readonly"
                        />
                      </Space>
                    ) : null}
                    <Space orientation="vertical" style={{ width: "100%" }}>
                      <label htmlFor="mcp-external-slot-display-name">Slot Display Name</label>
                      <input
                        id="mcp-external-slot-display-name"
                        aria-label="Slot Display Name"
                        value={slotDisplayNameValue}
                        onChange={(event) => setSlotDisplayNameValue(event.target.value)}
                        placeholder="Read-only token"
                      />
                    </Space>
                    <Space>
                      <Space orientation="vertical">
                        <span className="flex items-center gap-1">
                          <label htmlFor="mcp-external-slot-secret-kind">Secret Kind</label>
                          <Tooltip title="The type of credential needed. 'bearer_token' for API keys, 'api_key' for simple keys, 'client_secret' for OAuth.">
                            <button
                              type="button"
                              aria-label="Secret kind help"
                              style={{ border: 0, background: "transparent", padding: 0, cursor: "help", lineHeight: 1 }}
                            >
                              <Typography.Text type="secondary">
                                <QuestionCircleOutlined />
                              </Typography.Text>
                            </button>
                          </Tooltip>
                        </span>
                        <select
                          id="mcp-external-slot-secret-kind"
                          aria-label="Secret Kind"
                          value={slotSecretKindValue}
                          onChange={(event) => setSlotSecretKindValue(event.target.value)}
                        >
                          <option value="bearer_token">bearer_token</option>
                          <option value="api_key">api_key</option>
                          <option value="client_secret">client_secret</option>
                        </select>
                      </Space>
                      <Space orientation="vertical">
                        <label htmlFor="mcp-external-slot-privilege-class">Privilege Class</label>
                        <select
                          id="mcp-external-slot-privilege-class"
                          aria-label="Privilege Class"
                          value={slotPrivilegeClassValue}
                          onChange={(event) => setSlotPrivilegeClassValue(event.target.value)}
                        >
                          <option value="read">read</option>
                          <option value="write">write</option>
                          <option value="admin">admin</option>
                        </select>
                      </Space>
                    </Space>
                    <Checkbox checked={slotIsRequiredValue} onChange={(event) => setSlotIsRequiredValue(event.target.checked)}>
                      Required
                    </Checkbox>
                    <Space>
                      <Button type="primary" onClick={handleSaveSlot} loading={slotSaving}>
                        {editingSlotName ? "Update Slot" : "Save Slot"}
                      </Button>
                      <Button onClick={resetSlotForm}>Cancel</Button>
                    </Space>
                  </Space>
                </Card>
              ) : null}

              <List
                bordered
                dataSource={activeSlots}
                locale={{ emptyText: <Empty description="No credential slots yet." /> }}
                renderItem={(slot) => {
                  const slotKey = `${activeManagedServer.id}:${slot.slot_name}`
                  return (
                    <List.Item>
                      <Space wrap size="small" style={{ width: "100%", justifyContent: "space-between" }}>
                        <Space wrap size="small">
                          <Typography.Text strong>{slot.display_name}</Typography.Text>
                          <Tag>{slot.slot_name}</Tag>
                          <Tag>{slot.secret_kind}</Tag>
                          <Tag color={slot.privilege_class === "read" ? "green" : slot.privilege_class === "write" ? "gold" : "red"}>
                            {slot.privilege_class}
                          </Tag>
                          {slot.is_required ? <Tag color="blue">required</Tag> : <Tag>optional</Tag>}
                          {slot.secret_configured ? <Tag color="green">secret configured</Tag> : <Tag>no secret</Tag>}
                        </Space>
                        <Space>
                          <Button size="small" aria-label={`Edit ${slot.display_name}`} onClick={() => openEditSlotForm(slot)}>
                            Edit
                          </Button>
                          <Button
                            size="small"
                            danger
                            aria-label={`Delete ${slot.display_name}`}
                            loading={slotDeletingKey === slotKey}
                            onClick={() => void handleDeleteSlot(slot)}
                          >
                            Delete
                          </Button>
                        </Space>
                      </Space>
                    </List.Item>
                  )
                }}
              />
            </Space>
          </Card>

          <Card
            size="small"
            title="Auth Template"
            extra={
              <Button
                onClick={handleAddAuthTemplateMapping}
                disabled={activeCredentialState === "not_required" || !activeAuthTemplateTarget || activeSlots.length === 0}
              >
                Add Mapping
              </Button>
            }
          >
            <Space orientation="vertical" size="middle" style={{ width: "100%" }}>
              {activeCredentialState === "not_required" ? (
                <StatePanel
                  state="ready"
                  title="No credentials required"
                  message="This server does not need runtime credentials or an auth template."
                />
              ) : activeManagedServer.auth_template_valid ? (
                <StatePanel state="ready" title="Template valid" />
              ) : (
                <StatePanel
                  state={activeManagedServer.auth_template_present ? "degraded" : "setup_required"}
                  title={activeAuthTemplateBlockedReason || "No auth template"}
                  message={
                    activeSlots.length === 0
                      ? "Add credential slots before defining how this server hydrates runtime auth."
                      : activeManagedServer.auth_template_present
                        ? "Fix the template or missing slot secrets before this managed server becomes fully ready."
                        : "Create a transport-specific auth template to map granted credential slots into runtime auth."
                  }
                />
              )}
              <Space wrap size="small">
                <Tag>{`Transport: ${activeManagedServer.transport}`}</Tag>
                {activeAuthTemplateTarget ? (
                  <Tag color="blue">{`Template target: ${activeAuthTemplateTarget === "header" ? "header" : "env"}`}</Tag>
                ) : (
                  <Tag color="red">Unsupported transport</Tag>
                )}
              </Space>
              {activeCredentialState === "not_required" ? null : activeSlots.length === 0 ? (
                <Empty description="Add at least one credential slot before creating an auth template." />
              ) : authTemplateMappings.length === 0 && !authTemplateLoading ? (
                <Empty description="No auth template mappings configured." />
              ) : null}
              {authTemplateMappings.map((mapping, index) => (
                <Card key={`${mapping.slot_name || "slot"}-${index}`} size="small" title={`Mapping ${index + 1}`}>
                  <Space orientation="vertical" size="middle" style={{ width: "100%" }}>
                    <Space wrap>
                      <Space orientation="vertical">
                        <label htmlFor={`mcp-auth-template-slot-${index}`}>Credential Slot</label>
                        <select
                          id={`mcp-auth-template-slot-${index}`}
                          aria-label={`Credential Slot ${index + 1}`}
                          value={mapping.slot_name}
                          onChange={(event) =>
                            handleAuthTemplateMappingChange(index, "slot_name", event.target.value)
                          }
                        >
                          {activeSlots.map((slot) => (
                            <option key={slot.slot_name} value={slot.slot_name}>
                              {slot.display_name}
                            </option>
                          ))}
                        </select>
                      </Space>
                      <Space orientation="vertical">
                        <label htmlFor={`mcp-auth-template-target-${index}`}>Target</label>
                        <input
                          id={`mcp-auth-template-target-${index}`}
                          aria-label={`Target Name ${index + 1}`}
                          value={mapping.target_name}
                          onChange={(event) =>
                            handleAuthTemplateMappingChange(index, "target_name", event.target.value)
                          }
                          placeholder={activeAuthTemplateTarget === "header" ? "Authorization" : "API_KEY"}
                        />
                      </Space>
                    </Space>
                    <Space wrap>
                      <Space orientation="vertical">
                        <label htmlFor={`mcp-auth-template-prefix-${index}`}>Prefix</label>
                        <input
                          id={`mcp-auth-template-prefix-${index}`}
                          aria-label={`Prefix ${index + 1}`}
                          value={mapping.prefix || ""}
                          onChange={(event) =>
                            handleAuthTemplateMappingChange(index, "prefix", event.target.value)
                          }
                          placeholder={activeAuthTemplateTarget === "header" ? "Bearer " : ""}
                        />
                      </Space>
                      <Space orientation="vertical">
                        <label htmlFor={`mcp-auth-template-suffix-${index}`}>Suffix</label>
                        <input
                          id={`mcp-auth-template-suffix-${index}`}
                          aria-label={`Suffix ${index + 1}`}
                          value={mapping.suffix || ""}
                          onChange={(event) =>
                            handleAuthTemplateMappingChange(index, "suffix", event.target.value)
                          }
                        />
                      </Space>
                    </Space>
                    <Space wrap size="small" style={{ justifyContent: "space-between", width: "100%" }}>
                      <Checkbox
                        checked={mapping.required !== false}
                        onChange={(event) =>
                          handleAuthTemplateMappingChange(index, "required", event.target.checked)
                        }
                      >
                        Required
                      </Checkbox>
                      <Space wrap size="small">
                        <Tag>{mapping.target_type}</Tag>
                        <Button
                          size="small"
                          danger
                          aria-label={`Remove auth mapping ${index + 1}`}
                          onClick={() => handleRemoveAuthTemplateMapping(index)}
                        >
                          Remove
                        </Button>
                      </Space>
                    </Space>
                  </Space>
                </Card>
              ))}
              <Button
                type="primary"
                onClick={handleSaveAuthTemplate}
                disabled={!canSaveAuthTemplate}
                loading={authTemplateSaving || authTemplateLoading}
              >
                Save Auth Template
              </Button>
            </Space>
          </Card>

          {activeSlots.length > 0 ? (
            <Card size="small" title="Slot Secret">
              <Space orientation="vertical" size="middle" style={{ width: "100%" }}>
                <Space>
                  <label htmlFor="mcp-external-slot">Slot</label>
                  <select
                    id="mcp-external-slot"
                    aria-label="Credential Slot"
                    value={activeSlotName}
                    onChange={(event) => setActiveSlotName(event.target.value)}
                  >
                    {activeSlots.map((slot) => (
                      <option key={slot.slot_name} value={slot.slot_name}>
                        {slot.display_name}
                      </option>
                    ))}
                  </select>
                </Space>
                <Space orientation="vertical" style={{ width: "100%" }}>
                  <label htmlFor="mcp-external-slot-secret">Slot Secret</label>
                  <input
                    id="mcp-external-slot-secret"
                    aria-label="Slot Secret"
                    type="password"
                    value={slotSecretValue}
                    onChange={(event) => setSlotSecretValue(event.target.value)}
                    placeholder="Paste slot secret"
                  />
                </Space>
                <Space>
                  <Button type="primary" onClick={handleSaveSlotSecret} disabled={!canSaveSlotSecret} loading={slotSecretSaving}>
                    Save Slot Secret
                  </Button>
                  <Button onClick={handleClearSlotSecret} disabled={!activeSlotName} loading={slotSecretClearing}>
                    Clear Slot Secret
                  </Button>
                </Space>
              </Space>
            </Card>
          ) : activeCredentialState === "not_required" ? (
            <Card size="small" title="No credentials required">
              <StatePanel
                state="ready"
                title="No credentials required"
                message="This managed server can run without credential slots, auth templates, or a server-level secret."
              />
            </Card>
          ) : (
            <Card size="small" title="Legacy Secret Fallback">
              <Space orientation="vertical" style={{ width: "100%" }}>
                <Typography.Text type="secondary">
                  This managed server still uses the transitional server-level secret flow until credential slots are defined.
                </Typography.Text>
                <label htmlFor="mcp-external-secret">Secret</label>
                <input
                  id="mcp-external-secret"
                  aria-label="Secret"
                  type="password"
                  value={secretValue}
                  onChange={(event) => setSecretValue(event.target.value)}
                  placeholder="Paste secret token"
                />
                <Button type="primary" onClick={handleSaveSecret} disabled={!canSave} loading={saving}>
                  Save Secret
                </Button>
              </Space>
            </Card>
          )}
        </>
      ) : null}

      <List
        bordered
        loading={loading}
        dataSource={servers}
        locale={{
          emptyText: (
            <Empty
              description={
                <Space orientation="vertical" size={4}>
                  <Typography.Text type="secondary">No external servers configured</Typography.Text>
                  <Typography.Text type="secondary" style={{ fontSize: 13 }}>
                    External MCP servers extend your AI assistant with tools like web search, code execution, and more.
                  </Typography.Text>
                </Space>
              }
            >
              <Button type="primary" onClick={openCreateForm}>
                Add New Server
              </Button>
            </Empty>
          )
        }}
        renderItem={(server) => {
          const readiness = rowReadinessByServerId.get(server.id)
          const credentialTag = readiness ? CREDENTIAL_TAGS[readiness.credentialState] : null
          return (
            <List.Item>
              <Space wrap size="small" style={{ width: "100%", justifyContent: "space-between" }}>
                <Space wrap size="small">
                  <Typography.Text strong>{server.name}</Typography.Text>
                  {focusedServerId === server.id ? <Tag color="blue">focused from audit</Tag> : null}
                  {server.server_source === "legacy" ? (
                    <Tag>legacy read only</Tag>
                  ) : (
                    <Tag color="green">managed</Tag>
                  )}
                  {readiness ? (
                    <>
                      <Tag color={READINESS_DISPLAY_COLORS[readiness.displayState]}>
                        {READINESS_DISPLAY_LABELS[readiness.displayState]}
                      </Tag>
                      <Tag>{`${readiness.toolCount} ${readiness.toolCount === 1 ? "tool" : "tools"}`}</Tag>
                    </>
                  ) : null}
                  {credentialTag ? (
                    <Tag color={credentialTag.color}>{credentialTag.label}</Tag>
                  ) : server.secret_configured ? (
                    <Tag color="green">secret configured</Tag>
                  ) : null}
                  {server.auth_template_valid ? (
                    <Tag color="green">template valid</Tag>
                  ) : server.auth_template_present ? (
                    <Tag color="orange">
                      {getExternalAuthTemplateBlockedReasonLabel(server.auth_template_blocked_reason) || "Template needs review"}
                    </Tag>
                  ) : server.credential_slots?.length ? (
                    <Tag>template not configured</Tag>
                  ) : null}
                  {readiness?.message ? (
                    <Typography.Text type="secondary">{readiness.message}</Typography.Text>
                  ) : null}
                  {server.runtime_executable ? <Tag color="green">runtime executable</Tag> : <Tag>inventory only</Tag>}
                  <Tag>{`${server.binding_count || 0} ${(server.binding_count || 0) === 1 ? "binding" : "bindings"}`}</Tag>
                  {server.credential_slots?.length ? (
                    <Tag color="blue">{`${server.credential_slots.length} slot${server.credential_slots.length === 1 ? "" : "s"}`}</Tag>
                  ) : null}
                  {server.superseded_by_server_id ? (
                    <Tag color="blue">{`superseded by ${server.superseded_by_server_id}`}</Tag>
                  ) : null}
                </Space>
                {server.server_source === "legacy" && !server.superseded_by_server_id ? (
                  <Button
                    size="small"
                    onClick={() => void handleImport(server.id)}
                    loading={importingServerId === server.id}
                  >
                    Import to MCP Hub
                  </Button>
                ) : server.server_source !== "legacy" ? (
                  <Space wrap size="small">
                    {readiness?.allowedActions.map((action) =>
                      renderReadinessAction(action, server)
                    )}
                    <Button
                      size="small"
                      aria-label={`Edit ${server.name}`}
                      onClick={() => openEditForm(server)}
                    >
                      Edit
                    </Button>
                    <Button
                      size="small"
                      danger
                      aria-label={`Delete ${server.name}`}
                      onClick={() => handleDeleteServer(server)}
                    >
                      Delete
                    </Button>
                  </Space>
                ) : null}
              </Space>
            </List.Item>
          )
        }}
      />
      <Modal
        title={detailsServer ? `${detailsServer.name} readiness details` : "Server readiness details"}
        open={Boolean(detailsServer)}
        onCancel={() => setDetailsServerId(null)}
        footer={<Button onClick={() => setDetailsServerId(null)}>Close</Button>}
      >
        {detailsServer && detailsReadiness ? (
          <Space orientation="vertical" size="small" style={{ width: "100%" }}>
            <StatePanel
              state={DESIGN_STATE_BY_READINESS_DISPLAY[detailsReadiness.displayState]}
              title={READINESS_DISPLAY_LABELS[detailsReadiness.displayState]}
              message={detailsReadiness.message}
            />
            <Typography.Text>{`Server ID: ${detailsServer.id}`}</Typography.Text>
            <Typography.Text>{`Display state: ${
              detailsBackendReadiness?.display_state ?? detailsReadiness.displayState
            }`}</Typography.Text>
            <Typography.Text>{`Primary reason: ${
              detailsBackendReadiness?.primary_reason_code ??
              detailsReadiness.primaryReasonCode ??
              "none"
            }`}</Typography.Text>
            <Typography.Text>{`Credential state: ${detailsReadiness.credentialState}`}</Typography.Text>
            <Typography.Text>{`Reason codes: ${detailsReadiness.reasonCodes.join(", ") || "none"}`}</Typography.Text>
            <Typography.Text>{`Transport: ${detailsServer.transport}`}</Typography.Text>
            <Typography.Text>{`Tools: ${detailsReadiness.toolCount}`}</Typography.Text>
            <Typography.Text>{`Last validation: ${formatDiagnosticTimestamp(
              detailsBackendReadiness?.last_validation_at
            )}`}</Typography.Text>
            <Typography.Text>{`Last discovery: ${formatDiagnosticTimestamp(
              detailsBackendReadiness?.last_discovery_at
            )}`}</Typography.Text>
            <Typography.Text>{`Last successful discovery: ${formatDiagnosticTimestamp(
              detailsBackendReadiness?.last_successful_discovery_at
            )}`}</Typography.Text>
            <Typography.Text>{`Current operation: ${formatDiagnosticCurrentOperation(
              detailsBackendReadiness?.current_operation
            )}`}</Typography.Text>
            <Typography.Text>{`Last error category: ${formatDiagnosticNullable(
              detailsBackendReadiness?.last_error_category
            )}`}</Typography.Text>
            <Typography.Text>{`Last error message: ${formatDiagnosticNullable(
              detailsBackendReadiness?.last_error_message
            )}`}</Typography.Text>
            <Typography.Text>{`Deployment mode: ${environmentDiagnostics.deploymentMode}`}</Typography.Text>
            <Typography.Text>{`API origin: ${environmentDiagnostics.apiOrigin}`}</Typography.Text>
            <Typography.Text>{`Health endpoint: ${environmentDiagnostics.healthEndpoint}`}</Typography.Text>
            <Typography.Text>{`Latest health result: ${environmentDiagnostics.latestHealthResult}`}</Typography.Text>
            <Typography.Text>
              Audit details: Use the Governance Audit tab to inspect server changes and policy events.
            </Typography.Text>
            <Typography.Text>
              Setup isolation: Use an isolated test database and temporary MCP server config for local walkthroughs and E2E runs before refreshing discovery.
            </Typography.Text>
            <Typography.Text strong>Sanitized config</Typography.Text>
            <pre
              data-testid="mcp-server-diagnostics-config"
              style={{
                background: "var(--ant-color-fill-quaternary)",
                border: "1px solid var(--ant-color-border-secondary)",
                borderRadius: 6,
                margin: 0,
                maxHeight: 240,
                overflow: "auto",
                padding: 12,
                whiteSpace: "pre-wrap",
                wordBreak: "break-word"
              }}
            >
              {detailsDiagnosticConfig}
            </pre>
          </Space>
        ) : null}
      </Modal>
    </Space>
  )
}
