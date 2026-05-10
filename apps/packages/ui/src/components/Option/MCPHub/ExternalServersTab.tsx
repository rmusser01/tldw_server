import { useEffect, useMemo, useRef, useState } from "react"
import { Alert, Button, Card, Checkbox, Empty, List, Modal, Space, Tag, Tooltip, Typography } from "antd"
import { QuestionCircleOutlined } from "@ant-design/icons"
import { useQueryClient } from "@tanstack/react-query"

import {
  clearExternalServerSlotSecret,
  createExternalServer,
  createExternalServerCredentialSlot,
  deleteExternalServer,
  deleteExternalServerCredentialSlot,
  getExternalServerAuthTemplate,
  importExternalServer,
  listExternalServers,
  describeExternalServerDiscoveryRefreshFailure,
  refreshExternalServerDiscovery,
  setExternalServerSecret,
  setExternalServerSlotSecret,
  type McpHubDrillTarget,
  updateExternalServer,
  updateExternalServerAuthTemplate,
  updateExternalServerCredentialSlot,
  type McpHubExternalServer,
  type McpHubExternalServerAuthTemplateMapping,
  type McpHubExternalServerCredentialSlot
} from "@/services/tldw/mcp-hub"

import {
  getExternalAuthTemplateBlockedReasonLabel,
  getManagedExternalServers,
  getManagedExternalServerSlots
} from "./policyHelpers"
import { invalidateMcpRuntimeQueries } from "./runtimeRefresh"

const DEFAULT_SLOT_SECRET_KIND = "bearer_token"
const DEFAULT_SLOT_PRIVILEGE_CLASS = "read"
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

type ExternalServersTabProps = {
  drillTarget?: McpHubDrillTarget | null
  onDrillHandled?: (requestId: number) => void
}

type RuntimeRefreshResult = { ok: true } | { ok: false; message: string }

const getRuntimeRefreshFailureMessage = (result: RuntimeRefreshResult) =>
  "message" in result ? result.message : ""

export const ExternalServersTab = ({
  drillTarget = null,
  onDrillHandled
}: ExternalServersTabProps) => {
  const queryClient = useQueryClient()
  const handledDrillRequestRef = useRef<number | null>(null)
  const [servers, setServers] = useState<McpHubExternalServer[]>([])
  const [serversLoaded, setServersLoaded] = useState(false)
  const [loading, setLoading] = useState(false)
  const [activeServerId, setActiveServerId] = useState<string>("")
  const [secretValue, setSecretValue] = useState("")
  const [saving, setSaving] = useState(false)
  const [importingServerId, setImportingServerId] = useState<string | null>(null)
  const [successMessage, setSuccessMessage] = useState<string | null>(null)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [serverFormOpen, setServerFormOpen] = useState(false)
  const [editingServerId, setEditingServerId] = useState<string | null>(null)
  const [serverIdValue, setServerIdValue] = useState("")
  const [serverNameValue, setServerNameValue] = useState("")
  const [transportValue, setTransportValue] = useState("stdio")
  const [ownerScopeType, setOwnerScopeType] = useState<"global" | "org" | "team" | "user">("global")
  const [enabledValue, setEnabledValue] = useState(true)
  const [configText, setConfigText] = useState("{}")
  const [serverSaving, setServerSaving] = useState(false)
  const [runtimeRefreshCount, setRuntimeRefreshCount] = useState(0)
  const runtimeRefreshing = runtimeRefreshCount > 0
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
    try {
      const rows = await listExternalServers()
      const nextServers = Array.isArray(rows) ? rows : []
      setServers(nextServers)
      const managedRows = getManagedExternalServers(nextServers)
      if (managedRows.some((server) => server.id === activeServerId)) {
        return
      }
      setActiveServerId(managedRows[0]?.id || "")
    } catch (err) {
      setServers([])
      setActiveServerId("")
      const msg = err instanceof Error ? err.message : "Unknown error"
      setErrorMessage(`Failed to load external servers: ${msg}`)
    } finally {
      setLoading(false)
      setServersLoaded(true)
    }
  }

  const refreshRuntimeDiscovery = async (
    serverId?: string
  ): Promise<RuntimeRefreshResult> => {
    setRuntimeRefreshCount((count) => count + 1)
    try {
      const refreshResult = serverId
        ? await refreshExternalServerDiscovery(serverId)
        : await refreshExternalServerDiscovery()
      await invalidateMcpRuntimeQueries(queryClient)
      if (!refreshResult.ok) {
        return {
          ok: false,
          message: describeExternalServerDiscoveryRefreshFailure(refreshResult)
        }
      }
      return { ok: true }
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Unknown error"
      return { ok: false, message: msg }
    } finally {
      setRuntimeRefreshCount((count) => Math.max(0, count - 1))
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
        openEditForm(server)
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
      const refreshResult = await refreshRuntimeDiscovery(imported.id)
      await loadServers()
      setActiveServerId(imported.id)
      if (refreshResult.ok) {
        setSuccessMessage("Legacy server imported and tools refreshed")
      } else {
        setSuccessMessage(
          `Legacy server imported, but discovery refresh failed. Retry runtime refresh. ${getRuntimeRefreshFailureMessage(refreshResult)}`
        )
      }
    } catch {
      setErrorMessage("Failed to import legacy external server.")
    } finally {
      setImportingServerId(null)
    }
  }

  const resetServerForm = () => {
    setServerFormOpen(false)
    setEditingServerId(null)
    setServerIdValue("")
    setServerNameValue("")
    setTransportValue("stdio")
    setOwnerScopeType("global")
    setEnabledValue(true)
    setConfigText("{}")
    setServerSaving(false)
  }

  const openCreateForm = () => {
    resetServerForm()
    setServerFormOpen(true)
  }

  const openEditForm = (server: McpHubExternalServer) => {
    setServerFormOpen(true)
    setEditingServerId(server.id)
    setServerIdValue(server.id)
    setServerNameValue(server.name)
    setTransportValue(server.transport || "stdio")
    setOwnerScopeType(server.owner_scope_type)
    setEnabledValue(Boolean(server.enabled))
    setConfigText(JSON.stringify(server.config || {}, null, 2))
  }

  const handleSaveServer = async () => {
    if (!serverNameValue.trim() || !transportValue.trim()) {
      setErrorMessage("Server name and transport are required.")
      return
    }
    if (!editingServerId && !serverIdValue.trim()) {
      setErrorMessage("Server id is required.")
      return
    }
    let parsedConfig: Record<string, unknown> = {}
    try {
      parsedConfig = JSON.parse(configText || "{}") as Record<string, unknown>
      if (!parsedConfig || typeof parsedConfig !== "object" || Array.isArray(parsedConfig)) {
        throw new Error("config")
      }
    } catch {
      setErrorMessage("Server config JSON must decode to an object.")
      return
    }

    setServerSaving(true)
    setErrorMessage(null)
    setSuccessMessage(null)
    try {
      const payload = {
        name: serverNameValue.trim(),
        transport: transportValue,
        config: parsedConfig,
        owner_scope_type: ownerScopeType,
        enabled: enabledValue
      }
      let refreshServerId: string | undefined
      if (editingServerId) {
        await updateExternalServer(editingServerId, payload)
        refreshServerId = enabledValue ? editingServerId : undefined
      } else {
        const created = await createExternalServer({
          server_id: serverIdValue.trim(),
          ...payload
        })
        refreshServerId = enabledValue ? created.id : undefined
      }
      resetServerForm()
      const refreshResult = await refreshRuntimeDiscovery(refreshServerId)
      await loadServers()
      const action = editingServerId ? "updated" : "created"
      if (refreshResult.ok) {
        setSuccessMessage(`Server ${action} and tools refreshed`)
      } else {
        setSuccessMessage(
          `Server ${action}, but discovery refresh failed. Retry runtime refresh. ${getRuntimeRefreshFailureMessage(refreshResult)}`
        )
      }
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
          const refreshResult = await refreshRuntimeDiscovery()
          await loadServers()
          if (refreshResult.ok) {
            setSuccessMessage("Server deleted and tools refreshed")
          } else {
            setSuccessMessage(
              `Server deleted, but discovery refresh failed. Retry runtime refresh. ${getRuntimeRefreshFailureMessage(refreshResult)}`
            )
          }
        } catch (err) {
          const msg = err instanceof Error ? err.message : "Unknown error"
          setErrorMessage(`Failed to delete external server: ${msg}`)
        }
      }
    })
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
      {errorMessage ? <Alert type="error" title={errorMessage} showIcon /> : null}
      {successMessage ? <Alert type="success" title={successMessage} showIcon /> : null}

      <Button type="primary" onClick={openCreateForm}>
        New Managed Server
      </Button>

      {serverFormOpen ? (
        <Card title={editingServerId ? "Edit Managed Server" : "Create Managed Server"}>
          <Space orientation="vertical" size="middle" style={{ width: "100%" }}>
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
                      aria-label="Transport help"
                      style={{ border: 0, background: "transparent", padding: 0, cursor: "help", lineHeight: 1 }}
                    >
                      <QuestionCircleOutlined style={{ color: "rgba(0,0,0,0.45)" }} />
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

            <Checkbox checked={enabledValue} onChange={(event) => setEnabledValue(event.target.checked)}>
              Enabled
            </Checkbox>

            <Space>
              <Button type="primary" onClick={handleSaveServer} loading={serverSaving || runtimeRefreshing}>
                {editingServerId ? "Update Server" : "Save Server"}
              </Button>
              <Button onClick={resetServerForm}>Cancel</Button>
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
        <Alert
          type="info"
          showIcon
          title="No external servers connected"
          description="External MCP servers extend your AI assistant with tools like web search, code execution, and more. Click 'New Managed Server' above to add one, or import a legacy server from the list below."
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
                              <QuestionCircleOutlined style={{ color: "rgba(0,0,0,0.45)" }} />
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
              <Button onClick={handleAddAuthTemplateMapping} disabled={!activeAuthTemplateTarget || activeSlots.length === 0}>
                Add Mapping
              </Button>
            }
          >
            <Space orientation="vertical" size="middle" style={{ width: "100%" }}>
              {activeManagedServer.auth_template_valid ? (
                <Alert type="success" showIcon title="Template valid" />
              ) : (
                <Alert
                  type={activeManagedServer.auth_template_present ? "warning" : "info"}
                  showIcon
                  title={activeAuthTemplateBlockedReason || "No auth template"}
                  description={
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
              {activeSlots.length === 0 ? (
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
        renderItem={(server) => (
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
                {server.auth_template_valid ? (
                  <Tag color="green">template valid</Tag>
                ) : (
                  <Tag color={server.auth_template_present ? "orange" : "default"}>
                    {getExternalAuthTemplateBlockedReasonLabel(server.auth_template_blocked_reason) || "No auth template"}
                  </Tag>
                )}
                {server.secret_configured ? <Tag color="green">secret configured</Tag> : <Tag>no secret</Tag>}
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
                <Space>
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
        )}
      />
    </Space>
  )
}
