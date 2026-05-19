import React from "react"
import { useTranslation } from "react-i18next"
import { Input, Select, Tag } from "antd"

import { tldwClient } from "@/services/tldw/TldwApiClient"
import { toAllowedPath } from "@/services/tldw/path-utils"
import {
  buildDraftAssistSuggestions,
  type DraftAssistSuggestion
} from "@/utils/persona-command-drafts"

import {
  CommandAnalyticsSummary,
  formatFailureLabel,
  formatLastUsedLabel,
  formatRunLabel,
  type PersonaVoiceAnalytics,
  type PersonaVoiceCommandAnalyticsItem
} from "./CommandAnalyticsSummary"
import { McpToolPicker } from "./McpToolPicker"
import { PERSONA_STARTER_COMMAND_TEMPLATES } from "./personaStarterCommandTemplates"

type VoiceCommandActionType = "mcp_tool" | "workflow" | "custom" | "llm_chat"

type PersonaVoiceCommand = {
  id: string
  persona_id?: string | null
  connection_id?: string | null
  connection_status?: "ok" | "missing" | null
  connection_name?: string | null
  name: string
  phrases: string[]
  action_type: VoiceCommandActionType
  action_config: Record<string, unknown>
  priority: number
  enabled: boolean
  requires_confirmation: boolean
  description?: string | null
  created_at?: string | null
}

type PersonaConnectionSummary = {
  id: string
  name: string
  auth_type?: string | null
  key_hint?: string | null
  secret_configured?: boolean
}

export type CommandDraftSource = "test_lab" | "setup_no_match"

type CommandFormState = {
  commandId: string | null
  name: string
  description: string
  phrasesText: string
  actionType: VoiceCommandActionType
  toolName: string
  workflowId: string
  customAction: string
  connectionId: string
  requestMethod: "GET" | "POST" | "PUT" | "PATCH" | "DELETE"
  requestPath: string
  extractMode: "none" | "query" | "content"
  slotMapText: string
  defaultPayloadText: string
  priority: string
  enabled: boolean
  requiresConfirmation: boolean
}

type CommandTemplate = {
  key: string
  label: string
  description: string
  apply: () => CommandFormState
}

type CommandsPanelProps = {
  selectedPersonaId: string
  selectedPersonaName: string
  isActive?: boolean
  analytics?: PersonaVoiceAnalytics | null
  analyticsLoading?: boolean
  handoffFocusRequest?: {
    section: "command_form" | "command_list"
    token: number
  } | null
  onSetupHandoffFocusConsumed?: (token: number) => void
  openCommandId?: string | null
  onOpenCommandHandled?: (commandId: string) => void
  draftCommandPhrase?: string | null
  draftCommandSource?: CommandDraftSource | null
  onDraftCommandPhraseHandled?: (heardText: string) => void
  rerunAfterSaveCommandId?: string | null
  onRerunAfterSave?: (commandId: string) => void
  onCommandSaved?: (
    commandId: string,
    context: { fromDraft: boolean }
  ) => void
}

const REQUEST_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE"] as const

const DEFAULT_FORM_STATE: CommandFormState = {
  commandId: null,
  name: "",
  description: "",
  phrasesText: "",
  actionType: "mcp_tool",
  toolName: "",
  workflowId: "",
  customAction: "",
  connectionId: "",
  requestMethod: "POST",
  requestPath: "",
  extractMode: "none",
  slotMapText: "{}",
  defaultPayloadText: "{}",
  priority: "50",
  enabled: true,
  requiresConfirmation: false
}

const COMMAND_TEMPLATES: CommandTemplate[] = [
  ...PERSONA_STARTER_COMMAND_TEMPLATES.map((template) => ({
    key: template.key,
    label: template.label,
    description: template.description,
    apply: () => ({
      ...DEFAULT_FORM_STATE,
      name: template.name,
      description: template.commandDescription,
      phrasesText: template.phrases.join("\n"),
      toolName: template.toolName,
      slotMapText: JSON.stringify(template.slotMap, null, 2),
      requiresConfirmation: template.requiresConfirmation
    })
  })),
  {
    key: "external-api",
    label: "External API",
    description: "Call a saved connection with a direct request",
    apply: () => ({
      ...DEFAULT_FORM_STATE,
      name: "Call External API",
      description: "Send a direct request through a saved persona connection",
      phrasesText: "call external api for {query}\nsend api request for {query}",
      actionType: "custom",
      customAction: "external_request",
      requestMethod: "POST",
      requestPath: "",
      slotMapText: JSON.stringify({ query: "query" }, null, 2),
      requiresConfirmation: true
    })
  }
]

const splitPhrases = (value: string): string[] =>
  value
    .split("\n")
    .map((item) => item.trim())
    .filter((item) => item.length > 0)

const stringifyJson = (value: unknown): string =>
  JSON.stringify(
    value && typeof value === "object" && !Array.isArray(value) ? value : {},
    null,
    2
  )

const toDraftCommandName = (phrase: string): string => {
  const normalized = String(phrase || "").trim().replace(/\s+/g, " ")
  if (!normalized) return ""
  return normalized.charAt(0).toUpperCase() + normalized.slice(1)
}

const toDraftFormState = (phrase: string): CommandFormState => {
  const normalized = String(phrase || "").trim().replace(/\s+/g, " ")
  return {
    ...DEFAULT_FORM_STATE,
    name: toDraftCommandName(normalized),
    phrasesText: normalized
  }
}

const parseJsonRecord = (
  rawValue: string,
  label: string
): { ok: true; value: Record<string, unknown> } | { ok: false; error: string } => {
  const trimmed = rawValue.trim()
  if (!trimmed) {
    return { ok: true, value: {} }
  }
  try {
    const parsed = JSON.parse(trimmed)
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      return { ok: false, error: `${label} must be a JSON object.` }
    }
    return { ok: true, value: parsed as Record<string, unknown> }
  } catch {
    return { ok: false, error: `${label} must be valid JSON.` }
  }
}

const normalizeRequestMethod = (
  value: unknown
): CommandFormState["requestMethod"] => {
  const normalized = String(value || "POST").trim().toUpperCase()
  return REQUEST_METHODS.includes(normalized as CommandFormState["requestMethod"])
    ? (normalized as CommandFormState["requestMethod"])
    : "POST"
}

const resolveCommandConnectionStatus = (
  command: PersonaVoiceCommand,
  availableConnections: PersonaConnectionSummary[]
): "ok" | "missing" | null => {
  if (!command.connection_id) return null
  if (command.connection_status === "ok" || command.connection_status === "missing") {
    return command.connection_status
  }
  return availableConnections.some((connection) => connection.id === command.connection_id)
    ? "ok"
    : "missing"
}

const toCommandTargetLabel = (command: PersonaVoiceCommand): string => {
  if (command.action_type === "mcp_tool") {
    return String(command.action_config?.tool_name || "").trim() || "MCP tool"
  }
  if (command.action_type === "workflow") {
    return (
      String(command.action_config?.workflow_name || "").trim() ||
      String(command.action_config?.workflow_id || "").trim() ||
      "Workflow"
    )
  }
  if (command.action_type === "custom") {
    if (command.connection_id) {
      const method = String(command.action_config?.method || "POST").trim().toUpperCase()
      const path = String(command.action_config?.path || "").trim()
      return path ? `${method} ${path}` : `${method} connection base URL`
    }
    return String(command.action_config?.action || "").trim() || "Custom action"
  }
  return "Persona planner fallback"
}

const toFormState = (command: PersonaVoiceCommand): CommandFormState => {
  const rawSlotMap =
    (command.action_config?.slot_to_param_map as Record<string, unknown> | undefined) ||
    (command.action_config?.param_map as Record<string, unknown> | undefined) ||
    {}
  const rawDefaultPayload =
    (command.action_config?.default_payload as Record<string, unknown> | undefined) || {}
  const extractMode =
    command.action_config?.extract_query === true
      ? "query"
      : command.action_config?.extract_content === true
        ? "content"
        : "none"

  return {
    commandId: command.id,
    name: command.name || "",
    description: String(command.description || ""),
    phrasesText: Array.isArray(command.phrases) ? command.phrases.join("\n") : "",
    actionType: command.action_type,
    toolName: String(command.action_config?.tool_name || ""),
    workflowId:
      String(command.action_config?.workflow_id || "") ||
      String(command.action_config?.workflow_name || ""),
    customAction: String(command.action_config?.action || ""),
    connectionId: String(command.connection_id || ""),
    requestMethod: normalizeRequestMethod(command.action_config?.method),
    requestPath: String(command.action_config?.path || ""),
    extractMode,
    slotMapText: stringifyJson(rawSlotMap),
    defaultPayloadText: stringifyJson(rawDefaultPayload),
    priority: String(command.priority ?? 50),
    enabled: command.enabled !== false,
    requiresConfirmation: Boolean(command.requires_confirmation)
  }
}

export const CommandsPanel: React.FC<CommandsPanelProps> = ({
  selectedPersonaId,
  selectedPersonaName,
  isActive = false,
  analytics = null,
  analyticsLoading = false,
  handoffFocusRequest = null,
  onSetupHandoffFocusConsumed,
  openCommandId = null,
  onOpenCommandHandled,
  draftCommandPhrase = null,
  draftCommandSource = null,
  onDraftCommandPhraseHandled,
  rerunAfterSaveCommandId = null,
  onRerunAfterSave,
  onCommandSaved
}) => {
  const { t } = useTranslation(["sidepanel", "common"])
  const [commands, setCommands] = React.useState<PersonaVoiceCommand[]>([])
  const [connections, setConnections] = React.useState<PersonaConnectionSummary[]>([])
  const [loading, setLoading] = React.useState(false)
  const [commandsLoaded, setCommandsLoaded] = React.useState(false)
  const [saving, setSaving] = React.useState(false)
  const [error, setError] = React.useState<string | null>(null)
  const [validationError, setValidationError] = React.useState<string | null>(null)
  const [formState, setFormState] =
    React.useState<CommandFormState>(DEFAULT_FORM_STATE)
  const [draftSourcePhrase, setDraftSourcePhrase] = React.useState<string | null>(null)
  const [draftSourceKind, setDraftSourceKind] = React.useState<CommandDraftSource | null>(
    null
  )
  const commandNameInputRef = React.useRef<HTMLInputElement | null>(null)
  const commandFormRef = React.useRef<HTMLDivElement | null>(null)
  const commandEditButtonRefs = React.useRef<Map<string, HTMLButtonElement | null>>(
    new Map()
  )
  const lastHandledHandoffTokenRef = React.useRef(0)
  const editingCommand = React.useMemo(
    () =>
      formState.commandId
        ? commands.find((command) => command.id === formState.commandId) ?? null
        : null,
    [commands, formState.commandId]
  )
  const selectedConnectionMissing = React.useMemo(() => {
    if (!formState.connectionId.trim()) return false
    return !connections.some((connection) => connection.id === formState.connectionId.trim())
  }, [connections, formState.connectionId])
  const draftAssistSuggestions = React.useMemo(
    () =>
      draftSourcePhrase &&
      (formState.actionType === "mcp_tool" || formState.actionType === "custom")
        ? buildDraftAssistSuggestions(draftSourcePhrase)
        : [],
    [draftSourcePhrase, formState.actionType]
  )
  const analyticsByCommandId = React.useMemo(() => {
    const entries = Array.isArray(analytics?.commands) ? analytics.commands : []
    return new Map<string, PersonaVoiceCommandAnalyticsItem>(
      entries.map((item) => [String(item.command_id || "").trim(), item])
    )
  }, [analytics])

  const [cmdSearchText, setCmdSearchText] = React.useState("")
  const [cmdFilterBroken, setCmdFilterBroken] = React.useState(false)
  const [cmdFilterNeverUsed, setCmdFilterNeverUsed] = React.useState(false)
  const [cmdFilterNeedsConfirmation, setCmdFilterNeedsConfirmation] = React.useState(false)
  const [cmdFilterMostUsed, setCmdFilterMostUsed] = React.useState(false)
  const [cmdSortBy, setCmdSortBy] = React.useState<
    "most_used" | "recent" | "name" | "priority"
  >("name")

  const mostUsedThreshold = React.useMemo(() => {
    const allInvocations = commands
      .map((cmd) => analyticsByCommandId.get(cmd.id)?.total_invocations ?? 0)
      .sort((a, b) => b - a)
    const topQuartileIndex = Math.max(1, Math.ceil(allInvocations.length * 0.25))
    return allInvocations[topQuartileIndex - 1] ?? 0
  }, [commands, analyticsByCommandId])

  const cmdFilterCounts = React.useMemo(() => {
    let broken = 0
    let neverUsed = 0
    let needsConfirmation = 0
    let mostUsed = 0
    for (const cmd of commands) {
      const a = analyticsByCommandId.get(cmd.id)
      if (
        resolveCommandConnectionStatus(cmd, connections) === "missing" ||
        (a && a.total_invocations > 0 && a.error_count / a.total_invocations > 0.5)
      ) {
        broken++
      }
      if (!a || a.total_invocations === 0) neverUsed++
      if (cmd.requires_confirmation) needsConfirmation++
      if (mostUsedThreshold > 0 && (a?.total_invocations ?? 0) >= mostUsedThreshold) {
        mostUsed++
      }
    }
    return { broken, neverUsed, needsConfirmation, mostUsed }
  }, [commands, connections, analyticsByCommandId, mostUsedThreshold])

  const filteredCommands = React.useMemo(() => {
    let result = commands

    if (cmdSearchText.trim()) {
      const needle = cmdSearchText.trim().toLowerCase()
      result = result.filter((cmd) => {
        const haystack = [cmd.name, cmd.description ?? "", ...cmd.phrases]
          .join(" ")
          .toLowerCase()
        return haystack.includes(needle)
      })
    }

    if (cmdFilterBroken) {
      result = result.filter((cmd) => {
        if (resolveCommandConnectionStatus(cmd, connections) === "missing") return true
        const a = analyticsByCommandId.get(cmd.id)
        if (a && a.total_invocations > 0) {
          return a.error_count / a.total_invocations > 0.5
        }
        return false
      })
    }

    if (cmdFilterNeverUsed) {
      result = result.filter((cmd) => {
        const a = analyticsByCommandId.get(cmd.id)
        return !a || a.total_invocations === 0
      })
    }

    if (cmdFilterNeedsConfirmation) {
      result = result.filter((cmd) => cmd.requires_confirmation)
    }

    if (cmdFilterMostUsed) {
      if (mostUsedThreshold > 0) {
        result = result.filter(
          (cmd) => (analyticsByCommandId.get(cmd.id)?.total_invocations ?? 0) >= mostUsedThreshold
        )
      } else {
        result = []
      }
    }

    const sorted = [...result]
    switch (cmdSortBy) {
      case "most_used":
        sorted.sort((a, b) => {
          const aCount = analyticsByCommandId.get(a.id)?.total_invocations ?? 0
          const bCount = analyticsByCommandId.get(b.id)?.total_invocations ?? 0
          return bCount - aCount
        })
        break
      case "recent":
        sorted.sort((a, b) => {
          const aDate = analyticsByCommandId.get(a.id)?.last_used ?? null
          const bDate = analyticsByCommandId.get(b.id)?.last_used ?? null
          if (!aDate && !bDate) return 0
          if (!aDate) return 1
          if (!bDate) return -1
          return bDate.localeCompare(aDate)
        })
        break
      case "name":
        sorted.sort((a, b) => a.name.localeCompare(b.name))
        break
      case "priority":
        sorted.sort((a, b) => b.priority - a.priority)
        break
    }

    return sorted
  }, [
    commands,
    connections,
    analyticsByCommandId,
    cmdSearchText,
    cmdFilterBroken,
    cmdFilterNeverUsed,
    cmdFilterNeedsConfirmation,
    cmdFilterMostUsed,
    mostUsedThreshold,
    cmdSortBy
  ])

  const activeCmdFilterCount =
    (cmdSearchText.trim() ? 1 : 0) +
    (cmdFilterBroken ? 1 : 0) +
    (cmdFilterNeverUsed ? 1 : 0) +
    (cmdFilterNeedsConfirmation ? 1 : 0) +
    (cmdFilterMostUsed ? 1 : 0)

  const clearAllCmdFilters = React.useCallback(() => {
    setCmdSearchText("")
    setCmdFilterBroken(false)
    setCmdFilterNeverUsed(false)
    setCmdFilterNeedsConfirmation(false)
    setCmdFilterMostUsed(false)
  }, [])

  const clearCommandEditor = React.useCallback(() => {
    setFormState(DEFAULT_FORM_STATE)
    setDraftSourcePhrase(null)
    setDraftSourceKind(null)
    setValidationError(null)
  }, [])

  React.useEffect(() => {
    let cancelled = false

    const load = async () => {
      if (!isActive || !selectedPersonaId) {
        setCommands([])
        setConnections([])
        setCommandsLoaded(false)
        setError(null)
        clearCommandEditor()
        return
      }

      clearCommandEditor()
      setLoading(true)
      setCommands([])
      setConnections([])
      setCommandsLoaded(false)
      setError(null)
      try {
        const [commandsResp, connectionsResp] = await Promise.all([
          tldwClient.fetchWithAuth(
            toAllowedPath(
              `/api/v1/persona/profiles/${encodeURIComponent(selectedPersonaId)}/voice-commands`
            ),
            { method: "GET" }
          ),
          tldwClient.fetchWithAuth(
            toAllowedPath(
              `/api/v1/persona/profiles/${encodeURIComponent(selectedPersonaId)}/connections`
            ),
            { method: "GET" }
          )
        ])

        if (!commandsResp.ok) {
          throw new Error(
            commandsResp.error ||
              t("sidepanel:personaGarden.commands.loadError", {
                defaultValue: "Failed to load persona commands."
              })
          )
        }
        if (!connectionsResp.ok) {
          throw new Error(
            connectionsResp.error ||
              t("sidepanel:personaGarden.connections.loadError", {
                defaultValue: "Failed to load persona connections."
              })
          )
        }

        const commandPayload = await commandsResp.json()
        const connectionPayload = await connectionsResp.json()
        const nextCommands = Array.isArray(commandPayload?.commands)
          ? commandPayload.commands
          : []
        const nextConnections = Array.isArray(connectionPayload)
          ? connectionPayload
          : []
        if (!cancelled) {
          setCommands(nextCommands as PersonaVoiceCommand[])
          setConnections(nextConnections as PersonaConnectionSummary[])
          setCommandsLoaded(true)
        }
      } catch (loadError) {
        if (!cancelled) {
          setCommands([])
          setConnections([])
          setCommandsLoaded(false)
          clearCommandEditor()
          setError(
            loadError instanceof Error
              ? loadError.message
              : t("sidepanel:personaGarden.commands.loadError", {
                  defaultValue: "Failed to load persona commands."
                })
          )
        }
      } finally {
        if (!cancelled) {
          setLoading(false)
        }
      }
    }

    void load()
    return () => {
      cancelled = true
    }
  }, [clearCommandEditor, isActive, selectedPersonaId])

  const resetForm = React.useCallback(() => {
    clearCommandEditor()
  }, [clearCommandEditor])

  const updateFormField = React.useCallback(
    (field: keyof CommandFormState, value: string | boolean | null) => {
      setFormState((current) => ({
        ...current,
        [field]: value
      }))
    },
    []
  )

  const handleTemplateApply = React.useCallback((template: CommandTemplate) => {
    setFormState(template.apply())
    setDraftSourcePhrase(null)
    setDraftSourceKind(null)
    setValidationError(null)
  }, [])

  const handleEdit = React.useCallback((command: PersonaVoiceCommand) => {
    setFormState(toFormState(command))
    setDraftSourcePhrase(null)
    setDraftSourceKind(null)
    setValidationError(null)
    setError(null)
  }, [])

  const applyDraftAssistSuggestion = React.useCallback(
    (suggestion: DraftAssistSuggestion) => {
      setFormState((current) => ({
        ...current,
        phrasesText: suggestion.suggestedPhrase,
        slotMapText: stringifyJson(suggestion.suggestedSlotMap)
      }))
      setValidationError(null)
    },
    []
  )

  React.useEffect(() => {
    const normalizedDraftPhrase = String(draftCommandPhrase || "")
      .trim()
      .replace(/\s+/g, " ")
    if (!isActive || !selectedPersonaId || !normalizedDraftPhrase) return
    setFormState(toDraftFormState(normalizedDraftPhrase))
    setDraftSourcePhrase(normalizedDraftPhrase)
    setDraftSourceKind(draftCommandSource || "test_lab")
    setValidationError(null)
    setError(null)
    onDraftCommandPhraseHandled?.(normalizedDraftPhrase)
  }, [
    draftCommandSource,
    draftCommandPhrase,
    isActive,
    onDraftCommandPhraseHandled,
    selectedPersonaId
  ])

  React.useEffect(() => {
    const normalizedRequestedCommandId = String(openCommandId || "").trim()
    if (
      !isActive ||
      !selectedPersonaId ||
      !normalizedRequestedCommandId ||
      !commandsLoaded
    ) {
      return
    }
    const requestedCommand = commands.find(
      (command) => command.id === normalizedRequestedCommandId
    )
    if (requestedCommand) {
      handleEdit(requestedCommand)
    } else {
      setError("Requested voice command could not be found.")
    }
    onOpenCommandHandled?.(normalizedRequestedCommandId)
  }, [
    commands,
    commandsLoaded,
    handleEdit,
    isActive,
    onOpenCommandHandled,
    openCommandId,
    selectedPersonaId
  ])

  React.useEffect(() => {
    if (!isActive || !selectedPersonaId || !handoffFocusRequest) return
    if (lastHandledHandoffTokenRef.current === handoffFocusRequest.token) return

    const focusCommandForm = () => {
      commandFormRef.current?.scrollIntoView?.({ block: "start", behavior: "smooth" })
      commandNameInputRef.current?.focus()
    }

    if (handoffFocusRequest.section === "command_form") {
      focusCommandForm()
      lastHandledHandoffTokenRef.current = handoffFocusRequest.token
      onSetupHandoffFocusConsumed?.(handoffFocusRequest.token)
      return
    }

    if (!commandsLoaded) return

    const firstCommandId = commands[0]?.id
    const firstEditButton = firstCommandId
      ? commandEditButtonRefs.current.get(firstCommandId)
      : null

    if (firstEditButton) {
      firstEditButton.scrollIntoView?.({ block: "start", behavior: "smooth" })
      firstEditButton.focus()
    } else {
      focusCommandForm()
    }

    lastHandledHandoffTokenRef.current = handoffFocusRequest.token
    onSetupHandoffFocusConsumed?.(handoffFocusRequest.token)
  }, [
    commands,
    commandsLoaded,
    handoffFocusRequest,
    isActive,
    onSetupHandoffFocusConsumed,
    selectedPersonaId
  ])

  const handleToggle = React.useCallback(
    async (command: PersonaVoiceCommand) => {
      if (!selectedPersonaId) return
      setError(null)
      try {
        const response = await tldwClient.fetchWithAuth(
          toAllowedPath(
            `/api/v1/persona/profiles/${encodeURIComponent(selectedPersonaId)}/voice-commands/${encodeURIComponent(command.id)}/toggle`
          ),
          {
            method: "POST",
            body: { enabled: !command.enabled }
          }
        )
        if (!response.ok) {
          throw new Error(response.error || "Failed to update command status.")
        }
        const payload = await response.json()
        setCommands((current) =>
          current.map((item) =>
            item.id === command.id ? (payload as PersonaVoiceCommand) : item
          )
        )
      } catch (toggleError) {
        setError(
          toggleError instanceof Error
            ? toggleError.message
            : "Failed to update command status."
        )
      }
    },
    [selectedPersonaId]
  )

  const handleDelete = React.useCallback(
    async (commandId: string) => {
      if (!selectedPersonaId) return
      if (
        typeof window !== "undefined" &&
        !window.confirm("Delete this voice command?")
      ) {
        return
      }
      setError(null)
      try {
        const response = await tldwClient.fetchWithAuth(
          toAllowedPath(
            `/api/v1/persona/profiles/${encodeURIComponent(selectedPersonaId)}/voice-commands/${encodeURIComponent(commandId)}`
          ),
          {
            method: "DELETE"
          }
        )
        if (!response.ok) {
          throw new Error(response.error || "Failed to delete command.")
        }
        setCommands((current) => current.filter((item) => item.id !== commandId))
        setFormState((current) =>
          current.commandId === commandId ? DEFAULT_FORM_STATE : current
        )
      } catch (deleteError) {
        setError(
          deleteError instanceof Error
            ? deleteError.message
            : "Failed to delete command."
        )
      }
    },
    [selectedPersonaId]
  )

  const handleSave = React.useCallback(async () => {
    if (!selectedPersonaId) return

    const name = formState.name.trim()
    const phrases = splitPhrases(formState.phrasesText)
    const trimmedConnectionId = formState.connectionId.trim()
    if (!name) {
      setValidationError("Command name is required.")
      return
    }
    if (phrases.length === 0) {
      setValidationError("Add at least one trigger phrase.")
      return
    }
    if (
      trimmedConnectionId &&
      !connections.some((connection) => connection.id === trimmedConnectionId)
    ) {
      setValidationError(
        "Selected connection no longer exists. Choose another connection or clear it."
      )
      return
    }

    const slotMapResult = parseJsonRecord(formState.slotMapText, "Slot mapping")
    if (slotMapResult.ok === false) {
      setValidationError(slotMapResult.error)
      return
    }
    const defaultPayloadResult = parseJsonRecord(
      formState.defaultPayloadText,
      "Default payload"
    )
    if (defaultPayloadResult.ok === false) {
      setValidationError(defaultPayloadResult.error)
      return
    }

    const actionConfig: Record<string, unknown> = {}
    if (formState.actionType === "mcp_tool") {
      const toolName = formState.toolName.trim()
      if (!toolName) {
        setValidationError("Tool name is required for MCP tool commands.")
        return
      }
      actionConfig.tool_name = toolName
      if (formState.extractMode === "query") actionConfig.extract_query = true
      if (formState.extractMode === "content") actionConfig.extract_content = true
    } else if (formState.actionType === "workflow") {
      const workflowId = formState.workflowId.trim()
      if (!workflowId) {
        setValidationError("Workflow id is required for workflow commands.")
        return
      }
      actionConfig.workflow_id = workflowId
    } else if (formState.actionType === "custom") {
      const customAction = formState.customAction.trim()
      if (!customAction && !formState.connectionId.trim()) {
        setValidationError("Action name is required for custom commands.")
        return
      }
      if (customAction) {
        actionConfig.action = customAction
      }
      if (formState.connectionId.trim()) {
        actionConfig.method = formState.requestMethod
        if (formState.requestPath.trim()) {
          actionConfig.path = formState.requestPath.trim()
        }
      }
    }

    if (Object.keys(slotMapResult.value).length > 0) {
      actionConfig.slot_to_param_map = slotMapResult.value
    }
    if (Object.keys(defaultPayloadResult.value).length > 0) {
      actionConfig.default_payload = defaultPayloadResult.value
    }

    const payload = {
      connection_id: trimmedConnectionId || null,
      name,
      description: formState.description.trim() || null,
      phrases,
      action_type: formState.actionType,
      action_config: actionConfig,
      priority: Number.parseInt(formState.priority, 10) || 0,
      enabled: formState.enabled,
      requires_confirmation: formState.requiresConfirmation
    }

    setSaving(true)
    setValidationError(null)
    setError(null)
    try {
      const isEditing = Boolean(formState.commandId)
      const savedFromDraft = Boolean(draftSourcePhrase) && !isEditing
      const response = await tldwClient.fetchWithAuth(
        isEditing
          ? toAllowedPath(
              `/api/v1/persona/profiles/${encodeURIComponent(selectedPersonaId)}/voice-commands/${encodeURIComponent(formState.commandId || "")}`
            )
          : toAllowedPath(
              `/api/v1/persona/profiles/${encodeURIComponent(selectedPersonaId)}/voice-commands`
            ),
        {
          method: isEditing ? "PUT" : "POST",
          body: payload
        }
      )
      if (!response.ok) {
        throw new Error(
          response.error ||
            (isEditing
              ? "Failed to update voice command."
              : "Failed to create voice command.")
        )
      }
      const saved = (await response.json()) as PersonaVoiceCommand
      setCommands((current) => {
        const existingIndex = current.findIndex((item) => item.id === saved.id)
        if (existingIndex === -1) {
          return [saved, ...current]
        }
        const next = [...current]
        next[existingIndex] = saved
        return next
      })
      resetForm()
      onCommandSaved?.(saved.id, { fromDraft: savedFromDraft })
      if (
        String(rerunAfterSaveCommandId || "").trim() &&
        saved.id === String(rerunAfterSaveCommandId || "").trim()
      ) {
        onRerunAfterSave?.(saved.id)
      }
    } catch (saveError) {
      setError(
        saveError instanceof Error
          ? saveError.message
          : "Failed to save voice command."
      )
    } finally {
      setSaving(false)
    }
  }, [
    connections,
    draftSourcePhrase,
    formState,
    onCommandSaved,
    onRerunAfterSave,
    resetForm,
    rerunAfterSaveCommandId,
    selectedPersonaId
  ])

  return (
    <div className="rounded-lg border border-border bg-surface p-3">
      <div className="text-[11px] font-semibold uppercase tracking-wide text-text-subtle">
        {t("sidepanel:personaGarden.commands.heading", {
          defaultValue: "Commands"
        })}
      </div>
      <div className="mt-2 space-y-3 text-sm text-text">
        <p className="text-xs text-text-muted">
          {selectedPersonaId
            ? t("sidepanel:personaGarden.commands.description", {
                defaultValue:
                  "Register direct voice commands for {{personaName}} with phrases, tool targets, and optional slot mappings.",
                personaName:
                  selectedPersonaName ||
                  selectedPersonaId ||
                  t("sidepanel:personaGarden.commands.currentPersona", {
                    defaultValue: "this persona"
                  })
              })
            : t("sidepanel:personaGarden.commands.noPersona", {
                defaultValue:
                  "Select a persona to manage its voice command registry."
              })}
        </p>
        <CommandAnalyticsSummary
          analytics={analytics}
          loading={Boolean(selectedPersonaId) && analyticsLoading}
        />

        {error ? (
          <div className="rounded-md border border-red-500/40 bg-red-500/10 px-3 py-2 text-xs text-red-700">
            {error}
          </div>
        ) : null}
        {validationError ? (
          <div className="rounded-md border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-xs text-amber-700">
            {validationError}
          </div>
        ) : null}

        {selectedPersonaId ? (
          <>
            <div className="space-y-2">
              <div className="text-xs font-medium text-text">
                {t("sidepanel:personaGarden.commands.templates", {
                  defaultValue: "Quick templates"
                })}
              </div>
              <div className="grid gap-2 md:grid-cols-3">
                {COMMAND_TEMPLATES.map((template) => (
                  <button
                    key={template.key}
                    type="button"
                    data-testid={`persona-commands-template-${template.key}`}
                    className="rounded-md border border-border bg-bg px-3 py-2 text-left transition hover:border-primary/40 hover:bg-surface2"
                    onClick={() => handleTemplateApply(template)}
                  >
                    <div className="text-sm font-medium text-text">
                      {template.label}
                    </div>
                    <div className="text-xs text-text-muted">
                      {template.description}
                    </div>
                  </button>
                ))}
              </div>
            </div>

            <div className="grid gap-3 xl:grid-cols-[minmax(0,1.3fr)_minmax(0,1fr)]">
              <div className="space-y-2">
                <div className="flex items-center justify-between gap-2">
                  <div className="text-xs font-medium text-text">
                    {t("sidepanel:personaGarden.commands.existing", {
                      defaultValue: "Registered commands"
                    })}
                    {commands.length > 0 &&
                    filteredCommands.length !== commands.length ? (
                      <span className="ml-1 font-normal text-text-muted">
                        ({filteredCommands.length}/{commands.length})
                      </span>
                    ) : null}
                  </div>
                  {loading ? (
                    <span className="text-xs text-text-muted">
                      {t("common:loading", "Loading...")}
                    </span>
                  ) : null}
                </div>

                {commands.length > 0 ? (
                  <div className="space-y-2">
                    <div className="flex flex-wrap items-center gap-2">
                      <Input
                        allowClear
                        size="small"
                        value={cmdSearchText}
                        onChange={(e) => setCmdSearchText(e.target.value)}
                        placeholder={t(
                          "sidepanel:personaGarden.commands.searchPlaceholder",
                          { defaultValue: "Search commands..." }
                        )}
                        className="w-full sm:w-52"
                        data-testid="persona-commands-search"
                      />
                      <Select
                        size="small"
                        value={cmdSortBy}
                        onChange={(value) => setCmdSortBy(value)}
                        style={{ width: 140 }}
                        data-testid="persona-commands-sort"
                        options={[
                          {
                            label: t(
                              "sidepanel:personaGarden.commands.sort.mostUsed",
                              { defaultValue: "Most used" }
                            ),
                            value: "most_used" as const
                          },
                          {
                            label: t(
                              "sidepanel:personaGarden.commands.sort.recent",
                              { defaultValue: "Recent" }
                            ),
                            value: "recent" as const
                          },
                          {
                            label: t(
                              "sidepanel:personaGarden.commands.sort.name",
                              { defaultValue: "Name" }
                            ),
                            value: "name" as const
                          },
                          {
                            label: t(
                              "sidepanel:personaGarden.commands.sort.priority",
                              { defaultValue: "Priority" }
                            ),
                            value: "priority" as const
                          }
                        ]}
                      />
                    </div>
                    <div className="flex flex-wrap items-center gap-1.5">
                      <Tag.CheckableTag
                        checked={cmdFilterBroken}
                        onChange={(checked) => {
                          setCmdFilterBroken(checked)
                        }}
                        data-testid="persona-commands-filter-broken"
                        aria-label="Filter broken commands"
                      >
                        {t(
                          "sidepanel:personaGarden.commands.filter.broken",
                          { defaultValue: "Broken" }
                        )}
                        {cmdFilterCounts.broken > 0
                          ? ` (${cmdFilterCounts.broken})`
                          : ""}
                      </Tag.CheckableTag>
                      <Tag.CheckableTag
                        checked={cmdFilterNeverUsed}
                        onChange={(checked) => {
                          setCmdFilterNeverUsed(checked)
                          if (checked) setCmdFilterMostUsed(false)
                        }}
                        data-testid="persona-commands-filter-never-used"
                        aria-label="Filter never used commands"
                      >
                        {t(
                          "sidepanel:personaGarden.commands.filter.neverUsed",
                          { defaultValue: "Never used" }
                        )}
                        {cmdFilterCounts.neverUsed > 0
                          ? ` (${cmdFilterCounts.neverUsed})`
                          : ""}
                      </Tag.CheckableTag>
                      <Tag.CheckableTag
                        checked={cmdFilterNeedsConfirmation}
                        onChange={(checked) => {
                          setCmdFilterNeedsConfirmation(checked)
                        }}
                        data-testid="persona-commands-filter-needs-confirmation"
                        aria-label="Filter commands needing confirmation"
                      >
                        {t(
                          "sidepanel:personaGarden.commands.filter.needsConfirmation",
                          { defaultValue: "Needs confirmation" }
                        )}
                        {cmdFilterCounts.needsConfirmation > 0
                          ? ` (${cmdFilterCounts.needsConfirmation})`
                          : ""}
                      </Tag.CheckableTag>
                      <Tag.CheckableTag
                        checked={cmdFilterMostUsed}
                        onChange={(checked) => {
                          setCmdFilterMostUsed(checked)
                          if (checked) setCmdFilterNeverUsed(false)
                        }}
                        data-testid="persona-commands-filter-most-used"
                        aria-label="Filter most used commands"
                      >
                        {t(
                          "sidepanel:personaGarden.commands.filter.mostUsed",
                          { defaultValue: "Most used" }
                        )}
                        {cmdFilterCounts.mostUsed > 0
                          ? ` (${cmdFilterCounts.mostUsed})`
                          : ""}
                      </Tag.CheckableTag>
                      {activeCmdFilterCount > 0 ? (
                        <button
                          type="button"
                          className="ml-1 text-xs text-text-muted underline hover:text-text"
                          onClick={clearAllCmdFilters}
                          data-testid="persona-commands-clear-filters"
                        >
                          {t(
                            "sidepanel:personaGarden.commands.filter.clearAll",
                            { defaultValue: "Clear filters" }
                          )}
                        </button>
                      ) : null}
                    </div>
                  </div>
                ) : null}
                {filteredCommands.length > 0 ? (
                  filteredCommands.map((command) => {
                    const commandAnalytics = analyticsByCommandId.get(command.id)
                    const lastUsedLabel = formatLastUsedLabel(commandAnalytics?.last_used)

                    return (
                      <div
                        key={command.id}
                        data-testid={`persona-commands-row-${command.id}`}
                        data-selected={formState.commandId === command.id ? "true" : "false"}
                        className={`rounded-md border bg-bg p-3 transition ${
                          formState.commandId === command.id
                            ? "border-primary bg-primary/5 shadow-sm ring-1 ring-primary/20"
                            : "border-border"
                        }`}
                      >
                        <div className="flex flex-wrap items-start justify-between gap-2">
                          <div>
                            <div className="font-medium text-text">
                              {command.name}
                            </div>
                            <div className="text-xs text-text-muted">
                              {toCommandTargetLabel(command)}
                            </div>
                          </div>
                          <div className="flex flex-wrap gap-2 text-[11px]">
                            <span className="rounded-full border border-border px-2 py-0.5 text-text-muted">
                              {command.action_type}
                            </span>
                            {command.enabled ? (
                              <span className="rounded-full border border-emerald-500/40 bg-emerald-500/10 px-2 py-0.5 text-emerald-700">
                                enabled
                              </span>
                            ) : (
                              <span className="rounded-full border border-amber-500/40 bg-amber-500/10 px-2 py-0.5 text-amber-700">
                                disabled
                              </span>
                            )}
                            {command.requires_confirmation ? (
                              <span className="rounded-full border border-border px-2 py-0.5 text-text-muted">
                                confirm
                              </span>
                            ) : null}
                            {command.connection_id ? (
                              resolveCommandConnectionStatus(command, connections) === "missing" ? (
                                <span className="rounded-full border border-red-500/40 bg-red-500/10 px-2 py-0.5 text-red-700">
                                  missing connection
                                </span>
                              ) : (
                                <span className="rounded-full border border-sky-500/40 bg-sky-500/10 px-2 py-0.5 text-sky-700">
                                  connection
                                </span>
                              )
                            ) : null}
                          </div>
                        </div>
                        {command.description ? (
                          <p className="mt-2 text-xs text-text-muted">
                            {command.description}
                          </p>
                        ) : null}
                        {resolveCommandConnectionStatus(command, connections) === "missing" ? (
                          <div className="mt-2 rounded-md border border-red-500/30 bg-red-500/10 px-3 py-2 text-xs text-red-700">
                            {t("sidepanel:personaGarden.commands.missingConnectionHint", {
                              defaultValue:
                                "The saved connection for this command was deleted. Edit the command to choose a replacement connection."
                            })}
                          </div>
                        ) : null}
                        {commandAnalytics ? (
                          <div
                            data-testid={`persona-commands-analytics-${command.id}`}
                            className="mt-2 flex flex-wrap gap-2 text-[11px]"
                          >
                            <span className="rounded-full border border-border px-2 py-0.5 text-text-muted">
                              {formatRunLabel(commandAnalytics.total_invocations)}
                            </span>
                            {commandAnalytics.error_count > 0 ? (
                              <span className="rounded-full border border-amber-500/40 bg-amber-500/10 px-2 py-0.5 text-amber-800">
                                {formatFailureLabel(commandAnalytics.error_count)}
                              </span>
                            ) : (
                              <span className="rounded-full border border-emerald-500/40 bg-emerald-500/10 px-2 py-0.5 text-emerald-700">
                                healthy
                              </span>
                            )}
                            {lastUsedLabel ? (
                              <span className="rounded-full border border-border px-2 py-0.5 text-text-muted">
                                {lastUsedLabel}
                              </span>
                            ) : null}
                          </div>
                        ) : null}
                        <div className="mt-2 flex flex-wrap gap-2 text-[11px] text-text-muted">
                          {command.phrases.map((phrase) => (
                            <span
                              key={`${command.id}-${phrase}`}
                              className="rounded-full border border-border px-2 py-0.5"
                            >
                              {phrase}
                            </span>
                          ))}
                        </div>
                        <div className="mt-3 flex flex-wrap gap-2">
                          <button
                            type="button"
                            data-testid={`persona-commands-edit-${command.id}`}
                            ref={(node) => {
                              commandEditButtonRefs.current.set(command.id, node)
                            }}
                            className="rounded-md border border-border px-2 py-1 text-xs text-text transition hover:bg-surface2"
                            onClick={() => handleEdit(command)}
                          >
                            {t("common:edit", "Edit")}
                          </button>
                          <button
                            type="button"
                            data-testid={`persona-commands-toggle-${command.id}`}
                            className="rounded-md border border-border px-2 py-1 text-xs text-text transition hover:bg-surface2"
                            onClick={() => {
                              void handleToggle(command)
                            }}
                          >
                            {command.enabled
                              ? t("sidepanel:personaGarden.commands.disable", {
                                  defaultValue: "Disable"
                                })
                              : t("sidepanel:personaGarden.commands.enable", {
                                  defaultValue: "Enable"
                                })}
                          </button>
                          <button
                            type="button"
                            data-testid={`persona-commands-delete-${command.id}`}
                            className="rounded-md border border-red-500/40 px-2 py-1 text-xs text-red-700 transition hover:bg-red-500/10"
                            onClick={() => {
                              void handleDelete(command.id)
                            }}
                          >
                            {t("common:delete", "Delete")}
                          </button>
                        </div>
                      </div>
                    )
                  })
                ) : (
                  <div
                    data-testid="persona-commands-empty"
                    className="rounded-md border border-dashed border-border px-3 py-4 text-xs text-text-muted"
                  >
                    {loading
                      ? t("sidepanel:personaGarden.commands.loading", {
                          defaultValue: "Loading commands..."
                        })
                      : commands.length > 0
                        ? t("sidepanel:personaGarden.commands.noMatches", {
                            defaultValue:
                              "No commands match the current filters."
                          })
                        : t("sidepanel:personaGarden.commands.empty", {
                            defaultValue:
                              "No direct voice commands yet. Start from a template or create one manually."
                          })}
                  </div>
                )}
              </div>

              <div
                ref={commandFormRef}
                className="rounded-md border border-border bg-bg p-3"
              >
                <div className="flex items-center justify-between gap-2">
                  <div className="text-sm font-medium text-text">
                    {formState.commandId
                      ? t("sidepanel:personaGarden.commands.editHeading", {
                          defaultValue: "Edit command"
                        })
                      : t("sidepanel:personaGarden.commands.createHeading", {
                          defaultValue: "Create command"
                        })}
                  </div>
                  {formState.commandId ? (
                    <button
                      type="button"
                      data-testid="persona-commands-reset"
                      className="rounded-md border border-border px-2 py-1 text-xs text-text transition hover:bg-surface2"
                      onClick={resetForm}
                    >
                      {t("common:reset", "Reset")}
                    </button>
                  ) : null}
                </div>

                <div className="mt-3 space-y-3">
                  {draftSourcePhrase && !formState.commandId ? (
                    <div className="space-y-2">
                      <div
                        data-testid="persona-commands-draft-banner"
                        className="rounded-md border border-sky-500/30 bg-sky-500/10 px-3 py-2 text-xs text-sky-900"
                      >
                        {draftSourceKind === "setup_no_match"
                          ? t("sidepanel:personaGarden.commands.draftFromSetup", {
                              defaultValue:
                                "Drafted from assistant setup. Save this command, then return to finish setup."
                            })
                          : t("sidepanel:personaGarden.commands.draftFromTestLab", {
                              defaultValue:
                                "Drafted from Test Lab. Adjust the phrase, add placeholders like {topic} if needed, then choose a target."
                            })}
                      </div>
                      {draftAssistSuggestions.length > 0 ? (
                        <div className="rounded-md border border-border bg-surface px-3 py-2 text-xs text-text-muted">
                          <div className="font-medium text-text">
                            {t("sidepanel:personaGarden.commands.draftAssistHeading", {
                              defaultValue: "Suggested placeholders"
                            })}
                          </div>
                          <div className="mt-2 flex flex-wrap gap-2">
                            {draftAssistSuggestions.map((suggestion) => (
                              <button
                                key={`${suggestion.id}-${suggestion.suggestedPhrase}`}
                                type="button"
                                data-testid={`persona-commands-draft-assist-chip-${suggestion.id}`}
                                className="rounded-full border border-sky-500/40 bg-sky-500/10 px-2 py-1 text-xs font-medium text-sky-800 transition hover:bg-sky-500/20"
                                onClick={() =>
                                  applyDraftAssistSuggestion(suggestion)
                                }
                              >
                                {suggestion.label}
                              </button>
                            ))}
                          </div>
                        </div>
                      ) : null}
                    </div>
                  ) : null}

                  <label className="block text-xs text-text-muted">
                    {t("sidepanel:personaGarden.commands.name", {
                      defaultValue: "Command name"
                    })}
                    <input
                      ref={commandNameInputRef}
                      data-testid="persona-commands-name-input"
                      className="mt-1 w-full rounded-md border border-border bg-surface px-2 py-1 text-sm text-text"
                      value={formState.name}
                      onChange={(event) =>
                        updateFormField("name", event.target.value)
                      }
                      placeholder="Search notes"
                    />
                  </label>

                  <label className="block text-xs text-text-muted">
                    {t("sidepanel:personaGarden.commands.descriptionLabel", {
                      defaultValue: "Description"
                    })}
                    <input
                      data-testid="persona-commands-description-input"
                      className="mt-1 w-full rounded-md border border-border bg-surface px-2 py-1 text-sm text-text"
                      value={formState.description}
                      onChange={(event) =>
                        updateFormField("description", event.target.value)
                      }
                      placeholder="What this command should do"
                    />
                  </label>

                  <label className="block text-xs text-text-muted">
                    {t("sidepanel:personaGarden.commands.phrases", {
                      defaultValue: "Trigger phrases"
                    })}
                    <textarea
                      data-testid="persona-commands-phrases-input"
                      className="mt-1 min-h-[88px] w-full rounded-md border border-border bg-surface px-2 py-1 text-sm text-text"
                      value={formState.phrasesText}
                      onChange={(event) =>
                        updateFormField("phrasesText", event.target.value)
                      }
                      placeholder={"search notes for {topic}\nfind notes about {topic}"}
                    />
                  </label>

                  <div className="grid gap-3 md:grid-cols-2">
                    <label className="block text-xs text-text-muted">
                      {t("sidepanel:personaGarden.commands.actionType", {
                        defaultValue: "Action type"
                      })}
                      <select
                        data-testid="persona-commands-action-type-select"
                        className="mt-1 w-full rounded-md border border-border bg-surface px-2 py-1 text-sm text-text"
                        value={formState.actionType}
                        onChange={(event) =>
                          updateFormField(
                            "actionType",
                            event.target.value as VoiceCommandActionType
                          )
                        }
                      >
                        <option value="mcp_tool">mcp_tool</option>
                        <option value="workflow">workflow</option>
                        <option value="custom">custom</option>
                        <option value="llm_chat">llm_chat</option>
                      </select>
                    </label>

                    <label className="block text-xs text-text-muted">
                      {t("sidepanel:personaGarden.commands.connection", {
                        defaultValue: "Connection"
                      })}
                      <select
                        data-testid="persona-commands-connection-select"
                        className="mt-1 w-full rounded-md border border-border bg-surface px-2 py-1 text-sm text-text"
                        value={formState.connectionId}
                        onChange={(event) =>
                          updateFormField("connectionId", event.target.value)
                        }
                      >
                        <option value="">
                          {t("sidepanel:personaGarden.commands.noConnection", {
                            defaultValue: "No connection"
                          })}
                        </option>
                        {selectedConnectionMissing ? (
                          <option value={formState.connectionId}>
                            {t("sidepanel:personaGarden.commands.missingConnectionOption", {
                              defaultValue: "Missing connection ({{connectionId}})",
                              connectionId: formState.connectionId
                            })}
                          </option>
                        ) : null}
                        {connections.map((connection) => (
                          <option key={connection.id} value={connection.id}>
                            {connection.name}
                          </option>
                        ))}
                      </select>
                    </label>
                  </div>

                  {formState.actionType === "mcp_tool" ? (
                    <>
                      <McpToolPicker
                        value={formState.toolName}
                        onChange={(nextValue) =>
                          updateFormField("toolName", nextValue)
                        }
                        disabled={saving}
                      />
                      <label className="block text-xs text-text-muted">
                        {t("sidepanel:personaGarden.commands.extractMode", {
                          defaultValue: "Phrase extraction"
                        })}
                        <select
                          data-testid="persona-commands-extract-mode-select"
                          className="mt-1 w-full rounded-md border border-border bg-surface px-2 py-1 text-sm text-text"
                          value={formState.extractMode}
                          onChange={(event) =>
                            updateFormField(
                              "extractMode",
                              event.target.value as CommandFormState["extractMode"]
                            )
                          }
                        >
                          <option value="none">none</option>
                          <option value="query">extract_query</option>
                          <option value="content">extract_content</option>
                        </select>
                      </label>
                    </>
                  ) : null}

                  {formState.actionType === "workflow" ? (
                    <label className="block text-xs text-text-muted">
                      {t("sidepanel:personaGarden.commands.workflowId", {
                        defaultValue: "Workflow id"
                      })}
                      <input
                        data-testid="persona-commands-target-input"
                        className="mt-1 w-full rounded-md border border-border bg-surface px-2 py-1 text-sm text-text"
                        value={formState.workflowId}
                        onChange={(event) =>
                          updateFormField("workflowId", event.target.value)
                        }
                        placeholder="daily-research-digest"
                      />
                    </label>
                  ) : null}

                  {formState.actionType === "custom" ? (
                    <div className="space-y-3">
                      {selectedConnectionMissing ? (
                        <div className="rounded-md border border-red-500/30 bg-red-500/10 px-3 py-2 text-xs text-red-700">
                          {t("sidepanel:personaGarden.commands.missingConnectionWarning", {
                            defaultValue:
                              "Selected connection no longer exists. Choose another connection or clear it."
                          })}
                        </div>
                      ) : null}
                      <label className="block text-xs text-text-muted">
                        {t("sidepanel:personaGarden.commands.customAction", {
                          defaultValue: "Custom action"
                        })}
                        <input
                          data-testid="persona-commands-custom-action-input"
                          className="mt-1 w-full rounded-md border border-border bg-surface px-2 py-1 text-sm text-text"
                          value={formState.customAction}
                          onChange={(event) =>
                            updateFormField("customAction", event.target.value)
                          }
                          placeholder={
                            formState.connectionId
                              ? "external_request"
                              : "help"
                          }
                        />
                      </label>

                      {formState.connectionId ? (
                        <div className="space-y-3 rounded-md border border-sky-500/30 bg-sky-500/10 px-3 py-3">
                          <div className="text-xs text-sky-900">
                            {t("sidepanel:personaGarden.commands.externalRequestHint", {
                              defaultValue:
                                "This command will call the selected connection directly. Leave request path blank to call the connection base URL."
                            })}
                          </div>
                          <div className="grid gap-3 md:grid-cols-[minmax(0,140px)_minmax(0,1fr)]">
                            <label className="block text-xs text-text-muted">
                              {t("sidepanel:personaGarden.commands.httpMethod", {
                                defaultValue: "HTTP method"
                              })}
                              <select
                                data-testid="persona-commands-http-method-select"
                                className="mt-1 w-full rounded-md border border-border bg-surface px-2 py-1 text-sm text-text"
                                value={formState.requestMethod}
                                onChange={(event) =>
                                  updateFormField(
                                    "requestMethod",
                                    event.target.value as CommandFormState["requestMethod"]
                                  )
                                }
                              >
                                {REQUEST_METHODS.map((method) => (
                                  <option key={method} value={method}>
                                    {method}
                                  </option>
                                ))}
                              </select>
                            </label>

                            <label className="block text-xs text-text-muted">
                              {t("sidepanel:personaGarden.commands.requestPath", {
                                defaultValue: "Request path"
                              })}
                              <input
                                data-testid="persona-commands-request-path-input"
                                className="mt-1 w-full rounded-md border border-border bg-surface px-2 py-1 text-sm text-text"
                                value={formState.requestPath}
                                onChange={(event) =>
                                  updateFormField("requestPath", event.target.value)
                                }
                                placeholder="alerts/search"
                              />
                            </label>
                          </div>
                        </div>
                      ) : null}
                    </div>
                  ) : null}

                  {formState.actionType === "llm_chat" ? (
                    <div className="rounded-md border border-border bg-surface px-3 py-2 text-xs text-text-muted">
                      {t("sidepanel:personaGarden.commands.llmChatNote", {
                        defaultValue:
                          "LLM chat commands hand the utterance to the persona planner rather than a direct tool target."
                      })}
                    </div>
                  ) : null}

                  <details className="rounded-md border border-border bg-surface px-3 py-2">
                    <summary className="cursor-pointer text-xs font-medium text-text">
                      {t("sidepanel:personaGarden.commands.advanced", {
                        defaultValue: "Advanced mappings"
                      })}
                    </summary>
                    <div className="mt-3 space-y-3">
                      <label className="block text-xs text-text-muted">
                        {t("sidepanel:personaGarden.commands.slotMap", {
                          defaultValue: "Slot to param map (JSON)"
                        })}
                        <textarea
                          data-testid="persona-commands-slot-map-input"
                          className="mt-1 min-h-[96px] w-full rounded-md border border-border bg-bg px-2 py-1 text-sm text-text"
                          value={formState.slotMapText}
                          onChange={(event) =>
                            updateFormField("slotMapText", event.target.value)
                          }
                        />
                      </label>
                      <label className="block text-xs text-text-muted">
                        {t("sidepanel:personaGarden.commands.defaultPayload", {
                          defaultValue: "Default payload (JSON)"
                        })}
                        <textarea
                          data-testid="persona-commands-default-payload-input"
                          className="mt-1 min-h-[96px] w-full rounded-md border border-border bg-bg px-2 py-1 text-sm text-text"
                          value={formState.defaultPayloadText}
                          onChange={(event) =>
                            updateFormField(
                              "defaultPayloadText",
                              event.target.value
                            )
                          }
                        />
                      </label>
                    </div>
                  </details>

                  <div className="grid gap-3 md:grid-cols-2">
                    <label className="block text-xs text-text-muted">
                      {t("sidepanel:personaGarden.commands.priority", {
                        defaultValue: "Priority"
                      })}
                      <input
                        data-testid="persona-commands-priority-input"
                        className="mt-1 w-full rounded-md border border-border bg-surface px-2 py-1 text-sm text-text"
                        value={formState.priority}
                        onChange={(event) =>
                          updateFormField("priority", event.target.value)
                        }
                        inputMode="numeric"
                        placeholder="50"
                      />
                    </label>
                    <div className="grid gap-2 text-xs text-text-muted">
                      <label className="flex items-center gap-2">
                        <input
                          data-testid="persona-commands-enabled-toggle"
                          type="checkbox"
                          checked={formState.enabled}
                          onChange={(event) =>
                            updateFormField("enabled", event.target.checked)
                          }
                        />
                        {t("sidepanel:personaGarden.commands.enabled", {
                          defaultValue: "Enabled"
                        })}
                      </label>
                      <label className="flex items-center gap-2">
                        <input
                          data-testid="persona-commands-confirmation-toggle"
                          type="checkbox"
                          checked={formState.requiresConfirmation}
                          onChange={(event) =>
                            updateFormField(
                              "requiresConfirmation",
                              event.target.checked
                            )
                          }
                        />
                        {t("sidepanel:personaGarden.commands.requireConfirmation", {
                          defaultValue: "Require confirmation"
                        })}
                      </label>
                    </div>
                  </div>

                  {formState.connectionId ? (
                    formState.actionType === "custom" ? null : (
                      <div className="rounded-md border border-sky-500/30 bg-sky-500/10 px-3 py-2 text-xs text-sky-800">
                        {t("sidepanel:personaGarden.commands.connectionHint", {
                          defaultValue:
                            "Connection-backed live execution is configured through custom commands. Switch to custom to define the HTTP request for this connection."
                        })}
                      </div>
                    )
                  ) : null}

                  <div className="flex flex-wrap gap-2">
                    <button
                      type="button"
                      data-testid="persona-commands-save"
                      className="rounded-md bg-primary px-3 py-2 text-sm font-medium text-primary-foreground transition hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-60"
                      disabled={saving}
                      onClick={() => {
                        void handleSave()
                      }}
                    >
                      {saving
                        ? t("common:saving", "Saving...")
                        : formState.commandId
                          ? t("common:update", "Update")
                          : t("common:create", "Create")}
                    </button>
                    <button
                      type="button"
                      className="rounded-md border border-border px-3 py-2 text-sm text-text transition hover:bg-surface2"
                      onClick={resetForm}
                    >
                      {t("common:clear", "Clear")}
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </>
        ) : null}
      </div>
    </div>
  )
}
