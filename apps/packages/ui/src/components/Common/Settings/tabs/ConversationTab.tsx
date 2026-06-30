import { FileIcon, X } from "lucide-react"
import React from "react"
import { Button, Form, Input, InputNumber, Select } from "antd"
import { useQuery, useQueryClient } from "@tanstack/react-query"
import { useTranslation } from "react-i18next"
import { tldwClient, type ConversationState } from "@/services/tldw/TldwApiClient"
import {
  CONVERSATION_STATE_OPTIONS,
  normalizeConversationState
} from "@/utils/conversation-state"
import { useAntdNotification } from "@/hooks/useAntdNotification"
import { useChatSettingsRecord } from "@/hooks/chat/useChatSettingsRecord"
import { PromptAssemblyPreview } from "@/components/Common/Settings/PromptAssemblyPreview"
import { LorebookDebugPanel } from "@/components/Common/Settings/LorebookDebugPanel"
import {
  CHARACTER_PROMPT_PRESETS,
  DEFAULT_CHARACTER_PROMPT_PRESET,
  isCharacterPromptPresetId,
  type CharacterPromptPresetId
} from "@/data/character-prompt-presets"

interface UploadedFile {
  id: string
  filename: string
  size: number
  processed?: boolean
}

interface ConversationTabProps {
  useDrawer?: boolean
  historyId: string | null
  selectedSystemPrompt: string | null
  onSystemPromptChange: (value: string) => void
  onResetSystemPrompt?: () => void | Promise<void>
  uploadedFiles: UploadedFile[]
  onRemoveFile: (id: string) => void
  serverChatId: string | null
  serverChatAssistantKind?: "character" | "persona" | null
  serverChatPersonaMemoryMode?: "read_only" | "read_write" | null
  serverChatState: ConversationState | null
  onStateChange: (state: ConversationState) => void
  serverChatTopic: string | null
  onTopicChange: (topic: string | null) => void
  onVersionChange: (version: number | null) => void
  onPersonaMemoryModeChange?: (mode: "read_only" | "read_write") => void
}

interface UpdateChatResponse {
  version?: number | null
}

type CharacterLite = {
  id: string | number
  name?: string | null
}

export const CONVERSATION_TAB_QUERY_KEYS = {
  listCharacters: ["tldw:listCharacters", "conversation-tab"] as const,
  pinnedMessages: (serverChatId: string | null) =>
    ["conversation-tab:pinned-messages", serverChatId] as const
}

function isErrorWithMessage(error: unknown): error is { message: string } {
  return (
    typeof error === "object" &&
    error !== null &&
    "message" in error &&
    typeof (error as { message?: unknown }).message === "string"
  )
}

function isUpdateChatResponse(value: unknown): value is UpdateChatResponse {
  if (!value || typeof value !== "object") return false
  if (!("version" in value)) return true
  const version = (value as { version?: unknown }).version
  return version === null || version === undefined || typeof version === "number"
}

function getUpdateChatVersion(value: unknown): number | null {
  if (!isUpdateChatResponse(value)) return null
  return value.version ?? null
}

const sanitizeDepth = (value: unknown): number => {
  if (typeof value === "number" && Number.isFinite(value)) {
    return Math.max(0, Math.floor(value))
  }
  if (typeof value === "string") {
    const parsed = Number.parseInt(value, 10)
    if (Number.isFinite(parsed)) {
      return Math.max(0, Math.floor(parsed))
    }
  }
  return 1
}

const sanitizeAutoSummaryThreshold = (value: unknown): number => {
  if (typeof value === "number" && Number.isFinite(value)) {
    return Math.max(2, Math.min(5000, Math.floor(value)))
  }
  if (typeof value === "string") {
    const parsed = Number.parseInt(value, 10)
    if (Number.isFinite(parsed)) {
      return Math.max(2, Math.min(5000, Math.floor(parsed)))
    }
  }
  return 40
}

const sanitizeAutoSummaryWindow = (
  value: unknown,
  threshold: number
): number => {
  const maxWindow = Math.max(1, threshold - 1)
  if (typeof value === "number" && Number.isFinite(value)) {
    return Math.max(1, Math.min(maxWindow, Math.floor(value)))
  }
  if (typeof value === "string") {
    const parsed = Number.parseInt(value, 10)
    if (Number.isFinite(parsed)) {
      return Math.max(1, Math.min(maxWindow, Math.floor(parsed)))
    }
  }
  return Math.min(12, maxWindow)
}

const sanitizeGenerationFloat = (
  value: unknown,
  min: number,
  max: number
): number | null => {
  if (typeof value === "number" && Number.isFinite(value)) {
    return Math.max(min, Math.min(max, value))
  }
  if (typeof value === "string") {
    const parsed = Number.parseFloat(value)
    if (Number.isFinite(parsed)) {
      return Math.max(min, Math.min(max, parsed))
    }
  }
  return null
}

const normalizeAuthorNotePosition = (
  value: unknown
): { mode: "before_system" | "depth"; depth: number } => {
  if (typeof value === "number" && Number.isFinite(value)) {
    return { mode: "depth", depth: sanitizeDepth(value) }
  }

  if (typeof value === "string") {
    const normalized = value.trim().toLowerCase()
    if (
      normalized === "before_system" ||
      normalized === "before-system" ||
      normalized === "before system" ||
      normalized === "before"
    ) {
      return { mode: "before_system", depth: 1 }
    }
    if (normalized.startsWith("depth")) {
      const suffix = normalized.replace(/^depth\s*:?\s*/, "")
      return { mode: "depth", depth: sanitizeDepth(suffix) }
    }
  }

  if (value && typeof value === "object") {
    const record = value as Record<string, unknown>
    const modeRaw = record.mode
    const mode =
      typeof modeRaw === "string" ? modeRaw.trim().toLowerCase() : ""
    if (mode === "depth" || mode === "at_depth") {
      return { mode: "depth", depth: sanitizeDepth(record.depth) }
    }
    if (
      mode === "before_system" ||
      mode === "before-system" ||
      mode === "before"
    ) {
      return { mode: "before_system", depth: 1 }
    }
  }

  return { mode: "before_system", depth: 1 }
}

const normalizeGreetingScope = (value: unknown): "chat" | "character" => {
  if (typeof value === "string" && value.trim().toLowerCase() === "character") {
    return "character"
  }
  return "chat"
}

const normalizePresetScope = (value: unknown): "chat" | "character" => {
  if (typeof value === "string" && value.trim().toLowerCase() === "chat") {
    return "chat"
  }
  return "character"
}

const normalizeChatPresetOverrideId = (
  value: unknown
): CharacterPromptPresetId | null => {
  if (!isCharacterPromptPresetId(value)) return null
  return value
}

const normalizeMemoryScope = (
  value: unknown
): "shared" | "character" | "both" => {
  if (typeof value === "string") {
    const normalized = value.trim().toLowerCase()
    if (
      normalized === "shared" ||
      normalized === "character" ||
      normalized === "both"
    ) {
      return normalized
    }
  }
  return "shared"
}

const normalizeTurnTakingMode = (value: unknown): "single" | "round_robin" => {
  if (typeof value === "string") {
    const normalized = value.trim().toLowerCase()
    if (
      normalized === "round_robin" ||
      normalized === "round-robin" ||
      normalized === "round robin"
    ) {
      return "round_robin"
    }
  }
  return "single"
}

const normalizeCharacterIdValue = (value: unknown): string | null => {
  const text = String(value ?? "").trim()
  if (!text) return null
  return text
}

const normalizeParticipantCharacterIds = (value: unknown): string[] => {
  const normalizedIds: string[] = []
  const seen = new Set<string>()

  const append = (candidate: unknown) => {
    const normalizedId = normalizeCharacterIdValue(candidate)
    if (!normalizedId || seen.has(normalizedId)) return
    seen.add(normalizedId)
    normalizedIds.push(normalizedId)
  }

  if (Array.isArray(value)) {
    value.forEach(append)
    return normalizedIds
  }

  if (typeof value === "string") {
    const text = value.trim()
    if (!text) return normalizedIds
    try {
      const parsed = JSON.parse(text)
      if (Array.isArray(parsed)) {
        parsed.forEach(append)
        return normalizedIds
      }
    } catch {
      // Fallback to comma-separated input.
    }
    text
      .split(",")
      .map((entry) => entry.trim())
      .forEach(append)
    return normalizedIds
  }

  return normalizedIds
}

const normalizeDirectedCharacterId = (value: unknown): string | null => {
  return normalizeCharacterIdValue(value)
}

const normalizeCharacterMemoryById = (
  value: unknown
): Record<string, { note: string; updatedAt?: string }> => {
  if (!value || typeof value !== "object") return {}
  const normalized: Record<string, { note: string; updatedAt?: string }> = {}
  for (const [key, entry] of Object.entries(value as Record<string, unknown>)) {
    const normalizedId = normalizeCharacterIdValue(key)
    if (!normalizedId) continue
    if (entry && typeof entry === "object") {
      const note = String((entry as { note?: unknown }).note ?? "").trim()
      if (!note) continue
      const updatedAt = (entry as { updatedAt?: unknown }).updatedAt
      const normalizedUpdatedAt =
        typeof updatedAt === "string" ? updatedAt.trim() : ""
      normalized[normalizedId] = {
        note,
        updatedAt: normalizedUpdatedAt || undefined
      }
      continue
    }
    const note = String(entry ?? "").trim()
    if (!note) continue
    normalized[normalizedId] = { note }
  }
  return normalized
}

const normalizeGenerationStopList = (value: unknown): string[] => {
  if (!Array.isArray(value)) return []
  const seen = new Set<string>()
  const stops: string[] = []
  value.forEach((entry) => {
    if (typeof entry !== "string") return
    const normalized = entry.trim()
    if (!normalized || seen.has(normalized)) return
    seen.add(normalized)
    stops.push(normalized)
  })
  return stops
}

const parseGenerationStopListFromText = (text: string): string[] => {
  const seen = new Set<string>()
  const stops: string[] = []
  text
    .split("\n")
    .map((entry) => entry.trim())
    .forEach((entry) => {
      if (!entry || seen.has(entry)) return
      seen.add(entry)
      stops.push(entry)
    })
  return stops
}

export function ConversationTab({
  useDrawer,
  historyId,
  selectedSystemPrompt,
  onSystemPromptChange,
  onResetSystemPrompt,
  uploadedFiles,
  onRemoveFile,
  serverChatId,
  serverChatAssistantKind,
  serverChatPersonaMemoryMode,
  serverChatState,
  onStateChange,
  serverChatTopic,
  onTopicChange,
  onVersionChange,
  onPersonaMemoryModeChange
}: ConversationTabProps) {
  const { t } = useTranslation(["common", "playground"])
  const notification = useAntdNotification()
  const queryClient = useQueryClient()
  const { settings: chatSettings, updateSettings } = useChatSettingsRecord({
    historyId,
    serverChatId
  })
  const handleAsyncError = React.useCallback(
    (error: unknown) => {
      notification.error({
        message: t("common:error", { defaultValue: "Error" }),
        description:
          (isErrorWithMessage(error)
            ? error.message
            : t("common:somethingWentWrong", {
                defaultValue: "Something went wrong"
              }))
      })
    },
    [notification, t]
  )
  const [authorNoteDraft, setAuthorNoteDraft] = React.useState("")
  const [authorNoteMode, setAuthorNoteMode] = React.useState<
    "before_system" | "depth"
  >("before_system")
  const [authorNoteDepth, setAuthorNoteDepth] = React.useState(1)
  const [memoryCharacterId, setMemoryCharacterId] = React.useState<string | null>(
    null
  )
  const [memoryNoteDraft, setMemoryNoteDraft] = React.useState("")
  const [isResettingSystemPrompt, setIsResettingSystemPrompt] =
    React.useState(false)
  const summarySettings =
    chatSettings?.summary && typeof chatSettings.summary === "object"
      ? (chatSettings.summary as Record<string, unknown>)
      : null
  const persistedAutoSummaryEnabled = Boolean(
    chatSettings?.autoSummaryEnabled ?? summarySettings?.enabled ?? false
  )
  const persistedAutoSummaryThreshold = sanitizeAutoSummaryThreshold(
    chatSettings?.autoSummaryThresholdMessages ??
      summarySettings?.thresholdMessages
  )
  const persistedAutoSummaryWindow = sanitizeAutoSummaryWindow(
    chatSettings?.autoSummaryWindowMessages ?? summarySettings?.windowMessages,
    persistedAutoSummaryThreshold
  )
  const [autoSummaryEnabledDraft, setAutoSummaryEnabledDraft] = React.useState(
    persistedAutoSummaryEnabled
  )
  const [autoSummaryThresholdDraft, setAutoSummaryThresholdDraft] =
    React.useState(persistedAutoSummaryThreshold)
  const [autoSummaryWindowDraft, setAutoSummaryWindowDraft] = React.useState(
    persistedAutoSummaryWindow
  )
  const personaMemoryModeValue = serverChatPersonaMemoryMode ?? "read_only"
  const generationOverrideSource =
    chatSettings?.chatGenerationOverride &&
    typeof chatSettings.chatGenerationOverride === "object"
      ? (chatSettings.chatGenerationOverride as Record<string, unknown>)
      : chatSettings?.generationOverrides &&
          typeof chatSettings.generationOverrides === "object"
        ? (chatSettings.generationOverrides as Record<string, unknown>)
        : null
  const persistedGenerationOverrideEnabled = Boolean(
    generationOverrideSource?.enabled ?? false
  )
  const persistedGenerationTemperature = sanitizeGenerationFloat(
    generationOverrideSource?.temperature,
    0,
    2
  )
  const persistedGenerationTopP = sanitizeGenerationFloat(
    generationOverrideSource?.top_p,
    0,
    1
  )
  const persistedGenerationRepetitionPenalty = sanitizeGenerationFloat(
    generationOverrideSource?.repetition_penalty,
    0,
    3
  )
  const persistedGenerationStops = normalizeGenerationStopList(
    generationOverrideSource?.stop
  )
  const persistedGenerationStopsKey = persistedGenerationStops.join("\n")
  const [generationOverrideEnabledDraft, setGenerationOverrideEnabledDraft] =
    React.useState(persistedGenerationOverrideEnabled)
  const [generationTemperatureDraft, setGenerationTemperatureDraft] =
    React.useState<number | null>(persistedGenerationTemperature)
  const [generationTopPDraft, setGenerationTopPDraft] = React.useState<
    number | null
  >(persistedGenerationTopP)
  const [generationRepetitionPenaltyDraft, setGenerationRepetitionPenaltyDraft] =
    React.useState<number | null>(persistedGenerationRepetitionPenalty)
  const [generationStopsDraft, setGenerationStopsDraft] = React.useState(
    persistedGenerationStopsKey
  )
  const greetingScopeValue = normalizeGreetingScope(chatSettings?.greetingScope)
  const presetScopeValue = normalizePresetScope(chatSettings?.presetScope)
  const chatPresetOverrideValue = normalizeChatPresetOverrideId(
    chatSettings?.chatPresetOverrideId
  )
  const memoryScopeValue = normalizeMemoryScope(chatSettings?.memoryScope)
  const turnTakingModeValue = normalizeTurnTakingMode(chatSettings?.turnTakingMode)
  const directedCharacterIdValue = normalizeDirectedCharacterId(
    chatSettings?.directedCharacterId
  )
  const characterMemoryById = React.useMemo(
    () => normalizeCharacterMemoryById(chatSettings?.characterMemoryById),
    [chatSettings?.characterMemoryById]
  )
  const participantCharacterIdsValue = normalizeParticipantCharacterIds(
    chatSettings?.participantCharacterIds
  )
  const {
    data: availableCharacters = [],
    isError: isAvailableCharactersError
  } = useQuery<CharacterLite[]>({
    queryKey: CONVERSATION_TAB_QUERY_KEYS.listCharacters,
    queryFn: async () => {
      await tldwClient.initialize()
      const list = await tldwClient.listCharacters({ limit: 200 })
      return Array.isArray(list) ? (list as CharacterLite[]) : []
    },
    staleTime: 60_000
  })
  const {
    data: pinnedMessages = [],
    isError: isPinnedMessagesError
  } = useQuery<
    Array<{ id: string; role: string; content: string }>
  >({
    queryKey: CONVERSATION_TAB_QUERY_KEYS.pinnedMessages(serverChatId),
    enabled: Boolean(serverChatId),
    queryFn: async () => {
      if (!serverChatId) return []
      await tldwClient.initialize()
      const list = await tldwClient.listChatMessages(serverChatId, {
        include_deleted: "false",
        include_metadata: "true"
      })
      return list
        .filter((item) => Boolean(item.pinned))
        .map((item) => ({
          id: item.id,
          role: item.role,
          content: String(item.content ?? "")
        }))
    },
    staleTime: 10_000
  })

  React.useEffect(() => {
    setAuthorNoteDraft(
      typeof chatSettings?.authorNote === "string" ? chatSettings.authorNote : ""
    )
    const normalized = normalizeAuthorNotePosition(chatSettings?.authorNotePosition)
    setAuthorNoteMode(normalized.mode)
    setAuthorNoteDepth(normalized.depth)
  }, [chatSettings?.authorNote, chatSettings?.authorNotePosition])

  React.useEffect(() => {
    setAutoSummaryEnabledDraft(persistedAutoSummaryEnabled)
    setAutoSummaryThresholdDraft(persistedAutoSummaryThreshold)
    setAutoSummaryWindowDraft(persistedAutoSummaryWindow)
  }, [
    persistedAutoSummaryEnabled,
    persistedAutoSummaryThreshold,
    persistedAutoSummaryWindow
  ])

  React.useEffect(() => {
    setGenerationOverrideEnabledDraft(persistedGenerationOverrideEnabled)
    setGenerationTemperatureDraft(persistedGenerationTemperature)
    setGenerationTopPDraft(persistedGenerationTopP)
    setGenerationRepetitionPenaltyDraft(persistedGenerationRepetitionPenalty)
    setGenerationStopsDraft(persistedGenerationStopsKey)
  }, [
    persistedGenerationOverrideEnabled,
    persistedGenerationTemperature,
    persistedGenerationTopP,
    persistedGenerationRepetitionPenalty,
    persistedGenerationStopsKey
  ])

  const conversationStateOptions = CONVERSATION_STATE_OPTIONS.map((option) => ({
    value: option.value,
    label: t(option.labelToken, option.defaultLabel)
  }))

  const authorNotePositionOptions = [
    {
      value: "before_system",
      label: t("playground:composer.authorNotePosition.beforeSystem", {
        defaultValue: "Before system"
      })
    },
    {
      value: "depth",
      label: t("playground:composer.authorNotePosition.depthN", {
        defaultValue: "Depth N"
      })
    }
  ]
  const greetingScopeOptions = [
    {
      value: "chat",
      label: t("playground:composer.greetingScope.chat", {
        defaultValue: "Per chat"
      })
    },
    {
      value: "character",
      label: t("playground:composer.greetingScope.character", {
        defaultValue: "Per character"
      })
    }
  ]
  const presetScopeOptions = [
    {
      value: "chat",
      label: t("playground:composer.presetScope.chat", {
        defaultValue: "Chat override"
      })
    },
    {
      value: "character",
      label: t("playground:composer.presetScope.character", {
        defaultValue: "Per character"
      })
    }
  ]
  const chatPresetOptions = CHARACTER_PROMPT_PRESETS.map((preset) => ({
    value: preset.id,
    label: t(
      `settings:manageCharacters.promptPresets.${preset.id}.label`,
      preset.label
    ),
    title: t(
      `settings:manageCharacters.promptPresets.${preset.id}.description`,
      preset.description
    )
  }))
  const memoryScopeOptions = [
    {
      value: "shared",
      label: t("playground:composer.memoryScope.shared", {
        defaultValue: "Shared"
      })
    },
    {
      value: "character",
      label: t("playground:composer.memoryScope.character", {
        defaultValue: "Per character"
      })
    },
    {
      value: "both",
      label: t("playground:composer.memoryScope.both", {
        defaultValue: "Both"
      })
    }
  ]
  const turnTakingModeOptions = [
    {
      value: "single",
      label: t("playground:composer.turnTakingMode.single", {
        defaultValue: "Single speaker"
      })
    },
    {
      value: "round_robin",
      label: t("playground:composer.turnTakingMode.roundRobin", {
        defaultValue: "Round robin"
      })
    }
  ]
  const participantCharacterOptions = availableCharacters.map((character) => ({
    value: String(character.id),
    label: character.name || String(character.id)
  }))

  React.useEffect(() => {
    if (participantCharacterOptions.length === 0) {
      setMemoryCharacterId(null)
      return
    }
    if (
      memoryCharacterId &&
      participantCharacterOptions.some((option) => option.value === memoryCharacterId)
    ) {
      return
    }
    const optionValues = new Set(participantCharacterOptions.map((option) => option.value))
    const preferredFromMap = Object.keys(characterMemoryById).find((key) =>
      optionValues.has(key)
    )
    const fallback =
      preferredFromMap ||
      participantCharacterIdsValue.find((id) => optionValues.has(id)) ||
      participantCharacterOptions[0]?.value ||
      null
    setMemoryCharacterId(fallback)
  }, [
    characterMemoryById,
    memoryCharacterId,
    participantCharacterIdsValue,
    participantCharacterOptions
  ])

  React.useEffect(() => {
    if (!memoryCharacterId) {
      setMemoryNoteDraft("")
      return
    }
    setMemoryNoteDraft(characterMemoryById[memoryCharacterId]?.note || "")
  }, [characterMemoryById, memoryCharacterId])

  const persistAuthorNote = async (value: string) => {
    try {
      await updateSettings({
        authorNote: value.trim()
      })
    } catch (error: unknown) {
      handleAsyncError(error)
    }
  }

  const persistAuthorNotePosition = async (
    mode: "before_system" | "depth",
    depth: number
  ) => {
    try {
      if (mode === "depth") {
        await updateSettings({
          authorNotePosition: {
            mode: "depth",
            depth: sanitizeDepth(depth)
          }
        })
        return
      }
      await updateSettings({ authorNotePosition: "before_system" })
    } catch (error: unknown) {
      handleAsyncError(error)
    }
  }
  const persistGreetingScope = async (value: "chat" | "character") => {
    await updateSettings({ greetingScope: value })
  }
  const persistPresetScope = async (value: "chat" | "character") => {
    await updateSettings({ presetScope: value })
  }
  const persistChatPresetOverrideId = async (
    value: CharacterPromptPresetId | null
  ) => {
    await updateSettings({ chatPresetOverrideId: value })
  }
  const persistChatGenerationOverride = async (
    override: {
      enabled?: boolean
      temperature?: number | null
      top_p?: number | null
      repetition_penalty?: number | null
      stop?: string[]
    } = {}
  ) => {
    const nextEnabled =
      typeof override.enabled === "boolean"
        ? override.enabled
        : generationOverrideEnabledDraft
    const nextTemperature =
      override.temperature !== undefined
        ? sanitizeGenerationFloat(override.temperature, 0, 2)
        : generationTemperatureDraft
    const nextTopP =
      override.top_p !== undefined
        ? sanitizeGenerationFloat(override.top_p, 0, 1)
        : generationTopPDraft
    const nextRepetitionPenalty =
      override.repetition_penalty !== undefined
        ? sanitizeGenerationFloat(override.repetition_penalty, 0, 3)
        : generationRepetitionPenaltyDraft
    const nextStops =
      override.stop !== undefined
        ? normalizeGenerationStopList(override.stop)
        : parseGenerationStopListFromText(generationStopsDraft)

    await updateSettings({
      chatGenerationOverride: {
        enabled: nextEnabled,
        temperature: nextTemperature,
        top_p: nextTopP,
        repetition_penalty: nextRepetitionPenalty,
        stop: nextStops,
        updatedAt: new Date().toISOString()
      }
    })
  }
  const persistMemoryScope = async (
    value: "shared" | "character" | "both"
  ) => {
    await updateSettings({ memoryScope: value })
  }
  const persistTurnTakingMode = async (value: "single" | "round_robin") => {
    await updateSettings({ turnTakingMode: value })
  }
  const persistParticipantCharacterIds = async (values: string[]) => {
    const normalized = normalizeParticipantCharacterIds(values)
    await updateSettings({ participantCharacterIds: normalized })
  }
  const persistDirectedCharacterId = async (value: string | null) => {
    const normalized = normalizeDirectedCharacterId(value)
    await updateSettings({
      directedCharacterId:
        normalized !== null && /^\d+$/.test(normalized)
          ? Number.parseInt(normalized, 10)
          : null
    })
  }
  const persistCharacterMemoryNote = async (
    characterId: string,
    noteDraft: string
  ) => {
    const normalizedId = normalizeDirectedCharacterId(characterId)
    if (!normalizedId) return
    const nextMap = { ...characterMemoryById }
    const nextNote = noteDraft.trim()
    if (nextNote.length === 0) {
      delete nextMap[normalizedId]
    } else {
      nextMap[normalizedId] = {
        note: nextNote,
        updatedAt: new Date().toISOString()
      }
    }
    await updateSettings({ characterMemoryById: nextMap })
  }
  const persistAutoSummaryEnabled = async (enabled: boolean) => {
    await updateSettings({ autoSummaryEnabled: enabled })
  }
  const persistAutoSummaryThreshold = async (value: unknown) => {
    const normalizedThreshold = sanitizeAutoSummaryThreshold(value)
    const normalizedWindow = sanitizeAutoSummaryWindow(
      autoSummaryWindowDraft,
      normalizedThreshold
    )
    setAutoSummaryThresholdDraft(normalizedThreshold)
    setAutoSummaryWindowDraft(normalizedWindow)
    await updateSettings({
      autoSummaryThresholdMessages: normalizedThreshold,
      autoSummaryWindowMessages: normalizedWindow
    })
  }
  const persistAutoSummaryWindow = async (value: unknown) => {
    const normalizedThreshold = sanitizeAutoSummaryThreshold(
      autoSummaryThresholdDraft
    )
    const normalizedWindow = sanitizeAutoSummaryWindow(
      value,
      normalizedThreshold
    )
    setAutoSummaryWindowDraft(normalizedWindow)
    await updateSettings({
      autoSummaryThresholdMessages: normalizedThreshold,
      autoSummaryWindowMessages: normalizedWindow
    })
  }

  const handleStateChange = async (val: string) => {
    const next = normalizeConversationState(val)
    onStateChange(next)
    if (!serverChatId) return
    try {
      const updated = await tldwClient.updateChat(serverChatId, {
        state: next
      })
      onVersionChange(getUpdateChatVersion(updated))
      queryClient.invalidateQueries({ queryKey: ["serverChatHistory"] })
    } catch (error: unknown) {
      handleAsyncError(error)
    }
  }

  const handleTopicBlur = async (value: string) => {
    const normalized = value.trim()
    const topicValue = normalized.length > 0 ? normalized : null
    onTopicChange(topicValue)
    if (!serverChatId) return
    try {
      const updated = await tldwClient.updateChat(serverChatId, {
        topic_label: topicValue || undefined
      })
      onVersionChange(getUpdateChatVersion(updated))
      queryClient.invalidateQueries({ queryKey: ["serverChatHistory"] })
    } catch (error: unknown) {
      handleAsyncError(error)
    }
  }

  const handlePersonaMemoryModeChange = async (value: string) => {
    const nextMode = value === "read_write" ? "read_write" : "read_only"
    onPersonaMemoryModeChange?.(nextMode)
    if (!serverChatId) return
    try {
      const updated = await tldwClient.updateChat(serverChatId, {
        persona_memory_mode: nextMode
      })
      onVersionChange(getUpdateChatVersion(updated))
      queryClient.invalidateQueries({ queryKey: ["serverChatHistory"] })
    } catch (error: unknown) {
      handleAsyncError(error)
    }
  }

  const previewSettingsFingerprint = React.useMemo(
    () =>
      JSON.stringify({
        authorNote: chatSettings?.authorNote || "",
        authorNotePosition: chatSettings?.authorNotePosition || null,
        greetingScope: greetingScopeValue,
        presetScope: presetScopeValue,
        memoryScope: memoryScopeValue,
        turnTakingMode: turnTakingModeValue,
        participantCharacterIds: participantCharacterIdsValue.join(","),
        directedCharacterId: directedCharacterIdValue,
        characterMemoryById: JSON.stringify(characterMemoryById),
        autoSummaryEnabled: autoSummaryEnabledDraft,
        autoSummaryThresholdMessages: autoSummaryThresholdDraft,
        autoSummaryWindowMessages: autoSummaryWindowDraft,
        chatGenerationOverrideEnabled: generationOverrideEnabledDraft,
        chatGenerationOverrideTemperature: generationTemperatureDraft,
        chatGenerationOverrideTopP: generationTopPDraft,
        chatGenerationOverrideRepetitionPenalty:
          generationRepetitionPenaltyDraft,
        chatGenerationOverrideStop: generationStopsDraft,
        summaryUpdatedAt:
          typeof summarySettings?.updatedAt === "string"
            ? summarySettings.updatedAt
            : null,
        greetingEnabled: chatSettings?.greetingEnabled ?? true,
        greetingSelectionId: chatSettings?.greetingSelectionId || null,
        chatPresetOverrideId: chatSettings?.chatPresetOverrideId || null
      }),
    [
      chatSettings?.authorNote,
      chatSettings?.authorNotePosition,
      greetingScopeValue,
      presetScopeValue,
      memoryScopeValue,
      turnTakingModeValue,
      participantCharacterIdsValue,
      directedCharacterIdValue,
      characterMemoryById,
      autoSummaryEnabledDraft,
      autoSummaryThresholdDraft,
      autoSummaryWindowDraft,
      generationOverrideEnabledDraft,
      generationTemperatureDraft,
      generationTopPDraft,
      generationRepetitionPenaltyDraft,
      generationStopsDraft,
      summarySettings,
      chatSettings?.greetingEnabled,
      chatSettings?.greetingSelectionId,
      chatSettings?.chatPresetOverrideId
    ]
  )

  const handleResetSystemPrompt = React.useCallback(async () => {
    if (!onResetSystemPrompt) return
    try {
      setIsResettingSystemPrompt(true)
      await onResetSystemPrompt()
    } finally {
      setIsResettingSystemPrompt(false)
    }
  }, [onResetSystemPrompt])

  return (
    <div className="space-y-4">
      <Form.Item
        name="systemPrompt"
        help={t("common:modelSettings.form.systemPrompt.help", {
          defaultValue:
            "Applies persistently to this conversation and overrides the selected system prompt template until you reset it."
        })}
        label={t("common:modelSettings.form.systemPrompt.label", {
          defaultValue: "Conversation System Prompt"
        })}>
        <div className="space-y-1">
          <Input.TextArea
            rows={useDrawer ? 4 : 6}
            placeholder={t(
              "common:modelSettings.form.systemPrompt.placeholder",
              { defaultValue: "Enter System Prompt" }
            )}
            onChange={(e) => onSystemPromptChange(e.target.value)}
          />
          <div className="flex items-center justify-end">
            <Button
              type="link"
              size="small"
              className="px-0"
              loading={isResettingSystemPrompt}
              onClick={() => {
                void handleResetSystemPrompt()
              }}>
              {selectedSystemPrompt
                ? t("common:modelSettings.form.systemPrompt.resetTemplate", {
                    defaultValue: "Reset to template"
                  })
                : t("common:modelSettings.form.systemPrompt.resetDefault", {
                    defaultValue: "Reset to default"
                  })}
            </Button>
          </div>
          {selectedSystemPrompt && (
            <div className="inline-flex items-center gap-2 rounded-full border border-primary/30 bg-primary/10 px-2 py-0.5 text-[11px] text-primary">
              <span className="inline-block h-1.5 w-1.5 rounded-full bg-primary" />
              <span>
                {t(
                  "playground:composer.sceneTemplateActive",
                  "Scene template active: Actor respects template interaction settings."
                )}
              </span>
            </div>
          )}
        </div>
      </Form.Item>

      {uploadedFiles.length > 0 && (
        <div className="mb-4">
          <div className="flex items-center justify-between mb-3">
            <h4 className="font-medium text-text">
              {t("playground:composer.uploadedFiles", {
                count: uploadedFiles.length,
                defaultValue: "Uploaded Files ({{count}})"
              })}
            </h4>
            <span className="text-xs text-text-muted">
              {t(
                "playground:composer.manageContextHint",
                "Manage Knowledge Search in the Context tab."
              )}
            </span>
          </div>
          <div className="space-y-2 max-h-32 overflow-y-auto">
            {uploadedFiles.map((file) => (
              <div
                key={file.id}
                className="flex items-center justify-between p-2 bg-surface2 rounded-md">
                <div className="flex items-center gap-2 flex-1 min-w-0">
                  <FileIcon className="h-4 w-4 flex-shrink-0 text-text-subtle" />
                  <div className="min-w-0 flex-1">
                    <p className="text-sm font-medium text-text truncate">
                      {file.filename}
                    </p>
                    <div className="flex items-center gap-2 text-xs text-text-subtle">
                      <span>{(file.size / 1024).toFixed(1)} KB</span>
                      {typeof file.processed === "boolean" && (
                        <span className="flex items-center gap-1">
                          <span
                            className={`inline-block w-2 h-2 rounded-full ${
                              file.processed ? "bg-success" : "bg-warn"
                            }`}
                          />
                          {file.processed
                            ? t("playground:composer.fileStatusProcessed", {
                                defaultValue: "Processed"
                              })
                            : t("playground:composer.fileStatusProcessing", {
                                defaultValue: "Processing..."
                              })}
                        </span>
                      )}
                    </div>
                  </div>
                </div>
                <button
                  onClick={() => onRemoveFile(file.id)}
                  className="rounded p-1 text-danger hover:bg-danger/10 hover:text-danger">
                  <X className="h-4 w-4" />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      <Form.Item
        label={t("playground:composer.conversationTags", "Conversation state")}
        help={t(
          "playground:composer.stateHelp",
          'Default state is "in-progress." Update it as the conversation progresses.'
        )}>
        <Select
          data-testid="conversation-state-select"
          value={serverChatState || "in-progress"}
          options={conversationStateOptions}
          onChange={handleStateChange}
        />
      </Form.Item>

      {serverChatAssistantKind === "persona" ? (
        <Form.Item
          label={t(
            "playground:composer.personaMemoryMode.label",
            "Persona memory mode"
          )}
          help={t(
            "playground:composer.personaMemoryMode.help",
            "Read-only keeps persona memory available for context without writing new durable memories from this chat."
          )}
        >
          <Select
            data-testid="persona-memory-mode-select"
            value={personaMemoryModeValue}
            options={[
              {
                value: "read_only",
                label: t(
                  "playground:composer.personaMemoryMode.readOnly",
                  "Read only"
                )
              },
              {
                value: "read_write",
                label: t(
                  "playground:composer.personaMemoryMode.readWrite",
                  "Read + write"
                )
              }
            ]}
            onChange={(value) => {
              void handlePersonaMemoryModeChange(String(value))
            }}
          />
        </Form.Item>
      ) : null}

      <Form.Item
        label={t("playground:composer.topicPlaceholder", "Conversation tag")}
        help={t(
          "playground:composer.persistence.topicHelp",
          "Optional label for this chat; saved to the server when available."
        )}>
        <Input
          value={serverChatTopic || ""}
          onChange={(e) => onTopicChange(e.target.value || null)}
          onBlur={(e) => handleTopicBlur(e.target.value)}
          placeholder={t(
            "playground:composer.topicPlaceholder",
            "Conversation tag (optional)"
          )}
        />
      </Form.Item>

      <Form.Item
        label={t("playground:composer.authorNote.label", "Author note")}
        help={t(
          "playground:composer.authorNote.help",
          "Optional note injected into character prompts for this chat."
        )}
      >
        <Input.TextArea
          rows={3}
          showCount
          maxLength={1200}
          value={authorNoteDraft}
          onChange={(e) => setAuthorNoteDraft(e.target.value)}
          onBlur={() => {
            const persisted =
              typeof chatSettings?.authorNote === "string" ? chatSettings.authorNote : ""
            if (persisted.trim() === authorNoteDraft.trim()) return
            void persistAuthorNote(authorNoteDraft)
          }}
          placeholder={t(
            "playground:composer.authorNote.placeholder",
            "E.g., Keep responses grounded, avoid repetition, and progress the scene."
          )}
        />
      </Form.Item>

      <Form.Item
        label={t(
          "playground:composer.authorNotePosition.label",
          "Author note insertion"
        )}
        help={t(
          "playground:composer.authorNotePosition.help",
          "Choose whether the note is inserted before system instructions or at conversation depth N."
        )}
      >
        <div className="space-y-2">
          <Select
            value={authorNoteMode}
            options={authorNotePositionOptions}
            onChange={(value) => {
              const nextMode = value === "depth" ? "depth" : "before_system"
              setAuthorNoteMode(nextMode)
              void persistAuthorNotePosition(nextMode, authorNoteDepth)
            }}
          />
          {authorNoteMode === "depth" && (
            <InputNumber
              min={0}
              step={1}
              value={authorNoteDepth}
              onChange={(value) => {
                setAuthorNoteDepth(sanitizeDepth(value))
              }}
              onBlur={() => {
                void persistAuthorNotePosition("depth", authorNoteDepth)
              }}
              addonBefore={t(
                "playground:composer.authorNotePosition.depthLabel",
                "Depth"
              )}
            />
          )}
        </div>
      </Form.Item>

      <Form.Item
        label={t("playground:composer.greetingScope.label", "Greeting scope")}
        help={t("playground:composer.greetingScope.help", {
          defaultValue:
            "Choose whether greeting behavior is resolved per chat or per character."
        })}
      >
        <Select
          value={greetingScopeValue}
          options={greetingScopeOptions}
          onChange={(value) => {
            const next = value === "character" ? "character" : "chat"
            void persistGreetingScope(next)
          }}
        />
      </Form.Item>

      <Form.Item
        label={t("playground:composer.presetScope.label", "Preset scope")}
        help={t("playground:composer.presetScope.help", {
          defaultValue:
            "Choose whether chat preset override or per-character preset is used by default."
        })}
      >
        <Select
          value={presetScopeValue}
          options={presetScopeOptions}
          onChange={(value) => {
            const next = value === "chat" ? "chat" : "character"
            void persistPresetScope(next)
          }}
        />
      </Form.Item>

      {presetScopeValue === "chat" && (
        <Form.Item
          label={t(
            "playground:composer.chatPresetOverride.label",
            "Chat preset override"
          )}
          help={t("playground:composer.chatPresetOverride.help", {
            defaultValue:
              "When preset scope is chat, this preset applies to every turn until changed."
          })}
        >
          <Select
            value={chatPresetOverrideValue || DEFAULT_CHARACTER_PROMPT_PRESET}
            options={chatPresetOptions}
            onChange={(value) => {
              const next = normalizeChatPresetOverrideId(value)
              void persistChatPresetOverrideId(
                next || DEFAULT_CHARACTER_PROMPT_PRESET
              )
            }}
          />
        </Form.Item>
      )}

      <Form.Item
        label={t(
          "playground:composer.chatGenerationOverride.label",
          "Chat generation override"
        )}
        help={t("playground:composer.chatGenerationOverride.help", {
          defaultValue:
            "Override character generation settings for this chat with explicit values."
        })}
      >
        <div className="space-y-2" data-testid="chat-generation-override">
          <Select
            value={generationOverrideEnabledDraft ? "on" : "off"}
            options={[
              {
                value: "off",
                label: t("playground:composer.chatGenerationOverride.off", {
                  defaultValue: "Override disabled"
                })
              },
              {
                value: "on",
                label: t("playground:composer.chatGenerationOverride.on", {
                  defaultValue: "Override enabled"
                })
              }
            ]}
            onChange={(value) => {
              const enabled = value === "on"
              setGenerationOverrideEnabledDraft(enabled)
              void persistChatGenerationOverride({ enabled })
            }}
          />
          <div className="grid grid-cols-1 gap-2 sm:grid-cols-3">
            <InputNumber
              min={0}
              max={2}
              step={0.01}
              precision={2}
              disabled={!generationOverrideEnabledDraft}
              value={generationTemperatureDraft}
              addonBefore={t(
                "playground:composer.chatGenerationOverride.temperature",
                "Temp"
              )}
              onChange={(value) => {
                setGenerationTemperatureDraft(
                  sanitizeGenerationFloat(value, 0, 2)
                )
              }}
              onBlur={() => {
                void persistChatGenerationOverride({
                  temperature: generationTemperatureDraft
                })
              }}
            />
            <InputNumber
              min={0}
              max={1}
              step={0.01}
              precision={2}
              disabled={!generationOverrideEnabledDraft}
              value={generationTopPDraft}
              addonBefore={t(
                "playground:composer.chatGenerationOverride.topP",
                "Top-p"
              )}
              onChange={(value) => {
                setGenerationTopPDraft(sanitizeGenerationFloat(value, 0, 1))
              }}
              onBlur={() => {
                void persistChatGenerationOverride({ top_p: generationTopPDraft })
              }}
            />
            <InputNumber
              min={0}
              max={3}
              step={0.01}
              precision={2}
              disabled={!generationOverrideEnabledDraft}
              value={generationRepetitionPenaltyDraft}
              addonBefore={t(
                "playground:composer.chatGenerationOverride.repetitionPenalty",
                "Rep pen"
              )}
              onChange={(value) => {
                setGenerationRepetitionPenaltyDraft(
                  sanitizeGenerationFloat(value, 0, 3)
                )
              }}
              onBlur={() => {
                void persistChatGenerationOverride({
                  repetition_penalty: generationRepetitionPenaltyDraft
                })
              }}
            />
          </div>
          <Input.TextArea
            rows={2}
            disabled={!generationOverrideEnabledDraft}
            value={generationStopsDraft}
            onChange={(event) => setGenerationStopsDraft(event.target.value)}
            onBlur={() => {
              void persistChatGenerationOverride({
                stop: parseGenerationStopListFromText(generationStopsDraft)
              })
            }}
            placeholder={t(
              "playground:composer.chatGenerationOverride.stopPlaceholder",
              "Stop sequences, one per line"
            )}
          />
        </div>
      </Form.Item>

      <Form.Item
        label={t("playground:composer.memoryScope.label", "Memory scope")}
        help={t("playground:composer.memoryScope.help", {
          defaultValue:
            "Choose whether memory is shared, per-character, or both."
        })}
      >
        <Select
          value={memoryScopeValue}
          options={memoryScopeOptions}
          onChange={(value) => {
            const normalized = normalizeMemoryScope(value)
            void persistMemoryScope(normalized)
          }}
        />
      </Form.Item>

      <Form.Item
        label={t("playground:composer.turnTakingMode.label", "Turn-taking mode")}
        help={t("playground:composer.turnTakingMode.help", {
          defaultValue:
            "Choose whether responses use one active speaker or rotate across selected participants."
        })}
      >
        <Select
          value={turnTakingModeValue}
          options={turnTakingModeOptions}
          onChange={(value) => {
            const normalized = normalizeTurnTakingMode(value)
            void persistTurnTakingMode(normalized)
          }}
        />
      </Form.Item>

      <Form.Item
        label={t("playground:composer.participants.label", "Participants")}
        help={t("playground:composer.participants.help", {
          defaultValue:
            "Select additional characters for multi-character chats. The chat's primary character remains included."
        })}
      >
        <div className="space-y-1">
          <Select
            mode="multiple"
            allowClear
            showSearch
            value={participantCharacterIdsValue}
            options={participantCharacterOptions}
            placeholder={t("playground:composer.participants.placeholder", {
              defaultValue: "Select additional characters"
            })}
            onChange={(values) => {
              const nextValues = Array.isArray(values)
                ? values.map((value) => String(value))
                : []
              void persistParticipantCharacterIds(nextValues)
            }}
          />
          {isAvailableCharactersError && (
            <p className="text-[11px] text-danger">
              {t("playground:composer.participants.loadError", {
                defaultValue:
                  "Failed to load character list. Participant options may be incomplete."
              })}
            </p>
          )}
        </div>
      </Form.Item>

      <Form.Item
        label={t(
          "playground:composer.directedReply.label",
          "Directed responder"
        )}
        help={t("playground:composer.directedReply.help", {
          defaultValue:
            "Optional: choose which participant should generate the next response."
        })}
      >
        <Select
          allowClear
          showSearch
          value={directedCharacterIdValue ?? undefined}
          options={participantCharacterOptions}
          placeholder={t("playground:composer.directedReply.placeholder", {
            defaultValue: "Automatic by turn-taking"
          })}
          onChange={(value) => {
            const normalized = value != null ? String(value) : null
            void persistDirectedCharacterId(normalized)
          }}
        />
      </Form.Item>

      {memoryScopeValue !== "shared" && (
        <Form.Item
          label={t(
            "playground:composer.characterMemory.label",
            "Per-character memory"
          )}
          help={t("playground:composer.characterMemory.help", {
            defaultValue:
              "Edit memory notes for individual characters used when memory scope includes per-character injection."
          })}
        >
          <div className="space-y-2">
            <Select
              showSearch
              value={memoryCharacterId ?? undefined}
              options={participantCharacterOptions}
              placeholder={t("playground:composer.characterMemory.target", {
                defaultValue: "Select character"
              })}
              onChange={(value) => {
                const normalized = value ? String(value) : null
                setMemoryCharacterId(normalized)
                setMemoryNoteDraft(
                  normalized ? characterMemoryById[normalized]?.note || "" : ""
                )
              }}
            />
            <Input.TextArea
              rows={3}
              showCount
              maxLength={600}
              value={memoryNoteDraft}
              onChange={(e) => setMemoryNoteDraft(e.target.value)}
              onBlur={() => {
                if (!memoryCharacterId) return
                const persisted = characterMemoryById[memoryCharacterId]?.note || ""
                if (persisted.trim() === memoryNoteDraft.trim()) return
                void persistCharacterMemoryNote(memoryCharacterId, memoryNoteDraft)
              }}
              placeholder={t(
                "playground:composer.characterMemory.placeholder",
                "Character-specific memory note"
              )}
            />
          </div>
        </Form.Item>
      )}

      <Form.Item
        label={t("playground:composer.autoSummary.label", "Auto summary")}
        help={t("playground:composer.autoSummary.help", {
          defaultValue:
            "Compress older unpinned messages once the threshold is reached."
        })}
      >
        <div className="space-y-2">
          <Select
            value={autoSummaryEnabledDraft ? "on" : "off"}
            options={[
              {
                value: "on",
                label: t("playground:composer.autoSummary.on", {
                  defaultValue: "Enabled"
                })
              },
              {
                value: "off",
                label: t("playground:composer.autoSummary.off", {
                  defaultValue: "Disabled"
                })
              }
            ]}
            onChange={(value) => {
              const enabled = value === "on"
              setAutoSummaryEnabledDraft(enabled)
              void persistAutoSummaryEnabled(enabled)
            }}
          />
          <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
            <InputNumber
              min={2}
              max={5000}
              step={1}
              value={autoSummaryThresholdDraft}
              disabled={!autoSummaryEnabledDraft}
              addonBefore={t(
                "playground:composer.autoSummary.threshold",
                "Threshold"
              )}
              onChange={(value) => {
                setAutoSummaryThresholdDraft(
                  sanitizeAutoSummaryThreshold(value)
                )
              }}
              onBlur={() => {
                void persistAutoSummaryThreshold(autoSummaryThresholdDraft)
              }}
            />
            <InputNumber
              min={1}
              max={Math.max(1, autoSummaryThresholdDraft - 1)}
              step={1}
              value={autoSummaryWindowDraft}
              disabled={!autoSummaryEnabledDraft}
              addonBefore={t(
                "playground:composer.autoSummary.window",
                "Recent window"
              )}
              onChange={(value) => {
                setAutoSummaryWindowDraft(
                  sanitizeAutoSummaryWindow(value, autoSummaryThresholdDraft)
                )
              }}
              onBlur={() => {
                void persistAutoSummaryWindow(autoSummaryWindowDraft)
              }}
            />
          </div>
        </div>
      </Form.Item>

      <Form.Item
        label={t("playground:composer.pinnedMessages.label", "Pinned messages")}
        help={t("playground:composer.pinnedMessages.help", {
          defaultValue:
            "Pinned messages are excluded from auto-summary compression."
        })}
      >
        <div className="space-y-2 max-h-40 overflow-y-auto rounded-md border border-border/60 bg-surface2/30 p-2">
          {isPinnedMessagesError ? (
            <div className="text-xs text-danger">
              {t("playground:composer.pinnedMessages.loadError", {
                defaultValue: "Failed to load pinned messages."
              })}
            </div>
          ) : pinnedMessages.length === 0 ? (
            <div className="text-xs text-text-muted">
              {t("playground:composer.pinnedMessages.empty", {
                defaultValue: "No pinned messages in this chat."
              })}
            </div>
          ) : (
            pinnedMessages.map((entry) => (
              <div
                key={entry.id}
                className="rounded bg-surface px-2 py-1 text-xs text-text"
              >
                <span className="mr-1 uppercase text-[10px] text-text-subtle">
                  {entry.role}
                </span>
                <span>{entry.content.slice(0, 180)}</span>
              </div>
            ))
          )}
        </div>
      </Form.Item>

      {typeof summarySettings?.content === "string" &&
        summarySettings.content.trim().length > 0 && (
          <Form.Item
            label={t(
              "playground:composer.autoSummary.currentSummary",
              "Current summary"
            )}
            help={t(
              "playground:composer.autoSummary.currentSummaryHelp",
              "Persisted server-side summary used for prompt compression."
            )}
          >
            <Input.TextArea
              readOnly
              rows={4}
              value={summarySettings.content}
            />
          </Form.Item>
        )}

      <PromptAssemblyPreview
        serverChatId={serverChatId}
        settingsFingerprint={previewSettingsFingerprint}
        serverChatAssistantKind={serverChatAssistantKind}
      />

      <LorebookDebugPanel
        serverChatId={serverChatId}
        settingsFingerprint={previewSettingsFingerprint}
        serverChatAssistantKind={serverChatAssistantKind}
      />
    </div>
  )
}
