import { FileIcon, X } from "lucide-react"
import { getChatSettingsRecovery, type ChatSettingsTarget } from "@/services/chat-settings"
import type { ChatSettingsRecord, ChatGenerationOverride } from "@/types/chat-session-settings"
import { useStoreMessageOption } from "@/store/option"
import React from "react"
import { Button, Form, Input, InputNumber, Select } from "antd"
import { useQuery, useQueryClient } from "@tanstack/react-query"
import { useTranslation } from "react-i18next"
import { tldwClient, type ConversationState } from "@/services/tldw/TldwApiClient"
import {
  CONVERSATION_STATE_OPTIONS,
  normalizeConversationState
} from "@/utils/conversation-state"
import { useChatSettingsRecord } from "@/hooks/chat/useChatSettingsRecord"
import { fetchAllServerChatMessages } from "@/hooks/chat/useServerChatLoader"
import { serverChatMirrorOwnerKey } from "@/db/dexie/server-chat-mirror"
import { toChatScopeParams } from "@/types/chat-scope"
import { createServicePromptScopeChangedError } from "@/services/tldw/service-prompt-scope-error"
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

export type ConversationDraftStatus = { dirty: boolean; pending: boolean; revision: number }

interface ConversationTabProps {
  useDrawer?: boolean
  historyId: string | null
  selectedSystemPrompt: string | null
  onSystemPromptChange: (value: string) => void
  onResetSystemPrompt?: () => void | Promise<void>
  uploadedFiles: UploadedFile[]
  onRemoveFile: (id: string) => void
  serverChatId: string | null
  serverChatVersion?: number | null
  serverChatCharacterId?: string | number | null
  isCurrent?: () => boolean
  onDraftChange?: (status: ConversationDraftStatus) => void
  serverChatAssistantKind?: "character" | "persona" | null
  serverChatPersonaMemoryMode?: "read_only" | "read_write" | null
  serverChatState: ConversationState | null
  onStateChange: (state: ConversationState) => void
  serverChatTopic: string | null
  onTopicChange: (topic: string | null) => void
  onVersionChange: (version: number | null) => void
  onPersonaMemoryModeChange?: (mode: "read_only" | "read_write") => void
}

type ConversationDetailsPatch = { state?: ConversationState; topic_label?: string | null }
type ConversationDraft = { settings: Partial<ChatSettingsRecord>; details: ConversationDetailsPatch; revision: number; recoveryId?: string }
type ConversationProposal = {
  draft: ConversationDraft
  target?: ChatSettingsTarget
  expectedVersion?: number
  settingsOperationId?: string
  settingsVerificationOnly?: boolean
  settingsAcknowledged: boolean
  detailsAcknowledged: boolean
  detailsAttempted: boolean
  observedVersion: number | null
  isCurrent: () => boolean
}

type CharacterLite = {
  id: string | number
  name?: string | null
}

// H1's deployed self-contained projector; broader effects need its behavior protocol.
const NATIVE_CONVERSATION_SETTING_KEYS = new Set([
  "schemaVersion", "updatedAt", "chatGenerationOverride", "generationOverrides", "turnTakingMode", "participantCharacterIds",
  "presetScope", "chatPresetOverrideId", "greetingScope", "greetingEnabled", "greetingSelectionId", "greetingsChecksum", "useCharacterDefault"
])

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
  serverChatVersion,
  serverChatCharacterId,
  isCurrent = () => true,
  onDraftChange,
  serverChatAssistantKind,
  serverChatPersonaMemoryMode,
  serverChatState,
  onStateChange,
  serverChatTopic,
  onTopicChange,
  onVersionChange
}: ConversationTabProps) {
  const numericFieldId = React.useId()
  const { t } = useTranslation(["common", "playground"])
  const queryClient = useQueryClient()
  const { settings: appliedSettings, updateSettings, isReady, isCurrent: settingsIsCurrent, mode, captureTarget, target: settingsTarget, settingsRecovery, settingsRecoveryError, refreshRecovery } = useChatSettingsRecord({
    historyId,
    serverChatId
  })
  const [draft, setDraft] = React.useState<ConversationDraft>({ settings: {}, details: {}, revision: 0 })
  const draftRef = React.useRef(draft)
  const [pending, setPending] = React.useState(false)
  const pendingRef = React.useRef<object | null>(null)
  const proposalRef = React.useRef<ConversationProposal | null>(null)
  const metadataVersionRef = React.useRef<number | null>(null)
  const [saveError, setSaveError] = React.useState<string | null>(null)
  const mounted = React.useRef(true)
  const [memoryCharacterId, setMemoryCharacterId] = React.useState<string | null>(null)
  const [generationStopsText, setGenerationStopsText] = React.useState<string | null>(null)
  const [isResettingSystemPrompt, setIsResettingSystemPrompt] = React.useState(false)
  const hasDraft = Object.keys(draft.settings).length > 0 || Object.keys(draft.details).length > 0
  const canEdit = Boolean(isReady && settingsIsCurrent?.() && isCurrent())
  const nativeHistory = mode === "ordinary" && Boolean(serverChatId)
  const primaryValue = typeof serverChatCharacterId === "number" ? serverChatCharacterId
    : typeof serverChatCharacterId === "string" && /^\d+$/.test(serverChatCharacterId.trim()) ? Number(serverChatCharacterId.trim()) : null
  const nativePrimaryId = primaryValue !== null && Number.isSafeInteger(primaryValue) && primaryValue > 0 ? primaryValue : null
  const nativeUnsupportedHelp = t("common:modelSettings.conversation.nativeUnavailableEffects", "Author notes, memory, directed replies and summaries are unavailable in saved server chat. Existing values are shown for reference.")
  const canApplyNativeSettings = (patch: Partial<ChatSettingsRecord>) =>
    Object.keys(patch).every(key => NATIVE_CONVERSATION_SETTING_KEYS.has(key)) &&
    (patch.turnTakingMode === undefined || patch.turnTakingMode === "single") &&
    (patch.participantCharacterIds === undefined || (nativePrimaryId !== null && Array.isArray(patch.participantCharacterIds) &&
      patch.participantCharacterIds.length === 1 && patch.participantCharacterIds[0] === nativePrimaryId))
  const verifyRecoveryOnly = Boolean(nativeHistory && draft.recoveryId && !canApplyNativeSettings(draft.settings))
  const recoveryUnsupported = Boolean(nativeHistory && settingsRecovery && !canApplyNativeSettings(settingsRecovery.patch))
  const applyLabel = verifyRecoveryOnly ? t("common:modelSettings.conversation.verifyPreviousChange", "Verify previous change")
    : saveError ? t("playground:composer.retryConversation", "Retry conversation changes") : t("playground:composer.applyConversation", "Apply conversation changes")
  const reportDraft = (next: ConversationDraft, busy: boolean) => onDraftChange?.({
    dirty: Object.keys(next.settings).length > 0 || Object.keys(next.details).length > 0,
    pending: busy, revision: next.revision
  })
  React.useLayoutEffect(() => {
    mounted.current = true
    onDraftChange?.({ dirty: false, pending: false, revision: 0 })
    return () => { mounted.current = false }
  }, [onDraftChange])
  const stage = (settings: Partial<ChatSettingsRecord>, details: ConversationDetailsPatch = {}) => {
    if (!canEdit || pendingRef.current || proposalRef.current || draftRef.current.recoveryId) return
    if (nativeHistory && settings.participantCharacterIds !== undefined) {
      if (!nativePrimaryId || !Array.isArray(settings.participantCharacterIds) ||
          settings.participantCharacterIds.some(id => String(id) !== String(nativePrimaryId))) {
        setSaveError(t("common:modelSettings.conversation.nativePrimaryRequired", "Saved server chat supports only its verified primary character. Reload this chat to verify its character."))
        return
      }
      settings = { ...settings, participantCharacterIds: [nativePrimaryId] }
    }
    if (nativeHistory && !canApplyNativeSettings(settings)) { setSaveError(nativeUnsupportedHelp); return }
    setSaveError(null)
    const previous = draftRef.current
    const next = {
      settings: { ...previous.settings, ...settings }, details: { ...previous.details, ...details },
      revision: previous.revision + 1
    }
    if (settings.chatGenerationOverride) next.settings.chatGenerationOverride = {
      ...previous.settings.chatGenerationOverride, ...settings.chatGenerationOverride
    }
    draftRef.current = next
    setDraft(next)
    reportDraft(next, false)
  }
  const stageDetails = (details: ConversationDetailsPatch) => {
    if (!canEdit || mode !== "ordinary" || !serverChatId ||
      !Number.isSafeInteger(serverChatVersion) || (serverChatVersion ?? 0) <= 0) return
    if (!Object.keys(draftRef.current.details).length) metadataVersionRef.current = serverChatVersion!
    stage({}, details)
  }
  // Staged fields win over hydration; untouched generation siblings stay fresh.
  const appliedGeneration = appliedSettings?.chatGenerationOverride && typeof appliedSettings.chatGenerationOverride === "object"
    ? appliedSettings.chatGenerationOverride : appliedSettings?.generationOverrides
  const chatSettings = {
    ...appliedSettings, ...draft.settings,
    ...(draft.settings.chatGenerationOverride ? { chatGenerationOverride: { ...appliedGeneration, ...draft.settings.chatGenerationOverride } } : {})
  }
  const authorNoteDraft = chatSettings.authorNote ?? ""
  const { mode: authorNoteMode, depth: authorNoteDepth } = normalizeAuthorNotePosition(chatSettings.authorNotePosition)
  const summarySettings = appliedSettings?.summary && typeof appliedSettings.summary === "object"
    ? appliedSettings.summary as Record<string, unknown> : null
  const autoSummaryEnabledDraft = Boolean(chatSettings.autoSummaryEnabled ?? summarySettings?.enabled ?? false)
  const autoSummaryThresholdDraft = sanitizeAutoSummaryThreshold(chatSettings.autoSummaryThresholdMessages ?? summarySettings?.thresholdMessages)
  const autoSummaryWindowDraft = sanitizeAutoSummaryWindow(chatSettings.autoSummaryWindowMessages ?? summarySettings?.windowMessages, autoSummaryThresholdDraft)
  const personaMemoryModeValue = serverChatPersonaMemoryMode ?? "read_only"
  const generationOverrideSource =
    chatSettings?.chatGenerationOverride &&
    typeof chatSettings.chatGenerationOverride === "object"
      ? (chatSettings.chatGenerationOverride as Record<string, unknown>)
      : chatSettings?.generationOverrides &&
          typeof chatSettings.generationOverrides === "object"
        ? (chatSettings.generationOverrides as Record<string, unknown>)
        : null
  const generationOverrideEnabledDraft = generationOverrideSource ? generationOverrideSource.enabled !== false : false
  const generationTemperatureDraft = sanitizeGenerationFloat(generationOverrideSource?.temperature, 0, 2)
  const generationTopPDraft = sanitizeGenerationFloat(generationOverrideSource?.top_p, 0, 1)
  const generationRepetitionPenaltyDraft = sanitizeGenerationFloat(generationOverrideSource?.repetition_penalty, 0, 3)
  const generationStopsDraft = generationStopsText ?? normalizeGenerationStopList(generationOverrideSource?.stop).join("\n")
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
    () => ({ ...normalizeCharacterMemoryById(appliedSettings?.characterMemoryById), ...draft.settings.characterMemoryById }),
    [appliedSettings?.characterMemoryById, draft.settings.characterMemoryById]
  )
  const participantCharacterIdsValue = normalizeParticipantCharacterIds(
    chatSettings?.participantCharacterIds
  )
  const readOwnerKey = mode === "ordinary" && settingsTarget?.requestScope
    ? JSON.stringify([serverChatMirrorOwnerKey({ requestScope: settingsTarget.requestScope }), toChatScopeParams(settingsTarget.scope)]) : null
  const canRead = canEdit && Boolean(readOwnerKey)
  const captureRead = (querySignal: AbortSignal) => {
    const target = captureTarget()
    const signal = target.signal ? AbortSignal.any([target.signal, querySignal]) : querySignal
    const assertCurrent = () => {
      signal.throwIfAborted()
      if (!target.requestScope || target.validate_lease?.() === false || !settingsIsCurrent?.() || !isCurrent() ||
        JSON.stringify([serverChatMirrorOwnerKey({ requestScope: target.requestScope }), toChatScopeParams(target.scope)]) !== readOwnerKey)
        throw createServicePromptScopeChangedError()
    }
    assertCurrent()
    return { target, signal, assertCurrent }
  }
  const { data: characterRead, isError: isAvailableCharactersError } = useQuery<{ ownerKey: string; items: CharacterLite[] }>({
    queryKey: [...CONVERSATION_TAB_QUERY_KEYS.listCharacters, readOwnerKey],
    enabled: canRead,
    queryFn: async ({ signal }) => {
      const read = captureRead(signal)
      const list = await tldwClient.listCharacters({ limit: 200 }, { requestScope: read.target.requestScope, signal: read.signal })
      read.assertCurrent()
      if (!Array.isArray(list)) throw new Error("The character list could not be verified.")
      return { ownerKey: readOwnerKey!, items: list as CharacterLite[] }
    },
    staleTime: 60_000
  })
  const availableCharacters = canRead && characterRead?.ownerKey === readOwnerKey ? characterRead.items : []
  const { data: pinnedRead, isError: isPinnedMessagesError, isPending: isPinnedMessagesPending } = useQuery<{
    ownerKey: string; items: Array<{ id: string; role: string; content: string }>
  }>({
    queryKey: [...CONVERSATION_TAB_QUERY_KEYS.pinnedMessages(serverChatId), readOwnerKey],
    enabled: canRead && Boolean(serverChatId),
    queryFn: async ({ signal }) => {
      const read = captureRead(signal)
      if (!read.target.serverChatId) throw createServicePromptScopeChangedError()
      // ponytail: reuse the loader's 20k-message ceiling; add a server pin query if chats exceed it.
      const list = await fetchAllServerChatMessages(async ({ limit, offset }) => {
        read.assertCurrent()
        const page = await tldwClient.listChatMessages(read.target.serverChatId!, {
          include_deleted: "false", include_metadata: "true", limit, offset
        }, { requestScope: read.target.requestScope, scope: read.target.scope, signal: read.signal })
        read.assertCurrent()
        return page
      }, { signal: read.signal })
      read.assertCurrent()
      return { ownerKey: readOwnerKey!, items: list.filter(item => Boolean(item.pinned)).map(item => ({ id: item.id, role: item.role, content: String(item.content ?? "") })) }
    },
    staleTime: 10_000
  })
  const ownsPinnedRead = canRead && pinnedRead?.ownerKey === readOwnerKey
  const pinnedMessages = ownsPinnedRead ? pinnedRead.items : []

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
    label: character.name || String(character.id),
    disabled: nativeHistory && String(character.id) !== String(nativePrimaryId)
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

  const memoryNoteDraft = memoryCharacterId ? characterMemoryById[memoryCharacterId]?.note ?? "" : ""
  const stageCharacterMemory = (characterId: string, note: string) => {
    stage({ characterMemoryById: { ...draftRef.current.settings.characterMemoryById, [characterId]: { note } } })
  }
  const stageGeneration = (override: ChatGenerationOverride) => stage({ chatGenerationOverride: override })
  const stageSummaryThreshold = (value: unknown) => {
    const threshold = sanitizeAutoSummaryThreshold(value)
    stage({ autoSummaryThresholdMessages: threshold,
      autoSummaryWindowMessages: sanitizeAutoSummaryWindow(autoSummaryWindowDraft, threshold) })
  }
  const reviewRecovery = async () => {
    if (!canEdit || hasDraft || pendingRef.current || !settingsRecovery) return
    const recovery = settingsRecovery
    const revision = draftRef.current.revision
    let target: ChatSettingsTarget | undefined
    const attempt = {}
    pendingRef.current = attempt
    setPending(true)
    reportDraft(draftRef.current, true)
    try {
      target = captureTarget()
      const fresh = await getChatSettingsRecovery(target)
      if (!mounted.current || !isCurrent() || !settingsIsCurrent?.() || target.validate_lease?.() === false || draftRef.current.revision !== revision) return
      if (!fresh || fresh.operationId !== recovery.operationId || fresh.destination !== recovery.destination || JSON.stringify(fresh.patch) !== JSON.stringify(recovery.patch))
        throw new Error("The previous settings change changed elsewhere. Refresh recovery before reviewing it.")
      const next = { settings: structuredClone(fresh.patch), details: {}, revision: revision + 1, recoveryId: fresh.operationId }
      draftRef.current = next
      setDraft(next)
      setSaveError(null)
      reportDraft(next, true)
    } catch (error) {
      if (mounted.current && isCurrent() && settingsIsCurrent?.()) setSaveError(isErrorWithMessage(error) ? error.message : "The previous settings change could not be read.")
      await refreshRecovery?.(target)
    } finally {
      if (pendingRef.current === attempt) {
        pendingRef.current = null
        if (mounted.current) { setPending(false); reportDraft(draftRef.current, false) }
      }
    }
  }
  const cancelChanges = () => {
    if (!canEdit || pendingRef.current) return
    proposalRef.current = null
    metadataVersionRef.current = null
    setGenerationStopsText(null)
    setSaveError(null)
    const next = { settings: {}, details: {}, revision: draftRef.current.revision + 1 }
    draftRef.current = next
    setDraft(next)
    reportDraft(next, false)
  }
  const applyChanges = async () => {
    if (!canEdit || pendingRef.current || !hasDraft || (settingsRecovery && !draftRef.current.recoveryId && !proposalRef.current)) return
    let proposal = proposalRef.current
    try {
      if (!proposal) {
        const captured = structuredClone(draftRef.current)
        const hasDetails = Object.keys(captured.details).length > 0
        const target = mode === "ordinary" ? captureTarget() : undefined
        const expectedVersion = metadataVersionRef.current ?? undefined
        if (hasDetails && (!serverChatId || !target?.requestScope || !Number.isSafeInteger(expectedVersion) || (expectedVersion ?? 0) <= 0))
          throw new Error("Reload this saved conversation before changing its details.")
        if (captured.details.topic_label !== undefined) captured.details.topic_label = captured.details.topic_label?.trim() || null
        if (!captured.recoveryId && captured.settings.authorNote !== undefined) captured.settings.authorNote = captured.settings.authorNote.trim()
        if (!captured.recoveryId && captured.settings.characterMemoryById) {
          // Native history cannot project per-character memory; an omitted key is not a server deletion.
          if (serverChatId) throw new Error("This server cannot change per-character memory here. Use a local chat for these edits.")
          captured.settings.characterMemoryById = Object.fromEntries(Object.entries({ ...normalizeCharacterMemoryById(appliedSettings?.characterMemoryById), ...captured.settings.characterMemoryById })
            .filter(([, entry]) => entry.note.trim())
            .map(([id, entry]) => [id, { note: entry.note.trim(), updatedAt: new Date().toISOString() }]))
        }
        proposal = { draft: captured, target, expectedVersion, settingsOperationId: captured.recoveryId, settingsVerificationOnly: Boolean(nativeHistory && captured.recoveryId && !canApplyNativeSettings(captured.settings)), settingsAcknowledged: false, detailsAcknowledged: false, detailsAttempted: false, observedVersion: serverChatVersion ?? null,
          isCurrent: () => mounted.current && isCurrent() && Boolean(settingsIsCurrent?.()) &&
            target?.validate_lease?.() !== false }
        proposalRef.current = proposal
      }
      if (!proposal.isCurrent()) return
      const attempt = {}
      pendingRef.current = attempt
      setPending(true)
      setSaveError(null)
      reportDraft(draftRef.current, true)
      try {
        const details = proposal.draft.details
        if (!proposal.detailsAcknowledged && Object.keys(details).length) {
          const target = proposal.target!
          const options = { scope: target.scope, requestScope: target.requestScope, signal: target.signal }
          let reconciled: Awaited<ReturnType<typeof tldwClient.getChat>> | undefined
          if (proposal.detailsAttempted) {
            const fresh = await tldwClient.getChat(target.serverChatId!, { ...options, requiredMetadata: Object.keys(details) })
            if (!proposal.isCurrent()) return
            if (fresh.id !== target.serverChatId || !Number.isSafeInteger(fresh.version) || (fresh.version ?? 0) < proposal.expectedVersion!)
              throw new Error("The conversation metadata could not be verified. Reload before saving.")
            if (fresh.version! > proposal.expectedVersion! && Object.entries(details).every(([key, value]) => fresh[key as keyof typeof fresh] === value)) reconciled = fresh
            else if (fresh.version !== proposal.expectedVersion)
              throw new Error("This conversation changed elsewhere. Your edits remain staged; cancel and reload before applying them.")
          }
          proposal.detailsAttempted = true
          const updated = reconciled ?? await tldwClient.updateChat(target.serverChatId!, details, {
            ...options, expectedVersion: proposal.expectedVersion
          })
          if (!proposal.isCurrent()) return
          if (updated.id !== target.serverChatId || !Number.isSafeInteger(updated.version) || updated.version! <= proposal.expectedVersion! ||
            Object.entries(details).some(([key, value]) => !Object.prototype.hasOwnProperty.call(updated, key) || updated[key as keyof typeof updated] !== value))
            throw new Error("The server did not confirm these conversation changes. Reload before retrying.")
          proposal.detailsAcknowledged = true
          proposal.observedVersion = updated.version!
          const canPublish = () => {
            const live = useStoreMessageOption.getState()
            return proposal!.isCurrent() && live.serverChatId === target.serverChatId && (live.serverChatVersion ?? 0) <= updated.version!
          }
          if (canPublish() && details.state !== undefined) onStateChange(details.state)
          if (canPublish() && details.topic_label !== undefined) onTopicChange(details.topic_label)
          if (canPublish()) onVersionChange(updated.version!)
          void queryClient.invalidateQueries({ queryKey: ["serverChatHistory"] })
        }
        if (!proposal.isCurrent()) return
        // Both writes advance conversations.version. Preserve the original metadata
        // CAS by applying it first, then refresh only for the next user edit.
        if (!proposal.settingsAcknowledged && Object.keys(proposal.draft.settings).length) {
          await (proposal.settingsVerificationOnly
            ? updateSettings(proposal.draft.settings, proposal.settingsOperationId, true)
            : proposal.settingsOperationId ? updateSettings(proposal.draft.settings, proposal.settingsOperationId)
            : updateSettings(proposal.draft.settings))
          proposal.settingsAcknowledged = true
        }
        if (!proposal.isCurrent()) return
        if (proposal.settingsAcknowledged && proposal.target?.serverChatId) {
          const target = proposal.target
          const fresh = await tldwClient.getChat(target.serverChatId!, {
            scope: target.scope, requestScope: target.requestScope, signal: target.signal, requiredMetadata: ["state", "topic_label"]
          })
          if (!proposal.isCurrent()) return
          if (fresh.id !== target.serverChatId || !Number.isSafeInteger(fresh.version) || (fresh.version ?? 0) <= 0 || fresh.version! < (proposal.observedVersion ?? 0))
            throw new Error("Settings were applied, but the conversation version could not be verified. Retry to refresh it.")
          const canPublish = () => {
            const live = useStoreMessageOption.getState()
            return proposal!.isCurrent() && live.serverChatId === target.serverChatId && (live.serverChatVersion ?? 0) <= fresh.version!
          }
          if (canPublish() && fresh.state && useStoreMessageOption.getState().serverChatState !== fresh.state) onStateChange(normalizeConversationState(fresh.state))
          if (canPublish() && fresh.topic_label !== undefined && useStoreMessageOption.getState().serverChatTopic !== fresh.topic_label) onTopicChange(fresh.topic_label)
          if (canPublish() && (useStoreMessageOption.getState().serverChatVersion ?? 0) < fresh.version!) onVersionChange(fresh.version!)
        }
        if (!proposal.isCurrent()) return
        proposalRef.current = null
        metadataVersionRef.current = null
        setGenerationStopsText(null)
        const next = { settings: {}, details: {}, revision: draftRef.current.revision + 1 }
        draftRef.current = next
        setDraft(next)
        reportDraft(next, false)
      } catch (error) {
        if (error && typeof error === "object" && "settingsOperationId" in error && typeof error.settingsOperationId === "string") proposal.settingsOperationId = error.settingsOperationId
        if (proposal.isCurrent()) setSaveError(isErrorWithMessage(error)
          ? error.message === "chat_settings_recovery_requires_compatible_path" ? "The server has not confirmed this change. It is retained; this chat cannot resend these settings. Review the saved setup or start a compatible chat."
          : error.message === "chat_settings_recovery_changed" ? "That previous change was verified or replaced elsewhere. Cancel this proposal and refresh before editing." : error.message
          : "Conversation changes could not be applied.")
      } finally {
        if (pendingRef.current === attempt) {
          pendingRef.current = null
          if (mounted.current) { setPending(false); reportDraft(draftRef.current, false) }
        }
      }
    } catch (error) {
      if (canEdit) setSaveError(isErrorWithMessage(error) ? error.message : "Conversation changes could not be applied.")
    }
  }
  const previewSettingsFingerprint = JSON.stringify(appliedSettings)

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
        help={t("common:modelSettings.conversation.modelDefaultPromptHelp", {
          defaultValue:
            "Saved with model settings. Applies to chats using this model; conversation Apply does not save it."
        })}
        label={t("common:modelSettings.conversation.modelDefaultPromptLabel", {
          defaultValue: "Model default system prompt"
        })}>
        <div className="space-y-1">
          <Form.Item name="systemPrompt" noStyle>
            <Input.TextArea
              aria-label={t("common:modelSettings.conversation.modelDefaultPromptLabel", "Model default system prompt")}
              rows={useDrawer ? 4 : 6}
              placeholder={t(
                "common:modelSettings.form.systemPrompt.placeholder",
                { defaultValue: "Enter System Prompt" }
              )}
              onChange={(e) => onSystemPromptChange(e.target.value)}
            />
          </Form.Item>
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
          <p className="mb-2 text-xs text-text-muted">
            {t("playground:composer.attachments.immediateRemoval", "Attachment removal takes effect immediately.")}
          </p>
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
                  type="button"
                  aria-label={`${t("common:removeFile", "Remove file")}: ${file.filename}`}
                  onClick={() => onRemoveFile(file.id)}
                  className="rounded p-1 text-danger hover:bg-danger/10 hover:text-danger">
                  <X className="h-4 w-4" />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="sticky top-0 z-10 space-y-2 border-y border-border bg-surface py-3">
        <p className="text-xs text-text-muted">{t("playground:composer.conversationStagingHelp", "Conversation changes stay in this panel until you apply them.")}</p>
        {settingsRecoveryError && <p role="alert">{t("playground:composer.conversationRecoveryReadError", "The previous settings change could not be verified on this device. Retry the read before editing.")}
          <Button htmlType="button" onClick={() => void refreshRecovery?.()}>{t("playground:composer.retryRecoveryRead", "Retry recovery read")}</Button>
        </p>}
        {settingsRecovery && <div className="space-y-2">
          <p role="status">{recoveryUnsupported ? t("common:modelSettings.conversation.nativeRecoveryHelp", "A previous change uses behavior unavailable in saved server chat. Review it to verify whether the server accepted it. Verification does not resend it.") : t("playground:composer.conversationRecoveryHelp", "A previous settings change still needs verification. Review it, then Apply to reconcile it before applying new edits. Cancel does not undo a change already accepted by the server.")}</p>
          <Button className="min-h-[40px]" htmlType="button" disabled={!canEdit || hasDraft || pending} onClick={() => void reviewRecovery()}>{t("playground:composer.reviewRecovery", "Review previous settings change")}</Button>
        </div>}
        {verifyRecoveryOnly && <p role="status">{t("common:modelSettings.conversation.nativeRecoveryVerification", "This previous change is unavailable in saved server chat. Verify checks server status without resending it.")}</p>}
        {draft.recoveryId && <details><summary>{t("playground:composer.recoveredSettings", "Recovered settings to verify")}</summary><pre className="max-h-40 overflow-auto whitespace-pre-wrap break-all text-xs">{JSON.stringify(draft.settings, null, 2).slice(0, 4000)}</pre></details>}
      {!canEdit && <p role="status">{t("playground:composer.conversationUnready", "Conversation settings are not ready. Check the selected chat and account before editing.")}</p>}
      {saveError && <p role="alert">{saveError} {verifyRecoveryOnly ? t("common:modelSettings.conversation.nativeRecoveryRetained", "Your exact previous change is retained. Verify reads its server status; it does not resend this unsupported change.") : t("playground:composer.conversationRetryHelp", "Your proposal is retained. Retry applies only unconfirmed changes; Cancel discards unapplied edits.")}</p>}
      <div className="flex flex-wrap gap-2">
        <Button className="min-h-[40px]" htmlType="button" type="primary" aria-label={applyLabel} aria-busy={pending} loading={pending} disabled={!canEdit || !hasDraft || pending || Boolean(settingsRecovery && !draft.recoveryId && !proposalRef.current)} onClick={() => void applyChanges()}>
          {applyLabel}
        </Button>
        <Button className="min-h-[40px]" htmlType="button" disabled={!canEdit || !hasDraft || pending} onClick={cancelChanges}>
          {t("playground:composer.cancelConversation", "Cancel conversation changes")}
        </Button>
      </div>
      </div>

      <fieldset className="min-w-0 space-y-4 border-0 p-0" disabled={!canEdit || pending || Boolean(proposalRef.current) || Boolean(draft.recoveryId)}>
      <Form.Item
        label={t("playground:composer.conversationStatus", "Conversation status")}
        help={t(
          "playground:composer.stateHelp",
          'Default state is "in-progress." Update it as the conversation progresses.'
        )}>
        <Select
          aria-label={t("playground:composer.conversationStatus", "Conversation status")}
          data-testid="conversation-state-select"
          value={draft.details.state ?? serverChatState ?? "in-progress"}
          disabled={!serverChatId || mode !== "ordinary" || !Number.isSafeInteger(serverChatVersion) || (serverChatVersion ?? 0) <= 0}
          options={conversationStateOptions}
          onChange={(value) => stageDetails({ state: normalizeConversationState(value) })}
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
          <output data-testid="persona-memory-mode-value">
            {personaMemoryModeValue === "read_write"
              ? t("playground:composer.personaMemoryMode.readWrite", "Read + write")
              : t("playground:composer.personaMemoryMode.readOnly", "Read only")}
          </output>
          <p className="mt-1 text-xs text-text-muted">
            {t("playground:composer.personaMemoryMode.unsupportedEdit", "This server cannot change persona memory mode here.")}
          </p>
        </Form.Item>
      ) : null}

      <Form.Item
        label={t("playground:composer.topicPlaceholder", "Conversation tag")}
        help={t(
          "playground:composer.persistence.topicHelp",
          "Optional label for this chat; saved to the server when available."
        )}>
        <Input
          aria-label={t("playground:composer.topicPlaceholder", "Conversation tag")}
          value={draft.details.topic_label !== undefined ? draft.details.topic_label ?? "" : serverChatTopic ?? ""}
          disabled={!serverChatId || mode !== "ordinary" || !Number.isSafeInteger(serverChatVersion) || (serverChatVersion ?? 0) <= 0}
          onChange={(e) => stageDetails({ topic_label: e.target.value || null })}
          placeholder={t(
            "playground:composer.topicPlaceholder",
            "Conversation tag (optional)"
          )}
        />
      </Form.Item>

      {nativeHistory && <p className="text-sm text-text-muted">{nativeUnsupportedHelp}</p>}
      <Form.Item
        label={t("playground:composer.authorNote.label", "Author note")}
        help={nativeHistory ? undefined : t(
          "playground:composer.authorNote.help",
          "Optional note injected into character prompts for this chat."
        )}
      >
        <Input.TextArea
          aria-label={t("playground:composer.authorNote.label", "Author note")}
          rows={3}
          showCount
          maxLength={1200}
          value={authorNoteDraft}
          disabled={nativeHistory}
          onChange={(e) => stage({ authorNote: e.target.value })}
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
        help={nativeHistory ? undefined : t(
          "playground:composer.authorNotePosition.help",
          "Choose whether the note is inserted before system instructions or at conversation depth N."
        )}
      >
        <div className="space-y-2">
          <Select
            aria-label={t("playground:composer.authorNotePosition.label", "Author note insertion")}
            value={authorNoteMode}
            disabled={nativeHistory}
            options={authorNotePositionOptions}
            onChange={(value) => {
              const nextMode = value === "depth" ? "depth" : "before_system"
              stage({ authorNotePosition: nextMode === "depth" ? { mode: "depth", depth: authorNoteDepth } : "before_system" })
            }}
          />
          {authorNoteMode === "depth" && (
            <div className="flex min-w-0 flex-col gap-1">
              <label htmlFor={`${numericFieldId}-author-note-depth`}>
                {t(
                  "playground:composer.authorNotePosition.depthLabel",
                  "Depth"
                )}
              </label>
              <InputNumber
                id={`${numericFieldId}-author-note-depth`}
                style={{ width: "100%" }}
                min={0}
                step={1}
                value={authorNoteDepth}
                disabled={nativeHistory}
                onChange={(value) => {
                  stage({ authorNotePosition: { mode: "depth", depth: sanitizeDepth(value) } })
                }}
              />
            </div>
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
          aria-label={t("playground:composer.greetingScope.label", "Greeting scope")}
          value={greetingScopeValue}
          options={greetingScopeOptions}
          onChange={(value) => {
            const next = value === "character" ? "character" : "chat"
            stage({ greetingScope: next })
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
          aria-label={t("playground:composer.presetScope.label", "Preset scope")}
          value={presetScopeValue}
          options={presetScopeOptions}
          onChange={(value) => {
            const next = value === "chat" ? "chat" : "character"
            stage({ presetScope: next })
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
            aria-label={t("playground:composer.chatPresetOverride.label", "Chat preset override")}
            value={chatPresetOverrideValue || DEFAULT_CHARACTER_PROMPT_PRESET}
            options={chatPresetOptions}
            onChange={(value) => {
              const next = normalizeChatPresetOverrideId(value)
              stage({ chatPresetOverrideId: next || DEFAULT_CHARACTER_PROMPT_PRESET })
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
            aria-label={t("playground:composer.chatGenerationOverride.label", "Chat generation override")}
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
              stageGeneration({ enabled })
            }}
          />
          <div className="grid grid-cols-1 gap-2 sm:grid-cols-3">
            <div className="flex min-w-0 flex-col gap-1">
              <label htmlFor={`${numericFieldId}-temperature`}>
                {t(
                  "playground:composer.chatGenerationOverride.temperature",
                  "Temp"
                )}
              </label>
              <InputNumber
                id={`${numericFieldId}-temperature`}
                style={{ width: "100%" }}
                min={0}
                max={2}
                step={0.01}
                precision={2}
                disabled={!generationOverrideEnabledDraft}
                value={generationTemperatureDraft}
                onChange={(value) => {
                  stageGeneration({ temperature: sanitizeGenerationFloat(value, 0, 2) })
                }}
              />
            </div>
            <div className="flex min-w-0 flex-col gap-1">
              <label htmlFor={`${numericFieldId}-top-p`}>
                {t(
                  "playground:composer.chatGenerationOverride.topP",
                  "Top-p"
                )}
              </label>
              <InputNumber
                id={`${numericFieldId}-top-p`}
                style={{ width: "100%" }}
                min={0}
                max={1}
                step={0.01}
                precision={2}
                disabled={!generationOverrideEnabledDraft}
                value={generationTopPDraft}
                onChange={(value) => {
                  stageGeneration({ top_p: sanitizeGenerationFloat(value, 0, 1) })
                }}
              />
            </div>
            <div className="flex min-w-0 flex-col gap-1">
              <label htmlFor={`${numericFieldId}-repetition-penalty`}>
                {t(
                  "playground:composer.chatGenerationOverride.repetitionPenalty",
                  "Rep pen"
                )}
              </label>
              <InputNumber
                id={`${numericFieldId}-repetition-penalty`}
                style={{ width: "100%" }}
                min={0}
                max={3}
                step={0.01}
                precision={2}
                disabled={!generationOverrideEnabledDraft}
                value={generationRepetitionPenaltyDraft}
                onChange={(value) => {
                  stageGeneration({ repetition_penalty: sanitizeGenerationFloat(value, 0, 3) })
                }}
              />
            </div>
          </div>
          <Input.TextArea
            aria-label={t("playground:composer.chatGenerationOverride.stopLabel", "Stop sequences")}
            rows={2}
            disabled={!generationOverrideEnabledDraft}
            value={generationStopsDraft}
            onChange={(event) => {
              if (!canEdit || pendingRef.current || proposalRef.current || draftRef.current.recoveryId) return
              setGenerationStopsText(event.target.value)
              stageGeneration({ stop: parseGenerationStopListFromText(event.target.value) })
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
        help={nativeHistory ? undefined : t("playground:composer.memoryScope.help", {
          defaultValue:
            "Choose whether memory is shared, per-character, or both."
        })}
      >
        <Select
          aria-label={t("playground:composer.memoryScope.label", "Memory scope")}
          value={memoryScopeValue}
          disabled={nativeHistory}
          options={memoryScopeOptions}
          onChange={(value) => {
            const normalized = normalizeMemoryScope(value)
            stage({ memoryScope: normalized })
          }}
        />
      </Form.Item>

      <Form.Item
        label={t("playground:composer.turnTakingMode.label", "Turn-taking mode")}
        help={nativeHistory ? t("common:modelSettings.conversation.nativeSingleSpeaker", "Saved server chat uses its primary character as a single speaker.") : t("playground:composer.turnTakingMode.help", {
          defaultValue:
            "Choose whether responses use one active speaker or rotate across selected participants."
        })}
      >
        <Select
          aria-label={t("playground:composer.turnTakingMode.label", "Turn-taking mode")}
          value={turnTakingModeValue}
          disabled={nativeHistory}
          options={turnTakingModeOptions}
          onChange={(value) => {
            const normalized = normalizeTurnTakingMode(value)
            stage({ turnTakingMode: normalized })
          }}
        />
      </Form.Item>

      <Form.Item
        label={t("playground:composer.participants.label", "Participants")}
        help={nativeHistory ? t("common:modelSettings.conversation.nativeParticipantHelp", "Only the chat’s primary character is supported. Reset keeps this character selected.") : t("playground:composer.participants.help", {
          defaultValue:
            "Select additional characters for multi-character chats. The chat's primary character remains included."
        })}
      >
        <div className="space-y-1">
          <Select
            aria-label={t("playground:composer.participants.label", "Participants")}
            mode="multiple"
            allowClear={!nativeHistory}
            disabled={nativeHistory}
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
              stage({ participantCharacterIds: normalizeParticipantCharacterIds(nextValues) })
            }}
          />
          {nativeHistory && <Button htmlType="button" className="min-h-10" disabled={!nativePrimaryId}
            onClick={() => stage({ participantCharacterIds: [], turnTakingMode: "single" })}>
            {t("common:modelSettings.conversation.resetNativePrimary", "Reset to primary character")}
          </Button>}
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
        help={nativeHistory ? undefined : t("playground:composer.directedReply.help", {
          defaultValue:
            "Optional: choose which participant should generate the next response."
        })}
      >
        <Select
          aria-label={t("playground:composer.directedReply.label", "Directed responder")}
          allowClear
          showSearch
          value={directedCharacterIdValue ?? undefined}
          disabled={nativeHistory}
          options={participantCharacterOptions}
          placeholder={t("playground:composer.directedReply.placeholder", {
            defaultValue: "Automatic by turn-taking"
          })}
          onChange={(value) => {
            const normalized = value != null ? String(value) : null
            stage({ directedCharacterId: normalized !== null && /^\d+$/.test(normalized) ? Number.parseInt(normalized, 10) : null })
          }}
        />
      </Form.Item>

      {memoryScopeValue !== "shared" && (
        <Form.Item
          label={t(
            "playground:composer.characterMemory.label",
            "Per-character memory"
          )}
          help={nativeHistory ? t("common:modelSettings.conversation.nativeMemoryReference", "Saved memory notes are shown for reference; this server chat cannot apply them.") : t("playground:composer.characterMemory.help", {
            defaultValue:
              "Edit memory notes for individual characters used when memory scope includes per-character injection."
          })}
        >
          <div className="space-y-2">
            <Select
              aria-label={t("playground:composer.characterMemory.targetLabel", "Memory character")}
              showSearch
              value={memoryCharacterId ?? undefined}
              options={participantCharacterOptions}
              placeholder={t("playground:composer.characterMemory.target", {
                defaultValue: "Select character"
              })}
              onChange={(value) => {
                const normalized = value ? String(value) : null
                setMemoryCharacterId(normalized)

              }}
            />
            <Input.TextArea
              aria-label={t("playground:composer.characterMemory.noteLabel", "Memory note")}
              rows={3}
              showCount
              maxLength={600}
              value={memoryNoteDraft}
              disabled={!memoryCharacterId || Boolean(serverChatId)}
              onChange={(e) => { if (memoryCharacterId) stageCharacterMemory(memoryCharacterId, e.target.value) }}
              placeholder={t(
                "playground:composer.characterMemory.placeholder",
                "Character-specific memory note"
              )}
            />
            {serverChatId && <p className="text-xs text-text-muted">{t("playground:composer.characterMemory.localOnly", "Per-character memory editing is available for local chats.")}</p>}
          </div>
        </Form.Item>
      )}

      <Form.Item
        label={t("playground:composer.autoSummary.label", "Auto summary")}
        help={nativeHistory ? t("common:modelSettings.conversation.nativeSummaryReference", "Saved summary settings are shown for reference; this server chat does not compress prompt history.") : t("playground:composer.autoSummary.help", {
          defaultValue:
            "Compress older unpinned messages once the threshold is reached."
        })}
      >
        <div className="space-y-2">
          <Select
            aria-label={t("playground:composer.autoSummary.label", "Auto summary")}
            value={autoSummaryEnabledDraft ? "on" : "off"}
            disabled={nativeHistory}
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
              stage({ autoSummaryEnabled: enabled })
            }}
          />
          <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
            <div className="flex min-w-0 flex-col gap-1">
              <label htmlFor={`${numericFieldId}-summary-threshold`}>
                {t(
                  "playground:composer.autoSummary.threshold",
                  "Threshold"
                )}
              </label>
              <InputNumber
                id={`${numericFieldId}-summary-threshold`}
                style={{ width: "100%" }}
                min={2}
                max={5000}
                step={1}
                value={autoSummaryThresholdDraft}
                disabled={nativeHistory || !autoSummaryEnabledDraft}
                onChange={(value) => {
                  stageSummaryThreshold(value)
                }}
              />
            </div>
            <div className="flex min-w-0 flex-col gap-1">
              <label htmlFor={`${numericFieldId}-summary-window`}>
                {t(
                  "playground:composer.autoSummary.window",
                  "Recent window"
                )}
              </label>
              <InputNumber
                id={`${numericFieldId}-summary-window`}
                style={{ width: "100%" }}
                min={1}
                max={Math.max(1, autoSummaryThresholdDraft - 1)}
                step={1}
                value={autoSummaryWindowDraft}
                disabled={nativeHistory || !autoSummaryEnabledDraft}
                onChange={(value) => {
                  stage({ autoSummaryWindowMessages: sanitizeAutoSummaryWindow(value, autoSummaryThresholdDraft) })
                }}
              />
            </div>
          </div>
        </div>
      </Form.Item>

      <Form.Item
        label={t("playground:composer.pinnedMessages.label", "Pinned messages")}
        help={nativeHistory ? t("common:modelSettings.conversation.nativePinReference", "These are saved message markers; this server chat does not use them for prompt compression.") : t("playground:composer.pinnedMessages.help", {
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
          ) : !ownsPinnedRead ? (
            <div role="status" className="text-xs text-text-muted">{canRead && isPinnedMessagesPending
              ? t("playground:composer.pinnedMessages.loading", "Loading pinned messages...")
              : t("playground:composer.pinnedMessages.unverified", "Pinned messages have not been verified for this chat.")}</div>
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
            help={nativeHistory ? t("common:modelSettings.conversation.nativeSavedSummary", "This is a saved summary; this server chat does not apply it to prompt history.") : t(
              "playground:composer.autoSummary.currentSummaryHelp",
              "Persisted server-side summary used for prompt compression."
            )}
          >
            <Input.TextArea
              aria-label={t("playground:composer.autoSummary.currentSummary", "Current summary")}
              readOnly
              rows={4}
              value={summarySettings.content}
            />
          </Form.Item>
        )}

      </fieldset>
      <p className="text-xs text-text-muted">{t("playground:composer.appliedPreviewHelp", "Prompt and lore previews show applied conversation settings.")}</p>
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
