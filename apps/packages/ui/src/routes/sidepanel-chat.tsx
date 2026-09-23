import { useSidepanelChatOwner, type SidepanelChatOwner } from "@/hooks/useSidepanelChatOwner"
import { PageAssistLoader } from "@/components/Common/PageAssistLoader"
import {
  formatToChatHistory,
  formatToMessage,
  getTitleById,
  getRecentChatFromCopilot,
  generateID,
  getFullChatData
} from "@/db/dexie/helpers"
import { getDesignSystemState } from "@/design-system"
import useBackgroundMessage from "@/hooks/useBackgroundMessage"
import { useMigration } from "@/hooks/useMigration"
import { useSmartScroll } from "@/hooks/useSmartScroll"
import {
  useChatShortcuts,
  useSidebarShortcuts,
  useChatModeShortcuts,
  useWebSearchShortcuts
} from "@/hooks/keyboard/useKeyboardShortcuts"
import { useConnectionActions } from "@/hooks/useConnectionState"
import { useServerOnline } from "@/hooks/useServerOnline"
import { useAntdNotification } from "@/hooks/useAntdNotification"
import { useCharacterGreeting } from "@/hooks/useCharacterGreeting"
import { useTTS } from "@/hooks/useTTS"
import { copilotResumeLastChat } from "@/services/app"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import type { ServerChatMessage as ApiServerChatMessage } from "@/services/tldw/TldwApiClient"
import { createSafeStorage } from "@/utils/safe-storage"
import { requestQuickIngestOpen } from "@/utils/quick-ingest-open"
import {
  CHAT_BACKGROUND_IMAGE_SETTING,
  CHAT_CHARACTER_IMAGE_OPACITY_SETTING,
  CHAT_MESSAGE_OPACITY_SETTING,
  CHAT_WINDOW_OPACITY_SETTING,
  resolveOpacityAlpha
} from "@/services/settings/ui-settings"
import { useSetting } from "@/hooks/useSetting"
import { useStorage } from "@plasmohq/storage/hook"
import { ChevronDown, ExternalLink } from "lucide-react"
import React, { lazy, Suspense } from "react"
import { useTranslation } from "react-i18next"
import { SidePanelBody } from "~/components/Sidepanel/Chat/body"
import { SidepanelForm } from "~/components/Sidepanel/Chat/form"
import { SidepanelHeaderSimple } from "~/components/Sidepanel/Chat/SidepanelHeaderSimple"
import { ConnectionBanner } from "~/components/Sidepanel/Chat/ConnectionBanner"
import { useMessage } from "~/hooks/useMessage"
import { useSelectedCharacter } from "@/hooks/useSelectedCharacter"
import { useSelectedAssistant } from "@/hooks/useSelectedAssistant"
import { useSidepanelChatTabsStore } from "@/store/sidepanel-chat-tabs"
import { DEFAULT_CHAT_SETTINGS } from "@/types/chat-settings"
import type { Character } from "@/types/character"
import type {
  ChatModelSettingsSnapshot,
  SidepanelChatSnapshot,
  SidepanelChatTab
} from "@/store/sidepanel-chat-tabs"
import type { HistoryInfo } from "@/db/dexie/types"
import { useStoreChatModelSettings } from "@/store/model"
import { useStoreMessageOption } from "@/store/option"
import { useUiModeStore } from "@/store/ui-mode"
import { useArtifactsStore } from "@/store/artifacts"
import { normalizeConversationState } from "@/utils/conversation-state"
import { normalizeChatRole } from "@/utils/normalize-chat-role"
import { normalizeMessageMetadataExtra } from "@/utils/dynamic-ui"
import { restoreQueuedRequests } from "@/utils/chat-request-queue"
import { useFlashcardsGenerateTransfer, useStudyPackTransfer } from "@/hooks/useFlashcardsGenerateTransfer"
import { flashcardsHandoffAuthority, loadFlashcardsTransferSnapshot } from "@/services/tldw/flashcards-generate-transfer"
import type { ServicePromptSnapshot } from "@/services/service-prompts"
import { linkServerChatMirror, reconcileServerChatMirror } from "@/db/dexie/server-chat-mirror"
import {
  buildCapturedNoteContent,
  CAPTURED_NOTE_KEYWORD
} from "@/services/notes-capture"
import {
  clearPendingWebClipAnalyzeRequest,
  readPendingWebClipAnalyzeRequest
} from "@/services/web-clipper/enrichment"
import {
  collectWebClipAnalyzeMessageIds,
  hasSubmittedWebClipAnalyzeMessage,
  WEB_CLIPPER_ANALYZE_MESSAGE_TYPE
} from "@/services/web-clipper/analyze-handoff"
import { CommandPaletteHost } from "@/components/Common/CommandPaletteHost"
import type { CommandItem } from "@/components/Common/CommandPalette"
import type { ServerChatHistoryItem } from "@/hooks/useServerChatHistory"
import { buildSidepanelFullAppChatPath } from "@/utils/sidepanel-full-app-route"
import {
  OPEN_HISTORY_EVENT,
  TIMELINE_ACTION_EVENT,
  type OpenHistoryDetail,
  type TimelineActionDetail
} from "@/utils/timeline-actions"
import {
  getTabsStorageKey,
  readSidepanelTabs,
  type SidepanelTabsState,
  readSidepanelRuntimeTabId
} from "./sidepanel-chat-resume"

// Lazy-load Timeline to reduce initial bundle size (~1.2MB cytoscape)
const TimelineModal = lazy(() =>
  import("@/components/Timeline").then((m) => ({ default: m.TimelineModal }))
)
const ArtifactsPanel = lazy(() =>
  import("@/components/Sidepanel/Chat/ArtifactsPanel").then((m) => ({
    default: m.ArtifactsPanel
  }))
)
const LazySidepanelChatSidebar = lazy(() =>
  import("~/components/Sidepanel/Chat/Sidebar").then((module) => ({
    default: module.SidepanelChatSidebar
  }))
)
const LazyNoteQuickSaveModal = lazy(
  () => import("~/components/Sidepanel/Notes/NoteQuickSaveModal")
)

import type { ChatHistory, Message as ChatMessage } from "~/store/option"

type ServerChatMessageInput = Omit<ApiServerChatMessage, "id"> & {
  id: string | number
}

type IngestCardStatus =
  | "queued"
  | "running"
  | "completed"
  | "failed"
  | "cancelled"
  | "auth_required"

type IngestCardState = {
  funnelId: string
  status: IngestCardStatus
  url?: string
  progressPercent?: number
  progressMessage?: string
  error?: string
  mediaId?: number
  jobIds: number[]
  canCancel: boolean
  canRetry: boolean
  starterQuestions: string[]
  timestampSeconds?: number
}

const formatSecondsAsClock = (seconds: number): string => {
  const safe = Math.max(0, Math.trunc(seconds))
  const hrs = Math.floor(safe / 3600)
  const mins = Math.floor((safe % 3600) / 60)
  const secs = safe % 60
  if (hrs > 0) {
    return `${hrs}:${String(mins).padStart(2, "0")}:${String(secs).padStart(
      2,
      "0"
    )}`
  }
  return `${mins}:${String(secs).padStart(2, "0")}`
}

const buildStarterQuestions = (payload: {
  url?: string
  timestampSeconds?: number
}): string[] => {
  const ts =
    typeof payload.timestampSeconds === "number" && payload.timestampSeconds >= 0
      ? Math.trunc(payload.timestampSeconds)
      : null
  const around = ts != null ? formatSecondsAsClock(ts) : null
  if (around) {
    return [
      `What happens around ${around} in this video, and why is it important?`,
      `What leads up to ${around}, and what happens immediately after?`,
      "Give me the top 5 takeaways from this video with supporting evidence."
    ]
  }
  return [
    "Give me a concise summary of this media.",
    "What are the main claims or takeaways?",
    "What should I verify or fact-check from this content?"
  ]
}

const normalizeServerChatMessageId = (
  id: ServerChatMessageInput["id"]
) => (typeof id === "number" ? String(id) : id)

const mapServerChatMessages = (
  list: ServerChatMessageInput[],
  userDisplayName?: string
) => {
  const resolvedUserName = userDisplayName?.trim() || "You"
  const normalizeRole = (role: string) => normalizeChatRole(role)
  const history = list.map((m) => ({
    role: normalizeRole(m.role),
    content: m.content
  }))
  const mappedMessages = list.map((m) => {
    const meta = m as Record<string, unknown>
    const createdAt = Date.parse(m.created_at)
    const normalizedRole = normalizeRole(m.role)
    const normalizedId = normalizeServerChatMessageId(m.id)
    const metadataExtra = normalizeMessageMetadataExtra(m.metadata_extra)
    const speakerCharacterIdRaw = metadataExtra?.speaker_character_id
    const speakerCharacterId =
      typeof speakerCharacterIdRaw === "number" &&
      Number.isFinite(speakerCharacterIdRaw)
        ? speakerCharacterIdRaw
        : typeof speakerCharacterIdRaw === "string" &&
            speakerCharacterIdRaw.trim().length > 0 &&
            Number.isFinite(Number(speakerCharacterIdRaw))
          ? Number(speakerCharacterIdRaw)
          : null
    const moodConfidenceRaw = metadataExtra?.mood_confidence
    const moodConfidence =
      typeof moodConfidenceRaw === "number" && Number.isFinite(moodConfidenceRaw)
        ? moodConfidenceRaw
        : typeof moodConfidenceRaw === "string" &&
            moodConfidenceRaw.trim().length > 0 &&
            Number.isFinite(Number(moodConfidenceRaw))
          ? Number(moodConfidenceRaw)
          : null
    return {
      createdAt: Number.isNaN(createdAt) ? undefined : createdAt,
      isBot: normalizedRole === "assistant",
      role: normalizedRole,
      name:
        normalizedRole === "assistant"
          ? "Assistant"
          : normalizedRole === "system"
            ? "System"
            : resolvedUserName,
      message: m.content,
      sources: [],
      images: [],
      id: normalizedId,
      serverMessageId: normalizedId,
      serverMessageVersion: m.version,
      parentMessageId:
        (meta?.parent_message_id as string | null | undefined) ??
        (meta?.parentMessageId as string | null | undefined) ??
        null,
      messageType:
        (meta?.message_type as string | undefined) ??
        (meta?.messageType as string | undefined),
      clusterId:
        (meta?.cluster_id as string | undefined) ??
        (meta?.clusterId as string | undefined),
      modelId:
        (meta?.model_id as string | undefined) ??
        (meta?.modelId as string | undefined),
      modelName:
        (meta?.model_name as string | undefined) ??
        (meta?.modelName as string | undefined) ??
        "Assistant",
      modelImage:
        (meta?.model_image as string | undefined) ??
        (meta?.modelImage as string | undefined),
      metadataExtra,
      speakerCharacterId,
      speakerCharacterName:
        typeof metadataExtra?.speaker_character_name === "string"
          ? metadataExtra.speaker_character_name
          : undefined,
      moodLabel:
        typeof metadataExtra?.mood_label === "string"
          ? metadataExtra.mood_label
          : undefined,
      moodConfidence,
      moodTopic:
        typeof metadataExtra?.mood_topic === "string"
          ? metadataExtra.mood_topic
          : null
    }
  })

  return { history, mappedMessages }
}

const deriveNoteTitle = (
  content: string,
  pageTitle?: string,
  url?: string
): string => {
  const cleanedTitle = (pageTitle || "").trim()
  if (cleanedTitle) return cleanedTitle
  const normalized = (content || "").trim().replace(/\s+/g, " ")
  if (normalized) {
    const words = normalized.split(" ").slice(0, 8).join(" ")
    return words + (normalized.length > words.length ? "..." : "")
  }
  if (url) {
    try {
      return new URL(url).hostname
    } catch {
      return url
    }
  }
  return ""
}

const DEFAULT_COMPOSER_HEIGHT = 160

const MODEL_SETTINGS_KEYS = [
  "f16KV",
  "frequencyPenalty",
  "keepAlive",
  "logitsAll",
  "mirostat",
  "mirostatEta",
  "mirostatTau",
  "numBatch",
  "numCtx",
  "numGpu",
  "numGqa",
  "numKeep",
  "numPredict",
  "numThread",
  "penalizeNewline",
  "presencePenalty",
  "repeatLastN",
  "repeatPenalty",
  "ropeFrequencyBase",
  "ropeFrequencyScale",
  "temperature",
  "tfsZ",
  "topK",
  "topP",
  "typicalP",
  "useMLock",
  "useMMap",
  "vocabOnly",
  "seed",
  "minP",
  "systemPrompt",
  "useMlock",
  "reasoningEffort",
  "ocrLanguage",
  "historyMessageLimit",
  "historyMessageOrder",
  "slashCommandInjectionMode",
  "apiProvider",
  "extraHeaders",
  "extraBody",
  "llamaThinkingBudgetTokens",
  "llamaGrammarMode",
  "llamaGrammarId",
  "llamaGrammarInline",
  "llamaGrammarOverride"
] as const

type ModelSettingsKey = (typeof MODEL_SETTINGS_KEYS)[number]
type ChatModelSettingsState = ReturnType<typeof useStoreChatModelSettings>

const pickChatModelSettings = (
  state: ChatModelSettingsState
): ChatModelSettingsSnapshot => {
  const snapshot = {} as Record<
    ModelSettingsKey,
    ChatModelSettingsSnapshot[ModelSettingsKey]
  >
  MODEL_SETTINGS_KEYS.forEach((key) => {
    snapshot[key] = state[key] as ChatModelSettingsSnapshot[ModelSettingsKey]
  })
  return snapshot as ChatModelSettingsSnapshot
}

const applyChatModelSettingsSnapshot = (
  snapshot: ChatModelSettingsSnapshot | undefined
) => {
  const store = useStoreChatModelSettings.getState()
  store.reset()
  if (!snapshot) return
  MODEL_SETTINGS_KEYS.forEach((key) => {
    const value = snapshot[key]
    if (value !== undefined) {
      store.updateSetting(key, value as any)
    }
  })
}

type BuildHistorySnapshotParams = {
  historyInfo: HistoryInfo
  restoredHistory: ChatHistory
  restoredMessages: ChatMessage[]
  modelSettings: ChatModelSettingsSnapshot
  selectedModel: SidepanelChatSnapshot["selectedModel"]
  toolChoice: SidepanelChatSnapshot["toolChoice"]
  useOCR: SidepanelChatSnapshot["useOCR"]
  webSearch: SidepanelChatSnapshot["webSearch"]
}

const buildHistorySnapshot = ({
  historyInfo,
  restoredHistory,
  restoredMessages,
  modelSettings,
  selectedModel,
  toolChoice,
  useOCR,
  webSearch
}: BuildHistorySnapshotParams): SidepanelChatSnapshot => {
  const lastUsedPrompt = historyInfo.last_used_prompt
  const nextSelectedSystemPrompt = lastUsedPrompt?.prompt_id ?? null
  const nextSelectedQuickPrompt = lastUsedPrompt?.prompt_id
    ? null
    : lastUsedPrompt?.prompt_content ?? null
  const nextSelectedModel = historyInfo.model_id ?? selectedModel ?? null

  return {
    history: restoredHistory,
    messages: restoredMessages,
    chatMode: historyInfo.is_rag ? "rag" : "normal",
    historyId: historyInfo.id,
    webSearch,
    toolChoice,
    selectedModel: nextSelectedModel,
    selectedSystemPrompt: nextSelectedSystemPrompt,
    selectedQuickPrompt: nextSelectedQuickPrompt,
    temporaryChat: false,
    useOCR,
    serverChatId: historyInfo.server_chat_id ?? null,
    serverChatState: null,
    serverChatTopic: null,
    serverChatClusterId: null,
    serverChatSource: null,
    serverChatExternalRef: null,
    queuedMessages: [],
    modelSettings
  }
}

const SidepanelChat = () => {
  const owner = useSidepanelChatOwner()
  return owner
    ? <OwnedSidepanelChat key={`${owner.ownerKey}:${owner.revision}`} owner={owner} />
    : <PageAssistLoader />
}

const OwnedSidepanelChat = ({ owner }: { owner: SidepanelChatOwner }) => {
  const transferFlashcards = useFlashcardsGenerateTransfer()
  const transferStudyPack = useStudyPackTransfer()
  useServerOnline()
  const drop = React.useRef<HTMLDivElement>(null)
  const [dropedFile, setDropedFile] = React.useState<File | undefined>()
  const [sidebarOpen, setSidebarOpen] = React.useState(false)
  const [sidebarSearchQuery, setSidebarSearchQuery] = React.useState("")
  const sidebarSearchInputRef = React.useRef<HTMLInputElement>(null)
  const [sidebarSearchFocusNonce, setSidebarSearchFocusNonce] = React.useState(0)
  const [composerHeight, setComposerHeight] = React.useState(0)
  const { t } = useTranslation(["playground", "sidepanel", "common"])
  // Avoid depending on the `t` function in effects; it can be referentially unstable.
  const newChatLabel = React.useMemo(
    () => t("sidepanel:tabs.newChat", "New chat"),
    [t]
  )
  const notification = useAntdNotification()
  const { cancel: cancelNarration, isSpeaking: isNarrating, speak } = useTTS()
  React.useEffect(() => {
    void tldwClient.initialize().catch(() => null)
  }, [])
  // Per-tab storage (Chrome side panel) or per-window/global (Firefox sidebar).
  // tabId: undefined = not resolved yet, null = resolved but unavailable.
  const [tabId, setTabId] = React.useState<number | null | undefined>(undefined)
  const [isRestoringChat, setIsRestoringChat] = React.useState(true)
  const [restoreFailed, setRestoreFailed] = React.useState(false)
  const storageRef = React.useRef(
    createSafeStorage({
      area: "local"
    })
  )
  const backgroundImageStorageRef = React.useRef(createSafeStorage())
  const [dropState, setDropState] = React.useState<
    "idle" | "dragging" | "error"
  >("idle")
  const [dropFeedback, setDropFeedback] = React.useState<
    { type: "info" | "error"; message: string } | null
  >(null)
  const feedbackTimerRef = React.useRef<ReturnType<typeof setTimeout> | null>(
    null
  )
  // L20: Debounce timer for drag-leave to prevent false positives
  const dragLeaveTimerRef = React.useRef<ReturnType<typeof setTimeout> | null>(
    null
  )
  const showDropFeedback = React.useCallback(
    (feedback: { type: "info" | "error"; message: string }) => {
      setDropFeedback(feedback)
      if (feedbackTimerRef.current) {
        clearTimeout(feedbackTimerRef.current)
      }
      feedbackTimerRef.current = setTimeout(() => {
        // L16: Explicitly clear feedback on timer expiry
        setDropFeedback(null)
        feedbackTimerRef.current = null
      }, 4000)
    },
    []
  )
  useMigration()
  const {
    streaming,
    onSubmit,
    messages,
    history,
    setHistory,
    historyId,
    setHistoryId,
    setMessages,
    selectedModel,
    setSelectedModel,
    selectedQuickPrompt,
    setSelectedQuickPrompt,
    selectedSystemPrompt,
    setSelectedSystemPrompt,
    defaultChatWithWebsite,
    chatMode,
    setChatMode,
    toolChoice,
    setToolChoice,
    setIsEmbedding,
    setIsFirstMessage,
    setIsLoading,
    setIsProcessing,
    setIsSearchingInternet,
    setStreaming,
    setTemporaryChat,
    sidepanelTemporaryChat,
    stopStreamingRequest,
    temporaryChat,
    clearChat,
    queuedMessages,
    setQueuedMessages,
    serverChatClusterId,
    serverChatExternalRef,
    serverChatId,
    serverChatSource,
    serverChatState,
    serverChatTopic,
    setServerChatClusterId,
    setServerChatExternalRef,
    setServerChatId,
    setServerChatSource,
    setServerChatState,
    setServerChatTopic,
    setUseOCR,
    useOCR,
    webSearch,
    setWebSearch
  } = useMessage()
  const [selectedCharacter] =
    useSelectedCharacter<Character | null>(null)
  const [selectedAssistant] = useSelectedAssistant(null)
  const setRagMediaIds = useStoreMessageOption((state) => state.setRagMediaIds)
  const tabs = useSidepanelChatTabsStore((state) => state.tabs)
  const activeTabId = useSidepanelChatTabsStore((state) => state.activeTabId)
  const modelSettingsSnapshot = useStoreChatModelSettings((state) =>
    pickChatModelSettings(state)
  )
  const [timelineAction, setTimelineAction] =
    React.useState<TimelineActionDetail | null>(null)
  const isSwitchingTabRef = React.useRef(false)
  const loadGenerationRef = React.useRef(0)
  const invalidateLoads = React.useCallback(() => { loadGenerationRef.current++ }, [])
  const { containerRef, isAutoScrollToBottom, autoScrollToBottom } =
    useSmartScroll(messages, streaming, 100)
  const uiMode = useUiModeStore((state) => state.mode)
  const [isNarrow, setIsNarrow] = React.useState(false)
  const { checkOnce } = useConnectionActions()
  const [noteModalOpen, setNoteModalOpen] = React.useState(false)
  const [noteDraftContent, setNoteDraftContent] = React.useState("")
  const [noteDraftTitle, setNoteDraftTitle] = React.useState("")
  const [noteSuggestedTitle, setNoteSuggestedTitle] = React.useState("")
  const [noteSourceUrl, setNoteSourceUrl] = React.useState<string | undefined>()
  const [noteSourceMessageId, setNoteSourceMessageId] = React.useState<string | null>(null)
  const [noteSourceConversationId, setNoteSourceConversationId] = React.useState<string | null>(null)
  const noteCaptureRef = React.useRef<{
    controller: AbortController
    pending?: Promise<ServicePromptSnapshot>
    release?: () => void
  } | null>(null)
  const [noteSaving, setNoteSaving] = React.useState(false)
  const [noteError, setNoteError] = React.useState<string | null>(null)
  const [ingestCard, setIngestCard] = React.useState<IngestCardState | null>(null)
  const ingestFunnelRef = React.useRef<{
    funnelId: string
    baselineUserMessages: number
    tracked: boolean
    mediaId?: number
  } | null>(null)
  const textareaRef = React.useRef<HTMLTextAreaElement>(null)
  const [stickyChatInput] = useStorage(
    "stickyChatInput",
    DEFAULT_CHAT_SETTINGS.stickyChatInput
  )
  const [userDisplayName] = useStorage("chatUserDisplayName", "")
  useCharacterGreeting({
    playgroundReady: !isRestoringChat,
    selectedCharacter,
    selectedCharacterMode: null,
    serverChatId,
    historyId,
    messagesLength: messages.length,
    setMessages,
    setHistory
  })
  const composerPadding = composerHeight
    ? `${composerHeight + 16}px`
    : `${DEFAULT_COMPOSER_HEIGHT}px`
  const userMessageCount = React.useMemo(
    () => messages.reduce((count, item) => count + (item?.isBot ? 0 : 1), 0),
    [messages]
  )
  const scrollToLatestBottom = React.useMemo(() => {
    if (!stickyChatInput) return "8rem"
    const baseOffset = composerHeight
      ? composerHeight + 24
      : DEFAULT_COMPOSER_HEIGHT
    return `${Math.max(baseOffset, 128)}px`
  }, [composerHeight, stickyChatInput])

  const clearNoteCapture = React.useCallback(() => {
    const capture = noteCaptureRef.current
    noteCaptureRef.current = null
    capture?.controller.abort()
    capture?.release?.()
  }, [])
  React.useEffect(() => clearNoteCapture, [clearNoteCapture])

  const resetNoteModal = React.useCallback(() => {
    clearNoteCapture()
    transferFlashcards.cancel()
    transferStudyPack.cancel()
    setNoteModalOpen(false)
    setNoteDraftContent("")
    setNoteDraftTitle("")
    setNoteSuggestedTitle("")
    setNoteSourceUrl(undefined)
    setNoteSourceMessageId(null)
    setNoteSourceConversationId(null)
    setNoteSaving(false)
    setNoteError(null)
  }, [clearNoteCapture, transferFlashcards, transferStudyPack])

  const openOptionsHashRoute = React.useCallback((route: string) => {
    const normalizedRoute = route.startsWith("/") ? route : `/${route}`
    const path = `/options.html#${normalizedRoute}` as const

    try {
      if (browser?.runtime?.id && browser.runtime.getURL) {
        const url = browser.runtime.getURL(path)
        if (browser.tabs?.create) {
          void browser.tabs.create({ url })
        } else {
          window.open(url, "_blank")
        }
        return
      }
    } catch (error) {
      console.debug("[sidepanel] openOptionsHashRoute browser API unavailable:", error)
    }

    try {
      if (
        typeof chrome !== "undefined" &&
        chrome.runtime?.id &&
        chrome.runtime.getURL
      ) {
        const url = chrome.runtime.getURL(path)
        window.open(url, "_blank")
        return
      }
    } catch (error) {
      console.debug("[sidepanel] openOptionsHashRoute chrome API unavailable:", error)
    }

    window.open(normalizedRoute, "_blank")
  }, [])

  const handleGenerateFlashcardsFromSelection = React.useCallback(async () => {
    const content = noteDraftContent
    const capture = noteCaptureRef.current
    if (!content.trim()) {
      setNoteError(t("sidepanel:notes.emptyContent", "Nothing to save"))
      return
    }

    try {
      const messageId = noteSourceMessageId?.trim() || undefined
      await transferFlashcards(async current => {
        if (!capture?.pending) throw new Error("Capture this source again to verify its account. Your draft is unchanged.")
        const owner = await capture.pending
        owner.scopeSignal.throwIfAborted()
        if (noteCaptureRef.current !== capture || flashcardsHandoffAuthority(owner) !== flashcardsHandoffAuthority(current)) {
          throw new Error("The source account changed. Capture it again before transferring.")
        }
        return {
          text: content,
          sourceType: messageId ? "message" : "manual",
          sourceId: messageId,
          messageId,
          sourceTitle: (noteDraftTitle || noteSuggestedTitle) || undefined,
          conversationId: messageId ? noteSourceConversationId || undefined : undefined
        }
      }, { newTab: true })
      if (noteCaptureRef.current === capture) resetNoteModal()
    } catch (error) {
      if (noteCaptureRef.current === capture && !(error instanceof Error && error.name === "AbortError")) setNoteError(error instanceof Error ? error.message : "The transfer could not be opened. Your draft is unchanged.")
    }
  }, [
    noteSourceMessageId,
    noteSourceConversationId,
    noteDraftContent,
    noteDraftTitle,
    noteSuggestedTitle,
    transferFlashcards,
    resetNoteModal,
    t
  ])

  const handleCreateStudyPackFromSelection = React.useCallback(async () => {
    const capture = noteCaptureRef.current
    const title = (noteDraftTitle || noteSuggestedTitle).trim()
    const messageId = noteSourceMessageId?.trim() || ""
    if (!title || !messageId) {
      setNoteError(
        t(
          "sidepanel:notes.studyPackUnavailable",
          "This selection does not have a supported message source for study packs yet."
        )
      )
      return
    }

    try {
      await transferStudyPack(async current => {
        if (!capture?.pending) throw new Error("Capture this source again to verify its account. Your selection is unchanged.")
        const owner = await capture.pending
        owner.scopeSignal.throwIfAborted()
        if (noteCaptureRef.current !== capture || flashcardsHandoffAuthority(owner) !== flashcardsHandoffAuthority(current)) {
          throw new Error("The source account changed. Capture it again before transferring.")
        }
        return {
          title,
          sourceItems: [{ sourceType: "message", sourceId: messageId, sourceTitle: title }]
        }
      }, { newTab: true })
      if (noteCaptureRef.current === capture) resetNoteModal()
    } catch (error) {
      if (noteCaptureRef.current === capture && !(error instanceof Error && error.name === "AbortError")) setNoteError(error instanceof Error ? error.message : "The study pack transfer could not be opened. Your selection is unchanged.")
    }
  }, [noteDraftTitle, noteSuggestedTitle, noteSourceMessageId, transferStudyPack, resetNoteModal, t])

  const handleNoteSave = React.useCallback(async () => {
    const content = noteDraftContent.trim()
    const title = (noteDraftTitle || noteSuggestedTitle).trim()
    if (!content) {
      setNoteError(t("sidepanel:notes.emptyContent", "Nothing to save"))
      return
    }
    if (!title) {
      setNoteError(t("sidepanel:notes.titleRequired", "Add a title to save this note"))
      return
    }
    setNoteError(null)
    setNoteSaving(true)
    try {
      await tldwClient.createNote(buildCapturedNoteContent(content, noteSourceUrl), {
        title,
        keywords: [CAPTURED_NOTE_KEYWORD]
      })
      notification.success({
        message: t("sidepanel:notification.savedToNotes", "Saved to Notes")
      })
      resetNoteModal()
    } catch (e: any) {
      const msg = e?.message
        ? t("sidepanel:notes.createFailedWithReason", "Note creation failed: {{reason}}").replace(
            "{{reason}}",
            e.message
          )
        : t(
            "sidepanel:notes.createFailed",
            "Note creation failed. The captured selection was not saved."
          )
      setNoteError(msg)
      notification.error({ message: msg })
    } finally {
      setNoteSaving(false)
    }
  }, [
    noteDraftContent,
    noteDraftTitle,
    noteSuggestedTitle,
    noteSourceUrl,
    notification,
    resetNoteModal,
    t
  ])

  const handleNoteTitleChange = (value: string) => {
    setNoteDraftTitle(value)
    if (noteError) setNoteError(null)
  }

  const handleNoteContentChange = (value: string) => {
    setNoteDraftContent(value)
    if (noteError) setNoteError(null)
  }

  const buildSnapshot = React.useCallback((): SidepanelChatSnapshot => {
    return {
      history,
      messages,
      chatMode,
      historyId,
      webSearch,
      toolChoice,
      selectedModel: selectedModel ?? null,
      selectedSystemPrompt,
      selectedQuickPrompt,
      temporaryChat,
      useOCR,
      serverChatId,
      serverChatState,
      serverChatTopic,
      serverChatClusterId,
      serverChatSource,
      serverChatExternalRef,
      queuedMessages,
      modelSettings: modelSettingsSnapshot
    }
  }, [
    history,
    messages,
    chatMode,
    historyId,
    webSearch,
    toolChoice,
    selectedModel,
    selectedSystemPrompt,
    selectedQuickPrompt,
    temporaryChat,
    useOCR,
    serverChatId,
    serverChatState,
    serverChatTopic,
    serverChatClusterId,
    serverChatSource,
    serverChatExternalRef,
    queuedMessages,
    modelSettingsSnapshot
  ])

  const applySnapshot = React.useCallback(
    (snapshot: SidepanelChatSnapshot) => {
      if (!owner.isCurrent()) return
      setHistory(snapshot.history || [])
      setMessages(snapshot.messages || [])
      setHistoryId(snapshot.historyId ?? null)
      setChatMode(snapshot.chatMode || "normal")
      setWebSearch(snapshot.webSearch ?? false)
      setToolChoice(snapshot.toolChoice ?? "none")
      const snapshotModel =
        typeof snapshot.selectedModel === "string"
          ? snapshot.selectedModel.trim()
          : ""
      if (snapshotModel) {
        setSelectedModel(snapshotModel)
      }
      setSelectedSystemPrompt(snapshot.selectedSystemPrompt ?? null)
      setSelectedQuickPrompt(snapshot.selectedQuickPrompt ?? null)
      setTemporaryChat(snapshot.temporaryChat ?? false)
      setUseOCR(snapshot.useOCR ?? false)
      setServerChatId(snapshot.serverChatId ?? null)
      setServerChatState(snapshot.serverChatState ?? null)
      setServerChatTopic(snapshot.serverChatTopic ?? null)
      setServerChatClusterId(snapshot.serverChatClusterId ?? null)
      setServerChatSource(snapshot.serverChatSource ?? null)
      setServerChatExternalRef(snapshot.serverChatExternalRef ?? null)
      setQueuedMessages(restoreQueuedRequests(snapshot.queuedMessages ?? []))
      setIsFirstMessage((snapshot.history || []).length === 0)
      setIsLoading(false)
      setIsProcessing(false)
      setIsEmbedding(false)
      setStreaming(false)
      setIsSearchingInternet(false)
      applyChatModelSettingsSnapshot(snapshot.modelSettings)
    },
    [
      owner,
      setHistory,
      setMessages,
      setHistoryId,
      setChatMode,
      setWebSearch,
      setToolChoice,
      setSelectedModel,
      setSelectedSystemPrompt,
      setSelectedQuickPrompt,
      setTemporaryChat,
      setUseOCR,
      setServerChatId,
      setServerChatState,
      setServerChatTopic,
      setServerChatClusterId,
      setServerChatSource,
      setServerChatExternalRef,
      setQueuedMessages,
      setIsFirstMessage,
      setIsLoading,
      setIsProcessing,
      setIsEmbedding,
      setStreaming,
      setIsSearchingInternet
    ]
  )

  const saveActiveTabSnapshot = React.useCallback(() => {
    const currentTabId = useSidepanelChatTabsStore.getState().activeTabId
    if (!owner.isCurrent() || isSwitchingTabRef.current || !currentTabId || currentTabId !== activeTabId) return
    const snapshot = buildSnapshot()
    snapshot.modelSettings = pickChatModelSettings(
      useStoreChatModelSettings.getState()
    )
    useSidepanelChatTabsStore.getState().setSnapshot(currentTabId, snapshot)
  }, [activeTabId, buildSnapshot, owner])

  const toggleSidebar = () => {
    setSidebarOpen((prev) => !prev)
  }

  const requestSidebarSearchFocus = React.useCallback(() => {
    if (uiMode !== "pro" || isNarrow) {
      setSidebarOpen(true)
    }
    setSidebarSearchFocusNonce((prev) => prev + 1)
  }, [isNarrow, uiMode])

  const toggleChatMode = () => {
    setChatMode(chatMode === "rag" ? "normal" : "rag")
  }

  const toggleWebSearchMode = () => {
    setWebSearch(!webSearch)
  }

  useChatShortcuts(clearChat, true)
  useSidebarShortcuts(toggleSidebar, true)
  useChatModeShortcuts(toggleChatMode, true)
  useWebSearchShortcuts(toggleWebSearchMode, true)

  const [chatBackgroundImage] = useStorage({
    key: CHAT_BACKGROUND_IMAGE_SETTING.key,
    instance: backgroundImageStorageRef.current
  })
  const [chatWindowOpacity] = useSetting(CHAT_WINDOW_OPACITY_SETTING)
  const [chatMessageOpacity] = useSetting(CHAT_MESSAGE_OPACITY_SETTING)
  const [chatCharacterImageOpacity] = useSetting(
    CHAT_CHARACTER_IMAGE_OPACITY_SETTING
  )
  const chatWindowOpacityAlpha = resolveOpacityAlpha(
    chatWindowOpacity,
    CHAT_WINDOW_OPACITY_SETTING.defaultValue
  )
  const chatMessageOpacityAlpha = resolveOpacityAlpha(
    chatMessageOpacity,
    CHAT_MESSAGE_OPACITY_SETTING.defaultValue
  )
  const chatCharacterImageOpacityAlpha = resolveOpacityAlpha(
    chatCharacterImageOpacity,
    CHAT_CHARACTER_IMAGE_OPACITY_SETTING.defaultValue
  )
  const bgMsg = useBackgroundMessage()
  const lastBgMsgRef = React.useRef<typeof bgMsg | null>(null)
  const pendingWebClipAnalyzeRef = React.useRef<string | null>(null)
  const messagesRef = React.useRef(messages)

  React.useEffect(() => {
    messagesRef.current = messages
  }, [messages])

  const applySnapshotRef = React.useRef(applySnapshot)
  applySnapshotRef.current = applySnapshot
  const restoreDefaults = React.useRef({ modelSettingsSnapshot, selectedModel, toolChoice, useOCR, webSearch, newChatLabel })
  const restoreSidepanelState = React.useCallback(async () => {
    const { modelSettingsSnapshot, selectedModel, toolChoice, useOCR, webSearch, newChatLabel } = restoreDefaults.current
    if (tabId === undefined || !owner.isCurrent()) return
    const generation = ++loadGenerationRef.current
    const current = () => owner.isCurrent() && generation === loadGenerationRef.current
    setIsRestoringChat(true)
    setRestoreFailed(false)
    try {
      const tabsState = await readSidepanelTabs(storageRef.current, tabId, owner.ownerKey, current)
      if (!current()) return
      if (tabsState?.tabs.length) {
        const activeId = tabsState.tabs.some(tab => tab.id === tabsState.activeTabId)
          ? tabsState.activeTabId : tabsState.tabs[0].id
        useSidepanelChatTabsStore.getState().setTabsState({
          tabs: tabsState.tabs, activeTabId: activeId, snapshotsById: tabsState.snapshotsById
        })
        if (activeId && tabsState.snapshotsById[activeId]) applySnapshotRef.current(tabsState.snapshotsById[activeId])
      } else {
        const enabled = await copilotResumeLastChat()
        if (!current()) return
        if (enabled && messagesRef.current.length === 0) {
          const recent = await getRecentChatFromCopilot(owner.snapshot)
          if (!current()) return
          if (recent && messagesRef.current.length === 0) {
            const snapshot = buildHistorySnapshot({
              historyInfo: recent.history, restoredHistory: formatToChatHistory(recent.messages),
              restoredMessages: formatToMessage(recent.messages), modelSettings: modelSettingsSnapshot,
              selectedModel: selectedModel ?? null, toolChoice, useOCR, webSearch
            })
            const id = generateID()
            useSidepanelChatTabsStore.getState().setTabsState({
              tabs: [{ id, label: recent.history.title || newChatLabel, historyId: recent.history.id,
                serverChatId: snapshot.serverChatId, serverChatTopic: null, updatedAt: Date.now() }],
              activeTabId: id, snapshotsById: { [id]: snapshot }
            })
            applySnapshotRef.current(snapshot)
          }
        }
      }
      if (current()) setIsRestoringChat(false)
    } catch {
      // Keep persistence suspended: a failed read must not replace an owner's
      // saved tabs with an empty scaffold. Offer an explicit same-owner retry.
      if (current()) setRestoreFailed(true)
    }
  }, [owner, tabId])

  const persistSidepanelState = React.useCallback(() => {
    if (tabId === undefined || isRestoringChat || !owner.isCurrent()) return
    saveActiveTabSnapshot()
    const { tabs, activeTabId, snapshotsById } = useSidepanelChatTabsStore.getState()
    const snapshot: SidepanelTabsState = {
      version: 2, ownerKey: owner.ownerKey, tabs, activeTabId, snapshotsById
    }
    void storageRef.current.set(getTabsStorageKey(tabId, owner.ownerKey), snapshot).catch(() => {})
  }, [isRestoringChat, owner, saveActiveTabSnapshot, tabId])

  React.useEffect(() => {
    void checkOnce()
  }, [checkOnce])

  React.useEffect(() => {
    if (typeof window === "undefined" || !window.matchMedia) return
    const media = window.matchMedia("(max-width: 400px)")
    const update = () => setIsNarrow(media.matches)
    update()
    media.addEventListener("change", update)
    return () => media.removeEventListener("change", update)
  }, [])

  React.useEffect(() => {
    // Resolve the tab id associated with this sidepanel instance.
    const fetchTabId = async () => {
      const resolvedTabId = await readSidepanelRuntimeTabId()
      setTabId(resolvedTabId)
    }
    fetchTabId()
  }, [])

  React.useEffect(() => {
    void restoreSidepanelState()
    return invalidateLoads
  }, [restoreSidepanelState, invalidateLoads])

  const truncateTabLabel = React.useCallback((label: string) => {
    const trimmed = label.trim()
    if (trimmed.length <= 40) return trimmed
    return `${trimmed.slice(0, 40)}...`
  }, [])

  const fallbackLabel = React.useMemo(() => {
    if (historyId || serverChatTopic) return ""
    const firstUserMessage = messages.find((message) => !message.isBot)?.message
    return (firstUserMessage || "").trim()
  }, [historyId, serverChatTopic, messages])

  React.useEffect(() => {
    if (!owner.isCurrent() || !activeTabId || isRestoringChat || isSwitchingTabRef.current) return
    let isCurrent = true
    const updateLabel = async () => {
      const store = useSidepanelChatTabsStore.getState()
      const currentTab = store.tabs.find((tab) => tab.id === activeTabId)
      const isManualLabel = currentTab?.labelSource === "manual"
      let label = ""
      if (historyId) {
        try {
          label = (await getTitleById(historyId)) || ""
        } catch {
          label = ""
        }
      }
      if (!label && serverChatTopic) {
        label = serverChatTopic
      }
      if (!label && fallbackLabel) {
        label = fallbackLabel
      }
      if (!label) {
        label = newChatLabel
      }
      if (!isCurrent || !owner.isCurrent()) return
      useSidepanelChatTabsStore.getState().upsertTab({
        id: activeTabId,
        label:
          isManualLabel && currentTab?.label
            ? currentTab.label
            : truncateTabLabel(label),
        labelSource: isManualLabel ? "manual" : "auto",
        historyId: historyId ?? null,
        serverChatId: serverChatId ?? null,
        serverChatTopic: serverChatTopic ?? null,
        updatedAt: Date.now()
      })
    }
    void updateLabel()
    return () => {
      isCurrent = false
    }
  }, [
    activeTabId,
    owner,
    isRestoringChat,
    fallbackLabel,
    historyId,
    serverChatId,
    serverChatTopic,
    newChatLabel,
    truncateTabLabel
  ])

  const handleRenameActiveTab = React.useCallback(
    (nextLabel: string) => {
      const trimmed = nextLabel.trim()
      if (!owner.isCurrent() || isRestoringChat || !activeTabId || !trimmed) return
      useSidepanelChatTabsStore.getState().renameTab(activeTabId, trimmed)
    },
    [activeTabId, owner, isRestoringChat]
  )

  React.useEffect(() => {
    if (isRestoringChat) return
    saveActiveTabSnapshot()
  }, [isRestoringChat, saveActiveTabSnapshot])

  React.useEffect(() => {
    if (!owner.isCurrent() || isRestoringChat || isSwitchingTabRef.current) return
    if (tabs.length > 0 && activeTabId) return
    const initialTabId = generateID()
    const initialTab: SidepanelChatTab = {
      id: initialTabId,
      label: newChatLabel,
      historyId: historyId ?? null,
      serverChatId: serverChatId ?? null,
      serverChatTopic: serverChatTopic ?? null,
      updatedAt: Date.now()
    }
    useSidepanelChatTabsStore.getState().setTabsState({
      tabs: [initialTab],
      activeTabId: initialTabId,
      snapshotsById: { [initialTabId]: buildSnapshot() }
    })
  }, [
    activeTabId,
    owner,
    buildSnapshot,
    historyId,
    isRestoringChat,
    newChatLabel,
    serverChatId,
    serverChatTopic,
    tabs.length
  ])

  const handleNewTab = React.useCallback(() => {
    if (!owner.isCurrent() || isRestoringChat) return
    loadGenerationRef.current++
    saveActiveTabSnapshot()
    if (streaming) {
      stopStreamingRequest()
    }
    setDropedFile(undefined)
    const newTabId = generateID()
    useSidepanelChatTabsStore.getState().upsertTab({
      id: newTabId,
      label: newChatLabel,
      historyId: null,
      serverChatId: null,
      serverChatTopic: null,
      updatedAt: Date.now()
    })
    isSwitchingTabRef.current = true
    useSidepanelChatTabsStore.getState().setActiveTabId(newTabId)
    clearChat()
    setTimeout(() => {
      isSwitchingTabRef.current = false
    }, 0)
  }, [
    owner,
      isRestoringChat,
    clearChat,
    saveActiveTabSnapshot,
    setDropedFile,
    stopStreamingRequest,
    streaming,
    newChatLabel
  ])

  const handleSelectTab = React.useCallback(
    (tabId: string) => {
      if (!owner.isCurrent() || isRestoringChat || !tabId || tabId === activeTabId) return
      loadGenerationRef.current++
      saveActiveTabSnapshot()
      if (streaming) {
        stopStreamingRequest()
      }
      setDropedFile(undefined)
      const snapshot = useSidepanelChatTabsStore.getState().getSnapshot(tabId)
      isSwitchingTabRef.current = true
      useSidepanelChatTabsStore.getState().setActiveTabId(tabId)
      if (snapshot) {
        applySnapshot(snapshot)
      } else {
        clearChat()
      }
      setTimeout(() => {
        isSwitchingTabRef.current = false
      }, 0)
    },
    [
      activeTabId,
      owner,
      isRestoringChat,
      applySnapshot,
      clearChat,
      saveActiveTabSnapshot,
      stopStreamingRequest,
      streaming,
      setDropedFile
    ]
  )

  const handleCloseTab = React.useCallback(
    (tabId: string) => {
      if (!owner.isCurrent() || isRestoringChat) return
      const store = useSidepanelChatTabsStore.getState()
      if (store.tabs.length <= 1) {
        store.removeTab(tabId)
        handleNewTab()
        return
      }
      if (tabId === store.activeTabId) {
        const currentIndex = store.tabs.findIndex((tab) => tab.id === tabId)
        const nextTab =
          store.tabs[currentIndex + 1] || store.tabs[currentIndex - 1]
        if (nextTab) {
          handleSelectTab(nextTab.id)
        }
      }
      store.removeTab(tabId)
    },
    [handleNewTab, handleSelectTab, owner, isRestoringChat]
  )

  const openSnapshotTab = React.useCallback(
    (tab: SidepanelChatTab, snapshot: SidepanelChatSnapshot) => {
      if (!owner.isCurrent()) return
      const store = useSidepanelChatTabsStore.getState()
      store.upsertTab(tab)
      store.setSnapshot(tab.id, snapshot)
      isSwitchingTabRef.current = true
      store.setActiveTabId(tab.id)
      applySnapshot(snapshot)
      setTimeout(() => {
        isSwitchingTabRef.current = false
      }, 0)
    },
    [applySnapshot, owner]
  )

  const openLocalHistory = React.useCallback(
    async (targetHistoryId: string) => {
      if (!owner.isCurrent() || isRestoringChat) return
      const generation = ++loadGenerationRef.current
      const current = () => owner.isCurrent() && generation === loadGenerationRef.current
      const existingTab = tabs.find((tab) => tab.historyId === targetHistoryId)
      if (existingTab) {
        handleSelectTab(existingTab.id)
        return
      }
      saveActiveTabSnapshot()
      if (streaming) {
        stopStreamingRequest()
      }
      setDropedFile(undefined)
      setIsLoading(true)
      try {
        const chatData = await getFullChatData(targetHistoryId)
        if (!current()) return
        if (!chatData || chatData.historyInfo.server_scope_key !== owner.ownerKey) {
          notification.error({
            message: t("common:error", "Error"),
            description: t(
              "common:serverChatLoadError",
              "Failed to load conversation."
            )
          })
          return
        }

        const restoredHistory = formatToChatHistory(chatData.messages)
        const restoredMessages = formatToMessage(chatData.messages)
        const historyInfo = chatData.historyInfo
        const snapshot = buildHistorySnapshot({
          historyInfo,
          restoredHistory,
          restoredMessages,
          modelSettings: modelSettingsSnapshot,
          selectedModel: selectedModel ?? null,
          toolChoice,
          useOCR,
          webSearch
        })

        const newTabId = generateID()
        openSnapshotTab(
          {
            id: newTabId,
            label: truncateTabLabel(
              historyInfo.title || newChatLabel
            ),
            labelSource: "auto",
            historyId: historyInfo.id,
            serverChatId: historyInfo.server_chat_id ?? null,
            serverChatTopic: null,
            updatedAt: Date.now()
          },
          snapshot
        )
      } catch (err: any) {
        if (!current()) return
        notification.error({
          message: t("common:error", "Error"),
          description:
            err?.message ||
            t("common:serverChatLoadError", "Failed to load conversation.")
        })
      } finally {
        if (current()) setIsLoading(false)
      }
    },
    [
      owner,
      isRestoringChat,
      handleSelectTab,
      modelSettingsSnapshot,
      notification,
      openSnapshotTab,
      saveActiveTabSnapshot,
      selectedModel,
      setDropedFile,
      setIsLoading,
      stopStreamingRequest,
      streaming,
      newChatLabel,
      t,
      tabs,
      toolChoice,
      truncateTabLabel,
      useOCR,
      webSearch
    ]
  )

  const handleTimelineActionRequest = React.useCallback(
    async (detail: TimelineActionDetail) => {
      if (!detail?.historyId) return
      if (detail.historyId !== historyId) {
        await openLocalHistory(detail.historyId)
      }
      setTimelineAction(detail)
    },
    [historyId, openLocalHistory]
  )

  const openServerChat = React.useCallback(
    async (chat: ServerChatHistoryItem) => {
      if (!owner.isCurrent() || isRestoringChat) return
      const generation = ++loadGenerationRef.current
      const current = () => owner.isCurrent() && generation === loadGenerationRef.current
      const chatId = String(chat.id)
      const existingTab = tabs.find((tab) => tab.serverChatId === chatId)
      if (existingTab) {
        handleSelectTab(existingTab.id)
        return
      }
      saveActiveTabSnapshot()
      if (streaming) {
        stopStreamingRequest()
      }
      setDropedFile(undefined)
      setIsLoading(true)
      const snapshotOwner = owner.snapshot
      try {
        const list = await tldwClient.listChatMessages(chatId, {
          include_deleted: "false",
          include_metadata: "true"
        }, { signal: snapshotOwner.scopeSignal, requestScope: snapshotOwner.requestScope })
        if (!current()) return
        const messageList: ServerChatMessageInput[] = list
        const { history, mappedMessages } = mapServerChatMessages(
          messageList,
          userDisplayName
        )
        const ownerKey = owner.ownerKey
        const localHistoryId = await linkServerChatMirror({
          chatId, title: chat.title || newChatLabel, ownerKey, signal: snapshotOwner.scopeSignal
        })
        const mirror = await reconcileServerChatMirror({
          historyId: localHistoryId, chatId, ownerKey,
          messages: mappedMessages, signal: snapshotOwner.scopeSignal
        })
        if (!current()) return

        const snapshot: SidepanelChatSnapshot = {
          history,
          messages: mappedMessages.map(message => ({
            ...message, id: mirror.localIds.get(message.serverMessageId) || message.id
          })),
          chatMode,
          historyId: localHistoryId,
          webSearch,
          toolChoice,
          selectedModel: selectedModel ?? null,
          selectedSystemPrompt,
          selectedQuickPrompt,
          temporaryChat: false,
          useOCR,
          serverChatId: chatId,
          serverChatState: normalizeConversationState(chat.state),
          serverChatTopic: chat.topic_label ?? null,
          serverChatClusterId: chat.cluster_id ?? null,
          serverChatSource: chat.source ?? null,
          serverChatExternalRef: chat.external_ref ?? null,
          queuedMessages: [],
          modelSettings: modelSettingsSnapshot
        }

        const newTabId = generateID()
        openSnapshotTab(
          {
            id: newTabId,
            label: truncateTabLabel(
              chat.title || newChatLabel
            ),
            labelSource: "auto",
            historyId: localHistoryId,
            serverChatId: chatId,
            serverChatTopic: chat.topic_label ?? null,
            updatedAt: Date.now()
          },
          snapshot
        )
      } catch (err: any) {
        if (!current()) return
        notification.error({
          message: t("common:error", "Error"),
          description:
            err?.message ||
            t("common:serverChatLoadError", "Failed to load conversation.")
        })
      } finally {
        if (current()) setIsLoading(false)
      }
    },
    [
      owner,
      isRestoringChat,
      userDisplayName,
      chatMode,
      handleSelectTab,
      modelSettingsSnapshot,
      notification,
      openSnapshotTab,
      saveActiveTabSnapshot,
      selectedModel,
      selectedQuickPrompt,
      selectedSystemPrompt,
      setDropedFile,
      setIsLoading,
      stopStreamingRequest,
      streaming,
      newChatLabel,
      t,
      tabs,
      toolChoice,
      truncateTabLabel,
      useOCR,
      webSearch
    ]
  )

  React.useEffect(() => {
    persistSidepanelState()
    const handleBeforeUnload = () => {
      persistSidepanelState()
    }

    const handleVisibilityChange = () => {
      if (document.visibilityState === "hidden") {
        persistSidepanelState()
      }
    }

    window.addEventListener("beforeunload", handleBeforeUnload)
    document.addEventListener("visibilitychange", handleVisibilityChange)

    return () => {
      window.removeEventListener("beforeunload", handleBeforeUnload)
      document.removeEventListener("visibilitychange", handleVisibilityChange)
    }
  }, [persistSidepanelState])

  React.useEffect(() => {
    if (typeof window === "undefined") return

    const handleTimelineActionEvent = (event: Event) => {
      const detail = (event as CustomEvent<TimelineActionDetail>).detail
      if (!detail?.historyId) return
      void handleTimelineActionRequest(detail)
    }

    const handleOpenHistoryEvent = (event: Event) => {
      const detail = (event as CustomEvent<OpenHistoryDetail>).detail
      if (!detail?.historyId) return
      void handleTimelineActionRequest({
        action: "go",
        historyId: detail.historyId,
        messageId: detail.messageId
      })
    }

    window.addEventListener(TIMELINE_ACTION_EVENT, handleTimelineActionEvent)
    window.addEventListener(OPEN_HISTORY_EVENT, handleOpenHistoryEvent)
    return () => {
      window.removeEventListener(TIMELINE_ACTION_EVENT, handleTimelineActionEvent)
      window.removeEventListener(OPEN_HISTORY_EVENT, handleOpenHistoryEvent)
    }
  }, [handleTimelineActionRequest])

  React.useEffect(() => {
    if (!drop.current) {
      return
    }
    const handleDragOver = (e: DragEvent) => {
      e.preventDefault()
      e.stopPropagation()
    }

    const handleDrop = (e: DragEvent) => {
      e.preventDefault()
      e.stopPropagation()

      setDropState("idle")

      const files = Array.from(e.dataTransfer?.files || [])

      const isImage = files.every((file) => file.type.startsWith("image/"))

      if (!isImage) {
        setDropState("error")
        showDropFeedback({
          type: "error",
          message: t(
            "playground:drop.imageOnly",
            "Only images can be dropped here right now."
          )
        })
        return
      }

      const newFiles = Array.from(e.dataTransfer?.files || []).slice(0, 1)
      if (newFiles.length > 0) {
        setDropedFile(newFiles[0])
        showDropFeedback({
          type: "info",
          message: `${newFiles[0]?.name || "Image"} ready to send`
        })
      }
    }

    const handleDragEnter = (e: DragEvent) => {
      e.preventDefault()
      e.stopPropagation()
      // L20: Clear drag-leave debounce timer when re-entering
      if (dragLeaveTimerRef.current) {
        clearTimeout(dragLeaveTimerRef.current)
        dragLeaveTimerRef.current = null
      }
      setDropState("dragging")
      showDropFeedback({
        type: "info",
        message: t(
          "playground:drop.imageHint",
          "Drop an image to include it in your message"
        )
      })
    }

    const handleDragLeave = (e: DragEvent) => {
      e.preventDefault()
      e.stopPropagation()
      // L20: Debounce drag-leave by 50ms to prevent false positives from child elements
      if (dragLeaveTimerRef.current) {
        clearTimeout(dragLeaveTimerRef.current)
      }
      dragLeaveTimerRef.current = setTimeout(() => {
        setDropState("idle")
        dragLeaveTimerRef.current = null
      }, 50)
    }

    drop.current.addEventListener("dragover", handleDragOver)
    drop.current.addEventListener("drop", handleDrop)
    drop.current.addEventListener("dragenter", handleDragEnter)
    drop.current.addEventListener("dragleave", handleDragLeave)

    return () => {
      if (drop.current) {
        drop.current.removeEventListener("dragover", handleDragOver)
        drop.current.removeEventListener("drop", handleDrop)
        drop.current.removeEventListener("dragenter", handleDragEnter)
        drop.current.removeEventListener("dragleave", handleDragLeave)
      }
    }
  }, [])

  React.useEffect(() => {
    return () => {
      if (feedbackTimerRef.current) {
        clearTimeout(feedbackTimerRef.current)
        feedbackTimerRef.current = null
      }
      // L20: Clean up drag-leave debounce timer
      if (dragLeaveTimerRef.current) {
        clearTimeout(dragLeaveTimerRef.current)
        dragLeaveTimerRef.current = null
      }
    }
  }, [])

  React.useEffect(() => {
    if (defaultChatWithWebsite) {
      setChatMode("rag")
    }
    if (sidepanelTemporaryChat) {
      setTemporaryChat(true)
    }
  }, [defaultChatWithWebsite, sidepanelTemporaryChat])

  const seedComposerMessage = React.useCallback(
    (messageText: string, options?: { ifEmptyOnly?: boolean }) => {
      const message = String(messageText || "").trim()
      if (!message) return
      try {
        window.dispatchEvent(
          new CustomEvent("tldw:set-composer-message", {
            detail: {
              message,
              ifEmptyOnly: Boolean(options?.ifEmptyOnly)
            }
          })
        )
      } catch {
        // ignore
      }
      try {
        window.dispatchEvent(new CustomEvent("tldw:focus-composer"))
      } catch {
        // ignore
      }
    },
    []
  )

  const handleRetryIngest = React.useCallback(async () => {
    if (!ingestCard?.funnelId) return
    try {
      await browser.runtime.sendMessage({
        type: "tldw:media-ingest/retry",
        payload: { funnelId: ingestCard.funnelId }
      })
    } catch (error) {
      console.error("[sidepanel] retry ingest failed", error)
    }
  }, [ingestCard?.funnelId])

  const handleCancelIngest = React.useCallback(async () => {
    if (!ingestCard?.funnelId) return
    try {
      await browser.runtime.sendMessage({
        type: "tldw:media-ingest/cancel",
        payload: { funnelId: ingestCard.funnelId, reason: "user_cancelled" }
      })
    } catch (error) {
      console.error("[sidepanel] cancel ingest failed", error)
    }
  }, [ingestCard?.funnelId])

  const handleOpenAuthSettings = React.useCallback(async () => {
    try {
      await browser.runtime.sendMessage({
        type: "tldw:media-ingest/open-auth-settings"
      })
    } catch (error) {
      console.error("[sidepanel] open auth settings failed", error)
      window.open("/options.html#/settings/tldw", "_blank")
    }
  }, [])

  React.useEffect(() => {
    const funnel = ingestFunnelRef.current
    if (!funnel || funnel.tracked) return
    if (userMessageCount <= funnel.baselineUserMessages) return
    funnel.tracked = true
    void browser.runtime
      .sendMessage({
        type: "tldw:media-ingest/funnel-event",
        payload: {
          event: "first_chat_message",
          funnelId: funnel.funnelId,
          metadata: {
            mediaId: funnel.mediaId
          }
        }
      })
      .catch((error) => {
        console.error("[sidepanel] funnel metric failed", error)
      })
    setIngestCard((prev) => {
      if (!prev || prev.funnelId !== funnel.funnelId) return prev
      if (prev.starterQuestions.length === 0) return prev
      return { ...prev, starterQuestions: [] }
    })
  }, [userMessageCount])

  React.useEffect(() => {
    if (streaming || !selectedModel) return

    const pendingAnalyze = readPendingWebClipAnalyzeRequest()
    if (!pendingAnalyze) return
    if (pendingWebClipAnalyzeRef.current === pendingAnalyze.id) return

    const baselineMessageIds = collectWebClipAnalyzeMessageIds(messagesRef.current)
    pendingWebClipAnalyzeRef.current = pendingAnalyze.id

    void (async () => {
      try {
        await onSubmit({
          message: pendingAnalyze.message,
          messageType: WEB_CLIPPER_ANALYZE_MESSAGE_TYPE,
          image: pendingAnalyze.image || "",
          requestOverrides: pendingAnalyze.requestOverrides
        })

        if (
          hasSubmittedWebClipAnalyzeMessage(
            messagesRef.current,
            baselineMessageIds
          )
        ) {
          clearPendingWebClipAnalyzeRequest(pendingAnalyze.id)
        } else {
          pendingWebClipAnalyzeRef.current = null
        }
      } catch (error) {
        pendingWebClipAnalyzeRef.current = null
        notification.error({
          message: t(
            "sidepanel:notification.webClipAnalyzeFailedTitle",
            "Clip handoff failed"
          ),
          description:
            error instanceof Error && error.message.trim()
              ? error.message
              : t(
                  "sidepanel:notification.webClipAnalyzeFailedBody",
                  "The saved clip could not be sent to chat."
                )
        })
      }
    })()
  }, [notification, onSubmit, selectedModel, streaming, t])

  React.useEffect(() => {
    if (!bgMsg) return
    if (lastBgMsgRef.current === bgMsg) return

    if (bgMsg.type === "save-to-notes") {
      lastBgMsgRef.current = bgMsg
      const selected = (bgMsg.text || bgMsg.payload?.selectionText || "").trim()
      if (!selected) {
        notification.warning({
          message: t(
            "sidepanel:notification.noSelectionForNotes",
            "Select text to save to Notes"
          )
        })
        return
      }
      const sourceUrl = (bgMsg.payload?.pageUrl as string | undefined) || undefined
      const suggestedTitle = deriveNoteTitle(
        selected,
        bgMsg.payload?.pageTitle as string | undefined,
        sourceUrl
      )
      const rawMessageId =
        typeof bgMsg.payload?.messageId === "string" || typeof bgMsg.payload?.messageId === "number"
          ? String(bgMsg.payload.messageId)
          : null
      clearNoteCapture()
      transferFlashcards.cancel()
      transferStudyPack.cancel()
      const capture: NonNullable<typeof noteCaptureRef.current> = { controller: new AbortController() }
      noteCaptureRef.current = capture
      capture.pending = loadFlashcardsTransferSnapshot(capture.controller.signal, () => {
        if (noteCaptureRef.current === capture) resetNoteModal()
      }).then(snapshot => {
        if (noteCaptureRef.current !== capture) {
          snapshot.release()
          throw new DOMException("The source capture was cancelled.", "AbortError")
        }
        capture.release = snapshot.release
        return snapshot
      })
      void capture.pending.catch(error => {
        if (noteCaptureRef.current !== capture) return
        if (error instanceof Error && error.name === "AbortError") resetNoteModal()
        else setNoteError("The source account could not be verified. Reconnect and capture it again; your draft is unchanged.")
      })
      setNoteDraftContent(selected)
      setNoteSuggestedTitle(suggestedTitle)
      setNoteDraftTitle(suggestedTitle)
      setNoteSourceUrl(sourceUrl)
      setNoteSourceMessageId(rawMessageId)
      setNoteSourceConversationId(rawMessageId ? serverChatId || null : null)
      setNoteSaving(false)
      setNoteError(null)
        setNoteModalOpen(true)
      return
    }

    if (bgMsg.type === "narrate-selection") {
      lastBgMsgRef.current = bgMsg
      const selected = (bgMsg.text || bgMsg.payload?.selectionText || "").trim()
      if (!selected) {
        return
      }
      if (isNarrating) {
        cancelNarration()
      }
      void speak({ utterance: selected })
      return
    }

    if (bgMsg.type === "media-ingest-status") {
      lastBgMsgRef.current = bgMsg
      const payload = (bgMsg.payload || {}) as Record<string, unknown>
      const funnelId = String(payload.funnelId || "").trim()
      const rawStatus = String(payload.status || "").trim().toLowerCase()
      if (!funnelId) return
      if (
        rawStatus !== "queued" &&
        rawStatus !== "running" &&
        rawStatus !== "completed" &&
        rawStatus !== "failed" &&
        rawStatus !== "cancelled" &&
        rawStatus !== "auth_required"
      ) {
        return
      }
      const nextStatus = rawStatus as IngestCardStatus
      const mediaId = Number(payload.mediaId)
      const progressPercent = Number(payload.progressPercent)
      const timestampSeconds = Number(payload.timestampSeconds)
      const progressMessage = String(payload.progressMessage || "").trim()
      const error = String(payload.error || "").trim()
      const url = String(payload.url || "").trim()
      const jobIds = Array.isArray(payload.jobIds)
        ? payload.jobIds
            .map((value) => Number(value))
            .filter((value) => Number.isFinite(value) && value > 0)
            .map((value) => Math.trunc(value))
        : []
      const canCancel = Boolean(payload.canCancel)
      const canRetry = Boolean(payload.canRetry)
      setIngestCard((prev) => ({
        funnelId,
        status: nextStatus,
        url: url || prev?.url,
        progressPercent: Number.isFinite(progressPercent)
          ? Math.max(0, Math.min(100, progressPercent))
          : prev?.progressPercent,
        progressMessage: progressMessage || undefined,
        error: error || undefined,
        mediaId:
          Number.isFinite(mediaId) && mediaId > 0 ? Math.trunc(mediaId) : prev?.mediaId,
        jobIds: jobIds.length > 0 ? jobIds : prev?.jobIds || [],
        canCancel,
        canRetry,
        starterQuestions:
          nextStatus === "completed" ? prev?.starterQuestions || [] : [],
        timestampSeconds:
          Number.isFinite(timestampSeconds) && timestampSeconds >= 0
            ? Math.trunc(timestampSeconds)
            : prev?.timestampSeconds
      }))
      return
    }

    if (bgMsg.type === "media-ingest-ready") {
      lastBgMsgRef.current = bgMsg
      const payload = (bgMsg.payload || {}) as Record<string, unknown>
      const mediaId = Number(bgMsg.payload?.mediaId)
      const funnelId = String(payload.funnelId || "").trim()
      const sourceUrl = (payload.url as string | undefined) || ""
      const timestampSeconds = Number(payload.timestampSeconds)
      const hasTimestamp =
        Number.isFinite(timestampSeconds) && timestampSeconds >= 0
          ? Math.trunc(timestampSeconds)
          : undefined
      const starterQuestions = buildStarterQuestions({
        url: sourceUrl,
        timestampSeconds: hasTimestamp
      })
      if (Number.isFinite(mediaId) && mediaId > 0) {
        setChatMode("rag")
        setRagMediaIds([Math.trunc(mediaId)])
        if (funnelId) {
          ingestFunnelRef.current = {
            funnelId,
            baselineUserMessages: userMessageCount,
            tracked: false,
            mediaId: Math.trunc(mediaId)
          }
        }
      }
      setIngestCard({
        funnelId: funnelId || `media-${Date.now()}`,
        status: "completed",
        url: sourceUrl || undefined,
        progressPercent: 100,
        progressMessage: "Media is ready. Pick a starter question or ask your own.",
        error: undefined,
        mediaId:
          Number.isFinite(mediaId) && mediaId > 0 ? Math.trunc(mediaId) : undefined,
        jobIds: [],
        canCancel: false,
        canRetry: false,
        starterQuestions,
        timestampSeconds: hasTimestamp
      })
      if (starterQuestions[0]) {
        seedComposerMessage(starterQuestions[0], { ifEmptyOnly: true })
      } else {
        try {
          window.dispatchEvent(new CustomEvent("tldw:focus-composer"))
        } catch {
          // ignore
        }
      }
      notification.success({
        message: t(
          "sidepanel:notification.mediaIngestReadyTitle",
          "Media ready for chat"
        ),
        description: sourceUrl
          ? t(
              "sidepanel:notification.mediaIngestReadyWithUrl",
              "Ready to chat about: {{url}}",
              { url: sourceUrl }
            )
          : t(
              "sidepanel:notification.mediaIngestReadyDescription",
              "Your media finished processing. Ask questions in chat."
            )
      })
      return
    }

    if (streaming) return

    lastBgMsgRef.current = bgMsg

    if (bgMsg.type === "transcription" || bgMsg.type === "transcription+summary") {
      const transcript = (bgMsg.payload?.transcript || bgMsg.text || "").trim()
      const summaryText = (bgMsg.payload?.summary || "").trim()
      const url = (bgMsg.payload?.url as string | undefined) || ""
      const label =
        bgMsg.type === "transcription+summary"
          ? t("sidepanel:notification.transcriptionSummaryTitle", "Transcription + summary")
          : t("sidepanel:notification.transcriptionTitle", "Transcription")
      const parts: string[] = []
      if (url) {
        parts.push(`${t("sidepanel:notification.sourceLabel", "Source")}: ${url}`)
      }
      if (transcript) {
        parts.push(`${t("sidepanel:notification.transcriptLabel", "Transcript")}:\n${transcript}`)
      }
      if (summaryText) {
        parts.push(`${t("sidepanel:notification.summaryLabel", "Summary")}:\n${summaryText}`)
      }
      const messageBody =
        parts.filter(Boolean).join("\n\n") ||
        t(
          "sidepanel:notification.transcriptionFallback",
          "Transcription completed. Open Media in the Web UI to view it."
        )
      const id = generateID()
      setMessages((prev) => [
        ...prev,
        { isBot: true, name: label, message: messageBody, sources: [], id }
      ])
      setHistory([...history, { role: "assistant", content: messageBody }])
      return
    }

    if (selectedModel) {
      onSubmit({
        message: bgMsg.text,
        messageType: bgMsg.type,
        image: ""
      })
    } else {
      notification.error({
        message: t("formError.noModel")
      })
    }
  }, [
    bgMsg,
    clearNoteCapture,
    resetNoteModal,
    transferFlashcards,
    transferStudyPack,
    serverChatId,
    streaming,
    selectedModel,
    onSubmit,
    notification,
    t,
    setMessages,
    setHistory,
    history,
    setNoteDraftContent,
    setNoteSuggestedTitle,
    setNoteDraftTitle,
    setNoteSourceUrl,
    setNoteSaving,
    setNoteError,
    setNoteModalOpen,
    setChatMode,
    setRagMediaIds,
    setIngestCard,
    seedComposerMessage,
    userMessageCount,
    cancelNarration,
    isNarrating,
    speak
  ])

  const draftKey = activeTabId
    ? `tldw:sidepanelChatDraft:${activeTabId}`
    : "tldw:sidepanelChatDraft"

  const activeTabLabel = React.useMemo(() => {
    const active = tabs.find((tab) => tab.id === activeTabId)
    if (active?.label) return active.label
    return newChatLabel
  }, [activeTabId, tabs, newChatLabel])

  const commandPaletteChats = React.useMemo(
    () =>
      [...tabs]
        .sort((a, b) => b.updatedAt - a.updatedAt)
        .map((tab) => ({
          id: tab.id,
          label: tab.label || t("common:untitled", "Untitled")
        })),
    [tabs, t]
  )
  const selectedCharacterIdForHandoff =
    selectedCharacter?.id == null ? null : String(selectedCharacter.id)
  const characterChatFullAppPath = React.useMemo(
    () =>
      buildSidepanelFullAppChatPath({
        selectedAssistant,
        selectedCharacterId: selectedCharacterIdForHandoff
      }),
    [selectedAssistant, selectedCharacterIdForHandoff]
  )
  const openCharacterChatInFullAppCommand = React.useMemo<CommandItem[]>(
    () => {
      if (characterChatFullAppPath === "/chat") return []
      return [
        {
          id: "action-open-character-chat-full-app",
          label: t(
            "common:commandPalette.openCharacterChatFullApp",
            "Open Character Chat in full app"
          ),
          description: t(
            "common:commandPalette.openCharacterChatFullAppDesc",
            "Continue the active Character, Persona, or role-play setup in /chat."
          ),
          icon: <ExternalLink className="size-4" />,
          action: () => openOptionsHashRoute(characterChatFullAppPath),
          category: "action",
          keywords: ["character", "role-play", "persona", "full app", "chat"]
        }
      ]
    },
    [characterChatFullAppPath, openOptionsHashRoute, t]
  )
  const ingestStatusLabel = React.useMemo(() => {
    const status = ingestCard?.status
    switch (status) {
      case "queued":
        return t("sidepanel:ingest.queued", "Queued")
      case "running":
        return t("sidepanel:ingest.running", "Processing")
      case "completed":
        return t(
          "sidepanel:ingest.completed",
          getDesignSystemState("ready")?.label ?? ""
        )
      case "failed":
        return t("sidepanel:ingest.failed", "Failed")
      case "cancelled":
        return t("sidepanel:ingest.cancelled", "Cancelled")
      case "auth_required":
        return t("sidepanel:ingest.authRequired", "Auth required")
      default:
        return t("sidepanel:ingest.queued", "Queued")
    }
  }, [ingestCard?.status, t])
  const ingestStatusToneClass = React.useMemo(() => {
    const status = ingestCard?.status
    if (status === "completed") return "border-success/30 bg-success/10 text-success"
    if (status === "failed" || status === "cancelled" || status === "auth_required") {
      return "border-danger/30 bg-danger/10 text-danger"
    }
    return "border-primary/30 bg-primary/10 text-primary"
  }, [ingestCard?.status])

  const isDockedSidebar = uiMode === "pro" && !isNarrow
  const isSidebarVisible = isDockedSidebar || sidebarOpen
  const messagePadding = uiMode === "pro" ? "px-4" : "px-6"
  const artifactsOpen = useArtifactsStore((state) => state.isOpen)
  const closeArtifacts = useArtifactsStore((state) => state.closeArtifact)

  return (
    <div
      className="flex h-dvh w-full min-w-0 overflow-x-hidden"
      data-testid="chat-workspace"
    >
      {isSidebarVisible && (
        <Suspense fallback={null}>
          <LazySidepanelChatSidebar
            open={isSidebarVisible}
            variant={isDockedSidebar ? "docked" : "overlay"}
            owner={owner}
          tabs={tabs}
            activeTabId={activeTabId}
            onSelectTab={handleSelectTab}
            onCloseTab={handleCloseTab}
            onNewTab={handleNewTab}
            searchQuery={sidebarSearchQuery}
            onSearchQueryChange={setSidebarSearchQuery}
            searchInputRef={sidebarSearchInputRef}
            focusSearchTrigger={sidebarSearchFocusNonce}
            onOpenLocalHistory={openLocalHistory}
            onOpenServerChat={openServerChat}
            onClose={() => setSidebarOpen(false)}
          />
        </Suspense>
      )}
      {!isDockedSidebar && sidebarOpen && (
        <div
          className="fixed inset-0 z-30 bg-black/30"
          onClick={() => setSidebarOpen(false)}
          aria-hidden="true"
        />
      )}
      <main
        className="relative flex h-dvh min-w-0 flex-1 flex-col overflow-x-hidden bg-bg"
        data-testid="chat-main"
      >
        <div className="relative z-20 w-full">
          <SidepanelHeaderSimple
            sidebarOpen={sidebarOpen}
            setSidebarOpen={setSidebarOpen}
            activeTitle={activeTabLabel}
            onRenameTitle={handleRenameActiveTab}
          />
          {messages.length > 0 ? <ConnectionBanner className="pt-12" /> : null}
        </div>
        <div
          ref={drop}
          data-testid="chat-dropzone"
          className={`relative flex min-h-0 min-w-0 flex-1 flex-col items-center overflow-x-hidden bg-bg ${
            dropState === "dragging" ? "bg-surface2" : ""
          }`}
          style={{
            "--chat-message-opacity": String(chatMessageOpacityAlpha),
            "--chat-character-image-opacity": String(
              chatCharacterImageOpacityAlpha
            ),
            ...(chatBackgroundImage
              ? {
                  backgroundImage: `url(${chatBackgroundImage})`,
                  backgroundSize: "cover",
                  backgroundPosition: "center",
                  backgroundRepeat: "no-repeat"
                }
              : {})
          } as React.CSSProperties}>
          {/* Keep themed background images visible while preserving text contrast. */}
          {chatBackgroundImage && (
            <div
              className="pointer-events-none absolute inset-0 backdrop-blur-[1px]"
              style={{
                backgroundColor: `rgb(var(--color-bg) / ${chatWindowOpacityAlpha})`
              }}
            />
          )}

          {dropState === "dragging" && (
            <div className="pointer-events-none absolute inset-0 z-30 flex flex-col items-center justify-center">
              <div className="rounded-2xl border border-dashed border-white/50 bg-black/70 px-5 py-3 text-center text-sm font-medium text-white shadow-lg backdrop-blur-sm">
                {t(
                  "playground:drop.overlayInstruction",
                  "Drop the image to attach it to your next reply"
                )}
              </div>
            </div>
          )}

          {dropFeedback && (
            <div className="pointer-events-none absolute top-20 left-0 right-0 z-30 flex justify-center px-4">
              <div
                role="status"
                aria-live="polite"
                className={`max-w-lg rounded-full px-4 py-2 text-sm shadow-lg backdrop-blur-sm ${
                  dropFeedback.type === "error"
                    ? "bg-danger text-white"
                    : "bg-elevated text-text"
                }`}
              >
                {dropFeedback.message}
              </div>
            </div>
          )}

          <div
            ref={containerRef}
            role="log"
            aria-live="polite"
            aria-relevant="additions"
            aria-label={t("playground:aria.chatTranscript", "Chat messages")}
            data-testid="chat-messages"
            className={`custom-scrollbar relative z-10 flex min-w-0 flex-1 w-full flex-col items-center overflow-x-hidden overflow-y-auto ${messagePadding}`}
            style={
              {
                "--composer-padding": composerPadding,
                paddingBottom: stickyChatInput
                  ? "calc(var(--composer-padding) + env(safe-area-inset-bottom, 0px))"
                  : "0px"
              } as React.CSSProperties
            }
          >
            {ingestCard && (
              <div className="w-full max-w-5xl pt-4">
                <div className="rounded-xl border border-border bg-surface p-3 shadow-sm">
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0">
                      <div className="text-[11px] font-semibold uppercase tracking-wide text-text-subtle">
                        {t("sidepanel:ingest.title", "Media ingest")}
                      </div>
                      <div className="mt-1 flex items-center gap-2">
                        <span
                          className={`inline-flex rounded-full border px-2 py-0.5 text-[11px] font-semibold ${ingestStatusToneClass}`}
                        >
                          {ingestStatusLabel}
                        </span>
                        {typeof ingestCard.progressPercent === "number" && (
                          <span className="text-xs text-text-subtle">
                            {Math.max(0, Math.min(100, Math.trunc(ingestCard.progressPercent)))}%
                          </span>
                        )}
                      </div>
                      {ingestCard.url && (
                        <div className="mt-1 truncate text-xs text-text-muted">
                          {ingestCard.url}
                        </div>
                      )}
                    </div>
                    <button
                      type="button"
                      onClick={() => setIngestCard(null)}
                      className="rounded-md border border-border px-2 py-1 text-xs text-text-muted transition hover:bg-surface2"
                    >
                      {t("common:dismiss", "Dismiss")}
                    </button>
                  </div>

                  {typeof ingestCard.progressPercent === "number" && (
                    <div className="mt-3 h-1.5 w-full overflow-hidden rounded-full bg-surface2">
                      <div
                        className="h-full bg-primary transition-all"
                        style={{
                          width: `${Math.max(
                            2,
                            Math.min(100, ingestCard.progressPercent)
                          )}%`
                        }}
                      />
                    </div>
                  )}

                  {ingestCard.progressMessage && (
                    <div className="mt-2 text-xs text-text-muted">
                      {ingestCard.progressMessage}
                    </div>
                  )}
                  {ingestCard.error && (
                    <div className="mt-2 text-xs text-danger">{ingestCard.error}</div>
                  )}

                  {ingestCard.starterQuestions.length > 0 &&
                    ingestCard.status === "completed" && (
                      <div className="mt-3 space-y-2">
                        <div className="text-xs font-medium text-text-subtle">
                          {t(
                            "sidepanel:ingest.starterQuestions",
                            "Starter questions"
                          )}
                        </div>
                        <div className="flex flex-wrap gap-2">
                          {ingestCard.starterQuestions.slice(0, 3).map((question) => (
                            <button
                              key={question}
                              type="button"
                              onClick={() => seedComposerMessage(question)}
                              className="rounded-full border border-border bg-surface2 px-3 py-1 text-xs text-text transition hover:bg-surface"
                            >
                              {question}
                            </button>
                          ))}
                        </div>
                      </div>
                    )}

                  <div className="mt-3 flex flex-wrap gap-2">
                    {ingestCard.status === "auth_required" && (
                      <button
                        type="button"
                        onClick={handleOpenAuthSettings}
                        className="rounded-md border border-border px-3 py-1 text-xs text-text transition hover:bg-surface2"
                      >
                        {t("sidepanel:ingest.openAuthSettings", "Open auth settings")}
                      </button>
                    )}
                    {ingestCard.canCancel && (
                      <button
                        type="button"
                        onClick={handleCancelIngest}
                        className="rounded-md border border-border px-3 py-1 text-xs text-text transition hover:bg-surface2"
                      >
                        {t("common:cancel", "Cancel")}
                      </button>
                    )}
                    {ingestCard.canRetry && (
                      <button
                        type="button"
                        onClick={handleRetryIngest}
                        className="rounded-md border border-border px-3 py-1 text-xs text-text transition hover:bg-surface2"
                      >
                        {t("common:retry", "Retry")}
                      </button>
                    )}
                  </div>
                </div>
              </div>
            )}
            {isRestoringChat && restoreFailed ? (
              <div role="alert" className="flex flex-col items-center gap-3 px-4 py-8">
                <p>{t("sidepanel:chat.restoreFailed", "Could not restore your saved chat. Retry to load it.")}</p>
                <button
                  type="button"
                  className="rounded-md border border-border px-3 py-1 text-sm text-text hover:bg-surface2"
                  onClick={() => void restoreSidepanelState()}>
                  {t("common:retry", "Retry")}
                </button>
              </div>
            ) : isRestoringChat ? (
              <div
                className="relative flex w-full flex-col items-center pt-16 pb-4"
                aria-busy="true"
                aria-label={t("sidepanel:chat.restoringChat", "Restoring previous chat")}>
                <div className="w-full max-w-5xl space-y-4">
                  {[1, 2, 3].map((i) => (
                    <div key={i} className="flex gap-4 animate-pulse">
                      <div className="w-8 h-8 rounded-full bg-surface2 flex-shrink-0"></div>
                      <div className="flex-1 space-y-2">
                        <div className="h-4 bg-surface2 rounded w-1/4"></div>
                        <div className="h-4 bg-surface2 rounded w-full"></div>
                        <div className="h-4 bg-surface2 rounded w-3/4"></div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <SidePanelBody
                scrollParentRef={containerRef}
                searchQuery={sidebarSearchQuery}
                inputRef={textareaRef}
                timelineAction={timelineAction}
                onTimelineActionHandled={() => setTimelineAction(null)}
              />
            )}
            {!isRestoringChat && !stickyChatInput && (
              <div className="w-full min-w-0 pt-4 pb-6">
                <SidepanelForm
                  key={activeTabId || "sidepanel-chat"}
                  dropedFile={dropedFile}
                  inputRef={textareaRef}
                  onHeightChange={setComposerHeight}
                  draftKey={draftKey}
                />
              </div>
            )}
          </div>

          {!isAutoScrollToBottom && (
            <div
              className="fixed z-20 left-0 right-0 flex justify-center"
              style={{ bottom: scrollToLatestBottom }}
            >
              <button
                onClick={() => autoScrollToBottom()}
                aria-label={t("playground:composer.scrollToLatest", "Scroll to latest messages")}
                title={t("playground:composer.scrollToLatest", "Scroll to latest messages") as string}
                data-testid="chat-scroll-latest"
                className="bg-surface shadow border border-border p-1.5 rounded-full pointer-events-auto hover:bg-surface2 transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-warn">
                <ChevronDown className="size-4 text-text-muted" aria-hidden="true" />
              </button>
            </div>
          )}
          {!isRestoringChat && stickyChatInput && (
            <div
              className="absolute bottom-0 left-0 right-0 z-10 w-full min-w-0"
              style={{ paddingBottom: "env(safe-area-inset-bottom, 0px)" }}
            >
              <SidepanelForm
                key={activeTabId || "sidepanel-chat"}
                dropedFile={dropedFile}
                inputRef={textareaRef}
                onHeightChange={setComposerHeight}
                draftKey={draftKey}
              />
            </div>
          )}
        </div>
      </main>
      {artifactsOpen && (
        <>
          <button
            type="button"
            aria-label={t("common:close", "Close")}
            onClick={closeArtifacts}
            className="fixed inset-0 z-40 bg-black/40"
            title={t("common:close", "Close")}
          />
          <div className="fixed inset-y-0 right-0 z-50 w-full max-w-[520px]">
            <Suspense
              fallback={
                <div className="flex h-full items-center justify-center border-l border-border bg-surface text-sm text-text-muted">
                  Loading artifacts...
                </div>
              }
            >
              <ArtifactsPanel />
            </Suspense>
          </div>
        </>
      )}
      {noteModalOpen ? (
        <Suspense fallback={null}>
          <LazyNoteQuickSaveModal
            open={noteModalOpen}
            title={noteDraftTitle}
            content={noteDraftContent}
            suggestedTitle={noteSuggestedTitle}
            sourceUrl={noteSourceUrl}
            loading={noteSaving}
            error={noteError}
            onTitleChange={handleNoteTitleChange}
            onContentChange={handleNoteContentChange}
            onCancel={resetNoteModal}
            onSave={handleNoteSave}
            onGenerateFlashcards={handleGenerateFlashcardsFromSelection}
            onCreateStudyPack={noteSourceMessageId ? handleCreateStudyPackFromSelection : undefined}
            modalTitle={t("sidepanel:notes.saveToNotesTitle", "Save to Notes")}
            saveText={t("common:save", "Save")}
            cancelText={t("common:cancel", "Cancel")}
            generateFlashcardsText={t(
              "sidepanel:notes.generateFlashcards",
              "Generate flashcards"
            )}
            createStudyPackText={t(
              "sidepanel:notes.createStudyPack",
              "Create study pack"
            )}
            titleLabel={t("sidepanel:notes.titleLabel", "Title")}
            contentLabel={t("sidepanel:notes.contentLabel", "Content")}
            titleRequiredText={t("sidepanel:notes.titleRequired", "Title is required to create a note.")}
            helperText={t("sidepanel:notes.helperText", "Review or edit the selected text, then Save or Cancel.")}
            sourceLabel={t("sidepanel:notes.sourceLabel", "Source")}
          />
        </Suspense>
      ) : null}
      <CommandPaletteHost
        commandPaletteProps={{
          scope: "sidepanel",
          onNewChat: clearChat,
          onToggleRag: toggleChatMode,
          onToggleWebSearch: toggleWebSearchMode,
          onIngestPage: () => {
            requestQuickIngestOpen()
          },
          onSwitchModel: () => {
            if (typeof window !== "undefined") {
              window.dispatchEvent(new CustomEvent("tldw:open-model-settings"))
            }
          },
          onToggleSidebar: toggleSidebar,
          onSearchHistory: requestSidebarSearchFocus,
          onSwitchChat: handleSelectTab,
          sidepanelChats: commandPaletteChats,
          additionalCommands: openCharacterChatInFullAppCommand
        }}
      />
      <Suspense fallback={null}>
        <TimelineModal />
      </Suspense>
    </div>
  )
}

export default SidepanelChat
