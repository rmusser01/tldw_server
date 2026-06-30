import {
  formatToChatHistory,
  formatToMessage,
  getTitleById,
  getRecentChatFromCopilot,
  generateID,
  getFullChatData,
  getHistoryByServerChatId,
  getHistoriesWithMetadata,
  saveHistory,
  saveMessage
} from "@/db/dexie/helpers"
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
import { copilotResumeLastChat } from "@/services/app"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import type { ServerChatMessage as ApiServerChatMessage } from "@/services/tldw/TldwApiClient"
import { createSafeStorage } from "@/utils/safe-storage"
import { CHAT_BACKGROUND_IMAGE_SETTING } from "@/services/settings/ui-settings"
import { useStorage } from "@plasmohq/storage/hook"
import { browser } from "wxt/browser"
import { ChevronDown } from "lucide-react"
import React, { lazy, Suspense } from "react"
import { useTranslation } from "react-i18next"
import { SidePanelBody } from "~/components/Sidepanel/Chat/body"
import { SidepanelForm } from "~/components/Sidepanel/Chat/form"
import { SidepanelHeaderSimple } from "~/components/Sidepanel/Chat/SidepanelHeaderSimple"
import { ConnectionBanner } from "~/components/Sidepanel/Chat/ConnectionBanner"
import { SidepanelChatSidebar } from "~/components/Sidepanel/Chat/Sidebar"
import NoteQuickSaveModal from "~/components/Sidepanel/Notes/NoteQuickSaveModal"
import { useMessage } from "~/hooks/useMessage"
import { useSelectedCharacter } from "@/hooks/useSelectedCharacter"
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
import { useUiModeStore } from "@/store/ui-mode"
import { useArtifactsStore } from "@/store/artifacts"
import { ArtifactsPanel } from "@/components/Sidepanel/Chat/ArtifactsPanel"
import { normalizeConversationState } from "@/utils/conversation-state"
import { normalizeChatRole } from "@/utils/normalize-chat-role"
import type { ServerChatHistoryItem } from "@/hooks/useServerChatHistory"
import {
  OPEN_HISTORY_EVENT,
  TIMELINE_ACTION_EVENT,
  type OpenHistoryDetail,
  type TimelineActionDetail
} from "@/utils/timeline-actions"
import {
  buildSidepanelChatWebUiHandoffUrl,
  SIDEPANEL_CHAT_WEBUI_HANDOFF_SOURCE,
  type SidepanelChatWebUiConfig
} from "@/services/tldw/sidepanel-chat-webui-handoff"

// Lazy-load Timeline to reduce initial bundle size (~1.2MB cytoscape)
const TimelineModal = lazy(() =>
  import("@/components/Timeline").then((m) => ({ default: m.TimelineModal }))
)
const CommandPalette = lazy(() =>
  import("@/components/Common/CommandPalette").then((m) => ({
    default: m.CommandPalette
  }))
)
import type { ChatHistory, Message as ChatMessage } from "~/store/option"

type ServerChatMessageInput = Omit<ApiServerChatMessage, "id"> & {
  id: string | number
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
        (meta?.modelImage as string | undefined)
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
type ChatModelSettingsState = ReturnType<typeof useStoreChatModelSettings.getState>

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value)

const getErrorMessage = (error: unknown): string =>
  error instanceof Error ? error.message : typeof error === "string" ? error : ""

const getTabsStorageKey = (id: number | null | undefined) =>
  id != null ? `sidepanelChatTabsState:tab-${id}` : "sidepanelChatTabsState"
const getLegacyStorageKey = (id: number | null | undefined) =>
  id != null ? `sidepanelChatState:tab-${id}` : "sidepanelChatState"

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
      store.updateSetting(key, value as ChatModelSettingsState[ModelSettingsKey])
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
  useServerOnline()
  const drop = React.useRef<HTMLDivElement>(null)
  const [dropedFile, setDropedFile] = React.useState<File | undefined>()
  const [sidebarOpen, setSidebarOpen] = React.useState(false)
  const [sidebarSearchQuery, setSidebarSearchQuery] = React.useState("")
  const sidebarSearchInputRef = React.useRef<HTMLInputElement>(null)
  const [sidebarSearchFocusNonce, setSidebarSearchFocusNonce] = React.useState(0)
  const [composerHeight, setComposerHeight] = React.useState(0)
  const { t } = useTranslation(["playground", "sidepanel", "common"])
  const notification = useAntdNotification()
  React.useEffect(() => {
    void tldwClient.initialize().catch(() => null)
  }, [])
  // Per-tab storage (Chrome side panel) or per-window/global (Firefox sidebar).
  // tabId: undefined = not resolved yet, null = resolved but unavailable.
  const [tabId, setTabId] = React.useState<number | null | undefined>(undefined)
  const [isRestoringChat, setIsRestoringChat] = React.useState(false)
  const storageRef = React.useRef(
    createSafeStorage({
      area: "local"
    })
  )
  const restoredTabIdRef = React.useRef<number | null | undefined>(undefined)
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
  const [selectedCharacter, setSelectedCharacter] =
    useSelectedCharacter<Character | null>(null)
  const tabs = useSidepanelChatTabsStore((state) => state.tabs)
  const activeTabId = useSidepanelChatTabsStore((state) => state.activeTabId)
  const modelSettingsSnapshot = useStoreChatModelSettings((state) =>
    pickChatModelSettings(state)
  )
  const [timelineAction, setTimelineAction] =
    React.useState<TimelineActionDetail | null>(null)
  const isSwitchingTabRef = React.useRef(false)
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
  const [noteSaving, setNoteSaving] = React.useState(false)
  const [noteError, setNoteError] = React.useState<string | null>(null)
  const textareaRef = React.useRef<HTMLTextAreaElement>(null)
  const [stickyChatInput] = useStorage(
    "stickyChatInput",
    DEFAULT_CHAT_SETTINGS.stickyChatInput
  )
  const [userDisplayName] = useStorage("chatUserDisplayName", "")
  const setHistoryWithUpdater = setHistory as (
    historyOrUpdater: ChatHistory | ((prev: ChatHistory) => ChatHistory)
  ) => void

  useCharacterGreeting({
    playgroundReady: !isRestoringChat,
    selectedCharacter: selectedCharacter ?? null,
    serverChatId,
    messagesLength: messages.length,
    setMessages,
    setHistory: setHistoryWithUpdater,
    setSelectedCharacter
  })
  const composerPadding = composerHeight
    ? `${composerHeight + 16}px`
    : `${DEFAULT_COMPOSER_HEIGHT}px`
  const scrollToLatestBottom = React.useMemo(() => {
    if (!stickyChatInput) return "8rem"
    const baseOffset = composerHeight
      ? composerHeight + 24
      : DEFAULT_COMPOSER_HEIGHT
    return `${Math.max(baseOffset, 128)}px`
  }, [composerHeight, stickyChatInput])

  const resetNoteModal = React.useCallback(() => {
    setNoteModalOpen(false)
    setNoteDraftContent("")
    setNoteDraftTitle("")
    setNoteSuggestedTitle("")
    setNoteSourceUrl(undefined)
    setNoteSaving(false)
    setNoteError(null)
  }, [])

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
      await tldwClient.createNote(content, {
        title,
        metadata: {
          source_url: noteSourceUrl,
          origin: "context-menu"
        }
      })
      notification.success({
        message: t("sidepanel:notification.savedToNotes", "Saved to Notes")
      })
      resetNoteModal()
    } catch (error: unknown) {
      const msg = getErrorMessage(error) || "Failed to save note"
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
      setQueuedMessages(snapshot.queuedMessages ?? [])
      setIsFirstMessage((snapshot.history || []).length === 0)
      setIsLoading(false)
      setIsProcessing(false)
      setIsEmbedding(false)
      setStreaming(false)
      setIsSearchingInternet(false)
      applyChatModelSettingsSnapshot(snapshot.modelSettings)
    },
    [
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
    if (!currentTabId) return
    const snapshot = buildSnapshot()
    snapshot.modelSettings = pickChatModelSettings(
      useStoreChatModelSettings.getState()
    )
    useSidepanelChatTabsStore.getState().setSnapshot(currentTabId, snapshot)
  }, [buildSnapshot])

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

  const [chatBackgroundImage] = useStorage<string | null>(
    {
      key: CHAT_BACKGROUND_IMAGE_SETTING.key,
      instance: createSafeStorage()
    },
    null
  )
  const resolvedChatBackgroundImage = chatBackgroundImage ?? null
  const bgMsg = useBackgroundMessage()
  const lastBgMsgRef = React.useRef<typeof bgMsg | null>(null)

  type LegacySidepanelChatSnapshot = {
    history: ChatHistory
    messages: ChatMessage[]
    chatMode: typeof chatMode
    historyId: string | null
  }

  type SidepanelTabsState = {
    tabs: SidepanelChatTab[]
    activeTabId: string | null
    snapshotsById: Record<string, SidepanelChatSnapshot>
  }

  const restoreSidepanelState = React.useCallback(async () => {
    // Wait until we've attempted to resolve tab id so we don't
    // accidentally attach a tab-specific snapshot to the wrong key.
    if (tabId === undefined) {
      return
    }

    const storage = storageRef.current
    setIsRestoringChat(true)
    try {
      // Prefer a tab-specific snapshot; fall back to the legacy/global key
      // so existing users don't lose their last session.
      const keysToTry: string[] = [getTabsStorageKey(tabId)]
      if (tabId != null) {
        keysToTry.push(getTabsStorageKey(null))
      }

      let tabsState: SidepanelTabsState | null = null
      for (const key of keysToTry) {
        const candidate = (await storage.get(key)) as SidepanelTabsState | null
        if (candidate && Array.isArray(candidate.tabs)) {
          tabsState = candidate
          break
        }
      }

      if (tabsState && tabsState.tabs.length > 0) {
        const fallbackId = tabsState.tabs[0]?.id ?? null
        const resolvedActiveId =
          (tabsState.activeTabId &&
            tabsState.snapshotsById?.[tabsState.activeTabId] &&
            tabsState.activeTabId) ||
          fallbackId
        useSidepanelChatTabsStore
          .getState()
          .setTabsState({
            tabs: tabsState.tabs,
            activeTabId: resolvedActiveId,
            snapshotsById: tabsState.snapshotsById || {}
          })
        if (resolvedActiveId) {
          const snapshot = tabsState.snapshotsById?.[resolvedActiveId]
          if (snapshot) {
            applySnapshot(snapshot)
          }
        }
        setIsRestoringChat(false)
        return
      }

      const legacyKeysToTry: string[] = [getLegacyStorageKey(tabId)]
      if (tabId != null) {
        legacyKeysToTry.push(getLegacyStorageKey(null))
      }

      let legacySnapshot: LegacySidepanelChatSnapshot | null = null
      for (const key of legacyKeysToTry) {
        const candidate = (await storage.get(key)) as
          | LegacySidepanelChatSnapshot
          | null
        if (candidate && Array.isArray(candidate.messages)) {
          legacySnapshot = candidate
          break
        }
      }

      if (legacySnapshot && Array.isArray(legacySnapshot.messages)) {
        const restoredSnapshot: SidepanelChatSnapshot = {
          history: legacySnapshot.history || [],
          messages: legacySnapshot.messages || [],
          chatMode: legacySnapshot.chatMode || "normal",
          historyId: legacySnapshot.historyId ?? null,
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
        const initialTab: SidepanelChatTab = {
          id: generateID(),
          label: t("sidepanel:tabs.newChat", "New chat"),
          historyId: legacySnapshot.historyId ?? null,
          serverChatId: null,
          serverChatTopic: null,
          updatedAt: Date.now()
        }
        useSidepanelChatTabsStore.getState().setTabsState({
          tabs: [initialTab],
          activeTabId: initialTab.id,
          snapshotsById: { [initialTab.id]: restoredSnapshot }
        })
        applySnapshot(restoredSnapshot)
        setIsRestoringChat(false)
        return
      }
    } catch {
      // fall through to recent chat resume
    }

    try {
      const isEnabled = await copilotResumeLastChat()
      if (!isEnabled) {
        setIsRestoringChat(false)
        return
      }
      if (messages.length === 0) {
        const recentChat = await getRecentChatFromCopilot()
        if (recentChat) {
          const restoredHistory = formatToChatHistory(recentChat.messages)
          const restoredMessages = formatToMessage(recentChat.messages)
          const restoredSnapshot: SidepanelChatSnapshot = {
            history: restoredHistory,
            messages: restoredMessages,
            chatMode,
            historyId: recentChat.history.id,
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
          const initialTab: SidepanelChatTab = {
            id: generateID(),
            label: t("sidepanel:tabs.newChat", "New chat"),
            historyId: recentChat.history.id,
            serverChatId: null,
            serverChatTopic: null,
            updatedAt: Date.now()
          }
          useSidepanelChatTabsStore.getState().setTabsState({
            tabs: [initialTab],
            activeTabId: initialTab.id,
            snapshotsById: { [initialTab.id]: restoredSnapshot }
          })
          applySnapshot(restoredSnapshot)
        }
      }
    } finally {
      setIsRestoringChat(false)
    }
  }, [
    applySnapshot,
    chatMode,
    messages.length,
    modelSettingsSnapshot,
    queuedMessages,
    selectedModel,
    selectedQuickPrompt,
    selectedSystemPrompt,
    serverChatClusterId,
    serverChatExternalRef,
    serverChatId,
    serverChatSource,
    serverChatState,
    serverChatTopic,
    t,
    tabId,
    temporaryChat,
    toolChoice,
    useOCR,
    webSearch
  ])

  const persistSidepanelState = React.useCallback(() => {
    const storage = storageRef.current
    const key = getTabsStorageKey(tabId)
    saveActiveTabSnapshot()
    const { tabs, activeTabId, snapshotsById } =
      useSidepanelChatTabsStore.getState()
    const snapshot: SidepanelTabsState = {
      tabs,
      activeTabId,
      snapshotsById
    }
    void storage.set(key, snapshot).catch(() => {
      // ignore persistence errors in sidepanel
    })
  }, [saveActiveTabSnapshot, tabId])

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
      try {
        // browser is provided by the extension runtime (see wxt config).
        const resp: unknown = await browser.runtime.sendMessage({
          type: "tldw:get-tab-id"
        })
        if (isRecord(resp) && typeof resp.tabId === "number") {
          setTabId(resp.tabId)
        } else {
          setTabId(null)
        }
      } catch {
        setTabId(null)
      }
    }
    fetchTabId()
  }, [])

  React.useEffect(() => {
    if (tabId === undefined) return
    if (restoredTabIdRef.current === tabId) return
    restoredTabIdRef.current = tabId
    void restoreSidepanelState()
  }, [restoreSidepanelState, tabId])

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
    if (!activeTabId || isSwitchingTabRef.current) return
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
        label = t("sidepanel:tabs.newChat", "New chat")
      }
      if (!isCurrent) return
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
    fallbackLabel,
    historyId,
    serverChatId,
    serverChatTopic,
    t,
    truncateTabLabel
  ])

  const handleRenameActiveTab = React.useCallback(
    (nextLabel: string) => {
      const trimmed = nextLabel.trim()
      if (!activeTabId || !trimmed) return
      useSidepanelChatTabsStore.getState().renameTab(activeTabId, trimmed)
    },
    [activeTabId]
  )

  React.useEffect(() => {
    if (!activeTabId || isSwitchingTabRef.current) return
    useSidepanelChatTabsStore.getState().setSnapshot(
      activeTabId,
      buildSnapshot()
    )
  }, [activeTabId, buildSnapshot])

  React.useEffect(() => {
    if (isRestoringChat || isSwitchingTabRef.current) return
    if (tabs.length > 0 && activeTabId) return
    const initialTabId = generateID()
    const initialTab: SidepanelChatTab = {
      id: initialTabId,
      label: t("sidepanel:tabs.newChat", "New chat"),
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
    buildSnapshot,
    historyId,
    isRestoringChat,
    serverChatId,
    serverChatTopic,
    t,
    tabs.length
  ])

  const handleNewTab = React.useCallback(() => {
    saveActiveTabSnapshot()
    if (streaming) {
      stopStreamingRequest()
    }
    setDropedFile(undefined)
    const newTabId = generateID()
    useSidepanelChatTabsStore.getState().upsertTab({
      id: newTabId,
      label: t("sidepanel:tabs.newChat", "New chat"),
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
    clearChat,
    saveActiveTabSnapshot,
    setDropedFile,
    stopStreamingRequest,
    streaming,
    t
  ])

  const handleSelectTab = React.useCallback(
    (tabId: string) => {
      if (!tabId || tabId === activeTabId) return
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
    [handleNewTab, handleSelectTab]
  )

  const openSnapshotTab = React.useCallback(
    (tab: SidepanelChatTab, snapshot: SidepanelChatSnapshot) => {
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
    [applySnapshot]
  )

  const openLocalHistory = React.useCallback(
    async (targetHistoryId: string) => {
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
        if (!chatData) {
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
              historyInfo.title || t("sidepanel:tabs.newChat", "New chat")
            ),
            labelSource: "auto",
            historyId: historyInfo.id,
            serverChatId: historyInfo.server_chat_id ?? null,
            serverChatTopic: null,
            updatedAt: Date.now()
          },
          snapshot
        )
      } catch (err: unknown) {
        notification.error({
          message: t("common:error", "Error"),
          description:
            getErrorMessage(err) ||
            t("common:serverChatLoadError", "Failed to load conversation.")
        })
      } finally {
        setIsLoading(false)
      }
    },
    [
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

  const ensureLocalHistoryMirror = React.useCallback(
    async (
      chatId: string,
      chat: ServerChatHistoryItem,
      list: ServerChatMessageInput[]
    ) => {
      let localHistoryId: string | null = null
      try {
        const existingHistory = await getHistoryByServerChatId(chatId)
        if (existingHistory) {
          localHistoryId = existingHistory.id
        } else {
          const newHistory = await saveHistory(
            chat.title || t("sidepanel:tabs.newChat", "New chat"),
            false,
            "server",
            undefined,
            chatId
          )
          localHistoryId = newHistory.id
        }

        if (localHistoryId) {
          const resolvedHistoryId = localHistoryId
          const metadataMap = await getHistoriesWithMetadata([resolvedHistoryId])
          const existingMeta = metadataMap.get(resolvedHistoryId)
          if (!existingMeta || existingMeta.messageCount === 0) {
            const now = Date.now()
            const results = await Promise.allSettled(
              list.map((m, index) => {
                const meta = m as Record<string, unknown>
                const parsedCreatedAt = Date.parse(m.created_at)
                const resolvedCreatedAt = Number.isNaN(parsedCreatedAt)
                  ? now + index
                  : parsedCreatedAt
                const normalizedId = normalizeServerChatMessageId(m.id)
                const role =
                  m.role === "assistant" ||
                  m.role === "system" ||
                  m.role === "user"
                    ? m.role
                    : "user"
                const name =
                  role === "assistant"
                    ? "Assistant"
                    : role === "system"
                      ? "System"
                      : "You"
                return saveMessage({
                  id: normalizedId,
                  history_id: resolvedHistoryId,
                  name,
                  role,
                  content: m.content,
                  images: [],
                  source: [],
                  time: index,
                  message_type:
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
                  parent_message_id:
                    (meta?.parent_message_id as
                      | string
                      | null
                      | undefined) ??
                    (meta?.parentMessageId as
                      | string
                      | null
                      | undefined) ??
                    null,
                  createdAt: resolvedCreatedAt
                })
              })
            )
            const failed = results
              .map((result, index) => ({
                result,
                messageId:
                  list[index]?.id === undefined
                    ? String(index)
                    : normalizeServerChatMessageId(list[index].id)
              }))
              .filter((entry) => entry.result.status === "rejected")
            if (failed.length > 0) {
              console.warn(
                `[ensureLocalHistoryMirror] ${failed.length} messages failed to save`,
                failed.map(({ messageId, result }) => ({
                  messageId,
                  reason:
                    result.status === "rejected" ? result.reason : undefined
                }))
              )
            }
          }
        }
      } catch (err) {
        console.error("[ensureLocalHistoryMirror] Failed:", err)
      }
      return localHistoryId
    },
    [t]
  )

  const openServerChat = React.useCallback(
    async (chat: ServerChatHistoryItem) => {
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
      try {
        await tldwClient.initialize().catch(() => null)
        const list = await tldwClient.listChatMessages(chatId, {
          include_deleted: "false"
        })
        const messageList: ServerChatMessageInput[] = list
        const { history, mappedMessages } = mapServerChatMessages(
          messageList,
          userDisplayName
        )
        const localHistoryId = await ensureLocalHistoryMirror(
          chatId,
          chat,
          messageList
        )

        const snapshot: SidepanelChatSnapshot = {
          history,
          messages: mappedMessages,
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
              chat.title || t("sidepanel:tabs.newChat", "New chat")
            ),
            labelSource: "auto",
            historyId: localHistoryId,
            serverChatId: chatId,
            serverChatTopic: chat.topic_label ?? null,
            updatedAt: Date.now()
          },
          snapshot
        )
      } catch (err: unknown) {
        notification.error({
          message: t("common:error", "Error"),
          description:
            getErrorMessage(err) ||
            t("common:serverChatLoadError", "Failed to load conversation.")
        })
      } finally {
        setIsLoading(false)
      }
    },
    [
      chatMode,
      ensureLocalHistoryMirror,
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
      t,
      tabs,
      toolChoice,
      truncateTabLabel,
      userDisplayName,
      useOCR,
      webSearch
    ]
  )

  React.useEffect(() => {
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
      persistSidepanelState()
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
    const dropEl = drop.current
    if (!dropEl) {
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

    dropEl.addEventListener("dragover", handleDragOver)
    dropEl.addEventListener("drop", handleDrop)
    dropEl.addEventListener("dragenter", handleDragEnter)
    dropEl.addEventListener("dragleave", handleDragLeave)

    return () => {
      dropEl.removeEventListener("dragover", handleDragOver)
      dropEl.removeEventListener("drop", handleDrop)
      dropEl.removeEventListener("dragenter", handleDragEnter)
      dropEl.removeEventListener("dragleave", handleDragLeave)
    }
  }, [showDropFeedback, t])

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
  }, [defaultChatWithWebsite, setChatMode, setTemporaryChat, sidepanelTemporaryChat])

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
      setNoteDraftContent(selected)
      setNoteSuggestedTitle(suggestedTitle)
      setNoteDraftTitle(suggestedTitle)
      setNoteSourceUrl(sourceUrl)
      setNoteSaving(false)
      setNoteError(null)
        setNoteModalOpen(true)
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
    setNoteModalOpen
  ])

  const draftKey = activeTabId
    ? `tldw:sidepanelChatDraft:${activeTabId}`
    : "tldw:sidepanelChatDraft"

  const activeTabLabel = React.useMemo(() => {
    const active = tabs.find((tab) => tab.id === activeTabId)
    if (active?.label) return active.label
    return t("sidepanel:tabs.newChat", "New chat")
  }, [activeTabId, tabs, t])

  const readWebUiHandoffConfig = React.useCallback(async () => {
    const storage = storageRef.current
    const readStorageValue = async <T,>(key: string): Promise<T | null> => {
      try {
        const value = await storage.get<T>(key)
        return value ?? null
      } catch {
        return null
      }
    }
    const storedConfig = await readStorageValue<Record<string, unknown>>(
      "tldwConfig"
    )
    const legacyWebUiUrl =
      (await readStorageValue<string>("webUiUrl")) ??
      (await readStorageValue<string>("webuiUrl"))

    const config: SidepanelChatWebUiConfig = {
      serverUrl:
        typeof storedConfig?.serverUrl === "string"
          ? storedConfig.serverUrl
          : null,
      webUiUrl:
        typeof storedConfig?.webUiUrl === "string"
          ? storedConfig.webUiUrl
          : typeof storedConfig?.webuiUrl === "string"
            ? storedConfig.webuiUrl
            : legacyWebUiUrl
    }

    return config
  }, [])

  const openChatInWebUi = React.useCallback(async () => {
    const snapshot = buildSnapshot()
    const draft = textareaRef.current?.value ?? ""
    const config = await readWebUiHandoffConfig()
    const url = buildSidepanelChatWebUiHandoffUrl({
      config,
      payload: {
        source: SIDEPANEL_CHAT_WEBUI_HANDOFF_SOURCE,
        createdAt: Date.now(),
        draft,
        historyId: snapshot.historyId ?? null,
        serverChatId: snapshot.serverChatId ?? null,
        chatMode: snapshot.chatMode ?? "normal",
        webSearch: snapshot.webSearch ?? false,
        toolChoice: snapshot.toolChoice ?? "none",
        selectedModel: snapshot.selectedModel ?? null,
        selectedSystemPrompt: snapshot.selectedSystemPrompt ?? null,
        selectedQuickPrompt: snapshot.selectedQuickPrompt ?? null,
        temporaryChat: snapshot.temporaryChat ?? false,
        useOCR: snapshot.useOCR ?? false,
        title: activeTabLabel
      }
    })

    const openFallback = () => {
      const opened = window.open(url, "_blank")
      if (!opened) {
        throw new Error("Window popup was blocked")
      }
    }

    try {
      if (browser.tabs?.create) {
        await browser.tabs.create({ url })
        return
      }
    } catch (error) {
      console.error("Failed to open WebUI chat tab:", error)
    }

    openFallback()
  }, [activeTabLabel, buildSnapshot, readWebUiHandoffConfig])

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

  const isDockedSidebar = uiMode === "pro" && !isNarrow
  const isSidebarVisible = isDockedSidebar || sidebarOpen
  const messagePadding = uiMode === "pro" ? "px-4" : "px-6"
  const artifactsOpen = useArtifactsStore((state) => state.isOpen)
  const closeArtifacts = useArtifactsStore((state) => state.closeArtifact)

  return (
    <div className="flex h-dvh w-full" data-testid="chat-workspace">
      {isSidebarVisible && (
        <SidepanelChatSidebar
          open={isSidebarVisible}
          variant={isDockedSidebar ? "docked" : "overlay"}
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
      )}
      {!isDockedSidebar && sidebarOpen && (
        <div
          className="fixed inset-0 z-30 bg-black/30"
          onClick={() => setSidebarOpen(false)}
          aria-hidden="true"
        />
      )}
      <main className="relative flex h-dvh flex-1 flex-col bg-bg" data-testid="chat-main">
        <div className="relative z-20 w-full">
          <SidepanelHeaderSimple
            sidebarOpen={sidebarOpen}
            setSidebarOpen={setSidebarOpen}
            activeTitle={activeTabLabel}
            onRenameTitle={handleRenameActiveTab}
            onOpenChatInWebUi={openChatInWebUi}
          />
          <ConnectionBanner className="pt-12" />
        </div>
        <div
          ref={drop}
          data-testid="chat-dropzone"
          className={`relative flex min-h-0 flex-1 flex-col items-center bg-bg ${
            dropState === "dragging" ? "bg-surface2" : ""
          }`}
          style={
            resolvedChatBackgroundImage
              ? {
                  backgroundImage: `url(${resolvedChatBackgroundImage})`,
                  backgroundSize: "cover",
                  backgroundPosition: "center",
                  backgroundRepeat: "no-repeat"
                }
              : {}
          }>
          {/* Background overlay for opacity effect */}
          {resolvedChatBackgroundImage && (
            <div
              className="absolute inset-0 bg-bg"
              style={{ opacity: 0.9, pointerEvents: "none" }}
            />
          )}

          {dropState === "dragging" && (
            <div className="pointer-events-none absolute inset-0 z-30 flex flex-col items-center justify-center">
              <div className="rounded-2xl border border-dashed border-border bg-surface/90 px-5 py-3 text-center text-sm font-medium text-text shadow-lg backdrop-blur-sm">
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
                    : "border border-border bg-elevated text-text"
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
            className={`custom-scrollbar relative z-10 flex flex-1 w-full flex-col items-center overflow-x-hidden overflow-y-auto ${messagePadding}`}
            style={
              {
                "--composer-padding": composerPadding,
                paddingBottom: stickyChatInput
                  ? "calc(var(--composer-padding) + env(safe-area-inset-bottom, 0px))"
                  : "0px"
              } as React.CSSProperties
            }
          >
            {isRestoringChat ? (
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
            {!stickyChatInput && (
              <div className="w-full pt-4 pb-6">
                <SidepanelForm
                  key={activeTabId || "sidepanel-chat"}
                  dropedFile={dropedFile}
                  inputRef={textareaRef}
                  onHeightChange={setComposerHeight}
                  draftKey={draftKey}
                  onOpenChatInWebUi={openChatInWebUi}
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
                className="bg-surface shadow border border-border p-1.5 rounded-full pointer-events-auto hover:bg-elevated transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-warn">
                <ChevronDown className="size-4 text-text-muted" aria-hidden="true" />
              </button>
            </div>
          )}
          {stickyChatInput && (
            <div
              className="absolute bottom-0 w-full z-10"
              style={{ paddingBottom: "env(safe-area-inset-bottom, 0px)" }}
            >
              <SidepanelForm
                key={activeTabId || "sidepanel-chat"}
                dropedFile={dropedFile}
                inputRef={textareaRef}
                onHeightChange={setComposerHeight}
                draftKey={draftKey}
                onOpenChatInWebUi={openChatInWebUi}
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
            <ArtifactsPanel />
          </div>
        </>
      )}
      <NoteQuickSaveModal
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
        modalTitle={t("sidepanel:notes.saveToNotesTitle", "Save to Notes")}
        saveText={t("common:save", "Save")}
        cancelText={t("common:cancel", "Cancel")}
        titleLabel={t("sidepanel:notes.titleLabel", "Title")}
        contentLabel={t("sidepanel:notes.contentLabel", "Content")}
        titleRequiredText={t("sidepanel:notes.titleRequired", "Title is required to create a note.")}
        helperText={t("sidepanel:notes.helperText", "Review or edit the selected text, then Save or Cancel.")}
        sourceLabel={t("sidepanel:notes.sourceLabel", "Source")}
      />
      <Suspense fallback={null}>
        <CommandPalette
          scope="sidepanel"
          onNewChat={clearChat}
          onToggleRag={toggleChatMode}
          onToggleWebSearch={toggleWebSearchMode}
          onIngestPage={() => {
            if (typeof window !== "undefined") {
              window.dispatchEvent(new CustomEvent("tldw:open-quick-ingest"))
            }
          }}
          onSwitchModel={() => {
            if (typeof window !== "undefined") {
              window.dispatchEvent(new CustomEvent("tldw:open-model-settings"))
            }
          }}
          onToggleSidebar={toggleSidebar}
          onSearchHistory={requestSidebarSearchFocus}
          onSwitchChat={handleSelectTab}
          sidepanelChats={commandPaletteChats}
        />
      </Suspense>
      <Suspense fallback={null}>
        <TimelineModal />
      </Suspense>
    </div>
  )
}

export default SidepanelChat
