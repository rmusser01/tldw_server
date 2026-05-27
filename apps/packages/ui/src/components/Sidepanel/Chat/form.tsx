import { useMutation, useQuery } from "@tanstack/react-query"
import React from "react"
import useDynamicTextareaSize from "~/hooks/useDynamicTextareaSize"
import { useMessage } from "~/hooks/useMessage"
import {
  Checkbox,
  Dropdown,
  Input,
  InputNumber,
  Modal,
  Popover,
  Radio,
  Select,
  Space,
  Switch,
  Tooltip,
  message
} from "antd"
import { useWebUI } from "~/store/webui"
import { defaultEmbeddingModelForRag } from "~/services/tldw-server"
import {
  ImageIcon,
  MicIcon,
  StopCircleIcon,
  X,
  CornerUpLeft,
  EyeIcon,
  EyeOffIcon,
  Gauge,
  Search,
  FileText,
  Globe,
  Headphones,
  Settings2,
  UserCircle2
} from "lucide-react"
import { useTranslation } from "react-i18next"
import { getVariable } from "@/utils/select-variable"
import { useTldwStt } from "@/hooks/useTldwStt"
import { useMicStream } from "@/hooks/useMicStream"
import type { DictationModePreference } from "@/hooks/useDictationStrategy"
import { BsIncognito } from "react-icons/bs"
import { handleChatInputKeyDown } from "@/utils/key-down"
import { getIsSimpleInternetSearch } from "@/services/search"
import { useStorage } from "@plasmohq/storage/hook"
import { useSttSettings } from "@/hooks/useSttSettings"
import { useVoiceChatSettings } from "@/hooks/useVoiceChatSettings"
import { useVoiceChatStream } from "@/hooks/useVoiceChatStream"
import { useVoiceChatMessages } from "@/hooks/useVoiceChatMessages"
import { useComposerEvents } from "@/hooks/useComposerEvents"
import { useTemporaryChatToggle } from "@/hooks/useTemporaryChatToggle"
import { useSelectedCharacter } from "@/hooks/useSelectedCharacter"
import { useCanonicalConnectionConfig } from "@/hooks/useCanonicalConnectionConfig"
import { useComposerVoiceChat } from "@/components/Chat/composer/hooks/useComposerVoiceChat"
import {
  COMPOSER_CONSTANTS,
  SPACING,
  STORAGE_KEYS,
  getComposerGap
} from "@/config/ui-constants"
import { isFireFoxPrivateMode } from "@/utils/is-private-mode"
import { useFocusShortcuts } from "@/hooks/keyboard"
import { isFirefoxTarget } from "@/config/platform"
import { useComposerText } from "@/components/Chat/composer/hooks/useComposerText"
import { useComposerSubmit } from "@/components/Chat/composer/hooks/useComposerSubmit"
import { useComposerAttachments } from "@/components/Chat/composer/hooks/useComposerAttachments"
import { useSlashCommands, type SlashCommandItem } from "@/hooks/useSlashCommands"
import { useTabMentions, type TabInfo } from "~/hooks/useTabMentions"
import { useDeferredComposerInput } from "@/hooks/playground"
import { KnowledgePanel } from "@/components/Knowledge"
import { ChatQueuePanel } from "@/components/Common/ChatQueuePanel"
import { ConnectionStatusIndicator } from "@/components/Sidepanel/Chat/ConnectionStatusIndicator"
import { ControlRow } from "@/components/Sidepanel/Chat/ControlRow"
import { ContextChips } from "@/components/Sidepanel/Chat/ContextChips"
import { SlashCommandMenu } from "@/components/Sidepanel/Chat/SlashCommandMenu"
import { MentionsMenu, type MentionMenuItem } from "@/components/Sidepanel/Chat/MentionsMenu"
import {
  shouldEnableOptionalResource,
  useChatSurfaceCoordinatorStore
} from "@/store/chat-surface-coordinator"
import { ModelParamsPanel } from "@/components/Sidepanel/Chat/ModelParamsPanel"
import { CurrentChatModelSettings } from "@/components/Common/Settings/CurrentChatModelSettings"
import { ActorPopout } from "@/components/Common/Settings/ActorPopout"
import { DocumentGeneratorDrawer } from "@/components/Common/Playground/DocumentGeneratorDrawer"
import { QuickIngestWizardModal as QuickIngestModal } from "@/components/Common/QuickIngestWizardModal"
import {
  useConnectionState,
  useConnectionUxState
} from "@/hooks/useConnectionState"
import { ConnectionPhase } from "@/types/connection"
import { useServerCapabilities } from "@/hooks/useServerCapabilities"
import { useTldwAudioStatus } from "@/hooks/useTldwAudioStatus"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import {
  normalizeVoiceConversationRuntimeError,
  resolveVoiceConversationAvailability,
  resolveVoiceConversationTtsConfig,
  shouldProbeVoiceConversationAudioHealth
} from "@/services/tldw/voice-conversation"
import { fetchChatModels } from "@/services/tldw-server"
import { getProviderDisplayName } from "@/utils/provider-registry"
import { useAntdNotification } from "@/hooks/useAntdNotification"
import { useSetting } from "@/hooks/useSetting"
import { useFocusComposerOnConnect } from "@/hooks/useComposerFocus"
import { useQuickIngestStore } from "@/store/quick-ingest"
import { useQuickIngestSessionStore } from "@/store/quick-ingest-session"
import {
  createQuickIngestSessionSeedFromOpenDetail,
  type QuickIngestOpenDetail
} from "@/utils/quick-ingest-open"
import { useUiModeStore } from "@/store/ui-mode"
import { useStoreMessageOption } from "@/store/option"
import { shallow } from "zustand/shallow"
import { Button } from "@/components/Common/Button"
import { generateID } from "@/db/dexie/helpers"
import type { UploadedFile } from "@/db/dexie/types"
import type { ChatDocuments } from "@/models/ChatTypes"
import { formatFileSize } from "@/utils/format"
import { formatPinnedResults } from "@/utils/rag-format"
import { createRenderPerfTracker } from "@/utils/perf/render-profiler"
import { useComposerQueue } from "@/components/Chat/composer/hooks/useComposerQueue"
import { useConversationContextComposition } from "@/hooks/chat/useConversationContextComposition"
import { useChatSettingsRecord } from "@/hooks/chat/useChatSettingsRecord"
import { resolveEffectiveAssistantState } from "@/hooks/chat/effective-assistant-state"
import {
  buildAvailableChatModelIds,
  findUnavailableChatModel,
  normalizeChatModelId
} from "@/utils/chat-model-availability"
import { createSafeStorage } from "@/utils/safe-storage"
import {
  DEFAULT_CHARACTER_STORAGE_KEY,
  defaultCharacterStorage,
  isFreshChatState,
  resolveCharacterSelectionId,
  shouldApplyDefaultCharacter,
  shouldResetDefaultCharacterBootstrap
} from "@/utils/default-character-preference"
import { CONTEXT_FILE_SIZE_MB_SETTING } from "@/services/settings/ui-settings"
import {
  ChatComposer,
  useComposerVariantPreference,
} from "@/components/Chat/composer/ChatComposer"
import { useComposerEnabledPreference } from "@/components/Chat/composer/hooks/useComposerEnabledPreference"
import { browser } from "wxt/browser"
import type { Character } from "@/types/character"
import type { QueuedRequest } from "@/utils/chat-request-queue"
import { AudioSourcePicker } from "@/components/Common/AudioSourcePicker"
import { CharacterControlsSheet } from "@/components/Sidepanel/Chat/CharacterControlsSheet"
import { getSidepanelOverlayResumeMarkerKey } from "@/utils/sidepanel-overlay-resume"

type Props = {
  dropedFile: File | undefined
  inputRef?: React.RefObject<HTMLTextAreaElement>
  onHeightChange?: (height: number) => void
  draftKey?: string
}

type DefaultCharacterPreferenceQueryResult = {
  defaultCharacterId: string | null
}

type SidepanelQueuedSourceContext = {
  documents?: ChatDocuments
  imageBackendOverride?: string
  isImageCommand?: boolean
}

export const SidepanelForm = ({
  dropedFile,
  inputRef,
  onHeightChange,
  draftKey
}: Props) => {
  const formContainerRef = React.useRef<HTMLDivElement>(null)
  const localTextareaRef = React.useRef<HTMLTextAreaElement>(null)
  const textareaRef = inputRef ?? localTextareaRef
  const contextFileInputRef = React.useRef<HTMLInputElement>(null)
  const { sendWhenEnter, setSendWhenEnter } = useWebUI()
  const setOptionalPanelVisible = useChatSurfaceCoordinatorStore(
    (state) => state.setPanelVisible
  )
  const markOptionalPanelEngaged = useChatSurfaceCoordinatorStore(
    (state) => state.markPanelEngaged
  )
  const audioHealthEnabled = useChatSurfaceCoordinatorStore((state) =>
    shouldEnableOptionalResource(state, "audio-health")
  )
  const [typing, setTyping] = React.useState<boolean>(false)
  const [characterControlsOpen, setCharacterControlsOpen] =
    React.useState(false)
  const { t } = useTranslation(["playground", "common", "option", "sidepanel"])
  const notification = useAntdNotification()
  const [chatWithWebsiteEmbedding] = useStorage(
    "chatWithWebsiteEmbedding",
    false
  )
  const [imageBackendDefault] = useStorage("imageBackendDefault", "")
  const [storedCharacter, setStoredCharacter] =
    useSelectedCharacter<Character | null>(null)
  const [defaultCharacter, setDefaultCharacter] = useStorage<Character | null>(
    {
      key: DEFAULT_CHARACTER_STORAGE_KEY,
      instance: defaultCharacterStorage
    },
    null
  )
  const { data: defaultCharacterPreference } = useQuery<DefaultCharacterPreferenceQueryResult>({
    queryKey: ["tldw:defaultCharacterPreference:chat"],
    queryFn: async () => {
      await tldwClient.initialize()
      const defaultCharacterId = await tldwClient.getDefaultCharacterPreference()
      return { defaultCharacterId }
    },
    staleTime: 60 * 1000,
    throwOnError: false
  })
  const [contextFileMaxSizeMb] = useSetting(CONTEXT_FILE_SIZE_MB_SETTING)
  const maxContextFileSizeBytes = React.useMemo(
    () => contextFileMaxSizeMb * 1024 * 1024,
    [contextFileMaxSizeMb]
  )
  const maxContextFileSizeLabel = React.useMemo(
    () => formatFileSize(maxContextFileSizeBytes),
    [maxContextFileSizeBytes]
  )
  const imageBackendDefaultTrimmed = React.useMemo(
    () => (imageBackendDefault || "").trim(),
    [imageBackendDefault]
  )
  const { data: voiceChatModels } = useQuery({
    queryKey: ["voiceChatModels"],
    queryFn: async () => fetchChatModels({ returnEmpty: true }),
    enabled: audioHealthEnabled
  })
  const voiceChatModelOptions = React.useMemo(() => {
    const options = [
      {
        value: "",
        label: t("playground:voiceChat.useChatModel", "Use chat model")
      }
    ]
    for (const model of voiceChatModels || []) {
      const providerLabel = getProviderDisplayName(model.provider || "")
      const modelLabel = model.nickname || model.model || model.name
      const label = providerLabel
        ? `${providerLabel} - ${modelLabel}`
        : modelLabel
      options.push({
        value: model.model || model.name,
        label
      })
    }
    return options
  }, [voiceChatModels, t])
  const availableChatModelIds = React.useMemo(
    () => buildAvailableChatModelIds(voiceChatModels as any[]),
    [voiceChatModels]
  )
  // STT settings consolidated into a single hook
  const sttSettings = useSttSettings()
  const {
    voiceChatEnabled,
    setVoiceChatEnabled,
    voiceChatModel,
    setVoiceChatModel,
    voiceChatPauseMs,
    setVoiceChatPauseMs,
    voiceChatTriggerPhrases,
    setVoiceChatTriggerPhrases,
    voiceChatAutoResume,
    setVoiceChatAutoResume,
    voiceChatBargeIn,
    setVoiceChatBargeIn,
    voiceChatTtsMode,
    setVoiceChatTtsMode
  } = useVoiceChatSettings()
  const voiceChatMessages = useVoiceChatMessages()
  const { config: canonicalConnectionConfig, loading: canonicalConnectionLoading } =
    useCanonicalConnectionConfig()
  const [ttsProvider] = useStorage("ttsProvider", "browser")
  const [tldwTtsModel] = useStorage("tldwTtsModel", "kokoro")
  const [tldwTtsVoice] = useStorage("tldwTtsVoice", "af_heart")
  const [tldwTtsSpeed] = useStorage("tldwTtsSpeed", 1)
  const [tldwTtsResponseFormat] = useStorage("tldwTtsResponseFormat", "mp3")
  const [openAITTSModel] = useStorage("openAITTSModel", "tts-1")
  const [openAITTSVoice] = useStorage("openAITTSVoice", "alloy")
  const [elevenLabsModel] = useStorage("elevenLabsModel", "")
  const [elevenLabsVoiceId] = useStorage("elevenLabsVoiceId", "")
  const [speechPlaybackSpeed] = useStorage("speechPlaybackSpeed", 1)
  const [voiceChatTriggerInput, setVoiceChatTriggerInput] = React.useState(
    voiceChatTriggerPhrases.join(", ")
  )
  React.useEffect(() => {
    setVoiceChatTriggerInput(voiceChatTriggerPhrases.join(", "))
  }, [voiceChatTriggerPhrases])
  const queuedQuickIngestCount = useQuickIngestStore((s) => s.queuedCount)
  const quickIngestHadFailure = useQuickIngestStore((s) => s.hadRecentFailure)
  const uiMode = useUiModeStore((state) => state.mode)
  const isProMode = uiMode === "pro"
  const { replyTarget, clearReplyTarget, ragPinnedResults } = useStoreMessageOption(
    (state) => ({
      replyTarget: state.replyTarget,
      clearReplyTarget: state.clearReplyTarget,
      ragPinnedResults: state.ragPinnedResults
    }),
    shallow
  )
  const composerPadding = SPACING.COMPOSER_PADDING
  const composerGap = getComposerGap(isProMode)
  const cardPadding = SPACING.CARD_PADDING
  const textareaMaxHeight = isProMode
    ? COMPOSER_CONSTANTS.TEXTAREA_MAX_HEIGHT_PRO
    : COMPOSER_CONSTANTS.TEXTAREA_MAX_HEIGHT_CASUAL
  const textareaMinHeight = isProMode
    ? COMPOSER_CONSTANTS.TEXTAREA_MIN_HEIGHT_PRO
    : COMPOSER_CONSTANTS.TEXTAREA_MIN_HEIGHT_CASUAL
  const storageKey = draftKey || STORAGE_KEYS.SIDEPANEL_CHAT_DRAFT
  // Shared primitive: form state + draft persistence.
  // See apps/packages/ui/src/components/Chat/composer/hooks/useComposerText.ts.
  // Sidepanel intentionally keeps its own local `textAreaFocus` below (plain
  // .focus() without the mobile blur heuristic) to preserve exact behavior —
  // adopt the primitive's `textAreaFocus` in a follow-up if desired.
  const { form, draftSaved, clearDraft } = useComposerText({
    draftKey: storageKey,
    textareaRef,
    isProMode
  })

  // Experimental Primer composer wire-up — opt in via ?nextgenComposer=1
  // query param OR the Settings "Enable new composer" toggle. Mirrors
  // the Playground integration: when ON, renders <ChatComposer> via the
  // shared message + submitForm pipeline.
  //
  // URL flag is mount-static; the toggle uses the live-syncing hook so
  // cross-tab flips update this surface without a reload.
  const [urlFlagEnabled] = React.useState(() => {
    if (typeof window === "undefined") return false
    try {
      const params = new URLSearchParams(window.location.search)
      return params.get("nextgenComposer") === "1"
    } catch {
      return false
    }
  })
  const [toggleEnabled] = useComposerEnabledPreference()
  const nextgenComposerEnabled = urlFlagEnabled || toggleEnabled
  const [nextgenComposerVariant] = useComposerVariantPreference()
  const { deferredInput: deferredComposerInput } = useDeferredComposerInput(
    form.values.message || ""
  )
  const renderPerfTrackerRef = React.useRef(
    createRenderPerfTracker({
      enabled: Boolean((globalThis as any).__TLDW_CHAT_PERF__)
    })
  )
  const onComposerRenderProfile = React.useCallback<React.ProfilerOnRenderCallback>(
    (id, phase, actualDuration, baseDuration, startTime, commitTime) => {
      renderPerfTrackerRef.current.onRender(
        String(id),
        phase,
        actualDuration,
        baseDuration,
        startTime,
        commitTime
      )
    },
    []
  )
  const wrapComposerProfile = React.useCallback(
    (id: string, node: React.ReactNode): React.ReactNode => {
      if (!renderPerfTrackerRef.current.isEnabled()) {
        return node
      }
      return (
        <React.Profiler id={id} onRender={onComposerRenderProfile}>
          {node}
        </React.Profiler>
      )
    },
    [onComposerRenderProfile]
  )
  React.useEffect(() => {
    const tracker = renderPerfTrackerRef.current
    if (!tracker.isEnabled() || typeof window === "undefined") {
      return
    }
    ;(window as any).__TLDW_SIDEPANEL_CHAT_RENDER_PERF_SNAPSHOT__ = () =>
      tracker.snapshot()
    ;(window as any).__TLDW_SIDEPANEL_CHAT_RENDER_PERF_SUMMARY__ = () =>
      tracker.summarize()
    ;(window as any).__TLDW_SIDEPANEL_CHAT_RENDER_PERF_CLEAR__ = () =>
      tracker.clear()
    return () => {
      delete (window as any).__TLDW_SIDEPANEL_CHAT_RENDER_PERF_SNAPSHOT__
      delete (window as any).__TLDW_SIDEPANEL_CHAT_RENDER_PERF_SUMMARY__
      delete (window as any).__TLDW_SIDEPANEL_CHAT_RENDER_PERF_CLEAR__
    }
  }, [])
  const messageInputProps = form.getInputProps("message")
  const [knowledgeMentionActive, setKnowledgeMentionActive] = React.useState(false)
  const [knowledgePanelOpen, setKnowledgePanelOpen] = React.useState(false)
  const imageValueRef = React.useRef(form.values.image)
  React.useEffect(() => {
    imageValueRef.current = form.values.image
  }, [form.values.image])
  const [contextFiles, setContextFiles] = React.useState<UploadedFile[]>([])
  const [mentionActiveIndex, setMentionActiveIndex] = React.useState(0)
  const [dictationAutoFallbackEnabled] = useStorage(
    "dictation_auto_fallback",
    false
  )
  const [dictationModeOverride] = useStorage<DictationModePreference | null>(
    "dictationModeOverride",
    null
  )
  // Voice/dictation orchestration is wired below via `useComposerVoiceChat`,
  // after `canUseServerStt` and `speechToTextLanguage` are computed.

  const {
    tabMentionsEnabled,
    mentionPosition,
    filteredTabs,
    availableTabs,
    selectedDocuments,
    handleTextChange,
    insertMention,
    closeMentions,
    addDocument,
    removeDocument,
    clearSelectedDocuments,
    reloadTabs,
    handleMentionsOpen
  } = useTabMentions(textareaRef, { includeActive: true })

  const hasWarnedPrivateMode = React.useRef(false)

  // Warn Firefox private mode users on mount that data won't persist
  React.useEffect(() => {
    if (!isFireFoxPrivateMode || hasWarnedPrivateMode.current) return
    hasWarnedPrivateMode.current = true
    notification.warning({
      message: t(
        "sidepanel:errors.privateModeTitle",
        "tldw Assistant can't save data"
      ),
      description: t(
        "sidepanel:errors.privateModeDescription",
        "Firefox Private Mode does not support saving chat history. Your conversations won't be saved."
      ),
      duration: 6
    })
  }, [isFireFoxPrivateMode, notification, t])

  React.useEffect(() => {
    if (!onHeightChange) return
    const node = formContainerRef.current
    if (!node || typeof ResizeObserver === "undefined") return

    const notifyHeight = (height: number) => {
      onHeightChange(Math.max(0, Math.ceil(height)))
    }

    notifyHeight(node.getBoundingClientRect().height)
    const observer = new ResizeObserver((entries) => {
      const entry = entries[0]
      if (!entry) return
      notifyHeight(entry.contentRect.height)
    })
    observer.observe(node)

    return () => {
      observer.disconnect()
    }
  }, [onHeightChange])

  // tldw WS STT
  const {
    connect: sttConnect,
    sendAudio,
    close: sttClose,
    connected: sttConnected,
    lastError: sttError
  } = useTldwStt()
  const {
    start: micStart,
    stop: micStop,
    active: micActive
  } = useMicStream((chunk) => {
    try {
      sendAudio(chunk)
    } catch {}
  })
  const [wsSttActive, setWsSttActive] = React.useState(false)
  const [ingestOpen, setIngestOpen] = React.useState(false)
  const [autoProcessQueuedIngest, setAutoProcessQueuedIngest] =
    React.useState(false)
  const {
    quickIngestSession,
    createDraftQuickIngestSession,
    upsertQuickIngestSession,
    showQuickIngestSession,
    hideQuickIngestSession
  } = useQuickIngestSessionStore(
    (state) => ({
      quickIngestSession: state.session,
      createDraftQuickIngestSession: state.createDraftSession,
      upsertQuickIngestSession: state.upsertSession,
      showQuickIngestSession: state.showSession,
      hideQuickIngestSession: state.hideSession
    }),
    shallow
  )
  const shouldRenderQuickIngest = ingestOpen || Boolean(quickIngestSession)
  const quickIngestBtnRef = React.useRef<HTMLButtonElement>(null)
  const { phase, isConnected, serverUrl } = useConnectionState()
  const { uxState } = useConnectionUxState()
  const isConnectionReady = isConnected && phase === ConnectionPhase.CONNECTED
  const { capabilities, loading: capsLoading } = useServerCapabilities()
  const hasServerVoiceChat =
    isConnectionReady &&
    !capsLoading &&
    Boolean(capabilities?.hasVoiceChat ?? capabilities?.hasAudio)
  const hasServerStt =
    isConnectionReady &&
    !capsLoading &&
    Boolean(capabilities?.hasStt ?? capabilities?.hasAudio)
  const shouldProbeAudioHealth = React.useMemo(
    () =>
      shouldProbeVoiceConversationAudioHealth({
        isConnectionReady,
        hasServerVoiceChat,
        hasServerStt,
        optionalAudioHealthEnabled: audioHealthEnabled
      }),
    [
      audioHealthEnabled,
      hasServerStt,
      hasServerVoiceChat,
      isConnectionReady
    ]
  )
  const {
    healthState: audioHealthState,
    sttHealthState,
    hasVoiceConversationTransport
  } = useTldwAudioStatus({
    enabled: shouldProbeAudioHealth,
    ttsProvider,
    tldwTtsModel
  })
  const canUseServerAudio =
    hasServerVoiceChat && audioHealthState !== "unhealthy"
  const canUseServerStt = hasServerStt && sttHealthState !== "unhealthy"
  // `hasVoiceInputControls` is computed below, after `useComposerVoiceChat`
  // exposes `browserSupportsSpeechRecognition`.
  const voiceConversationTtsConfig = React.useMemo(
    () =>
      resolveVoiceConversationTtsConfig({
        ttsProvider,
        tldwTtsModel,
        tldwTtsVoice,
        tldwTtsSpeed,
        tldwTtsResponseFormat,
        openAITTSModel,
        openAITTSVoice,
        elevenLabsModel,
        elevenLabsVoiceId,
        speechPlaybackSpeed,
        voiceChatTtsMode
      }),
    [
      elevenLabsModel,
      elevenLabsVoiceId,
      openAITTSModel,
      openAITTSVoice,
      speechPlaybackSpeed,
      tldwTtsModel,
      tldwTtsResponseFormat,
      tldwTtsSpeed,
      tldwTtsVoice,
      ttsProvider,
      voiceChatTtsMode
    ]
  )
  const voiceConversationAvailability = React.useMemo(
    () =>
      resolveVoiceConversationAvailability({
        isConnectionReady: isConnectionReady && !canonicalConnectionLoading,
        hasVoiceConversationTransport,
        authReady: Boolean(
          canonicalConnectionConfig?.serverUrl &&
            (canonicalConnectionConfig?.authMode === "multi-user"
              ? canonicalConnectionConfig.accessToken
              : canonicalConnectionConfig.apiKey)
        ),
        sttHealthState,
        ttsHealthState: audioHealthState,
        selectedModel: String(voiceChatModel || "").trim(),
        allowBackendDefaultModel: true,
        ttsConfigReady: voiceConversationTtsConfig.ok
      }),
    [
      audioHealthState,
      canonicalConnectionConfig?.apiKey,
      canonicalConnectionConfig?.authMode,
      canonicalConnectionConfig?.accessToken,
      canonicalConnectionConfig?.serverUrl,
      canonicalConnectionLoading,
      hasVoiceConversationTransport,
      isConnectionReady,
      sttHealthState,
      voiceChatModel,
      voiceConversationTtsConfig.ok
    ]
  )
  const voiceChatAvailable = voiceConversationAvailability.available

  const voiceChat = useVoiceChatStream({
    active: voiceChatEnabled && voiceChatAvailable,
    onTranscript: (text) => {
      voiceChatMessages.beginTurn(text)
    },
    onAssistantDelta: (delta) => {
      voiceChatMessages.appendAssistantDelta(delta)
    },
    onAssistantMessage: (text) => {
      void voiceChatMessages.finalizeAssistant(text)
    },
    onError: (msg) => {
      const runtimeError = normalizeVoiceConversationRuntimeError(msg)
      notification.error({
        message: t("playground:voiceChat.errorTitle", "Voice chat error"),
        description: runtimeError.message
      })
      void voiceChatMessages.failTurn(runtimeError.reason)
      setVoiceChatEnabled(false)
    },
    onWarning: (msg) => {
      notification.warning({
        message: t("playground:voiceChat.warningTitle", "Voice chat warning"),
        description: msg
      })
    }
  })

  React.useEffect(() => {
    setOptionalPanelVisible("audio-health", voiceChatEnabled)
    if (voiceChatEnabled) {
      markOptionalPanelEngaged("audio-health")
    }

    return () => {
      setOptionalPanelVisible("audio-health", false)
    }
  }, [markOptionalPanelEngaged, setOptionalPanelVisible, voiceChatEnabled])

  const [debouncedPlaceholder, setDebouncedPlaceholder] = React.useState<string>(
    t("form.textarea.placeholder")
  )
  const placeholderTimeoutRef = React.useRef<ReturnType<typeof setTimeout> | null>(null)
  const {
    onSubmit,
    selectedModel,
    setSelectedModel,
    chatMode,
    stopStreamingRequest,
    streaming,
    setChatMode,
    webSearch,
    setWebSearch,
    selectedQuickPrompt,
    setSelectedQuickPrompt,
    selectedSystemPrompt,
    setSelectedSystemPrompt,
    speechToTextLanguage,
    useOCR,
    setUseOCR,
    defaultInternetSearchOn,
    defaultChatWithWebsite,
    temporaryChat,
    setTemporaryChat,
    toolChoice,
    setToolChoice,
    historyId,
    history,
    chatLoopState = {
      status: "idle",
      pendingApprovals: [],
      inflightToolCallIds: []
    },
    messages,
    clearChat,
    queuedMessages,
    setQueuedMessages,
    serverChatId
  } = useMessage()
  const serverChatAssistantKind = useStoreMessageOption(
    (state) => state.serverChatAssistantKind
  )
  const serverChatAssistantId = useStoreMessageOption(
    (state) => state.serverChatAssistantId
  )
  const serverChatCharacterId = useStoreMessageOption(
    (state) => state.serverChatCharacterId
  )
  const { settings: conversationContextSettings, updateSettings } =
    useChatSettingsRecord({
      historyId,
      serverChatId
    })
  const effectiveAssistantState = React.useMemo(
    () =>
      resolveEffectiveAssistantState({
        tracked: {
          assistantKind: serverChatAssistantKind,
          assistantId: serverChatAssistantId,
          characterId: serverChatCharacterId
        },
        settings: conversationContextSettings ?? null
      }),
    [
      conversationContextSettings,
      serverChatAssistantId,
      serverChatAssistantKind,
      serverChatCharacterId
    ]
  )
  const sidepanelOverlayResumeStorageRef = React.useRef(
    createSafeStorage({ area: "local" })
  )
  const overlayResumeMarkerKey = React.useMemo(
    () => getSidepanelOverlayResumeMarkerKey(storageKey),
    [storageKey]
  )
  const handlePrepareTrackedStart = React.useCallback(async () => {
    if (!overlayResumeMarkerKey) return
    try {
      await sidepanelOverlayResumeStorageRef.current.remove(overlayResumeMarkerKey)
    } catch {
      // no-op
    }
  }, [overlayResumeMarkerKey])
  const previousServerChatIdRef = React.useRef<string | null | undefined>(
    serverChatId
  )

  React.useEffect(() => {
    if (!overlayResumeMarkerKey) return

    const storage = sidepanelOverlayResumeStorageRef.current
    const syncOverlayResumeMarker = async () => {
      try {
        if (effectiveAssistantState.mode === "overlay") {
          await storage.set(overlayResumeMarkerKey, {
            updatedAt: new Date().toISOString()
          })
          return
        }
        await storage.remove(overlayResumeMarkerKey)
      } catch {
        // no-op
      }
    }

    void syncOverlayResumeMarker()
  }, [effectiveAssistantState.mode, overlayResumeMarkerKey])

  React.useEffect(() => {
    const previous = previousServerChatIdRef.current
    const current = serverChatId

    if (previous && previous !== "" && !current && !temporaryChat) {
      notification.warning({
        message: t(
          "sidepanel:saveStatus.saveFailed",
          "Failed to save chat to server"
        ),
        description: t(
          "sidepanel:saveStatus.saveFailedDescription",
          "Chat is now saving locally only. Check your connection and try again."
        ),
        placement: "bottomRight",
        duration: 4
      })
    }

    previousServerChatIdRef.current = current
  }, [notification, serverChatId, t, temporaryChat])
  const hasImage = form.values.image.length > 0
  const replyLabel = replyTarget
    ? [
        t("common:replyingTo", "Replying to"),
        replyTarget.name ? `${replyTarget.name}:` : null,
        replyTarget.preview
      ]
        .filter(Boolean)
        .join(" ")
    : ""
  const pageContextActive = chatMode === "rag" && chatWithWebsiteEmbedding
  const contextChips = [
    ...(replyTarget && isProMode
      ? [
          {
            id: "reply",
            label: replyLabel,
            icon: <CornerUpLeft className="h-3 w-3 text-text-subtle" />,
            onRemove: clearReplyTarget,
            removeLabel: t("common:clearReply", "Clear reply target")
          }
        ]
      : []),
    ...(pageContextActive
      ? [
          {
            id: "page-context",
            label: t("sidepanel:composer.pageContext", "Current page"),
            icon: <Globe className="h-3 w-3 text-text-subtle" />,
            onRemove: () => setChatMode("normal"),
            removeLabel: t(
              "sidepanel:composer.removePageContext",
              "Remove page context"
            )
          }
        ]
      : []),
    ...selectedDocuments.map((doc) => ({
      id: `tab-${doc.id}`,
      label: doc.title,
      icon: <Globe className="h-3 w-3 text-text-subtle" />,
      onRemove: () => removeDocument(doc.id),
      removeLabel: t("sidepanel:composer.removeDocument", "Remove page")
    })),
    ...(knowledgeMentionActive
      ? [
          {
            id: "knowledge",
            label: t("sidepanel:composer.knowledgeContext", "Knowledge base"),
            icon: <Search className="h-3 w-3 text-text-subtle" />,
            onRemove: () => {
              setKnowledgeMentionActive(false)
              window.dispatchEvent(new CustomEvent("tldw:toggle-rag"))
            },
            removeLabel: t(
              "sidepanel:composer.removeKnowledge",
              "Remove knowledge context"
            )
          }
        ]
      : []),
    ...contextFiles.map((file) => ({
      id: `file-${file.id}`,
      label: file.filename,
      icon: <FileText className="h-3 w-3 text-text-subtle" />,
      onRemove: () =>
        setContextFiles((prev) => prev.filter((item) => item.id !== file.id)),
      removeLabel: t("sidepanel:composer.removeFile", "Remove file")
    })),
    ...(hasImage
      ? [
          {
            id: "image",
            label: t("playground:actions.upload", "Attach image"),
            previewSrc: form.values.image,
            onRemove: () => {
              form.setFieldValue("image", "")
            },
            removeLabel: t(
              "sidepanel:composer.removeImage",
              "Remove uploaded image"
            )
          }
        ]
      : [])
  ]

  const sendButtonTitle = !isConnectionReady
    ? (t(
        "playground:composer.connectToSend",
        "Connect to your tldw server to start chatting."
      ) as string)
    : sendWhenEnter
      ? (t("playground:sendWhenEnter") as string)
      : undefined

  // Sidepanel is image-only — images are read to base64 and written to the
  // form; non-images trigger a toast. The shared primitive centralizes
  // the decision tree (unsupported-type check, image vs non-image branch,
  // base64 encoding); we supply Sidepanel-specific toast callbacks.
  const attachmentHandler = useComposerAttachments({
    chatMode: "normal",
    setImageField: (base64) => form.setFieldValue("image", base64),
    onImageAccepted: (file) => {
      message.success({
        content: t("sidepanel:composer.imageUploaded", {
          defaultValue: "Image added: {{name}}",
          name:
            file.name.length > 20
              ? `${file.name.slice(0, 17)}...`
              : file.name
        }),
        duration: 2
      })
    },
    onNonImageRejected: () => {
      message.error({
        content: t(
          "sidepanel:composer.imageTypeError",
          "Please select an image file"
        ),
        duration: 3
      })
    },
    onImageReadError: () => {
      message.error({
        content: t(
          "sidepanel:composer.imageUploadError",
          "Failed to process image"
        ),
        duration: 3
      })
    },
  })
  const onInputChange = attachmentHandler.onInputChange
  const openUploadDialog = attachmentHandler.handleDocumentUpload
  const textAreaFocus = React.useCallback(() => {
    if (textareaRef.current) {
      textareaRef.current.focus()
    }
  }, [])

  // When sidepanel connection transitions to CONNECTED, focus the composer
  useFocusComposerOnConnect(phase)

  // Voice/dictation orchestration — delegates the full browser+server STT
  // pipeline (source preferences, capture plan, strategy, diagnostics,
  // toggle handler, transcript streaming) to the shared composer primitive.
  // Sidepanel writes transcripts straight into the message field; it does
  // not auto-submit on dictation end.
  const {
    isListening,
    browserSupportsSpeechRecognition,
    isServerDictating,
    stopServerDictation,
    dictationAudioSourcePreference,
    setDictationAudioSourcePreference,
    dictationResolvedSourceKind: resolvedDictationSourceKind,
    audioInputDevices,
    speechAvailable,
    speechUsesServer,
    stopListening,
    handleDictationToggle
  } = useComposerVoiceChat({
    surface: "sidepanel",
    canUseServerStt,
    speechToTextLanguage,
    sttSettings,
    dictationModeOverride,
    dictationAutoFallbackEnabled: Boolean(dictationAutoFallbackEnabled),
    onTranscript: (text) => form.setFieldValue("message", text)
  })
  const hasVoiceInputControls =
    browserSupportsSpeechRecognition || hasServerStt || hasServerVoiceChat

  // Composer window events hook
  const handleOpenQuickIngest = React.useCallback((detail?: QuickIngestOpenDetail) => {
    const seed = createQuickIngestSessionSeedFromOpenDetail(detail)
    setAutoProcessQueuedIngest(false)
    if (quickIngestSession) {
      if (seed) {
        upsertQuickIngestSession(seed)
      }
      showQuickIngestSession()
    } else {
      createDraftQuickIngestSession(seed ?? undefined)
    }
    setIngestOpen(true)
    requestAnimationFrame(() => {
      quickIngestBtnRef.current?.focus()
    })
  }, [
    createDraftQuickIngestSession,
    quickIngestSession,
    showQuickIngestSession,
    upsertQuickIngestSession
  ])

  const {
    openActorSettings,
    setOpenActorSettings,
    openModelSettings,
    setOpenModelSettings,
    documentGeneratorOpen,
    setDocumentGeneratorOpen,
    documentGeneratorSeed,
    setDocumentGeneratorSeed
  } = useComposerEvents({
    serverChatId,
    onFocusComposer: textAreaFocus,
    onOpenQuickIngest: handleOpenQuickIngest
  })

  // Temporary chat toggle hook
  const { handleToggleTemporaryChat } = useTemporaryChatToggle({
    temporaryChat,
    setTemporaryChat,
    messagesLength: messages.length,
    clearChat
  })
  const temporaryChatLocked = temporaryChat && messages.length > 0
  const temporaryChatToggleLabel = temporaryChat
    ? t("playground:actions.temporaryOn", "Temporary chat (not saved)")
    : t("playground:actions.temporaryOff", "Save chat to history")
  const persistenceModeLabel = React.useMemo(() => {
    if (temporaryChat) {
      return t(
        "playground:composer.persistence.ephemeral",
        "Not saved: cleared when you close this window."
      )
    }
    if (serverChatId || isConnectionReady) {
      return t(
        "playground:composer.persistence.server",
        "Saved to your tldw server (and locally)."
      )
    }
    return t(
      "playground:composer.persistence.local",
      "Saved locally until your tldw server is connected."
    )
  }, [isConnectionReady, serverChatId, t, temporaryChat])
  const persistencePillLabel = React.useMemo(() => {
    if (temporaryChat) {
      return t("playground:composer.persistence.ephemeralPill", "Not saved")
    }
    if (serverChatId || isConnectionReady) {
      return t("playground:composer.persistence.serverPill", "Server")
    }
    return t("playground:composer.persistence.localPill", "Local")
  }, [isConnectionReady, serverChatId, t, temporaryChat])
  const persistenceTooltip = React.useMemo(
    () => (
      <div className="flex flex-col gap-0.5 text-xs">
        <span className="font-medium">{persistencePillLabel}</span>
        <span className="text-text-subtle">{persistenceModeLabel}</span>
      </div>
    ),
    [persistenceModeLabel, persistencePillLabel]
  )

  // Character selection state
  const storedCharacterId = React.useMemo(
    () => resolveCharacterSelectionId(storedCharacter),
    [storedCharacter]
  )
  const localDefaultCharacterId = React.useMemo(
    () => resolveCharacterSelectionId(defaultCharacter),
    [defaultCharacter]
  )
  const serverDefaultCharacterId = defaultCharacterPreference?.defaultCharacterId
  const effectiveDefaultCharacter = React.useMemo<Character | null>(() => {
    if (typeof serverDefaultCharacterId === "undefined") {
      return defaultCharacter
    }
    if (!serverDefaultCharacterId) {
      return null
    }
    if (
      localDefaultCharacterId === serverDefaultCharacterId &&
      defaultCharacter
    ) {
      return defaultCharacter
    }
    return { id: serverDefaultCharacterId } as Character
  }, [
    defaultCharacter,
    localDefaultCharacterId,
    serverDefaultCharacterId
  ])
  const effectiveDefaultCharacterId = React.useMemo(
    () => resolveCharacterSelectionId(effectiveDefaultCharacter),
    [effectiveDefaultCharacter]
  )
  const isFreshChat = React.useMemo(
    () => isFreshChatState(serverChatId, messages.length),
    [messages.length, serverChatId]
  )
  const defaultCharacterBootstrapAppliedRef = React.useRef(false)
  const previousFreshChatRef = React.useRef(isFreshChat)
  const [selectedCharacterId, setSelectedCharacterId] = React.useState<
    string | null
  >(storedCharacterId)

  React.useEffect(() => {
    setSelectedCharacterId((prev) =>
      prev === storedCharacterId ? prev : storedCharacterId
    )
  }, [storedCharacterId])

  React.useEffect(() => {
    if (typeof serverDefaultCharacterId === "undefined") return

    if (!serverDefaultCharacterId) {
      if (localDefaultCharacterId) {
        void setDefaultCharacter(null)
      }
      return
    }

    if (localDefaultCharacterId === serverDefaultCharacterId) return
    void setDefaultCharacter({ id: serverDefaultCharacterId } as Character)
  }, [
    localDefaultCharacterId,
    serverDefaultCharacterId,
    setDefaultCharacter
  ])

  React.useEffect(() => {
    defaultCharacterBootstrapAppliedRef.current = false
  }, [effectiveDefaultCharacterId])

  React.useEffect(() => {
    if (
      shouldResetDefaultCharacterBootstrap(
        previousFreshChatRef.current,
        isFreshChat
      )
    ) {
      defaultCharacterBootstrapAppliedRef.current = false
    }
    previousFreshChatRef.current = isFreshChat
  }, [isFreshChat])

  React.useEffect(() => {
    if (!effectiveDefaultCharacter || !effectiveDefaultCharacterId) return
    if (
      !shouldApplyDefaultCharacter({
        defaultCharacterId: effectiveDefaultCharacterId,
        selectedCharacterId: storedCharacterId,
        isFreshChat,
        hasAppliedInSession: defaultCharacterBootstrapAppliedRef.current
      })
    ) {
      return
    }

    defaultCharacterBootstrapAppliedRef.current = true
    void setStoredCharacter(effectiveDefaultCharacter)
  }, [
    effectiveDefaultCharacter,
    effectiveDefaultCharacterId,
    isFreshChat,
    setStoredCharacter,
    storedCharacterId
  ])

  const conversationContextComposition = useConversationContextComposition({
    draftMessage: form.values.message,
    selection: {
      chatId: serverChatId ?? undefined,
      characterId: selectedCharacterId,
      worldBookIds: [],
      dictionaryIds: []
    },
    settings: conversationContextSettings,
    debounceMs: 250,
    updateSettings
  })

  const {
    filteredSlashCommands,
    showSlashMenu,
    slashActiveIndex,
    setSlashActiveIndex,
    applySlashCommand,
    handleSlashCommandSelect
  } = useSlashCommands({
    chatMode,
    webSearch,
    setChatMode,
    setWebSearch,
    onOpenModelSettings: () => setOpenModelSettings(true),
    inputValue: form.values.message,
    setInputValue: (value) => form.setFieldValue("message", value)
  })

  const handleSlashCommandPick = React.useCallback(
    (command: SlashCommandItem) => {
      handleSlashCommandSelect(command)
      requestAnimationFrame(() => textareaRef.current?.focus())
    },
    [handleSlashCommandSelect, textareaRef]
  )

  const removeMentionToken = React.useCallback(() => {
    if (!mentionPosition || !textareaRef.current) return
    const current = form.values.message || ""
    const before = current.substring(0, mentionPosition.start)
    const after = current.substring(mentionPosition.end)
    const nextValue = before + after
    form.setFieldValue("message", nextValue)
    requestAnimationFrame(() => {
      textareaRef.current?.focus()
      textareaRef.current?.setSelectionRange(
        mentionPosition.start,
        mentionPosition.start
      )
    })
    closeMentions()
  }, [closeMentions, form, mentionPosition, textareaRef])

  const handleCurrentPageMention = React.useCallback(async () => {
    try {
      const tabs = await browser.tabs.query({
        active: true,
        currentWindow: true
      })
      const activeTab = tabs.find((tab) => tab.id && tab.title && tab.url)
      if (!activeTab) {
        console.error("[Sidepanel] No active tab found for page mention.")
        return
      }
      const tabInfo: TabInfo = {
        id: activeTab.id!,
        title: activeTab.title!,
        url: activeTab.url!,
        favIconUrl: activeTab.favIconUrl
      }
      addDocument(tabInfo)
    } catch (error) {
      console.error("[Sidepanel] Failed to fetch active tab for mention:", error)
    } finally {
      removeMentionToken()
    }
  }, [addDocument, removeMentionToken])

  const handleKnowledgeMention = React.useCallback(() => {
    setKnowledgeMentionActive(true)
    window.dispatchEvent(new CustomEvent("tldw:toggle-rag"))
    removeMentionToken()
  }, [removeMentionToken])

  const handleFileMention = React.useCallback(() => {
    contextFileInputRef.current?.click()
    removeMentionToken()
  }, [removeMentionToken])

  const handleContextFileChange = React.useCallback(
    async (event: React.ChangeEvent<HTMLInputElement>) => {
      const files = Array.from(event.target.files || [])
      if (files.length === 0) return
      const oversized = files.find(
        (file) => file.size > maxContextFileSizeBytes
      )
      if (oversized) {
        notification.error({
          message: t("option:upload.fileTooLargeTitle", "File Too Large"),
          description: t(
            "option:upload.fileTooLargeDescription",
            {
              defaultValue: "File size must be less than {{size}}",
              size: maxContextFileSizeLabel
            }
          )
        })
        event.target.value = ""
        return
      }
      try {
        const { processFileUpload } = await import("~/utils/file-processor")
        const nextFiles: UploadedFile[] = []
        const failedFiles: string[] = []
        for (const file of files) {
          try {
            const source = await processFileUpload(file)
            const content =
              source && typeof (source as any).content === "string"
                ? (source as any).content
                : null
            if (!content) {
              failedFiles.push(file.name)
              continue
            }
            nextFiles.push({
              id: generateID(),
              filename: file.name,
              type: file.type,
              content,
              size: file.size,
              uploadedAt: Date.now(),
              processed: false
            })
          } catch {
            failedFiles.push(file.name)
          }
        }
        if (nextFiles.length > 0) {
          setContextFiles((prev) => [...prev, ...nextFiles])
          notification.success({
            message: t("sidepanel:composer.filesAdded", {
              defaultValue: "{{count}} file(s) added to context",
              count: nextFiles.length
            })
          })
        }
        if (failedFiles.length > 0) {
          notification.warning({
            message: t("sidepanel:composer.someFilesFailed", {
              defaultValue: "Failed to process: {{files}}",
              files: failedFiles.join(", ")
            })
          })
        }
      } catch (error: any) {
        notification.error({
          message: t("sidepanel:composer.fileAddError", "Failed to add file"),
          description: error?.message || ""
        })
      } finally {
        event.target.value = ""
      }
    },
    [
      maxContextFileSizeBytes,
      maxContextFileSizeLabel,
      notification,
      setContextFiles,
      t
    ]
  )

  const mentionQuery = (mentionPosition?.query || "").toLowerCase()
  const staticMentionItems = React.useMemo<MentionMenuItem[]>(
    () => [
      {
        id: "mention-current-page",
        label: t("sidepanel:composer.mentionCurrentPage", "Current page"),
        description: t(
          "sidepanel:composer.mentionCurrentPageDesc",
          "Use the active tab as context"
        ),
        icon: <Globe className="size-3.5" />,
        kind: "page"
      },
      {
        id: "mention-knowledge",
        label: t("sidepanel:composer.mentionKnowledge", "Knowledge base"),
        description: t(
          "sidepanel:composer.mentionKnowledgeDesc",
          "Search your knowledge base"
        ),
        icon: <Search className="size-3.5" />,
        kind: "knowledge"
      },
      {
        id: "mention-file",
        label: t("sidepanel:composer.mentionFile", "File"),
        description: t(
          "sidepanel:composer.mentionFileDesc",
          "Attach a file as context"
        ),
        icon: <FileText className="size-3.5" />,
        kind: "file"
      }
    ],
    [t]
  )

  const mentionItems = React.useMemo<MentionMenuItem[]>(() => {
    if (!tabMentionsEnabled) return []
    const filteredStatic =
      mentionQuery.length === 0
        ? staticMentionItems
        : staticMentionItems.filter((item) =>
            item.label.toLowerCase().includes(mentionQuery)
          )
    const tabItems: MentionMenuItem[] = filteredTabs.map((tab) => ({
      id: `tab-${tab.id}`,
      label: tab.title,
      description: tab.url,
      icon: <Globe className="size-3.5" />,
      kind: "tab",
      payload: tab
    }))
    return [...filteredStatic, ...tabItems]
  }, [filteredTabs, mentionQuery, staticMentionItems, tabMentionsEnabled])

  const showMentionMenu = Boolean(mentionPosition) && tabMentionsEnabled

  React.useEffect(() => {
    if (!showMentionMenu) {
      setMentionActiveIndex(0)
      return
    }
    setMentionActiveIndex((prev) => {
      if (mentionItems.length === 0) return 0
      return Math.min(prev, mentionItems.length - 1)
    })
  }, [mentionItems.length, showMentionMenu])

  React.useEffect(() => {
    if (!showMentionMenu) return
    let cancelled = false
    void handleMentionsOpen().catch((error) => {
      if (cancelled) return
      console.error("Failed to open mentions menu:", error)
    })
    return () => {
      cancelled = true
    }
  }, [handleMentionsOpen, showMentionMenu])

  const handleMentionSelect = React.useCallback(
    (item: MentionMenuItem) => {
      if (item.kind === "tab" && item.payload) {
        const tab = item.payload as TabInfo
        const alreadySelected = selectedDocuments.some((doc) => doc.id === tab.id)
        if (alreadySelected) {
          removeMentionToken()
          return
        }
        insertMention(tab, form.values.message, (value) =>
          form.setFieldValue("message", value)
        )
        return
      }
      if (item.kind === "page") {
        void handleCurrentPageMention()
        return
      }
      if (item.kind === "knowledge") {
        handleKnowledgeMention()
        return
      }
      if (item.kind === "file") {
        handleFileMention()
      }
    },
    [
      form,
      handleCurrentPageMention,
      handleFileMention,
      handleKnowledgeMention,
      insertMention,
      removeMentionToken,
      selectedDocuments
    ]
  )


  const handlePaste = (e: React.ClipboardEvent) => {
    if (e.clipboardData.files.length > 0) {
      const file = e.clipboardData.files[0]
      // Only handle image files from paste
      if (file.type.startsWith("image/")) {
        e.preventDefault()
        onInputChange(file)
      }
    }
  }

  useFocusShortcuts(textareaRef, true)

  const ensureEmbeddingModelAvailable = async (): Promise<boolean> => {
    // Fast path: no RAG or web search enabled
    if (chatMode !== "rag" && !webSearch) {
      return true
    }

    let defaultEM: string | null | undefined

    // When chatting with the current page via embeddings, require a default embedding model
    if (chatMode === "rag" && chatWithWebsiteEmbedding) {
      defaultEM = await defaultEmbeddingModelForRag()
      if (!defaultEM) {
        form.setFieldError("message", t("formError.noEmbeddingModel"))
        return false
      }
    }

    // When web search is enabled and not in simple-search mode, also require an embedding model
    if (webSearch) {
      if (typeof defaultEM === "undefined") {
        defaultEM = await defaultEmbeddingModelForRag()
      }
      const simpleSearch = await getIsSimpleInternetSearch()
      if (!defaultEM && !simpleSearch) {
        form.setFieldError("message", t("formError.noEmbeddingModel"))
        return false
      }
    }

    return true
  }

  const buildPinnedMessage = React.useCallback(
    (message: string, options?: { ignorePinnedResults?: boolean }) => {
      if (options?.ignorePinnedResults) return message
      if (!ragPinnedResults || ragPinnedResults.length === 0) return message
      const pinnedText = formatPinnedResults(ragPinnedResults, "markdown")
      return message ? `${message}\n\n${pinnedText}` : pinnedText
    },
    [ragPinnedResults]
  )

  const parseImageSlashCommand = React.useCallback(
    (text: string) => {
      const trimmed = text.trim()
      if (!trimmed.toLowerCase().startsWith("/generate-image")) return null
      const remainder = trimmed.slice("/generate-image".length)
      const colonMatch = remainder.match(
        /^\s*:\s*([^\s]+)(?:\s+([\s\S]*))?$/i
      )
      if (colonMatch) {
        const provider = colonMatch[1]?.trim() || ""
        const prompt = (colonMatch[2] || "").trim()
        const missingProvider = provider.length === 0
        return {
          provider,
          prompt,
          invalid: missingProvider,
          missingProvider
        }
      }

      const prompt = remainder.trim()
      if (imageBackendDefaultTrimmed) {
        return {
          provider: imageBackendDefaultTrimmed,
          prompt,
          invalid: false,
          missingProvider: false
        }
      }

      return {
        provider: "",
        prompt,
        invalid: true,
        missingProvider: true
      }
    },
    [imageBackendDefaultTrimmed]
  )

  const resolveSubmissionIntent = React.useCallback(
    (rawMessage: string, options?: { ignorePinnedResults?: boolean }) => {
      const imageCommand = parseImageSlashCommand(rawMessage)
      if (imageCommand) {
        return {
          handled: true,
          message: imageCommand.prompt,
          imageBackendOverride: imageCommand.provider,
          isImageCommand: true,
          invalidImageCommand: imageCommand.invalid,
          imageCommandMissingProvider: Boolean(imageCommand.missingProvider),
          combinedMessage: imageCommand.prompt
        }
      }
      const slashResult = applySlashCommand(rawMessage)
      if (slashResult.handled) {
        form.setFieldValue("message", slashResult.message)
      }
      const nextMessage = slashResult.handled
        ? slashResult.message
        : rawMessage
      const combinedMessage = buildPinnedMessage(nextMessage, options)
      return {
        handled: slashResult.handled,
        message: nextMessage,
        imageBackendOverride: undefined,
        isImageCommand: false,
        invalidImageCommand: false,
        imageCommandMissingProvider: false,
        combinedMessage
      }
    },
    [applySlashCommand, buildPinnedMessage, form, parseImageSlashCommand]
  )

  async function sendCurrentFormMessage(
    rawMessage: string,
    image: string,
    options?: { ignorePinnedResults?: boolean }
  ): Promise<void> {
    const intent = resolveSubmissionIntent(rawMessage, options)
    if (intent.invalidImageCommand) {
      notification.error({
        message: t("error", { defaultValue: "Error" }),
        description: intent.imageCommandMissingProvider
          ? t(
              "imageCommand.missingProvider",
              "Pick an Image provider in More tools or use /generate-image:<provider> <prompt>."
            )
          : t(
              "imageCommand.invalidUsage",
              "Use /generate-image:<provider> <prompt>."
            )
      })
      return
    }
    const trimmed = intent.combinedMessage.trim()
    if (
      !intent.isImageCommand &&
      trimmed.length === 0 &&
      image.length === 0 &&
      selectedDocuments.length === 0 &&
      contextFiles.length === 0
    ) {
      return
    }
    if (intent.isImageCommand && trimmed.length === 0) {
      notification.error({
        message: t("error", { defaultValue: "Error" }),
        description: t(
          "imageCommand.missingPrompt",
          "Image prompt is required."
        )
      })
      return
    }
    await stopListening()
    if (!intent.isImageCommand) {
      const normalizedSelectedModel = normalizeChatModelId(selectedModel)
      if (!normalizedSelectedModel) {
        form.setFieldError("message", t("formError.noModel"))
        return
      }
      const unavailableModel = findUnavailableChatModel(
        [normalizedSelectedModel],
        availableChatModelIds
      )
      if (unavailableModel) {
        form.setFieldError(
          "message",
          t(
            "playground:composer.validationModelUnavailableInline",
            "Selected model is not available on this server. Refresh models or choose a different model."
          )
        )
        return
      }
    }
    if (!intent.isImageCommand) {
      const hasEmbedding = await ensureEmbeddingModelAvailable()
      if (!hasEmbedding) {
        return
      }
    }
    const contextSend = intent.isImageCommand
      ? null
      : await conversationContextComposition.composeForSend({
          message: trimmed,
          history
        })
    await submitDispatch(
      {
        image: intent.isImageCommand ? "" : image,
        message: trimmed,
        docs: intent.isImageCommand
          ? []
          : selectedDocuments.map((doc) => ({
              type: "tab",
              tabId: doc.id,
              title: doc.title,
              url: doc.url,
              favIconUrl: doc.favIconUrl
            })),
        uploadedFiles: intent.isImageCommand ? [] : contextFiles,
        imageBackendOverride: intent.isImageCommand
          ? intent.imageBackendOverride
          : undefined,
        requestOverrides: contextSend?.requestOverrides
      },
      {
        beforeSend: () => {
          form.reset()
          textAreaFocus()
        },
        afterSend: () => {
          clearDraft()
          clearSelectedDocuments()
          setContextFiles([])
          setKnowledgeMentionActive(false)
        }
      }
    )
  }
  const sendCurrentFormMessageRef = React.useRef(sendCurrentFormMessage)
  React.useEffect(() => {
    sendCurrentFormMessageRef.current = sendCurrentFormMessage
  }, [sendCurrentFormMessage])

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (showMentionMenu) {
      if (e.key === "ArrowDown" && mentionItems.length > 0) {
        e.preventDefault()
        setMentionActiveIndex((prev) =>
          prev + 1 >= mentionItems.length ? 0 : prev + 1
        )
        return
      }
      if (e.key === "ArrowUp" && mentionItems.length > 0) {
        e.preventDefault()
        setMentionActiveIndex((prev) =>
          prev <= 0 ? mentionItems.length - 1 : prev - 1
        )
        return
      }
      if (
        (e.key === "Enter" || (e.key === "Tab" && !e.shiftKey)) &&
        mentionItems.length > 0
      ) {
        e.preventDefault()
        const item = mentionItems[mentionActiveIndex]
        if (item) {
          handleMentionSelect(item)
        }
        return
      }
      if (e.key === "Escape") {
        e.preventDefault()
        closeMentions()
        return
      }
    }
    if (showSlashMenu) {
      if (e.key === "ArrowDown" && filteredSlashCommands.length > 0) {
        e.preventDefault()
        setSlashActiveIndex((prev) =>
          prev + 1 >= filteredSlashCommands.length ? 0 : prev + 1
        )
        return
      }
      if (e.key === "ArrowUp" && filteredSlashCommands.length > 0) {
        e.preventDefault()
        setSlashActiveIndex((prev) =>
          prev <= 0 ? filteredSlashCommands.length - 1 : prev - 1
        )
        return
      }
      if (
        (e.key === "Enter" || (e.key === "Tab" && !e.shiftKey)) &&
        filteredSlashCommands.length > 0
      ) {
        e.preventDefault()
        const command = filteredSlashCommands[slashActiveIndex]
        if (command) {
          handleSlashCommandPick(command)
        }
        return
      }
      if (e.key === "Escape") {
        e.preventDefault()
        form.setFieldValue(
          "message",
          form.values.message.replace(/^\s*\//, "")
        )
        return
      }
    }
    if (!isConnectionReady) {
      if (e.key === "Enter") {
        e.preventDefault()
        void submitForm()
      }
      return
    }
    if (e.key === "Process" || e.key === "229") return
    if (
      handleChatInputKeyDown({
        e,
        sendWhenEnter,
        typing,
        isSending: false
      })
    ) {
      e.preventDefault()
      void submitForm()
    }
  }

  const openSettings = React.useCallback(() => {
    try {
      if (typeof chrome !== "undefined" && chrome.runtime?.openOptionsPage) {
        chrome.runtime.openOptionsPage()
        return
      }
    } catch {}
    window.open("/options.html#/", "_blank")
  }, [])

  const openDiagnostics = React.useCallback(() => {
    window.open("/options.html#/settings/health", "_blank")
  }, [])

  const handleOpenModelSettings = React.useCallback(() => {
    setOpenModelSettings(true)
  }, [setOpenModelSettings])

  const handleWebSearchToggle = React.useCallback(() => {
    setWebSearch(!webSearch)
  }, [setWebSearch, webSearch])

  const handleKnowledgeInsert = React.useCallback(
    (text: string) => {
      const current = textareaRef.current?.value || ""
      const next = current ? `${current}\n\n${text}` : text
      form.setFieldValue("message", next)
      textareaRef.current?.focus()
    },
    [form.setFieldValue, textareaRef]
  )

  async function handleKnowledgeAsk(
    text: string,
    options?: { ignorePinnedResults?: boolean }
  ) {
    const trimmed = text.trim()
    if (!trimmed) return
    form.setFieldValue("message", text)
    await submitCurrentRequest(trimmed, "", options)
  }
  const handleKnowledgePanelOpenChange = React.useCallback(
    (nextOpen: boolean) => {
      setKnowledgePanelOpen(nextOpen)
    },
    []
  )
  const handleKnowledgeAddFile = React.useCallback(() => {
    contextFileInputRef.current?.click()
  }, [])
  const handleKnowledgeRemoveFile = React.useCallback((fileId: string) => {
    setContextFiles((prev) => prev.filter((item) => item.id !== fileId))
  }, [])
  const handleKnowledgeClearFiles = React.useCallback(() => {
    setContextFiles([])
  }, [])

  // Dictation toggle + transcript streaming + pending-start handling are
  // owned by `useComposerVoiceChat` (see top of component); nothing extra to
  // wire here. `handleDictationToggle` and `stopListening` are destructured
  // above for the controls below to consume.

  const voiceChatStatusLabel = React.useMemo(() => {
    switch (voiceChat.state) {
      case "connecting":
        return t("playground:voiceChat.statusConnecting", "Connecting")
      case "listening":
        return t("playground:voiceChat.statusListening", "Listening")
      case "thinking":
        return t("playground:voiceChat.statusThinking", "Thinking")
      case "speaking":
        return t("playground:voiceChat.statusSpeaking", "Speaking")
      case "error":
        return t("playground:voiceChat.statusError", "Error")
      default:
        return t("playground:voiceChat.statusIdle", "Voice chat")
    }
  }, [t, voiceChat.state])

  const voiceChatToneClass = React.useMemo(() => {
    if (voiceChat.state === "error") {
      return "border-danger text-danger"
    }
    if (voiceChatEnabled && voiceChat.state !== "idle") {
      return "border-primary text-primaryStrong"
    }
    return "border-border text-text-muted"
  }, [voiceChat.state, voiceChatEnabled])

  const voiceChatUnavailableMessage = React.useMemo(() => {
    const fallback = t(
      "playground:voiceChat.unavailableBody",
      "Connect to a tldw server with audio chat streaming enabled."
    )
    return voiceConversationAvailability.message
      ? t(voiceConversationAvailability.message, fallback)
      : fallback
  }, [t, voiceConversationAvailability.message])

  const handleVoiceChatToggle = React.useCallback(() => {
    if (!voiceChatAvailable) {
      notification.error({
        message: t("playground:voiceChat.unavailableTitle", "Voice chat unavailable"),
        description: voiceChatUnavailableMessage
      })
      return
    }
    if (!voiceChatEnabled) {
      if (isListening) stopListening()
      if (isServerDictating) stopServerDictation()
    }
    if (voiceChatEnabled) {
      voiceChatMessages.abandonTurn()
    }
    setVoiceChatEnabled(!voiceChatEnabled)
  }, [
    voiceChatAvailable,
    voiceChatEnabled,
    isListening,
    isServerDictating,
    notification,
    setVoiceChatEnabled,
    stopListening,
    stopServerDictation,
    t,
    voiceChatMessages,
    voiceChatUnavailableMessage
  ])

  const handleLiveCaptionsToggle = React.useCallback(async () => {
    if (wsSttActive) {
      try {
        micStop()
      } catch {}
      try {
        sttClose()
      } catch {}
      setWsSttActive(false)
    } else {
      try {
        sttConnect()
        await micStart()
        setWsSttActive(true)
      } catch (e: any) {
        notification.error({
          message: t(
            "playground:actions.streamErrorTitle",
            "Live captions unavailable"
          ),
          description:
            e?.message ||
            t(
              "playground:actions.streamMicError",
              "Unable to start live captions. Check microphone permissions and server health, then try again."
            )
        })
        try {
          micStop()
        } catch {}
        try {
          sttClose()
        } catch {}
        setWsSttActive(false)
      }
    }
  }, [micStart, micStop, notification, sttClose, sttConnect, t, wsSttActive])

  const voiceChatSettingsContent = (
    <div className="flex w-64 flex-col gap-2 text-xs">
      <div className="text-[11px] font-semibold uppercase tracking-wide text-text-muted">
        {t("playground:voiceChat.settingsTitle", "Voice chat settings")}
      </div>
      <div className="flex flex-col gap-1">
        <span className="text-[11px] text-text-muted">
          {t("playground:voiceChat.modelLabel", "Voice model")}
        </span>
        <Select
          size="small"
          value={voiceChatModel || ""}
          options={voiceChatModelOptions}
          onChange={(value) => setVoiceChatModel(value)}
        />
      </div>
      <div className="flex flex-col gap-1">
        <span className="text-[11px] text-text-muted">
          {t("playground:voiceChat.pauseLabel", "Auto-send pause (ms)")}
        </span>
        <InputNumber
          size="small"
          min={200}
          max={5000}
          value={voiceChatPauseMs}
          onChange={(value) =>
            setVoiceChatPauseMs(typeof value === "number" ? value : 900)
          }
        />
      </div>
      <div className="flex flex-col gap-1">
        <span className="text-[11px] text-text-muted">
          {t("playground:voiceChat.triggerLabel", "Trigger phrases")}
        </span>
        <Input
          size="small"
          placeholder={t(
            "playground:voiceChat.triggerPlaceholder",
            "e.g. send it, over"
          )}
          value={voiceChatTriggerInput}
          onChange={(e) => {
            const value = e.target.value
            setVoiceChatTriggerInput(value)
            const parsed = value
              .split(",")
              .map((entry) => entry.trim())
              .filter(Boolean)
            setVoiceChatTriggerPhrases(parsed)
          }}
        />
      </div>
      <div className="flex flex-col gap-1">
        <span className="text-[11px] text-text-muted">
          {t("playground:voiceChat.ttsModeLabel", "TTS mode")}
        </span>
        <Radio.Group
          size="small"
          optionType="button"
          value={voiceChatTtsMode}
          onChange={(e) => setVoiceChatTtsMode(e.target.value)}
        >
          <Radio.Button value="stream">
            {t("playground:voiceChat.ttsModeStream", "Stream")}
          </Radio.Button>
          <Radio.Button value="full">
            {t("playground:voiceChat.ttsModeFull", "Full")}
          </Radio.Button>
        </Radio.Group>
      </div>
      <div className="flex flex-col gap-1">
        <span className="text-[11px] text-text-muted">
          {t("playground:voiceChat.sourceLabel", "Input source")}
        </span>
        <AudioSourcePicker
          ariaLabel={t(
            "playground:voiceChat.sourcePickerLabel",
            "Dictation input source"
          )}
          devices={audioInputDevices}
          requestedSourceKind={dictationAudioSourcePreference.sourceKind}
          resolvedSourceKind={resolvedDictationSourceKind}
          requestedDeviceId={dictationAudioSourcePreference.deviceId}
          lastKnownLabel={dictationAudioSourcePreference.lastKnownLabel}
          onChange={(nextValue) =>
            setDictationAudioSourcePreference({
              featureGroup: "dictation",
              sourceKind: nextValue.sourceKind,
              deviceId: nextValue.deviceId ?? null,
              lastKnownLabel: nextValue.lastKnownLabel ?? null
            })
          }
        />
      </div>
      <div className="flex items-center justify-between gap-2">
        <span className="text-[11px] text-text-muted">
          {t("playground:voiceChat.autoResume", "Auto resume")}
        </span>
        <Switch
          size="small"
          checked={voiceChatAutoResume}
          onChange={(checked) => setVoiceChatAutoResume(checked)}
        />
      </div>
      <div className="flex items-center justify-between gap-2">
        <span className="text-[11px] text-text-muted">
          {t("playground:voiceChat.bargeIn", "Barge-in")}
        </span>
        <Switch
          size="small"
          checked={voiceChatBargeIn}
          onChange={(checked) => setVoiceChatBargeIn(checked)}
        />
      </div>
    </div>
  )

  const handleVisionToggle = React.useCallback(() => {
    setChatMode(chatMode === "vision" ? "normal" : "vision")
  }, [chatMode, setChatMode])

  const handleRagToggle = React.useCallback(() => {
    window.dispatchEvent(new CustomEvent("tldw:toggle-rag"))
  }, [])

  const handleQuickIngestOpen = React.useCallback((detail?: QuickIngestOpenDetail) => {
    const seed = createQuickIngestSessionSeedFromOpenDetail(detail)
    setAutoProcessQueuedIngest(false)
    if (quickIngestSession) {
      if (seed) {
        upsertQuickIngestSession(seed)
      }
      showQuickIngestSession()
    } else {
      createDraftQuickIngestSession(seed ?? undefined)
    }
    setIngestOpen(true)
  }, [
    createDraftQuickIngestSession,
    quickIngestSession,
    showQuickIngestSession,
    upsertQuickIngestSession
  ])

  const handleProcessQueuedIngest = React.useCallback(() => {
    if (!isConnectionReady) return

    // Snapshot the current queue size; if it has been cleared between
    // render and click, we still open the modal but skip auto-processing.
    if (queuedQuickIngestCount <= 0) {
      setAutoProcessQueuedIngest(false)
      if (quickIngestSession) {
        showQuickIngestSession()
      } else {
        createDraftQuickIngestSession()
      }
      setIngestOpen(true)
      return
    }

    setAutoProcessQueuedIngest(true)
    if (quickIngestSession) {
      showQuickIngestSession()
    } else {
      createDraftQuickIngestSession()
    }
    setIngestOpen(true)
  }, [
    createDraftQuickIngestSession,
    isConnectionReady,
    queuedQuickIngestCount,
    quickIngestSession,
    showQuickIngestSession
  ])

  React.useEffect(() => {
    if (!sttError) return
    notification.error({
      message: t(
        "playground:actions.streamErrorTitle",
        "Live captions unavailable"
      ),
      description: sttError
    })
    try {
      micStop()
    } catch {}
    try {
      sttClose()
    } catch {}
    setWsSttActive(false)
  }, [micStop, setWsSttActive, sttClose, sttError, t])

  React.useEffect(() => {
    if (dropedFile) {
      onInputChange(dropedFile)
    }
  }, [dropedFile])

  useDynamicTextareaSize(textareaRef, form.values.message, textareaMaxHeight)

  // Browser-dictation transcript streaming is handled by `useComposerVoiceChat`
  // via its `onTranscript` callback above; no local effect needed here.

  React.useEffect(() => {
    if (selectedQuickPrompt) {
      const word = getVariable(selectedQuickPrompt)
      form.setFieldValue("message", selectedQuickPrompt)
      if (word) {
        textareaRef.current?.focus()
        const interval = setTimeout(() => {
          textareaRef.current?.setSelectionRange(word.start, word.end)
          setSelectedQuickPrompt(null)
        }, 100)
        return () => {
          clearInterval(interval)
        }
      }
    }
  }, [selectedQuickPrompt])
  const { mutateAsync: sendMessage, isPending: isSending } = useMutation({
    mutationFn: onSubmit,
    onSuccess: () => {
      textAreaFocus()
    },
    onError: (error) => {
      textAreaFocus()
    }
  })

  // Shared composer dispatch. Wraps sendMessage with before/after hooks so
  // Sidepanel's "reset + focus before, clear attachments after" pattern
  // mirrors Playground's. See Chat/composer/hooks/useComposerSubmit.ts.
  const { dispatch: submitDispatch } = useComposerSubmit({ sendMessage })

  const buildQueuedDocuments = React.useCallback(
    (): ChatDocuments =>
      selectedDocuments.map((doc) => ({
        type: "tab",
        tabId: doc.id,
        title: doc.title,
        url: doc.url,
        favIconUrl: doc.favIconUrl
      })),
    [selectedDocuments]
  )

  const buildQueuedRequestSnapshot = React.useCallback(
    () => ({
      selectedModel,
      chatMode,
      webSearch,
      compareMode: false,
      compareSelectedModels: [],
      selectedSystemPrompt,
      selectedQuickPrompt,
      toolChoice,
      useOCR
    }),
    [
      chatMode,
      selectedModel,
      selectedQuickPrompt,
      selectedSystemPrompt,
      toolChoice,
      useOCR,
      webSearch
    ]
  )

  const isQueuedDispatchBlockedByComposerState = React.useMemo(
    () => contextFiles.length > 0,
    [contextFiles.length]
  )

  const validateQueuedRequest = React.useCallback(
    (item: QueuedRequest) => {
      if (isQueuedDispatchBlockedByComposerState) {
        return t(
          "playground:composer.queue.currentDraftAttachmentConflict",
          "Clear the current draft attachments/context before sending queued requests."
        )
      }

      const sourceContext = (item.sourceContext ??
        null) as SidepanelQueuedSourceContext | null

      if (sourceContext?.isImageCommand && item.promptText.trim().length === 0) {
        return t(
          "imageCommand.missingPrompt",
          "Image prompt is required."
        )
      }

      if (!sourceContext?.isImageCommand) {
        const normalizedSelectedModel = normalizeChatModelId(
          item.snapshot.selectedModel
        )
        if (!normalizedSelectedModel) {
          return t("formError.noModel")
        }
        const unavailableModel = findUnavailableChatModel(
          [normalizedSelectedModel],
          availableChatModelIds
        )
        if (unavailableModel) {
          return t(
            "playground:composer.validationModelUnavailableInline",
            "Selected model is not available on this server. Refresh models or choose a different model."
          )
        }
      }

      return null
    },
    [availableChatModelIds, isQueuedDispatchBlockedByComposerState, t]
  )

  const sendQueuedRequest = React.useCallback(
    async (item: QueuedRequest) => {
      const validationError = validateQueuedRequest(item)
      if (validationError) {
        form.setFieldError("message", validationError)
        throw new Error(validationError)
      }

      setSelectedModel(item.snapshot.selectedModel)
      setChatMode(item.snapshot.chatMode)
      setWebSearch(item.snapshot.webSearch)
      setSelectedSystemPrompt(item.snapshot.selectedSystemPrompt ?? "")
      setSelectedQuickPrompt(item.snapshot.selectedQuickPrompt ?? "")
      if (
        item.snapshot.toolChoice === "auto" ||
        item.snapshot.toolChoice === "required" ||
        item.snapshot.toolChoice === "none"
      ) {
        setToolChoice(item.snapshot.toolChoice)
      }
      setUseOCR(item.snapshot.useOCR)

      await stopListening()
      stopServerDictation()

      const sourceContext = (item.sourceContext ??
        null) as SidepanelQueuedSourceContext | null
      const documents = Array.isArray(sourceContext?.documents)
        ? sourceContext.documents
        : []

      await sendMessage({
        image: sourceContext?.isImageCommand ? "" : item.image,
        message: item.promptText,
        docs: sourceContext?.isImageCommand ? [] : documents,
        requestOverrides: {
          chatMode: item.snapshot.chatMode,
          selectedModel: item.snapshot.selectedModel,
          selectedSystemPrompt: item.snapshot.selectedSystemPrompt,
          toolChoice:
            item.snapshot.toolChoice === "auto" ||
            item.snapshot.toolChoice === "required" ||
            item.snapshot.toolChoice === "none"
              ? item.snapshot.toolChoice
              : undefined,
          useOCR: item.snapshot.useOCR,
          webSearch: item.snapshot.webSearch
        },
        imageBackendOverride: sourceContext?.isImageCommand
          ? sourceContext.imageBackendOverride
          : undefined
      })
    },
    [
      form,
      sendMessage,
      setChatMode,
      setSelectedModel,
      setSelectedQuickPrompt,
      setSelectedSystemPrompt,
      setToolChoice,
      setUseOCR,
      setWebSearch,
      stopListening,
      stopServerDictation,
      validateQueuedRequest
    ]
  )

  const resolveQueueConversationId = React.useCallback(
    () => historyId ?? serverChatId ?? null,
    [historyId, serverChatId]
  )

  const handleQueueEnqueueBlocked = React.useCallback(() => {
    notification.warning({
      message: t(
        "playground:composer.queue.attachmentsNeedManualRepairTitle",
        "Queue needs a simpler draft"
      ),
      description: t(
        "playground:composer.queue.attachmentsNeedManualRepairBody",
        "Queued requests currently support text, images, and selected tabs. Clear attached files/context before queueing this draft."
      )
    })
  }, [notification, t])

  const handleQueueEnqueueSuccess = React.useCallback(
    (isStreamingAtEnqueue: boolean) => {
      clearDraft()
      form.reset()
      clearSelectedDocuments()
      setContextFiles([])
      setKnowledgeMentionActive(false)
      textAreaFocus()
      notification.info({
        message: t("playground:composer.queue.requestQueued", "Request queued"),
        description: isStreamingAtEnqueue
          ? t(
              "playground:composer.queue.requestQueuedWhileBusy",
              "We'll run it after the current response finishes."
            )
          : t(
              "playground:composer.queue.requestQueuedWhileOffline",
              "We'll send it when your tldw server reconnects."
            )
      })
    },
    [
      clearDraft,
      clearSelectedDocuments,
      form,
      notification,
      setContextFiles,
      setKnowledgeMentionActive,
      t,
      textAreaFocus
    ]
  )

  const sidepanelQueueCancelReasonText =
    isSending && serverChatId
      ? t(
          "playground:composer.queue.cancelAndRunDisabled",
          "Cancel current & run now is not available for server-backed turns yet."
        )
      : null

  const queue = useComposerQueue({
    isConnectionReady,
    isStreaming: isSending,
    queuedMessages,
    setQueuedMessages,
    sendQueuedRequest,
    stopStreamingRequest,
    resolveConversationId: resolveQueueConversationId,
    buildQueuedDocuments,
    buildQueuedRequestSnapshot,
    isQueuedDispatchBlocked: isQueuedDispatchBlockedByComposerState,
    onEnqueueBlocked: handleQueueEnqueueBlocked,
    onEnqueueSuccess: handleQueueEnqueueSuccess,
    cancelCurrentAndRunDisabledReasonText: sidepanelQueueCancelReasonText
  })

  const queuedRequestActions = queue.queuedRequestActions
  const cancelCurrentAndRunDisabledReason =
    queue.cancelCurrentAndRunDisabledReason
  const handleRunQueuedRequest = queue.handleRunQueuedRequest
  const handleRunNextQueuedRequest = queue.handleRunNextQueuedRequest

  const queueSubmission = React.useCallback(
    ({
      promptText,
      image,
      intent
    }: {
      promptText: string
      image: string
      intent: ReturnType<typeof resolveSubmissionIntent>
    }) => {
      const documents = buildQueuedDocuments()
      return queue.enqueue({
        promptText,
        image: intent.isImageCommand ? "" : image,
        attachments: documents,
        sourceContext: {
          documents,
          imageBackendOverride: intent.isImageCommand
            ? intent.imageBackendOverride
            : undefined,
          isImageCommand: intent.isImageCommand
        },
        blockedReason: isQueuedDispatchBlockedByComposerState
          ? "draft-attachments-conflict"
          : null
      })
    },
    [
      buildQueuedDocuments,
      isQueuedDispatchBlockedByComposerState,
      queue
    ]
  )

  const submitCurrentRequest = React.useCallback(
    async (
      rawMessage: string,
      image: string,
      options?: { ignorePinnedResults?: boolean }
    ) => {
      const intent = resolveSubmissionIntent(rawMessage, options)
      if (intent.invalidImageCommand) {
        notification.error({
          message: t("error", { defaultValue: "Error" }),
          description: intent.imageCommandMissingProvider
            ? t(
                "imageCommand.missingProvider",
                "Pick an Image provider in More tools or use /generate-image:<provider> <prompt>."
              )
            : t(
                "imageCommand.invalidUsage",
                "Use /generate-image:<provider> <prompt>."
              )
        })
        return
      }

      const trimmed = intent.combinedMessage.trim()
      if (
        !intent.isImageCommand &&
        trimmed.length === 0 &&
        image.length === 0 &&
        selectedDocuments.length === 0 &&
        contextFiles.length === 0
      ) {
        return
      }

      if (intent.isImageCommand && trimmed.length === 0) {
        notification.error({
          message: t("error", { defaultValue: "Error" }),
          description: t(
            "imageCommand.missingPrompt",
            "Image prompt is required."
          )
        })
        return
      }

      if (!intent.isImageCommand) {
        const normalizedSelectedModel = normalizeChatModelId(selectedModel)
        if (!normalizedSelectedModel) {
          form.setFieldError("message", t("formError.noModel"))
          return
        }
        const unavailableModel = findUnavailableChatModel(
          [normalizedSelectedModel],
          availableChatModelIds
        )
        if (unavailableModel) {
          form.setFieldError(
            "message",
            t(
              "playground:composer.validationModelUnavailableInline",
              "Selected model is not available on this server. Refresh models or choose a different model."
            )
          )
          return
        }
      }

      const shouldQueueInsteadOfSend = isSending || !isConnectionReady
      if (shouldQueueInsteadOfSend) {
        await stopListening()
        stopServerDictation()
        queueSubmission({
          promptText: trimmed,
          image,
          intent
        })
        return
      }

      await sendCurrentFormMessageRef.current(rawMessage, image, options)
    },
    [
      availableChatModelIds,
      contextFiles.length,
      form,
      isConnectionReady,
      isSending,
      notification,
      queueSubmission,
      resolveSubmissionIntent,
      selectedDocuments.length,
      selectedModel,
      stopListening,
      stopServerDictation,
      t
    ]
  )

  const submitForm = React.useCallback(
    (options?: { ignorePinnedResults?: boolean }) => {
      form.onSubmit(async (value) => {
        await submitCurrentRequest(value.message, value.image, options)
      })()
    },
    [form, submitCurrentRequest]
  )

  const shouldQueuePrimaryAction = isSending || !isConnectionReady
  const primaryActionLabel = shouldQueuePrimaryAction
    ? t("common:queue", "Queue")
    : t("common:send", "Send")
  const primaryActionTitle = shouldQueuePrimaryAction
    ? (isSending
        ? t(
            "playground:composer.queue.primaryWhileBusy",
            "Queue this request to run after the current response."
          )
        : t(
            "playground:composer.queue.primaryWhileOffline",
            "Queue this request until your tldw server reconnects."
          )) as string
    : sendButtonTitle
  const primaryActionAriaLabel = shouldQueuePrimaryAction
    ? (t("playground:composer.queue.primaryAria", "Queue request") as string)
    : (t("playground:composer.submitAria", "Send message") as string)

  React.useEffect(() => {
    const handleDrop = (e: DragEvent) => {
      e.preventDefault()
      if (e.dataTransfer?.items) {
        for (let i = 0; i < e.dataTransfer.items.length; i++) {
          if (e.dataTransfer.items[i].type === "text/plain") {
            e.dataTransfer.items[i].getAsString((text) => {
              form.setFieldValue("message", text)
            })
          }
        }
      }
    }
    const handleDragOver = (e: DragEvent) => {
      e.preventDefault()
    }
    const el = textareaRef.current
    if (el) {
      el.addEventListener("drop", handleDrop)
      el.addEventListener("dragover", handleDragOver)
    }

    if (defaultInternetSearchOn) {
      setWebSearch(true)
    }

    if (defaultChatWithWebsite) {
      setChatMode("rag")
    }

    return () => {
      if (el) {
        el.removeEventListener("drop", handleDrop)
        el.removeEventListener("dragover", handleDragOver)
      }
    }
  }, [])

  React.useEffect(() => {
    if (defaultInternetSearchOn) {
      setWebSearch(true)
    }
  }, [defaultInternetSearchOn])

  React.useEffect(() => {
    const handler = (
      event: CustomEvent<{ message?: string; append?: boolean; ifEmptyOnly?: boolean }>
    ) => {
      const incoming = String(event.detail?.message || "").trim()
      if (!incoming) return
      const current = String(form.values.message || "")
      if (event.detail?.ifEmptyOnly && current.trim().length > 0) return
      const nextMessage =
        event.detail?.append && current.trim().length > 0
          ? `${current}\n\n${incoming}`
          : incoming
      form.setFieldValue("message", nextMessage)
      requestAnimationFrame(() => textareaRef.current?.focus())
    }
    window.addEventListener(
      "tldw:set-composer-message",
      handler as EventListener
    )
    return () => {
      window.removeEventListener(
        "tldw:set-composer-message",
        handler as EventListener
      )
    }
  }, [form, form.values.message, textareaRef])

  // Clear error messages when user starts typing (they're taking action)
  // Errors persist until user interaction rather than auto-dismissing
  React.useEffect(() => {
    if (form.values.message && form.errors.message) {
      form.clearFieldError("message")
    }
  }, [form.values.message, form.errors.message, form.clearFieldError])

  // Clear "no model" error when a model is selected
  React.useEffect(() => {
    if (selectedModel && form.errors.message) {
      form.clearFieldError("message")
    }
  }, [selectedModel, form.errors.message, form.clearFieldError])

  // Debounce placeholder changes to prevent flashing on flaky connections
  React.useEffect(() => {
    const targetPlaceholder = isConnectionReady
      ? t("form.textarea.placeholder")
      : uxState === "testing"
        ? t(
            "sidepanel:composer.connectingPlaceholder",
            "Connecting..."
          )
        : t(
            "sidepanel:composer.disconnectedPlaceholder",
            "Not connected — open Settings to connect"
          )

    // Clear any existing timeout
    if (placeholderTimeoutRef.current) {
      clearTimeout(placeholderTimeoutRef.current)
    }

    // Debounce by ~400ms to avoid flashing while keeping the UI responsive
    placeholderTimeoutRef.current = setTimeout(() => {
      setDebouncedPlaceholder(targetPlaceholder)
    }, 400)

    return () => {
      if (placeholderTimeoutRef.current) {
        clearTimeout(placeholderTimeoutRef.current)
        placeholderTimeoutRef.current = null
      }
    }
  }, [isConnectionReady, uxState, t])

  return (
    <React.Profiler id="sidepanel-form-root" onRender={onComposerRenderProfile}>
      <div
        ref={formContainerRef}
        className={`flex w-full min-w-0 flex-col items-center ${composerPadding}`}>
      <div
        className={`relative z-10 flex w-full min-w-0 flex-col items-center justify-center ${composerGap} text-body`}>
        <div className="relative flex w-full min-w-0 flex-row justify-center gap-2">
          <div
            aria-disabled={!isConnectionReady}
            className={`relative w-full min-w-0 max-w-[64rem] rounded-3xl border border-border/80 bg-surface/95 shadow-card backdrop-blur-lg duration-100 ${cardPadding}`}>
            <div>
              {/* Inline Model Parameters Panel (Pro mode only) */}
              {wrapComposerProfile(
                "sidepanel-model-params-panel",
                <ModelParamsPanel
                  onOpenFullSettings={handleOpenModelSettings}
                  selectedModel={selectedModel}
                />
              )}
              <div className="flex">
                <form
                  onSubmit={(event) => {
                    event.preventDefault()
                    void submitForm()
                  }}
                  className="min-w-0 flex-1 flex flex-col items-center"
                >
                  <input
                    id="file-upload"
                    name="file-upload"
                    type="file"
                    className="sr-only"
                    ref={attachmentHandler.fileInputRef}
                    accept="image/*"
                    multiple={false}
                    tabIndex={-1}
                    aria-hidden="true"
                    aria-label={t("playground:actions.attachImage", "Attach image") as string}
                    onChange={attachmentHandler.onFileInputChange}
                  />
                  <input
                    id="context-file-upload"
                    name="context-file-upload"
                    type="file"
                    className="sr-only"
                    ref={contextFileInputRef}
                    multiple
                    tabIndex={-1}
                    aria-hidden="true"
                    aria-label={t("playground:actions.attachDocument", "Attach document") as string}
                    onChange={handleContextFileChange}
                  />
                  <div
                    className={`flex w-full min-w-0 flex-col px-1 ${
                      !isConnectionReady
                        ? "rounded-md border border-dashed border-warn bg-warn/10"
                        : ""
                    }`}>
                    {/* Connection status indicator when disconnected */}
                    {wrapComposerProfile(
                      "sidepanel-connection-status",
                      <ConnectionStatusIndicator
                        isConnectionReady={isConnectionReady}
                        uxState={uxState}
                        onOpenSettings={openSettings}
                      />
                    )}
                    {/* Knowledge Search: search KB, insert snippets, ask directly */}
                    {isProMode && (
                      wrapComposerProfile(
                        "sidepanel-knowledge-panel",
                        <KnowledgePanel
                          onInsert={handleKnowledgeInsert}
                          onAsk={handleKnowledgeAsk}
                          open={knowledgePanelOpen}
                          onOpenChange={handleKnowledgePanelOpenChange}
                          currentMessage={knowledgePanelOpen ? deferredComposerInput : ""}
                          showAttachedContext
                          attachedTabs={selectedDocuments}
                          availableTabs={availableTabs}
                          attachedFiles={contextFiles}
                          onAddTab={addDocument}
                          onRemoveTab={removeDocument}
                          onClearTabs={clearSelectedDocuments}
                          onRefreshTabs={reloadTabs}
                          onAddFile={handleKnowledgeAddFile}
                          onRemoveFile={handleKnowledgeRemoveFile}
                          onClearFiles={handleKnowledgeClearFiles}
                        />
                      )
                    )}
                    {/* Queued messages banner - shown above input area */}
                    {wrapComposerProfile(
                      "sidepanel-queued-banner",
                      <ChatQueuePanel
                        queue={queuedMessages}
                        isConnectionReady={isConnectionReady}
                        isStreaming={isSending}
                        onRunNext={handleRunNextQueuedRequest}
                        onRunNow={handleRunQueuedRequest}
                        onDelete={queuedRequestActions.remove}
                        onMove={queuedRequestActions.move}
                        onUpdate={queuedRequestActions.update}
                        onClearAll={queuedRequestActions.clear}
                        onOpenDiagnostics={openDiagnostics}
                        forceRunDisabledReason={cancelCurrentAndRunDisabledReason}
                      />
                    )}
                    {contextChips.length > 0 && (
                      <div className="px-2 pb-2">
                        <div className="mb-1 text-[10px] font-semibold uppercase tracking-[0.2em] text-text-subtle">
                          {t("playground:composer.contextLabel", "Context")}
                        </div>
                        {wrapComposerProfile(
                          "sidepanel-context-chips",
                          <ContextChips
                            items={contextChips}
                            ariaLabel={t("playground:composer.contextLabel", "Context:")}
                            className="flex flex-wrap items-center gap-2"
                          />
                        )}
                      </div>
                    )}
                    {(() => {
                      const composerTextareaShellNode = (
                        <div className="relative min-w-0">
                      {wrapComposerProfile(
                        "sidepanel-textarea-shell",
                        <div className="relative min-w-0 rounded-2xl border border-border/70 bg-surface/80 px-1 py-1.5 transition focus-within:border-focus/60 focus-within:ring-2 focus-within:ring-focus/30">
                          <SlashCommandMenu
                            open={showSlashMenu}
                            commands={filteredSlashCommands}
                            activeIndex={slashActiveIndex}
                            onActiveIndexChange={setSlashActiveIndex}
                            onSelect={handleSlashCommandPick}
                            emptyLabel={t(
                              "common:commandPalette.noResults",
                              "No results found"
                            )}
                            className="absolute bottom-full left-3 right-3 mb-2"
                          />
                          <MentionsMenu
                            open={showMentionMenu}
                            items={mentionItems}
                            activeIndex={mentionActiveIndex}
                            onActiveIndexChange={setMentionActiveIndex}
                            onSelect={handleMentionSelect}
                            emptyLabel={t(
                              "sidepanel:composer.noMentions",
                              "No matches found"
                            )}
                            className="absolute bottom-full left-3 right-3 mb-2"
                          />
                          <textarea
                            id="textarea-message"
                            onKeyDown={(e) => handleKeyDown(e)}
                            ref={textareaRef}
                            data-testid="chat-input"
                            className={`w-full resize-none border-0 bg-transparent px-3 py-2 text-body text-text placeholder:text-text-muted/80 focus-within:outline-none focus:ring-0 focus-visible:ring-0 ring-0 dark:ring-0 ${
                              !isConnectionReady
                                ? "cursor-not-allowed text-text-muted placeholder:text-text-subtle"
                                : ""
                            }`}
                            readOnly={!isConnectionReady}
                            aria-readonly={!isConnectionReady}
                            aria-disabled={!isConnectionReady}
                            aria-label={
                              !isConnectionReady
                                ? t(
                                    "sidepanel:composer.disconnectedAriaLabel",
                                    "Message input (read-only: not connected to server)"
                                  )
                                : t("sidepanel:composer.messageAriaLabel", "Message input")
                            }
                            onPaste={handlePaste}
                            rows={1}
                            style={{ minHeight: `${textareaMinHeight}px` }}
                            tabIndex={0}
                            onCompositionStart={() => {
                              if (!isFirefoxTarget) {
                                setTyping(true)
                              }
                            }}
                            onCompositionEnd={() => {
                              if (!isFirefoxTarget) {
                                setTyping(false)
                              }
                            }}
                            placeholder={debouncedPlaceholder || t("form.textarea.placeholder")}
                            {...messageInputProps}
                            onChange={(event) => {
                              messageInputProps.onChange(event)
                              if (tabMentionsEnabled && textareaRef.current) {
                                handleTextChange(
                                  event.target.value,
                                  textareaRef.current.selectionStart || 0
                                )
                              }
                            }}
                            onSelect={() => {
                              if (tabMentionsEnabled && textareaRef.current) {
                                handleTextChange(
                                  textareaRef.current.value,
                                  textareaRef.current.selectionStart || 0
                                )
                              }
                            }}
                          />
                        </div>
                      )}
                      {/* Draft saved indicator */}
                      {draftSaved && (
                        <span
                          className="absolute bottom-1 right-2 text-label text-text-subtle transition-opacity pointer-events-none"
                          role="status"
                          aria-live="polite"
                        >
                          {t("sidepanel:composer.draftSaved", "Draft saved")}
                        </span>
                      )}
                        </div>
                      )

                      const composerInlineMessagesNode = (
                        <>
                          {form.errors.message && (
                            <div
                              role="alert"
                              aria-live="assertive"
                              aria-atomic="true"
                              className="flex items-center justify-between gap-2 px-2 py-1 text-xs text-danger animate-shake"
                            >
                              <div className="flex items-center gap-2">
                                <svg className="h-3.5 w-3.5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                                  <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                                </svg>
                                <span>{form.errors.message}</span>
                              </div>
                              <button
                                type="button"
                                onClick={() => form.clearFieldError("message")}
                                className="flex-shrink-0 text-danger hover:text-danger"
                                aria-label={t("common:dismiss", "Dismiss")}
                                title={t("common:dismiss", "Dismiss")}
                              >
                                <X className="h-3 w-3" />
                              </button>
                            </div>
                          )}
                          {!form.errors.message && isConnectionReady && !streaming && isProMode && (
                            <div className="px-2 py-1 text-label text-text-subtle">
                              {!selectedModel ? (
                                <span className="flex items-center gap-1">
                                  <svg className="h-3 w-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                                  </svg>
                                  {t("sidepanel:composer.hints.selectModel", "Select a model above to start chatting")}
                                </span>
                              ) : form.values.message.trim().length === 0 && form.values.image.length === 0 ? (
                                <span>
                                  {sendWhenEnter
                                    ? t("sidepanel:composer.hints.typeAndEnter", "Type a message and press Enter to send")
                                    : t("sidepanel:composer.hints.typeAndClick", "Type a message and click Send")}
                                </span>
                              ) : null}
                            </div>
                          )}
                        </>
                      )

                      const composerControlAreaNode = (
                        <div className="mt-2 flex min-w-0 flex-col gap-2">
                      <Tooltip title={persistenceTooltip}>
                        <div className="flex items-center gap-2">
                          <Switch
                            size="small"
                            checked={!temporaryChat}
                            disabled={temporaryChatLocked || isFireFoxPrivateMode}
                            onChange={(checked) =>
                              handleToggleTemporaryChat(!checked)
                            }
                            aria-label={temporaryChatToggleLabel as string}
                          />
                          <span className="text-xs text-text whitespace-nowrap">
                            {temporaryChatToggleLabel}
                          </span>
                        </div>
                      </Tooltip>
                      <div className="flex w-full min-w-0 flex-row flex-wrap items-center justify-between gap-1.5">
                      {isProMode ? (
                        <>
                          {/* Control Row - contains Prompt, Model, RAG, and More tools */}
                          {wrapComposerProfile(
                            "sidepanel-control-row",
                            <ControlRow
                              selectedSystemPrompt={selectedSystemPrompt}
                              setSelectedSystemPrompt={setSelectedSystemPrompt}
                              setSelectedQuickPrompt={setSelectedQuickPrompt}
                              selectedCharacterId={selectedCharacterId}
                              setSelectedCharacterId={setSelectedCharacterId}
                              serverChatId={serverChatId}
                              conversationContextComposition={
                                conversationContextComposition.composition
                              }
                              conversationContextStatus={
                                conversationContextComposition.status
                              }
                              conversationContextSaveSelection={
                                conversationContextComposition.saveSelection
                              }
                              webSearch={webSearch}
                              setWebSearch={setWebSearch}
                              chatMode={chatMode}
                              setChatMode={setChatMode}
                              onImageUpload={onInputChange}
                              onToggleRag={handleRagToggle}
                              isConnected={isConnectionReady}
                              toolChoice={toolChoice}
                              setToolChoice={setToolChoice}
                              chatLoopStatus={chatLoopState.status}
                              pendingApprovalsCount={chatLoopState.pendingApprovals.length}
                              runningToolCount={chatLoopState.inflightToolCallIds.length}
                            />
                          )}
                          <div className="flex min-w-0 flex-wrap items-center justify-end gap-2">
                            <div
                              role="group"
                              aria-label={t(
                                "playground:composer.actions",
                                "Send options"
                              )}
                              className="flex min-w-0 flex-wrap items-center gap-2">
                              {/* L15: gap-2 provides visual separation */}
                              <>
                                {!streaming ? (
                                  <>
                                    <Tooltip
                                      title={t("playground:actions.upload", "Attach image")}
                                    >
                                      <button
                                        type="button"
                                        onClick={openUploadDialog}
                                        data-testid="chat-upload-image-inline"
                                        className="inline-flex min-h-[44px] min-w-[44px] items-center justify-center rounded-md border border-border bg-surface text-text-muted transition-colors hover:bg-surface2 hover:text-text focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
                                        aria-label={t(
                                          "playground:actions.upload",
                                          "Attach image"
                                        )}
                                        title={t(
                                          "playground:actions.upload",
                                          "Attach image"
                                        )}
                                      >
                                        <ImageIcon className="h-4 w-4" />
                                      </button>
                                    </Tooltip>
                                    <div className="flex items-center gap-1">
                                      <Tooltip
                                        title={
                                          voiceChatAvailable
                                            ? voiceChatStatusLabel
                                            : voiceChatUnavailableMessage
                                        }
                                      >
                                        <button
                                          type="button"
                                          onClick={handleVoiceChatToggle}
                                          disabled={!voiceChatAvailable || streaming}
                                          className={`rounded-md border p-1 transition hover:bg-surface2 disabled:cursor-not-allowed disabled:opacity-50 ${voiceChatToneClass}`}
                                          aria-label={voiceChatStatusLabel}
                                        >
                                          <Headphones className="h-4 w-4" />
                                        </button>
                                      </Tooltip>
                                      <Popover content={voiceChatSettingsContent} trigger="click">
                                        <button
                                          type="button"
                                          className="rounded-md border border-border p-1 text-text-muted hover:bg-surface2 hover:text-text"
                                          aria-label={t(
                                            "playground:voiceChat.settingsButton",
                                            "Voice chat settings"
                                          )}
                                        >
                                          <Settings2 className="h-3.5 w-3.5" />
                                        </button>
                                      </Popover>
                                    </div>
                                    {hasVoiceInputControls && (
                                      <Tooltip
                                        title={
                                          !speechAvailable
                                            ? t(
                                                "playground:actions.speechUnavailableBody",
                                                "Connect to a tldw server that exposes the audio transcriptions API to use dictation."
                                              )
                                            : speechUsesServer
                                              ? t(
                                                  "playground:tooltip.speechToTextServer",
                                                  "Dictation via your tldw server"
                                                )
                                              : t(
                                                  "playground:tooltip.speechToTextBrowser",
                                                  "Dictation via browser speech recognition"
                                                )
                                        }
                                      >
                                        <button
                                          type="button"
                                          onClick={handleDictationToggle}
                                          disabled={!speechAvailable || voiceChatEnabled}
                                          className={`rounded-md border border-border p-1 text-text-muted hover:bg-surface2 hover:text-text disabled:cursor-not-allowed disabled:opacity-50 ${
                                            speechAvailable &&
                                            ((speechUsesServer && isServerDictating) ||
                                              (!speechUsesServer && isListening))
                                              ? "border-primary text-primaryStrong"
                                              : ""
                                          }`}
                                          aria-label={
                                            !speechAvailable
                                              ? (t(
                                                  "playground:actions.speechUnavailableTitle",
                                                  "Dictation unavailable"
                                                ) as string)
                                              : speechUsesServer
                                                ? (isServerDictating
                                                    ? (t("playground:actions.speechStop", "Stop dictation") as string)
                                                    : (t("playground:actions.speechStart", "Start dictation") as string))
                                                : (isListening
                                                    ? (t("playground:actions.speechStop", "Stop dictation") as string)
                                                    : (t("playground:actions.speechStart", "Start dictation") as string))
                                          }
                                          title={
                                            !speechAvailable
                                              ? (t(
                                                  "playground:actions.speechUnavailableTitle",
                                                  "Dictation unavailable"
                                                ) as string)
                                              : speechUsesServer
                                                ? (isServerDictating
                                                    ? (t("playground:actions.speechStop", "Stop dictation") as string)
                                                    : (t("playground:actions.speechStart", "Start dictation") as string))
                                                : (isListening
                                                    ? (t("playground:actions.speechStop", "Stop dictation") as string)
                                                    : (t("playground:actions.speechStart", "Start dictation") as string))
                                          }
                                        >
                                          <MicIcon className="h-4 w-4" />
                                        </button>
                                      </Tooltip>
                                    )}
                                    <Tooltip
                                      title={t(
                                        "playground:characterRail.title",
                                        "Character controls"
                                      )}
                                    >
                                      <button
                                        type="button"
                                        data-testid="chat-character-controls-trigger"
                                        onClick={() =>
                                          setCharacterControlsOpen(true)
                                        }
                                        className="inline-flex h-11 w-11 min-h-[44px] min-w-[44px] items-center justify-center rounded-md border border-border bg-surface text-text-muted transition-colors hover:bg-surface2 hover:text-text focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
                                        aria-label={t(
                                          "playground:characterRail.title",
                                          "Character controls"
                                        )}
                                      >
                                        <UserCircle2 className="h-4 w-4" />
                                      </button>
                                    </Tooltip>
                                  </>
                                ) : (
                                  <Tooltip title={t("tooltip.stopStreaming")}>
                                    <button
                                      type="button"
                                      onClick={stopStreamingRequest}
                                      data-testid="chat-stop-streaming"
                                      className="rounded-md border border-border p-1 text-text-muted hover:bg-surface2 hover:text-text"
                                      title={t(
                                        "playground:composer.stopStreaming",
                                        "Stop streaming response"
                                      )}
                                    >
                                      <StopCircleIcon className="h-5 w-5" />
                                      <span className="sr-only">
                                        {t(
                                          "playground:composer.stopStreaming",
                                          "Stop streaming response"
                                        )}
                                      </span>
                                    </button>
                                  </Tooltip>
                                )}
                                <Space.Compact>
                                  <button
                                    aria-label={primaryActionAriaLabel}
                                    data-testid="chat-send"
                                    title={primaryActionTitle}
                                    type={shouldQueuePrimaryAction ? "button" : "submit"}
                                    onClick={
                                      shouldQueuePrimaryAction
                                        ? () => {
                                            void submitForm()
                                          }
                                        : undefined
                                    }
                                    className="inline-flex min-h-[44px] items-center gap-2 rounded-l-md border border-border bg-surface px-3 text-sm text-text transition-colors hover:bg-surface2"
                                  >
                                    {!shouldQueuePrimaryAction && sendWhenEnter ? (
                                      <svg
                                        xmlns="http://www.w3.org/2000/svg"
                                        fill="none"
                                        stroke="currentColor"
                                        strokeLinecap="round"
                                        strokeLinejoin="round"
                                        strokeWidth="2"
                                        className="h-4 w-4"
                                        viewBox="0 0 24 24"
                                      >
                                        <path d="M9 10L4 15 9 20"></path>
                                        <path d="M20 4v7a4 4 0 01-4 4H4"></path>
                                      </svg>
                                    ) : null}
                                    {primaryActionLabel}
                                  </button>
                                  <Dropdown
                                    trigger={["click"]}
                                    menu={{
                                      items: [
                                        {
                                          key: "send-section",
                                          type: "group",
                                          label: t(
                                            "playground:composer.actions",
                                            "Send options"
                                          ),
                                          children: [
                                            {
                                              key: 1,
                                              label: (
                                                <Checkbox
                                                  checked={sendWhenEnter}
                                                  onChange={(e) =>
                                                    setSendWhenEnter(e.target.checked)
                                                  }>
                                                  {t("sendWhenEnter")}
                                                </Checkbox>
                                              )
                                            }
                                          ]
                                        },
                                        {
                                          type: "divider",
                                          key: "divider-1"
                                        },
                                        {
                                          key: "context-section",
                                          type: "group",
                                          label: t(
                                            "playground:composer.coreTools",
                                            "Conversation options"
                                          ),
                                          children: [
                                            {
                                              key: 2,
                                              label: (
                                                <Checkbox
                                                  checked={chatMode === "rag"}
                                                  onChange={(e) => {
                                                    setChatMode(
                                                      e.target.checked
                                                        ? "rag"
                                                        : "normal"
                                                    )
                                                  }}>
                                                  {t("common:chatWithCurrentPage")}
                                                </Checkbox>
                                              )
                                            },
                                            {
                                              key: 3,
                                              label: (
                                                <Checkbox
                                                  checked={useOCR}
                                                  onChange={(e) =>
                                                    setUseOCR(e.target.checked)
                                                  }>
                                                  {t("useOCR")}
                                                </Checkbox>
                                              )
                                            }
                                          ]
                                        }
                                      ]
                                    }}
                                  >
                                    <button
                                      type="button"
                                      aria-label={
                                        t(
                                          "playground:composer.sendOptions",
                                          "Open send options"
                                        ) as string
                                      }
                                      title={
                                        t(
                                          "playground:composer.sendOptions",
                                          "Open send options"
                                        ) as string
                                      }
                                      className="inline-flex min-h-[44px] items-center rounded-r-md border border-l-0 border-border bg-surface px-2 text-text transition-colors hover:bg-surface2"
                                    >
                                      <svg
                                        xmlns="http://www.w3.org/2000/svg"
                                        fill="none"
                                        viewBox="0 0 24 24"
                                        strokeWidth={1.5}
                                        stroke="currentColor"
                                        className="w-4 h-4"
                                      >
                                        <path
                                          strokeLinecap="round"
                                          strokeLinejoin="round"
                                          d="m19.5 8.25-7.5 7.5-7.5-7.5"
                                        />
                                      </svg>
                                    </button>
                                  </Dropdown>
                                </Space.Compact>
                                <Tooltip
                                  title={
                                    t("common:currentChatModelSettings") as string
                                  }>
                                  <button
                                    type="button"
                                    onClick={() => setOpenModelSettings(true)}
                                    className={`rounded-md p-1 text-text-muted hover:bg-surface2 hover:text-text ${
                                      streaming ? "border border-border" : ""
                                    }`}
                                    title={t(
                                      "playground:composer.openModelSettings",
                                      "Open current chat settings"
                                    )}
                                  >
                                    <Gauge className="h-5 w-5" />
                                    <span className="sr-only">
                                      {t(
                                        "playground:composer.openModelSettings",
                                        "Open current chat settings"
                                      )}
                                    </span>
                                  </button>
                                </Tooltip>
                              </>
                            </div>
                          </div>
                        </>
                      ) : (
                        <>
                          <div className="flex flex-wrap items-end gap-2">
                            <div className="flex flex-col items-center gap-1">
                              <Tooltip
                                title={t("playground:actions.upload", "Attach image")}
                              >
                                <button
                                  type="button"
                                  onClick={openUploadDialog}
                                  className="h-11 w-11 min-h-[44px] min-w-[44px] rounded-full border border-border p-0 text-text-muted hover:bg-surface2 hover:text-text focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
                                  aria-label={t(
                                    "playground:actions.upload",
                                    "Attach image"
                                  )}
                                  title={t(
                                    "playground:actions.upload",
                                    "Attach image"
                                  )}
                                >
                                  <ImageIcon className="h-4 w-4" />
                                </button>
                              </Tooltip>
                              <span className="text-[10px] font-medium leading-none text-text-subtle">
                                {t("playground:actions.uploadShort", "Image")}
                              </span>
                            </div>
                            {hasVoiceInputControls && (
                              <div className="flex flex-wrap items-end gap-1.5">
                                <div className="flex flex-col items-center gap-1">
                                  <Tooltip
                                    title={
                                      voiceChatAvailable
                                        ? voiceChatStatusLabel
                                        : voiceChatUnavailableMessage
                                    }
                                  >
                                    <button
                                      type="button"
                                      onClick={handleVoiceChatToggle}
                                      disabled={!voiceChatAvailable || streaming}
                                      className={`h-11 w-11 min-h-[44px] min-w-[44px] rounded-full border p-0 transition hover:bg-surface2 disabled:cursor-not-allowed disabled:opacity-50 ${voiceChatToneClass}`}
                                      aria-label={voiceChatStatusLabel}
                                    >
                                      <Headphones className="h-4 w-4" />
                                    </button>
                                  </Tooltip>
                                  <span className="text-[10px] font-medium leading-none text-text-subtle">
                                    {t("playground:voiceChat.toggleShort", "Voice")}
                                  </span>
                                </div>
                                <div className="flex flex-col items-center gap-1">
                                  <Popover content={voiceChatSettingsContent} trigger="click">
                                    <button
                                      type="button"
                                      className="h-11 w-11 min-h-[44px] min-w-[44px] rounded-full border border-border p-0 text-text-muted hover:bg-surface2"
                                      aria-label={t(
                                        "playground:voiceChat.settingsButton",
                                        "Voice chat settings"
                                      )}
                                    >
                                      <Settings2 className="h-3.5 w-3.5" />
                                    </button>
                                  </Popover>
                                  <span className="text-[10px] font-medium leading-none text-text-subtle">
                                    {t("playground:voiceChat.settingsShort", "Config")}
                                  </span>
                                </div>
                                <div className="flex flex-col items-center gap-1">
                                  <Tooltip
                                    title={
                                      !speechAvailable
                                        ? t(
                                            "playground:actions.speechUnavailableBody",
                                            "Connect to a tldw server that exposes the audio transcriptions API to use dictation."
                                          )
                                        : speechUsesServer
                                          ? t(
                                              "playground:tooltip.speechToTextServer",
                                              "Dictation via your tldw server"
                                            )
                                          : t(
                                              "playground:tooltip.speechToTextBrowser",
                                              "Dictation via browser speech recognition"
                                            )
                                    }
                                  >
                                    <button
                                      type="button"
                                      onClick={handleDictationToggle}
                                      disabled={!speechAvailable || voiceChatEnabled}
                                      className={`h-11 w-11 min-h-[44px] min-w-[44px] rounded-full border border-border p-0 text-text-muted hover:bg-surface2 hover:text-text focus:outline-none focus-visible:ring-2 focus-visible:ring-focus disabled:cursor-not-allowed disabled:opacity-50 ${
                                        speechAvailable &&
                                        ((speechUsesServer && isServerDictating) ||
                                          (!speechUsesServer && isListening))
                                          ? "border-primary text-primaryStrong"
                                          : ""
                                      }`}
                                      aria-label={
                                        !speechAvailable
                                          ? (t(
                                              "playground:actions.speechUnavailableTitle",
                                              "Dictation unavailable"
                                            ) as string)
                                          : speechUsesServer
                                            ? (isServerDictating
                                                ? (t("playground:actions.speechStop", "Stop dictation") as string)
                                                : (t("playground:actions.speechStart", "Start dictation") as string))
                                            : (isListening
                                                ? (t("playground:actions.speechStop", "Stop dictation") as string)
                                                : (t("playground:actions.speechStart", "Start dictation") as string))
                                      }
                                      title={
                                        !speechAvailable
                                          ? (t(
                                              "playground:actions.speechUnavailableTitle",
                                              "Dictation unavailable"
                                            ) as string)
                                          : speechUsesServer
                                            ? (isServerDictating
                                                ? (t("playground:actions.speechStop", "Stop dictation") as string)
                                                : (t("playground:actions.speechStart", "Start dictation") as string))
                                            : (isListening
                                                ? (t("playground:actions.speechStop", "Stop dictation") as string)
                                                : (t("playground:actions.speechStart", "Start dictation") as string))
                                      }
                                    >
                                      <MicIcon className="h-4 w-4" />
                                    </button>
                                  </Tooltip>
                                  <span className="text-[10px] font-medium leading-none text-text-subtle">
                                    {t("playground:actions.speechShort", "Dictate")}
                                  </span>
                                </div>
                                <div className="flex flex-col items-center gap-1">
                                  <Tooltip
                                    title={t(
                                      "playground:characterRail.title",
                                      "Character controls"
                                    )}
                                  >
                                    <button
                                      type="button"
                                      data-testid="chat-character-controls-trigger"
                                      onClick={() =>
                                        setCharacterControlsOpen(true)
                                      }
                                      className="h-11 w-11 min-h-[44px] min-w-[44px] rounded-full border border-border p-0 text-text-muted transition-colors hover:bg-surface2 hover:text-text focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
                                      aria-label={t(
                                        "playground:characterRail.title",
                                        "Character controls"
                                      )}
                                    >
                                      <UserCircle2 className="h-4 w-4" />
                                    </button>
                                  </Tooltip>
                                  <span className="text-[10px] font-medium leading-none text-text-subtle">
                                    {t("playground:characterRail.shortLabel", "Character")}
                                  </span>
                                </div>
                              </div>
                            )}
                          </div>
                          <div className="flex items-end gap-2">
                            {streaming && (
                              <div className="flex flex-col items-center gap-1">
                                <Tooltip title={t("tooltip.stopStreaming")}>
                                  <button
                                    type="button"
                                    onClick={stopStreamingRequest}
                                    data-testid="chat-stop-streaming"
                                    className="h-11 w-11 min-h-[44px] min-w-[44px] rounded-full border border-border p-0 text-text-muted hover:bg-surface2 hover:text-text"
                                    aria-label={t(
                                      "playground:composer.stopStreaming",
                                      "Stop streaming response"
                                    )}
                                    title={t(
                                      "playground:composer.stopStreaming",
                                      "Stop streaming response"
                                    )}
                                  >
                                    <StopCircleIcon className="h-4 w-4" />
                                  </button>
                                </Tooltip>
                                <span className="text-[10px] font-medium leading-none text-text-subtle">
                                  {t("playground:composer.stopShort", "Stop")}
                                </span>
                              </div>
                            )}
                            <Button
                              type={shouldQueuePrimaryAction ? "button" : "submit"}
                              onClick={
                                shouldQueuePrimaryAction
                                  ? () => {
                                      void submitForm()
                                    }
                                  : undefined
                              }
                              variant="primary"
                              size="sm"
                              ariaLabel={primaryActionAriaLabel}
                              title={primaryActionTitle}
                              className="min-h-[44px] rounded-full px-4 text-[11px] font-semibold uppercase tracking-[0.12em]"
                            >
                              {primaryActionLabel}
                            </Button>
                          </div>
                        </>
                      )}
                        </div>
                        </div>
                      )

                      if (nextgenComposerEnabled) {
                        const sharedNoticesSlot = composerInlineMessagesNode

                        // Compact brief for V3 in the sidepanel — same idea
                        // as Playground's, trimmed to the data Sidepanel
                        // actually has (no character picker, no source list
                        // beyond chat-with-page mode). Click handlers route
                        // to the same actions the legacy controls expose.
                        const briefSections = [
                          {
                            id: "sidepanel-brief",
                            label: "Brief",
                            fields: [
                              {
                                id: "model",
                                fieldKey: "mdl",
                                value: selectedModel || "—",
                                active: Boolean(selectedModel),
                                onClick: () => setOpenModelSettings(true),
                                "aria-label": selectedModel
                                  ? `Change model (current: ${selectedModel})`
                                  : "Pick a model",
                              },
                              {
                                id: "mode",
                                fieldKey: "mod",
                                value: chatMode,
                                active: chatMode !== "normal",
                                onClick: () =>
                                  setChatMode(
                                    chatMode === "rag" ? "normal" : "rag"
                                  ),
                                "aria-label":
                                  chatMode === "rag"
                                    ? "Switch to normal chat mode"
                                    : "Switch to chat-with-page (RAG) mode",
                              },
                              {
                                id: "web",
                                fieldKey: "web",
                                value: webSearch ? "on" : "off",
                                active: webSearch,
                                onClick: () => setWebSearch(!webSearch),
                                "aria-label": webSearch
                                  ? "Turn web search off"
                                  : "Turn web search on",
                              },
                            ],
                          },
                        ]

                        const variantNode =
                          nextgenComposerVariant === "v5" ? (
                            <ChatComposer
                              variant="v5"
                              message={form.values.message}
                              onMessageChange={(value) =>
                                form.setFieldValue("message", value)
                              }
                              onSend={() => void submitForm()}
                              sending={isSending}
                              stopStreaming={stopStreamingRequest}
                              onPaletteTrigger={() => {
                                // V5 design: ⌘K injects `/` to open the
                                // textarea's existing slash menu.
                                const current = form.values.message
                                form.setFieldValue(
                                  "message",
                                  current.endsWith("/") ? current : `${current}/`
                                )
                                textareaRef.current?.focus()
                              }}
                              textareaSlot={composerTextareaShellNode}
                              facetsSlot={composerControlAreaNode}
                              noticesSlot={sharedNoticesSlot}
                            />
                          ) : nextgenComposerVariant === "v3" ? (
                            <ChatComposer
                              variant="v3"
                              message={form.values.message}
                              onMessageChange={(value) =>
                                form.setFieldValue("message", value)
                              }
                              onSend={() => void submitForm()}
                              sending={isSending}
                              stopStreaming={stopStreamingRequest}
                              briefSections={briefSections}
                              textareaSlot={composerTextareaShellNode}
                              bottomBarSlot={composerControlAreaNode}
                              noticesSlot={sharedNoticesSlot}
                            />
                          ) : (
                            <ChatComposer
                              variant="v1"
                              message={form.values.message}
                              onMessageChange={(value) =>
                                form.setFieldValue("message", value)
                              }
                              onSend={() => void submitForm()}
                              sending={isSending}
                              stopStreaming={stopStreamingRequest}
                              density="compact"
                              textareaSlot={composerTextareaShellNode}
                              bottomBarSlot={composerControlAreaNode}
                              noticesSlot={sharedNoticesSlot}
                            />
                          )

                        return (
                          <div data-testid="nextgen-composer-wrapper">
                            {variantNode}
                          </div>
                        )
                      }

                      return (
                        <>
                          {composerTextareaShellNode}
                          {composerInlineMessagesNode}
                          {composerControlAreaNode}
                        </>
                      )
                    })()}
                  </div>
                </form>
              </div>
            </div>
          </div>
        </div>
      </div>
      {/* Mount heavy overlays only when open to avoid keystroke-time rerenders. */}
      {openModelSettings && (
        <CurrentChatModelSettings
          open={openModelSettings}
          setOpen={setOpenModelSettings}
          isOCREnabled={useOCR}
        />
      )}
      {openActorSettings && (
        <ActorPopout open={openActorSettings} setOpen={setOpenActorSettings} />
      )}
      <Modal
        open={characterControlsOpen}
        onCancel={() => setCharacterControlsOpen(false)}
        footer={null}
        title={t("playground:characterRail.title", "Character controls")}
        centered
      >
        <CharacterControlsSheet
          beforeTrackedStart={handlePrepareTrackedStart}
          onRequestClose={() => setCharacterControlsOpen(false)}
        />
      </Modal>
      {documentGeneratorOpen && (
        <DocumentGeneratorDrawer
          open={documentGeneratorOpen}
          onClose={() => {
            setDocumentGeneratorOpen(false)
            setDocumentGeneratorSeed({})
          }}
          conversationId={
            documentGeneratorSeed?.conversationId ?? serverChatId ?? null
          }
          defaultModel={selectedModel || null}
          seedMessage={documentGeneratorSeed?.message ?? null}
          seedMessageId={documentGeneratorSeed?.messageId ?? null}
        />
      )}
        {shouldRenderQuickIngest && (
          <QuickIngestModal
            open={ingestOpen}
            autoProcessQueued={autoProcessQueuedIngest}
            onClose={() => {
              hideQuickIngestSession()
              setIngestOpen(false)
              setAutoProcessQueuedIngest(false)
              requestAnimationFrame(() => quickIngestBtnRef.current?.focus())
            }}
          />
        )}
      </div>
    </React.Profiler>
  )
}
