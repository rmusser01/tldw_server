// @vitest-environment jsdom
import React from "react"
import { render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { AssistantSelect } from "@/components/Common/AssistantSelect"
import { PlaygroundEmpty } from "../PlaygroundEmpty"
import { PlaygroundForm } from "../PlaygroundForm"

const onSubmitMock = vi.hoisted(() => vi.fn(async (_payload: unknown) => null))
const createChatCompletionMock = vi.hoisted(() =>
  vi.fn(async () => ({
    json: async () => ({ choices: [] })
  }))
)
const speechRecognitionState = vi.hoisted(() => ({
  transcript: "",
  isListening: false,
  supported: false,
  start: vi.fn(),
  stop: vi.fn(),
  resetTranscript: vi.fn()
}))
const serverDictationState = vi.hoisted(() => ({
  isServerDictating: false,
  startServerDictation: vi.fn(async () => undefined),
  stopServerDictation: vi.fn(),
  lastOptions: null as null | Record<string, any>
}))
const dictationStrategyState = vi.hoisted(() => ({
  value: {
    requestedMode: "auto",
    resolvedMode: "unavailable",
    speechAvailable: false,
    speechUsesServer: false,
    isDictating: false,
    toggleIntent: "unavailable",
    autoFallbackActive: false,
    autoFallbackErrorClass: null,
    recordServerError: vi.fn(() => ({
      errorClass: "unknown_error",
      appliedFallback: false
    })),
    recordServerSuccess: vi.fn(),
    clearAutoFallback: vi.fn()
  } as any
}))
const playgroundFormMessageOptionState = vi.hoisted(() => ({
  value: null as any
}))
const playgroundFormConnectionState = vi.hoisted(() => ({
  phase: "connected",
  isConnected: true
}))
const selectedAssistantMock = vi.hoisted(() => ({
  setSelectedAssistant: vi.fn(async (_next: unknown) => undefined)
}))

const createMessageOptionState = () => ({
  onSubmit: onSubmitMock,
  messages: [],
  selectedModel: "deepseek-chat",
  selectedModelIsLoading: false,
  setSelectedModel: vi.fn(),
  chatMode: "normal",
  setChatMode: vi.fn(),
  compareMode: false,
  setCompareMode: vi.fn(),
  compareFeatureEnabled: false,
  setCompareFeatureEnabled: vi.fn(),
  compareSelectedModels: [],
  setCompareSelectedModels: vi.fn(),
  compareMaxModels: 3,
  setCompareMaxModels: vi.fn(),
  speechToTextLanguage: "en-US",
  stopStreamingRequest: vi.fn(),
  streaming: false,
  webSearch: false,
  setWebSearch: vi.fn(),
  toolChoice: "auto",
  setToolChoice: vi.fn(),
  selectedQuickPrompt: null,
  textareaRef: { current: null },
  setSelectedQuickPrompt: vi.fn(),
  selectedSystemPrompt: null,
  setSelectedSystemPrompt: vi.fn(),
  temporaryChat: false,
  setTemporaryChat: vi.fn(),
  clearChat: vi.fn(),
  useOCR: false,
  setUseOCR: vi.fn(),
  defaultInternetSearchOn: false,
  setHistory: vi.fn(),
  history: [],
  uploadedFiles: [],
  fileRetrievalEnabled: false,
  setFileRetrievalEnabled: vi.fn(),
  handleFileUpload: vi.fn(),
  removeUploadedFile: vi.fn(),
  clearUploadedFiles: vi.fn(),
  queuedMessages: [],
  addQueuedMessage: vi.fn(),
  setQueuedMessages: vi.fn(),
  clearQueuedMessages: vi.fn(),
  serverChatId: null,
  setServerChatId: vi.fn(),
  serverChatState: "in-progress",
  setServerChatState: vi.fn(),
  serverChatSource: null,
  setServerChatSource: vi.fn(),
  setServerChatVersion: vi.fn(),
  replyTarget: null,
  clearReplyTarget: vi.fn(),
  ragPinnedResults: [],
  messageSteeringMode: "default",
  messageSteeringForceNarrate: false,
  contextFiles: [],
  documentContext: [],
  selectedKnowledge: null,
  ragMediaIds: []
})

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, fallbackOrOptions?: string | Record<string, unknown>, options?: Record<string, unknown>) => {
      const normalizedOptions =
        typeof fallbackOrOptions === "object" && fallbackOrOptions !== null
          ? fallbackOrOptions
          : options
      const template =
        typeof fallbackOrOptions === "string"
          ? fallbackOrOptions
          : typeof normalizedOptions?.defaultValue === "string"
            ? normalizedOptions.defaultValue
            : key
      if (!normalizedOptions) return template
      return template.replace(/\{\{(\w+)\}\}/g, (_match, token) => {
        const value = normalizedOptions[token]
        return value == null ? "" : String(value)
      })
    }
  })
}))

vi.mock("@tanstack/react-query", () => ({
  useQuery: () => ({ data: [] }),
  useQueryClient: () => ({ invalidateQueries: vi.fn() }),
  useMutation: ({
    mutationFn,
    onSuccess,
    onError
  }: {
    mutationFn: (args: any) => Promise<any>
    onSuccess?: () => void
    onError?: (error: unknown) => void
  }) => ({
    mutateAsync: async (args: any) => {
      try {
        const result = await mutationFn(args)
        onSuccess?.()
        return result
      } catch (error) {
        onError?.(error)
        throw error
      }
    }
  })
}))

vi.mock("antd", () => {
  const React = require("react") as typeof import("react")

  const InputComponent = React.forwardRef<HTMLInputElement, any>(
    (
      {
        value,
        onChange,
        onKeyDown,
        placeholder,
        disabled,
        className,
        "aria-label": ariaLabel,
        "data-testid": dataTestId
      },
      ref
    ) => (
      <input
        ref={ref}
        value={value ?? ""}
        onChange={(event) => onChange?.(event)}
        onKeyDown={onKeyDown}
        placeholder={placeholder}
        disabled={disabled}
        className={className}
        aria-label={ariaLabel ?? placeholder}
        data-testid={dataTestId}
      />
    )
  )
  ;(InputComponent as any).TextArea = ({
    value,
    onChange,
    placeholder,
    readOnly,
    disabled,
    className,
    "data-testid": dataTestId
  }: any) => (
    <textarea
      value={value ?? ""}
      onChange={(event) => onChange?.(event)}
      placeholder={placeholder}
      readOnly={readOnly}
      disabled={disabled}
      className={className}
      data-testid={dataTestId}
    />
  )

  const Dropdown = ({
    open,
    onOpenChange,
    popupRender,
    children
  }: any) => {
    const containerRef = React.useRef<HTMLDivElement | null>(null)

    React.useEffect(() => {
      if (!open) return
      const onMouseDown = (event: MouseEvent) => {
        if (!containerRef.current?.contains(event.target as Node)) {
          onOpenChange?.(false)
        }
      }
      const onKeyDown = (event: KeyboardEvent) => {
        if (event.key === "Escape") {
          onOpenChange?.(false)
        }
      }
      document.addEventListener("mousedown", onMouseDown)
      document.addEventListener("keydown", onKeyDown)
      return () => {
        document.removeEventListener("mousedown", onMouseDown)
        document.removeEventListener("keydown", onKeyDown)
      }
    }, [open, onOpenChange])

    return (
      <div ref={containerRef}>
        <div onClick={() => onOpenChange?.(!open)}>{children}</div>
        {open ? popupRender?.(null) : null}
      </div>
    )
  }

  const ButtonComponent = ({
    children,
    onClick,
    disabled,
    loading,
    htmlType,
    className,
    "data-testid": dataTestId,
    title,
    "aria-label": ariaLabel
  }: any) => (
    <button
      type={htmlType === "submit" ? "submit" : "button"}
      onClick={onClick}
      disabled={disabled || loading}
      className={className}
      data-testid={dataTestId}
      title={title}
      aria-label={ariaLabel}
    >
      {children}
    </button>
  )

  const ModalComponent = ({
    open,
    title,
    children,
    footer
  }: {
    open?: boolean
    title?: React.ReactNode
    children?: React.ReactNode
    footer?: React.ReactNode
  }) =>
    open ? (
      <div role="dialog" aria-label={typeof title === "string" ? title : "modal"}>
        {title ? <h2>{title}</h2> : null}
        <div>{children}</div>
        {footer ? <div>{footer}</div> : null}
      </div>
    ) : null
  ModalComponent.confirm = vi.fn()

  return {
    Button: ButtonComponent,
    Checkbox: ({
      children,
      checked,
      onChange,
      disabled
    }: {
      children: React.ReactNode
      checked?: boolean
      onChange?: (event: { target: { checked: boolean } }) => void
      disabled?: boolean
    }) => (
      <label>
        <input
          type="checkbox"
          checked={Boolean(checked)}
          onChange={(event) =>
            onChange?.({ target: { checked: event.target.checked } })
          }
          disabled={disabled}
        />
        {children}
      </label>
    ),
    Dropdown,
    Input: InputComponent,
    InputNumber: ({ value, onChange, disabled, ...rest }: any) => (
      <input
        type="number"
        value={value ?? ""}
        onChange={(event) => {
          const next = event.target.value
          onChange?.(next === "" ? undefined : Number(next))
        }}
        disabled={disabled}
        {...rest}
      />
    ),
    Modal: ModalComponent,
    Popover: ({ children }: { children: React.ReactNode }) => <>{children}</>,
    Radio: {
      Group: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
      Button: ({ children }: { children: React.ReactNode }) => (
        <button type="button">{children}</button>
      )
    },
    Select: ({ value, options = [], onChange, disabled, ...rest }: any) => (
      <select
        value={value ?? ""}
        onChange={(event) => onChange?.(event.target.value)}
        disabled={disabled}
        {...rest}
      >
        {Array.isArray(options)
          ? options.map((option: any) => (
              <option
                key={String(option?.value)}
                value={String(option?.value || "")}
              >
                {typeof option?.label === "string"
                  ? option.label
                  : String(option?.value || "")}
              </option>
            ))
          : null}
      </select>
    ),
    Space: {
      Compact: ({ children }: { children: React.ReactNode }) => <div>{children}</div>
    },
    Switch: ({ checked, onChange, disabled }: any) => (
      <input
        type="checkbox"
        checked={Boolean(checked)}
        onChange={(event) => onChange?.(event.target.checked)}
        disabled={disabled}
      />
    ),
    Tooltip: ({ children }: { children: React.ReactNode }) => <>{children}</>
  }
})

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (_key: unknown, defaultValue: unknown) => React.useState(defaultValue)
}))

vi.mock("react-router-dom", () => ({
  Link: ({ children, to, ...rest }: any) => (
    <a href={typeof to === "string" ? to : "#"} {...rest}>
      {children}
    </a>
  ),
  useNavigate: () => vi.fn()
}))

vi.mock("@/context/demo-mode", () => ({
  useDemoMode: () => ({ demoEnabled: false })
}))

vi.mock("@/store/tutorials", () => ({
  useHelpModal: () => ({ open: vi.fn() })
}))

vi.mock("@/components/ui/feedback", () => ({
  EmptyState: ({
    title,
    description,
    primaryAction,
    secondaryAction,
    children,
    ...rest
  }: any) => (
    <section {...rest}>
      <h1>{title}</h1>
      <div>{description}</div>
      {primaryAction ? (
        <button type="button" onClick={primaryAction.onClick}>
          {primaryAction.label}
        </button>
      ) : null}
      {secondaryAction ? (
        <button type="button" onClick={secondaryAction.onClick}>
          {secondaryAction.label}
        </button>
      ) : null}
      {children}
    </section>
  )
}))

vi.mock("~/hooks/useDynamicTextareaSize", () => ({
  default: vi.fn()
}))

vi.mock("~/hooks/useMessageOption", () => ({
  useMessageOption: () => playgroundFormMessageOptionState.value
}))

vi.mock("@/store/option", () => ({
  useStoreMessageOption: (selector: (state: any) => unknown) =>
    selector({
      setRagMediaIds: vi.fn(),
      setRagPinnedResults: vi.fn()
    })
}))

vi.mock("@/store/model", () => ({
  useStoreChatModelSettings: (selector?: (state: any) => unknown) => {
    const state = {
      systemPrompt: "",
      setSystemPrompt: vi.fn(),
      temperature: 0.7,
      numPredict: 512,
      topP: 0.9,
      topK: 40,
      frequencyPenalty: 0,
      presencePenalty: 0,
      repeatPenalty: 1,
      reasoningEffort: "medium",
      historyMessageLimit: 20,
      historyMessageOrder: "recent_first",
      slashCommandInjectionMode: "append",
      apiProvider: "custom",
      extraHeaders: "",
      extraBody: "",
      jsonMode: false,
      numCtx: 8192,
      updateSetting: vi.fn(),
      updateSettings: vi.fn(),
      setActiveSettingsScope: vi.fn(),
      updateScopedSetting: vi.fn(),
      getEffectiveSettings: vi.fn(() => ({}))
    }
    return selector ? selector(state) : state
  }
}))

vi.mock("~/store/webui", () => ({
  useWebUI: () => ({
    sendWhenEnter: true,
    setSendWhenEnter: vi.fn(),
    ttsEnabled: false
  })
}))

vi.mock("@/hooks/useConnectionState", () => ({
  useConnectionState: () => playgroundFormConnectionState,
  useIsConnected: () => true
}))

vi.mock("@/types/connection", () => ({
  ConnectionPhase: { CONNECTED: "connected" },
  deriveConnectionUxState: () => ({ label: "Connected", tone: "ok" })
}))

vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => ({
    loading: false,
    capabilities: { hasAudio: false, hasWebSearch: false }
  })
}))

vi.mock("@/hooks/useTldwAudioStatus", () => ({
  useTldwAudioStatus: () => ({ healthState: "ready" })
}))

vi.mock("@/hooks/useMcpTools", () => ({
  useMcpTools: () => ({
    hasMcp: false,
    healthState: "ready",
    discoveredTools: [],
    chatTools: [],
    toolCounts: { enabled: 0, total: 0 },
    tools: [],
    toolsLoading: false,
    catalogs: [],
    catalogsLoading: false,
    toolCatalog: "none",
    toolCatalogId: null,
    toolModules: [],
    moduleOptions: [],
    moduleOptionsLoading: false,
    toolCatalogStrict: false,
    setToolCatalog: vi.fn(),
    setToolCatalogId: vi.fn(),
    setToolModules: vi.fn(),
    setToolCatalogStrict: vi.fn(),
    setToolEnabled: vi.fn(),
    resetToolFilter: vi.fn()
  })
}))

vi.mock("@/hooks/useSpeechRecognition", () => ({
  useSpeechRecognition: () => ({
    transcript: speechRecognitionState.transcript,
    isListening: speechRecognitionState.isListening,
    resetTranscript: speechRecognitionState.resetTranscript,
    start: speechRecognitionState.start,
    stop: speechRecognitionState.stop,
    supported: speechRecognitionState.supported
  })
}))

vi.mock("@/hooks/useServerDictation", () => ({
  useServerDictation: (options: Record<string, any>) => {
    serverDictationState.lastOptions = options
    return {
      isServerDictating: serverDictationState.isServerDictating,
      startServerDictation: serverDictationState.startServerDictation,
      stopServerDictation: serverDictationState.stopServerDictation
    }
  }
}))

vi.mock("@/hooks/useDictationStrategy", () => ({
  useDictationStrategy: () => dictationStrategyState.value
}))

vi.mock("~/hooks/useTabMentions", () => ({
  useTabMentions: () => ({
    tabMentionsEnabled: false,
    showMentions: false,
    mentionPosition: null,
    filteredTabs: [],
    availableTabs: [],
    selectedDocuments: [],
    handleTextChange: vi.fn(),
    insertMention: vi.fn(),
    closeMentions: vi.fn(),
    addDocument: vi.fn(),
    removeDocument: vi.fn(),
    clearSelectedDocuments: vi.fn(),
    reloadTabs: vi.fn(),
    handleMentionsOpen: vi.fn()
  })
}))

vi.mock("~/hooks/keyboard", () => ({
  useFocusShortcuts: vi.fn()
}))

vi.mock("@/hooks/useKeyboardShortcuts", () => ({
  isMac: false
}))

vi.mock("@/hooks/useDraftPersistence", () => ({
  useDraftPersistence: () => ({ draftSaved: false })
}))

vi.mock("@/hooks/useSelectedCharacter", () => ({
  useSelectedCharacter: () => [null, vi.fn()]
}))

vi.mock("@/hooks/useSelectedAssistant", () => ({
  useSelectedAssistant: (initialValue: any = null) => {
    const [selectedAssistant, setSelectedAssistant] = React.useState(initialValue)
    const setSelectedAssistantWithBroadcast = async (next: any) => {
      selectedAssistantMock.setSelectedAssistant(next)
      setSelectedAssistant(next)
    }
    return [
      selectedAssistant,
      setSelectedAssistantWithBroadcast,
      { isLoading: false, setRenderValue: setSelectedAssistant }
    ] as const
  }
}))

vi.mock("@/hooks/useVoiceChatSettings", () => ({
  useVoiceChatSettings: () => ({
    voiceChatEnabled: false,
    setVoiceChatEnabled: vi.fn(),
    voiceChatModel: "chat",
    setVoiceChatModel: vi.fn(),
    voiceChatPauseMs: 800,
    setVoiceChatPauseMs: vi.fn(),
    voiceChatTriggerPhrases: [],
    setVoiceChatTriggerPhrases: vi.fn(),
    voiceChatAutoResume: false,
    setVoiceChatAutoResume: vi.fn(),
    voiceChatBargeIn: false,
    setVoiceChatBargeIn: vi.fn(),
    voiceChatTtsMode: "stream",
    setVoiceChatTtsMode: vi.fn()
  })
}))

vi.mock("@/hooks/useVoiceChatStream", () => ({
  useVoiceChatStream: () => ({ state: "idle" })
}))

vi.mock("@/hooks/useVoiceChatMessages", () => ({
  useVoiceChatMessages: () => ({
    beginTurn: vi.fn(),
    appendAssistantDelta: vi.fn(),
    finalizeAssistant: vi.fn(async () => undefined),
    abandonTurn: vi.fn()
  })
}))

vi.mock("../MentionsDropdown", () => ({
  MentionsDropdown: () => null
}))

vi.mock("../ComposerTextarea", () => ({
  ComposerTextarea: ({
    value,
    onChange,
    placeholder
  }: {
    value: string
    onChange: (event: React.ChangeEvent<HTMLTextAreaElement>) => void
    placeholder?: string
  }) => (
    <textarea
      value={value}
      onChange={onChange}
      placeholder={placeholder}
      data-testid="composer-textarea"
    />
  )
}))

vi.mock("../ComposerToolbar", () => ({
  ComposerToolbar: ({
    researchLaunchButton,
    toolsButton,
    sendControl
  }: {
    researchLaunchButton?: React.ReactNode
    toolsButton?: React.ReactNode
    sendControl?: React.ReactNode
  }) => (
    <div data-testid="composer-toolbar">
      {researchLaunchButton}
      {toolsButton}
      {sendControl}
    </div>
  )
}))

vi.mock("../CompareToggle", () => ({
  CompareToggle: () => null
}))

vi.mock("../ParameterPresets", () => ({
  detectCurrentPreset: () => "balanced",
  getPresetByKey: () => ({ key: "balanced", settings: {} })
}))

vi.mock("../useMobileComposerViewport", () => ({
  useMobileComposerViewport: () => ({
    keyboardOpen: false,
    keyboardInsetPx: 0
  })
}))

vi.mock("@/components/Common/Settings/CurrentChatModelSettings", () => ({
  CurrentChatModelSettings: () => null
}))

vi.mock("@/components/Common/Settings/ActorPopout", () => ({
  ActorPopout: () => null
}))

vi.mock("@/services/tldw-server", () => ({
  defaultEmbeddingModelForRag: vi.fn(async () => "embedding"),
  fetchChatModels: vi.fn(async () => []),
  fetchImageModels: vi.fn(async () => [])
}))

vi.mock("@/services/search", () => ({
  getIsSimpleInternetSearch: vi.fn(async () => true)
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: vi.fn(async () => undefined),
    createChatCompletion: createChatCompletionMock,
    listAllCharacters: vi.fn(async () => [
      { id: "default-assistant", name: "Helpful AI Assistant" }
    ]),
    listPersonaProfiles: vi.fn(async () => []),
    updateChat: vi.fn(async () => ({}))
  }
}))

vi.mock("@/components/Common/CharacterSelect", () => ({
  CharacterSelect: () => null
}))

vi.mock("@/components/Common/ProviderIcon", () => ({
  ProviderIcons: () => null
}))

vi.mock("@/components/Knowledge", () => ({
  KnowledgePanel: () => null
}))

vi.mock("@/components/Common/Beta", () => ({
  BetaTag: () => null
}))

vi.mock("@/components/Common/Playground/DocumentGeneratorDrawer", () => ({
  DocumentGeneratorDrawer: () => null
}))

vi.mock("@/store/ui-mode", () => ({
  useUiModeStore: (selector: (state: { mode: string }) => unknown) =>
    selector({ mode: "pro" })
}))

vi.mock("../TokenProgressBar", () => ({
  TokenProgressBar: () => null
}))

vi.mock("../AttachmentsSummary", () => ({
  AttachmentsSummary: () => null
}))

vi.mock("../VoiceChatIndicator", () => ({
  VoiceChatIndicator: () => null
}))

vi.mock("@/hooks/useMediaQuery", () => ({
  useMobile: () => false
}))

vi.mock("@/components/Common/Button", () => ({
  Button: ({
    children,
    onClick,
    ariaLabel,
    title,
    disabled,
    className,
    "data-testid": dataTestId
  }: any) => (
    <button
      type="button"
      onClick={onClick}
      aria-label={ariaLabel}
      title={title}
      disabled={Boolean(disabled)}
      className={className}
      data-testid={dataTestId}
    >
      {children}
    </button>
  )
}))

vi.mock("@/hooks/useSimpleForm", () => ({
  useSimpleForm: ({ initialValues }: { initialValues: Record<string, string> }) => {
    const [values, setValues] = React.useState(initialValues)
    const [errors, setErrors] = React.useState<Record<string, string>>({})
    const setFieldValue = (field: string, value: string) =>
      setValues((prev) => ({ ...prev, [field]: value }))
    const reset = () => setValues(initialValues)
    return {
      values,
      errors,
      setFieldValue,
      setFieldError: (field: string, value: string) =>
        setErrors((prev) => ({ ...prev, [field]: value })),
      clearFieldError: (field: string) =>
        setErrors((prev) => {
          const next = { ...prev }
          delete next[field]
          return next
        }),
      reset,
      onSubmit:
        (handler: (values: Record<string, string>) => void | Promise<void>) =>
        async (event?: React.FormEvent) => {
          event?.preventDefault?.()
          await handler(values)
        },
      getInputProps: (field: string) => ({
        value: values[field] ?? "",
        onChange: (event: React.ChangeEvent<HTMLTextAreaElement | HTMLInputElement>) =>
          setFieldValue(field, event.target.value)
      })
    }
  }
}))

vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => ({
    success: vi.fn(),
    error: vi.fn(),
    warning: vi.fn(),
    info: vi.fn()
  })
}))

vi.mock("@/utils/onboarding-ingestion-telemetry", () => ({
  trackOnboardingChatSubmitSuccess: vi.fn(async () => undefined)
}))

vi.mock("@/hooks/playground", () => ({
  useModelSelector: () => ({
    modelDropdownOpen: false,
    setModelDropdownOpen: vi.fn(),
    modelSearchQuery: "",
    setModelSearchQuery: vi.fn(),
    modelSortMode: "favorites",
    setModelSortMode: vi.fn(),
    selectedModelMeta: null,
    modelContextLength: 8192,
    modelCapabilities: ["streaming"],
    resolvedMaxContext: 8192,
    resolvedProviderKey: "custom",
    providerLabel: "Custom",
    modelSummaryLabel: "deepseek-chat",
    apiModelLabel: "deepseek-chat",
    modelSelectorWarning: null,
    favoriteModels: [],
    favoriteModelsIsLoading: false,
    favoriteModelSet: new Set(),
    toggleFavoriteModel: vi.fn(),
    filteredModels: [],
    modelDropdownMenuItems: [],
    isSmallModel: false
  }),
  useComposerTokens: () => ({
    draftTokenCount: 0,
    conversationTokenCount: 0,
    tokenUsageLabel: "0 tokens",
    tokenUsageCompactLabel: "~0 tokens",
    tokenUsageTooltip: "0 tokens",
    estimateTokensForText: (value: string) => Math.ceil((value || "").length / 4)
  }),
  useImageBackend: () => ({
    imageBackendDefault: "mock-backend",
    setImageBackendDefault: vi.fn(),
    imageBackendOptions: [{ value: "mock-backend", label: "Mock Backend", provider: "custom" }],
    imageBackendLabel: "Mock Backend",
    imageBackendActiveKey: "mock-backend",
    imageBackendMenuItems: [],
    imageBackendBadgeLabel: "Mock Backend"
  }),
  useActionBarVisibility: () => ({
    actionBarVisible: true,
    actionBarVisibilityClass: "",
    handlers: {
      onMouseEnter: vi.fn(),
      onMouseLeave: vi.fn(),
      onFocusCapture: vi.fn(),
      onBlurCapture: vi.fn()
    }
  }),
  usePersistenceMode: () => ({
    persistenceTooltip: "Saved",
    focusConnectionCard: vi.fn(),
    getPersistenceModeLabel: () => "Saved"
  }),
  useSlashCommands: () => ({
    showSlashMenu: false,
    slashActiveIndex: 0,
    setSlashActiveIndex: vi.fn(),
    filteredSlashCommands: [],
    resolveSubmissionIntent: (message: string) => ({
      message,
      handled: false,
      invalidImageCommand: false,
      imageCommandMissingProvider: false,
      isImageCommand: false,
      imageBackendOverride: undefined
    }),
    activeImageCommand: null,
    handleSlashCommandSelect: vi.fn()
  }),
  useMessageCollapse: () => ({
    isMessageCollapsed: false,
    setIsMessageCollapsed: vi.fn(),
    collapsedRange: null,
    setCollapsedRange: vi.fn(),
    hasExpandedLargeText: false,
    setHasExpandedLargeText: vi.fn(),
    pendingCaretRef: { current: null },
    lastDisplaySelectionRef: { current: null },
    pendingCollapsedStateRef: { current: null },
    pointerDownRef: { current: false },
    selectionFromPointerRef: { current: null },
    normalizeCollapsedRange: vi.fn(() => null),
    parseCollapsedRange: vi.fn(() => null),
    buildCollapsedMessageLabel: vi.fn(() => ""),
    getCollapsedDisplayMeta: vi.fn((message: string) => ({ display: message })),
    getDisplayCaretFromMessage: vi.fn((value: number) => value),
    getMessageCaretFromDisplay: vi.fn((value: number) => value),
    collapseLargeMessage: vi.fn(),
    expandLargeMessage: vi.fn(),
    restoreMessageValue: vi.fn()
  }),
  useDeferredComposerInput: (value: string) => ({ deferredInput: value }),
  useMcpToolsControl: () => ({
    mcpSettingsOpen: false,
    setMcpSettingsOpen: vi.fn(),
    mcpPopoverOpen: false,
    setMcpPopoverOpen: vi.fn(),
    mcpSummaryLabel: "MCP none",
    mcpAriaLabel: "MCP",
    mcpChoiceLabel: "None",
    mcpDisabledReason: "",
    mcpStatusLabel: "Ready",
    handleCatalogSelect: vi.fn(),
    catalogGroups: { team: [], org: [], global: [] },
    catalogDraft: "",
    setCatalogDraft: vi.fn(),
    commitCatalog: vi.fn()
  })
}))

beforeEach(() => {
  onSubmitMock.mockClear()
  createChatCompletionMock.mockClear()
  selectedAssistantMock.setSelectedAssistant.mockClear()
  playgroundFormMessageOptionState.value = createMessageOptionState()
  playgroundFormConnectionState.phase = "connected"
  playgroundFormConnectionState.isConnected = true
  speechRecognitionState.transcript = ""
  speechRecognitionState.isListening = false
  speechRecognitionState.supported = false
  speechRecognitionState.start.mockClear()
  speechRecognitionState.stop.mockClear()
  speechRecognitionState.resetTranscript.mockClear()
  serverDictationState.isServerDictating = false
  serverDictationState.startServerDictation.mockClear()
  serverDictationState.stopServerDictation.mockClear()
  serverDictationState.lastOptions = null
})

const renderRolePlayStarterHarness = () =>
  render(
    <>
      <PlaygroundEmpty />
      <PlaygroundForm droppedFiles={[]} />
      <AssistantSelect variant="dropdown" />
    </>
  )

describe("PlaygroundForm role-play starter", () => {
  it("does not crash role-play state derivation when document context is null", () => {
    playgroundFormMessageOptionState.value = {
      ...createMessageOptionState(),
      documentContext: null
    }

    expect(() => render(<PlaygroundForm droppedFiles={[]} />)).not.toThrow()
    expect(document.querySelector("#composer-options-panel")).toBeInTheDocument()
  })

  it("opens character selection from the starter and returns focus after selecting the default assistant", async () => {
    const user = userEvent.setup()
    renderRolePlayStarterHarness()

    await user.click(screen.getByRole("button", { name: /chat as a character/i }))

    expect(
      await screen.findByRole("button", { name: /select character or persona/i })
    ).toBeInTheDocument()
    expect(screen.getByRole("tab", { name: "Characters" })).toHaveAttribute(
      "aria-selected",
      "true"
    )

    const assistantMenu = await screen.findByTestId("assistant-select-menu")
    await user.click(
      within(assistantMenu).getByRole("button", {
        name: /^(default assistant|helpful ai assistant)$/i
      })
    )

    await waitFor(() => {
      expect(screen.queryByTestId("assistant-select-menu")).toBeNull()
    })
    expect(screen.queryByText(/something went wrong/i)).not.toBeInTheDocument()
    expect(selectedAssistantMock.setSelectedAssistant).toHaveBeenCalledWith(
      expect.objectContaining({
        kind: "character",
        id: "default-assistant",
        name: "Helpful AI Assistant"
      })
    )
    await waitFor(() => {
      expect(document.activeElement).toBe(
        screen.getByRole("button", { name: /helpful ai assistant/i })
      )
    })
  })
})
