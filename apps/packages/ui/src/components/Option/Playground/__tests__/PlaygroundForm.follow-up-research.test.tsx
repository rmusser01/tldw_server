// @vitest-environment jsdom
import React from "react"
import { render, screen, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { PlaygroundForm } from "../PlaygroundForm"

const onSubmitMock = vi.hoisted(() => vi.fn(async (_payload: unknown) => null))
const createChatCompletionMock = vi.hoisted(() =>
  vi.fn(async () => ({
    json: async () => ({ choices: [] })
  }))
)
const createResearchRunMock = vi.hoisted(() =>
  vi.fn(async () => ({
    id: "rs_follow_up",
    status: "running",
    phase: "planning",
    control_state: "running"
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

const buildAttachedResearchContext = (overrides: Record<string, unknown> = {}) => ({
  attached_at: "2026-03-08T20:00:00Z",
  run_id: "run_123",
  query: "Battery recycling supply chain",
  question: "Battery recycling supply chain",
  outline: [{ title: "Overview" }],
  key_claims: [{ text: "Claim one" }],
  unresolved_questions: ["What changed in Europe?"],
  verification_summary: { unsupported_claim_count: 0 },
  source_trust_summary: { high_trust_count: 2 },
  research_url: "/research?run=run_123",
  ...overrides
})

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, fallback?: string, options?: Record<string, unknown>) => {
      const template = fallback || key
      if (!options) return template
      return template.replace(/\{\{(\w+)\}\}/g, (_match, token) => {
        const value = options[token]
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
  const InputComponent = ({
    value,
    onChange,
    placeholder,
    disabled,
    className,
    "data-testid": dataTestId
  }: any) => (
    <input
      value={value ?? ""}
      onChange={(event) => onChange?.(event)}
      placeholder={placeholder}
      disabled={disabled}
      className={className}
      data-testid={dataTestId}
    />
  )
  InputComponent.TextArea = ({
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

  const SelectComponent = ({
    value,
    options = [],
    onChange,
    disabled,
    ...rest
  }: any) => (
    <select
      value={value ?? ""}
      onChange={(event) => onChange?.(event.target.value)}
      disabled={disabled}
      {...rest}
    >
      {Array.isArray(options)
        ? options.map((option: any) => (
            <option key={String(option?.value)} value={String(option?.value || "")}>
              {typeof option?.label === "string"
                ? option.label
                : String(option?.value || "")}
            </option>
          ))
        : null}
    </select>
  )

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
    Dropdown: ({ children }: { children: React.ReactNode }) => <>{children}</>,
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
    Radio: {
      Group: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
      Button: ({ children }: { children: React.ReactNode }) => <button type="button">{children}</button>
    },
    Select: SelectComponent,
    Switch: ({ checked, onChange, disabled }: any) => (
      <input
        type="checkbox"
        checked={Boolean(checked)}
        onChange={(event) => onChange?.(event.target.checked)}
        disabled={disabled}
      />
    ),
    Tooltip: ({ children }: { children: React.ReactNode }) => <>{children}</>,
    Popover: ({ children }: { children: React.ReactNode }) => <>{children}</>,
    Modal: ModalComponent,
    Button: ButtonComponent,
    Space: {
      Compact: ({ children }: { children: React.ReactNode }) => <div>{children}</div>
    }
  }
})

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (_key: string, defaultValue: unknown) =>
    React.useState(defaultValue)
}))

vi.mock("react-router-dom", () => ({
  Link: ({ children, to, ...rest }: any) => (
    <a href={typeof to === "string" ? to : "#"} {...rest}>
      {children}
    </a>
  ),
  useNavigate: () => vi.fn()
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
  useStoreChatModelSettings: (selector: (state: any) => unknown) =>
    selector({
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
      updateSettings: vi.fn()
    })
}))

vi.mock("~/store/webui", () => ({
  useWebUI: () => ({
    sendWhenEnter: true,
    setSendWhenEnter: vi.fn(),
    ttsEnabled: false
  })
}))

vi.mock("@/hooks/useConnectionState", () => ({
  useConnectionState: () => playgroundFormConnectionState
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
    setToolCatalogStrict: vi.fn()
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

vi.mock("../ContextFootprintPanel", () => ({
  ContextFootprintPanel: () => null
}))

vi.mock("../CompareToggle", () => ({
  CompareToggle: () => null
}))

vi.mock("../ParameterPresets", () => ({
  detectCurrentPreset: () => ({ key: "balanced", label: "Balanced" }),
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
    createResearchRun: createResearchRunMock,
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

vi.mock("../VoiceModeSelector", () => ({
  VoiceModeSelector: () => null
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

vi.mock("@/utils/resolve-api-provider", () => ({
  resolveApiProviderForModel: vi.fn(async () => "custom")
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
  createResearchRunMock.mockClear()
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
  dictationStrategyState.value = {
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
  }
})

describe("PlaygroundForm follow-up research", () => {
  beforeEach(() => {
    playgroundFormMessageOptionState.value = createMessageOptionState()
    playgroundFormMessageOptionState.value.serverChatId = "chat_123"
    playgroundFormMessageOptionState.value.temporaryChat = false
  })

  it("hides follow-up research for empty drafts", () => {
    render(
      <PlaygroundForm
        droppedFiles={[]}
        attachedResearchContext={buildAttachedResearchContext()}
      />
    )

    expect(
      screen.queryByRole("button", { name: "Follow-up Research" })
    ).not.toBeInTheDocument()
  })

  it("hides follow-up research for temporary chats", () => {
    playgroundFormMessageOptionState.value.temporaryChat = true

    render(
      <PlaygroundForm
        droppedFiles={[]}
        attachedResearchContext={buildAttachedResearchContext()}
      />
    )

    expect(
      screen.queryByRole("button", { name: "Follow-up Research" })
    ).not.toBeInTheDocument()
  })

  it("opens the follow-up confirmation surface for a saved chat with attached background", async () => {
    const user = userEvent.setup()
    render(
      <PlaygroundForm
        droppedFiles={[]}
        attachedResearchContext={buildAttachedResearchContext()}
      />
    )

    await user.type(screen.getByTestId("composer-textarea"), "Investigate battery recycling")

    const followUpButton = screen.getByRole("button", {
      name: "Follow-up Research"
    })

    expect(followUpButton).toBeEnabled()

    await user.click(followUpButton)

    expect(
      screen.getByRole("checkbox", {
        name: "Use attached research as background"
      })
    ).toBeChecked()
    expect(
      screen.getByRole("button", { name: "Start research" })
    ).toBeEnabled()
  })

  it("hides follow-up research when no research context is attached", async () => {
    const user = userEvent.setup()
    render(<PlaygroundForm droppedFiles={[]} attachedResearchContext={null} />)

    await user.type(
      screen.getByTestId("composer-textarea"),
      "Investigate battery recycling"
    )

    expect(
      screen.queryByRole("button", { name: "Follow-up Research" })
    ).not.toBeInTheDocument()
  })

  it("starts follow-up research once with bounded attached background", async () => {
    const user = userEvent.setup()
    let resolveLaunch: ((value: unknown) => void) | null = null
    createResearchRunMock.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          resolveLaunch = resolve
        })
    )

    render(
      <PlaygroundForm
        droppedFiles={[]}
        attachedResearchContext={buildAttachedResearchContext()}
      />
    )

    await user.type(
      screen.getByTestId("composer-textarea"),
      "Investigate battery recycling"
    )
    await user.click(
      screen.getByRole("button", { name: "Follow-up Research" })
    )

    const startResearchButton = screen.getByRole("button", {
      name: "Start research"
    })
    await user.click(startResearchButton)
    await user.click(startResearchButton)

    expect(createResearchRunMock).toHaveBeenCalledTimes(1)
    expect(createResearchRunMock).toHaveBeenCalledWith(
      expect.objectContaining({
        query: "Investigate battery recycling",
        chat_handoff: expect.objectContaining({
          chat_id: "chat_123"
        }),
        follow_up: expect.objectContaining({
          question: "Investigate battery recycling",
          background: expect.objectContaining({
            question: "Battery recycling supply chain"
          })
        })
      })
    )

    resolveLaunch?.({
      id: "rs_follow_up",
      status: "running",
      phase: "planning",
      control_state: "running"
    })
  })
})
