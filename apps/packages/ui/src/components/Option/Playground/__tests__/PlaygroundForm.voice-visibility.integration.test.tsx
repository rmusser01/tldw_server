// @vitest-environment jsdom
import React from "react"
import { render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

const createMessageOptionState = () => ({
  onSubmit: vi.fn(async () => null),
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

const messageOptionOverrides = vi.hoisted(() => ({
  value: {} as Record<string, unknown>
}))

let capturedModeLauncherProps: { voiceChatUnavailableReason?: string | null } | null =
  null

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, fallback?: string, options?: Record<string, unknown>) => {
      const translations: Record<string, string> = {
        "playground:voiceChat.unavailableReasons.transportMissing":
          "This server does not advertise voice conversation streaming."
      }
      const template = translations[key] || fallback || key
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
  )

  return {
    Button: ({
      children,
      onClick,
      disabled,
      loading,
      htmlType,
      className,
      title,
      "aria-label": ariaLabel
    }: any) => (
      <button
        type={htmlType || "button"}
        onClick={onClick}
        disabled={disabled || loading}
        className={className}
        title={title}
        aria-label={ariaLabel}
      >
        {children}
      </button>
    ),
    Checkbox: ({ children, checked, onChange, disabled }: any) => (
      <label>
        <input
          type="checkbox"
          checked={Boolean(checked)}
          onChange={(event) => onChange?.(event.target.checked)}
          disabled={disabled}
        />
        {children}
      </label>
    ),
    Dropdown: ({ children }: any) => <>{children}</>,
    Input: InputComponent,
    InputNumber: ({ value, onChange, disabled }: any) => (
      <input
        type="number"
        value={value ?? ""}
        onChange={(event) => onChange?.(Number(event.target.value))}
        disabled={disabled}
      />
    ),
    Radio: {
      Group: ({ children, value, onChange }: any) => (
        <div data-value={value} onChange={onChange}>
          {children}
        </div>
      ),
      Button: ({ children }: any) => <button type="button">{children}</button>
    },
    Select: SelectComponent,
    Switch: ({ checked, onChange }: any) => (
      <button type="button" aria-pressed={checked} onClick={() => onChange?.(!checked)}>
        switch
      </button>
    ),
    Tooltip: ({ children }: any) => <>{children}</>,
    Modal: Object.assign(
      ({
        open,
        children
      }: {
        open?: boolean
        children: React.ReactNode
      }) => (open ? <div data-testid="modal">{children}</div> : null),
      {
        confirm: vi.fn()
      }
    )
  }
})

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (key: string, defaultValue: unknown) => React.useState(defaultValue)
}))

vi.mock("~/hooks/useMessageOption", () => ({
  useMessageOption: () => ({
    ...createMessageOptionState(),
    ...messageOptionOverrides.value
  })
}))

vi.mock("@/hooks/useChatSettingsRecord", () => ({
  useChatSettingsRecord: () => ({
    settings: null,
    updateSettings: vi.fn()
  })
}))

vi.mock("@/hooks/useSelectedCharacter", () => ({
  useSelectedCharacter: () => [null, vi.fn()]
}))

vi.mock("@/hooks/useVoiceChatSettings", () => ({
  useVoiceChatSettings: () => ({
    voiceChatEnabled: true,
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
    finalizeAssistant: vi.fn(),
    failTurn: vi.fn(),
    abandonTurn: vi.fn()
  })
}))

vi.mock("@/hooks/useAudioSourceCatalog", () => ({
  useAudioSourceCatalog: () => ({ devices: [], isSettled: true })
}))

vi.mock("@/hooks/useCanonicalConnectionConfig", () => ({
  useCanonicalConnectionConfig: () => ({
    config: {
      serverUrl: "http://localhost:8000",
      authMode: "single-user",
      apiKey: "test-api-key"
    },
    loading: false
  })
}))

vi.mock("@/hooks/useConnectionState", () => ({
  useConnectionState: () => ({
    phase: "connected",
    isConnected: true
  })
}))

vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => ({
    capabilities: {
      hasVoiceChat: true,
      hasStt: true,
      hasTts: true,
      hasWebSearch: false,
      hasMcp: false,
      hasAudio: true
    },
    loading: false
  })
}))

vi.mock("@/hooks/useTldwAudioStatus", () => ({
  useTldwAudioStatus: () => ({
    healthState: "ready",
    sttHealthState: "ready",
    hasVoiceConversationTransport: false
  })
}))

vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => ({
    success: vi.fn(),
    error: vi.fn(),
    info: vi.fn(),
    warning: vi.fn()
  })
}))

vi.mock("@/hooks/useMcpTools", () => ({
  useMcpTools: () => ({
    hasMcp: false,
    healthState: "unavailable",
    tools: [],
    toolsLoading: false,
    catalogs: [],
    catalogsLoading: false,
    toolCatalog: null,
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

vi.mock("@/hooks/playground", () => ({
  useModelSelector: () => ({
    modelDropdownOpen: false,
    setModelDropdownOpen: vi.fn(),
    modelSearchQuery: "",
    setModelSearchQuery: vi.fn(),
    modelSortMode: "default",
    setModelSortMode: vi.fn(),
    selectedModelMeta: null,
    modelContextLength: 0,
    modelCapabilities: [],
    resolvedMaxContext: 0,
    resolvedProviderKey: "openai",
    providerLabel: "OpenAI",
    modelSummaryLabel: "deepseek-chat",
    apiModelLabel: "deepseek-chat",
    modelSelectorWarning: null,
    favoriteModels: [],
    favoriteModelsIsLoading: false,
    favoriteModelSet: new Set<string>(),
    toggleFavoriteModel: vi.fn(),
    filteredModels: [],
    modelDropdownMenuItems: [],
    isSmallModel: false
  }),
  useComposerTokens: () => ({
    draftTokenCount: 0,
    conversationTokenCount: 0,
    tokenUsageLabel: "0 tokens",
    tokenUsageCompactLabel: "0 tokens",
    tokenUsageTooltip: "0 tokens",
    estimateTokensForText: () => 0
  }),
  useImageBackend: () => ({}),
  useActionBarVisibility: () => ({
    handlers: {
      onMouseEnter: vi.fn(),
      onMouseLeave: vi.fn(),
      onFocusCapture: vi.fn(),
      onBlurCapture: vi.fn()
    }
  }),
  useSlashCommands: () => ({}),
  useMessageCollapse: () => ({}),
  useDeferredComposerInput: () => ({ deferredInput: "" }),
  useMcpToolsControl: () => ({}),
  useModelComparison: () => ({
    compareModeActive: false,
    compareModelMetaById: {},
    availableCompareModels: [],
    compareModelLabelById: {},
    compareSelectedModelLabels: [],
    compareNeedsMoreModels: false,
    compareModelsSupportCapability: true,
    compareCapabilityIncompatibilities: [],
    toggleCompareMode: vi.fn(),
    handleAddCompareModel: vi.fn(),
    handleRemoveCompareModel: vi.fn(),
    sendLabel: "Send"
  })
}))

vi.mock("@/hooks/useMcpToolsControl", () => ({
  useMcpToolsControl: () => ({
    mcpAriaLabel: "MCP",
    mcpSummaryLabel: "MCP",
    mcpChoiceLabel: "MCP",
    mcpDisabledReason: null,
    mcpPopoverOpen: false,
    setMcpPopoverOpen: vi.fn()
  })
}))

vi.mock("~/store/webui", () => ({
  useWebUI: () => ({
    sendWhenEnter: true,
    setSendWhenEnter: vi.fn()
  })
}))

vi.mock("@/store/ui-mode", () => ({
  useUiModeStore: (selector: (state: { mode: string }) => unknown) =>
    selector({ mode: "focus" })
}))

vi.mock("@/store/model", () => ({
  useStoreChatModelSettings: (selector: (state: Record<string, unknown>) => unknown) =>
    selector({
      systemPrompt: "",
      setSystemPrompt: vi.fn(),
      temperature: 1,
      numPredict: 0,
      topP: 1,
      topK: 0,
      frequencyPenalty: 0,
      presencePenalty: 0,
      repeatPenalty: 1,
      reasoningEffort: "medium",
      historyMessageLimit: 20,
      historyMessageOrder: "newest",
      slashCommandInjectionMode: "none",
      apiProvider: "openai",
      extraHeaders: {},
      extraBody: {},
      llamaThinkingBudgetTokens: 0,
      llamaGrammarMode: "off",
      llamaGrammarId: "",
      llamaGrammarInline: "",
      llamaGrammarOverride: "",
      jsonMode: false,
      numCtx: 4096,
      updateSetting: vi.fn(),
      updateSettings: vi.fn()
    })
}))

vi.mock("@/store/option", () => ({
  useStoreMessageOption: (selector: (state: Record<string, unknown>) => unknown) =>
    selector({
      setRagMediaIds: vi.fn(),
      setRagPinnedResults: vi.fn()
    })
}))

vi.mock("@/store/chat-surface-coordinator", () => ({
  shouldEnableOptionalResource: () => false,
  useChatSurfaceCoordinatorStore: (selector: (state: Record<string, unknown>) => unknown) =>
    selector({
      setPanelVisible: vi.fn(),
      markPanelEngaged: vi.fn()
    })
}))

vi.mock("../hooks", () => ({
  useModelComparison: () => ({
    compareModeActive: false,
    compareModelMetaById: {},
    availableCompareModels: [],
    compareModelLabelById: {},
    compareSelectedModelLabels: [],
    compareNeedsMoreModels: false,
    compareModelsSupportCapability: true,
    compareCapabilityIncompatibilities: [],
    toggleCompareMode: vi.fn(),
    handleAddCompareModel: vi.fn(),
    handleRemoveCompareModel: vi.fn(),
    sendLabel: "Send"
  }),
  useContextWindow: () => ({
    contextWindowModalOpen: false,
    setContextWindowModalOpen: vi.fn(),
    contextWindowDraftValue: "",
    setContextWindowDraftValue: vi.fn(),
    sessionInsightsOpen: false,
    setSessionInsightsOpen: vi.fn(),
    sessionUsageSummary: { totalTokens: 0 },
    sessionUsageLabel: "0 tokens",
    sessionInsights: { totals: { totalTokens: 0 } },
    projectedBudget: { utilizationPercent: 0 },
    tokenBudgetRisk: { level: "low" },
    tokenBudgetRiskLabel: "Low",
    showTokenBudgetWarning: false,
    tokenBudgetWarningText: "",
    characterContextTokenEstimate: 0,
    systemPromptTokenEstimate: 0,
    pinnedSourceTokenEstimate: 0,
    historyTokenEstimate: 0,
    summaryCheckpointSuggestion: null,
    modelRecommendations: [],
    visibleModelRecommendations: [],
    dismissModelRecommendation: vi.fn(),
    contextFootprintRows: [],
    nonMessageContextTokenEstimate: 0,
    nonMessageContextPercent: 0,
    showNonMessageContextWarning: false,
    largestContextContributor: null,
    formatContextWindowValue: (value: number) => String(value),
    isContextWindowOverrideActive: false,
    requestedContextWindowOverride: null,
    isContextWindowOverrideClamped: false,
    openContextWindowModal: vi.fn(),
    saveContextWindowSetting: vi.fn(),
    resetContextWindowSetting: vi.fn(),
    openSessionInsightsModal: vi.fn()
  }),
  usePlaygroundVoiceChat: () => ({
    isListening: false,
    browserSupportsSpeechRecognition: false,
    dictationAudioSourcePreference: "auto",
    dictationResolvedSourceKind: "browser",
    setDictationAudioSourcePreference: vi.fn(),
    isServerDictating: false,
    speechAvailable: false,
    speechUsesServer: false,
    voiceChatStatusLabel: "Voice chat",
    speechTooltipText: "Voice chat unavailable",
    handleVoiceChatToggle: vi.fn(),
    handleDictationToggle: vi.fn(),
    stopListening: vi.fn()
  }),
  usePromptTemplates: () => ({
    currentPresetKey: "custom",
    currentPreset: null,
    startupTemplates: [],
    startupTemplateDraftName: "",
    setStartupTemplateDraftName: vi.fn(),
    startupTemplatePreview: null,
    setStartupTemplatePreview: vi.fn(),
    startupTemplateNameFallback: "",
    selectedSystemPromptRecord: null,
    handleSaveStartupTemplate: vi.fn(),
    handleOpenStartupTemplatePreview: vi.fn(),
    handleApplyStartupTemplate: vi.fn(),
    handleDeleteStartupTemplate: vi.fn(),
    handleTemplateSelect: vi.fn(),
    promptSummaryLabel: "Prompt"
  }),
  usePlaygroundAttachments: () => ({
    attachments: [],
    attachmentCount: 0,
    useDroppedFiles: vi.fn()
  }),
  useComposerInput: () => ({
    form: {
      values: { message: "" },
      errors: { message: null },
      getInputProps: () => ({})
    },
    typing: false,
    setMessageValue: vi.fn(),
    restoreMessageValue: vi.fn(),
    messageDisplayValue: "",
    collapsedDisplayMeta: null,
    textareaRef: { current: null },
    textAreaFocus: vi.fn(),
    syncCollapsedCaret: vi.fn(),
    commitCollapsedEdit: vi.fn(),
    replaceCollapsedRange: vi.fn(),
    handleCompositionStart: vi.fn(),
    handleCompositionEnd: vi.fn(),
    handleTextareaMouseDown: vi.fn(),
    handleTextareaMouseUp: vi.fn(),
    handleTextareaChange: vi.fn(),
    handleTextareaSelect: vi.fn(),
    markComposerPerf: vi.fn(),
    measureComposerPerf: vi.fn(),
    onComposerRenderProfile: vi.fn(),
    wrapComposerProfile: (_label: string, node: React.ReactNode) => node,
    draftSaved: false,
    selectedQuickPrompt: null,
    setSelectedQuickPrompt: vi.fn()
  }),
  usePlaygroundImageGen: () => ({}),
  usePlaygroundPersistence: () => ({
    persistenceTooltip: "Persist",
    focusConnectionCard: vi.fn(),
    getPersistenceModeLabel: () => "Saved",
    privateChatLocked: false,
    showServerPersistenceHint: false,
    handleToggleTemporaryChat: vi.fn(),
    handleSaveChatToServer: vi.fn(),
    persistChatMetadata: vi.fn(),
    handleDismissServerPersistenceHint: vi.fn()
  }),
  usePlaygroundRawPreview: () => ({
    rawRequestSnapshot: null,
    rawRequestSnapshotLoading: false,
    rawRequestSnapshotError: null,
    refreshRawRequestSnapshot: vi.fn()
  }),
  usePlaygroundQueueManagement: () => ({
    availableChatModelIds: [],
    isQueuedDispatchBlockedByComposerState: false,
    queuedRequestActions: {
      remove: vi.fn(),
      move: vi.fn(),
      update: vi.fn(),
      clear: vi.fn()
    },
    queueSubmission: vi.fn(),
    cancelCurrentAndRunDisabledReason: null,
    handleRunQueuedRequest: vi.fn(),
    handleRunNextQueuedRequest: vi.fn(),
    validateSelectedChatModelsAvailability: vi.fn()
  }),
  usePlaygroundSettings: () => ({
    startupTemplatesRaw: [],
    setStartupTemplatesRaw: vi.fn(),
    startupTemplatePreviewOpen: false,
    setStartupTemplatePreviewOpen: vi.fn(),
    compareSharedContextLabels: [],
    compareInteroperabilityNotices: [],
    contextConflictWarnings: []
  }),
  usePlaygroundContextItems: () => ({
    contextItems: [],
    contextItemsLoading: false
  }),
  usePlaygroundSubmit: () => ({
    submitForm: vi.fn(),
    submitFormRef: { current: vi.fn() }
  }),
  toText: (value: unknown) => String(value ?? ""),
  estimateTokensFromText: () => 0
}))

vi.mock("@/components/Common/CharacterSelect", () => ({
  CharacterSelect: () => <div data-testid="character-select" />
}))

vi.mock("@/components/Common/ProviderIcon", () => ({
  ProviderIcons: ({ className }: { className?: string }) => (
    <span data-testid="provider-icons" className={className} />
  )
}))

vi.mock("@/components/Common/Beta", () => ({
  BetaTag: () => <div data-testid="beta-tag" />
}))

vi.mock("../ComposerToolbar", () => ({
  ComposerToolbar: ({
    modeLauncherButton,
    voiceChatButton
  }: {
    modeLauncherButton?: React.ReactNode
    voiceChatButton: React.ReactNode
  }) => (
    <div data-testid="composer-toolbar">
      {modeLauncherButton}
      {voiceChatButton}
    </div>
  )
}))

vi.mock("../PlaygroundModeLauncher", () => ({
  PlaygroundModeLauncher: (props: { voiceChatUnavailableReason?: string | null }) => {
    capturedModeLauncherProps = props
    return null
  }
}))

vi.mock("../PlaygroundToolsPopover", () => ({
  PlaygroundToolsPopover: () => null
}))

vi.mock("../PlaygroundMcpControl", () => ({
  PlaygroundMcpControl: () => null
}))

vi.mock("../PlaygroundSendControl", () => ({
  PlaygroundSendControl: () => null,
  PlaygroundAttachmentButton: () => null
}))

vi.mock("../PlaygroundComposerNotices", () => ({
  PlaygroundComposerNotices: () => null
}))

vi.mock("../PlaygroundKnowledgeSection", () => ({
  PlaygroundKnowledgeSection: () => null
}))

vi.mock("../CompareToggle", () => ({
  CompareToggle: () => null
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

vi.mock("@/components/Common/AudioSourcePicker", () => ({
  AudioSourcePicker: () => null
}))

vi.mock("@/components/Common/ChatQueuePanel", () => ({
  ChatQueuePanel: () => null
}))

vi.mock("../MentionsDropdown", () => ({
  MentionsDropdown: () => null
}))

vi.mock("../ComposerTextarea", () => ({
  ComposerTextarea: () => null
}))

vi.mock("../AttachedResearchContextChip", () => ({
  AttachedResearchContextChip: () => null
}))

vi.mock("@/components/Common/Settings/ActorPopout", () => ({
  ActorPopout: () => null
}))

vi.mock("@/components/Common/Settings/CurrentChatModelSettings", () => ({
  CurrentChatModelSettings: () => null
}))

vi.mock("react-router-dom", () => ({
  Link: ({
    children,
    to,
    ...props
  }: {
    children: React.ReactNode
    to?: string
    [key: string]: unknown
  }) => (
    <a href={to} {...props}>
      {children}
    </a>
  ),
  useNavigate: () => vi.fn()
}))

import { PlaygroundForm } from "../PlaygroundForm"

describe("PlaygroundForm voice visibility", () => {
  beforeEach(() => {
    messageOptionOverrides.value = {}
  })

  it("hides the main voice button but forwards the shared unavailable reason when voice transport is missing", () => {
    capturedModeLauncherProps = null
    render(<PlaygroundForm droppedFiles={[]} />)

    expect(screen.queryByTestId("voice-chat-button")).toBeNull()
    expect(capturedModeLauncherProps?.voiceChatUnavailableReason).toBe(
      "This server does not advertise voice conversation streaming."
    )
  })

  it("shows the most recent chat error above the composer with a diagnostics link", () => {
    messageOptionOverrides.value = {
      messages: [
        {
          id: "user-1",
          isBot: false,
          message: "Hello"
        },
        {
          id: "assistant-error-1",
          isBot: true,
          message:
            "__tldw_error__:" +
            JSON.stringify({
              summary: "The selected model is not available.",
              hint: "Choose a different model or refresh the model list.",
              detail: "model_not_found"
            })
        }
      ]
    }

    render(<PlaygroundForm droppedFiles={[]} />)

    const banner = screen.getByTestId("playground-chat-error-banner")
    expect(banner).toHaveTextContent("The selected model is not available.")
    expect(banner).toHaveTextContent(
      "Choose a different model or refresh the model list."
    )
    expect(
      screen.getByRole("link", { name: "View in Health & Diagnostics" })
    ).toHaveAttribute("href", "/settings/health")
  })
})
