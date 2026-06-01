// @vitest-environment jsdom
import React from "react"
import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { beforeEach, describe, expect, it, vi } from "vitest"

const onSubmitMock = vi.hoisted(() => vi.fn(async (_payload: unknown) => null))

const messageOptionState = vi.hoisted(() => ({
  value: null as any
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
  historyId: null,
  history: [],
  uploadedFiles: [],
  fileRetrievalEnabled: false,
  setFileRetrievalEnabled: vi.fn(),
  handleFileUpload: vi.fn(),
  removeUploadedFile: vi.fn(),
  clearUploadedFiles: vi.fn(),
  queuedMessages: [],
  setQueuedMessages: vi.fn(),
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
    onMutate,
    onSuccess,
    onError
  }: {
    mutationFn: (args: any) => Promise<any>
    onMutate?: (args: any) => unknown
    onSuccess?: (data: unknown, variables: unknown, context: any) => void
    onError?: (error: unknown) => void
  }) => ({
    mutateAsync: async (args: any) => {
      const context = onMutate?.(args)
      try {
        const result = await mutationFn(args)
        onSuccess?.(result, args, context)
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
    "data-testid": dataTestId
  }: any) => (
    <input
      value={value ?? ""}
      onChange={(event) => onChange?.(event)}
      placeholder={placeholder}
      disabled={disabled}
      data-testid={dataTestId}
    />
  )
  InputComponent.TextArea = ({
    value,
    onChange,
    placeholder,
    readOnly,
    disabled,
    "data-testid": dataTestId
  }: any) => (
    <textarea
      value={value ?? ""}
      onChange={(event) => onChange?.(event)}
      placeholder={placeholder}
      readOnly={readOnly}
      disabled={disabled}
      data-testid={dataTestId}
    />
  )

  return {
    Button: ({
      children,
      onClick,
      disabled,
      loading,
      htmlType,
      title,
      "aria-label": ariaLabel,
      "aria-pressed": ariaPressed,
      "data-testid": dataTestId
    }: any) => (
      <button
        type={htmlType === "submit" ? "submit" : "button"}
        onClick={onClick}
        disabled={disabled || loading}
        title={title}
        aria-label={ariaLabel}
        aria-pressed={ariaPressed}
        data-testid={dataTestId}
      >
        {children}
      </button>
    ),
    Checkbox: ({ children, checked, onChange, disabled }: any) => (
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
    InputNumber: ({ value, onChange, disabled }: any) => (
      <input
        type="number"
        value={value ?? ""}
        onChange={(event) => onChange?.(Number(event.target.value))}
        disabled={disabled}
      />
    ),
    Modal: Object.assign(
      ({
        open,
        children
      }: {
        open?: boolean
        children: React.ReactNode
      }) => (open ? <div role="dialog">{children}</div> : null),
      {
        confirm: vi.fn()
      }
    ),
    Radio: {
      Group: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
      Button: ({ children }: { children: React.ReactNode }) => (
        <button type="button">{children}</button>
      )
    },
    Select: ({
      value,
      options = [],
      onChange,
      disabled
    }: {
      value?: string
      options?: Array<{ value: string; label: string }>
      onChange?: (value: string) => void
      disabled?: boolean
    }) => (
      <select
        value={value ?? ""}
        onChange={(event) => onChange?.(event.target.value)}
        disabled={disabled}
      >
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    ),
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
  useStorage: (_key: string, defaultValue: unknown) =>
    React.useState(defaultValue)
}))

vi.mock("~/hooks/useMessageOption", () => ({
  useMessageOption: () => messageOptionState.value
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
      llamaThinkingBudgetTokens: 0,
      llamaGrammarMode: "off",
      llamaGrammarId: "",
      llamaGrammarInline: "",
      llamaGrammarOverride: "",
      jsonMode: false,
      numCtx: 8192,
      updateSetting: vi.fn(),
      updateSettings: vi.fn()
    })
}))

vi.mock("@/store/chat-surface-coordinator", () => ({
  shouldEnableOptionalResource: () => false,
  useChatSurfaceCoordinatorStore: (selector: (state: any) => unknown) =>
    selector({
      setPanelVisible: vi.fn(),
      markPanelEngaged: vi.fn()
    })
}))

vi.mock("@/store/ui-mode", () => ({
  useUiModeStore: (selector: (state: { mode: string }) => unknown) =>
    selector({ mode: "pro" })
}))

vi.mock("~/store/webui", () => ({
  useWebUI: () => ({
    sendWhenEnter: true,
    setSendWhenEnter: vi.fn(),
    ttsEnabled: false
  })
}))

vi.mock("@/hooks/useConnectionState", () => ({
  useConnectionState: () => ({
    phase: "connected",
    isConnected: true
  })
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

vi.mock("@/hooks/useTldwAudioStatus", () => ({
  useTldwAudioStatus: () => ({ healthState: "ready" })
}))

vi.mock("@/hooks/useMcpTools", () => ({
  useMcpTools: () => ({
    hasMcp: false,
    healthState: "ready",
    tools: [],
    discoveredTools: [],
    chatTools: [],
    toolCounts: { total: 0, enabled: 0 },
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

vi.mock("@/hooks/useChatSettingsRecord", () => ({
  useChatSettingsRecord: () => ({ settings: null, updateSettings: vi.fn() })
}))

vi.mock("@/hooks/useChatMoodBadgePreference", () => ({
  useChatMoodBadgePreference: () => [false, vi.fn()]
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

vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => ({
    success: vi.fn(),
    error: vi.fn(),
    warning: vi.fn(),
    info: vi.fn()
  })
}))

vi.mock("@/components/Chat/composer/ChatComposer", () => ({
  ChatComposer: ({ textareaSlot, bottomBarSlot, facetsSlot }: any) => (
    <div>
      {textareaSlot}
      {bottomBarSlot}
      {facetsSlot}
    </div>
  ),
  useComposerVariantPreference: () => ["classic"]
}))

vi.mock("@/components/Chat/composer/hooks/useComposerEnabledPreference", () => ({
  useComposerEnabledPreference: () => [false]
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

vi.mock("react-router-dom", () => ({
  Link: ({ children, to, ...rest }: any) => (
    <a href={typeof to === "string" ? to : "#"} {...rest}>
      {children}
    </a>
  ),
  useNavigate: () => vi.fn()
}))

vi.mock("@/services/settings/registry", async () => {
  const actual = await vi.importActual<
    typeof import("@/services/settings/registry")
  >("@/services/settings/registry")

  return {
    ...actual,
    clearSetting: vi.fn(),
    getSetting: vi.fn(async () => undefined)
  }
})

vi.mock("@/db/dexie/helpers", () => ({
  getAllPrompts: vi.fn(async () => [])
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
    createResearchRun: vi.fn(async () => ({})),
    updateChat: vi.fn(async () => ({}))
  }
}))

vi.mock("@/components/Common/AudioSourcePicker", () => ({
  AudioSourcePicker: () => null
}))

vi.mock("@/components/Common/Beta", () => ({
  BetaTag: () => null
}))

vi.mock("@/components/Common/CharacterSelect", () => ({
  CharacterSelect: () => null
}))

vi.mock("@/components/Common/ChatQueuePanel", () => ({
  ChatQueuePanel: () => null
}))

vi.mock("@/components/Common/Settings/ActorPopout", () => ({
  ActorPopout: () => null
}))

vi.mock("@/components/Common/Settings/CurrentChatModelSettings", () => ({
  CurrentChatModelSettings: () => null
}))

vi.mock("@/components/Common/Playground/DocumentGeneratorDrawer", () => ({
  default: () => null
}))

vi.mock("../VoiceModeSelector", () => ({
  VoiceModeSelector: () => null
}))

vi.mock("../PlaygroundImageGenModal", () => ({
  PlaygroundImageGenModal: () => null
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
    openUIRequestButton,
    sendControl
  }: {
    openUIRequestButton?: React.ReactNode
    sendControl?: React.ReactNode
  }) => (
    <div data-testid="composer-toolbar">
      {openUIRequestButton}
      {sendControl}
    </div>
  )
}))

vi.mock("../PlaygroundSendControl", () => ({
  PlaygroundAttachmentButton: () => null,
  PlaygroundSendControl: ({
    onSubmitForm,
    sendLabel
  }: {
    onSubmitForm: () => void
    sendLabel?: string
  }) => (
    <button type="button" onClick={onSubmitForm}>
      {sendLabel || "Send"}
    </button>
  )
}))

vi.mock("../PlaygroundToolsPopover", () => ({
  PlaygroundToolsPopover: () => null
}))

vi.mock("../PlaygroundMcpControl", () => ({
  PlaygroundMcpControl: () => null
}))

vi.mock("../PlaygroundModeLauncher", () => ({
  PlaygroundModeLauncher: () => null
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

vi.mock("../AttachedResearchContextChip", () => ({
  AttachedResearchContextChip: () => null
}))

vi.mock("../hooks", async () => {
  const submitHook = await vi.importActual<
    typeof import("../hooks/usePlaygroundSubmit")
  >("../hooks/usePlaygroundSubmit")

  return {
    usePlaygroundSubmit: submitHook.usePlaygroundSubmit,
    toText: (value: unknown) => String(value ?? ""),
    estimateTokensFromText: () => 0,
    useModelComparison: () => ({
      compareModeActive: false,
      compareModelMetaById: {},
      availableCompareModels: [],
      compareModelLabelById: {},
      compareSelectedModelLabels: [],
      compareNeedsMoreModels: false,
      compareModelsSupportCapability: () => true,
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
      dictationAudioSourcePreference: { sourceKind: "browser" },
      dictationResolvedSourceKind: "browser",
      setDictationAudioSourcePreference: vi.fn(),
      isServerDictating: false,
      speechAvailable: false,
      speechUsesServer: false,
      voiceChatAvailable: false,
      voiceChatUnavailableReason: null,
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
    useComposerInput: () => {
      const [values, setValues] = React.useState({ message: "", image: "" })
      const form = {
        values,
        errors: { message: null },
        setFieldValue: (field: string, value: string) =>
          setValues((prev) => ({ ...prev, [field]: value })),
        setFieldError: vi.fn(),
        clearFieldError: vi.fn(),
        reset: () => setValues({ message: "", image: "" }),
        onSubmit:
          (handler: (values: { message: string; image: string }) => void) =>
          async (event?: React.FormEvent) => {
            event?.preventDefault?.()
            await handler(values)
          },
        getInputProps: (field: "message" | "image") => ({
          value: values[field],
          onChange: (
            event: React.ChangeEvent<HTMLTextAreaElement | HTMLInputElement>
          ) =>
            setValues((prev) => ({
              ...prev,
              [field]: event.target.value
            }))
        })
      }

      return {
        form,
        typing: false,
        setMessageValue: (value: string) =>
          setValues((prev) => ({ ...prev, message: value })),
        restoreMessageValue: vi.fn(),
        messageDisplayValue: values.message,
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
        handleTextareaChange: (
          event: React.ChangeEvent<HTMLTextAreaElement>
        ) => setValues((prev) => ({ ...prev, message: event.target.value })),
        handleTextareaSelect: vi.fn(),
        markComposerPerf: vi.fn(),
        measureComposerPerf: vi.fn(),
        onComposerRenderProfile: vi.fn(),
        wrapComposerProfile: (_label: string, node: React.ReactNode) => node,
        draftSaved: false,
        selectedQuickPrompt: null,
        setSelectedQuickPrompt: vi.fn()
      }
    },
    usePlaygroundImageGen: () => ({
      imageGenerateModalOpen: false,
      imageGenerateSubmitting: false,
      imageGenerateBackend: "",
      imageGeneratePrompt: "",
      imageGeneratePromptMode: "scene",
      imageGeneratePromptStrategies: [],
      imageGenerateFormat: "png",
      imageGenerateNegativePrompt: "",
      imageGenerateWidth: undefined,
      imageGenerateHeight: undefined,
      imageGenerateSteps: undefined,
      imageGenerateCfgScale: undefined,
      imageGenerateSeed: undefined,
      imageGenerateSampler: "",
      imageGenerateModel: "",
      imageGenerateExtraParams: "",
      imageGenerateReferenceFileId: undefined,
      imageGenerateReferenceCandidates: [],
      imageGenerateReferenceCandidatesLoading: false,
      imageGenerateSyncPolicy: "inherit",
      imageGenerateResolvedSyncMode: "off",
      imageGenerateRefineSubmitting: false,
      imageGenerateRefineBaseline: "",
      imageGenerateRefineCandidate: null,
      imageGenerateRefineModel: null,
      imageGenerateRefineLatencyMs: null,
      imageGenerateRefineDiff: null,
      setImageGenerateBackend: vi.fn(),
      setImageGeneratePrompt: vi.fn(),
      setImageGeneratePromptMode: vi.fn(),
      setImageGenerateFormat: vi.fn(),
      setImageGenerateNegativePrompt: vi.fn(),
      setImageGenerateWidth: vi.fn(),
      setImageGenerateHeight: vi.fn(),
      setImageGenerateSteps: vi.fn(),
      setImageGenerateCfgScale: vi.fn(),
      setImageGenerateSeed: vi.fn(),
      setImageGenerateSampler: vi.fn(),
      setImageGenerateModel: vi.fn(),
      setImageGenerateExtraParams: vi.fn(),
      setImageGenerateReferenceFileId: vi.fn(),
      setImageGenerateSyncPolicy: vi.fn(),
      setImageGenerateEventSyncGlobalDefault: vi.fn(),
      closeImageGenerateModal: vi.fn(),
      hydrateImageGenerateSettings: vi.fn(),
      openImageGenerateModal: vi.fn(),
      createImagePromptDraft: vi.fn(),
      clearImagePromptRefineState: vi.fn(),
      refineImagePromptWithLlm: vi.fn(),
      applyRefinedImagePromptCandidate: vi.fn(),
      rejectRefinedImagePromptCandidate: vi.fn(),
      submitImageGenerateModal: vi.fn(),
      normalizeImageGenerationEventSyncMode: (value: string) => value,
      normalizeImageGenerationEventSyncPolicy: (value: string) => value
    }),
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
      availableChatModelIds: ["deepseek-chat"],
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
      validateSelectedChatModelsAvailability: () => true
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
    })
  }
})

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
    estimateTokensForText: (value: string) =>
      Math.ceil((value || "").length / 4)
  }),
  useImageBackend: () => ({
    imageBackendDefault: "mock-backend",
    setImageBackendDefault: vi.fn(),
    imageBackendOptions: [],
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
    getCollapsedDisplayMeta: vi.fn((message: string) => ({
      display: message
    })),
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
    handleCatalogSelect: vi.fn(),
    catalogGroups: { team: [], org: [], global: [] },
    catalogDraft: "",
    setCatalogDraft: vi.fn(),
    commitCatalog: vi.fn()
  })
}))

vi.mock("@/hooks/useMediaQuery", () => ({
  useMobile: () => false
}))

vi.mock("@/utils/onboarding-ingestion-telemetry", () => ({
  trackOnboardingChatSubmitSuccess: vi.fn(async () => undefined)
}))

import { PlaygroundForm } from "../PlaygroundForm"

describe("PlaygroundForm OpenUI mode", () => {
  beforeEach(() => {
    onSubmitMock.mockClear()
    messageOptionState.value = createMessageOptionState()
  })

  it("sends the next prompt with OpenUI request overrides", async () => {
    const user = userEvent.setup()
    render(<PlaygroundForm droppedFiles={[]} />)

    const openUIButton = screen.getByRole("button", { name: /OpenUI/i })
    expect(openUIButton).toBeInTheDocument()

    await user.click(openUIButton)
    await user.type(screen.getByTestId("composer-textarea"), "Build a settings form")
    await user.click(screen.getAllByRole("button", { name: "Send" })[0])

    await waitFor(() => expect(onSubmitMock).toHaveBeenCalledTimes(1))
    expect(onSubmitMock).toHaveBeenCalledWith(
      expect.objectContaining({
        message: "Build a settings form",
        requestOverrides: {
          dynamicUIRequest: { renderer: "openui" }
        }
      })
    )
    await waitFor(() =>
      expect(screen.getByRole("button", { name: /OpenUI/i })).toHaveAttribute(
        "aria-pressed",
        "false"
      )
    )
  })
})
