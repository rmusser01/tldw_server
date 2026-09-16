// @vitest-environment jsdom
import React from "react"
import { act, cleanup, render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { useMilestoneStore } from "@/store/milestones"
import { DISCUSS_MEDIA_PROMPT_SETTING } from "@/services/settings/ui-settings"
import { useChatActions } from "@/hooks/chat/useChatActions"
import { useStoreMessageOption } from "@/store/option"
import { usePlaygroundSessionStore } from "@/store/playground-session"
import { usePlaygroundSessionPersistence } from "@/hooks/usePlaygroundSessionPersistence"
import { chatRagMethods } from "@/services/tldw/domains/chat-rag"
import { buildChatSurfaceScopeKeyFromConfig } from "@/services/chat-surface-scope"
import type { ServicePromptSnapshot } from "@/services/service-prompts"
import type { TldwApiClientCore } from "@/services/tldw/TldwApiClient"
const handoff = vi.hoisted(() => ({ scope: "server-a:alice" as string | null, get: vi.fn(), clear: vi.fn(), setRagMediaIds: vi.fn() }))
vi.mock("@/hooks/useHomeMilestoneScope", () => ({ useHomeMilestoneScope: () => handoff.scope }))

// These cases keep the Form, submit hook, action router, session restore, RAG
// pipeline and request serializer real. Only network/DB and unrelated UI are fixtures.
const sourceFlow = vi.hoisted(() => ({
  enabled: false,
  request: vi.fn(),
  stream: vi.fn(),
  getFullChatData: vi.fn(),
  snapshot: null as ServicePromptSnapshot | null,
  restore: null as null | (() => Promise<unknown>),
  sessionReady: false,
  config: { serverUrl: "https://handoff.test", authMode: "single-user" as const, apiKey: "synthetic-handoff" }
}))
vi.mock("@/services/background-proxy", () => ({ bgRequest: (...args: unknown[]) => sourceFlow.request(...args) }))
vi.mock("@/services/service-prompts", async () => ({
  ...await vi.importActual<typeof import("@/services/service-prompts")>("@/services/service-prompts"),
  loadServicePromptSnapshot: async () => sourceFlow.snapshot
}))
vi.mock("@/models", () => ({ pageAssistModel: async () => ({ stream: sourceFlow.stream }) }))
vi.mock("@/services/actor-settings", () => ({ getActorSettingsForChat: async () => null }))
vi.mock("@/db/dexie/nickname", () => ({ getModelNicknameByID: async () => null }))
vi.mock("@/services/chat-settings", () => ({ syncChatSettingsForServerChat: async () => null }))
vi.mock("@/services/tldw/server-capabilities", () => ({ getServerCapabilities: async () => ({ hasChatSaveToDb: true }) }))
vi.mock("@/services/title", () => ({ generateTitle: async () => "Cedar saved chat" }))
vi.mock("@/services/app", async () => ({
  ...await vi.importActual<typeof import("@/services/app")>("@/services/app"),
  getNoOfRetrievedDocs: async () => 8
}))
vi.mock("@/utils/resolve-api-provider", async () => ({
  ...await vi.importActual<typeof import("@/utils/resolve-api-provider")>("@/utils/resolve-api-provider"),
  resolveApiProviderForModel: async () => "openai"
}))
vi.mock("@/hooks/utils/messageHelpers", async () => ({
  ...await vi.importActual<typeof import("@/hooks/utils/messageHelpers")>("@/hooks/utils/messageHelpers"),
  createSaveMessageOnSuccess: () => async () => "cedar-local",
  createSaveMessageOnError: () => async () => "cedar-local"
}))

const onSubmitMock = vi.hoisted(() =>
  vi.fn(async (_payload: unknown) => ({ status: "submitted" as const }))
)

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

vi.mock("@/components/Chat/composer/PromptAssistComposerAction", () => ({ PromptAssistComposerAction: () => null }))

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
  useMessageOption: () => {
    // The fixture is selected before mount and reset only after cleanup.
    const useOptions = sourceFlow.enabled ? useSourceFlowActions : useStaticMessageOptions
    return useOptions()
  }
}))

vi.mock("@/store/option", async () => {
  const actual = await vi.importActual<typeof import("@/store/option")>("@/store/option")
  return { ...actual, useStoreMessageOption: Object.assign(
    (selector: (state: any) => unknown, equality?: (left: unknown, right: unknown) => boolean) => sourceFlow.enabled
      ? actual.useStoreMessageOption(selector, equality)
      : selector({
      setRagMediaIds: handoff.setRagMediaIds,
      setRagPinnedResults: vi.fn()
    }), actual.useStoreMessageOption) }
})

vi.mock("@/store/model", () => ({
  useStoreChatModelSettings: (selector: (state: any) => unknown) =>
    (selector ?? ((state: unknown) => state))({
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
      setActiveSettingsScope: vi.fn(),
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
    isConnected: true,
    serverUrl: sourceFlow.config.serverUrl,
    lastConfigUpdatedAt: 0
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

vi.mock("@/hooks/chat/useChatSettingsRecord", () => ({
  useChatSettingsRecord: () => ({ settings: null, updateSettings: vi.fn() })
}))

vi.mock("@/hooks/useChatMoodBadgePreference", () => ({
  useChatMoodBadgePreference: () => [false, vi.fn()]
}))

vi.mock("@/hooks/useSelectedAssistant", () => ({
  useSelectedAssistant: () => [null, vi.fn()]
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
  useLocation: () => ({ pathname: "/chat", search: "", hash: "" }),
  useNavigate: () => vi.fn()
}))

vi.mock("@/services/settings/registry", async () => {
  const actual = await vi.importActual<
    typeof import("@/services/settings/registry")
  >("@/services/settings/registry")

  return {
    ...actual,
    clearSetting: handoff.clear,
    getSetting: handoff.get
  }
})

vi.mock("@/db/dexie/helpers", async () => ({
  ...await vi.importActual<typeof import("@/db/dexie/helpers")>("@/db/dexie/helpers"),
  getAllPrompts: vi.fn(async () => []),
  getFullChatData: (...args: unknown[]) => sourceFlow.getFullChatData(...args),
  getSessionFiles: async () => [],
  updateLastUsedModel: async () => undefined,
  updateChatHistoryCreatedAt: async () => undefined
}))

vi.mock("@/services/tldw-server", () => ({
  defaultEmbeddingModelForRag: vi.fn(async () => "embedding"),
  fetchChatModels: vi.fn(async () => []),
  fetchImageModels: vi.fn(async () => []),
  systemPromptForNonRagOption: async () => ""
}))

vi.mock("@/services/search", () => ({
  getIsSimpleInternetSearch: vi.fn(async () => true)
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: vi.fn(async () => undefined),
    getConfig: async () => sourceFlow.config,
    ragSearch: (query: string, options: unknown) => chatRagMethods.ragSearch.call({ normalizeRagQuery: (value: string) => value } as unknown as TldwApiClientCore, query, options),
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
  PlaygroundKnowledgeSection: ({ fileRetrievalEnabled, onFileRetrievalChange }: {
    fileRetrievalEnabled: boolean
    onFileRetrievalChange: (enabled: boolean) => void
  }) => sourceFlow.enabled ? (
    <label>
      Source retrieval
      <input type="checkbox" checked={fileRetrievalEnabled} onChange={event => onFileRetrievalChange(event.target.checked)} />
    </label>
  ) : null
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
        beginPromptAssistReset: vi.fn(() => 1),
        markPromptAssistAttemptSaved: vi.fn(),
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

function useSourceFlowActions() {
  const state = useStoreMessageOption((value) => value)
  const [abortController, setAbortController] = React.useState<AbortController | null>(null)
  const actions = useChatActions({
    ...state,
    abortController,
    setAbortController,
    t: (_key: string, fallback?: string) => fallback || _key,
    notification: { error: vi.fn(), warning: vi.fn(), info: vi.fn(), success: vi.fn() },
    currentChatModelSettings: { apiProvider: "openai", setSystemPrompt: vi.fn() },
    ensureServerChatHistoryId: async () => "cedar-local",
    compareModeActive: false,
    compareFeatureEnabled: false,
    markCompareHistoryCreated: vi.fn(),
    selectedAssistant: null,
    selectedCharacter: null,
    isCharacterConversation: false,
    setSelectedModel: state.setSelectedModel
  } as Parameters<typeof useChatActions>[0])
  return { ...createMessageOptionState(), ...state, ...actions }
}

function useStaticMessageOptions() {
  return messageOptionState.value
}

function SourceFlowHarness() {
  const session = usePlaygroundSessionPersistence()
  sourceFlow.restore = session.restoreSession
  sourceFlow.sessionReady = session.sessionScopeReady
  return <PlaygroundForm droppedFiles={[]} />
}

const defer = <T,>() => {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((done) => { resolve = done })
  return { promise, resolve }
}

const cedarRows = [
  { id: "cedar-question", history_id: "cedar-local", role: "user", content: "Tell me about the old Cedar garden.", images: [], sources: [], createdAt: 1 },
  { id: "cedar-answer", history_id: "cedar-local", role: "assistant", content: "Cedar has a green gate.", images: [], sources: [], createdAt: 2 }
]
const cedarData = { historyInfo: { id: "cedar-local", title: "Cedar saved chat", server_chat_id: "cedar-server" }, messages: cedarRows }
const rowanPayload = { ownerScope: "server-a:alice", mediaId: "42", title: "Rowan.md", content: "Summarize this source.", mode: "rag_media" }
const rowanText = "The Rowan archive keeps violet maps in the east room."

describe("Home source handoff through the real Chat and RAG send boundary", () => {
  beforeEach(() => {
    sourceFlow.enabled = true
    sourceFlow.sessionReady = false
    sourceFlow.restore = null
    sourceFlow.config.serverUrl = "https://handoff.test"
    sourceFlow.config.apiKey = "synthetic-handoff"
    localStorage.clear()
    useStoreMessageOption.setState(useStoreMessageOption.getInitialState(), true)
    useStoreMessageOption.setState({ selectedModel: "openai:gpt-4o-mini", temporaryChat: false, fileRetrievalEnabled: false })
    usePlaygroundSessionStore.getState().clearSession()
    const scopeKey = buildChatSurfaceScopeKeyFromConfig(sourceFlow.config)
    usePlaygroundSessionStore.getState().saveSession({
      scopeKey, historyId: "cedar-local", serverChatId: "cedar-server", chatMode: "normal", ragMediaIds: null
    })
    const controller = new AbortController()
    sourceFlow.snapshot = {
      scopeKey, requestScope: { config: sourceFlow.config, userId: 1 },
      scopeSignal: controller.signal, scopeInvalidatedSignal: controller.signal,
      capability: "supported", release: vi.fn(),
      definitions: Object.fromEntries(["chat.rag.answer", "chat.rag.question_rewrite"].map((id) => [id, {
        definition: { id, parts: [{ key: "template", mode: "template", required_variables: ["context", "question"] }] },
        parts: { template: "Source evidence: {context}\nQuestion: {question}" }, source: "packaged", revision: null
      }]))
    }
    handoff.scope = "server-a:alice"
    handoff.get.mockReset().mockResolvedValue(undefined)
    handoff.clear.mockReset().mockResolvedValue(undefined)
    sourceFlow.getFullChatData.mockReset().mockResolvedValue(cedarData)
    sourceFlow.request.mockReset().mockResolvedValue({ results: [{ content: rowanText, metadata: { media_id: 42, title: "Rowan.md" } }] })
    sourceFlow.stream.mockReset().mockImplementation(async function* () { yield { content: "Rowan keeps violet maps in the east room." } })
    messageOptionState.value = createMessageOptionState()
  })

  afterEach(() => { cleanup(); sourceFlow.enabled = false })

  const consume = async (payload: unknown = rowanPayload) => {
    await act(async () => window.dispatchEvent(new CustomEvent("tldw:discuss-media", { detail: payload })))
    await waitFor(() => expect(screen.getByTestId("composer-textarea")).toHaveValue("Chat with this media: Rowan.md\n\nSummarize this source."))
  }
  const restore = async () => {
    await waitFor(() => expect(sourceFlow.sessionReady).toBe(true))
    await act(async () => { await sourceFlow.restore!() })
  }
  const send = async () => {
    await userEvent.setup().click(screen.getAllByRole("button", { name: "Send" })[0])
    await waitFor(() => expect(useStoreMessageOption.getState().streaming).toBe(false))
  }

  const persistSourceSession = async (enabled = true, failedRetrieval = false) => {
    const view = render(<SourceFlowHarness />)
    await restore()
    await consume()
    if (failedRetrieval) {
      sourceFlow.request.mockRejectedValueOnce(new Error("Source unavailable"))
      await send()
      expect(sourceFlow.stream).not.toHaveBeenCalled()
    }
    await act(async () => useStoreMessageOption.getState().setFileRetrievalEnabled(enabled))
    // Exercise the real unmount flush, including its asynchronous scope lookup.
    await act(async () => view.unmount())
    await waitFor(() => expect(usePlaygroundSessionStore.getState().ragMediaIds).toEqual([42]))
    return localStorage.getItem("tldw-playground-session")!
  }

  const rehydrateColdSession = async (serialized: string) => {
    await act(async () => {
      useStoreMessageOption.setState(useStoreMessageOption.getInitialState(), true)
      useStoreMessageOption.setState({ selectedModel: "openai:gpt-4o-mini", temporaryChat: false })
      usePlaygroundSessionStore.setState(usePlaygroundSessionStore.getInitialState(), true)
      localStorage.setItem("tldw-playground-session", serialized)
      await usePlaygroundSessionStore.persist.rehydrate()
    })
    sourceFlow.restore = null
    sourceFlow.sessionReady = false
    sourceFlow.request.mockClear()
    sourceFlow.stream.mockClear()
  }

  it.each([false, true])("restores source activation through cold persistence and Send (prior retrieval failure=%s)", async failedRetrieval => {
    await rehydrateColdSession(await persistSourceSession(true, failedRetrieval))
    render(<SourceFlowHarness />)
    await restore()
    await userEvent.setup().type(screen.getByTestId("composer-textarea"), "Where does Rowan keep the maps?")
    await send()

    expect(sourceFlow.request).toHaveBeenCalledWith(expect.objectContaining({
      path: "/api/v1/rag/search",
      body: expect.objectContaining({ include_media_ids: [42], sources: ["media_db"] })
    }))
    expect(JSON.stringify(sourceFlow.stream.mock.calls[0][0])).toContain(rowanText)
    expect(useStoreMessageOption.getState().messages.at(-1)?.sources[0]?.pageContent).toBe(rowanText)
    expect(useStoreMessageOption.getState().serverChatId).toBe("cedar-server")
  })

  it.each(["explicit-false", "legacy-missing"])("keeps %s retrieval disabled after cold rehydrate", async savedFlag => {
    const persisted = JSON.parse(await persistSourceSession(false))
    if (savedFlag === "legacy-missing") delete persisted.state.fileRetrievalEnabled
    await rehydrateColdSession(JSON.stringify(persisted))
    render(<SourceFlowHarness />)
    await restore()
    await userEvent.setup().type(screen.getByTestId("composer-textarea"), "Continue the ordinary conversation.")
    await send()

    expect(sourceFlow.request).not.toHaveBeenCalled()
    expect(sourceFlow.stream).toHaveBeenCalledTimes(1)
    expect(useStoreMessageOption.getState().fileRetrievalEnabled).toBe(false)
  })

  it.each(["account", "server"])("rejects source activation from a cold session owned by another %s", async changed => {
    await rehydrateColdSession(await persistSourceSession())
    if (changed === "account") sourceFlow.config.apiKey = "synthetic-other-owner"
    else sourceFlow.config.serverUrl = "https://other-handoff.test"
    render(<SourceFlowHarness />)
    await restore()

    expect(useStoreMessageOption.getState().serverChatId).toBeNull()
    expect(useStoreMessageOption.getState().ragMediaIds).toBeNull()
    expect(useStoreMessageOption.getState().fileRetrievalEnabled).toBe(false)
  })

  it.each(["before-restore", "during-restore"])("preserves a newer explicit retrieval toggle off %s", async order => {
    await rehydrateColdSession(await persistSourceSession())
    const historyRead = defer<typeof cedarData>()
    sourceFlow.getFullChatData.mockClear()
    if (order === "during-restore") sourceFlow.getFullChatData.mockReturnValue(historyRead.promise)
    render(<SourceFlowHarness />)
    await waitFor(() => expect(sourceFlow.sessionReady).toBe(true))
    let restoring: Promise<unknown> | undefined
    if (order === "during-restore") {
      await act(async () => { restoring = sourceFlow.restore!() })
      await waitFor(() => expect(sourceFlow.getFullChatData).toHaveBeenCalledTimes(1))
    }
    // Reach the real Form's user callback, including a return to the initial
    // false value. A value comparison alone cannot protect this newer intent.
    const user = userEvent.setup()
    await user.click(screen.getByRole("checkbox", { name: "Source retrieval" }))
    await user.click(screen.getByRole("checkbox", { name: "Source retrieval" }))
    if (order === "during-restore") await act(async () => { historyRead.resolve(cedarData); await restoring })
    else await restore()
    await user.type(screen.getByTestId("composer-textarea"), "Continue the ordinary conversation.")
    await send()

    expect(sourceFlow.request).not.toHaveBeenCalled()
    expect(useStoreMessageOption.getState().fileRetrievalEnabled).toBe(false)
    expect(useStoreMessageOption.getState().serverChatId).toBe("cedar-server")
  })

  it.each([
    { order: "restore-first", priorSource: false },
    { order: "handoff-first", priorSource: false },
    { order: "restore-first", priorSource: true },
    { order: "handoff-first", priorSource: true },
    { order: "handoff-before-restore", priorSource: true }
  ])("retrieves Rowan before generation with old Cedar history ($order, prior source=$priorSource)", async ({ order, priorSource }) => {
    if (priorSource) usePlaygroundSessionStore.getState().saveSession({ chatMode: "rag", ragMediaIds: [7] })
    const payload = defer<unknown>()
    handoff.get.mockImplementation((key) => key === DISCUSS_MEDIA_PROMPT_SETTING ? payload.promise : Promise.resolve(undefined))
    handoff.clear.mockImplementation(async () => { handoff.get.mockResolvedValue(undefined) })
    const historyRead = defer<typeof cedarData>()
    if (order === "handoff-first") sourceFlow.getFullChatData.mockReturnValue(historyRead.promise)
    render(<SourceFlowHarness />)
    await waitFor(() => expect(sourceFlow.sessionReady).toBe(true))
    let restoring: Promise<unknown> | undefined
    if (order === "restore-first") await restore()
    else if (order === "handoff-first") await act(async () => { restoring = sourceFlow.restore!() })
    await act(async () => { payload.resolve(rowanPayload) })
    await waitFor(() => expect(screen.getByTestId("composer-textarea")).toHaveValue("Chat with this media: Rowan.md\n\nSummarize this source."))
    if (order === "handoff-first") await act(async () => { historyRead.resolve(cedarData); await restoring })
    if (order === "handoff-before-restore") await restore()
    expect(useStoreMessageOption.getState().history).toHaveLength(2)
    await send()
    expect(sourceFlow.request).toHaveBeenCalledWith(expect.objectContaining({
      path: "/api/v1/rag/search", method: "POST",
      body: expect.objectContaining({ include_media_ids: [42], sources: ["media_db"] }),
      servicePromptConfig: expect.objectContaining({ serverUrl: sourceFlow.config.serverUrl }),
      headers: expect.objectContaining({ "X-TLDW-Expected-User-ID": "1" })
    }))
    expect(sourceFlow.stream).toHaveBeenCalledTimes(1)
    expect(JSON.stringify(sourceFlow.stream.mock.calls[0][0])).toContain(rowanText)
    expect(useStoreMessageOption.getState().messages.at(-1)?.sources[0]?.pageContent).toBe(rowanText)
    expect(useStoreMessageOption.getState().chatMode).toBe("rag")
  })

  it.each(["empty", "error"])("does not generate from the old conversation after %s selected-source retrieval", async (outcome) => {
    if (outcome === "empty") sourceFlow.request.mockResolvedValue({ results: [] })
    else sourceFlow.request.mockRejectedValue(new Error("Source unavailable"))
    render(<SourceFlowHarness />)
    await restore()
    await consume()
    await send()
    expect(sourceFlow.request).toHaveBeenCalledTimes(1)
    expect(sourceFlow.stream).not.toHaveBeenCalled()
    expect(useStoreMessageOption.getState().messages.at(-1)?.message).toMatch(/selected source|selected media/i)
  })

  it("keeps a full-content ordinary handoff on the ordinary path", async () => {
    render(<SourceFlowHarness />)
    await restore()
    await act(async () => useStoreMessageOption.setState({ fileRetrievalEnabled: true, ragMediaIds: [7], chatMode: "rag" }))
    await act(async () => window.dispatchEvent(new CustomEvent("tldw:discuss-media", { detail: { mediaId: "42", title: "Rowan.md", content: rowanText, mode: "chat" } })))
    await send()
    expect(sourceFlow.request).not.toHaveBeenCalled()
    expect(sourceFlow.stream).toHaveBeenCalledTimes(1)
    expect(JSON.stringify(sourceFlow.stream.mock.calls[0][0])).toContain(rowanText)
  })

  it("does not arm retrieval or seed the composer for a different owner's handoff", async () => {
    render(<SourceFlowHarness />)
    await restore()
    const intentRevision = usePlaygroundSessionStore.getState().sourceSelectionRevision
    await act(async () => window.dispatchEvent(new CustomEvent("tldw:discuss-media", { detail: { ...rowanPayload, ownerScope: "server-a:bob" } })))
    expect(screen.getByTestId("composer-textarea")).toHaveValue("")
    expect(useStoreMessageOption.getState().fileRetrievalEnabled).toBe(false)
    expect(useStoreMessageOption.getState().ragMediaIds).toBeNull()
    expect(sourceFlow.request).not.toHaveBeenCalled()
    expect(usePlaygroundSessionStore.getState().sourceSelectionRevision).toBe(intentRevision)
  })

  it("discards an earlier owner's storage read through A to B to A", async () => {
    const intentRevision = usePlaygroundSessionStore.getState().sourceSelectionRevision
    const oldRead = defer<unknown>()
    let readStarted = false
    handoff.get.mockImplementation((key) => {
      if (key === DISCUSS_MEDIA_PROMPT_SETTING && !readStarted) {
        readStarted = true
        return oldRead.promise
      }
      return Promise.resolve(undefined)
    })
    const view = render(<SourceFlowHarness />)
    await waitFor(() => expect(readStarted).toBe(true))
    handoff.scope = "server-a:bob"
    view.rerender(<SourceFlowHarness />)
    await act(async () => {})
    handoff.scope = "server-a:alice"
    view.rerender(<SourceFlowHarness />)
    await act(async () => { oldRead.resolve(rowanPayload) })
    expect(screen.getByTestId("composer-textarea")).toHaveValue("")
    expect(useStoreMessageOption.getState().fileRetrievalEnabled).toBe(false)
    expect(sourceFlow.request).not.toHaveBeenCalled()
    expect(usePlaygroundSessionStore.getState().sourceSelectionRevision).toBe(intentRevision)
  })

  it("does not publish a late source answer after the captured request authority is invalidated", async () => {
    const retrieval = defer<unknown>()
    const authority = new AbortController()
    sourceFlow.snapshot = { ...sourceFlow.snapshot!, scopeSignal: authority.signal, scopeInvalidatedSignal: authority.signal }
    sourceFlow.request.mockReturnValue(retrieval.promise)
    sourceFlow.stream.mockImplementation(async function* (_history, options) {
      options.signal.throwIfAborted()
      yield { content: "late answer" }
    })
    render(<SourceFlowHarness />)
    await restore()
    await consume()
    await userEvent.setup().click(screen.getAllByRole("button", { name: "Send" })[0])
    await waitFor(() => expect(sourceFlow.request).toHaveBeenCalledTimes(1))
    await act(async () => {
      authority.abort()
      retrieval.resolve({ results: [{ content: rowanText, metadata: { media_id: 42 } }] })
    })
    await waitFor(() => expect(useStoreMessageOption.getState().streaming).toBe(false))
    for (const [, options] of sourceFlow.stream.mock.calls) expect(options.signal.aborted).toBe(true)
    expect(useStoreMessageOption.getState().messages.map((message) => message.id)).toEqual(["cedar-question", "cedar-answer"])
    expect(useStoreMessageOption.getState().historyId).toBe("cedar-local")
  })

  it.each(["before-restore", "during-restore"])("preserves an accepted same-value Rowan handoff %s", async (order) => {
    useStoreMessageOption.setState({ chatMode: "rag", ragMediaIds: [42], fileRetrievalEnabled: true })
    usePlaygroundSessionStore.getState().saveSession({ chatMode: "rag", ragMediaIds: [7] })
    const historyRead = defer<typeof cedarData>()
    if (order === "during-restore") sourceFlow.getFullChatData.mockReturnValue(historyRead.promise)
    render(<SourceFlowHarness />)
    await waitFor(() => expect(sourceFlow.sessionReady).toBe(true))
    let restoring: Promise<unknown> | undefined
    if (order === "during-restore") await act(async () => { restoring = sourceFlow.restore!() })
    await consume()
    if (order === "during-restore") await act(async () => { historyRead.resolve(cedarData); await restoring })
    else await restore()
    await send()
    expect(sourceFlow.request.mock.calls[0][0].body.include_media_ids).toEqual([42])
    expect(JSON.stringify(sourceFlow.stream.mock.calls[0][0])).toContain(rowanText)
  })

  it.each(["before-restore", "during-restore"])("preserves an accepted same-value ordinary handoff %s", async (order) => {
    useStoreMessageOption.setState({ chatMode: "normal", ragMediaIds: null, fileRetrievalEnabled: true })
    usePlaygroundSessionStore.getState().saveSession({ chatMode: "rag", ragMediaIds: [7] })
    const historyRead = defer<typeof cedarData>()
    if (order === "during-restore") sourceFlow.getFullChatData.mockReturnValue(historyRead.promise)
    render(<SourceFlowHarness />)
    await waitFor(() => expect(sourceFlow.sessionReady).toBe(true))
    let restoring: Promise<unknown> | undefined
    if (order === "during-restore") await act(async () => { restoring = sourceFlow.restore!() })
    await act(async () => window.dispatchEvent(new CustomEvent("tldw:discuss-media", { detail: { mediaId: "42", title: "Rowan.md", content: rowanText, mode: "chat" } })))
    if (order === "during-restore") await act(async () => { historyRead.resolve(cedarData); await restoring })
    else await restore()
    await send()
    expect(sourceFlow.request).not.toHaveBeenCalled()
    expect(JSON.stringify(sourceFlow.stream.mock.calls[0][0])).toContain(rowanText)
    expect(useStoreMessageOption.getState().chatMode).toBe("normal")
    expect(useStoreMessageOption.getState().history.slice(0, 2).map((row) => row.content)).toEqual(cedarRows.map((row) => row.content))
  })

  it("preserves accepted source intent through Rowan to Cedar to Rowan while restore is pending", async () => {
    useStoreMessageOption.setState({ chatMode: "rag", ragMediaIds: [42], fileRetrievalEnabled: true })
    usePlaygroundSessionStore.getState().saveSession({ chatMode: "rag", ragMediaIds: [7] })
    const historyRead = defer<typeof cedarData>()
    sourceFlow.getFullChatData.mockReturnValue(historyRead.promise)
    render(<SourceFlowHarness />)
    await waitFor(() => expect(sourceFlow.sessionReady).toBe(true))
    let restoring!: Promise<unknown>
    await act(async () => { restoring = sourceFlow.restore!() })
    await consume()
    await act(async () => window.dispatchEvent(new CustomEvent("tldw:discuss-media", { detail: { ...rowanPayload, title: "Cedar.md", mediaId: "7" } })))
    await consume()
    await act(async () => { historyRead.resolve(cedarData); await restoring })
    await send()
    expect(sourceFlow.request.mock.calls[0][0].body.include_media_ids).toEqual([42])
  })

  it("allows a later requested restore on the same hook to replay saved source selection", async () => {
    render(<SourceFlowHarness />)
    await restore()
    await consume()
    await act(async () => usePlaygroundSessionStore.getState().saveSession({ chatMode: "rag", ragMediaIds: [7] }))
    await restore()
    expect(useStoreMessageOption.getState().ragMediaIds).toEqual([7])
    expect(useStoreMessageOption.getState().chatMode).toBe("rag")
  })

  it.each(["conversation", "account"])("does not apply the initial selection baseline to a different persisted %s", async (changed) => {
    render(<SourceFlowHarness />)
    await waitFor(() => expect(sourceFlow.sessionReady).toBe(true))
    await consume()
    if (changed === "account") sourceFlow.config.apiKey = "synthetic-other-owner"
    sourceFlow.getFullChatData.mockResolvedValue({ ...cedarData, historyInfo: { ...cedarData.historyInfo, id: "other-local", server_chat_id: "other-server" } })
    await act(async () => usePlaygroundSessionStore.getState().saveSession({ scopeKey: buildChatSurfaceScopeKeyFromConfig(sourceFlow.config), historyId: "other-local", serverChatId: "other-server", chatMode: "rag", ragMediaIds: [7] }))
    await restore()
    expect(useStoreMessageOption.getState().ragMediaIds).toEqual([7])
    expect(useStoreMessageOption.getState().serverChatId).toBe("other-server")
  })

  it("hydrates the real localStorage session synchronously before the handoff owner mounts", async () => {
    const persisted = JSON.parse(localStorage.getItem("tldw-playground-session")!)
    persisted.state.chatMode = "rag"
    persisted.state.ragMediaIds = [7]
    localStorage.setItem("tldw-playground-session", JSON.stringify(persisted))
    let hydrated = false
    const unsubscribe = usePlaygroundSessionStore.persist.onFinishHydration(() => { hydrated = true })
    void usePlaygroundSessionStore.persist.rehydrate()
    // No await: the configured browser storage and migration both hydrate synchronously.
    expect(hydrated).toBe(true)
    expect(usePlaygroundSessionStore.persist.hasHydrated()).toBe(true)
    expect(usePlaygroundSessionStore.getState().ragMediaIds).toEqual([7])
    unsubscribe()
    render(<SourceFlowHarness />)
    await consume()
    await restore()
    await send()
    expect(sourceFlow.request.mock.calls[0][0].body.include_media_ids).toEqual([42])
  })
})

describe("PlaygroundForm OpenUI mode", () => {
  beforeEach(() => {
    onSubmitMock.mockReset().mockResolvedValue({ status: "submitted" })
    handoff.scope = "server-a:alice"
    handoff.get.mockReset().mockResolvedValue(undefined)
    handoff.clear.mockReset().mockResolvedValue(undefined)
    handoff.setRagMediaIds.mockReset()
    useMilestoneStore.getState().resetMilestones()
    messageOptionState.value = createMessageOptionState()
  })

  it("consumes legacy unowned handoffs when token-derived identity is unavailable", async () => {
    handoff.scope = null
    handoff.get.mockImplementation(async (setting) => setting === DISCUSS_MEDIA_PROMPT_SETTING ? { mediaId: "42", title: "Legacy review", content: "Discuss", mode: "rag_media" } : undefined)
    render(<PlaygroundForm droppedFiles={[]} />)
    await waitFor(() => expect(screen.getByTestId("composer-textarea")).toHaveValue("Chat with this media: Legacy review\n\nDiscuss"))
    expect(handoff.setRagMediaIds).toHaveBeenCalledWith([42])
    expect(handoff.clear).toHaveBeenCalledWith(DISCUSS_MEDIA_PROMPT_SETTING)
  })

  it("waits for a resolved identity before applying or clearing an owned handoff", async () => {
    handoff.scope = null
    handoff.get.mockImplementation(async (setting) => setting === DISCUSS_MEDIA_PROMPT_SETTING ? { ownerScope: "server-a:alice", mediaId: "42", title: "Owned source", content: "Discuss", mode: "rag_media" } : undefined)
    const view = render(<PlaygroundForm droppedFiles={[]} />)
    await act(async () => {})
    expect(screen.getByTestId("composer-textarea")).toHaveValue("")
    expect(handoff.setRagMediaIds).not.toHaveBeenCalled()
    expect(handoff.clear).not.toHaveBeenCalledWith(DISCUSS_MEDIA_PROMPT_SETTING)
    handoff.scope = "server-a:alice"
    view.rerender(<PlaygroundForm droppedFiles={[]} />)
    await waitFor(() => expect(screen.getByTestId("composer-textarea")).toHaveValue("Chat with this media: Owned source\n\nDiscuss"))
    expect(handoff.clear).toHaveBeenCalledWith(DISCUSS_MEDIA_PROMPT_SETTING)
  })

  it.each(["server-a:alice", "server-a:bob"])("only applies a saved source owned by %s when identity matches", async (ownerScope) => {
    handoff.get.mockImplementation(async (setting) => setting === DISCUSS_MEDIA_PROMPT_SETTING ? { ownerScope, mediaId: "42", title: "Private source", content: "Summarize", mode: "rag_media" } : undefined)
    render(<PlaygroundForm droppedFiles={[]} />)
    await waitFor(() => expect(handoff.clear).toHaveBeenCalledWith(DISCUSS_MEDIA_PROMPT_SETTING))
    if (ownerScope === handoff.scope) {
      expect(screen.getByTestId("composer-textarea")).toHaveValue("Chat with this media: Private source\n\nSummarize")
      expect(handoff.setRagMediaIds).toHaveBeenCalledWith([42])
    } else {
      expect(screen.getByTestId("composer-textarea")).toHaveValue("")
      expect(handoff.setRagMediaIds).not.toHaveBeenCalled()
    }
  })

  it("does not apply a saved source after identity changes while storage resolves", async () => {
    let resolve!: (value: unknown) => void
    handoff.get.mockImplementation((setting) => setting === DISCUSS_MEDIA_PROMPT_SETTING ? new Promise(done => { resolve = done }) : Promise.resolve(undefined))
    const view = render(<PlaygroundForm droppedFiles={[]} />)
    await waitFor(() => expect(resolve).toBeTypeOf("function"))
    const firstResolve = resolve
    handoff.scope = "server-a:bob"
    view.rerender(<PlaygroundForm droppedFiles={[]} />)
    await act(async () => { firstResolve({ ownerScope: "server-a:alice", mediaId: "42", title: "Private source", content: "Summarize" }) })
    expect(screen.getByTestId("composer-textarea")).toHaveValue("")
  })

  it.each(["submitted", "failed", "skipped"])("records normal Chat milestone only for %s and its original owner", async (status) => {
    let resolve!: (value: unknown) => void
    onSubmitMock.mockImplementation(() => new Promise(done => { resolve = done }) as never)
    const user = userEvent.setup()
    const view = render(<PlaygroundForm droppedFiles={[]} />)
    await user.type(screen.getByTestId("composer-textarea"), "Hello")
    await user.click(screen.getAllByRole("button", { name: "Send" })[0])
    await waitFor(() => expect(resolve).toBeTypeOf("function"))
    handoff.scope = "server-a:bob"
    view.rerender(<PlaygroundForm droppedFiles={[]} />)
    await act(async () => { resolve({ status, errorMessage: "Failed", reason: "Skipped" }) })
    expect(useMilestoneStore.getState().scopedMilestones["server-a:alice"]?.first_chat != null).toBe(status === "submitted")
    expect(useMilestoneStore.getState().scopedMilestones["server-a:bob"]?.first_chat).toBeUndefined()
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
