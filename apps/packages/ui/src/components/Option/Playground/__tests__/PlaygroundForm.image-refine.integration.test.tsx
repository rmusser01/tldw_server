// @vitest-environment jsdom
import React from "react"
import { render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { PlaygroundForm } from "../PlaygroundForm"
import type { AttachedResearchContext } from "../research-chat-context"
import { DICTATION_DIAGNOSTICS_EVENT } from "@/utils/dictation-diagnostics"
import { fetchChatModels } from "@/services/tldw-server"

const onSubmitMock = vi.hoisted(() => vi.fn(async (_payload: unknown) => null))
const createChatCompletionMock = vi.hoisted(() =>
  vi.fn(async () => ({
    json: async () => ({
      choices: [
        {
          message: {
            content:
              "Prompt: cinematic portrait of Lana, neon rain, dramatic rim light, 50mm lens"
          }
        }
      ]
    })
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
  lastOptions: null as null | Record<string, any>,
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
const serverCapabilitiesState = vi.hoisted(() => ({
  loading: false,
  capabilities: { hasAudio: false, hasWebSearch: false }
}))
const audioSourceCatalogState = vi.hoisted(() => ({
  devices: [] as Array<{ deviceId: string; label: string }>,
  isSettled: true
}))
const { storageValues, storageLoadingKeys } = vi.hoisted(() => ({
  storageValues: new Map<string, unknown>(),
  storageLoadingKeys: new Set<string>()
}))

const updateStoredValue = (
  key: string,
  defaultValue: unknown,
  nextValue: unknown
) => {
  const currentValue = storageValues.has(key)
    ? storageValues.get(key)
    : defaultValue
  const resolvedValue =
    typeof nextValue === "function"
      ? (nextValue as (current: unknown) => unknown)(currentValue)
      : nextValue
  storageValues.set(key, resolvedValue)
}

const createMessageOptionState = () => ({
  onSubmit: onSubmitMock,
  messages: [
    {
      id: "assistant-1",
      isBot: true,
      message: "Lana stands by the rainy neon alley.",
      moodLabel: "focused"
    }
  ],
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

type SubmitPayload = {
  message?: string
  imageEventSyncPolicy?: string
  imageGenerationRefine?: {
    model?: string
    latencyMs?: number
    diffStats?: Record<string, unknown>
  }
}

const getLastSubmitPayload = (): SubmitPayload => {
  const payload = onSubmitMock.mock.calls.at(-1)?.[0] as SubmitPayload | undefined
  if (!payload) {
    throw new Error("Expected submit payload")
  }
  return payload
}

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
      {Array.isArray(options) &&
        options.map((option: any) => (
          <option key={String(option?.value)} value={String(option?.value || "")}>
            {typeof option?.label === "string"
              ? option.label
              : String(option?.value || "")}
          </option>
        ))}
    </select>
  )
  const RadioButton = ({
    value,
    children,
    onSelect,
    active,
    disabled
  }: {
    value: string
    children: React.ReactNode
    onSelect?: (value: string) => void
    active?: boolean
    disabled?: boolean
  }) => (
    <button
      type="button"
      aria-pressed={active}
      disabled={disabled}
      onClick={() => onSelect?.(value)}
    >
      {children}
    </button>
  )
  const RadioGroup = ({
    children,
    value,
    onChange,
    disabled
  }: {
    children: React.ReactNode
    value?: string
    onChange?: (event: { target: { value: string } }) => void
    disabled?: boolean
  }) => (
    <div>
      {React.Children.map(children, (child: any) => {
        if (!React.isValidElement(child)) {
          return child
        }
        const typedChild = child as React.ReactElement<{
          value?: string
          disabled?: boolean
          onSelect?: (value: string) => void
          active?: boolean
        }>
        return React.cloneElement(typedChild, {
              onSelect: (nextValue: string) =>
                onChange?.({ target: { value: nextValue } }),
              active: typedChild.props?.value === value,
              disabled: disabled || typedChild.props?.disabled
            })
      })}
    </div>
  )
  const Radio = { Group: RadioGroup, Button: RadioButton }

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

  const DropdownComponent = ({ children, menu }: any) => (
    <div>
      {children}
      {Array.isArray(menu?.items)
        ? menu.items.map((item: any) => (
            <button
              key={String(item?.key)}
              type="button"
              onClick={() => item?.onClick?.()}
            >
              {item?.key === "image"
                ? "Image"
                : typeof item?.label === "string"
                  ? item.label
                  : String(item?.key || "menu-item")}
            </button>
          ))
        : null}
    </div>
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
    Checkbox: ({ children }: { children: React.ReactNode }) => <label>{children}</label>,
    Dropdown: DropdownComponent,
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
    Radio,
    Select: SelectComponent,
    Switch: ({
      checked,
      onChange,
      disabled
    }: {
      checked?: boolean
      onChange?: (checked: boolean) => void
      disabled?: boolean
    }) => (
      <input
        type="checkbox"
        checked={Boolean(checked)}
        onChange={(event) => onChange?.(event.target.checked)}
        disabled={disabled}
      />
    ),
    Tooltip: ({ children }: { children: React.ReactNode }) => <>{children}</>,
    Popover: ({
      children,
      content
    }: {
      children: React.ReactNode
      content?: React.ReactNode
    }) => (
      <>
        {children}
        {content}
      </>
    ),
    Modal: ModalComponent,
    Button: ButtonComponent,
    Space: {
      Compact: ({ children }: { children: React.ReactNode }) => <div>{children}</div>
    }
  }
})

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (key: string, defaultValue: unknown) => [
    storageValues.has(key) ? storageValues.get(key) : defaultValue,
    (nextValue: unknown) => updateStoredValue(key, defaultValue, nextValue),
    { isLoading: storageLoadingKeys.has(key) }
  ]
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
  useStoreMessageOption: (
    selector: (state: {
      setRagMediaIds: (...args: unknown[]) => void
      setRagPinnedResults: (...args: unknown[]) => void
    }) => unknown
  ) =>
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
  useServerCapabilities: () => serverCapabilitiesState
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

vi.mock("@/hooks/useAudioSourceCatalog", () => ({
  useAudioSourceCatalog: () => ({
    devices: audioSourceCatalogState.devices,
    isLoading: false,
    isSettled: audioSourceCatalogState.isSettled
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
  useDictationStrategy: (options: Record<string, any>) => {
    dictationStrategyState.lastOptions = options
    return dictationStrategyState.value
  }
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
  useVoiceChatStream: () => ({
    state: "idle"
  })
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
    toolsButton,
    sendControl,
    speechAvailable,
    speechUsesServer,
    isListening,
    isServerDictating,
    onDictationToggle
  }: {
    toolsButton?: React.ReactNode
    sendControl?: React.ReactNode
    speechAvailable?: boolean
    speechUsesServer?: boolean
    isListening?: boolean
    isServerDictating?: boolean
    onDictationToggle?: () => void
  }) => {
    const isActive = speechUsesServer
      ? Boolean(isServerDictating)
      : Boolean(isListening)
    const dictationLabel = !speechAvailable
      ? "Dictation unavailable"
      : isActive
        ? "Stop dictation"
        : "Start dictation"
    return (
      <div data-testid="composer-toolbar">
        {toolsButton}
        {sendControl}
        <button
          type="button"
          data-testid="dictation-button"
          aria-label={dictationLabel}
          onClick={onDictationToggle}
          disabled={!speechAvailable}
        >
          Dictation
        </button>
      </div>
    )
  }
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
    iconOnly,
    variant,
    size,
    shape,
    icon,
    loading,
    disabled,
    className,
    "data-testid": dataTestId,
  }: {
    children?: React.ReactNode
    onClick?: () => void
    ariaLabel?: string
    title?: string
    iconOnly?: boolean
    variant?: string
    size?: string
    shape?: string
    icon?: React.ReactNode
    loading?: boolean
    disabled?: boolean
    className?: string
    "data-testid"?: string
  }) => (
    <button
      type="button"
      onClick={onClick}
      aria-label={ariaLabel}
      title={title}
      disabled={Boolean(disabled) || Boolean(loading)}
      className={className}
      data-testid={dataTestId}
    >
      {icon}
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
      onSubmit: (handler: (values: Record<string, string>) => void | Promise<void>) =>
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
    imageBackendOptions: [
      {
        value: "mock-backend",
        label: "Mock Backend",
        provider: "custom"
      }
    ],
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
  useDeferredComposerInput: (value: string) => ({
    deferredInput: value
  }),
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
    catalogGroups: {
      team: [],
      org: [],
      global: []
    },
    catalogDraft: "",
    setCatalogDraft: vi.fn(),
    commitCatalog: vi.fn()
  })
}))

describe("PlaygroundForm image prompt refinement modal integration", () => {
  beforeEach(() => {
    storageValues.clear()
    onSubmitMock.mockClear()
    createChatCompletionMock.mockClear()
    vi.mocked(fetchChatModels).mockClear()
    playgroundFormMessageOptionState.value = createMessageOptionState()
    playgroundFormConnectionState.phase = "connected"
    playgroundFormConnectionState.isConnected = true
    serverCapabilitiesState.loading = false
    serverCapabilitiesState.capabilities.hasAudio = false
    serverCapabilitiesState.capabilities.hasWebSearch = false
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
    dictationStrategyState.lastOptions = null
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

  it("queues the current draft instead of sending when disconnected", async () => {
    playgroundFormConnectionState.phase = "disconnected"
    playgroundFormConnectionState.isConnected = false

    const user = userEvent.setup()
    render(<PlaygroundForm droppedFiles={[]} />)

    await user.type(
      screen.getByTestId("composer-textarea"),
      "Queue this while offline"
    )
    await user.click(screen.getByRole("button", { name: "Queue request" }))

    expect(onSubmitMock).not.toHaveBeenCalled()
    const queuedUpdate =
      playgroundFormMessageOptionState.value.setQueuedMessages.mock.calls.at(-1)?.[0]
    expect(queuedUpdate).toBeTypeOf("function")
    expect(queuedUpdate([])).toEqual([
      expect.objectContaining({
        promptText: "Queue this while offline",
        message: "Queue this while offline"
      })
    ])
  })

  it("does not force a provider refresh on the first successful send", async () => {
    const user = userEvent.setup()
    render(<PlaygroundForm droppedFiles={[]} />)

    await user.type(screen.getByTestId("composer-textarea"), "hello")
    await user.click(screen.getByRole("button", { name: "Send message" }))

    await waitFor(() => {
      expect(onSubmitMock).toHaveBeenCalled()
    })

    expect(vi.mocked(fetchChatModels)).not.toHaveBeenCalledWith(
      expect.objectContaining({ forceRefresh: true })
    )
    expect(vi.mocked(fetchChatModels)).not.toHaveBeenCalledWith(
      expect.objectContaining({ refreshOpenRouter: true })
    )
  })

  it("supports create -> refine -> accept flow and submits refine metadata", async () => {
    const user = userEvent.setup()
    render(<PlaygroundForm droppedFiles={[]} />)

    await user.click(screen.getByRole("button", { name: "Generate image" }))
    expect(
      await screen.findByRole("dialog", { name: "Generate image" })
    ).toBeInTheDocument()

    await user.click(screen.getByRole("button", { name: "Create prompt" }))
    const promptArea = screen.getByPlaceholderText(
      "Describe the image you want to generate."
    ) as HTMLTextAreaElement
    await waitFor(() => {
      expect(promptArea.value.trim().length).toBeGreaterThan(0)
    })

    await user.click(screen.getByTestId("image-refine-with-llm"))
    expect(await screen.findByTestId("image-prompt-refine-diff")).toBeInTheDocument()
    expect(createChatCompletionMock).toHaveBeenCalledTimes(1)

    await user.click(screen.getByTestId("image-refine-accept"))
    await waitFor(() => {
      expect(screen.queryByTestId("image-prompt-refine-diff")).toBeNull()
    })
    expect(promptArea.value).toContain("cinematic portrait of Lana")

    await user.click(
      within(screen.getByRole("dialog", { name: "Generate image" })).getByRole(
        "button",
        { name: "Generate image" }
      )
    )
    await waitFor(() => {
      expect(onSubmitMock).toHaveBeenCalled()
    })

    const submitPayload = getLastSubmitPayload()
    expect(submitPayload.imageGenerationRefine).toBeTruthy()
    expect(submitPayload.imageEventSyncPolicy).toBe("inherit")
    expect(submitPayload.imageGenerationRefine.model).toBe("deepseek-chat")
    expect(submitPayload.imageGenerationRefine.latencyMs).toBeGreaterThan(0)
    expect(submitPayload.imageGenerationRefine.diffStats).toMatchObject({
      baselineSegments: expect.any(Number),
      candidateSegments: expect.any(Number),
      sharedSegments: expect.any(Number),
      overlapRatio: expect.any(Number),
      addedCount: expect.any(Number),
      removedCount: expect.any(Number)
    })
  })

  it("supports reject flow and keeps original draft without refine metadata", async () => {
    const user = userEvent.setup()
    render(<PlaygroundForm droppedFiles={[]} />)

    await user.click(screen.getByRole("button", { name: "Generate image" }))
    await user.click(screen.getByRole("button", { name: "Create prompt" }))

    const promptArea = screen.getByPlaceholderText(
      "Describe the image you want to generate."
    ) as HTMLTextAreaElement
    await waitFor(() => {
      expect(promptArea.value.trim().length).toBeGreaterThan(0)
    })
    const originalPrompt = promptArea.value

    await user.click(screen.getByTestId("image-refine-with-llm"))
    expect(await screen.findByTestId("image-prompt-refine-diff")).toBeInTheDocument()

    await user.click(screen.getByRole("button", { name: "Keep original" }))
    await waitFor(() => {
      expect(screen.queryByTestId("image-prompt-refine-diff")).toBeNull()
    })
    expect(promptArea.value).toBe(originalPrompt)

    await user.click(
      within(screen.getByRole("dialog", { name: "Generate image" })).getByRole(
        "button",
        { name: "Generate image" }
      )
    )
    await waitFor(() => {
      expect(onSubmitMock).toHaveBeenCalled()
    })

    const submitPayload = getLastSubmitPayload()
    expect(submitPayload.imageGenerationRefine).toBeUndefined()
  })

  it("shows and applies attached research context edits in the raw request preview", async () => {
    const user = userEvent.setup()

    const PreviewHarness = () => {
      const [attachedContext, setAttachedContext] = React.useState<AttachedResearchContext>(
        buildAttachedResearchContext() as AttachedResearchContext
      )
      const [baselineContext] = React.useState<AttachedResearchContext>(
        buildAttachedResearchContext() as AttachedResearchContext
      )

      return (
        <PlaygroundForm
          droppedFiles={[]}
          attachedResearchContext={attachedContext}
          attachedResearchContextBaseline={baselineContext}
          onApplyAttachedResearchContext={(next) => setAttachedContext(next)}
          onResetAttachedResearchContext={() =>
            setAttachedContext(baselineContext)
          }
        />
      )
    }

    render(<PreviewHarness />)

    await user.click(screen.getByRole("button", { name: "Edit attached research" }))
    expect(
      await screen.findByRole("dialog", { name: "Current chat request JSON" })
    ).toBeInTheDocument()
    expect(screen.getByTestId("attached-research-context-panel")).toBeInTheDocument()

    const questionInput = screen.getByTestId(
      "attached-research-context-question-input"
    ) as HTMLInputElement
    expect(questionInput.value).toBe("Battery recycling supply chain")

    const rawJson = screen.getByTestId("raw-chat-request-json") as HTMLTextAreaElement
    await waitFor(() => {
      expect(rawJson.value).toContain('"run_id": "run_123"')
    })

    await user.clear(questionInput)
    await user.type(questionInput, "Edited attached question")
    await user.click(screen.getByRole("button", { name: "Apply" }))
    await user.click(screen.getByRole("button", { name: "Refresh" }))

    await waitFor(() => {
      expect(
        (screen.getByTestId(
          "attached-research-context-question-input"
        ) as HTMLInputElement).value
      ).toBe("Edited attached question")
      expect(
        (screen.getByTestId("raw-chat-request-json") as HTMLTextAreaElement).value
      ).toContain('"question": "Edited attached question"')
    })

    await user.click(screen.getByRole("button", { name: "Reset to Attached Run" }))
    await user.click(screen.getByRole("button", { name: "Refresh" }))

    await waitFor(() => {
      expect(
        (screen.getByTestId(
          "attached-research-context-question-input"
        ) as HTMLInputElement).value
      ).toBe("Battery recycling supply chain")
      expect(
        (screen.getByTestId("raw-chat-request-json") as HTMLTextAreaElement).value
      ).toContain('"question": "Battery recycling supply chain"')
    })
  })

  it("clears refine candidate when prompt is manually edited before submit", async () => {
    const user = userEvent.setup()
    render(<PlaygroundForm droppedFiles={[]} />)

    await user.click(screen.getByRole("button", { name: "Generate image" }))
    await user.click(screen.getByRole("button", { name: "Create prompt" }))
    await user.click(screen.getByTestId("image-refine-with-llm"))
    expect(await screen.findByTestId("image-prompt-refine-diff")).toBeInTheDocument()

    const promptArea = screen.getByPlaceholderText(
      "Describe the image you want to generate."
    ) as HTMLTextAreaElement
    await user.clear(promptArea)
    await user.type(promptArea, "manual prompt override from user edit")

    await waitFor(() => {
      expect(screen.queryByTestId("image-prompt-refine-diff")).toBeNull()
    })

    await user.click(
      within(screen.getByRole("dialog", { name: "Generate image" })).getByRole(
        "button",
        { name: "Generate image" }
      )
    )
    await waitFor(() => {
      expect(onSubmitMock).toHaveBeenCalled()
    })

    const submitPayload = getLastSubmitPayload()
    expect(submitPayload.message).toBe("manual prompt override from user edit")
    expect(submitPayload.imageGenerationRefine).toBeUndefined()
  })
})

describe("PlaygroundForm dictation integration", () => {
  beforeEach(() => {
    storageValues.clear()
    storageLoadingKeys.clear()
    speechRecognitionState.transcript = ""
    speechRecognitionState.isListening = false
    speechRecognitionState.supported = true
    speechRecognitionState.start.mockClear()
    speechRecognitionState.stop.mockClear()
    speechRecognitionState.resetTranscript.mockClear()
    serverDictationState.isServerDictating = false
    serverDictationState.startServerDictation.mockClear()
    serverDictationState.stopServerDictation.mockClear()
    serverDictationState.lastOptions = null
    dictationStrategyState.lastOptions = null
    serverCapabilitiesState.loading = false
    serverCapabilitiesState.capabilities.hasAudio = false
    serverCapabilitiesState.capabilities.hasWebSearch = false
    audioSourceCatalogState.devices = [
      { deviceId: "default", label: "Default microphone" },
      { deviceId: "usb-1", label: "USB microphone" },
      { deviceId: "usb-2", label: "USB microphone 2" },
      { deviceId: "usb-3", label: "USB microphone 3" }
    ]
    audioSourceCatalogState.isSettled = true
  })

  it("routes server dictation intent through the shared toggle handler", async () => {
    const user = userEvent.setup()
    storageValues.set("dictationAudioSourcePreference", {
      featureGroup: "dictation",
      sourceKind: "default_mic",
      deviceId: null,
      lastKnownLabel: null
    })
    dictationStrategyState.value = {
      requestedMode: "auto",
      resolvedMode: "server",
      speechAvailable: true,
      speechUsesServer: true,
      isDictating: false,
      toggleIntent: "start_server",
      autoFallbackActive: false,
      autoFallbackErrorClass: null,
      recordServerError: vi.fn(() => ({
        errorClass: "unknown_error",
        appliedFallback: false
      })),
      recordServerSuccess: vi.fn(),
      clearAutoFallback: vi.fn()
    }

    render(<PlaygroundForm droppedFiles={[]} />)
    await user.click(screen.getByTestId("dictation-button"))

    expect(serverDictationState.startServerDictation).toHaveBeenCalledTimes(1)
    expect(speechRecognitionState.start).not.toHaveBeenCalled()
  })

  it("routes explicit mic dictation through server capture with the selected device", async () => {
    const user = userEvent.setup()
    storageValues.set("dictationAudioSourcePreference", {
      featureGroup: "dictation",
      sourceKind: "mic_device",
      deviceId: "usb-1",
      lastKnownLabel: "USB microphone"
    })
    dictationStrategyState.value = {
      requestedMode: "auto",
      resolvedMode: "server",
      speechAvailable: true,
      speechUsesServer: true,
      isDictating: false,
      toggleIntent: "start_server",
      autoFallbackActive: false,
      autoFallbackErrorClass: null,
      recordServerError: vi.fn(() => ({
        errorClass: "unknown_error",
        appliedFallback: false
      })),
      recordServerSuccess: vi.fn(),
      clearAutoFallback: vi.fn()
    }

    render(<PlaygroundForm droppedFiles={[]} />)
    await user.click(screen.getByTestId("dictation-button"))

    expect(speechRecognitionState.start).not.toHaveBeenCalled()
    expect(serverDictationState.startServerDictation).toHaveBeenCalledWith({
      featureGroup: "dictation",
      sourceKind: "mic_device",
      deviceId: "usb-1",
      lastKnownLabel: "USB microphone"
    })
  })

  it("waits for dictation source hydration before starting with a remembered mic", async () => {
    const user = userEvent.setup()
    storageValues.set("dictationAudioSourcePreference", {
      featureGroup: "dictation",
      sourceKind: "mic_device",
      deviceId: "usb-1",
      lastKnownLabel: "USB microphone"
    })
    storageLoadingKeys.add("dictationAudioSourcePreference")
    dictationStrategyState.value = {
      requestedMode: "auto",
      resolvedMode: "server",
      speechAvailable: true,
      speechUsesServer: true,
      isDictating: false,
      toggleIntent: "start_server",
      autoFallbackActive: false,
      autoFallbackErrorClass: null,
      recordServerError: vi.fn(() => ({
        errorClass: "unknown_error",
        appliedFallback: false
      })),
      recordServerSuccess: vi.fn(),
      clearAutoFallback: vi.fn()
    }

    const { rerender } = render(<PlaygroundForm droppedFiles={[]} />)
    await user.click(screen.getByTestId("dictation-button"))

    expect(serverDictationState.startServerDictation).not.toHaveBeenCalled()

    storageLoadingKeys.delete("dictationAudioSourcePreference")
    rerender(<PlaygroundForm droppedFiles={[]} />)

    await waitFor(() => {
      expect(serverDictationState.startServerDictation).toHaveBeenCalledWith({
        featureGroup: "dictation",
        sourceKind: "mic_device",
        deviceId: "usb-1",
        lastKnownLabel: "USB microphone"
      })
    })
  })

  it("falls back to the default mic when the remembered dictation device is missing after catalog settle", async () => {
    const user = userEvent.setup()
    storageValues.set("dictationAudioSourcePreference", {
      featureGroup: "dictation",
      sourceKind: "mic_device",
      deviceId: "usb-missing",
      lastKnownLabel: "Studio microphone"
    })
    audioSourceCatalogState.devices = [
      { deviceId: "default", label: "Default microphone" }
    ]
    dictationStrategyState.value = {
      requestedMode: "auto",
      resolvedMode: "server",
      speechAvailable: true,
      speechUsesServer: true,
      isDictating: false,
      toggleIntent: "start_server",
      autoFallbackActive: false,
      autoFallbackErrorClass: null,
      recordServerError: vi.fn(() => ({
        errorClass: "unknown_error",
        appliedFallback: false
      })),
      recordServerSuccess: vi.fn(),
      clearAutoFallback: vi.fn()
    }

    render(<PlaygroundForm droppedFiles={[]} />)
    await user.click(screen.getByTestId("dictation-button"))

    expect(dictationStrategyState.lastOptions).toMatchObject({
      browserDictationCompatible: true
    })
    expect(serverDictationState.startServerDictation).toHaveBeenCalledWith(undefined)
  })

  it("keeps auto dictation browser-compatible for the default mic when server STT is unavailable", async () => {
    storageValues.set("dictationAudioSourcePreference", {
      featureGroup: "dictation",
      sourceKind: "default_mic",
      deviceId: null,
      lastKnownLabel: null
    })
    speechRecognitionState.supported = true
    serverCapabilitiesState.capabilities.hasAudio = false

    render(<PlaygroundForm droppedFiles={[]} />)

    expect(dictationStrategyState.lastOptions).toMatchObject({
      browserDictationCompatible: true,
      resolvedModeOverride: null
    })
  })

  it("forces browser dictation unavailable for a mic device when server STT is unavailable", async () => {
    storageValues.set("dictationModeOverride", "browser")
    storageValues.set("dictationAudioSourcePreference", {
      featureGroup: "dictation",
      sourceKind: "mic_device",
      deviceId: "usb-3",
      lastKnownLabel: "USB microphone 3"
    })
    serverCapabilitiesState.capabilities.hasAudio = false

    render(<PlaygroundForm droppedFiles={[]} />)

    expect(dictationStrategyState.lastOptions).toMatchObject({
      browserDictationCompatible: false,
      resolvedModeOverride: "unavailable"
    })
  })

  it("emits server-error diagnostics for the updated dictation source after rerender", async () => {
    const user = userEvent.setup()
    const diagnosticsEvents: any[] = []
    const handleDiagnostics = (event: Event) => {
      diagnosticsEvents.push((event as CustomEvent).detail)
    }

    storageValues.set("dictationAudioSourcePreference", {
      featureGroup: "dictation",
      sourceKind: "default_mic",
      deviceId: null,
      lastKnownLabel: null
    })
    dictationStrategyState.value = {
      requestedMode: "auto",
      resolvedMode: "server",
      speechAvailable: true,
      speechUsesServer: true,
      isDictating: false,
      toggleIntent: "start_server",
      autoFallbackActive: false,
      autoFallbackErrorClass: null,
      recordServerError: vi.fn(() => ({
        errorClass: "provider_unavailable",
        appliedFallback: false
      })),
      recordServerSuccess: vi.fn(),
      clearAutoFallback: vi.fn()
    }
    serverDictationState.startServerDictation.mockImplementationOnce(async () => {
      serverDictationState.lastOptions?.onError?.({
        details: {
          detail: {
            dictation_error_class: "provider_unavailable",
            status: "provider_unavailable",
            message: "server unavailable"
          }
        }
      })
    })

    window.addEventListener(DICTATION_DIAGNOSTICS_EVENT, handleDiagnostics)
    try {
      const { rerender } = render(<PlaygroundForm droppedFiles={[]} />)
      storageValues.set("dictationAudioSourcePreference", {
        featureGroup: "dictation",
        sourceKind: "mic_device",
        deviceId: "usb-2",
        lastKnownLabel: "USB microphone 2"
      })
      rerender(<PlaygroundForm droppedFiles={[]} />)

      await user.click(screen.getByTestId("dictation-button"))

      const serverErrorEvent = diagnosticsEvents.find(
        (event) => event.kind === "server_error"
      )
      expect(serverErrorEvent).toBeTruthy()
      expect(serverErrorEvent.requested_source_kind).toBe("mic_device")
      expect(serverErrorEvent.resolved_source_kind).toBe("mic_device")
    } finally {
      window.removeEventListener(DICTATION_DIAGNOSTICS_EVENT, handleDiagnostics)
    }
  })

  it("routes browser dictation intent through speech recognition start", async () => {
    const user = userEvent.setup()
    storageValues.set("dictationAudioSourcePreference", {
      featureGroup: "dictation",
      sourceKind: "default_mic",
      deviceId: null,
      lastKnownLabel: null
    })
    dictationStrategyState.value = {
      requestedMode: "auto",
      resolvedMode: "browser",
      speechAvailable: true,
      speechUsesServer: false,
      isDictating: false,
      toggleIntent: "start_browser",
      autoFallbackActive: false,
      autoFallbackErrorClass: null,
      recordServerError: vi.fn(() => ({
        errorClass: "unknown_error",
        appliedFallback: false
      })),
      recordServerSuccess: vi.fn(),
      clearAutoFallback: vi.fn()
    }

    render(<PlaygroundForm droppedFiles={[]} />)
    await user.click(screen.getByTestId("dictation-button"))

    expect(speechRecognitionState.start).toHaveBeenCalledTimes(1)
    expect(speechRecognitionState.start).toHaveBeenCalledWith(
      expect.objectContaining({
        continuous: true,
        lang: "en-US"
      })
    )
    expect(serverDictationState.startServerDictation).not.toHaveBeenCalled()
  })

  it("writes transcript text into the composer when server dictation resolves", async () => {
    dictationStrategyState.value = {
      requestedMode: "auto",
      resolvedMode: "server",
      speechAvailable: true,
      speechUsesServer: true,
      isDictating: false,
      toggleIntent: "start_server",
      autoFallbackActive: false,
      autoFallbackErrorClass: null,
      recordServerError: vi.fn(() => ({
        errorClass: "unknown_error",
        appliedFallback: false
      })),
      recordServerSuccess: vi.fn(),
      clearAutoFallback: vi.fn()
    }

    render(<PlaygroundForm droppedFiles={[]} />)
    await waitFor(() => {
      expect(serverDictationState.lastOptions).toBeTruthy()
    })

    serverDictationState.lastOptions?.onTranscript?.("dictation transcript text")

    const textarea = screen.getByTestId("composer-textarea") as HTMLTextAreaElement
    await waitFor(() => {
      expect(textarea.value).toBe("dictation transcript text")
    })
  })
})
