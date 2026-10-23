// @vitest-environment jsdom
import React from "react"
import { BuddyManagementButton } from "@/components/Common/PersonaBuddy/BuddyManagementButton"
import { resolveEffectiveAssistantState } from "@/hooks/chat/effective-assistant-state"
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { ConfigProvider } from "antd"
import { openModelSelector } from "../playground-cockpit-actions"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { AssistantSelect } from "@/components/Common/AssistantSelect"
import { PlaygroundEmpty } from "../PlaygroundEmpty"
import { PlaygroundForm, type ApplyCharacterRoute } from "../PlaygroundForm"
import type { Character } from "@/types/character"
import type { AssistantSelection } from "@/types/assistant-selection"
import type {
  RolePlayApplyContext,
  RolePlaySetupApplyPayload
} from "../RolePlaySetupDrawer"
import {
  CHAT_ROUTE_REPLACEMENT_EVENT,
  type ChatRouteReplacementDetail
} from "@/utils/character-chat-mode-intent"
import { resolveServerChatAssistantIdentity } from "@/hooks/chat/useServerChatLoader"
import {
  createStartupTemplateBundle,
  type StartupTemplateBundle
} from "../startup-template-bundles"
import { usePlaygroundSessionStore } from "@/store/playground-session"
import { useBuddyManagementStore } from "@/store/buddy-management"
import { useActorStore } from "@/store/actor"
import { buildQueuedRequest } from "@/utils/chat-request-queue"
import { createDefaultActorSettings, type ActorSettings } from "@/types/actor"
import type { ActorSceneRecord } from "@/services/actor-settings"
import type {
  HistoryOwnerV1,
  NativeHistoryOwnerV1
} from "@/services/chat-history-selection"
import type { HistoryViewSelectionV1 } from "@/types/history-selection"
import type { ServicePromptRequestScope } from "@/services/tldw/domains/service-prompts"
import type { State as MessageOptionState } from "@/store/option/types"
import {
  captureActorSettingsTarget,
  type ActorSettingsTarget
} from "@/utils/actor-settings-target"
import {
  getLocalHistoryOwner,
  type LocalHistoryOwnerV1
} from "@/db/dexie/history-selection"
import { installSettingsLocks } from "@/services/__tests__/chat-settings-fixture"
import { summarizeRolePlayScene } from "../role-play-scene"
import {
  chatDraftAddressKey,
  freshChatDraftAddress,
  type ChatDraftAddress
} from "@/utils/chat-draft-address"
import { resolveModelSelectorSelection } from "@/hooks/playground/modelSelectorUtils"
import type { UseModelSelectorParams } from "@/hooks/playground/useModelSelector"

// Local sources use the owner returned for acknowledged seeded rows. Retained
// destinations supply their complete owner/view/account settings locator; the
// controller double does not issue generation or history-load capabilities.
type SettingsSelectionState = {
  status: string
  owner: HistoryOwnerV1 | null
  view: HistoryViewSelectionV1 | null
}

// Declared local sources have acknowledged rows/profile; only the table engine is doubled.
vi.mock("@/db/dexie/schema", async () => ({ db: (await import("@/hooks/chat/__tests__/local-history-fixture")).memory }))
import { seedLocalFork } from "@/hooks/chat/__tests__/local-history-fixture"

const queuedDocumentPreparation = vi.hoisted(() => vi.fn())
vi.mock("@/services/chat-document-processing", async original => ({
  ...await original<typeof import("@/services/chat-document-processing")>(),
  prepareChatDocumentAttachmentsForSend: queuedDocumentPreparation
}))

const saveActorSettingsMock = vi.hoisted(() =>
  vi.fn<typeof import("@/services/actor-settings").saveActorSettingsForChat>()
)
const ownerGuardFixture = vi.hoisted(() => ({ capture: null as null | (() => () => boolean) }))
type ModelSettingsStore = ReturnType<typeof import("@/store/model").useStoreChatModelSettings.getState>
const nativeSetupFixture = vi.hoisted(() => ({
  realSettings: false,
  realModelPicker: false,
  realDrawer: false,
  realRolePlayPresets: false,
  repeatActionStarted: false,
  actionAcknowledgements: [] as boolean[],
  drawerApplications: [] as Promise<boolean | void>[],
  drawerProps: null as React.ComponentProps<
    typeof import("../RolePlaySetupDrawer").RolePlaySetupDrawer
  > | null,
  realDrafts: false,
  ownerCurrent: true,
  destinationCurrent: null as null | (() => boolean),
  actualSettingsStore: null as
    | typeof import("@/store/model").useStoreChatModelSettings
    | null,
  formattingGuide: false,
  identityGuard: null as null | (() => boolean),
  assistantRevision: 0,
  retain: vi.fn(async (_key: string, _draft: unknown) => true),
  getCharacter: vi.fn(
    async (_id: string, _options?: unknown): Promise<unknown> => ({
      id: "route-card",
      name: "Route Card"
    })
  ),
  error: vi.fn(),
  models: [
    { model: "deepseek-chat", provider: "openai", is_configured: true },
    { model: "gpt-4.1", provider: "openai", is_configured: true }
  ],
  controller: null as null | {
    getCurrent: () => SettingsSelectionState
    getSignal: () => AbortSignal
    settingsMode: (chatId: string | null) => "ordinary" | "pending"
    reset: () => void
    fence: () => () => boolean
    draftAccount: {
      requestScope: ServicePromptRequestScope
      isCurrent: () => boolean
    }
  },
  onDraftPublication: null as null | ((address: ChatDraftAddress) => void),
  selectionApply: null as Promise<boolean | void> | null,
  scene:
    vi.fn<typeof import("@/services/actor-settings").getActorSceneForChat>(),
  actualSceneReader: null as
    | typeof import("@/services/actor-settings").getActorSceneForChat
    | null,
  actualSceneWriter: null as
    | typeof import("@/services/actor-settings").saveActorSettingsForChat
    | null,
  localOwner: null as LocalHistoryOwnerV1 | null,
  settings: null as ModelSettingsStore | null,
  fallbackSettings: null as ModelSettingsStore | null,
  tools: { tools: [] as unknown[] },
  actor:
    vi.fn<typeof import("@/services/actor-settings").getActorSettingsForChat>(),
  snapshot: vi.fn<
    () => Promise<{
      scopeSignal: AbortSignal
      scopeInvalidatedSignal: AbortSignal
      requestScope: ServicePromptRequestScope
      release: () => void
    }>
  >(async () => ({
    scopeSignal: new AbortController().signal,
    scopeInvalidatedSignal: new AbortController().signal,
    requestScope: {
      config: { serverUrl: "http://localhost:8000", authMode: "single-user" },
      userId: null
    },
    release: vi.fn()
  }))
}))
const assistantOwnerIsCurrent = vi.hoisted(() => () => nativeSetupFixture.ownerCurrent)
vi.mock("@/hooks/chat/useHistorySelection", async importOriginal => ({
  ...await importOriginal<typeof import("@/hooks/chat/useHistorySelection")>(),
  useHistorySelectionContext: () => nativeSetupFixture.controller
}))
vi.mock("@/store/mcp-tools", () => ({ useMcpToolsStore: { getState: () => nativeSetupFixture.tools } }))
vi.mock("@/services/service-prompts", async importOriginal => ({
  ...await importOriginal<typeof import("@/services/service-prompts")>(),
  loadServicePromptSnapshot: nativeSetupFixture.snapshot
}))
const savedSetupFixture = vi.hoisted(() => ({ raw: "[]" }))
const startupPreviewFixture = vi.hoisted(() => ({
  apply: null as null | (() => Promise<unknown>),
  template: null as StartupTemplateBundle | null
}))
const viewportState = vi.hoisted(() => ({ mobile: false, mode: "pro", realToolbar: false }))
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
  isLoading: false,
  defaultCharacter: null as Character | null,
  ownedDefaultCharacter: null as Character | null,
  initialSelection: null as any,
  currentSelection: undefined as AssistantSelection | null | undefined,
  externalMirror: null as null | ((value: AssistantSelection | null) => void),
  setSelectedAssistant: vi.fn(async (_next: unknown): Promise<boolean | void> => undefined)
}))
const selectedCharacterState = vi.hoisted(() => ({
  value: null as any,
  setSelectedCharacter: vi.fn(async (_next: unknown) => undefined)
}))
const chatModelSettingsState = vi.hoisted(() => ({
  systemPrompt: "",
  setSystemPrompt: vi.fn<ModelSettingsStore["setSystemPrompt"]>(),
  updateSetting: vi.fn(),
  updateSettings: vi.fn()
}))

const createMessageOptionState = () => ({
  chatDraftAddress: {
    kind: "fresh",
    id: "form-fixture",
    scope: { type: "global" },
    temporary: false
  } satisfies ChatDraftAddress,
  chatDraftRevision: 0,
  chatDraftTransition: null,
  startNewChatDraft: vi.fn((temporary?: boolean) => {
    const state = playgroundFormMessageOptionState.value
    state.setChatDraftAddress(
      freshChatDraftAddress(temporary ?? state.temporaryChat)
    )
  }),
  // Declared setter follows the real post-subscriber acceptance contract. The
  // compact companion exercises that same boundary through the actual store.
  setChatDraftAddress: vi.fn<MessageOptionState["setChatDraftAddress"]>(
    (address, options = {}) => {
      const state = playgroundFormMessageOptionState.value
      if (
        options.sourceAddress !== undefined &&
        options.sourceAddress !== state.chatDraftAddress
      )
        return false
      if (
        options.expectedRevision !== undefined &&
        options.expectedRevision !== state.chatDraftRevision
      )
        return false
      let acceptedAddress = address
      if (
        state.chatDraftAddress &&
        chatDraftAddressKey(state.chatDraftAddress) ===
          chatDraftAddressKey(address)
      )
        acceptedAddress = state.chatDraftAddress
      else {
        const previousAddress = state.chatDraftAddress
        state.chatDraftAddress = address
        state.chatDraftRevision++
        if (previousAddress && options.kind !== "bind") {
          state.selectedQuickPrompt = null
          state.quickPromptRevision++
        }
        state.temporaryChat = address.kind === "fresh" && address.temporary
        state.chatDraftTransition = {
          kind: options.kind ?? "switch",
          sourceAddress: previousAddress,
          destinationKey: chatDraftAddressKey(address),
          revision: state.chatDraftRevision
        }
      }
      const acceptedRevision = state.chatDraftRevision
      nativeSetupFixture.onDraftPublication?.(acceptedAddress)
      return (
        state.chatDraftAddress === acceptedAddress &&
        state.chatDraftRevision === acceptedRevision
      )
    }
  ),
  onSubmit: onSubmitMock,
  messages: [],
  selectedModel: "deepseek-chat",
  selectedModelRevision: 0,
  selectedModelIsLoading: false,
  setSelectedModel: vi.fn<MessageOptionState["setSelectedModel"]>(
    (model, options) => {
      const state = playgroundFormMessageOptionState.value
      state.selectedModel = model
      state.selectedModelRevision++
      state.selectedModelStorageHandled = options?.storageHandled === true
    }
  ),
  beginSelectedModelSave: vi.fn((model: string) => {
    // Declared Form model boundary; the actual device lease is checked separately.
    const source = playgroundFormMessageOptionState.value
    source.setSelectedModel(model)
    const revision = source.selectedModelRevision
    const isCurrent = () =>
      source.selectedModel === model &&
      source.selectedModelRevision === revision
    return {
      completion: Promise.resolve(isCurrent()),
      isCurrent,
      retry: async () => isCurrent()
    }
  }),
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
  quickPromptRevision: 0,
  textareaRef: { current: null },
  setSelectedQuickPrompt: vi.fn<MessageOptionState["setSelectedQuickPrompt"]>(
    (prompt, options) => {
      const state = playgroundFormMessageOptionState.value
      if (
        options?.expectedRevision !== undefined &&
        options.expectedRevision !== state.quickPromptRevision
      )
        return false
      state.selectedQuickPrompt = prompt || null
      state.quickPromptRevision++
      return true
    }
  ),
  selectedSystemPrompt: null,
  setSelectedSystemPrompt: vi.fn<(prompt: string | null | undefined) => void>(
    (prompt) => {
      playgroundFormMessageOptionState.value.selectedSystemPrompt = prompt
    }
  ),
  temporaryChat: false,
  setTemporaryChat: vi.fn<MessageOptionState["setTemporaryChat"]>(
    (temporary) => {
      const state = playgroundFormMessageOptionState.value
      if (state.temporaryChat !== temporary) {
        state.chatDraftRevision++
        state.chatDraftTransition = null
        if (state.chatDraftAddress?.kind === "fresh")
          state.chatDraftAddress = { ...state.chatDraftAddress, temporary }
      }
      state.temporaryChat = temporary
    }
  ),
  clearChat: vi.fn(),
  useOCR: false,
  setUseOCR: vi.fn(),
  defaultInternetSearchOn: false,
  setHistory: vi.fn<MessageOptionState["setHistory"]>((next) => {
    const state = playgroundFormMessageOptionState.value
    state.history = typeof next === "function" ? next(state.history) : next
  }),
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
  historyId: "history-1",
  serverChatAssistantKind: "character",
  serverChatAssistantId: "char-original",
  serverChatCharacterId: "char-original",
  setHistoryId: vi.fn<MessageOptionState["setHistoryId"]>((id, options) => {
    const state = playgroundFormMessageOptionState.value
    state.historyId = id
    if (!options?.preserveServerChatId && id)
      Object.assign(state, {
        serverChatId: null,
        serverChatState: null,
        serverChatVersion: null,
        serverChatTitle: null,
        serverChatCharacterId: null,
        serverChatMetaLoaded: false,
        serverChatLoadState: "idle",
        serverChatLoadError: null,
        serverChatTopic: null,
        serverChatClusterId: null,
        serverChatSource: null,
        serverChatExternalRef: null
      })
  }),
  setMessages: vi.fn<MessageOptionState["setMessages"]>((next) => {
    const state = playgroundFormMessageOptionState.value
    state.messages = typeof next === "function" ? next(state.messages) : next
  }),
  setServerChatCharacterId: vi.fn<
    MessageOptionState["setServerChatCharacterId"]
  >((id) => {
    playgroundFormMessageOptionState.value.serverChatCharacterId = id ?? null
  }),
  setServerChatAssistantKind: vi.fn<
    MessageOptionState["setServerChatAssistantKind"]
  >((kind) => {
    playgroundFormMessageOptionState.value.serverChatAssistantKind =
      kind ?? null
  }),
  setServerChatAssistantId: vi.fn<
    MessageOptionState["setServerChatAssistantId"]
  >((id) => {
    playgroundFormMessageOptionState.value.serverChatAssistantId = id ?? null
  }),
  setServerChatPersonaMemoryMode: vi.fn<
    MessageOptionState["setServerChatPersonaMemoryMode"]
  >((mode) => {
    playgroundFormMessageOptionState.value.serverChatPersonaMemoryMode =
      mode ?? null
  }),
  setServerChatMetaLoaded: vi.fn<MessageOptionState["setServerChatMetaLoaded"]>(
    (loaded) => {
      playgroundFormMessageOptionState.value.serverChatMetaLoaded = loaded
    }
  ),
  setServerChatId: vi.fn<MessageOptionState["setServerChatId"]>((id) => {
    Object.assign(playgroundFormMessageOptionState.value, {
      serverChatId: id,
      serverChatState: id ? "in-progress" : null,
      serverChatVersion: null,
      serverChatTitle: null,
      serverChatCharacterId: null,
      serverChatAssistantKind: null,
      serverChatAssistantId: null,
      serverChatPersonaMemoryMode: null,
      serverChatMetaLoaded: false,
      serverChatLoadState: id ? "loading" : "idle",
      serverChatLoadError: null,
      serverChatTopic: null,
      serverChatClusterId: null,
      serverChatSource: null,
      serverChatExternalRef: null
    })
  }),
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

vi.mock("@/hooks/useChatDraftOwner", () => {
  const owner = {
    ownerKey: "a".repeat(64),
    isCurrent: () => nativeSetupFixture.ownerCurrent
  }
  return { useChatDraftOwner: () => owner }
})

const modelCatalogQueryOptions = vi.hoisted(() => ({ enabled: [] as boolean[], loading: false, imageCommand: false }))
vi.mock("@tanstack/react-query", () => ({
  useQuery: ({ queryKey, enabled }: { queryKey: string[]; enabled?: boolean }) => {
    if (queryKey[0] === "playground:chatModels") modelCatalogQueryOptions.enabled.push(Boolean(enabled))
    return { data: queryKey[0] === "playground:chatModels" && nativeSetupFixture.settings
      ? nativeSetupFixture.models : [], isSuccess: !modelCatalogQueryOptions.loading, isFetching: queryKey[0] === "playground:chatModels" && modelCatalogQueryOptions.loading, isError: false, refetch: vi.fn() }
  },
  useQueryClient: () => ({ invalidateQueries: vi.fn(), getQueryState: vi.fn(() => undefined) }),
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

vi.mock("antd", async (original) => {
  const actual = await original<typeof import("antd")>()
  const React = await import("react")

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
    "data-testid": dataTestId,
    "aria-label": ariaLabel
  }: any) => (
    <textarea
      value={value ?? ""}
      onChange={(event) => onChange?.(event)}
      placeholder={placeholder}
      readOnly={readOnly}
      disabled={disabled}
      className={className}
      data-testid={dataTestId}
      aria-label={ariaLabel}
    />
  )

  const Dropdown = ({ open, onOpenChange, popupRender, children }: any) => {
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
      aria-label={ariaLabel}>
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
      <div
        role="dialog"
        aria-label={typeof title === "string" ? title : "modal"}>
        {title ? <h2>{title}</h2> : null}
        <div>{children}</div>
        {footer ? <div>{footer}</div> : null}
      </div>
    ) : null
  ModalComponent.confirm = vi.fn()
  ModalComponent.useModal = actual.Modal.useModal

  return {
    ConfigProvider: actual.ConfigProvider,
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
    Dropdown: (props: React.ComponentProps<typeof actual.Dropdown>) =>
      nativeSetupFixture.realModelPicker ? (
        <actual.Dropdown {...props} />
      ) : (
        <Dropdown {...props} />
      ),
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
    Drawer: ({
      open,
      title,
      children,
      footer
    }: React.ComponentProps<typeof actual.Drawer>) =>
      open ? (
        <section
          role="dialog"
          aria-label={typeof title === "string" ? title : "Drawer"}>
          {title}
          {children}
          {footer}
        </section>
      ) : null,
    Skeleton: actual.Skeleton,
    Popover: ({ children }: { children: React.ReactNode }) => <>{children}</>,
    Radio: {
      Group: ({ children }: { children: React.ReactNode }) => (
        <div>{children}</div>
      ),
      Button: ({ children }: { children: React.ReactNode }) => (
        <button type="button">{children}</button>
      )
    },
    Select: ({ value, options = [], onChange, disabled, ...rest }: any) => (
      <select
        value={value ?? ""}
        onChange={(event) => onChange?.(event.target.value)}
        disabled={disabled}
        {...rest}>
        {Array.isArray(options)
          ? options.map((option: any) => (
              <option
                key={String(option?.value)}
                value={String(option?.value || "")}>
                {typeof option?.label === "string"
                  ? option.label
                  : String(option?.value || "")}
              </option>
            ))
          : null}
      </select>
    ),
    Space: {
      Compact: ({ children }: { children: React.ReactNode }) => (
        <div>{children}</div>
      )
    },
    Switch: ({ checked, onChange, disabled }: any) => (
      <input
        type="checkbox"
        checked={Boolean(checked)}
        onChange={(event) => onChange?.(event.target.checked)}
        disabled={disabled}
      />
    ),
    Tooltip: (props: React.ComponentProps<typeof actual.Tooltip>) =>
      nativeSetupFixture.realModelPicker ? (
        <actual.Tooltip {...props} />
      ) : (
        <>{props.children}</>
      )
  }
})

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (key: unknown, defaultValue: unknown) =>
    React.useState(
      key === "playgroundAppendFormattingGuidePrompt"
        ? nativeSetupFixture.formattingGuide
        : key === "playgroundStartupTemplateBundles"
          ? savedSetupFixture.raw
          : (key as { key?: string })?.key ===
              `defaultCharacterSelection:owner:${"a".repeat(64)}`
            ? {
                ownerKey: "a".repeat(64),
                selection: selectedAssistantMock.ownedDefaultCharacter
              }
            : (key as { key?: string })?.key === "defaultCharacterSelection"
              ? selectedAssistantMock.defaultCharacter
              : defaultValue
    )
}))

vi.mock("react-router-dom", () => ({
  Link: ({ children, to, ...rest }: any) => (
    <a href={typeof to === "string" ? to : "#"} {...rest}>
      {children}
    </a>
  ),
  useNavigate: () => vi.fn(),
  useLocation: () => ({ pathname: "/playground", search: "", hash: "" })
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
  useMessageOption: () => {
    const state = playgroundFormMessageOptionState.value
    const selection = selectedAssistantMock.currentSelection === undefined
      ? selectedAssistantMock.initialSelection
      : selectedAssistantMock.currentSelection
    return {
      ...state,
      selectedAssistant: state.selectedAssistant ?? selection,
      effectiveAssistantState: state.effectiveAssistantState ?? resolveEffectiveAssistantState({
        tracked: { assistantKind: state.serverChatAssistantKind, assistantId: state.serverChatAssistantId, characterId: state.serverChatCharacterId },
        draftSelection: selection
      })
    }
  }
}))

vi.mock("@/store/option", () => ({
  useStoreMessageOption: Object.assign(
    (selector: (state: any) => unknown) =>
      selector({
        ...playgroundFormMessageOptionState.value,
        setRagMediaIds: vi.fn(),
        setRagPinnedResults: vi.fn()
      }),
    {
      getState: () => ({ ...playgroundFormMessageOptionState.value }),
      subscribe: vi.fn(() => () => {})
    }
  )
}))

vi.mock("@/store/model", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/store/model")>()
  nativeSetupFixture.actualSettingsStore = actual.useStoreChatModelSettings
  const getDeclaredSettings = (): ModelSettingsStore => {
    if (nativeSetupFixture.settings) return nativeSetupFixture.settings
    return (nativeSetupFixture.fallbackSettings ??= {
      ...actual.useStoreChatModelSettings.getState(),
      systemPrompt: chatModelSettingsState.systemPrompt,
      setSystemPrompt: chatModelSettingsState.setSystemPrompt,
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
      updateSetting: chatModelSettingsState.updateSetting,
      updateSettings: chatModelSettingsState.updateSettings,
      setActiveSettingsScope: vi.fn(),
      updateScopedSetting: vi.fn(),
      getEffectiveSettings: vi.fn(() => ({}))
    })
  }
  return {
    useStoreChatModelSettings: Object.assign(
      (selector?: (state: ModelSettingsStore) => unknown) => {
        const state = nativeSetupFixture.realSettings
          ? actual.useStoreChatModelSettings.getState()
          : getDeclaredSettings()
        return selector ? selector(state) : state
      },
      {
        getState: () =>
          nativeSetupFixture.realSettings
            ? actual.useStoreChatModelSettings.getState()
            : getDeclaredSettings(),
        subscribe: actual.useStoreChatModelSettings.subscribe
      }
    )
  }
})

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

// Real persistence/hydration can be mounted over acknowledged in-memory bucket IO;
// this is not native browser durability or account verification evidence.
const draftBucket = vi.hoisted(() => ({ records: new Map<string, { value: unknown; updatedAt: number }>() }))
const sceneBucket = vi.hoisted(() => ({
  records: new Map<string, unknown>(),
  preferenceRecords: new Map<string, unknown>(),
  get: vi.fn<(key: string) => Promise<unknown>>(),
  set: vi.fn<(key: string, value: unknown) => Promise<void>>()
}))
// Scene storage is a declared durable physical engine. The real Scene service,
// rich read, conditional bucket write, lock and Applied store remain connected.
vi.mock("@/utils/safe-storage", async (original) => ({
  ...(await original<typeof import("@/utils/safe-storage")>()),
  createSafeStorage: () => ({
    hasPersistentBackend: true,
    get: async (key: string) =>
      key.startsWith("actorSettings:") || key.startsWith("actorProfile:")
        ? sceneBucket.get(key)
        : sceneBucket.preferenceRecords.get(key),
    set: async (key: string, value: unknown) => {
      if (key.startsWith("actorSettings:") || key.startsWith("actorProfile:"))
        await sceneBucket.set(key, value)
      else sceneBucket.preferenceRecords.set(key, structuredClone(value))
    },
    getAll: async () =>
      Object.fromEntries([
        ...sceneBucket.preferenceRecords,
        ...sceneBucket.records
      ]),
    remove: async (key: string) => {
      sceneBucket.records.delete(key)
      sceneBucket.preferenceRecords.delete(key)
    },
    watch: () => () => {},
    unwatch: () => {}
  })
}))
vi.mock("@/services/settings/local-bucket", async (original) => {
  const actual =
    await original<typeof import("@/services/settings/local-bucket")>()
  return {
    ...actual,
    createLocalRegistryBucket: (
      options: Parameters<typeof actual.createLocalRegistryBucket>[0]
    ) => {
      const { prefix } = options
      if (prefix === "actorSettings:owned:" || prefix === "actorProfile:owned:")
        return actual.createLocalRegistryBucket(options)
      return {
        isDurable: () => true,
        buildKey: (key: string) => prefix + key,
        get: async (key: string) =>
          draftBucket.records.get(prefix + key) ?? null,
        set: async (key: string, value: unknown) => {
          draftBucket.records.set(prefix + key, {
            value,
            updatedAt: Date.now()
          })
          return true
        },
        list: async () => ({
          records: Object.fromEntries(
            [...draftBucket.records]
              .filter(([key]) => key.startsWith(prefix))
              .map(([key, value]) => [key.slice(prefix.length), value])
          ),
          unavailable: false
        }),
        remove: async (key: string) => {
          draftBucket.records.delete(prefix + key)
        },
        cleanup: async () => 0
      }
    }
  }
})
vi.mock("@/hooks/useDraftPersistence", async original => {
  const actual = await original<typeof import("@/hooks/useDraftPersistence")>()
  return { ...actual, useDraftPersistence: (...args: Parameters<typeof actual.useDraftPersistence>) => nativeSetupFixture.realDrafts
    ? actual.useDraftPersistence(...args)
    : { draftSaved: false, draftReady: true, retainDraft: nativeSetupFixture.retain, flushDraft: vi.fn() } }
})

vi.mock("@/hooks/useSelectedCharacter", () => ({
  useSelectedCharacter: () => [
    selectedCharacterState.value,
    selectedCharacterState.setSelectedCharacter
  ]
}))

vi.mock("@/hooks/useSelectedAssistant", () => ({
  getSelectedAssistantOperationRevision: () => nativeSetupFixture.assistantRevision,
  useSelectedAssistant: (initialValue: any = null) => {
    const [selectedAssistant, setSelectedAssistant] = React.useState(selectedAssistantMock.initialSelection ?? initialValue)
    const publishSelection = (next: AssistantSelection | null) => {
      selectedAssistantMock.currentSelection = next
      setSelectedAssistant(next)
    }
    selectedAssistantMock.externalMirror = publishSelection
    const setSelectedAssistantWithBroadcast = async (next: any, options: { isCurrent?: () => boolean } = {}) => {
      // Match the real owned writer: check caller at entry, then allocate its own revision.
      if (options.isCurrent?.() === false) return false
      nativeSetupFixture.assistantRevision++
      nativeSetupFixture.identityGuard = options.isCurrent ?? null
      const acknowledged = await selectedAssistantMock.setSelectedAssistant(next)
      if (acknowledged === false) return false
      if (options.isCurrent?.() === false) return false
      publishSelection(next)
      return true
    }
    return [
      selectedAssistant,
      setSelectedAssistantWithBroadcast,
      { isLoading: selectedAssistantMock.isLoading, isCurrent: assistantOwnerIsCurrent, setRenderValue: publishSelection }
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

vi.mock("../ComposerToolbar", async importOriginal => {
  const actual = await importOriginal<typeof import("../ComposerToolbar")>()
  const StubToolbar = ({
    modelSelectButton,
    serverChatId,
    onRolePlayTriggerCapture,
    researchLaunchButton,
    toolsButton,
    sendControl,
    rolePlayActions,
    contextItems,
    setSelectedSystemPrompt,
    onOpenModelSettings
  }: {
    onOpenModelSettings?: () => void
    serverChatId?: string | null
    onRolePlayTriggerCapture?: (element: HTMLButtonElement) => void
    modelSelectButton?: React.ReactNode
    researchLaunchButton?: React.ReactNode
    toolsButton?: React.ReactNode
    sendControl?: React.ReactNode
    contextItems?: { label?: React.ReactNode; value?: React.ReactNode }[]
    rolePlayActions?: { onOpenRolePlaySetup: () => void }
    setSelectedSystemPrompt?: (id: string | undefined) => void
  }) => (
    <div data-testid="composer-toolbar">
      <span data-testid="composer-context-items">{JSON.stringify(contextItems?.map(item => ({label: item.label, value: item.value})))}</span>
      <button type="button" onClick={onOpenModelSettings}>Open default model settings</button>
      {modelSelectButton}
      {researchLaunchButton}
      {toolsButton}
      <div data-testid="composer-inline-send-control">{sendControl}</div>
      <div onClickCapture={(event) => {
        const button = event.currentTarget.querySelector("button")
        if (button) onRolePlayTriggerCapture?.(button)
      }}>
        <BuddyManagementButton target={serverChatId ? { scope_type: "conversation", scope_id: serverChatId } : null} onConversationSettings={rolePlayActions?.onOpenRolePlaySetup} />
      </div>
      <button type="button" onClick={rolePlayActions?.onOpenRolePlaySetup}>Open role-play setup</button>
      <button
        type="button"
        onClick={() => setSelectedSystemPrompt?.("prompt-1")}>
        Reselect system prompt
      </button>
      <button
        type="button"
        onClick={() => setSelectedSystemPrompt?.("prompt-2")}>
        Change system prompt
      </button>
    </div>
  )

  return { ComposerToolbar: (props: React.ComponentProps<typeof actual.ComposerToolbar>) =>
    viewportState.realToolbar ? <actual.ComposerToolbar {...props} /> : <StubToolbar {...props} /> }
})

vi.mock("@/components/Common/PromptSelect", () => ({ PromptSelect: () => <button>System prompt selector</button> }))
vi.mock("@/components/Layouts/ConnectionStatus", () => ({ ConnectionStatus: () => <span>Connection status</span> }))
vi.mock("../SystemPromptTemplates", async importOriginal => ({
  ...await importOriginal<typeof import("../SystemPromptTemplates")>(),
  SystemPromptTemplatesModal: () => null
}))

vi.mock("../CompareToggle", () => ({
  CompareToggle: () => null
}))

vi.mock("@/services/actor-settings", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("@/services/actor-settings")>()
  nativeSetupFixture.actualSceneReader = actual.getActorSceneForChat
  nativeSetupFixture.actualSceneWriter = actual.saveActorSettingsForChat
  return {
    ...actual,
    saveActorSettingsForChat: saveActorSettingsMock,
    getActorSettingsForChat: nativeSetupFixture.actor,
    getActorSceneForChat: nativeSetupFixture.scene
  }
})

vi.mock("@/hooks/useHomeMilestoneScope", () => ({ useHomeMilestoneScope: () => null }))

vi.mock("@/components/Chat/composer/PromptAssistComposerAction", () => ({ PromptAssistComposerAction: () => null }))

vi.mock("../RolePlaySetupDrawer", async (original) => {
  const actual = await original<typeof import("../RolePlaySetupDrawer")>()
  type Props = React.ComponentProps<typeof actual.RolePlaySetupDrawer>
  const StubDrawer = ({
    characterId,
    onApply,
    onClose,
    returnFocusRef,
    savedRolePlaySetups,
    onPreviewSavedSetup,
    onApplySavedSetup,
    captureApplyGuard
  }: Props) => {
    ownerGuardFixture.capture = captureApplyGuard ?? null
    const [result, setResult] = React.useState("")
    const mounted = React.useRef(true)
    React.useEffect(
      () => () => {
        mounted.current = false
      },
      []
    )
    const apply = (payload: RolePlaySetupApplyPayload) => {
      const pending = Promise.resolve(
        onApply(payload, {
          isCurrent: () => mounted.current,
          onDestination: (next) => {
            nativeSetupFixture.destinationCurrent = next.isCurrent
          }
        })
      )
      nativeSetupFixture.selectionApply = pending
      void pending.then(
        (accepted) =>
          setResult(accepted === false ? "Apply failed" : "Applied"),
        () => setResult("Apply failed")
      )
    }
    const applySaved = () => {
      const saved = savedRolePlaySetups?.[0]
      if (!saved || !onApplySavedSetup)
        throw new Error(
          "fixture requires the actual saved-setup Apply callback"
        )
      const pending = Promise.resolve(
        onApplySavedSetup(saved, {
          isCurrent: () => mounted.current,
          onDestination: (next) => {
            nativeSetupFixture.destinationCurrent = next.isCurrent
          }
        })
      )
      nativeSetupFixture.selectionApply = pending
      void pending.then(
        (accepted) =>
          setResult(accepted === false ? "Apply failed" : "Applied"),
        () => setResult("Apply failed")
      )
    }
    return (
      <section aria-label="Role-play setup">
        <span data-testid="setup-character">{characterId ?? "None"}</span>
        <button
          onClick={() => {
            onClose()
            returnFocusRef?.current?.focus()
          }}>
          Cancel setup
        </button>
        <button
          onClick={() =>
            apply({
              identitySelection: {
                kind: "persona",
                id: "persona-guide",
                name: "Guide",
                metadata: { selectionMode: "tracked" }
              },
              generationPresetKey: "precise"
            })
          }>
          Apply Persona setup
        </button>
        <button
          onClick={() =>
            apply({
              identitySelection: {
                kind: "persona",
                id: "persona-guide",
                name: "Guide",
                metadata: { selectionMode: "overlay" }
              },
              generationPresetKey: "precise"
            })
          }>
          Apply Persona overlay setup
        </button>
        <button onClick={() => apply({ clearIdentity: true })}>
          Apply clear identity
        </button>
        <button
          onClick={() =>
            apply({
              identitySelection: {
                kind: "character",
                id: "char-new",
                name: "New character",
                metadata: { selectionMode: "tracked" }
              }
            })
          }>
          Apply Character setup
        </button>
        <button
          onClick={() =>
            apply({
              identitySelection: {
                kind: "character",
                id: "char-new",
                name: "New character",
                metadata: { selectionMode: "tracked" }
              },
              sceneSettings: {
                ...createDefaultActorSettings(),
                isEnabled: true
              }
            })
          }>
          Apply Character enabled Scene
        </button>
        <button
          onClick={() =>
            apply({
              identitySelection: {
                kind: "character",
                id: "char-new",
                name: "New character",
                metadata: { selectionMode: "tracked" }
              },
              sceneSettings: createDefaultActorSettings()
            })
          }>
          Apply Character neutral Scene
        </button>
        <button
          onClick={() =>
            apply({
              identitySelection: {
                kind: "persona",
                id: "persona-guide",
                name: "Guide",
                metadata: { selectionMode: "tracked" }
              },
              behaviorTemplate: {
                id: "guarded-behavior",
                title: "Guarded behavior",
                content: "Original behavior proposal",
                category: "roleplay"
              },
              generationPresetKey: "precise"
            })
          }>
          Apply behavior and generation setup
        </button>
        <button
          onClick={() =>
            apply({
              identitySelection: {
                kind: "persona",
                id: "persona-guide",
                name: "Guide",
                metadata: { selectionMode: "overlay" }
              },
              behaviorTemplate: {
                id: "guarded-behavior",
                title: "Guarded behavior",
                content: "Original behavior proposal",
                category: "roleplay"
              },
              generationPresetKey: "precise"
            })
          }>
          Apply overlay behavior and generation setup
        </button>
        <button onClick={applySaved}>Apply owned model saved setup</button>
        <button
          onClick={() => onPreviewSavedSetup?.(savedRolePlaySetups![0].id)}>
          Preview saved setup
        </button>
        <span role="status">{result}</span>
      </section>
    )
  }
  return {
    ...actual,
    RolePlaySetupDrawer: (props: Props) => {
      if (!nativeSetupFixture.realDrawer) return <StubDrawer {...props} />
      nativeSetupFixture.drawerProps = props
      return (
        <actual.RolePlaySetupDrawer
          {...props}
          onApply={(payload, originalContext) => {
            // Record the actual callback and optionally exercise duplicate delivery of
            // its own real action-start acknowledgement; no receipt is manufactured.
            type StartedContext = RolePlayApplyContext & {
              onActionStarted?: (revision: number) => boolean
            }
            const context: StartedContext | undefined = originalContext
            const acknowledge = context?.onActionStarted
            const forwarded =
              nativeSetupFixture.repeatActionStarted && acknowledge
                ? {
                    ...context,
                    onActionStarted: (revision: number) => {
                      const accepted = acknowledge(revision)
                      nativeSetupFixture.actionAcknowledgements.push(
                        accepted,
                        acknowledge(revision)
                      )
                      return accepted
                    }
                  }
                : context
            const pending = Promise.resolve(props.onApply(payload, forwarded))
            nativeSetupFixture.drawerApplications.push(pending)
            nativeSetupFixture.selectionApply = pending
            return pending
          }}
        />
      )
    }
  }
})

vi.mock("../PlaygroundStartupTemplateModal", () => ({
  PlaygroundStartupTemplateModal: ({
    preview,
    onApply,
    onClose,
    busy = false
  }: {
    preview: StartupTemplateBundle
    onApply: () => Promise<unknown>
    onClose: () => void
    busy?: boolean
  }) => {
    startupPreviewFixture.apply = onApply
    startupPreviewFixture.template = preview
    return (
      <>
        <button
          disabled={busy}
          onClick={() => {
            void onApply()
          }}>
          Apply saved template
        </button>
        <button onClick={onClose}>Cancel template preview</button>
      </>
    )
  }
}))

vi.mock("../ParameterPresets", async (original) => {
  const actual = await original<typeof import("../ParameterPresets")>()
  return {
    ...actual,
    detectCurrentPreset: () => "balanced",
    getPresetByKey: (key: import("../ParameterPresets").PresetKey) =>
      nativeSetupFixture.realRolePlayPresets
        ? actual.getPresetByKey(key)
        : { key: "balanced", settings: {} }
  }
})

vi.mock("../useMobileComposerViewport", () => ({
  useMobileComposerViewport: () => ({
    keyboardOpen: false,
    keyboardInsetPx: 0
  })
}))

const modelSettingsBoundary = vi.hoisted(() => ({ enabled: false, afterClose: null as null | (() => void) }))
vi.mock("@/components/Common/Settings/CurrentChatModelSettings", () => ({
  CurrentChatModelSettings: ({ open, setOpen, afterClose, settingsScope }: {
    open: boolean; setOpen: (value: boolean) => void; afterClose?: () => void; settingsScope?: string | null
  }) => {
    React.useEffect(() => {
      if (modelSettingsBoundary.enabled && !open) modelSettingsBoundary.afterClose = afterClose ?? null
    }, [open, afterClose])
    return modelSettingsBoundary.enabled ? <section data-testid="held-model-editor" data-open={String(open)} data-scope={settingsScope ?? ""}>
      <button onClick={() => setOpen(false)}>Close held model editor</button>
    </section> : null
  }
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
    getCharacter: nativeSetupFixture.getCharacter,
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

vi.mock("../PlaygroundKnowledgeSection", () => ({
  PlaygroundKnowledgeSection: ({ onAsk }: { onAsk?: (text: string, options?: { ignorePinnedResults?: boolean }) => void }) => (
    <button
      data-testid="knowledge-ask"
      type="button"
      onClick={() =>
        onAsk?.("Ask with this pinned source.", { ignorePinnedResults: true })
      }
    >
      Ask knowledge
    </button>
  )
}))

vi.mock("@/components/Common/Beta", () => ({
  BetaTag: () => null
}))

vi.mock("@/components/Common/Playground/DocumentGeneratorDrawer", () => ({
  DocumentGeneratorDrawer: () => null
}))

vi.mock("@/store/ui-mode", () => ({
  useUiModeStore: (selector: (state: { mode: string }) => unknown) =>
    selector({ mode: viewportState.mode })
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
  useMobile: () => viewportState.mobile
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

vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => ({
    success: vi.fn(),
    error: nativeSetupFixture.error,
    warning: vi.fn(),
    info: vi.fn()
  })
}))

vi.mock("@/utils/onboarding-ingestion-telemetry", () => ({
  trackOnboardingChatSubmitSuccess: vi.fn(async () => undefined)
}))

vi.mock("@/hooks/playground", () => ({
  useModelSelector: ({
    composerModels,
    selectedModel
  }: UseModelSelectorParams) => {
    const [open, setOpen] = React.useState(false)
    const settings = nativeSetupFixture.realSettings
      ? nativeSetupFixture.actualSettingsStore!.getState()
      : null
    const selectedModelSettingsScope = settings
      ? resolveModelSelectorSelection(
          composerModels,
          selectedModel,
          settings.apiProvider,
          {
            scope: settings.activeSettingsScope,
            providerOverride: settings.activeSettingsScope
              ? settings.scopedSettingsByModelKey[settings.activeSettingsScope]
                  ?.apiProvider
              : undefined
          }
        ).settingsScope
      : undefined
    return {
      selectedModelKey: nativeSetupFixture.controller
        ? String(playgroundFormMessageOptionState.value.selectedModel).includes(
            ":"
          )
          ? playgroundFormMessageOptionState.value.selectedModel
          : `openai:${playgroundFormMessageOptionState.value.selectedModel}`
        : undefined,
      selectedModelSettingsScope,
      modelDropdownOpen: nativeSetupFixture.realModelPicker && open,
      setModelDropdownOpen: nativeSetupFixture.realModelPicker
        ? setOpen
        : vi.fn(),
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
    }
  },
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
    imageBackendOptions: [
      { value: "mock-backend", label: "Mock Backend", provider: "custom" }
    ],
    imageBackendLabel: "Mock Backend",
    imageBackendActiveKey: "mock-backend",
    imageBackendMenuItems: [],
    imageBackendBadgeLabel: "Mock Backend"
  }),
  useActionBarVisibility: () => ({
    composerTextareaExpanded: false,
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
    activeImageCommand: modelCatalogQueryOptions.imageCommand,
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

beforeEach(async () => {
  modelCatalogQueryOptions.loading = false
  modelCatalogQueryOptions.imageCommand = false
  modelSettingsBoundary.enabled = false
  modelSettingsBoundary.afterClose = null
  await seedLocalFork(false)
  nativeSetupFixture.localOwner = await getLocalHistoryOwner("history-1", {
    profileId: "profile",
    serverScopeKey: null,
    isCurrent: () => true
  })
  installSettingsLocks()
  useActorStore.getState().reset()
  localStorage.clear()
  nativeSetupFixture.realSettings = false
  nativeSetupFixture.realModelPicker = false
  nativeSetupFixture.realDrawer = false
  nativeSetupFixture.realRolePlayPresets = false
  nativeSetupFixture.repeatActionStarted = false
  nativeSetupFixture.actionAcknowledgements = []
  nativeSetupFixture.drawerApplications = []
  nativeSetupFixture.drawerProps = null
  queuedDocumentPreparation.mockReset()
  nativeSetupFixture.realDrafts = false
  nativeSetupFixture.ownerCurrent = true
  nativeSetupFixture.destinationCurrent = null
  draftBucket.records.clear()
  nativeSetupFixture.formattingGuide = false
  nativeSetupFixture.identityGuard = null
  nativeSetupFixture.assistantRevision = 0
  nativeSetupFixture.retain.mockReset().mockResolvedValue(true)
  nativeSetupFixture.getCharacter
    .mockReset()
    .mockResolvedValue({ id: "route-card", name: "Route Card" })
  nativeSetupFixture.error.mockClear()
  nativeSetupFixture.actualSettingsStore!.getState().reset()
  nativeSetupFixture.controller = null
  nativeSetupFixture.settings = null
  nativeSetupFixture.fallbackSettings = null
  nativeSetupFixture.tools = { tools: [] }
  nativeSetupFixture.actor
    .mockReset()
    .mockResolvedValue(createDefaultActorSettings())
  nativeSetupFixture.scene
    .mockReset()
    .mockImplementation(nativeSetupFixture.actualSceneReader!)
  nativeSetupFixture.onDraftPublication = null
  nativeSetupFixture.selectionApply = null
  sceneBucket.records.clear()
  sceneBucket.preferenceRecords.clear()
  sceneBucket.get.mockReset().mockImplementation(async (key) => {
    const record = sceneBucket.records.get(key)
    return record === undefined ? undefined : structuredClone(record)
  })
  sceneBucket.set.mockReset().mockImplementation(async (key, value) => {
    sceneBucket.records.set(key, structuredClone(value))
  })
  nativeSetupFixture.snapshot
    .mockReset()
    .mockResolvedValue({
      scopeSignal: new AbortController().signal,
      scopeInvalidatedSignal: new AbortController().signal,
      requestScope: {
        config: { serverUrl: "http://localhost:8000", authMode: "single-user" },
        userId: null
      },
      release: vi.fn()
    })
  selectedAssistantMock.isLoading = false
  selectedAssistantMock.defaultCharacter = null
  selectedAssistantMock.ownedDefaultCharacter = null
  savedSetupFixture.raw = "[]"
  startupPreviewFixture.apply = null
  startupPreviewFixture.template = null
  saveActorSettingsMock
    .mockReset()
    .mockImplementation(nativeSetupFixture.actualSceneWriter!)
  nativeSetupFixture.realDrawer = false
  viewportState.mobile = false
  viewportState.mode = "pro"
  viewportState.realToolbar = false
  useBuddyManagementStore.getState().close()
  onSubmitMock.mockClear()
  createChatCompletionMock.mockClear()
  selectedAssistantMock.initialSelection = null
  selectedAssistantMock.currentSelection = undefined
  selectedAssistantMock.externalMirror = null
  selectedAssistantMock.setSelectedAssistant
    .mockReset()
    .mockResolvedValue(undefined)
  selectedCharacterState.value = null
  selectedCharacterState.setSelectedCharacter.mockClear()
  chatModelSettingsState.systemPrompt = ""
  chatModelSettingsState.setSystemPrompt
    .mockReset()
    .mockImplementation((prompt) => {
      chatModelSettingsState.systemPrompt = prompt ?? ""
      if (nativeSetupFixture.settings)
        nativeSetupFixture.settings.systemPrompt = prompt
      else if (nativeSetupFixture.fallbackSettings)
        nativeSetupFixture.fallbackSettings.systemPrompt = prompt
    })
  chatModelSettingsState.updateSetting.mockClear()
  chatModelSettingsState.updateSettings.mockReset()
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

afterEach(() => {
  vi.unstubAllGlobals()
})

const prepareNativeSource = () => {
  let epoch = 0
  let selectionController = new AbortController()
  const originalOwner = nativeSetupFixture.localOwner
  if (!originalOwner)
    throw new Error("fixture requires acknowledged local ownership")
  const view: HistoryViewSelectionV1 = {
    owner_key: originalOwner.owner_key,
    conversation_id: originalOwner.conversation_id,
    view_session_id: "local-settings-view",
    interpretation: { kind: "parent_graph_v1" },
    cursor: { kind: "empty" },
    selection_revision: 0
  }
  let current: SettingsSelectionState = {
    status: "ready",
    owner: originalOwner,
    view
  }
  const reset = vi.fn(() => {
    epoch++
    selectionController.abort()
    selectionController = new AbortController()
    current = { status: "idle", owner: null, view: null }
  })
  nativeSetupFixture.controller = {
    getCurrent: () => current,
    getSignal: () => selectionController.signal,
    settingsMode: (chatId) => !chatId ? "ordinary" :
      current.owner?.kind === "native" && current.owner.conversation_id === chatId &&
      current.view?.conversation_id === chatId && current.owner.validate_lease()
        ? "ordinary" : "pending",
    reset,
    fence: () => {
      const captured = epoch
      return () => epoch === captured
    },
    draftAccount: {
      requestScope: {
        config: { serverUrl: "http://localhost:8000", authMode: "single-user" },
        userId: null
      },
      isCurrent: assistantOwnerIsCurrent
    }
  }
  nativeSetupFixture.settings = {
    ...nativeSetupFixture.actualSettingsStore!.getState(),
    setSystemPrompt: chatModelSettingsState.setSystemPrompt,
    apiProvider: "openai",
    temperature: 0,
    topP: 0,
    getEffectiveSettings: (_scope, updates) => ({
      ...nativeSetupFixture.settings!,
      ...updates
    }),
    updateSettings: chatModelSettingsState.updateSettings,
    updateSetting: chatModelSettingsState.updateSetting,
    setActiveSettingsScope: vi.fn()
  }
  const source = playgroundFormMessageOptionState.value
  source.chatDraftAddress = {
    kind: "local",
    id: originalOwner.conversation_id,
    profileId: originalOwner.profile_id
  } satisfies ChatDraftAddress
  source.historyId = originalOwner.conversation_id
  source.setQueuedMessages.mockImplementation(
    (
      next:
        | import("@/utils/chat-request-queue").QueuedRequest[]
        | ((
            previous: import("@/utils/chat-request-queue").QueuedRequest[]
          ) => import("@/utils/chat-request-queue").QueuedRequest[])
    ) => {
      source.queuedMessages =
        typeof next === "function" ? next(source.queuedMessages) : next
    }
  )
  return { reset, source, originalOwner, owner: () => current.owner }
}

const prepareQualifiedSceneSource = (
  kind: "native" | "fresh" = "native",
  nativeChatId = "native-current",
  nativeHistoryId = "native-current-mirror",
  freshTemporary = false
) => {
  const base = prepareNativeSource()
  const source = base.source
  const controller = nativeSetupFixture.controller!
  let accountCurrent = true
  const isAccountCurrent = () =>
    accountCurrent && nativeSetupFixture.ownerCurrent
  const requestScope: ServicePromptRequestScope = {
    config: {
      serverUrl: "http://localhost:8000",
      authMode: "single-user",
      apiKey: "original-selection-key",
      expectedSingleUserApiKeyScope: "original-selection-scope"
    },
    userId: null
  }
  const scope = { type: "global" } as const
  const owner: NativeHistoryOwnerV1 = {
    kind: "native",
    conversation_id: nativeChatId,
    owner_key: "native-selection-owner",
    request_scope: requestScope,
    scope,
    validate_lease: isAccountCurrent
  }
  const view: HistoryViewSelectionV1 = {
    owner_key: owner.owner_key!,
    conversation_id: owner.conversation_id,
    view_session_id: "original-settings-view",
    interpretation: { kind: "parent_graph_v1" },
    cursor: { kind: "empty" },
    selection_revision: 4
  }
  if (kind === "native") {
    Object.assign(controller.getCurrent(), { status: "ready", owner, view })
    source.chatDraftAddress = {
      kind: "native",
      id: owner.conversation_id,
      ownerKey: view.owner_key,
      scope
    } satisfies ChatDraftAddress
    source.serverChatId = owner.conversation_id
    source.historyId = nativeHistoryId
  } else {
    controller.reset()
    base.reset.mockClear()
    source.chatDraftAddress = freshChatDraftAddress(freshTemporary)
    source.temporaryChat = freshTemporary
    source.serverChatId = null
    source.historyId = null
  }
  source.selectedModel = "openai:deepseek-chat"
  source.serverChatAssistantKind = "character"
  source.serverChatCharacterId = "char-new"
  source.serverChatAssistantId = "char-new"
  source.serverChatMetaLoaded = true
  controller.draftAccount = { requestScope, isCurrent: isAccountCurrent }
  nativeSetupFixture.snapshot.mockResolvedValue({
    scopeSignal: new AbortController().signal,
    scopeInvalidatedSignal: new AbortController().signal,
    requestScope,
    release: vi.fn()
  })
  const address: ChatDraftAddress = source.chatDraftAddress
  const revision: number = source.chatDraftRevision
  const target = captureActorSettingsTarget({
    address,
    owner: kind === "native" ? owner : null,
    view: kind === "native" ? view : null,
    requestScope,
    accountIsCurrent: isAccountCurrent,
    isCurrent: () =>
      isAccountCurrent() &&
      source.chatDraftAddress === address &&
      source.chatDraftRevision === revision
  })
  if (!target)
    throw new Error("fixture requires an original qualified Scene locator")
  return {
    ...base,
    target,
    address,
    revision,
    requestScope,
    retireOriginalAccount: () => {
      accountCurrent = false
    }
  }
}

const sceneSelectionGate = <T,>() => {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((done) => {
    resolve = done
  })
  return { promise, resolve }
}

const ownedStartupBundle = (
  scene: ActorSettings,
  identity: NonNullable<StartupTemplateBundle["rolePlay"]>["identity"] = {
    kind: "character",
    id: "char-new",
    name: "New character"
  }
) =>
  createStartupTemplateBundle({
    name: "Owned startup Scene",
    selectedModel: null,
    systemPrompt: "",
    rolePlay: {
      source: "role-play-setup",
      identity,
      behavior: null,
      scene,
      generation: null,
      context: null
    }
  })

const openOwnedStartupPreview = async () => {
  const user = userEvent.setup()
  const view = render(<PlaygroundForm droppedFiles={[]} />)
  await user.click(screen.getByRole("button", { name: "Open role-play setup" }))
  await user.click(
    await screen.findByRole("button", { name: "Preview saved setup" })
  )
  await screen.findByRole("button", { name: "Apply saved template" })
  const apply = startupPreviewFixture.apply
  if (!apply)
    throw new Error("fixture requires the actual startup Apply callback")
  return { user, view, apply }
}

const openActualSceneDrawer = async (
  kind: "native" | "fresh" | "neutral fresh" = "native",
  options: { temporary?: boolean; nonempty?: boolean; persona?: boolean; unavailableFreshModel?: boolean } = {}
) => {
  const f = prepareQualifiedSceneSource(
    kind === "native" ? "native" : "fresh",
    "native-current",
    "native-current-mirror",
    options.temporary ?? false
  )
  if (options.nonempty) {
    f.source.history = [
      { role: "user", content: "Original Temporary transcript" }
    ]
    f.source.messages = [
      {
        id: "temporary-message",
        isBot: false,
        name: "You",
        message: "Original Temporary presentation",
        sources: []
      }
    ]
  }
  if (kind === "neutral fresh") {
    f.source.serverChatAssistantKind = null
    f.source.serverChatAssistantId = null
    f.source.serverChatCharacterId = null
  }
  if (options.persona) {
    f.source.serverChatAssistantKind = "persona"
    f.source.serverChatAssistantId = "persona-guide"
    f.source.serverChatCharacterId = null
    selectedAssistantMock.initialSelection = {
      kind: "persona",
      id: "persona-guide",
      name: "Persona guide",
      metadata: { selectionMode: "tracked" }
    } satisfies AssistantSelection
  }
  if (options.unavailableFreshModel) {
    f.source.selectedModel = null
    nativeSetupFixture.models = []
  }
  nativeSetupFixture.realDrawer = true
  const original: ActorSceneRecord = {
    value: {
      ...createDefaultActorSettings(),
      notes: "Original hydrated Scene"
    },
    updatedAt: 1730000000000
  }
  const key = `actorSettings:owned:${f.target.key}`
  sceneBucket.records.set(key, original)
  const user = userEvent.setup()
  const view = render(<PlaygroundForm droppedFiles={[]} />)
  await user.click(screen.getByRole("button", { name: "Open role-play setup" }))
  const notes = await screen.findByRole("textbox", { name: "Scene notes" })
  await waitFor(() => expect(notes).toHaveValue(original.value.notes))
  await waitFor(() =>
    expect(
      screen.getByRole("button", { name: "Apply", exact: true })
    ).toBeEnabled()
  )
  fireEvent.change(notes, { target: { value: "Staged connected Scene" } })
  return { ...f, key, original, user, view }
}

// The actual Form emits this event; this declared router-listener binding uses
// the real session cancellation and exact callback from the dispatched detail.
// It does not claim a mounted Playground router or issue any H1 capability.
const installRouteReceiptListener = (externalRestore: boolean) => {
  const events: ChatRouteReplacementDetail[] = []
  const acknowledgements: boolean[] = []
  const listener = (event: Event) => {
    const detail = (event as CustomEvent<ChatRouteReplacementDetail>).detail
    const source = playgroundFormMessageOptionState.value
    const session = usePlaygroundSessionStore.getState()
    if (
      detail.href !== window.location.href ||
      detail.serverChatId !== source.serverChatId ||
      detail.historyId !== source.historyId ||
      detail.restoreRevision !== session.restoreRevision
    )
      return
    events.push(detail)
    session.cancelPendingRestore()
    const revision = usePlaygroundSessionStore.getState().restoreRevision
    acknowledgements.push(detail.onRestoreCancelled?.(revision) ?? false)
    if (externalRestore)
      usePlaygroundSessionStore.getState().cancelPendingRestore()
  }
  window.addEventListener(CHAT_ROUTE_REPLACEMENT_EVENT, listener)
  return {
    events,
    acknowledgements,
    remove: () =>
      window.removeEventListener(CHAT_ROUTE_REPLACEMENT_EVENT, listener)
  }
}

describe("PlaygroundForm role-play starter", () => {
  it("disables ordinary Send while the required catalog is loading and enables it after settlement", async () => {
    prepareNativeSource()
    modelCatalogQueryOptions.loading = true
    const view = render(<PlaygroundForm droppedFiles={[]} />)
    try {
    const sends = within(view.container).getAllByRole("button", { name: "Send message", exact: true })
    for (const send of sends) {
      expect(send).toBeDisabled()
      expect(send).toHaveAttribute("title", "Loading models…")
    }
    modelCatalogQueryOptions.loading = false
    view.rerender(<PlaygroundForm droppedFiles={[]} />)
    for (const send of sends) expect(send).toBeEnabled()
    } finally { view.unmount() }
  })
  it.each(["image", "saved character"] as const)("keeps %s Send independent of the ordinary catalog load", kind => {
    if (kind === "saved character") {
      const { source } = prepareQualifiedSceneSource("native", "saved-source")
      source.serverChatAssistantKind = "character"
      source.serverChatCharacterId = "saved-character"
    } else {
      prepareNativeSource()
      modelCatalogQueryOptions.imageCommand = true
    }
    modelCatalogQueryOptions.loading = true
    const view = render(<PlaygroundForm droppedFiles={[]} />)
    try {
      for (const send of within(view.container).getAllByRole("button", { name: "Send message", exact: true })) expect(send).toBeEnabled()
    } finally { view.unmount() }
  })

  it.each([true, false])("requires the model catalog independently of an unopened picker while connected=%s", async connected => {
    const { useChatSurfaceCoordinatorStore } = await import("@/store/chat-surface-coordinator")
    useChatSurfaceCoordinatorStore.setState({ visiblePanels: { "model-catalog": false }, engagedPanels: { "model-catalog": false } })
    playgroundFormConnectionState.isConnected = connected
    playgroundFormConnectionState.phase = connected ? "connected" : "disconnected"
    modelCatalogQueryOptions.enabled = []
    const view = render(<PlaygroundForm droppedFiles={[]} />)
    expect(modelCatalogQueryOptions.enabled.length).toBeGreaterThan(0)
    expect(modelCatalogQueryOptions.enabled.every(enabled => enabled === connected)).toBe(true)
    expect(useChatSurfaceCoordinatorStore.getState().visiblePanels["model-catalog"]).toBe(false)
    expect(useChatSurfaceCoordinatorStore.getState().engagedPanels["model-catalog"]).toBe(false)
    view.unmount()
  })

  it.each(["enabled", "unavailable"] as const)(
    "Scene selection isolates a full replacement from %s legacy scratch and the departing owned Scene",
    async (legacy) => {
      const f = prepareQualifiedSceneSource()
      f.source.serverChatCharacterId = "departing-character"
      f.source.serverChatAssistantId = "departing-character"
      const original: ActorSceneRecord = {
        value: {
          ...createDefaultActorSettings(),
          isEnabled: true,
          notes: "Private departing Scene"
        },
        updatedAt: 1730000000000
      }
      const key = `actorSettings:owned:${f.target.key}`
      sceneBucket.records.set(key, original)
      if (legacy === "enabled")
        nativeSetupFixture.actor.mockResolvedValue({
          ...createDefaultActorSettings(),
          isEnabled: true,
          notes: "Private legacy scratch"
        })
      else
        nativeSetupFixture.actor.mockRejectedValue(
          new Error("actor_settings_unavailable")
        )
      const user = userEvent.setup()
      const view = render(<PlaygroundForm droppedFiles={[]} />)
      try {
        await user.type(
          screen.getByTestId("composer-textarea"),
          "Exact replacement source  "
        )
        await user.click(
          screen.getByRole("button", { name: "Open role-play setup" })
        )
        await user.click(
          await screen.findByRole("button", { name: "Apply Character setup" })
        )
        await screen.findByText("Applied")
        expect(nativeSetupFixture.actor).not.toHaveBeenCalled()
        expect(nativeSetupFixture.scene).not.toHaveBeenCalled()
        expect(sceneBucket.get).not.toHaveBeenCalled()
        expect(sceneBucket.records.get(key)).toBe(original)
        expect(f.source.chatDraftAddress).not.toBe(f.address)
        expect(f.reset).toHaveBeenCalledOnce()
        expect(nativeSetupFixture.retain).toHaveBeenCalledWith(
          expect.any(String),
          expect.objectContaining({ content: "Exact replacement source  " })
        )
        expect(screen.getByTestId("composer-textarea")).toHaveValue(
          "Exact replacement source  "
        )
        expect(saveActorSettingsMock).not.toHaveBeenCalled()
        expect(onSubmitMock).not.toHaveBeenCalled()
        expect(createChatCompletionMock).not.toHaveBeenCalled()
      } finally {
        view.unmount()
      }
    }
  )

  it.each(
    ["native", "fresh"].flatMap((kind) =>
      [false, true].map((enabled) => ({
        kind: kind as "native" | "fresh",
        enabled
      }))
    )
  )(
    "Scene selection uses the retained $kind owned Scene (enabled: $enabled) instead of legacy scratch",
    async ({ kind, enabled }) => {
      const f = prepareQualifiedSceneSource(kind)
      const original: ActorSceneRecord = {
        value: {
          ...createDefaultActorSettings(),
          isEnabled: enabled,
          notes: "Exact original owned Scene"
        },
        updatedAt: 1730000000000
      }
      const key = `actorSettings:owned:${f.target.key}`
      sceneBucket.records.set(key, original)
      nativeSetupFixture.actor.mockResolvedValue({
        ...createDefaultActorSettings(),
        isEnabled: !enabled
      })
      const user = userEvent.setup()
      const view = render(<PlaygroundForm droppedFiles={[]} />)
      try {
        await user.type(
          screen.getByTestId("composer-textarea"),
          "Keep retained source  "
        )
        await user.click(
          screen.getByRole("button", { name: "Open role-play setup" })
        )
        await user.click(
          await screen.findByRole("button", { name: "Apply Character setup" })
        )
        await screen.findByText(enabled ? "Apply failed" : "Applied")
        expect(nativeSetupFixture.actor).not.toHaveBeenCalled()
        expect(nativeSetupFixture.scene).toHaveBeenCalledExactlyOnceWith(
          expect.objectContaining({
            target: expect.objectContaining({ key: f.target.key })
          })
        )
        expect(
          nativeSetupFixture.scene.mock.calls[0][0].characterId
        ).toBeUndefined()
        expect(sceneBucket.get).toHaveBeenCalledExactlyOnceWith(key)
        expect(sceneBucket.records.get(key)).toBe(original)
        expect(f.source.chatDraftAddress).toBe(f.address)
        expect(f.source.chatDraftRevision).toBe(f.revision)
        expect(f.reset).not.toHaveBeenCalled()
        expect(f.source.setHistoryId).not.toHaveBeenCalled()
        expect(selectedAssistantMock.setSelectedAssistant).toHaveBeenCalledTimes(
          enabled ? 0 : 1
        )
        expect(screen.getByTestId("composer-textarea")).toHaveValue(
          "Keep retained source  "
        )
        expect(saveActorSettingsMock).not.toHaveBeenCalled()
        expect(onSubmitMock).not.toHaveBeenCalled()
        expect(createChatCompletionMock).not.toHaveBeenCalled()
      } finally {
        view.unmount()
      }
    }
  )

  it.each(["native", "fresh"] as const)(
    "Scene selection stages neutral defaults for an absent retained $kind Scene without a profile fallback or write",
    async (kind) => {
      const f = prepareQualifiedSceneSource(kind)
      nativeSetupFixture.actor.mockResolvedValue({
        ...createDefaultActorSettings(),
        isEnabled: true
      })
      const user = userEvent.setup()
      const view = render(<PlaygroundForm droppedFiles={[]} />)
      try {
        await user.click(
          screen.getByRole("button", { name: "Open role-play setup" })
        )
        await user.click(
          await screen.findByRole("button", { name: "Apply Character setup" })
        )
        await screen.findByText("Applied")
        expect(nativeSetupFixture.actor).not.toHaveBeenCalled()
        expect(sceneBucket.get).toHaveBeenCalledExactlyOnceWith(
          `actorSettings:owned:${f.target.key}`
        )
        expect(
          nativeSetupFixture.scene.mock.calls[0][0].characterId
        ).toBeUndefined()
        expect(sceneBucket.records.size).toBe(0)
        expect(f.source.chatDraftAddress).toBe(f.address)
        expect(f.reset).not.toHaveBeenCalled()
        expect(saveActorSettingsMock).not.toHaveBeenCalled()
      } finally {
        view.unmount()
      }
    }
  )

  it.each(["unqualified target", "owned read unavailable"] as const)(
    "Scene selection refuses a retained native destination after %s",
    async (failure) => {
      const f = prepareQualifiedSceneSource()
      if (failure === "unqualified target")
        nativeSetupFixture.controller!.getCurrent().view = {
          owner_key: "foreign-owner",
          conversation_id: "native-current",
          view_session_id: "foreign-view",
          interpretation: { kind: "parent_graph_v1" },
          cursor: { kind: "empty" },
          selection_revision: 5
        } satisfies HistoryViewSelectionV1
      else
        sceneBucket.get.mockRejectedValueOnce(
          new Error("physical Scene storage unavailable")
        )
      const user = userEvent.setup()
      const view = render(<PlaygroundForm droppedFiles={[]} />)
      try {
        await user.type(
          screen.getByTestId("composer-textarea"),
          "Keep refused source  "
        )
        await user.click(
          screen.getByRole("button", { name: "Open role-play setup" })
        )
        await user.click(
          await screen.findByRole("button", { name: "Apply Character setup" })
        )
        await screen.findByText("Apply failed")
        expect(nativeSetupFixture.actor).not.toHaveBeenCalled()
        expect(sceneBucket.get).toHaveBeenCalledTimes(
          failure === "unqualified target" ? 0 : 1
        )
        expect(f.source.chatDraftAddress).toBe(f.address)
        expect(f.reset).not.toHaveBeenCalled()
        expect(f.source.setHistoryId).not.toHaveBeenCalled()
        expect(selectedAssistantMock.setSelectedAssistant).not.toHaveBeenCalled()
        expect(screen.getByTestId("composer-textarea")).toHaveValue(
          "Keep refused source  "
        )
        expect(saveActorSettingsMock).not.toHaveBeenCalled()
      } finally {
        view.unmount()
      }
    }
  )

  it.each(
    ["native", "fresh"].flatMap((kind) =>
      [false, true].map((enabled) => ({
        kind: kind as "native" | "fresh",
        enabled
      }))
    )
  )(
    "Scene selection honors an explicit $kind proposal (enabled: $enabled) without reading the stored Scene",
    async ({ kind, enabled }) => {
      const f = prepareQualifiedSceneSource(kind)
      const original: ActorSceneRecord = {
        value: { ...createDefaultActorSettings(), isEnabled: !enabled },
        updatedAt: 1730000000000
      }
      const key = `actorSettings:owned:${f.target.key}`
      sceneBucket.records.set(key, original)
      const user = userEvent.setup()
      const view = render(<PlaygroundForm droppedFiles={[]} />)
      try {
        await user.click(
          screen.getByRole("button", { name: "Open role-play setup" })
        )
        await user.click(
          await screen.findByRole("button", {
            name: enabled
              ? "Apply Character enabled Scene"
              : "Apply Character neutral Scene"
          })
        )
        await screen.findByText(enabled ? "Apply failed" : "Applied")
        expect(nativeSetupFixture.actor).not.toHaveBeenCalled()
        expect(nativeSetupFixture.scene).not.toHaveBeenCalled()
        expect(sceneBucket.records.get(key)).toBe(original)
        expect(f.source.chatDraftAddress).toBe(f.address)
        expect(f.reset).not.toHaveBeenCalled()
        expect(selectedAssistantMock.setSelectedAssistant).toHaveBeenCalledTimes(
          enabled ? 0 : 1
        )
        expect(saveActorSettingsMock).not.toHaveBeenCalled()
      } finally {
        view.unmount()
      }
    }
  )

  it.each([
    "uninterrupted",
    "original account lease",
    "New",
    "New then original address"
  ] as const)(
    "Scene selection retains the original account and destination through a held owned read: %s",
    async (retirement) => {
      const f = prepareQualifiedSceneSource()
      const original: ActorSceneRecord = {
        value: createDefaultActorSettings(),
        updatedAt: 1730000000000
      }
      const key = `actorSettings:owned:${f.target.key}`
      sceneBucket.records.set(key, original)
      const held = sceneSelectionGate<void>()
      // Before migration the legacy boundary reaches this same physical wait.
      // After migration the actual rich service checks its original target on return.
      nativeSetupFixture.actor.mockImplementationOnce(async () => {
        await held.promise
        return createDefaultActorSettings()
      })
      sceneBucket.get.mockImplementationOnce(async (readKey) => {
        await held.promise
        const record = sceneBucket.records.get(readKey)
        return record ? structuredClone(record) : null
      })
      const user = userEvent.setup()
      const view = render(<PlaygroundForm droppedFiles={[]} />)
      try {
        await user.type(
          screen.getByTestId("composer-textarea"),
          "Held original source  "
        )
        await user.click(
          screen.getByRole("button", { name: "Open role-play setup" })
        )
        await user.click(
          await screen.findByRole("button", { name: "Apply Character setup" })
        )
        await waitFor(() =>
          expect(
            nativeSetupFixture.actor.mock.calls.length +
              sceneBucket.get.mock.calls.length
          ).toBe(1)
        )
        if (retirement === "original account lease") f.retireOriginalAccount()
        if (retirement === "New" || retirement === "New then original address") {
          f.source.startNewChatDraft(false)
          if (retirement === "New then original address")
            f.source.setChatDraftAddress(f.address)
        }
        const destination = f.source.chatDraftAddress
        const revision = f.source.chatDraftRevision
        await act(async () => {
          held.resolve(undefined)
          await nativeSetupFixture.selectionApply
        })
        await screen.findByText(
          retirement === "uninterrupted" ? "Applied" : "Apply failed"
        )
        expect(f.source.chatDraftAddress).toBe(destination)
        expect(f.source.chatDraftRevision).toBe(revision)
        expect(f.reset).not.toHaveBeenCalled()
        expect(f.source.setHistoryId).not.toHaveBeenCalled()
        expect(selectedAssistantMock.setSelectedAssistant).toHaveBeenCalledTimes(
          retirement === "uninterrupted" ? 1 : 0
        )
        expect(sceneBucket.records.get(key)).toBe(original)
        expect(screen.getByTestId("composer-textarea")).toHaveValue(
          "Held original source  "
        )
        expect(saveActorSettingsMock).not.toHaveBeenCalled()
      } finally {
        held.resolve(undefined)
        await nativeSetupFixture.selectionApply
        view.unmount()
      }
    }
  )

  it.each([
    "New",
    "New then same raw IDs",
    "New then same issued address"
  ] as const)(
    "Scene selection refuses full destination publication retired by %s",
    async (retirement) => {
      const { source, reset } = prepareNativeSource()
      const originalHistoryId: string | null = source.historyId
      const originalServerId: string | null = source.serverChatId
      source.setHistory.mockImplementation(
        (history: MessageOptionState["history"]) => {
          source.history = history
        }
      )
      source.setMessages.mockImplementation(
        (messages: MessageOptionState["messages"]) => {
          source.messages = messages
        }
      )
      let armed = true
      let replacement:
        | {
            address: ChatDraftAddress
            revision: number
            historyId: string | null
            serverChatId: string | null
            history: MessageOptionState["history"]
            messages: MessageOptionState["messages"]
          }
        | undefined
      nativeSetupFixture.onDraftPublication = (issuedAddress) => {
        if (!armed) return
        armed = false
        source.startNewChatDraft(false)
        if (retirement === "New then same issued address")
          source.setChatDraftAddress(issuedAddress)
        source.historyId =
          retirement === "New then same raw IDs"
            ? originalHistoryId
            : "replacement-local"
        source.serverChatId =
          retirement === "New then same raw IDs"
            ? originalServerId
            : "replacement-server"
        source.history = [{ role: "user", content: "REPLACEMENT TURN" }]
        source.messages = [
          {
            id: "replacement-message",
            isBot: false,
            name: "You",
            message: "REPLACEMENT VIEW",
            sources: []
          }
        ]
        replacement = {
          address: source.chatDraftAddress,
          revision: source.chatDraftRevision,
          historyId: source.historyId,
          serverChatId: source.serverChatId,
          history: source.history,
          messages: source.messages
        }
      }
      const user = userEvent.setup()
      const view = render(<PlaygroundForm droppedFiles={[]} />)
      try {
        await user.type(
          screen.getByTestId("composer-textarea"),
          "Keep source after publication  "
        )
        await user.click(
          screen.getByRole("button", { name: "Open role-play setup" })
        )
        await user.click(
          await screen.findByRole("button", { name: "Apply Character setup" })
        )
        await screen.findByText("Apply failed")
        expect(armed).toBe(false)
        expect(replacement).toBeDefined()
        expect(source.chatDraftAddress).toBe(replacement!.address)
        expect(source.chatDraftRevision).toBe(replacement!.revision)
        expect(source.historyId).toBe(replacement!.historyId)
        expect(source.serverChatId).toBe(replacement!.serverChatId)
        expect(source.history).toBe(replacement!.history)
        expect(source.messages).toBe(replacement!.messages)
        expect(reset).not.toHaveBeenCalled()
        expect(source.setHistoryId).not.toHaveBeenCalled()
        expect(source.setHistory).not.toHaveBeenCalled()
        expect(source.setMessages).not.toHaveBeenCalled()
        expect(nativeSetupFixture.destinationCurrent).toBeNull()
        expect(selectedAssistantMock.setSelectedAssistant).toHaveBeenCalledOnce()
        expect(screen.getByTestId("composer-textarea")).toHaveValue(
          "Keep source after publication  "
        )
        expect(saveActorSettingsMock).not.toHaveBeenCalled()
        expect(onSubmitMock).not.toHaveBeenCalled()
      } finally {
        nativeSetupFixture.onDraftPublication = null
        view.unmount()
      }
    }
  )

  it("keeps the original native Scene read current when its acknowledged mirror hydrates while held", async () => {
    const f = prepareQualifiedSceneSource("native")
    nativeSetupFixture.realDrawer = true
    const original = { value: { ...createDefaultActorSettings(), notes: "Held original Scene" }, updatedAt: 1730000000000 }
    const key = "actorSettings:owned:"+f.target.key
    sceneBucket.records.set(key, original)
    const held = sceneSelectionGate<void>()
    sceneBucket.get.mockImplementationOnce(async readKey => { await held.promise; return structuredClone(sceneBucket.records.get(readKey)) })
    const user = userEvent.setup()
    const view = render(<PlaygroundForm droppedFiles={[]} />)
    try {
      await user.click(screen.getByRole("button", { name: "Open role-play setup" }))
      await waitFor(() => expect(sceneBucket.get).toHaveBeenCalledOnce())
      const mirror = await import("@/db/dexie/server-chat-mirror")
      const alias = await mirror.linkServerChatMirror({ chatId: f.source.serverChatId!, title: "Held Scene mirror", ownerKey: mirror.serverChatMirrorOwnerKey({ requestScope: f.requestScope }), scope: { type: "global" } })
      act(() => { f.source.historyId = alias; view.rerender(<PlaygroundForm droppedFiles={[]} />) })
      await act(async () => { held.resolve(); await held.promise })
      await waitFor(() => expect(screen.getByRole("textbox", { name: "Scene notes" })).toHaveValue(original.value.notes))
      expect(sceneBucket.get).toHaveBeenCalledOnce()
      expect(screen.getByRole("textbox", { name: "Scene notes" })).toBeEnabled()
    } finally { held.resolve(); view.unmount() }
  })

  it.each(["unacknowledged", "owner", "CID", "scope"])("refuses a native Scene alias outside its original receipt (%s)", async mismatch => {
    const f = await openActualSceneDrawer("native")
    const mirror = await import("@/db/dexie/server-chat-mirror")
    const alias = mismatch === "unacknowledged" ? "raw-unacknowledged-13679" : await mirror.linkServerChatMirror({ chatId: mismatch === "CID" ? "other-native" : f.source.serverChatId!, title: "Foreign alias", ownerKey: mismatch === "owner" ? "other-owner" : mirror.serverChatMirrorOwnerKey({ requestScope: f.requestScope }), scope: mismatch === "scope" ? { type: "workspace", workspaceId: "7" } : { type: "global" } })
    try {
      act(() => { f.source.historyId = alias; f.view.rerender(<PlaygroundForm droppedFiles={[]} />) })
      expect(screen.getByRole("button", { name: "Apply", exact: true })).toBeDisabled()
      expect(sceneBucket.records.get(f.key)).toEqual(f.original)
      expect(saveActorSettingsMock).not.toHaveBeenCalled()
    } finally { f.view.unmount() }
  })

  it.each(["staged", "held save"])("preserves the original native Scene across acknowledged mirror hydration (%s)", async phase => {
    const f = await openActualSceneDrawer("native")
    const mirror = await import("@/db/dexie/server-chat-mirror")
    const alias = await mirror.linkServerChatMirror({ chatId: f.source.serverChatId!, title: "Scene mirror hydration", ownerKey: mirror.serverChatMirrorOwnerKey({ requestScope: f.requestScope }), scope: { type: "global" } })
    const held = sceneSelectionGate<void>()
    const persist = sceneBucket.set.getMockImplementation()!
    if (phase === "held save") sceneBucket.set.mockImplementationOnce(async (...args) => { await held.promise; return persist(...args) })
    try {
      if (phase === "held save") { await f.user.click(screen.getByRole("button", { name: "Apply", exact: true })); await waitFor(() => expect(sceneBucket.set).toHaveBeenCalledOnce()) }
      act(() => { f.source.historyId = alias; f.view.rerender(<PlaygroundForm droppedFiles={[]} />) })
      if (phase === "staged") {
        await waitFor(() => expect(screen.getByRole("textbox", { name: "Scene notes" })).toHaveValue("Staged connected Scene"))
        await f.user.click(screen.getByRole("button", { name: "Apply", exact: true }))
      }
      await act(async () => { held.resolve(); await held.promise })
      await waitFor(() => expect(screen.queryByRole("dialog", { name: "Role-play setup", exact: true })).not.toBeInTheDocument())
      expect(sceneBucket.records.get(f.key)?.value.notes).toBe("Staged connected Scene")
      expect(f.source.chatDraftAddress).toBe(f.address)
      expect(f.source.chatDraftRevision).toBe(f.revision)
    } finally { held.resolve(); await Promise.allSettled(nativeSetupFixture.drawerApplications); f.view.unmount() }
  })

  it("applies a saved owned Scene without an unrelated fresh model or catalog", async () => {
    const originalModels = nativeSetupFixture.models
    const f = await openActualSceneDrawer("native", { unavailableFreshModel: true })
    const initialScope = nativeSetupFixture.settings!.activeSettingsScope
    try {
      await f.user.click(screen.getByRole("button", { name: "Apply", exact: true }))
      await act(async () => { await nativeSetupFixture.selectionApply })
      expect(saveActorSettingsMock).toHaveBeenCalledOnce()
      expect(sceneBucket.records.get(f.key)).toEqual(expect.objectContaining({
        value: { ...f.original.value, notes: "Staged connected Scene" }
      }))
      expect(f.source.chatDraftAddress).toBe(f.address)
      expect(f.source.chatDraftRevision).toBe(f.revision)
      expect(f.source.serverChatId).toBe("native-current")
      expect(f.source.selectedModel).toBeNull()
      expect(nativeSetupFixture.settings!.activeSettingsScope).toBe(initialScope)
      expect(f.reset).not.toHaveBeenCalled()
      expect(nativeSetupFixture.snapshot).not.toHaveBeenCalled()
      expect(selectedAssistantMock.setSelectedAssistant).not.toHaveBeenCalled()
    } finally {
      await Promise.allSettled(nativeSetupFixture.drawerApplications)
      f.view.unmount()
      nativeSetupFixture.models = originalModels
    }
  })

  it.each(["native", "fresh", "neutral fresh"] as const)(
    "Form Drawer action ownership admits one unchanged hydrated %s Apply",
    async (kind) => {
      const f = await openActualSceneDrawer(kind)
      try {
        await f.user.click(
          screen.getByRole("button", { name: "Apply", exact: true })
        )
        await act(async () => {
          await nativeSetupFixture.selectionApply
        })
        expect(saveActorSettingsMock).toHaveBeenCalledOnce()
        expect(sceneBucket.records.get(f.key)).toEqual(
          expect.objectContaining({
            value: { ...f.original.value, notes: "Staged connected Scene" }
          })
        )
        expect(useActorStore.getState().applied?.target.key).toBe(f.target.key)
        expect(useActorStore.getState().applied?.target.isCurrent()).toBe(true)
        expect(
          screen.queryByTestId("role-play-setup-drawer")
        ).not.toBeInTheDocument()
        expect(f.source.chatDraftAddress).toBe(f.address)
        expect(f.source.chatDraftRevision).toBe(f.revision)
        expect(f.reset).not.toHaveBeenCalled()
        expect(onSubmitMock).not.toHaveBeenCalled()
      } finally {
        await Promise.allSettled(nativeSetupFixture.drawerApplications)
        f.view.unmount()
      }
    }
  )

  it.each(["uninterrupted", "same-value model revision"] as const)(
    "Form manual Scene CAS preserves its original selected-model lease after behavior settles (%s)",
    async (retirement) => {
      nativeSetupFixture.realSettings = true
      const modelStore = nativeSetupFixture.actualSettingsStore!
      modelStore.getState().setActiveSettingsScope("openai:deepseek-chat")
      modelStore
        .getState()
        .updateSettings({
          apiProvider: "openai",
          systemPrompt: "Original manual behavior",
          temperature: 0.4
        })
      const entrySettings = modelStore.getState()
      const f = await openActualSceneDrawer("fresh", { persona: true })
      const originalModel = f.source.selectedModel
      const originalModelRevision = f.source.selectedModelRevision
      const held = sceneSelectionGate<void>()
      let casKey: string | undefined
      let savePromise:
        | ReturnType<
            typeof import("@/services/actor-settings").saveActorSettingsForChat
          >
        | undefined
      // The hydrated read has already finished. Only physical CAS IO is held;
      // the real Drawer, Form, Scene writer and bucket lease remain connected.
      sceneBucket.get.mockImplementationOnce(async (key) => {
        casKey = key
        await held.promise
        const record = sceneBucket.records.get(key)
        return record === undefined ? undefined : structuredClone(record)
      })
      try {
        expect(
          screen.getByRole("button", { name: "Clear behavior" })
        ).toBeEnabled()
        await f.user.click(screen.getByRole("button", { name: "Clear behavior" }))
        await f.user.click(
          screen.getByRole("button", { name: "Apply", exact: true })
        )
        await waitFor(() => expect(casKey).toBe(f.key))
        expect(saveActorSettingsMock).toHaveBeenCalledOnce()
        const save = saveActorSettingsMock.mock.calls[0][0]
        savePromise = saveActorSettingsMock.mock.results[0].value
        expect(await nativeSetupFixture.selectionApply).toBe(true)
        const settledSettings = modelStore.getState()
        expect(settledSettings).not.toBe(entrySettings)
        expect(entrySettings.systemPrompt).toBe("Original manual behavior")
        expect(settledSettings).toEqual(
          expect.objectContaining({
            activeSettingsScope: "openai:deepseek-chat",
            apiProvider: "openai",
            systemPrompt: "",
            systemPromptTemplateId: undefined,
            temperature: 0.4
          })
        )
        expect(save.target?.key).toBe(f.target.key)
        expect(save.target?.isCurrent()).toBe(true)
        expect(save.expectedRecord).toEqual(f.original)
        expect(save.settings).toEqual({
          ...f.original.value,
          notes: "Staged connected Scene"
        })
        expect(save.historyId).toBeNull()
        expect(save.serverChatId).toBeNull()
        expect(save.validate_lease?.()).toBe(true)
        expect(sceneBucket.set).not.toHaveBeenCalled()
        expect(useActorStore.getState().applied).toBeNull()
        if (retirement === "same-value model revision") {
          act(() => f.source.setSelectedModel(originalModel))
          expect(f.source.selectedModel).toBe(originalModel)
          expect(f.source.selectedModelRevision).toBe(originalModelRevision + 1)
          expect(modelStore.getState()).toBe(settledSettings)
          expect(save.target?.isCurrent()).toBe(true)
          // The accepted Scene owner remains valid; this old command's exact
          // physical save lease must retire before the bucket can commit.
          expect(save.validate_lease?.()).toBe(false)
        }
        let saved: boolean | undefined
        await act(async () => {
          held.resolve(undefined)
          saved = await savePromise
        })
        expect(saved).toBe(retirement === "uninterrupted")
        if (retirement === "uninterrupted") {
          await waitFor(() =>
            expect(useActorStore.getState().applied?.target.key).toBe(
              f.target.key
            )
          )
          expect(sceneBucket.set).toHaveBeenCalledExactlyOnceWith(
            f.key,
            expect.objectContaining({ value: save.settings })
          )
          expect(useActorStore.getState().applied?.target.isCurrent()).toBe(true)
          expect(
            screen.queryByTestId("role-play-setup-drawer")
          ).not.toBeInTheDocument()
        } else {
          expect(sceneBucket.records.get(f.key)).toBe(f.original)
          expect(sceneBucket.set).not.toHaveBeenCalled()
          expect(useActorStore.getState().applied).toBeNull()
          expect(
            screen.getByRole("textbox", { name: "Scene notes" })
          ).toHaveValue("Staged connected Scene")
          expect(screen.getByTestId("role-play-setup-drawer")).toBeInTheDocument()
        }
        expect(modelStore.getState()).toBe(settledSettings)
        expect(f.source.chatDraftAddress).toBe(f.address)
        expect(f.source.chatDraftRevision).toBe(f.revision)
        expect(f.source.historyId).toBeNull()
        expect(f.source.serverChatId).toBeNull()
        expect(f.reset).not.toHaveBeenCalled()
        expect(f.source.setHistoryId).not.toHaveBeenCalled()
        expect(f.source.setServerChatId).not.toHaveBeenCalled()
        expect(f.source.beginSelectedModelSave).not.toHaveBeenCalled()
        expect(nativeSetupFixture.actor).not.toHaveBeenCalled()
        expect(onSubmitMock).not.toHaveBeenCalled()
        expect(createChatCompletionMock).not.toHaveBeenCalled()
      } finally {
        f.view.unmount()
        held.resolve(undefined)
        await Promise.allSettled([
          savePromise,
          ...nativeSetupFixture.drawerApplications
        ])
      }
    }
  )

  it("Form Drawer action ownership refuses duplicate delivery of its exact action-start acknowledgement", async () => {
    const f = await openActualSceneDrawer()
    nativeSetupFixture.repeatActionStarted = true
    try {
      await f.user.click(
        screen.getByRole("button", { name: "Apply", exact: true })
      )
      await act(async () => {
        await nativeSetupFixture.selectionApply
      })
      expect(nativeSetupFixture.actionAcknowledgements).toEqual([true, false])
      expect(saveActorSettingsMock).toHaveBeenCalledOnce()
      expect(useActorStore.getState().applied?.target.key).toBe(f.target.key)
      expect(f.source.chatDraftAddress).toBe(f.address)
      expect(f.source.chatDraftRevision).toBe(f.revision)
    } finally {
      await Promise.allSettled(nativeSetupFixture.drawerApplications)
      f.view.unmount()
    }
  })

  it.each(["before Apply", "during preparation"] as const)(
    "Form Drawer action ownership retires on an external same-address action %s",
    async (boundary) => {
      const f = await openActualSceneDrawer("fresh")
      const originalWait =
        sceneSelectionGate<
          Awaited<ReturnType<typeof nativeSetupFixture.snapshot>>
        >()
      const externalWait =
        sceneSelectionGate<
          Awaited<ReturnType<typeof nativeSetupFixture.snapshot>>
        >()
      const snapshot = {
        scopeSignal: new AbortController().signal,
        scopeInvalidatedSignal: new AbortController().signal,
        requestScope: f.requestScope,
        release: vi.fn()
      }
      nativeSetupFixture.snapshot.mockReturnValueOnce(
        boundary === "before Apply" ? externalWait.promise : originalWait.promise
      )
      if (boundary === "during preparation")
        nativeSetupFixture.snapshot.mockReturnValueOnce(externalWait.promise)
      let external: Promise<boolean | void> | undefined
      try {
        if (boundary === "during preparation") {
          await f.user.click(
            screen.getByRole("button", { name: "Apply", exact: true })
          )
          await waitFor(() =>
            expect(nativeSetupFixture.snapshot).toHaveBeenCalledOnce()
          )
        }
        const callback = nativeSetupFixture.drawerProps?.onApply
        if (!callback)
          throw new Error("fixture requires the current actual Form callback")
        act(() => {
          external = Promise.resolve(
            callback({
              identitySelection: {
                kind: "character",
                id: "char-new",
                name: "New character",
                metadata: { selectionMode: "tracked" }
              },
              sceneSettings: createDefaultActorSettings()
            })
          )
        })
        await waitFor(() =>
          expect(nativeSetupFixture.snapshot).toHaveBeenCalledTimes(
            boundary === "before Apply" ? 1 : 2
          )
        )
        expect(f.source.chatDraftAddress).toBe(f.address)
        expect(f.source.chatDraftRevision).toBe(f.revision)
        if (boundary === "before Apply") {
          await f.user.click(
            screen.getByRole("button", { name: "Apply", exact: true })
          )
          expect(nativeSetupFixture.drawerApplications).toHaveLength(0)
        } else {
          await act(async () => {
            originalWait.resolve(snapshot)
            await nativeSetupFixture.selectionApply
          })
          expect(await nativeSetupFixture.selectionApply).toBe(false)
        }
        expect(saveActorSettingsMock).not.toHaveBeenCalled()
        expect(selectedAssistantMock.setSelectedAssistant).not.toHaveBeenCalled()
        expect(sceneBucket.records.get(f.key)).toBe(f.original)
        expect(useActorStore.getState().applied).toBeNull()
        expect(f.reset).not.toHaveBeenCalled()
        expect(f.source.setHistoryId).not.toHaveBeenCalled()
        expect(f.source.setServerChatId).not.toHaveBeenCalled()
      } finally {
        f.view.unmount()
        originalWait.resolve(snapshot)
        externalWait.resolve(snapshot)
        await Promise.allSettled([nativeSetupFixture.selectionApply, external])
      }
    }
  )

  it("Form Drawer action ownership permits Retry after its own failed preparation", async () => {
    const f = await openActualSceneDrawer("fresh")
    nativeSetupFixture.snapshot.mockRejectedValueOnce(
      new Error("Declared physical snapshot preparation failure")
    )
    try {
      await f.user.click(
        screen.getByRole("button", { name: "Apply", exact: true })
      )
      await act(async () => {
        await expect(nativeSetupFixture.selectionApply).rejects.toThrow(
          "Declared physical snapshot preparation failure"
        )
      })
      expect(nativeSetupFixture.snapshot).toHaveBeenCalledOnce()
      expect(saveActorSettingsMock).not.toHaveBeenCalled()
      expect(await screen.findByRole("alert")).toHaveTextContent(
        "Settings could not be applied"
      )
      await f.user.click(
        screen.getByRole("button", { name: "Apply", exact: true })
      )
      await act(async () => {
        await nativeSetupFixture.selectionApply
      })
      expect(nativeSetupFixture.snapshot).toHaveBeenCalledTimes(2)
      expect(saveActorSettingsMock).toHaveBeenCalledOnce()
      expect(sceneBucket.records.get(f.key)).toEqual(
        expect.objectContaining({
          value: expect.objectContaining({ notes: "Staged connected Scene" })
        })
      )
      expect(useActorStore.getState().applied?.target.key).toBe(f.target.key)
      expect(
        screen.queryByTestId("role-play-setup-drawer")
      ).not.toBeInTheDocument()
    } finally {
      await Promise.allSettled(nativeSetupFixture.drawerApplications)
      f.view.unmount()
    }
  })

  it.each([
    "no active route",
    "owned restore receipt",
    "owned receipt then external restore"
  ] as const)(
    "Form route receipt ownership preserves native selection with %s",
    async (route) => {
      const f = prepareQualifiedSceneSource()
      const originalHref = window.location.href
      window.history.replaceState(
        {},
        "",
        route === "no active route"
          ? "/chat"
          : "/chat?mode=character&characterId=char-new&chatId=native-current"
      )
      const originalRestore = usePlaygroundSessionStore.getState().restoreRevision
      const receipt =
        route === "no active route"
          ? null
          : installRouteReceiptListener(
              route === "owned receipt then external restore"
            )
      const user = userEvent.setup()
      const view = render(<PlaygroundForm droppedFiles={[]} />)
      try {
        await user.click(
          screen.getByRole("button", { name: "Open role-play setup" })
        )
        await user.click(
          await screen.findByRole("button", { name: "Apply Character setup" })
        )
        let accepted: boolean | void
        await act(async () => {
          accepted = await nativeSetupFixture.selectionApply
        })
        expect(accepted!).toBe(route !== "owned receipt then external restore")
        if (receipt) {
          expect(receipt.events).toHaveLength(1)
          expect(receipt.events[0]).toEqual(
            expect.objectContaining({
              href: window.location.href,
              serverChatId: f.source.serverChatId,
              historyId: f.source.historyId,
              restoreRevision: originalRestore
            })
          )
          expect(receipt.acknowledgements).toEqual([true])
          expect(usePlaygroundSessionStore.getState().restoreRevision).toBe(
            originalRestore +
              (route === "owned receipt then external restore" ? 2 : 1)
          )
        } else
          expect(usePlaygroundSessionStore.getState().restoreRevision).toBe(
            originalRestore
          )
        expect(f.source.chatDraftAddress).toBe(f.address)
        expect(f.source.chatDraftRevision).toBe(f.revision)
        expect(f.reset).not.toHaveBeenCalled()
        expect(f.source.setHistoryId).not.toHaveBeenCalled()
        expect(f.source.setServerChatId).not.toHaveBeenCalled()
        expect(saveActorSettingsMock).not.toHaveBeenCalled()
        expect(onSubmitMock).not.toHaveBeenCalled()
      } finally {
        receipt?.remove()
        await Promise.allSettled([nativeSetupFixture.selectionApply])
        view.unmount()
        window.history.replaceState({}, "", originalHref)
      }
    }
  )

  it.each(["owned receipt", "later external restore"] as const)(
    "Form saved settings preserves the hydrated Drawer without replacing its route: %s",
    async (retirement) => {
      const originalHref = window.location.href
      window.history.replaceState(
        {},
        "",
        "/chat?mode=character&characterId=char-new&chatId=native-current"
      )
      const f = await openActualSceneDrawer()
      const originalRestore = usePlaygroundSessionStore.getState().restoreRevision
      const receipt = installRouteReceiptListener(false)
      const held = sceneSelectionGate<void>()
      let observedSave: Promise<boolean> | undefined
      let saveParams:
        | Parameters<NonNullable<typeof nativeSetupFixture.actualSceneWriter>>[0]
        | undefined
      const leasesAtSave: boolean[] = []
      saveActorSettingsMock.mockImplementationOnce((params) => {
        saveParams = params
        leasesAtSave.push(params.validate_lease?.() === true)
        observedSave = (async () => {
          await held.promise
          return nativeSetupFixture.actualSceneWriter!(params)
        })()
        return observedSave
      })
      try {
        await f.user.click(
          screen.getByRole("button", { name: "Apply", exact: true })
        )
        await waitFor(() => expect(saveActorSettingsMock).toHaveBeenCalledOnce())
        expect(receipt.events).toHaveLength(0)
        expect(receipt.acknowledgements).toEqual([])
        expect(usePlaygroundSessionStore.getState().restoreRevision).toBe(
          originalRestore
        )
        expect(leasesAtSave).toEqual([true])
        expect(saveParams?.target?.key).toBe(f.target.key)
        expect(saveParams?.expectedRecord).toEqual(f.original)
        expect(saveParams?.settings).toEqual({
          ...f.original.value,
          notes: "Staged connected Scene"
        })
        expect(sceneBucket.set).not.toHaveBeenCalled()
        // Allow the actual session publication to rerender the parent and Drawer.
        f.view.rerender(<PlaygroundForm droppedFiles={[]} />)
        await act(async () => {
          await Promise.resolve()
        })
        expect(saveParams?.validate_lease?.()).toBe(true)
        if (retirement === "later external restore") {
          act(() => usePlaygroundSessionStore.getState().cancelPendingRestore())
          f.view.rerender(<PlaygroundForm droppedFiles={[]} />)
          expect(usePlaygroundSessionStore.getState().restoreRevision).toBe(
            originalRestore + 1
          )
          expect(saveParams?.validate_lease?.()).toBe(false)
        }
        let saved: boolean | undefined
        await act(async () => {
          held.resolve(undefined)
          saved = await observedSave
        })
        expect(saved).toBe(retirement === "owned receipt")
        if (retirement === "owned receipt") {
          await waitFor(() =>
            expect(useActorStore.getState().applied?.target.key).toBe(
              f.target.key
            )
          )
          expect(useActorStore.getState().applied?.target.isCurrent()).toBe(true)
          expect(sceneBucket.set).toHaveBeenCalledExactlyOnceWith(
            f.key,
            expect.objectContaining({ value: saveParams!.settings })
          )
          expect(
            screen.queryByTestId("role-play-setup-drawer")
          ).not.toBeInTheDocument()
        } else {
          expect(sceneBucket.records.get(f.key)).toBe(f.original)
          expect(sceneBucket.set).not.toHaveBeenCalled()
          expect(useActorStore.getState().applied).toBeNull()
        }
        expect(f.source.chatDraftAddress).toBe(f.address)
        expect(f.source.chatDraftRevision).toBe(f.revision)
        expect(f.reset).not.toHaveBeenCalled()
        expect(nativeSetupFixture.actor).not.toHaveBeenCalled()
        expect(onSubmitMock).not.toHaveBeenCalled()
      } finally {
        receipt.remove()
        f.view.unmount()
        held.resolve(undefined)
        await Promise.allSettled([
          observedSave,
          nativeSetupFixture.selectionApply
        ])
        window.history.replaceState({}, "", originalHref)
      }
    }
  )

  it.each(["owned receipt", "later external restore"] as const)(
    "Form route receipt ownership carries startup's nested source and intended model through %s",
    async (retirement) => {
      const f = prepareQualifiedSceneSource()
      nativeSetupFixture.realSettings = true
      const modelStore = nativeSetupFixture.actualSettingsStore!
      modelStore.getState().setActiveSettingsScope("openai:deepseek-chat")
      modelStore
        .getState()
        .updateSettings({ apiProvider: "openai", temperature: 0, topP: 0 })
      const original: ActorSceneRecord = {
        value: {
          ...createDefaultActorSettings(),
          notes: "Original nested startup Scene"
        },
        updatedAt: 1730000000000
      }
      const proposal: ActorSettings = {
        ...createDefaultActorSettings(),
        notes: "Nested receipt startup proposal"
      }
      const key = `actorSettings:owned:${f.target.key}`
      sceneBucket.records.set(key, original)
      savedSetupFixture.raw = JSON.stringify([
        { ...ownedStartupBundle(proposal), selectedModel: "openai:gpt-4.1" }
      ])
      const originalHref = window.location.href
      window.history.replaceState(
        {},
        "",
        "/chat?mode=character&characterId=char-new&chatId=native-current"
      )
      const originalRestore = usePlaygroundSessionStore.getState().restoreRevision
      const receipt = installRouteReceiptListener(false)
      const held = sceneSelectionGate<void>()
      let observedSave: Promise<boolean> | undefined
      let saveParams:
        | Parameters<NonNullable<typeof nativeSetupFixture.actualSceneWriter>>[0]
        | undefined
      const leasesAtSave: boolean[] = []
      saveActorSettingsMock.mockImplementationOnce((params) => {
        saveParams = params
        leasesAtSave.push(params.validate_lease?.() === true)
        observedSave = (async () => {
          await held.promise
          return nativeSetupFixture.actualSceneWriter!(params)
        })()
        return observedSave
      })
      const { view, apply } = await openOwnedStartupPreview()
      let application: Promise<unknown> | undefined
      try {
        act(() => {
          application = apply()
        })
        await waitFor(() => expect(saveActorSettingsMock).toHaveBeenCalledOnce())
        expect(receipt.events).toHaveLength(1)
        expect(receipt.events[0].restoreRevision).toBe(originalRestore)
        expect(receipt.acknowledgements).toEqual([true])
        expect(usePlaygroundSessionStore.getState().restoreRevision).toBe(
          originalRestore + 1
        )
        expect(leasesAtSave).toEqual([true])
        expect(saveParams?.target?.key).toBe(f.target.key)
        expect(saveParams?.expectedRecord).toEqual(original)
        expect(saveParams?.settings).toEqual(proposal)
        expect(f.source.selectedModel).toBe("openai:gpt-4.1")
        expect(modelStore.getState().activeSettingsScope).toBe("openai:gpt-4.1")
        expect(sceneBucket.set).not.toHaveBeenCalled()
        view.rerender(<PlaygroundForm droppedFiles={[]} />)
        await act(async () => {
          await Promise.resolve()
        })
        expect(saveParams?.validate_lease?.()).toBe(true)
        if (retirement === "later external restore") {
          act(() => usePlaygroundSessionStore.getState().cancelPendingRestore())
          view.rerender(<PlaygroundForm droppedFiles={[]} />)
          expect(usePlaygroundSessionStore.getState().restoreRevision).toBe(
            originalRestore + 2
          )
          expect(saveParams?.validate_lease?.()).toBe(false)
        }
        let accepted: unknown
        await act(async () => {
          held.resolve(undefined)
          accepted = await application
        })
        expect(accepted).toBe(retirement === "owned receipt")
        if (retirement === "owned receipt") {
          expect(sceneBucket.set).toHaveBeenCalledExactlyOnceWith(
            key,
            expect.objectContaining({ value: proposal })
          )
          expect(useActorStore.getState().applied?.target.key).toBe(f.target.key)
          expect(useActorStore.getState().applied?.target.isCurrent()).toBe(true)
          expect(
            screen.queryByRole("button", { name: "Apply saved template" })
          ).not.toBeInTheDocument()
        } else {
          expect(sceneBucket.records.get(key)).toBe(original)
          expect(sceneBucket.set).not.toHaveBeenCalled()
          expect(useActorStore.getState().applied).toBeNull()
          expect(
            screen.getByRole("button", { name: "Apply saved template" })
          ).toBeInTheDocument()
        }
        expect(f.source.chatDraftAddress).toBe(f.address)
        expect(f.source.chatDraftRevision).toBe(f.revision)
        expect(f.source.serverChatId).toBe("native-current")
        expect(f.source.historyId).toBe("native-current-mirror")
        expect(f.reset).not.toHaveBeenCalled()
        expect(nativeSetupFixture.scene).toHaveBeenCalledOnce()
        expect(nativeSetupFixture.actor).not.toHaveBeenCalled()
        expect(onSubmitMock).not.toHaveBeenCalled()
      } finally {
        receipt.remove()
        view.unmount()
        held.resolve(undefined)
        await Promise.allSettled([application, observedSave])
        window.history.replaceState({}, "", originalHref)
      }
    }
  )

  it.each([false, true])(
    "Form route receipt ownership keeps replacement's exact R+2 after owned cancellation (external clear-session subscriber: %s)",
    async (externalRestore) => {
      const f = prepareQualifiedSceneSource()
      f.source.serverChatCharacterId = "departing-character"
      f.source.serverChatAssistantId = "departing-character"
      const originalHref = window.location.href
      window.history.replaceState(
        {},
        "",
        "/chat?mode=character&characterId=departing-character&chatId=native-current"
      )
      const originalRestore = usePlaygroundSessionStore.getState().restoreRevision
      const receipt = installRouteReceiptListener(false)
      let sawClear = false
      const unsubscribe = usePlaygroundSessionStore.subscribe((state) => {
        if (sawClear || state.restoreRevision !== originalRestore + 2) return
        sawClear = true
        if (externalRestore) state.cancelPendingRestore()
      })
      const user = userEvent.setup()
      const view = render(<PlaygroundForm droppedFiles={[]} />)
      try {
        await user.click(
          screen.getByRole("button", { name: "Open role-play setup" })
        )
        await user.click(
          await screen.findByRole("button", {
            name: "Apply Character neutral Scene"
          })
        )
        let accepted: boolean | void
        await act(async () => {
          accepted = await nativeSetupFixture.selectionApply
        })
        expect(receipt.events).toHaveLength(1)
        expect(receipt.events[0]).toEqual(
          expect.objectContaining({
            restoreRevision: originalRestore,
            serverChatId: "native-current",
            historyId: "native-current-mirror"
          })
        )
        expect(receipt.acknowledgements).toEqual([true])
        expect(sawClear).toBe(true)
        expect(usePlaygroundSessionStore.getState().restoreRevision).toBe(
          originalRestore + (externalRestore ? 3 : 2)
        )
        expect(accepted!).toBe(!externalRestore)
        expect(f.source.chatDraftAddress).not.toBe(f.address)
        expect(f.source.chatDraftAddress.kind).toBe("fresh")
        expect(f.source.chatDraftRevision).toBe(f.revision + 1)
        expect(f.reset).toHaveBeenCalledOnce()
        if (externalRestore) {
          expect(nativeSetupFixture.destinationCurrent).toBeNull()
          expect(f.source.historyId).toBe("native-current-mirror")
          expect(f.source.serverChatId).toBe("native-current")
          expect(f.source.setHistoryId).not.toHaveBeenCalled()
          expect(f.source.setServerChatId).not.toHaveBeenCalled()
        } else {
          expect(nativeSetupFixture.destinationCurrent?.()).toBe(true)
          expect(f.source.historyId).toBeNull()
          expect(f.source.serverChatId).toBeNull()
        }
        expect(saveActorSettingsMock).not.toHaveBeenCalled()
        expect(nativeSetupFixture.actor).not.toHaveBeenCalled()
        expect(onSubmitMock).not.toHaveBeenCalled()
      } finally {
        receipt.remove()
        unsubscribe()
        view.unmount()
        await Promise.allSettled([nativeSetupFixture.selectionApply])
        window.history.replaceState({}, "", originalHref)
      }
    }
  )

  it.each([
    "uninterrupted",
    "first behavior setter",
    "behavior settings setter"
  ] as const)(
    "Form role-play continuation fences behavior and generation after %s",
    async (boundary) => {
      const f = prepareQualifiedSceneSource("fresh")
      f.source.serverChatAssistantKind = null
      f.source.serverChatAssistantId = null
      f.source.serverChatCharacterId = null
      nativeSetupFixture.realRolePlayPresets = true
      const settings = nativeSetupFixture.settings!
      const originalSystemPromptWriter =
        chatModelSettingsState.setSystemPrompt.getMockImplementation()!
      let foreign:
        | {
            address: ChatDraftAddress
            revision: number
            history: MessageOptionState["history"]
            messages: MessageOptionState["messages"]
            quickPromptRevision: number
          }
        | undefined
      const replace = () => {
        f.source.startNewChatDraft(false)
        f.source.history = [{ role: "user", content: "Foreign transcript" }]
        f.source.messages = [
          {
            id: "foreign-message",
            isBot: false,
            name: "You",
            message: "Foreign presentation",
            sources: []
          }
        ]
        f.source.selectedSystemPrompt = "foreign-template"
        f.source.setSelectedQuickPrompt("foreign-quick")
        settings.systemPrompt = "Foreign behavior"
        settings.systemPromptTemplateId = "foreign-template"
        settings.temperature = 1.7
        settings.topP = 0.34
        settings.topK = 77
        foreign = {
          address: f.source.chatDraftAddress,
          revision: f.source.chatDraftRevision,
          history: f.source.history,
          messages: f.source.messages,
          quickPromptRevision: f.source.quickPromptRevision
        }
      }
      chatModelSettingsState.setSystemPrompt.mockImplementation((prompt) => {
        originalSystemPromptWriter(prompt)
        if (boundary === "first behavior setter") replace()
      })
      chatModelSettingsState.updateSettings.mockImplementation(
        (updates: Partial<import("@/store/model").ChatModelSettings>) => {
          Object.assign(settings, updates)
          if (
            boundary === "behavior settings setter" &&
            "systemPromptTemplateId" in updates
          )
            replace()
        }
      )
      const user = userEvent.setup()
      const view = render(<PlaygroundForm droppedFiles={[]} />)
      try {
        await user.click(
          screen.getByRole("button", { name: "Open role-play setup" })
        )
        await user.click(
          await screen.findByRole("button", {
            name: "Apply overlay behavior and generation setup"
          })
        )
        let accepted: boolean | void
        await act(async () => {
          accepted = await nativeSetupFixture.selectionApply
        })
        expect(
          chatModelSettingsState.setSystemPrompt
        ).toHaveBeenCalledExactlyOnceWith("Original behavior proposal")
        if (boundary === "uninterrupted") {
          expect(accepted!).toBe(true)
          expect(chatModelSettingsState.updateSettings).toHaveBeenCalledTimes(2)
          expect(settings).toEqual(
            expect.objectContaining({
              systemPrompt: "Original behavior proposal",
              systemPromptTemplateId: "guarded-behavior",
              temperature: 0.2,
              topP: 0.8,
              topK: 20
            })
          )
          expect(
            f.source.setSelectedSystemPrompt
          ).toHaveBeenCalledExactlyOnceWith(undefined)
          expect(f.source.setSelectedQuickPrompt).toHaveBeenCalledExactlyOnceWith(
            null
          )
          expect(f.source.chatDraftAddress).toBe(f.address)
        } else {
          expect(foreign).toBeDefined()
          expect(accepted!).toBe(false)
          expect(chatModelSettingsState.updateSettings).toHaveBeenCalledTimes(
            boundary === "first behavior setter" ? 0 : 1
          )
          expect(f.source.setSelectedSystemPrompt).not.toHaveBeenCalled()
          // The new composer may acquire its own command once. Old Apply must
          // never use the unadorned clear that bypasses command ownership.
          expect(f.source.setSelectedQuickPrompt.mock.calls).toEqual([
            ["foreign-quick"],
            [null, { expectedRevision: foreign!.quickPromptRevision }]
          ])
          expect(f.source.chatDraftAddress).toBe(foreign!.address)
          expect(f.source.chatDraftRevision).toBe(foreign!.revision)
          expect(f.source.history).toBe(foreign!.history)
          expect(f.source.messages).toBe(foreign!.messages)
          expect(f.source.selectedSystemPrompt).toBe("foreign-template")
          expect(f.source.selectedQuickPrompt).toBeNull()
          expect(f.source.quickPromptRevision).toBe(
            foreign!.quickPromptRevision + 1
          )
          expect(screen.getByLabelText("Prompt preview")).toHaveTextContent(
            "foreign-quick"
          )
          expect(settings).toEqual(
            expect.objectContaining({
              systemPrompt: "Foreign behavior",
              systemPromptTemplateId: "foreign-template",
              temperature: 1.7,
              topP: 0.34,
              topK: 77
            })
          )
        }
        expect(selectedAssistantMock.setSelectedAssistant).toHaveBeenCalledOnce()
        expect(saveActorSettingsMock).not.toHaveBeenCalled()
        expect(onSubmitMock).not.toHaveBeenCalled()
      } finally {
        await Promise.allSettled([nativeSetupFixture.selectionApply])
        view.unmount()
      }
    }
  )

  it.each([false, true])(
    "Form Temporary ownership applies a disabled Scene through the actual Drawer (nonempty: %s)",
    async (nonempty) => {
      // The account-owned preference is already this exact tracked character;
      // Temporary conversion must keep it without issuing H1 capabilities.
      selectedAssistantMock.initialSelection = {
        kind: "character",
        id: "char-new",
        name: "New character",
        metadata: { selectionMode: "tracked" }
      }
      const f = await openActualSceneDrawer("fresh", {
        temporary: true,
        nonempty
      })
      const originalHistory = f.source.history
      const originalMessages = f.source.messages
      let issued: { address: ChatDraftAddress; revision: number } | undefined
      nativeSetupFixture.onDraftPublication = (address) => {
        issued = { address, revision: f.source.chatDraftRevision }
      }
      try {
        expect(f.source.temporaryChat).toBe(true)
        expect(f.address).toEqual(
          expect.objectContaining({ kind: "fresh", temporary: true })
        )
        expect(f.original.value.isEnabled).toBe(false)
        await f.user.click(
          screen.getByRole("button", { name: "Apply", exact: true })
        )
        if (nonempty) {
          await act(async () => {
            await expect(nativeSetupFixture.selectionApply).rejects.toThrow(
              "temporary_history_unavailable"
            )
          })
          expect(await screen.findByRole("alert")).toHaveTextContent(
            "Start a new saved chat before applying this character"
          )
          expect(f.source.chatDraftAddress).toBe(f.address)
          expect(f.source.chatDraftRevision).toBe(f.revision)
          expect(f.source.temporaryChat).toBe(true)
          expect(f.source.history).toBe(originalHistory)
          expect(f.source.messages).toBe(originalMessages)
          expect(issued).toBeUndefined()
          expect(f.source.setChatDraftAddress).not.toHaveBeenCalled()
          expect(f.source.setTemporaryChat).not.toHaveBeenCalled()
          expect(f.reset).not.toHaveBeenCalled()
          expect(nativeSetupFixture.snapshot).not.toHaveBeenCalled()
          expect(saveActorSettingsMock).not.toHaveBeenCalled()
          expect(sceneBucket.set).not.toHaveBeenCalled()
          expect(useActorStore.getState().applied).toBeNull()
        } else {
          await act(async () => {
            await nativeSetupFixture.selectionApply
          })
          expect(saveActorSettingsMock).toHaveBeenCalledOnce()
          expect(issued).toBeDefined()
          const destination = issued!
          const save = saveActorSettingsMock.mock.calls[0][0]
          expect(f.source.chatDraftAddress).toBe(destination.address)
          expect(f.source.chatDraftRevision).toBe(destination.revision)
          expect(destination.revision).toBe(f.revision + 1)
          expect(destination.address).toEqual(
            expect.objectContaining({ kind: "fresh", temporary: false })
          )
          expect(destination.address).not.toBe(f.address)
          expect(f.source.temporaryChat).toBe(false)
          expect(f.source.history).toEqual([])
          expect(f.source.messages).toEqual([])
          expect(f.source.setChatDraftAddress).toHaveBeenCalledExactlyOnceWith(
            destination.address,
            { sourceAddress: f.address, expectedRevision: f.revision }
          )
          expect(f.source.setTemporaryChat).toHaveBeenCalledExactlyOnceWith(false)
          expect(f.reset).toHaveBeenCalledOnce()
          expect(save.target?.address).toEqual(destination.address)
          expect(save.target?.key).not.toBe(f.target.key)
          expect(save.target?.account?.requestScope).toEqual(f.requestScope)
          expect(save.expectedRecord).toBeNull()
          expect(save.settings).toEqual({
            ...f.original.value,
            notes: "Staged connected Scene"
          })
          expect(sceneBucket.records.get(f.key)).toBe(f.original)
          expect(sceneBucket.set).toHaveBeenCalledExactlyOnceWith(
            `actorSettings:owned:${save.target!.key}`,
            expect.objectContaining({ value: save.settings })
          )
          expect(useActorStore.getState().applied?.target.key).toBe(
            save.target?.key
          )
          expect(useActorStore.getState().applied?.target.isCurrent()).toBe(true)
          expect(
            screen.queryByTestId("role-play-setup-drawer")
          ).not.toBeInTheDocument()
        }
        expect(f.source.historyId).toBeNull()
        expect(f.source.serverChatId).toBeNull()
        expect(nativeSetupFixture.actor).not.toHaveBeenCalled()
        expect(selectedAssistantMock.setSelectedAssistant).not.toHaveBeenCalled()
        expect(onSubmitMock).not.toHaveBeenCalled()
      } finally {
        nativeSetupFixture.onDraftPublication = null
        f.view.unmount()
        await Promise.allSettled(nativeSetupFixture.drawerApplications)
      }
    }
  )

  it.each([
    "uninterrupted",
    "same-model settings edit",
    "same-value model revision",
    "tool choice",
    "MCP state replacement"
  ] as const)(
    "Startup Scene first read retains the original settings, model and tool command (%s)",
    async (retirement) => {
      const f = prepareQualifiedSceneSource("fresh")
      Object.assign(f.source, {
        serverChatAssistantKind: null,
        serverChatAssistantId: null,
        serverChatCharacterId: null
      })
      nativeSetupFixture.realSettings = true
      const store = nativeSetupFixture.actualSettingsStore!
      store.getState().setActiveSettingsScope("openai:deepseek-chat")
      store
        .getState()
        .updateSettings({
          apiProvider: "openai",
          temperature: 0.4,
          topP: 0.7,
          systemPrompt: "Original first-read behavior"
        })
      const original: ActorSceneRecord = {
        value: {
          ...createDefaultActorSettings(),
          notes: "Original first-read Scene"
        },
        updatedAt: 1730000000000
      }
      const proposal: ActorSettings = {
        ...createDefaultActorSettings(),
        notes: "Frozen first-read proposal"
      }
      const key = `actorSettings:owned:${f.target.key}`
      sceneBucket.records.set(key, original)
      const bundle = ownedStartupBundle(proposal, null)
      bundle.selectedModel = f.source.selectedModel
      if (!bundle.rolePlay)
        throw new Error("fixture requires the frozen role-play proposal")
      bundle.rolePlay.behavior = {
        source: "custom",
        templateId: null,
        templateTitle: null,
        templateCategory: null,
        systemPrompt: "Frozen first-read behavior proposal",
        modified: false
      }
      savedSetupFixture.raw = JSON.stringify([bundle])
      const held = sceneSelectionGate<void>()
      let readEntered = false
      // Hold actual rich Scene IO, before the hook factory or any identity/model
      // publication. The original service and conditional bucket write remain real.
      sceneBucket.get.mockImplementationOnce(async (readKey) => {
        expect(readKey).toBe(key)
        readEntered = true
        await held.promise
        return structuredClone(original)
      })
      const { view, apply } = await openOwnedStartupPreview()
      const entrySettings = store.getState()
      const entryModel = f.source.selectedModel
      const entryRevision: number = f.source.selectedModelRevision
      const entryToolChoice: MessageOptionState["toolChoice"] =
        f.source.toolChoice
      const entryTools = nativeSetupFixture.tools
      let retainedTools = entryTools
      let retainedSettings = entrySettings
      f.source.setToolChoice.mockImplementation(
        (choice: MessageOptionState["toolChoice"]) => {
          f.source.toolChoice = choice
        }
      )
      let application: Promise<unknown> | undefined
      try {
        expect(entrySettings.activeSettingsScope).toBe("openai:deepseek-chat")
        expect(entrySettings.systemPrompt).toBe("Original first-read behavior")
        act(() => {
          application = apply()
        })
        await waitFor(() => expect(readEntered).toBe(true))
        expect(nativeSetupFixture.scene).toHaveBeenCalledOnce()
        expect(nativeSetupFixture.scene.mock.calls[0][0].target?.key).toBe(
          f.target.key
        )
        expect(nativeSetupFixture.scene.mock.calls[0][0].validate_lease?.()).toBe(
          true
        )
        expect(selectedAssistantMock.setSelectedAssistant).not.toHaveBeenCalled()
        expect(f.source.beginSelectedModelSave).not.toHaveBeenCalled()
        expect(saveActorSettingsMock).not.toHaveBeenCalled()
        expect(sceneBucket.set).not.toHaveBeenCalled()
        if (retirement === "same-model settings edit") {
          act(() =>
            store
              .getState()
              .updateSettings({
                systemPrompt: "Foreign first-read behavior",
                temperature: 1.5,
                topP: 0.31
              })
          )
          retainedSettings = store.getState()
          expect(retainedSettings).not.toBe(entrySettings)
          expect(entrySettings.systemPrompt).toBe("Original first-read behavior")
          expect(entrySettings.temperature).toBe(0.4)
          view.rerender(<PlaygroundForm droppedFiles={[]} />)
        } else if (retirement === "same-value model revision") {
          act(() => f.source.setSelectedModel(entryModel))
          expect(f.source.selectedModel).toBe(entryModel)
          expect(f.source.selectedModelRevision).toBe(entryRevision + 1)
          expect(store.getState()).toBe(entrySettings)
          view.rerender(<PlaygroundForm droppedFiles={[]} />)
        } else if (retirement === "tool choice") {
          act(() => f.source.setToolChoice("none"))
          expect(f.source.toolChoice).toBe("none")
          expect(store.getState()).toBe(entrySettings)
          view.rerender(<PlaygroundForm droppedFiles={[]} />)
        } else if (retirement === "MCP state replacement") {
          retainedTools = { tools: [{ name: "newer-first-read-tool" }] }
          nativeSetupFixture.tools = retainedTools
          expect(nativeSetupFixture.tools).not.toBe(entryTools)
          expect(store.getState()).toBe(entrySettings)
          view.rerender(<PlaygroundForm droppedFiles={[]} />)
        }
        let accepted: unknown
        await act(async () => {
          held.resolve(undefined)
          accepted = await application
        })
        expect(accepted).toBe(retirement === "uninterrupted")
        if (retirement === "uninterrupted") {
          expect(
            selectedAssistantMock.setSelectedAssistant
          ).toHaveBeenCalledOnce()
          expect(f.source.beginSelectedModelSave).toHaveBeenCalledExactlyOnceWith(
            entryModel
          )
          expect(saveActorSettingsMock).toHaveBeenCalledOnce()
          expect(saveActorSettingsMock.mock.calls[0][0]).toEqual(
            expect.objectContaining({
              target: expect.objectContaining({ key: f.target.key }),
              expectedRecord: original,
              settings: proposal
            })
          )
          expect(sceneBucket.set).toHaveBeenCalledExactlyOnceWith(
            key,
            expect.objectContaining({ value: proposal })
          )
          expect(store.getState().systemPrompt).toBe(
            "Frozen first-read behavior proposal"
          )
          expect(useActorStore.getState().applied?.target.key).toBe(f.target.key)
          expect(
            screen.queryByRole("button", { name: "Apply saved template" })
          ).not.toBeInTheDocument()
        } else {
          expect(
            selectedAssistantMock.setSelectedAssistant
          ).not.toHaveBeenCalled()
          expect(f.source.beginSelectedModelSave).not.toHaveBeenCalled()
          expect(saveActorSettingsMock).not.toHaveBeenCalled()
          expect(sceneBucket.set).not.toHaveBeenCalled()
          expect(sceneBucket.records.get(key)).toBe(original)
          expect(store.getState()).toBe(retainedSettings)
          expect(useActorStore.getState().applied).toBeNull()
          expect(
            screen.getByRole("button", { name: "Apply saved template" })
          ).toBeInTheDocument()
          expect(f.source.selectedModelRevision).toBe(
            entryRevision + (retirement === "same-value model revision" ? 1 : 0)
          )
        }
        expect(f.source.selectedModel).toBe(entryModel)
        expect(f.source.toolChoice).toBe(
          retirement === "tool choice" ? "none" : entryToolChoice
        )
        expect(nativeSetupFixture.tools).toBe(retainedTools)
        expect(f.source.chatDraftAddress).toBe(f.address)
        expect(f.source.chatDraftRevision).toBe(f.revision)
        expect(f.reset).not.toHaveBeenCalled()
        expect(f.source.setHistoryId).not.toHaveBeenCalled()
        expect(nativeSetupFixture.actor).not.toHaveBeenCalled()
        expect(onSubmitMock).not.toHaveBeenCalled()
      } finally {
        view.unmount()
        held.resolve(undefined)
        await Promise.allSettled([application])
      }
    }
  )

  it.each([
    "uninterrupted",
    "same-value model revision",
    "foreign model",
    "tool choice",
    "native MCP state"
  ] as const)(
    "Form fresh publication ownership refuses destructive continuation after %s",
    async (retirement) => {
      const f = prepareQualifiedSceneSource()
      Object.assign(f.source, {
        serverChatAssistantId: "departing-character",
        serverChatCharacterId: "departing-character",
        history: [{ role: "user", content: "Original saved transcript" }],
        messages: [
          {
            id: "source-publication",
            isBot: false,
            name: "You",
            message: "Original saved presentation",
            sources: []
          }
        ]
      })
      nativeSetupFixture.realSettings = true
      const store = nativeSetupFixture.actualSettingsStore!
      store.getState().setActiveSettingsScope("openai:deepseek-chat")
      store
        .getState()
        .updateSettings({ apiProvider: "openai", temperature: 0.4, topP: 0.7 })
      const entrySettings = store.getState()
      const entryModel = f.source.selectedModel
      const entryModelRevision: number = f.source.selectedModelRevision
      const originalHistory: MessageOptionState["history"] = f.source.history
      const originalMessages: MessageOptionState["messages"] = f.source.messages
      const originalServerId: string = f.source.serverChatId
      const originalHistoryId: string = f.source.historyId
      const originalTools = nativeSetupFixture.tools
      let retainedTools = originalTools
      let issued: { address: ChatDraftAddress; revision: number } | undefined
      let armed = true
      f.source.setToolChoice.mockImplementation(
        (choice: MessageOptionState["toolChoice"]) => {
          f.source.toolChoice = choice
        }
      )
      // This is the declared real setter's synchronous post-publication seam;
      // the actual model settings store is immutable and is not recaptured.
      nativeSetupFixture.onDraftPublication = (address) => {
        if (!armed) return
        armed = false
        issued = { address, revision: f.source.chatDraftRevision }
        if (
          retirement === "same-value model revision" ||
          retirement === "foreign model"
        )
          f.source.setSelectedModel(
            retirement === "foreign model" ? "openai:gpt-4.1" : entryModel
          )
        if (retirement === "tool choice") f.source.setToolChoice("none")
        if (retirement === "native MCP state") {
          retainedTools = { tools: [{ name: "foreign-client-tool" }] }
          nativeSetupFixture.tools = retainedTools
        }
      }
      const user = userEvent.setup()
      const view = render(<PlaygroundForm droppedFiles={[]} />)
      try {
        await user.type(
          screen.getByTestId("composer-textarea"),
          "Exact publication source  "
        )
        await user.click(
          screen.getByRole("button", { name: "Open role-play setup" })
        )
        await user.click(
          await screen.findByRole("button", { name: "Apply Character setup" })
        )
        let accepted: boolean | void
        await act(async () => {
          accepted = await nativeSetupFixture.selectionApply
        })
        expect(armed).toBe(false)
        expect(issued).toBeDefined()
        expect(f.source.setChatDraftAddress).toHaveBeenCalledExactlyOnceWith(
          issued!.address,
          { sourceAddress: f.address, expectedRevision: f.revision }
        )
        expect(f.source.chatDraftAddress).toBe(issued!.address)
        expect(f.source.chatDraftRevision).toBe(f.revision + 1)
        expect(issued!.revision).toBe(f.revision + 1)
        expect(accepted!).toBe(retirement === "uninterrupted")
        if (retirement === "uninterrupted") {
          expect(f.reset).toHaveBeenCalledOnce()
          expect(f.source.setHistoryId).toHaveBeenCalledExactlyOnceWith(null, {
            preserveServerChatId: false
          })
          expect(f.source.setServerChatId).toHaveBeenCalledExactlyOnceWith(null)
          expect(f.source.setHistory).toHaveBeenCalledExactlyOnceWith([])
          expect(f.source.setMessages).toHaveBeenCalledExactlyOnceWith([])
          expect(nativeSetupFixture.destinationCurrent?.()).toBe(true)
        } else {
          expect(f.reset).not.toHaveBeenCalled()
          expect(f.source.setHistoryId).not.toHaveBeenCalled()
          expect(f.source.setServerChatId).not.toHaveBeenCalled()
          expect(f.source.setHistory).not.toHaveBeenCalled()
          expect(f.source.setMessages).not.toHaveBeenCalled()
          expect(f.source.historyId).toBe(originalHistoryId)
          expect(f.source.serverChatId).toBe(originalServerId)
          expect(f.source.history).toBe(originalHistory)
          expect(f.source.messages).toBe(originalMessages)
          expect(nativeSetupFixture.destinationCurrent).toBeNull()
        }
        expect(f.source.selectedModel).toBe(
          retirement === "foreign model" ? "openai:gpt-4.1" : entryModel
        )
        expect(f.source.selectedModelRevision).toBe(
          entryModelRevision +
            (["same-value model revision", "foreign model"].includes(retirement)
              ? 1
              : 0)
        )
        expect(f.source.toolChoice).toBe(
          retirement === "tool choice" ? "none" : "auto"
        )
        expect(nativeSetupFixture.tools).toBe(retainedTools)
        if (retirement === "foreign model") {
          // The user's newer choice owns the Form's automatic scope projection.
          // No old command may alter either original scoped or global settings.
          expect(store.getState().activeSettingsScope).toBe("openai:gpt-4.1")
          expect(store.getState().scopedSettingsByModelKey).toBe(
            entrySettings.scopedSettingsByModelKey
          )
          expect(store.getState().globalSettings).toBe(
            entrySettings.globalSettings
          )
          expect(entrySettings.activeSettingsScope).toBe("openai:deepseek-chat")
          expect(entrySettings.temperature).toBe(0.4)
          expect(entrySettings.topP).toBe(0.7)
        } else expect(store.getState()).toBe(entrySettings)
        expect(selectedAssistantMock.setSelectedAssistant).toHaveBeenCalledOnce()
        expect(nativeSetupFixture.retain).toHaveBeenCalledWith(
          expect.any(String),
          expect.objectContaining({ content: "Exact publication source  " })
        )
        expect(screen.getByTestId("composer-textarea")).toHaveValue(
          "Exact publication source  "
        )
        expect(nativeSetupFixture.scene).not.toHaveBeenCalled()
        expect(nativeSetupFixture.actor).not.toHaveBeenCalled()
        expect(saveActorSettingsMock).not.toHaveBeenCalled()
        expect(sceneBucket.set).not.toHaveBeenCalled()
        expect(onSubmitMock).not.toHaveBeenCalled()
      } finally {
        nativeSetupFixture.onDraftPublication = null
        view.unmount()
        await Promise.allSettled([nativeSetupFixture.selectionApply])
      }
    }
  )

  it.each([
    "uninterrupted",
    "same-value model revision",
    "foreign model"
  ] as const)(
    "Form identity publication ownership holds the original model during nonnative storage (%s)",
    async (retirement) => {
      const f = prepareQualifiedSceneSource("fresh")
      Object.assign(f.source, {
        serverChatAssistantKind: null,
        serverChatAssistantId: null,
        serverChatCharacterId: null
      })
      nativeSetupFixture.realSettings = true
      const store = nativeSetupFixture.actualSettingsStore!
      store.getState().setActiveSettingsScope("openai:deepseek-chat")
      store
        .getState()
        .updateSettings({ apiProvider: "openai", temperature: 0.4, topP: 0.7 })
      const entrySettings = store.getState()
      const entryModel = f.source.selectedModel
      const entryRevision: number = f.source.selectedModelRevision
      const originalSelection = selectedAssistantMock.currentSelection
      const held = sceneSelectionGate<boolean>()
      const publications: unknown[] = []
      selectedAssistantMock.setSelectedAssistant.mockImplementationOnce(
        async (next) => {
          await held.promise
          // Physical preference storage honors the exact guard received by the
          // real Form caller; this double issues no ownership or generation grant.
          if (nativeSetupFixture.identityGuard?.() === false) return false
          publications.push(next)
          return true
        }
      )
      const user = userEvent.setup()
      const view = render(<PlaygroundForm droppedFiles={[]} />)
      try {
        await user.click(
          screen.getByRole("button", { name: "Open role-play setup" })
        )
        await user.click(
          await screen.findByRole("button", { name: "Apply Persona overlay setup" })
        )
        await waitFor(() =>
          expect(
            selectedAssistantMock.setSelectedAssistant
          ).toHaveBeenCalledOnce()
        )
        expect(nativeSetupFixture.identityGuard?.()).toBe(true)
        expect(store.getState()).toBe(entrySettings)
        if (retirement !== "uninterrupted") {
          act(() =>
            f.source.setSelectedModel(
              retirement === "foreign model" ? "openai:gpt-4.1" : entryModel
            )
          )
          expect(f.source.selectedModelRevision).toBe(entryRevision + 1)
          // The guard must refuse at the held physical write, before the outer
          // command's eventual false result or any later hook recapture.
          expect(nativeSetupFixture.identityGuard?.()).toBe(false)
        }
        let accepted: boolean | void
        await act(async () => {
          held.resolve(true)
          accepted = await nativeSetupFixture.selectionApply
        })
        expect(accepted!).toBe(retirement === "uninterrupted")
        expect(publications).toHaveLength(retirement === "uninterrupted" ? 1 : 0)
        if (retirement === "uninterrupted") {
          expect(publications[0]).toEqual(
            expect.objectContaining({ kind: "persona", id: "persona-guide" })
          )
          expect(selectedAssistantMock.currentSelection).toEqual(
            expect.objectContaining({ kind: "persona", id: "persona-guide" })
          )
        } else
          expect(selectedAssistantMock.currentSelection).toBe(originalSelection)
        expect(f.source.chatDraftAddress).toBe(f.address)
        expect(f.source.chatDraftRevision).toBe(f.revision)
        expect(f.reset).not.toHaveBeenCalled()
        expect(f.source.setHistoryId).not.toHaveBeenCalled()
        expect(f.source.beginSelectedModelSave).not.toHaveBeenCalled()
        expect(saveActorSettingsMock).not.toHaveBeenCalled()
        expect(sceneBucket.set).not.toHaveBeenCalled()
        expect(onSubmitMock).not.toHaveBeenCalled()
      } finally {
        view.unmount()
        held.resolve(true)
        await Promise.allSettled([nativeSetupFixture.selectionApply])
      }
    }
  )

  it.each([
    "absent global provider",
    "global provider",
    "raw target provider"
  ] as const)(
    "Legacy startup bare-model scope accepts its intended destination with %s",
    async (ownership) => {
      const source = playgroundFormMessageOptionState.value
      source.historyId = null
      nativeSetupFixture.realSettings = true
      const store = nativeSetupFixture.actualSettingsStore!
      const originalScope = "openai:deepseek-chat"
      const targetModel = "legacy-target-model"
      const globalProvider =
        ownership === "absent global provider" ? undefined : "anthropic"
      const targetProvider =
        ownership === "raw target provider" ? "custom" : globalProvider
      const targetScope =
        ownership === "global provider" ? `anthropic:${targetModel}` : targetModel
      if (globalProvider)
        store.getState().updateSettings({ apiProvider: globalProvider })
      store.getState().setActiveSettingsScope(originalScope)
      store
        .getState()
        .updateSettings({
          apiProvider: "openai",
          systemPrompt: "Departing scoped behavior",
          temperature: 0.41
        })
      if (ownership === "raw target provider") {
        store.getState().updateScopedSetting(targetModel, "apiProvider", "custom")
        store.getState().updateScopedSetting(targetModel, "temperature", 0.92)
      }
      const originalGlobal = store.getState().globalSettings
      const originalScoped =
        store.getState().scopedSettingsByModelKey[originalScope]
      expect(originalGlobal.apiProvider).toBe(globalProvider)
      expect(
        store.getState().scopedSettingsByModelKey[targetModel]?.apiProvider
      ).toBe(ownership === "raw target provider" ? "custom" : undefined)
      savedSetupFixture.raw = JSON.stringify([
        createStartupTemplateBundle({
          name: "Ordinary bare-model startup",
          selectedModel: targetModel,
          systemPrompt: "Raw target behavior",
          character: null
        })
      ])
      const held = sceneSelectionGate<boolean>()
      source.beginSelectedModelSave.mockImplementation((model: string) => {
        source.setSelectedModel(model)
        const revision: number = source.selectedModelRevision
        const isCurrent = () =>
          source.selectedModel === model &&
          source.selectedModelRevision === revision
        return { completion: held.promise, isCurrent, retry: () => held.promise }
      })
      const { view, apply } = await openOwnedStartupPreview()
      let application: Promise<unknown> | undefined
      try {
        expect(source.selectedModel).toBe("deepseek-chat")
        expect(store.getState().activeSettingsScope).toBe(originalScope)
        act(() => {
          application = apply()
        })
        await waitFor(() =>
          expect(source.beginSelectedModelSave).toHaveBeenCalledExactlyOnceWith(
            targetModel
          )
        )
        // Keep the real Form scope effect active during the actual declared
        // model ACK. It must agree with the original target's effective provider.
        view.rerender(<PlaygroundForm droppedFiles={[]} />)
        await waitFor(() =>
          expect(store.getState().activeSettingsScope).toBe(targetScope)
        )
        expect(store.getState().apiProvider).toBe(targetProvider)
        expect(store.getState().globalSettings).toBe(originalGlobal)
        expect(
          store.getState().scopedSettingsByModelKey[targetModel]?.apiProvider
        ).toBe(ownership === "raw target provider" ? "custom" : undefined)
        let accepted: unknown
        await act(async () => {
          held.resolve(true)
          accepted = await application
        })
        expect(accepted).toBe(true)
        expect(source.selectedModel).toBe(targetModel)
        expect(store.getState().activeSettingsScope).toBe(targetScope)
        expect(store.getState().systemPrompt).toBe("Raw target behavior")
        expect(
          store.getState().scopedSettingsByModelKey[targetScope]?.systemPrompt
        ).toBe("Raw target behavior")
        expect(store.getState().scopedSettingsByModelKey[originalScope]).toBe(
          originalScoped
        )
        expect(store.getState().globalSettings).toBe(originalGlobal)
        expect(store.getState().apiProvider).toBe(targetProvider)
        expect(
          store.getState().scopedSettingsByModelKey[targetScope]?.apiProvider
        ).toBe(ownership === "raw target provider" ? "custom" : undefined)
        if (ownership === "raw target provider")
          expect(
            store.getState().scopedSettingsByModelKey[targetModel]?.temperature
          ).toBe(0.92)
        expect(
          screen.queryByRole("button", { name: "Apply saved template" })
        ).not.toBeInTheDocument()
        expect(nativeSetupFixture.scene).not.toHaveBeenCalled()
        expect(saveActorSettingsMock).not.toHaveBeenCalled()
        expect(sceneBucket.set).not.toHaveBeenCalled()
        expect(onSubmitMock).not.toHaveBeenCalled()
      } finally {
        view.unmount()
        held.resolve(true)
        await Promise.allSettled([application])
      }
    }
  )

  it("Startup Scene refuses a bare-ID role-play source before all identity and Scene publications", async () => {
    const source = playgroundFormMessageOptionState.value
    source.serverChatId = "unqualified-original"
    source.historyId = "unqualified-mirror"
    const address: ChatDraftAddress = source.chatDraftAddress
    savedSetupFixture.raw = JSON.stringify([
      ownedStartupBundle(createDefaultActorSettings())
    ])
    const { view, apply } = await openOwnedStartupPreview()
    try {
      let accepted: unknown
      await act(async () => {
        accepted = await apply()
      })
      expect(accepted).toBe(false)
      expect(source.chatDraftAddress).toBe(address)
      expect(source.serverChatId).toBe("unqualified-original")
      expect(source.historyId).toBe("unqualified-mirror")
      expect(source.setChatDraftAddress).not.toHaveBeenCalled()
      expect(source.setServerChatId).not.toHaveBeenCalled()
      expect(source.setHistoryId).not.toHaveBeenCalled()
      expect(selectedAssistantMock.setSelectedAssistant).not.toHaveBeenCalled()
      expect(source.beginSelectedModelSave).not.toHaveBeenCalled()
      expect(nativeSetupFixture.scene).not.toHaveBeenCalled()
      expect(sceneBucket.get).not.toHaveBeenCalled()
      expect(sceneBucket.set).not.toHaveBeenCalled()
      expect(saveActorSettingsMock).not.toHaveBeenCalled()
      expect(useActorStore.getState().applied).toBeNull()
      expect(
        screen.getByRole("button", { name: "Apply saved template" })
      ).toBeInTheDocument()
    } finally {
      view.unmount()
    }
  })

  it.each(["uninterrupted", "same-model settings edit"] as const)(
    "Form settings continuation preserves immutable response settings during held identity (%s)",
    async (retirement) => {
      const f = prepareQualifiedSceneSource("fresh")
      f.source.serverChatAssistantKind = null
      f.source.serverChatAssistantId = null
      f.source.serverChatCharacterId = null
      nativeSetupFixture.realSettings = true
      nativeSetupFixture.realRolePlayPresets = true
      const modelStore = nativeSetupFixture.actualSettingsStore!
      modelStore.getState().setActiveSettingsScope("openai:deepseek-chat")
      modelStore
        .getState()
        .updateSettings({
          apiProvider: "openai",
          temperature: 0.4,
          topP: 0.7,
          systemPrompt: "Original behavior"
        })
      const entrySettings = modelStore.getState()
      const originalModel = f.source.selectedModel
      const originalModelRevision = f.source.selectedModelRevision
      const originalHistory = f.source.history
      const originalMessages = f.source.messages
      const held = sceneSelectionGate<boolean>()
      selectedAssistantMock.setSelectedAssistant.mockReturnValueOnce(held.promise)
      let editedSettings: ModelSettingsStore | undefined
      const user = userEvent.setup()
      const view = render(<PlaygroundForm droppedFiles={[]} />)
      try {
        await user.click(
          screen.getByRole("button", { name: "Open role-play setup" })
        )
        await user.click(
          await screen.findByRole("button", {
            name: "Apply overlay behavior and generation setup"
          })
        )
        await waitFor(() =>
          expect(
            selectedAssistantMock.setSelectedAssistant
          ).toHaveBeenCalledOnce()
        )
        expect(nativeSetupFixture.destinationCurrent?.()).toBe(true)
        expect(modelStore.getState()).toBe(entrySettings)
        expect(f.source.selectedModel).toBe(originalModel)
        expect(f.source.selectedModelRevision).toBe(originalModelRevision)
        if (retirement === "same-model settings edit") {
          act(() =>
            modelStore
              .getState()
              .updateSettings({
                temperature: 1.3,
                topP: 0.34,
                topK: 77,
                systemPrompt: "Newer same-model behavior",
                systemPromptTemplateId: "newer-behavior"
              })
          )
          editedSettings = modelStore.getState()
          expect(editedSettings).not.toBe(entrySettings)
          expect(entrySettings.temperature).toBe(0.4)
          expect(entrySettings.systemPrompt).toBe("Original behavior")
          view.rerender(<PlaygroundForm droppedFiles={[]} />)
        }
        let accepted: boolean | void
        await act(async () => {
          held.resolve(true)
          accepted = await nativeSetupFixture.selectionApply
        })
        expect(accepted!).toBe(retirement === "uninterrupted")
        if (retirement === "uninterrupted") {
          expect(modelStore.getState()).toEqual(
            expect.objectContaining({
              systemPrompt: "Original behavior proposal",
              systemPromptTemplateId: "guarded-behavior",
              temperature: 0.2,
              topP: 0.8,
              topK: 20
            })
          )
          expect(
            f.source.setSelectedSystemPrompt
          ).toHaveBeenCalledExactlyOnceWith(undefined)
          expect(f.source.setSelectedQuickPrompt).toHaveBeenCalledExactlyOnceWith(
            null
          )
        } else {
          expect(modelStore.getState()).toBe(editedSettings)
          expect(modelStore.getState()).toEqual(
            expect.objectContaining({
              systemPrompt: "Newer same-model behavior",
              systemPromptTemplateId: "newer-behavior",
              temperature: 1.3,
              topP: 0.34,
              topK: 77
            })
          )
          expect(f.source.setSelectedSystemPrompt).not.toHaveBeenCalled()
          expect(f.source.setSelectedQuickPrompt).not.toHaveBeenCalled()
        }
        expect(f.source.selectedModel).toBe(originalModel)
        expect(f.source.selectedModelRevision).toBe(originalModelRevision)
        expect(f.source.chatDraftAddress).toBe(f.address)
        expect(f.source.chatDraftRevision).toBe(f.revision)
        expect(f.source.history).toBe(originalHistory)
        expect(f.source.messages).toBe(originalMessages)
        expect(f.source.serverChatAssistantKind).toBeNull()
        expect(f.source.serverChatAssistantId).toBeNull()
        expect(f.reset).not.toHaveBeenCalled()
        expect(nativeSetupFixture.actor).not.toHaveBeenCalled()
        expect(nativeSetupFixture.scene).not.toHaveBeenCalled()
        expect(saveActorSettingsMock).not.toHaveBeenCalled()
        expect(sceneBucket.set).not.toHaveBeenCalled()
        expect(useActorStore.getState().applied).toBeNull()
        expect(onSubmitMock).not.toHaveBeenCalled()
      } finally {
        view.unmount()
        held.resolve(true)
        await Promise.allSettled([nativeSetupFixture.selectionApply])
      }
    }
  )

  it.each(["uninterrupted", "same-model settings edit"] as const)(
    "Form settings continuation preserves immutable startup settings during held model persistence (%s)",
    async (retirement) => {
      const f = prepareQualifiedSceneSource("fresh")
      f.source.serverChatAssistantKind = null
      f.source.serverChatAssistantId = null
      f.source.serverChatCharacterId = null
      nativeSetupFixture.realSettings = true
      const modelStore = nativeSetupFixture.actualSettingsStore!
      modelStore.getState().setActiveSettingsScope("openai:deepseek-chat")
      modelStore
        .getState()
        .updateSettings({
          apiProvider: "openai",
          temperature: 0.4,
          topP: 0.7,
          systemPrompt: "Original startup behavior"
        })
      const entrySettings = modelStore.getState()
      const originalModel = f.source.selectedModel
      const originalHistory = f.source.history
      const originalMessages = f.source.messages
      const original: ActorSceneRecord = {
        value: {
          ...createDefaultActorSettings(),
          notes: "Original settings-owned Scene"
        },
        updatedAt: 1730000000000
      }
      const proposal: ActorSettings = {
        ...createDefaultActorSettings(),
        notes: "Frozen settings-owned startup proposal"
      }
      const key = `actorSettings:owned:${f.target.key}`
      sceneBucket.records.set(key, original)
      const bundle = ownedStartupBundle(proposal, null)
      if (!bundle.rolePlay)
        throw new Error("fixture requires the original role-play proposal")
      bundle.selectedModel = originalModel
      bundle.rolePlay.behavior = {
        source: "custom",
        templateId: null,
        templateTitle: null,
        templateCategory: null,
        systemPrompt: "Frozen startup behavior proposal",
        modified: false
      }
      savedSetupFixture.raw = JSON.stringify([bundle])
      const held = sceneSelectionGate<boolean>()
      let modelSaveRevision = -1
      f.source.beginSelectedModelSave.mockImplementation((model: string) => {
        f.source.setSelectedModel(model)
        const revision: number = f.source.selectedModelRevision
        modelSaveRevision = revision
        const isCurrent = () =>
          f.source.selectedModel === model &&
          f.source.selectedModelRevision === revision
        return { completion: held.promise, isCurrent, retry: () => held.promise }
      })
      let editedSettings: ModelSettingsStore | undefined
      const { view, apply } = await openOwnedStartupPreview()
      let application: Promise<unknown> | undefined
      try {
        act(() => {
          application = apply()
        })
        await waitFor(() =>
          expect(f.source.beginSelectedModelSave).toHaveBeenCalledOnce()
        )
        expect(selectedAssistantMock.setSelectedAssistant).toHaveBeenCalledOnce()
        expect(modelStore.getState()).toBe(entrySettings)
        expect(f.source.selectedModel).toBe(originalModel)
        expect(f.source.selectedModelRevision).toBe(modelSaveRevision)
        expect(saveActorSettingsMock).not.toHaveBeenCalled()
        if (retirement === "same-model settings edit") {
          act(() =>
            modelStore
              .getState()
              .updateSettings({
                temperature: 1.6,
                topP: 0.29,
                topK: 81,
                systemPrompt: "Newer same-model startup behavior",
                systemPromptTemplateId: "newer-startup-behavior"
              })
          )
          editedSettings = modelStore.getState()
          expect(editedSettings).not.toBe(entrySettings)
          expect(entrySettings.temperature).toBe(0.4)
          expect(entrySettings.systemPrompt).toBe("Original startup behavior")
          view.rerender(<PlaygroundForm droppedFiles={[]} />)
        }
        let accepted: unknown
        await act(async () => {
          held.resolve(true)
          accepted = await application
        })
        expect(accepted).toBe(retirement === "uninterrupted")
        if (retirement === "uninterrupted") {
          expect(modelStore.getState()).toEqual(
            expect.objectContaining({
              systemPrompt: "Frozen startup behavior proposal",
              temperature: 0.4,
              topP: 0.7
            })
          )
          expect(sceneBucket.set).toHaveBeenCalledExactlyOnceWith(
            key,
            expect.objectContaining({ value: proposal })
          )
          expect(saveActorSettingsMock).toHaveBeenCalledOnce()
          expect(saveActorSettingsMock.mock.calls[0][0].expectedRecord).toEqual(
            original
          )
          expect(useActorStore.getState().applied?.target.key).toBe(f.target.key)
          expect(useActorStore.getState().applied?.target.isCurrent()).toBe(true)
          expect(
            screen.queryByRole("button", { name: "Apply saved template" })
          ).not.toBeInTheDocument()
        } else {
          expect(modelStore.getState()).toBe(editedSettings)
          expect(modelStore.getState()).toEqual(
            expect.objectContaining({
              systemPrompt: "Newer same-model startup behavior",
              systemPromptTemplateId: "newer-startup-behavior",
              temperature: 1.6,
              topP: 0.29,
              topK: 81
            })
          )
          expect(f.source.setSelectedSystemPrompt).not.toHaveBeenCalled()
          expect(sceneBucket.records.get(key)).toBe(original)
          expect(saveActorSettingsMock).not.toHaveBeenCalled()
          expect(sceneBucket.set).not.toHaveBeenCalled()
          expect(useActorStore.getState().applied).toBeNull()
          expect(
            screen.getByRole("button", { name: "Apply saved template" })
          ).toBeInTheDocument()
        }
        expect(f.source.selectedModel).toBe(originalModel)
        expect(f.source.selectedModelRevision).toBe(modelSaveRevision)
        expect(f.source.chatDraftAddress).toBe(f.address)
        expect(f.source.chatDraftRevision).toBe(f.revision)
        expect(f.source.history).toBe(originalHistory)
        expect(f.source.messages).toBe(originalMessages)
        expect(f.reset).not.toHaveBeenCalled()
        expect(nativeSetupFixture.scene).toHaveBeenCalledOnce()
        expect(nativeSetupFixture.actor).not.toHaveBeenCalled()
        expect(onSubmitMock).not.toHaveBeenCalled()
      } finally {
        view.unmount()
        held.resolve(true)
        await Promise.allSettled([application])
      }
    }
  )

  // These controls use the real Form/usePromptTemplates callbacks and Scene
  // service/bucket CAS/Applied store. Account, owner and physical storage are
  // declared boundaries; they do not claim H1 admission or browser durability.
  it.each(
    (["native", "fresh"] as const).flatMap((kind) =>
      [false, true].map((conflict) => ({ kind, conflict }))
    )
  )(
    "Startup Scene ownership retains the original $kind record across identity preparation (conflict: $conflict)",
    async ({ kind, conflict }) => {
      const f = prepareQualifiedSceneSource(kind)
      const original: ActorSceneRecord = {
        value: { ...createDefaultActorSettings(), notes: "Original owned Scene" },
        updatedAt: 1730000000000
      }
      const newer: ActorSceneRecord = {
        value: {
          ...createDefaultActorSettings(),
          notes: "Independent newer Scene"
        },
        updatedAt: original.updatedAt + 1
      }
      const proposal: ActorSettings = {
        ...createDefaultActorSettings(),
        notes: "Frozen startup proposal"
      }
      const key = `actorSettings:owned:${f.target.key}`
      sceneBucket.records.set(key, original)
      savedSetupFixture.raw = JSON.stringify([ownedStartupBundle(proposal)])
      let readBeforeIdentity = false
      const leasesAtSave: boolean[] = []
      saveActorSettingsMock.mockImplementation((params) => {
        leasesAtSave.push(params.validate_lease?.() === true)
        return nativeSetupFixture.actualSceneWriter!(params)
      })
      selectedAssistantMock.setSelectedAssistant.mockImplementationOnce(
        async () => {
          readBeforeIdentity = nativeSetupFixture.scene.mock.calls.length === 1
          if (conflict) sceneBucket.records.set(key, newer)
          return true
        }
      )
      const { view, apply } = await openOwnedStartupPreview()
      try {
        let accepted: unknown
        await act(async () => {
          accepted = await apply()
        })
        expect(accepted).toBe(!conflict)
        expect(readBeforeIdentity).toBe(true)
        expect(nativeSetupFixture.scene).toHaveBeenCalledOnce()
        const save = saveActorSettingsMock.mock.calls[0]?.[0]
        expect(save?.target?.key).toBe(f.target.key)
        expect(save?.expectedRecord).toEqual(original)
        expect(leasesAtSave).toEqual([true])
        expect(save?.settings).toEqual(proposal)
        expect(f.source.chatDraftAddress).toBe(f.address)
        expect(f.source.chatDraftRevision).toBe(f.revision)
        expect(f.reset).not.toHaveBeenCalled()
        if (conflict) {
          expect(sceneBucket.records.get(key)).toBe(newer)
          expect(sceneBucket.set).not.toHaveBeenCalled()
          expect(useActorStore.getState().applied).toBeNull()
          expect(
            screen.getByRole("button", { name: "Apply saved template" })
          ).toBeInTheDocument()
          await act(async () => {
            expect(await apply()).toBe(false)
          })
          expect(
            selectedAssistantMock.setSelectedAssistant
          ).toHaveBeenCalledOnce()
          expect(nativeSetupFixture.scene).toHaveBeenCalledOnce()
          expect(saveActorSettingsMock).toHaveBeenCalledTimes(2)
          expect(leasesAtSave).toEqual([true, true])
          expect(saveActorSettingsMock.mock.calls[1][0].expectedRecord).toEqual(
            original
          )
          expect(sceneBucket.records.get(key)).toBe(newer)
          expect(sceneBucket.set).not.toHaveBeenCalled()
        } else {
          expect(sceneBucket.records.get(key)).toEqual(
            expect.objectContaining({ value: proposal })
          )
          expect(sceneBucket.set).toHaveBeenCalledOnce()
          expect(useActorStore.getState().applied).toEqual(
            expect.objectContaining({ settings: proposal })
          )
          expect(useActorStore.getState().applied?.target.key).toBe(f.target.key)
          expect(useActorStore.getState().applied?.target.isCurrent()).toBe(true)
          expect(
            screen.queryByRole("button", { name: "Apply saved template" })
          ).not.toBeInTheDocument()
        }
        expect(onSubmitMock).not.toHaveBeenCalled()
        expect(createChatCompletionMock).not.toHaveBeenCalled()
      } finally {
        view.unmount()
      }
    }
  )

  it.each([false, true])(
    "Startup Scene ownership saves only the exact notified fresh destination with expected absence (conflict: %s)",
    async (conflict) => {
      const f = prepareQualifiedSceneSource()
      f.source.serverChatCharacterId = "departing-character"
      f.source.serverChatAssistantId = "departing-character"
      const departing: ActorSceneRecord = {
        value: {
          ...createDefaultActorSettings(),
          isEnabled: true,
          notes: "Departing private Scene"
        },
        updatedAt: 1730000000000
      }
      const departingKey = `actorSettings:owned:${f.target.key}`
      sceneBucket.records.set(departingKey, departing)
      const proposal: ActorSettings = {
        ...createDefaultActorSettings(),
        notes: "Fresh destination proposal"
      }
      const newer: ActorSceneRecord = {
        value: {
          ...createDefaultActorSettings(),
          notes: "Fresh destination already saved"
        },
        updatedAt: departing.updatedAt + 1
      }
      savedSetupFixture.raw = JSON.stringify([ownedStartupBundle(proposal)])
      let issued:
        | {
            address: ChatDraftAddress
            revision: number
            target: ActorSettingsTarget
          }
        | undefined
      nativeSetupFixture.onDraftPublication = (address) => {
        const revision: number = f.source.chatDraftRevision
        const target = captureActorSettingsTarget({
          address,
          requestScope: f.requestScope,
          accountIsCurrent: f.target.account!.isCurrent,
          isCurrent: () =>
            f.source.chatDraftAddress === address &&
            f.source.chatDraftRevision === revision
        })
        if (!target)
          throw new Error("fixture requires the exact issued destination locator")
        issued = { address, revision, target }
      }
      const leasesAtSave: boolean[] = []
      saveActorSettingsMock.mockImplementationOnce(async (params) => {
        if (!issued)
          throw new Error("Save must follow the actual fresh draft publication")
        leasesAtSave.push(params.validate_lease?.() === true)
        if (conflict)
          sceneBucket.records.set(
            `actorSettings:owned:${issued.target.key}`,
            newer
          )
        return nativeSetupFixture.actualSceneWriter!(params)
      })
      const { view, apply } = await openOwnedStartupPreview()
      try {
        let accepted: unknown
        await act(async () => {
          accepted = await apply()
        })
        expect(accepted).toBe(!conflict)
        expect(issued).toBeDefined()
        const destination = issued!
        expect(f.source.chatDraftAddress).toBe(destination.address)
        expect(f.source.chatDraftRevision).toBe(destination.revision)
        expect(f.source.historyId).toBeNull()
        expect(f.source.serverChatId).toBeNull()
        expect(f.reset).toHaveBeenCalledOnce()
        const save = saveActorSettingsMock.mock.calls[0]?.[0]
        expect(save?.target?.key).toBe(destination.target.key)
        expect(save?.target?.address).toEqual(destination.address)
        expect(save?.target?.account?.requestScope).toEqual(f.requestScope)
        expect(save?.expectedRecord).toBeNull()
        expect(leasesAtSave).toEqual([true])
        expect(sceneBucket.records.get(departingKey)).toBe(departing)
        const destinationKey = `actorSettings:owned:${destination.target.key}`
        if (conflict) {
          expect(sceneBucket.records.get(destinationKey)).toBe(newer)
          expect(sceneBucket.set).not.toHaveBeenCalled()
          expect(useActorStore.getState().applied).toBeNull()
          expect(
            screen.getByRole("button", { name: "Apply saved template" })
          ).toBeInTheDocument()
        } else {
          expect(sceneBucket.records.get(destinationKey)).toEqual(
            expect.objectContaining({ value: proposal })
          )
          expect(sceneBucket.set).toHaveBeenCalledExactlyOnceWith(
            destinationKey,
            expect.objectContaining({ value: proposal })
          )
          expect(useActorStore.getState().applied?.target.key).toBe(
            destination.target.key
          )
          expect(useActorStore.getState().applied?.target.isCurrent()).toBe(true)
          expect(
            screen.queryByRole("button", { name: "Apply saved template" })
          ).not.toBeInTheDocument()
        }
        expect(
          [...sceneBucket.records.keys()].every((key) =>
            key.startsWith("actorSettings:owned:")
          )
        ).toBe(true)
        expect(onSubmitMock).not.toHaveBeenCalled()
      } finally {
        nativeSetupFixture.onDraftPublication = null
        view.unmount()
      }
    }
  )

  it("Startup Scene ownership publishes one complete Applied receipt for an enabled Scene", async () => {
    const f = prepareQualifiedSceneSource("fresh")
    f.source.serverChatAssistantKind = null
    f.source.serverChatAssistantId = null
    f.source.serverChatCharacterId = null
    const proposal: ActorSettings = {
      ...createDefaultActorSettings(),
      isEnabled: true,
      notes: "A complete Scene"
    }
    savedSetupFixture.raw = JSON.stringify([ownedStartupBundle(proposal, null)])
    type ActorState = ReturnType<typeof useActorStore.getState>
    const publications: Pick<
      ActorState,
      "applied" | "settings" | "preview" | "tokenCount"
    >[] = []
    const unsubscribe = useActorStore.subscribe((state) =>
      publications.push({
        applied: state.applied,
        settings: state.settings,
        preview: state.preview,
        tokenCount: state.tokenCount
      })
    )
    const { view, apply } = await openOwnedStartupPreview()
    try {
      await act(async () => {
        expect(await apply()).toBe(true)
      })
      const preview = summarizeRolePlayScene(proposal)
      expect(publications).toHaveLength(1)
      expect(publications[0]).toEqual({
        applied: expect.objectContaining({
          target: expect.objectContaining({ key: f.target.key }),
          settings: proposal,
          preview: preview.prompt,
          tokenCount: preview.tokenCount
        }),
        settings: proposal,
        preview: preview.prompt,
        tokenCount: preview.tokenCount
      })
      expect(useActorStore.getState().applied?.target.isCurrent()).toBe(true)
      expect(
        sceneBucket.records.get(`actorSettings:owned:${f.target.key}`)
      ).toEqual(expect.objectContaining({ value: proposal }))
      expect(
        screen.queryByRole("button", { name: "Apply saved template" })
      ).not.toBeInTheDocument()
      expect(onSubmitMock).not.toHaveBeenCalled()
    } finally {
      unsubscribe()
      view.unmount()
    }
  })

  it("Startup Scene ownership freezes the proposal before an identity callback mutates the visible preview", async () => {
    const f = prepareQualifiedSceneSource("fresh")
    const proposal: ActorSettings = {
      ...createDefaultActorSettings(),
      notes: "Original frozen notes"
    }
    savedSetupFixture.raw = JSON.stringify([ownedStartupBundle(proposal)])
    let mutated = false
    selectedAssistantMock.setSelectedAssistant.mockImplementationOnce(
      async () => {
        const scene = startupPreviewFixture.template?.rolePlay?.scene
        if (!scene)
          throw new Error(
            "fixture requires the actual callback-visible preview Scene"
          )
        scene.notes = "Callback replacement notes"
        mutated = true
        return true
      }
    )
    const { view, apply } = await openOwnedStartupPreview()
    try {
      await act(async () => {
        expect(await apply()).toBe(true)
      })
      expect(mutated).toBe(true)
      expect(startupPreviewFixture.template?.rolePlay?.scene?.notes).toBe(
        "Callback replacement notes"
      )
      expect(saveActorSettingsMock.mock.calls[0]?.[0].settings).toEqual(proposal)
      expect(
        sceneBucket.records.get(`actorSettings:owned:${f.target.key}`)
      ).toEqual(expect.objectContaining({ value: proposal }))
      expect(useActorStore.getState().applied?.settings).toEqual(proposal)
      expect(useActorStore.getState().applied?.target.key).toBe(f.target.key)
      expect(nativeSetupFixture.scene).toHaveBeenCalledOnce()
      expect(selectedAssistantMock.setSelectedAssistant).toHaveBeenCalledOnce()
    } finally {
      view.unmount()
    }
  })

  it.each([
    "uninterrupted",
    "original account lease",
    "New",
    "New then original address and raw IDs"
  ] as const)(
    "Startup Scene ownership guards actual persistence after a held Save: %s",
    async (retirement) => {
      const f = prepareQualifiedSceneSource("fresh")
      const original: ActorSceneRecord = {
        value: {
          ...createDefaultActorSettings(),
          notes: "Original retained record"
        },
        updatedAt: 1730000000000
      }
      const key = `actorSettings:owned:${f.target.key}`
      sceneBucket.records.set(key, original)
      const proposal: ActorSettings = {
        ...createDefaultActorSettings(),
        notes: "Held Scene proposal"
      }
      savedSetupFixture.raw = JSON.stringify([ownedStartupBundle(proposal)])
      const held = sceneSelectionGate<void>()
      // Only the service-entry wait is declared. After it settles, the actual
      // writer performs target validation and the real conditional bucket write.
      saveActorSettingsMock.mockImplementationOnce(async (params) => {
        await held.promise
        return nativeSetupFixture.actualSceneWriter!(params)
      })
      const { view, apply } = await openOwnedStartupPreview()
      let pending: Promise<unknown> | undefined
      try {
        act(() => {
          pending = apply()
        })
        await waitFor(() => expect(saveActorSettingsMock).toHaveBeenCalledOnce())
        if (retirement === "original account lease") f.retireOriginalAccount()
        if (
          retirement === "New" ||
          retirement === "New then original address and raw IDs"
        ) {
          f.source.startNewChatDraft(false)
          if (retirement === "New then original address and raw IDs") {
            f.source.setChatDraftAddress(f.address)
            f.source.historyId = null
            f.source.serverChatId = null
          }
          f.source.history = [{ role: "user", content: "Replacement history" }]
          f.source.messages = [
            {
              id: "replacement-message",
              isBot: false,
              name: "You",
              message: "Replacement view",
              sources: []
            }
          ]
        }
        const destination = {
          address: f.source.chatDraftAddress,
          revision: f.source.chatDraftRevision,
          historyId: f.source.historyId,
          serverChatId: f.source.serverChatId,
          history: f.source.history,
          messages: f.source.messages
        }
        let accepted: unknown
        await act(async () => {
          held.resolve(undefined)
          accepted = await pending
        })
        expect(accepted).toBe(retirement === "uninterrupted")
        expect(f.source.chatDraftAddress).toBe(destination.address)
        expect(f.source.chatDraftRevision).toBe(destination.revision)
        expect(f.source.historyId).toBe(destination.historyId)
        expect(f.source.serverChatId).toBe(destination.serverChatId)
        expect(f.source.history).toBe(destination.history)
        expect(f.source.messages).toBe(destination.messages)
        expect(f.reset).not.toHaveBeenCalled()
        if (retirement === "uninterrupted") {
          expect(sceneBucket.records.get(key)).toEqual(
            expect.objectContaining({ value: proposal })
          )
          expect(sceneBucket.set).toHaveBeenCalledExactlyOnceWith(
            key,
            expect.objectContaining({ value: proposal })
          )
          expect(useActorStore.getState().applied?.target.key).toBe(f.target.key)
          expect(useActorStore.getState().applied?.target.isCurrent()).toBe(true)
          expect(
            screen.queryByRole("button", { name: "Apply saved template" })
          ).not.toBeInTheDocument()
        } else {
          expect(sceneBucket.records.get(key)).toBe(original)
          expect(sceneBucket.set).not.toHaveBeenCalled()
          expect(useActorStore.getState().applied).toBeNull()
          expect(
            screen.getByRole("button", { name: "Apply saved template" })
          ).toBeInTheDocument()
        }
        expect(onSubmitMock).not.toHaveBeenCalled()
        expect(createChatCompletionMock).not.toHaveBeenCalled()
      } finally {
        held.resolve(undefined)
        await pending
        view.unmount()
      }
    }
  )

  it("keeps a loaded protected persona readable with unavailable Send, exact draft and owner", async () => {
    const f = prepareQualifiedSceneSource("native", "loaded-persona", "loaded-persona-mirror")
    Object.assign(f.source, { serverChatAssistantKind: "persona", serverChatAssistantId: "persona-guide", serverChatCharacterId: null,
      messages: [{ id: "persona-row", name: "Guide", message: "Saved persona response", isBot: true }],
      history: [{ role: "assistant", content: "Saved persona response" }] })
    selectedAssistantMock.initialSelection = null
    const user = userEvent.setup()
    const view = render(<PlaygroundForm droppedFiles={[]} />)
    try {
      const input = screen.getByTestId("composer-textarea")
      await user.type(input, "Keep this exact persona draft  ")
      const before = { chatDraftAddress: f.source.chatDraftAddress, chatDraftRevision: f.source.chatDraftRevision,
        historyId: f.source.historyId, serverChatId: f.source.serverChatId, history: f.source.history, messages: f.source.messages }
      expect(screen.getByText(/Tracked persona chat is unavailable here/)).toBeVisible()
      const sends = screen.getAllByRole("button", { name: /send message/i })
      sends.forEach(send => expect(send).toBeDisabled())
      await user.click(sends[0])
      await user.keyboard("{Enter}")
      expect(onSubmitMock).not.toHaveBeenCalled()
      expect(createChatCompletionMock).not.toHaveBeenCalled()
      expect(input).toHaveValue("Keep this exact persona draft  ")
      expect(f.source).toMatchObject(before)
      expect(f.reset).not.toHaveBeenCalled()
      expect(nativeSetupFixture.retain).not.toHaveBeenCalled()
    } finally { view.unmount() }
  })

  it.each(["drawer", "startup"])("refuses protected persona through the retained %s application before identity, model, prompt, RAG or destination writes", async entry => {
    const persona = { kind: "persona" as const, id: "persona-guide", name: "Guide Persona" }
    let view: ReturnType<typeof render>
    let apply: () => Promise<unknown>
    if (entry === "drawer") {
      const fixture = await openActualSceneDrawer("neutral fresh")
      view = fixture.view
      const callback = nativeSetupFixture.drawerProps?.onApply
      if (!callback) throw new Error("expected actual parent Apply callback")
      apply = async () => callback({ identitySelection: { ...persona, metadata: { selectionMode: "tracked" } } })
    } else {
      prepareQualifiedSceneSource("fresh")
      savedSetupFixture.raw = JSON.stringify([ownedStartupBundle(createDefaultActorSettings(), persona)])
      const fixture = await openOwnedStartupPreview()
      view = fixture.view
      apply = fixture.apply
    }
    const source = playgroundFormMessageOptionState.value
    const snapshot = () => structuredClone({ historyId: source.historyId, serverChatId: source.serverChatId,
      chatDraftAddress: source.chatDraftAddress, chatDraftRevision: source.chatDraftRevision, selectedModel: source.selectedModel,
      selectedSystemPrompt: source.selectedSystemPrompt, history: source.history, messages: source.messages,
      ragPinnedResults: source.ragPinnedResults, toolChoice: source.toolChoice })
    const before = snapshot()
    const modelBefore = nativeSetupFixture.actualSettingsStore!.getState()
    const recordsBefore = structuredClone(Array.from(sceneBucket.records.entries()))
    const route = vi.fn()
    window.addEventListener(CHAT_ROUTE_REPLACEMENT_EVENT, route)
    try {
      await act(async () => {
        if (entry === "drawer") await expect(apply()).rejects.toThrow("native_history_persona_unsupported")
        else {
          expect(await apply()).toBe(false)
          expect(nativeSetupFixture.error).toHaveBeenCalledWith(expect.objectContaining({
            description: expect.stringMatching(/persona.*not supported.*overlay/i)
          }))
        }
      })
      expect(selectedAssistantMock.setSelectedAssistant).not.toHaveBeenCalled()
      expect(nativeSetupFixture.actualSettingsStore!.getState()).toBe(modelBefore)
      expect(snapshot()).toEqual(before)
      expect(Array.from(sceneBucket.records.entries())).toEqual(recordsBefore)
      expect(sceneBucket.set).not.toHaveBeenCalled()
      expect(saveActorSettingsMock).not.toHaveBeenCalled()
      expect(nativeSetupFixture.retain).not.toHaveBeenCalled()
      expect(route).not.toHaveBeenCalled()
    } finally { window.removeEventListener(CHAT_ROUTE_REPLACEMENT_EVENT, route); view.unmount() }
  })

  it("clears an explicit scope when the default composer opener reopens during close", async () => {
    prepareNativeSource()
    modelSettingsBoundary.enabled = true
    const user = userEvent.setup()
    const view = render(<PlaygroundForm droppedFiles={[]} />)
    try {
      act(() => window.dispatchEvent(new CustomEvent("tldw:open-model-settings", {
        detail: { settingsScope: "openai:previous" }
      })))
      await screen.findByTestId("held-model-editor")
      await user.click(screen.getByRole("button", { name: "Close held model editor" }))
      const completion = modelSettingsBoundary.afterClose
      if (!completion) throw new Error("expected held native close completion")
      await user.click(screen.getByRole("button", { name: "Open default model settings" }))
      const editor = screen.getByTestId("held-model-editor")
      expect(editor).toHaveAttribute("data-open", "true")
      expect(editor).toHaveAttribute("data-scope", "")
      act(() => completion())
      expect(editor).toHaveAttribute("data-open", "true")
      expect(editor).toHaveAttribute("data-scope", "")
    } finally { view.unmount() }
  })

  it.each(["current", "history", "account", "draft", "model", "reopen", "unmount"])("returns model settings focus only for its %s close owner", async boundary => {
    prepareNativeSource()
    modelSettingsBoundary.enabled = true
    const user = userEvent.setup()
    const open = (scope = "openai:original") => window.dispatchEvent(new CustomEvent("tldw:open-model-settings", {
      detail: { settingsScope: scope, returnFocusSelector: "[data-testid='model-settings-opener']" }
    }))
    const view = render(<>
      <button data-testid="model-settings-opener" onClick={() => open()}>Open owned model settings</button>
      <button>Another task</button>
      <PlaygroundForm droppedFiles={[]} />
    </>)
    const opener = screen.getByRole("button", { name: "Open owned model settings" })
    await user.click(opener)
    const editor = await screen.findByTestId("held-model-editor")
    await user.click(screen.getByRole("button", { name: "Close held model editor" }))
    expect(editor).toHaveAttribute("data-open", "false")
    expect(editor).toHaveAttribute("data-scope", "openai:original")
    const completion = modelSettingsBoundary.afterClose
    if (!completion) throw new Error("expected held native close completion")
    screen.getByRole("button", { name: "Another task" }).focus()
    const frames: FrameRequestCallback[] = []
    const frame = vi.spyOn(window, "requestAnimationFrame").mockImplementation(callback => { frames.push(callback); return frames.length })
    try {
      if (boundary === "reopen") act(() => { open("openai:new") })
      act(() => completion())
      if (boundary === "history") act(() => nativeSetupFixture.controller!.reset())
      if (boundary === "account") nativeSetupFixture.ownerCurrent = false
      if (boundary === "draft") act(() => playgroundFormMessageOptionState.value.startNewChatDraft())
      if (boundary === "model") act(() => playgroundFormMessageOptionState.value.setSelectedModel("openai:gpt-4.1"))
      if (boundary === "unmount") view.unmount()
      act(() => frames.splice(0).forEach(callback => callback(0)))
      if (boundary === "current") expect(opener).toHaveFocus()
      else if (boundary !== "unmount") expect(screen.getByRole("button", { name: "Another task" })).toHaveFocus()
      if (boundary === "reopen") {
        expect(screen.getByTestId("held-model-editor")).toHaveAttribute("data-open", "true")
        expect(screen.getByTestId("held-model-editor")).toHaveAttribute("data-scope", "openai:new")
      } else expect(screen.queryByTestId("held-model-editor")).toBeNull()
    } finally { view.unmount(); frame.mockRestore() }
  })

  it.each(["click", "Tab"])("preserves outside focus when an externally opened real picker departs by %s", async departure => {
    nativeSetupFixture.realModelPicker = true
    const user = userEvent.setup()
    render(<ConfigProvider theme={{ token: { motion: false } }}>
      <button>Another task</button>
      <button data-testid="external-model-opener" onClick={() => openModelSelector({ returnFocusSelector: "[data-testid='external-model-opener']" })}>Choose another model</button>
      <PlaygroundForm droppedFiles={[]} />
    </ConfigProvider>)
    await user.click(screen.getByRole("button", { name: "Choose another model" }))
    const trigger = screen.getByTestId("model-selector")
    const search = await screen.findByRole("textbox", { name: "Search models" })
    await act(() => new Promise(resolve => window.requestAnimationFrame(resolve)))
    search.focus()
    if (departure === "click") {
      await user.click(screen.getByRole("button", { name: "Another task" }))
    } else {
      screen.getByRole("link", { name: "Help me choose a model" }).focus()
      await user.tab()
      await user.tab()
    }
    await act(() => new Promise(resolve => window.requestAnimationFrame(resolve)))
    expect(screen.getByRole("button", { name: "Another task" })).toHaveFocus()
    expect(trigger).toHaveAttribute("aria-expanded", "false")
  })

  it("returns externally opened picker Escape to its visible opener", async () => {
    nativeSetupFixture.realModelPicker = true
    const user = userEvent.setup()
    render(<ConfigProvider theme={{ token: { motion: false } }}>
      <button data-testid="external-model-opener" onClick={() => openModelSelector({ returnFocusSelector: "[data-testid='external-model-opener']" })}>Choose another model</button>
      <PlaygroundForm droppedFiles={[]} />
    </ConfigProvider>)
    const opener = screen.getByRole("button", { name: "Choose another model" })
    await user.click(opener)
    const search = await screen.findByRole("textbox", { name: "Search models" })
    await waitFor(() => expect(search).toHaveFocus())
    fireEvent.keyDown(search, { key: "Escape", keyCode: 27 })
    await waitFor(() => expect(screen.getByTestId("model-selector")).toHaveAttribute("aria-expanded", "false"))
    await waitFor(() => expect(opener).toHaveFocus())
  })

  it.each(["reopen", "unmount", "outside focus"])("retires a held external return on %s", async retirement => {
    nativeSetupFixture.realModelPicker = true
    const user = userEvent.setup()
    const view = (show = true) => <ConfigProvider theme={{ token: { motion: false } }}>
      <button>Another task</button>
      <button data-testid="external-model-opener" onClick={() => openModelSelector({ returnFocusSelector: "[data-testid='external-model-opener']" })}>Choose another model</button>
      {show && <PlaygroundForm droppedFiles={[]} />}
    </ConfigProvider>
    const { rerender, unmount } = render(view())
    await user.click(screen.getByRole("button", { name: "Choose another model" }))
    const search = await screen.findByRole("textbox", { name: "Search models" })
    await waitFor(() => expect(search).toHaveFocus())
    const frames = new Map<number, FrameRequestCallback>()
    let id = 0
    const request = vi.spyOn(window, "requestAnimationFrame").mockImplementation(callback => { frames.set(++id, callback); return id })
    const cancel = vi.spyOn(window, "cancelAnimationFrame").mockImplementation(frame => { frames.delete(frame) })
    try {
      fireEvent.keyDown(search, { key: "Escape", keyCode: 27 })
      expect(screen.getByTestId("model-selector")).toHaveAttribute("aria-expanded", "false")
      if (retirement === "reopen") {
        act(() => openModelSelector({ returnFocusSelector: "[data-testid='external-model-opener']" }))
      } else {
        if (retirement === "unmount") rerender(view(false))
        screen.getByRole("button", { name: "Another task" }).focus()
      }
      for (let frame = 0; frame < 4; frame++) {
        const pending = [...frames.values()]
        frames.clear()
        act(() => pending.forEach(callback => callback(0)))
      }
      if (retirement === "reopen") expect(screen.getByRole("textbox", { name: "Search models" })).toHaveFocus()
      else expect(screen.getByRole("button", { name: "Another task" })).toHaveFocus()
    } finally {
      unmount()
      request.mockRestore()
      cancel.mockRestore()
    }
  })

  it.each([false, true])("keeps one real Casual customization trigger and primary controls in the mobile cockpit: %s", async mobile => {
    viewportState.mobile = mobile
    viewportState.mode = "casual"
    viewportState.realToolbar = true
    const user = userEvent.setup()
    render(<PlaygroundForm droppedFiles={[]} mobileCockpitModeActive />)
    expect(screen.queryByRole("button", { name: /(?:Hide|Show) composer options/ })).not.toBeInTheDocument()
    const trigger = screen.getByTestId("composer-options-toggle")
    await waitFor(() => expect(trigger).toBeEnabled())
    expect(trigger).toBeVisible()
    expect(screen.getByRole("button", { name: /Select model|deepseek-chat/i })).toBeVisible()
    expect(screen.getByTestId("composer-inline-send-control")).toBeVisible()
    await user.click(trigger)
    const customization = screen.getByRole("region", { name: "Composer tools" })
    expect(customization).toBeVisible()
    const buddy = within(customization).getByRole("button", { name: "Buddy & Persona", hidden: true })
    expect(buddy).toBeVisible()
    expect(within(screen.getByRole("region", { name: "Prompts & people" })).getByRole("button", { name: "Role-play setup" })).toBeVisible()
  })

  it("allows Customize Escape while the real embedded Search & Context section remains open", async () => {
    viewportState.mode = "casual"
    viewportState.realToolbar = true
    const user = userEvent.setup()
    render(<PlaygroundForm droppedFiles={[]} />)
    const trigger = screen.getByTestId("composer-options-toggle")
    await waitFor(() => expect(trigger).toBeEnabled())
    await user.click(trigger)
    await user.click(screen.getByRole("button", { name: "Search & Context" }))
    const search = screen.getByRole("button", { name: "Close Search & Context" })
    search.focus()
    await user.keyboard("{Escape}")
    expect(trigger).toHaveAttribute("aria-expanded", "false")
    expect(trigger).toHaveFocus()
    expect(screen.getByTestId("knowledge-ask")).toBeVisible()
  })

  it("keeps the actual character starter event and popup keyboard layers when Customize begins closed", async () => {
    viewportState.mode = "casual"
    viewportState.realToolbar = true
    const user = userEvent.setup()
    render(<PlaygroundForm droppedFiles={[]} />)
    const trigger = screen.getByTestId("composer-options-toggle")
    await waitFor(() => expect(trigger).toBeEnabled())
    expect(trigger).toHaveAttribute("aria-expanded", "false")
    act(() => window.dispatchEvent(new CustomEvent("tldw:open-assistant-select", { detail: { tab: "character" } })))
    const tab = await screen.findByRole("tab", { name: "Characters" })
    expect(trigger).toHaveAttribute("aria-expanded", "true")
    expect(tab).toHaveAttribute("aria-selected", "true")
    tab.focus()
    await user.keyboard("{Escape}")
    expect(screen.queryByTestId("assistant-select-panel")).not.toBeInTheDocument()
    expect(trigger).toHaveAttribute("aria-expanded", "true")
    await waitFor(() => expect(screen.getByTestId("character-select")).toHaveFocus())
    await user.keyboard("{Escape}")
    expect(trigger).toHaveAttribute("aria-expanded", "false")
    expect(trigger).toHaveFocus()
    expect(selectedAssistantMock.setSelectedAssistant).not.toHaveBeenCalled()
    expect(playgroundFormMessageOptionState.value.setMessages).not.toHaveBeenCalled()
  })

  it("allows acknowledged destination hydration while a saved setup scene finishes", async () => {
    prepareQualifiedSceneSource("native", "source-chat")
    nativeSetupFixture.realDrafts = true
    savedSetupFixture.raw = JSON.stringify([
      createStartupTemplateBundle({
        name: "Hydrated scene",
        selectedModel: null,
        systemPrompt: "",
        rolePlay: {
          source: "role-play-setup",
          identity: null,
          behavior: null,
          scene: null,
          generation: null,
          context: null
        }
      })
    ])
    let finish!: () => void
    const pending = new Promise<boolean>((resolve) => {
      finish = () => resolve(true)
    })
    saveActorSettingsMock.mockReturnValueOnce(pending)
    const user = userEvent.setup()
    const view = render(<PlaygroundForm droppedFiles={[]} />)
    try {
      await user.type(
        screen.getByTestId("composer-textarea"),
        "Source draft for scene"
      )
      await user.click(
        screen.getByRole("button", { name: "Open role-play setup" })
      )
      await user.click(
        await screen.findByRole("button", { name: "Preview saved setup" })
      )
      await user.click(
        await screen.findByRole("button", { name: "Apply saved template" })
      )
      await waitFor(() => expect(saveActorSettingsMock).toHaveBeenCalledOnce())
      await waitFor(() =>
        expect(screen.getByTestId("composer-textarea")).toHaveValue("")
      )
      await act(async () => {
        finish()
        await pending
      })
      await waitFor(() =>
        expect(
          screen.queryByRole("button", { name: "Apply saved template" })
        ).not.toBeInTheDocument()
      )
    } finally {
      view.unmount()
      finish()
      await pending
    }
  })

  it.each([1, 2])(
    "keeps the accepted startup identity while retrying %s failed model device writes",
    async (failures) => {
      const { source } = prepareQualifiedSceneSource(
        "native",
        "partial-model-source"
      )
      const bundle = createStartupTemplateBundle({
        name: "Partial model Retry",
        selectedModel: "openai:proposal",
        systemPrompt: "",
        rolePlay: {
          source: "role-play-setup",
          identity: null,
          behavior: null,
          scene: null,
          generation: null,
          context: null
        }
      })
      savedSetupFixture.raw = JSON.stringify([bundle])
      let revision = -1
      const retry = vi.fn(async () => true)
      for (let i = 1; i < failures; i++)
        retry.mockRejectedValueOnce(new Error("Device model retry failed"))
      source.beginSelectedModelSave.mockImplementation((model) => {
        source.setSelectedModel(model)
        revision = source.selectedModelRevision
        const completion = Promise.reject<boolean>(
          new Error("Device model write failed")
        )
        void completion.catch(() => undefined)
        return {
          completion,
          retry,
          isCurrent: () =>
            source.selectedModelRevision === revision &&
            source.selectedModel === model
        }
      })
      startupPreviewFixture.apply = null
      const user = userEvent.setup()
      const view = render(<PlaygroundForm droppedFiles={[]} />)
      try {
        await user.click(
          screen.getByRole("button", { name: "Open role-play setup" })
        )
        await user.click(
          await screen.findByRole("button", { name: "Preview saved setup" })
        )
        await user.click(
          await screen.findByRole("button", { name: "Apply saved template" })
        )
        await waitFor(() =>
          expect(nativeSetupFixture.error).toHaveBeenCalledOnce()
        )
        expect(saveActorSettingsMock).not.toHaveBeenCalled()
        const originalRevision = revision
        for (let i = 1; i < failures; i++) {
          await act(async () => {
            await startupPreviewFixture.apply!()
          })
          expect(retry).toHaveBeenCalledTimes(i)
          expect(
            screen.getByRole("button", { name: "Apply saved template" })
          ).toBeInTheDocument()
        }
        await act(async () => {
          await startupPreviewFixture.apply!()
        })
        expect(retry).toHaveBeenCalledTimes(failures)
        expect({
          identity: selectedAssistantMock.setSelectedAssistant.mock.calls.length,
          modelOperations: source.beginSelectedModelSave.mock.calls.length,
          revision: source.selectedModelRevision,
          scenes: saveActorSettingsMock.mock.calls.length
        }).toEqual({
          identity: 1,
          modelOperations: 1,
          revision: originalRevision,
          scenes: 1
        })
        expect(
          screen.queryByRole("button", { name: "Apply saved template" })
        ).not.toBeInTheDocument()
      } finally {
        view.unmount()
      }
    }
  )

  it("retries only the failed scene save for the same startup preview", async () => {
    prepareNativeSource()
    savedSetupFixture.raw = JSON.stringify([
      createStartupTemplateBundle({
        name: "Scene retry",
        selectedModel: null,
        systemPrompt: "",
        rolePlay: {
          source: "role-play-setup",
          identity: null,
          behavior: null,
          scene: null,
          generation: null,
          context: null
        }
      })
    ])
    saveActorSettingsMock.mockResolvedValueOnce(false)
    const user = userEvent.setup()
    render(<PlaygroundForm droppedFiles={[]} />)
    await user.click(screen.getByRole("button", { name: "Open role-play setup" }))
    await user.click(
      await screen.findByRole("button", { name: "Preview saved setup" })
    )
    await user.click(
      await screen.findByRole("button", { name: "Apply saved template" })
    )
    await waitFor(() => expect(nativeSetupFixture.error).toHaveBeenCalledOnce())
    expect(
      screen.getByRole("button", { name: "Apply saved template" })
    ).toBeInTheDocument()
    await user.click(screen.getByRole("button", { name: "Apply saved template" }))
    await waitFor(() =>
      expect(
        screen.queryByRole("button", { name: "Apply saved template" })
      ).not.toBeInTheDocument()
    )
    expect({
      identityWrites:
        selectedAssistantMock.setSelectedAssistant.mock.calls.length,
      sceneWrites: saveActorSettingsMock.mock.calls.length,
      settingsWrites: chatModelSettingsState.updateSettings.mock.calls.length
    }).toEqual({ identityWrites: 1, sceneWrites: 2, settingsWrites: 1 })
  })

  it("admits one startup preview attempt while identity persistence is held", async () => {
    prepareNativeSource()
    savedSetupFixture.raw = JSON.stringify([
      createStartupTemplateBundle({
        name: "One attempt",
        selectedModel: null,
        systemPrompt: "",
        rolePlay: {
          source: "role-play-setup",
          identity: null,
          behavior: null,
          scene: null,
          generation: null,
          context: null
        }
      })
    ])
    let finish!: () => void
    const pending = new Promise<void>((resolve) => {
      finish = resolve
    })
    selectedAssistantMock.setSelectedAssistant.mockReturnValueOnce(pending)
    const user = userEvent.setup()
    render(<PlaygroundForm droppedFiles={[]} />)
    await user.click(screen.getByRole("button", { name: "Open role-play setup" }))
    await user.click(
      await screen.findByRole("button", { name: "Preview saved setup" })
    )
    const apply = startupPreviewFixture.apply!
    try {
      act(() => {
        void apply()
        void apply()
      })
      await waitFor(() =>
        expect(selectedAssistantMock.setSelectedAssistant).toHaveBeenCalled()
      )
      expect(selectedAssistantMock.setSelectedAssistant).toHaveBeenCalledTimes(1)
      expect(
        screen.getByRole("button", { name: "Apply saved template" })
      ).toBeDisabled()
    } finally {
      await act(async () => {
        finish()
        await pending
      })
    }
  })

  it("keeps a reopened same-template preview when an older scene save completes", async () => {
    prepareNativeSource()
    savedSetupFixture.raw = JSON.stringify([
      createStartupTemplateBundle({
        name: "Same preview",
        selectedModel: null,
        systemPrompt: "",
        rolePlay: {
          source: "role-play-setup",
          identity: null,
          behavior: null,
          scene: null,
          generation: null,
          context: null
        }
      })
    ])
    let finish!: () => void
    const pending = new Promise<boolean>((resolve) => {
      finish = () => resolve(true)
    })
    saveActorSettingsMock.mockReturnValueOnce(pending)
    const user = userEvent.setup()
    render(<PlaygroundForm droppedFiles={[]} />)
    await user.click(screen.getByRole("button", { name: "Open role-play setup" }))
    await user.click(
      await screen.findByRole("button", { name: "Preview saved setup" })
    )
    await user.click(
      await screen.findByRole("button", { name: "Apply saved template" })
    )
    try {
      await waitFor(() => expect(saveActorSettingsMock).toHaveBeenCalledOnce())
      await user.click(
        screen.getByRole("button", { name: "Cancel template preview" })
      )
      await user.click(
        screen.getByRole("button", { name: "Preview saved setup" })
      )
      await screen.findByRole("button", { name: "Apply saved template" })
      await act(async () => {
        finish()
        await pending
      })
      expect(
        screen.getByRole("button", { name: "Apply saved template" })
      ).toBeEnabled()
    } finally {
      await act(async () => {
        finish()
        await pending
      })
    }
  })

  it.each(["new destination draft", "destination edit and Undo"])(
    "retires a held startup scene after accepted hydration and %s",
    async (change) => {
      prepareQualifiedSceneSource("native", "hydration-source")
      nativeSetupFixture.realDrafts = true
      savedSetupFixture.raw = JSON.stringify([
        createStartupTemplateBundle({
          name: "Owned hydration",
          selectedModel: null,
          systemPrompt: "",
          rolePlay: {
            source: "role-play-setup",
            identity: null,
            behavior: null,
            scene: null,
            generation: null,
            context: null
          }
        })
      ])
      let finish!: () => void
      const pending = new Promise<boolean>((resolve) => {
        finish = () => resolve(true)
      })
      saveActorSettingsMock.mockReturnValueOnce(pending)
      const publish = vi.fn<(settings: ActorSettings | null) => void>()
      const unsubscribe = useActorStore.subscribe((state, previous) => {
        if (state.settings !== previous.settings) publish(state.settings)
      })
      const user = userEvent.setup()
      const view = render(<PlaygroundForm droppedFiles={[]} />)
      try {
        await user.type(
          screen.getByTestId("composer-textarea"),
          "Source to retain"
        )
        await user.click(
          screen.getByRole("button", { name: "Open role-play setup" })
        )
        await user.click(
          await screen.findByRole("button", { name: "Preview saved setup" })
        )
        await user.click(
          await screen.findByRole("button", { name: "Apply saved template" })
        )
        await waitFor(() => expect(saveActorSettingsMock).toHaveBeenCalledOnce())
        await waitFor(() =>
          expect(screen.getByTestId("composer-textarea")).toHaveValue("")
        )
        await user.type(screen.getByTestId("composer-textarea"), "My newer draft")
        if (change === "destination edit and Undo")
          await user.clear(screen.getByTestId("composer-textarea"))
        await act(async () => {
          finish()
          await pending
        })
        expect(publish).not.toHaveBeenCalled()
        expect(
          screen.getByRole("button", { name: "Apply saved template" })
        ).toBeInTheDocument()
        expect(screen.getByTestId("composer-textarea")).toHaveValue(
          change === "new destination draft" ? "My newer draft" : ""
        )
      } finally {
        view.unmount()
        finish()
        await pending
        unsubscribe()
      }
    }
  )

  it.each(["other model", "same model newer revision"])(
    "does not adopt a synchronous settings subscriber's %s after startup model ACK",
    async (change) => {
      const { source } = prepareNativeSource()
      nativeSetupFixture.realSettings = true
      const store = nativeSetupFixture.actualSettingsStore!
      savedSetupFixture.raw = JSON.stringify([
        createStartupTemplateBundle({
          name: "Frozen model",
          selectedModel: "openai:proposal",
          systemPrompt: "Proposed behavior",
          rolePlay: {
            source: "role-play-setup",
            identity: null,
            behavior: {
              source: "template",
              templateId: "proposal",
              templateTitle: "Proposal",
              templateCategory: "roleplay",
              systemPrompt: "Proposed behavior",
              modified: false
            },
            scene: null,
            generation: null,
            context: null
          }
        })
      ])
      let armed = true
      const unsubscribe = store.subscribe((state) => {
        if (armed && state.systemPrompt === "Proposed behavior") {
          armed = false
          source.setSelectedModel(
            change === "other model" ? "openai:newer" : "openai:proposal"
          )
        }
      })
      const user = userEvent.setup()
      const view = render(<PlaygroundForm droppedFiles={[]} />)
      try {
        await user.click(
          screen.getByRole("button", { name: "Open role-play setup" })
        )
        await user.click(
          await screen.findByRole("button", { name: "Preview saved setup" })
        )
        const apply = startupPreviewFixture.apply!
        await act(async () => {
          await apply()
        })
        expect(armed).toBe(false)
        expect(saveActorSettingsMock).not.toHaveBeenCalled()
        expect(
          screen.getByRole("button", { name: "Apply saved template" })
        ).toBeInTheDocument()
      } finally {
        view.unmount()
        unsubscribe()
      }
    }
  )

  it.each(
    ["inherited", "explicit"].flatMap((model) =>
      ["other model", "same model newer revision"].map((change) => ({
        model,
        change
      }))
    )
  )(
    "retires $model startup model ownership when selecting $change during held identity persistence",
    async ({ model, change }) => {
      const { source } = prepareNativeSource()
      savedSetupFixture.raw = JSON.stringify([
        createStartupTemplateBundle({
          name: "Held identity",
          selectedModel: model === "explicit" ? "openai:proposal" : null,
          systemPrompt: "",
          rolePlay: {
            source: "role-play-setup",
            identity: null,
            behavior: null,
            scene: null,
            generation: null,
            context: null
          }
        })
      ])
      let finish!: () => void
      const pending = new Promise<void>((resolve) => {
        finish = resolve
      })
      selectedAssistantMock.setSelectedAssistant.mockReturnValueOnce(pending)
      const user = userEvent.setup()
      const view = render(<PlaygroundForm droppedFiles={[]} />)
      try {
        await user.click(
          screen.getByRole("button", { name: "Open role-play setup" })
        )
        await user.click(
          await screen.findByRole("button", { name: "Preview saved setup" })
        )
        await user.click(
          await screen.findByRole("button", { name: "Apply saved template" })
        )
        await waitFor(() =>
          expect(
            selectedAssistantMock.setSelectedAssistant
          ).toHaveBeenCalledOnce()
        )
        const chosen =
          change === "other model" ? "openai:newer" : source.selectedModel
        act(() => source.setSelectedModel(chosen))
        await act(async () => {
          finish()
          await pending
        })
        expect(source.setSelectedModel).toHaveBeenCalledExactlyOnceWith(chosen)
        expect(source.beginSelectedModelSave).not.toHaveBeenCalled()
        expect(saveActorSettingsMock).not.toHaveBeenCalled()
        expect(
          screen.getByRole("button", { name: "Apply saved template" })
        ).toBeInTheDocument()
      } finally {
        view.unmount()
        finish()
        await pending
      }
    }
  )

  it.each(["other model", "same model newer revision"])(
    "retires an inherited-model startup scene when selecting %s while its save is held",
    async (change) => {
      const { source } = prepareNativeSource()
      savedSetupFixture.raw = JSON.stringify([
        createStartupTemplateBundle({
          name: "Inherited model",
          selectedModel: null,
          systemPrompt: "",
          rolePlay: {
            source: "role-play-setup",
            identity: null,
            behavior: null,
            scene: null,
            generation: null,
            context: null
          }
        })
      ])
      let finish!: () => void
      const pending = new Promise<boolean>((resolve) => {
        finish = () => resolve(true)
      })
      saveActorSettingsMock.mockReturnValueOnce(pending)
      const publish = vi.fn<(settings: ActorSettings | null) => void>()
      const unsubscribe = useActorStore.subscribe((state, previous) => {
        if (state.settings !== previous.settings) publish(state.settings)
      })
      const user = userEvent.setup()
      const view = render(<PlaygroundForm droppedFiles={[]} />)
      try {
        await user.click(
          screen.getByRole("button", { name: "Open role-play setup" })
        )
        await user.click(
          await screen.findByRole("button", { name: "Preview saved setup" })
        )
        await user.click(
          await screen.findByRole("button", { name: "Apply saved template" })
        )
        await waitFor(() => expect(saveActorSettingsMock).toHaveBeenCalledOnce())
        act(() =>
          source.setSelectedModel(
            change === "other model" ? "openai:newer" : source.selectedModel
          )
        )
        await act(async () => {
          finish()
          await pending
        })
        expect(publish).not.toHaveBeenCalled()
        expect(
          screen.getByRole("button", { name: "Apply saved template" })
        ).toBeInTheDocument()
        expect(source.beginSelectedModelSave).not.toHaveBeenCalled()
      } finally {
        view.unmount()
        finish()
        await pending
        unsubscribe()
      }
    }
  )

  it.each(["other model", "same model newer revision"])(
    "refuses an inherited-model scene Retry after explicitly selecting %s",
    async (change) => {
      const { source } = prepareNativeSource()
      savedSetupFixture.raw = JSON.stringify([
        createStartupTemplateBundle({
          name: "Inherited Retry",
          selectedModel: null,
          systemPrompt: "",
          rolePlay: {
            source: "role-play-setup",
            identity: null,
            behavior: null,
            scene: null,
            generation: null,
            context: null
          }
        })
      ])
      saveActorSettingsMock.mockResolvedValueOnce(false)
      const user = userEvent.setup()
      render(<PlaygroundForm droppedFiles={[]} />)
      await user.click(
        screen.getByRole("button", { name: "Open role-play setup" })
      )
      await user.click(
        await screen.findByRole("button", { name: "Preview saved setup" })
      )
      await user.click(
        await screen.findByRole("button", { name: "Apply saved template" })
      )
      await waitFor(() => expect(nativeSetupFixture.error).toHaveBeenCalledOnce())
      act(() =>
        source.setSelectedModel(
          change === "other model" ? "openai:newer" : source.selectedModel
        )
      )
      await act(async () => {
        await startupPreviewFixture.apply!()
      })
      expect({
        identities: selectedAssistantMock.setSelectedAssistant.mock.calls.length,
        scenes: saveActorSettingsMock.mock.calls.length
      }).toEqual({ identities: 1, scenes: 1 })
      expect(
        screen.getByRole("button", { name: "Apply saved template" })
      ).toBeInTheDocument()
      expect(source.beginSelectedModelSave).not.toHaveBeenCalled()
    }
  )

  it("stops startup preview publication when the Scene store subscriber synchronously starts New", async () => {
    const { source } = prepareNativeSource()
    savedSetupFixture.raw = JSON.stringify([
      createStartupTemplateBundle({
        name: "Retired publication",
        selectedModel: null,
        systemPrompt: "",
        rolePlay: {
          source: "role-play-setup",
          identity: null,
          behavior: null,
          scene: null,
          generation: null,
          context: null
        }
      })
    ])
    let publications = 0
    let retiredSettings: ActorSettings | null = null
    const unsubscribe = useActorStore.subscribe((state, previous) => {
      if (state.settings === previous.settings) return
      publications++
      retiredSettings = state.settings
      source.startNewChatDraft(false)
    })
    const user = userEvent.setup()
    const view = render(<PlaygroundForm droppedFiles={[]} />)
    try {
      await user.click(
        screen.getByRole("button", { name: "Open role-play setup" })
      )
      await user.click(
        await screen.findByRole("button", { name: "Preview saved setup" })
      )
      await user.click(
        await screen.findByRole("button", { name: "Apply saved template" })
      )
      await waitFor(() => expect(publications).toBe(1))
      expect(useActorStore.getState().settings).toBe(retiredSettings)
      expect(useActorStore.getState().applied?.target.isCurrent() ?? false).toBe(
        false
      )
      expect(
        screen.getByRole("button", { name: "Apply saved template" })
      ).toBeInTheDocument()
    } finally {
      view.unmount()
      unsubscribe()
    }
  })

  it.each(["New", "input edit and Undo"])(
    "refuses a retained startup Apply after its original source retires by %s",
    async (change) => {
      const { source } = prepareNativeSource()
      savedSetupFixture.raw = JSON.stringify([
        createStartupTemplateBundle({
          name: "Original source",
          selectedModel: null,
          systemPrompt: "",
          rolePlay: {
            source: "role-play-setup",
            identity: null,
            behavior: null,
            scene: null,
            generation: null,
            context: null
          }
        })
      ])
      const user = userEvent.setup()
      const view = render(<PlaygroundForm droppedFiles={[]} />)
      await user.click(
        screen.getByRole("button", { name: "Open role-play setup" })
      )
      await user.click(
        await screen.findByRole("button", { name: "Preview saved setup" })
      )
      const retainedApply = startupPreviewFixture.apply!
      if (change === "New") {
        act(() => source.startNewChatDraft(false))
        view.rerender(<PlaygroundForm droppedFiles={[]} />)
      } else {
        await user.type(screen.getByTestId("composer-textarea"), "Changed")
        await user.clear(screen.getByTestId("composer-textarea"))
      }
      await act(async () => {
        await retainedApply()
      })
      expect({
        identityWrites:
          selectedAssistantMock.setSelectedAssistant.mock.calls.length,
        sceneWrites: saveActorSettingsMock.mock.calls.length
      }).toEqual({ identityWrites: 0, sceneWrites: 0 })
    }
  )

  it.each(["accepted", "declined"])(
    "retires an obsolete route only after an %s fresh picker identity",
    async (outcome) => {
      const { source, reset } = prepareNativeSource()
      source.historyId = null
      source.chatDraftAddress = freshChatDraftAddress(false)
      nativeSetupFixture.controller!.reset()
      reset.mockClear()
      selectedAssistantMock.setSelectedAssistant.mockResolvedValueOnce(
        outcome === "accepted"
      )
      const events: unknown[] = []
      const listener = (event: Event) =>
        events.push((event as CustomEvent).detail)
      window.addEventListener("tldw:chat-route-replacement", listener)
      const user = userEvent.setup()
      render(<PlaygroundForm droppedFiles={[]} />)
      await user.click(
        screen.getByRole("button", { name: "Open role-play setup" })
      )
      await user.click(
        await screen.findByRole("button", { name: "Apply Character setup" })
      )
      await screen.findByText(outcome === "accepted" ? "Applied" : "Apply failed")
      window.removeEventListener("tldw:chat-route-replacement", listener)
      expect(events).toEqual(
        outcome === "accepted"
          ? [
              expect.objectContaining({
                characterId: "char-new",
                historyId: null,
                serverChatId: null
              })
            ]
          : []
      )
      expect(reset).not.toHaveBeenCalled()
    }
  )

  it("retires a saved source URL after identity ACK and before releasing its owner", async () => {
    const { source, reset } = prepareNativeSource()
    source.serverChatId = "saved-source"
    source.historyId = "saved-history"
    window.history.replaceState({}, "", "/chat?mode=character&characterId=old&chatId=saved-source")
    const events: unknown[] = []
    const listener = (event: Event) => events.push({ detail: (event as CustomEvent).detail,
      serverChatId: source.serverChatId, historyId: source.historyId, resetCalls: reset.mock.calls.length,
      selected: selectedAssistantMock.currentSelection?.id })
    window.addEventListener("tldw:chat-route-replacement", listener)
    const user = userEvent.setup()
    const view = render(<PlaygroundForm droppedFiles={[]} />)
    try {
      await user.click(screen.getByRole("button", { name: "Open role-play setup" }))
      await user.click(await screen.findByRole("button", { name: "Apply Character setup" }))
      await screen.findByText("Applied")
      expect(events).toEqual([{ detail: expect.objectContaining({ href: window.location.href,
        characterId: "char-new", serverChatId: "saved-source", historyId: "saved-history" }),
        serverChatId: "saved-source", historyId: "saved-history", resetCalls: 0, selected: "char-new" }])
      expect(reset).toHaveBeenCalledOnce()
    } finally { window.removeEventListener("tldw:chat-route-replacement", listener); view.unmount() }
  })

  it("keeps queued work and source draft while character entry blocks dispatch", async () => {
    const { source } = prepareNativeSource()
    source.serverChatId = "saved-source"
    const item = { ...buildQueuedRequest({ promptText: "Keep queued request", snapshot: { selectedModel: "openai:gpt-4.1" } }), status: "blocked", blockedReason: "restored_request" }
    source.queuedMessages = [item]
    const blocker = { active: true, title: "Finish character setup before sending.", actionLabel: "Choose character", onAction: vi.fn() }
    const user = userEvent.setup()
    render(<PlaygroundForm droppedFiles={[]} characterChatSendBlocker={blocker} />)
    await user.type(screen.getByTestId("composer-textarea"), "Source draft  ")
    const next = screen.getByRole("button", { name: "Retry next" })
    await user.click(screen.getByRole("button", { name: "View queue" }))
    const now = screen.getByRole("button", { name: "Retry now" })
    await user.click(next); await user.click(now)
    expect(next).toBeDisabled()
    expect(now).toBeDisabled()
    expect(next).toHaveAttribute("title", blocker.title)
    expect(source.queuedMessages).toEqual([item])
    expect(source.setSelectedModel).not.toHaveBeenCalled()
    expect(onSubmitMock).not.toHaveBeenCalled()
    expect(source.serverChatId).toBe("saved-source")
    expect(screen.getByTestId("composer-textarea")).toHaveValue("Source draft  ")
  })

  it("retains automatically queued work for review after character setup is cancelled", async () => {
    const { source } = prepareNativeSource()
    const item = buildQueuedRequest({ promptText: "Keep automatic queue", snapshot: { selectedModel: "openai:gpt-4.1" } })
    source.queuedMessages = [item]
    const view = render(<PlaygroundForm droppedFiles={[]} characterChatSendBlocker={{ active: true, title: "Finish character setup before sending.", actionLabel: "Choose character", onAction: vi.fn() }} />)
    expect(onSubmitMock).not.toHaveBeenCalled()
    view.rerender(<PlaygroundForm droppedFiles={[]} />)
    await act(async () => { await Promise.resolve() })
    expect(source.queuedMessages).toEqual([expect.objectContaining({ id: item.id, promptText: item.promptText, status: "blocked" })])
    expect(source.setSelectedModel).not.toHaveBeenCalled()
    expect(onSubmitMock).not.toHaveBeenCalled()
  })

  it("keeps saved native Character uploads blocked before live preparation", async () => {
    const { source } = prepareQualifiedSceneSource("native", "saved-source")
    const originalMessages = source.messages
    const item = {
      ...buildQueuedRequest({
        promptText: "Keep native queued upload",
        sourceContext: {
          uploadedFiles: [
            {
              id: "queued-file",
              filename: "scene.txt",
              content: "data:text/plain;base64,c2NlbmU=",
              processingMode: "add_to_chat"
            }
          ]
        },
        snapshot: { selectedModel: "openai:gpt-4.1" }
      }),
      status: "blocked",
      blockedReason: "restored_request"
    }
    source.queuedMessages = [item]
    const user = userEvent.setup()
    const view = render(<PlaygroundForm droppedFiles={[]} />)
    try {
      await user.type(screen.getByTestId("composer-textarea"), "Source draft  ")
      await user.click(screen.getByRole("button", { name: "Retry next" }))
      await waitFor(() =>
        expect(source.queuedMessages).toEqual([
          expect.objectContaining({
            id: item.id,
            promptText: item.promptText,
            status: "blocked",
            blockedReason: expect.stringContaining(
              "uses its saved roleplay setup"
            )
          })
        ])
      )
      expect(queuedDocumentPreparation).not.toHaveBeenCalled()
      expect(source.setSelectedModel).not.toHaveBeenCalled()
      expect(onSubmitMock).not.toHaveBeenCalled()
      expect(source.messages).toBe(originalMessages)
      expect(source.serverChatId).toBe("saved-source")
      expect(source.historyId).toBe("native-current-mirror")
      expect(screen.getByTestId("composer-textarea")).toHaveValue(
        "Source draft  "
      )
    } finally {
      view.unmount()
    }
  })

  it.each([false, true])(
    "retires held local queue preparation when a character setup blocker appears (then clears: %s)",
    async (clears) => {
      const { source, originalOwner } = prepareNativeSource()
      nativeSetupFixture.realSettings = true
      const modelStore = nativeSetupFixture.actualSettingsStore!
      modelStore.getState().setActiveSettingsScope("openai:deepseek-chat")
      const item = {
        ...buildQueuedRequest({
          promptText: "Keep queued upload",
          sourceContext: {
            uploadedFiles: [
              {
                id: "queued-file",
                filename: "scene.txt",
                content: "data:text/plain;base64,c2NlbmU=",
                processingMode: "add_to_chat"
              }
            ]
          },
          snapshot: { selectedModel: "openai:gpt-4.1" }
        }),
        status: "blocked",
        blockedReason: "restored_request"
      }
      source.queuedMessages = [item]
      const held =
        sceneSelectionGate<
          import("@/services/chat-document-processing").PreparedChatDocumentAttachments
        >()
      const prepared: import("@/services/chat-document-processing").PreparedChatDocumentAttachments =
        {
          blockedFiles: [],
          failedFiles: [],
          recoveryActions: [],
          contextFiles: [],
          requestOverrides: { contextFiles: [], uploadedFiles: [] },
          turnMetadata: { status: "ready", files: [] }
        }
      queuedDocumentPreparation.mockReturnValueOnce(held.promise)
      const user = userEvent.setup()
      const view = render(<PlaygroundForm droppedFiles={[]} />)
      try {
        await user.type(screen.getByTestId("composer-textarea"), "Source draft  ")
        await user.click(screen.getByRole("button", { name: "Retry next" }))
        await waitFor(() =>
          expect(queuedDocumentPreparation).toHaveBeenCalledOnce()
        )
        // Queue execution is allowed to publish its own declared model/scope
        // synchronously before preparation. All later publication remains fenced.
        expect(source.setSelectedModel).toHaveBeenCalledExactlyOnceWith(
          "openai:gpt-4.1"
        )
        expect(modelStore.getState().activeSettingsScope).toBe("openai:gpt-4.1")
        view.rerender(
          <PlaygroundForm
            droppedFiles={[]}
            characterChatSendBlocker={{
              active: true,
              title: "Finish character setup before sending.",
              actionLabel: "Choose character",
              onAction: vi.fn()
            }}
          />
        )
        if (clears) view.rerender(<PlaygroundForm droppedFiles={[]} />)
        await act(async () => {
          held.resolve(prepared)
          await held.promise
        })
        expect(source.setSelectedModel).toHaveBeenCalledExactlyOnceWith(
          "openai:gpt-4.1"
        )
        expect(onSubmitMock).not.toHaveBeenCalled()
        expect(source.setHistoryId).not.toHaveBeenCalled()
        expect(source.setServerChatId).not.toHaveBeenCalled()
        expect(source.setHistory).not.toHaveBeenCalled()
        expect(source.setMessages).not.toHaveBeenCalled()
        expect(source.queuedMessages).toEqual([
          expect.objectContaining({
            id: item.id,
            promptText: item.promptText,
            status: "blocked"
          })
        ])
        expect(source.serverChatId).toBeNull()
        expect(source.historyId).toBe(originalOwner.conversation_id)
        expect(screen.getByTestId("composer-textarea")).toHaveValue(
          "Source draft  "
        )
      } finally {
        held.resolve(prepared)
        await held.promise
        view.unmount()
      }
    }
  )

  it("refuses a fresh route while the current response is still running", async () => {
    const { source, reset } = prepareNativeSource()
    source.isProcessing = true
    let apply!: ApplyCharacterRoute
    const register = (next: ApplyCharacterRoute | null) => { if (next) apply = next }
    render(<PlaygroundForm droppedFiles={[]} onCharacterRouteApplyReady={register} />)
    await waitFor(() => expect(apply).toBeTypeOf("function"))
    await act(async () => expect(await apply("route-card", () => true)).toBe("failed"))
    expect(selectedAssistantMock.setSelectedAssistant).not.toHaveBeenCalled()
    expect(reset).not.toHaveBeenCalled()
  })

  it.each([
    'card:source',
    'card:history',
    'card:account',
    'card:choice',
    'card:tools',
    'card:connection hydration',
    'card:catalog hydration',
    'card:source hydration',
    'retention:text ABA',
    'identity:settings',
    'identity:account',
  ])('retires held fresh entry at %s', async (boundary) => {
    const { source, reset } = prepareNativeSource();
    const address = source.chatDraftAddress;
    let finish!: () => void;
    if (boundary.startsWith('card'))
      nativeSetupFixture.getCharacter.mockReturnValueOnce(
        new Promise((resolve) => {
          finish = () => resolve({ id: 'route-card', name: 'Route Card' });
        })
      );
    else {
      const pending = new Promise<boolean>((resolve) => {
        finish = () => resolve(true);
      });
      if (boundary.startsWith('retention')) nativeSetupFixture.retain.mockReturnValueOnce(pending);
      else selectedAssistantMock.setSelectedAssistant.mockReturnValueOnce(pending);
    }
    let apply!: ApplyCharacterRoute;
    const register = (next: ApplyCharacterRoute | null) => {
      if (next) apply = next;
    };
    const user = userEvent.setup();
    const view = render(<PlaygroundForm droppedFiles={[]} onCharacterRouteApplyReady={register} />);
    await user.type(screen.getByTestId('composer-textarea'), 'Held source  ');
    await waitFor(() => expect(apply).toBeTypeOf('function'));
    let result!: Promise<unknown>;
    act(() => {
      result = apply('route-card', () => true);
    });
    const barrier = boundary.startsWith('card')
      ? nativeSetupFixture.getCharacter
      : boundary.startsWith('retention')
        ? nativeSetupFixture.retain
        : selectedAssistantMock.setSelectedAssistant;
    await waitFor(() => expect(barrier).toHaveBeenCalledOnce());
    if (boundary === 'card:source') {
      source.chatDraftAddress = { ...address, id: 'newer-draft' };
      source.chatDraftRevision++;
    }
    if (boundary === 'card:history') {
      nativeSetupFixture.controller!.reset();
      reset.mockClear();
    }
    if (boundary.endsWith('account')) nativeSetupFixture.ownerCurrent = false;
    if (boundary === 'card:choice') {
      nativeSetupFixture.assistantRevision++;
      act(() =>
        selectedAssistantMock.externalMirror?.({
          kind: 'character',
          id: 'newer',
          name: 'Newer',
          metadata: { selectionMode: 'tracked' },
        })
      );
    }
    if (boundary === 'card:tools') nativeSetupFixture.tools = { tools: [{}] };
    if (boundary === 'card:connection hydration') {
      playgroundFormConnectionState.phase = 'connecting';
      view.rerender(<PlaygroundForm droppedFiles={[]} onCharacterRouteApplyReady={register} />);
      playgroundFormConnectionState.phase = 'connected';
      view.rerender(<PlaygroundForm droppedFiles={[]} onCharacterRouteApplyReady={register} />);
    }
    if (boundary === 'card:catalog hydration') {
      nativeSetupFixture.models = [...nativeSetupFixture.models];
      view.rerender(<PlaygroundForm droppedFiles={[]} onCharacterRouteApplyReady={register} />);
    }
    if (boundary === 'card:source hydration') {
      source.history = [...source.history];
      source.messages = [...source.messages];
      view.rerender(<PlaygroundForm droppedFiles={[]} onCharacterRouteApplyReady={register} />);
    }
    if (boundary === 'identity:settings')
      nativeSetupFixture.settings = { ...nativeSetupFixture.settings!, topK: 20 };
    if (boundary === 'retention:text ABA') {
      await user.clear(screen.getByTestId('composer-textarea'));
      await user.type(screen.getByTestId('composer-textarea'), 'Held source  ');
    }
    const destination = source.chatDraftAddress;
    await act(async () => {
      finish();
      expect(await result).toBe('retired');
    });
    expect(source.chatDraftAddress).toBe(destination);
    expect(reset).not.toHaveBeenCalled();
    expect(source.setHistoryId).not.toHaveBeenCalled();
    expect(source.setMessages).not.toHaveBeenCalled();
    expect(screen.getByTestId('composer-textarea')).toHaveValue('Held source  ');
    expect(onSubmitMock).not.toHaveBeenCalled();
    if (boundary.endsWith('hydration')) {
      await act(async () => expect(await apply('route-card', () => true)).toBe('applied'));
      expect(reset).toHaveBeenCalledOnce();
      expect(nativeSetupFixture.retain).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({ content: 'Held source  ' })
      );
      expect(onSubmitMock).not.toHaveBeenCalled();
    }
    view.unmount();
  });

  it("retains exact source via real draft persistence before accepting a fresh route", async () => {
    const { source, reset } = prepareNativeSource()
    nativeSetupFixture.realDrafts = true
    const address = source.chatDraftAddress
    let apply!: ApplyCharacterRoute
    const register = (next: ApplyCharacterRoute | null) => { if (next) apply = next }
    const user = userEvent.setup()
    render(<PlaygroundForm droppedFiles={[]} onCharacterRouteApplyReady={register} />)
    await waitFor(() => expect(apply).toBeTypeOf("function"))
    await user.type(screen.getByTestId("composer-textarea"), "Real retention\nExact spaces  ")
    let outcome: unknown
    await act(async () => { outcome = await apply("route-card", () => true) })
    expect(outcome).toBe("applied")
    expect([...draftBucket.records.values()].map(record => record.value)).toContainEqual(expect.objectContaining({ content: "Real retention\nExact spaces  " }))
    expect(source.chatDraftAddress).not.toBe(address)
    expect(reset).toHaveBeenCalledOnce()
    await waitFor(() => expect(screen.getByTestId("composer-textarea")).toHaveValue(""))
    expect(onSubmitMock).not.toHaveBeenCalled()
  })

  it.each(["saved same card", "fresh source"])("accepts Library entry as a new draft after retaining %s", async sourceKind => {
    const { reset, source } = prepareNativeSource()
    if (sourceKind === "saved same card") {
      source.serverChatAssistantKind = "character"
      source.serverChatCharacterId = "route-card"
      source.serverChatAssistantId = "route-card"
    } else {
      source.historyId = null
      source.serverChatId = null
      nativeSetupFixture.controller!.reset()
      reset.mockClear()
    }
    const address = source.chatDraftAddress
    const user = userEvent.setup()
    let apply!: ApplyCharacterRoute
    const register = (next: ApplyCharacterRoute | null) => { if (next) apply = next }
    render(<PlaygroundForm droppedFiles={[]} onCharacterRouteApplyReady={register} />)
    await user.type(screen.getByTestId("composer-textarea"), "Exact source\nTrailing spaces  ")
    await waitFor(() => expect(apply).toBeTypeOf("function"))
    await act(async () => expect(await apply("route-card", () => true)).toBe("applied"))
    expect(nativeSetupFixture.retain).toHaveBeenCalledWith(expect.any(String), expect.objectContaining({ content: "Exact source\nTrailing spaces  " }))
    expect(nativeSetupFixture.retain.mock.invocationCallOrder[0]).toBeLessThan(selectedAssistantMock.setSelectedAssistant.mock.invocationCallOrder[0])
    expect(source.chatDraftAddress).not.toBe(address)
    expect(reset).toHaveBeenCalledOnce()
    expect(source.setHistoryId).toHaveBeenCalledWith(null, { preserveServerChatId: false })
    expect(source.setSelectedModel).not.toHaveBeenCalled()
    expect(onSubmitMock).not.toHaveBeenCalled()
    expect(createChatCompletionMock).not.toHaveBeenCalled()
  })

  it.each(["missing", "retention failure", "identity declined", "navigation declined", "input while loading"])("preserves the actual composer and source on Library %s", async outcome => {
    const { reset, source } = prepareNativeSource()
    const address = source.chatDraftAddress
    const user = userEvent.setup()
    let apply!: ApplyCharacterRoute
    const register = (next: ApplyCharacterRoute | null) => { if (next) apply = next }
    if (outcome === "missing") nativeSetupFixture.getCharacter.mockResolvedValueOnce(null)
    if (outcome === "retention failure") nativeSetupFixture.retain.mockResolvedValueOnce(false)
    if (outcome === "identity declined") selectedAssistantMock.setSelectedAssistant.mockResolvedValueOnce(false)
    const block = (event: Event) => event.preventDefault()
    if (outcome === "navigation declined") window.addEventListener("tldw:settings-navigation-request", block)
    let finish!: (value: unknown) => void
    if (outcome === "input while loading") nativeSetupFixture.getCharacter.mockReturnValueOnce(new Promise(resolve => { finish = resolve }))
    render(<PlaygroundForm droppedFiles={[]} onCharacterRouteApplyReady={register} />)
    await user.type(screen.getByTestId("composer-textarea"), "Keep source draft  ")
    await waitFor(() => expect(apply).toBeTypeOf("function"))
    let result: unknown
    if (outcome === "input while loading") {
      let pending!: Promise<unknown>
      act(() => { pending = apply("route-card", () => true) })
      await waitFor(() => expect(nativeSetupFixture.getCharacter).toHaveBeenCalledOnce())
      await user.type(screen.getByTestId("composer-textarea"), "newer")
      await act(async () => { finish({ id: "route-card", name: "Route Card" }); result = await pending })
    } else await act(async () => { result = await apply("route-card", () => true) })
    window.removeEventListener("tldw:settings-navigation-request", block)
    expect(result).toBe(outcome === "missing" ? "missing" : outcome === "input while loading" ? "retired" : "failed")
    expect(source.chatDraftAddress).toBe(address)
    expect(reset).not.toHaveBeenCalled()
    expect(source.setHistoryId).not.toHaveBeenCalled()
    expect(source.setMessages).not.toHaveBeenCalled()
    expect(screen.getByTestId("composer-textarea")).toHaveValue(outcome === "input while loading" ? "Keep source draft  newer" : "Keep source draft  ")
    expect(onSubmitMock).not.toHaveBeenCalled()
  })

  it.each(["accepted", "topK", "formatting guide"])(
    "preflights a fresh character before leaving a local source (%s)",
    async (blocker) => {
      const { reset, source, owner } = prepareNativeSource()
      const blocked = blocker !== "accepted"
      if (blocker === "topK") nativeSetupFixture.settings!.topK = 20
      nativeSetupFixture.formattingGuide = blocker === "formatting guide"
      const user = userEvent.setup()
      render(<PlaygroundForm droppedFiles={[]} />)
      await user.type(
        screen.getByTestId("composer-textarea"),
        "Source unsent draft"
      )
      await user.click(
        screen.getByRole("button", { name: "Open role-play setup" })
      )
      await user.click(
        await screen.findByRole("button", { name: "Apply Character setup" })
      )
      await screen.findByText(blocked ? "Apply failed" : "Applied")
      expect(reset).toHaveBeenCalledTimes(blocked ? 0 : 1)
      expect(source.setHistoryId).toHaveBeenCalledTimes(blocked ? 0 : 1)
      expect(selectedAssistantMock.setSelectedAssistant).toHaveBeenCalledTimes(
        blocked ? 0 : 1
      )
      expect(screen.getByTestId("composer-textarea")).toHaveValue(
        "Source unsent draft"
      )
      if (!blocked) {
        expect(source.setHistoryId).toHaveBeenCalledWith(null, {
          preserveServerChatId: false
        })
        expect(owner()).toBeNull()
        expect(nativeSetupFixture.actor).not.toHaveBeenCalled()
        expect(nativeSetupFixture.scene).not.toHaveBeenCalled()
      }
    }
  )

  it.each([
    "option during preparation",
    "settings during identity",
    "tools during identity",
    "scope during identity"
  ])("keeps the source when configuration changes: %s", async (change) => {
    const { reset, source, owner } = prepareNativeSource()
    let finish!: () => void
    const pending = new Promise<void>((resolve) => {
      finish = resolve
    })
    const invalidated = new AbortController()
    if (change === "scope during identity")
      nativeSetupFixture.snapshot.mockResolvedValueOnce({
        scopeSignal: new AbortController().signal,
        scopeInvalidatedSignal: invalidated.signal,
        requestScope: {
          config: { serverUrl: "http://localhost:8000", authMode: "single-user" },
          userId: null
        },
        release: vi.fn()
      })
    if (change === "option during preparation") {
      const snapshot = await nativeSetupFixture.snapshot()
      nativeSetupFixture.snapshot.mockClear().mockImplementationOnce(async () => {
        await pending
        return snapshot
      })
    } else selectedAssistantMock.setSelectedAssistant.mockReturnValueOnce(pending)
    const user = userEvent.setup()
    render(<PlaygroundForm droppedFiles={[]} />)
    await user.click(screen.getByRole("button", { name: "Open role-play setup" }))
    await user.click(
      await screen.findByRole("button", { name: "Apply Character setup" })
    )
    await waitFor(() =>
      expect(
        change === "option during preparation"
          ? nativeSetupFixture.snapshot
          : selectedAssistantMock.setSelectedAssistant
      ).toHaveBeenCalledOnce()
    )
    if (change === "option during preparation") source.toolChoice = "required"
    if (change === "settings during identity")
      nativeSetupFixture.settings = { ...nativeSetupFixture.settings!, topK: 20 }
    if (change === "tools during identity")
      nativeSetupFixture.tools = { tools: [{}] }
    if (change === "scope during identity") invalidated.abort()
    await act(async () => {
      finish()
      await pending
    })
    await screen.findByText("Apply failed")
    expect(reset).not.toHaveBeenCalled()
    expect(source.setHistoryId).not.toHaveBeenCalled()
    expect(owner()).toBe(nativeSetupFixture.localOwner)
    if (change !== "option during preparation") {
      expect(nativeSetupFixture.identityGuard?.()).toBe(false)
      expect(selectedAssistantMock.currentSelection).not.toEqual(
        expect.objectContaining({ id: "char-new" })
      )
    }
  })

  it.each(["cancelled", "failed", "declined"])(
    "retains the source when character identity persistence is %s",
    async (outcome) => {
      const { reset, source, owner } = prepareNativeSource()
      let finish!: (accepted?: boolean) => void
      let fail!: (error: Error) => void
      const pending = new Promise<boolean | void>((resolve, reject) => {
        finish = resolve
        fail = reject
      })
      selectedAssistantMock.setSelectedAssistant.mockReturnValueOnce(pending)
      const user = userEvent.setup()
      render(<PlaygroundForm droppedFiles={[]} />)
      await user.type(
        screen.getByTestId("composer-textarea"),
        "Keep source draft"
      )
      await user.click(
        screen.getByRole("button", { name: "Open role-play setup" })
      )
      await user.click(
        await screen.findByRole("button", { name: "Apply Character setup" })
      )
      await waitFor(() =>
        expect(selectedAssistantMock.setSelectedAssistant).toHaveBeenCalledOnce()
      )
      if (outcome === "cancelled") {
        await user.click(screen.getByRole("button", { name: "Cancel setup" }))
        expect(nativeSetupFixture.identityGuard?.()).toBe(false)
        await act(async () => {
          finish()
          await pending
        })
      } else if (outcome === "declined") {
        await act(async () => {
          finish(false)
          await pending
        })
        await screen.findByText("Apply failed")
      } else {
        await act(async () => {
          fail(new Error("storage unavailable"))
          await pending.catch(() => undefined)
        })
        await screen.findByText("Apply failed")
      }
      expect(reset).not.toHaveBeenCalled()
      expect(source.setHistoryId).not.toHaveBeenCalled()
      expect(owner()).toBe(nativeSetupFixture.localOwner)
      expect(screen.getByTestId("composer-textarea")).toHaveValue(
        "Keep source draft"
      )
      expect(selectedAssistantMock.currentSelection).not.toEqual(
        expect.objectContaining({ id: "char-new" })
      )
    }
  )

  it.each(["topK", "inherited template", "accepted"])("uses the proposed model's real settings scope (%s)", async blocker => {
    const blocked = blocker !== "accepted"
    const { reset, source } = prepareNativeSource()
    nativeSetupFixture.realSettings = true
    const store = nativeSetupFixture.actualSettingsStore!
    const sourceScope = "openai:deepseek-chat"
    const targetScope = "openai:gpt-4.1"
    if (blocker === "inherited template") store.getState().updateSettings({ systemPromptTemplateId: "global-template" })
    store.getState().setActiveSettingsScope(sourceScope)
    store.getState().updateSettings({ temperature: 0.4 })
    store.getState().updateScopedSetting(targetScope, "topK", blocker === "topK" ? 20 : undefined)
    savedSetupFixture.raw = JSON.stringify([createStartupTemplateBundle({ name: "Target model", selectedModel: targetScope, systemPrompt: "Target instructions", presetKey: "custom", character: { id: "new", name: "New" } as Character })])
    const user = userEvent.setup()
    render(<PlaygroundForm droppedFiles={[]} />)
    await user.click(screen.getByRole("button", { name: "Open role-play setup" }))
    await user.click(await screen.findByRole("button", { name: "Preview saved setup" }))
    await user.click(await screen.findByRole("button", { name: "Apply saved template" }))
    if (blocked) await waitFor(() => expect(nativeSetupFixture.error).toHaveBeenCalledOnce())
    else await waitFor(() => expect(reset).toHaveBeenCalledOnce())
    expect(reset).toHaveBeenCalledTimes(blocked ? 0 : 1)
    expect(store.getState().getEffectiveSettings(sourceScope).systemPrompt).not.toBe("Target instructions")
    if (!blocked) {
      expect(store.getState().getEffectiveSettings(targetScope).systemPrompt).toBe("Target instructions")
      expect(source.setSelectedModel).toHaveBeenCalledWith(targetScope)
    }
  })

  it.each(
    ["model setter", "settings scope subscriber"].flatMap((boundary) =>
      ["other model", "same model newer revision"].map((change) => ({
        boundary,
        change
      }))
    )
  )(
    "does not certify native startup destination after $boundary selects $change",
    async ({ boundary, change }) => {
      const { source } = prepareNativeSource()
      nativeSetupFixture.realSettings = true
      const store = nativeSetupFixture.actualSettingsStore!
      const targetScope = "openai:gpt-4.1"
      const newerScope =
        change === "other model" ? "openai:deepseek-chat" : targetScope
      store.getState().setActiveSettingsScope("openai:deepseek-chat")
      savedSetupFixture.raw = JSON.stringify([
        createStartupTemplateBundle({
          name: "Native model ownership",
          selectedModel: targetScope,
          systemPrompt: "Proposed native behavior",
          presetKey: "custom",
          character: { id: "new", name: "New" } as Character
        })
      ])
      let armed = true
      const replace = () => {
        armed = false
        source.setSelectedModel(newerScope)
        store.getState().setActiveSettingsScope(newerScope)
      }
      source.setSelectedModel.mockImplementation((model: string) => {
        source.selectedModel = model
        source.selectedModelRevision++
        if (armed && boundary === "model setter" && model === targetScope)
          replace()
      })
      const unsubscribe = store.subscribe((state) => {
        if (
          armed &&
          boundary === "settings scope subscriber" &&
          state.activeSettingsScope === targetScope
        )
          replace()
      })
      startupPreviewFixture.apply = null
      const user = userEvent.setup()
      const view = render(<PlaygroundForm droppedFiles={[]} />)
      try {
        await user.click(
          screen.getByRole("button", { name: "Open role-play setup" })
        )
        await user.click(
          await screen.findByRole("button", { name: "Preview saved setup" })
        )
        await screen.findByRole("button", { name: "Apply saved template" })
        const apply = startupPreviewFixture.apply!
        await act(async () => {
          await apply()
        })
        expect(armed).toBe(false)
        expect(store.getState().activeSettingsScope).toBe(newerScope)
        expect(store.getState().systemPrompt).not.toBe("Proposed native behavior")
        expect(source.beginSelectedModelSave).not.toHaveBeenCalled()
        expect(saveActorSettingsMock).not.toHaveBeenCalled()
        const identityWrites =
          selectedAssistantMock.setSelectedAssistant.mock.calls.length
        const modelWrites = source.setSelectedModel.mock.calls.length
        await act(async () => {
          await apply()
        })
        expect(selectedAssistantMock.setSelectedAssistant).toHaveBeenCalledTimes(
          identityWrites
        )
        expect(source.setSelectedModel).toHaveBeenCalledTimes(modelWrites)
        expect(source.beginSelectedModelSave).not.toHaveBeenCalled()
        expect(
          screen.getByRole("button", { name: "Apply saved template" })
        ).toBeInTheDocument()
      } finally {
        view.unmount()
        unsubscribe()
      }
    }
  )

  it.each(["other model", "same model newer revision"] as const)(
    "does not export native ownership after a scope subscriber selects %s",
    async (change) => {
      const { source } = prepareNativeSource()
      nativeSetupFixture.realSettings = true
      source.selectedModel = "openai:deepseek-chat"
      const store = nativeSetupFixture.actualSettingsStore!
      const targetModel = "openai:gpt-4.1"
      const foreignModel =
        change === "other model" ? "openai:deepseek-chat" : targetModel
      store.getState().setActiveSettingsScope("openai:deepseek-chat")
      store.getState().updateSettings({ apiProvider: "openai" })
      savedSetupFixture.raw = JSON.stringify([
        createStartupTemplateBundle({
          name: "Owned changed-model setup",
          selectedModel: targetModel,
          systemPrompt: "Proposed scoped behavior",
          presetKey: "custom",
          character: { id: "char-new", name: "New character" }
        })
      ])
      let armed = false
      let publicationEntered = false
      const unsubscribe = store.subscribe((state) => {
        if (armed && state.activeSettingsScope === targetModel) {
          armed = false
          publicationEntered = true
          source.setSelectedModel(foreignModel)
        }
      })
      const user = userEvent.setup()
      const view = render(<PlaygroundForm droppedFiles={[]} />)
      try {
        await user.click(
          screen.getByRole("button", { name: "Open role-play setup" })
        )
        expect(store.getState().activeSettingsScope).toBe("openai:deepseek-chat")
        expect(source.selectedModel).toBe("openai:deepseek-chat")
        expect(source.selectedModelRevision).toBe(0)
        armed = true
        await user.click(
          await screen.findByRole("button", {
            name: "Apply owned model saved setup"
          })
        )
        await screen.findByText("Apply failed")
        expect(publicationEntered).toBe(true)
        expect(armed).toBe(false)
        expect(source.selectedModel).toBe(foreignModel)
        expect(source.selectedModelRevision).toBe(2)
        expect(nativeSetupFixture.destinationCurrent).toBeNull()
        expect(source.beginSelectedModelSave).not.toHaveBeenCalled()
        expect(store.getState().systemPrompt).not.toBe("Proposed scoped behavior")
        expect(saveActorSettingsMock).not.toHaveBeenCalled()
        expect(sceneBucket.set).not.toHaveBeenCalled()
      } finally {
        view.unmount()
        unsubscribe()
      }
    }
  )

  it("does not publish late native settings after an owned route receipt and a scope subscriber's newer model", async () => {
    const { source, address, revision } = prepareQualifiedSceneSource()
    nativeSetupFixture.realSettings = true
    const store = nativeSetupFixture.actualSettingsStore!
    const targetModel = "openai:gpt-4.1"
    store.getState().setActiveSettingsScope("openai:deepseek-chat")
    store.getState().updateSettings({ apiProvider: "openai" })
    savedSetupFixture.raw = JSON.stringify([
      createStartupTemplateBundle({
        name: "Retained native changed-model setup",
        selectedModel: targetModel,
        systemPrompt: "Retired route behavior",
        presetKey: "custom",
        character: { id: "char-new", name: "New character" }
      })
    ])
    const originalHref = window.location.href
    window.history.replaceState(
      {},
      "",
      "/chat?mode=character&characterId=char-new&chatId=native-current"
    )
    const receipt = installRouteReceiptListener(false)
    let armed = false
    let publicationsAtRetirement:
      | { routes: number; identities: number; scenes: number }
      | undefined
    const unsubscribe = store.subscribe((state) => {
      if (armed && state.activeSettingsScope === targetModel) {
        armed = false
        source.setSelectedModel("openai:deepseek-chat")
        publicationsAtRetirement = {
          routes: receipt.events.length,
          identities:
            selectedAssistantMock.setSelectedAssistant.mock.calls.length,
          scenes: saveActorSettingsMock.mock.calls.length
        }
      }
    })
    const user = userEvent.setup()
    const view = render(<PlaygroundForm droppedFiles={[]} />)
    try {
      await user.click(
        screen.getByRole("button", { name: "Open role-play setup" })
      )
      expect(store.getState().activeSettingsScope).toBe("openai:deepseek-chat")
      expect(source.selectedModelRevision).toBe(0)
      armed = true
      await user.click(
        await screen.findByRole("button", {
          name: "Apply owned model saved setup"
        })
      )
      await screen.findByText("Apply failed")
      expect(armed).toBe(false)
      // Route cancellation was accepted under original ownership, before the
      // subsequent model/scope writer. Retirement forbids further publication.
      expect(publicationsAtRetirement).toEqual({
        routes: 1,
        identities: 1,
        scenes: 0
      })
      expect(receipt.acknowledgements).toEqual([true])
      expect(receipt.events).toHaveLength(publicationsAtRetirement!.routes)
      expect(selectedAssistantMock.setSelectedAssistant).toHaveBeenCalledTimes(
        publicationsAtRetirement!.identities
      )
      expect(saveActorSettingsMock).toHaveBeenCalledTimes(
        publicationsAtRetirement!.scenes
      )
      expect(source.selectedModel).toBe("openai:deepseek-chat")
      expect(source.selectedModelRevision).toBe(2)
      expect(source.chatDraftAddress).toBe(address)
      expect(source.chatDraftRevision).toBe(revision)
      expect(source.beginSelectedModelSave).not.toHaveBeenCalled()
      expect(store.getState().systemPrompt).not.toBe("Retired route behavior")
      expect(sceneBucket.set).not.toHaveBeenCalled()
      expect(nativeSetupFixture.destinationCurrent).toBeNull()
    } finally {
      view.unmount()
      unsubscribe()
      receipt.remove()
      window.history.replaceState({}, "", originalHref)
    }
  })

  it("does not adopt an unowned default Character while this account preference is unresolved", async () => {
    selectedAssistantMock.defaultCharacter = { id: "4", name: "Alice private default", system_prompt: "Alice private prompt" }
    render(<PlaygroundForm droppedFiles={[]} />)
    await act(async () => { await Promise.resolve() })
    expect(selectedAssistantMock.setSelectedAssistant).not.toHaveBeenCalled()
  })

  it("waits for account verification before applying the configured default Character", async () => {
    selectedAssistantMock.isLoading = true
    selectedAssistantMock.ownedDefaultCharacter = { id: "7", name: "Default Archivist" }
    const view = render(<PlaygroundForm droppedFiles={[]} />)
    expect(selectedAssistantMock.setSelectedAssistant).not.toHaveBeenCalled()
    selectedAssistantMock.isLoading = false
    view.rerender(<PlaygroundForm droppedFiles={[]} />)
    await waitFor(() => expect(selectedAssistantMock.setSelectedAssistant).toHaveBeenCalledExactlyOnceWith(expect.objectContaining({ kind: "character", id: "7", name: "Default Archivist" })))
  })

  it("restores the opening control after the conditionally mounted setup drawer closes", async () => {
    const user = userEvent.setup()
    render(<PlaygroundForm droppedFiles={[]} />)
    const trigger = screen.getByRole("button", { name: "Open role-play setup" })
    await user.click(trigger)
    await user.click(await screen.findByRole("button", { name: "Cancel setup" }))
    expect(trigger).toHaveFocus()
  })

  it("keeps Buddy management reachable in mobile cockpit with the current conversation and setup action", async () => {
    viewportState.mobile = true
    playgroundFormMessageOptionState.value = { ...createMessageOptionState(), serverChatId: "mobile-chat" }
    const user = userEvent.setup()
    render(<>
      <PlaygroundForm droppedFiles={[]} mobileCockpitModeActive />
      <div role="dialog"><button onClick={() => useBuddyManagementStore.getState().conversationSettings?.()}>Edit conversation settings</button></div>
    </>)
    const trigger = screen.getByRole("button", { name: "Buddy & Persona" })
    expect(trigger).toBeVisible()
    await user.click(trigger)
    expect(useBuddyManagementStore.getState()).toMatchObject({ open: true, target: { scope_type: "conversation", scope_id: "mobile-chat" } })
    await user.click(screen.getByRole("button", { name: "Edit conversation settings" }))
    await user.click(await screen.findByRole("button", { name: "Cancel setup" }))
    expect(trigger).toHaveFocus()
  })

  it("detaches the previous tracked chat before persisting a deliberate replacement", async () => {
    const user = userEvent.setup()
    playgroundFormMessageOptionState.value.serverChatId = "chat-original"
    let finishCommit: () => void = () => undefined
    selectedAssistantMock.setSelectedAssistant.mockImplementationOnce(() => new Promise<void>((resolve) => { finishCommit = resolve }))
    render(<PlaygroundForm droppedFiles={[]} />)
    await user.click(screen.getByRole("button", { name: "Open role-play setup" }))
    await user.click(await screen.findByRole("button", { name: "Apply Persona setup" }))
    expect(selectedAssistantMock.setSelectedAssistant).toHaveBeenCalledWith(expect.objectContaining({ kind: "persona", id: "persona-guide" }))
    expect(playgroundFormMessageOptionState.value.setHistoryId).toHaveBeenCalledWith(null, { preserveServerChatId: false })
    expect(playgroundFormMessageOptionState.value.setServerChatId).toHaveBeenCalledWith(null)
    expect(chatModelSettingsState.updateSettings).not.toHaveBeenCalled()
    finishCommit()
    await screen.findByText("Applied")
    expect(playgroundFormMessageOptionState.value.setHistoryId).toHaveBeenCalledWith(null, { preserveServerChatId: false })
    expect(playgroundFormMessageOptionState.value.setMessages).toHaveBeenCalledWith([])
    expect(playgroundFormMessageOptionState.value.setServerChatId).toHaveBeenCalledWith(null)
    expect(chatModelSettingsState.updateSettings).toHaveBeenCalledOnce()
  })

  it("keeps the deliberate replacement detached and skips generation changes when persistence rejects", async () => {
    const user = userEvent.setup()
    playgroundFormMessageOptionState.value.serverChatId = "chat-original"
    selectedAssistantMock.setSelectedAssistant.mockRejectedValueOnce(new Error("storage unavailable"))
    render(<PlaygroundForm droppedFiles={[]} />)
    await user.click(screen.getByRole("button", { name: "Open role-play setup" }))
    await user.click(await screen.findByRole("button", { name: "Apply Persona setup" }))
    await screen.findByText("Apply failed")
    expect(playgroundFormMessageOptionState.value.setServerChatId).toHaveBeenCalledWith(null)
    expect(chatModelSettingsState.updateSettings).not.toHaveBeenCalled()
  })

  it("retains the active saved identity when inline setup chooses that same persona", async () => {
    playgroundFormMessageOptionState.value.serverChatId = "chat-original"
    playgroundFormMessageOptionState.value.serverChatAssistantKind = "persona"
    playgroundFormMessageOptionState.value.serverChatAssistantId = "persona-guide"
    const user = userEvent.setup()
    render(<PlaygroundForm droppedFiles={[]} />)
    await user.click(screen.getByRole("button", { name: "Open role-play setup" }))
    await user.click(await screen.findByRole("button", { name: "Apply Persona setup" }))
    await screen.findByText("Applied")
    expect(playgroundFormMessageOptionState.value.setServerChatId).not.toHaveBeenCalled()
  })

  it.each([
    "legacy character startup",
    "owned clear-identity role-play startup"
  ] as const)(
    "detaches through the real Form before one selection write for %s",
    async (path) => {
      const kind = path === "legacy character startup" ? "character" : null
      const identity = kind ? { kind, id: "new-identity", name: "New identity" } : null
      if (kind === null)
        prepareQualifiedSceneSource("native", "chat-original")
      else playgroundFormMessageOptionState.value.serverChatId = "chat-original"
      savedSetupFixture.raw = JSON.stringify([
        createStartupTemplateBundle({
          name: "Saved setup",
          selectedModel: null,
          systemPrompt: "New behavior",
          character:
            kind === "character"
              ? { id: identity!.id, name: identity!.name }
              : null,
          // This ordinary SDK startup has no Scene proposal. Its character identity
          // still uses the nonnative writer; role-play character ordering is tested
          // separately with original qualified native ownership.
          rolePlay:
            kind === null
              ? {
                  source: "role-play-setup",
                  identity,
                  behavior: null,
                  scene: null,
                  generation: null,
                  context: null
                }
              : null
        })
      ])
      const held = sceneSelectionGate<void>()
      selectedAssistantMock.setSelectedAssistant.mockReturnValueOnce(held.promise)
      const { view, apply } = await openOwnedStartupPreview()
      let application: Promise<unknown> | undefined
      try {
        act(() => {
          application = apply()
        })
        await waitFor(() =>
          expect(
            selectedAssistantMock.setSelectedAssistant
          ).toHaveBeenCalledOnce()
        )
        expect(
          playgroundFormMessageOptionState.value.setServerChatId
        ).toHaveBeenCalledWith(null)
        expect(
          selectedAssistantMock.setSelectedAssistant
        ).toHaveBeenCalledExactlyOnceWith(identity ? expect.objectContaining(identity) : null)
        if (kind === "character") {
          expect(nativeSetupFixture.scene).not.toHaveBeenCalled()
          expect(saveActorSettingsMock).not.toHaveBeenCalled()
        }
        let accepted: unknown
        await act(async () => {
          held.resolve(undefined)
          accepted = await application
        })
        expect(accepted).toBe(true)
        expect(
          screen.queryByRole("button", { name: "Apply saved template" })
        ).not.toBeInTheDocument()
      } finally {
        view.unmount()
        held.resolve(undefined)
        await Promise.allSettled([application])
      }
    }
  )

  it.each(["inline", "template"].flatMap(origin => ["target", "history", "roundtrip"].map(change => ({ origin, change }))))("ignores late $origin settings after $change changes", async ({ origin, change }) => {
    savedSetupFixture.raw = JSON.stringify([createStartupTemplateBundle({ name: "Replacement", selectedModel: "other-model", systemPrompt: "Stale template", character: { id: "new", name: "New" } as Character, rolePlay: { source: "role-play-setup", identity: { kind: "character", id: "new", name: "New" }, behavior: null, scene: null, generation: null, context: null } })])
    playgroundFormMessageOptionState.value.serverChatId = "original"
    let finishCommit!: () => void
    const pending = new Promise<void>(resolve => { finishCommit = resolve })
    selectedAssistantMock.setSelectedAssistant.mockReturnValueOnce(pending)
    const user = userEvent.setup()
    render(<PlaygroundForm droppedFiles={[]} />)
    await user.click(screen.getByRole("button", { name: "Open role-play setup" }))
    if (origin === "template") {
      await user.click(await screen.findByRole("button", { name: "Preview saved setup" }))
      await user.click(await screen.findByRole("button", { name: "Apply saved template" }))
    } else await user.click(await screen.findByRole("button", { name: "Apply Persona setup" }))
    const updatesBefore = chatModelSettingsState.updateSettings.mock.calls.length
    const promptsBefore = chatModelSettingsState.updateSetting.mock.calls.length
    if (change === "target") playgroundFormMessageOptionState.value.serverChatId = "newer-chat"
    if (change === "history") playgroundFormMessageOptionState.value.historyId = "newer-history"
    if (change === "roundtrip") usePlaygroundSessionStore.getState().cancelPendingRestore()
    await act(async () => { finishCommit(); await pending })
    expect(chatModelSettingsState.updateSettings.mock.calls.length).toBe(updatesBefore)
    expect(chatModelSettingsState.updateSetting.mock.calls.length).toBe(promptsBefore)
    expect(playgroundFormMessageOptionState.value.setSelectedModel).not.toHaveBeenCalledWith("other-model")
  })

  it.each(["inline", "template"])("ignores late %s settings after the Form unmounts", async origin => {
    savedSetupFixture.raw = JSON.stringify([createStartupTemplateBundle({ name: "Replacement", selectedModel: "other-model", systemPrompt: "Stale template", character: { id: "new", name: "New" } as Character, rolePlay: { source: "role-play-setup", identity: { kind: "character", id: "new", name: "New" }, behavior: null, scene: null, generation: null, context: null } })])
    let finishCommit!: () => void
    const pending = new Promise<void>(resolve => { finishCommit = resolve })
    selectedAssistantMock.setSelectedAssistant.mockReturnValueOnce(pending)
    const user = userEvent.setup()
    const view = render(<PlaygroundForm droppedFiles={[]} />)
    await user.click(screen.getByRole("button", { name: "Open role-play setup" }))
    if (origin === "template") {
      await user.click(await screen.findByRole("button", { name: "Preview saved setup" }))
      await user.click(await screen.findByRole("button", { name: "Apply saved template" }))
    } else await user.click(await screen.findByRole("button", { name: "Apply Persona setup" }))
    const updatesBefore = chatModelSettingsState.updateSettings.mock.calls.length
    view.unmount()
    await act(async () => { finishCommit(); await pending })
    expect(chatModelSettingsState.updateSettings.mock.calls.length).toBe(updatesBefore)
    expect(playgroundFormMessageOptionState.value.setSelectedModel).not.toHaveBeenCalledWith("other-model")
  })

  it.each([true, false])("retains none-to-none identity only for metadata-ready neutral saved chats: %s", async ready => {
    const canonical = resolveServerChatAssistantIdentity({ id: "neutral", source: "webui", character_id: null, assistant_kind: null, assistant_id: null })
    Object.assign(playgroundFormMessageOptionState.value, { serverChatId: "neutral", serverChatMetaLoaded: ready, serverChatAssistantKind: canonical.assistantKind, serverChatAssistantId: canonical.assistantId, serverChatCharacterId: canonical.characterId })
    const user = userEvent.setup()
    render(<PlaygroundForm droppedFiles={[]} />)
    await user.click(screen.getByRole("button", { name: "Open role-play setup" }))
    await user.click(await screen.findByRole("button", { name: "Apply clear identity" }))
    await screen.findByText("Applied")
    expect(playgroundFormMessageOptionState.value.setServerChatId).toHaveBeenCalledTimes(ready ? 0 : 1)
  })

  it.each(["identity", "scene"])("refuses owned tracked-persona %s startup before detachment or any writer", async proposal => {
    const f = prepareQualifiedSceneSource("native", "original-persona-refusal", "original-persona-history")
    const setup = ownedStartupBundle({ ...createDefaultActorSettings(), notes: "Unsupported persona Scene proposal" }, {
      kind: "persona", id: "guide", name: "Guide"
    })
    if (!setup.rolePlay) throw new Error("expected role-play proposal")
    if (proposal === "identity") setup.rolePlay.scene = null
    savedSetupFixture.raw = JSON.stringify([setup])
    const record = { value: createDefaultActorSettings(), updatedAt: 1730000000000 }
    sceneBucket.records.set(`actorSettings:owned:${f.target.key}`, record)
    const before = { chatDraftAddress: f.source.chatDraftAddress, chatDraftRevision: f.source.chatDraftRevision, historyId: f.source.historyId,
      serverChatId: f.source.serverChatId, history: f.source.history, messages: f.source.messages, selectedModel: f.source.selectedModel,
      selectedSystemPrompt: f.source.selectedSystemPrompt, ragPinnedResults: f.source.ragPinnedResults }
    const { view, apply } = await openOwnedStartupPreview()
    const modelBefore = nativeSetupFixture.actualSettingsStore!.getState()
    try {
      await act(async () => { expect(await apply()).toBe(false) })
      expect(nativeSetupFixture.error).toHaveBeenCalledWith(expect.objectContaining({ description: expect.stringMatching(/persona.*not supported.*overlay/i) }))
      expect(selectedAssistantMock.setSelectedAssistant).not.toHaveBeenCalled()
      expect(f.source.setChatDraftAddress).not.toHaveBeenCalled()
      expect(f.source.setHistoryId).not.toHaveBeenCalled()
      expect(f.source.setServerChatId).not.toHaveBeenCalled()
      expect(f.reset).not.toHaveBeenCalled()
      expect(saveActorSettingsMock).not.toHaveBeenCalled()
      expect(sceneBucket.set).not.toHaveBeenCalled()
      expect(sceneBucket.records.get(`actorSettings:owned:${f.target.key}`)).toBe(record)
      expect(nativeSetupFixture.actualSettingsStore!.getState()).toBe(modelBefore)
      expect(f.source).toMatchObject(before)
      expect(useActorStore.getState().applied).toBeNull()
      expect(screen.getByRole("button", { name: "Apply saved template" })).toBeInTheDocument()
    } finally { view.unmount() }
  })

  it("detaches an owned clear-identity startup before Scene persistence and suppresses its late presentation", async () => {
    const f = prepareQualifiedSceneSource("native", "old-chat", "old-history")
    savedSetupFixture.raw = JSON.stringify([
      createStartupTemplateBundle({
        name: "Setup with scene",
        selectedModel: null,
        systemPrompt: "",
        rolePlay: {
          source: "role-play-setup",
          identity: null,
          behavior: null,
          scene: null,
          generation: null,
          context: null
        }
      })
    ])
    const held = sceneSelectionGate<boolean>()
    saveActorSettingsMock.mockReturnValueOnce(held.promise)
    const { view, apply } = await openOwnedStartupPreview()
    let application: Promise<unknown> | undefined
    try {
      act(() => {
        application = apply()
      })
      await waitFor(() => expect(saveActorSettingsMock).toHaveBeenCalledOnce())
      expect(saveActorSettingsMock).toHaveBeenCalledWith(
        expect.objectContaining({
          historyId: null,
          serverChatId: null,
          target: expect.objectContaining({
            address: expect.objectContaining({ kind: "fresh" }),
            accountKey: f.target.accountKey
          }),
          expectedRecord: null
        })
      )
      expect(selectedAssistantMock.setSelectedAssistant).toHaveBeenCalledTimes(1)
      view.unmount()
      await act(async () => {
        held.resolve(true)
        await application
      })
      expect(useActorStore.getState().applied).toBeNull()
    } finally {
      view.unmount()
      held.resolve(true)
      await Promise.allSettled([application])
    }
  })

  it("passes the Form action guard to the drawer and invalidates it on a newer same-destination action", async () => {
    let release!: () => void
    const pending = new Promise<void>(resolve => { release = resolve })
    selectedAssistantMock.setSelectedAssistant.mockReturnValueOnce(pending)
    const user = userEvent.setup()
    render(<PlaygroundForm droppedFiles={[]} />)
    await user.click(screen.getByRole("button", { name: "Open role-play setup" }))
    await user.click(screen.getByRole("button", { name: "Apply Persona setup" }))
    const isCurrent = ownerGuardFixture.capture!()
    expect(isCurrent()).toBe(true)
    await user.click(screen.getByRole("button", { name: "Apply clear identity" }))
    expect(isCurrent()).toBe(false)
    await act(async () => { release(); await pending })
  })

  it.each(["principal", "authority", "equivalent"] as const)("fences the actual Drawer scene continuation at the %s boundary", async (boundary) => {
    const f = prepareQualifiedSceneSource("fresh")
    Object.assign(f.source, { serverChatAssistantKind: null, serverChatAssistantId: null, serverChatCharacterId: null })
    sceneBucket.records.set(`actorSettings:owned:${f.target.key}`, {
      value: { ...createDefaultActorSettings(), isEnabled: true, notes: "Prior account scene" },
      updatedAt: 1730000000000
    })
    nativeSetupFixture.realDrawer = true
    let finishScene!: () => void
    const pending = new Promise<boolean>(resolve => { finishScene = () => resolve(true) })
    saveActorSettingsMock.mockReturnValueOnce(pending)
    const user = userEvent.setup()
    const view = render(<PlaygroundForm droppedFiles={[]} />)
    try {
      await user.click(screen.getByRole("button", { name: "Open role-play setup" }))
      const drawer = await screen.findByRole("dialog", { name: "Role-play setup" })
      await waitFor(() => expect(within(drawer).getByRole("textbox", { name: "Scene notes" })).toHaveValue("Prior account scene"))
      await user.clear(within(drawer).getByRole("textbox", { name: "Scene notes" }))
      await user.type(within(drawer).getByRole("textbox", { name: "Scene notes" }), "Original account scene")
      await waitFor(() => expect(within(drawer).getByRole("button", { name: "Apply" })).not.toBeDisabled())
      await user.click(within(drawer).getByRole("button", { name: "Apply" }))
      await waitFor(() => expect(saveActorSettingsMock.mock.calls, JSON.stringify({ errors: nativeSetupFixture.error.mock.calls, text: drawer.textContent })).toHaveLength(1))
      const successor = { ...createDefaultActorSettings(), notes: "Successor account scene" }
      act(() => {
        useActorStore.setState({ settings: successor, preview: "Successor account preview", tokenCount: 77 })
        window.dispatchEvent(boundary === "principal" ? new Event("tldw:auth-principal-changed") : new CustomEvent("tldw:config-updated", { detail: { authorityChanged: boundary === "authority" } }))
      })
      await act(async () => { finishScene(); await pending })
      if (boundary === "equivalent") {
        expect(useActorStore.getState().settings?.notes).toBe("Original account scene")
        expect(useActorStore.getState().preview).toContain("Original account scene")
      } else {
        expect(useActorStore.getState()).toMatchObject({ settings: successor, preview: "Successor account preview", tokenCount: 77 })
      }
    } finally {
      view.unmount()
      finishScene()
      await pending
      useActorStore.getState().reset()
    }
  })

  it("clears identity with one awaited canonical selection write", async () => {
    const user = userEvent.setup()
    selectedAssistantMock.initialSelection = { kind: "character", id: "ada", name: "Ada Lovelace" }
    render(<PlaygroundForm droppedFiles={[]} />)
    await user.click(screen.getByRole("button", { name: "Open role-play setup" }))
    expect(screen.getByTestId("setup-character")).toHaveTextContent("ada")
    await user.click(await screen.findByRole("button", { name: "Apply clear identity" }))
    await screen.findByText("Applied")
    expect(selectedAssistantMock.setSelectedAssistant).toHaveBeenCalledExactlyOnceWith(null)
    expect(screen.getByTestId("setup-character")).toHaveTextContent("None")
  })

  it("clears a custom override only when the selected system template identity changes", async () => {
    const user = userEvent.setup()
    playgroundFormMessageOptionState.value = {
      ...createMessageOptionState(),
      selectedSystemPrompt: "prompt-1"
    }
    chatModelSettingsState.systemPrompt = "Conversation override"

    render(<PlaygroundForm droppedFiles={[]} />)

    await user.click(screen.getByRole("button", { name: "Reselect system prompt" }))

    expect(chatModelSettingsState.updateSetting).not.toHaveBeenCalled()

    await user.click(screen.getByRole("button", { name: "Change system prompt" }))

    expect(chatModelSettingsState.updateSetting.mock.calls).toEqual([
      ["systemPromptTemplateId", undefined],
      ["systemPrompt", undefined]
    ])
  })

  it("does not crash role-play state derivation when document context is null", () => {
    playgroundFormMessageOptionState.value = {
      ...createMessageOptionState(),
      documentContext: null
    }

    expect(() => render(<PlaygroundForm droppedFiles={[]} />)).not.toThrow()
    expect(screen.getByTestId("composer-toolbar")).toBeInTheDocument()
  })

  it("opens character selection from the starter and returns focus after selecting the default assistant", async () => {
    const user = userEvent.setup()
    renderRolePlayStarterHarness()

    await user.click(screen.getByRole("button", { name: /explore chat modes/i }))
    await user.click(
      await screen.findByRole("button", { name: /character chat/i })
    )

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

  it.each(["loading", "recovery"])("B02 retains the real composer draft through character %s", async (phase) => {
    const user = userEvent.setup()
    const onAction = vi.fn()
    onSubmitMock.mockResolvedValueOnce({ status: "submitted" } as never)
    const { source } = prepareNativeSource()
    source.selectedModel = "deepseek-chat"
    Object.assign(source, { serverChatAssistantKind: null, serverChatAssistantId: null, serverChatCharacterId: null })
    selectedCharacterState.value = { id: "ada", name: "Ada Lovelace" }
    const blocker = { active: true, title: "Wait for the selected character. Your draft is retained.", actionLabel: phase === "loading" ? "Loading character" : "Retry character", disabled: phase === "loading", onAction }
    const form = (blocked: typeof blocker | null) => <PlaygroundForm droppedFiles={[]} characterChatSendBlocker={blocked} />
    const { rerender } = render(form(blocker))
    const composer = screen.getByTestId("composer-textarea")
    await user.type(composer, "Keep this B02 question.")
    fireEvent.submit(composer.closest("form")!)
    expect(onSubmitMock).not.toHaveBeenCalled()
    expect(composer).toHaveValue("Keep this B02 question.")
    const button = within(screen.getByTestId("composer-inline-send-control")).getByRole("button", { name: blocker.actionLabel })
    if (phase === "loading") expect(button).toBeDisabled()
    else {
      onAction.mockClear()
      await user.click(button)
      expect(onAction).toHaveBeenCalledTimes(1)
    }
    rerender(form(null))
    expect(onSubmitMock).not.toHaveBeenCalled()
    expect(composer).toHaveValue("Keep this B02 question.")
    const send = within(screen.getByTestId("composer-inline-send-control")).getByRole("button", { name: "Send message", exact: true }) as HTMLButtonElement
    const recoveredForm = (screen.getByTestId("composer-textarea") as HTMLTextAreaElement).form!
    expect(send).toBeEnabled()
    expect(send.type).toBe("submit")
    expect(send.form).toBe(recoveredForm)
    expect(recoveredForm.checkValidity()).toBe(true)
    await user.click(send)
    await waitFor(() => expect(onSubmitMock.mock.calls, JSON.stringify({ errors: nativeSetupFixture.error.mock.calls })).toHaveLength(1))
  })

  it("turns blocked character chat composer SEND into setup without dropping the draft", async () => {
    const user = userEvent.setup()
    const onAction = vi.fn()
    selectedCharacterState.value = {
      id: "ada",
      name: "Ada Lovelace"
    }

    render(
      <PlaygroundForm
        droppedFiles={[]}
        characterChatSendBlocker={{
          active: true,
          title: "Configure the selected model provider before chatting as Ada Lovelace",
          actionLabel: "Open model settings",
          onAction
        }}
      />
    )

    const composer = screen.getByTestId("composer-textarea")
    await user.type(composer, "Continue the analytical role-play scene.")

    const setupButton = within(
      screen.getByTestId("composer-inline-send-control")
    ).getByRole("button", {
      name: /open model settings/i
    })
    expect(setupButton).toHaveAttribute(
      "title",
      expect.stringContaining("Ada Lovelace")
    )

    await user.click(setupButton)

    expect(onAction).toHaveBeenCalledTimes(1)
    expect(onSubmitMock).not.toHaveBeenCalled()
    expect(composer).toHaveValue("Continue the analytical role-play scene.")
    expect(setupButton).toHaveAttribute(
      "title",
      expect.stringContaining("Ada Lovelace")
    )
  })

  it("keeps knowledge ask sends behind the character chat setup blocker", async () => {
    const user = userEvent.setup()
    const onAction = vi.fn()
    selectedCharacterState.value = {
      id: "ada",
      name: "Ada Lovelace"
    }

    render(
      <PlaygroundForm
        droppedFiles={[]}
        characterChatSendBlocker={{
          active: true,
          title: "Configure the selected model provider before chatting as Ada Lovelace",
          actionLabel: "Open model settings",
          onAction
        }}
      />
    )

    await user.click(screen.getByTestId("knowledge-ask"))

    await waitFor(() => {
      expect(onAction).toHaveBeenCalledTimes(1)
    })
    expect(onSubmitMock).not.toHaveBeenCalled()
    expect(screen.getByTestId("composer-textarea")).toHaveValue(
      "Ask with this pinned source."
    )
  })

  it("shows character model-usability copy in the model selector", () => {
    render(
      <PlaygroundForm
        droppedFiles={[]}
        characterChatModelUsability={{
          status: "provider_unconfigured",
          canSend: false,
          selectedModelId: "deepseek-chat",
          providerQualifiedModelId: "custom:deepseek-chat",
          matchedModelId: "deepseek-chat",
          matchedProvider: "custom",
          recommendedAction: "open-model-settings",
          detailReason: "provider-unconfigured"
        }}
        characterChatModelUsabilityLabel="Provider setup needed"
        characterChatModelUsabilityTitle="Configure the selected model provider before chatting as Ada"
      />
    )

    const selector = screen.getByTestId("model-selector")
    expect(selector).toHaveTextContent("deepseek-chat")
    expect(selector).toHaveTextContent("Provider setup needed")
    expect(selector).toHaveAttribute(
      "title",
      "Configure the selected model provider before chatting as Ada"
    )
    expect(selector).not.toHaveTextContent("Healthy")
    expect(selector).not.toHaveTextContent("Ready")
  })
})

it("UAT355: a plain composer ignores an untracked cross-tab Character mirror", async () => {
  const { source } = prepareQualifiedSceneSource("fresh")
  const originalMessages = source.messages
  Object.assign(playgroundFormMessageOptionState.value, {
    historyId: null,
    serverChatId: null,
    serverChatAssistantKind: null,
    serverChatAssistantId: null,
    serverChatCharacterId: null,
    serverChatMetaLoaded: false,
    selectedAssistant: null,
    selectedCharacter: null,
    effectiveAssistantState: {
      mode: "plain",
      kind: null,
      id: null,
      displayName: null,
      avatarUrl: null,
      systemPromptSnapshot: null,
      source: "none"
    }
  })
  const view = render(<PlaygroundForm droppedFiles={[]} />)
  try {
    await act(async () => {
      selectedAssistantMock.externalMirror!({
        kind: "character",
        id: "foreign-robot",
        name: "Foreign Robot",
        greeting: "Foreign greeting",
        system_prompt: "Foreign persona prompt"
      })
    })
    expect
      .soft(screen.getByTestId("composer-context-items").textContent)
      .not.toContain("Foreign Robot")
    expect
      .soft(screen.getByTestId("composer-context-items").textContent)
      .not.toContain("Character setup")
    expect
      .soft(screen.queryByText("Add expression images for Foreign Robot."))
      .toBeNull()
    expect(
      screen.queryByText("Foreign greeting", { exact: true })
    ).not.toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: "Start with this greeting" })
    ).not.toBeInTheDocument()
    expect(nativeSetupFixture.getCharacter).not.toHaveBeenCalled()
    expect(source.messages).toBe(originalMessages)
    expect(onSubmitMock).not.toHaveBeenCalled()
  } finally {
    view.unmount()
  }
})
it.each(["tracked_character", "overlay", "fresh_character"])(
  "UAT355: preserves owned composer %s context",
  async (mode) => {
    const hasOwner = mode === "tracked_character"
    const { source, requestScope } = prepareQualifiedSceneSource(
      hasOwner ? "native" : "fresh",
      "owned-chat",
      "owned-chat-mirror"
    )
    if (hasOwner) {
      source.messages = [
        {
          id: "saved-owned-row",
          isBot: true,
          name: "Owned Character",
          message: "Saved owned scene",
          sources: []
        }
      ]
      source.history = [{ role: "assistant", content: "Saved owned scene" }]
    }
    const originalMessages = source.messages
    const originalTranscript = structuredClone(source.messages)
    const originalHistory = structuredClone(source.history)
    const selection = {
      kind: "character",
      id: "owned",
      name: "Owned Character",
      greeting: "Owned greeting",
      metadata: { selectionMode: mode === "overlay" ? "overlay" : "tracked" }
    }
    selectedAssistantMock.initialSelection = selection
    nativeSetupFixture.getCharacter.mockResolvedValue({
      id: "owned",
      name: "Owned Character",
      first_message: "Verified owned opening"
    })
    Object.assign(playgroundFormMessageOptionState.value, {
      historyId: hasOwner ? "owned-chat-mirror" : null,
      serverChatId: hasOwner ? "owned-chat" : null,
      serverChatAssistantKind: hasOwner ? "character" : null,
      serverChatAssistantId: hasOwner ? "owned" : null,
      serverChatCharacterId: hasOwner ? "owned" : null,
      serverChatMetaLoaded: true,
      selectedAssistant: selection,
      selectedCharacter: selection,
      effectiveAssistantState: {
        mode: mode === "overlay" ? "overlay" : "tracked_character",
        kind: "character",
        id: "owned",
        displayName: "Owned Character",
        avatarUrl: null,
        systemPromptSnapshot: null,
        source: mode === "overlay" ? "overlay" : "tracked"
      }
    })
    const view = render(<PlaygroundForm droppedFiles={[]} />)
    try {
      expect(screen.getByTestId("composer-context-items")).toHaveTextContent(
        "Owned Character"
      )
      expect(screen.getByTestId("composer-context-items")).toHaveTextContent(
        "Character setup"
      )
      if (mode === "fresh_character") {
        expect(
          screen.queryByRole("link", { name: "Edit expressions" })
        ).not.toBeInTheDocument()
        expect(
          await screen.findByText("Verified owned opening", { selector: "div" })
        ).toBeInTheDocument()
        expect(
          screen.getByRole("button", { name: "Start with this greeting" })
        ).toBeEnabled()
        expect(nativeSetupFixture.getCharacter).toHaveBeenCalledWith(
          "owned",
          expect.objectContaining({ requestScope })
        )
      } else {
        expect(
          screen.queryByText("Verified owned opening", { exact: true })
        ).not.toBeInTheDocument()
        expect(
          screen.queryByRole("button", { name: "Start with this greeting" })
        ).not.toBeInTheDocument()
        expect(nativeSetupFixture.getCharacter).not.toHaveBeenCalled()
      }
      expect(source.messages).toBe(originalMessages)
      expect(source.messages).toEqual(originalTranscript)
      expect(source.history).toEqual(originalHistory)
      expect(onSubmitMock).not.toHaveBeenCalled()
      expect(createChatCompletionMock).not.toHaveBeenCalled()
    } finally {
      view.unmount()
    }
  }
)


it("names only the server connection when selected-model readiness is unknown", () => {
  render(<PlaygroundForm droppedFiles={[]} />)

  const selector = screen.getByTestId("model-selector")
  expect(selector).toHaveTextContent("Server connected")
  expect(selector).not.toHaveTextContent("Healthy")
  expect(within(selector).getByText("Server connected")).toHaveAttribute(
    "title",
    "Status"
  )
})

it("keeps offline connection guidance with a selected model", () => {
  playgroundFormConnectionState.phase = "disconnected"
  playgroundFormConnectionState.isConnected = false
  render(<PlaygroundForm droppedFiles={[]} />)

  expect(screen.getByTestId("model-selector")).toHaveTextContent("Offline")
})
