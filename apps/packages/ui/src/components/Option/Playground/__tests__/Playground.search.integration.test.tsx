// @vitest-environment jsdom
import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, fireEvent, render as renderComponent, renderHook, screen, waitFor, within } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { mockReducedMotion } from "@/test-utils/reducedMotion"

import { useSimpleForm } from "@/hooks/useSimpleForm"
import { useSidepanelChatHandoffImport } from "../hooks/useSidepanelChatHandoffImport"
import { useHistorySelection, useHistorySelectionContext, type HistorySelectionController } from "@/hooks/chat/useHistorySelection"
import { useStoreMessageOption } from "@/store/option"
import { chatDraftOwnerKey } from "@/hooks/useChatDraftOwner"
import { buildChatSurfaceScopeKeyFromConfig } from "@/services/chat-surface-scope"
import { usePlaygroundSessionStore } from "@/store/playground-session"
import { resolveHistorySelection } from "@/utils/history-selection"
import { Playground } from "../Playground"
import { PageHelpModalHost } from "@/components/Common/PageHelpModalHost"
import { useTutorialStore } from "@/store/tutorials"
import {
  encodeSidepanelChatWebUiHandoff,
  SIDEPANEL_CHAT_WEBUI_HANDOFF_SOURCE
} from "@/services/tldw/sidepanel-chat-webui-handoff"

const h1 = vi.hoisted(() => ({ enabled: false, legacyNative: false, controller: null as HistorySelectionController | null, bookmarks: new Map<string, any>(), confirm: vi.fn(), historyInfo: vi.fn(), sawFreshHandle: false }))
const verifiedOwner = { config: { serverUrl: "http://chat.test", authMode: "multi-user" as const }, userId: "A", scopeKey: "verified-A" }
vi.mock("@/services/service-prompts", async original => {
  const actual = await original<typeof import("@/services/service-prompts")>()
  return { ...actual, resolveServicePromptScope: (...args: Parameters<typeof actual.resolveServicePromptScope>) => h1.enabled ? Promise.resolve(verifiedOwner) : actual.resolveServicePromptScope(...args) }
})
const timelineReveal = vi.hoisted(() => ({ current: null as null | (() => Promise<boolean>) }))
const h1Key = (scope: any, owner: any) => JSON.stringify([scope.profile_id, scope.client_session_id, owner.owner_key, owner.conversation_id])

const forkPresentation = vi.hoisted(() => ({controller: null as HistorySelectionController | null, mode: null as "ordinary" | "pending" | "fork" | null, settings: null as {authorNote: string} | null, blockAutomaticLoad: false, storedReferenceInvalid: false, opened: [] as unknown[]}))
vi.mock("@/hooks/chat/useHistorySelection", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("@/hooks/chat/useHistorySelection")>()
  return {
    ...actual,
    useHistorySelectionContext: () => {
      const controller = actual.useHistorySelectionContext()
      const {
        blockAutomaticLoad,
        storedReferenceInvalid,
        controller: override,
        mode,
        settings
      } = forkPresentation
      // A real context keeps its provider value through child-only state updates.
      return React.useMemo(() => {
        if (blockAutomaticLoad)
          return {
            ...controller,
            canAutomaticallyLoad: () => false,
            ...(storedReferenceInvalid
              ? {
                  getStoredReference: () => {
                    throw new Error("invalid_history_reference")
                  }
                }
              : {}),
            open: (...args: Parameters<typeof controller.open>) => {
              forkPresentation.opened.push(args[0])
              return controller.open(...args)
            }
          }
        return (
          override ??
          (mode
            ? {
                ...controller,
                settingsMode: () => mode,
                forkSettings: settings
              }
            : controller)
        )
      }, [
        controller,
        blockAutomaticLoad,
        storedReferenceInvalid,
        override,
        mode,
        settings
      ])
    }
  }
})

const messageOptionState = vi.hoisted(() => ({
  value: {
    messages: [
      { id: "m-1", message: "alpha message", isBot: false, role: "user" },
      { id: "m-2", message: "beta response", isBot: true, role: "assistant" }
    ],
    history: [],
    historyId: "history-1",
    serverChatId: "chat-1",
    isLoading: false,
    selectedModel: "model-1",
    selectedSystemPrompt: "prompt-1",
    selectedQuickPrompt: "quick-1",
    chatMode: "normal",
    webSearch: false,
    toolChoice: "none",
    temporaryChat: false,
    useOCR: false,
    fileRetrievalEnabled: false,
    ragMediaIds: null as number[] | null,
    setHistoryId: vi.fn(),
    setHistory: vi.fn(),
    setMessages: vi.fn(),
    setSelectedSystemPrompt: vi.fn(),
    setSelectedQuickPrompt: vi.fn(),
    setSelectedModel: vi.fn(),
    setServerChatId: vi.fn(),
    setChatMode: vi.fn(),
    setWebSearch: vi.fn(),
    setToolChoice: vi.fn(),
    setTemporaryChat: vi.fn(),
    setUseOCR: vi.fn(),
    setFileRetrievalEnabled: vi.fn(),
    setRagMediaIds: vi.fn(),
    setContextFiles: vi.fn(),
    createChatBranch: vi.fn(),
    streaming: false,
    effectiveAssistantState: {
      mode: "plain",
      kind: null,
      id: null,
      displayName: null,
      avatarUrl: null,
      systemPromptSnapshot: null,
      source: "none"
    },
    selectedAssistant: null,
    selectedCharacter: null,
    setSelectedCharacter: vi.fn(),
    compareMode: false,
    compareFeatureEnabled: false
  }
}))

const artifactsState = vi.hoisted(() => ({
  value: {
    isOpen: false,
    active: null,
    isPinned: false,
    history: [],
    unreadCount: 0,
    setOpen: vi.fn(),
    closeArtifact: vi.fn(),
    markRead: vi.fn()
  }
}))

const smartScrollState = vi.hoisted(() => ({
  value: {
    containerRef: { current: null } as React.MutableRefObject<HTMLDivElement | null>,
    isAutoScrollToBottom: true,
    autoScrollToBottom: vi.fn(),
    pauseAutoScroll: vi.fn()
  }
}))

const mobileViewportState = vi.hoisted(() => ({
  value: false
}))

const desktopViewportState = vi.hoisted(() => ({
  value: true
}))

const storageState = vi.hoisted(() => ({
  value: new Map<string, unknown>()
}))

const artifactFixture = vi.hoisted(() => ({
  id: "artifact-1",
  title: "Generated table",
  content: "a,b\n1,2",
  kind: "table" as const
}))

const storeOptionState = vi.hoisted(() => ({
  value: {
    compareParentByHistory: {} as Record<
      string,
      { parentHistoryId: string; clusterId?: string }
    >,
    uploadedFiles: [] as Array<Record<string, unknown>>,
    contextFiles: [] as Array<Record<string, unknown>>,
    setUploadedFiles: vi.fn(),
    setContextFiles: vi.fn()
  }
}))

const tldwClientState = vi.hoisted(() => ({
  getConfig: vi.fn(async () => h1.enabled ? { serverUrl: "http://chat.test", authMode: "multi-user" } : ({})),
  getProvidersStatus: vi.fn(async () => ({})),
  initialize: vi.fn(async () => null),
  getResearchBundle: vi.fn(async () => null),
  getDocumentUploadDraft: vi.fn(async () => ({ payload: {} })),
  deleteDocumentUploadDraft: vi.fn(async () => undefined)
}))

const routerState = vi.hoisted(() => ({
  hashRouter: false,
  navigate: vi.fn()
}))

type ChatSettingsSyncParams = {
  historyId: string | null
  serverChatId: string | null
}

type ChatSettingsPatchParams = ChatSettingsSyncParams & {
  patch: Record<string, unknown>
}

const chatSettingsState = vi.hoisted(() => ({
  syncChatSettingsForServerChat: vi.fn(
    async (_params: ChatSettingsSyncParams): Promise<unknown> => null
  ),
  applyChatSettingsPatch: vi.fn(
    async (_params: ChatSettingsPatchParams): Promise<unknown> => null
  )
}))

const loadLocalConversationMock = vi.hoisted(() => vi.fn(async () => {}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, defaultValue?: string, options?: Record<string, unknown>) => {
      const template = defaultValue || key
      if (!options) return template
      return template.replace(/\{\{(\w+)\}\}/g, (_match, token) => {
        const value = options[token]
        return value == null ? "" : String(value)
      })
    }
  })
}))

const fragmentComposer = vi.hoisted(() => ({ enabled: false, ready: false }))
// Actual import hook/form state; the large surrounding Form UI is an explicit boundary.
vi.mock("@/components/Option/Playground/PlaygroundForm", () => {
  const Composer = () => {
    const form = useSimpleForm({ initialValues: { message: "RESTORED DESTINATION  " } })
    const imported = useSidepanelChatHandoffImport({ ready: fragmentComposer.ready, destinationKey: "owned-destination", captureGuard: () => () => true,
      draftValue: form.values.message, getDraftRevision: form.getRevision, setMessageValue: value => form.setFieldValue("message", value), t: (_key, fallback) => fallback })
    return <div data-testid="playground-form"><textarea aria-label="Fragment composer" value={form.values.message} readOnly />
      {imported.conflict ? <div role="status" aria-label="Fragment review"><pre>{imported.conflict.handoff.draft.text}</pre><p>{imported.skippedFeedback}</p><button onClick={imported.replaceWithHandoffDraft}>Accept fragment draft</button><button onClick={imported.cancelHandoffImport}>Cancel fragment import</button></div> : null}</div>
  }
  return { PlaygroundForm: () => fragmentComposer.enabled ? <Composer /> : <div data-testid="playground-form" /> }
})

vi.mock("@/components/Option/Playground/PlaygroundChat", () => ({
  PlaygroundChat: React.forwardRef(function MockPlaygroundChat(
    props: {
      navigationRef?: React.MutableRefObject<any>
      searchQuery?: string
      matchedMessageIndices?: Set<number>
      activeSearchMessageIndex?: number | null
    },
    _ref
  ) {
    React.useEffect(() => { if (props.navigationRef && timelineReveal.current) props.navigationRef.current = { reveal: timelineReveal.current } }, [props.navigationRef])
    const selection = useHistorySelectionContext()
    if (h1.enabled) h1.controller = selection
    return (
      <div
        data-selected-history={h1.enabled ? selection?.capture?.status === "captured" ? selection.capture.selected_content.map(row => row.message).join(" / ") : selection?.status : ""}
        data-testid="playground-chat"
        data-search-query={props.searchQuery || ""}
        data-search-count={props.matchedMessageIndices?.size || 0}
        data-search-active-index={
          props.activeSearchMessageIndex == null ? "" : props.activeSearchMessageIndex
        }
      />
    )
  })
}))

vi.mock("@/components/Sidepanel/Chat/ArtifactsPanel", () => ({
  ArtifactsPanel: () => <div data-testid="artifacts-panel" />
}))

vi.mock("@/hooks/useMessageOption", () => ({
  useMessageOption: () => {
    const store = useStoreMessageOption()
    return h1.enabled ? { ...messageOptionState.value, ...store } : messageOptionState.value
  }
}))

vi.mock("@/hooks/usePlaygroundSessionPersistence", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/hooks/usePlaygroundSessionPersistence")>()
  return { usePlaygroundSessionPersistence: () => {
    if (h1.enabled) {
      const state = actual.usePlaygroundSessionPersistence()
      if (state.hasPersistedFreshDraft) h1.sawFreshHandle = true
      return state
    }
    return ({
    restoreSession: vi.fn(async () => false), sessionScopeReady: true, hasPersistedSession: false, persistedHistoryId: null, persistedServerChatId: null
  }) } }
})

vi.mock("@/hooks/useSelectedAssistant", () => ({
  useSelectedAssistant: () => [null, async () => undefined, {
    isLoading: false,
    readSelection: async () => null,
    isCurrent: () => true,
    setRenderValue: () => undefined
  }]
}))

vi.mock("@/hooks/playground-session-restore", () => ({
  shouldRestorePersistedPlaygroundSession: () => false
}))

vi.mock("@/services/app", () => ({
  webUIResumeLastChat: vi.fn(async () => false)
}))

vi.mock("@/db/dexie/helpers", () => ({
  getFullChatData: vi.fn(async (id: string) => ({ historyInfo: { id }, messages: [] })),
  getSessionFiles: vi.fn(async () => []),
  formatSelectedHistory: (capture: any) => ({ history: capture.selected_content.map((row: any) => ({ role: "assistant", content: row.message })), messages: capture.selected_content.map((row: any) => ({ id: row.id, message: row.message, isBot: true, name: "Assistant", sources: [] })) }),
  formatToChatHistory: vi.fn(),
  formatToMessage: vi.fn(),
  getPromptById: vi.fn(async () => null),
  getRecentChatFromWebUI: vi.fn(async () => null)
}))

vi.mock("@/store/model", () => ({
  useStoreChatModelSettings: () => ({ setSystemPrompt: vi.fn() })
}))

vi.mock("@/hooks/useSmartScroll", () => ({
  useSmartScroll: () => smartScrollState.value
}))

vi.mock("@/services/settings/ui-settings", () => ({
  CHAT_BACKGROUND_IMAGE_SETTING: "chatBackgroundImage",
  THEME_SETTING: {
    key: "theme",
    defaultValue: "dark"
  },
  HEADER_SHORTCUT_IDS: [],
  SIDEBAR_SHORTCUT_IDS: [],
  CHAT_WINDOW_OPACITY_SETTING: "chatWindowOpacity",
  CHAT_MESSAGE_OPACITY_SETTING: "chatMessageOpacity",
  CHAT_CHARACTER_IMAGE_OPACITY_SETTING: "chatCharacterImageOpacity",
  resolveOpacityAlpha: (value: unknown, fallback = 35) =>
    typeof value === "number" && Number.isFinite(value)
      ? value / 100
      : fallback / 100
}))

vi.mock("../Knowledge/utils/unsupported-types", () => ({
  otherUnsupportedTypes: []
}))

vi.mock("@/store/option", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/store/option")>()
  const useStoreMessageOption = (selector: any = (state: any) => state) => h1.enabled ? actual.useStoreMessageOption(selector) : selector(storeOptionState.value)
  useStoreMessageOption.getState = () => h1.enabled ? actual.useStoreMessageOption.getState() : storeOptionState.value
  useStoreMessageOption.setState = actual.useStoreMessageOption.setState
  useStoreMessageOption.subscribe = actual.useStoreMessageOption.subscribe
  return { useStoreMessageOption }
})

vi.mock("@/services/tldw/TldwApiClient", async original => ({
  ...await original<typeof import("@/services/tldw/TldwApiClient")>(),
  tldwClient: tldwClientState
}))

vi.mock("@/store/artifacts", () => ({
  useArtifactsStore: (selector: (state: typeof artifactsState.value) => unknown) =>
    selector(artifactsState.value)
}))

vi.mock("@/hooks/useSetting", () => ({
  useSetting: (setting: string) => {
    if (setting === "chatWindowOpacity") return [35]
    if (setting === "chatMessageOpacity") return [60]
    if (setting === "chatCharacterImageOpacity") return [100]
    return [""]
  }
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (key: string, defaultValue: unknown) => [
    storageState.value.has(key) ? storageState.value.get(key) : defaultValue,
    vi.fn()
  ]
}))

vi.mock("@/hooks/useMediaQuery", () => ({
  useMobile: () => mobileViewportState.value,
  useDesktop: () => desktopViewportState.value
}))

vi.mock("@/services/chat-settings", async importOriginal => ({
  ...await importOriginal<typeof import("@/services/chat-settings")>(),
  syncChatSettingsForServerChat: (params: ChatSettingsSyncParams) =>
    chatSettingsState.syncChatSettingsForServerChat(params),
  applyChatSettingsPatch: (params: ChatSettingsPatchParams) =>
    chatSettingsState.applyChatSettingsPatch(params)
}))

vi.mock("@/hooks/useLoadLocalConversation", () => ({
  useLoadLocalConversation: () => loadLocalConversationMock,
  restoreReadableLocalComparison: vi.fn(async () => false)
}))

vi.mock("../playground-shortcuts", () => ({
  resolvePlaygroundShortcutAction: () => null
}))

vi.mock("@/hooks/useCharacterGreeting", () => ({
  useCharacterGreeting: () => undefined
}))

vi.mock("react-router-dom", async () => {
  const actual = await vi.importActual<typeof import("react-router-dom")>(
    "react-router-dom"
  )
  return {
    ...actual,
    useNavigate: () => routerState.navigate,
    useLocation: () => {
      const href = window.location.href
      const hashRouter = routerState.hashRouter
      return React.useMemo(() => {
        const current = new URL(href)
        if (hashRouter && current.hash.startsWith("#/")) {
          const route = new URL(current.hash.slice(1), current.origin)
          return { pathname: route.pathname, search: route.search, hash: route.hash, state: null, key: "hash-location" }
        }
        return { pathname: current.pathname || "/chat", search: current.search, hash: current.hash, state: null, key: "test-location" }
      }, [href, hashRouter])
    }
  }
})

let queryClient: QueryClient
beforeEach(() => { queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } }) })
afterEach(() => queryClient.clear())
const render = (ui: React.ReactElement) => renderComponent(ui, {
  wrapper: ({ children }) => <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
})

describe("Playground thread search integration", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    fragmentComposer.enabled = false
    fragmentComposer.ready = false
    h1.confirm.mockClear()
    h1.enabled = false
    forkPresentation.controller = null
    timelineReveal.current = null
    h1.legacyNative = false
    forkPresentation.mode = null
    forkPresentation.settings = null
    forkPresentation.blockAutomaticLoad = false
    routerState.hashRouter = false
    storageState.value.clear()
    mobileViewportState.value = false
    desktopViewportState.value = true
    loadLocalConversationMock.mockClear()
    messageOptionState.value.historyId = "history-1"
    messageOptionState.value.serverChatId = "chat-1"
    messageOptionState.value.selectedModel = "model-1"
    messageOptionState.value.selectedSystemPrompt = "prompt-1"
    messageOptionState.value.selectedQuickPrompt = "quick-1"
    messageOptionState.value.chatMode = "normal"
    messageOptionState.value.webSearch = false
    messageOptionState.value.toolChoice = "none"
    messageOptionState.value.temporaryChat = false
    messageOptionState.value.useOCR = false
    messageOptionState.value.fileRetrievalEnabled = false
    messageOptionState.value.ragMediaIds = null
    artifactsState.value.isOpen = false
    artifactsState.value.active = null
    artifactsState.value.history = []
    artifactsState.value.unreadCount = 0
    storeOptionState.value.compareParentByHistory = {}
    storeOptionState.value.uploadedFiles = []
    storeOptionState.value.contextFiles = []
    storeOptionState.value.setUploadedFiles.mockImplementation((files) => {
      storeOptionState.value.uploadedFiles = files
    })
    storeOptionState.value.setContextFiles.mockImplementation((files) => {
      storeOptionState.value.contextFiles = files
    })
    tldwClientState.getDocumentUploadDraft.mockResolvedValue({ payload: {} })
    window.history.replaceState(null, "", "/chat")
  })

  it("does not import ambient attachment settings while a copied child is pending or verified", async () => {
    forkPresentation.mode = "pending"
    messageOptionState.value.serverChatId = "child"
    const mounted = render(<Playground />)
    await act(async () => {})
    expect(chatSettingsState.syncChatSettingsForServerChat).not.toHaveBeenCalled()
    expect(chatSettingsState.applyChatSettingsPatch).not.toHaveBeenCalled()
    forkPresentation.mode = "fork"
    forkPresentation.settings = {authorNote: "server"}
    mounted.rerender(<Playground />)
    await act(async () => {})
    expect(chatSettingsState.syncChatSettingsForServerChat).not.toHaveBeenCalled()
    expect(chatSettingsState.applyChatSettingsPatch).not.toHaveBeenCalled()
    forkPresentation.mode = "ordinary"
    forkPresentation.settings = null
    mounted.rerender(<Playground />)
    await waitFor(() => expect(chatSettingsState.syncChatSettingsForServerChat).toHaveBeenCalled())
  })

  it("keeps ordinary attachment lookup owner-fenced through a legacy review controller without accepting ancestry", async () => {
    h1.legacyNative = true
    const actual = renderHook(() => useHistorySelection())
    await act(async () => {await actual.result.current.open({kind: "native", owner_key: "local-owner", conversation_id: "chat-1", request_scope: verifiedOwner, scope: {type: "global"}, validate_lease: () => true} as any)})
    forkPresentation.controller = actual.result.current
    render(<Playground />)
    await waitFor(() => expect(chatSettingsState.syncChatSettingsForServerChat).toHaveBeenCalledWith(expect.objectContaining({historyId: "history-1", serverChatId: "chat-1", requestScope: verifiedOwner, validate_lease: expect.any(Function)})))
    expect(actual.result.current.status).toBe("legacy_review_required")
    expect(h1.confirm).not.toHaveBeenCalled()
  })

  it("does not scroll or edit after a held reveal outlives the initiating owner", async () => {
    const actual = renderHook(() => useHistorySelection())
    forkPresentation.controller = actual.result.current
    let release!: (value: boolean) => void
    const reveal = vi.fn(() => new Promise<boolean>(resolve => { release = resolve }))
    timelineReveal.current = reveal
    const edit = vi.fn()
    window.addEventListener("tldw:edit-message", edit)
    const target = document.createElement("div")
    target.dataset.messageId = "m-2"
    const scroll = vi.fn()
    target.scrollIntoView = scroll
    const page = render(<Playground />)
    smartScrollState.value.containerRef.current!.append(target)
    act(() => window.dispatchEvent(new CustomEvent("tldw:timeline-action", { detail: {
      action: "edit", historyId: "history-1", messageId: "m-2"
    } })))
    await waitFor(() => expect(reveal).toHaveBeenCalledWith(1))
    expect(smartScrollState.value.pauseAutoScroll).toHaveBeenCalled()
    expect(smartScrollState.value.pauseAutoScroll.mock.invocationCallOrder.at(-1)).toBeLessThan(reveal.mock.invocationCallOrder[0])
    act(() => actual.result.current.reset())
    await act(async () => { release(true); await Promise.resolve() })
    expect(scroll).not.toHaveBeenCalled()
    expect(edit).not.toHaveBeenCalled()
    window.removeEventListener("tldw:edit-message", edit)
    page.unmount()
  })

  it("does not scroll an indexed search result after a held reveal outlives its owner", async () => {
    const actual = renderHook(() => useHistorySelection())
    forkPresentation.controller = actual.result.current
    let release!: (value: boolean) => void
    const reveal = vi.fn(() => new Promise<boolean>(resolve => { release = resolve }))
    timelineReveal.current = reveal
    const target = document.createElement("div")
    target.dataset.testid = "chat-message"
    target.dataset.index = "1"
    const scroll = vi.fn()
    target.scrollIntoView = scroll
    const page = render(<Playground />)
    smartScrollState.value.containerRef.current!.append(target)
    fireEvent.keyDown(window, { key: "f", ctrlKey: true })
    fireEvent.change(screen.getByPlaceholderText("Search messages in this conversation"), { target: { value: "beta" } })
    await waitFor(() => expect(reveal).toHaveBeenCalledWith(1))
    expect(smartScrollState.value.pauseAutoScroll).toHaveBeenCalled()
    expect(smartScrollState.value.pauseAutoScroll.mock.invocationCallOrder.at(-1)).toBeLessThan(reveal.mock.invocationCallOrder[0])
    act(() => actual.result.current.reset())
    await act(async () => { release(true); await Promise.resolve() })
    expect(scroll).not.toHaveBeenCalled()
    page.unmount()
  })

  it.each([2, 101])("uses settled indexed navigation for %i messages", async (count) => {
    const previous = messageOptionState.value.messages
    messageOptionState.value.messages = [...previous, ...Array.from({length: count - previous.length}, (_, index) => ({id: `extra-${index}`, message: "Other row", isBot: false, role: "user"}))]
    timelineReveal.current = vi.fn(async () => true)
    const target = document.createElement("div")
    target.dataset.testid = "chat-message"
    target.dataset.index = "1"
    const scroll = vi.fn()
    target.scrollIntoView = scroll
    const restoreMatchMedia = mockReducedMotion(false)
    const page = render(<Playground />)
    try {
      smartScrollState.value.containerRef.current!.append(target)
      fireEvent.keyDown(window, { key: "f", ctrlKey: true })
      fireEvent.change(screen.getByPlaceholderText("Search messages in this conversation"), {target: {value: "beta"}})
      await waitFor(() => expect(scroll).toHaveBeenCalledWith({block: "center", behavior: count > 100 ? "auto" : "smooth"}))
    } finally {
      page.unmount()
      restoreMatchMedia()
      messageOptionState.value.messages = previous
    }
  })

  // AX-19 (#3124): search jumps smooth-scrolled even when the OS asks for
  // reduced motion.
  it("jumps to a search result without smooth scrolling under reduced motion", async () => {
    const restoreMatchMedia = mockReducedMotion(true)
    timelineReveal.current = vi.fn(async () => true)
    const target = document.createElement("div")
    target.dataset.testid = "chat-message"
    target.dataset.index = "1"
    const scroll = vi.fn()
    target.scrollIntoView = scroll
    const page = render(<Playground />)
    try {
      smartScrollState.value.containerRef.current!.append(target)
      fireEvent.keyDown(window, { key: "f", ctrlKey: true })
      fireEvent.change(screen.getByPlaceholderText("Search messages in this conversation"), {target: {value: "beta"}})
      await waitFor(() => expect(scroll).toHaveBeenCalledWith({block: "center", behavior: "auto"}))
    } finally {
      page.unmount()
      restoreMatchMedia()
    }
  })

  it("forks timeline messages using stable IDs rather than rendered positions", async () => {
    render(<Playground />)
    act(() => window.dispatchEvent(new CustomEvent("tldw:timeline-action", {detail: {
      action: "branch", historyId: "history-1", messageId: "m-2"
    }})))
    await waitFor(() => expect(messageOptionState.value.createChatBranch).toHaveBeenCalledWith("m-2"))
  })

  it("opens in-thread search on Cmd/Ctrl+F and forwards query to PlaygroundChat", async () => {
    render(<Playground />)

    fireEvent.keyDown(window, { key: "f", ctrlKey: true })

    const input = screen.getByPlaceholderText(
      "Search messages in this conversation"
    ) as HTMLInputElement
    expect(input).toBeInTheDocument()

    fireEvent.change(input, { target: { value: "beta" } })

    expect(screen.getByTestId("playground-chat")).toHaveAttribute(
      "data-search-query",
      "beta"
    )
    await waitFor(() => {
      expect(screen.getByTestId("playground-chat")).toHaveAttribute(
        "data-search-count",
        "1"
      )
    })
  })

  // AX-09 (#3122): closing in-thread search must hand focus back to where the
  // user was, instead of dropping it on <body>.
  it.each(["Escape", "Close button"])(
    "returns focus to the previously focused control after closing thread search with %s",
    async (closeWith) => {
      render(<Playground />)
      const trigger = screen.getByTestId("playground-shortcuts-help-trigger")
      trigger.focus()

      fireEvent.keyDown(trigger, { key: "f", ctrlKey: true })
      const input = screen.getByPlaceholderText(
        "Search messages in this conversation"
      )
      await waitFor(() => expect(input).toHaveFocus())

      if (closeWith === "Escape") {
        fireEvent.keyDown(input, { key: "Escape" })
      } else {
        fireEvent.click(screen.getByRole("button", { name: "Close" }))
      }

      expect(
        screen.queryByPlaceholderText("Search messages in this conversation")
      ).not.toBeInTheDocument()
      await waitFor(() => expect(trigger).toHaveFocus())
    }
  )

  it("opens the single shortcut reference from the chat header and closes back to its trigger", async () => {
    useTutorialStore.getState().closeHelpModal()
    render(
      <>
        <Playground />
        <PageHelpModalHost />
      </>
    )
    const trigger = screen.getByTestId("playground-shortcuts-help-trigger")
    trigger.focus()
    fireEvent.click(trigger)
    const dialog = await screen.findByRole("dialog", { name: "Page Help" })
    expect(screen.getAllByRole("dialog")).toHaveLength(1)
    expect(
      screen.queryByTestId("playground-shortcuts-help-panel")
    ).not.toBeInTheDocument()
    expect(within(dialog).getByText("Search this conversation")).toBeVisible()
    fireEvent.keyDown(dialog, { key: "Escape" })
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
    expect(trigger).toHaveFocus()
  })

  it("preserves the legacy chat help command through the one shared host", async () => {
    useTutorialStore.getState().closeHelpModal()
    render(
      <>
        <Playground />
        <PageHelpModalHost />
      </>
    )
    act(() =>
      window.dispatchEvent(new CustomEvent("tldw:open-playground-shortcuts"))
    )
    const dialog = await screen.findByRole("dialog", { name: "Page Help" })
    expect(screen.getAllByRole("dialog")).toHaveLength(1)
    expect(
      within(dialog).getByText("Fork from the focused message")
    ).toBeVisible()
    fireEvent.keyDown(dialog, { key: "Escape" })
  })

  it("does not render the deprecated chat workflows header action", () => {
    render(<Playground />)

    expect(
      screen.queryByTestId("playground-chat-workflows-trigger")
    ).not.toBeInTheDocument()
    expect(routerState.navigate).not.toHaveBeenCalled()
  })

  it("shows a desktop right-edge artifacts expand button only when an artifact is active and the rail is closed", () => {
    artifactsState.value.active = artifactFixture
    artifactsState.value.isOpen = false

    render(<Playground />)

    expect(
      screen.getByRole("button", { name: "Expand artifacts rail" })
    ).toBeInTheDocument()
    expect(screen.queryByTestId("artifacts-panel")).not.toBeInTheDocument()
  })

  it("does not show the right-edge artifacts expand button without an active artifact", () => {
    artifactsState.value.active = null
    artifactsState.value.isOpen = false

    render(<Playground />)

    expect(
      screen.queryByRole("button", { name: "Expand artifacts rail" })
    ).not.toBeInTheDocument()
  })

  it("opens artifacts from the right edge and marks them read", () => {
    artifactsState.value.active = artifactFixture
    artifactsState.value.isOpen = false

    render(<Playground />)
    fireEvent.click(screen.getByRole("button", { name: "Expand artifacts rail" }))

    expect(artifactsState.value.setOpen).toHaveBeenCalledWith(true)
    expect(artifactsState.value.markRead).toHaveBeenCalledTimes(1)
  })

  it("routes artifact focus events to the edge button when the rail is closed", async () => {
    artifactsState.value.active = artifactFixture
    artifactsState.value.isOpen = false

    render(<Playground />)
    const edgeButton = screen.getByRole("button", {
      name: "Expand artifacts rail"
    })

    window.dispatchEvent(new CustomEvent("tldw:focus-artifacts-trigger"))

    await waitFor(() => {
      expect(document.activeElement).toBe(edgeButton)
    })
  })

  it("shows mobile artifacts sheet context and returns focus to trigger when closing", async () => {
    mobileViewportState.value = true
    desktopViewportState.value = false
    storageState.value.set("playgroundChatLayoutMode", "cockpit")
    artifactsState.value.isOpen = true

    render(<Playground />)

    expect(
      screen.getByTestId("playground-mobile-artifacts-sheet")
    ).toBeInTheDocument()
    expect(
      screen.getByTestId("playground-mobile-artifacts-title")
    ).toHaveTextContent("Artifacts panel")

    fireEvent.click(screen.getByTestId("playground-mobile-artifacts-return"))
    expect(artifactsState.value.closeArtifact).toHaveBeenCalledTimes(1)

    await waitFor(() => {
      expect(document.activeElement).toBe(
        screen.getByTestId("playground-artifacts-trigger")
      )
    })
  })

  it("shows branch fork context and returns to parent history in one action", () => {
    storeOptionState.value.compareParentByHistory = {
      "history-1": {
        parentHistoryId: "history-parent",
        clusterId: "cluster-a"
      },
      "history-parent": {
        parentHistoryId: "history-root"
      }
    }

    const openHistorySpy = vi.fn()
    const onOpenHistory = ((event: Event) => {
      openHistorySpy((event as CustomEvent).detail)
    }) as EventListener
    window.addEventListener("tldw:open-history", onOpenHistory)

    render(<Playground />)

    expect(screen.getByTestId("playground-branch-fork-point")).toHaveTextContent(
      "Fork point: cluster-a"
    )
    expect(screen.getByTestId("playground-branch-depth")).toHaveTextContent(
      "Depth 2"
    )

    fireEvent.click(
      screen.getByRole("button", { name: "Back to comparison chat" })
    )

    expect(openHistorySpy).toHaveBeenCalledWith({
      historyId: "history-parent"
    })

    window.removeEventListener("tldw:open-history", onOpenHistory)
  })

  it("keeps mixed sidepanel options and references untouched before review and after Cancel", async () => {
    fragmentComposer.enabled = true
    fragmentComposer.ready = true
    const encoded = encodeSidepanelChatWebUiHandoff({ source: SIDEPANEL_CHAT_WEBUI_HANDOFF_SOURCE, createdAt: Date.now(),
      draft: "COPIED DRAFT  ", historyId: "foreign-history", serverChatId: "foreign-server", selectedModel: "foreign-model",
      selectedSystemPrompt: "foreign-prompt", selectedQuickPrompt: "literal prompt", chatMode: "rag", webSearch: true,
      toolChoice: "required", temporaryChat: true, useOCR: true, chatDocumentDraftId: "foreign-document", ragMediaIds: [101], fileRetrievalEnabled: true })
    window.history.replaceState(null, "", `/chat#handoff=${encoded}`)
    render(<Playground />)
    await screen.findByRole("status", { name: "Fragment review" })
    const writes = [messageOptionState.value.setSelectedModel, messageOptionState.value.setSelectedSystemPrompt,
      messageOptionState.value.setSelectedQuickPrompt, messageOptionState.value.setChatMode, messageOptionState.value.setWebSearch,
      messageOptionState.value.setToolChoice, messageOptionState.value.setTemporaryChat, messageOptionState.value.setUseOCR,
      messageOptionState.value.setRagMediaIds, messageOptionState.value.setFileRetrievalEnabled, messageOptionState.value.setContextFiles,
      storeOptionState.value.setUploadedFiles, tldwClientState.getDocumentUploadDraft, tldwClientState.deleteDocumentUploadDraft]
    for (const write of writes) expect(write).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole("button", { name: "Cancel fragment import" }))
    expect(screen.getByLabelText("Fragment composer")).toHaveValue("RESTORED DESTINATION  ")
    for (const write of writes) expect(write).not.toHaveBeenCalled()
    expect(routerState.navigate).toHaveBeenCalledWith(expect.objectContaining({ hash: "" }), { replace: true })
  })

  it("does not let an unavailable foreign local reference block its transferable draft", async () => {
    loadLocalConversationMock.mockResolvedValueOnce(false)
    fragmentComposer.enabled = true
    fragmentComposer.ready = true
    const draftReceived = vi.fn()
    window.addEventListener("tldw:set-composer-message", draftReceived)
    const encoded = encodeSidepanelChatWebUiHandoff({ source: SIDEPANEL_CHAT_WEBUI_HANDOFF_SOURCE, createdAt: Date.now(), historyId: "foreign-local-id", serverChatId: "unqualified-server-id", draft: "TRANSFERABLE DRAFT" })
    window.history.replaceState(null, "", `/chat#handoff=${encoded}`)
    try {
      render(<Playground />)
      await screen.findByRole("status", { name: "Fragment review" })
      expect(screen.getByRole("status", { name: "Fragment review" }).querySelector("pre")?.textContent).toBe("TRANSFERABLE DRAFT")
      expect(draftReceived.mock.calls.some(([event]) => (event as CustomEvent).detail?.message === "TRANSFERABLE DRAFT")).toBe(false)
      expect(loadLocalConversationMock).not.toHaveBeenCalled()
      expect(messageOptionState.value.setServerChatId).not.toHaveBeenCalledWith("unqualified-server-id")
      expect(messageOptionState.value.historyId).toBe("history-1")
      expect(messageOptionState.value.serverChatId).toBe("chat-1")
      expect(screen.getByRole("status", { name: "Fragment review" })).toHaveTextContent("Conversation history")
    } finally {
      window.removeEventListener("tldw:set-composer-message", draftReceived)
      loadLocalConversationMock.mockReset().mockResolvedValue(undefined)
    }
  })

  it("waits for hydrated draft review without starting an unqualified document read", async () => {
    fragmentComposer.enabled = true
    const copied = "  {{user}}\n\tREVIEW NOW  "
    const encoded = encodeSidepanelChatWebUiHandoff({ source: SIDEPANEL_CHAT_WEBUI_HANDOFF_SOURCE, createdAt: Date.now(), draft: copied, selectedModel: "different-model", chatDocumentDraftId: "unqualified-document" })
    window.history.replaceState(null, "", `/chat#handoff=${encoded}`)
    const view = render(<Playground />)
    await act(async () => {})
    expect(screen.queryByRole("status", { name: "Fragment review" })).not.toBeInTheDocument()
    expect(tldwClientState.getDocumentUploadDraft).not.toHaveBeenCalled()
    expect(messageOptionState.value.setSelectedModel).not.toHaveBeenCalled()
    fragmentComposer.ready = true
    view.rerender(<Playground />)
    await screen.findByRole("status", { name: "Fragment review" })
    expect(screen.getByRole("status", { name: "Fragment review" }).querySelector("pre")?.textContent).toBe(copied)
    expect(screen.getByLabelText("Fragment composer")).toHaveValue("RESTORED DESTINATION  ")
    expect(tldwClientState.deleteDocumentUploadDraft).not.toHaveBeenCalled()
  })

  it("keeps a document-only copy reviewable without consuming or merging unqualified files", async () => {
    fragmentComposer.enabled = true
    fragmentComposer.ready = true
    const existingFile = { id: "existing", filename: "keep.md", content: "KEEP" }
    storeOptionState.value.uploadedFiles = [existingFile]
    storeOptionState.value.contextFiles = [existingFile]
    const encoded = encodeSidepanelChatWebUiHandoff({ source: SIDEPANEL_CHAT_WEBUI_HANDOFF_SOURCE, createdAt: Date.now(), chatDocumentDraftId: "unqualified-document" })
    window.history.replaceState(null, "", `/chat#handoff=${encoded}`)
    render(<Playground />)
    await screen.findByRole("status", { name: "Fragment review" })
    expect(tldwClientState.getDocumentUploadDraft).not.toHaveBeenCalled()
    expect(tldwClientState.deleteDocumentUploadDraft).not.toHaveBeenCalled()
    expect(storeOptionState.value.setUploadedFiles).not.toHaveBeenCalled()
    expect(messageOptionState.value.setContextFiles).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole("button", { name: "Cancel fragment import" }))
    expect(screen.getByLabelText("Fragment composer")).toHaveValue("RESTORED DESTINATION  ")
    expect(storeOptionState.value.uploadedFiles).toEqual([existingFile])
    expect(routerState.navigate).toHaveBeenCalledWith(expect.objectContaining({ hash: "" }), { replace: true })
  })

})

vi.mock("@/db/dexie/history-selection", () => ({
  ensureLocalProfileId: async () => "profile",
  getLocalHistoryOwner: async (id: string) => ({
    kind: "local",
    profile_id: "profile",
    owner_key: "local-owner",
    conversation_id: id
  }),
  loadHistoryBookmark: async (scope: any, owner: any) =>
    h1.bookmarks.get(h1Key(scope, owner)) || null,
  saveHistoryBookmark: async (scope: any, view: any) =>
    h1.bookmarks.set(h1Key(scope, view), { ...scope, view })
}))
vi.mock("@/db/dexie/fork-operations", () => ({findForkCandidate: async () => null, loadForkOperations: async () => []}))
vi.mock("@/db/dexie/chat", () => ({
  PageAssistDatabase: class {
    getHistoryInfo = h1.historyInfo
  }
}))
vi.mock("@/services/chat-history-selection", () => ({
  confirmLegacyHistoryProjection: (...args: any[]) => h1.confirm(...args),
  captureHistorySnapshot: async (owner: any, view: any) => {
    const bound = { ...view, owner_key: "local-owner" }
    const snapshot: any = {
      version: 1,
      owner_key: "local-owner",
      conversation_id: owner.conversation_id,
      source_digest: "source",
      storage_context_digest: "storage",
      fences: {},
      interpretation_status: { kind: "parent_graph_v1" },
      nodes: ["A", "B"].map((id) => ({
        id,
        revision: "1",
        role: "assistant",
        parent_id: null,
        settled: true,
        preview: owner.conversation_id + " answer " + id
      }))
    }
    if (h1.legacyNative) return {status: "legacy_review_required", code: "legacy_review_required", snapshot, view: bound}
    const result = resolveHistorySelection(snapshot, bound, "send", "")
    if (result.status !== "ready") return { ...result, snapshot, view: bound }
    return {
      status: "captured",
      snapshot,
      view: bound,
      rows: result.rows,
      selected_content: result.rows.map((row) => ({
        id: row.id,
        revision: row.revision,
        message: row.preview,
        images: []
      })),
      storage_context_digest: "storage",
      purpose: "send"
    }
  }
}))

describe("Playground H1 URL initialization with the mounted session/controller", () => {
  beforeEach(() => {
    routerState.hashRouter = false
    h1.enabled = true
    h1.sawFreshHandle = false
    forkPresentation.controller = null
    h1.legacyNative = false
    forkPresentation.mode = null
    forkPresentation.settings = null
    h1.controller = null
    h1.bookmarks.clear()
    h1.confirm.mockClear()
    h1.historyInfo.mockReset().mockImplementation(async (id: string) =>
      ["chat-one", "chat-two"].includes(id)
        ? { id, title: id, createdAt: 1, is_rag: false, message_source: "branch", local_owner_key: "local-owner" }
        : null
    )
    localStorage.clear()
    sessionStorage.clear()
    storageState.value.clear()
    window.history.replaceState(null, "", "/chat")
    usePlaygroundSessionStore.getState().clearSession()
    useStoreMessageOption.setState({
      chatDraftAddress: null,
      chatDraftTransition: null,
      chatDraftRevision: 0,
      historyId: null,
      serverChatId: null,
      history: [],
      messages: [],
      temporaryChat: false,
      queuedMessages: [],
      compareMode: false,
      compareSelectedModels: [],
      contextFiles: [],
      serverChatAssistantId: null,
      serverChatAssistantKind: null,
      serverChatCharacterId: null,
      serverChatMetaLoaded: false
    })
  })
  function seed(pending = false) {
    const reference = {
      profile_id: "profile",
      client_session_id: "frozen-source",
      owner_key: "local-owner",
      conversation_id: "chat-one",
      owner_kind: "local"
    }
    const view = {
      owner_key: reference.owner_key,
      conversation_id: reference.conversation_id,
      view_session_id: "source-view",
      selection_revision: 3,
      interpretation: { kind: "parent_graph_v1" },
      cursor: { kind: "after_message", message_id: "A" }
    }
    const record: any = { ...reference, view }
    if (pending)
      Object.assign(record, {
        pending_confirmation: {
          version: 1,
          projection_id: "pending-original",
          selection_revision: 3,
          ordered_path_ids: ["A"],
          cursor: view.cursor
        },
        pending_view_session_id: "source-view",
        pending_dispatch_started: true
      })
    h1.bookmarks.set(h1Key(reference, view), record)
    return reference
  }
  it("TASK-13390: a blocked automatic restore stays silent even when the stored reference is malformed", async () => {
    // Without a link, a blocked restore must return quietly. Reading the stored
    // reference before the gate turned a malformed stored value into a visible
    // "unavailable" state after every account change.
    forkPresentation.blockAutomaticLoad = true
    forkPresentation.storedReferenceInvalid = true
    forkPresentation.opened = []
    try {
      window.history.replaceState({}, "", "/chat")
      const page = render(<Playground />)
      await waitFor(() => expect(screen.getByTestId("playground-chat")).toBeInTheDocument())
      await act(async () => { await new Promise(resolve => setTimeout(resolve, 50)) })
      expect(forkPresentation.opened).not.toContainEqual(expect.objectContaining({ code: "invalid_history_reference" }))
      page.unmount()
    } finally {
      forkPresentation.blockAutomaticLoad = false
      forkPresentation.storedReferenceInvalid = false
      forkPresentation.opened = []
    }
  })

  it("TASK-13390: an explicit history link still opens after an account change invalidates automatic loading", async () => {
    // After an account/config change the native-history lease is invalid, so
    // canAutomaticallyLoad() is false. That must stop AUTOMATIC restores only; a
    // historySelection link the user followed is deliberate, as the loader already
    // treats deliberate selections. initializePlayground runs once, so an early
    // return here drops the link for as long as the page stays mounted.
    forkPresentation.blockAutomaticLoad = true
    try {
      const reference = seed()
      window.history.replaceState({}, "", "/chat?historySelection=" + encodeURIComponent(JSON.stringify(reference)))
      const page = render(<Playground />)
      await waitFor(() =>
        expect(screen.getByTestId("playground-chat")).toHaveAttribute("data-selected-history", "chat-one answer A")
      )
      page.unmount()
    } finally {
      // This describe's beforeEach does not reset the flag, so reset it here.
      forkPresentation.blockAutomaticLoad = false
    }
  })

  it.each([
    { name: "missing history", details: null, code: "owner_conversation_mismatch" },
    { name: "unowned history", details: { id: "chat-one", message_source: "branch" }, code: "owner_conversation_mismatch" },
    { name: "unbound server mirror", details: { id: "chat-one", message_source: "server", server_chat_id: "foreign-chat" }, code: "unbound_server_mirror" }
  ])("rejects a bookmarked $name without publishing selected rows", async ({ details, code }) => {
    const reference = seed()
    h1.historyInfo.mockResolvedValue(details)
    window.history.replaceState(null, "", "/chat?historySelection=" + encodeURIComponent(JSON.stringify(reference)))
    const page = render(<Playground />)
    await waitFor(() => expect(h1.controller?.error).toBe(code))
    expect(h1.controller?.capture).toBeNull()
    expect(h1.controller?.view).toBeNull()
    expect(useStoreMessageOption.getState().messages).toEqual([])
    expect(h1.bookmarks.size).toBe(1)
    expect(window.location.search).toContain("historySelection")
    page.unmount()
  })
  it.each(["query", "hash", "hash-router"])(
    "consumes a %s handoff once and restores destination choice, conversation switch and empty reset",
    async (route) => {
      routerState.hashRouter = route === "hash-router"
      const reference = seed()
      const query =
        "keep=value&historySelection=" +
        encodeURIComponent(JSON.stringify(reference))
      window.history.replaceState(
        { preserved: true },
        "",
        route === "query"
          ? "/chat?" + query + "#anchor"
          : "/options.html?outer=value#/chat?" + query + "#anchor"
      )
      let page = render(<Playground />)
      await waitFor(() =>
        expect(screen.getByTestId("playground-chat")).toHaveAttribute(
          "data-selected-history",
          "chat-one answer A"
        )
      )
      await act(async () => {
        await h1.controller!.choose({ kind: "after_message", message_id: "B" })
      })
      page.unmount()
      page = render(<Playground />)
      await waitFor(() =>
        expect(screen.getByTestId("playground-chat")).toHaveAttribute(
          "data-selected-history",
          "chat-one answer B"
        )
      )
      expect(window.location.href).not.toContain("historySelection")
      expect(window.location.href).toContain("keep=value")
      if (route === "query") expect(window.location.hash).toBe("#anchor")
      else {
        expect(window.location.search).toBe("?outer=value")
        expect(window.location.hash).toContain("#anchor")
      }
      expect(window.history.state).toEqual({ preserved: true })
      await act(async () => {
        await h1.controller!.loadConversation({ historyId: "chat-two" }, null)
      })
      page.unmount()
      page = render(<Playground />)
      await waitFor(() =>
        expect(h1.controller?.view?.conversation_id).toBe("chat-two")
      )
      await act(async () => {
        h1.controller!.reset()
        usePlaygroundSessionStore.getState().clearSession()
        useStoreMessageOption.setState({
          historyId: null,
          serverChatId: null,
          messages: [],
          history: []
        })
      })
      page.unmount()
      page = render(<Playground />)
      await waitFor(() => expect(h1.controller?.status).toBe("idle"))
      expect(useStoreMessageOption.getState().messages).toEqual([])
      page.unmount()
      // A different destination URL still represents a new initialization.
      window.history.replaceState(null, "", "/chat?" + query)
      page = render(<Playground />)
      await waitFor(() =>
        expect(h1.controller?.view?.conversation_id).toBe("chat-one")
      )
      page.unmount()
    }
  )
  it("keeps an explicit owned handoff ahead of a valid old fresh handle after provider remount", async () => {
    const reference = seed()
    const fresh = { kind: "fresh" as const, id: "old-unsent", scope: { type: "global" as const }, temporary: false }
    usePlaygroundSessionStore.getState().saveSession({
      chatDraftHandle: { address: fresh, ownerHash: chatDraftOwnerKey(verifiedOwner as never) },
      scopeKey: buildChatSurfaceScopeKeyFromConfig(verifiedOwner.config), historyId: null, serverChatId: null
    })
    expect(usePlaygroundSessionStore.getState().isSessionValid(buildChatSurfaceScopeKeyFromConfig(verifiedOwner.config))).toBe(true)
    // Layout's settings controller has accepted B; its chat controller now remounts.
    useStoreMessageOption.getState().startNewChatDraft(false)
    const source = useStoreMessageOption.getState()
    useStoreMessageOption.getState().setChatDraftAddress({ kind: "local", id: reference.conversation_id, profileId: reference.profile_id }, { kind: "switch", expectedRevision: source.chatDraftRevision, sourceAddress: source.chatDraftAddress })
    window.history.replaceState(null, "", "/chat?historySelection=" + encodeURIComponent(JSON.stringify(reference)))
    let page = render(<Playground />)
    await waitFor(() => expect(screen.getByTestId("playground-chat")).toHaveAttribute("data-selected-history", "chat-one answer A"))
    expect(h1.sawFreshHandle).toBe(true)
    expect(useStoreMessageOption.getState().chatDraftAddress).toMatchObject({ kind: "local", id: "chat-one", profileId: "profile" })
    expect(window.location.search).toBe("")
    page.unmount()
    page = render(<Playground />)
    await waitFor(() => expect(screen.getByTestId("playground-chat")).toHaveAttribute("data-selected-history", "chat-one answer A"))
    expect(h1.controller?.getCurrent().owner?.conversation_id).toBe("chat-one")
    expect(useStoreMessageOption.getState().chatDraftAddress?.kind).toBe("local")
    page.unmount()
  })
  it("retains the exact pending origin after consuming its URL and reloading", async () => {
    const reference = seed(true)
    window.history.replaceState(
      null,
      "",
      "/chat?historySelection=" + encodeURIComponent(JSON.stringify(reference))
    )
    let page = render(<Playground />)
    await waitFor(() => expect(h1.controller?.status).toBe("pending_unknown"))
    page.unmount()
    page = render(<Playground />)
    await waitFor(() =>
      expect(h1.controller?.pending?.intent.projection_id).toBe(
        "pending-original"
      )
    )
    expect(window.location.search).toBe("")
    expect(h1.controller?.pending?.scope.client_session_id).toBe(
      "frozen-source"
    )
    expect(h1.controller?.pending?.view.view_session_id).toBe("source-view")
    expect(
      h1.bookmarks.get(h1Key(reference, reference)).pending_dispatch_started
    ).toBe(true)
    expect(h1.confirm).not.toHaveBeenCalled()
    page.unmount()
  })
})

vi.mock("@/services/tldw-server", async original => ({
  ...await original<typeof import("@/services/tldw-server")>(),
  fetchChatModels: async () => []
}))
