// @vitest-environment jsdom
import React from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { App } from "antd"
import { CharacterSelect } from "@/components/Common/CharacterSelect"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"

import { Playground } from "../Playground"
import { useStoreMessageOption } from "@/store/option"
import { useServerChatLoader } from "@/hooks/chat/useServerChatLoader"
import { resolveEffectiveAssistantState, effectiveAssistantStateToSelection } from "@/hooks/chat/effective-assistant-state"
import type { AssistantSelection } from "@/types/assistant-selection"
import { useSelectedAssistant } from "@/hooks/useSelectedAssistant"
import { Header } from "@/components/Layouts/Header"
import { CHAT_ROUTE_REPLACEMENT_EVENT } from "@/utils/character-chat-mode-intent"
import { SETTINGS_NAVIGATION_REQUEST_EVENT } from "@/utils/settings-return"
import { useClearChat } from "@/hooks/chat/useClearChat"
import { useSelectServerChat } from "@/hooks/chat/useSelectServerChat"
import { selectedAssistantStorage } from "@/utils/selected-assistant-storage"
import { AssistantSelect } from "@/components/Common/AssistantSelect"
import { useChatSurfaceCoordinatorStore } from "@/store/chat-surface-coordinator"
import { usePlaygroundSessionStore } from "@/store/playground-session"
import { webUIResumeLastChat } from "@/services/app"
import { getRecentChatFromWebUI } from "@/db/dexie/helpers"
import {
  encodeSidepanelChatWebUiHandoff,
  SIDEPANEL_CHAT_WEBUI_HANDOFF_PARAM
} from "@/services/tldw/sidepanel-chat-webui-handoff"
import { SETTINGS_SERVER_CHAT_ID_PARAM } from "@/utils/settings-return"
import type { PlaygroundSessionRestoreOutcome } from "@/hooks/usePlaygroundSessionPersistence"
import { getFlashcardSourceMeta } from "@/components/Flashcards/utils/source-reference"
import { useChatActions } from "@/hooks/chat/useChatActions"

const ordinaryCompletion = vi.hoisted(() => ({ owner: "A", complete: vi.fn(), save: vi.fn(async () => "bob-history") }))
vi.mock("@/hooks/chat-modes/normalChatMode", () => ({ normalChatMode: (...args: unknown[]) => ordinaryCompletion.complete(...args) }))
vi.mock("@/hooks/utils/messageHelpers", async original => ({
  ...await original<typeof import("@/hooks/utils/messageHelpers")>(),
  createSaveMessageOnSuccess: () => ordinaryCompletion.save,
  validateBeforeSubmit: () => true
}))
vi.mock("@/services/actor-settings", () => ({ getActorSettingsForChat: async () => null }))
vi.mock("@/services/tldw/server-capabilities", () => ({ getServerCapabilities: async () => ({ hasChatSaveToDb: true }) }))

const OrdinarySend = () => {
  const state = useStoreMessageOption()
  const actions = useChatActions({
    ...state, t: stableTranslation, notification: { error: vi.fn(), warning: vi.fn(), info: vi.fn(), success: vi.fn() },
    abortController: null, setAbortController: vi.fn(), setIsSearchingInternet: vi.fn(),
    contextFiles: [], setContextFiles: vi.fn(), documentContext: null, setDocumentContext: vi.fn(),
    currentChatModelSettings: {}, ensureServerChatHistoryId: async () => "bob-history",
    selectedCharacter: null, selectedAssistant: null, compareModeActive: false, compareSelectedModels: [],
    compareMaxModels: 3, markCompareHistoryCreated: vi.fn(), invalidateServerChatHistory: vi.fn(),
    clearReplyTarget: vi.fn(), clearMessageSteering: vi.fn(), replyTarget: null,
    messageSteeringMode: "none", messageSteeringForceNarrate: false, messageSteeringPrompts: null
  } as unknown as Parameters<typeof useChatActions>[0])
  return <button onClick={() => void actions.onSubmit({ message: "Bob ordinary question", image: "" })}>Send ordinary turn</button>
}

const useMessageOptionMock = vi.hoisted(() => vi.fn())
const realLoader = vi.hoisted(() => ({ enabled: false, webStorage: false, session: false, route: false, additionalLoader: false, storageBarrier: null as Promise<void> | null, storageWrites: 0, invalidated: new AbortController(), setSelection: null as null | ReturnType<typeof useSelectedAssistant>[1] }))
const ensureTestHistory = async () => null
const loaderNotification = { error: vi.fn() }
const loaderTranslation = ((key: string) => key) as never
type SelectionTestState = ReturnType<typeof useStoreMessageOption.getState> & { testAssistant: AssistantSelection | null }
const setStorageAssistant = async (selection: AssistantSelection | null) => {
  realLoader.storageWrites++
  if (realLoader.storageBarrier) await realLoader.storageBarrier
  useStoreMessageOption.setState({ testAssistant: selection } as Partial<SelectionTestState>)
}
const setTestAssistant = async (selection: AssistantSelection | null) => { await realLoader.setSelection?.(selection) }
const useRealServerConversation = () => {
  const clearChat = useClearChat()
  const store = useStoreMessageOption() as SelectionTestState
  const [assistant, setAssistant] = useSelectedAssistant()
  realLoader.setSelection = setAssistant
  useServerChatLoader({ ensureServerChatHistoryId: ensureTestHistory, notification: loaderNotification, t: loaderTranslation })
  const resolved = resolveEffectiveAssistantState({
    tracked: { assistantKind: store.serverChatAssistantKind, assistantId: store.serverChatAssistantId, characterId: store.serverChatCharacterId },
    draftSelection: assistant
  })
  return { ...messageOptionState.value, ...store, clearChat, selectedAssistant: effectiveAssistantStateToSelection(resolved) ?? assistant, setSelectedAssistant: setAssistant, ...(realLoader.session || realLoader.route ? { setSelectedCharacter: async (character: { id: string | number; name: string }) => setAssistant({ kind: "character", id: String(character.id), name: character.name, metadata: { selectionMode: "tracked" } }) } : {}) }
}

const AdditionalServerLoader = () => { useRealServerConversation(); return null }
const SavedChatSelection = ({ ordinary = false }: { ordinary?: boolean }) => {
  const select = useSelectServerChat()
  return <button onClick={() => select((ordinary ? { id: "ordinary", title: "Ordinary saved chat", source: "webui-chat" } : { id: "robot", title: "Robot chat", character_id: 5, assistant_kind: "character", assistant_id: "5", source: "webui-character-chat" }) as Parameters<typeof select>[0])}>Open saved {ordinary ? "ordinary chat" : "Robot"}</button>
}

const messageOptionState = vi.hoisted(() => ({
  value: {
    messages: [],
    history: [],
    historyId: null,
    serverChatId: null,
    isLoading: false,
    setHistoryId: vi.fn(),
    setHistory: vi.fn(),
    setMessages: vi.fn(),
    setSelectedSystemPrompt: vi.fn(),
    setSelectedModel: vi.fn(),
    setServerChatId: vi.fn(),
    setServerChatCharacterId: vi.fn(),
    setServerChatAssistantKind: vi.fn(),
    setServerChatAssistantId: vi.fn(),
    setServerChatPersonaMemoryMode: vi.fn(),
    setServerChatMetaLoaded: vi.fn(),
    setContextFiles: vi.fn(),
    createChatBranch: vi.fn(),
    streaming: false,
    selectedCharacter: null,
    setSelectedCharacter: vi.fn(),
    compareMode: false,
    compareFeatureEnabled: false
  }
}))

const sessionPersistenceState = vi.hoisted(() => ({
  value: {
    restoreSession: vi.fn<() => Promise<PlaygroundSessionRestoreOutcome>>(
      async () => "not-restored"
    ),
    clearPersistedSession: vi.fn(async () => undefined),
    sessionScopeReady: true,
    hasPersistedSession: false,
    persistedHistoryId: null as string | null,
    persistedServerChatId: null as string | null
  }
}))

const restoreDecisionState = vi.hoisted(() => ({
  value: false as boolean | null
}))

const tldwClientState = vi.hoisted(() => ({
  createChat: vi.fn(),
  getConfig: vi.fn(async () => ({ serverUrl: "http://chat.test", authMode: "multi-user", accessToken: "test." + btoa(JSON.stringify({ sub: "A" })) + ".signature" })),
  initialize: vi.fn(async () => undefined),
  getProvidersStatus: vi.fn(async () => null),
  getChatSettings: vi.fn(async () => ({ settings: {} })),
  listConversationShareLinks: vi.fn(async () => ({ links: [] })),
  ensureConfigForRequest: vi.fn(async () => ({ serverUrl: "http://chat.test", authMode: "multi-user", accessToken: "test." + btoa(JSON.stringify({ sub: "A" })) + ".signature" })),
  getChat: vi.fn(),
  listAllCharacters: vi.fn(async () => [{ id: 5, name: "Robot" }, { id: 7, name: "New choice" }]),
  listPersonaProfiles: vi.fn(async () => []),
  listChatMessages: vi.fn(),
  getCharacter: vi.fn(async (id: string | number) => ({
    id,
    name: "Route Character"
  }))
}))


const stableTranslation = vi.hoisted(() => (key: string, fallback?: string | { defaultValue?: string }) => typeof fallback === "string" ? fallback : fallback?.defaultValue ?? key)
vi.mock("react-i18next", () => ({ useTranslation: () => ({ t: stableTranslation }) }))

vi.mock("@/components/Option/Playground/PlaygroundForm", () => ({
  PlaygroundForm: ({ characterWorkflowActive, characterChatSendBlocker }: { characterWorkflowActive?: boolean; characterChatSendBlocker?: unknown }) => <div data-testid="playground-form" data-character-workflow={String(characterWorkflowActive)} data-character-blocked={String(Boolean(characterChatSendBlocker))}>{realLoader.additionalLoader ? Array.from({ length: realLoader.webStorage ? 5 : 1 }, (_, index) => <AdditionalServerLoader key={index} />) : null}</div>
}))

vi.mock("@/components/Option/Playground/PlaygroundChat", () => ({
  PlaygroundChat: () => <div data-testid="playground-chat" />
}))

vi.mock("@/components/Sidepanel/Chat/ArtifactsPanel", () => ({
  ArtifactsPanel: () => null
}))

vi.mock("@/hooks/useMessageOption", () => ({
  useMessageOption: useMessageOptionMock
}))

vi.mock("@/hooks/useConnectionState", () => ({ useConnectionState: () => ({ serverUrl: "http://chat.test", lastConfigUpdatedAt: 0 }) }))
vi.mock("@/hooks/usePlaygroundSessionPersistence", async importOriginal => {
  const actual = await importOriginal<typeof import("@/hooks/usePlaygroundSessionPersistence")>()
  const useFixture = () => { usePlaygroundSessionStore(); return sessionPersistenceState.value }
  return { usePlaygroundSessionPersistence: () => {
    const useImplementation = realLoader.session ? actual.usePlaygroundSessionPersistence : useFixture
    return useImplementation()
  } }
})

vi.mock("@/hooks/playground-session-restore", async () => {
  const actual = await vi.importActual<
    typeof import("@/hooks/playground-session-restore")
  >("@/hooks/playground-session-restore")
  return {
    shouldRestorePersistedPlaygroundSession: (
      input: Parameters<
        typeof actual.shouldRestorePersistedPlaygroundSession
      >[0]
    ) =>
      restoreDecisionState.value ??
      actual.shouldRestorePersistedPlaygroundSession(input)
  }
})

vi.mock("@/services/model-settings", () => ({ lastUsedChatModelEnabled: async () => false }))

vi.mock("@/services/app", () => ({
  webUIResumeLastChat: vi.fn(async () => false)
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: tldwClientState
}))

vi.mock("@/services/tldw-server", async (importOriginal) => {
  const actual = await importOriginal<
    typeof import("@/services/tldw-server")
  >()
  return {
    ...actual,
    fetchChatModels: vi.fn(async () => [])
  }
})

vi.mock("@/db/dexie/helpers", () => ({
  generateID: () => "owned-message",
  formatToChatHistory: vi.fn(),
  formatToMessage: vi.fn(),
  getHistoryByServerChatId: vi.fn(async () => null),
  getPromptById: vi.fn(async () => null),
  getSessionFiles: vi.fn(async () => []),
  getRecentChatFromWebUI: vi.fn(async () => null)
}))

vi.mock("@/store/model", () => ({
  useStoreChatModelSettings: () => ({ setSystemPrompt: vi.fn(), reset: vi.fn() })
}))

vi.mock("@/components/Layouts/HeaderShortcuts", () => ({ HeaderShortcuts: () => null }))
vi.mock("@/hooks/useActiveChatTitle", () => ({ useActiveChatTitle: () => ({ title: "Saved Chat", ready: true, owner: { saving: false }, renameTitle: vi.fn() }) }))
vi.mock("@/hooks/useSmartScroll", () => ({
  useSmartScroll: () => ({
    containerRef: { current: null },
    isAutoScrollToBottom: true,
    autoScrollToBottom: vi.fn()
  })
}))

vi.mock("@/services/settings/ui-settings", () => ({
  CHAT_BACKGROUND_IMAGE_SETTING: "chatBackgroundImage",
  CHAT_WINDOW_OPACITY_SETTING: "chatWindowOpacity",
  CHAT_MESSAGE_OPACITY_SETTING: "chatMessageOpacity",
  CHAT_CHARACTER_IMAGE_OPACITY_SETTING: "chatCharacterImageOpacity",
  resolveOpacityAlpha: (value: unknown, fallback = 35) =>
    typeof value === "number" && Number.isFinite(value)
      ? value / 100
      : fallback / 100,
  THEME_SETTING: {
    key: "theme",
    defaultValue: "dark"
  },
  HEADER_SHORTCUTS_EXPANDED_SETTING: { key: "headerShortcutsExpanded", defaultValue: false },
  HEADER_SHORTCUT_IDS: [],
  SIDEBAR_SHORTCUT_IDS: []
}))

vi.mock("../Knowledge/utils/unsupported-types", () => ({
  otherUnsupportedTypes: []
}))

vi.mock("@/store/option", async importOriginal => {
  const actual = await importOriginal<typeof import("@/store/option")>()
  return { ...actual, useStoreMessageOption: Object.assign((selector?: (state: unknown) => unknown) => {
    if (realLoader.enabled) return actual.useStoreMessageOption(selector as never)
    return typeof selector === "function" ? selector({ compareParentByHistory: {} }) : { compareParentByHistory: {} }
  }, actual.useStoreMessageOption) }
})

vi.mock("@/services/service-prompts", () => ({
  loadServicePromptSnapshot: async (_ids: unknown, { signal }: { signal: AbortSignal }) => ({
    scopeKey: `scope-${ordinaryCompletion.owner}`, scopeSignal: signal, scopeInvalidatedSignal: realLoader.invalidated.signal,
    requestScope: { config: { serverUrl: "http://chat.test", authMode: "multi-user" }, userId: ordinaryCompletion.owner }, release: vi.fn()
  })
}))
vi.mock("@/services/chat-settings", () => ({ syncChatSettingsForServerChat: async () => null }))
vi.mock("@/hooks/chat/useChatSettingsRecord", () => ({ useChatSettingsRecord: () => ({ settings: null, updateSettings: vi.fn(async () => null) }) }))

vi.mock("@/store/artifacts", () => ({
  useArtifactsStore: (selector: (state: {
    isOpen: boolean
    active: null
    isPinned: boolean
    history: never[]
    unreadCount: number
    setOpen: ReturnType<typeof vi.fn>
    closeArtifact: ReturnType<typeof vi.fn>
    markRead: ReturnType<typeof vi.fn>
  }) => unknown) =>
    selector({
      isOpen: false,
      active: null,
      isPinned: false,
      history: [],
      unreadCount: 0,
      setOpen: vi.fn(),
      closeArtifact: vi.fn(),
      markRead: vi.fn()
    })
}))

vi.mock("@/hooks/useSetting", () => ({
  useSetting: (setting: string) => {
    if (setting === "chatWindowOpacity") return [35]
    if (setting === "chatMessageOpacity") return [60]
    if (setting === "chatCharacterImageOpacity") return [100]
    return [""]
  }
}))

vi.mock("@plasmohq/storage", () => import("../../../../../../../tldw-frontend/extension/shims/plasmo-storage"))
vi.mock("@plasmohq/storage/hook", async () => {
  const web = await import("../../../../../../../tldw-frontend/extension/shims/plasmo-storage-hook")
  const useControlledStorage = (key: string | { key: string }, defaultValue: unknown) => {
    const value = useStoreMessageOption(state => (state as SelectionTestState).testAssistant)
    if (realLoader.enabled && typeof key === "object" && key.key === "selectedAssistant") {
      return [value, setStorageAssistant, { isLoading: false, setRenderValue: setStorageAssistant }]
    }
    return [defaultValue, vi.fn()]
  }
  return { useStorage: (key: string | { key: string }, defaultValue: unknown) => {
    const useImplementation = realLoader.webStorage ? web.useStorage : useControlledStorage
    return useImplementation(key, defaultValue)
  } }
})

vi.mock("@/hooks/useMediaQuery", () => ({
  useMobile: () => false,
  useDesktop: () => true
}))


vi.mock("@/hooks/useServerChatHistory", () => ({
  useServerChatHistory: () => ({
    data: [],
    total: 0,
    isLoading: false,
    sidebarRefreshState: "ready",
    hasUsableData: true,
    isShowingStaleData: false
  })
}))

vi.mock("../playground-shortcuts", () => ({
  resolvePlaygroundShortcutAction: () => null
}))

vi.mock("@/hooks/useCharacterGreeting", () => ({
  useCharacterGreeting: () => undefined
}))

const subscribeRoute = (notify: () => void) => {
  window.addEventListener("popstate", notify)
  return () => window.removeEventListener("popstate", notify)
}
const readRoute = () => window.location.href
let routeCommitDelay = 0
const navigateRoute = (to: string | { pathname?: string; search?: string; hash?: string }) => {
  const commit = () => {
    window.history.replaceState({}, "", typeof to === "string" ? to : `${to.pathname || "/chat"}${to.search || ""}${to.hash || ""}`)
    window.dispatchEvent(new PopStateEvent("popstate"))
  }
  if (routeCommitDelay) setTimeout(commit, routeCommitDelay)
  else commit()
}
vi.mock("react-router-dom", async () => {
  const actual = await vi.importActual<typeof import("react-router-dom")>("react-router-dom")
  return { ...actual,
    useNavigate: () => realLoader.route ? navigateRoute : vi.fn(),
    useLocation: () => {
      React.useSyncExternalStore(subscribeRoute, readRoute)
      return { pathname: window.location.pathname || "/chat", search: window.location.search || "", hash: window.location.hash || "", state: null, key: "test-location" }
    }
  }
})

class RouteTestBoundary extends React.Component<{ children: React.ReactNode }, { failed: boolean }> {
  state = { failed: false }
  static getDerivedStateFromError() { return { failed: true } }
  render() { return this.state.failed ? <p role="alert">Chat route crashed</p> : this.props.children }
}

describe("Playground coordinator integration", () => {
  beforeEach(() => {
    ordinaryCompletion.owner = "A"
    ordinaryCompletion.save.mockClear()
    tldwClientState.createChat.mockResolvedValue({ id: "bob-created", title: "Bob chat", source: "webui-chat" })
    realLoader.session = false
    routeCommitDelay = 0
    realLoader.route = false
    realLoader.enabled = false
    realLoader.webStorage = false
    realLoader.additionalLoader = false
    realLoader.storageBarrier = null
    realLoader.storageWrites = 0
    useMessageOptionMock.mockImplementation(() => messageOptionState.value)
    realLoader.invalidated = new AbortController()
    window.history.pushState({}, "", "/chat")
    messageOptionState.value.messages = []
    messageOptionState.value.history = []
    messageOptionState.value.historyId = null
    messageOptionState.value.serverChatId = null
    messageOptionState.value.selectedCharacter = null
    messageOptionState.value.setMessages.mockClear()
    messageOptionState.value.setHistoryId.mockClear()
    messageOptionState.value.setServerChatId.mockClear()
    messageOptionState.value.setServerChatCharacterId.mockClear()
    messageOptionState.value.setServerChatAssistantKind.mockClear()
    messageOptionState.value.setServerChatAssistantId.mockClear()
    messageOptionState.value.setServerChatPersonaMemoryMode.mockClear()
    messageOptionState.value.setServerChatMetaLoaded.mockClear()
    messageOptionState.value.setSelectedCharacter.mockClear()
    tldwClientState.getChat.mockReset()
    tldwClientState.listChatMessages.mockReset()
    tldwClientState.initialize.mockClear()
    tldwClientState.getProvidersStatus.mockClear()
    tldwClientState.getCharacter.mockClear()
    tldwClientState.getCharacter.mockImplementation(async (id: string | number) => ({
      id,
      name: "Route Character"
    }))
    sessionPersistenceState.value.restoreSession = vi.fn(
      async () => "not-restored" as const
    )
    sessionPersistenceState.value.clearPersistedSession = vi.fn(
      async () => undefined
    )
    sessionPersistenceState.value.sessionScopeReady = true
    sessionPersistenceState.value.hasPersistedSession = false
    sessionPersistenceState.value.persistedHistoryId = null
    sessionPersistenceState.value.persistedServerChatId = null
    restoreDecisionState.value = false
    usePlaygroundSessionStore.setState({ restoreRevision: 0 })
    vi.mocked(webUIResumeLastChat).mockReset()
    vi.mocked(webUIResumeLastChat).mockResolvedValue(false)
    vi.mocked(getRecentChatFromWebUI).mockReset()
    vi.mocked(getRecentChatFromWebUI).mockResolvedValue(null)

    useChatSurfaceCoordinatorStore.setState({
      routeId: null,
      surface: null,
      visiblePanels: {
        "server-history": false,
        "mcp-tools": false,
        "audio-health": false,
        "model-catalog": false
      },
      engagedPanels: {
        "server-history": false,
        "mcp-tools": false,
        "audio-health": false,
        "model-catalog": false
      }
    })
  })

  it("uses ordinary workflow immediately after Bob creates and completes a chat following account clear", async () => {
    realLoader.enabled = realLoader.webStorage = true
    localStorage.clear()
    await selectedAssistantStorage.set("playgroundChatWorkflowMode", "character")
    useStoreMessageOption.setState({ ...useStoreMessageOption.getInitialState(), selectedModel: "test-model", temporaryChat: false,
      serverChatId: "alice-character", serverChatMetaLoaded: true, serverChatCharacterId: 4, serverChatAssistantKind: "character",
      messages: [{ id: "alice-private", isBot: true, message: "Alice private answer", sources: [] }]
    }, true)
    useMessageOptionMock.mockImplementation(() => ({ ...messageOptionState.value, ...useStoreMessageOption(), selectedAssistant: null }))
    const view = render(<><Playground /><OrdinarySend /></>)
    try {
      act(() => {
        window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed", { detail: { kind: "logout" } }))
        usePlaygroundSessionStore.getState().clearSession()
        useStoreMessageOption.getState().setServerChatId(null)
        useStoreMessageOption.setState({ messages: [], history: [], historyId: null })
        ordinaryCompletion.owner = "B"
        window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed", { detail: { kind: "login" } }))
      })
      await waitFor(() => expect(screen.getByTestId("playground-form")).toHaveAttribute("data-character-workflow", "true"))
      ordinaryCompletion.complete.mockImplementation(async (_message, _image, _regenerate, _messages, _history, _signal, params) => {
        params.setMessages([{ id: "bob-user", isBot: false, message: "Bob ordinary question", sources: [] }, { id: "bob-answer", isBot: true, message: "Bob answer", sources: [] }])
        await params.saveMessageOnSuccess({ message: "Bob ordinary question", fullText: "Bob answer", saveToDb: true, conversationId: params.serverChatId, userServerMessageId: "user-receipt", assistantServerMessageId: "answer-receipt" })
        return { status: "submitted" }
      })
      fireEvent.click(screen.getByRole("button", { name: "Send ordinary turn" }))
      await waitFor(() => expect(ordinaryCompletion.save).toHaveBeenCalledTimes(1))
      expect(tldwClientState.createChat).toHaveBeenCalledTimes(1)
      expect(tldwClientState.createChat.mock.calls[0][1]).toMatchObject({ requestScope: { userId: "B" } })
      expect(useStoreMessageOption.getState().serverChatId).toBe("bob-created")
      expect(useStoreMessageOption.getState().messages.map(row => row.message)).toEqual(["Bob ordinary question", "Bob answer"])
      expect(screen.getByTestId("playground-form")).toHaveAttribute("data-character-workflow", "false")
      expect(screen.getByTestId("playground-form")).toHaveAttribute("data-character-blocked", "false")
      expect(await selectedAssistantStorage.get("playgroundChatWorkflowMode")).toBe("character")
    } finally { view.unmount(); localStorage.clear() }
  })

  it.each(["/chat?mode=character&characterId=4&keep=1", "/options.html#/chat?mode=character&characterId=4&keep=1"])("promotes accepted entry to saved route and restores its transcript on remount: %s", async url => {
    realLoader.enabled = realLoader.webStorage = realLoader.session = realLoader.route = true
    restoreDecisionState.value = null
    localStorage.clear()
    usePlaygroundSessionStore.getState().clearSession()
    useStoreMessageOption.setState({ historyId: null, serverChatId: null, history: [], messages: [], serverChatMetaLoaded: false, serverChatCharacterId: null, serverChatAssistantKind: null, serverChatAssistantId: null, temporaryChat: false, streaming: false, isLoading: false, isProcessing: false, queuedMessages: [] })
    useMessageOptionMock.mockImplementation(useRealServerConversation)
    tldwClientState.getChat.mockResolvedValue({ id: "saved-cedar", title: "Cedar saved", scope_type: "global", character_id: 4, assistant_kind: "character", assistant_id: "4", source: "webui-character-chat" })
    tldwClientState.listChatMessages.mockResolvedValue([{ id: "question", role: "user", content: "Question", version: 1 }, { id: "answer", role: "assistant", content: "Saved final answer", version: 1 }])
    window.history.replaceState({}, "", url)
    const view = render(<Playground />)
    await waitFor(() => expect(tldwClientState.getCharacter).toHaveBeenCalledWith("4"))
    await waitFor(() => expect(screen.queryByTestId("playground-chat")).toBeInTheDocument())
    // The inference transport is outside this fixture; deliver its acknowledged owned target.
    await act(async () => { useStoreMessageOption.setState({ serverChatId: "saved-cedar", serverChatCharacterId: 4, serverChatAssistantKind: "character", serverChatAssistantId: "4", serverChatMetaLoaded: true }) })
    await waitFor(() => expect(window.location.href).toContain("chatId=saved-cedar"), { timeout: 2500 })
    expect(window.location.href).toContain("keep=1")
    expect(usePlaygroundSessionStore.getState().serverChatId).toBe("saved-cedar")
    view.unmount()
    useStoreMessageOption.setState({ historyId: null, serverChatId: null, messages: [], history: [], serverChatMetaLoaded: false, serverChatCharacterId: null, serverChatAssistantKind: null, serverChatAssistantId: null })
    const reloaded = render(<Playground />)
    await waitFor(() => expect(useStoreMessageOption.getState().messages.map(row => row.message)).toEqual(["Question", "Saved final answer"]))
    expect(useStoreMessageOption.getState().serverChatId).toBe("saved-cedar")
    // The same route command is still deliberate when selected again, including
    // the currently selected character. It must not resurrect the saved target.
    for (const id of ["4", "5"]) {
      const priorCalls = tldwClientState.getCharacter.mock.calls.length
      act(() => navigateRoute({ pathname: "/chat", search: `?mode=character&characterId=${id}` }))
      await waitFor(() => expect(tldwClientState.getCharacter.mock.calls.length).toBeGreaterThan(priorCalls))
      await waitFor(() => expect(useStoreMessageOption.getState().serverChatId).toBeNull())
      expect(useStoreMessageOption.getState().messages).toEqual([])
      expect(window.location.href).not.toContain("chatId=")
    }
    reloaded.unmount()
  })

  it.each([false, true])("does not publish a superseded saved target after delayed scope hydration (return to A: %s)", async returnToA => {
    realLoader.enabled = realLoader.webStorage = realLoader.session = realLoader.route = true
    restoreDecisionState.value = null
    localStorage.clear()
    usePlaygroundSessionStore.getState().clearSession()
    useStoreMessageOption.setState({ historyId: null, serverChatId: null, history: [], messages: [], serverChatMetaLoaded: false, serverChatCharacterId: null, serverChatAssistantKind: null, serverChatAssistantId: null, temporaryChat: false, streaming: false, isLoading: false, isProcessing: false, queuedMessages: [] })
    useMessageOptionMock.mockImplementation(useRealServerConversation)
    let release!: (config: Awaited<ReturnType<typeof tldwClientState.getConfig>>) => void
    const held = new Promise<Awaited<ReturnType<typeof tldwClientState.getConfig>>>(resolve => { release = resolve })
    tldwClientState.getConfig.mockImplementationOnce(() => held)
    window.history.replaceState({}, "", "/chat?mode=character&characterId=4")
    const view = render(<Playground />)
    await waitFor(() => expect(tldwClientState.getCharacter).toHaveBeenCalledWith("4"))
    act(() => useStoreMessageOption.setState({ serverChatId: "old-A", serverChatCharacterId: 4, serverChatAssistantKind: "character", serverChatAssistantId: "4", serverChatMetaLoaded: true }))
    expect(window.location.href).not.toContain("chatId=")
    // Apply the real logout boundary's clear before its old configuration read resolves.
    act(() => {
      window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed", { detail: { reason: "logout" } }))
      usePlaygroundSessionStore.getState().clearSession()
      useStoreMessageOption.setState({ serverChatId: null, historyId: null, messages: [], history: [], serverChatMetaLoaded: false, serverChatCharacterId: null, serverChatAssistantKind: null, serverChatAssistantId: null })
      if (returnToA) window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed", { detail: { reason: "login" } }))
    })
    await act(async () => { release({ serverUrl: "http://chat.test", authMode: "multi-user", accessToken: "test." + btoa(JSON.stringify({ sub: "A" })) + ".signature" }); await held })
    expect(window.location.href).not.toContain("chatId=")
    expect(useStoreMessageOption.getState().serverChatId).toBeNull()
    view.unmount()
  })

  it("registers the webui chat route context on mount", () => {
    render(<Playground />)

    expect(useChatSurfaceCoordinatorStore.getState().routeId).toBe("chat")
    expect(useChatSurfaceCoordinatorStore.getState().surface).toBe("webui")
  })

  it("waits for session scope readiness before consuming the one-time restore pass", async () => {
    const restoreSession = vi.fn(async () => "restored" as const)
    sessionPersistenceState.value.restoreSession = restoreSession
    sessionPersistenceState.value.sessionScopeReady = false
    sessionPersistenceState.value.hasPersistedSession = false
    restoreDecisionState.value = false

    const { rerender } = render(<Playground />)

    expect(restoreSession).not.toHaveBeenCalled()

    sessionPersistenceState.value.sessionScopeReady = true
    sessionPersistenceState.value.hasPersistedSession = true
    sessionPersistenceState.value.persistedHistoryId = "history-123"
    restoreDecisionState.value = true
    rerender(<Playground />)

    await waitFor(() => {
      expect(restoreSession).toHaveBeenCalledTimes(1)
    })
  })

  it("falls back to recent chat only when restore reports nothing restored", async () => {
    sessionPersistenceState.value.restoreSession = vi.fn(
      async () => "not-restored" as const
    )
    sessionPersistenceState.value.hasPersistedSession = true
    restoreDecisionState.value = true
    vi.mocked(webUIResumeLastChat).mockResolvedValue(true)

    render(<Playground />)

    await waitFor(() => {
      expect(getRecentChatFromWebUI).toHaveBeenCalledTimes(1)
    })
  })

  it("stops recent-chat initialization when restore reports cancellation", async () => {
    sessionPersistenceState.value.restoreSession = vi.fn(
      async () => "cancelled" as const
    )
    sessionPersistenceState.value.hasPersistedSession = true
    restoreDecisionState.value = true
    vi.mocked(webUIResumeLastChat).mockResolvedValue(true)

    render(<Playground />)

    await waitFor(() => {
      expect(sessionPersistenceState.value.restoreSession).toHaveBeenCalledTimes(1)
    })
    expect(webUIResumeLastChat).not.toHaveBeenCalled()
    expect(getRecentChatFromWebUI).not.toHaveBeenCalled()
  })

  it("does not overwrite a server chat selected while session scope initializes", async () => {
    const restoreSession = vi.fn(async () => "restored" as const)
    sessionPersistenceState.value.restoreSession = restoreSession
    sessionPersistenceState.value.sessionScopeReady = false
    sessionPersistenceState.value.hasPersistedSession = true
    sessionPersistenceState.value.persistedServerChatId = "persisted-chat"
    restoreDecisionState.value = null

    const { rerender } = render(<Playground />)

    messageOptionState.value.serverChatId = "selected-chat"
    sessionPersistenceState.value.sessionScopeReady = true
    rerender(<Playground />)

    await waitFor(() => {
      expect(screen.getByTestId("playground-chat")).toBeInTheDocument()
    })
    expect(restoreSession).not.toHaveBeenCalled()
  })

  it("applies explicit character chat route ids before persisted session restore", async () => {
    const restoreSession = vi.fn(async () => "restored" as const)
    sessionPersistenceState.value.restoreSession = restoreSession
    sessionPersistenceState.value.hasPersistedSession = true
    sessionPersistenceState.value.persistedServerChatId = "persisted-chat"
    restoreDecisionState.value = true
    window.history.pushState(
      {},
      "",
      "/chat?mode=character&chatId=route-chat&characterId=stale-character"
    )

    render(<Playground />)

    await waitFor(() => {
      expect(messageOptionState.value.setServerChatId).toHaveBeenCalledWith(
        "route-chat"
      )
    })
    expect(restoreSession).not.toHaveBeenCalled()
    expect(messageOptionState.value.setSelectedCharacter).not.toHaveBeenCalled()
  })

  it("applies character route ids before persisted session restore", async () => {
    const restoreSession = vi.fn(async () => "restored" as const)
    sessionPersistenceState.value.restoreSession = restoreSession
    sessionPersistenceState.value.hasPersistedSession = true
    sessionPersistenceState.value.persistedServerChatId = "persisted-chat"
    restoreDecisionState.value = true
    window.history.pushState(
      {},
      "",
      "/chat?mode=character&characterId=route-character"
    )

    render(<Playground />)

    await waitFor(() => {
      expect(tldwClientState.getCharacter).toHaveBeenCalledWith(
        "route-character"
      )
    })
    expect(restoreSession).not.toHaveBeenCalled()
    expect(usePlaygroundSessionStore.getState().restoreRevision).toBeGreaterThan(0)
    expect(messageOptionState.value.setSelectedCharacter).toHaveBeenCalledWith(
      expect.objectContaining({
        id: "route-character",
        name: "Route Character"
      })
    )
  })

  it("applies a settings-return server chat before persisted session restore", async () => {
    const restoreSession = vi.fn(async () => "restored" as const)
    sessionPersistenceState.value.restoreSession = restoreSession
    sessionPersistenceState.value.hasPersistedSession = true
    sessionPersistenceState.value.persistedServerChatId = "persisted-chat"
    restoreDecisionState.value = true
    window.history.pushState(
      {},
      "",
      `/chat?${SETTINGS_SERVER_CHAT_ID_PARAM}=settings-chat`
    )

    render(<Playground />)

    await waitFor(() => {
      expect(messageOptionState.value.setServerChatId).toHaveBeenCalledWith(
        "settings-chat"
      )
    })
    expect(restoreSession).not.toHaveBeenCalled()
  })

  it("selects the saved Flashcard source conversation through the actual neutral Chat route consumer", async () => {
    const source = getFlashcardSourceMeta({ source_ref_type: "message", source_ref_id: "source-message", conversation_id: "source-chat" })!
    window.history.pushState({}, "", source.href!)
    render(<Playground />)
    await waitFor(() => expect(messageOptionState.value.setServerChatId).toHaveBeenCalledWith("source-chat"))
    expect(tldwClientState.getCharacter).not.toHaveBeenCalled()
  })

  it("applies a sidepanel handoff before persisted session restore", async () => {
    const restoreSession = vi.fn(async () => "restored" as const)
    sessionPersistenceState.value.restoreSession = restoreSession
    sessionPersistenceState.value.hasPersistedSession = true
    sessionPersistenceState.value.persistedServerChatId = "persisted-chat"
    restoreDecisionState.value = true
    const handoff = encodeSidepanelChatWebUiHandoff({
      source: "sidepanel-chat",
      createdAt: Date.now(),
      serverChatId: "handoff-chat"
    })
    window.history.pushState(
      {},
      "",
      `/chat?${SIDEPANEL_CHAT_WEBUI_HANDOFF_PARAM}=${encodeURIComponent(handoff)}`
    )

    render(<Playground />)

    await waitFor(() => {
      expect(messageOptionState.value.setServerChatId).toHaveBeenCalledWith(
        "handoff-chat"
      )
    })
    expect(restoreSession).not.toHaveBeenCalled()
  })

  it("starts a fresh character route chat over an active server chat", async () => {
    messageOptionState.value.serverChatId = "active-chat"
    messageOptionState.value.historyId = "active-history"
    messageOptionState.value.messages = [
      {
        isBot: true,
        name: "Assistant",
        role: "assistant",
        message: "Prior reply",
        sources: []
      }
    ]
    messageOptionState.value.history = [
      {
        role: "assistant",
        content: "Prior reply"
      }
    ]
    window.history.pushState(
      {},
      "",
      "/chat?mode=character&characterId=route-character"
    )

    render(<Playground />)

    await waitFor(() => {
      expect(tldwClientState.getCharacter).toHaveBeenCalledWith(
        "route-character"
      )
    })
    expect(messageOptionState.value.setHistoryId).toHaveBeenCalledWith(null, {
      preserveServerChatId: false
    })
    expect(messageOptionState.value.setHistory).toHaveBeenCalledWith([])
    expect(messageOptionState.value.setMessages).toHaveBeenCalledWith([])
    expect(messageOptionState.value.setServerChatId).toHaveBeenCalledWith(null)
    expect(messageOptionState.value.setSelectedCharacter).toHaveBeenCalledWith(
      expect.objectContaining({
        id: "route-character",
        name: "Route Character"
      })
    )
  })

  it.each([false, true])("settles canonical saved entry with the real session subscription and local loader (StrictMode=%s)", async (strict) => {
    messageOptionState.value.serverChatId = "settings-chat"
    window.history.pushState({}, "", "/chat?settingsServerChatId=settings-chat")
    const error = vi.spyOn(console, "error").mockImplementation(() => undefined)
    const element = strict ? <React.StrictMode><Playground /></React.StrictMode> : <Playground />
    const view = render(<RouteTestBoundary>{element}</RouteTestBoundary>)
    await act(async () => { await Promise.resolve() })
    expect(screen.queryByRole("alert")).not.toBeInTheDocument()
    expect(usePlaygroundSessionStore.getState().restoreRevision).toBeLessThanOrEqual(2)
    await waitFor(() => expect(window.location.search).toBe(""))
    const settledRevision = usePlaygroundSessionStore.getState().restoreRevision
    view.rerender(<RouteTestBoundary>{element}</RouteTestBoundary>)
    await act(async () => { await Promise.resolve() })
    expect(usePlaygroundSessionStore.getState().restoreRevision).toBe(settledRevision)
    expect(error.mock.calls.flat().join(" ")).not.toContain("Maximum update depth")
    view.unmount()
  })

  it("cancels pending generic restoration before applying an explicit settings target", async () => {
    const capturedRevision = usePlaygroundSessionStore.getState().restoreRevision
    let release!: () => void
    const held = new Promise<void>(resolve => { release = resolve })
    let staleRestorePublished = false
    const pendingRestore = held.then(() => {
      if (usePlaygroundSessionStore.getState().restoreRevision === capturedRevision) staleRestorePublished = true
    })
    let revisionWhenApplied: number | null = null
    messageOptionState.value.setServerChatId.mockImplementationOnce(() => {
      revisionWhenApplied = usePlaygroundSessionStore.getState().restoreRevision
    })
    window.history.pushState({}, "", "/chat?settingsServerChatId=settings-chat")
    const view = render(<Playground />)
    await waitFor(() => expect(revisionWhenApplied).not.toBeNull())
    await act(async () => { release(); await pendingRestore })
    expect(revisionWhenApplied).toBeGreaterThan(capturedRevision)
    expect(staleRestorePublished).toBe(false)
    view.unmount()
  })

  it.each(["settings", "timeline"] as const)("finishes a delayed current %s local-history load", async origin => {
    const { PageAssistDatabase } = await import("@/db/dexie/chat")
    let release!: () => void
    const held = new Promise<void>(resolve => { release = resolve })
    const read = vi.spyOn(PageAssistDatabase.prototype, "getChatHistory").mockImplementation(async () => { await held; return [] })
    vi.spyOn(PageAssistDatabase.prototype, "getHistoryInfo").mockResolvedValue({ id: "owned-local", title: "Owned local title", createdAt: 1, is_rag: false })
    if (origin === "settings") window.history.pushState({}, "", "/chat?settingsHistoryId=owned-local")
    const view = render(<Playground />)
    if (origin === "timeline") {
      await act(async () => { await Promise.resolve() })
      act(() => window.dispatchEvent(new CustomEvent("tldw:open-history", { detail: { historyId: "owned-local" } })))
    }
    await waitFor(() => expect(read).toHaveBeenCalledWith("owned-local"))
    await act(async () => { release(); await held })
    await waitFor(() => expect(document.title).toBe("Owned local title"))
    expect(messageOptionState.value.setHistoryId).toHaveBeenCalledWith("owned-local", { preserveServerChatId: false })
    view.unmount()
  })

  it("does not complete a failed local settings return as a successful server selection", async () => {
    const { PageAssistDatabase } = await import("@/db/dexie/chat")
    let fail!: () => void
    const held = new Promise<never>((_resolve, reject) => { fail = () => reject(new Error("Synthetic local failure")) })
    const read = vi.spyOn(PageAssistDatabase.prototype, "getChatHistory").mockReturnValue(held)
    vi.spyOn(PageAssistDatabase.prototype, "getHistoryInfo").mockResolvedValue({ id: "missing-local", title: "Missing local title", createdAt: 1, is_rag: false })
    vi.spyOn(console, "error").mockImplementation(() => undefined)
    window.history.pushState({}, "", "/chat?settingsHistoryId=missing-local&settingsServerChatId=target-server")
    const view = render(<Playground />)
    await waitFor(() => expect(read).toHaveBeenCalled())
    messageOptionState.value.setServerChatId.mockClear()
    await act(async () => { fail(); await held.catch(() => undefined) })
    expect(messageOptionState.value.setServerChatId).not.toHaveBeenCalled()
    expect(window.location.search).toContain("settingsHistoryId=missing-local")
    view.unmount()
  })


  it.each(["history", "cold"])("uses canonical ordinary workflow after %s load despite a persisted Character preference", async entry => {
    realLoader.enabled = realLoader.webStorage = true
    useMessageOptionMock.mockImplementation(useRealServerConversation)
    localStorage.clear()
    await selectedAssistantStorage.set("playgroundChatWorkflowMode", "character")
    await selectedAssistantStorage.set("selectedAssistant", { kind: "character", id: "4", name: "Cedar", metadata: { selectionMode: "tracked" } })
    useStoreMessageOption.setState({ serverChatId: entry === "cold" ? "ordinary" : null, historyId: null, messages: [], history: [], serverChatMetaLoaded: false, serverChatCharacterId: null, serverChatAssistantKind: null, serverChatAssistantId: null, temporaryChat: false, streaming: false, isProcessing: false })
    tldwClientState.getChat.mockResolvedValue({ id: "ordinary", title: "Ordinary saved chat", source: "webui-chat", scope_type: "global" })
    tldwClientState.listChatMessages.mockResolvedValue([{ id: "answer", role: "assistant", content: "Ordinary answer", version: 1 }])
    const view = render(<><Playground /><SavedChatSelection ordinary /></>)
    try {
      if (entry === "history") fireEvent.click(screen.getByRole("button", { name: "Open saved ordinary chat" }))
      await waitFor(() => expect(useStoreMessageOption.getState().messages.map(row => row.message)).toContain("Ordinary answer"))
      await waitFor(() => expect(screen.getByTestId("playground-form")).toHaveAttribute("data-character-workflow", "false"))
      expect(screen.getByTestId("playground-form")).toHaveAttribute("data-character-blocked", "false")
      expect(screen.queryByText("Choose a character to start character chat")).not.toBeInTheDocument()
      expect(await selectedAssistantStorage.get("playgroundChatWorkflowMode")).toBe("character")
    } finally { view.unmount(); localStorage.clear() }
  })

  it.each(["pending", "scope-pending", "fresh", "unsaved", "character", "persona"])("preserves the intended workflow for %s chat state", async state => {
    realLoader.webStorage = true
    localStorage.clear()
    await selectedAssistantStorage.set("playgroundChatWorkflowMode", "character")
    const fresh = state === "fresh" || state === "unsaved"
    if (state === "fresh") window.history.pushState({}, "", "/chat?mode=character&characterId=4")
    sessionPersistenceState.value.sessionScopeReady = state !== "scope-pending"
    const base = messageOptionState.value
    useMessageOptionMock.mockImplementation(() => ({ ...base,
      serverChatId: fresh ? null : "owned-chat", serverChatMetaLoaded: state !== "pending",
      serverChatAssistantKind: state === "character" ? "character" : state === "persona" ? "persona" : null,
      serverChatCharacterId: state === "character" ? "4" : null,
      messages: state === "unsaved" ? [{ id: "draft", isBot: false, message: "Keep my unsaved thought" }] : []
    }))
    const view = render(<Playground />)
    try {
      await act(async () => { await new Promise(resolve => setTimeout(resolve, 30)) })
      await waitFor(() => expect(screen.getByTestId("playground-form")).toHaveAttribute("data-character-workflow", state === "persona" ? "false" : "true"))
      if (state === "unsaved") expect(base.setMessages).not.toHaveBeenCalled()
      expect(await selectedAssistantStorage.get("playgroundChatWorkflowMode")).toBe("character")
    } finally { view.unmount(); localStorage.clear() }
  })

  it.each(["profile", "messages"])("keeps a saved target over a previous character while %s is delayed", async delayed => {
    realLoader.enabled = true
    useMessageOptionMock.mockImplementation(useRealServerConversation)
    useStoreMessageOption.setState({ serverChatId: null, historyId: null, messages: [], history: [], serverChatMetaLoaded: false, serverChatCharacterId: null, serverChatAssistantKind: null, serverChatAssistantId: null, temporaryChat: true, streaming: false, isProcessing: false, testAssistant: { kind: "character", id: "4", name: "Cedar", metadata: { selectionMode: "tracked" } } } as Partial<SelectionTestState>)
    const profile = { id: 5, name: "Robot" }
    const messages = [{ id: "answer", role: "assistant", content: "BEEP BOOP", version: 1 }]
    let release!: () => void
    const pending = new Promise<void>(resolve => { release = resolve })
    tldwClientState.getChat.mockResolvedValue({ id: "robot", title: "Robot chat", character_id: 5, assistant_kind: "character", assistant_id: "5", source: "webui-character-chat", scope_type: "global" })
    tldwClientState.listChatMessages.mockImplementation(async () => { if (delayed === "messages") await pending; return messages })
    tldwClientState.getCharacter.mockImplementation(async () => { if (delayed === "profile") await pending; return profile })
    window.history.pushState({}, "", "/chat?settingsServerChatId=robot")
    const changes: Array<{ id: string | null; meta: boolean }> = []
    const stop = useStoreMessageOption.subscribe(state => changes.push({ id: state.serverChatId, meta: state.serverChatMetaLoaded }))
    const view = render(<Playground />)
    try {
      await waitFor(() => expect(tldwClientState.listChatMessages).toHaveBeenCalled())
      await act(async () => { await new Promise(resolve => setTimeout(resolve, 30)) })
      const readyIndex = changes.findIndex(row => row.meta)
      expect(readyIndex).toBeGreaterThan(-1)
      expect(changes.slice(readyIndex).some(row => row.id === null)).toBe(false)
      if (delayed === "profile") {
        expect(useStoreMessageOption.getState().serverChatLoadState).toBe("loaded")
        expect(useStoreMessageOption.getState().isLoading).toBe(false)
        expect(useStoreMessageOption.getState().messages.map(message => message.message)).toContain("BEEP BOOP")
      }
      await act(async () => { release(); await pending })
      await waitFor(() => expect(useStoreMessageOption.getState().messages.map(message => message.message)).toContain("BEEP BOOP"))
      await waitFor(() => expect((useStoreMessageOption.getState() as SelectionTestState).testAssistant).toMatchObject({ id: "5", name: "Robot" }))
      expect(tldwClientState.getChat).toHaveBeenCalledTimes(1)
    } finally { release(); stop(); view.unmount() }
  })

  it.each(["picker", "picker-roundtrip", "replacement", "principal"])("does not install a late saved identity after %s changes during metadata fetch", async change => {
    realLoader.enabled = true
    useMessageOptionMock.mockImplementation(useRealServerConversation)
    useStoreMessageOption.setState({ serverChatId: null, historyId: null, messages: [], history: [], serverChatMetaLoaded: false, serverChatCharacterId: null, serverChatAssistantKind: null, serverChatAssistantId: null, temporaryChat: true, streaming: false, isProcessing: false, testAssistant: { kind: "character", id: "4", name: "Cedar", metadata: { selectionMode: "tracked" } } } as Partial<SelectionTestState>)
    let release!: () => void
    const pending = new Promise<void>(resolve => { release = resolve })
    tldwClientState.getChat.mockImplementation(async () => { await pending; return { id: "robot", title: "Robot chat", character_id: 5, assistant_kind: "character", assistant_id: "5", source: "webui-character-chat", scope_type: "global" } })
    tldwClientState.listChatMessages.mockResolvedValue([])
    const view = render(<Playground />)
    await act(async () => useStoreMessageOption.getState().setServerChatId("robot"))
    try {
      await waitFor(() => expect(tldwClientState.getChat).toHaveBeenCalled())
      await act(async () => {
        if (change === "picker") await setTestAssistant({ kind: "character", id: "7", name: "New choice", metadata: { selectionMode: "tracked" } })
        if (change === "picker-roundtrip") { await setTestAssistant({ kind: "character", id: "7", name: "Other" }); await setTestAssistant({ kind: "character", id: "4", name: "Cedar" }) }
        if (change === "replacement") useStoreMessageOption.getState().setServerChatId(null)
        if (change === "principal") realLoader.invalidated.abort()
        release(); await pending
        await new Promise(resolve => setTimeout(resolve, 30))
      })
      expect((useStoreMessageOption.getState() as SelectionTestState).testAssistant?.id).toBe(change === "picker" ? "7" : "4")
    } finally { release(); view.unmount() }
  })

  it.each([false, true])("keeps the saved target while a raw preference change suppresses a late profile (same turn %s)", async sameTurn => {
    realLoader.enabled = true
    useMessageOptionMock.mockImplementation(useRealServerConversation)
    useStoreMessageOption.setState({ serverChatId: null, historyId: null, messages: [], history: [], serverChatMetaLoaded: false, serverChatCharacterId: null, serverChatAssistantKind: null, serverChatAssistantId: null, temporaryChat: true, streaming: false, isProcessing: false, testAssistant: { kind: "character", id: "4", name: "Cedar", metadata: { selectionMode: "tracked" } } } as Partial<SelectionTestState>)
    let release!: () => void
    const pending = new Promise<void>(resolve => { release = resolve })
    tldwClientState.getChat.mockResolvedValue({ id: "robot", title: "Robot chat", character_id: 5, assistant_kind: "character", assistant_id: "5", source: "webui-character-chat", scope_type: "global" })
    tldwClientState.listChatMessages.mockResolvedValue([])
    tldwClientState.getCharacter.mockImplementation(async () => { await pending; return { id: 5, name: "Robot" } })
    window.history.pushState({}, "", "/chat?settingsServerChatId=robot")
    const view = render(<Playground />)
    try {
      await waitFor(() => expect(tldwClientState.getCharacter).toHaveBeenCalled())
      await act(async () => {
        const selected = setTestAssistant({ kind: "character", id: "7", name: "New choice", metadata: { selectionMode: "tracked" } })
        if (sameTurn) { release(); await pending }
        await selected
      })
      await act(async () => { release(); await pending; await new Promise(resolve => setTimeout(resolve, 300)) })
      expect((useStoreMessageOption.getState() as SelectionTestState).testAssistant?.id).toBe("7")
      expect(useStoreMessageOption.getState().serverChatId).toBe("robot")
    } finally { release(); view.unmount() }
  })

  it.each(["replacement", "principal", "unmount"])("discards optional profile completion after %s", async change => {
    realLoader.enabled = true
    useMessageOptionMock.mockImplementation(useRealServerConversation)
    useStoreMessageOption.setState({ serverChatId: null, historyId: null, messages: [], history: [], serverChatMetaLoaded: false, serverChatCharacterId: null, serverChatAssistantKind: null, serverChatAssistantId: null, temporaryChat: true, streaming: false, isProcessing: false, testAssistant: { kind: "character", id: "4", name: "Cedar", metadata: { selectionMode: "tracked" } } } as Partial<SelectionTestState>)
    let release!: () => void
    const pending = new Promise<void>(resolve => { release = resolve })
    tldwClientState.getChat.mockResolvedValue({ id: "robot", title: "Robot chat", character_id: 5, assistant_kind: "character", assistant_id: "5", source: "webui-character-chat", scope_type: "global" })
    tldwClientState.listChatMessages.mockResolvedValue([])
    tldwClientState.getCharacter.mockImplementation(async () => { await pending; return { id: 5, name: "Late Robot profile" } })
    const view = render(<Playground />)
    await act(async () => useStoreMessageOption.getState().setServerChatId("robot"))
    try {
      await waitFor(() => expect(tldwClientState.getCharacter).toHaveBeenCalled())
      await act(async () => {
        if (change === "replacement") useStoreMessageOption.getState().setServerChatId(null)
        if (change === "principal") realLoader.invalidated.abort()
        if (change === "unmount") view.unmount()
        release(); await pending
        await new Promise(resolve => setTimeout(resolve, 30))
      })
      expect((useStoreMessageOption.getState() as SelectionTestState).testAssistant?.name).not.toBe("Late Robot profile")
    } finally { release(); view.unmount() }
  })

  it("loads canonical messages without waiting for optional global selection persistence", async () => {
    realLoader.enabled = true
    realLoader.additionalLoader = true
    useMessageOptionMock.mockImplementation(useRealServerConversation)
    useStoreMessageOption.setState({ serverChatId: null, historyId: null, messages: [], history: [], serverChatMetaLoaded: false, serverChatCharacterId: null, serverChatAssistantKind: null, serverChatAssistantId: null, temporaryChat: true, streaming: false, isProcessing: false, testAssistant: { kind: "character", id: "4", name: "Cedar", metadata: { selectionMode: "tracked" } } } as Partial<SelectionTestState>)
    let releaseStorage!: () => void
    realLoader.storageBarrier = new Promise<void>(resolve => { releaseStorage = resolve })
    tldwClientState.getChat.mockResolvedValue({ id: "robot", title: "Robot chat", character_id: 5, assistant_kind: "character", assistant_id: "5", source: "webui-character-chat", scope_type: "global" })
    tldwClientState.listChatMessages.mockResolvedValue([{ id: "answer", role: "assistant", content: "BEEP BOOP", version: 1 }])
    tldwClientState.getCharacter.mockResolvedValue({ id: 5, name: "Robot" })
    window.history.pushState({}, "", "/chat?settingsServerChatId=robot")
    const view = render(<Playground />)
    try {
      await waitFor(() => expect(realLoader.storageWrites).toBeGreaterThan(0))
      await waitFor(() => expect(useStoreMessageOption.getState().serverChatLoadState).toBe("loaded"))
      expect(useStoreMessageOption.getState().serverChatId).toBe("robot")
      expect(useStoreMessageOption.getState().messages.map(message => message.message)).toContain("BEEP BOOP")
      expect((useStoreMessageOption.getState() as SelectionTestState).testAssistant?.id).toBe("4")
      await act(async () => { releaseStorage(); await realLoader.storageBarrier })
      await waitFor(() => expect((useStoreMessageOption.getState() as SelectionTestState).testAssistant).toMatchObject({ id: "5", name: "Robot" }))
    } finally { releaseStorage(); view.unmount() }
  })

  it.each(["immediate", "profile", "messages", "sidebar", "cross-tab"])("preserves canonical saved identity with actual WebUI storage and six loaders: %s", async variant => {
    realLoader.enabled = true
    realLoader.webStorage = true
    realLoader.additionalLoader = true
    useMessageOptionMock.mockImplementation(useRealServerConversation)
    window.localStorage.clear()
    await selectedAssistantStorage.set("selectedAssistant", { kind: "character", id: "4", name: "Cedar", metadata: { selectionMode: "tracked" } })
    useStoreMessageOption.setState({ serverChatId: null, historyId: null, messages: [], history: [], serverChatMetaLoaded: false, serverChatCharacterId: null, serverChatAssistantKind: null, serverChatAssistantId: null, temporaryChat: true, streaming: false, isProcessing: false })
    let release!: () => void
    const pending = new Promise<void>(resolve => { release = resolve })
    tldwClientState.getChat.mockResolvedValue({ id: "robot", title: "Robot chat", character_id: 5, assistant_kind: "character", assistant_id: "5", source: "webui-character-chat", scope_type: "global" })
    tldwClientState.listChatMessages.mockImplementation(async () => { if (variant === "messages") await pending; return [{ id: "answer", role: "assistant", content: "BEEP BOOP", version: 1 }] })
    tldwClientState.getCharacter.mockImplementation(async () => { if (variant === "profile") await pending; return { id: 5, name: "Robot" } })
    if (variant !== "sidebar") window.history.pushState({}, "", "/chat?settingsServerChatId=robot")
    const view = render(<><Playground /><SavedChatSelection /></>)
    try {
      if (variant === "sidebar") fireEvent.click(screen.getByRole("button", { name: "Open saved Robot" }))
      await waitFor(() => expect(tldwClientState.listChatMessages).toHaveBeenCalled())
      await act(async () => { await new Promise(resolve => setTimeout(resolve, 40)) })
      expect(useStoreMessageOption.getState().serverChatId).toBe("robot")
      if (variant === "profile") {
        expect(useStoreMessageOption.getState().isLoading).toBe(false)
        expect(useStoreMessageOption.getState().messages.map(message => message.message)).toContain("BEEP BOOP")
      }
      await act(async () => { release(); await pending })
      await waitFor(() => expect(useStoreMessageOption.getState().messages.map(message => message.message)).toContain("BEEP BOOP"))
      if (variant === "cross-tab") {
        const replacement = JSON.stringify({ kind: "character", id: "7", name: "Another tab", metadata: { selectionMode: "tracked" } })
        await act(async () => {
          window.localStorage.setItem("selectedAssistant", replacement)
          window.dispatchEvent(new StorageEvent("storage", { key: "selectedAssistant", newValue: replacement }))
        })
        const current = useMessageOptionMock.mock.results.at(-1)?.value
        expect(current.selectedAssistant).toMatchObject({ kind: "character", id: "5" })
        expect(useStoreMessageOption.getState().serverChatId).toBe("robot")
        expect(useStoreMessageOption.getState().messages.map(message => message.message)).toContain("BEEP BOOP")
      }
    } finally { release(); view.unmount() }
  })

  it.each(["Robot", "New choice"])("uses the real picker to deliberately select %s with a saved target", async choice => {
    realLoader.enabled = true
    realLoader.webStorage = true
    realLoader.additionalLoader = true
    useMessageOptionMock.mockImplementation(useRealServerConversation)
    window.localStorage.clear()
    await selectedAssistantStorage.set("selectedAssistant", { kind: "character", id: "5", name: "Robot", metadata: { selectionMode: "tracked" } })
    useStoreMessageOption.setState({ serverChatId: null, historyId: null, messages: [], history: [], serverChatMetaLoaded: false, serverChatCharacterId: null, serverChatAssistantKind: null, serverChatAssistantId: null, temporaryChat: true, streaming: false, isProcessing: false })
    let release!: () => void
    const pending = new Promise<void>(resolve => { release = resolve })
    tldwClientState.getChat.mockResolvedValue({ id: "robot", title: "Robot chat", character_id: 5, assistant_kind: "character", assistant_id: "5", source: "webui-character-chat", scope_type: "global" })
    tldwClientState.listChatMessages.mockResolvedValue([{ id: "answer", role: "assistant", content: "BEEP BOOP", version: 1 }])
    tldwClientState.getCharacter.mockImplementation(async () => { await pending; return { id: 5, name: "Late Robot" } })
    window.history.pushState({}, "", "/chat?settingsServerChatId=robot")
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const view = render(<QueryClientProvider client={queryClient}><Playground /><AssistantSelect /></QueryClientProvider>)
    try {
      await waitFor(() => expect(tldwClientState.getCharacter).toHaveBeenCalled())
      fireEvent.click(await screen.findByRole("button", { name: "Robot" }))
      const buttons = await screen.findAllByRole("button", { name: choice })
      fireEvent.click(buttons.at(-1)!)
      // The handler detaches synchronously, before selection persistence or the held profile.
      expect(useStoreMessageOption.getState().serverChatId).toBe(choice === "Robot" ? "robot" : null)
      await act(async () => { release(); await pending })
      await waitFor(async () => expect(await selectedAssistantStorage.get("selectedAssistant")).toMatchObject({ id: choice === "Robot" ? "5" : "7", name: choice }))
      expect(useStoreMessageOption.getState().serverChatId).toBe(choice === "Robot" ? "robot" : null)
    } finally { release(); view.unmount(); queryClient.clear() }
  })

  it.each(["search", "hash", "roundtrip"])("UAT068 replaces a fully loaded saved Character from its %s route without restoring it", async routeKind => {
    const choice = "New choice"
    realLoader.route = true
    routeCommitDelay = routeKind === "roundtrip" ? 250 : 70
    realLoader.enabled = true
    realLoader.webStorage = true
    realLoader.additionalLoader = true
    useMessageOptionMock.mockImplementation(useRealServerConversation)
    window.localStorage.clear()
    await selectedAssistantStorage.set("selectedAssistant", { kind: "character", id: "5", name: "Robot", metadata: { selectionMode: "tracked" } })
    useStoreMessageOption.setState({ serverChatId: null, historyId: null, messages: [], history: [], serverChatMetaLoaded: false, serverChatCharacterId: null, serverChatAssistantKind: null, serverChatAssistantId: null, temporaryChat: true, streaming: false, isProcessing: false })
    let release!: () => void
    const pending = new Promise<void>(resolve => { release = resolve })
    tldwClientState.getChat.mockResolvedValue({ id: "robot", title: "Robot chat", character_id: 5, assistant_kind: "character", assistant_id: "5", source: "webui-character-chat", scope_type: "global" })
    tldwClientState.listChatMessages.mockResolvedValue([{ id: "answer", role: "assistant", content: "BEEP BOOP", version: 1 }])
    tldwClientState.getCharacter.mockImplementation(async id => ({ id, name: String(id) === "5" ? "Robot" : "New choice" }))
    window.history.pushState({}, "", routeKind !== "hash" ? "/chat?mode=character&characterId=5&chatId=robot&keep=1#anchor" : "/options.html?keep=1#/chat?mode=character&characterId=5&chatId=robot&tab=2")
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const view = render(<QueryClientProvider client={queryClient}><Playground /><AssistantSelect /></QueryClientProvider>)
    try {
      await waitFor(() => expect(useStoreMessageOption.getState().serverChatLoadState).toBe("loaded"))
      await waitFor(async () => expect(await selectedAssistantStorage.get("selectedAssistant")).toMatchObject({ id: "5", name: "Robot" }))
      fireEvent.click(await screen.findByRole("button", { name: "Robot" }))
      const buttons = await screen.findAllByRole("button", { name: choice })
      fireEvent.click(buttons.at(-1)!)
      // The handler detaches synchronously, before selection persistence or the held profile.
      expect(useStoreMessageOption.getState().serverChatId).toBe(null)
      await act(async () => { release(); await pending })
      await waitFor(async () => expect(await selectedAssistantStorage.get("selectedAssistant")).toMatchObject({ id: "7", name: choice }))
      if (routeKind === "roundtrip") {
        fireEvent.click(await screen.findByRole("button", { name: "New choice" }))
        fireEvent.click((await screen.findAllByRole("button", { name: "Robot" })).at(-1)!)
      }
      await act(async () => { await new Promise(resolve => setTimeout(resolve, routeKind === "roundtrip" ? 500 : 180)) })
      expect(useStoreMessageOption.getState().serverChatId).toBeNull()
      expect(useStoreMessageOption.getState().messages).toEqual([])
      expect(window.location.search + window.location.hash).not.toContain("chatId=robot")
      expect(window.location.search + window.location.hash).toContain(routeKind === "roundtrip" ? "characterId=5" : "characterId=7")
      expect(window.location.search).toContain("keep=1")
      expect(await selectedAssistantStorage.get("selectedAssistant")).toMatchObject({ id: routeKind === "roundtrip" ? "5" : "7", name: routeKind === "roundtrip" ? "Robot" : choice })
      // A later deliberate revisit must still restore the untouched old saved chat.
      act(() => navigateRoute(routeKind !== "hash" ? "/chat?mode=character&characterId=5&chatId=robot&keep=1#anchor" : "/options.html?keep=1#/chat?mode=character&characterId=5&chatId=robot&tab=2"))
      await waitFor(() => expect(useStoreMessageOption.getState().serverChatId).toBe("robot"))
      await waitFor(() => expect(useStoreMessageOption.getState().messages.map(row => row.message)).toEqual(["BEEP BOOP"]))
    } finally { release(); view.unmount(); queryClient.clear() }
  })

  it.each([0, 70])("UAT068 clear survives a %sms route commit without restoring the saved Character", async delay => {
    const routeKind = "search"
    realLoader.route = true
    routeCommitDelay = delay
    realLoader.enabled = true
    realLoader.webStorage = true
    realLoader.additionalLoader = true
    useMessageOptionMock.mockImplementation(useRealServerConversation)
    window.localStorage.clear()
    await selectedAssistantStorage.set("selectedAssistant", { kind: "character", id: "5", name: "Robot", metadata: { selectionMode: "tracked" } })
    useStoreMessageOption.setState({ serverChatId: null, historyId: null, messages: [], history: [], serverChatMetaLoaded: false, serverChatCharacterId: null, serverChatAssistantKind: null, serverChatAssistantId: null, temporaryChat: true, streaming: false, isProcessing: false })
    let release!: () => void
    new Promise<void>(resolve => { release = resolve })
    tldwClientState.getChat.mockResolvedValue({ id: "robot", title: "Robot chat", character_id: 5, assistant_kind: "character", assistant_id: "5", source: "webui-character-chat", scope_type: "global" })
    tldwClientState.listChatMessages.mockResolvedValue([{ id: "answer", role: "assistant", content: "BEEP BOOP", version: 1 }])
    tldwClientState.getCharacter.mockImplementation(async () => { return { id: 5, name: "Robot" } })
    window.history.pushState({}, "", routeKind === "search" ? "/chat?mode=character&characterId=5&chatId=robot&keep=1#anchor" : "/options.html?keep=1#/chat?mode=character&characterId=5&chatId=robot&tab=2")
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const view = render(<QueryClientProvider client={queryClient}><Playground /><Header /></QueryClientProvider>)
    try {
      await waitFor(() => expect(useStoreMessageOption.getState().serverChatLoadState).toBe("loaded"))
      await waitFor(async () => expect(await selectedAssistantStorage.get("selectedAssistant")).toMatchObject({ id: "5", name: "Robot" }))
      fireEvent.click(screen.getByRole("button", { name: "New saved chat" }))
      await act(async () => { await new Promise(resolve => setTimeout(resolve, 180)) })
      expect(window.location.pathname + window.location.search).toBe("/chat")
      expect(useStoreMessageOption.getState().serverChatId).toBeNull()
      expect(useStoreMessageOption.getState().messages).toEqual([])
    } finally { release(); view.unmount(); queryClient.clear() }
  })

  it.each(["cancel", "other-surface", "other-target", "other-history", "authority-roundtrip"])("UAT068 rejects %s before retiring the owned saved route", async boundary => {
    realLoader.enabled = realLoader.webStorage = realLoader.route = true
    useMessageOptionMock.mockImplementation(useRealServerConversation)
    localStorage.clear()
    await selectedAssistantStorage.set("selectedAssistant", { kind: "character", id: "5", name: "Robot", metadata: { selectionMode: "tracked" } })
    useStoreMessageOption.setState({ serverChatId: null, historyId: null, messages: [], history: [], serverChatMetaLoaded: false, serverChatCharacterId: null, serverChatAssistantKind: null, serverChatAssistantId: null, temporaryChat: false, streaming: false, isProcessing: false })
    tldwClientState.getChat.mockResolvedValue({ id: "robot", title: "Robot chat", character_id: 5, assistant_kind: "character", assistant_id: "5", source: "webui-character-chat", scope_type: "global" })
    tldwClientState.listChatMessages.mockResolvedValue([{ id: "answer", role: "assistant", content: "BEEP BOOP", version: 1 }])
    tldwClientState.getCharacter.mockResolvedValue({ id: 5, name: "Robot" })
    window.history.replaceState({}, "", "/chat?mode=character&characterId=5&chatId=robot")
    const view = render(<><Playground /><Header /></>)
    const decline = (event: Event) => event.preventDefault()
    try {
      await waitFor(() => expect(useStoreMessageOption.getState().serverChatLoadState).toBe("loaded"))
      const before = useStoreMessageOption.getState()
      const href = window.location.href
      const detail = { href, serverChatId: before.serverChatId, historyId: before.historyId, restoreRevision: usePlaygroundSessionStore.getState().restoreRevision, characterId: "7" }
      if (boundary === "cancel") {
        window.addEventListener(SETTINGS_NAVIGATION_REQUEST_EVENT, decline)
        fireEvent.click(screen.getByRole("button", { name: "New saved chat" }))
      } else {
        if (boundary === "other-surface") detail.href = "http://localhost/sidepanel.html"
        if (boundary === "other-target") detail.serverChatId = "another-chat"
        if (boundary === "other-history") detail.historyId = "another-history"
        if (boundary === "authority-roundtrip") {
          act(() => { usePlaygroundSessionStore.getState().cancelPendingRestore(); usePlaygroundSessionStore.getState().cancelPendingRestore() })
        }
        act(() => window.dispatchEvent(new CustomEvent(CHAT_ROUTE_REPLACEMENT_EVENT, { detail })))
      }
      await act(async () => { await new Promise(resolve => setTimeout(resolve, 40)) })
      expect(window.location.href).toBe(href)
      expect(useStoreMessageOption.getState().serverChatId).toBe("robot")
      expect(useStoreMessageOption.getState().temporaryChat).toBe(false)
      expect(useStoreMessageOption.getState().messages.map(row => row.message)).toEqual(["BEEP BOOP"])
      expect(await selectedAssistantStorage.get("selectedAssistant")).toMatchObject({ id: "5", name: "Robot" })
    } finally { window.removeEventListener(SETTINGS_NAVIGATION_REQUEST_EVENT, decline); view.unmount() }
  })

  it("keeps the saved conversation and preference after cancelling the real legacy picker confirmation", async () => {
    realLoader.enabled = true
    realLoader.webStorage = true
    window.localStorage.clear()
    await selectedAssistantStorage.set("selectedAssistant", { kind: "character", id: "5", name: "Robot", metadata: { selectionMode: "tracked" } })
    useStoreMessageOption.setState({ serverChatId: "robot", historyId: null, messages: [], history: [], serverChatMetaLoaded: true, serverChatCharacterId: 5, serverChatAssistantKind: "character", serverChatAssistantId: "5", temporaryChat: true, streaming: false, isProcessing: false })
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const view = render(<App><QueryClientProvider client={queryClient}><CharacterSelect /></QueryClientProvider></App>)
    try {
      fireEvent.click(await screen.findByRole("button", { name: /^Robot —/ }))
      fireEvent.click(await screen.findByRole("menuitem", { name: /New choice/ }))
      fireEvent.click(await screen.findByRole("button", { name: "Cancel" }))
      await waitFor(() => expect(screen.queryByRole("dialog", { name: "Switch character?" })).not.toBeInTheDocument())
      expect(useStoreMessageOption.getState().serverChatId).toBe("robot")
      expect(await selectedAssistantStorage.get("selectedAssistant")).toMatchObject({ id: "5", name: "Robot" })
    } finally { view.unmount(); queryClient.clear() }
  })

})
