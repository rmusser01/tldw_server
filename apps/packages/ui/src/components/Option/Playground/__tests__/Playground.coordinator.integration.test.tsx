// @vitest-environment jsdom
import React from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { render, screen, waitFor } from "@testing-library/react"

import { Playground } from "../Playground"
import { useChatSurfaceCoordinatorStore } from "@/store/chat-surface-coordinator"

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
    restoreSession: vi.fn(async () => false),
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
  initialize: vi.fn(async () => undefined),
  getProvidersStatus: vi.fn(async () => null),
  getCharacter: vi.fn(async (id: string | number) => ({
    id,
    name: "Route Character"
  }))
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, defaultValue?: string) => defaultValue || key
  })
}))

vi.mock("@/components/Option/Playground/PlaygroundForm", () => ({
  PlaygroundForm: () => <div data-testid="playground-form" />
}))

vi.mock("@/components/Option/Playground/PlaygroundChat", () => ({
  PlaygroundChat: () => <div data-testid="playground-chat" />
}))

vi.mock("@/components/Sidepanel/Chat/ArtifactsPanel", () => ({
  ArtifactsPanel: () => null
}))

vi.mock("@/hooks/useMessageOption", () => ({
  useMessageOption: () => messageOptionState.value
}))

vi.mock("@/hooks/usePlaygroundSessionPersistence", () => ({
  usePlaygroundSessionPersistence: () => sessionPersistenceState.value
}))

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

vi.mock("@/services/app", () => ({
  webUIResumeLastChat: vi.fn(async () => false)
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: tldwClientState
}))

vi.mock("@/services/tldw-server", () => ({
  fetchChatModels: vi.fn(async () => [])
}))

vi.mock("@/db/dexie/helpers", () => ({
  formatToChatHistory: vi.fn(),
  formatToMessage: vi.fn(),
  getHistoryByServerChatId: vi.fn(async () => null),
  getPromptById: vi.fn(async () => null),
  getRecentChatFromWebUI: vi.fn(async () => null)
}))

vi.mock("@/store/model", () => ({
  useStoreChatModelSettings: () => ({ setSystemPrompt: vi.fn() })
}))

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
  HEADER_SHORTCUT_IDS: [],
  SIDEBAR_SHORTCUT_IDS: []
}))

vi.mock("../Knowledge/utils/unsupported-types", () => ({
  otherUnsupportedTypes: []
}))

vi.mock("@/store/option", () => ({
  useStoreMessageOption: (selector?: (state: { compareParentByHistory: Record<string, never> }) => unknown) =>
    typeof selector === "function" ? selector({ compareParentByHistory: {} }) : { compareParentByHistory: {} }
}))

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

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (_key: string, defaultValue: unknown) => [defaultValue, vi.fn()]
}))

vi.mock("@/hooks/useMediaQuery", () => ({
  useMobile: () => false,
  useDesktop: () => true
}))

vi.mock("@/hooks/useLoadLocalConversation", () => ({
  useLoadLocalConversation: () => vi.fn(async () => {})
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

vi.mock("react-router-dom", async () => {
  const actual = await vi.importActual<typeof import("react-router-dom")>(
    "react-router-dom"
  )
  return {
    ...actual,
    useNavigate: () => vi.fn(),
    useLocation: () => ({
      pathname: window.location.pathname || "/chat",
      search: window.location.search || "",
      hash: window.location.hash || "",
      state: null,
      key: "test-location"
    })
  }
})

describe("Playground coordinator integration", () => {
  beforeEach(() => {
    window.history.pushState({}, "", "/chat")
    messageOptionState.value.messages = []
    messageOptionState.value.history = []
    messageOptionState.value.historyId = null
    messageOptionState.value.serverChatId = null
    messageOptionState.value.selectedCharacter = null
    messageOptionState.value.setServerChatId.mockClear()
    messageOptionState.value.setServerChatCharacterId.mockClear()
    messageOptionState.value.setServerChatAssistantKind.mockClear()
    messageOptionState.value.setServerChatAssistantId.mockClear()
    messageOptionState.value.setServerChatPersonaMemoryMode.mockClear()
    messageOptionState.value.setServerChatMetaLoaded.mockClear()
    messageOptionState.value.setSelectedCharacter.mockClear()
    tldwClientState.initialize.mockClear()
    tldwClientState.getProvidersStatus.mockClear()
    tldwClientState.getCharacter.mockClear()
    tldwClientState.getCharacter.mockImplementation(async (id: string | number) => ({
      id,
      name: "Route Character"
    }))
    sessionPersistenceState.value.restoreSession = vi.fn(async () => false)
    sessionPersistenceState.value.clearPersistedSession = vi.fn(
      async () => undefined
    )
    sessionPersistenceState.value.sessionScopeReady = true
    sessionPersistenceState.value.hasPersistedSession = false
    sessionPersistenceState.value.persistedHistoryId = null
    sessionPersistenceState.value.persistedServerChatId = null
    restoreDecisionState.value = false

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

  it("registers the webui chat route context on mount", () => {
    render(<Playground />)

    expect(useChatSurfaceCoordinatorStore.getState().routeId).toBe("chat")
    expect(useChatSurfaceCoordinatorStore.getState().surface).toBe("webui")
  })

  it("waits for session scope readiness before consuming the one-time restore pass", async () => {
    const restoreSession = vi.fn(async () => true)
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

  it("does not overwrite a server chat selected while session scope initializes", async () => {
    const restoreSession = vi.fn(async () => true)
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
    const restoreSession = vi.fn(async () => true)
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
    const restoreSession = vi.fn(async () => true)
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
    expect(messageOptionState.value.setSelectedCharacter).toHaveBeenCalledWith(
      expect.objectContaining({
        id: "route-character",
        name: "Route Character"
      })
    )
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
})
