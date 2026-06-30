// @vitest-environment jsdom
import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { Playground } from "../Playground"
import {
  decodeSidepanelChatWebUiHandoff,
  encodeSidepanelChatWebUiHandoff,
  SIDEPANEL_CHAT_WEBUI_HANDOFF_PARAM,
  SIDEPANEL_CHAT_WEBUI_HANDOFF_SOURCE
} from "@/services/tldw/sidepanel-chat-webui-handoff"

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
    setContextFiles: vi.fn(),
    createChatBranch: vi.fn(),
    streaming: false,
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
    autoScrollToBottom: vi.fn()
  }
}))

const mobileViewportState = vi.hoisted(() => ({
  value: false
}))

const desktopViewportState = vi.hoisted(() => ({
  value: true
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
    >
  }
}))

const routerState = vi.hoisted(() => ({
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

vi.mock("@/components/Option/Playground/PlaygroundForm", () => ({
  PlaygroundForm: () => <div data-testid="playground-form" />
}))

vi.mock("@/components/Option/Playground/PlaygroundChat", () => ({
  PlaygroundChat: React.forwardRef(function MockPlaygroundChat(
    props: {
      searchQuery?: string
      matchedMessageIndices?: Set<number>
      activeSearchMessageIndex?: number | null
    },
    _ref
  ) {
    return (
      <div
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
  useMessageOption: () => messageOptionState.value
}))

vi.mock("@/hooks/usePlaygroundSessionPersistence", () => ({
  usePlaygroundSessionPersistence: () => ({
    restoreSession: vi.fn(async () => false),
    sessionScopeReady: true,
    hasPersistedSession: false,
    persistedHistoryId: null,
    persistedServerChatId: null
  })
}))

vi.mock("@/hooks/playground-session-restore", () => ({
  shouldRestorePersistedPlaygroundSession: () => false
}))

vi.mock("@/services/app", () => ({
  webUIResumeLastChat: vi.fn(async () => false)
}))

vi.mock("@/db/dexie/helpers", () => ({
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
  CHAT_BACKGROUND_IMAGE_SETTING: "chatBackgroundImage"
}))

vi.mock("../Knowledge/utils/unsupported-types", () => ({
  otherUnsupportedTypes: []
}))

vi.mock("@/store/option", () => ({
  useStoreMessageOption: (
    selector: (state: typeof storeOptionState.value) => unknown
  ) => selector(storeOptionState.value)
}))

vi.mock("@/store/artifacts", () => ({
  useArtifactsStore: (selector: (state: typeof artifactsState.value) => unknown) =>
    selector(artifactsState.value)
}))

vi.mock("@/hooks/useSetting", () => ({
  useSetting: () => [""]
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (_key: string, defaultValue: unknown) => [defaultValue]
}))

vi.mock("@/hooks/useMediaQuery", () => ({
  useMobile: () => mobileViewportState.value,
  useDesktop: () => desktopViewportState.value
}))

vi.mock("@/services/chat-settings", () => ({
  syncChatSettingsForServerChat: (params: ChatSettingsSyncParams) =>
    chatSettingsState.syncChatSettingsForServerChat(params),
  applyChatSettingsPatch: (params: ChatSettingsPatchParams) =>
    chatSettingsState.applyChatSettingsPatch(params)
}))

vi.mock("@/hooks/useLoadLocalConversation", () => ({
  useLoadLocalConversation: () => loadLocalConversationMock
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
    useLocation: () => ({
      pathname: window.location.pathname || "/chat",
      search: window.location.search || "",
      hash: window.location.hash || "",
      state: null,
      key: "test-location"
    })
  }
})

describe("Playground thread search integration", () => {
  beforeEach(() => {
    vi.clearAllMocks()
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
    artifactsState.value.isOpen = false
    artifactsState.value.active = null
    artifactsState.value.history = []
    artifactsState.value.unreadCount = 0
    storeOptionState.value.compareParentByHistory = {}
    window.history.replaceState(null, "", "/chat")
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

  it("opens shortcut help from the header and closes with Escape", () => {
    render(<Playground />)

    fireEvent.click(screen.getByTestId("playground-shortcuts-help-trigger"))
    expect(
      screen.getByTestId("playground-shortcuts-help-panel")
    ).toBeInTheDocument()

    fireEvent.keyDown(window, { key: "Escape" })
    expect(
      screen.queryByTestId("playground-shortcuts-help-panel")
    ).not.toBeInTheDocument()
  })

  it("opens shortcut help when a global open-shortcuts event is dispatched", async () => {
    render(<Playground />)

    window.dispatchEvent(new CustomEvent("tldw:open-playground-shortcuts"))
    await waitFor(() => {
      expect(
        screen.getByTestId("playground-shortcuts-help-panel")
      ).toBeInTheDocument()
    })
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

  it("restores sidepanel WebUI handoff state from the URL fragment and clears the fragment", async () => {
    messageOptionState.value.historyId = "history-1"
    messageOptionState.value.serverChatId = null
    messageOptionState.value.selectedSystemPrompt = "stale-system"
    messageOptionState.value.selectedQuickPrompt = "stale-quick"

    const encodedHandoff = encodeSidepanelChatWebUiHandoff({
      source: SIDEPANEL_CHAT_WEBUI_HANDOFF_SOURCE,
      createdAt: Date.now(),
      draft: "continue from the sidepanel",
      historyId: "history-handoff",
      serverChatId: "server-handoff",
      selectedSystemPrompt: null,
      selectedQuickPrompt: "",
      chatMode: "rag",
      webSearch: true,
      toolChoice: "auto",
      temporaryChat: true,
      useOCR: true
    })
    const hashParams = new URLSearchParams()
    hashParams.set(SIDEPANEL_CHAT_WEBUI_HANDOFF_PARAM, encodedHandoff)
    window.history.replaceState(null, "", `/chat#${hashParams.toString()}`)
    expect(decodeSidepanelChatWebUiHandoff(encodedHandoff)).toMatchObject({
      historyId: "history-handoff"
    })

    render(<Playground />)

    await waitFor(
      () => {
        expect(loadLocalConversationMock).toHaveBeenCalledWith(
          "history-handoff"
        )
      },
      { timeout: 5_000 }
    )
    await waitFor(() => {
      expect(messageOptionState.value.setServerChatId).toHaveBeenCalledWith(
        "server-handoff"
      )
    })
    expect(
      messageOptionState.value.setSelectedSystemPrompt
    ).toHaveBeenCalledWith(null)
    expect(messageOptionState.value.setSelectedQuickPrompt).toHaveBeenCalledWith(
      null
    )
    expect(messageOptionState.value.setChatMode).toHaveBeenCalledWith("rag")
    expect(messageOptionState.value.setWebSearch).toHaveBeenCalledWith(true)
    expect(messageOptionState.value.setToolChoice).toHaveBeenCalledWith("auto")
    expect(messageOptionState.value.setTemporaryChat).toHaveBeenCalledWith(true)
    expect(messageOptionState.value.setUseOCR).toHaveBeenCalledWith(true)
    expect(window.location.hash).toBe("")
  })
})
