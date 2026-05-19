// @vitest-environment jsdom
import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { Playground } from "../Playground"

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
    setHistoryId: vi.fn(),
    setHistory: vi.fn(),
    setMessages: vi.fn(),
    setSelectedSystemPrompt: vi.fn(),
    setSelectedModel: vi.fn(),
    setServerChatId: vi.fn(),
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
  useMobile: () => mobileViewportState.value
}))

vi.mock("@/hooks/useLoadLocalConversation", () => ({
  useLoadLocalConversation: () => vi.fn(async () => {})
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
    artifactsState.value.isOpen = false
    storeOptionState.value.compareParentByHistory = {}
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
})
