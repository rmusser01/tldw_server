// @vitest-environment jsdom
import React from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { render, waitFor } from "@testing-library/react"

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
    sessionScopeReady: true,
    hasPersistedSession: false,
    persistedHistoryId: null as string | null,
    persistedServerChatId: null as string | null
  }
}))

const restoreDecisionState = vi.hoisted(() => ({
  value: false
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

vi.mock("@/hooks/playground-session-restore", () => ({
  shouldRestorePersistedPlaygroundSession: () => restoreDecisionState.value
}))

vi.mock("@/services/app", () => ({
  webUIResumeLastChat: vi.fn(async () => false)
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
  CHAT_BACKGROUND_IMAGE_SETTING: "chatBackgroundImage"
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
  useSetting: () => [""]
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (_key: string, defaultValue: unknown) => [defaultValue]
}))

vi.mock("@/hooks/useMediaQuery", () => ({
  useMobile: () => false
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
    sessionPersistenceState.value.restoreSession = vi.fn(async () => false)
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
})
