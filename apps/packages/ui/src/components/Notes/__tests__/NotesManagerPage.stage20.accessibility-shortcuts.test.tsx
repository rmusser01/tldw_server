import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest"
import { ConfigProvider } from "antd"
import NotesManagerPage from "../NotesManagerPage"

const {
  mockBgRequest,
  mockMessageSuccess,
  mockMessageError,
  mockMessageWarning,
  mockNavigate,
  mockConfirmDanger,
  mockGetSetting,
  mockClearSetting,
  mockStorageGet,
  mockStorageSet,
  mockStorageRemove,
  mockStartTutorial
} = vi.hoisted(() => {
  return {
    mockBgRequest: vi.fn(),
    mockMessageSuccess: vi.fn(),
    mockMessageError: vi.fn(),
    mockMessageWarning: vi.fn(),
    mockNavigate: vi.fn(),
    mockConfirmDanger: vi.fn(),
    mockGetSetting: vi.fn(),
    mockClearSetting: vi.fn(),
    mockStorageGet: vi.fn(),
    mockStorageSet: vi.fn(),
    mockStorageRemove: vi.fn(),
    mockStartTutorial: vi.fn()
  }
})

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      defaultValueOrOptions?:
        | string
        | {
            defaultValue?: string
            [key: string]: unknown
          }
    ) => {
      if (typeof defaultValueOrOptions === "string") return defaultValueOrOptions
      if (defaultValueOrOptions?.defaultValue) return defaultValueOrOptions.defaultValue
      return key
    }
  })
}))

vi.mock("react-router-dom", () => ({
  useNavigate: () => mockNavigate
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: mockBgRequest
}))

vi.mock("@/hooks/useServerOnline", () => ({
  useServerOnline: () => true
}))

vi.mock("@/context/demo-mode", () => ({
  useDemoMode: () => ({ demoEnabled: false })
}))

vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => ({
    capabilities: { hasNotes: true },
    loading: false
  })
}))

vi.mock("@/components/Common/confirm-danger", () => ({
  useConfirmDanger: () => mockConfirmDanger
}))

vi.mock("@/hooks/useAntdMessage", () => ({
  useAntdMessage: () => ({
    success: mockMessageSuccess,
    error: mockMessageError,
    warning: mockMessageWarning,
    info: vi.fn()
  })
}))

vi.mock("@/services/note-keywords", () => ({
  getAllNoteKeywordStats: vi.fn(async () => []),
  searchNoteKeywords: vi.fn(async () => [])
}))

vi.mock("@/store/option", () => ({
  useStoreMessageOption: (selector: (state: Record<string, unknown>) => unknown) =>
    selector({
      setHistory: vi.fn(),
      setMessages: vi.fn(),
      setHistoryId: vi.fn(),
      setServerChatId: vi.fn(),
      setServerChatState: vi.fn(),
      setServerChatTopic: vi.fn(),
      setServerChatClusterId: vi.fn(),
      setServerChatSource: vi.fn(),
      setServerChatExternalRef: vi.fn()
    })
}))

vi.mock("@/services/settings/registry", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/services/settings/registry")>()
  return {
    ...actual,
    getSetting: mockGetSetting,
    clearSetting: mockClearSetting
  }
})

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: vi.fn(async () => undefined),
    getChat: vi.fn(async () => null),
    listChatMessages: vi.fn(async () => []),
    getCharacter: vi.fn(async () => null)
  }
}))

vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: () => ({
    get: mockStorageGet,
    set: mockStorageSet,
    remove: mockStorageRemove,
    watch: vi.fn(() => vi.fn()),
    unwatch: vi.fn()
  })
}))

// This suite qualifies Notes storage/shortcut behavior; actual source/AntD handoff has a dedicated integration test.
vi.mock("@/hooks/useStartTutorial", () => ({
  useStartTutorial:
    () =>
    (...args: unknown[]) => {
      mockStartTutorial(...args)
      return true
    }
}))

vi.mock("@/store/tutorials", () => {
  const store = Object.assign(() => ({}), {
    getState: () => ({
      startTutorial: mockStartTutorial
    })
  })
  return { useTutorialStore: store }
})

vi.mock("@/components/Notes/NotesListPanel", () => ({
  default: () => <div data-testid="notes-list-panel" />
}))

const renderPage = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false }
    }
  })
  return render(
    <QueryClientProvider client={queryClient}>
      <NotesManagerPage />
    </QueryClientProvider>
  )
}

describe("NotesManagerPage stage 20 accessibility shortcut discovery", () => {
  // The dialogs live in a lazily loaded overlay host; warm its chunk so the
  // first dialog lookup does not race the default findBy timeout.
  beforeAll(async () => {
    await import("../NotesManagerOverlays")
  }, 60_000)

  beforeEach(() => {
    vi.clearAllMocks()
    mockConfirmDanger.mockResolvedValue(true)
    mockGetSetting.mockResolvedValue(null)
    mockClearSetting.mockResolvedValue(undefined)
    mockStorageGet.mockResolvedValue(null)
    mockStorageSet.mockResolvedValue(undefined)
    mockStorageRemove.mockResolvedValue(undefined)

    mockBgRequest.mockImplementation(async (request: { path?: string }) => {
      const path = String(request.path || "")
      if (path.startsWith("/api/v1/notes/?")) {
        return {
          items: [],
          pagination: { total_items: 0, total_pages: 1 }
        }
      }
      return {}
    })
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it("adds shortcut summary semantics and opens shortcut help from the toolbar", async () => {
    renderPage()

    const editorRegion = screen.getByTestId("notes-editor-region")
    expect(editorRegion).toHaveAttribute("aria-describedby", "notes-shortcuts-summary")
    expect(document.getElementById("notes-shortcuts-summary")).toHaveTextContent(
      "Ctrl or Command plus S to save"
    )

    fireEvent.click(screen.getByTestId("notes-shortcuts-help-button"))

    expect(await screen.findByRole("dialog", {}, { timeout: 10_000 })).toBeInTheDocument()
    expect(await screen.findByTestId("notes-shortcuts-modal")).toHaveTextContent(
      "Ctrl/Cmd + S"
    )
    expect(screen.getByTestId("notes-shortcuts-modal")).toHaveTextContent(
      "Alt + Shift + D"
    )
  })

  it("ignores '?' while typing in the editor and opens help from global context", async () => {
    renderPage()

    const textarea = screen.getByLabelText("Note content")
    fireEvent.keyDown(textarea, { key: "?", shiftKey: true })

    expect(screen.queryByTestId("notes-shortcuts-modal")).not.toBeInTheDocument()

    fireEvent.keyDown(window, { key: "?", shiftKey: true })

    expect(await screen.findByRole("dialog", {}, { timeout: 10_000 })).toBeInTheDocument()
    expect(await screen.findByTestId("notes-shortcuts-modal")).toBeInTheDocument()
  })

  it("ignores Ctrl/Cmd+K and Alt+N while typing, but still focuses search globally", async () => {
    vi.useFakeTimers()
    renderPage()

    const textarea = screen.getByLabelText("Note content")
    const searchInput = screen.getByPlaceholderText(
      "Search notes... (use quotes for exact match)"
    )

    textarea.focus()
    fireEvent.keyDown(textarea, { key: "k", ctrlKey: true })
    expect(textarea).toHaveFocus()

    fireEvent.keyDown(textarea, { key: "N", altKey: true })
    await act(async () => {
      vi.runOnlyPendingTimers()
    })
    expect(textarea).toHaveFocus()

    fireEvent.keyDown(window, { key: "k", ctrlKey: true })
    expect(searchInput).toHaveFocus()
  })

  it("persists tutorial state through shared storage on first visit", async () => {
    vi.useFakeTimers()
    renderPage()

    await act(async () => {
      await Promise.resolve()
    })
    expect(mockStorageGet).toHaveBeenCalledWith("notes-tutorial-shown")

    await act(async () => {
      vi.advanceTimersByTime(1000)
      await Promise.resolve()
      await Promise.resolve()
    })

    expect(mockStorageSet).toHaveBeenCalledWith("notes-tutorial-shown", "1")
    expect(mockStartTutorial).toHaveBeenCalledWith("notes-basics")
  })

  it("suppresses the first-visit tutorial when the user starts interacting before the timer fires", async () => {
    vi.useFakeTimers()
    renderPage()

    await act(async () => {
      await Promise.resolve()
    })

    fireEvent.pointerDown(window)

    await act(async () => {
      vi.advanceTimersByTime(1000)
      await Promise.resolve()
      await Promise.resolve()
    })

    expect(mockStorageSet).toHaveBeenCalledWith("notes-tutorial-shown", "1")
    expect(mockStartTutorial).not.toHaveBeenCalled()
  })

  it("removes tutorial state from shared storage when restarting the tutorial", async () => {
    renderPage()

    const opener = screen.getByTestId("notes-shortcuts-help-button")
    opener.focus()
    fireEvent.click(opener)
    fireEvent.click(await screen.findByTestId("notes-restart-tutorial"))

    await waitFor(() => {
      expect(mockStorageRemove).toHaveBeenCalledWith("notes-tutorial-shown")
    })
    expect(mockStartTutorial).toHaveBeenCalledWith("notes-basics", opener)
  })
  // AX-09 (#3122): closing the shortcuts dialog must return focus to the
  // control that opened it, not drop it on <body>.
  it("returns focus to the shortcuts help button after the dialog closes", async () => {
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false }, mutations: { retry: false } }
    })
    // Without motion the dialog closes in the same pass, as it does once the
    // browser's close animation ends.
    render(
      <QueryClientProvider client={queryClient}>
        <ConfigProvider theme={{ token: { motion: false } }}>
          <NotesManagerPage />
        </ConfigProvider>
      </QueryClientProvider>
    )

    const opener = screen.getByTestId("notes-shortcuts-help-button")
    opener.focus()
    fireEvent.click(opener)

    const dialog = await screen.findByRole("dialog", {}, { timeout: 10_000 })
    await screen.findByTestId("notes-shortcuts-modal")
    // A keyboard user is inside the dialog when they close it.
    const closeButton = within(dialog).getByRole("button", { name: /close/i })
    closeButton.focus()
    expect(closeButton).toHaveFocus()
    fireEvent.click(closeButton)

    await waitFor(() => {
      expect(screen.queryByTestId("notes-shortcuts-modal")).not.toBeInTheDocument()
    })
    await waitFor(() => {
      expect(opener).toHaveFocus()
    })
  })
})
