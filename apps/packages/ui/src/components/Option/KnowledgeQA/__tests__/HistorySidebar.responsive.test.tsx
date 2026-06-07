import { act, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { HistorySidebar } from "../HistorySidebar"

const messageOpenMock = vi.fn()

const state = {
  isMobile: false,
  historySidebarOpen: true,
  historyHydrated: true,
  currentThreadId: null as string | null,
  searchHistory: [] as Array<{
    id: string
    query: string
    timestamp: string
    keywords: string[]
    sourcesCount: number
    hasAnswer: boolean
    answerPreview?: string
    pinned?: boolean
    conversationId?: string
    trustState?:
      | "cited_answer"
      | "uncited_degraded_answer"
      | "no_answer_insufficient_evidence"
      | "no_results"
      | "failed_search"
      | "unsynced_local_result"
      | "unknown_trust"
  }>,
  setHistorySidebarOpen: vi.fn(),
  restoreFromHistory: vi.fn(),
  deleteHistoryItem: vi.fn(),
  toggleHistoryPin: vi.fn(),
  setSettingsPanelOpen: vi.fn(),
}

vi.mock("@/hooks/useMediaQuery", () => ({
  useMobile: () => state.isMobile,
}))

vi.mock("@/hooks/useAntdMessage", () => ({
  useAntdMessage: () => ({
    open: messageOpenMock,
  }),
}))

vi.mock("../KnowledgeQAProvider", () => ({
  useKnowledgeQA: () => ({
    searchHistory: state.searchHistory,
    historyHydrated: state.historyHydrated,
    currentThreadId: state.currentThreadId,
    historySidebarOpen: state.historySidebarOpen,
    setHistorySidebarOpen: state.setHistorySidebarOpen,
    restoreFromHistory: state.restoreFromHistory,
    deleteHistoryItem: state.deleteHistoryItem,
    toggleHistoryPin: state.toggleHistoryPin,
    preset: "balanced",
    setSettingsPanelOpen: state.setSettingsPanelOpen,
  }),
}))

describe("HistorySidebar responsive layout", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    state.isMobile = false
    state.historySidebarOpen = true
    state.historyHydrated = true
    state.currentThreadId = null
    state.searchHistory = []
    localStorage.clear()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it("renders desktop sidebar as fixed-width panel when open", () => {
    render(<HistorySidebar />)
    expect(screen.getByTestId("knowledge-history-desktop-open")).toBeInTheDocument()
  })

  it("uses mobile overlay drawer when history is open on mobile", () => {
    state.isMobile = true
    state.historySidebarOpen = true

    render(<HistorySidebar />)
    expect(screen.getByTestId("knowledge-history-mobile-overlay")).toBeInTheDocument()
    expect(
      screen.queryByTestId("knowledge-history-desktop-open")
    ).not.toBeInTheDocument()
  })

  it("shows mobile open button when history is collapsed on mobile", () => {
    state.isMobile = true
    state.historySidebarOpen = false

    render(<HistorySidebar />)
    const openButton = screen.getByTestId("knowledge-history-mobile-open")
    expect(openButton).toBeInTheDocument()
    expect(openButton.className).toContain("fixed")
    expect(openButton.className).toContain("env(safe-area-inset-top)")
    expect(
      screen.queryByTestId("knowledge-history-mobile-overlay")
    ).not.toBeInTheDocument()
  })

  it("keeps the history skeleton visible until provider hydration completes", () => {
    state.searchHistory = []
    state.historyHydrated = false

    const { rerender } = render(<HistorySidebar />)
    expect(screen.getByLabelText("Loading search history")).toBeInTheDocument()
    expect(screen.queryByText("No search history yet")).not.toBeInTheDocument()

    state.historyHydrated = true
    rerender(<HistorySidebar />)

    expect(screen.queryByLabelText("Loading search history")).not.toBeInTheDocument()
    expect(screen.getByText("No search history yet")).toBeInTheDocument()
  })

  it("restores a history item when selected", async () => {
    state.searchHistory = [
      {
        id: "h1",
        query: "How does retrieval ranking work?",
        timestamp: new Date().toISOString(),
        keywords: ["__knowledge_QA__"],
        sourcesCount: 2,
        hasAnswer: true,
      },
    ]

    const { findByText } = render(<HistorySidebar />)
    const item = await findByText("How does retrieval ranking work?")
    item.click()

    expect(state.restoreFromHistory).toHaveBeenCalledTimes(1)
    expect(state.restoreFromHistory).toHaveBeenCalledWith(state.searchHistory[0])
  })

  it("keeps the delete affordance discoverable for keyboard users", async () => {
    state.searchHistory = [
      {
        id: "h2",
        query: "Accessibility checks",
        timestamp: new Date().toISOString(),
        keywords: ["__knowledge_QA__"],
        sourcesCount: 1,
        hasAnswer: true,
      },
    ]

    render(<HistorySidebar />)
    const deleteButton = await screen.findByLabelText("Delete from history")
    expect(deleteButton.className).toContain("focus-visible:opacity-100")
    expect(deleteButton.className).toContain("group-focus-within:opacity-100")
  })

  it("keeps delete actions visible on touch/mobile without hover dependency", async () => {
    state.isMobile = true
    state.historySidebarOpen = true
    state.searchHistory = [
      {
        id: "h2-mobile",
        query: "Touch delete visibility",
        timestamp: new Date().toISOString(),
        keywords: ["__knowledge_QA__"],
        sourcesCount: 1,
        hasAnswer: true,
      },
    ]

    render(<HistorySidebar />)
    const deleteButton = await screen.findByLabelText("Delete from history")
    expect(deleteButton.className).toContain("opacity-100")
    expect(deleteButton.className).not.toContain("group-hover:opacity-100")
  })

  it("shows compact trust labels for history items with trust state", async () => {
    state.searchHistory = [
      {
        id: "h-trust",
        query: "Trust-labeled query",
        timestamp: new Date().toISOString(),
        keywords: ["__knowledge_QA__"],
        sourcesCount: 2,
        hasAnswer: true,
        trustState: "uncited_degraded_answer",
      },
    ]

    render(<HistorySidebar />)

    expect(await screen.findByText("Uncited answer")).toBeInTheDocument()
  })

  it("marks the active history thread with aria-current", async () => {
    state.currentThreadId = "thread-123"
    state.searchHistory = [
      {
        id: "h3",
        query: "Current thread query",
        timestamp: new Date().toISOString(),
        keywords: ["__knowledge_QA__"],
        sourcesCount: 3,
        hasAnswer: true,
        conversationId: "thread-123",
      },
    ]

    render(<HistorySidebar />)
    const activeButton = await screen.findByRole("button", {
      name: /Current thread query/i,
    })
    expect(activeButton).toHaveAttribute("aria-current", "true")
  })

  it("filters history items by query and answer preview", async () => {
    state.searchHistory = [
      {
        id: "h4",
        query: "Timeline extraction",
        timestamp: new Date().toISOString(),
        keywords: ["__knowledge_QA__"],
        sourcesCount: 2,
        hasAnswer: true,
      },
      {
        id: "h5",
        query: "Different topic",
        answerPreview: "Mentions citation confidence",
        timestamp: new Date().toISOString(),
        keywords: ["__knowledge_QA__"],
        sourcesCount: 4,
        hasAnswer: true,
      },
    ]

    render(<HistorySidebar />)
    fireEvent.change(screen.getByLabelText("Filter history"), {
      target: { value: "citation confidence" },
    })

    expect(screen.queryByText("Timeline extraction")).not.toBeInTheDocument()
    expect(screen.getByText("Different topic")).toBeInTheDocument()
  })

  it("shows pinned history group and toggles pin action", async () => {
    state.searchHistory = [
      {
        id: "h6",
        query: "Pinned query",
        pinned: true,
        timestamp: new Date().toISOString(),
        keywords: ["__knowledge_QA__"],
        sourcesCount: 1,
        hasAnswer: true,
      },
      {
        id: "h7",
        query: "Regular query",
        timestamp: new Date().toISOString(),
        keywords: ["__knowledge_QA__"],
        sourcesCount: 1,
        hasAnswer: true,
      },
    ]

    render(<HistorySidebar />)
    expect(screen.getByText("Pinned")).toBeInTheDocument()

    fireEvent.click(screen.getByLabelText("Unpin history item"))
    expect(state.toggleHistoryPin).toHaveBeenCalledWith("h6")
  })

  it("keeps two-click delete confirmation behavior", async () => {
    state.searchHistory = [
      {
        id: "h8",
        query: "Delete candidate",
        timestamp: new Date().toISOString(),
        keywords: ["__knowledge_QA__"],
        sourcesCount: 1,
        hasAnswer: false,
      },
    ]

    render(<HistorySidebar />)
    const deleteButton = await screen.findByLabelText("Delete from history")
    fireEvent.click(deleteButton)

    expect(screen.getByLabelText("Click again to confirm deletion")).toBeInTheDocument()

    fireEvent.click(screen.getByLabelText("Click again to confirm deletion"))
    expect(state.deleteHistoryItem).toHaveBeenCalledWith("h8")
  })

  it("exports all history entries", async () => {
    const createObjectURL = vi.fn(() => "blob:history-export")
    const revokeObjectURL = vi.fn()
    Object.defineProperty(URL, "createObjectURL", {
      writable: true,
      value: createObjectURL,
    })
    Object.defineProperty(URL, "revokeObjectURL", {
      writable: true,
      value: revokeObjectURL,
    })

    state.searchHistory = [
      {
        id: "h9",
        query: "Export query",
        answerPreview: "Export preview",
        timestamp: new Date().toISOString(),
        keywords: ["__knowledge_QA__"],
        sourcesCount: 3,
        hasAnswer: true,
      },
    ]

    const clickSpy = vi
      .spyOn(HTMLAnchorElement.prototype, "click")
      .mockImplementation(() => {})

    render(<HistorySidebar />)
    fireEvent.click(screen.getByRole("button", { name: "Export All" }))

    expect(createObjectURL).toHaveBeenCalledTimes(1)
    expect(revokeObjectURL).toHaveBeenCalledTimes(1)
    expect(clickSpy).toHaveBeenCalledTimes(1)
    expect(messageOpenMock).toHaveBeenCalledWith(
      expect.objectContaining({ type: "success" })
    )

    clickSpy.mockRestore()
  })

  it("dismisses the collapsed-sidebar hint without crashing when storage writes are blocked", () => {
    vi.useFakeTimers()
    state.historySidebarOpen = false

    const setItemSpy = vi
      .spyOn(Storage.prototype, "setItem")
      .mockImplementation(() => {
        throw new DOMException("Blocked", "SecurityError")
      })

    try {
      render(<HistorySidebar />)
      expect(screen.getByRole("status")).toHaveTextContent("Expand history sidebar")

      expect(() => {
        act(() => {
          vi.advanceTimersByTime(5000)
        })
      }).not.toThrow()

      expect(screen.queryByRole("status")).not.toBeInTheDocument()
    } finally {
      setItemSpy.mockRestore()
    }
  })
})
