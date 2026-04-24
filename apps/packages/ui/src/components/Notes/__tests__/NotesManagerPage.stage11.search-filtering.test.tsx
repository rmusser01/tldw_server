import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import NotesManagerPage from "../NotesManagerPage"

const {
  mockBgRequest,
  mockMessageSuccess,
  mockMessageError,
  mockMessageWarning,
  mockMessageInfo,
  mockNavigate,
  mockConfirmDanger,
  mockGetSetting,
  mockSetSetting,
  mockClearSetting
} = vi.hoisted(() => {
  return {
    mockBgRequest: vi.fn(),
    mockMessageSuccess: vi.fn(),
    mockMessageError: vi.fn(),
    mockMessageWarning: vi.fn(),
    mockMessageInfo: vi.fn(),
    mockNavigate: vi.fn(),
    mockConfirmDanger: vi.fn(),
    mockGetSetting: vi.fn(),
    mockSetSetting: vi.fn(),
    mockClearSetting: vi.fn()
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
    info: mockMessageInfo
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
    setSetting: mockSetSetting,
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

vi.mock("@/components/Common/MarkdownPreview", () => ({
  MarkdownPreview: ({ content }: { content: string }) => (
    <div data-testid="markdown-preview-content">{content}</div>
  )
}))

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
const isNotesSearchPath = (request: { path?: string } | undefined) =>
  /\/api\/v1\/notes\/search\/?\?/.test(String(request?.path || ""))
const getSearchCalls = () =>
  mockBgRequest.mock.calls.filter(([request]) =>
    isNotesSearchPath(request as { path?: string })
  )

describe("NotesManagerPage stage 11 search filtering", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockConfirmDanger.mockResolvedValue(true)
    mockGetSetting.mockResolvedValue(null)
    mockSetSetting.mockResolvedValue(undefined)
    mockClearSetting.mockResolvedValue(undefined)
    mockBgRequest.mockImplementation(async (request: { path?: string; method?: string }) => {
      const path = String(request.path || "")
      const method = String(request.method || "GET").toUpperCase()
      if (path.startsWith("/api/v1/notes/?")) {
        return { items: [], pagination: { total_items: 0, total_pages: 1 } }
      }
      if (isNotesSearchPath(request)) {
        return { items: [], pagination: { total_items: 0, total_pages: 1 } }
      }
      if (path === "/api/v1/admin/notes/title-settings" && method === "GET") {
        return {
          llm_enabled: false,
          default_strategy: "heuristic",
          effective_strategy: "heuristic",
          strategies: ["heuristic", "llm", "llm_fallback"]
        }
      }
      return {}
    })
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it("shows search input", () => {
    renderPage()
    expect(screen.getByPlaceholderText("Search notes... (use quotes for exact match)")).toBeInTheDocument()
  })

  it("opens search tips popover with phrase and AND guidance", async () => {
    renderPage()
    fireEvent.click(screen.getByTestId("notes-search-tips-button"))

    await waitFor(() => {
      expect(
        screen.getByText('Use quotes for phrases, e.g. "project roadmap".')
      ).toBeInTheDocument()
    })
    expect(
      screen.getByText("Text query + selected tags are combined with AND.")
    ).toBeInTheDocument()
  })

  it("filters search tips by text inside the popover", async () => {
    renderPage()
    fireEvent.click(screen.getByTestId("notes-search-tips-button"))

    const filterInput = await screen.findByTestId("notes-search-tips-filter")
    fireEvent.change(filterInput, { target: { value: "prefix" } })

    await waitFor(() => {
      expect(
        screen.getByText("Use prefix terms (like analy*) for broader matches.")
      ).toBeInTheDocument()
    })
    expect(
      screen.queryByText('Use quotes for phrases, e.g. "project roadmap".')
    ).not.toBeInTheDocument()
  })

  it("debounces rapid typing and issues only one final search request", async () => {
    vi.useFakeTimers()
    renderPage()
    const input = screen.getByPlaceholderText("Search notes... (use quotes for exact match)")

    await act(async () => {
      fireEvent.change(input, { target: { value: "a" } })
      await Promise.resolve()
    })
    await act(async () => {
      fireEvent.change(input, { target: { value: "al" } })
      await Promise.resolve()
    })
    await act(async () => {
      fireEvent.change(input, { target: { value: "alpha" } })
      await Promise.resolve()
    })

    expect(getSearchCalls()).toHaveLength(0)

    await act(async () => {
      vi.advanceTimersByTime(349)
      await Promise.resolve()
    })
    expect(getSearchCalls()).toHaveLength(0)

    await act(async () => {
      vi.advanceTimersByTime(1)
      await Promise.resolve()
      await Promise.resolve()
    })

    expect(getSearchCalls()).toHaveLength(1)

    const finalSearchCall = getSearchCalls()[0]
    expect(String(finalSearchCall?.[0]?.path || "")).toContain("query=alpha")
  })

  it("sends search requests for search queries", async () => {
    vi.useFakeTimers()
    renderPage()
    const input = screen.getByPlaceholderText("Search notes... (use quotes for exact match)")

    fireEvent.change(input, { target: { value: "alpha" } })
    await act(async () => {
      vi.advanceTimersByTime(350)
      await Promise.resolve()
      await Promise.resolve()
    })
    expect(getSearchCalls()).toHaveLength(1)

    fireEvent.change(input, { target: { value: "alpha beta" } })
    await act(async () => {
      vi.advanceTimersByTime(350)
      await Promise.resolve()
      await Promise.resolve()
    })
    expect(getSearchCalls()).toHaveLength(2)
  })
})
