import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
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
  mockClearSetting,
  mockGetAllNoteKeywords,
  mockSearchNoteKeywords
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
    mockClearSetting: vi.fn(),
    mockGetAllNoteKeywords: vi.fn(),
    mockSearchNoteKeywords: vi.fn()
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
  getAllNoteKeywordStats: mockGetAllNoteKeywords,
  searchNoteKeywords: mockSearchNoteKeywords
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
  default: ({
    notes,
    onSelectNote
  }: {
    notes?: Array<{ id: string | number; title?: string }>
    onSelectNote: (id: string | number) => void
  }) => (
    <div data-testid="notes-list-panel">
      {(notes || []).map((note) => (
        <button
          key={String(note.id)}
          type="button"
          data-testid={`mock-note-${String(note.id)}`}
          onClick={() => onSelectNote(note.id)}
        >
          {note.title || String(note.id)}
        </button>
      ))}
    </div>
  )
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

const buildSearchResponse = (path: string) => {
  const url = new URL(`http://localhost${path}`)
  const query = (url.searchParams.get("query") || "").trim().toLowerCase()
  const tokens = url.searchParams
    .getAll("tokens")
    .map((token) => token.trim().toLowerCase())
    .filter(Boolean)

  if (query === "ml" && tokens.includes("research")) {
    return {
      items: [{ id: "c-1", title: "Combined result", content: "combined" }],
      pagination: { total_items: 3, total_pages: 1 }
    }
  }
  if (query === "ml") {
    return {
      items: [
        { id: "q-1", title: "Query result 1", content: "query one" },
        { id: "q-2", title: "Query result 2", content: "query two" }
      ],
      pagination: { total_items: 7, total_pages: 1 }
    }
  }
  if (tokens.includes("research")) {
    return {
      items: [{ id: "k-1", title: "Keyword result", content: "keyword only" }],
      pagination: { total_items: 4, total_pages: 1 }
    }
  }
  return { items: [], pagination: { total_items: 0, total_pages: 1 } }
}

describe("NotesManagerPage stage 13 navigation filter summary", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockConfirmDanger.mockResolvedValue(true)
    mockGetSetting.mockResolvedValue(null)
    mockSetSetting.mockResolvedValue(undefined)
    mockClearSetting.mockResolvedValue(undefined)
    mockGetAllNoteKeywords.mockResolvedValue([
      { keyword: "research", noteCount: 4 },
      { keyword: "ml", noteCount: 2 },
      { keyword: "planning", noteCount: 0 }
    ])
    mockSearchNoteKeywords.mockResolvedValue(["research", "ml", "planning"])
    mockBgRequest.mockImplementation(async (request: { path?: string; method?: string }) => {
      const path = String(request.path || "")
      const method = String(request.method || "GET").toUpperCase()
      if (path.startsWith("/api/v1/notes/?")) {
        return {
          items: [
            { id: "n1", title: "Alpha note", content: "alpha content", version: 1 },
            { id: "n2", title: "Beta note", content: "beta content", version: 1 },
            { id: "n3", title: "Gamma note", content: "gamma content", version: 1 }
          ],
          pagination: { total_items: 3, total_pages: 1 }
        }
      }
      if (path.startsWith("/api/v1/notes/search/?")) {
        return buildSearchResponse(path)
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

  it("renders query-only summary with accessible live-region metadata", async () => {
    renderPage()
    fireEvent.change(screen.getByPlaceholderText("Search notes... (use quotes for exact match)"), {
      target: { value: "ML" }
    })

    await waitFor(() => {
      expect(screen.getByTestId("notes-active-filter-summary")).toHaveTextContent(
        "Showing 2 of 7 notes"
      )
    })

    const summary = screen.getByTestId("notes-active-filter-summary")
    expect(summary).toHaveAttribute("role", "status")
    expect(summary).toHaveAttribute("aria-live", "polite")
    expect(summary).toHaveTextContent('Query: "ML"')
  })

  it("renders keyword-only summary after applying keyword picker selection", async () => {
    renderPage()
    fireEvent.click(screen.getByRole("button", { name: "Browse tags" }))
    await screen.findByText("Apply filters")
    fireEvent.click(screen.getByText(/^research\b/i))
    fireEvent.click(screen.getByRole("button", { name: "Apply filters" }))

    await waitFor(() => {
      expect(screen.getByTestId("notes-active-filter-summary")).toHaveTextContent(
        "Showing 1 of 4 notes"
      )
    })
    expect(screen.getByTestId("notes-active-filter-summary-details")).toHaveTextContent(
      "Tags: research"
    )
  })

  it("renders combined query + keyword summary and preserves clear action labeling", async () => {
    renderPage()
    fireEvent.change(screen.getByPlaceholderText("Search notes... (use quotes for exact match)"), {
      target: { value: "ML" }
    })
    await waitFor(() => {
      expect(screen.getByTestId("notes-active-filter-summary")).toHaveTextContent(
        "Showing 2 of 7 notes"
      )
    })

    fireEvent.click(screen.getByRole("button", { name: "Browse tags" }))
    await screen.findByText("Apply filters")
    fireEvent.click(screen.getByText(/^research\b/i))
    fireEvent.click(screen.getByRole("button", { name: "Apply filters" }))

    await waitFor(() => {
      expect(screen.getByTestId("notes-active-filter-summary")).toHaveTextContent(
        "Showing 1 of 3 notes"
      )
    })
    expect(screen.getByTestId("notes-active-filter-summary-details")).toHaveTextContent(
      'Query: "ML" + Tags: research'
    )
    expect(screen.getByLabelText("Clear active note filters")).toBeInTheDocument()
  }, 12000)
})
