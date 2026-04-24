import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
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
  mockGetAllNoteKeywordStats,
  mockSearchNoteKeywords
} = vi.hoisted(() => ({
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
  mockGetAllNoteKeywordStats: vi.fn(),
  mockSearchNoteKeywords: vi.fn()
}))

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
  getAllNoteKeywordStats: mockGetAllNoteKeywordStats,
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

const readKeywordOptionOrder = () => {
  const nodes = document.querySelectorAll<HTMLInputElement>(
    '[data-testid^="notes-keyword-picker-option-"] input[type="checkbox"], input[type="checkbox"][data-testid^="notes-keyword-picker-option-"]'
  )
  return Array.from(nodes)
    .map((node) => (node.closest("label")?.textContent || "").trim())
    .filter(Boolean)
}

describe("NotesManagerPage stage 24 keyword picker prioritization", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockConfirmDanger.mockResolvedValue(true)
    mockGetSetting.mockResolvedValue(null)
    mockSetSetting.mockResolvedValue(undefined)
    mockClearSetting.mockResolvedValue(undefined)
    mockGetAllNoteKeywordStats.mockResolvedValue([
      { keyword: "alpha", noteCount: 2 },
      { keyword: "zeta", noteCount: 9 },
      { keyword: "beta", noteCount: 5 }
    ])
    mockSearchNoteKeywords.mockResolvedValue(["alpha", "zeta", "beta"])

    mockBgRequest.mockImplementation(async (request: { path?: string; method?: string }) => {
      const path = String(request.path || "")
      const method = String(request.method || "GET").toUpperCase()
      if (path.startsWith("/api/v1/notes/?")) {
        return { items: [], pagination: { total_items: 0, total_pages: 1 } }
      }
      if (path === "/api/v1/admin/notes/title-settings" && method === "GET") {
        return {
          llm_enabled: false,
          default_strategy: "heuristic",
          effective_strategy: "heuristic",
          strategies: ["heuristic"]
        }
      }
      return {}
    })
  })

  it("defaults to frequency sorting and supports alphabetical sort toggles", async () => {
    renderPage()

    fireEvent.click(screen.getByRole("button", { name: "Browse tags" }))
    await waitFor(() => {
      expect(screen.getByTestId("notes-keyword-picker-sort-select")).toBeInTheDocument()
    })

    expect(readKeywordOptionOrder().slice(0, 3)).toEqual([
      "zeta (9)",
      "beta (5)",
      "alpha (2)"
    ])

    fireEvent.change(screen.getByTestId("notes-keyword-picker-sort-select"), {
      target: { value: "alpha_asc" }
    })

    await waitFor(() => {
      expect(readKeywordOptionOrder().slice(0, 3)).toEqual([
        "alpha (2)",
        "beta (5)",
        "zeta (9)"
      ])
    })
  })

  it("surfaces recently used keywords at the top of the picker", async () => {
    renderPage()

    fireEvent.click(screen.getByRole("button", { name: "Browse tags" }))
    await waitFor(() => {
      expect(screen.getByText("Apply filters")).toBeInTheDocument()
    })

    fireEvent.click(screen.getByText("beta (5)"))
    fireEvent.click(screen.getByText("alpha (2)"))
    fireEvent.click(screen.getByText("Apply filters"))

    fireEvent.click(screen.getByRole("button", { name: "Browse tags" }))
    const recentSection = await screen.findByTestId("notes-keyword-picker-recent-section")
    const recentButtons = within(recentSection).getAllByRole("button")

    expect(recentButtons.map((button) => button.textContent?.trim())).toContain("beta (5)")
    expect(recentButtons.map((button) => button.textContent?.trim())).toContain("alpha (2)")
  })
})
