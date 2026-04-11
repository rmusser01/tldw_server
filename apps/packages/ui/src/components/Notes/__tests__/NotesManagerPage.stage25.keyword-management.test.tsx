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

describe("NotesManagerPage stage 25 keyword management", () => {
  let keywordRows: Array<{ id: number; keyword: string; version: number; note_count: number }>

  beforeEach(() => {
    vi.clearAllMocks()
    mockConfirmDanger.mockResolvedValue(true)
    mockGetSetting.mockResolvedValue(null)
    mockSetSetting.mockResolvedValue(undefined)
    mockClearSetting.mockResolvedValue(undefined)
    mockSearchNoteKeywords.mockResolvedValue(["alpha", "beta", "gamma"])

    keywordRows = [
      { id: 1, keyword: "alpha", version: 1, note_count: 3 },
      { id: 2, keyword: "beta", version: 1, note_count: 2 },
      { id: 3, keyword: "gamma", version: 1, note_count: 1 }
    ]

    mockGetAllNoteKeywordStats.mockImplementation(async () =>
      keywordRows.map((row) => ({
        keyword: row.keyword,
        noteCount: row.note_count
      }))
    )

    mockBgRequest.mockImplementation(
      async (request: {
        path?: string
        method?: string
        body?: Record<string, any>
        headers?: Record<string, string>
      }) => {
        const path = String(request.path || "")
        const method = String(request.method || "GET").toUpperCase()
        const headers = request.headers || {}

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
        if (path.startsWith("/api/v1/notes/keywords/?") && method === "GET") {
          return keywordRows.map((row) => ({
            id: row.id,
            keyword: row.keyword,
            version: row.version,
            note_count: row.note_count
          }))
        }

        const keywordIdMatch = path.match(/^\/api\/v1\/notes\/keywords\/(\d+)$/)
        if (keywordIdMatch && method === "PATCH") {
          const keywordId = Number(keywordIdMatch[1])
          const expectedVersion = Number(headers["expected-version"] || 0)
          const row = keywordRows.find((item) => item.id === keywordId)
          if (!row || row.version !== expectedVersion) {
            throw { status: 409, message: "Keyword conflict" }
          }
          row.keyword = String(request.body?.keyword || row.keyword)
          row.version += 1
          return {
            id: row.id,
            keyword: row.keyword,
            version: row.version,
            note_count: row.note_count
          }
        }

        const keywordMergeMatch = path.match(/^\/api\/v1\/notes\/keywords\/(\d+)\/merge$/)
        if (keywordMergeMatch && method === "POST") {
          const sourceKeywordId = Number(keywordMergeMatch[1])
          const targetKeywordId = Number(request.body?.target_keyword_id || 0)
          const source = keywordRows.find((item) => item.id === sourceKeywordId)
          const target = keywordRows.find((item) => item.id === targetKeywordId)
          if (!source || !target) {
            throw { status: 404, message: "Keyword not found" }
          }
          const expectedVersion = Number(headers["expected-version"] || 0)
          if (source.version !== expectedVersion) {
            throw { status: 409, message: "Keyword merge conflict" }
          }
          target.note_count += source.note_count
          keywordRows = keywordRows.filter((item) => item.id !== sourceKeywordId)
          return {
            source_keyword_id: sourceKeywordId,
            target_keyword_id: targetKeywordId,
            source_deleted_version: source.version + 1,
            target_version: target.version,
            merged_note_links: Math.max(1, source.note_count),
            merged_conversation_links: 0,
            merged_collection_links: 0,
            merged_flashcard_links: 0
          }
        }

        if (keywordIdMatch && method === "DELETE") {
          const keywordId = Number(keywordIdMatch[1])
          const expectedVersion = Number(headers["expected-version"] || 0)
          const row = keywordRows.find((item) => item.id === keywordId)
          if (!row || row.version !== expectedVersion) {
            throw { status: 409, message: "Keyword delete conflict" }
          }
          keywordRows = keywordRows.filter((item) => item.id !== keywordId)
          return {}
        }

        return {}
      }
    )
  })

  it("opens keyword manager from the browse-keywords modal", async () => {
    renderPage()

    fireEvent.click(screen.getByRole("button", { name: "Browse keywords" }))
    const managerButton = await screen.findByTestId("notes-keyword-picker-open-manager")
    fireEvent.click(managerButton)

    await waitFor(() => {
      expect(screen.getByTestId("notes-keyword-manager-modal")).toBeInTheDocument()
    })
    expect(screen.getByTestId("notes-keyword-manager-item-1")).toBeInTheDocument()
    expect(screen.getByTestId("notes-keyword-manager-item-2")).toBeInTheDocument()
  })

  it("supports rename, merge, and delete workflows from keyword manager", async () => {
    renderPage()

    fireEvent.click(screen.getByRole("button", { name: "Browse keywords" }))
    fireEvent.click(await screen.findByTestId("notes-keyword-picker-open-manager"))

    await waitFor(() => {
      expect(screen.getByTestId("notes-keyword-manager-modal")).toBeInTheDocument()
    })

    fireEvent.click(screen.getByTestId("notes-keyword-manager-rename-1"))
    const renameInput = await screen.findByTestId("notes-keyword-manager-rename-input")
    fireEvent.change(renameInput, { target: { value: "alpha-renamed" } })

    const renameButtons = screen.getAllByRole("button", { name: "Rename" })
    fireEvent.click(renameButtons[renameButtons.length - 1])

    await waitFor(() => {
      expect(mockMessageSuccess).toHaveBeenCalledWith('Renamed keyword to "alpha-renamed"')
    })

    fireEvent.click(screen.getByTestId("notes-keyword-manager-merge-2"))
    const mergeTargetSelect = await screen.findByTestId("notes-keyword-manager-merge-target")
    fireEvent.change(mergeTargetSelect, { target: { value: "1" } })

    const mergeButtons = screen.getAllByRole("button", { name: "Merge" })
    fireEvent.click(mergeButtons[mergeButtons.length - 1])

    await waitFor(() => {
      expect(mockMessageSuccess).toHaveBeenCalledWith('Merged "beta" into "alpha-renamed"')
    })

    fireEvent.click(screen.getByTestId("notes-keyword-manager-delete-1"))
    await waitFor(() => {
      expect(mockMessageSuccess).toHaveBeenCalledWith('Deleted keyword "alpha-renamed"')
    })

    const requestCalls = mockBgRequest.mock.calls.map((call) => call[0])
    expect(
      requestCalls.some(
        (req: any) => req.method === "PATCH" && String(req.path).startsWith("/api/v1/notes/keywords/1")
      )
    ).toBe(true)
    expect(
      requestCalls.some(
        (req: any) =>
          req.method === "POST" && String(req.path).startsWith("/api/v1/notes/keywords/2/merge")
      )
    ).toBe(true)
    expect(
      requestCalls.some(
        (req: any) => req.method === "DELETE" && String(req.path).startsWith("/api/v1/notes/keywords/1")
      )
    ).toBe(true)
  })
})

