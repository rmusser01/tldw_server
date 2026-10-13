/**
 * NL-01 (#3103): the /notes list must page through the whole library.
 *
 * The bgRequest mock mirrors the FastAPI notes list endpoint
 * (tldw_Server_API/app/api/v1/endpoints/notes.py `list_notes`): it honours only
 * `limit` (default 100) / `offset` and the whitelisted `sort_by`/`sort_order`,
 * ignores unknown params such as `page`/`results_per_page`, and reports the
 * library size as `total` and `pagination.total` (never `total_items`).
 * The real NotesListPanel renders, so the footer text is asserted end to end.
 */
import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import NotesManagerPage from "../NotesManagerPage"

// The list query is gated on a verified notes authority scope; pin one owner.
vi.mock("@/components/Notes/hooks/useNotesGraphAuthorityScope", async (importOriginal) => {
  const actual = await importOriginal<
    typeof import("@/components/Notes/hooks/useNotesGraphAuthorityScope")
  >()
  return { ...actual, useNotesGraphAuthorityScope: () => "notes-list-contract-owner" }
})

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
  mockClearSetting: vi.fn()
}))

const interpolate = (template: string, values: Record<string, unknown>) =>
  template.replace(/\{\{(\w+)\}\}/g, (match, name: string) =>
    name in values ? String(values[name]) : match
  )

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      defaultValueOrOptions?: string | { defaultValue?: string; [key: string]: unknown }
    ) => {
      if (typeof defaultValueOrOptions === "string") return defaultValueOrOptions
      if (defaultValueOrOptions?.defaultValue) {
        return interpolate(defaultValueOrOptions.defaultValue, defaultValueOrOptions)
      }
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

type ServerNote = {
  id: string
  title: string
  content: string
  version: number
  created_at: string
  last_modified: string
}

/** `note-1` is the most recently modified note, `note-N` the oldest. */
const makeServerNotes = (count: number): ServerNote[] =>
  Array.from({ length: count }, (_, index) => {
    const timestamp = new Date(Date.UTC(2026, 0, 1) - index * 60_000).toISOString()
    return {
      id: `note-${index + 1}`,
      title: `Note ${index + 1}`,
      content: `Body ${index + 1}`,
      version: 1,
      created_at: timestamp,
      last_modified: timestamp
    }
  })

const SORT_COLUMNS = new Set(["last_modified", "created_at", "title"])
const SORT_ORDERS = new Set(["asc", "desc"])

const isBrowseListPath = (path: string) => /^\/api\/v1\/notes\/?\?/.test(path)

/** Mirrors `list_notes` in tldw_Server_API/app/api/v1/endpoints/notes.py. */
const serveNotesList = (path: string, notes: ServerNote[]) => {
  const params = new URL(`https://server.test${path}`).searchParams
  const limit = Math.min(Number(params.get("limit") ?? 100), 1000)
  const offset = Number(params.get("offset") ?? 0)
  const sortBy = params.get("sort_by") ?? "last_modified"
  const sortOrder = params.get("sort_order") ?? "desc"
  if (!SORT_COLUMNS.has(sortBy) || !SORT_ORDERS.has(sortOrder)) {
    throw new Error(`422: unsupported sort ${sortBy} ${sortOrder}`)
  }
  const sorted = [...notes].sort((a, b) => {
    const left = sortBy === "title" ? a.title.toLowerCase() : a[sortBy as keyof ServerNote]
    const right = sortBy === "title" ? b.title.toLowerCase() : b[sortBy as keyof ServerNote]
    if (left !== right) {
      const ascending = left < right ? -1 : 1
      return sortOrder === "asc" ? ascending : -ascending
    }
    return a.id < b.id ? -1 : 1
  })
  const page = sorted.slice(offset, offset + limit)
  const hasMore = offset + page.length < notes.length
  const nextOffset = hasMore ? offset + limit : null
  return {
    notes: page,
    items: page,
    results: page,
    count: page.length,
    limit,
    offset,
    total: notes.length,
    has_more: hasMore,
    next_offset: nextOffset,
    pagination: {
      mode: "offset",
      limit,
      offset,
      total: notes.length,
      has_more: hasMore,
      next_offset: nextOffset
    }
  }
}

const installNotesServer = (notes: ServerNote[]) => {
  mockBgRequest.mockImplementation(async (request: { path?: string; method?: string }) => {
    const path = String(request.path || "")
    const method = String(request.method || "GET").toUpperCase()
    if (path === "/api/v1/admin/notes/title-settings" && method === "GET") {
      return {
        llm_enabled: false,
        default_strategy: "heuristic",
        effective_strategy: "heuristic",
        strategies: ["heuristic"]
      }
    }
    if (isBrowseListPath(path) && method === "GET") {
      return serveNotesList(path, notes)
    }
    return {}
  })
}

const browseRequestParams = () =>
  mockBgRequest.mock.calls
    .map(([request]) => String(request?.path || ""))
    .filter(isBrowseListPath)
    .map((path) => new URL(`https://server.test${path}`).searchParams)

const renderedNoteIds = () =>
  screen
    .queryAllByTestId(/^notes-open-button-/)
    .map((button) => button.getAttribute("data-testid")?.replace("notes-open-button-", ""))

const queryClients: QueryClient[] = []

const renderPage = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false }
    }
  })
  queryClients.push(queryClient)
  return render(
    <QueryClientProvider client={queryClient}>
      <NotesManagerPage />
    </QueryClientProvider>
  )
}

const LIST_TIMEOUT = { timeout: 10_000 }

describe("NotesManagerPage list contract (NL-01, #3103)", { timeout: 60_000 }, () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockConfirmDanger.mockResolvedValue(true)
    mockGetSetting.mockResolvedValue(null)
    mockSetSetting.mockResolvedValue(undefined)
    mockClearSetting.mockResolvedValue(undefined)
  })

  afterEach(async () => {
    cleanup()
    while (queryClients.length > 0) {
      const queryClient = queryClients.pop()
      if (!queryClient) continue
      await queryClient.cancelQueries()
      queryClient.clear()
    }
  })

  it("requests the browse page with limit/offset and the selected sort", async () => {
    installNotesServer(makeServerNotes(105))
    renderPage()

    await waitFor(() => {
      expect(browseRequestParams().length).toBeGreaterThan(0)
    }, LIST_TIMEOUT)
    const params = browseRequestParams()[0]
    expect({
      limit: params.get("limit"),
      offset: params.get("offset"),
      sort_by: params.get("sort_by"),
      sort_order: params.get("sort_order"),
      page: params.get("page"),
      results_per_page: params.get("results_per_page")
    }).toEqual({
      limit: "20",
      offset: "0",
      sort_by: "last_modified",
      sort_order: "desc",
      page: null,
      results_per_page: null
    })
  })

  it("reports the server's pagination.total for a library larger than 100", async () => {
    installNotesServer(makeServerNotes(105))
    renderPage()

    expect(await screen.findByText("Showing 1-20 of 105", {}, LIST_TIMEOUT)).toBeInTheDocument()
    expect(renderedNoteIds()).toHaveLength(20)
  })

  it("renders only the current page of a library smaller than the server default", async () => {
    installNotesServer(makeServerNotes(25))
    renderPage()

    expect(await screen.findByText("Showing 1-20 of 25", {}, LIST_TIMEOUT)).toBeInTheDocument()
    expect(renderedNoteIds()).toEqual(
      Array.from({ length: 20 }, (_, index) => `note-${index + 1}`)
    )
  })

  it("shows the next slice of notes on page 2", async () => {
    installNotesServer(makeServerNotes(105))
    renderPage()
    await screen.findByText("Showing 1-20 of 105", {}, LIST_TIMEOUT)

    fireEvent.click(screen.getByTitle("2"))

    expect(await screen.findByText("Showing 21-40 of 105", {}, LIST_TIMEOUT)).toBeInTheDocument()
    await waitFor(() => {
      expect(renderedNoteIds()).toEqual(
        Array.from({ length: 20 }, (_, index) => `note-${index + 21}`)
      )
    }, LIST_TIMEOUT)
    expect(browseRequestParams().at(-1)?.get("offset")).toBe("20")
  })

  it("steps back to the last page when the current page empties", async () => {
    const notes = makeServerNotes(21)
    installNotesServer(notes)
    renderPage()
    await screen.findByText("Showing 1-20 of 21", {}, LIST_TIMEOUT)
    fireEvent.click(screen.getByTitle("2"))
    await screen.findByText("Showing 21-21 of 21", {}, LIST_TIMEOUT)

    // The only note on page 2 is deleted, so page 2 no longer exists.
    notes.pop()
    await act(async () => {
      await queryClients[0].invalidateQueries({ queryKey: ["notes"] })
    })

    expect(await screen.findByText("Showing 1-20 of 20", {}, LIST_TIMEOUT)).toBeInTheDocument()
    expect(renderedNoteIds()).toHaveLength(20)
  })
})
