/**
 * UX review 2026-10 contract reproductions for the /notes list (#3103).
 *
 * Unlike the stage tests, the bgRequest mock here behaves like the real
 * FastAPI notes endpoints (tldw_Server_API/app/api/v1/endpoints/notes.py):
 * `GET /api/v1/notes/` honours only `limit` (default 100) / `offset`, ignores
 * unknown query params, returns `pagination.total` (never `total_items`) and
 * only inlines keywords when `include_keywords=true`; `PATCH /api/v1/notes/{id}`
 * replaces the note's keyword set.
 *
 * These began as red-first `it.fails` reproductions. NL-01, NL-02 and NL-03
 * are fixed (#3148), so each is now a plain `it(...)` with its assertions
 * unchanged; the comment above each test records the defect it reproduced.
 */
import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import NotesManagerPage from "../NotesManagerPage"

// The list query is gated on a verified notes authority scope; pin one owner.
vi.mock("@/components/Notes/hooks/useNotesGraphAuthorityScope", () => ({
  useNotesGraphAuthorityScope: () => "notes-contract-owner"
}))

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
  mockPromptModal
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
    mockPromptModal: vi.fn()
  }
})

vi.mock("@/components/Notes/notes-manager-utils", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/components/Notes/notes-manager-utils")>()
  return { ...actual, promptModal: mockPromptModal }
})

// Bulk "Add tags" asks through its own themed prompt since NL-03 was fixed.
vi.mock("@/components/Notes/NotesBulkAddTagsPrompt", () => ({
  promptBulkAddTags: vi.fn(async () => ["c"])
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
  default: ({
    notes,
    total,
    pageSize,
    exportProgress,
    onChangePage,
    onExportAllJson,
    onToggleBulkSelection
  }: {
    notes?: Array<{ id: string | number; title?: string }>
    total: number
    pageSize: number
    exportProgress?: unknown
    onChangePage: (page: number, nextPageSize: number) => void
    onExportAllJson?: () => void
    onToggleBulkSelection?: (id: string | number, checked: boolean, shiftKey: boolean) => void
  }) => (
    <div data-testid="notes-list-panel">
      <div data-testid="notes-rendered-count">{String((notes || []).length)}</div>
      <div data-testid="notes-rendered-ids">{(notes || []).map((note) => String(note.id)).join("|")}</div>
      <div data-testid="notes-total">{String(total)}</div>
      <div data-testid="notes-export-state">{exportProgress ? "exporting" : "idle"}</div>
      <button
        type="button"
        data-testid="notes-go-page-2"
        onClick={() => onChangePage(2, pageSize)}
      >
        Page 2
      </button>
      <button
        type="button"
        data-testid="notes-export-json"
        onClick={() => onExportAllJson?.()}
      >
        Export JSON
      </button>
      {(notes || []).map((note) => (
        <button
          key={String(note.id)}
          type="button"
          data-testid={`mock-select-${String(note.id)}`}
          onClick={() => onToggleBulkSelection?.(note.id, true, false)}
        >
          Select {String(note.id)}
        </button>
      ))}
    </div>
  )
}))

type ServerNote = {
  id: string
  title: string
  content: string
  version: number
  created_at: string
  last_modified: string
  keywords: string[]
}

const makeServerNotes = (count: number): ServerNote[] =>
  Array.from({ length: count }, (_, index) => {
    const timestamp = new Date(Date.UTC(2026, 0, 1) - index * 60_000).toISOString()
    return {
      id: `note-${index + 1}`,
      title: `Note ${index + 1}`,
      content: `Body ${index + 1}`,
      version: 1,
      created_at: timestamp,
      last_modified: timestamp,
      keywords: []
    }
  })

const toKeywordRows = (keywords: string[]) =>
  keywords.map((keyword, index) => ({
    id: index + 1,
    keyword,
    sync_id: `kw-${keyword}`,
    created_at: "2026-01-01T00:00:00Z",
    last_modified: "2026-01-01T00:00:00Z",
    version: 1,
    client_id: "1",
    deleted: false
  }))

const toNoteResponse = (note: ServerNote, includeKeywords: boolean) => {
  const { keywords, ...rest } = note
  return includeKeywords ? { ...rest, keywords: toKeywordRows(keywords) } : rest
}

// The route is "/api/v1/notes/"; accept the slashless alias a client may use.
const isBrowseListPath = (path: string) => /^\/api\/v1\/notes\/?\?/.test(path)

/** Mirrors list_notes in tldw_Server_API/app/api/v1/endpoints/notes.py. */
const serveNotesList = (path: string, notes: ServerNote[]) => {
  const params = new URL(`https://server.test${path}`).searchParams
  const limit = Math.min(Number(params.get("limit") ?? 100), 1000)
  const offset = Number(params.get("offset") ?? 0)
  const includeKeywords = params.get("include_keywords") === "true"
  const page = notes.slice(offset, offset + limit).map((note) => toNoteResponse(note, includeKeywords))
  const hasMore = offset + page.length < notes.length
  return {
    notes: page,
    items: page,
    results: page,
    count: page.length,
    limit,
    offset,
    total: notes.length,
    has_more: hasMore,
    next_offset: hasMore ? offset + limit : null,
    pagination: {
      mode: "offset",
      limit,
      offset,
      total: notes.length,
      has_more: hasMore,
      next_offset: hasMore ? offset + limit : null
    }
  }
}

/**
 * `maxListRequests` is a test-only circuit breaker: a client that ignores the
 * server's pagination would otherwise loop to MAX_EXPORT_PAGES (1000 requests)
 * and make the suite slow. A correct client never reaches it.
 */
const installContractServer = (
  notes: ServerNote[],
  { maxListRequests = Number.POSITIVE_INFINITY }: { maxListRequests?: number } = {}
) => {
  let listRequests = 0
  mockBgRequest.mockImplementation(
    async (request: { path?: string; method?: string; body?: { keywords?: string[] } }) => {
      const path = String(request.path || "")
      const method = String(request.method || "GET").toUpperCase()
      if (path === "/api/v1/admin/notes/title-settings" && method === "GET") {
        return {
          llm_enabled: false,
          default_strategy: "heuristic",
          effective_strategy: "heuristic",
          strategies: ["heuristic", "llm", "llm_fallback"]
        }
      }
      if (isBrowseListPath(path) && method === "GET") {
        listRequests += 1
        if (listRequests > maxListRequests) {
          throw new Error("test circuit breaker: runaway notes list pagination")
        }
        return serveNotesList(path, notes)
      }
      const noteMatch = path.match(/^\/api\/v1\/notes\/([^/?]+)(\?.*)?$/)
      const note = noteMatch ? notes.find((row) => row.id === decodeURIComponent(noteMatch[1])) : undefined
      if (note && method === "GET") {
        return toNoteResponse(note, true)
      }
      if (note && method === "PATCH") {
        // The server replaces the keyword set with exactly what was sent.
        if (Array.isArray(request.body?.keywords)) note.keywords = [...request.body.keywords]
        note.version += 1
        return toNoteResponse(note, true)
      }
      return {}
    }
  )
}

const listRequestPaths = () =>
  mockBgRequest.mock.calls
    .map(([request]) => String(request?.path || ""))
    .filter(isBrowseListPath)

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

const waitForFirstPage = async () => {
  await waitFor(() => {
    expect(Number(screen.getByTestId("notes-rendered-count").textContent)).toBeGreaterThan(0)
  })
}

describe("NotesManagerPage UX contract reproductions (#3103)", { timeout: 60_000 }, () => {
  let capturedBlobs: Blob[] = []

  beforeEach(() => {
    vi.clearAllMocks()
    capturedBlobs = []
    mockConfirmDanger.mockResolvedValue(true)
    mockGetSetting.mockResolvedValue(null)
    mockSetSetting.mockResolvedValue(undefined)
    mockClearSetting.mockResolvedValue(undefined)
    Object.defineProperty(URL, "createObjectURL", {
      configurable: true,
      writable: true,
      value: vi.fn((blob: Blob) => {
        capturedBlobs.push(blob)
        return "blob:notes-export"
      })
    })
    Object.defineProperty(URL, "revokeObjectURL", {
      configurable: true,
      writable: true,
      value: vi.fn()
    })
    vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => undefined)
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  // NL-01 (#3103): useNotesListManagement.tsx:394-397 sends page/results_per_page, but list_notes (endpoints/notes.py:2309-2316) only reads limit/offset.
  it("NL-01 (#3103): browse request pages with the server's limit/offset params", async () => {
    installContractServer(makeServerNotes(250))
    renderPage()
    await waitForFirstPage()

    const params = new URL(`https://server.test${listRequestPaths()[0]}`).searchParams
    expect({ limit: params.get("limit"), offset: params.get("offset") }).toEqual({
      limit: "20",
      offset: "0"
    })
  })

  // NL-01 (#3103): useNotesListManagement.tsx:405 reads pagination.total_items, but list_notes (endpoints/notes.py:2353-2358) returns pagination.total.
  it("NL-01 (#3103): displayed total comes from pagination.total", async () => {
    installContractServer(makeServerNotes(250))
    renderPage()
    await waitForFirstPage()

    expect(screen.getByTestId("notes-total")).toHaveTextContent(/^250$/)
  })

  // NL-01 (#3103): page/results_per_page from useNotesListManagement.tsx:394-397 are ignored by endpoints/notes.py:2309-2316, so every page is the first 100 notes.
  it("NL-01 (#3103): page 2 shows the next slice of notes, not page 1 again", async () => {
    installContractServer(makeServerNotes(250))
    renderPage()
    await waitForFirstPage()

    fireEvent.click(screen.getByTestId("notes-go-page-2"))

    const expectedPage2 = Array.from({ length: 20 }, (_, index) => `note-${index + 21}`).join("|")
    await waitFor(() => {
      expect(screen.getByTestId("notes-rendered-ids").textContent).toBe(expectedPage2)
    })
  })

  // NL-02 (#3103): useNotesExport.tsx:135-170 pages with page/results_per_page and stops on pagination.total_pages (never sent), so it re-reads the first 100 notes.
  it("NL-02 (#3103): export pages with limit/offset and stops at the true total", async () => {
    const totalNotes = 250
    installContractServer(makeServerNotes(totalNotes), { maxListRequests: 25 })
    renderPage()
    await waitForFirstPage()
    const requestsBeforeExport = listRequestPaths().length

    fireEvent.click(screen.getByTestId("notes-export-json"))
    await waitFor(() => {
      expect(capturedBlobs).toHaveLength(1)
    })
    await waitFor(() => {
      expect(screen.getByTestId("notes-export-state")).toHaveTextContent("idle")
    })

    const exported = JSON.parse(await capturedBlobs[0].text()) as Array<{ id: string }>
    const exportRequests = listRequestPaths().length - requestsBeforeExport
    expect({
      exportRequests,
      exportedNotes: exported.length,
      uniqueNotes: new Set(exported.map((note) => note.id)).size
    }).toEqual({
      exportRequests: Math.ceil(totalNotes / 100),
      exportedNotes: totalNotes,
      uniqueNotes: totalNotes
    })
  })

  // NL-03 (#3103): NotesManagerPage.tsx:1421-1431 PATCHes only the new tags and the server replaces the set (_sync_note_keywords, endpoints/notes.py:1710-1720).
  it("NL-03 (#3103): bulk Assign tags keeps each note's existing tags", async () => {
    const notes = makeServerNotes(3)
    notes[0].keywords = ["a", "b"]
    installContractServer(notes)
    mockPromptModal.mockResolvedValue("c")
    renderPage()
    await waitForFirstPage()

    fireEvent.click(await screen.findByTestId("mock-select-note-1"))
    fireEvent.click(await screen.findByTestId("notes-bulk-assign-keywords"))

    await waitFor(() => {
      expect(
        mockBgRequest.mock.calls.some(
          ([request]) => String(request?.method || "").toUpperCase() === "PATCH"
        )
      ).toBe(true)
    })
    const patchCall = mockBgRequest.mock.calls.find(
      ([request]) => String(request?.method || "").toUpperCase() === "PATCH"
    )
    expect(String(patchCall?.[0]?.path || "")).toContain("/api/v1/notes/note-1")
    expect([...(patchCall?.[0]?.body?.keywords ?? [])].sort()).toEqual(["a", "b", "c"])
  })
})
