import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import NotesManagerPage from "../NotesManagerPage"

// NL-05 (#3101): bulk actions and Trash at scale.

const notesConnectionConfig = {
  serverUrl: "https://notes.example.test",
  authMode: "multi-user" as const,
  accessToken: "test-access-token"
}

vi.mock("@/hooks/useCanonicalConnectionConfig", () => ({
  useCanonicalConnectionConfig: () => ({
    config: notesConnectionConfig,
    loading: false,
    authorityLoading: false
  })
}))

vi.mock("@/services/tldw/TldwAuth", () => ({
  tldwAuth: {
    getCurrentUser: vi.fn(async () => ({ id: 1, is_active: true }))
  }
}))

vi.mock("@/components/Notes/hooks/useNotesGraphAuthorityScope", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/components/Notes/hooks/useNotesGraphAuthorityScope")>()
  return {
    ...actual,
    useNotesGraphAuthorityScope: () =>
      actual.createNotesGraphAuthorityScope(notesConnectionConfig.serverUrl, 1)
  }
})

const {
  mockBgRequest,
  mockMessageSuccess,
  mockMessageError,
  mockMessageWarning,
  mockMessageInfo,
  mockMessageOpen,
  mockNavigate,
  mockConfirmDanger,
  mockGetSetting,
  mockSetSetting,
  mockClearSetting,
  mockPromptBulkAddTags,
  mockShowUndoNotification
} = vi.hoisted(() => ({
  mockBgRequest: vi.fn(),
  mockMessageSuccess: vi.fn(),
  mockMessageError: vi.fn(),
  mockMessageWarning: vi.fn(),
  mockMessageInfo: vi.fn(),
  mockMessageOpen: vi.fn(),
  mockNavigate: vi.fn(),
  mockConfirmDanger: vi.fn(),
  mockGetSetting: vi.fn(),
  mockSetSetting: vi.fn(),
  mockClearSetting: vi.fn(),
  mockPromptBulkAddTags: vi.fn(),
  mockShowUndoNotification: vi.fn()
}))

vi.mock("@/components/Notes/NotesBulkAddTagsPrompt", () => ({
  promptBulkAddTags: mockPromptBulkAddTags
}))

vi.mock("@/hooks/useUndoNotification", () => ({
  useUndoNotification: () => ({ showUndoNotification: mockShowUndoNotification })
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, defaultValueOrOptions?: string | { defaultValue?: string; [key: string]: unknown }) => {
      const template =
        typeof defaultValueOrOptions === "string" ? defaultValueOrOptions : defaultValueOrOptions?.defaultValue
      if (!template) return key
      if (typeof defaultValueOrOptions !== "object") return template
      // Like i18next, a placeholder without a value stays as written.
      return template.replace(/\{\{(\w+)\}\}/g, (match, name: string) => String(defaultValueOrOptions[name] ?? match))
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
    info: mockMessageInfo,
    open: mockMessageOpen,
    destroy: vi.fn()
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

type Request = { path?: string; method?: string; body?: Record<string, unknown> }
type Route = (request: Request, path: string, method: string) => unknown

const ACTIVE_NOTES = [
  { id: "n1", title: "Alpha", content: "a", version: 1 },
  { id: "n2", title: "Beta", content: "b", version: 1 },
  { id: "n3", title: "Gamma", content: "c", version: 1 }
]

/** Routes requests through `route` first; anything it leaves undefined gets a default. */
const serve = (route: Route = () => undefined) => {
  mockBgRequest.mockImplementation(async (request: Request) => {
    const path = String(request.path || "")
    const method = String(request.method || "GET").toUpperCase()
    const routed = await route(request, path, method)
    if (routed !== undefined) return routed
    if (path.startsWith("/api/v1/notes/?")) {
      return { items: ACTIVE_NOTES, pagination: { total: ACTIVE_NOTES.length } }
    }
    if (path === "/api/v1/admin/notes/title-settings") {
      return { llm_enabled: false, default_strategy: "heuristic", effective_strategy: "heuristic", strategies: ["heuristic"] }
    }
    const detail = path.match(/^\/api\/v1\/notes\/(n\d)$/)
    if (detail && method === "GET") {
      const note = ACTIVE_NOTES.find((row) => row.id === detail[1])
      return { ...note, keywords: [] }
    }
    return {}
  })
}

const deferred = <T = unknown,>() => {
  let resolve!: (value: T) => void
  let reject!: (error: unknown) => void
  const promise = new Promise<T>((res, rej) => {
    resolve = res
    reject = rej
  })
  return { promise, resolve, reject }
}

const noteIdOf = (path: string) => path.match(/^\/api\/v1\/notes\/([^/?]+)/)?.[1] ?? ""

const callsTo = (method: string, id?: string) =>
  mockBgRequest.mock.calls.filter(([request]: [Request]) => {
    const path = String(request?.path || "")
    return String(request?.method || "GET").toUpperCase() === method && (id == null || noteIdOf(path) === id)
  })

const renderPage = () => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } }
  })
  return render(
    <QueryClientProvider client={queryClient}>
      <NotesManagerPage />
    </QueryClientProvider>
  )
}

const selectRows = async (...ids: string[]) => {
  for (const id of ids) fireEvent.click(await screen.findByTestId(`notes-select-checkbox-${id}`))
  await waitFor(() => {
    expect(screen.getByTestId("notes-bulk-selected-count")).toHaveTextContent(`${ids.length} selected`)
  })
}

describe("NotesManagerPage NL-05 bulk runs", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockConfirmDanger.mockResolvedValue(true)
    mockGetSetting.mockResolvedValue(null)
    mockSetSetting.mockResolvedValue(undefined)
    mockClearSetting.mockResolvedValue(undefined)
  })

  it("shows n/N progress and disables the bar while a bulk delete runs", async () => {
    const pending: Record<string, ReturnType<typeof deferred>> = { n1: deferred(), n2: deferred() }
    serve((_request, path, method) => (method === "DELETE" ? pending[noteIdOf(path)].promise : undefined))
    renderPage()
    await selectRows("n1", "n2")

    fireEvent.click(screen.getByTestId("notes-bulk-delete"))

    const progress = await screen.findByTestId("notes-bulk-progress")
    expect(progress).toHaveTextContent("Deleting 0/2")
    for (const testId of ["notes-bulk-export", "notes-bulk-assign-keywords", "notes-bulk-delete", "notes-bulk-clear-selection"]) {
      expect(screen.getByTestId(testId)).toBeDisabled()
    }
    // A second bulk action cannot start while this one runs.
    fireEvent.click(screen.getByTestId("notes-bulk-assign-keywords"))
    expect(mockPromptBulkAddTags).not.toHaveBeenCalled()

    pending.n1.resolve({})
    await waitFor(() => expect(screen.getByTestId("notes-bulk-progress")).toHaveTextContent("Deleting 1/2"))
    pending.n2.resolve({})

    await waitFor(() => expect(screen.queryByTestId("notes-bulk-progress")).not.toBeInTheDocument())
    expect(mockMessageSuccess).toHaveBeenCalledWith("Deleted 2 selected notes")
    expect(callsTo("DELETE")).toHaveLength(2)
  })

  it("lists the notes a bulk delete could not delete, and Retry deletes only those", async () => {
    let failBeta = true
    serve((_request, path, method) => {
      if (method !== "DELETE") return undefined
      if (noteIdOf(path) === "n2" && failBeta) {
        failBeta = false
        throw new Error("server unavailable")
      }
      return {}
    })
    renderPage()
    await selectRows("n1", "n2")
    fireEvent.click(screen.getByTestId("notes-bulk-delete"))

    const failures = await screen.findByTestId("notes-bulk-failures")
    expect(failures).toHaveTextContent("Delete failed for 1:")
    expect(within(failures).getByText("Beta")).toBeInTheDocument()
    expect(failures).not.toHaveTextContent("Alpha")

    fireEvent.click(within(failures).getByTestId("notes-bulk-retry"))

    await waitFor(() => expect(callsTo("DELETE", "n2")).toHaveLength(2))
    await waitFor(() => expect(screen.queryByTestId("notes-bulk-failures")).not.toBeInTheDocument())
    expect(callsTo("DELETE", "n1")).toHaveLength(1)
    expect(mockConfirmDanger).toHaveBeenCalledTimes(1)
  })

  it("retries failed tag additions with the same tags, without asking again", async () => {
    mockPromptBulkAddTags.mockResolvedValue(["research"])
    let failAlpha = true
    serve((_request, path, method) => {
      if (method !== "PATCH") return undefined
      if (noteIdOf(path) === "n1" && failAlpha) {
        failAlpha = false
        throw new Error("server unavailable")
      }
      return { id: noteIdOf(path), version: 2, keywords: ["research"] }
    })
    renderPage()
    await selectRows("n1", "n2")
    fireEvent.click(screen.getByTestId("notes-bulk-assign-keywords"))

    const failures = await screen.findByTestId("notes-bulk-failures")
    expect(failures).toHaveTextContent("Adding tags failed for 1:")
    expect(within(failures).getByText("Alpha")).toBeInTheDocument()

    fireEvent.click(within(failures).getByTestId("notes-bulk-retry"))

    await waitFor(() => expect(callsTo("PATCH", "n1")).toHaveLength(2))
    await waitFor(() => expect(screen.queryByTestId("notes-bulk-failures")).not.toBeInTheDocument())
    expect(mockPromptBulkAddTags).toHaveBeenCalledTimes(1)
    const retried = callsTo("PATCH", "n1")[1][0] as Request
    expect(retried.body?.keywords).toEqual(["research"])
  })

  it("dismisses the failure list", async () => {
    serve((_request, path, method) => {
      if (method === "DELETE" && noteIdOf(path) === "n1") throw new Error("server unavailable")
      return undefined
    })
    renderPage()
    await selectRows("n1")
    fireEvent.click(screen.getByTestId("notes-bulk-delete"))
    const failures = await screen.findByTestId("notes-bulk-failures")
    fireEvent.click(within(failures).getByTestId("notes-bulk-dismiss-failures"))
    await waitFor(() => expect(screen.queryByTestId("notes-bulk-failures")).not.toBeInTheDocument())
  })
})

const listParams = (path: string) => new URLSearchParams(path.split("?")[1] ?? "")

/** A library bigger than the visible page: the list shows 3 rows, gathers read pages of 100. */
const serveLibrary = (library: Array<{ id: string; title: string; content: string; version: number }>) =>
  serve((_request, path, method) => {
    if (method === "GET" && path.startsWith("/api/v1/notes/?")) {
      const params = listParams(path)
      const limit = Number(params.get("limit"))
      const offset = Number(params.get("offset"))
      const rows = limit >= 100 ? library.slice(offset, offset + limit) : library.slice(0, 3)
      return { items: rows, pagination: { total: library.length } }
    }
    if (method === "DELETE") return {}
    return undefined
  })

const gatherRequests = () =>
  mockBgRequest.mock.calls.filter(([request]: [Request]) => {
    const path = String(request?.path || "")
    return path.startsWith("/api/v1/notes/?") && listParams(path).get("limit") === "100"
  })

const makeLibrary = (size: number) =>
  Array.from({ length: size }, (_, index) => ({ id: `m${index + 1}`, title: `Match ${index + 1}`, content: "x", version: 1 }))

describe("NotesManagerPage NL-05 select all", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockConfirmDanger.mockResolvedValue(true)
    mockGetSetting.mockResolvedValue(null)
    mockSetSetting.mockResolvedValue(undefined)
    mockClearSetting.mockResolvedValue(undefined)
  })

  it("selects every row on the page from the header checkbox, and clears them again", async () => {
    serve()
    renderPage()
    const selectAll = await screen.findByTestId("notes-select-all-page")
    await screen.findByTestId("notes-select-checkbox-n3")

    fireEvent.click(selectAll)
    await waitFor(() => expect(screen.getByTestId("notes-bulk-selected-count")).toHaveTextContent("3 selected"))
    for (const id of ["n1", "n2", "n3"]) expect(screen.getByTestId(`notes-select-checkbox-${id}`)).toBeChecked()
    expect(selectAll).toBeChecked()

    fireEvent.click(selectAll)
    await waitFor(() => expect(screen.queryByTestId("notes-bulk-actions-bar")).not.toBeInTheDocument())
  })

  it("selects all matching notes beyond the page and runs the bulk action on every one", async () => {
    serveLibrary(makeLibrary(5))
    renderPage()
    await screen.findByTestId("notes-select-checkbox-m3")
    fireEvent.click(screen.getByTestId("notes-select-all-page"))

    const selectMatching = await screen.findByTestId("notes-bulk-select-all-matching")
    expect(selectMatching).toHaveTextContent("Select all 5 matching")
    fireEvent.click(selectMatching)

    await waitFor(() => expect(screen.getByTestId("notes-bulk-selected-count")).toHaveTextContent("5 selected"))
    expect(screen.getByTestId("notes-bulk-matching-status")).toHaveTextContent("All 5 matching notes are selected.")

    fireEvent.click(screen.getByTestId("notes-bulk-delete"))
    await waitFor(() => expect(callsTo("DELETE")).toHaveLength(5))
    expect(mockConfirmDanger).toHaveBeenCalledWith(
      expect.objectContaining({ content: expect.stringContaining("Delete 5 selected notes") })
    )
  })

  it("goes back to the page selection when a row is unchecked after Select all matching", async () => {
    serveLibrary(makeLibrary(5))
    renderPage()
    await screen.findByTestId("notes-select-checkbox-m3")
    fireEvent.click(screen.getByTestId("notes-select-all-page"))
    fireEvent.click(await screen.findByTestId("notes-bulk-select-all-matching"))
    await waitFor(() => expect(screen.getByTestId("notes-bulk-selected-count")).toHaveTextContent("5 selected"))

    fireEvent.click(screen.getByTestId("notes-select-checkbox-m1"))

    await waitFor(() => expect(screen.getByTestId("notes-bulk-selected-count")).toHaveTextContent("2 selected"))
    expect(screen.queryByTestId("notes-bulk-matching-status")).not.toBeInTheDocument()
  })

  it("does not bring an all-matching selection back after Notes -> Trash -> Notes", async () => {
    serveLibrary(makeLibrary(5))
    renderPage()
    await screen.findByTestId("notes-select-checkbox-m3")
    fireEvent.click(screen.getByTestId("notes-select-all-page"))
    fireEvent.click(await screen.findByTestId("notes-bulk-select-all-matching"))
    await waitFor(() => expect(screen.getByTestId("notes-bulk-selected-count")).toHaveTextContent("5 selected"))

    fireEvent.click(screen.getByTestId("notes-mode-trash"))
    await waitFor(() => expect(screen.getByTestId("notes-mode-trash")).toHaveAttribute("aria-pressed", "true"))
    fireEvent.click(screen.getByTestId("notes-mode-active"))
    await screen.findByTestId("notes-select-checkbox-m3")

    expect(screen.queryByTestId("notes-bulk-actions-bar")).not.toBeInTheDocument()
    expect(screen.getByTestId("notes-select-checkbox-m1")).not.toBeChecked()
  })

  it("caps Select all matching at the first 1000 notes", async () => {
    serveLibrary(makeLibrary(1500))
    renderPage()
    await screen.findByTestId("notes-select-checkbox-m3")
    fireEvent.click(screen.getByTestId("notes-select-all-page"))

    const selectMatching = await screen.findByTestId("notes-bulk-select-all-matching")
    expect(selectMatching).toHaveTextContent("Select the first 1000 matching")
    fireEvent.click(selectMatching)

    await waitFor(() => expect(screen.getByTestId("notes-bulk-selected-count")).toHaveTextContent("1000 selected"))
    expect(screen.getByTestId("notes-bulk-matching-status")).toHaveTextContent(
      "The first 1000 of 1500 matching notes are selected."
    )
    expect(gatherRequests()).toHaveLength(10)
  })
})

const TRASH_NOTES = [
  { id: "t1", title: "Old idea", content: "o", version: 3, deleted: true },
  { id: "t2", title: "Draft plan", content: "d", version: 5, deleted: true },
  { id: "t3", title: "Scratch", content: "s", version: 2, deleted: true }
]

const restoreCalls = () =>
  mockBgRequest.mock.calls
    .map(([request]: [Request]) => request)
    .filter((request) => String(request?.method || "").toUpperCase() === "POST" && String(request?.path || "").includes("/restore"))
    .map((request) => String(request.path))

const serveTrash = (onRestore: (id: string) => unknown = (id) => ({ id, version: 9, deleted: false })) => {
  let trash = TRASH_NOTES.slice()
  serve((_request, path, method) => {
    if (path.startsWith("/api/v1/notes/trash?")) {
      const text = (listParams(path).get("query") || "").toLowerCase()
      const rows = text ? trash.filter((note) => `${note.title} ${note.content}`.toLowerCase().includes(text)) : trash
      return { items: rows, total: rows.length }
    }
    const restore = path.match(/^\/api\/v1\/notes\/(t\d)\/restore\?expected_version=(\d+)$/)
    if (restore && method === "POST") {
      const result = onRestore(restore[1])
      trash = trash.filter((note) => note.id !== restore[1])
      return result
    }
    const detail = path.match(/^\/api\/v1\/notes\/(t\d)$/)
    if (detail && method === "GET") {
      const note = TRASH_NOTES.find((row) => row.id === detail[1])
      return { ...note, version: 9, deleted: false, keywords: [] }
    }
    return undefined
  })
}

const openTrash = async () => {
  fireEvent.click(await screen.findByTestId("notes-mode-trash"))
  await screen.findByTestId("notes-trash-row-t1")
}

describe("NotesManagerPage NL-05 Trash", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockConfirmDanger.mockResolvedValue(true)
    mockGetSetting.mockResolvedValue(null)
    mockSetSetting.mockResolvedValue(undefined)
    mockClearSetting.mockResolvedValue(undefined)
  })

  it("stays in Trash after a restore and offers Open note in the toast", async () => {
    serveTrash()
    renderPage()
    await openTrash()

    fireEvent.click(screen.getByTestId("notes-restore-t1"))

    await waitFor(() => expect(mockMessageOpen).toHaveBeenCalled())
    expect(restoreCalls()).toEqual(["/api/v1/notes/t1/restore?expected_version=3"])
    await waitFor(() => expect(screen.queryByTestId("notes-trash-row-t1")).not.toBeInTheDocument())
    expect(screen.getByTestId("notes-mode-trash")).toHaveAttribute("aria-pressed", "true")
    expect(screen.getByTestId("notes-trash-row-t2")).toBeInTheDocument()
    expect(callsTo("GET", "t1")).toHaveLength(0)

    const toast = render(<>{mockMessageOpen.mock.calls[0][0].content}</>)
    expect(toast.container).toHaveTextContent("Note restored")
    fireEvent.click(within(toast.container).getByTestId("notes-restored-open-note"))

    await waitFor(() => expect(screen.getByTestId("notes-mode-trash")).toHaveAttribute("aria-pressed", "false"))
    await waitFor(() => expect(callsTo("GET", "t1").length).toBeGreaterThan(0))
  })

  it("selects Trash notes with checkboxes and restores them together", async () => {
    serveTrash()
    renderPage()
    await openTrash()

    fireEvent.click(screen.getByTestId("notes-select-checkbox-t1"))
    fireEvent.click(screen.getByTestId("notes-select-checkbox-t2"))
    await waitFor(() => expect(screen.getByTestId("notes-bulk-selected-count")).toHaveTextContent("2 selected"))
    expect(screen.queryByTestId("notes-bulk-delete")).not.toBeInTheDocument()
    expect(screen.queryByTestId("notes-bulk-export")).not.toBeInTheDocument()

    fireEvent.click(screen.getByTestId("notes-bulk-restore"))

    await waitFor(() => expect(mockMessageSuccess).toHaveBeenCalledWith("Restored 2 notes"))
    expect(restoreCalls()).toEqual([
      "/api/v1/notes/t1/restore?expected_version=3",
      "/api/v1/notes/t2/restore?expected_version=5"
    ])
    expect(screen.getByTestId("notes-mode-trash")).toHaveAttribute("aria-pressed", "true")
    await waitFor(() => expect(screen.queryByTestId("notes-trash-row-t1")).not.toBeInTheDocument())
    expect(screen.getByTestId("notes-trash-row-t3")).toBeInTheDocument()
  })

  it("selects every Trash note on the page from the header checkbox", async () => {
    serveTrash()
    renderPage()
    await openTrash()
    fireEvent.click(screen.getByTestId("notes-select-all-page"))
    await waitFor(() => expect(screen.getByTestId("notes-bulk-selected-count")).toHaveTextContent("3 selected"))
    expect(screen.getByTestId("notes-bulk-restore")).toBeInTheDocument()
  })

  it("lists notes that could not be restored and retries only those", async () => {
    let failDraft = true
    serveTrash((id) => {
      if (id === "t2" && failDraft) {
        failDraft = false
        throw new Error("server unavailable")
      }
      return { id, version: 9, deleted: false }
    })
    renderPage()
    await openTrash()
    fireEvent.click(screen.getByTestId("notes-select-checkbox-t1"))
    fireEvent.click(screen.getByTestId("notes-select-checkbox-t2"))
    fireEvent.click(await screen.findByTestId("notes-bulk-restore"))

    const failures = await screen.findByTestId("notes-bulk-failures")
    expect(failures).toHaveTextContent("Restore failed for 1:")
    expect(within(failures).getByText("Draft plan")).toBeInTheDocument()

    fireEvent.click(within(failures).getByTestId("notes-bulk-retry"))
    await waitFor(() =>
      expect(restoreCalls().filter((path) => path.includes("/t2/"))).toEqual([
        "/api/v1/notes/t2/restore?expected_version=5",
        "/api/v1/notes/t2/restore?expected_version=5"
      ])
    )
    await waitFor(() => expect(screen.queryByTestId("notes-bulk-failures")).not.toBeInTheDocument())
  })

  it("searches Trash on the server, and shows a no-results state with Clear", async () => {
    serveTrash()
    renderPage()
    await openTrash()

    const search = screen.getByTestId("notes-trash-search")
    fireEvent.change(search, { target: { value: "draft" } })
    fireEvent.keyDown(search, { key: "Enter", code: "Enter", keyCode: 13 })

    await waitFor(() => {
      expect(screen.queryByTestId("notes-trash-row-t1")).not.toBeInTheDocument()
      expect(screen.getByTestId("notes-trash-row-t2")).toBeInTheDocument()
    })
    const searched = mockBgRequest.mock.calls
      .map(([request]: [Request]) => String(request?.path || ""))
      .filter((path) => path.startsWith("/api/v1/notes/trash?") && listParams(path).get("query") === "draft")
    expect(searched.length).toBeGreaterThan(0)

    fireEvent.change(search, { target: { value: "zzz" } })
    fireEvent.keyDown(search, { key: "Enter", code: "Enter", keyCode: 13 })
    const empty = await screen.findByTestId("notes-list-no-results-state")
    expect(empty).toHaveTextContent('No notes match "zzz".')

    fireEvent.click(within(empty).getByRole("button", { name: /Clear search/ }))
    expect(await screen.findByTestId("notes-trash-row-t1")).toBeInTheDocument()
    expect(screen.getByTestId("notes-trash-search")).toHaveValue("")
  })
})
