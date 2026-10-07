/**
 * NL-03 (#3103): bulk "Add tags" must add to each selected note's tags, not
 * replace them, and must offer an Undo that never clobbers later edits.
 *
 * The bgRequest mock mirrors the real notes endpoints
 * (tldw_Server_API/app/api/v1/endpoints/notes.py): `PATCH /api/v1/notes/{id}`
 * REPLACES the keyword set, checks the `expected-version` header, and a
 * keyword-only PATCH does not bump the note version.
 */
import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import NotesManagerPage from "../NotesManagerPage"

// Pin the notes authority scope like the other Notes stage suites (see stage44).
// Without it the notes list never loads when this file runs on its own.
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
  mockNavigate,
  mockConfirmDanger,
  mockGetSetting,
  mockSetSetting,
  mockClearSetting,
  mockPromptBulkAddTags,
  mockShowUndoNotification
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
    mockPromptBulkAddTags: vi.fn(),
    mockShowUndoNotification: vi.fn()
  }
})

vi.mock("@/components/Notes/NotesBulkAddTagsPrompt", () => ({
  promptBulkAddTags: mockPromptBulkAddTags
}))

vi.mock("@/hooks/useUndoNotification", () => ({
  useUndoNotification: () => ({ showUndoNotification: mockShowUndoNotification })
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
  getAllNoteKeywordStats: vi.fn(async () => [
    { keyword: "research", noteCount: 4 },
    { keyword: "summary", noteCount: 1 }
  ]),
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
    bulkSelectedIds,
    onSelectNote,
    onToggleBulkSelection
  }: {
    notes?: Array<{ id: string | number; title?: string }>
    bulkSelectedIds?: string[]
    onSelectNote?: (id: string | number) => void
    onToggleBulkSelection?: (id: string | number, checked: boolean, shiftKey: boolean) => void
  }) => (
    <div data-testid="notes-list-panel">
      <div data-testid="notes-list-bulk-selected">{(bulkSelectedIds || []).join("|")}</div>
      {(notes || []).map((note) => (
        <div key={String(note.id)}>
          <button
            type="button"
            data-testid={`mock-open-${String(note.id)}`}
            onClick={() => onSelectNote?.(note.id)}
          >
            Open {String(note.id)}
          </button>
          <button
            type="button"
            data-testid={`mock-select-${String(note.id)}`}
            onClick={() => onToggleBulkSelection?.(note.id, true, false)}
          >
            Select {String(note.id)}
          </button>
        </div>
      ))}
    </div>
  )
}))

type ServerNote = {
  id: string
  title: string
  content: string
  version: number
  keywords: string[]
}

type BgRequest = {
  path?: string
  method?: string
  headers?: Record<string, string>
  body?: { keywords?: string[] }
}

let serverNotes: ServerNote[] = []
/** Keywords the list endpoint reports; lets a test make list rows stale. */
let listKeywordsOverride: Record<string, string[]> = {}
let patchFailures: Map<string, Error> = new Map()

const versionConflict = (noteId: string) =>
  Object.assign(
    new Error(`Note ID ${noteId} update failed: version mismatch`),
    { status: 409 }
  )

const keywordRows = (keywords: string[]) =>
  keywords.map((keyword, index) => ({ id: index + 1, keyword, deleted: false, version: 1 }))

const toNoteResponse = (note: ServerNote, keywords = note.keywords) => ({
  id: note.id,
  title: note.title,
  content: note.content,
  version: note.version,
  last_modified: "2026-10-01T00:00:00Z",
  keywords: keywordRows(keywords)
})

const findServerNote = (id: string) => serverNotes.find((note) => note.id === id)

const installNotesServer = () => {
  mockBgRequest.mockImplementation(async (request: BgRequest) => {
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
    if (/^\/api\/v1\/notes\/?\?/.test(path) && method === "GET") {
      const items = serverNotes.map((note) =>
        toNoteResponse(note, listKeywordsOverride[note.id] ?? note.keywords)
      )
      return {
        items,
        pagination: { total: items.length, limit: 20, offset: 0, has_more: false }
      }
    }
    const noteMatch = path.match(/^\/api\/v1\/notes\/([^/?]+)(\?.*)?$/)
    const note = noteMatch ? findServerNote(decodeURIComponent(noteMatch[1])) : undefined
    if (note && method === "GET") {
      return toNoteResponse(note)
    }
    if (note && method === "PATCH") {
      const failure = patchFailures.get(note.id)
      if (failure) throw failure
      const expected = request.headers?.["expected-version"]
      if (expected != null && Number(expected) !== note.version) {
        throw versionConflict(note.id)
      }
      if (Array.isArray(request.body?.keywords)) {
        // The server replaces the whole keyword set with what was sent.
        note.keywords = [...request.body.keywords]
      }
      // Keyword-only PATCH does not bump the note version on the server.
      return toNoteResponse(note)
    }
    if (note && method === "PUT") {
      note.version += 1
      return toNoteResponse(note)
    }
    return {}
  })
}

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

const patchCallsFor = (noteId: string) =>
  mockBgRequest.mock.calls
    .map(([request]) => request as BgRequest)
    .filter(
      (request) =>
        String(request?.method || "").toUpperCase() === "PATCH" &&
        String(request?.path || "").startsWith(`/api/v1/notes/${noteId}`)
    )

const selectNotes = async (ids: string[]) => {
  for (const id of ids) {
    fireEvent.click(await screen.findByTestId(`mock-select-${id}`))
  }
  await waitFor(() => {
    expect(screen.getByTestId("notes-list-bulk-selected")).toHaveTextContent(ids.join("|"))
  })
}

const addTagsToSelection = async (ids: string[], tags: string[]) => {
  await selectNotes(ids)
  mockPromptBulkAddTags.mockResolvedValueOnce(tags)
  fireEvent.click(screen.getByTestId("notes-bulk-assign-keywords"))
  await waitFor(() => {
    expect(mockPromptBulkAddTags).toHaveBeenCalled()
  })
}

const waitForUndoOffer = async () => {
  await waitFor(() => {
    expect(mockShowUndoNotification).toHaveBeenCalledTimes(1)
  })
  return mockShowUndoNotification.mock.calls[0][0] as {
    title: string
    description?: string
    onUndo: () => Promise<void>
  }
}

describe("NotesManagerPage stage 16 bulk add tags (NL-03)", { timeout: 30_000 }, () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockConfirmDanger.mockResolvedValue(true)
    mockGetSetting.mockResolvedValue(null)
    mockSetSetting.mockResolvedValue(undefined)
    mockClearSetting.mockResolvedValue(undefined)
    listKeywordsOverride = {}
    patchFailures = new Map()
    serverNotes = [
      { id: "n1", title: "Alpha", content: "a", version: 3, keywords: ["alpha", "beta"] },
      { id: "n2", title: "Beta", content: "b", version: 1, keywords: [] },
      { id: "n3", title: "Gamma", content: "c", version: 7, keywords: ["x"] }
    ]
    installNotesServer()
  })

  it("labels the bulk action 'Add tags'", async () => {
    renderPage()
    await selectNotes(["n1"])
    expect(screen.getByTestId("notes-bulk-assign-keywords")).toHaveTextContent(/^Add tags$/)
  })

  it("adds the chosen tags to each note's current tags using the note's version", async () => {
    // The list row is stale: the server already has "beta" on n1.
    listKeywordsOverride = { n1: ["alpha"] }
    renderPage()
    await addTagsToSelection(["n1", "n2"], ["gamma"])

    await waitFor(() => {
      expect(patchCallsFor("n2")).toHaveLength(1)
    })
    const [n1Patch] = patchCallsFor("n1")
    expect(n1Patch.body?.keywords).toEqual(["alpha", "beta", "gamma"])
    expect(n1Patch.headers?.["expected-version"]).toBe("3")
    expect(patchCallsFor("n2")[0].body?.keywords).toEqual(["gamma"])
    expect(findServerNote("n1")?.keywords).toEqual(["alpha", "beta", "gamma"])

    const undoOffer = await waitForUndoOffer()
    expect(undoOffer.title).toContain("gamma")
    expect(undoOffer.title).toContain("2 notes")
    expect(undoOffer.description).toMatch(/existing tags were kept/i)
    expect(mockConfirmDanger).not.toHaveBeenCalled()
  })

  it("offers existing tags as suggestions in the prompt", async () => {
    renderPage()
    await addTagsToSelection(["n1", "n2"], ["gamma"])

    const [modalApi, options] = mockPromptBulkAddTags.mock.calls[0]
    expect(modalApi).toBeTruthy()
    expect(options.noteCount).toBe(2)
    await waitFor(() => {
      expect(options.suggestions).toEqual(expect.arrayContaining(["research", "summary"]))
    })
  })

  it("does not add a tag a note already has, ignoring case", async () => {
    serverNotes[0].keywords = ["Research", "beta"]
    serverNotes[1].keywords = ["research", "gamma"]
    renderPage()
    await addTagsToSelection(["n1", "n2"], ["research", "Gamma", " gamma "])

    await waitFor(() => {
      expect(patchCallsFor("n1")).toHaveLength(1)
    })
    expect(patchCallsFor("n1")[0].body?.keywords).toEqual(["Research", "beta", "Gamma"])
    // n2 already has every tag, so it is left untouched.
    await waitFor(() => {
      expect(mockMessageInfo).toHaveBeenCalledWith(expect.stringMatching(/1 selected note already had/))
    })
    expect(patchCallsFor("n2")).toHaveLength(0)
    const undoOffer = await waitForUndoOffer()
    expect(undoOffer.title).toContain("1 note")
  })

  it("reports a note that fails and still updates the others", async () => {
    patchFailures.set("n2", versionConflict("n2"))
    renderPage()
    await addTagsToSelection(["n1", "n2", "n3"], ["gamma"])

    await waitFor(() => {
      expect(mockMessageWarning).toHaveBeenCalledWith(expect.stringContaining('"Beta"'))
    })
    expect(findServerNote("n1")?.keywords).toEqual(["alpha", "beta", "gamma"])
    expect(findServerNote("n2")?.keywords).toEqual([])
    expect(findServerNote("n3")?.keywords).toEqual(["x", "gamma"])
    const undoOffer = await waitForUndoOffer()
    expect(undoOffer.title).toContain("2 notes")
  })

  it("Undo restores each note's previous tags", async () => {
    renderPage()
    await addTagsToSelection(["n1", "n2"], ["gamma"])
    const undoOffer = await waitForUndoOffer()
    const patchesBeforeUndo = mockBgRequest.mock.calls.length

    await expect(undoOffer.onUndo()).resolves.toBeUndefined()

    expect(findServerNote("n1")?.keywords).toEqual(["alpha", "beta"])
    expect(findServerNote("n2")?.keywords).toEqual([])
    const undoPatches = mockBgRequest.mock.calls
      .slice(patchesBeforeUndo)
      .map(([request]) => request as BgRequest)
      .filter((request) => String(request.method).toUpperCase() === "PATCH")
    expect(undoPatches.map((request) => request.headers?.["expected-version"])).toEqual(["3", "1"])
  })

  it("Undo skips notes edited since the tags were added and reports them", async () => {
    renderPage()
    await addTagsToSelection(["n1", "n2", "n3"], ["gamma"])
    const undoOffer = await waitForUndoOffer()

    // n1: content edited elsewhere (version bump). n2: tags edited elsewhere
    // (keyword-only edits keep the version, so Undo must compare tags too).
    const n1 = findServerNote("n1")!
    n1.content = "edited elsewhere"
    n1.version += 1
    findServerNote("n2")!.keywords = ["gamma", "delta"]

    await expect(undoOffer.onUndo()).rejects.toThrow(/"Alpha".*"Beta"|"Beta".*"Alpha"/)

    expect(findServerNote("n1")?.keywords).toEqual(["alpha", "beta", "gamma"])
    expect(findServerNote("n2")?.keywords).toEqual(["gamma", "delta"])
    expect(findServerNote("n3")?.keywords).toEqual(["x"])
  })

  it("Undo skips the open note while it has unsaved edits", async () => {
    renderPage()
    fireEvent.click(await screen.findByTestId("mock-open-n1"))
    await waitFor(() => {
      expect(screen.getByDisplayValue("Alpha")).toBeInTheDocument()
    })
    await addTagsToSelection(["n1", "n2"], ["gamma"])
    const undoOffer = await waitForUndoOffer()

    fireEvent.change(screen.getByPlaceholderText("Write your note here... (Markdown supported)"), {
      target: { value: "unsaved local edit" }
    })
    await waitFor(() => {
      expect(screen.getByDisplayValue("unsaved local edit")).toBeInTheDocument()
    })

    await expect(undoOffer.onUndo()).rejects.toThrow(/"Alpha"/)
    expect(findServerNote("n1")?.keywords).toEqual(["alpha", "beta", "gamma"])
    expect(findServerNote("n2")?.keywords).toEqual([])
  })
})
