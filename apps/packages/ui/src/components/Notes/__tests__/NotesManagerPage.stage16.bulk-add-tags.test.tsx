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
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import NotesManagerPage from "../NotesManagerPage"
import userEvent from "@testing-library/user-event"
import type { BgRequestInit } from "@/services/background-proxy"

// Pin the notes authority scope like the other Notes stage suites (see stage44).
// Without it the notes list never loads when this file runs on its own.
const notesAuthority = vi.hoisted(() => ({ userId: 1 }))
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
    getCurrentUser: mockCurrentUser
  }
}))

vi.mock("@/components/Notes/hooks/useNotesGraphAuthorityScope", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/components/Notes/hooks/useNotesGraphAuthorityScope")>()
  return {
    ...actual,
    useNotesGraphAuthorityScope: () =>
      actual.createNotesGraphAuthorityScope(notesConnectionConfig.serverUrl, notesAuthority.userId)
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
  mockShowUndoNotification,
  mockCurrentUser,
  mockMessageOpen,
  mockMessageDestroy
} = vi.hoisted(() => {
  return {
    mockBgRequest: vi.fn<(request: BgRequest) => Promise<unknown>>(),
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
    mockShowUndoNotification: vi.fn(),
    mockCurrentUser: vi.fn<() => Promise<{ id: number; is_active: boolean }>>(),
    mockMessageOpen: vi.fn<(options: { key: string; content: React.ReactNode }) => void>(),
    mockMessageDestroy: vi.fn()
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
    info: mockMessageInfo,
    open: mockMessageOpen,
    destroy: mockMessageDestroy
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
    onRestoreNote,
    onToggleBulkSelection
  }: {
    notes?: Array<{ id: string | number; title?: string; version?: number; deleted?: boolean }>
    bulkSelectedIds?: string[]
    onSelectNote?: (id: string | number) => void
    onRestoreNote?: (id: string | number, version?: number) => void
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
          {note.deleted && <button type="button" data-testid={`mock-restore-${String(note.id)}`} onClick={() => onRestoreNote?.(note.id, note.version)}>Restore {String(note.id)}</button>}
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
  deleted?: boolean
}

type BgRequest = Pick<BgRequestInit, "servicePromptConfig" | "abortSignal"> & {
  path?: string
  method?: string
  headers?: Record<string, string>
  body?: { keywords?: string[]; title?: string; content?: string }
}

let serverNotes: ServerNote[] = []
/** Keywords the list endpoint reports; lets a test make list rows stale. */
let listKeywordsOverride: Record<string, string[]> = {}
let patchFailures: Map<string, Error> = new Map()
let listVersionlessIds = new Set<string>()
let notesByOwner = new Map<string, ServerNote[]>()
let initialServerNotes: ServerNote[] = []
const originalOwner = "https://notes.example.test|1"
const ownerKey = (serverUrl: string, userId: string | number) => `${serverUrl}|${userId}`

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
  deleted: Boolean(note.deleted),
  last_modified: "2026-10-01T00:00:00Z",
  keywords: keywordRows(keywords)
})

const findServerNote = (id: string) => serverNotes.find((note) => note.id === id)

const notesServerResponse = async (request: BgRequest): Promise<unknown> => {
    const key = ownerKey(request.servicePromptConfig?.serverUrl ?? notesConnectionConfig.serverUrl, request.servicePromptConfig?.expectedUserId ?? notesAuthority.userId)
    let notes = notesByOwner.get(key)
    if (!notes) {
      notes = initialServerNotes.map(note => ({ ...note, keywords: ["other owner"] }))
      notesByOwner.set(key, notes)
    }
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
      const items = notes.filter(note => !note.deleted).map((note) => ({
        ...toNoteResponse(note, listKeywordsOverride[note.id] ?? note.keywords),
        ...(listVersionlessIds.has(note.id) ? { version: undefined } : {})
      }))
      return {
        items,
        pagination: { total: items.length, limit: 20, offset: 0, has_more: false }
      }
    }
    if (path.startsWith("/api/v1/notes/trash?") && method === "GET") {
      const items = notes.filter(note => note.deleted).map(note => toNoteResponse(note))
      return { items, pagination: { total_items: items.length } }
    }
    const restoreMatch = path.match(/^\/api\/v1\/notes\/([^/]+)\/restore\?expected_version=(\d+)$/)
    if (restoreMatch && method === "POST") {
      const note = notes.find(note => note.id === decodeURIComponent(restoreMatch[1]))
      if (!note || note.version !== Number(restoreMatch[2])) throw versionConflict(restoreMatch[1])
      note.deleted = false
      note.version += 1
      return toNoteResponse(note)
    }
    const noteMatch = path.match(/^\/api\/v1\/notes\/([^/?]+)(\?.*)?$/)
    const note = noteMatch ? notes.find(note => note.id === decodeURIComponent(noteMatch[1])) : undefined
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
    if (note && method === "DELETE") {
      if (Number(request.headers?.["expected-version"]) !== note.version) throw versionConflict(note.id)
      note.deleted = true
      note.version += 1
      return undefined
    }
    if (note && method === "PUT") {
      if (Number(request.headers?.["expected-version"]) !== note.version) throw versionConflict(note.id)
      note.title = request.body?.title ?? note.title
      note.content = request.body?.content ?? note.content
      if (Array.isArray(request.body?.keywords)) note.keywords = [...request.body.keywords]
      note.version += 1
      return toNoteResponse(note)
    }
    return {}
}

const installNotesServer = () => mockBgRequest.mockImplementation(notesServerResponse)

const renderPage = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false }
    }
  })
  const page = () => <QueryClientProvider client={queryClient}><NotesManagerPage /></QueryClientProvider>
  const view = render(page())
  return { ...view, rerenderPage: () => view.rerender(page()) }
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
    notesAuthority.userId = 1
    notesConnectionConfig.serverUrl = "https://notes.example.test"
    mockCurrentUser.mockImplementation(async () => ({ id: notesAuthority.userId, is_active: true }))
    mockPromptBulkAddTags.mockReset()
    listVersionlessIds = new Set()
    notesByOwner = new Map()
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
    initialServerNotes = serverNotes.map(note => ({ ...note, keywords: [...note.keywords] }))
    notesByOwner.set(originalOwner, serverNotes)
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

  describe("Notes mutator original ownership", () => {
    let user: ReturnType<typeof userEvent.setup>
    beforeEach(() => { user = userEvent.setup() })
    const deferred = <T,>() => {
      let resolve!: (value: T) => void
      const promise = new Promise<T>(done => { resolve = done })
      return { promise, resolve }
    }
    const requests = (method: string) => mockBgRequest.mock.calls.map(([request]) => request)
      .filter(request => String(request.method || "GET").toUpperCase() === method)
    const mutations = () => mockBgRequest.mock.calls.map(([request]) => request)
      .filter(request => ["PATCH", "DELETE", "POST"].includes(String(request.method || "GET").toUpperCase()) && String(request.path).startsWith("/api/v1/notes/"))
    const choose = async (ids = ["n1"]) => {
      for (const id of ids) await user.click(await screen.findByTestId(`mock-select-${id}`))
      expect(screen.getByTestId("notes-list-bulk-selected")).toHaveTextContent(ids.join("|"))
    }
    const open = async (id = "n1") => {
      await user.click(await screen.findByTestId(`mock-open-${id}`))
      await screen.findByDisplayValue(id === "n1" ? "Alpha" : "Beta")
    }
    const changeOwner = async (view: ReturnType<typeof renderPage>, change: string) => {
      if (change === "unmount") { view.unmount(); return }
      const changeOnce = async (away: boolean) => {
        await act(async () => {
          if (change.startsWith("account")) notesAuthority.userId = away ? 2 : 1
          else notesConnectionConfig.serverUrl = away ? "https://other-notes.example.test" : "https://notes.example.test"
          view.rerenderPage()
          window.dispatchEvent(change.startsWith("account") ? new Event("tldw:auth-principal-changed")
            : new CustomEvent("tldw:config-updated", { detail: { authorityChanged: true } }))
        })
      }
      await changeOnce(true)
      if (change.endsWith("return")) await changeOnce(false)
    }
    const gateRequest = (method: string, path: string, applied = false) => {
      const gate = deferred<void>()
      const seen: BgRequest[] = []
      mockBgRequest.mockImplementation(async request => {
        if (String(request.method || "GET").toUpperCase() === method && (method === "GET" ? String(request.path) === path : String(request.path).startsWith(path)) && seen.length === 0) {
          seen.push(request)
          const result = applied ? await notesServerResponse(request) : undefined
          await gate.promise
          // A transport may still deliver after abort; the page must fence publication.
          return applied ? result : notesServerResponse(request)
        }
        return notesServerResponse(request)
      })
      return { gate, seen }
    }
    const release = async (gate: ReturnType<typeof deferred<void>>) => {
      await act(async () => { gate.resolve() })
    }
    const clearMessages = () => {
      mockShowUndoNotification.mockClear()
      mockMessageSuccess.mockClear()
      mockMessageWarning.mockClear()
      mockMessageError.mockClear()
      mockMessageInfo.mockClear()
    }
    const expectOwned = (request: BgRequest) => {
      expect(request.servicePromptConfig).toMatchObject({ serverUrl: "https://notes.example.test", expectedUserId: 1 })
      expect(request.headers?.["X-TLDW-Expected-User-ID"]).toBe("1")
      expect(request.abortSignal).toBeInstanceOf(AbortSignal)
    }
    const offerUndo = async () => {
      await choose()
      mockPromptBulkAddTags.mockResolvedValueOnce(["gamma"])
      await user.click(screen.getByTestId("notes-bulk-assign-keywords"))
      return waitForUndoOffer()
    }

    it.each(["account-return", "server-return", "unmount"])("abandons Add tags after %s while its original prompt is pending", async change => {
      const view = renderPage()
      await choose()
      const picked = deferred<string[]>()
      mockPromptBulkAddTags.mockReturnValueOnce(picked.promise)
      await user.click(screen.getByTestId("notes-bulk-assign-keywords"))
      expect(mockPromptBulkAddTags).toHaveBeenCalledTimes(1)
      await changeOwner(view, change)
      const before = mutations().length
      await act(async () => { picked.resolve(["gamma"]) })
      expect(mutations()).toHaveLength(before)
      expect(findServerNote("n1")?.keywords).toEqual(["alpha", "beta"])
      expect(mockShowUndoNotification).not.toHaveBeenCalled()
    })

    it("adds tags only to the original targets when selection changes during the prompt", async () => {
      renderPage()
      await choose()
      const picked = deferred<string[]>()
      mockPromptBulkAddTags.mockReturnValueOnce(picked.promise)
      await user.click(screen.getByTestId("notes-bulk-assign-keywords"))
      expect(mockPromptBulkAddTags).toHaveBeenCalledTimes(1)
      await user.click(screen.getByTestId("notes-bulk-clear-selection"))
      await choose(["n2"])
      await act(async () => { picked.resolve(["gamma"]) })
      await waitForUndoOffer()
      expect(findServerNote("n1")?.keywords).toEqual(["alpha", "beta", "gamma"])
      expect(findServerNote("n2")?.keywords).toEqual([])
      expect(screen.getByTestId("notes-list-bulk-selected")).toHaveTextContent("n2")
    })

    it("does not open Add tags after the original discard confirmation loses its owner", async () => {
      const view = renderPage()
      await open()
      await user.clear(screen.getByDisplayValue("Alpha"))
      await user.clear(screen.getByPlaceholderText("Write your note here... (Markdown supported)"))
      await choose()
      const discard = deferred<boolean>()
      mockConfirmDanger.mockReturnValueOnce(discard.promise)
      await user.click(screen.getByTestId("notes-bulk-assign-keywords"))
      await waitFor(() => expect(mockConfirmDanger).toHaveBeenCalled())
      await changeOwner(view, "account-return")
      await act(async () => { discard.resolve(true) })
      expect(mockPromptBulkAddTags).not.toHaveBeenCalled()
      expect(mutations()).toHaveLength(0)
    })

    it("does not PATCH after a fresh Add tags GET crosses a server away-and-return", async () => {
      const view = renderPage()
      await choose()
      const read = gateRequest("GET", "/api/v1/notes/n1")
      mockPromptBulkAddTags.mockResolvedValueOnce(["gamma"])
      await user.click(screen.getByTestId("notes-bulk-assign-keywords"))
      await waitFor(() => expect(read.seen).toHaveLength(1))
      await changeOwner(view, "server-return")
      await release(read.gate)
      expect(requests("PATCH")).toHaveLength(0)
      expect(findServerNote("n1")?.keywords).toEqual(["alpha", "beta"])
      expect(mockShowUndoNotification).not.toHaveBeenCalled()
    })

    it("does not publish an Add tags Undo offer after a late applied PATCH loses its owner", async () => {
      const view = renderPage()
      await choose()
      const write = gateRequest("PATCH", "/api/v1/notes/n1", true)
      mockPromptBulkAddTags.mockResolvedValueOnce(["gamma"])
      await user.click(screen.getByTestId("notes-bulk-assign-keywords"))
      await waitFor(() => expect(write.seen).toHaveLength(1))
      await changeOwner(view, "account-return")
      clearMessages()
      const before = requests("GET").length
      await release(write.gate)
      expect(mockShowUndoNotification).not.toHaveBeenCalled()
      expect(requests("GET")).toHaveLength(before)
      expect(write.seen[0].abortSignal?.aborted).toBe(true)
    })

    it("keeps a newly edited selected note intact while Add tags finishes", async () => {
      renderPage()
      await open()
      await choose()
      const read = gateRequest("GET", "/api/v1/notes/n1")
      mockPromptBulkAddTags.mockResolvedValueOnce(["gamma"])
      await user.click(screen.getByTestId("notes-bulk-assign-keywords"))
      await waitFor(() => expect(read.seen).toHaveLength(1))
      const body = screen.getByPlaceholderText("Write your note here... (Markdown supported)")
      await user.clear(body)
      await user.type(body, "Newer local draft")
      await release(read.gate)
      await waitForUndoOffer()
      expect(body).toHaveValue("Newer local draft")
      expect(findServerNote("n1")?.keywords).toEqual(["alpha", "beta", "gamma"])
    })

    it("uses the original owner, request signal, and tag/version CAS for Add tags and legitimate Undo", async () => {
      renderPage()
      const before = mockBgRequest.mock.calls.length
      const undo = await offerUndo()
      await act(async () => { await undo.onUndo() })
      const operationRequests = mockBgRequest.mock.calls.slice(before).map(([request]) => request)
        .filter(request => /^\/api\/v1\/notes\/n1(?:\?|$)/.test(String(request.path)))
      expect(operationRequests.length).toBeGreaterThanOrEqual(4)
      for (const request of operationRequests) expectOwned(request)
      expect(patchCallsFor("n1").map(request => request.headers?.["expected-version"])).toEqual(["3", "3"])
      expect(findServerNote("n1")?.keywords).toEqual(["alpha", "beta"])
    })

    it.each(["account", "server-return", "unmount"])("rejects a retained Add tags Undo callback after %s without a foreign write", async change => {
      const view = renderPage()
      const undo = await offerUndo()
      await changeOwner(view, change)
      const before = mutations().length
      const readsBefore = requests("GET").length
      await expect(undo.onUndo()).rejects.toThrow()
      expect(mutations()).toHaveLength(before)
      if (change === "account") expect(requests("GET")).toHaveLength(readsBefore)
      expect(findServerNote("n1")?.keywords).toEqual(["alpha", "beta", "gamma"])
    })

    it.each(["mismatch", "inactive"])("rejects Add tags Undo when fresh principal validation returns %s", async status => {
      renderPage()
      const undo = await offerUndo()
      mockCurrentUser.mockResolvedValue(status === "mismatch" ? { id: 2, is_active: true } : { id: 1, is_active: false })
      const before = mutations().length
      await expect(undo.onUndo()).rejects.toThrow()
      expect(mutations()).toHaveLength(before)
      expect(findServerNote("n1")?.keywords).toEqual(["alpha", "beta", "gamma"])
    })

    it("rejects Undo after the original detail GET loses its owner", async () => {
      const view = renderPage()
      const undo = await offerUndo()
      const read = gateRequest("GET", "/api/v1/notes/n1")
      const pending = undo.onUndo().then(() => true, () => false)
      await waitFor(() => expect(read.seen).toHaveLength(1))
      await changeOwner(view, "account-return")
      const before = requests("PATCH").length
      let restored = true
      await act(async () => { read.gate.resolve(); restored = await pending })
      expect(restored).toBe(false)
      expect(requests("PATCH")).toHaveLength(before)
    })

    it("does not Undo the open note after newer typing during its detail GET", async () => {
      renderPage()
      await open()
      const undo = await offerUndo()
      const read = gateRequest("GET", "/api/v1/notes/n1")
      const pending = undo.onUndo().then(() => true, () => false)
      await waitFor(() => expect(read.seen).toHaveLength(1))
      const body = screen.getByPlaceholderText("Write your note here... (Markdown supported)")
      await user.clear(body)
      await user.type(body, "Newer draft during Undo")
      const before = requests("PATCH").length
      let restored = true
      await act(async () => { read.gate.resolve(); restored = await pending })
      expect(restored).toBe(false)
      expect(requests("PATCH")).toHaveLength(before)
      expect(body).toHaveValue("Newer draft during Undo")
      expect(findServerNote("n1")?.keywords).toEqual(["alpha", "beta", "gamma"])
    })

    it("rejects a late Undo acknowledgement after the owner changed without refreshing the new editor", async () => {
      const view = renderPage()
      const undo = await offerUndo()
      const write = gateRequest("PATCH", "/api/v1/notes/n1", true)
      const pending = undo.onUndo().then(() => true, () => false)
      await waitFor(() => expect(write.seen).toHaveLength(1))
      await changeOwner(view, "account-return")
      await open("n2")
      const before = requests("GET").length
      let restored = true
      await act(async () => { write.gate.resolve(); restored = await pending })
      expect(restored).toBe(false)
      expect(requests("GET")).toHaveLength(before)
      expect(screen.getByDisplayValue("Beta")).toBeInTheDocument()
    })

    it.each(["account-return", "server", "unmount"])("abandons bulk Delete after %s while its confirmation is pending", async change => {
      const view = renderPage()
      await choose()
      const confirm = deferred<boolean>()
      mockConfirmDanger.mockReturnValueOnce(confirm.promise)
      await user.click(screen.getByTestId("notes-bulk-delete"))
      await waitFor(() => expect(mockConfirmDanger).toHaveBeenCalledWith(expect.objectContaining({ title: "Delete selected notes?" })))
      await changeOwner(view, change)
      const before = requests("DELETE").length
      await act(async () => { confirm.resolve(true) })
      expect(requests("DELETE")).toHaveLength(before)
      expect(findServerNote("n1")?.deleted).not.toBe(true)
      expect(mockMessageSuccess).not.toHaveBeenCalled()
    })

    it("does not DELETE after its missing-version lookup crosses an owner change", async () => {
      listVersionlessIds.add("n1")
      const view = renderPage()
      await choose()
      const read = gateRequest("GET", "/api/v1/notes/n1")
      await user.click(screen.getByTestId("notes-bulk-delete"))
      await waitFor(() => expect(read.seen).toHaveLength(1))
      await changeOwner(view, "server-return")
      await release(read.gate)
      expect(requests("DELETE")).toHaveLength(0)
      expect(findServerNote("n1")?.deleted).not.toBe(true)
    })

    it("does not publish a late bulk Delete success or clear a newer selected editor", async () => {
      const view = renderPage()
      await choose()
      const write = gateRequest("DELETE", "/api/v1/notes/n1", true)
      await user.click(screen.getByTestId("notes-bulk-delete"))
      await waitFor(() => expect(write.seen).toHaveLength(1))
      await changeOwner(view, "account-return")
      await open("n2")
      clearMessages()
      const before = requests("GET").length
      await release(write.gate)
      expect(mockMessageSuccess).not.toHaveBeenCalled()
      expect(requests("GET")).toHaveLength(before)
      expect(screen.getByDisplayValue("Beta")).toBeInTheDocument()
    })

    it("uses the original owner and expected version for a legitimate bulk Delete", async () => {
      renderPage()
      await choose()
      await user.click(screen.getByTestId("notes-bulk-delete"))
      await waitFor(() => expect(requests("DELETE")).toHaveLength(1))
      expectOwned(requests("DELETE")[0])
      expect(requests("DELETE")[0].headers?.["expected-version"]).toBe("3")
      expect(findServerNote("n1")?.deleted).toBe(true)
    })

    it("abandons a single Delete after its original confirmation loses its owner", async () => {
      const view = renderPage()
      await open()
      const confirm = deferred<boolean>()
      mockConfirmDanger.mockReturnValueOnce(confirm.promise)
      await user.click(screen.getByTestId("notes-overflow-menu-button"))
      await user.click(await screen.findByRole("menuitem", { name: "Delete" }))
      await waitFor(() => expect(mockConfirmDanger).toHaveBeenCalledWith(expect.objectContaining({ content: expect.stringContaining("to Trash?") })))
      await changeOwner(view, "account-return")
      await open("n2")
      await act(async () => { confirm.resolve(true) })
      expect(requests("DELETE")).toHaveLength(0)
      expect(screen.getByDisplayValue("Beta")).toBeInTheDocument()
    })

    it("does not publish a late applied single Delete acknowledgement into a newer owner/editor", async () => {
      const view = renderPage()
      await open()
      const write = gateRequest("DELETE", "/api/v1/notes/n1", true)
      await user.click(screen.getByTestId("notes-overflow-menu-button"))
      await user.click(await screen.findByRole("menuitem", { name: "Delete" }))
      await waitFor(() => expect(write.seen).toHaveLength(1))
      await changeOwner(view, "account-return")
      await open("n2")
      clearMessages()
      mockMessageOpen.mockClear()
      const before = requests("GET").length
      await release(write.gate)
      expect(mockMessageOpen).not.toHaveBeenCalled()
      expect(mockMessageSuccess).not.toHaveBeenCalled()
      expect(requests("GET")).toHaveLength(before)
      expect(screen.getByDisplayValue("Beta")).toBeInTheDocument()
    })

    it("preserves a newer same-owner Beta draft after an already-applied single Delete acknowledgement", async () => {
      renderPage()
      await open()
      const write = gateRequest("DELETE", "/api/v1/notes/n1", true)
      await user.click(screen.getByTestId("notes-overflow-menu-button"))
      await user.click(await screen.findByRole("menuitem", { name: "Delete" }))
      await waitFor(() => expect(write.seen).toHaveLength(1))
      await open("n2")
      const body = screen.getByPlaceholderText("Write your note here... (Markdown supported)")
      await user.clear(body)
      await user.type(body, "Newer Beta draft after Delete")
      await release(write.gate)
      expect(screen.queryByDisplayValue("Beta")).toBeInTheDocument()
      expect(screen.getByPlaceholderText("Write your note here... (Markdown supported)")).toHaveValue("Newer Beta draft after Delete")
      expect(findServerNote("n1")?.deleted).toBe(true)
    })

    it("uses the original owner and versions for a legitimate single Delete and actual toast Undo", async () => {
      renderPage()
      await open()
      await user.click(screen.getByTestId("notes-overflow-menu-button"))
      await user.click(await screen.findByRole("menuitem", { name: "Delete" }))
      await waitFor(() => expect(mockMessageOpen).toHaveBeenCalledTimes(1))
      expect(findServerNote("n1")?.deleted).toBe(true)
      const toast = mockMessageOpen.mock.calls[0][0]
      render(<>{toast.content}</>)
      await user.click(screen.getByTestId("notes-delete-undo-n1"))
      await screen.findByDisplayValue("Alpha")
      const deleted = requests("DELETE")[0]
      const trashRead = requests("GET").find(request => String(request.path).startsWith("/api/v1/notes/trash?"))
      const restored = requests("POST").find(request => String(request.path).startsWith("/api/v1/notes/n1/restore?"))
      expect(trashRead).toBeDefined()
      expect(restored).toBeDefined()
      expectOwned(deleted)
      if (trashRead) expectOwned(trashRead)
      if (restored) expectOwned(restored)
      expect(deleted.headers?.["expected-version"]).toBe("3")
      expect(restored?.path).toBe("/api/v1/notes/n1/restore?expected_version=4")
      expect(findServerNote("n1")).toMatchObject({ deleted: false, version: 5, keywords: ["alpha", "beta"] })
      expect(mockMessageSuccess).toHaveBeenCalledWith("Note restored")
    })

    it("uses the original owner, signal, and deleted version for a legitimate direct Restore", async () => {
      findServerNote("n1")!.deleted = true
      renderPage()
      await user.click(await screen.findByTestId("notes-mode-trash"))
      await user.click(await screen.findByTestId("mock-restore-n1"))
      await screen.findByDisplayValue("Alpha")
      const restored = requests("POST").find(request => String(request.path).startsWith("/api/v1/notes/n1/restore?"))
      expect(restored).toBeDefined()
      if (restored) expectOwned(restored)
      expect(restored?.path).toBe("/api/v1/notes/n1/restore?expected_version=3")
      expect(findServerNote("n1")).toMatchObject({ deleted: false, version: 4, keywords: ["alpha", "beta"] })
      expect(mockMessageSuccess).toHaveBeenCalledWith("Note restored")
    })

    it("preserves a newer same-owner Beta draft after an already-applied direct Restore acknowledgement", async () => {
      findServerNote("n1")!.deleted = true
      renderPage()
      await user.click(await screen.findByTestId("notes-mode-trash"))
      const restore = gateRequest("POST", "/api/v1/notes/n1/restore", true)
      await user.click(await screen.findByTestId("mock-restore-n1"))
      await waitFor(() => expect(restore.seen).toHaveLength(1))
      await user.click(screen.getByTestId("notes-mode-active"))
      await open("n2")
      const body = screen.getByPlaceholderText("Write your note here... (Markdown supported)")
      await user.clear(body)
      await user.type(body, "Newer Beta draft after Restore")
      await release(restore.gate)
      expect(screen.queryByDisplayValue("Beta")).toBeInTheDocument()
      expect(screen.getByPlaceholderText("Write your note here... (Markdown supported)")).toHaveValue("Newer Beta draft after Restore")
      expect(findServerNote("n1")?.deleted).toBe(false)
    })

    describe("Notes conflict Reload original ownership", () => {
      it("continues owned Add tags after the dirty-note dialog successfully saves the original draft", async () => {
        renderPage()
        await open()
        await choose()
        const body = screen.getByPlaceholderText("Write your note here... (Markdown supported)")
        await user.clear(body)
        await user.type(body, "Original draft saved before Add tags")
        let failFirstSave = true
        mockBgRequest.mockImplementation(async request => {
          if (request.method === "PUT" && failFirstSave) {
            failFirstSave = false
            throw Object.assign(new Error("The first save needs correction"), { status: 400 })
          }
          return notesServerResponse(request)
        })
        mockPromptBulkAddTags.mockResolvedValueOnce(["gamma"])
        await user.click(screen.getByTestId("notes-bulk-assign-keywords"))
        await user.click(await screen.findByRole("button", { name: "Try saving again" }))
        await waitForUndoOffer()
        const saves = requests("PUT")
        expect(saves).toHaveLength(2)
        for (const request of saves) expectOwned(request)
        expect(saves.map(request => request.headers?.["expected-version"])).toEqual(["3", "3"])
        expect(saves[1].body?.content).toBe("Original draft saved before Add tags")
        expect(patchCallsFor("n1").at(-1)?.headers?.["expected-version"]).toBe("4")
        expect(findServerNote("n1")).toMatchObject({ title: "Alpha", content: "Original draft saved before Add tags", version: 4, keywords: ["alpha", "beta", "gamma"] })
        expect(screen.getByDisplayValue("Alpha")).toBeInTheDocument()
        expect(body).toHaveValue("Original draft saved before Add tags")
      })

      it("does not publish an Alpha DELETE conflict after a newer Beta selection and draft", async () => {
        renderPage()
        await open()
        const gate = deferred<void>()
        const seen: BgRequest[] = []
        mockBgRequest.mockImplementation(async request => {
          if (request.method === "DELETE" && String(request.path).startsWith("/api/v1/notes/n1?") && seen.length === 0) {
            seen.push(request)
            await gate.promise
          }
          return notesServerResponse(request)
        })
        await user.click(screen.getByTestId("notes-overflow-menu-button"))
        await user.click(await screen.findByRole("menuitem", { name: "Delete" }))
        await waitFor(() => expect(seen).toHaveLength(1))
        const alpha = findServerNote("n1")
        expect(alpha).toBeDefined()
        if (!alpha) throw new Error("Alpha server note missing")
        alpha.version += 1
        await open("n2")
        const body = screen.getByPlaceholderText("Write your note here... (Markdown supported)")
        await user.clear(body)
        await user.type(body, "Newer Beta draft during Alpha conflict")
        mockMessageError.mockClear()
        await release(gate)
        expect(mockMessageError).not.toHaveBeenCalled()
        expect(screen.getByDisplayValue("Beta")).toBeInTheDocument()
        expect(body).toHaveValue("Newer Beta draft during Alpha conflict")
        expect(findServerNote("n1")?.deleted).not.toBe(true)
        expect(seen[0].headers?.["expected-version"]).toBe("3")
      })
    })

    it("does not publish a late Restore acknowledgement into a newer owner/editor", async () => {
      findServerNote("n1")!.deleted = true
      const view = renderPage()
      await user.click(await screen.findByTestId("notes-mode-trash"))
      const restore = gateRequest("POST", "/api/v1/notes/n1/restore", true)
      await user.click(await screen.findByTestId("mock-restore-n1"))
      await waitFor(() => expect(restore.seen).toHaveLength(1))
      await changeOwner(view, "account-return")
      await user.click(screen.getByTestId("notes-mode-active"))
      await open("n2")
      clearMessages()
      const before = requests("GET").length
      await release(restore.gate)
      expect(mockMessageSuccess).not.toHaveBeenCalled()
      expect(requests("GET")).toHaveLength(before)
      expect(screen.getByDisplayValue("Beta")).toBeInTheDocument()
    })

    it("does not restore through a prior single Delete toast after the owner changed", async () => {
      const view = renderPage()
      await open()
      await user.click(screen.getByTestId("notes-overflow-menu-button"))
      await user.click(await screen.findByRole("menuitem", { name: "Delete" }))
      await waitFor(() => expect(mockMessageOpen).toHaveBeenCalledTimes(1))
      const toast = mockMessageOpen.mock.calls[0][0]
      await changeOwner(view, "account-return")
      const before = requests("POST").length
      render(<>{toast.content}</>)
      await user.click(screen.getByTestId("notes-delete-undo-n1"))
      await act(async () => {})
      expect(requests("POST")).toHaveLength(before)
      expect(findServerNote("n1")?.deleted).toBe(true)
      expect(mockMessageSuccess).not.toHaveBeenCalled()
    })
  })

})
