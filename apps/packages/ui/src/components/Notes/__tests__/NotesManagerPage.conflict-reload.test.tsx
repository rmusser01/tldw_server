import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import NotesManagerPage from "../NotesManagerPage"
import { NOTE_AUTOSAVE_DELAY_MS } from "../notes-manager-utils"

// NS-N1 (#3102): "Reload notes" in the 409 toast must load the server's note
// into the editor, so tab A's next autosave cannot overwrite tab B's change.

const {
  mockBgRequest,
  mockMessageSuccess,
  mockMessageError,
  mockMessageWarning,
  mockMessageInfo,
  mockNavigate,
  mockConfirmDanger,
  mockGetSetting,
  mockClearSetting,
  server
} = vi.hoisted(() => ({
  mockBgRequest: vi.fn(),
  mockMessageSuccess: vi.fn(),
  mockMessageError: vi.fn(),
  mockMessageWarning: vi.fn(),
  mockMessageInfo: vi.fn(),
  mockNavigate: vi.fn(),
  mockConfirmDanger: vi.fn(),
  mockGetSetting: vi.fn(),
  mockClearSetting: vi.fn(),
  server: { title: "", content: "", version: 0 }
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

vi.mock("@/components/Notes/NotesListPanel", () => ({
  default: ({ onSelectNote }: { onSelectNote: (id: string) => void }) => (
    <button data-testid="notes-list-panel" onClick={() => onSelectNote("11")}>
      Open shared note
    </button>
  )
}))

vi.mock("@/hooks/useCanonicalConnectionConfig", () => {
  const config = {
    serverUrl: "https://notes.test",
    authMode: "multi-user",
    accessToken: `test.${btoa(JSON.stringify({ sub: "7" }))}.signature`
  }
  return { useCanonicalConnectionConfig: () => ({ config, loading: false }) }
})
vi.mock("../hooks/useNotesGraphAuthorityScope", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../hooks/useNotesGraphAuthorityScope")>()
  return {
    ...actual,
    useNotesGraphAuthorityScope: () => actual.createNotesGraphAuthorityScope("https://notes.test", 7)
  }
})
vi.mock("@/services/tldw/TldwAuth", () => ({
  tldwAuth: { getCurrentUser: vi.fn(async () => ({ id: 7, is_active: true })) }
}))
vi.mock("@/hooks/useCallerCapabilities", () => {
  const capabilities = {
    monitoringAlerts: "allowed",
    userId: 7,
    refreshAfterForbidden: vi.fn(async () => undefined)
  }
  return { useCallerCapabilities: () => capabilities }
})

const EDITOR_PLACEHOLDER = "Write your note here... (Markdown supported)"
const STALE_LOCAL_TEXT = "Tab A stale local edit"
const OTHER_TAB_TEXT = "Tab B saved this change"
// The full page renders slowly under jsdom; each case drives several saves.
const PAGE_TEST_TIMEOUT_MS = 20_000

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

type NoteRequest = {
  path?: string
  method?: string
  headers?: Record<string, string>
  body?: { title?: string; content?: string }
}

const updateCalls = () =>
  mockBgRequest.mock.calls
    .map(([request]) => request as NoteRequest)
    .filter((request) => {
      const method = String(request?.method || "GET").toUpperCase()
      return String(request?.path || "") === "/api/v1/notes/11" && method === "PUT"
    })

const serveSharedNote = () => {
  server.title = "Shared note"
  server.content = "Original body"
  server.version = 1
  mockBgRequest.mockImplementation(async (request: NoteRequest) => {
    const path = String(request.path || "")
    const method = String(request.method || "GET").toUpperCase()

    if (path.startsWith("/api/v1/notes/?")) {
      return { items: [], pagination: { total_items: 0, total_pages: 1 } }
    }

    if (path === "/api/v1/notes/11" && method === "GET") {
      return {
        id: 11,
        title: server.title,
        content: server.content,
        metadata: { keywords: [] },
        version: server.version,
        last_modified: `2026-10-03T10:0${server.version}:00.000Z`
      }
    }

    if (path === "/api/v1/notes/11" && method === "PUT") {
      // Optimistic locking, as the server does it.
      if (Number(request.headers?.["expected-version"]) !== server.version) {
        throw { status: 409, message: "Version conflict" }
      }
      server.title = String(request.body?.title ?? server.title)
      server.content = String(request.body?.content ?? "")
      server.version += 1
      return {
        id: 11,
        version: server.version,
        last_modified: `2026-10-03T10:0${server.version}:00.000Z`
      }
    }

    return {}
  })
}

/** Tab B saves a change to the same note: the server moves to version 2. */
const saveFromOtherTab = () => {
  server.title = "Tab B title"
  server.content = OTHER_TAB_TEXT
  server.version = 2
}

const editor = () => screen.getByPlaceholderText(EDITOR_PLACEHOLDER)

/**
 * Open the note in "tab A", edit it, save manually and get the 409. The single
 * conflict panel (NS-03) replaced the toast; its "Use their version" is the
 * NS-N1 reload path.
 */
const reachConflictPanel = async () => {
  renderPage()
  fireEvent.click(screen.getByText("Open shared note"))
  await waitFor(() => expect(editor()).toHaveValue("Original body"))
  await waitFor(() =>
    expect(screen.getByTestId("notes-editor-revision-meta")).toHaveTextContent("Version 1")
  )

  saveFromOtherTab()

  fireEvent.change(editor(), { target: { value: STALE_LOCAL_TEXT } })
  fireEvent.click(screen.getByTestId("notes-save-button"))
  await waitFor(() => expect(updateCalls()).toHaveLength(1))

  const panel = await screen.findByTestId("notes-save-issue")
  expect(panel).toHaveAttribute("data-kind", "conflict")
  return within(panel).getByTestId("notes-conflict-take-theirs")
}

const stubClipboard = (writeText: (text: string) => Promise<void>) => {
  const spy = vi.fn(writeText)
  Object.defineProperty(navigator, "clipboard", {
    configurable: true,
    value: { writeText: spy }
  })
  return spy
}

describe("NotesManagerPage NS-N1 conflict reload", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    localStorage.clear()
    mockConfirmDanger.mockResolvedValue(true)
    mockGetSetting.mockResolvedValue(null)
    mockClearSetting.mockResolvedValue(undefined)
    serveSharedNote()
  })

  afterEach(async () => {
    cleanup()
    while (queryClients.length > 0) {
      const queryClient = queryClients.pop()
      if (!queryClient) continue
      await queryClient.cancelQueries()
      queryClient.clear()
    }
    vi.useRealTimers()
    Reflect.deleteProperty(navigator, "clipboard")
  })

  it("loads the other tab's change, clears the dirty state and never autosaves the stale text", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true })
    const writeText = stubClipboard(async () => undefined)

    const reloadAction = await reachConflictPanel()
    const putsBeforeReload = updateCalls().length
    fireEvent.click(reloadAction)

    await waitFor(() => expect(editor()).toHaveValue(OTHER_TAB_TEXT))
    expect(screen.getByPlaceholderText("Title")).toHaveValue("Tab B title")
    expect(screen.getByTestId("notes-editor-revision-meta")).toHaveTextContent("Version 2")
    expect(screen.getByTestId("notes-save-status")).toHaveAttribute("data-state", "saved")

    // The user's unsaved text is not dropped: it goes to the clipboard and a
    // toast says so.
    expect(writeText).toHaveBeenCalledTimes(1)
    expect(writeText.mock.calls[0][0]).toContain(STALE_LOCAL_TEXT)
    expect(mockMessageInfo).toHaveBeenCalledWith(
      expect.objectContaining({ content: expect.stringMatching(/copied to the clipboard/i) })
    )

    // Let tab A's autosave debounce run out.
    await act(async () => {
      vi.advanceTimersByTime(NOTE_AUTOSAVE_DELAY_MS * 2)
    })
    await act(async () => {})

    const stalePuts = updateCalls()
      .slice(putsBeforeReload)
      .filter((request) => String(request.body?.content).includes(STALE_LOCAL_TEXT))
    expect(stalePuts).toEqual([])
    expect(server.content).toBe(OTHER_TAB_TEXT)
  }, PAGE_TEST_TIMEOUT_MS)

  it("keeps the editor and its base version when the unsaved text cannot be copied and the user keeps editing", async () => {
    stubClipboard(async () => {
      throw new Error("Write permission denied")
    })
    // "Keep editing" in the "use their version" confirm.
    mockConfirmDanger.mockImplementation(
      async (options: { title?: string }) => options.title !== "Use their version?"
    )

    const reloadAction = await reachConflictPanel()
    fireEvent.click(reloadAction)

    await waitFor(() =>
      expect(mockConfirmDanger).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Use their version?",
          content: expect.stringMatching(/could not be copied/i)
        })
      )
    )
    await act(async () => {})
    expect(editor()).toHaveValue(STALE_LOCAL_TEXT)
    expect(screen.getByTestId("notes-editor-revision-meta")).toHaveTextContent("Version 1")
    // Still in the conflict: the local edits remain unsaved.
    expect(screen.getByTestId("notes-save-status")).toHaveAttribute("data-state", "conflict")

    // Save never resends the stale base version during a conflict (NS-03).
    fireEvent.click(screen.getByTestId("notes-save-button"))
    await act(async () => {})
    expect(updateCalls()).toHaveLength(1)
    expect(server.content).toBe(OTHER_TAB_TEXT)
  }, PAGE_TEST_TIMEOUT_MS)
})
