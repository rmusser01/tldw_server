import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import NotesManagerPage from "../NotesManagerPage"
import type { NotesLeaveGuardState } from "../hooks/useNotesEditorState"

// The Notes page on the save state machine (#3102: NS-01, NS-03, NS-N2,
// NS-05, NS-02, NS-06).

// Pin the notes authority scope like the other Notes stage suites, so the
// list and saves never depend on ambient connection state.
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
  server: {
    note: { id: 11, title: "Saved note", content: "Saved body", version: 1 },
    created: 0
  }
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, defaultValueOrOptions?: string | { defaultValue?: string }) => {
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

vi.mock("@/hooks/useCallerCapabilities", () => {
  const capabilities = { monitoringAlerts: "denied", userId: 1, refreshAfterForbidden: vi.fn(async () => undefined) }
  return { useCallerCapabilities: () => capabilities }
})

vi.mock("@/components/Notes/NotesListPanel", () => ({
  default: ({ onSelectNote }: { onSelectNote: (id: string) => void }) => (
    <button data-testid="notes-list-panel" onClick={() => onSelectNote("11")}>
      Open saved note
    </button>
  )
}))

type NoteBody = { title?: string; content?: string; auto_title?: boolean; [key: string]: unknown }
type RequestInit = { path?: string; method?: string; headers?: Record<string, string>; body?: NoteBody }

const EDITOR_PLACEHOLDER = "Write your note here... (Markdown supported)"

const requestsTo = (method: string, path: string) =>
  mockBgRequest.mock.calls
    .map(([request]) => request as RequestInit)
    .filter((request) => String(request.method || "GET").toUpperCase() === method && request.path === path)

const serverResponds = async (request: RequestInit) => {
  const path = String(request.path || "")
  const method = String(request.method || "GET").toUpperCase()
  if (path.startsWith("/api/v1/notes/?")) {
    return { items: [], pagination: { total_items: 0, total_pages: 1 } }
  }
  if (path === "/api/v1/notes/" && method === "POST") {
    server.created += 1
    const body = request.body || {}
    return {
      id: 50 + server.created,
      title: body.title || "Named by the server",
      content: body.content,
      version: 1,
      last_modified: "2026-10-03T12:00:00.000Z"
    }
  }
  if (/^\/api\/v1\/notes\/5\d$/.test(path) && method === "GET") {
    const created = requestsTo("POST", "/api/v1/notes/").at(-1)?.body || {}
    return {
      id: Number(path.split("/").at(-1)),
      title: created.title || "Named by the server",
      content: created.content,
      metadata: { keywords: [] },
      version: 1,
      last_modified: "2026-10-03T12:00:00.000Z"
    }
  }
  if (path === "/api/v1/notes/11" && method === "GET") {
    return { ...server.note, metadata: { keywords: [] }, last_modified: "2026-10-03T11:00:00.000Z" }
  }
  if (path === "/api/v1/notes/11" && method === "PUT") {
    if (Number(request.headers?.["expected-version"]) !== server.note.version) {
      throw { status: 409, message: "Version conflict (PUT /api/v1/notes/11)" }
    }
    server.note = {
      ...server.note,
      content: String(request.body?.content),
      version: server.note.version + 1
    }
    return { ...server.note, last_modified: "2026-10-03T12:30:00.000Z" }
  }
  return {}
}

const leaveGuard: { current: NotesLeaveGuardState | null } = { current: null }
const CaptureLeaveGuard: React.FC<NotesLeaveGuardState> = (props) => {
  leaveGuard.current = props
  return null
}

const renderPage = () => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } }
  })
  return render(
    <QueryClientProvider client={queryClient}>
      <NotesManagerPage LeaveGuard={CaptureLeaveGuard} />
    </QueryClientProvider>
  )
}

const openSavedNote = async () => {
  fireEvent.click(screen.getByText("Open saved note"))
  await waitFor(() => expect(screen.getByPlaceholderText(EDITOR_PLACEHOLDER)).toHaveValue(server.note.content))
  await waitFor(() => expect(screen.getByTestId("notes-save-status")).toHaveAttribute("data-state", "saved"))
}

const type = (value: string) => {
  fireEvent.change(screen.getByPlaceholderText(EDITOR_PLACEHOLDER), { target: { value } })
}

// antd-heavy page: each interaction can take seconds on a loaded runner.
describe("NotesManagerPage save state machine", { timeout: 60_000 }, () => {
  beforeEach(() => {
    vi.clearAllMocks()
    localStorage.clear()
    leaveGuard.current = null
    server.note = { id: 11, title: "Saved note", content: "Saved body", version: 1 }
    server.created = 0
    mockConfirmDanger.mockResolvedValue(true)
    mockGetSetting.mockResolvedValue(null)
    mockClearSetting.mockResolvedValue(undefined)
    mockBgRequest.mockImplementation(serverResponds)
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText: vi.fn(async () => undefined) }
    })
  })

  afterEach(() => {
    Reflect.deleteProperty(navigator, "clipboard")
  })

  it("flushes an edit typed just before navigating away, then lets the navigation go (NS-01)", async () => {
    renderPage()
    await openSavedNote()
    expect(leaveGuard.current?.when).toBe(false)

    type("Typed one second before clicking Chat")
    expect(leaveGuard.current?.when).toBe(true)

    let leave = false
    await act(async () => {
      leave = (await leaveGuard.current?.onLeave()) ?? false
    })

    expect(leave).toBe(true)
    const [put] = requestsTo("PUT", "/api/v1/notes/11")
    expect(put.body?.content).toBe("Typed one second before clicking Chat")
    expect(put.headers?.["expected-version"]).toBe("1")
    expect(leaveGuard.current?.when).toBe(false)
  })

  it("guards a full page unload while edits are unsaved", async () => {
    renderPage()
    await openSavedNote()
    type("Not saved yet")

    const unload = new Event("beforeunload", { cancelable: true })
    window.dispatchEvent(unload)

    expect(unload.defaultPrevented).toBe(true)
  })

  it("says Saved only after the server acknowledges, and announces outcomes only (NS-06)", async () => {
    let acknowledge!: () => void
    mockBgRequest.mockImplementation(async (request: RequestInit) => {
      if (String(request.method).toUpperCase() === "PUT") {
        await new Promise<void>((resolve) => {
          acknowledge = resolve
        })
      }
      return serverResponds(request)
    })
    renderPage()
    await openSavedNote()

    type("An edit to save")
    const status = screen.getByTestId("notes-save-status")
    const announcement = screen.getByTestId("notes-save-status-announcement")
    expect(status).toHaveAttribute("data-state", "dirty")
    expect(announcement).toBeEmptyDOMElement()
    expect(screen.queryByTestId("notes-save-feedback")).not.toBeInTheDocument()

    fireEvent.click(screen.getByTestId("notes-save-button"))
    await waitFor(() => expect(status).toHaveAttribute("data-state", "saving"))
    expect(status).not.toHaveTextContent(/^Saved/)
    expect(announcement).toBeEmptyDOMElement()

    await act(async () => {
      acknowledge()
    })
    await waitFor(() => expect(status).toHaveAttribute("data-state", "saved"))
    expect(announcement).toHaveTextContent(/^Saved/)
    expect(screen.getByTestId("notes-editor-revision-meta")).toHaveTextContent("Version 2")
    expect(screen.getByTestId("notes-editor-revision-meta")).toHaveTextContent("Stored on notes.example.test")
  })

  it("makes Save the primary action only while there are unsaved edits (NS-06)", async () => {
    renderPage()
    await openSavedNote()
    expect(screen.getByTestId("notes-save-button")).not.toHaveClass("ant-btn-primary")

    type("Now there is something to save")

    expect(screen.getByTestId("notes-save-button")).toHaveClass("ant-btn-primary")
  })

  it("turns a 409 into one conflict panel and resolves it with the server's latest version (NS-03)", async () => {
    renderPage()
    await openSavedNote()
    server.note = { ...server.note, content: "Their body", version: 2 }

    type("My body")
    fireEvent.click(screen.getByTestId("notes-save-button"))

    const panel = await screen.findByTestId("notes-save-issue")
    expect(panel).toHaveAttribute("data-kind", "conflict")
    expect(panel).toHaveTextContent(/changed in another tab or device/i)
    expect(panel).not.toHaveTextContent(/connection/i)
    expect(screen.getByTestId("notes-save-status")).toHaveAttribute("data-state", "conflict")
    expect(screen.queryByTestId("notes-save-feedback")).not.toBeInTheDocument()
    expect(mockMessageError).not.toHaveBeenCalled()
    expect(screen.getAllByTestId("notes-save-issue")).toHaveLength(1)

    // Save again never resends the stale version.
    fireEvent.click(screen.getByTestId("notes-save-button"))
    await act(async () => {})
    expect(requestsTo("PUT", "/api/v1/notes/11")).toHaveLength(1)

    fireEvent.click(within(panel).getByTestId("notes-conflict-keep-mine"))

    await waitFor(() => expect(screen.getByTestId("notes-save-status")).toHaveAttribute("data-state", "saved"))
    const puts = requestsTo("PUT", "/api/v1/notes/11")
    expect(puts).toHaveLength(2)
    expect(puts[1].headers?.["expected-version"]).toBe("2")
    expect(server.note).toMatchObject({ content: "My body", version: 3 })
    expect(screen.queryByTestId("notes-save-issue")).not.toBeInTheDocument()
  })

  it("lets the user take their version after a conflict, copying the local text first", async () => {
    renderPage()
    await openSavedNote()
    server.note = { ...server.note, content: "Their body", version: 2 }
    type("My body")
    fireEvent.click(screen.getByTestId("notes-save-button"))
    const panel = await screen.findByTestId("notes-save-issue")

    fireEvent.click(within(panel).getByTestId("notes-conflict-take-theirs"))

    await waitFor(() => expect(screen.getByPlaceholderText(EDITOR_PLACEHOLDER)).toHaveValue("Their body"))
    expect(navigator.clipboard.writeText).toHaveBeenCalledWith(expect.stringContaining("My body"))
    await waitFor(() => expect(screen.getByTestId("notes-save-status")).toHaveAttribute("data-state", "saved"))
    expect(screen.queryByTestId("notes-save-issue")).not.toBeInTheDocument()
  })

  it("copies my text from the conflict panel without leaving the conflict", async () => {
    renderPage()
    await openSavedNote()
    server.note = { ...server.note, content: "Their body", version: 2 }
    type("My body")
    fireEvent.click(screen.getByTestId("notes-save-button"))
    const panel = await screen.findByTestId("notes-save-issue")

    fireEvent.click(within(panel).getByTestId("notes-conflict-copy-mine"))

    await waitFor(() =>
      expect(navigator.clipboard.writeText).toHaveBeenCalledWith(expect.stringContaining("My body"))
    )
    expect(screen.getByTestId("notes-save-issue")).toHaveAttribute("data-kind", "conflict")
    expect(screen.getByPlaceholderText(EDITOR_PLACEHOLDER)).toHaveValue("My body")
  })

  it("lets the server name a note typed without a title (NS-02)", async () => {
    renderPage()
    type("First line names the note")

    fireEvent.click(screen.getByTestId("notes-save-button"))

    await waitFor(() => expect(requestsTo("POST", "/api/v1/notes/")).toHaveLength(1))
    const [post] = requestsTo("POST", "/api/v1/notes/")
    expect(post.body).toMatchObject({ auto_title: true })
    expect(post.body).not.toHaveProperty("title")
    await waitFor(() => expect(screen.getByPlaceholderText("Title")).toHaveValue("Named by the server"))
    expect(screen.queryByTestId("notes-save-issue")).not.toBeInTheDocument()
  })
})
