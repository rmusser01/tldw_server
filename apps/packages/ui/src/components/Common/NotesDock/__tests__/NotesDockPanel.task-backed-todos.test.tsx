import React from "react"
import { ConfigProvider } from "antd"
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { NotesDockPanel } from "../NotesDockPanel"
import { useNotesDockStore, type NotesDockNote } from "@/store/notes-dock"
import { NotesDockHost } from "../NotesDockHost"
import { getQueryClient } from "@/services/query-client"
import type { BgRequestInit } from "@/services/background-proxy"
import type { TldwConfig } from "@/services/tldw/TldwApiClient"

const {
  mockBgRequest,
  mockNavigate,
  stableTranslate,
  stableMessageApi
} = vi.hoisted(() => ({
  mockBgRequest: vi.fn(),
  mockNavigate: vi.fn(),
  stableTranslate: (
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
  },
  stableMessageApi: {
    error: vi.fn(),
    success: vi.fn(),
    warning: vi.fn(),
    info: vi.fn()
  }
}))

const dockAuthority = vi.hoisted(() => ({
  userId: 1,
  serverUrl: "https://notes.example.test",
  initialize: vi.fn(async () => undefined),
  ensureConfig: vi.fn<() => Promise<TldwConfig>>(),
  currentUser: vi.fn<() => Promise<{ id: number; is_active: boolean }>>(),
  watches: new Set<{ tldwConfig: (value: { newValue?: unknown }) => void }>()
}))
vi.mock("@/services/tldw/TldwApiClient", () => ({ tldwClient: {
  initialize: dockAuthority.initialize, ensureConfigForRequest: dockAuthority.ensureConfig
} }))
vi.mock("@/services/tldw/TldwAuth", () => ({ tldwAuth: { getCurrentUser: dockAuthority.currentUser } }))
vi.mock("@/services/tldw/deployment-mode", () => ({ isHostedTldwDeployment: () => false }))
vi.mock("@/utils/safe-storage", () => ({ createSafeStorage: () => ({
  get: vi.fn(async () => null), set: vi.fn(async () => undefined), remove: vi.fn(async () => undefined),
  watch: (watcher: { tldwConfig: (value: { newValue?: unknown }) => void }) => { dockAuthority.watches.add(watcher) },
  unwatch: (watcher: { tldwConfig: (value: { newValue?: unknown }) => void }) => { dockAuthority.watches.delete(watcher) }
}) }))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: stableTranslate
  })
}))

vi.mock("react-router-dom", () => ({
  useNavigate: () => mockNavigate
}))

vi.mock("@/hooks/useServerOnline", () => ({
  useServerOnline: () => true
}))

vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => ({
    capabilities: { hasNotes: true },
    loading: false
  })
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: mockBgRequest
}))

vi.mock("@/hooks/useAntdMessage", () => ({
  useAntdMessage: () => stableMessageApi
}))

class ResizeObserverMock {
  observe() {}
  unobserve() {}
  disconnect() {}
}

const dockContentEditor = () =>
  screen.getByPlaceholderText("Jot down notes, ideas, or observations...")

const seedDockNote = (overrides: Partial<NotesDockNote> = {}) => {
  const snapshot = overrides.snapshot ?? {
    title: "Dock task note",
    content: "- [ ] Draft PRD",
    keywords: ["planning"],
    version: 2
  }
  useNotesDockStore.setState({
    isOpen: true,
    position: { x: 24, y: 80 },
    size: { width: 640, height: 520 },
    notes: [
      {
        localId: "local-1",
        id: 101,
        title: "Dock task note",
        content: "- [ ] Draft PRD",
        keywords: ["planning"],
        version: 2,
        snapshot,
        isDirty: false,
        ...overrides
      }
    ],
    activeNoteId: "local-1"
  })
}

const setupTaskBackedDockMock = (options: {
  conflictOnSave?: boolean
  includeActivity?: boolean
} = {}) => {
  let noteContent = "- [ ] Draft PRD"
  let noteVersion = 2
  let taskStatus: "open" | "done" = "open"
  let taskVersion = 5
  let activityDismissed = false

  mockBgRequest.mockImplementation(async (request: { path?: string; method?: string; body?: any }) => {
    const path = String(request.path || "")
    const method = String(request.method || "GET").toUpperCase()

    if (path.startsWith("/api/v1/notes/?page=1")) {
      return { items: [{ id: 101, title: "Dock task note", content: noteContent, keywords: ["planning"], version: noteVersion }], pagination: { total_items: 1, total_pages: 1 } }
    }

    if (path === "/api/v1/notes/101" && method === "GET") {
      return {
        id: 101,
        title: "Dock task note",
        content: noteContent,
        keywords: ["planning"],
        version: noteVersion
      }
    }

    if (path.startsWith("/api/v1/notes/101/tasks") && method === "GET") {
      return {
        tasks: [
          {
            id: "task-1",
            note_id: "101",
            text: "Draft PRD",
            status: taskStatus,
            metadata: {},
            projection_status: "live",
            version: taskVersion,
            projection: {
              note_id: "101",
              note_version: noteVersion,
              line_number: 1,
              start_offset: 0,
              end_offset: noteContent.length,
              raw_line: noteContent,
              has_child_content: false,
              projection_status: "live"
            }
          }
        ],
        reconciliation: { status: "clean", note_id: "101", note_version: noteVersion }
      }
    }

    if (path.startsWith("/api/v1/notes/tasks/activity") && method === "GET") {
      return {
        events:
          options.includeActivity && !activityDismissed
            ? [
                {
                  id: "event-1",
                  task_id: "task-1",
                  note_id: "101",
                  event_type: "status_changed",
                  actor_type: "agent",
                  actor_id: "agent-1",
                  tool_name: "notes.tasks.set_status",
                  created_at: "2026-06-05T07:05:00Z"
                }
              ]
            : []
      }
    }

    if (path === "/api/v1/notes/tasks/activity/event-1" && method === "PATCH") {
      activityDismissed = true
      return {
        event_id: "event-1",
        user_id: "1",
        dismissed_at: "2026-06-05T07:06:00Z"
      }
    }

    if (path === "/api/v1/notes/tasks/status" && method === "POST") {
      taskStatus = request.body?.updates?.[0]?.status || "done"
      taskVersion += 1
      noteVersion += 1
      noteContent = taskStatus === "done" ? "- [x] Draft PRD" : "- [ ] Draft PRD"
      return { tasks: [] }
    }

    if (path.startsWith("/api/v1/notes/101?expected_version=") && method === "PUT") {
      if (options.conflictOnSave) {
        const error = new Error("version mismatch") as Error & { status?: number }
        error.status = 409
        throw error
      }
      noteContent = String(request.body?.content || noteContent)
      noteVersion += 1
      return {
        id: 101,
        title: "Dock task note",
        content: noteContent,
        keywords: ["planning"],
        version: noteVersion
      }
    }

    return {}
  })
}

const openTaskBackedDockNote = async (overrides: Partial<NotesDockNote> = {}) => {
  const user = userEvent.setup()
  useNotesDockStore.setState({ isOpen: true, notes: [], activeNoteId: null,
    position: { x: 24, y: 80 }, size: { width: 640, height: 520 } })
  render(<ConfigProvider theme={{ token: { motion: false } }}><NotesDockPanel /></ConfigProvider>)
  await user.click(await screen.findByRole("button", { name: /^Dock task note -/ }))
  await waitFor(() => expect(screen.getByRole("checkbox", { name: /Draft PRD/ })).toBeEnabled())
  await waitFor(() => expect(mockBgRequest).toHaveBeenCalledWith(expect.objectContaining({ path: expect.stringContaining("/api/v1/notes/tasks/activity") })))
  const row = useNotesDockStore.getState().notes.find((note) => note.id === 101)
  expect(row?.snapshot).toEqual(overrides.snapshot ?? { title: "Dock task note", content: "- [ ] Draft PRD", keywords: ["planning"], version: 2 })
  if (overrides.isDirty) {
    await user.type(dockContentEditor(), "{end}{enter}local note")
    expect(dockContentEditor()).toHaveValue(overrides.content)
  }
}

describe("NotesDockPanel task-backed todos", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.stubGlobal("ResizeObserver", ResizeObserverMock as any)
    dockAuthority.userId = 1
    dockAuthority.serverUrl = "https://notes.example.test"
    dockAuthority.watches.clear()
    dockAuthority.ensureConfig.mockImplementation(async () => ({ serverUrl: dockAuthority.serverUrl, authMode: "multi-user", accessToken: "fixture-token" }))
    dockAuthority.currentUser.mockImplementation(async () => ({ id: dockAuthority.userId, is_active: true }))
  })

  afterEach(() => {
    useNotesDockStore.setState({
      isOpen: false,
      notes: [],
      activeNoteId: null
    })
    vi.unstubAllGlobals()
  })

  it("uses the backend task status endpoint for clean dock checkbox toggles", async () => {
    setupTaskBackedDockMock()
    await openTaskBackedDockNote()
    fireEvent.click(await screen.findByRole("checkbox", { name: /Draft PRD/ }))

    await waitFor(() => {
      expect(mockBgRequest).toHaveBeenCalledWith(
        expect.objectContaining({
          path: "/api/v1/notes/tasks/status",
          method: "POST",
          body: {
            updates: [
              expect.objectContaining({
                task_id: "task-1",
                status: "done",
                expected_task_version: 5,
                expected_note_version: 2
              })
            ]
          }
        })
      )
    })
    expect(await screen.findByDisplayValue("- [x] Draft PRD")).toBeInTheDocument()
    expect(screen.queryByText("Unsaved changes")).not.toBeInTheDocument()
  })

  it("keeps dirty dock checkbox toggles local without writing task status", async () => {
    setupTaskBackedDockMock()
    await openTaskBackedDockNote({
      content: "- [ ] Draft PRD\nlocal note",
      snapshot: {
        title: "Dock task note",
        content: "- [ ] Draft PRD",
        keywords: ["planning"],
        version: 2
      },
      isDirty: true
    })

    fireEvent.click(await screen.findByRole("checkbox", { name: /Draft PRD/ }))

    expect(mockBgRequest).not.toHaveBeenCalledWith(
      expect.objectContaining({ path: "/api/v1/notes/tasks/status" })
    )
    expect(dockContentEditor()).toHaveValue("- [x] Draft PRD\nlocal note")
    expect(screen.getByText("Unsaved changes")).toBeInTheDocument()
  })

  it("preserves dirty local task edits when saving hits a version conflict", async () => {
    setupTaskBackedDockMock({ conflictOnSave: true })
    await openTaskBackedDockNote({
      content: "- [ ] Draft PRD\nlocal note",
      snapshot: {
        title: "Dock task note",
        content: "- [ ] Draft PRD",
        keywords: ["planning"],
        version: 2
      },
      isDirty: true
    })

    fireEvent.click(await screen.findByRole("checkbox", { name: /Draft PRD/ }))
    fireEvent.click(screen.getByRole("button", { name: "Update" }))

    await waitFor(() => {
      expect(stableMessageApi.error).toHaveBeenCalled()
    })
    expect(mockBgRequest).not.toHaveBeenCalledWith(
      expect.objectContaining({ path: "/api/v1/notes/tasks/status" })
    )
    expect(dockContentEditor()).toHaveValue("- [x] Draft PRD\nlocal note")
  })

  it("shows agent task activity without overwriting a dirty dock note and can dismiss it", async () => {
    setupTaskBackedDockMock({ includeActivity: true })
    await openTaskBackedDockNote({
      content: "- [ ] Draft PRD\nlocal note",
      snapshot: {
        title: "Dock task note",
        content: "- [ ] Draft PRD",
        keywords: ["planning"],
        version: 2
      },
      isDirty: true
    })

    expect(await screen.findByTestId("notes-dock-task-activity-notice")).toHaveTextContent(
      "agent-1 via notes.tasks.set_status changed 1 task in Dock task note."
    )
    expect(dockContentEditor()).toHaveValue("- [ ] Draft PRD\nlocal note")

    fireEvent.click(screen.getByRole("button", { name: "Dismiss task activity" }))
    await waitFor(() => {
      expect(mockBgRequest).toHaveBeenCalledWith(
        expect.objectContaining({
          path: "/api/v1/notes/tasks/activity/event-1",
          method: "PATCH",
          body: { dismissed: true }
        })
      )
    })
    await waitFor(() => {
      expect(screen.queryByTestId("notes-dock-task-activity-notice")).not.toBeInTheDocument()
    })
  })
})

type OwnedDockServerNote = { id: number; title: string; content: string; keywords: string[]; version: number }
const ownedDockNotes = new Map<number, OwnedDockServerNote>()
const dockGates = new Set<() => void>()
const dockPendingRequests = new Set<Promise<unknown>>()
const dockDeferred = <T,>(fallback: T) => {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((done) => { resolve = done })
  const drain = () => resolve(fallback)
  dockGates.add(drain)
  return { promise, resolve: (value: T) => { dockGates.delete(drain); resolve(value) } }
}
const dockRequests = (): BgRequestInit[] => mockBgRequest.mock.calls.map(([request]) => request)
const dockStatusWrites = () => dockRequests().filter((request) => String(request.path) === "/api/v1/notes/tasks/status")
const dockNoteWrites = () => dockRequests().filter((request) => ["POST", "PUT"].includes(String(request.method)) && String(request.path) !== "/api/v1/notes/tasks/status")
const dockRow = (id = 101) => useNotesDockStore.getState().notes.find((note) => note.id === id)
let dockTaskVersion = 5
let dockScript: ((request: BgRequestInit, response: unknown) => Promise<unknown>) | undefined
const ownedDockResponse = async (request: BgRequestInit): Promise<unknown> => {
  const url = new URL(String(request.path), "https://fixture.test")
  const method = String(request.method || "GET").toUpperCase()
  if (url.pathname === "/api/v1/notes/tasks/status") {
    const updates: unknown = request.body?.updates
    if (!Array.isArray(updates) || updates.length !== 1) throw new Error("One task update required")
    const update: unknown = updates[0]
    if (!update || typeof update !== "object" || !("task_id" in update) || !("status" in update) ||
      !("expected_task_version" in update) || !("expected_note_version" in update)) throw new Error("Task CAS required")
    const note = ownedDockNotes.get(Number(String(update.task_id).replace("task-", "")))
    if (!note || update.expected_task_version !== dockTaskVersion || update.expected_note_version !== note.version) throw { status: 409, message: "Task version conflict" }
    note.content = note.content.replace(/^- \[[ x]\]/, update.status === "done" ? "- [x]" : "- [ ]")
    note.version += 1
    dockTaskVersion += 1
    return { tasks: [] }
  }
  if (url.pathname === "/api/v1/notes/tasks/activity") return { events: [] }
  if (url.pathname === "/api/v1/notes/" && method === "POST") {
    const note = { id: 103, title: String(request.body?.title), content: String(request.body?.content), keywords: [], version: 1 }
    ownedDockNotes.set(note.id, note)
    return { ...note }
  }
  if (url.pathname === "/api/v1/notes/" || url.pathname === "/api/v1/notes/search/") {
    const query = url.searchParams.get("query")?.toLowerCase() || ""
    return { items: [...ownedDockNotes.values()].filter((note) => note.title.toLowerCase().includes(query)).map((note) => ({ ...note })) }
  }
  const id = Number(url.pathname.split("/")[4])
  const note = ownedDockNotes.get(id)
  if (!note) throw { status: 404, message: "Missing note" }
  if (url.pathname.endsWith("/tasks")) {
    return { tasks: [{ id: `task-${id}`, note_id: String(id), text: id === 101 ? "Alpha" : id === 102 ? "Beta" : "New task",
      status: note.content.startsWith("- [x]") ? "done" : "open", metadata: {}, projection_status: "live", version: dockTaskVersion,
      projection: { note_id: String(id), note_version: note.version, line_number: 1, start_offset: 0,
        end_offset: note.content.length, raw_line: note.content, has_child_content: false, projection_status: "live" } }],
      reconciliation: { status: "clean", note_id: String(id), note_version: note.version } }
  }
  if (method === "PUT") {
    const expected = url.searchParams.get("expected_version") || request.headers?.["expected-version"]
    if (Number(expected) !== note.version) throw { status: 409, message: "Note version conflict" }
    note.title = String(request.body?.title)
    note.content = String(request.body?.content)
    note.version += 1
  }
  return { ...note }
}
const settleDock = async () => {
  await act(async () => { await Promise.allSettled([...dockPendingRequests]); await Promise.resolve() })
}
const mountOwnedDock = () => render(
  <ConfigProvider theme={{ token: { motion: false } }}><NotesDockHost /></ConfigProvider>
)
const openOwnedDockNote = async (user: ReturnType<typeof userEvent.setup>, title = "Alpha", id = 101) => {
  await user.click(await screen.findByRole("button", { name: new RegExp(`^${title} note -`) }))
  await waitFor(() => expect(dockRow(id)?.isDirty, JSON.stringify({
    rows: useNotesDockStore.getState().notes.map((note) => ({ id: note.id, localId: note.localId })),
    activeNoteId: useNotesDockStore.getState().activeNoteId,
    isOpen: useNotesDockStore.getState().isOpen,
    requests: dockRequests().map((request) => ({ path: request.path, method: request.method || "GET" }))
  })).toBe(false))
  await waitFor(() => expect(screen.getByRole("checkbox", { name: `Task: ${title}` })).toBeEnabled())
}
const changeDockAuthority = (kind: "account" | "server", returnToOriginal = true) => {
  act(() => {
    if (kind === "account") dockAuthority.userId = 2
    else dockAuthority.serverUrl = "https://other.example.test"
    for (const watcher of dockAuthority.watches) watcher.tldwConfig({ newValue: { serverUrl: dockAuthority.serverUrl, authMode: "multi-user", accessToken: "fixture-token" } })
    window.dispatchEvent(new Event("tldw:auth-credentials-changed"))
    if (returnToOriginal) {
      dockAuthority.userId = 1
      dockAuthority.serverUrl = "https://notes.example.test"
      window.dispatchEvent(new Event("tldw:auth-credentials-changed"))
    }
  })
}
const expectScopedDock = (request: BgRequestInit) => {
  expect(request.servicePromptConfig).toMatchObject({ serverUrl: "https://notes.example.test", expectedUserId: 1 })
  expect(request.headers?.["X-TLDW-Expected-User-ID"]).toBe("1")
  expect(request.abortSignal).toBeInstanceOf(AbortSignal)
}

describe("Notes checkbox original ownership", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    localStorage.clear()
    vi.stubGlobal("ResizeObserver", ResizeObserverMock)
    dockAuthority.userId = 1
    dockAuthority.serverUrl = "https://notes.example.test"
    dockAuthority.watches.clear()
    dockAuthority.ensureConfig.mockImplementation(async () => ({ serverUrl: dockAuthority.serverUrl, authMode: "multi-user", accessToken: "fixture-token" }))
    dockAuthority.currentUser.mockImplementation(async () => ({ id: dockAuthority.userId, is_active: true }))
    ownedDockNotes.clear()
    ownedDockNotes.set(101, { id: 101, title: "Alpha note", content: "- [ ] Alpha", keywords: [], version: 2 })
    ownedDockNotes.set(102, { id: 102, title: "Beta note", content: "- [ ] Beta", keywords: [], version: 2 })
    dockTaskVersion = 5
    dockScript = undefined
    mockBgRequest.mockImplementation((request: BgRequestInit) => {
      const promise = ownedDockResponse(request).then((response) => dockScript ? dockScript(request, response) : response)
      dockPendingRequests.add(promise)
      void promise.then(() => dockPendingRequests.delete(promise), () => dockPendingRequests.delete(promise))
      return promise
    })
    useNotesDockStore.setState({ isOpen: true, notes: [], activeNoteId: null, position: { x: 24, y: 80 }, size: { width: 640, height: 520 } })
  })
  afterEach(async () => {
    for (const drain of dockGates) drain()
    dockGates.clear()
    await settleDock()
    cleanup()
    useNotesDockStore.setState({ isOpen: false, notes: [], activeNoteId: null })
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it("loads the initial archive once through the debounce boundary and opens Alpha then Beta", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] })
    try {
      await act(async () => { mountOwnedDock() })
      await settleDock()
      expect(screen.getByRole("button", { name: /^Alpha note -/ })).toBeEnabled()
      expect(screen.getByRole("button", { name: /^Beta note -/ })).toBeEnabled()
      await act(async () => { await vi.advanceTimersByTimeAsync(250) })
      await settleDock()
      const initialLists = dockRequests().filter((request) => {
        const url = new URL(String(request.path), "https://fixture.test")
        return url.pathname === "/api/v1/notes/" && !url.searchParams.get("query") &&
          String(request.method || "GET").toUpperCase() === "GET"
      })
      expect(initialLists).toHaveLength(1)
      vi.useRealTimers()
      const user = userEvent.setup()
      await openOwnedDockNote(user)
      await settleDock()
      await openOwnedDockNote(user, "Beta", 102)
      await settleDock()
      expect(dockRequests()).toContainEqual(expect.objectContaining({ path: "/api/v1/notes/102", method: "GET" }))
      expect(useNotesDockStore.getState().activeNoteId).toBe(dockRow(102)?.localId)
      expect(screen.getByRole("checkbox", { name: "Task: Beta" })).toBeEnabled()
    } finally {
      vi.useRealTimers()
    }
  })

  it("opens and toggles twice under the original owner, including reads after its own markSaved", async () => {
    const user = userEvent.setup()
    mountOwnedDock()
    await openOwnedDockNote(user)
    mockBgRequest.mockClear()
    await user.click(screen.getByRole("checkbox", { name: "Task: Alpha" }))
    await waitFor(() => expect(dockRow()?.version).toBe(3))
    await waitFor(() => expect(screen.getByRole("checkbox", { name: "Task: Alpha" })).toBeChecked())
    const first = dockStatusWrites()[0]
    expect(first.body).toEqual({ updates: [{ task_id: "task-101", status: "done", expected_task_version: 5, expected_note_version: 2 }] })
    for (const request of dockRequests()) expectScopedDock(request)
    await user.click(screen.getByRole("checkbox", { name: "Task: Alpha" }))
    await waitFor(() => expect(dockRow()?.version).toBe(4))
    expect(dockStatusWrites()[1].body).toEqual({ updates: [{ task_id: "task-101", status: "open", expected_task_version: 6, expected_note_version: 3 }] })
    expect(dockRow()?.isDirty).toBe(false)
    expect(stableMessageApi.error).not.toHaveBeenCalled()
  })

  it("lets a brand-new draft Save establish ownership before its first task write", async () => {
    const user = userEvent.setup()
    mountOwnedDock()
    await user.click(await screen.findByRole("button", { name: "New note" }))
    await user.type(screen.getByPlaceholderText("Note title..."), "New task note")
    await user.type(dockContentEditor(), "- [[ ] New task")
    await user.click(screen.getByRole("button", { name: "Save" }))
    await waitFor(() => expect(dockRow(103)?.isDirty).toBe(false))
    expectScopedDock(dockNoteWrites()[0])
    await user.click(await screen.findByRole("checkbox", { name: "Task: New task" }))
    await waitFor(() => expect(dockRow(103)?.version).toBe(2))
    expectScopedDock(dockStatusWrites()[0])
  })

  it("saves all original dirty rows including a non-active owned row before closing", async () => {
    const user = userEvent.setup()
    mountOwnedDock()
    await openOwnedDockNote(user)
    await user.type(dockContentEditor(), " Alpha draft")
    await openOwnedDockNote(user, "Beta", 102)
    await user.type(dockContentEditor(), " Beta draft")
    await user.click(screen.getByRole("button", { name: "Close" }))
    await user.click(await screen.findByRole("button", { name: "Save & Close" }))
    await waitFor(() => expect(useNotesDockStore.getState().isOpen).toBe(false))
    expect(dockNoteWrites()).toHaveLength(2)
    for (const request of dockNoteWrites()) expectScopedDock(request)
    expect(dockRow()?.content).toBe("- [ ] Alpha Alpha draft")
    expect(dockRow(102)?.content).toBe("- [ ] Beta Beta draft")
    expect(useNotesDockStore.getState().notes.every((note) => !note.isDirty)).toBe(true)
  })

  it("refuses an unbound existing server row until an owned Open", async () => {
    const user = userEvent.setup()
    seedDockNote({ id: 101, title: "Alpha note", content: "- [ ] Alpha" })
    mountOwnedDock()
    await screen.findByRole("button", { name: "Update" })
    await settleDock()
    expect(dockRequests().filter((request) => String(request.path).includes("/tasks"))).toHaveLength(0)
    await user.click(screen.getByRole("button", { name: "Update" }))
    await settleDock()
    expect(dockNoteWrites()).toHaveLength(0)
    expect(dockRow()?.version).toBe(2)
  })

  it.each([false, true])("reopens a retained verified row with foreign owner=%s without fresh attestation", async (foreign) => {
    const user = userEvent.setup()
    mountOwnedDock()
    await openOwnedDockNote(user)
    const retained = dockRow()
    await user.click(screen.getByRole("button", { name: "Close" }))
    await waitFor(() => expect(screen.queryByRole("dialog", { name: "Notes Dock" })).not.toBeInTheDocument())
    if (foreign) changeDockAuthority("account", false)
    mockBgRequest.mockClear()
    act(() => useNotesDockStore.getState().setOpen(true))
    await screen.findByRole("button", { name: "Update" })
    await settleDock()
    const taskReads = dockRequests().filter((request) => String(request.path).includes("/tasks"))
    if (foreign) {
      expect(taskReads).toHaveLength(0)
      expect(dockRow()).toBe(retained)
    } else {
      expect(taskReads.length).toBeGreaterThan(0)
      for (const request of taskReads) expectScopedDock(request)
    }
  })

  it("does not stamp a delayed Open response after server away and return", async () => {
    const user = userEvent.setup()
    mountOwnedDock()
    const gate = dockDeferred(undefined)
    dockScript = async (request, response) => { if (String(request.path) === "/api/v1/notes/101") await gate.promise; return response }
    await user.click(await screen.findByRole("button", { name: /^Alpha note -/ }))
    await waitFor(() => expect(dockRequests().some((request) => String(request.path) === "/api/v1/notes/101")).toBe(true))
    changeDockAuthority("server")
    await act(async () => gate.resolve(undefined))
    await settleDock()
    expect(dockRow()).toBeUndefined()
    expect(stableMessageApi.error).not.toHaveBeenCalled()
  })

  it("captures Open intent before delayed principal lookup and account away-return", async () => {
    const user = userEvent.setup()
    mountOwnedDock()
    await screen.findByRole("button", { name: /^Alpha note -/ })
    const gate = dockDeferred({ id: 1, is_active: true })
    dockAuthority.currentUser.mockClear()
    dockAuthority.currentUser.mockImplementation(() => gate.promise)
    mockBgRequest.mockClear()
    await user.click(screen.getByRole("button", { name: /^Alpha note -/ }))
    expect(dockAuthority.currentUser).toHaveBeenCalled()
    expect(dockRequests().filter((request) => String(request.path) === "/api/v1/notes/101")).toHaveLength(0)
    changeDockAuthority("account")
    await act(async () => gate.resolve({ id: 1, is_active: true }))
    await settleDock()
    expect(dockRequests().filter((request) => String(request.path) === "/api/v1/notes/101")).toHaveLength(0)
    expect(dockRow()).toBeUndefined()
  })

  it("refuses a checkbox write if its original row changed during fresh principal lookup", async () => {
    const user = userEvent.setup()
    mountOwnedDock()
    await openOwnedDockNote(user)
    const gate = dockDeferred({ id: 1, is_active: true })
    dockAuthority.currentUser.mockClear()
    dockAuthority.currentUser.mockImplementation(() => gate.promise)
    mockBgRequest.mockClear()
    await user.click(screen.getByRole("checkbox", { name: "Task: Alpha" }))
    expect(dockAuthority.currentUser).toHaveBeenCalled()
    expect(dockStatusWrites()).toHaveLength(0)
    await user.type(dockContentEditor(), " New local text")
    await act(async () => gate.resolve({ id: 1, is_active: true }))
    await settleDock()
    expect(dockStatusWrites()).toHaveLength(0)
    expect(dockRow()?.content).toBe("- [ ] Alpha New local text")
    expect(dockRow()?.isDirty).toBe(true)
  })

  it.each(["account", "active", "edit", "unmount"] as const)("rejects a late applied checkbox ACK after %s invalidation", async (change) => {
    const user = userEvent.setup()
    const view = mountOwnedDock()
    await openOwnedDockNote(user)
    if (change === "active") {
      await openOwnedDockNote(user, "Beta", 102)
      await user.click(screen.getByRole("button", { name: "Alpha note" }))
      await waitFor(() => expect(screen.getByRole("checkbox", { name: "Task: Alpha" })).toBeEnabled())
    }
    const gate = dockDeferred(undefined)
    dockScript = async (request, response) => { if (String(request.path) === "/api/v1/notes/tasks/status") await gate.promise; return response }
    await user.click(screen.getByRole("checkbox", { name: "Task: Alpha" }))
    await waitFor(() => expect(ownedDockNotes.get(101)?.version).toBe(3))
    if (change === "account") changeDockAuthority("account")
    if (change === "active") {
      await user.click(screen.getByRole("button", { name: "Beta note" }))
      await user.click(screen.getByRole("button", { name: "Alpha note" }))
    }
    if (change === "edit") await user.type(dockContentEditor(), " New local text")
    const original = dockRow()
    if (change === "unmount") view.unmount()
    mockBgRequest.mockClear()
    stableMessageApi.error.mockClear()
    const refresh = vi.spyOn(getQueryClient(), "invalidateQueries")
    await act(async () => gate.resolve(undefined))
    await settleDock()
    expect(dockRequests()).toHaveLength(0)
    expect(dockRow()).toBe(original)
    expect(stableMessageApi.error).not.toHaveBeenCalled()
    expect(refresh).not.toHaveBeenCalled()
  })

  it("keeps a delayed post-status detail failure quiet after server away-return", async () => {
    const user = userEvent.setup()
    mountOwnedDock()
    await openOwnedDockNote(user)
    const gate = dockDeferred(undefined)
    dockScript = async (request, response) => {
      if (String(request.path) === "/api/v1/notes/101") { await gate.promise; throw new Error("Old detail failed") }
      return response
    }
    await user.click(screen.getByRole("checkbox", { name: "Task: Alpha" }))
    await waitFor(() => expect(dockRequests().filter((request) => String(request.path) === "/api/v1/notes/101")).toHaveLength(2))
    const original = dockRow()
    changeDockAuthority("server")
    stableMessageApi.error.mockClear()
    await act(async () => gate.resolve(undefined))
    await settleDock()
    expect(dockRow()).toBe(original)
    expect(stableMessageApi.error).not.toHaveBeenCalled()
  })

  it.each(["edit", "server"] as const)("does not stamp or publish a late Save ACK after %s invalidation", async (change) => {
    const user = userEvent.setup()
    mountOwnedDock()
    await openOwnedDockNote(user)
    await user.type(dockContentEditor(), " Saved candidate")
    const gate = dockDeferred(undefined)
    dockScript = async (request, response) => { if (String(request.method) === "PUT") await gate.promise; return response }
    await user.click(screen.getByRole("button", { name: "Update" }))
    await waitFor(() => expect(ownedDockNotes.get(101)?.version).toBe(3))
    if (change === "edit") await user.type(dockContentEditor(), " Newer local text")
    else changeDockAuthority("server")
    const original = dockRow()
    mockBgRequest.mockClear()
    stableMessageApi.success.mockClear()
    await act(async () => gate.resolve(undefined))
    await settleDock()
    expect(dockRow()).toBe(original)
    expect(dockRow()?.isDirty).toBe(true)
    expect(dockRequests()).toHaveLength(0)
    expect(stableMessageApi.success).not.toHaveBeenCalled()
  })

  it.each([false, true])("retains the row after Cancel revokes a pending tab Save, newer edit=%s", async (newerEdit) => {
    const user = userEvent.setup()
    mountOwnedDock()
    await openOwnedDockNote(user)
    await user.type(dockContentEditor(), " Saved candidate")
    const gate = dockDeferred(undefined)
    let refreshStarted = false
    dockScript = async (request, response) => {
      if (String(request.path).startsWith("/api/v1/notes/101/tasks")) { refreshStarted = true; await gate.promise }
      return response
    }
    await user.click(screen.getByRole("button", { name: "Close note" }))
    const dialog = await screen.findByRole("dialog", { name: "Unsaved note" })
    await user.click(within(dialog).getByRole("button", { name: "Save" }))
    await waitFor(() => expect(refreshStarted).toBe(true))
    await user.click(within(dialog).getByRole("button", { name: "Cancel" }))
    await waitFor(() => expect(screen.queryByRole("dialog", { name: "Unsaved note" })).not.toBeInTheDocument())
    if (newerEdit) await user.type(dockContentEditor(), " Newer local text")
    const original = dockRow()
    await act(async () => gate.resolve(undefined))
    await settleDock()
    expect(dockRow()).toBe(original)
    expect(useNotesDockStore.getState().isOpen).toBe(true)
    if (newerEdit) expect(dockRow()?.isDirty).toBe(true)
  })

  it.each([false, true])("keeps the dock open after Cancel revokes an all-Save, newer rows=%s", async (newerRows) => {
    const user = userEvent.setup()
    mountOwnedDock()
    await openOwnedDockNote(user)
    await user.type(dockContentEditor(), " Alpha candidate")
    await openOwnedDockNote(user, "Beta", 102)
    await user.type(dockContentEditor(), " Beta candidate")
    const gate = dockDeferred(undefined)
    dockScript = async (request, response) => { if (String(request.method) === "PUT" && String(request.path).includes("/102")) await gate.promise; return response }
    await user.click(screen.getByRole("button", { name: "Close" }))
    const dialog = await screen.findByRole("dialog", { name: "Unsaved notes" })
    await user.click(within(dialog).getByRole("button", { name: "Save & Close" }))
    await waitFor(() => expect(ownedDockNotes.get(102)?.version).toBe(3))
    await user.click(within(dialog).getByRole("button", { name: "Cancel" }))
    await waitFor(() => expect(screen.queryByRole("dialog", { name: "Unsaved notes" })).not.toBeInTheDocument())
    if (newerRows) {
      await user.click(screen.getByRole("button", { name: "Alpha note" }))
      await user.type(dockContentEditor(), " Newer Alpha text")
      await user.click(screen.getByRole("button", { name: "New note" }))
      await user.type(dockContentEditor(), "New unsaved draft")
    }
    await act(async () => gate.resolve(undefined))
    await settleDock()
    expect(useNotesDockStore.getState().isOpen).toBe(true)
    if (newerRows) {
      expect(dockRow()?.content).toBe("- [ ] Alpha Alpha candidate Newer Alpha text")
      expect(dockRow()?.isDirty).toBe(true)
      expect(useNotesDockStore.getState().notes.some((note) => !note.id && note.content === "New unsaved draft" && note.isDirty)).toBe(true)
    }
  })

  it("scopes archive list and search transport to the verified owner", async () => {
    const user = userEvent.setup()
    mountOwnedDock()
    await screen.findByRole("button", { name: /^Alpha note -/ })
    const listing = dockRequests().find((request) => String(request.path).startsWith("/api/v1/notes/?"))
    expect(listing).toBeDefined()
    if (listing) expectScopedDock(listing)
    await user.type(screen.getByPlaceholderText("Search notes..."), "Alpha")
    await waitFor(() => expect(dockRequests().some((request) => String(request.path).startsWith("/api/v1/notes/search/"))).toBe(true))
    const search = dockRequests().find((request) => String(request.path).startsWith("/api/v1/notes/search/"))
    expect(search).toBeDefined()
    if (search) expectScopedDock(search)
  })

  it("clears revoked archive results", async () => {
    mountOwnedDock()
    await screen.findByRole("button", { name: /^Alpha note -/ })
    changeDockAuthority("account", false)
    await act(async () => { await Promise.resolve() })
    expect(screen.queryByRole("button", { name: /^Alpha note -/ })).not.toBeInTheDocument()
    expect(dockRow()).toBeUndefined()
  })

  it.each(["account", "server"] as const)("rejects a live archive Open after %s changes during principal lookup", async (change) => {
    const user = userEvent.setup()
    mountOwnedDock()
    const button = await screen.findByRole("button", { name: /^Alpha note -/ })
    const gate = dockDeferred({ id: 1, is_active: true })
    dockAuthority.currentUser.mockImplementation(() => gate.promise)
    mockBgRequest.mockClear()
    dockAuthority.currentUser.mockClear()
    await user.click(button)
    // Missing fresh lookup fails immediately, without waiting for an unreachable gate.
    expect(dockAuthority.currentUser).toHaveBeenCalled()
    expect(dockRequests().filter((request) => String(request.path) === "/api/v1/notes/101")).toHaveLength(0)
    changeDockAuthority(change, false)
    mockBgRequest.mockClear()
    stableMessageApi.error.mockClear()
    stableMessageApi.success.mockClear()
    await act(async () => gate.resolve({ id: 1, is_active: true }))
    await settleDock()
    expect(dockRequests().filter((request) => String(request.path) === "/api/v1/notes/101")).toHaveLength(0)
    expect(dockRow()).toBeUndefined()
    expect(useNotesDockStore.getState().notes).toHaveLength(0)
    expect(stableMessageApi.error).not.toHaveBeenCalled()
    expect(stableMessageApi.success).not.toHaveBeenCalled()
  })
})
