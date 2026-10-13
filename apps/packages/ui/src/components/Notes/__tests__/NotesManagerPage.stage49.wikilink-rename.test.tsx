import React from "react"
import { App, ConfigProvider } from "antd"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import NotesManagerPage from "../NotesManagerPage"
import type { NotesLeaveGuardState } from "../hooks/useNotesEditorState"

// Renaming a note leaves [[Old title]] links in other notes unresolved
// (#3110, follow-up to NE-02). The owner chose to offer the update: after the
// renamed note saves, a non-blocking prompt offers "Update links", the result
// toast offers Undo, and a dismissed prompt does not come back.

// Pin the notes authority scope like stage44. Without it, whether the page
// talks to the server depends on ambient connection state from whichever test
// files ran beside this one.
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

vi.mock("@/hooks/useCallerCapabilities", () => {
  const capabilities = {
    monitoringAlerts: "allowed",
    userId: 1,
    refreshAfterForbidden: vi.fn(async () => undefined)
  }
  return { useCallerCapabilities: () => capabilities }
})

const { mockBgRequest, mockNavigate, mockConfirmDanger, mockGetSetting, mockClearSetting } = vi.hoisted(() => ({
  mockBgRequest: vi.fn(),
  mockNavigate: vi.fn(),
  mockConfirmDanger: vi.fn(),
  mockGetSetting: vi.fn(),
  mockClearSetting: vi.fn()
}))

vi.mock("react-i18next", () => {
  // One stable `t`: a new function per render would re-run every effect keyed on it.
  const t = (
    key: string,
    defaultValueOrOptions?: string | { defaultValue?: string; [key: string]: unknown }
  ) => {
    if (typeof defaultValueOrOptions === "string") return defaultValueOrOptions
    const template = defaultValueOrOptions?.defaultValue
    if (!template) return key
    return template.replace(/\{\{\s*(\w+)\s*\}\}/g, (_match, name: string) =>
      String(defaultValueOrOptions?.[name] ?? "")
    )
  }
  return { useTranslation: () => ({ t }) }
})

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

vi.mock("@/components/Common/MarkdownPreview", () => ({
  MarkdownPreview: ({ content }: { content: string }) => (
    <div data-testid="markdown-preview-content">{content}</div>
  )
}))

vi.mock("@/components/Notes/NotesListPanel", () => ({
  default: () => <div data-testid="notes-list-panel" />
}))

const REFERRERS = "/api/v1/notes/wikilinks/referrers"
const REWRITE = "/api/v1/notes/wikilinks/rewrite"
const UNDO = "/api/v1/notes/wikilinks/rewrite/undo"

type ServerRequest = {
  path?: string
  method?: string
  body?: Record<string, unknown>
  headers?: Record<string, string>
}

// The one note the editor creates and renames, as the server holds it.
let serverNote: { id: string; title: string; content: string; version: number } | null = null
// Notes that link to a title, by lower-cased title.
let linkersByTitle: Record<string, Array<{ id: string; title: string; version: number }>> = {}

// The route's leave guard, captured so a test can navigate away like the router does.
const leaveGuard: { current: NotesLeaveGuardState | null } = { current: null }
const CaptureLeaveGuard: React.FC<NotesLeaveGuardState> = (props) => {
  leaveGuard.current = props
  return null
}

// Without motion a closed prompt leaves the DOM at once; jsdom never ends a CSS transition.
const appShell = (page: React.ReactNode) => (
  <ConfigProvider theme={{ token: { motion: false } }}>
    <App>{page}</App>
  </ConfigProvider>
)

const renderPage = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false }
    }
  })
  return render(
    appShell(
      <QueryClientProvider client={queryClient}>
        <NotesManagerPage LeaveGuard={CaptureLeaveGuard} />
      </QueryClientProvider>
    )
  )
}

const requestsTo = (path: string, method = "POST"): ServerRequest[] =>
  mockBgRequest.mock.calls
    .map(([request]) => request as ServerRequest)
    .filter(
      (request) => request.path === path && String(request.method || "GET").toUpperCase() === method
    )

const titleInput = () => screen.getByPlaceholderText("Title")
const editor = () => screen.getByPlaceholderText("Write your note here... (Markdown supported)")

// The page DOM is large, and a role query walks all of it on every poll. The
// offer is found by test id and text instead; the hook suite covers its roles.
const PROMPT = "notes-wikilink-rename-prompt"
const CONFIRM = "notes-wikilink-rename-confirm"

const expectNoPrompt = () => {
  expect(screen.queryByTestId(PROMPT)).not.toBeInTheDocument()
  expect(screen.queryByTestId(CONFIRM)).not.toBeInTheDocument()
}

/** The prompt's own Close control: the nearest one around the prompt text. */
const findPromptCloseButton = async (): Promise<HTMLElement> => {
  let node: HTMLElement | null = await screen.findByTestId(PROMPT)
  while (node) {
    const close = within(node).queryAllByLabelText("Close").find((element) => element.tagName === "BUTTON")
    if (close) return close
    node = node.parentElement
  }
  throw new Error("The prompt has no Close control")
}

const findUndoButton = async (): Promise<HTMLElement> => {
  const button = (await screen.findByText("Undo")).closest("button")
  if (!button) throw new Error("Undo is not a button")
  return button
}

const saveAndWaitForVersion = async (version: number) => {
  fireEvent.click(screen.getByTestId("notes-save-button"))
  await waitFor(() => {
    expect(screen.getByTestId("notes-editor-revision-meta")).toHaveTextContent(`Version ${version}`)
  })
}

const createNote = async (title: string) => {
  fireEvent.change(titleInput(), { target: { value: title } })
  fireEvent.change(editor(), { target: { value: "The renamed note." } })
  await saveAndWaitForVersion(1)
}

const renameNote = async (title: string, version: number) => {
  fireEvent.change(titleInput(), { target: { value: title } })
  await saveAndWaitForVersion(version)
}

// antd-heavy page: each interaction can take seconds on a loaded runner.
describe("NotesManagerPage stage 49 wikilink rename offer", { timeout: 60_000 }, () => {
  beforeEach(() => {
    vi.clearAllMocks()
    leaveGuard.current = null
    serverNote = null
    linkersByTitle = {
      "old title": [
        { id: "linker-a", title: "Linker A", version: 4 },
        { id: "linker-b", title: "Linker B", version: 2 }
      ]
    }
    mockConfirmDanger.mockResolvedValue(true)
    mockGetSetting.mockResolvedValue(null)
    mockClearSetting.mockResolvedValue(undefined)

    mockBgRequest.mockImplementation(async (request: ServerRequest) => {
      const path = String(request.path || "")
      const method = String(request.method || "GET").toUpperCase()

      if (path === REFERRERS && method === "POST") {
        const notes = linkersByTitle[String(request.body?.title || "").toLowerCase()] ?? []
        return { title: request.body?.title, count: notes.length, notes, next_after_note_id: null }
      }
      if (path === REWRITE && method === "POST") {
        // Linker A is rewritten. Linker B was edited after the count (it is at
        // version 3 now), so it is skipped and still links to the old title.
        const sent = (request.body?.notes ?? []) as Array<{ id: string; expected_version: number }>
        linkersByTitle = { "old title": [{ id: "linker-b", title: "Linker B", version: 3 }] }
        const results = sent.map((note) =>
          note.id === "linker-a"
            ? {
                id: "linker-a",
                title: "Linker A",
                status: "updated",
                version: 5,
                replaced_count: 1,
                replacements: [{ token_index: 0, original: "[[Old title]]" }]
              }
            : {
                id: note.id,
                title: "Linker B",
                status: note.expected_version === 3 ? "updated" : "skipped_conflict",
                version: note.expected_version === 3 ? 4 : 3,
                replaced_count: note.expected_version === 3 ? 1 : 0,
                replacements:
                  note.expected_version === 3 ? [{ token_index: 0, original: "[[Old title]]" }] : []
              }
        )
        return {
          old_title: request.body?.old_title,
          new_title: serverNote?.title,
          link_form: "title",
          replacement: `[[${serverNote?.title}]]`,
          new_title_shared: false,
          updated_count: results.filter((result) => result.status === "updated").length,
          skipped_count: results.filter((result) => result.status !== "updated").length,
          results
        }
      }
      if (path === UNDO && method === "POST") {
        return {
          restored_count: 1,
          skipped_count: 0,
          results: [{ id: "linker-a", title: "Linker A", status: "restored", version: 6 }]
        }
      }
      if (path === "/api/v1/notes/wikilinks/resolve") return { titles: [], ids: [] }
      if (path.startsWith("/api/v1/notes/?")) {
        return {
          items: serverNote ? [serverNote] : [],
          pagination: { total_items: serverNote ? 1 : 0, total_pages: 1 }
        }
      }
      if (path === "/api/v1/notes/" && method === "POST") {
        serverNote = {
          id: "note-a",
          title: String(request.body?.title || ""),
          content: String(request.body?.content || ""),
          version: 1
        }
        return { ...serverNote, metadata: { keywords: [] }, last_modified: "2026-02-18T11:00:00.000Z" }
      }
      if (path === "/api/v1/notes/note-a" && method === "PUT" && serverNote) {
        serverNote = {
          ...serverNote,
          title: String(request.body?.title || serverNote.title),
          content: String(request.body?.content ?? serverNote.content),
          version: serverNote.version + 1
        }
        return { ...serverNote, metadata: { keywords: [] }, last_modified: "2026-02-18T11:05:00.000Z" }
      }
      if (path === "/api/v1/notes/note-a" && method === "GET" && serverNote) {
        return { ...serverNote, metadata: { keywords: [] }, last_modified: "2026-02-18T11:00:00.000Z" }
      }
      if (path.startsWith("/api/v1/notes/note-a/neighbors")) {
        return { nodes: [{ id: "note-a", type: "note", label: serverNote?.title }], edges: [] }
      }
      return {}
    })
  })

  it("offers to update links only when other notes link to the old title", async () => {
    renderPage()
    await createNote("Old title")

    await renameNote("New title", 2)

    expect(await screen.findByText('2 notes link to "Old title"')).toBeInTheDocument()
    expect(screen.getByTestId(CONFIRM)).toHaveTextContent("Update links")
    const [request] = requestsTo(REFERRERS)
    expect(request.body).toEqual({ title: "Old title", exclude_note_id: "note-a", unresolved_only: true })
    expect(request.headers?.["X-TLDW-Expected-User-ID"]).toBe("1")
    // The offer changes nothing by itself.
    expect(requestsTo(REWRITE)).toHaveLength(0)
  })

  it("shows no prompt when no other note links to the old title", async () => {
    linkersByTitle = {}
    renderPage()
    await createNote("Old title")

    await renameNote("New title", 2)

    await waitFor(() => expect(requestsTo(REFERRERS)).toHaveLength(1))
    expectNoPrompt()
    expect(screen.queryByText(/link to "Old title"/)).not.toBeInTheDocument()
  })

  it("does not ask about links when a save leaves the title alone", async () => {
    renderPage()
    await createNote("Old title")

    fireEvent.change(editor(), { target: { value: "Only the body changed." } })
    await saveAndWaitForVersion(2)

    expect(requestsTo(REFERRERS)).toHaveLength(0)
    expectNoPrompt()
  })

  it("rewrites the links on confirm and shows a result toast with Undo and the skipped note", async () => {
    renderPage()
    await createNote("Old title")
    await renameNote("New title", 2)

    fireEvent.click(await screen.findByTestId(CONFIRM))

    expect(await screen.findByText("Updated links in 1 note")).toBeInTheDocument()
    expect(screen.getByText(/Skipped: Linker B \(edited since\)/)).toBeInTheDocument()
    expect(await findUndoButton()).toBeInTheDocument()
    const [request] = requestsTo(REWRITE)
    expect(request.body).toEqual({
      note_id: "note-a",
      old_title: "Old title",
      notes: [
        { id: "linker-a", expected_version: 4 },
        { id: "linker-b", expected_version: 2 }
      ]
    })
    // The skipped note was not overwritten. It is offered again at its current version.
    expect(await screen.findByText('1 note links to "Old title"')).toBeInTheDocument()
    fireEvent.click(screen.getByTestId(CONFIRM))
    await waitFor(() => expect(requestsTo(REWRITE)).toHaveLength(2))
    expect(requestsTo(REWRITE)[1].body?.notes).toEqual([{ id: "linker-b", expected_version: 3 }])
  })

  it("restores the previous text when Undo is clicked", async () => {
    renderPage()
    await createNote("Old title")
    await renameNote("New title", 2)
    fireEvent.click(await screen.findByTestId(CONFIRM))

    fireEvent.click(await findUndoButton())

    expect(await screen.findByText("Restored successfully")).toBeInTheDocument()
    const [request] = requestsTo(UNDO)
    expect(request.body).toEqual({
      old_title: "Old title",
      replacement: "[[New title]]",
      notes: [
        {
          id: "linker-a",
          expected_version: 5,
          replacements: [{ token_index: 0, original: "[[Old title]]" }]
        }
      ]
    })
  })

  it("does not prompt again for a rename the user dismissed", async () => {
    renderPage()
    await createNote("Old title")
    await renameNote("New title", 2)

    fireEvent.click(await findPromptCloseButton())
    await waitFor(() => {
      expectNoPrompt()
    })
    // Later saves of the note, and the same rename made again, stay quiet.
    fireEvent.change(editor(), { target: { value: "Edited after dismissing." } })
    await saveAndWaitForVersion(3)
    await renameNote("Old title", 4)
    await renameNote("New title", 5)

    await waitFor(() => expect(requestsTo(REFERRERS)).toHaveLength(2))
    expect(requestsTo(REFERRERS).map((request) => request.body?.title)).toEqual(["Old title", "New title"])
    expect(requestsTo(REWRITE)).toHaveLength(0)
    expectNoPrompt()
  })

  it("opens no offer on the next page for a rename saved by the leave flush", async () => {
    const view = renderPage()
    await createNote("Old title")
    // The count is still on its way when the page goes away.
    const respond = mockBgRequest.getMockImplementation()!
    let answerCount: () => void = () => {}
    mockBgRequest.mockImplementation((request: ServerRequest) =>
      request.path === REFERRERS
        ? new Promise((resolve) => {
            answerCount = () => resolve(respond(request))
          })
        : respond(request)
    )
    fireEvent.change(titleInput(), { target: { value: "New title" } })

    // Navigating away flushes the unsaved rename first (NS-01), then leaves.
    let leave = false
    await act(async () => {
      leave = (await leaveGuard.current?.onLeave()) ?? false
    })
    expect(leave).toBe(true)
    expect(requestsTo("/api/v1/notes/note-a", "PUT").at(-1)?.body?.title).toBe("New title")
    await waitFor(() => expect(requestsTo(REFERRERS)).toHaveLength(1))
    view.rerender(appShell(<span>Another page</span>))

    await act(async () => {
      answerCount()
      await new Promise((resolve) => setTimeout(resolve, 50))
    })

    expect(screen.getByText("Another page")).toBeInTheDocument()
    expectNoPrompt()
    expect(screen.queryByText(/link to "Old title"/)).not.toBeInTheDocument()
  })
})
