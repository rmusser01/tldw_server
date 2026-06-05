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
  mockClearSetting
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
    mockClearSetting: vi.fn()
  }
})

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
  bgRequest: (...args: unknown[]) => mockBgRequest(...args)
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

const setupTaskBackedNoteMock = (options: { conflictOnSave?: boolean } = {}) => {
  let noteContent = "- [ ] Draft PRD"
  let noteVersion = 2
  let taskStatus: "open" | "done" = "open"
  let taskVersion = 5
  let activityDismissed = false

  mockGetSetting.mockImplementation(async (setting: { key?: string }) => {
    if (setting?.key === "tldw:lastNoteId") return "note-1"
    return null
  })

  mockBgRequest.mockImplementation(async (request: { path?: string; method?: string; body?: any }) => {
    const path = String(request.path || "")
    const method = String(request.method || "GET").toUpperCase()

    if (path.startsWith("/api/v1/notes/?")) {
      return {
        items: [
          {
            id: "note-1",
            title: "Task note",
            content_preview: noteContent,
            version: noteVersion,
            updated_at: "2026-06-05T07:00:00Z"
          }
        ],
        pagination: { total_items: 1, total_pages: 1 }
      }
    }

    if (path.startsWith("/api/v1/notes/note-1/neighbors")) {
      return { nodes: [], edges: [] }
    }

    if (path === "/api/v1/notes/note-1" && method === "GET") {
      return {
        id: "note-1",
        title: "Task note",
        content: noteContent,
        metadata: { keywords: [] },
        version: noteVersion,
        last_modified: "2026-06-05T07:00:00Z"
      }
    }

    if (path.startsWith("/api/v1/notes/note-1/tasks") && method === "GET") {
      return {
        tasks: [
          {
            id: "task-1",
            note_id: "note-1",
            text: "Draft PRD",
            status: taskStatus,
            metadata: {},
            projection_status: "live",
            version: taskVersion,
            projection: {
              note_id: "note-1",
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
        reconciliation: { status: "clean", note_id: "note-1", note_version: noteVersion }
      }
    }

    if (path.startsWith("/api/v1/notes/tasks/activity") && method === "GET") {
      return {
        events: activityDismissed
          ? []
          : [
              {
                id: "event-1",
                task_id: "task-1",
                note_id: "note-1",
                event_type: "status_changed",
                actor_type: "agent",
                actor_id: "agent-1",
                created_at: "2026-06-05T07:05:00Z"
              }
            ]
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

    if (path.startsWith("/api/v1/notes/note-1?expected_version=") && method === "PUT") {
      if (options.conflictOnSave) {
        const error = new Error("version mismatch") as Error & { status?: number }
        error.status = 409
        throw error
      }
      noteContent = String(request.body?.content || noteContent)
      noteVersion += 1
      return {
        id: "note-1",
        title: "Task note",
        content: noteContent,
        metadata: { keywords: [] },
        version: noteVersion,
        last_modified: "2026-06-05T07:10:00Z"
      }
    }

    return {}
  })
}

describe("NotesManagerPage task-backed todos", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockConfirmDanger.mockResolvedValue(true)
    mockClearSetting.mockResolvedValue(undefined)
  })

  it("renders task-backed checkboxes in preview while edit mode keeps raw markdown", async () => {
    setupTaskBackedNoteMock()
    renderPage()

    expect(await screen.findByDisplayValue("- [ ] Draft PRD")).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Preview" }))

    expect(await screen.findByRole("checkbox", { name: /Draft PRD/ })).not.toBeChecked()

    fireEvent.click(screen.getByRole("button", { name: "Edit" }))

    expect(screen.getByDisplayValue("- [ ] Draft PRD")).toBeInTheDocument()
  }, 10000)

  it("uses the backend status endpoint for clean checkbox toggles and refreshes the note", async () => {
    setupTaskBackedNoteMock()
    renderPage()

    await screen.findByDisplayValue("- [ ] Draft PRD")
    fireEvent.click(screen.getByRole("button", { name: "Preview" }))
    fireEvent.click(await screen.findByRole("checkbox", { name: /Draft PRD/ }))

    await waitFor(() => {
      expect(mockBgRequest).toHaveBeenCalledWith(
        expect.objectContaining({
          path: "/api/v1/notes/tasks/status",
          method: "POST",
          body: {
            updates: [
              {
                task_id: "task-1",
                status: "done",
                expected_task_version: 5,
                expected_note_version: 2
              }
            ]
          }
        })
      )
    })

    fireEvent.click(screen.getByRole("button", { name: "Edit" }))
    expect(await screen.findByDisplayValue("- [x] Draft PRD")).toBeInTheDocument()
  }, 10000)

  it("keeps dirty checkbox toggles local and preserves the draft on save conflict", async () => {
    setupTaskBackedNoteMock({ conflictOnSave: true })
    renderPage()

    const textarea = await screen.findByDisplayValue("- [ ] Draft PRD")
    fireEvent.change(textarea, { target: { value: "- [ ] Draft PRD\nlocal note" } })
    fireEvent.click(screen.getByRole("button", { name: "Preview" }))
    fireEvent.click(await screen.findByRole("checkbox", { name: /Draft PRD/ }))

    expect(mockBgRequest).not.toHaveBeenCalledWith(
      expect.objectContaining({ path: "/api/v1/notes/tasks/status" })
    )

    fireEvent.click(screen.getByRole("button", { name: "Edit" }))
    expect(await screen.findByLabelText("Note content")).toHaveValue("- [x] Draft PRD\nlocal note")

    fireEvent.click(screen.getByTestId("notes-save-button"))

    await waitFor(() => {
      expect(mockMessageError).toHaveBeenCalled()
    })
    expect(screen.getByLabelText("Note content")).toHaveValue("- [x] Draft PRD\nlocal note")
  }, 10000)

  it("shows unread agent task activity and can dismiss it", async () => {
    setupTaskBackedNoteMock()
    renderPage()

    expect(await screen.findByText("Agent updated this note's tasks.")).toBeInTheDocument()

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
      expect(screen.queryByText("Agent updated this note's tasks.")).not.toBeInTheDocument()
    })
  }, 10000)
})
