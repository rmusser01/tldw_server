import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { NotesDockPanel } from "../NotesDockPanel"
import { useNotesDockStore, type NotesDockNote } from "@/store/notes-dock"

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
      return { items: [], pagination: { total_items: 0, total_pages: 1 } }
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

describe("NotesDockPanel task-backed todos", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.stubGlobal("ResizeObserver", ResizeObserverMock as any)
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
    seedDockNote()

    render(<NotesDockPanel />)
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
    seedDockNote({
      content: "- [ ] Draft PRD\nlocal note",
      snapshot: {
        title: "Dock task note",
        content: "- [ ] Draft PRD",
        keywords: ["planning"],
        version: 2
      },
      isDirty: true
    })

    render(<NotesDockPanel />)
    fireEvent.click(await screen.findByRole("checkbox", { name: /Draft PRD/ }))

    expect(mockBgRequest).not.toHaveBeenCalledWith(
      expect.objectContaining({ path: "/api/v1/notes/tasks/status" })
    )
    expect(dockContentEditor()).toHaveValue("- [x] Draft PRD\nlocal note")
    expect(screen.getByText("Unsaved changes")).toBeInTheDocument()
  })

  it("preserves dirty local task edits when saving hits a version conflict", async () => {
    setupTaskBackedDockMock({ conflictOnSave: true })
    seedDockNote({
      content: "- [ ] Draft PRD\nlocal note",
      snapshot: {
        title: "Dock task note",
        content: "- [ ] Draft PRD",
        keywords: ["planning"],
        version: 2
      },
      isDirty: true
    })

    render(<NotesDockPanel />)
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
    seedDockNote({
      content: "- [ ] Draft PRD\nlocal note",
      snapshot: {
        title: "Dock task note",
        content: "- [ ] Draft PRD",
        keywords: ["planning"],
        version: 2
      },
      isDirty: true
    })

    render(<NotesDockPanel />)

    expect(await screen.findByTestId("notes-dock-task-activity-notice")).toHaveTextContent(
      "Agent updated this note's tasks."
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
