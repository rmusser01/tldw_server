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
  mockSetSetting,
  mockClearSetting
} = vi.hoisted(() => ({
  mockBgRequest: vi.fn(),
  mockMessageSuccess: vi.fn(),
  mockMessageError: vi.fn(),
  mockMessageWarning: vi.fn(),
  mockMessageInfo: vi.fn(),
  mockNavigate: vi.fn(),
  mockConfirmDanger: vi.fn(),
  mockGetSetting: vi.fn(),
  mockSetSetting: vi.fn(),
  mockClearSetting: vi.fn()
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
    onSelectNote,
    onToggleBulkSelection
  }: {
    notes?: Array<{ id: string | number; title?: string }>
    onSelectNote: (id: string | number) => void
    onToggleBulkSelection?: (id: string | number, checked: boolean, shiftKey: boolean) => void
  }) => (
    <div data-testid="notes-list-panel">
      {(notes || []).map((note) => (
        <div key={String(note.id)}>
          <button
            type="button"
            data-testid={`mock-note-${String(note.id)}`}
            onClick={() => onSelectNote(note.id)}
          >
            {note.title || String(note.id)}
          </button>
          <button
            type="button"
            data-testid={`mock-bulk-select-${String(note.id)}`}
            onClick={() => onToggleBulkSelection?.(note.id, true, false)}
          >
            Select {note.title || String(note.id)}
          </button>
        </div>
      ))}
    </div>
  )
}))

vi.mock("@/components/Notes/NotesEditorHeader", () => ({
  default: ({ onDelete }: { onDelete: () => void }) => (
    <button type="button" data-testid="mock-delete-note" onClick={onDelete}>
      Delete note
    </button>
  )
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

describe("NotesManagerPage stage 46 list reliability", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockConfirmDanger.mockResolvedValue(true)
    mockGetSetting.mockResolvedValue(null)
    mockSetSetting.mockResolvedValue(undefined)
    mockClearSetting.mockResolvedValue(undefined)
  })

  it("removes a deleted selected note from recent notes and persisted recent state", async () => {
    let activeNotes = [
      { id: "note-delete", title: "Delete me", content: "body", version: 1 }
    ]

    mockBgRequest.mockImplementation(async (request: { path?: string; method?: string }) => {
      const path = String(request.path || "")
      const method = String(request.method || "GET").toUpperCase()

      if (path.startsWith("/api/v1/notes/?")) {
        return {
          items: activeNotes,
          pagination: { total_items: activeNotes.length, total_pages: 1 }
        }
      }

      if (path === "/api/v1/admin/notes/title-settings" && method === "GET") {
        return {
          llm_enabled: false,
          default_strategy: "heuristic",
          effective_strategy: "heuristic",
          strategies: ["heuristic"]
        }
      }

      if (path === "/api/v1/notes/note-delete" && method === "GET") {
        return {
          id: "note-delete",
          title: "Delete me",
          content: "body",
          version: 1,
          metadata: { keywords: [] }
        }
      }

      if (path.startsWith("/api/v1/notes/note-delete/neighbors")) {
        return { nodes: [{ id: "note-delete", type: "note", label: "Delete me" }], edges: [] }
      }

      if (path.startsWith("/api/v1/notes/note-delete?expected_version=1") && method === "DELETE") {
        activeNotes = []
        return {}
      }

      return {}
    })

    renderPage()
    fireEvent.click(await screen.findByTestId("mock-note-note-delete"))

    await waitFor(() => {
      expect(screen.getByTestId("notes-recent-item-note-delete")).toBeInTheDocument()
    })

    fireEvent.click(screen.getByTestId("mock-delete-note"))

    await waitFor(() => {
      expect(mockMessageSuccess).toHaveBeenCalledWith("Note deleted")
    })

    await waitFor(() => {
      expect(screen.queryByTestId("notes-recent-item-note-delete")).not.toBeInTheDocument()
    })

    const latestRecentWrite = mockSetSetting.mock.calls
      .filter(([setting]) => setting?.key === "tldw:notesRecentOpened")
      .at(-1)

    expect(latestRecentWrite?.[1]).toEqual([])
  })

  it("removes bulk-deleted notes from recent notes and persisted recent state", async () => {
    let activeNotes = [
      { id: "note-one", title: "One", content: "first", version: 1 },
      { id: "note-two", title: "Two", content: "second", version: 2 }
    ]

    mockBgRequest.mockImplementation(async (request: { path?: string; method?: string }) => {
      const path = String(request.path || "")
      const method = String(request.method || "GET").toUpperCase()

      if (path.startsWith("/api/v1/notes/?")) {
        return {
          items: activeNotes,
          pagination: { total_items: activeNotes.length, total_pages: 1 }
        }
      }

      if (path === "/api/v1/admin/notes/title-settings" && method === "GET") {
        return {
          llm_enabled: false,
          default_strategy: "heuristic",
          effective_strategy: "heuristic",
          strategies: ["heuristic"]
        }
      }

      if (path === "/api/v1/notes/note-one" && method === "GET") {
        return {
          id: "note-one",
          title: "One",
          content: "first",
          version: 1,
          metadata: { keywords: [] }
        }
      }

      if (path === "/api/v1/notes/note-two" && method === "GET") {
        return {
          id: "note-two",
          title: "Two",
          content: "second",
          version: 2,
          metadata: { keywords: [] }
        }
      }

      if (path.includes("/neighbors")) {
        return { nodes: [], edges: [] }
      }

      if (path.startsWith("/api/v1/notes/note-one?expected_version=1") && method === "DELETE") {
        activeNotes = activeNotes.filter((note) => note.id !== "note-one")
        return {}
      }

      if (path.startsWith("/api/v1/notes/note-two?expected_version=2") && method === "DELETE") {
        activeNotes = activeNotes.filter((note) => note.id !== "note-two")
        return {}
      }

      return {}
    })

    renderPage()
    fireEvent.click(await screen.findByTestId("mock-note-note-one"))
    fireEvent.click(await screen.findByTestId("mock-note-note-two"))

    await waitFor(() => {
      expect(screen.getByTestId("notes-recent-item-note-one")).toBeInTheDocument()
      expect(screen.getByTestId("notes-recent-item-note-two")).toBeInTheDocument()
    })

    fireEvent.click(screen.getByTestId("mock-bulk-select-note-one"))
    fireEvent.click(screen.getByTestId("mock-bulk-select-note-two"))
    fireEvent.click(screen.getByTestId("notes-bulk-delete"))

    await waitFor(() => {
      expect(mockMessageSuccess).toHaveBeenCalledWith("Deleted 2 selected notes")
    })

    await waitFor(() => {
      expect(screen.queryByTestId("notes-recent-item-note-one")).not.toBeInTheDocument()
      expect(screen.queryByTestId("notes-recent-item-note-two")).not.toBeInTheDocument()
    })

    const latestRecentWrite = mockSetSetting.mock.calls
      .filter(([setting]) => setting?.key === "tldw:notesRecentOpened")
      .at(-1)

    expect(latestRecentWrite?.[1]).toEqual([])
  })
})
