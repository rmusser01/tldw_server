import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import NotesManagerPage from "../NotesManagerPage"

vi.mock("@/components/Notes/hooks/useNotesGraphAuthorityScope", () => ({
  useNotesGraphAuthorityScope: () => "test-notes-authority"
}))

const {
  mockBgRequest,
  mockMessageOpen,
  mockMessageDestroy,
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
  mockMessageOpen: vi.fn(),
  mockMessageDestroy: vi.fn(),
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
    open: mockMessageOpen,
    destroy: mockMessageDestroy,
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

vi.mock("@/components/Notes/NotesListPanel", () => ({
  default: ({ onSelectNote }: any) => (
    <button data-testid="mock-select-note" onClick={() => onSelectNote("note-undo")}>
      Select note
    </button>
  )
}))

vi.mock("@/components/Notes/NotesEditorHeader", () => ({
  default: ({ onDelete }: any) => (
    <button data-testid="mock-delete-note" onClick={() => onDelete()}>
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

const getUndoContent = () => {
  const payload = mockMessageOpen.mock.calls[0]?.[0]
  expect(payload).toBeTruthy()
  return payload.content as React.ReactNode
}

describe("NotesManagerPage stage 32 delete undo", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockConfirmDanger.mockResolvedValue(true)
    mockGetSetting.mockResolvedValue(null)
    mockSetSetting.mockResolvedValue(undefined)
    mockClearSetting.mockResolvedValue(undefined)
  })

  it("restores deleted notes when Undo is pressed from toast action", async () => {
    mockBgRequest.mockImplementation(async (request: { path?: string; method?: string }) => {
      const path = String(request.path || "")
      const method = String(request.method || "GET").toUpperCase()
      if (path.startsWith("/api/v1/notes/?")) {
        return {
          items: [
            {
              id: "note-undo",
              title: "Undo note",
              content: "body",
              version: 1,
              metadata: { keywords: [] }
            }
          ],
          pagination: { total_items: 1, total_pages: 1 }
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
      if (path === "/api/v1/notes/note-undo" && method === "GET") {
        return {
          id: "note-undo",
          title: "Undo note",
          content: "body",
          version: 1,
          last_modified: "2026-02-18T10:00:00.000Z",
          metadata: { keywords: [] }
        }
      }
      if (path.startsWith("/api/v1/notes/note-undo/neighbors")) {
        return { nodes: [{ id: "note-undo", type: "note", label: "Undo note" }], edges: [] }
      }
      if (path.startsWith("/api/v1/notes/note-undo?expected_version=1") && method === "DELETE") {
        return {}
      }
      if (path.startsWith("/api/v1/notes/trash?limit=100&offset=0") && method === "GET") {
        return {
          items: [{ id: "note-undo", version: 2, title: "Undo note", content: "body" }],
          total: 1
        }
      }
      if (path.startsWith("/api/v1/notes/note-undo/restore?expected_version=2") && method === "POST") {
        return {
          id: "note-undo",
          title: "Undo note",
          content: "body",
          version: 3,
          metadata: { keywords: [] }
        }
      }
      return {}
    })

    renderPage()
    fireEvent.click(await screen.findByTestId("mock-select-note"))
    fireEvent.click(await screen.findByTestId("mock-delete-note"))

    await waitFor(() => {
      expect(mockMessageOpen).toHaveBeenCalledTimes(1)
    })

    const undoToast = render(<>{getUndoContent()}</>)
    fireEvent.click(undoToast.getByTestId("notes-delete-undo-note-undo"))

    await waitFor(() => {
      const restoreCalled = mockBgRequest.mock.calls.some(([request]) =>
        String(request?.path || "").startsWith(
          "/api/v1/notes/note-undo/restore?expected_version=2"
        )
      )
      expect(restoreCalled).toBe(true)
    })
    expect(mockMessageSuccess).toHaveBeenCalledWith("Note restored")
  })

  it("looks up deleted versions across paginated trash results", async () => {
    const firstTrashPage = Array.from({ length: 100 }, (_, index) => ({
      id: `other-note-${index}`,
      version: 1,
      title: `Other ${index}`,
      content: "body"
    }))

    mockBgRequest.mockImplementation(async (request: { path?: string; method?: string }) => {
      const path = String(request.path || "")
      const method = String(request.method || "GET").toUpperCase()
      if (path.startsWith("/api/v1/notes/?")) {
        return {
          items: [
            {
              id: "note-undo",
              title: "Undo note",
              content: "body",
              version: 1,
              metadata: { keywords: [] }
            }
          ],
          pagination: { total_items: 1, total_pages: 1 }
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
      if (path === "/api/v1/notes/note-undo" && method === "GET") {
        return {
          id: "note-undo",
          title: "Undo note",
          content: "body",
          version: 1,
          last_modified: "2026-02-18T10:00:00.000Z",
          metadata: { keywords: [] }
        }
      }
      if (path.startsWith("/api/v1/notes/note-undo/neighbors")) {
        return { nodes: [{ id: "note-undo", type: "note", label: "Undo note" }], edges: [] }
      }
      if (path.startsWith("/api/v1/notes/note-undo?expected_version=1") && method === "DELETE") {
        return {}
      }
      if (path.startsWith("/api/v1/notes/trash?limit=100&offset=0") && method === "GET") {
        return {
          items: firstTrashPage,
          pagination: { total_items: 101, total_pages: 2 }
        }
      }
      if (path.startsWith("/api/v1/notes/trash?limit=100&offset=100") && method === "GET") {
        return {
          items: [{ id: "note-undo", version: 7, title: "Undo note", content: "body" }],
          pagination: { total_items: 101, total_pages: 2 }
        }
      }
      if (path.startsWith("/api/v1/notes/note-undo/restore?expected_version=7") && method === "POST") {
        return {
          id: "note-undo",
          title: "Undo note",
          content: "body",
          version: 8,
          metadata: { keywords: [] }
        }
      }
      return {}
    })

    renderPage()
    fireEvent.click(await screen.findByTestId("mock-select-note"))
    fireEvent.click(await screen.findByTestId("mock-delete-note"))

    await waitFor(() => {
      expect(mockMessageOpen).toHaveBeenCalledTimes(1)
    })

    const undoToast = render(<>{getUndoContent()}</>)
    fireEvent.click(undoToast.getByTestId("notes-delete-undo-note-undo"))

    await waitFor(() => {
      const restoreCalled = mockBgRequest.mock.calls.some(([request]) =>
        String(request?.path || "").startsWith(
          "/api/v1/notes/note-undo/restore?expected_version=7"
        )
      )
      expect(restoreCalled).toBe(true)
    })
    expect(mockMessageSuccess).toHaveBeenCalledWith("Note restored")
  })

  it("shows fallback warning when undo version cannot be resolved", async () => {
    mockBgRequest.mockImplementation(async (request: { path?: string; method?: string }) => {
      const path = String(request.path || "")
      const method = String(request.method || "GET").toUpperCase()
      if (path.startsWith("/api/v1/notes/?")) {
        return {
          items: [{ id: "note-undo", title: "Undo note", content: "body", version: 1 }],
          pagination: { total_items: 1, total_pages: 1 }
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
      if (path === "/api/v1/notes/note-undo" && method === "GET") {
        return { id: "note-undo", title: "Undo note", content: "body", version: 1 }
      }
      if (path.startsWith("/api/v1/notes/note-undo/neighbors")) {
        return { nodes: [], edges: [] }
      }
      if (path.startsWith("/api/v1/notes/note-undo?expected_version=1") && method === "DELETE") {
        return {}
      }
      if (path.startsWith("/api/v1/notes/trash?limit=100&offset=0") && method === "GET") {
        return { items: [], total: 0 }
      }
      return {}
    })

    renderPage()
    fireEvent.click(await screen.findByTestId("mock-select-note"))
    fireEvent.click(await screen.findByTestId("mock-delete-note"))

    await waitFor(() => {
      expect(mockMessageOpen).toHaveBeenCalledTimes(1)
    })

    const undoToast = render(<>{getUndoContent()}</>)
    fireEvent.click(undoToast.getByTestId("notes-delete-undo-note-undo"))

    await waitFor(() => {
      expect(mockMessageWarning).toHaveBeenCalledWith(
        "Undo unavailable. Open Trash to restore this note."
      )
    })
  })
})
