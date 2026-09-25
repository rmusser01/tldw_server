import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import NotesManagerPage from "../NotesManagerPage"

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
    info: vi.fn()
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

describe("NotesManagerPage stage 8 trash/restore", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockConfirmDanger.mockResolvedValue(true)
    mockGetSetting.mockResolvedValue(null)
    mockClearSetting.mockResolvedValue(undefined)
  })

  it("restores a trashed note and opens it in active mode", async () => {
    let activeNotes = [{ id: "note-active", title: "Active note", content: "active", version: 1, deleted: false }]
    let trashNotes = [{ id: "note-trash", title: "Trash note", content: "trashed", version: 7, deleted: true }]

    mockBgRequest.mockImplementation(async (request: { path?: string; method?: string }) => {
      const path = String(request.path || "")
      const method = String(request.method || "GET").toUpperCase()

      if (path.startsWith("/api/v1/notes/trash?")) {
        return {
          items: trashNotes,
          total: trashNotes.length
        }
      }

      if (path.startsWith("/api/v1/notes/?")) {
        return {
          items: activeNotes,
          pagination: { total_items: activeNotes.length, total_pages: 1 }
        }
      }

      if (path === "/api/v1/notes/note-trash/restore?expected_version=7" && method === "POST") {
        trashNotes = []
        activeNotes = [
          { id: "note-active", title: "Active note", content: "active", version: 1, deleted: false },
          { id: "note-trash", title: "Trash note", content: "restored", version: 8, deleted: false }
        ]
        return {
          id: "note-trash",
          title: "Trash note",
          content: "restored",
          version: 8,
          last_modified: "2026-02-18T12:00:00.000Z",
          deleted: false,
          metadata: { keywords: [] }
        }
      }

      if (path === "/api/v1/notes/note-trash" && method === "GET") {
        return {
          id: "note-trash",
          title: "Trash note",
          content: "restored",
          metadata: { keywords: [] },
          version: 8,
          last_modified: "2026-02-18T12:00:00.000Z",
          deleted: false
        }
      }

      if (path.startsWith("/api/v1/notes/note-trash/neighbors")) {
        return { nodes: [{ id: "note-trash", type: "note", label: "Trash note" }], edges: [] }
      }

      return {}
    })

    renderPage()
    fireEvent.click(screen.getByTestId("notes-mode-trash"))

    expect(await screen.findByText("Trash note")).toBeInTheDocument()
    fireEvent.click(screen.getByTestId("notes-restore-note-trash"))

    await waitFor(() => {
      expect(mockMessageSuccess).toHaveBeenCalledWith("Note restored")
    })

    await waitFor(() => {
      const restoreCalled = mockBgRequest.mock.calls.some(([request]) => {
        return (
          String(request?.path || "") ===
            "/api/v1/notes/note-trash/restore?expected_version=7" &&
          String(request?.method || "GET").toUpperCase() === "POST"
        )
      })
      expect(restoreCalled).toBe(true)
    })

    await waitFor(() => {
      const detailCalled = mockBgRequest.mock.calls.some(([request]) => {
        return (
          String(request?.path || "") === "/api/v1/notes/note-trash" &&
          String(request?.method || "GET").toUpperCase() === "GET"
        )
      })
      expect(detailCalled).toBe(true)
    })

    expect(await screen.findByTestId("notes-editor-revision-meta")).toHaveTextContent("Version 8")
  })

  it("shows trash empty state", async () => {
    mockBgRequest.mockImplementation(async (request: { path?: string }) => {
      const path = String(request.path || "")
      if (path.startsWith("/api/v1/notes/trash?")) {
        return { items: [], total: 0 }
      }
      if (path.startsWith("/api/v1/notes/?")) {
        return { items: [], pagination: { total_items: 0, total_pages: 1 } }
      }
      return {}
    })

    renderPage()
    fireEvent.click(screen.getByTestId("notes-mode-trash"))
    expect(await screen.findByText("Trash is empty")).toBeInTheDocument()
  })
})
