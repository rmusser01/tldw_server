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
  mockClearSetting,
  mockGetAllNoteKeywordStats,
  mockSearchNoteKeywords,
  mockGetCurrentUser,
  mockCytoscapeFactory
} = vi.hoisted(() => {
  const cyInstance: Record<string, any> = {
    on: vi.fn(),
    fit: vi.fn(),
    destroy: vi.fn(),
    zoom: vi.fn(() => 1)
  }
  const cytoscapeFactory: any = vi.fn(() => cyInstance)
  cytoscapeFactory.use = vi.fn()

  return {
    mockBgRequest: vi.fn(),
    mockMessageSuccess: vi.fn(),
    mockMessageError: vi.fn(),
    mockMessageWarning: vi.fn(),
    mockMessageInfo: vi.fn(),
    mockNavigate: vi.fn(),
    mockConfirmDanger: vi.fn(),
    mockGetSetting: vi.fn(),
    mockSetSetting: vi.fn(),
    mockClearSetting: vi.fn(),
    mockGetAllNoteKeywordStats: vi.fn(),
    mockSearchNoteKeywords: vi.fn(),
    mockGetCurrentUser: vi.fn(),
    mockCytoscapeFactory: cytoscapeFactory
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
    info: mockMessageInfo
  })
}))

vi.mock("@/services/note-keywords", () => ({
  getAllNoteKeywordStats: mockGetAllNoteKeywordStats,
  searchNoteKeywords: mockSearchNoteKeywords
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

vi.mock("@/services/tldw/TldwAuth", () => ({
  tldwAuth: { getCurrentUser: mockGetCurrentUser }
}))

vi.mock("@/components/Common/MarkdownPreview", () => ({
  MarkdownPreview: ({ content }: { content: string }) => (
    <div data-testid="markdown-preview-content">{content}</div>
  )
}))

vi.mock("@/components/Notes/NotesListPanel", () => ({
  default: () => <div data-testid="notes-list-panel" />
}))

vi.mock("cytoscape", () => ({
  default: mockCytoscapeFactory
}))

vi.mock("cytoscape-dagre", () => ({
  default: {}
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

const seedAndSaveNote = async () => {
  fireEvent.change(screen.getByPlaceholderText("Title"), {
    target: { value: "Focus handoff note" }
  })
  fireEvent.change(screen.getByPlaceholderText("Write your note here... (Markdown supported)"), {
    target: { value: "Seed content" }
  })
  fireEvent.click(screen.getByTestId("notes-save-button"))

  await waitFor(() => {
    expect(screen.getByTestId("notes-editor-revision-meta")).toHaveTextContent("Version 1")
  })
}

describe("NotesManagerPage stage 21 accessibility overlay and view focus handoff", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockConfirmDanger.mockResolvedValue(true)
    mockGetSetting.mockResolvedValue(null)
    mockSetSetting.mockResolvedValue(undefined)
    mockClearSetting.mockResolvedValue(undefined)
    mockGetAllNoteKeywordStats.mockResolvedValue([{ keyword: "research", noteCount: 3 }])
    mockSearchNoteKeywords.mockResolvedValue(["research"])
    mockGetCurrentUser.mockResolvedValue({
      id: 1,
      username: "notes-user",
      is_active: true
    })

    mockBgRequest.mockImplementation(async (request: { path?: string; method?: string }) => {
      const path = String(request.path || "")
      const method = String(request.method || "GET").toUpperCase()

      if (path.startsWith("/api/v1/notes/?")) {
        return { items: [], pagination: { total_items: 0, total_pages: 1 } }
      }
      if (path === "/api/v1/admin/notes/title-settings" && method === "GET") {
        return {
          llm_enabled: false,
          default_strategy: "heuristic",
          effective_strategy: "heuristic",
          strategies: ["heuristic"]
        }
      }
      if (path === "/api/v1/notes/" && method === "POST") {
        return { id: "note-a", version: 1, last_modified: "2026-02-18T10:00:00.000Z" }
      }
      if (path === "/api/v1/notes/note-a" && method === "GET") {
        return {
          id: "note-a",
          title: "Focus handoff note",
          content: "Seed content",
          metadata: { keywords: [] },
          version: 1,
          last_modified: "2026-02-18T10:00:00.000Z"
        }
      }
      if (path.startsWith("/api/v1/notes/note-a/neighbors")) {
        return {
          nodes: [{ id: "note-a", type: "note", label: "Focus handoff note" }],
          edges: []
        }
      }
      if (path.startsWith("/api/v1/notes/graph?")) {
        return {
          elements: {
            nodes: [{ data: { id: "note:note-a", type: "note", label: "Focus handoff note" } }],
            edges: []
          },
          truncated: false
        }
      }
      return {}
    })
  })

  it("closes keyword picker with Escape and restores focus to Browse tags trigger", async () => {
    renderPage()

    const browseButton = screen.getByRole("button", { name: "Browse tags" })
    browseButton.focus()
    fireEvent.click(browseButton)

    await waitFor(() => {
      expect(screen.getByTestId("notes-keyword-picker-modal")).toBeInTheDocument()
    })

    fireEvent.keyDown(document, { key: "Escape" })

    await waitFor(() => {
      expect(screen.queryByTestId("notes-keyword-picker-modal")).not.toBeInTheDocument()
    })
    await waitFor(() => {
      expect(browseButton).toHaveFocus()
    })
  })

  it("moves focus into the first-class Graph view without opening a dialog", async () => {
    renderPage()
    await seedAndSaveNote()

    expect(screen.getByTestId("notes-view-mode-list")).toBeInTheDocument()
    expect(screen.getByTestId("notes-view-mode-timeline")).toBeInTheDocument()
    expect(screen.getByTestId("notes-view-mode-inbox")).toBeInTheDocument()
    expect(screen.getByTestId("notes-view-mode-moodboard")).toBeInTheDocument()
    expect(screen.getByTestId("notes-view-mode-graph")).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Split" }))
    const openGraphButton = await screen.findByTestId("notes-open-graph-view")
    openGraphButton.focus()
    fireEvent.click(openGraphButton)

    await waitFor(() => {
      expect(screen.getByTestId("notes-graph-workspace")).toHaveFocus()
    })
    expect(screen.getByTestId("notes-list-panel")).toBeInTheDocument()
    expect(mockNavigate).not.toHaveBeenCalled()
  })

  it("does not request or reveal prior graph data before principal authority is ready", async () => {
    mockGetCurrentUser.mockReset()
    mockGetCurrentUser.mockReturnValue(new Promise(() => undefined))
    renderPage()
    await seedAndSaveNote()

    fireEvent.click(screen.getByRole("button", { name: "Split" }))
    fireEvent.click(await screen.findByTestId("notes-open-graph-view"))
    await screen.findByTestId("notes-graph-workspace")

    expect(
      mockBgRequest.mock.calls.some(([request]) =>
        String(request?.path || "").startsWith("/api/v1/notes/graph?")
      )
    ).toBe(false)
    expect(
      mockBgRequest.mock.calls.some(([request]) =>
        String(request?.path || "").includes("/graph/suggestions")
      )
    ).toBe(false)
    expect(screen.queryByTestId("notes-graph-canvas")).not.toBeInTheDocument()
  })
})
