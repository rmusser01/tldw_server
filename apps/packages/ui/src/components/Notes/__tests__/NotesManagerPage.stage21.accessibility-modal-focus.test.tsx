import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
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
  mockCytoscapeFactory,
  mockAuthorityState,
  mockCanonicalConfig,
  mockDetailState
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
    mockCytoscapeFactory: cytoscapeFactory,
    mockAuthorityState: { principalId: 1 },
    mockCanonicalConfig: {
      serverUrl: "https://notes.example.test",
      authMode: "single-user" as const,
      authSource: "cookie-session" as const
    },
    mockDetailState: { promise: null as Promise<Record<string, unknown>> | null }
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

vi.mock("@/hooks/useCanonicalConnectionConfig", () => ({
  useCanonicalConnectionConfig: () => ({
    config: mockCanonicalConfig,
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
  default: ({
    notes,
    selectedId,
    onSelectNote
  }: {
    notes?: Array<{ id: string; title?: string }>
    selectedId: string | number | null
    onSelectNote: (id: string) => void
  }) => (
    <div data-testid="notes-list-panel">
      <output data-testid="notes-list-selected-id">{selectedId ?? "none"}</output>
      {(notes ?? []).map((note) => (
        <button
          key={note.id}
          type="button"
          data-testid={`notes-list-note-${note.id}`}
          onClick={() => onSelectNote(note.id)}>
          {note.title ?? note.id}
        </button>
      ))}
    </div>
  )
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
  await screen.findByTestId("notes-list-note-note-a")
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
    mockAuthorityState.principalId = 1
    mockDetailState.promise = null
    mockGetCurrentUser.mockImplementation(async () => ({
      id: mockAuthorityState.principalId,
      username: `notes-user-${mockAuthorityState.principalId}`,
      is_active: true
    }))

    mockBgRequest.mockImplementation(async (request: { path?: string; method?: string }) => {
      const path = String(request.path || "")
      const method = String(request.method || "GET").toUpperCase()

      if (path.startsWith("/api/v1/notes/?")) {
        return mockAuthorityState.principalId === 1
          ? {
              items: [
                {
                  id: "note-a",
                  title: "Focus handoff note",
                  content: "Seed content",
                  deleted: false
                }
              ],
              pagination: { total_items: 1, total_pages: 1 }
            }
          : { items: [], pagination: { total_items: 0, total_pages: 1 } }
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
        if (mockDetailState.promise) return mockDetailState.promise
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
          nodes: [
            {
              id: "note:note-a",
              type: "note",
              label: "Focus handoff note",
              created_at: null,
              deleted: false,
              degree: 0,
              tag_count: 0,
              primary_source_id: null
            }
          ],
          edges: [],
          truncated: false,
          truncated_by: [],
          has_more: false,
          cursor: null,
          limits: { max_nodes: 120, max_edges: 480, max_degree: 40 },
          radius_cap_applied: false,
          active_note_count: 1,
          all_notes_note_cap: 100,
          all_notes_eligible: true
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
    fireEvent.click(await screen.findByTestId("notes-view-mode-graph"))
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

  it("drops account A list, selection, and canvas evidence before empty account B can request graph data", async () => {
    renderPage()
    fireEvent.click(await screen.findByTestId("notes-list-note-note-a"))
    await waitFor(() => {
      expect(screen.getByPlaceholderText("Title")).toHaveValue(
        "Focus handoff note"
      )
    })

    fireEvent.click(screen.getByRole("button", { name: "Split" }))
    fireEvent.click(await screen.findByTestId("notes-open-graph-view"))
    await screen.findByTestId("notes-graph-canvas")
    const boundaryCallIndex = mockBgRequest.mock.calls.length

    mockAuthorityState.principalId = 2
    act(() => {
      window.dispatchEvent(
        new CustomEvent("tldw:auth-principal-changed", {
          detail: { kind: "switch" }
        })
      )
    })

    expect(screen.queryByTestId("notes-list-note-note-a")).not.toBeInTheDocument()
    expect(screen.queryByTestId("notes-graph-canvas")).not.toBeInTheDocument()
    await waitFor(() => {
      expect(screen.getByTestId("notes-list-selected-id")).toHaveTextContent(
        "none"
      )
    })
    expect(screen.getByTestId("notes-editor-empty-state")).toBeInTheDocument()

    const accountBCalls = mockBgRequest.mock.calls.slice(boundaryCallIndex)
    expect(
      accountBCalls.some(([request]) => {
        const path = String(request?.path || "")
        return (
          (path.includes("/graph?") || path.includes("/graph/suggestions")) &&
          path.includes("note-a")
        )
      })
    ).toBe(false)
  })

  it("keeps the selected editor note through same-principal cookie revalidation", async () => {
    renderPage()
    fireEvent.click(await screen.findByTestId("notes-list-note-note-a"))
    const title = screen.getByPlaceholderText("Title")
    await waitFor(() => expect(title).toHaveValue("Focus handoff note"))
    const resume = new Promise<{
      id: number
      username: string
      is_active: boolean
    }>((resolve) => {
      window.setTimeout(
        () => resolve({ id: 1, username: "notes-user-1", is_active: true }),
        20
      )
    })
    mockGetCurrentUser.mockReturnValueOnce(resume)

    act(() => window.dispatchEvent(new Event("focus")))

    expect(title).toHaveValue("Focus handoff note")
    await waitFor(() => {
      expect(screen.getByTestId("notes-list-selected-id")).toHaveTextContent(
        "note-a"
      )
    })
    expect(title).toHaveValue("Focus handoff note")
  })

  it("ignores an account A detail completion after account B becomes authoritative", async () => {
    let resolveDetail!: (value: Record<string, unknown>) => void
    mockDetailState.promise = new Promise((resolve) => {
      resolveDetail = resolve
    })
    renderPage()
    fireEvent.click(await screen.findByTestId("notes-list-note-note-a"))
    await waitFor(() => {
      expect(mockBgRequest).toHaveBeenCalledWith(
        expect.objectContaining({ path: "/api/v1/notes/note-a" })
      )
    })

    mockAuthorityState.principalId = 2
    act(() => {
      window.dispatchEvent(
        new CustomEvent("tldw:auth-principal-changed", {
          detail: { kind: "switch" }
        })
      )
    })
    await waitFor(() => {
      expect(screen.queryByTestId("notes-list-note-note-a")).not.toBeInTheDocument()
    })

    await act(async () => {
      resolveDetail({
        id: "note-a",
        title: "Stale account A detail",
        content: "Must not cross the account boundary",
        metadata: { keywords: [] },
        version: 1
      })
      await Promise.resolve()
    })

    expect(screen.getByTestId("notes-list-selected-id")).toHaveTextContent(
      "none"
    )
    expect(screen.getByPlaceholderText("Title")).toHaveValue("")
  })
})
