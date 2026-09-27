import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import NotesManagerPage from "../NotesManagerPage"

const authority = vi.hoisted(() => ({ scope: "alice-notes" as string | null, online: true }))
vi.mock("../hooks/useNotesGraphAuthorityScope", () => ({
  useNotesGraphAuthorityScope: () => authority.scope
}))

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
  mockInitialize,
  mockGetChat,
  mockListChatMessages,
  mockGetCharacter
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
  mockClearSetting: vi.fn(),
  mockGetAllNoteKeywordStats: vi.fn(),
  mockSearchNoteKeywords: vi.fn(),
  mockInitialize: vi.fn(),
  mockGetChat: vi.fn(),
  mockListChatMessages: vi.fn(),
  mockGetCharacter: vi.fn()
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
  useServerOnline: () => authority.online
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
    initialize: mockInitialize,
    getChat: mockGetChat,
    listChatMessages: mockListChatMessages,
    getCharacter: mockGetCharacter
  }
}))

const renderPage = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false }
    }
  })
  const page = () => (
    <QueryClientProvider client={queryClient}>
      <NotesManagerPage />
    </QueryClientProvider>
  )
  const view = render(page())
  return { ...view, queryClient, rerenderPage: () => view.rerender(page()) }
}

const configureCommonRequests = (neighborsPayload: Record<string, any>) => {
  mockBgRequest.mockImplementation(async (request: { path?: string; method?: string }) => {
    const path = String(request.path || "")
    const method = String(request.method || "GET").toUpperCase()
    if (path.startsWith("/api/v1/notes/?")) {
      return {
        items: [
          {
            id: "note-source-1",
            title: "Source note",
            content: "source content",
            metadata: { keywords: [] },
            version: 1,
            last_modified: "2026-02-18T10:00:00.000Z"
          }
        ],
        pagination: { total_items: 1, total_pages: 1 }
      }
    }
    if (path === "/api/v1/notes/note-source-1" && method === "GET") {
      return {
        id: "note-source-1",
        title: "Source note",
        content: "source content",
        metadata: { keywords: [] },
        version: 1,
        last_modified: "2026-02-18T10:00:00.000Z"
      }
    }
    if (path.startsWith("/api/v1/notes/note-source-1/neighbors")) {
      return neighborsPayload
    }
    if (path === "/api/v1/admin/notes/title-settings" && method === "GET") {
      return {
        llm_enabled: false,
        default_strategy: "heuristic",
        effective_strategy: "heuristic",
        strategies: ["heuristic"]
      }
    }
    return {}
  })
}

describe("NotesManagerPage stage 27 source link surfacing", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    authority.scope = "alice-notes"
    authority.online = true
    mockConfirmDanger.mockResolvedValue(true)
    mockGetSetting.mockResolvedValue(null)
    mockSetSetting.mockResolvedValue(undefined)
    mockClearSetting.mockResolvedValue(undefined)
    mockGetAllNoteKeywordStats.mockResolvedValue([])
    mockSearchNoteKeywords.mockResolvedValue([])
    mockInitialize.mockResolvedValue(undefined)
    mockGetChat.mockResolvedValue(null)
    mockListChatMessages.mockResolvedValue([])
    mockGetCharacter.mockResolvedValue(null)
  })

  it("opens and edits an ordinary note without requesting graph data", async () => {
    configureCommonRequests({ nodes: [], edges: [] })
    localStorage.setItem("notes-section-connections", "true")
    renderPage()
    fireEvent.click(await screen.findByTestId("notes-open-button-note-source-1"))
    const title = await screen.findByDisplayValue("Source note")
    fireEvent.change(title, { target: { value: "Updated title" } })
    await act(async () => {})
    expect(title).toHaveValue("Updated title")
    expect(screen.getByTestId("notes-section-connections-toggle")).toHaveAttribute("aria-expanded", "false")
    expect(mockBgRequest.mock.calls.filter(([request]) => String(request.path).includes("/neighbors"))).toHaveLength(0)
    localStorage.removeItem("notes-section-connections")
  })

  it("reports denied connections without retrying after the user opens the section", async () => {
    configureCommonRequests({ nodes: [], edges: [] })
    const requests = mockBgRequest.getMockImplementation()!
    mockBgRequest.mockImplementation((request) => {
      if (String(request.path).includes("/neighbors")) {
        return Promise.reject(Object.assign(new Error("Permission denied: missing notes.graph.read"), { status: 403 }))
      }
      return requests(request)
    })
    const view = renderPage()
    view.queryClient.setQueryData(["note-graph-neighbors", "alice-notes", "note-source-1", 0], {
      nodes: [{ id: "source:web:media-77", type: "source", label: "Previously loaded source" }],
      edges: [{ id: "cached-edge", source: "note-source-1", target: "source:web:media-77", type: "source_membership" }]
    })
    fireEvent.click(await screen.findByTestId("notes-open-button-note-source-1"))
    await screen.findByDisplayValue("Source note")
    fireEvent.click(screen.getByTestId("notes-section-connections-toggle"))
    expect(await screen.findByText("Note connections are unavailable for this account.")).toBeVisible()
    expect(screen.queryByText("Previously loaded source")).toBeNull()
    expect(screen.queryByTestId("notes-related-retry")).toBeNull()
    fireEvent.click(screen.getByTestId("notes-section-connections-toggle"))
    fireEvent.click(screen.getByTestId("notes-section-connections-toggle"))
    await act(async () => { window.dispatchEvent(new Event("focus")) })
    authority.online = false
    view.rerenderPage()
    authority.online = true
    view.rerenderPage()
    await act(async () => {})
    expect(mockBgRequest.mock.calls.filter(([request]) => String(request.path).includes("/neighbors"))).toHaveLength(1)
  })

  it("shows connections as not loaded when opened offline, then loads after reconnecting", async () => {
    configureCommonRequests({ nodes: [], edges: [] })
    const view = renderPage()
    fireEvent.click(await screen.findByTestId("notes-open-button-note-source-1"))
    await screen.findByDisplayValue("Source note")
    authority.online = false
    view.rerenderPage()
    fireEvent.click(screen.getByTestId("notes-section-connections-toggle"))
    expect(await screen.findByText("Connect to the server to load note connections.")).toBeVisible()
    expect(screen.queryByTestId("notes-manual-links-empty")).toBeNull()
    expect(screen.queryByTestId("notes-related-empty")).toBeNull()
    expect(screen.queryByTestId("notes-backlinks-empty")).toBeNull()
    expect(mockBgRequest.mock.calls.filter(([request]) => String(request.path).includes("/neighbors"))).toHaveLength(0)
    authority.online = true
    view.rerenderPage()
    expect(await screen.findByTestId("notes-related-empty")).toBeVisible()
    expect(mockBgRequest.mock.calls.filter(([request]) => String(request.path).includes("/neighbors"))).toHaveLength(1)
  })

  it("discards a graph response after authority changes and requires a new explicit request", async () => {
    configureCommonRequests({ nodes: [], edges: [] })
    const requests = mockBgRequest.getMockImplementation()!
    let resolveGraph!: (value: unknown) => void
    mockBgRequest.mockImplementation((request) => {
      if (String(request.path).includes("/neighbors")) return new Promise(resolve => { resolveGraph = resolve })
      return requests(request)
    })
    const view = renderPage()
    fireEvent.click(await screen.findByTestId("notes-open-button-note-source-1"))
    fireEvent.click(await screen.findByTestId("notes-section-connections-toggle"))
    await waitFor(() => expect(resolveGraph).toBeDefined())
    authority.scope = "bob-notes"
    view.rerenderPage()
    await act(async () => { resolveGraph({ nodes: [{ id: "source:web:private", type: "source", label: "Private source" }], edges: [] }) })
    await waitFor(() => expect(view.queryClient.getQueryData(["note-graph-neighbors", "alice-notes", "note-source-1", 0])).toBeNull())
    expect(screen.queryByText("Private source")).toBeNull()
    fireEvent.click(await screen.findByTestId("notes-open-button-note-source-1"))
    await screen.findByDisplayValue("Source note")
    expect(screen.getByTestId("notes-section-connections-toggle")).toHaveAttribute("aria-expanded", "false")
    expect(mockBgRequest.mock.calls.filter(([request]) => String(request.path).includes("/neighbors"))).toHaveLength(1)
    authority.scope = "alice-notes"
    view.rerenderPage()
    fireEvent.click(await screen.findByTestId("notes-open-button-note-source-1"))
    await screen.findByDisplayValue("Source note")
    expect(screen.getByTestId("notes-section-connections-toggle")).toHaveAttribute("aria-expanded", "false")
    expect(mockBgRequest.mock.calls.filter(([request]) => String(request.path).includes("/neighbors"))).toHaveLength(1)
  })

  it("renders sorted source chips and navigates to media permalink IDs", async () => {
    configureCommonRequests({
      nodes: [
        { id: "note-source-1", type: "note", label: "Source note" },
        { id: "source:web:media-77", type: "source", label: "web: media-77" },
        { id: "source:yt:media-21", type: "source", label: "yt: media-21" }
      ],
      edges: [
        {
          id: "sm-1",
          source: "note-source-1",
          target: "source:web:media-77",
          type: "source_membership",
          directed: false
        },
        {
          id: "sm-2",
          source: "note-source-1",
          target: "source:yt:media-21",
          type: "source_membership",
          directed: false
        }
      ]
    })

    renderPage()

    fireEvent.click(await screen.findByTestId("notes-open-button-note-source-1"))
    fireEvent.click(await screen.findByTestId("notes-section-connections-toggle"))
    const sourceChipNodes = await screen.findAllByTestId(/notes-source-link-/)
    const sourceLabels = sourceChipNodes.map((node) => node.textContent?.trim())
    expect(sourceLabels).toEqual(["Web source: media-77", "YouTube source: media-21"])

    fireEvent.click(screen.getByText("Web source: media-77"))
    expect(mockNavigate).toHaveBeenCalledWith("/media?id=media-77")
  })

  it("opens external URLs in a new tab when source external ref is a URL", async () => {
    configureCommonRequests({
      nodes: [
        { id: "note-source-1", type: "note", label: "Source note" },
        {
          id: "source:web:https://example.com/article",
          type: "source",
          label: "web: https://example.com/article"
        }
      ],
      edges: [
        {
          id: "sm-1",
          source: "note-source-1",
          target: "source:web:https://example.com/article",
          type: "source_membership",
          directed: false
        }
      ]
    })

    const openSpy = vi.spyOn(window, "open").mockImplementation(() => null)
    renderPage()

    fireEvent.click(await screen.findByTestId("notes-open-button-note-source-1"))
    fireEvent.click(await screen.findByTestId("notes-section-connections-toggle"))
    fireEvent.click(await screen.findByText("Web source: https://example.com/article"))

    expect(openSpy).toHaveBeenCalledWith(
      "https://example.com/article",
      "_blank",
      "noopener,noreferrer"
    )
    expect(mockNavigate).not.toHaveBeenCalledWith("/media?id=https%3A%2F%2Fexample.com%2Farticle")
    openSpy.mockRestore()
  })

  it("preserves human-readable source labels when the graph provides them", async () => {
    configureCommonRequests({
      nodes: [
        { id: "note-source-1", type: "note", label: "Source note" },
        {
          id: "source:web:media-77",
          type: "source",
          label: "Captured article title"
        }
      ],
      edges: [
        {
          id: "sm-1",
          source: "note-source-1",
          target: "source:web:media-77",
          type: "source_membership",
          directed: false
        }
      ]
    })

    renderPage()

    fireEvent.click(await screen.findByTestId("notes-open-button-note-source-1"))
    fireEvent.click(await screen.findByTestId("notes-section-connections-toggle"))
    expect(await screen.findByText("Captured article title")).toBeInTheDocument()
  })
})
