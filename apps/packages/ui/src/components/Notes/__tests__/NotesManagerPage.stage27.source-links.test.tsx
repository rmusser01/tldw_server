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
  return render(
    <QueryClientProvider client={queryClient}>
      <NotesManagerPage />
    </QueryClientProvider>
  )
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
    expect(await screen.findByText("Captured article title")).toBeInTheDocument()
  })
})
