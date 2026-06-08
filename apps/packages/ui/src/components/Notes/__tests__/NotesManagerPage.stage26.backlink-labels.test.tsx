import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
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

const buildNoteListResponse = (conversationId: string) => ({
  items: [
    {
      id: "note-backlink-1",
      title: "Backlink note",
      content: "note content",
      conversation_id: conversationId,
      message_id: "msg-42",
      metadata: { keywords: [] },
      version: 1,
      last_modified: "2026-02-18T10:00:00.000Z"
    }
  ],
  pagination: { total_items: 1, total_pages: 1 }
})

const buildDetailResponse = (conversationId: string) => ({
  id: "note-backlink-1",
  title: "Backlink note",
  content: "note content",
  conversation_id: conversationId,
  message_id: "msg-42",
  metadata: { keywords: [] },
  version: 1,
  last_modified: "2026-02-18T10:00:00.000Z"
})

describe("NotesManagerPage stage 26 conversation backlink labels", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockConfirmDanger.mockResolvedValue(true)
    mockGetSetting.mockResolvedValue(null)
    mockSetSetting.mockResolvedValue(undefined)
    mockClearSetting.mockResolvedValue(undefined)
    mockGetAllNoteKeywordStats.mockResolvedValue([])
    mockSearchNoteKeywords.mockResolvedValue([])
    mockInitialize.mockResolvedValue(undefined)
    mockListChatMessages.mockResolvedValue([])
    mockGetCharacter.mockResolvedValue(null)
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  const configureCommonRequests = (conversationId: string) => {
    mockBgRequest.mockImplementation(async (request: { path?: string; method?: string }) => {
      const path = String(request.path || "")
      const method = String(request.method || "GET").toUpperCase()
      if (path.startsWith("/api/v1/notes/?")) {
        return buildNoteListResponse(conversationId)
      }
      if (path === "/api/v1/notes/note-backlink-1" && method === "GET") {
        return buildDetailResponse(conversationId)
      }
      if (path.startsWith("/api/v1/notes/note-backlink-1/neighbors")) {
        return {
          nodes: [{ id: "note-backlink-1", type: "note", label: "Backlink note" }],
          edges: []
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
      return {}
    })
  }

  it("shows conversation title labels with UUID debug tooltip in list and header", async () => {
    configureCommonRequests("conv-1234")
    mockGetChat.mockResolvedValue({
      id: "conv-1234",
      title: "Research session",
      topic_label: "Topic fallback"
    })

    renderPage()

    const listLabel = await screen.findByText("Research session")
    fireEvent.mouseEnter(listLabel)
    expect(await screen.findByText("Conversation ID: conv-1234")).toBeInTheDocument()

    fireEvent.click(await screen.findByTestId("notes-open-button-note-backlink-1"))
    await waitFor(() => {
      const labels = screen.getAllByText("Research session")
      expect(labels.length).toBeGreaterThan(1)
    })
    expect(mockGetChat).toHaveBeenCalledWith("conv-1234")
  })

  it("falls back to topic label when conversation title is empty", async () => {
    configureCommonRequests("conv-topic")
    mockGetChat.mockResolvedValue({
      id: "conv-topic",
      title: "",
      topic_label: "Topic label"
    })

    renderPage()
    expect(await screen.findByText("Topic label")).toBeInTheDocument()
  })

  it("falls back to raw conversation ID when metadata lookup fails", async () => {
    configureCommonRequests("conv-unavailable")
    mockGetChat.mockRejectedValue(new Error("missing chat"))

    renderPage()
    expect(await screen.findByText("conv-unavailable")).toBeInTheDocument()
    await waitFor(() => {
      expect(mockGetChat).toHaveBeenCalledWith("conv-unavailable")
    })
  })

  it("retries transient conversation label lookups and hydrates once they recover", async () => {
    configureCommonRequests("conv-retry")
    mockGetChat
      .mockRejectedValueOnce(new Error("temporary upstream failure"))
      .mockResolvedValueOnce({
        id: "conv-retry",
        title: "Recovered session",
        topic_label: ""
      })

    renderPage()

    expect(await screen.findByText("conv-retry")).toBeInTheDocument()
    expect(mockGetChat).toHaveBeenCalledTimes(1)

    await waitFor(
      () => {
        expect(screen.getByText("Recovered session")).toBeInTheDocument()
      },
      { timeout: 3000 }
    )
    expect(mockGetChat).toHaveBeenCalledTimes(2)
  }, 7000)

  it("does not re-request a missing conversation label after the backlink is selected", async () => {
    configureCommonRequests("conv-unavailable")
    mockGetChat.mockRejectedValue(new Error("Chat session conv-unavailable not found"))

    renderPage()
    expect(await screen.findByText("conv-unavailable")).toBeInTheDocument()

    fireEvent.click(await screen.findByTestId("notes-open-button-note-backlink-1"))
    await waitFor(() => {
      expect(mockGetChat).toHaveBeenCalledTimes(1)
    })

    expect(mockGetChat).toHaveBeenCalledTimes(1)
  })

  it("does not auto-fetch UUID conversation labels until the linked note is opened", async () => {
    const uuidConversationId = "acd86f3a-492e-4a8b-90fb-ff282b9721fb"
    configureCommonRequests(uuidConversationId)
    mockGetChat.mockRejectedValue(
      new Error(`Chat session ${uuidConversationId} not found`)
    )

    renderPage()
    expect(await screen.findByText(uuidConversationId)).toBeInTheDocument()

    await waitFor(() => {
      expect(mockGetChat).not.toHaveBeenCalled()
    })

    fireEvent.click(await screen.findByTestId("notes-open-button-note-backlink-1"))

    await waitFor(() => {
      expect(mockGetChat).toHaveBeenCalledTimes(1)
    })
    expect(mockGetChat).toHaveBeenCalledWith(uuidConversationId)
  })

  it("opens linked conversations in the same tab by default", async () => {
    configureCommonRequests("conv-same-tab")
    mockGetChat.mockResolvedValue({
      id: "conv-same-tab",
      title: "Research session",
      topic_label: "Topic fallback",
      state: "in-progress",
      source: "chat",
      external_ref: null
    })
    mockListChatMessages.mockResolvedValue([
      {
        id: "msg-1",
        role: "user",
        content: "hello",
        created_at: "2026-02-18T10:01:00.000Z",
        version: 1
      }
    ])

    const openSpy = vi.spyOn(window, "open").mockImplementation(() => null)
    renderPage()

    fireEvent.click(await screen.findByTestId("notes-open-button-note-backlink-1"))
    // "Open linked conversation" is now inside the overflow menu
    fireEvent.click(await screen.findByTestId("notes-overflow-menu-button"))
    fireEvent.click(await screen.findByText(/open linked conversation/i))

    await waitFor(() => {
      expect(mockNavigate).toHaveBeenCalledWith("/")
    })
    expect(openSpy).not.toHaveBeenCalled()
    openSpy.mockRestore()
  })
})
