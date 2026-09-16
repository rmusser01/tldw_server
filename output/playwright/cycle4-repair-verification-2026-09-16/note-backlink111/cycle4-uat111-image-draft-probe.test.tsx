import type { AssistantSelection } from "@/types/assistant-selection"
import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import NotesManagerPage from "@/components/Notes/NotesManagerPage"
import { useStoreMessageOption } from "@/store/option"
import { usePlaygroundSessionStore } from "@/store/playground-session"
import { Playground } from "@/components/Option/Playground/Playground"

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
  useNavigate: () => mockNavigate,
  useLocation: () => ({ pathname: "/chat", search: "", hash: "", state: null, key: "chat" })
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: mockBgRequest, bgRequestClient: mockBgRequest, bgStream: vi.fn()
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

const chatAuthority = vi.hoisted(() => ({
  owner: "A", selection: { kind: "character", id: "5", name: "Robot", metadata: { selectionMode: "tracked" } } as AssistantSelection | null,
  controller: new AbortController(), setSelection: vi.fn()
}))
vi.mock("@/hooks/useSelectedAssistant", () => ({ useSelectedAssistant: () => [chatAuthority.selection, chatAuthority.setSelection] }))
vi.mock("@/components/Notes/hooks/useNotesGraphAuthorityScope", () => ({ useNotesGraphAuthorityScope: () => chatAuthority.owner }))
vi.mock("@/services/service-prompts", () => ({ loadServicePromptSnapshot: async () => ({
  scopeKey: "scope-A", scopeSignal: chatAuthority.controller.signal, scopeInvalidatedSignal: chatAuthority.controller.signal,
  requestScope: { config: { serverUrl: "http://server", authMode: "multi-user" }, userId: "A" }, release: vi.fn()
}) }))

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
    getCharacter: mockGetCharacter,
    getConfig: async () => ({ serverUrl: "http://server", authMode: "multi-user", accessToken: "test-token" }),
    getProvidersStatus: async () => null
  }
}))


vi.mock("@/hooks/useMessageOption", () => ({ useMessageOption: () => ({
  ...useStoreMessageOption(), selectedAssistant: chatAuthority.selection,
  selectedCharacter: chatAuthority.selection?.kind === "character" ? { id: chatAuthority.selection.id, name: chatAuthority.selection.name } : null,
  setSelectedAssistant: chatAuthority.setSelection, setSelectedCharacter: vi.fn(), onSubmit: vi.fn(), regenerateLastMessage: vi.fn()
}) }))
vi.mock("@/components/Option/Playground/PlaygroundForm", () => ({ PlaygroundForm: () => <div data-testid="loaded-chat-composer" /> }))
vi.mock("@/components/Option/Playground/PlaygroundChat", () => ({ PlaygroundChat: () => <div data-testid="loaded-chat-body">{useStoreMessageOption(state => state.messages).map(row => <p key={row.id}>{row.message}</p>)}</div> }))
vi.mock("@/components/Option/Playground/CharacterChatSessionsPanel", () => ({ CharacterChatSessionsPanel: () => null }))
vi.mock("@/components/Sidepanel/Chat/ArtifactsPanel", () => ({ ArtifactsPanel: () => null }))
vi.mock("@/hooks/useCharacterGreeting", () => ({ useCharacterGreeting: () => undefined }))
vi.mock("@/hooks/useLoadLocalConversation", () => ({ useLoadLocalConversation: () => vi.fn() }))
vi.mock("@/services/app", () => ({ webUIResumeLastChat: async () => false }))
vi.mock("@/services/chat-settings", () => ({ syncChatSettingsForServerChat: async () => null }))
vi.mock("@/services/chat-surface-scope", async original => ({ ...await original<typeof import("@/services/chat-surface-scope")>(), buildChatSurfaceScopeKeyFromConfig: () => "scope-A" }))
vi.mock("@/services/tldw-server", async original => ({ ...await original<typeof import("@/services/tldw-server")>(), fetchChatModels: async () => [] }))

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
    chatAuthority.owner = "A"
    chatAuthority.controller = new AbortController()
    chatAuthority.selection = { kind: "character", id: "5", name: "Robot", metadata: { selectionMode: "tracked" } }
    chatAuthority.setSelection.mockImplementation(async (selection, options) => { if (options?.isCurrent?.() !== false) chatAuthority.selection = selection })
    useStoreMessageOption.setState({ messages: [], history: [], historyId: null, serverChatId: "robot", serverChatCharacterId: "5", serverChatMetaLoaded: true, streaming: false, isProcessing: false })
    usePlaygroundSessionStore.getState().saveSession({ historyId: "robot-history", serverChatId: "robot", scopeKey: "scope-A" })
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
      expect(mockNavigate).toHaveBeenCalledWith("/chat")
    })
    expect(openSpy).not.toHaveBeenCalled()
    openSpy.mockRestore()
  })
  it("restores Cedar identity and canonical saved-message IDs before navigation, replacing the Robot restore target", async () => {
    configureCommonRequests("cedar")
    mockGetChat.mockResolvedValue({ id: "cedar", title: "Cedar chat", character_id: 4, source: "webui-character-chat", version: 3 })
    mockListChatMessages.mockResolvedValue([{ id: "q", role: "user", content: "When?", created_at: "2026-02-18T10:00:00Z", version: 1 }, { id: "a", role: "assistant", content: "08:30", created_at: "2026-02-18T10:01:00Z", version: 1 }])
    const atNavigation: { chat: string | null; selection: AssistantSelection | null }[] = []
    mockNavigate.mockImplementation(() => atNavigation.push({ chat: useStoreMessageOption.getState().serverChatId, selection: chatAuthority.selection }))
    renderPage()
    fireEvent.click(await screen.findByTestId("notes-open-button-note-backlink-1"))
    fireEvent.click(await screen.findByTestId("notes-overflow-menu-button"))
    fireEvent.click(await screen.findByText(/open linked conversation/i))
    await waitFor(() => expect(mockNavigate).toHaveBeenCalledWith("/chat"))
    expect(atNavigation[0]).toMatchObject({ chat: "cedar", selection: { kind: "character", id: "4", metadata: { selectionMode: "tracked" } } })
    expect(useStoreMessageOption.getState()).toMatchObject({ serverChatId: "cedar", serverChatCharacterId: 4, serverChatVersion: 3, serverChatMetaLoaded: true, temporaryChat: false })
    expect(useStoreMessageOption.getState().messages).toMatchObject([{ serverMessageId: "q" }, { serverMessageId: "a", message: "08:30" }])
    expect(usePlaygroundSessionStore.getState()).toMatchObject({ serverChatId: "cedar", historyId: null, trackedCharacterId: "4", scopeKey: "scope-A" })
  })

  it("keeps the linked Cedar reply through the real Playground mismatch and persisted-session initialization effects", async () => {
    configureCommonRequests("cedar")
    mockGetChat.mockResolvedValue({ id: "cedar", title: "Cedar chat", character_id: 4, source: "webui-character-chat", version: 3 })
    mockListChatMessages.mockResolvedValue([{ id: "q", role: "user", content: "When?", created_at: "2026-02-18T10:00:00Z", version: 1 }, { id: "a", role: "assistant", content: "Cedar opens at 08:30.", created_at: "2026-02-18T10:01:00Z", version: 1 }])
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
    const Route = () => {
      const [chat, setChat] = React.useState(false)
      mockNavigate.mockImplementation(() => setChat(true))
      return chat ? <Playground /> : <NotesManagerPage />
    }
    render(<QueryClientProvider client={queryClient}><Route /></QueryClientProvider>)
    fireEvent.click(await screen.findByTestId("notes-open-button-note-backlink-1"))
    fireEvent.click(await screen.findByTestId("notes-overflow-menu-button"))
    fireEvent.click(await screen.findByText(/open linked conversation/i))
    expect(await screen.findByText("Cedar opens at 08:30.")).toBeInTheDocument()
    await waitFor(() => expect(screen.getByTestId("loaded-chat-composer")).toBeInTheDocument())
    expect(useStoreMessageOption.getState()).toMatchObject({ serverChatId: "cedar", serverChatCharacterId: 4 })
    expect(usePlaygroundSessionStore.getState()).toMatchObject({ serverChatId: "cedar", trackedCharacterId: "4" })
  })

  it.each(["stream", "new-draft", "note-edit", "account"])("does not replace current work when %s changes during a linked read", async change => {
    configureCommonRequests("cedar")
    const chat = { id: "cedar", title: "Cedar chat", character_id: 4, source: "webui-character-chat" }
    let release!: (rows: unknown[]) => void
    mockGetChat.mockResolvedValue(chat)
    mockListChatMessages.mockReturnValue(new Promise(resolve => { release = resolve }))
    renderPage()
    fireEvent.click(await screen.findByTestId("notes-open-button-note-backlink-1"))
    fireEvent.click(await screen.findByTestId("notes-overflow-menu-button"))
    fireEvent.click(await screen.findByText(/open linked conversation/i))
    await waitFor(() => expect(mockListChatMessages).toHaveBeenCalled())
    act(() => {
      if (change === "stream") useStoreMessageOption.setState({ streaming: true })
      if (change === "new-draft") useStoreMessageOption.setState({ messages: [{ id: "draft", role: "user", isBot: false, name: "You", message: "Keep this thought" }] })
      if (change === "account") chatAuthority.controller.abort()
    })
    if (change === "note-edit") fireEvent.change(screen.getByPlaceholderText("Write your note here... (Markdown supported)"), { target: { value: "New note edits" } })
    await act(async () => { release([{ id: "a", role: "assistant", content: "Late Cedar", version: 1 }]); await Promise.resolve(); await Promise.resolve() })
    expect(mockNavigate).not.toHaveBeenCalled()
    expect(useStoreMessageOption.getState().serverChatId).toBe("robot")
    expect(chatAuthority.selection.id).toBe("5")
  })

  it.each(["image-only-draft", "image-error-draft"])("opens past only a retained assistant display error (%s)", async kind => {
    configureCommonRequests("normal-saved")
    mockGetChat.mockResolvedValue({ id: "normal-saved", title: "Recovered chat", source: "webui-chat" })
    mockListChatMessages.mockResolvedValue([{ id: "server-user", role: "user", content: "Question" }, { id: "server-answer", role: "assistant", content: "Answer" }])
    const error = '__tldw_error__:{"summary":"Provider failed","hint":"Try again","detail":"Stream completion failed"}'
    useStoreMessageOption.setState({ historyId: "local-normal", serverChatId: "normal-saved", messages: [
      { id: "local-user", serverMessageId: "server-user", isBot: false, name: "You", message: "Question" },
      { id: "local-error", isBot: kind !== "user-error-text", role: kind === "user-error-text" ? "user" : "assistant", name: "Assistant", images: ["data:image/png;base64,YQ=="], message: kind === "image-only-draft" ? "" : error },
      { id: "local-answer", serverMessageId: "server-answer", isBot: true, name: "Assistant", message: "Answer" }
    ], streaming: false, isProcessing: false })
    renderPage()
    fireEvent.click(await screen.findByTestId("notes-open-button-note-backlink-1"))
    fireEvent.click(await screen.findByTestId("notes-overflow-menu-button"))
    fireEvent.click(await screen.findByText(/open linked conversation/i))
    if (kind === "display-error") {
      await waitFor(() => expect(mockNavigate).toHaveBeenCalledWith("/chat"))
      expect(useStoreMessageOption.getState().messages).toMatchObject([{ serverMessageId: "server-user" }, { serverMessageId: "server-answer" }])
    } else {
      expect(mockNavigate).not.toHaveBeenCalled()
      expect(mockListChatMessages).not.toHaveBeenCalled()
      expect(mockMessageWarning).toHaveBeenCalled()
    }
  })

  it("does not cancel an owned linked read when only an assistant display error is restored", async () => {
    configureCommonRequests("normal-saved")
    mockGetChat.mockResolvedValue({ id: "normal-saved", title: "Recovered chat", source: "webui-chat" })
    let release!: (rows: unknown[]) => void
    mockListChatMessages.mockReturnValue(new Promise(resolve => { release = resolve }))
    renderPage()
    fireEvent.click(await screen.findByTestId("notes-open-button-note-backlink-1"))
    fireEvent.click(await screen.findByTestId("notes-overflow-menu-button"))
    fireEvent.click(await screen.findByText(/open linked conversation/i))
    await waitFor(() => expect(mockListChatMessages).toHaveBeenCalled())
    act(() => useStoreMessageOption.setState({ messages: [{ id: "restored-error", role: "assistant", isBot: true, name: "Assistant", message: '__tldw_error__:{"summary":"Provider failed","hint":"Try again"}' }] }))
    await act(async () => { release([{ id: "saved-answer", role: "assistant", content: "Recovered answer" }]); await Promise.resolve(); await Promise.resolve() })
    await waitFor(() => expect(mockNavigate).toHaveBeenCalledWith("/chat"))
    expect(useStoreMessageOption.getState().messages).toMatchObject([{ serverMessageId: "saved-answer", message: "Recovered answer" }])
  })

  it.each([false, true])("opens a normally acknowledged saved Chat while retaining the unsent-draft guard (draft=%s)", async (withDraft) => {
    configureCommonRequests("normal-saved")
    mockGetChat.mockResolvedValue({ id: "normal-saved", title: "Normal saved chat", character_id: null, assistant_kind: null, assistant_id: null, source: "webui-chat", version: 1 })
    mockListChatMessages.mockResolvedValue([
      { id: "server-user", sender: "user", content: "Question" },
      { id: "server-answer", sender: "assistant", content: "Answer" }
    ])
    chatAuthority.selection = null
    useStoreMessageOption.setState({ historyId: "local-normal", serverChatId: "normal-saved", serverChatCharacterId: null, serverChatMetaLoaded: true, messages: [
      { id: "local-user", serverMessageId: "server-user", isBot: false, name: "You", message: "Question" },
      { id: "local-answer", serverMessageId: "server-answer", isBot: true, name: "Assistant", message: "Answer" },
      ...(withDraft ? [{ id: "unsent", isBot: false, name: "You", message: "Question" }] : [])
    ], history: [], streaming: false, isProcessing: false })
    renderPage()
    fireEvent.click(await screen.findByTestId("notes-open-button-note-backlink-1"))
    fireEvent.click(await screen.findByTestId("notes-overflow-menu-button"))
    fireEvent.click(await screen.findByText(/open linked conversation/i))
    if (withDraft) {
      expect(mockMessageWarning).toHaveBeenCalledWith("Finish or save the current chat before opening the linked conversation.")
      expect(mockNavigate).not.toHaveBeenCalled()
      expect(mockListChatMessages).not.toHaveBeenCalled()
    } else {
      await waitFor(() => expect(mockNavigate).toHaveBeenCalledWith("/chat"))
      expect(useStoreMessageOption.getState().serverChatId).toBe("normal-saved")
      expect(chatAuthority.selection).toBeNull()
    }
  })

})
