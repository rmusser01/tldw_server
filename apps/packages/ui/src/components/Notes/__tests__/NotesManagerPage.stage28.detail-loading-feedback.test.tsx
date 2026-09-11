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

const commonListResponse = {
  items: [
    {
      id: "note-1",
      title: "First note",
      content: "First content",
      metadata: { keywords: [] },
      version: 1,
      last_modified: "2026-02-18T10:00:00.000Z"
    },
    {
      id: "note-2",
      title: "Second note",
      content: "Second content",
      metadata: { keywords: [] },
      version: 1,
      last_modified: "2026-02-18T10:05:00.000Z"
    }
  ],
  pagination: { total_items: 2, total_pages: 1 }
}

describe("NotesManagerPage stage 28 editor detail loading feedback", () => {
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

  it("announces detail loading state with polite status semantics", async () => {
    let resolveDetail: ((value: unknown) => void) | null = null
    mockBgRequest.mockImplementation(async (request: { path?: string; method?: string }) => {
      const path = String(request.path || "")
      const method = String(request.method || "GET").toUpperCase()
      if (path.startsWith("/api/v1/notes/?")) return commonListResponse
      if (path === "/api/v1/notes/note-1" && method === "GET") {
        return await new Promise((resolve) => {
          resolveDetail = resolve
        })
      }
      if (path.startsWith("/api/v1/notes/note-1/neighbors")) {
        return { nodes: [{ id: "note-1", type: "note", label: "First note" }], edges: [] }
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

    renderPage()
    fireEvent.click(await screen.findByTestId("notes-open-button-note-1"))

    const loading = await screen.findByTestId("notes-editor-loading-detail")
    expect(loading).toHaveAttribute("role", "status")
    expect(loading).toHaveAttribute("aria-live", "polite")
    expect(screen.getByTestId("notes-editor-region")).toHaveAttribute("aria-busy", "true")

    resolveDetail?.({
      id: "note-1",
      title: "First note",
      content: "First content",
      metadata: { keywords: [] },
      version: 1,
      last_modified: "2026-02-18T10:00:00.000Z"
    })

    await waitFor(() => {
      expect(screen.queryByTestId("notes-editor-loading-detail")).not.toBeInTheDocument()
    })
    expect(screen.getByTestId("notes-editor-region")).toHaveAttribute("aria-busy", "false")
  })

  it("keeps previous editor content visible while next note details are loading", async () => {
    let resolveSecondDetail: ((value: unknown) => void) | null = null
    mockBgRequest.mockImplementation(async (request: { path?: string; method?: string }) => {
      const path = String(request.path || "")
      const method = String(request.method || "GET").toUpperCase()
      if (path.startsWith("/api/v1/notes/?")) return commonListResponse
      if (path === "/api/v1/notes/note-1" && method === "GET") {
        return {
          id: "note-1",
          title: "First note",
          content: "First content",
          metadata: { keywords: [] },
          version: 1,
          last_modified: "2026-02-18T10:00:00.000Z"
        }
      }
      if (path === "/api/v1/notes/note-2" && method === "GET") {
        return await new Promise((resolve) => {
          resolveSecondDetail = resolve
        })
      }
      if (path.includes("/neighbors")) {
        return { nodes: [], edges: [] }
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

    renderPage()

    fireEvent.click(await screen.findByTestId("notes-open-button-note-1"))
    await waitFor(() => {
      expect((screen.getByLabelText("Note content") as HTMLTextAreaElement).value).toBe(
        "First content"
      )
    })

    fireEvent.click(await screen.findByTestId("notes-open-button-note-2"))
    expect(await screen.findByTestId("notes-editor-loading-detail")).toBeInTheDocument()
    expect((screen.getByLabelText("Note content") as HTMLTextAreaElement).value).toBe("First content")

    resolveSecondDetail?.({
      id: "note-2",
      title: "Second note",
      content: "Second content",
      metadata: { keywords: [] },
      version: 1,
      last_modified: "2026-02-18T10:05:00.000Z"
    })

    await waitFor(() => {
      expect((screen.getByLabelText("Note content") as HTMLTextAreaElement).value).toBe(
        "Second content"
      )
    })
  })
})
