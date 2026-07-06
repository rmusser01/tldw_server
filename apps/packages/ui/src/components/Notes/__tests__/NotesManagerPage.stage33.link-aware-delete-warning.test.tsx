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
  mockEndTutorial,
  mockStartTutorial,
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
  mockEndTutorial: vi.fn(),
  mockStartTutorial: vi.fn(),
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

vi.mock("@/store/tutorials", () => {
  const store = Object.assign(() => ({}), {
    getState: () => ({
      endTutorial: mockEndTutorial,
      startTutorial: mockStartTutorial
    })
  })
  return { useTutorialStore: store }
})

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

vi.mock("@/components/Notes/NotesListPanel", () => ({
  default: ({ onSelectNote }: any) => (
    <button data-testid="mock-select-note" onClick={() => onSelectNote("note-delete")}>
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

const baseNoteResponse = {
  id: "note-delete",
  title: "Delete me",
  content: "content",
  version: 3,
  metadata: { keywords: [] },
  last_modified: "2026-02-18T10:00:00.000Z"
}

describe("NotesManagerPage stage 33 link-aware delete warnings", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockConfirmDanger.mockResolvedValue(false)
    mockGetSetting.mockResolvedValue(null)
    mockSetSetting.mockResolvedValue(undefined)
    mockClearSetting.mockResolvedValue(undefined)
  })

  it("adds reference-count warning text when inbound links exist", async () => {
    mockBgRequest.mockImplementation(async (request: { path?: string; method?: string }) => {
      const path = String(request.path || "")
      const method = String(request.method || "GET").toUpperCase()
      if (path.startsWith("/api/v1/notes/?")) {
        return {
          items: [baseNoteResponse],
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
      if (path === "/api/v1/notes/note-delete" && method === "GET") return baseNoteResponse
      if (path.startsWith("/api/v1/notes/note-delete/neighbors")) {
        return {
          nodes: [],
          edges: [
            { id: "e1", source: "note:a", target: "note-delete", type: "wikilink" },
            { id: "e2", source: "note-delete", target: "note:b", type: "backlink" }
          ]
        }
      }
      return {}
    })

    renderPage()
    fireEvent.click(await screen.findByTestId("mock-select-note"))
    fireEvent.click(await screen.findByTestId("mock-delete-note"))

    await waitFor(() => {
      expect(mockConfirmDanger).toHaveBeenCalled()
    })
    const deleteDialogArgs = mockConfirmDanger.mock.calls[0][0]
    expect(String(deleteDialogArgs.content)).toContain("This note is referenced by 2 other notes")
  })

  it("keeps baseline delete confirmation copy when inbound links are absent", async () => {
    mockBgRequest.mockImplementation(async (request: { path?: string; method?: string }) => {
      const path = String(request.path || "")
      const method = String(request.method || "GET").toUpperCase()
      if (path.startsWith("/api/v1/notes/?")) {
        return {
          items: [baseNoteResponse],
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
      if (path === "/api/v1/notes/note-delete" && method === "GET") return baseNoteResponse
      if (path.startsWith("/api/v1/notes/note-delete/neighbors")) {
        return { nodes: [], edges: [] }
      }
      return {}
    })

    renderPage()
    fireEvent.click(await screen.findByTestId("mock-select-note"))
    fireEvent.click(await screen.findByTestId("mock-delete-note"))

    await waitFor(() => {
      expect(mockConfirmDanger).toHaveBeenCalled()
    })
    expect(mockEndTutorial).toHaveBeenCalled()
    expect(mockEndTutorial.mock.invocationCallOrder[0]).toBeLessThan(
      mockConfirmDanger.mock.invocationCallOrder[0]
    )
    const deleteDialogArgs = mockConfirmDanger.mock.calls[0][0]
    expect(String(deleteDialogArgs.content)).toBe("Delete this note?")
  })
})
