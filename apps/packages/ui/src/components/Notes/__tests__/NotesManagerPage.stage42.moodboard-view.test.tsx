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
  mockPromptModal
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
  mockPromptModal: vi.fn()
}))

vi.mock("@/components/Notes/notes-manager-utils", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/components/Notes/notes-manager-utils")>()
  return { ...actual, promptModal: mockPromptModal }
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

vi.mock("@/components/Notes/NotesEditorHeader", () => ({
  default: () => <div data-testid="notes-editor-header-mock" />
}))

vi.mock("@/components/Notes/NotesListPanel", () => ({
  default: () => <div data-testid="notes-list-panel-mock" />
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

let seededMoodboards: Array<{ id: number; name: string; description?: string | null; version: number }> = []
let allowMoodboardFetch = false

describe("NotesManagerPage stage 42 moodboard view", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    window.localStorage.clear()
    seededMoodboards = [{ id: 7, name: "Inspiration", description: "Visual ideas", version: 1 }]
    allowMoodboardFetch = false
    mockConfirmDanger.mockResolvedValue(true)
    mockGetSetting.mockResolvedValue(null)
    mockSetSetting.mockResolvedValue(undefined)
    mockClearSetting.mockResolvedValue(undefined)

    const moodboardNotes = [
      {
        id: 1,
        title: "Manual card",
        content_preview: "manual preview",
        membership_source: "manual",
        keywords: ["alpha"]
      },
      {
        id: 2,
        title: "Smart card",
        content_preview: "smart preview",
        membership_source: "both",
        keywords: ["palette"]
      },
      {
        id: 3,
        title: "Third card",
        content_preview: "third preview",
        membership_source: "smart",
        keywords: ["gamma"]
      },
      ...Array.from({ length: 22 }, (_, index) => {
        const noteId = index + 4
        return {
          id: noteId,
          title: `Card ${noteId}`,
          content_preview: `preview ${noteId}`,
          membership_source: noteId % 2 === 0 ? "manual" : "smart",
          keywords: [`kw-${noteId}`]
        }
      })
    ]

    mockBgRequest.mockImplementation(async (request: { path?: string; method?: string; body?: any }) => {
      const path = String(request.path || "")
      const method = String(request.method || "GET").toUpperCase()

      if (path.startsWith("/api/v1/notes/moodboards?") && method === "GET") {
        if (!allowMoodboardFetch) {
          throw new Error("unexpected eager moodboard fetch")
        }
        return {
          moodboards: seededMoodboards
        }
      }

      if (path.startsWith("/api/v1/notes/moodboards/7/notes?") && method === "GET") {
        const queryString = path.split("?")[1] || ""
        const params = new URLSearchParams(queryString)
        const limit = Number(params.get("limit") || "20")
        const offset = Number(params.get("offset") || "0")
        const boundedLimit = Number.isFinite(limit) && limit > 0 ? Math.floor(limit) : 20
        const boundedOffset = Number.isFinite(offset) && offset >= 0 ? Math.floor(offset) : 0
        const notes = moodboardNotes.slice(boundedOffset, boundedOffset + boundedLimit)
        return {
          notes,
          count: notes.length,
          total: moodboardNotes.length
        }
      }

      if (path === "/api/v1/notes/moodboards" && method === "POST") {
        const body = request.body || {}
        const nextId = seededMoodboards.reduce((max, item) => Math.max(max, item.id), 0) + 1
        const created = {
          id: nextId,
          name: String(body.name || "").trim() || `Collection ${nextId}`,
          description: null,
          version: 1
        }
        seededMoodboards = [...seededMoodboards, created]
        return created
      }

      if (path.startsWith("/api/v1/notes/moodboards/") && method === "PATCH") {
        const id = Number(path.replace("/api/v1/notes/moodboards/", "").split("?")[0])
        const body = request.body || {}
        seededMoodboards = seededMoodboards.map((item) =>
          item.id === id
            ? {
                ...item,
                name: String(body.name || item.name),
                version: item.version + 1
              }
            : item
        )
        return seededMoodboards.find((item) => item.id === id) || {}
      }

      if (path.startsWith("/api/v1/notes/moodboards/") && method === "DELETE") {
        const id = Number(path.replace("/api/v1/notes/moodboards/", "").split("?")[0])
        seededMoodboards = seededMoodboards.filter((item) => item.id !== id)
        return {}
      }

      if (path.startsWith("/api/v1/notes/?")) {
        return {
          items: [],
          pagination: { total_items: 0, total_pages: 1 }
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

      if (path === "/api/v1/notes/2" && method === "GET") {
        return {
          id: 2,
          title: "Smart card",
          content: "smart preview",
          metadata: { keywords: [] },
          version: 1
        }
      }

      if (path.includes("/neighbors?")) {
        return { nodes: [], edges: [] }
      }

      return {}
    })
  })

  it("renders moodboard cards and opens selected note from masonry wall", async () => {
    renderPage()

    allowMoodboardFetch = true
    fireEvent.click(screen.getByTestId("notes-view-mode-moodboard"))

    await waitFor(() => {
      expect(screen.getByTestId("notes-moodboard-view")).toBeInTheDocument()
    })

    await waitFor(() => {
      expect(screen.getByTestId("notes-moodboard-card-1")).toBeInTheDocument()
      expect(screen.getByTestId("notes-moodboard-card-2")).toBeInTheDocument()
      expect(screen.getByTestId("notes-moodboard-card-3")).toBeInTheDocument()
    })

    fireEvent.click(screen.getByTestId("notes-moodboard-card-2"))

    await waitFor(() => {
      expect(mockBgRequest).toHaveBeenCalledWith(
        expect.objectContaining({
          path: "/api/v1/notes/2"
        })
      )
    })
  })

  it("supports create, rename, and delete actions for moodboards", async () => {
    mockPromptModal
      .mockResolvedValueOnce("Fresh Board")
      .mockResolvedValueOnce("Renamed Board")

    renderPage()

    allowMoodboardFetch = true
    fireEvent.click(screen.getByTestId("notes-view-mode-moodboard"))

    // Open the nested Organize section (collapsed by default)
    await waitFor(() => {
      expect(screen.getByTestId("notes-section-organize-toggle")).toBeInTheDocument()
    })
    fireEvent.click(screen.getByTestId("notes-section-organize-toggle"))

    await waitFor(() => {
      expect(screen.getByTestId("notes-moodboard-controls")).toBeInTheDocument()
    })

    fireEvent.click(screen.getByTestId("notes-moodboard-create"))

    await waitFor(() => {
      expect(mockBgRequest).toHaveBeenCalledWith(
        expect.objectContaining({
          path: "/api/v1/notes/moodboards",
          method: "POST"
        })
      )
    })

    fireEvent.click(screen.getByTestId("notes-moodboard-rename"))

    await waitFor(() => {
      expect(mockBgRequest).toHaveBeenCalledWith(
        expect.objectContaining({
          path: expect.stringMatching(/^\/api\/v1\/notes\/moodboards\/\d+$/),
          method: "PATCH"
        })
      )
    })

    fireEvent.click(screen.getByTestId("notes-moodboard-delete"))

    await waitFor(() => {
      expect(mockBgRequest).toHaveBeenCalledWith(
        expect.objectContaining({
          path: expect.stringMatching(/^\/api\/v1\/notes\/moodboards\/\d+$/),
          method: "DELETE"
        })
      )
    })

  })

  it("supports moodboard pagination controls and page navigation", async () => {
    renderPage()

    allowMoodboardFetch = true
    fireEvent.click(screen.getByTestId("notes-view-mode-moodboard"))

    await waitFor(() => {
      expect(screen.getByTestId("notes-moodboard-pagination")).toBeInTheDocument()
    })

    const pageSizeContainer = screen.getByTestId("notes-moodboard-page-size")
    const pageSizeContent = pageSizeContainer.querySelector('.ant-select-content') || pageSizeContainer
    fireEvent.mouseDown(pageSizeContent)
    await waitFor(() => {
      // Options are [10, 20, 50, 100]; labels are identical due to i18n mock
      // Click first option (value=10) which is first in DOM order
      const options = document.querySelectorAll('.ant-select-item-option')
      if (options.length === 0) throw new Error('No dropdown options found')
      fireEvent.click(options[0])
    })

    const hasMoodboardPageRequest = (limit: number, offset: number) =>
      mockBgRequest.mock.calls.some(([arg]) => {
        const path = String(arg?.path || "")
        if (!path.startsWith("/api/v1/notes/moodboards/7/notes?")) return false
        const queryString = path.split("?")[1] || ""
        const params = new URLSearchParams(queryString)
        return params.get("limit") === String(limit) && params.get("offset") === String(offset)
      })

    await waitFor(() => {
      expect(hasMoodboardPageRequest(10, 0)).toBe(true)
    })

    await waitFor(() => {
      expect(screen.getByTestId("notes-moodboard-page-next")).toBeInTheDocument()
    })

    fireEvent.click(screen.getByTestId("notes-moodboard-page-next"))

    await waitFor(() => {
      expect(hasMoodboardPageRequest(10, 10)).toBe(true)
    })
  })
})
