import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
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
      if (defaultValueOrOptions?.defaultValue) {
        if (
          typeof defaultValueOrOptions.count === "number" &&
          defaultValueOrOptions.defaultValue.includes("{{count}}")
        ) {
          return defaultValueOrOptions.defaultValue.replace(
            "{{count}}",
            String(defaultValueOrOptions.count)
          )
        }
        return defaultValueOrOptions.defaultValue
      }
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
  getAllNoteKeywordStats: vi.fn(async () => [
    { keyword: "research", noteCount: 4 },
    { keyword: "ml", noteCount: 2 },
    { keyword: "planning", noteCount: 1 }
  ]),
  searchNoteKeywords: vi.fn(async () => ["research", "ml", "planning"])
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
  default: ({ notes }: { notes?: Array<{ id: string | number }> }) => (
    <div data-testid="notes-list-panel-mock">{(notes || []).length}</div>
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

let seededNotebookSettings: Array<{ id: number; name: string; keywords: string[] }> = []
let seededServerNotebooks: Array<{ id: number; name: string; keywords: string[] }> = []

describe("NotesManagerPage stage 39 organization model", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    window.localStorage.clear()
    mockConfirmDanger.mockResolvedValue(true)
    mockSetSetting.mockResolvedValue(undefined)
    mockClearSetting.mockResolvedValue(undefined)
    seededNotebookSettings = []
    seededServerNotebooks = []
    mockGetSetting.mockImplementation(async (setting: { key?: string }) => {
      const key = String(setting?.key || "")
      if (key === "tldw:notesRecentOpened") return []
      if (key === "tldw:notesPinnedIds") return []
      if (key === "tldw:notesNotebooks") return seededNotebookSettings
      if (key === "tldw:notesPageSize") return 20
      if (key === "tldw:lastNoteId") return null
      if (key === "tldw:notesTitleSuggestStrategy") return null
      return null
    })
    mockBgRequest.mockImplementation(async (request: { path?: string; method?: string }) => {
      const path = String(request.path || "")
      const method = String(request.method || "GET").toUpperCase()
      if (path.startsWith("/api/v1/notes/collections?") && method === "GET") {
        return {
          collections: seededServerNotebooks
        }
      }
      if (path === "/api/v1/notes/collections" && method === "POST") {
        const body = (request as any).body || {}
        const nextId =
          seededServerNotebooks.reduce((max, item) => Math.max(max, item.id), 0) + 1
        const created = {
          id: nextId,
          name: String(body?.name || "").trim(),
          keywords: Array.isArray(body?.keywords) ? body.keywords : []
        }
        seededServerNotebooks = [...seededServerNotebooks, created]
        return created
      }
      if (path.startsWith("/api/v1/notes/collections/") && method === "PATCH") {
        const idSegment = path.replace("/api/v1/notes/collections/", "").split("?")[0]
        const collectionId = Number(idSegment)
        const body = (request as any).body || {}
        const existing = seededServerNotebooks.find((entry) => entry.id === collectionId)
        if (!existing) throw new Error("Collection not found")
        const updated = {
          ...existing,
          name: String(body?.name || existing.name),
          keywords: Array.isArray(body?.keywords) ? body.keywords : existing.keywords
        }
        seededServerNotebooks = seededServerNotebooks.map((entry) =>
          entry.id === collectionId ? updated : entry
        )
        return updated
      }
      if (path.startsWith("/api/v1/notes/collections/") && method === "DELETE") {
        const idSegment = path.replace("/api/v1/notes/collections/", "").split("?")[0]
        const collectionId = Number(idSegment)
        seededServerNotebooks = seededServerNotebooks.filter((entry) => entry.id !== collectionId)
        return {}
      }
      if (path.startsWith("/api/v1/notes/?")) {
        return {
          items: [
            {
              id: 1,
              title: "Alpha",
              content: "alpha body",
              updated_at: "2026-02-10T12:00:00.000Z",
              metadata: { keywords: [] }
            },
            {
              id: 2,
              title: "Beta",
              content: "beta body",
              updated_at: "2026-01-05T12:00:00.000Z",
              metadata: { keywords: [] }
            }
          ],
          pagination: { total_items: 2, total_pages: 1 }
        }
      }
      if (path.startsWith("/api/v1/notes/search/?")) {
        return {
          items: [
            {
              id: 1,
              title: "Notebook scoped result",
              content: "alpha body",
              updated_at: "2026-02-10T12:00:00.000Z",
              metadata: { keywords: [] }
            }
          ],
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
      if (path === "/api/v1/notes/2" && method === "GET") {
        return {
          id: 2,
          title: "Beta",
          content: "beta body",
          updated_at: "2026-01-05T12:00:00.000Z",
          metadata: { keywords: [] },
          version: 1
        }
      }
      return {}
    })
  })

  it("applies notebook keywords to search requests and filter summary", async () => {
    seededServerNotebooks = [{ id: 11, name: "Research", keywords: ["research", "ml"] }]
    renderPage()

    // Open the nested Organize section (collapsed by default)
    await waitFor(() => {
      expect(screen.getByTestId("notes-section-organize-toggle")).toBeInTheDocument()
    })
    fireEvent.click(screen.getByTestId("notes-section-organize-toggle"))

    await waitFor(() => {
      expect(screen.getByTestId("notes-notebook-select")).toBeInTheDocument()
    })

    // Select option from Ant Design Select
    const notebookContainer = screen.getByTestId("notes-notebook-select")
    const notebookContent = notebookContainer.querySelector('.ant-select-content') || notebookContainer
    fireEvent.mouseDown(notebookContent)
    await waitFor(() => {
      const options = document.querySelectorAll('.ant-select-item-option')
      const match = Array.from(options).find(el => el.textContent === 'Research (2)')
      if (!match) throw new Error('Option "Research (2)" not found')
      fireEvent.click(match)
    })

    await waitFor(() => {
      const searchPaths = mockBgRequest.mock.calls
        .map((args) => String((args[0] as { path?: string }).path || ""))
        .filter((path) => path.startsWith("/api/v1/notes/search/?"))
      expect(
        searchPaths.some(
          (path) => path.includes("tokens=research") && path.includes("tokens=ml")
        )
      ).toBe(true)
    })

    expect(screen.getByTestId("notes-active-filter-summary-details")).toHaveTextContent(
      "Saved filter: Research"
    )
  })

  it("saves current keyword filters as a notebook and persists it", async () => {
    mockPromptModal.mockResolvedValue("Research Notebook")
    renderPage()

    // Open the nested Organize section (collapsed by default)
    await waitFor(() => {
      expect(screen.getByTestId("notes-section-organize-toggle")).toBeInTheDocument()
    })
    fireEvent.click(screen.getByTestId("notes-section-organize-toggle"))

    fireEvent.click(screen.getByRole("button", { name: "Browse tags" }))
    const modalBody = await screen.findByTestId("notes-keyword-picker-modal")
    const pickerDialog =
      (modalBody.closest(".ant-modal") as HTMLElement | null) ??
      (modalBody.closest(".ant-modal-root") as HTMLElement | null) ??
      document.body
    fireEvent.click(within(pickerDialog).getByText(/^research\b/i))
    fireEvent.click(within(pickerDialog).getByRole("button", { name: "Apply filters" }))
    await waitFor(() => {
      expect(screen.getByTestId("notes-save-notebook")).not.toBeDisabled()
    })
    fireEvent.click(screen.getByTestId("notes-save-notebook"))

    await waitFor(() => {
      expect(mockMessageSuccess).toHaveBeenCalledWith('Saved filter "Research Notebook"')
    })

    expect(mockBgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/notes/collections",
        method: "POST"
      })
    )

    const notebookPersistCall = mockSetSetting.mock.calls.find((args) => {
      const payload = args[1]
      return (
        Array.isArray(payload) &&
        payload.some((entry) => entry && typeof entry === "object" && entry.name === "Research Notebook")
      )
    })
    expect(notebookPersistCall).toBeTruthy()
    expect(screen.getByTestId("notes-active-filter-summary-details")).toHaveTextContent(
      "Saved filter: Research Notebook"
    )

  }, 10000)

  it("renders timeline view groups for date-based browsing", async () => {
    renderPage()

    fireEvent.click(screen.getByTestId("notes-view-mode-timeline"))

    await waitFor(() => {
      expect(screen.getByTestId("notes-timeline-view")).toBeInTheDocument()
    })

    expect(screen.getByTestId("notes-timeline-group-2026-02")).toHaveTextContent("Alpha")
    expect(screen.getByTestId("notes-timeline-group-2026-01")).toHaveTextContent("Beta")

    fireEvent.click(screen.getByTestId("notes-timeline-item-2"))

    await waitFor(() => {
      expect(mockBgRequest).toHaveBeenCalledWith(
        expect.objectContaining({
          path: "/api/v1/notes/2"
        })
      )
    })
  })

  it("renders keyboard-focusable help buttons for organize and tags guidance", async () => {
    renderPage()

    expect(await screen.findByRole("button", { name: "Organize help" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Tags help" })).toBeInTheDocument()
  })
})
