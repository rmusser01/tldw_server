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
  mockClearSetting
} = vi.hoisted(() => {
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
    mockClearSetting: vi.fn()
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

vi.mock("@/components/Common/MarkdownPreview", () => ({
  MarkdownPreview: ({ content }: { content: string }) => (
    <div data-testid="markdown-preview-content">{content}</div>
  )
}))

vi.mock("@/components/Notes/NotesListPanel", () => ({
  default: ({
    notes,
    onSelectNote
  }: {
    notes?: Array<{ id: string | number; title?: string }>
    onSelectNote: (id: string | number) => void
  }) => (
    <div data-testid="notes-list-panel">
      {(notes || []).map((note) => (
        <button
          key={String(note.id)}
          type="button"
          data-testid={`mock-note-${String(note.id)}`}
          onClick={() => onSelectNote(note.id)}
        >
          {note.title || String(note.id)}
        </button>
      ))}
    </div>
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

describe("NotesManagerPage stage 12 recent notes", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockConfirmDanger.mockResolvedValue(true)
    mockGetSetting.mockResolvedValue(null)
    mockSetSetting.mockResolvedValue(undefined)
    mockClearSetting.mockResolvedValue(undefined)
    mockBgRequest.mockImplementation(async (request: { path?: string; method?: string }) => {
      const path = String(request.path || "")
      const method = String(request.method || "GET").toUpperCase()
      if (path.startsWith("/api/v1/notes/?")) {
        return {
          items: [
            { id: "n1", title: "Alpha note", content: "alpha content", version: 1 },
            { id: "n2", title: "Beta note", content: "beta content", version: 1 }
          ],
          pagination: { total_items: 2, total_pages: 1 }
        }
      }
      if (path === "/api/v1/admin/notes/title-settings" && method === "GET") {
        return {
          llm_enabled: false,
          default_strategy: "heuristic",
          effective_strategy: "heuristic",
          strategies: ["heuristic", "llm", "llm_fallback"]
        }
      }
      if (path === "/api/v1/notes/n1" && method === "GET") {
        return {
          id: "n1",
          title: "Alpha note",
          content: "alpha content",
          version: 1,
          last_modified: new Date().toISOString()
        }
      }
      if (path === "/api/v1/notes/n2" && method === "GET") {
        return {
          id: "n2",
          title: "Beta note",
          content: "beta content",
          version: 1,
          last_modified: new Date().toISOString()
        }
      }
      return {}
    })
  })

  it("tracks recent note ordering and persists it after selections", async () => {
    renderPage()
    fireEvent.click(await screen.findByTestId("mock-note-n1"))
    await waitFor(() => {
      expect(screen.getByTestId("notes-recent-section")).toBeInTheDocument()
    })
    fireEvent.click(await screen.findByTestId("mock-note-n2"))

    await waitFor(() => {
      const items = screen.getAllByTestId(/notes-recent-item-/)
      expect(items[0]).toHaveTextContent("Beta note")
      expect(items[1]).toHaveTextContent("Alpha note")
    })

    const persistedCall = mockSetSetting.mock.calls
      .filter(([setting]) => setting?.key === "tldw:notesRecentOpened")
      .at(-1)
    expect(persistedCall).toBeTruthy()
    expect(persistedCall?.[1]?.[0]?.id).toBe("n2")
    expect(persistedCall?.[1]?.[1]?.id).toBe("n1")
  })

  it("handles recent-note persistence promise failures", async () => {
    const catchSpy = vi.fn()
    mockSetSetting.mockReturnValue({ catch: catchSpy })

    renderPage()
    fireEvent.click(await screen.findByTestId("mock-note-n1"))

    await waitFor(() => {
      expect(screen.getByTestId("notes-recent-item-n1")).toHaveTextContent("Alpha note")
    })
    await waitFor(() => {
      expect(mockSetSetting).toHaveBeenCalledWith(
        expect.objectContaining({ key: "tldw:notesRecentOpened" }),
        expect.arrayContaining([expect.objectContaining({ id: "n1" })])
      )
    })
    expect(catchSpy).toHaveBeenCalled()
  })

  it("loads persisted recent notes and keeps in-note search guidance visible", async () => {
    mockGetSetting.mockImplementation(async (setting: { key?: string }) => {
      if (setting?.key === "tldw:notesRecentOpened") {
        return [{ id: "seed-note", title: "Seeded recent note" }]
      }
      return null
    })

    renderPage()
    await waitFor(() => {
      expect(screen.getByTestId("notes-recent-section")).toBeInTheDocument()
    })
    expect(screen.getByText("Seeded recent note")).toBeInTheDocument()
    expect(screen.getByTestId("notes-in-note-search-guidance")).toHaveTextContent(
      "For in-note search, use browser Ctrl/Cmd+F."
    )
  })
})
