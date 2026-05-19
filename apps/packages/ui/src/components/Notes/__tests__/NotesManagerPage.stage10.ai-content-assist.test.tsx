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
    onSelectNote?: (id: string | number) => void
  }) => (
    <div data-testid="notes-list-panel">
      {(notes || []).map((note) => (
        <button
          key={String(note.id)}
          type="button"
          data-testid={`notes-open-button-${String(note.id)}`}
          onClick={() => onSelectNote?.(note.id)}
        >
          {note.title || `Note ${String(note.id)}`}
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

describe("NotesManagerPage stage 10 AI content assist actions", () => {
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
            { id: "note-a", title: "Alpha note", content: "Alpha source body", metadata: { keywords: [] }, version: 1 },
            { id: "note-b", title: "Beta note", content: "Beta source body", metadata: { keywords: [] }, version: 1 }
          ],
          pagination: { total_items: 2, total_pages: 1 }
        }
      }
      if (path === "/api/v1/notes/note-a" && method === "GET") {
        return {
          id: "note-a",
          title: "Alpha note",
          content: "Alpha source body",
          metadata: { keywords: [] },
          version: 1
        }
      }
      if (path === "/api/v1/notes/note-b" && method === "GET") {
        return {
          id: "note-b",
          title: "Beta note",
          content: "Beta source body",
          metadata: { keywords: [] },
          version: 1
        }
      }
      if (path.startsWith("/api/v1/notes/note-a/neighbors")) {
        return {
          nodes: [{ id: "note-a", type: "note", label: "Alpha note" }],
          edges: []
        }
      }
      if (path.startsWith("/api/v1/notes/note-b/neighbors")) {
        return {
          nodes: [{ id: "note-b", type: "note", label: "Beta note" }],
          edges: []
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
      return {}
    })
  })

  it("applies summarize assist only after confirmation and tracks provenance", async () => {
    renderPage()
    const textarea = screen.getByPlaceholderText("Write your note here... (Markdown supported)")

    fireEvent.change(textarea, {
      target: {
        value:
          "Research notes explain baseline outcomes. They compare two model variants. Final section captures open questions."
      }
    })
    fireEvent.click(screen.getByTestId("notes-assist-summarize"))

    await waitFor(() => {
      expect(mockConfirmDanger).toHaveBeenCalled()
    })

    await waitFor(() => {
      expect(
        (screen.getByPlaceholderText("Write your note here... (Markdown supported)") as HTMLTextAreaElement).value
      ).toContain("Summary:")
    })
    expect(screen.getByTestId("notes-editor-provenance")).toHaveTextContent(
      "Origin: AI-generated (Summarize"
    )

    fireEvent.change(screen.getByPlaceholderText("Write your note here... (Markdown supported)"), {
      target: { value: "manual update after summary" }
    })
    await waitFor(() => {
      expect(screen.getByTestId("notes-editor-provenance")).toHaveTextContent("Origin: Typed manually")
    })
  })

  it("does not mutate content when summarize confirmation is rejected", async () => {
    mockConfirmDanger.mockResolvedValueOnce(false)
    renderPage()
    const textarea = screen.getByPlaceholderText(
      "Write your note here... (Markdown supported)"
    ) as HTMLTextAreaElement
    fireEvent.change(textarea, {
      target: { value: "Keep this draft exactly as typed." }
    })

    fireEvent.click(screen.getByTestId("notes-assist-summarize"))

    await waitFor(() => {
      expect(mockConfirmDanger).toHaveBeenCalled()
    })
    expect(textarea.value).toBe("Keep this draft exactly as typed.")
    expect(screen.getByTestId("notes-editor-provenance")).toHaveTextContent("Origin: Typed manually")
  })

  it("suggests keywords with explicit selection and marks generated provenance after apply", async () => {
    renderPage()
    fireEvent.change(screen.getByPlaceholderText("Write your note here... (Markdown supported)"), {
      target: {
        value:
          "Quantum models track entanglement and decoherence. Quantum experiments repeat entanglement checks."
      }
    })

    fireEvent.click(screen.getByTestId("notes-assist-suggest-keywords"))

    await waitFor(() => {
      expect(screen.getByTestId("notes-assist-keyword-suggestions-modal")).toBeInTheDocument()
    })

    expect(mockConfirmDanger).not.toHaveBeenCalled()
    fireEvent.click(screen.getByTestId("notes-assist-keyword-option-checks"))
    fireEvent.click(screen.getByRole("button", { name: "Apply selected" }))

    await waitFor(() => {
      expect(mockMessageSuccess).toHaveBeenCalledWith("Applied suggested tags.")
    })
    expect(screen.getByTestId("notes-editor-provenance")).toHaveTextContent(
      "Origin: AI-generated (Suggest tags"
    )
    const editorKeywordsControl = screen.getByTestId("notes-keywords-editor")
    expect(within(editorKeywordsControl).getByText("quantum")).toBeInTheDocument()
    expect(within(editorKeywordsControl).queryByText("checks")).not.toBeInTheDocument()
  })

  it("does not attach suggested keywords when the review modal is canceled", async () => {
    renderPage()
    fireEvent.change(screen.getByPlaceholderText("Write your note here... (Markdown supported)"), {
      target: {
        value:
          "Quantum models track entanglement and decoherence. Quantum experiments repeat entanglement checks."
      }
    })

    fireEvent.click(screen.getByTestId("notes-assist-suggest-keywords"))
    const modal = await screen.findByTestId("notes-assist-keyword-suggestions-modal")
    expect(modal).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }))

    await waitFor(() => {
      expect(screen.getByTestId("notes-editor-provenance")).toHaveTextContent("Origin: Typed manually")
    })
    const editorKeywordsControl = screen.getByTestId("notes-keywords-editor")
    expect(within(editorKeywordsControl).queryByText("quantum")).not.toBeInTheDocument()
  })

  it("clears AI undo state when switching to a different note", async () => {
    renderPage()

    fireEvent.click(await screen.findByTestId("notes-open-button-note-a"))
    await waitFor(() => {
      expect(screen.getByDisplayValue("Alpha note")).toBeInTheDocument()
    })

    fireEvent.click(screen.getByTestId("notes-assist-summarize"))
    expect(await screen.findByTestId("notes-undo-assist")).toBeInTheDocument()

    fireEvent.click(screen.getByTestId("notes-open-button-note-b"))

    await waitFor(() => {
      expect(screen.getByDisplayValue("Beta note")).toBeInTheDocument()
    })
    expect(screen.getByDisplayValue("Beta source body")).toBeInTheDocument()
    expect(screen.queryByTestId("notes-undo-assist")).not.toBeInTheDocument()
  })
})
