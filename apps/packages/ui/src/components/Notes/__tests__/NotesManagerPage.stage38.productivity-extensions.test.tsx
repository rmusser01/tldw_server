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
  default: ({ onApplyTemplate, onDuplicate }: any) => (
    <div>
      <button
        data-testid="notes-apply-template-research-brief"
        onClick={() => onApplyTemplate?.("research_brief")}
      >
        Apply research template
      </button>
      <button data-testid="notes-duplicate-action" onClick={() => onDuplicate?.()}>
        Duplicate note
      </button>
    </div>
  )
}))

vi.mock("@/components/Notes/NotesListPanel", () => ({
  default: ({ notes, onTogglePinned }: any) => (
    <div>
      <div data-testid="notes-order">{(notes || []).map((note: any) => String(note.title || note.id)).join("|")}</div>
      <button data-testid="notes-pin-second" onClick={() => onTogglePinned?.(2)}>
        Pin second note
      </button>
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

describe("NotesManagerPage stage 38 productivity extensions", () => {
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
            {
              id: 1,
              title: "Alpha note",
              content: "Alpha body",
              metadata: { keywords: [] }
            },
            {
              id: 2,
              title: "Beta note",
              content: "Beta body",
              metadata: { keywords: [] }
            }
          ],
          pagination: { total_items: 2, total_pages: 1 }
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
  })

  it("applies a research template into a new draft", async () => {
    renderPage()

    fireEvent.click(screen.getByTestId("notes-apply-template-research-brief"))

    await waitFor(() => {
      expect(screen.getByPlaceholderText("Title")).toHaveValue("Research Brief")
    })
    expect(
      String(
        (screen.getByPlaceholderText("Write your note here... (Markdown supported)") as HTMLTextAreaElement)
          .value
      )
    ).toContain("## Research Question")
    expect(mockMessageSuccess).toHaveBeenCalledWith("Applied template: Research Brief")
  })

  it("duplicates the current draft as a copy", async () => {
    renderPage()

    fireEvent.change(screen.getByPlaceholderText("Title"), {
      target: { value: "Original note" }
    })
    fireEvent.change(screen.getByPlaceholderText("Write your note here... (Markdown supported)"), {
      target: { value: "Original body text" }
    })

    fireEvent.click(screen.getByTestId("notes-duplicate-action"))

    await waitFor(() => {
      expect(screen.getByPlaceholderText("Title")).toHaveValue("Original note (Copy)")
    })
    expect(screen.getByPlaceholderText("Write your note here... (Markdown supported)")).toHaveValue(
      "Original body text"
    )
    expect(mockMessageSuccess).toHaveBeenCalledWith("Created duplicate draft. Save to keep it.")
  })

  it("duplicates the current draft from a guarded power-user shortcut", async () => {
    renderPage()

    const titleInput = screen.getByPlaceholderText("Title")
    const contentInput = screen.getByPlaceholderText("Write your note here... (Markdown supported)")

    fireEvent.change(titleInput, {
      target: { value: "Shortcut source" }
    })
    fireEvent.change(contentInput, {
      target: { value: "Shortcut body" }
    })

    contentInput.focus()
    fireEvent.keyDown(contentInput, { key: "d", altKey: true, shiftKey: true })
    expect(titleInput).toHaveValue("Shortcut source")

    fireEvent.keyDown(window, { key: "d", altKey: true, shiftKey: true })

    await waitFor(() => {
      expect(titleInput).toHaveValue("Shortcut source (Copy)")
    })
    expect(contentInput).toHaveValue("Shortcut body")
    expect(mockMessageSuccess).toHaveBeenCalledWith("Created duplicate draft. Save to keep it.")
  })

  it("pins a note and reorders the visible list with persistence", async () => {
    renderPage()

    await waitFor(() => {
      expect(screen.getByTestId("notes-order")).toHaveTextContent("Alpha note|Beta note")
    })

    fireEvent.click(screen.getByTestId("notes-pin-second"))

    await waitFor(() => {
      expect(screen.getByTestId("notes-order")).toHaveTextContent("Beta note|Alpha note")
    })
    expect(mockSetSetting).toHaveBeenCalledWith(
      expect.anything(),
      expect.arrayContaining(["2"])
    )
  })
})
