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
    mockClearSetting: vi.fn(),
    mockPromptModal: vi.fn()
  }
})

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

vi.mock("@/components/Common/MarkdownPreview", () => ({
  MarkdownPreview: ({ content }: { content: string }) => (
    <div data-testid="markdown-preview-content">{content}</div>
  )
}))

vi.mock("@/components/Notes/NotesListPanel", () => ({
  default: ({
    notes,
    bulkSelectedIds,
    onToggleBulkSelection
  }: {
    notes?: Array<{ id: string | number; title?: string }>
    bulkSelectedIds?: string[]
    onToggleBulkSelection?: (id: string | number, checked: boolean, shiftKey: boolean) => void
  }) => (
    <div data-testid="notes-list-panel">
      <div data-testid="notes-list-bulk-selected">{(bulkSelectedIds || []).join("|")}</div>
      {(notes || []).map((note) => (
        <div key={String(note.id)}>
          <button
            type="button"
            data-testid={`mock-select-${String(note.id)}`}
            onClick={() => onToggleBulkSelection?.(note.id, true, false)}
          >
            Select {String(note.id)}
          </button>
          <button
            type="button"
            data-testid={`mock-shift-select-${String(note.id)}`}
            onClick={() => onToggleBulkSelection?.(note.id, true, true)}
          >
            Shift Select {String(note.id)}
          </button>
        </div>
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

describe("NotesManagerPage stage 16 bulk actions", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockConfirmDanger.mockResolvedValue(true)
    mockGetSetting.mockResolvedValue(null)
    mockSetSetting.mockResolvedValue(undefined)
    mockClearSetting.mockResolvedValue(undefined)
    if (!URL.createObjectURL) {
      ;(URL as any).createObjectURL = vi.fn(() => "blob:notes")
    }
    if (!URL.revokeObjectURL) {
      ;(URL as any).revokeObjectURL = vi.fn(() => undefined)
    }

    mockBgRequest.mockImplementation(async (request: { path?: string; method?: string }) => {
      const path = String(request.path || "")
      const method = String(request.method || "GET").toUpperCase()
      if (path.startsWith("/api/v1/notes/?")) {
        return {
          items: [
            { id: "n1", title: "Alpha", content: "a", version: 1 },
            { id: "n2", title: "Beta", content: "b", version: 1 },
            { id: "n3", title: "Gamma", content: "c", version: 1 }
          ],
          pagination: { total_items: 3, total_pages: 1 }
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
      if (path.includes("/api/v1/notes/") && method === "DELETE") {
        return {}
      }
      if (path.includes("/api/v1/notes/") && method === "PATCH") {
        return {}
      }
      if (path.startsWith("/api/v1/notes/") && method === "GET") {
        return { id: "n1", title: "Alpha", content: "a", version: 1 }
      }
      return {}
    })
  })

  it("supports shift-range selection with live selected-count announcement", async () => {
    renderPage()
    fireEvent.click(await screen.findByTestId("mock-select-n1"))
    fireEvent.click(await screen.findByTestId("mock-shift-select-n3"))

    await waitFor(() => {
      expect(screen.getByTestId("notes-list-bulk-selected")).toHaveTextContent("n1|n2|n3")
    })
    const bar = screen.getByTestId("notes-bulk-actions-bar")
    expect(bar).toHaveAttribute("role", "status")
    expect(bar).toHaveAttribute("aria-live", "polite")

    fireEvent.click(screen.getByTestId("notes-bulk-clear-selection"))
    await waitFor(() => {
      expect(screen.queryByTestId("notes-bulk-actions-bar")).not.toBeInTheDocument()
    })
  })

  it("dispatches bulk export and bulk delete operations", async () => {
    renderPage()
    fireEvent.click(await screen.findByTestId("mock-select-n1"))
    fireEvent.click(await screen.findByTestId("mock-select-n2"))

    await waitFor(() => {
      expect(screen.getByTestId("notes-list-bulk-selected")).toHaveTextContent("n1|n2")
    })

    fireEvent.click(screen.getByTestId("notes-bulk-export"))
    expect(mockMessageSuccess).toHaveBeenCalledWith(
      expect.stringContaining("Exported")
    )

    fireEvent.click(screen.getByTestId("notes-bulk-delete"))
    await waitFor(() => {
      const deleteCalls = mockBgRequest.mock.calls.filter(
        ([request]) =>
          String(request?.method || "").toUpperCase() === "DELETE" &&
          String(request?.path || "").includes("/api/v1/notes/")
      )
      expect(deleteCalls.length).toBe(2)
    })
    expect(mockConfirmDanger).toHaveBeenCalledWith(
      expect.objectContaining({ title: "Delete selected notes?" })
    )
  })

  it("confirms and dispatches bulk keyword assignment patches", async () => {
    mockPromptModal.mockResolvedValue("research, summary")
    renderPage()
    fireEvent.click(await screen.findByTestId("mock-select-n1"))

    await waitFor(() => {
      expect(screen.getByTestId("notes-list-bulk-selected")).toHaveTextContent("n1")
    })

    fireEvent.click(screen.getByTestId("notes-bulk-assign-keywords"))
    await waitFor(() => {
      const patchCalls = mockBgRequest.mock.calls.filter(
        ([request]) =>
          String(request?.method || "").toUpperCase() === "PATCH" &&
          String(request?.path || "").includes("/api/v1/notes/")
      )
      expect(patchCalls.length).toBe(1)
    })

    const patchCall = mockBgRequest.mock.calls.find(
      ([request]) => String(request?.method || "").toUpperCase() === "PATCH"
    )
    expect(patchCall?.[0]?.body?.keywords).toEqual(["research", "summary"])
    expect(mockConfirmDanger).toHaveBeenCalledWith(
      expect.objectContaining({ title: "Apply tags to selected notes?" })
    )
  })
})
