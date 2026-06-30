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
  mockNavigate,
  mockConfirmDanger,
  mockGetSetting,
  mockClearSetting
} = vi.hoisted(() => {
  return {
    mockBgRequest: vi.fn(),
    mockMessageSuccess: vi.fn(),
    mockMessageError: vi.fn(),
    mockMessageWarning: vi.fn(),
    mockNavigate: vi.fn(),
    mockConfirmDanger: vi.fn(),
    mockGetSetting: vi.fn(),
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
    info: vi.fn()
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
  default: () => <div data-testid="notes-list-panel" />
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

describe("NotesManagerPage stage 19 accessibility skip links and labels", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockConfirmDanger.mockResolvedValue(true)
    mockGetSetting.mockResolvedValue(null)
    mockClearSetting.mockResolvedValue(undefined)

    mockBgRequest.mockImplementation(async (request: { path?: string }) => {
      const path = String(request.path || "")
      if (path.startsWith("/api/v1/notes/?")) {
        return {
          items: [],
          pagination: { total_items: 0, total_pages: 1 }
        }
      }
      return {}
    })
  })

  it("renders skip links and focuses list/editor regions when activated", async () => {
    renderPage()

    const skipToList = screen.getByRole("link", { name: "Skip to notes list" })
    const skipToEditor = screen.getByRole("link", { name: "Skip to editor" })
    const listRegion = screen.getByTestId("notes-list-region")
    const editorRegion = screen.getByTestId("notes-editor-region")

    expect(skipToList).toHaveAttribute("href", "#notes-list-region")
    expect(skipToEditor).toHaveAttribute("href", "#notes-editor-region")
    expect(listRegion).toHaveAttribute("tabindex", "-1")
    expect(editorRegion).toHaveAttribute("tabindex", "-1")

    fireEvent.click(skipToList)
    await waitFor(() => {
      expect(listRegion).toHaveFocus()
    })

    fireEvent.click(skipToEditor)
    await waitFor(() => {
      expect(editorRegion).toHaveFocus()
    })
  })

  it("exposes an explicit note content label in edit and split editor modes", () => {
    renderPage()

    const editTextarea = screen.getByLabelText("Note content")
    expect(editTextarea.tagName).toBe("TEXTAREA")

    fireEvent.click(screen.getByRole("button", { name: "Split" }))

    const splitTextarea = screen.getByLabelText("Note content")
    expect(splitTextarea.tagName).toBe("TEXTAREA")
  })

  it("labels the primary note fields and sidebar search controls", () => {
    renderPage()

    expect(screen.getByLabelText("Note title")).toHaveAttribute("placeholder", "Title")
    expect(screen.getByRole("combobox", { name: "Note tags" })).toBeInTheDocument()
    expect(screen.getByLabelText("Search notes")).toHaveAttribute(
      "placeholder",
      "Search notes... (use quotes for exact match)"
    )
    expect(screen.getByRole("combobox", { name: "Filter by tag" })).toBeInTheDocument()
  })

  it("moves focus to save failure recovery and links fields to the save status", async () => {
    mockBgRequest.mockImplementation(async (request: { path?: string; method?: string }) => {
      const path = String(request.path || "")
      const method = String(request.method || "GET").toUpperCase()
      if (path.startsWith("/api/v1/notes/?")) {
        return {
          items: [],
          pagination: { total_items: 0, total_pages: 1 }
        }
      }
      if (path === "/api/v1/notes/" && method === "POST") {
        throw new Error("Server unavailable")
      }
      return {}
    })

    renderPage()

    const titleInput = screen.getByLabelText("Note title")
    const contentInput = screen.getByLabelText("Note content")
    fireEvent.change(titleInput, { target: { value: "Failed save note" } })
    fireEvent.change(contentInput, { target: { value: "Keep this draft visible." } })
    fireEvent.click(screen.getByTestId("notes-save-button"))

    const status = await screen.findByRole("alert", { name: "Note save status" })
    expect(status).toHaveTextContent("Could not save")
    await waitFor(() => {
      expect(status).toHaveFocus()
    })
    expect(titleInput.getAttribute("aria-describedby")).toContain(
      "notes-save-status-message"
    )
    expect(contentInput.getAttribute("aria-describedby")).toContain(
      "notes-save-status-message"
    )
  })
})
