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

vi.mock("@/components/Common/MarkdownPreview", () => ({
  MarkdownPreview: ({ content }: { content: string }) => (
    <div data-testid="markdown-preview-content">{content}</div>
  )
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

describe("NotesManagerPage stage 2 editor modes", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockConfirmDanger.mockResolvedValue(true)
    mockGetSetting.mockResolvedValue(null)
    mockClearSetting.mockResolvedValue(undefined)

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
        return { id: 14, version: 1 }
      }

      if (path === "/api/v1/notes/14" && method === "GET") {
        return {
          id: 14,
          title: "Mode note",
          content: "## Header",
          metadata: { keywords: [] },
          version: 1
        }
      }

      return {}
    })
  })

  it("switches between Edit, Split, and Preview modes", async () => {
    renderPage()

    const textareaPlaceholder = "Write your note here... (Markdown supported)"
    const textarea = screen.getByPlaceholderText(textareaPlaceholder)

    fireEvent.change(textarea, { target: { value: "## Heading\n\ncontent" } })

    expect(screen.getByText("Markdown + LaTeX supported")).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Preview" }))

    expect(screen.queryByPlaceholderText(textareaPlaceholder)).not.toBeInTheDocument()
    expect(screen.getByText("Preview (Markdown + LaTeX)")).toBeInTheDocument()
    expect(await screen.findByTestId("markdown-preview-content")).toHaveTextContent("## Heading")

    fireEvent.click(screen.getByRole("button", { name: "Split" }))

    expect(screen.getByPlaceholderText(textareaPlaceholder)).toBeInTheDocument()
    expect(screen.getByText("Preview (Markdown + LaTeX)")).toBeInTheDocument()
    expect(screen.getByText("Markdown + LaTeX supported")).toBeInTheDocument()
  }, 10000)

  it("auto-resizes editor textarea with bounded height", async () => {
    renderPage()

    const textareaPlaceholder = "Write your note here... (Markdown supported)"
    const textarea = screen.getByPlaceholderText(textareaPlaceholder) as HTMLTextAreaElement

    Object.defineProperty(textarea, "scrollHeight", {
      configurable: true,
      get: () => 360
    })

    fireEvent.change(textarea, { target: { value: "Resize me" } })

    await waitFor(() => {
      expect(textarea.style.height).toBe("360px")
    })

    fireEvent.click(screen.getByRole("button", { name: "Split" }))

    const splitTextarea = screen.getByPlaceholderText(textareaPlaceholder) as HTMLTextAreaElement
    Object.defineProperty(splitTextarea, "scrollHeight", {
      configurable: true,
      get: () => 2000
    })

    fireEvent.change(splitTextarea, { target: { value: "x".repeat(120) } })

    await waitFor(() => {
      const height = Number.parseInt(splitTextarea.style.height.replace("px", ""), 10)
      expect(height).toBeGreaterThanOrEqual(220)
      expect(height).toBeLessThan(500)
    })
  })

  it("ignores editor-mode shortcuts while typing in the search input", async () => {
    renderPage()

    const textareaPlaceholder = "Write your note here... (Markdown supported)"
    const textarea = screen.getByPlaceholderText(textareaPlaceholder)
    fireEvent.change(textarea, { target: { value: "## Heading\n\ncontent" } })

    const searchInput = screen.getByPlaceholderText(
      "Search notes... (use quotes for exact match)"
    )
    fireEvent.focus(searchInput)
    fireEvent.keyDown(searchInput, { key: "p", ctrlKey: true, shiftKey: true })

    expect(screen.getByPlaceholderText(textareaPlaceholder)).toBeInTheDocument()
    expect(screen.queryByText("Preview (Markdown + LaTeX)")).not.toBeInTheDocument()
  })
})
