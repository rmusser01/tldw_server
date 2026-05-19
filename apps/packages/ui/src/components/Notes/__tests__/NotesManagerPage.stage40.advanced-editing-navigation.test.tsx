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
  mockSetSetting,
  mockClearSetting,
  mockPromptModal
} = vi.hoisted(() => ({
  mockBgRequest: vi.fn(),
  mockMessageSuccess: vi.fn(),
  mockMessageError: vi.fn(),
  mockMessageWarning: vi.fn(),
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

describe("NotesManagerPage stage 40 advanced editing and navigation", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    window.localStorage.clear()
    mockConfirmDanger.mockResolvedValue(true)
    mockGetSetting.mockResolvedValue(null)
    mockSetSetting.mockResolvedValue(undefined)
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

  it("preserves markdown content when switching to WYSIWYG and back without edits", async () => {
    renderPage()

    const textarea = screen.getByPlaceholderText(
      "Write your note here... (Markdown supported)"
    ) as HTMLTextAreaElement
    const sourceMarkdown = "# Title\n\n- item\n\nParagraph with **bold** and [link](https://example.com)"
    fireEvent.change(textarea, { target: { value: sourceMarkdown } })

    fireEvent.click(screen.getByTestId("notes-input-mode-wysiwyg"))
    expect(await screen.findByTestId("notes-wysiwyg-editor")).toBeInTheDocument()

    fireEvent.click(screen.getByTestId("notes-input-mode-markdown"))

    await waitFor(() => {
      expect(
        screen.getByPlaceholderText("Write your note here... (Markdown supported)")
      ).toHaveValue(sourceMarkdown)
    })
  }, 10000)

  it("generates table of contents entries and jumps to heading offsets in markdown mode", async () => {
    renderPage()

    const textarea = screen.getByPlaceholderText(
      "Write your note here... (Markdown supported)"
    ) as HTMLTextAreaElement
    fireEvent.change(textarea, {
      target: {
        value: "# Intro\n\n## Section A\nalpha\n\n## Section B\nbeta"
      }
    })

    expect(await screen.findByTestId("notes-toc-panel")).toBeInTheDocument()
    fireEvent.click(screen.getByTestId("notes-toc-item-section-b"))

    await waitFor(() => {
      const updatedTextarea = screen.getByPlaceholderText(
        "Write your note here... (Markdown supported)"
      ) as HTMLTextAreaElement
      expect(updatedTextarea.selectionStart).toBeGreaterThan(10)
      expect(updatedTextarea.selectionStart).toBe(updatedTextarea.selectionEnd)
    })
  })

  it("converts WYSIWYG edits back to markdown with heading and inline formatting", async () => {
    renderPage()

    const textarea = screen.getByPlaceholderText(
      "Write your note here... (Markdown supported)"
    ) as HTMLTextAreaElement
    fireEvent.change(textarea, { target: { value: "# Title\n\nParagraph" } })

    fireEvent.click(screen.getByTestId("notes-input-mode-wysiwyg"))
    const richEditor = await screen.findByTestId("notes-wysiwyg-editor")

    ;(richEditor as HTMLDivElement).innerHTML =
      '<h1 data-md-slug="title">Title</h1><p>Paragraph</p><h2 data-md-slug="new-section">New Section</h2><p>Added <strong>bold</strong></p>'
    fireEvent.input(richEditor)

    fireEvent.click(screen.getByTestId("notes-input-mode-markdown"))

    await waitFor(() => {
      const markdownTextarea = screen.getByPlaceholderText(
        "Write your note here... (Markdown supported)"
      ) as HTMLTextAreaElement
      expect(markdownTextarea.value).toContain("## New Section")
      expect(markdownTextarea.value).toContain("Added **bold**")
    })
  })

  it("restores the WYSIWYG selection before creating a link after prompt focus steals it", async () => {
    renderPage()

    const textarea = screen.getByPlaceholderText(
      "Write your note here... (Markdown supported)"
    ) as HTMLTextAreaElement
    fireEvent.change(textarea, { target: { value: "Link target here" } })

    fireEvent.click(screen.getByTestId("notes-input-mode-wysiwyg"))
    const richEditor = await screen.findByTestId("notes-wysiwyg-editor")
    ;(richEditor as HTMLDivElement).innerHTML = "<p>Link target here</p>"

    const textNode = richEditor.querySelector("p")?.firstChild
    expect(textNode?.nodeType).toBe(Node.TEXT_NODE)

    const selection = window.getSelection()
    const range = document.createRange()
    range.setStart(textNode as Text, 0)
    range.setEnd(textNode as Text, "Link target".length)
    selection?.removeAllRanges()
    selection?.addRange(range)

    const selectionSnapshots: string[] = []
    const docWithExec = document as Document & {
      execCommand?: ReturnType<typeof vi.fn>
    }
    const originalExecCommand = docWithExec.execCommand
    docWithExec.execCommand = vi.fn((command: string) => {
      if (command === "createLink") {
        selectionSnapshots.push(window.getSelection()?.toString() || "")
      }
      return true
    })

    mockPromptModal.mockImplementation(async () => {
      window.getSelection()?.removeAllRanges()
      return "https://example.com"
    })

    try {
      fireEvent.click(screen.getByTestId("notes-toolbar-link"))

      await waitFor(() => {
        expect(docWithExec.execCommand).toHaveBeenCalledWith(
          "createLink",
          false,
          "https://example.com"
        )
      })
      expect(selectionSnapshots[0]).toBe("Link target")
    } finally {
      docWithExec.execCommand = originalExecCommand
    }
  })
})
