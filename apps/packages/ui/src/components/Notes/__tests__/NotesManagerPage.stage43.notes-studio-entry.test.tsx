import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import NotesManagerPage from "../NotesManagerPage"

const notesConnectionConfig = {
  serverUrl: "https://notes.example.test",
  authMode: "multi-user" as const,
  accessToken: "test-access-token"
}

vi.mock("@/hooks/useCanonicalConnectionConfig", () => ({
  useCanonicalConnectionConfig: () => ({
    config: notesConnectionConfig,
    loading: false,
    authorityLoading: false
  })
}))

vi.mock("@/services/tldw/TldwAuth", () => ({
  tldwAuth: {
    getCurrentUser: vi.fn(async () => ({ id: 1, is_active: true }))
  }
}))

vi.mock("@/components/Notes/hooks/useNotesGraphAuthorityScope", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/components/Notes/hooks/useNotesGraphAuthorityScope")>()
  return {
    ...actual,
    useNotesGraphAuthorityScope: () =>
      actual.createNotesGraphAuthorityScope(notesConnectionConfig.serverUrl, 1)
  }
})

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

vi.mock("@/components/Common/MarkdownPreview", () => ({
  MarkdownPreview: ({ content }: { content: string }) => (
    <div data-testid="markdown-preview-content">{content}</div>
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

const sourceNote = {
  id: "source-1",
  title: "Source note",
  content: "Alpha paragraph.\n\nSelected excerpt for studio.\n\nOmega paragraph.",
  metadata: { keywords: [] },
  version: 3
}

const derivedNote = {
  id: "derived-1",
  title: "Source note Study Notes",
  content: "# Source note Study Notes\n\nStructured content",
  metadata: { keywords: [] },
  version: 1,
  studio: {
    note_id: "derived-1",
    template_type: "cornell",
    handwriting_mode: "off",
    source_note_id: "source-1",
    excerpt_hash: "sha256:demo",
    companion_content_hash: "sha256:markdown",
    render_version: 1
  }
}

describe("NotesManagerPage stage 43 notes studio entry", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockConfirmDanger.mockResolvedValue(true)
    mockGetSetting.mockResolvedValue("source-1")
    mockSetSetting.mockResolvedValue(undefined)
    mockClearSetting.mockResolvedValue(undefined)

    mockBgRequest.mockImplementation(async (request: { path?: string; method?: string; body?: any }) => {
      const path = String(request.path || "")
      const method = String(request.method || "GET").toUpperCase()

      if (path.startsWith("/api/v1/notes/?")) {
        return {
          items: [
            {
              id: "source-1",
              title: "Source note",
              content_preview: "Selected excerpt for studio.",
              version: 3
            },
            {
              id: "derived-1",
              title: "Source note Study Notes",
              content_preview: "Structured content",
              version: 1,
              studio: derivedNote.studio
            }
          ],
          pagination: { total_items: 2, total_pages: 1 }
        }
      }

      if (path === "/api/v1/notes/source-1" && method === "GET") {
        return sourceNote
      }

      if (path === "/api/v1/notes/derived-1" && method === "GET") {
        return derivedNote
      }

      if (path === "/api/v1/notes/studio/derive" && method === "POST") {
        return {
          note: derivedNote,
          studio_document: {
            note_id: "derived-1",
            payload_json: { sections: [] },
            template_type: request.body?.template_type,
            handwriting_mode: request.body?.handwriting_mode,
            source_note_id: "source-1",
            excerpt_snapshot: request.body?.excerpt_text,
            excerpt_hash: "sha256:demo",
            companion_content_hash: "sha256:markdown",
            render_version: 1,
            created_at: "2026-03-28T10:00:00Z",
            last_modified: "2026-03-28T10:00:00Z"
          },
          is_stale: false,
          stale_reason: null
        }
      }

      if (path === "/api/v1/notes/derived-1/studio" && method === "GET") {
        return {
          note: derivedNote,
          studio_document: {
            note_id: "derived-1",
            payload_json: { sections: [] },
            template_type: "cornell",
            handwriting_mode: "off",
            source_note_id: "source-1",
            excerpt_snapshot: "Selected excerpt for studio.",
            excerpt_hash: "sha256:demo",
            companion_content_hash: "sha256:markdown",
            render_version: 1,
            created_at: "2026-03-28T10:00:00Z",
            last_modified: "2026-03-28T10:00:00Z"
          },
          is_stale: false,
          stale_reason: null
        }
      }

      return {}
    })
  })

  it("shows Notes Studio in more actions and blocks empty markdown selections", async () => {
    renderPage()

    const textarea = await screen.findByPlaceholderText("Write your note here... (Markdown supported)")
    await waitFor(() => {
      expect((textarea as HTMLTextAreaElement).value).toContain("Selected excerpt for studio.")
    })

    fireEvent.click(screen.getByTestId("notes-overflow-menu-button"))
    const studioAction = await screen.findByText("Notes Studio")
    expect(studioAction).toBeInTheDocument()

    fireEvent.click(studioAction)

    await waitFor(() => {
      expect(mockMessageWarning).toHaveBeenCalledWith("Select Markdown text before opening Notes Studio.")
    })

    expect(
      mockBgRequest.mock.calls.some(([request]) => (request as { path?: string }).path === "/api/v1/notes/studio/derive")
    ).toBe(false)
  })

  it("prompts WYSIWYG users to switch to markdown before launching Notes Studio", async () => {
    renderPage()

    await screen.findByPlaceholderText("Write your note here... (Markdown supported)")

    fireEvent.click(screen.getByTestId("notes-input-mode-wysiwyg"))

    fireEvent.click(screen.getByTestId("notes-overflow-menu-button"))
    fireEvent.click(await screen.findByText("Notes Studio"))

    expect(await screen.findByText("Notes Studio works from Markdown selections only.")).toBeInTheDocument()

    const switchButton = screen.getByRole("button", { name: "Switch to Markdown" })
    fireEvent.click(switchButton)

    await waitFor(() => {
      expect(screen.queryByText("Notes Studio works from Markdown selections only.")).not.toBeInTheDocument()
    })
  })

  it("requires saving dirty notes before opening Notes Studio", async () => {
    renderPage()

    const textarea = (await screen.findByPlaceholderText(
      "Write your note here... (Markdown supported)"
    )) as HTMLTextAreaElement

    fireEvent.change(textarea, {
      target: { value: `${sourceNote.content}\n\nUnsaved local-only sentence.` }
    })

    const selectionStart = textarea.value.indexOf("Unsaved local-only sentence.")
    const selectionEnd = selectionStart + "Unsaved local-only sentence.".length
    textarea.focus()
    textarea.setSelectionRange(selectionStart, selectionEnd)
    fireEvent.select(textarea)

    fireEvent.click(screen.getByTestId("notes-overflow-menu-button"))
    fireEvent.click(await screen.findByText("Notes Studio"))

    await waitFor(() => {
      expect(mockMessageWarning).toHaveBeenCalledWith("Save this note before opening Notes Studio.")
    })

    expect(screen.queryByTestId("notes-studio-create-modal")).not.toBeInTheDocument()
    expect(
      mockBgRequest.mock.calls.some(([request]) => (request as { path?: string }).path === "/api/v1/notes/studio/derive")
    ).toBe(false)
  })

  it("captures studio options, derives a note, and reloads selected studio state", async () => {
    renderPage()

    const textarea = (await screen.findByPlaceholderText(
      "Write your note here... (Markdown supported)"
    )) as HTMLTextAreaElement

    const selectionStart = textarea.value.indexOf("Selected excerpt for studio.")
    const selectionEnd = selectionStart + "Selected excerpt for studio.".length
    textarea.focus()
    textarea.setSelectionRange(selectionStart, selectionEnd)
    fireEvent.select(textarea)

    fireEvent.click(screen.getByTestId("notes-overflow-menu-button"))
    fireEvent.click(await screen.findByText("Notes Studio"))

    const modal = await screen.findByTestId("notes-studio-create-modal")
    expect(within(modal).getByText("Choose notebook template")).toBeInTheDocument()

    fireEvent.click(within(modal).getByLabelText("Cornell"))
    fireEvent.click(within(modal).getByLabelText("Off"))
    fireEvent.click(within(modal).getByRole("button", { name: "Create Notes Studio note" }))

    await waitFor(() => {
      expect(
        mockBgRequest.mock.calls.some(([request]) => {
          const payload = request as { path?: string; method?: string; body?: any }
          return (
            payload.path === "/api/v1/notes/studio/derive" &&
            payload.method === "POST" &&
            payload.body?.template_type === "cornell" &&
            payload.body?.handwriting_mode === "off" &&
            payload.body?.excerpt_text === "Selected excerpt for studio."
          )
        })
      ).toBe(true)
    })

    expect(await screen.findByTestId("notes-studio-heading")).toHaveTextContent(
      "Source note Study Notes"
    )
    expect((await screen.findAllByText("Notes Studio")).length).toBeGreaterThan(0)

    await waitFor(() => {
      expect(
        mockBgRequest.mock.calls.some(([request]) => {
          const payload = request as { path?: string; method?: string }
          return payload.path === "/api/v1/notes/derived-1/studio" && payload.method === "GET"
        })
      ).toBe(true)
    })
  })
})
