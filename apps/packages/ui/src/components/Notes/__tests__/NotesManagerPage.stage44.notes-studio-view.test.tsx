import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import NotesManagerPage from "../NotesManagerPage"

// Pin the notes authority scope like the other Notes stage suites. Without it,
// whether the notes list loads depends on ambient connection state, so this
// suite passed or failed depending on which test files ran beside it.
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

vi.mock("@/components/Notes/NotesListPanel", () => ({
  default: () => <div data-testid="notes-list-panel" />
}))

type TemplateType = "lined" | "grid" | "cornell"
type HandwritingMode = "off" | "accented"

const titleFromMarkdown = (markdown: string): string => {
  const headingMatch = markdown.match(/^#\s+(.+)$/m)
  return headingMatch?.[1]?.trim() || "Studio note"
}

const makeStudioState = (options: {
  template: TemplateType
  handwritingMode: HandwritingMode
  isStale: boolean
  cachedSvg?: string | null
  noteTitle?: string
  noteContent?: string
}) => ({
  note: {
    id: "derived-1",
    title: options.noteTitle || "Studio note",
    content: options.noteContent || "# Studio note\n\nCanonical Markdown companion",
    metadata: { keywords: [] },
    studio: {
      note_id: "derived-1",
      template_type: options.template,
      handwriting_mode: options.handwritingMode,
      source_note_id: "source-1",
      excerpt_hash: "sha256:excerpt",
      companion_content_hash: "sha256:companion",
      render_version: 1
    }
  },
  studio_document: {
    note_id: "derived-1",
    template_type: options.template,
    handwriting_mode: options.handwritingMode,
    source_note_id: "source-1",
    excerpt_hash: "sha256:excerpt",
    companion_content_hash: "sha256:companion",
    render_version: 1,
    created_at: "2026-03-28T10:00:00Z",
    last_modified: "2026-03-28T10:00:00Z",
    payload_json: {
      layout: {
        template_type: options.template,
        handwriting_mode: options.handwritingMode,
        render_version: 1
      },
      sections: [
        {
          id: "cue-1",
          kind: "cue",
          title: "Key Questions",
          items: ["What is retrieval?", "Why does reranking help?"]
        },
        {
          id: "notes-1",
          kind: "notes",
          title: "Notes",
          content: "Dense body text should stay readable."
        },
        {
          id: "prompt-1",
          kind: "prompt",
          title: "Try it yourself",
          content: "Sketch the ranking flow from memory."
        },
        {
          id: "summary-1",
          kind: "summary",
          title: "Summary",
          content: "Short summary sentence."
        }
      ]
    },
    diagram_manifest_json: options.cachedSvg
      ? {
          diagram_type: "flowchart",
          source_section_ids: ["notes-1"],
          source_graph: "graph TD;A-->B;",
          cached_svg: options.cachedSvg,
          render_hash: "hash-1",
          generation_status: "ready"
        }
      : null
  },
  is_stale: options.isStale,
  stale_reason: options.isStale ? "companion_content_hash_mismatch" : null
})

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

// antd-heavy page: each interaction can take seconds on a loaded runner.
describe("NotesManagerPage stage 44 notes studio view", { timeout: 60_000 }, () => {
  let currentTemplate: TemplateType = "cornell"
  let currentHandwritingMode: HandwritingMode = "accented"
  let currentStale = false
  let currentCachedSvg: string | null = null
  let currentNoteTitle = "Studio note"
  let currentNoteContent = "# Studio note\n\nCanonical Markdown companion"

  beforeEach(() => {
    cleanup()
    vi.clearAllMocks()
    mockConfirmDanger.mockResolvedValue(true)
    mockGetSetting.mockResolvedValue("derived-1")
    mockSetSetting.mockResolvedValue(undefined)
    mockClearSetting.mockResolvedValue(undefined)

    currentTemplate = "cornell"
    currentHandwritingMode = "accented"
    currentStale = false
    currentCachedSvg = null
    currentNoteTitle = "Studio note"
    currentNoteContent = "# Studio note\n\nCanonical Markdown companion"

    mockBgRequest.mockImplementation(async (request: { path?: string; method?: string }) => {
      const path = String(request.path || "")
      const method = String(request.method || "GET").toUpperCase()

      if (path.startsWith("/api/v1/notes/?")) {
        return {
          items: [
            {
              id: "derived-1",
              title: "Studio note",
              content_preview: "Canonical Markdown companion",
              version: 1,
              studio: {
                note_id: "derived-1",
                template_type: currentTemplate,
                handwriting_mode: currentHandwritingMode,
                source_note_id: "source-1",
                excerpt_hash: "sha256:excerpt",
                companion_content_hash: "sha256:companion",
                render_version: 1
              }
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

      if (path === "/api/v1/notes/derived-1" && method === "GET") {
        return {
          id: "derived-1",
          title: currentNoteTitle,
          content: currentNoteContent,
          version: 1,
          metadata: { keywords: [] },
          studio: {
            note_id: "derived-1",
            template_type: currentTemplate,
            handwriting_mode: currentHandwritingMode,
            source_note_id: "source-1",
            excerpt_hash: "sha256:excerpt",
            companion_content_hash: "sha256:companion",
            render_version: 1
          }
        }
      }

      if (path === "/api/v1/notes/derived-1/studio" && method === "GET") {
        return makeStudioState({
          template: currentTemplate,
          handwritingMode: currentHandwritingMode,
          isStale: currentStale,
          cachedSvg: currentCachedSvg,
          noteTitle: currentNoteTitle,
          noteContent: currentNoteContent
        })
      }

      if (path === "/api/v1/notes/derived-1/studio/regenerate" && method === "POST") {
        const body = (request as { body?: { current_markdown?: string } }).body
        if (typeof body?.current_markdown === "string") {
          currentNoteTitle = titleFromMarkdown(body.current_markdown)
          currentNoteContent = body.current_markdown.trim().length > 0
            ? body.current_markdown
            : `# ${currentNoteTitle}`
        }
        currentStale = false
        return makeStudioState({
          template: currentTemplate,
          handwritingMode: currentHandwritingMode,
          isStale: false,
          cachedSvg: currentCachedSvg,
          noteTitle: currentNoteTitle,
          noteContent: currentNoteContent
        })
      }

      return {}
    })
  })

  it.each([
    ["lined" as const],
    ["grid" as const],
    ["cornell" as const]
  ])("renders a Studio shell with %s template chrome from canonical payload", async (template) => {
    currentTemplate = template

    renderPage()

    const studioView = await screen.findByTestId("notes-studio-view")
    expect(studioView).toBeInTheDocument()
    expect(studioView).toHaveClass(`studio-template-${template}`)
    expect(screen.getByTestId(`notes-studio-template-${template}`)).toBeInTheDocument()
    if (template === "lined") {
      expect(studioView).toHaveStyle({ backgroundImage: expect.stringContaining("linear-gradient") })
    }
    if (template === "grid") {
      expect(studioView).toHaveStyle({ backgroundSize: "24px 24px" })
    }
    if (template === "cornell") {
      expect(studioView).toHaveStyle({ backgroundImage: expect.stringContaining("linear-gradient") })
    }
  })

  it("applies accented handwriting to headings, cues, and prompts, not dense body text", async () => {
    currentTemplate = "cornell"
    currentHandwritingMode = "accented"

    renderPage()

    const cueHeading = await screen.findByTestId("notes-studio-section-title-cue-1")
    const cueItem = screen.getByTestId("notes-studio-cue-item-cue-1-0")
    const promptBody = screen.getByTestId("notes-studio-section-content-prompt-1")
    const denseBody = screen.getByTestId("notes-studio-section-content-notes-1")

    expect(cueHeading).toHaveClass("studio-handwriting-accent")
    expect(cueItem).toHaveClass("studio-handwriting-accent")
    expect(promptBody).toHaveClass("studio-handwriting-accent")
    expect(denseBody).not.toHaveClass("studio-handwriting-accent")
  })

  it("renders cue sections without duplicating cue items as body paragraphs", async () => {
    renderPage()

    const cueSection = await screen.findByTestId("notes-studio-section-cue-1")
    expect(cueSection).toBeInTheDocument()
    expect(screen.queryByTestId("notes-studio-section-content-cue-1")).not.toBeInTheDocument()
  })

  it("shows stale warning only when the selected studio state is stale", async () => {
    currentStale = false

    renderPage()

    await screen.findByTestId("notes-studio-view")
    expect(screen.queryByTestId("notes-studio-stale-banner")).not.toBeInTheDocument()
  })

  it("posts the current markdown when regenerating a stale Studio preview", async () => {
    currentStale = false

    renderPage()

    await screen.findByTestId("notes-studio-view")
    fireEvent.click(screen.getByRole("button", { name: "Continue editing plain note" }))

    const nextMarkdown = ""
    fireEvent.change(
      await screen.findByPlaceholderText("Write your note here... (Markdown supported)"),
      {
        target: { value: nextMarkdown }
      }
    )

    fireEvent.click(screen.getByRole("button", { name: "Preview" }))
    await screen.findByTestId("notes-studio-stale-banner")
    fireEvent.click(screen.getByRole("button", { name: "Regenerate Studio view from current Markdown" }))

    await waitFor(() => {
      expect(
        mockBgRequest.mock.calls.some(([request]) => {
          const payload = request as {
            path?: string
            method?: string
            body?: { current_markdown?: string }
          }
          return (
            payload.path === "/api/v1/notes/derived-1/studio/regenerate" &&
            payload.method === "POST" &&
            payload.body?.current_markdown === nextMarkdown
          )
        })
      ).toBe(true)
    })

    await waitFor(() => {
      expect(screen.queryByTestId("notes-studio-stale-banner")).not.toBeInTheDocument()
    })
    expect(await screen.findByTestId("notes-studio-heading")).toHaveTextContent("Studio note")
  })

  it("lets a stale Studio note return to the plain Markdown editor", async () => {
    currentStale = true

    renderPage()
    // The WYSIWYG toggle is hidden until NE-01 is fixed (D5, #3102). When it
    // returns, click "notes-input-mode-wysiwyg" here again so this test also
    // proves that continuing the plain note resets WYSIWYG back to Markdown.

    const staleBanner = await screen.findByTestId("notes-studio-stale-banner")
    fireEvent.click(
      within(staleBanner).getByRole("button", { name: "Continue editing plain note" })
    )

    await waitFor(() => {
      expect(screen.queryByTestId("notes-studio-view")).not.toBeInTheDocument()
    })

    expect(
      screen.getByPlaceholderText("Write your note here... (Markdown supported)")
    ).toBeInTheDocument()
    expect(screen.queryByTestId("notes-wysiwyg-editor")).not.toBeInTheDocument()
  })

  it("lets a non-stale Studio note reopen Markdown editing and marks preview stale after body edits", async () => {
    currentStale = false

    renderPage()

    await screen.findByTestId("notes-studio-view")
    fireEvent.click(screen.getByRole("button", { name: "Continue editing plain note" }))

    const textarea = await screen.findByPlaceholderText("Write your note here... (Markdown supported)")
    fireEvent.change(textarea, {
      target: { value: "# Studio note\n\nEdited markdown body" }
    })

    fireEvent.click(screen.getByRole("button", { name: "Preview" }))

    await waitFor(() => {
      expect(screen.getByTestId("notes-studio-stale-banner")).toBeInTheDocument()
    })
  })

  it("renders diagram card content from cached SVG in the canonical manifest", async () => {
    currentCachedSvg =
      '<svg viewBox="0 0 120 80" xmlns="http://www.w3.org/2000/svg"><text x="10" y="20">Diagram</text></svg>'

    renderPage()

    const diagramCard = await screen.findByTestId("notes-studio-diagram-card")
    expect(diagramCard).toBeInTheDocument()
    expect(diagramCard.innerHTML).toContain("<svg")
    expect(diagramCard.innerHTML).toContain("Diagram")
  })
})
