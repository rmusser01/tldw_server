import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
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

vi.mock("@/components/Notes/NotesListPanel", () => ({
  default: ({ notes = [], onSelectNote }: Pick<React.ComponentProps<typeof import("../NotesListPanel").default>, "notes" | "onSelectNote">) => (
    <div data-testid="notes-list-panel">
      {notes.map((note) => (
        <button
          key={note.id}
          data-testid={`notes-list-item-${note.id}`}
          onClick={() => onSelectNote(note.id)}
        >
          {note.title}
        </button>
      ))}
    </div>
  )
}))

vi.mock("@/components/Notes/NotesEditorHeader", () => ({
  default: ({ onExport }: Pick<React.ComponentProps<typeof import("../NotesEditorHeader").default>, "onExport">) => (
    <div>
      <button data-testid="print-note-action" onClick={() => onExport("print")}>
        Print note
      </button>
    </div>
  )
}))

type TemplateType = "lined" | "grid" | "cornell"
type HandwritingMode = "off" | "accented"

const makeStudioState = (options: {
  noteId?: string
  noteTitle?: string
  noteContent?: string
  template: TemplateType
  handwritingMode: HandwritingMode
  isStale: boolean
  cachedSvg?: string | null
}) => ({
  note: {
    id: options.noteId || "derived-1",
    title: options.noteTitle || "Studio note",
    content: options.noteContent || "# Studio note\n\nCanonical Markdown companion",
    metadata: { keywords: ["notes"] },
    studio: {
      note_id: options.noteId || "derived-1",
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
          title: "Cue",
          items: ["Prompt"]
        },
        {
          id: "notes-1",
          kind: "notes",
          title: "Notes",
          content: "Main notes section."
        },
        {
          id: "summary-1",
          kind: "summary",
          title: "Summary",
          content: "Summary content."
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

const setNavigatorLanguage = (language: string) => {
  Object.defineProperty(window.navigator, "language", {
    configurable: true,
    value: language
  })
}

const noteTitleForId = (noteId: string, useStudioNote: boolean): string => {
  if (!useStudioNote) return "Plain note"
  return noteId === "derived-2" ? "Second studio note" : "Studio note"
}

const noteContentForId = (noteId: string, useStudioNote: boolean): string => {
  if (!useStudioNote) return "# Plain note\n\nPlain markdown body"
  return noteId === "derived-2"
    ? "# Second studio note\n\nCanonical Markdown companion"
    : "# Studio note\n\nCanonical Markdown companion"
}

describe("NotesManagerPage stage 45 notes studio export", () => {
  let useStudioNote = true
  let currentTemplate: TemplateType = "cornell"
  let currentHandwritingMode: HandwritingMode = "off"
  let currentCachedSvg: string | null = null
  let currentStale = false

  beforeEach(() => {
    cleanup()
    vi.clearAllMocks()
    setNavigatorLanguage("en-US")
    mockConfirmDanger.mockResolvedValue(true)
    mockGetSetting.mockResolvedValue("derived-1")
    mockSetSetting.mockResolvedValue(undefined)
    mockClearSetting.mockResolvedValue(undefined)

    useStudioNote = true
    currentTemplate = "cornell"
    currentHandwritingMode = "off"
    currentStale = false
    currentCachedSvg =
      '<svg viewBox="0 0 120 80" xmlns="http://www.w3.org/2000/svg"><text x="10" y="20">Diagram</text></svg>'

    mockBgRequest.mockImplementation(async (request: { path?: string; method?: string }) => {
      const path = String(request.path || "")
      const method = String(request.method || "GET").toUpperCase()

      if (path.startsWith("/api/v1/notes/?")) {
        return {
          items: (useStudioNote ? ["derived-1", "derived-2"] : ["derived-1"]).map((noteId) => ({
            id: noteId,
            title: noteTitleForId(noteId, useStudioNote),
            content_preview: "Canonical Markdown companion",
            version: 1,
            studio: useStudioNote
              ? {
                  note_id: noteId,
                  template_type: currentTemplate,
                  handwriting_mode: currentHandwritingMode,
                  source_note_id: "source-1",
                  excerpt_hash: "sha256:excerpt",
                  companion_content_hash: "sha256:companion",
                  render_version: 1
                }
              : null
          })),
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

      const noteMatch = path.match(/^\/api\/v1\/notes\/(derived-[12])$/)
      if (noteMatch && method === "GET") {
        const noteId = noteMatch[1]
        return {
          id: noteId,
          title: noteTitleForId(noteId, useStudioNote),
          content: noteContentForId(noteId, useStudioNote),
          version: 1,
          metadata: { keywords: ["notes"] },
          studio: useStudioNote
            ? {
                note_id: noteId,
                template_type: currentTemplate,
                handwriting_mode: currentHandwritingMode,
                source_note_id: "source-1",
                excerpt_hash: "sha256:excerpt",
                companion_content_hash: "sha256:companion",
                render_version: 1
              }
            : null
        }
      }

      const studioMatch = path.match(/^\/api\/v1\/notes\/(derived-[12])\/studio$/)
      if (studioMatch && method === "GET") {
        const noteId = studioMatch[1]
        return makeStudioState({
          noteId,
          noteTitle: noteTitleForId(noteId, true),
          noteContent: noteContentForId(noteId, true),
          template: currentTemplate,
          handwritingMode: currentHandwritingMode,
          isStale: currentStale,
          cachedSvg: currentCachedSvg
        })
      }

      return {}
    })
  })

  it("prints selected Studio note with notebook HTML, selected paper size, Cornell layout, and SVG diagram", async () => {
    const mockPrintWindow = {
      document: {
        open: vi.fn(),
        write: vi.fn(),
        close: vi.fn()
      },
      focus: vi.fn(),
      print: vi.fn()
    }
    const openSpy = vi.spyOn(window, "open").mockReturnValue(mockPrintWindow as unknown as Window)

    renderPage()

    const paperSizeSelect = await screen.findByTestId("notes-studio-paper-size-select")
    fireEvent.change(paperSizeSelect, {
      target: { value: "A5" }
    })
    fireEvent.click(screen.getByTestId("print-note-action"))

    await waitFor(() => {
      expect(openSpy).toHaveBeenCalled()
      expect(mockPrintWindow.document.write).toHaveBeenCalledTimes(1)
    })

    const html = String(mockPrintWindow.document.write.mock.calls[0]?.[0] || "")
    expect(html).toContain('data-paper-size="A5"')
    expect(html).toContain("studio-template-cornell")
    expect(html).toContain("studio-cornell-layout")
    expect(html).toContain("<svg")
    expect(mockPrintWindow.focus).toHaveBeenCalled()
    expect(mockPrintWindow.print).toHaveBeenCalled()
  })

  it("uses locale-driven default paper size: US locale -> US Letter, non-US -> A4", async () => {
    const usPrintWindow = {
      document: {
        open: vi.fn(),
        write: vi.fn(),
        close: vi.fn()
      },
      focus: vi.fn(),
      print: vi.fn()
    }
    const openSpy = vi.spyOn(window, "open").mockReturnValue(usPrintWindow as unknown as Window)

    setNavigatorLanguage("en-US")
    renderPage()
    await screen.findByTestId("notes-studio-view")
    fireEvent.click(screen.getByTestId("print-note-action"))

    await waitFor(() => {
      expect(usPrintWindow.document.write).toHaveBeenCalledTimes(1)
    })
    const usHtml = String(usPrintWindow.document.write.mock.calls[0]?.[0] || "")
    expect(usHtml).toContain('data-paper-size="US Letter"')

    cleanup()
    vi.clearAllMocks()
    const intlPrintWindow = {
      document: {
        open: vi.fn(),
        write: vi.fn(),
        close: vi.fn()
      },
      focus: vi.fn(),
      print: vi.fn()
    }
    openSpy.mockReturnValue(intlPrintWindow as unknown as Window)
    setNavigatorLanguage("de-DE")
    renderPage()
    await screen.findByTestId("notes-studio-view")
    fireEvent.click(screen.getByTestId("print-note-action"))

    await waitFor(() => {
      expect(intlPrintWindow.document.write).toHaveBeenCalledTimes(1)
    })
    const intlHtml = String(intlPrintWindow.document.write.mock.calls[0]?.[0] || "")
    expect(intlHtml).toContain('data-paper-size="A4"')
  })

  it("resets Studio paper size when switching to a different selected note", async () => {
    renderPage()

    const paperSizeSelect = await screen.findByTestId("notes-studio-paper-size-select")
    fireEvent.change(paperSizeSelect, {
      target: { value: "A5" }
    })

    fireEvent.click(screen.getByTestId("notes-list-item-derived-2"))

    await waitFor(() => {
      expect(screen.getByTestId("notes-studio-heading")).toHaveTextContent("Second studio note")
    })
    expect(await screen.findByTestId("notes-studio-paper-size-select")).toHaveValue("US Letter")
  })

  it("shows an actionable error when the print pop-up cannot be opened", async () => {
    vi.spyOn(window, "open").mockReturnValue(null)

    renderPage()
    await screen.findByTestId("notes-studio-view")
    fireEvent.click(screen.getByTestId("print-note-action"))

    await waitFor(() => {
      expect(mockMessageError).toHaveBeenCalledWith(
        "Unable to open print view. Please allow pop-ups and try again."
      )
    })
  })

  it("falls back to plain single-note print path for non-Studio notes", async () => {
    useStudioNote = false
    const mockPrintWindow = {
      document: {
        open: vi.fn(),
        write: vi.fn(),
        close: vi.fn()
      },
      focus: vi.fn(),
      print: vi.fn()
    }
    vi.spyOn(window, "open").mockReturnValue(mockPrintWindow as unknown as Window)

    renderPage()

    await screen.findByPlaceholderText("Write your note here... (Markdown supported)")
    fireEvent.click(screen.getByTestId("print-note-action"))

    await waitFor(() => {
      expect(mockPrintWindow.document.write).toHaveBeenCalledTimes(1)
    })

    const html = String(mockPrintWindow.document.write.mock.calls[0]?.[0] || "")
    expect(html).toContain("print-shell")
    expect(html).not.toContain("notes-studio-print-shell")
    expect(html).toContain("Plain note")
  })

  it("falls back to plain single-note print when the Studio view is stale", async () => {
    currentStale = true
    const mockPrintWindow = {
      document: {
        open: vi.fn(),
        write: vi.fn(),
        close: vi.fn()
      },
      focus: vi.fn(),
      print: vi.fn()
    }
    vi.spyOn(window, "open").mockReturnValue(mockPrintWindow as unknown as Window)

    renderPage()

    await screen.findByTestId("notes-studio-stale-banner")
    fireEvent.click(screen.getByTestId("print-note-action"))

    await waitFor(() => {
      expect(mockPrintWindow.document.write).toHaveBeenCalledTimes(1)
    })

    const html = String(mockPrintWindow.document.write.mock.calls[0]?.[0] || "")
    expect(html).toContain("print-shell")
    expect(html).not.toContain("notes-studio-print-shell")
  })
})
