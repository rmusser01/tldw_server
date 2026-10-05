/**
 * NE-01 (#3102): the WYSIWYG editor must be uncontrolled.
 *
 * It used to re-render its contentEditable from state on every input, which
 * replaced the DOM and collapsed the caret to offset 0, so typed text came out
 * reversed ("Hello" -> "olleH") and autosave persisted it.
 */
import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
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
  mockNavigate,
  mockConfirmDanger,
  mockGetSetting,
  mockSetSetting,
  mockClearSetting
} = vi.hoisted(() => ({
  mockBgRequest: vi.fn(),
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
    success: vi.fn(),
    error: vi.fn(),
    warning: vi.fn(),
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

const MARKDOWN_PLACEHOLDER = "Write your note here... (Markdown supported)"

const INITIAL_NOTES: Record<string, { title: string; content: string }> = {
  "note-a": { title: "Alpha note", content: "Alpha source body" },
  "note-b": { title: "Beta note", content: "Beta source body" }
}

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

/** Text between the start of `root` and the collapsed caret, or null if the caret is elsewhere. */
const textBeforeCaret = (root: HTMLElement): string | null => {
  const selection = window.getSelection()
  if (!selection || selection.rangeCount === 0 || !selection.isCollapsed) return null
  const caret = selection.getRangeAt(0)
  if (!root.contains(caret.startContainer)) return null
  const prefix = document.createRange()
  prefix.selectNodeContents(root)
  prefix.setEnd(caret.startContainer, caret.startOffset)
  return prefix.toString()
}

const openWysiwygEditor = async () => {
  fireEvent.click(await screen.findByTestId("notes-input-mode-wysiwyg"))
  return (await screen.findByTestId("notes-wysiwyg-editor")) as HTMLDivElement
}

describe("NotesManagerPage WYSIWYG typing (NE-01)", { timeout: 60_000 }, () => {
  let innerHtmlSetter: ReturnType<typeof vi.spyOn> | null = null
  let createdNote: { title: string; content: string } | null = null
  let holdCreate: Promise<void> | null = null
  let serverNotes: Record<string, { title: string; content: string }> = {}
  let conflictOnPut = false

  beforeEach(() => {
    vi.clearAllMocks()
    window.localStorage.clear()
    mockConfirmDanger.mockResolvedValue(true)
    mockGetSetting.mockResolvedValue(null)
    mockSetSetting.mockResolvedValue(undefined)
    mockClearSetting.mockResolvedValue(undefined)

    createdNote = null
    holdCreate = null
    serverNotes = Object.fromEntries(
      Object.entries(INITIAL_NOTES).map(([id, note]) => [id, { ...note }])
    )
    conflictOnPut = false

    mockBgRequest.mockImplementation(
      async (request: {
        path?: string
        method?: string
        body?: { title?: string; content?: string }
      }) => {
        const path = String(request.path || "")
        const method = String(request.method || "GET").toUpperCase()
        if (path === "/api/v1/notes/" && method === "POST") {
          if (holdCreate) await holdCreate
          createdNote = {
            title: String(request.body?.title ?? ""),
            content: String(request.body?.content ?? "")
          }
          return { id: "note-new", ...createdNote, version: 1, last_modified: "2026-10-03T00:00:00Z" }
        }
        if (path === "/api/v1/notes/note-new" && method === "GET" && createdNote) {
          return {
            id: "note-new",
            ...createdNote,
            metadata: { keywords: [] },
            version: 1,
            last_modified: "2026-10-03T00:00:00Z"
          }
        }
        if (path.startsWith("/api/v1/notes/?")) {
          return {
            items: Object.entries(serverNotes).map(([id, note]) => ({
              id,
              title: note.title,
              content: note.content,
              metadata: { keywords: [] },
              version: 1
            })),
            pagination: { total_items: Object.keys(serverNotes).length, total_pages: 1 }
          }
        }
        const noteMatch = path.match(/^\/api\/v1\/notes\/(note-[ab])$/)
        if (noteMatch && serverNotes[noteMatch[1]]) {
          const id = noteMatch[1]
          if (method === "PUT" && conflictOnPut) {
            throw Object.assign(new Error("version mismatch"), { status: 409 })
          }
          return {
            id,
            title: serverNotes[id].title,
            content: method === "PUT" ? String(request.body?.content ?? "") : serverNotes[id].content,
            metadata: { keywords: [] },
            version: method === "PUT" ? 2 : 1,
            last_modified: "2026-10-03T00:00:00Z"
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
      }
    )
  })

  afterEach(() => {
    innerHtmlSetter?.mockRestore()
    innerHtmlSetter = null
  })

  it("types forward character by character and keeps the caret at the end", async () => {
    const user = userEvent.setup()
    renderPage()
    await screen.findByPlaceholderText(MARKDOWN_PLACEHOLDER)

    const editor = await openWysiwygEditor()
    await user.click(editor)
    await user.type(editor, "Hello world")

    expect(editor.textContent).toBe("Hello world")
    expect(textBeforeCaret(editor)).toBe("Hello world")

    fireEvent.click(screen.getByTestId("notes-input-mode-markdown"))
    await waitFor(() => {
      expect(screen.getByPlaceholderText(MARKDOWN_PLACEHOLDER)).toHaveValue("Hello world")
    })
  })

  it("never rewrites the editor DOM in response to the user's own input", async () => {
    const user = userEvent.setup()
    renderPage()
    await screen.findByPlaceholderText(MARKDOWN_PLACEHOLDER)

    const editor = await openWysiwygEditor()
    const paragraph = editor.querySelector("p")
    expect(paragraph).not.toBeNull()

    await user.click(editor)
    innerHtmlSetter = vi.spyOn(Element.prototype, "innerHTML", "set")
    await user.type(editor, "abc")

    const editorWrites = innerHtmlSetter.mock.contexts.filter((context) => context === editor)
    expect(editorWrites).toHaveLength(0)
    expect(paragraph?.isConnected).toBe(true)
    expect(editor.firstChild).toBe(paragraph)
    expect(editor.textContent).toBe("abc")
  })

  it("replaces the WYSIWYG content when another note is loaded", async () => {
    const user = userEvent.setup()
    renderPage()

    fireEvent.click(await screen.findByTestId("notes-open-button-note-a"))
    await waitFor(() => {
      expect(screen.getByPlaceholderText(MARKDOWN_PLACEHOLDER)).toHaveValue("Alpha source body")
    })

    const editor = await openWysiwygEditor()
    expect(editor.textContent).toBe("Alpha source body")

    await user.click(editor)
    await user.type(editor, " edited")
    expect(editor.textContent).toBe("Alpha source body edited")

    fireEvent.click(screen.getByTestId("notes-open-button-note-b"))

    await waitFor(() => {
      expect(screen.getByTestId("notes-wysiwyg-editor").textContent).toBe("Beta source body")
    })
    expect(mockBgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/notes/note-a",
        method: "PUT",
        body: expect.objectContaining({ content: "Alpha source body edited" })
      })
    )
  })

  it("replaces the WYSIWYG content with the server version on a conflict reload", async () => {
    const user = userEvent.setup()
    renderPage()

    fireEvent.click(await screen.findByTestId("notes-open-button-note-a"))
    await waitFor(() => {
      expect(screen.getByPlaceholderText(MARKDOWN_PLACEHOLDER)).toHaveValue("Alpha source body")
    })
    const editor = await openWysiwygEditor()
    await user.click(editor)
    await user.type(editor, " local")

    conflictOnPut = true
    serverNotes["note-a"].content = "Server version body"
    await user.keyboard("{Control>}s{/Control}")
    // The single conflict panel (NS-03) replaced the "Reload" notice; "Use
    // their version" is the reload. jsdom has no clipboard, so the discard
    // confirm (mocked to accept) stands in for the copy.
    fireEvent.click(await screen.findByTestId("notes-conflict-take-theirs"))

    await waitFor(() => {
      expect(screen.getByTestId("notes-wysiwyg-editor").textContent).toBe("Server version body")
    })
  })

  it("keeps typed text when the editor remounts in split view", async () => {
    const user = userEvent.setup()
    renderPage()
    await screen.findByPlaceholderText(MARKDOWN_PLACEHOLDER)

    const editor = await openWysiwygEditor()
    await user.click(editor)
    await user.type(editor, "Kept text")

    fireEvent.click(screen.getByRole("button", { name: "Split" }))

    await waitFor(() => {
      const remounted = screen.getByTestId("notes-wysiwyg-editor")
      expect(remounted).not.toBe(editor)
      expect(remounted.textContent).toBe("Kept text")
    })
  })

  it("keeps the caret in place when a new note is reloaded after its first save", async () => {
    const user = userEvent.setup()
    renderPage()
    fireEvent.change(await screen.findByPlaceholderText("Title"), {
      target: { value: "Fresh note" }
    })

    const editor = await openWysiwygEditor()
    await user.click(editor)
    await user.type(editor, "Hello")
    await user.keyboard("{Control>}s{/Control}")

    // The first save creates the note and reloads it into the focused editor.
    await waitFor(() => {
      expect(mockBgRequest).toHaveBeenCalledWith(
        expect.objectContaining({ path: "/api/v1/notes/note-new", method: "GET" })
      )
    })
    await waitFor(() => {
      expect(screen.queryByTestId("notes-editor-loading-detail")).not.toBeInTheDocument()
    })
    expect(document.activeElement).toBe(editor)

    await user.keyboard(" world")

    expect(editor.textContent).toBe("Hello world")
    expect(textBeforeCaret(editor)).toBe("Hello world")
  })

  it("keeps text typed while the first save is in flight when switching back to Markdown", async () => {
    const user = userEvent.setup()
    let releaseCreate: () => void = () => undefined
    holdCreate = new Promise<void>((resolve) => {
      releaseCreate = resolve
    })
    renderPage()
    fireEvent.change(await screen.findByPlaceholderText("Title"), {
      target: { value: "Fresh note" }
    })

    const editor = await openWysiwygEditor()
    await user.click(editor)
    await user.type(editor, "Hello")
    await user.keyboard("{Control>}s{/Control}")
    await waitFor(() => {
      expect(mockBgRequest).toHaveBeenCalledWith(
        expect.objectContaining({ path: "/api/v1/notes/", method: "POST" })
      )
    })

    await user.keyboard(" world")
    releaseCreate()
    await waitFor(() => {
      expect(createdNote?.content).toBe("Hello")
    })

    fireEvent.click(screen.getByTestId("notes-input-mode-markdown"))

    await waitFor(() => {
      expect(screen.getByPlaceholderText(MARKDOWN_PLACEHOLDER)).toHaveValue("Hello world")
    })
  })
})
