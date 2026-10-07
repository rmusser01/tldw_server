import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import NotesManagerPage from "../NotesManagerPage"

// Pin the notes authority scope like stage44. Without it, whether the notes
// list (and so every wikilink lookup) loads depends on ambient connection
// state from whichever test files ran beside this one.
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

const FAR_NOTE_ID = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
const WEEKLY_OLD_ID = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
const WEEKLY_NEW_ID = "cccccccc-cccc-4ccc-8ccc-cccccccccccc"

// The whole library. Only "Linked note" is on the loaded list page; the rest
// are reachable only through the server (resolve and title search).
const LIBRARY = [
  { id: "note-b", title: "Linked note" },
  { id: FAR_NOTE_ID, title: "Attention is all you need" },
  { id: WEEKLY_OLD_ID, title: "Weekly sync" },
  { id: WEEKLY_NEW_ID, title: "Weekly sync" }
]

const normalizeTitle = (title: string) => title.trim().replace(/\s+/g, " ").toLowerCase()

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

// A stand-in preview that renders the wikilink anchors the page emitted. The
// real Markdown rendering is covered by wikilinks.ux-contract.test.tsx and
// wikilinks.markdown-render.test.tsx.
vi.mock("@/components/Common/MarkdownPreview", () => ({
  MarkdownPreview: ({ content }: { content: string }) => {
    const links = Array.from(
      content.matchAll(/\]\((#note(?:-new)?:[^\s)]+)(?:\s+"([^"]*)")?\)/g)
    )
    if (links.length > 0) {
      return (
        <div>
          {links.map(([, href, title]) => (
            <a
              key={href}
              data-testid={
                href.startsWith("#note-new:")
                  ? "markdown-preview-create-link"
                  : "markdown-preview-note-link"
              }
              href={href}
              title={title}
            >
              wikilink
            </a>
          ))}
        </div>
      )
    }
    return <div data-testid="markdown-preview-content">{content}</div>
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

// antd-heavy page: each interaction can take seconds on a loaded runner.
describe("NotesManagerPage stage 7 wikilinks", { timeout: 60_000 }, () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockConfirmDanger.mockResolvedValue(true)
    mockGetSetting.mockResolvedValue(null)
    mockClearSetting.mockResolvedValue(undefined)

    mockBgRequest.mockImplementation(async (request: {
      path?: string
      method?: string
      body?: { titles?: unknown; ids?: unknown }
    }) => {
      const path = String(request.path || "")
      const method = String(request.method || "GET").toUpperCase()

      if (path === "/api/v1/notes/wikilinks/resolve" && method === "POST") {
        const titles: string[] = Array.isArray(request.body?.titles) ? request.body.titles : []
        const ids: string[] = Array.isArray(request.body?.ids) ? request.body.ids : []
        return {
          titles: titles.map((title) => {
            const matches = LIBRARY.filter((note) => normalizeTitle(note.title) === normalizeTitle(title))
            return {
              title,
              note_id: matches[0]?.id ?? null,
              note_title: matches[0]?.title ?? null,
              candidate_count: matches.length
            }
          }),
          ids: ids.map((id) => {
            const match = LIBRARY.find((note) => note.id === id)
            return { id, note_id: match?.id ?? null, note_title: match?.title ?? null }
          })
        }
      }
      if (path.startsWith("/api/v1/notes/search/?") && path.includes("title_only=true")) {
        const query = new URLSearchParams(path.split("?")[1]).get("query") || ""
        const items = LIBRARY.filter((note) =>
          note.title.toLowerCase().includes(query.toLowerCase())
        ).map((note) => ({ ...note, content: "", version: 1 }))
        return { items, total: items.length }
      }
      if (path.startsWith("/api/v1/notes/?")) {
        return {
          items: [{ id: "note-b", title: "Linked note", content: "Linked body", version: 1 }],
          pagination: { total_items: 1, total_pages: 1 }
        }
      }
      if (path.startsWith("/api/v1/notes/note-b/neighbors")) {
        // The server projected a [[Linked note]] title link from the far note.
        return {
          nodes: [
            { id: "note-b", type: "note", label: "Linked note" },
            { id: FAR_NOTE_ID, type: "note", label: "Attention is all you need" }
          ],
          edges: [
            { id: "wl-1", source: FAR_NOTE_ID, target: "note-b", type: "wikilink", directed: true },
            { id: "bl-1", source: "note-b", target: FAR_NOTE_ID, type: "backlink", directed: true }
          ]
        }
      }
      if (path === "/api/v1/notes/note-b" && method === "GET") {
        return {
          id: "note-b",
          title: "Linked note",
          content: "Loaded linked content",
          metadata: { keywords: [] },
          version: 2,
          last_modified: "2026-02-18T11:00:00.000Z"
        }
      }
      if (path === `/api/v1/notes/${FAR_NOTE_ID}` && method === "GET") {
        return {
          id: FAR_NOTE_ID,
          title: "Attention is all you need",
          content: "A note beyond the loaded page",
          metadata: { keywords: [] },
          version: 1,
          last_modified: "2026-02-18T11:00:00.000Z"
        }
      }
      return {}
    })
  })

  const requestedPaths = (method: string) =>
    mockBgRequest.mock.calls
      .map(([request]) => request)
      .filter((request) => String(request?.method || "GET").toUpperCase() === method)
      .map((request) => String(request?.path || ""))

  const getEditor = () =>
    screen.getByPlaceholderText("Write your note here... (Markdown supported)")

  it("shows wikilink suggestions and inserts selected title", async () => {
    renderPage()

    const textarea = screen.getByPlaceholderText("Write your note here... (Markdown supported)")
    fireEvent.change(textarea, { target: { value: "See [[Li" } })

    const suggestions = await screen.findByTestId("notes-wikilink-suggestions")
    expect(suggestions).toHaveTextContent("Linked note")

    fireEvent.keyDown(textarea, { key: "Enter" })

    await waitFor(() => {
      expect(
        screen.getByPlaceholderText("Write your note here... (Markdown supported)")
      ).toHaveValue("See [[Linked note]]")
    })
  })

  it("opens resolved wikilink when clicked in preview mode", async () => {
    renderPage()

    const textarea = screen.getByPlaceholderText("Write your note here... (Markdown supported)")
    fireEvent.change(textarea, { target: { value: "[[Linked note]]" } })

    fireEvent.click(screen.getByRole("button", { name: "Preview" }))
    fireEvent.click(await screen.findByTestId("markdown-preview-note-link"))

    await waitFor(() => {
      const openedLinkedNote = mockBgRequest.mock.calls.some(([request]) => {
        const path = String(request?.path || "")
        const method = String(request?.method || "GET").toUpperCase()
        return path === "/api/v1/notes/note-b" && method === "GET"
      })
      expect(openedLinkedNote).toBe(true)
    })
  })

  it("opens resolved wikilink when clicked in split preview", async () => {
    renderPage()

    const textarea = screen.getByPlaceholderText("Write your note here... (Markdown supported)")
    fireEvent.change(textarea, { target: { value: "Reference [[Linked note]] here" } })

    fireEvent.click(screen.getByRole("button", { name: "Split" }))
    fireEvent.click(await screen.findByTestId("markdown-preview-note-link"))

    await waitFor(() => {
      const openedLinkedNote = mockBgRequest.mock.calls.some(([request]) => {
        const path = String(request?.path || "")
        const method = String(request?.method || "GET").toUpperCase()
        return path === "/api/v1/notes/note-b" && method === "GET"
      })
      expect(openedLinkedNote).toBe(true)
    })
  })

  // NE-02 (#3110): links and autocomplete must not depend on the loaded page.

  it("opens a [[Title]] link to a note beyond the loaded page", async () => {
    renderPage()

    fireEvent.change(getEditor(), { target: { value: "See [[attention is all you need]]" } })
    fireEvent.click(screen.getByRole("button", { name: "Preview" }))

    const link = await screen.findByTestId("markdown-preview-note-link")
    expect(link.getAttribute("href")).toBe(`#note:${FAR_NOTE_ID}`)
    fireEvent.click(link)

    await waitFor(() => {
      expect(requestedPaths("GET")).toContain(`/api/v1/notes/${FAR_NOTE_ID}`)
    })
  })

  it("opens an [[id:UUID]] link", async () => {
    renderPage()

    fireEvent.change(getEditor(), { target: { value: `See [[id:${FAR_NOTE_ID}]]` } })
    fireEvent.click(screen.getByRole("button", { name: "Preview" }))
    fireEvent.click(await screen.findByTestId("markdown-preview-note-link"))

    await waitFor(() => {
      expect(requestedPaths("GET")).toContain(`/api/v1/notes/${FAR_NOTE_ID}`)
    })
  })

  it("suggests notes from the whole library through the server title search", async () => {
    renderPage()

    fireEvent.change(getEditor(), { target: { value: "Read [[atten" } })

    const suggestions = await screen.findByTestId("notes-wikilink-suggestions")
    await waitFor(() => {
      expect(suggestions).toHaveTextContent("Attention is all you need")
    })
    expect(
      requestedPaths("GET").some(
        (path) =>
          path.startsWith("/api/v1/notes/search/?") &&
          path.includes("title_only=true") &&
          path.includes("query=atten")
      )
    ).toBe(true)

    fireEvent.keyDown(getEditor(), { key: "Enter" })

    await waitFor(() => {
      expect(getEditor()).toHaveValue("Read [[Attention is all you need]]")
    })
  })

  it("inserts an id link when the picked title is shared by several notes", async () => {
    renderPage()

    fireEvent.change(getEditor(), { target: { value: "[[weekly" } })

    const suggestions = await screen.findByTestId("notes-wikilink-suggestions")
    await waitFor(() => {
      expect(suggestions.querySelectorAll("button")).toHaveLength(2)
    })

    fireEvent.keyDown(getEditor(), { key: "Enter" })

    await waitFor(() => {
      expect(getEditor()).toHaveValue(`[[id:${WEEKLY_OLD_ID}]]`)
    })
  })

  it("offers to create the note for an unresolved link", async () => {
    renderPage()

    fireEvent.change(getEditor(), { target: { value: "Plan: [[Plain Title Test]]" } })
    fireEvent.click(screen.getByRole("button", { name: "Preview" }))

    const createLink = await screen.findByTestId("markdown-preview-create-link")
    expect(createLink.getAttribute("title")).toContain("Create note")
    // The preview styles create-note links as missing, so they look different from resolved links.
    expect(screen.getByTestId("notes-preview-surface").className).toContain("a[href^='#note-new:']")
    fireEvent.click(createLink)

    await waitFor(() => {
      expect(screen.getByPlaceholderText("Title")).toHaveValue("Plain Title Test")
    })
    // The new note is a draft: its body is empty and ready for typing.
    expect(getEditor()).toHaveValue("")
  })

  it("shows the note that links here by [[Title]] under Backlinks", async () => {
    renderPage()

    fireEvent.change(getEditor(), { target: { value: "[[Linked note]]" } })
    fireEvent.click(screen.getByRole("button", { name: "Preview" }))
    fireEvent.click(await screen.findByTestId("markdown-preview-note-link"))
    await waitFor(() => {
      expect(screen.getByPlaceholderText("Title")).toHaveValue("Linked note")
    })

    // Connections load on demand: the section must be opened first.
    fireEvent.click(screen.getByTestId("notes-section-connections-toggle"))

    const backlinks = await screen.findByTestId("notes-backlinks-list")
    expect(
      within(backlinks).getByRole("button", { name: "Attention is all you need" })
    ).toBeInTheDocument()
  })

  it("keeps an unknown title as plain text when the server can't answer", async () => {
    mockBgRequest.mockImplementation(async (request: { path?: string }) => {
      const path = String(request.path || "")
      if (path === "/api/v1/notes/wikilinks/resolve") throw new Error("offline")
      return {}
    })
    renderPage()

    fireEvent.change(getEditor(), { target: { value: "Plan: [[Plain Title Test]]" } })
    fireEvent.click(screen.getByRole("button", { name: "Preview" }))

    expect(await screen.findByTestId("markdown-preview-content")).toHaveTextContent(
      "Plan: [[Plain Title Test]]"
    )
    expect(screen.queryByTestId("markdown-preview-create-link")).toBeNull()
  })
})
