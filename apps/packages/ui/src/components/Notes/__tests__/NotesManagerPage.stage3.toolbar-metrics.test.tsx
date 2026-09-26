import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import NotesManagerPage from "../NotesManagerPage"
import { consumeFlashcardsGenerateHandoff, consumeStudyPackHandoff } from "@/services/tldw/flashcards-generate-handoff"
import { loadServicePromptSnapshot } from "@/services/service-prompts"
import { flashcardsHandoffAuthority } from "@/services/tldw/flashcards-generate-transfer"
vi.mock("@plasmohq/storage", async () => import("../../../../../../tldw-frontend/extension/shims/plasmo-storage"))
vi.mock("@/services/tldw/deployment-mode", () => ({ isHostedTldwDeployment: () => false }))
vi.mock("@/services/tldw/TldwAuth", () => ({ tldwAuth: { getCurrentUser: async () => ({ id: 1, is_active: true }) } }))

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
    getConfig: async () => JSON.parse(window.localStorage.getItem("tldwConfig") || "null"),
    ensureConfigForRequest: async () => JSON.parse(window.localStorage.getItem("tldwConfig") || "null"),
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
  default: ({ onSelectNote }: { onSelectNote: (id: string) => void }) => <div data-testid="notes-list-panel"><button onClick={() => onSelectNote("11")}>Open saved note</button></div>
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

describe("NotesManagerPage stage 3 toolbar and metrics", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    window.localStorage.clear()
    window.localStorage.setItem("tldwConfig", JSON.stringify({ serverUrl: "https://notes.test", authMode: "multi-user", accessToken: `test.${btoa(JSON.stringify({ sub: "1" }))}.signature` }))
    let tail = Promise.resolve()
    const locks = { request: (_name: string, work: () => unknown) => { const next = tail.then(work); tail = next.then(() => undefined, () => undefined); return next } }
    vi.stubGlobal("navigator", new Proxy(window.navigator, { get: (target, key) => key === "locks" ? locks : Reflect.get(target, key, target) }))
    mockConfirmDanger.mockResolvedValue(true)
    mockGetSetting.mockResolvedValue(null)
    mockClearSetting.mockResolvedValue(undefined)

    mockBgRequest.mockImplementation(async (request: { path?: string; method?: string }) => {
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

  afterEach(() => vi.unstubAllGlobals())

  it.each(["single-user", "multi-user"])("transfers the selected saved Note into a Study Pack with verified ownership (%s)", async authMode => {
    if (authMode === "single-user") window.localStorage.setItem("tldwConfig", JSON.stringify({ serverUrl: "https://notes.test", authMode, apiKey: "synthetic-key" }))
    const note = { id: "11", title: "Private saved note", content: "Private content", metadata: { keywords: [] }, version: 1 }
    mockBgRequest.mockImplementation(async ({ path }: { path: string }) => {
      if (path.startsWith("/api/v1/notes/?")) return { items: [note], pagination: { total_items: 1, total_pages: 1 } }
      if (path === "/api/v1/notes/11") return note
      return {}
    })
    renderPage()
    await waitFor(() => expect(mockBgRequest.mock.calls.some(([request]) => request.path.startsWith("/api/v1/notes/?"))).toBe(true))
    fireEvent.click(await screen.findByRole("button", { name: "Open saved note" }))
    await screen.findByDisplayValue(note.title)
    const button = screen.getByTestId("notes-create-study-pack-button")
    await waitFor(() => expect(button).toBeEnabled())
    fireEvent.click(button)
    await waitFor(() => expect(mockNavigate).toHaveBeenCalledTimes(1))
    const route = new URL(mockNavigate.mock.calls[0][0], "https://app.test")
    expect(route.href).not.toContain("Private")
    const snapshot = await loadServicePromptSnapshot([])
    try {
      expect(await consumeStudyPackHandoff(route.searchParams.get("study_pack_handoff")!, flashcardsHandoffAuthority(snapshot))).toEqual({
        title: note.title, sourceItems: [{ sourceType: "note", sourceId: "11", sourceTitle: note.title }]
      })
    } finally { snapshot.release() }
    expect(screen.getByDisplayValue(note.title)).toBeInTheDocument()
  })

  it("inserts markdown syntax at cursor/selection via toolbar", async () => {
    renderPage()

    const textareaPlaceholder = "Write your note here... (Markdown supported)"
    const textarea = screen.getByPlaceholderText(textareaPlaceholder) as HTMLTextAreaElement

    fireEvent.change(textarea, { target: { value: "hello world" } })
    textarea.focus()
    textarea.setSelectionRange(0, 5)

    fireEvent.click(screen.getByTestId("notes-toolbar-bold"))

    await waitFor(() => {
      expect(textarea.value).toContain("**hello** world")
    })

    const start = textarea.value.indexOf("world")
    textarea.setSelectionRange(start, start + 5)
    fireEvent.click(screen.getByTestId("notes-toolbar-link"))

    await waitFor(() => {
      expect(textarea.value).toContain("[world](https://)")
    })
  })

  it("shows and updates editor metrics footer", async () => {
    renderPage()

    const textarea = screen.getByPlaceholderText(
      "Write your note here... (Markdown supported)"
    ) as HTMLTextAreaElement

    expect(screen.getByTestId("notes-editor-metrics")).toHaveTextContent(
      "0 words · 0 chars · 0 mins read"
    )

    fireEvent.change(textarea, { target: { value: "hello world" } })

    await waitFor(() => {
      expect(screen.getByTestId("notes-editor-metrics")).toHaveTextContent(
        "2 words · 11 chars · 1 min read"
      )
    })
  })

  it("renders save and overflow menu button in the toolbar", () => {
    renderPage()

    const saveButton = screen.getByTestId("notes-save-button")
    const overflowButton = screen.getByTestId("notes-overflow-menu-button")

    expect(saveButton).toBeInTheDocument()
    expect(overflowButton).toBeInTheDocument()

    // Save should appear before the overflow menu button in DOM order
    const position = saveButton.compareDocumentPosition(overflowButton)
    expect(position & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
  })
  it("transfers the actual unsaved Note exactly through an opaque route", async () => {
    renderPage()
    const text = " \nPrivate unsaved note\n\t "
    const textarea = screen.getByPlaceholderText("Write your note here... (Markdown supported)")
    fireEvent.change(textarea, { target: { value: text } })
    fireEvent.click(screen.getByTestId("notes-overflow-menu-button"))
    fireEvent.click(await screen.findByText("Generate flashcards"))
    await waitFor(() => expect(mockNavigate).toHaveBeenCalledTimes(1))
    const route = mockNavigate.mock.calls[0][0]
    expect(route).not.toContain("Private")
    const token = new URL(route, "https://app.test").searchParams.get("generate_handoff")!
    const snapshot = await loadServicePromptSnapshot([])
    try { expect(await consumeFlashcardsGenerateHandoff(token, flashcardsHandoffAuthority(snapshot))).toMatchObject({ text, sourceType: "note" }) }
    finally { snapshot.release() }
    expect(textarea).toHaveValue(text)
  })

  it("keeps the unsaved Note and reports unavailable shared transfer storage", async () => {
    renderPage()
    const textarea = screen.getByPlaceholderText("Write your note here... (Markdown supported)")
    fireEvent.change(textarea, { target: { value: "Keep my draft" } })
    vi.stubGlobal("navigator", new Proxy(window.navigator, { get: (target, key) => key === "locks" ? undefined : Reflect.get(target, key, target) }))
    fireEvent.click(screen.getByTestId("notes-overflow-menu-button"))
    fireEvent.click(await screen.findByText("Generate flashcards"))
    await waitFor(() => expect(mockMessageError).toHaveBeenCalled())
    expect(mockNavigate).not.toHaveBeenCalled()
    expect(textarea).toHaveValue("Keep my draft")
  })

})
