import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import NotesManagerPage from "../NotesManagerPage"

// NL-05 (#3101): bulk actions and Trash at scale.

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
  mockMessageOpen,
  mockNavigate,
  mockConfirmDanger,
  mockGetSetting,
  mockSetSetting,
  mockClearSetting,
  mockPromptBulkAddTags,
  mockShowUndoNotification
} = vi.hoisted(() => ({
  mockBgRequest: vi.fn(),
  mockMessageSuccess: vi.fn(),
  mockMessageError: vi.fn(),
  mockMessageWarning: vi.fn(),
  mockMessageInfo: vi.fn(),
  mockMessageOpen: vi.fn(),
  mockNavigate: vi.fn(),
  mockConfirmDanger: vi.fn(),
  mockGetSetting: vi.fn(),
  mockSetSetting: vi.fn(),
  mockClearSetting: vi.fn(),
  mockPromptBulkAddTags: vi.fn(),
  mockShowUndoNotification: vi.fn()
}))

vi.mock("@/components/Notes/NotesBulkAddTagsPrompt", () => ({
  promptBulkAddTags: mockPromptBulkAddTags
}))

vi.mock("@/hooks/useUndoNotification", () => ({
  useUndoNotification: () => ({ showUndoNotification: mockShowUndoNotification })
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, defaultValueOrOptions?: string | { defaultValue?: string; [key: string]: unknown }) => {
      const template =
        typeof defaultValueOrOptions === "string" ? defaultValueOrOptions : defaultValueOrOptions?.defaultValue
      if (!template) return key
      if (typeof defaultValueOrOptions !== "object") return template
      return template.replace(/\{\{(\w+)\}\}/g, (_, name: string) => String(defaultValueOrOptions[name] ?? ""))
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
    info: mockMessageInfo,
    open: mockMessageOpen,
    destroy: vi.fn()
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

type Request = { path?: string; method?: string; body?: Record<string, unknown> }
type Route = (request: Request, path: string, method: string) => unknown

const ACTIVE_NOTES = [
  { id: "n1", title: "Alpha", content: "a", version: 1 },
  { id: "n2", title: "Beta", content: "b", version: 1 },
  { id: "n3", title: "Gamma", content: "c", version: 1 }
]

/** Routes requests through `route` first; anything it leaves undefined gets a default. */
const serve = (route: Route = () => undefined) => {
  mockBgRequest.mockImplementation(async (request: Request) => {
    const path = String(request.path || "")
    const method = String(request.method || "GET").toUpperCase()
    const routed = await route(request, path, method)
    if (routed !== undefined) return routed
    if (path.startsWith("/api/v1/notes/?")) {
      return { items: ACTIVE_NOTES, pagination: { total: ACTIVE_NOTES.length } }
    }
    if (path === "/api/v1/admin/notes/title-settings") {
      return { llm_enabled: false, default_strategy: "heuristic", effective_strategy: "heuristic", strategies: ["heuristic"] }
    }
    const detail = path.match(/^\/api\/v1\/notes\/(n\d)$/)
    if (detail && method === "GET") {
      const note = ACTIVE_NOTES.find((row) => row.id === detail[1])
      return { ...note, keywords: [] }
    }
    return {}
  })
}

const deferred = <T = unknown,>() => {
  let resolve!: (value: T) => void
  let reject!: (error: unknown) => void
  const promise = new Promise<T>((res, rej) => {
    resolve = res
    reject = rej
  })
  return { promise, resolve, reject }
}

const noteIdOf = (path: string) => path.match(/^\/api\/v1\/notes\/([^/?]+)/)?.[1] ?? ""

const callsTo = (method: string, id?: string) =>
  mockBgRequest.mock.calls.filter(([request]: [Request]) => {
    const path = String(request?.path || "")
    return String(request?.method || "GET").toUpperCase() === method && (id == null || noteIdOf(path) === id)
  })

const renderPage = () => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } }
  })
  return render(
    <QueryClientProvider client={queryClient}>
      <NotesManagerPage />
    </QueryClientProvider>
  )
}

const selectRows = async (...ids: string[]) => {
  for (const id of ids) fireEvent.click(await screen.findByTestId(`notes-select-checkbox-${id}`))
  await waitFor(() => {
    expect(screen.getByTestId("notes-bulk-selected-count")).toHaveTextContent(`${ids.length} selected`)
  })
}

describe("NotesManagerPage NL-05 bulk runs", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockConfirmDanger.mockResolvedValue(true)
    mockGetSetting.mockResolvedValue(null)
    mockSetSetting.mockResolvedValue(undefined)
    mockClearSetting.mockResolvedValue(undefined)
  })

  it("shows n/N progress and disables the bar while a bulk delete runs", async () => {
    const pending: Record<string, ReturnType<typeof deferred>> = { n1: deferred(), n2: deferred() }
    serve((_request, path, method) => (method === "DELETE" ? pending[noteIdOf(path)].promise : undefined))
    renderPage()
    await selectRows("n1", "n2")

    fireEvent.click(screen.getByTestId("notes-bulk-delete"))

    const progress = await screen.findByTestId("notes-bulk-progress")
    expect(progress).toHaveTextContent("Deleting 0/2")
    for (const testId of ["notes-bulk-export", "notes-bulk-assign-keywords", "notes-bulk-delete", "notes-bulk-clear-selection"]) {
      expect(screen.getByTestId(testId)).toBeDisabled()
    }
    // A second bulk action cannot start while this one runs.
    fireEvent.click(screen.getByTestId("notes-bulk-assign-keywords"))
    expect(mockPromptBulkAddTags).not.toHaveBeenCalled()

    pending.n1.resolve({})
    await waitFor(() => expect(screen.getByTestId("notes-bulk-progress")).toHaveTextContent("Deleting 1/2"))
    pending.n2.resolve({})

    await waitFor(() => expect(screen.queryByTestId("notes-bulk-progress")).not.toBeInTheDocument())
    expect(mockMessageSuccess).toHaveBeenCalledWith("Deleted 2 selected notes")
    expect(callsTo("DELETE")).toHaveLength(2)
  })

  it("lists the notes a bulk delete could not delete, and Retry deletes only those", async () => {
    let failBeta = true
    serve((_request, path, method) => {
      if (method !== "DELETE") return undefined
      if (noteIdOf(path) === "n2" && failBeta) {
        failBeta = false
        throw new Error("server unavailable")
      }
      return {}
    })
    renderPage()
    await selectRows("n1", "n2")
    fireEvent.click(screen.getByTestId("notes-bulk-delete"))

    const failures = await screen.findByTestId("notes-bulk-failures")
    expect(failures).toHaveTextContent("Delete failed for 1:")
    expect(within(failures).getByText("Beta")).toBeInTheDocument()
    expect(failures).not.toHaveTextContent("Alpha")

    fireEvent.click(within(failures).getByTestId("notes-bulk-retry"))

    await waitFor(() => expect(callsTo("DELETE", "n2")).toHaveLength(2))
    await waitFor(() => expect(screen.queryByTestId("notes-bulk-failures")).not.toBeInTheDocument())
    expect(callsTo("DELETE", "n1")).toHaveLength(1)
    expect(mockConfirmDanger).toHaveBeenCalledTimes(1)
  })

  it("retries failed tag additions with the same tags, without asking again", async () => {
    mockPromptBulkAddTags.mockResolvedValue(["research"])
    let failAlpha = true
    serve((_request, path, method) => {
      if (method !== "PATCH") return undefined
      if (noteIdOf(path) === "n1" && failAlpha) {
        failAlpha = false
        throw new Error("server unavailable")
      }
      return { id: noteIdOf(path), version: 2, keywords: ["research"] }
    })
    renderPage()
    await selectRows("n1", "n2")
    fireEvent.click(screen.getByTestId("notes-bulk-assign-keywords"))

    const failures = await screen.findByTestId("notes-bulk-failures")
    expect(failures).toHaveTextContent("Adding tags failed for 1:")
    expect(within(failures).getByText("Alpha")).toBeInTheDocument()

    fireEvent.click(within(failures).getByTestId("notes-bulk-retry"))

    await waitFor(() => expect(callsTo("PATCH", "n1")).toHaveLength(2))
    await waitFor(() => expect(screen.queryByTestId("notes-bulk-failures")).not.toBeInTheDocument())
    expect(mockPromptBulkAddTags).toHaveBeenCalledTimes(1)
    const retried = callsTo("PATCH", "n1")[1][0] as Request
    expect(retried.body?.keywords).toEqual(["research"])
  })

  it("dismisses the failure list", async () => {
    serve((_request, path, method) => {
      if (method === "DELETE" && noteIdOf(path) === "n1") throw new Error("server unavailable")
      return undefined
    })
    renderPage()
    await selectRows("n1")
    fireEvent.click(screen.getByTestId("notes-bulk-delete"))
    const failures = await screen.findByTestId("notes-bulk-failures")
    fireEvent.click(within(failures).getByTestId("notes-bulk-dismiss-failures"))
    await waitFor(() => expect(screen.queryByTestId("notes-bulk-failures")).not.toBeInTheDocument())
  })
})
