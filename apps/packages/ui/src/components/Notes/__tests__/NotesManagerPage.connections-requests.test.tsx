import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import NotesManagerPage from "../NotesManagerPage"

// Connections (G09 follow-up): a note switch aborts the neighbour request of the
// note the user left, and a 403 is remembered for the account, so switching
// notes shows "unavailable" without asking the server again.

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
  mockClearSetting,
  responsiveState
} = vi.hoisted(() => ({
  mockBgRequest: vi.fn(),
  mockNavigate: vi.fn(),
  mockConfirmDanger: vi.fn(),
  mockGetSetting: vi.fn(),
  mockClearSetting: vi.fn(),
  responsiveState: { isMobile: false }
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      defaultValueOrOptions?: string | { defaultValue?: string; [key: string]: unknown }
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

vi.mock("@/hooks/useMediaQuery", () => ({
  useMobile: () => responsiveState.isMobile
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

const queryClients: QueryClient[] = []

const renderPage = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false }
    }
  })
  queryClients.push(queryClient)
  return render(
    <QueryClientProvider client={queryClient}>
      <NotesManagerPage />
    </QueryClientProvider>
  )
}

const noteRows = ["a", "b", "c"].map((suffix, index) => ({
  id: `note-${suffix}`,
  title: `Note ${suffix.toUpperCase()}`,
  content: `Body ${suffix}`,
  version: 1,
  last_modified: `2026-10-0${3 - index}T10:00:00.000Z`,
  metadata: { keywords: [] }
}))

type NeighborRequest = { noteId: string; abortSignal?: AbortSignal }

describe("NotesManagerPage Connections requests", { timeout: 60_000 }, () => {
  let neighborRequests: NeighborRequest[]
  let answerNeighbors: (request: NeighborRequest) => Promise<unknown>

  beforeEach(() => {
    vi.clearAllMocks()
    neighborRequests = []
    mockConfirmDanger.mockResolvedValue(true)
    mockGetSetting.mockResolvedValue(null)
    mockClearSetting.mockResolvedValue(undefined)
    mockBgRequest.mockImplementation(async (request: { path?: string; method?: string; abortSignal?: AbortSignal }) => {
      const path = String(request.path || "")
      const method = String(request.method || "GET").toUpperCase()
      if (path.startsWith("/api/v1/notes/?")) {
        return { items: noteRows, pagination: { total: noteRows.length } }
      }
      const neighbors = path.match(/^\/api\/v1\/notes\/([^/?]+)\/neighbors/)
      if (neighbors) {
        const neighborRequest = { noteId: decodeURIComponent(neighbors[1]), abortSignal: request.abortSignal }
        neighborRequests.push(neighborRequest)
        return answerNeighbors(neighborRequest)
      }
      const note = noteRows.find((row) => path === `/api/v1/notes/${row.id}` && method === "GET")
      if (note) return note
      return {}
    })
  })

  afterEach(async () => {
    cleanup()
    while (queryClients.length > 0) {
      const queryClient = queryClients.pop()
      if (!queryClient) continue
      await queryClient.cancelQueries()
      queryClient.clear()
    }
  })

  const openNote = async (id: string, title: string) => {
    fireEvent.click(await screen.findByTestId(`notes-open-button-${id}`))
    await waitFor(() => {
      expect(screen.getByPlaceholderText("Title")).toHaveValue(title)
    })
  }

  it("aborts the neighbour request of a note the user switched away from", async () => {
    // Held until aborted, like a slow graph query.
    answerNeighbors = ({ abortSignal }) =>
      new Promise((_resolve, reject) => {
        abortSignal?.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")), { once: true })
      })
    renderPage()
    await openNote("note-a", "Note A")
    fireEvent.click(screen.getByTestId("notes-section-connections-toggle"))
    await waitFor(() => expect(neighborRequests.map((request) => request.noteId)).toEqual(["note-a"]))

    await openNote("note-b", "Note B")
    await openNote("note-c", "Note C")

    await waitFor(() => {
      expect(neighborRequests.map((request) => request.noteId)).toEqual(["note-a", "note-b", "note-c"])
    })
    expect(neighborRequests.map((request) => request.abortSignal?.aborted)).toEqual([true, true, false])
  })

  it("remembers a 403 for the account instead of asking again on each note switch", async () => {
    answerNeighbors = async () => {
      throw Object.assign(new Error("Forbidden (GET /api/v1/notes/note-a/neighbors)"), { status: 403 })
    }
    renderPage()
    await openNote("note-a", "Note A")
    fireEvent.click(screen.getByTestId("notes-section-connections-toggle"))
    expect(await screen.findByText("Note connections are unavailable for this account.")).toBeInTheDocument()

    await openNote("note-b", "Note B")
    await openNote("note-c", "Note C")

    expect(await screen.findByText("Note connections are unavailable for this account.")).toBeInTheDocument()
    expect(screen.getByTestId("notes-section-connections-toggle")).toHaveAttribute("aria-expanded", "true")
    expect(neighborRequests.map((request) => request.noteId)).toEqual(["note-a"])
  })
})
