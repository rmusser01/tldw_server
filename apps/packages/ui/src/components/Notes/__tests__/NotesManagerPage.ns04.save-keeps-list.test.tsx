import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import NotesManagerPage from "../NotesManagerPage"

// NS-04: saving a note updates its row in place; the list is refreshed in the
// background instead of being replaced by a spinner (which reset its scroll).

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

const listRows = [
  { id: "n1", title: "Old title", content: "Old body", version: 1, last_modified: "2026-10-01T10:00:00.000Z", metadata: { keywords: [] } },
  { id: "n2", title: "Other note", content: "Other body", version: 1, last_modified: "2026-09-30T10:00:00.000Z", metadata: { keywords: [] } }
]

const trashedRows = [
  { id: "t1", title: "Trashed note", content: "Trashed body", version: 3, deleted: true, last_modified: "2026-09-29T10:00:00.000Z", metadata: { keywords: [] } }
]

describe("NotesManagerPage NS-04 save keeps the list in place", { timeout: 60_000 }, () => {
  let holdListRefetch: boolean
  let holdTrash: boolean
  let holdSearch: boolean
  let releaseList: Array<() => void>

  beforeEach(() => {
    vi.clearAllMocks()
    holdListRefetch = false
    holdTrash = false
    holdSearch = false
    releaseList = []
    mockConfirmDanger.mockResolvedValue(true)
    mockGetSetting.mockResolvedValue(null)
    mockClearSetting.mockResolvedValue(undefined)
    mockBgRequest.mockImplementation(async (request: { path?: string; method?: string }) => {
      const path = String(request.path || "")
      const method = String(request.method || "GET").toUpperCase()
      const held = (payload: unknown, hold: boolean) =>
        hold
          ? new Promise((resolve) => {
              releaseList.push(() => resolve(payload))
            })
          : payload
      if (path.startsWith("/api/v1/notes/trash?")) {
        return held({ items: trashedRows, total: trashedRows.length }, holdTrash)
      }
      if (path.startsWith("/api/v1/notes/search/")) {
        return held({ items: [listRows[1]], pagination: { total_items: 1, total_pages: 1 } }, holdSearch)
      }
      if (path.startsWith("/api/v1/notes/?")) {
        const payload = { items: listRows, pagination: { total_items: listRows.length, total_pages: 1 } }
        if (!holdListRefetch) return payload
        return new Promise((resolve) => {
          releaseList.push(() => resolve(payload))
        })
      }
      if (path === "/api/v1/notes/n1" && method === "GET") return listRows[0]
      if (path === "/api/v1/notes/n1" && method === "PUT") {
        return {
          id: "n1",
          title: "New title",
          content: "Old body",
          version: 2,
          last_modified: "2026-10-02T10:00:00.000Z",
          keywords: []
        }
      }
      return {}
    })
  })

  afterEach(async () => {
    releaseList.forEach((release) => release())
    cleanup()
    while (queryClients.length > 0) {
      const queryClient = queryClients.pop()
      if (!queryClient) continue
      await queryClient.cancelQueries()
      queryClient.clear()
    }
  })

  it("updates the saved row immediately and keeps rows visible while the list refreshes", async () => {
    renderPage()

    fireEvent.click(await screen.findByTestId("notes-open-button-n1"))
    const titleInput = screen.getByPlaceholderText("Title") as HTMLInputElement
    await waitFor(() => {
      expect(titleInput.value).toBe("Old title")
    })

    holdListRefetch = true
    fireEvent.change(titleInput, { target: { value: "New title" } })
    fireEvent.keyDown(titleInput, { key: "s", ctrlKey: true })

    await waitFor(() => {
      expect(
        mockBgRequest.mock.calls.some(
          ([request]) => request?.path === "/api/v1/notes/n1" && String(request?.method).toUpperCase() === "PUT"
        )
      ).toBe(true)
    })

    // The row shows the saved title without waiting for the list to reload.
    await waitFor(() => {
      expect(screen.getByTestId("notes-open-button-n1")).toHaveTextContent("New title")
    })
    expect(screen.queryByTestId("notes-list-loading")).not.toBeInTheDocument()
    expect(screen.getByTestId("notes-open-button-n2")).toBeInTheDocument()
  })

  it("does not show active notes as Trash rows while Trash loads", async () => {
    renderPage()
    await screen.findByTestId("notes-open-button-n1")

    holdTrash = true
    fireEvent.click(screen.getByTestId("notes-mode-trash"))

    await waitFor(() => {
      expect(
        mockBgRequest.mock.calls.some(([request]) => String(request?.path || "").startsWith("/api/v1/notes/trash?"))
      ).toBe(true)
    })
    expect(screen.getByTestId("notes-list-loading")).toBeInTheDocument()
    expect(screen.queryByTestId("notes-open-button-n1")).not.toBeInTheDocument()
    expect(screen.queryByTestId("notes-restore-n1")).not.toBeInTheDocument()
    expect(screen.getByTestId("notes-header-count")).toHaveTextContent("Loading…")

    releaseList.forEach((release) => release())
    expect(await screen.findByTestId("notes-restore-t1")).toBeInTheDocument()
  })

  it("does not show the previous results as matches while a search loads", async () => {
    renderPage()
    await screen.findByTestId("notes-open-button-n1")

    holdSearch = true
    const searchInput = screen.getByRole("textbox", { name: "Search notes" })
    fireEvent.change(searchInput, { target: { value: "other" } })
    fireEvent.keyDown(searchInput, { key: "Enter", code: "Enter", keyCode: 13, charCode: 13 })

    await waitFor(() => {
      expect(
        mockBgRequest.mock.calls.some(([request]) => String(request?.path || "").startsWith("/api/v1/notes/search/"))
      ).toBe(true)
    })
    expect(screen.getByTestId("notes-list-loading")).toBeInTheDocument()
    expect(screen.queryByTestId("notes-open-button-n1")).not.toBeInTheDocument()

    releaseList.forEach((release) => release())
    expect(await screen.findByTestId("notes-open-button-n2")).toBeInTheDocument()
    expect(screen.queryByTestId("notes-open-button-n1")).not.toBeInTheDocument()
  })

  it("asks the list endpoint to skip its duplicate arrays", async () => {
    renderPage()

    await screen.findByTestId("notes-open-button-n1")

    const listPaths = mockBgRequest.mock.calls
      .map(([request]) => String(request?.path || ""))
      .filter((path) => path.startsWith("/api/v1/notes/?"))
    expect(listPaths.length).toBeGreaterThan(0)
    listPaths.forEach((path) => {
      expect(new URLSearchParams(path.split("?")[1]).get("legacy_aliases")).toBe("false")
    })
  })
})
