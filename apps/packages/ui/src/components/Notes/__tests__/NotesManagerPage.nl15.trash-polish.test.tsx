import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import NotesManagerPage from "../NotesManagerPage"

// NL-15: Trash is labelled as Trash, shows Trash guidance instead of a blank
// editor, and restoring a note clears its stale "Note deleted · Undo" toast.

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
  mockMessageDestroy,
  responsiveState
} = vi.hoisted(() => ({
  mockBgRequest: vi.fn(),
  mockNavigate: vi.fn(),
  mockConfirmDanger: vi.fn(),
  mockGetSetting: vi.fn(),
  mockClearSetting: vi.fn(),
  mockMessageDestroy: vi.fn(),
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
    info: vi.fn(),
    open: vi.fn(),
    destroy: mockMessageDestroy
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

const trashedNote = {
  id: "n-trash",
  title: "Old draft",
  content: "Gone for now",
  version: 3,
  deleted: true,
  last_modified: "2026-10-02T17:00:00.000Z",
  metadata: { keywords: [] }
}

describe("NotesManagerPage NL-15 Trash polish", { timeout: 60_000 }, () => {
  beforeEach(() => {
    vi.clearAllMocks()
    responsiveState.isMobile = false
    mockConfirmDanger.mockResolvedValue(true)
    mockGetSetting.mockResolvedValue(null)
    mockClearSetting.mockResolvedValue(undefined)
    mockBgRequest.mockImplementation(async (request: { path?: string; method?: string }) => {
      const path = String(request.path || "")
      const method = String(request.method || "GET").toUpperCase()
      if (path.startsWith("/api/v1/notes/trash?")) {
        return { items: [trashedNote], total: 1 }
      }
      if (path.startsWith("/api/v1/notes/?")) {
        return { items: [], pagination: { total_items: 0, total_pages: 1 } }
      }
      if (path.startsWith("/api/v1/notes/n-trash/restore") && method === "POST") {
        return { ...trashedNote, deleted: false, version: 4 }
      }
      if (path === "/api/v1/notes/n-trash" && method === "GET") {
        return { ...trashedNote, deleted: false, version: 4 }
      }
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

  it("labels the panel Trash and replaces the blank editor with Trash guidance", async () => {
    renderPage()

    fireEvent.click(await screen.findByTestId("notes-mode-trash"))

    expect(await screen.findByTestId("notes-restore-n-trash", undefined, { timeout: 10_000 })).toBeInTheDocument()
    expect(screen.getByTestId("notes-sidebar-header-label")).toHaveTextContent("Trash")
    expect(screen.getByTestId("notes-trash-guidance")).toHaveAttribute("id", "notes-editor-region")
    expect(screen.queryByTestId("notes-editor-empty-create")).not.toBeInTheDocument()
    expect(screen.queryByRole("button", { name: "Import" })).not.toBeInTheDocument()
    expect(screen.queryByRole("button", { name: "Export" })).not.toBeInTheDocument()
  })

  it("goes back to the notes from the Trash guidance", async () => {
    renderPage()

    fireEvent.click(await screen.findByTestId("notes-mode-trash"))
    fireEvent.click(await screen.findByRole("button", { name: "Back to notes" }))

    await waitFor(() => {
      expect(screen.getByTestId("notes-sidebar-header-label")).toHaveTextContent("Notes")
    })
    expect(screen.queryByTestId("notes-trash-guidance")).not.toBeInTheDocument()
  })

  it("dismisses the note's \"Note deleted · Undo\" toast when it is restored from Trash", async () => {
    renderPage()

    fireEvent.click(await screen.findByTestId("notes-mode-trash"))
    fireEvent.click(await screen.findByTestId("notes-restore-n-trash", undefined, { timeout: 10_000 }))

    await waitFor(() => {
      expect(mockMessageDestroy).toHaveBeenCalledWith("notes-delete-n-trash")
    })
  })
})
