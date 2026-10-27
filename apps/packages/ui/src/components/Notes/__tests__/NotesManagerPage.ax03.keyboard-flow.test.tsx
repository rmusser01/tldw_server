import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import NotesManagerPage from "../NotesManagerPage"

// AX-03: keyboard-only use of Notes. The list is one Tab stop, Enter opens a
// note into the editor, Escape goes back to the list, Alt+Arrow steps notes.

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

const NOTES = [
  { id: "n-alpha", title: "Alpha", content: "Alpha body", version: 1, last_modified: "2026-10-02T12:00:00.000Z", metadata: { keywords: [] } },
  { id: "n-beta", title: "Beta", content: "Beta body", version: 1, last_modified: "2026-10-01T12:00:00.000Z", metadata: { keywords: [] } },
  { id: "n-gamma", title: "Gamma", content: "Gamma body", version: 1, last_modified: "2026-09-30T12:00:00.000Z", metadata: { keywords: [] } }
]

const rowTestId = (id: string) => `notes-open-button-${id}`
const openButton = (id: string) => screen.findByTestId(rowTestId(id))
const rowFor = (id: string) => screen.getByTestId(rowTestId(id))

describe("NotesManagerPage AX-03 keyboard flow", { timeout: 60_000 }, () => {
  beforeEach(() => {
    vi.clearAllMocks()
    responsiveState.isMobile = false
    mockConfirmDanger.mockResolvedValue(true)
    mockGetSetting.mockResolvedValue(null)
    mockClearSetting.mockResolvedValue(undefined)
    mockBgRequest.mockImplementation(async (request: { path?: string; method?: string }) => {
      const path = String(request.path || "")
      const method = String(request.method || "GET").toUpperCase()
      if (path.startsWith("/api/v1/notes/?") || path.startsWith("/api/v1/notes/search/?")) {
        return { items: NOTES, pagination: { total_items: NOTES.length, total_pages: 1 } }
      }
      const detail = NOTES.find((note) => path === `/api/v1/notes/${note.id}` && method === "GET")
      if (detail) return detail
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

  it("opens a note with Enter, focuses the editor body, and returns to the row with Escape", async () => {
    renderPage()

    const alphaRow = await openButton("n-alpha")
    expect(alphaRow).toHaveAccessibleName("Open note Alpha")
    expect(alphaRow).toHaveAttribute("tabindex", "0")
    alphaRow.focus()
    fireEvent.keyDown(alphaRow, { key: "Enter" })

    const editor = screen.getByLabelText("Note content") as HTMLTextAreaElement
    await waitFor(() => {
      expect(editor.value).toBe("Alpha body")
    })
    await waitFor(() => {
      expect(editor).toHaveFocus()
    })

    fireEvent.keyDown(editor, { key: "Escape" })
    await waitFor(() => {
      expect(rowFor("n-alpha")).toHaveFocus()
    })
  })

  it("steps to the next and previous note with Alt+ArrowDown/Up and keeps focus on the list", async () => {
    renderPage()

    const alphaRow = await openButton("n-alpha")
    fireEvent.click(alphaRow)
    const editor = screen.getByLabelText("Note content") as HTMLTextAreaElement
    await waitFor(() => {
      expect(editor.value).toBe("Alpha body")
    })

    alphaRow.focus()
    fireEvent.keyDown(alphaRow, { key: "ArrowDown", altKey: true })
    await waitFor(() => {
      expect(editor.value).toBe("Beta body")
    })
    const betaRow = rowFor("n-beta")
    await waitFor(() => {
      expect(betaRow).toHaveFocus()
    })
    expect(betaRow).toHaveAttribute("tabindex", "0")

    fireEvent.keyDown(betaRow, { key: "ArrowUp", altKey: true })
    await waitFor(() => {
      expect(editor.value).toBe("Alpha body")
    })
  })

  it("clears search on the first Escape and moves to the results on the second", async () => {
    renderPage()
    await openButton("n-alpha")

    const search = screen.getByLabelText("Search notes") as HTMLInputElement
    search.focus()
    fireEvent.change(search, { target: { value: "gam" } })
    expect(search.value).toBe("gam")

    fireEvent.keyDown(search, { key: "Escape" })
    expect(search.value).toBe("")
    expect(search).toHaveFocus()

    fireEvent.keyDown(search, { key: "Escape" })
    await waitFor(() => {
      expect(rowFor("n-alpha")).toHaveFocus()
    })
  })

  it("lands the notes-list skip link on the active row", async () => {
    renderPage()
    await openButton("n-alpha")

    fireEvent.click(screen.getByRole("link", { name: "Skip to notes list" }))
    await waitFor(() => {
      expect(rowFor("n-alpha")).toHaveFocus()
    })
  })
})
