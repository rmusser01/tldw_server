import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import NotesManagerPage from "../NotesManagerPage"

// NL-12 (decision 16): Inbox is the one way to list captured notes, and the
// any/all toggle counts and applies to the tags the reader picked, not the
// Inbox's hidden "captured" tag.

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
  getAllNoteKeywordStats: vi.fn(async () => [
    { keyword: "thesis", noteCount: 1 },
    { keyword: "reading", noteCount: 2 }
  ]),
  searchNoteKeywords: vi.fn(async (text: string) =>
    ["thesis", "reading"].filter((keyword) => keyword.includes(String(text || "").toLowerCase()))
  )
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

const tokenFilters = () =>
  mockBgRequest.mock.calls
    .map(([request]) => String(request?.path || ""))
    .filter((path) => path.startsWith("/api/v1/notes/search"))
    .map((path) => new URLSearchParams(path.split("?")[1]).getAll("tokens"))

const tagSearches = () =>
  mockBgRequest.mock.calls
    .map(([request]) => String(request?.path || ""))
    .filter((path) => path.startsWith("/api/v1/notes/search"))
    .map((path) => new URLSearchParams(path.split("?")[1]))

const typeIntoTagFilter = async (text: string) => {
  const input = screen.getByRole("combobox", { name: "Filter by tag" })
  fireEvent.mouseDown(input)
  fireEvent.change(input, { target: { value: text } })
  return input
}

describe("NotesManagerPage NL-12 Inbox and tag match", { timeout: 60_000 }, () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockConfirmDanger.mockResolvedValue(true)
    mockGetSetting.mockResolvedValue(null)
    mockClearSetting.mockResolvedValue(undefined)
    mockBgRequest.mockImplementation(async (request: { path?: string }) => {
      const path = String(request.path || "")
      if (path.startsWith("/api/v1/notes/?") || path.startsWith("/api/v1/notes/search")) {
        return {
          items: [
            {
              id: "n-thesis",
              title: "Thesis outline",
              content: "Chapters",
              version: 1,
              last_modified: "2026-10-02T17:00:00.000Z",
              keywords: [{ keyword: "thesis" }]
            }
          ],
          pagination: { total_items: 1, total_pages: 1 }
        }
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

  const pickTag = async (partial: string, tag: string) => {
    await typeIntoTagFilter(partial)
    fireEvent.click(await screen.findByTestId(`notes-keyword-filter-option-label-${tag}`))
    await waitFor(() => {
      expect(tokenFilters().some((tokens) => tokens.includes(tag))).toBe(true)
    })
  }

  const openInbox = async () => {
    fireEvent.click(screen.getByTestId("notes-view-mode-inbox"))
    await waitFor(() => {
      expect(tokenFilters().at(-1)).toEqual(["captured"])
    })
  }

  it("has no Captured chip next to the Inbox view", async () => {
    renderPage()
    await screen.findByTestId("notes-open-button-n-thesis")

    expect(screen.queryByTestId("notes-captured-filter")).not.toBeInTheDocument()
    expect(screen.getByTestId("notes-view-mode-inbox")).toBeInTheDocument()
  })

  it("offers no match choice for Inbox plus one tag, and keeps both required", async () => {
    renderPage()
    await screen.findByTestId("notes-open-button-n-thesis")
    await openInbox()

    await pickTag("thes", "thesis")

    expect(screen.queryByRole("group", { name: "Show notes with" })).not.toBeInTheDocument()
    const last = tagSearches().at(-1)
    expect(last?.getAll("tokens")).toEqual(["captured", "thesis"])
    expect(last?.get("tokens_match")).toBe("all")
    expect(last?.has("required_tokens")).toBe(false)
  })

  it("keeps Inbox notes only when the reader asks for any of two tags", async () => {
    renderPage()
    await screen.findByTestId("notes-open-button-n-thesis")
    await openInbox()
    await pickTag("thes", "thesis")
    await pickTag("read", "reading")

    fireEvent.click(screen.getByRole("button", { name: "Any of these tags" }))

    await waitFor(() => {
      const last = tagSearches().at(-1)
      expect(last?.getAll("tokens")).toEqual(["thesis", "reading"])
      expect(last?.get("tokens_match")).toBe("any")
      expect(last?.getAll("required_tokens")).toEqual(["captured"])
    })
  })
})
