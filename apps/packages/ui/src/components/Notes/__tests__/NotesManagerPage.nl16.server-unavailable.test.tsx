import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { cleanup, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import NotesManagerPage from "../NotesManagerPage"

// NL-16: when the notes server is unreachable the editor says so and that
// edits are kept on this device.

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

const failListWith = (error: Error) => {
  mockBgRequest.mockImplementation(async (request: { path?: string }) => {
    const path = String(request.path || "")
    if (path.startsWith("/api/v1/notes/?")) throw error
    return {}
  })
}

describe("NotesManagerPage NL-16 server unavailable", { timeout: 60_000 }, () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockConfirmDanger.mockResolvedValue(true)
    mockGetSetting.mockResolvedValue(null)
    mockClearSetting.mockResolvedValue(undefined)
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

  it("tells the writer the server is unreachable and that edits stay on this device", async () => {
    failListWith(new Error("Failed to fetch (GET /api/v1/notes/?limit=20&offset=0)"))
    renderPage()

    const banner = await screen.findByTestId("notes-server-unavailable-banner", undefined, { timeout: 10_000 })
    expect(banner).toHaveTextContent("Notes server unavailable")
    expect(banner).toHaveTextContent("kept on this device")
  })

  it("does not blame the server for a permission error", async () => {
    failListWith(Object.assign(new Error("Forbidden (GET /api/v1/notes/?limit=20&offset=0)"), { status: 403 }))
    renderPage()

    await waitFor(
      () => {
        expect(screen.getAllByTestId("notes-list-error-state").length).toBeGreaterThan(0)
      },
      { timeout: 10_000 }
    )
    expect(screen.queryByTestId("notes-server-unavailable-banner")).not.toBeInTheDocument()
  })
})
