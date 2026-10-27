import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import NotesManagerPage from "../NotesManagerPage"

const {
  mockBgRequest,
  mockMessageSuccess,
  mockMessageError,
  mockMessageWarning,
  mockMessageInfo,
  mockNavigate,
  mockConfirmDanger,
  mockGetSetting,
  mockSetSetting,
  mockClearSetting,
  canonicalConfig
} = vi.hoisted(() => ({
  canonicalConfig: { serverUrl: "https://notes.test", authMode: "single-user", authSource: "manual", apiKey: "synthetic-notes-key" },
  mockBgRequest: vi.fn(),
  mockMessageSuccess: vi.fn(),
  mockMessageError: vi.fn(),
  mockMessageWarning: vi.fn(),
  mockMessageInfo: vi.fn(),
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
    success: mockMessageSuccess,
    error: mockMessageError,
    warning: mockMessageWarning,
    info: mockMessageInfo
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
    getConfig: vi.fn(async () => canonicalConfig),
    getChat: vi.fn(async () => null),
    listChatMessages: vi.fn(async () => []),
    getCharacter: vi.fn(async () => null)
  }
}))

// Hold I/O while keeping the real canonical config and verified Notes owner hooks.
vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (key: string, fallback: unknown) => [
    key === "tldwConfig" ? canonicalConfig : canonicalConfig[key] ?? fallback,
    vi.fn()
  ]
}))
vi.mock("@/services/tldw/TldwAuth", () => ({
  tldwAuth: { getCurrentUser: vi.fn(async () => ({ id: 7, is_active: true })) }
}))

vi.mock("@/components/Notes/NotesListPanel", () => ({
  default: ({ onExportAllMd }: any) => (
    <button data-testid="trigger-export-md" onClick={() => onExportAllMd()}>
      Export MD
    </button>
  )
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

const makeNotes = (count: number, offset = 0) =>
  Array.from({ length: count }, (_, index) => ({
    id: offset + index + 1,
    title: `Note ${offset + index + 1}`,
    content: `Content ${offset + index + 1}`,
    metadata: { keywords: [] }
  }))

describe("NotesManagerPage stage 35 export preflight", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockGetSetting.mockResolvedValue(null)
    mockSetSetting.mockResolvedValue(undefined)
    mockClearSetting.mockResolvedValue(undefined)

    Object.defineProperty(URL, "createObjectURL", {
      configurable: true,
      writable: true,
      value: vi.fn(() => "blob:notes-export")
    })
    Object.defineProperty(URL, "revokeObjectURL", {
      configurable: true,
      writable: true,
      value: vi.fn()
    })
    vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => undefined)
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("prompts for confirmation and cancels large export when user declines", async () => {
    mockConfirmDanger.mockResolvedValue(false)
    mockBgRequest.mockImplementation(async (request: { path?: string; method?: string }) => {
      const path = String(request.path || "")
      const method = String(request.method || "GET").toUpperCase()
      if (path === "/api/v1/admin/notes/title-settings" && method === "GET") {
        return {
          llm_enabled: false,
          default_strategy: "heuristic",
          effective_strategy: "heuristic",
          strategies: ["heuristic"]
        }
      }
      if (!path.startsWith("/api/v1/notes/?")) return {}
      // list_notes pages with limit/offset and reports pagination.total.
      const params = new URL(`https://example.local${path}`).searchParams
      const limit = Number(params.get("limit") || "100")
      if (limit === 20) {
        return {
          items: makeNotes(20, 0),
          pagination: { limit, offset: 0, total: 100_001 }
        }
      }
      return { items: [], pagination: { limit, offset: 0, total: 100_001 } }
    })

    renderPage()
    await screen.findByTestId("notes-large-list-pagination-hint")
    fireEvent.click(screen.getByTestId("trigger-export-md"))

    await waitFor(() => {
      expect(mockConfirmDanger).toHaveBeenCalledWith(
        expect.objectContaining({
          title: expect.stringContaining("Large MD export"),
          okText: "Start export",
          cancelText: "Cancel"
        })
      )
    })

    expect(
      mockBgRequest.mock.calls.some(([request]) =>
        String(request?.path || "").includes("limit=100")
      )
    ).toBe(false)
    expect(mockMessageSuccess).not.toHaveBeenCalled()
  })

  it("starts large export when user confirms preflight dialog", async () => {
    mockConfirmDanger.mockResolvedValue(true)
    mockBgRequest.mockImplementation(async (request: { path?: string; method?: string }) => {
      const path = String(request.path || "")
      const method = String(request.method || "GET").toUpperCase()
      if (path === "/api/v1/admin/notes/title-settings" && method === "GET") {
        return {
          llm_enabled: false,
          default_strategy: "heuristic",
          effective_strategy: "heuristic",
          strategies: ["heuristic"]
        }
      }
      if (!path.startsWith("/api/v1/notes/?")) return {}
      const params = new URL(`https://example.local${path}`).searchParams
      const limit = Number(params.get("limit") || "100")
      const offset = Number(params.get("offset") || "0")
      if (limit === 20) {
        return {
          items: makeNotes(20, 0),
          pagination: { limit, offset, total: 100_001 }
        }
      }
      if (limit === 100 && offset === 0) {
        return {
          items: makeNotes(5, 0),
          pagination: { limit, offset, total: 100_001 }
        }
      }
      return { items: [], pagination: { limit, offset, total: 100_001 } }
    })

    renderPage()
    await screen.findByTestId("notes-large-list-pagination-hint")
    fireEvent.click(screen.getByTestId("trigger-export-md"))

    await waitFor(() => {
      expect(mockConfirmDanger).toHaveBeenCalled()
    })
    await waitFor(() => {
      expect(
        mockBgRequest.mock.calls.some(([request]) =>
          String(request?.path || "").includes("limit=100")
        )
      ).toBe(true)
    })
    expect(mockMessageSuccess).toHaveBeenCalled()
  })
})
