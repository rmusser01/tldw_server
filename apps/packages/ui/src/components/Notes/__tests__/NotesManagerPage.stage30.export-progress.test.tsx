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
  mockClearSetting
} = vi.hoisted(() => ({
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
    getChat: vi.fn(async () => null),
    listChatMessages: vi.fn(async () => []),
    getCharacter: vi.fn(async () => null)
  }
}))

vi.mock("@/components/Notes/NotesListPanel", () => ({
  default: ({ onExportAllMd, onCancelExport, exportProgress }: any) => (
    <div>
      <button data-testid="trigger-export-md" onClick={() => onExportAllMd()}>
        Export MD
      </button>
      <button data-testid="trigger-export-cancel" onClick={() => onCancelExport?.()}>
        Cancel export
      </button>
      <div data-testid="notes-export-progress-prop">
        {exportProgress
          ? `${exportProgress.format}:${exportProgress.fetchedPages}:${exportProgress.fetchedNotes}:${exportProgress.failedBatches}`
          : "idle"}
      </div>
    </div>
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

describe("NotesManagerPage stage 30 export progress", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockConfirmDanger.mockResolvedValue(true)
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

  it("updates export progress during chunked export and clears progress after completion", async () => {
    let resolvePage2: ((value: unknown) => void) | null = null
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
      const offset = Number(params.get("offset") || "0")
      if (limit !== 100) {
        return { items: [], pagination: { limit, offset, total: 0 } }
      }
      if (offset === 0) {
        return { items: makeNotes(100, 0), pagination: { limit, offset, total: 120 } }
      }
      if (offset === 100) {
        return await new Promise((resolve) => {
          resolvePage2 = resolve
        })
      }
      return { items: [], pagination: { limit, offset, total: 120 } }
    })

    renderPage()
    fireEvent.click(screen.getByTestId("trigger-export-md"))

    await waitFor(() => {
      expect(screen.getByTestId("notes-export-progress-prop")).toHaveTextContent("md:1:100:0")
    })

    resolvePage2?.({
      items: makeNotes(20, 100),
      pagination: { limit: 100, offset: 100, total: 120 }
    })

    await waitFor(() => {
      expect(screen.getByTestId("notes-export-progress-prop")).toHaveTextContent("md:2:120:0")
    })
    await waitFor(() => {
      expect(screen.getByTestId("notes-export-progress-prop")).toHaveTextContent("idle")
    })
    expect(mockMessageSuccess).toHaveBeenCalled()
  })

  it("surfaces partial-failure warning when a later export batch fails", async () => {
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
      if (limit !== 100) {
        return { items: [], pagination: { limit, offset, total: 0 } }
      }
      if (offset === 0) {
        return { items: makeNotes(100, 0), pagination: { limit, offset, total: 200 } }
      }
      if (offset === 100) {
        throw new Error("batch timeout")
      }
      return { items: [], pagination: { limit, offset, total: 200 } }
    })

    renderPage()
    fireEvent.click(screen.getByTestId("trigger-export-md"))

    await waitFor(() => {
      expect(mockMessageWarning).toHaveBeenCalledWith(
        expect.stringContaining("partial data")
      )
    })
    expect(screen.getByTestId("notes-export-progress-prop")).toHaveTextContent("idle")
    expect(mockMessageSuccess).toHaveBeenCalled()
  })

  it("Cancel aborts the in-flight batch and stops the export without a download (NL-02)", async () => {
    let page2Signal: AbortSignal | undefined
    mockBgRequest.mockImplementation(
      async (request: { path?: string; method?: string; abortSignal?: AbortSignal }) => {
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
        if (limit !== 100) {
          return { items: [], pagination: { limit, offset, total: 0 } }
        }
        if (offset === 0) {
          return { items: makeNotes(100, 0), pagination: { limit, offset, total: 500 } }
        }
        page2Signal = request.abortSignal
        return await new Promise((_resolve, reject) => {
          request.abortSignal?.addEventListener("abort", () =>
            reject(new DOMException("Aborted", "AbortError"))
          )
        })
      }
    )
    const exportRequests = () =>
      mockBgRequest.mock.calls.filter(([request]) =>
        String(request?.path || "").includes("limit=100")
      ).length

    renderPage()
    fireEvent.click(screen.getByTestId("trigger-export-md"))
    await waitFor(() => {
      expect(screen.getByTestId("notes-export-progress-prop")).toHaveTextContent("md:1:100:0")
    })
    await waitFor(() => {
      expect(exportRequests()).toBe(2)
    })

    fireEvent.click(screen.getByTestId("trigger-export-cancel"))

    await waitFor(() => {
      expect(screen.getByTestId("notes-export-progress-prop")).toHaveTextContent("idle")
    })
    expect(page2Signal?.aborted).toBe(true)
    expect(exportRequests()).toBe(2)
    expect(mockMessageInfo).toHaveBeenCalledWith("Export cancelled")
    expect(mockMessageSuccess).not.toHaveBeenCalled()
    expect(URL.createObjectURL).not.toHaveBeenCalled()
  })
})
