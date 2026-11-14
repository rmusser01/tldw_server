import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import NotesManagerPage from "../NotesManagerPage"

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
  mockNavigate,
  mockConfirmDanger,
  mockGetSetting,
  mockSetSetting,
  mockClearSetting
} = vi.hoisted(() => {
  return {
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

vi.mock("@/components/Common/MarkdownPreview", () => ({
  MarkdownPreview: ({ content }: { content: string }) => (
    <div data-testid="markdown-preview-content">{content}</div>
  )
}))

vi.mock("@/components/Notes/NotesListPanel", () => ({
  default: ({
    notes,
    pageSize,
    onChangePage
  }: {
    notes?: Array<{ title?: string }>
    pageSize: number
    onChangePage: (page: number, nextPageSize: number) => void
  }) => (
    <div data-testid="notes-list-panel">
      <div data-testid="notes-order">{(notes || []).map((note) => note.title).join("|")}</div>
      <div data-testid="notes-page-size">{String(pageSize)}</div>
      <button
        type="button"
        data-testid="notes-set-page-size-50"
        onClick={() => onChangePage(1, 50)}
      >
        Set page size 50
      </button>
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

const listItems = [
  {
    id: "n1",
    title: "Zulu note",
    content: "zulu content",
    version: 1,
    created_at: "2026-02-02T00:00:00.000Z",
    last_modified: "2026-02-01T00:00:00.000Z"
  },
  {
    id: "n2",
    title: "Alpha note",
    content: "alpha content",
    version: 1,
    created_at: "2026-01-01T00:00:00.000Z",
    last_modified: "2026-02-03T00:00:00.000Z"
  },
  {
    id: "n3",
    title: "Gamma note",
    content: "gamma content",
    version: 1,
    created_at: "2026-02-04T00:00:00.000Z",
    last_modified: "2026-02-02T00:00:00.000Z"
  }
]

describe("NotesManagerPage stage 14 sorting and pagination", () => {
  let storedPageSize: number | null = null

  beforeEach(() => {
    vi.clearAllMocks()
    storedPageSize = null
    mockConfirmDanger.mockResolvedValue(true)
    mockGetSetting.mockImplementation(async (setting: { key?: string }) => {
      if (setting?.key === "tldw:notesPageSize") {
        return storedPageSize
      }
      return null
    })
    mockSetSetting.mockImplementation(async (setting: { key?: string }, value: unknown) => {
      if (setting?.key === "tldw:notesPageSize") {
        storedPageSize = Number(value)
      }
      return undefined
    })
    mockClearSetting.mockResolvedValue(undefined)
    mockBgRequest.mockImplementation(async (request: { path?: string; method?: string }) => {
      const path = String(request.path || "")
      const method = String(request.method || "GET").toUpperCase()
      if (path.startsWith("/api/v1/notes/?")) {
        return {
          items: listItems,
          pagination: { total_items: listItems.length, total_pages: 1 }
        }
      }
      if (path === "/api/v1/admin/notes/title-settings" && method === "GET") {
        return {
          llm_enabled: false,
          default_strategy: "heuristic",
          effective_strategy: "heuristic",
          strategies: ["heuristic", "llm", "llm_fallback"]
        }
      }
      return {}
    })
  })

  it("applies all sort options and sends matching API query parameters", async () => {
    renderPage()
    await waitFor(() => {
      expect(screen.getByTestId("notes-order")).toHaveTextContent(
        "Alpha note|Gamma note|Zulu note"
      )
    })

    const latestListPath = () => {
      const listCall = [...mockBgRequest.mock.calls]
        .map(([request]) => String(request?.path || ""))
        .filter((path) => path.startsWith("/api/v1/notes/?"))
        .at(-1)
      return String(listCall || "")
    }

    expect(latestListPath()).toContain("sort_by=last_modified")
    expect(latestListPath()).toContain("sort_order=desc")

    // Helper to change an Ant Design Select value by option label
    const selectOption = async (testId: string, optionLabel: string) => {
      const container = screen.getByTestId(testId)
      // Open the dropdown by clicking the content area
      const content = container.querySelector('.ant-select-content') || container
      fireEvent.mouseDown(content)
      await waitFor(() => {
        const options = document.querySelectorAll('.ant-select-item-option')
        const match = Array.from(options).find(el => el.textContent === optionLabel)
        if (!match) throw new Error(`Option "${optionLabel}" not found in dropdown`)
        fireEvent.click(match)
      })
    }

    await selectOption("notes-sort-select", "Date created (newest first)")
    await waitFor(() => {
      expect(screen.getByTestId("notes-order")).toHaveTextContent(
        "Gamma note|Zulu note|Alpha note"
      )
    })
    expect(latestListPath()).toContain("sort_by=created_at")
    expect(latestListPath()).toContain("sort_order=desc")

    await selectOption("notes-sort-select", "Title (A-Z)")
    await waitFor(() => {
      expect(screen.getByTestId("notes-order")).toHaveTextContent(
        "Alpha note|Gamma note|Zulu note"
      )
    })
    expect(latestListPath()).toContain("sort_by=title")
    expect(latestListPath()).toContain("sort_order=asc")

    await selectOption("notes-sort-select", "Title (Z-A)")
    await waitFor(() => {
      expect(screen.getByTestId("notes-order")).toHaveTextContent(
        "Zulu note|Gamma note|Alpha note"
      )
    })
    expect(latestListPath()).toContain("sort_by=title")
    expect(latestListPath()).toContain("sort_order=desc")
  })

  it("persists page-size preference and rehydrates it on next load", async () => {
    const firstRender = renderPage()
    await waitFor(() => {
      expect(screen.getByTestId("notes-page-size")).toHaveTextContent("20")
    })

    fireEvent.click(screen.getByTestId("notes-set-page-size-50"))
    await waitFor(() => {
      expect(mockSetSetting).toHaveBeenCalledWith(
        expect.objectContaining({ key: "tldw:notesPageSize" }),
        50
      )
    })
    firstRender.unmount()

    renderPage()
    await waitFor(() => {
      expect(screen.getByTestId("notes-page-size")).toHaveTextContent("50")
    })
  })
})
