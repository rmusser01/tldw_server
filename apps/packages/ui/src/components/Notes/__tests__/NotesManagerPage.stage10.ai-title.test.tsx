import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
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
  default: () => <div data-testid="notes-list-panel" />
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

describe("NotesManagerPage stage 10 AI title generation", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockConfirmDanger.mockResolvedValue(true)
    mockGetSetting.mockResolvedValue(null)
    mockSetSetting.mockResolvedValue(undefined)
    mockClearSetting.mockResolvedValue(undefined)
    mockBgRequest.mockImplementation(async (request: { path?: string; method?: string; body?: any }) => {
      const path = String(request.path || "")
      const method = String(request.method || "GET").toUpperCase()
      if (path.startsWith("/api/v1/notes/?")) {
        return {
          items: [],
          pagination: { total_items: 0, total_pages: 1 }
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
      if (path === "/api/v1/notes/title/suggest" && method === "POST") {
        return { title: "AI Suggested Title" }
      }
      return {}
    })
  })

  it("generates title suggestion and applies it after confirmation", async () => {
    renderPage()
    fireEvent.change(screen.getByPlaceholderText("Write your note here... (Markdown supported)"), {
      target: { value: "This note discusses model evaluation reliability." }
    })

    fireEvent.click(screen.getByTestId("notes-generate-title-button"))

    await waitFor(() => {
      expect(mockConfirmDanger).toHaveBeenCalled()
    })
    expect(screen.getByPlaceholderText("Title")).toHaveValue("AI Suggested Title")

    const suggestCall = mockBgRequest.mock.calls.find(([request]) => {
      return (
        String(request?.path || "") === "/api/v1/notes/title/suggest" &&
        String(request?.method || "GET").toUpperCase() === "POST"
      )
    })
    expect(suggestCall).toBeTruthy()
    expect(suggestCall?.[0]?.body?.content).toContain("model evaluation")
  })

  it("keeps existing title when suggestion is rejected", async () => {
    renderPage()
    fireEvent.change(screen.getByPlaceholderText("Title"), {
      target: { value: "Manual Title" }
    })
    fireEvent.change(screen.getByPlaceholderText("Write your note here... (Markdown supported)"), {
      target: { value: "Some content for title generation." }
    })
    mockConfirmDanger.mockResolvedValueOnce(false)

    fireEvent.click(screen.getByTestId("notes-generate-title-button"))

    await waitFor(() => {
      expect(mockConfirmDanger).toHaveBeenCalled()
    })
    expect(screen.getByPlaceholderText("Title")).toHaveValue("Manual Title")
  })

  it("shows backend errors for suggestion request failures", async () => {
    mockBgRequest.mockImplementation(async (request: { path?: string; method?: string }) => {
      const path = String(request.path || "")
      const method = String(request.method || "GET").toUpperCase()
      if (path.startsWith("/api/v1/notes/?")) {
        return { items: [], pagination: { total_items: 0, total_pages: 1 } }
      }
      if (path === "/api/v1/notes/title/suggest" && method === "POST") {
        throw new Error("title service unavailable")
      }
      return {}
    })

    renderPage()
    fireEvent.change(screen.getByPlaceholderText("Write your note here... (Markdown supported)"), {
      target: { value: "Generate title please." }
    })
    fireEvent.click(screen.getByTestId("notes-generate-title-button"))

    await waitFor(() => {
      expect(mockMessageError).toHaveBeenCalledWith("title service unavailable")
    })
  })

  it("shows strategy selector and uses persisted strategy when server allows switching", async () => {
    mockGetSetting.mockImplementation(async (setting: { key?: string }) => {
      if (setting?.key === "tldw:notesTitleSuggestStrategy") return "llm_fallback"
      return null
    })
    mockBgRequest.mockImplementation(async (request: { path?: string; method?: string }) => {
      const path = String(request.path || "")
      const method = String(request.method || "GET").toUpperCase()
      if (path.startsWith("/api/v1/notes/?")) {
        return { items: [], pagination: { total_items: 0, total_pages: 1 } }
      }
      if (path === "/api/v1/admin/notes/title-settings" && method === "GET") {
        return {
          llm_enabled: true,
          default_strategy: "heuristic",
          effective_strategy: "heuristic",
          strategies: ["heuristic", "llm", "llm_fallback"]
        }
      }
      if (path === "/api/v1/notes/title/suggest" && method === "POST") {
        return { title: "AI Suggested Title" }
      }
      return {}
    })

    renderPage()
    await waitFor(() => {
      expect(screen.getByTestId("notes-title-strategy-select")).toBeTruthy()
    })
    fireEvent.change(screen.getByPlaceholderText("Write your note here... (Markdown supported)"), {
      target: { value: "strategy aware title generation content" }
    })

    fireEvent.click(screen.getByTestId("notes-generate-title-button"))

    await waitFor(() => {
      expect(mockConfirmDanger).toHaveBeenCalled()
    })
    const suggestCall = mockBgRequest.mock.calls.find(([request]) => {
      return (
        String(request?.path || "") === "/api/v1/notes/title/suggest" &&
        String(request?.method || "GET").toUpperCase() === "POST"
      )
    })
    expect(suggestCall?.[0]?.body?.title_strategy).toBe("llm_fallback")
  })

  it("hides strategy selector and falls back to heuristic when policy disables llm", async () => {
    renderPage()
    expect(screen.queryByTestId("notes-title-strategy-select")).toBeNull()
    fireEvent.change(screen.getByPlaceholderText("Write your note here... (Markdown supported)"), {
      target: { value: "heuristic fallback content" }
    })

    fireEvent.click(screen.getByTestId("notes-generate-title-button"))

    await waitFor(() => {
      expect(mockConfirmDanger).toHaveBeenCalled()
    })
    const suggestCall = mockBgRequest.mock.calls.find(([request]) => {
      return (
        String(request?.path || "") === "/api/v1/notes/title/suggest" &&
        String(request?.method || "GET").toUpperCase() === "POST"
      )
    })
    expect(suggestCall?.[0]?.body?.title_strategy).toBe("heuristic")
  })

  it("persists selected strategy changes when switching is enabled", async () => {
    mockBgRequest.mockImplementation(async (request: { path?: string; method?: string }) => {
      const path = String(request.path || "")
      const method = String(request.method || "GET").toUpperCase()
      if (path.startsWith("/api/v1/notes/?")) {
        return { items: [], pagination: { total_items: 0, total_pages: 1 } }
      }
      if (path === "/api/v1/admin/notes/title-settings" && method === "GET") {
        return {
          llm_enabled: true,
          default_strategy: "heuristic",
          effective_strategy: "heuristic",
          strategies: ["heuristic", "llm", "llm_fallback"]
        }
      }
      if (path === "/api/v1/notes/title/suggest" && method === "POST") {
        return { title: "AI Suggested Title" }
      }
      return {}
    })

    renderPage()
    const strategySelect = await screen.findByTestId("notes-title-strategy-select")
    const selectTrigger = strategySelect.closest(".ant-select") || strategySelect
    fireEvent.mouseDown(selectTrigger)
    fireEvent.click(await screen.findByText("AI-powered"))

    await waitFor(() => {
      expect(mockSetSetting).toHaveBeenCalled()
    })
    const persistedCall = mockSetSetting.mock.calls.find(([setting]) => {
      return setting?.key === "tldw:notesTitleSuggestStrategy"
    })
    expect(persistedCall?.[1]).toBe("llm")
  })
})
