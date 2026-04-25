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
  mockNavigate,
  mockConfirmDanger,
  mockGetSetting,
  mockClearSetting
} = vi.hoisted(() => ({
  mockBgRequest: vi.fn(),
  mockMessageSuccess: vi.fn(),
  mockMessageError: vi.fn(),
  mockMessageWarning: vi.fn(),
  mockNavigate: vi.fn(),
  mockConfirmDanger: vi.fn(),
  mockGetSetting: vi.fn(),
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

const createCalls = () =>
  mockBgRequest.mock.calls.filter(([request]) => {
    const path = String(request?.path || "")
    const method = String(request?.method || "GET").toUpperCase()
    return path === "/api/v1/notes/" && method === "POST"
  })

const updateCalls = () =>
  mockBgRequest.mock.calls.filter(([request]) => {
    const path = String(request?.path || "")
    const method = String(request?.method || "GET").toUpperCase()
    return path.startsWith("/api/v1/notes/11?expected_version=") && method === "PUT"
  })

describe("NotesManagerPage stage 34 keyword partial save warnings", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockConfirmDanger.mockResolvedValue(true)
    mockGetSetting.mockResolvedValue(null)
    mockClearSetting.mockResolvedValue(undefined)

    mockBgRequest.mockImplementation(async (request: { path?: string; method?: string }) => {
      const path = String(request.path || "")
      const method = String(request.method || "GET").toUpperCase()

      if (path.startsWith("/api/v1/notes/?")) {
        return {
          items: [],
          pagination: { total_items: 0, total_pages: 1 }
        }
      }

      if (path === "/api/v1/notes/" && method === "POST") {
        return {
          id: 11,
          version: 1,
          keyword_sync: {
            failed_count: 1,
            failed_keywords: ["beta"]
          }
        }
      }

      if (path === "/api/v1/notes/11" && method === "GET") {
        return {
          id: 11,
          title: "Partial keyword note",
          content: "Initial content",
          metadata: { keywords: [] },
          version: 1
        }
      }

      if (path.startsWith("/api/v1/notes/11?expected_version=") && method === "PUT") {
        return {
          id: 11,
          version: 2,
          keyword_sync: {
            failed_count: 2,
            failed_keywords: ["beta", "gamma"]
          }
        }
      }

      return {}
    })
  })

  it("shows warning toasts when note save succeeds with keyword attach failures", async () => {
    renderPage()

    const titleInput = screen.getByPlaceholderText("Title")
    const contentInput = screen.getByPlaceholderText(
      "Write your note here... (Markdown supported)"
    )

    fireEvent.change(titleInput, {
      target: { value: "Partial keyword note" }
    })
    fireEvent.change(contentInput, {
      target: { value: "Initial content" }
    })

    fireEvent.focus(contentInput)
    fireEvent.keyDown(contentInput, { key: "s", ctrlKey: true })

    await waitFor(() => {
      expect(createCalls()).toHaveLength(1)
    })
    expect(mockMessageSuccess).toHaveBeenCalledWith("Note created")
    expect(mockMessageWarning).toHaveBeenCalledWith(
      expect.stringContaining("1 tag failed to attach (beta)")
    )

    fireEvent.change(contentInput, {
      target: { value: "Updated content" }
    })

    fireEvent.focus(titleInput)
    fireEvent.keyDown(titleInput, { key: "s", metaKey: true })

    await waitFor(() => {
      expect(updateCalls()).toHaveLength(1)
    })
    expect(mockMessageSuccess).toHaveBeenCalledWith("Note updated")
    expect(mockMessageWarning).toHaveBeenCalledWith(
      expect.stringContaining("2 tags failed to attach (beta, gamma)")
    )
  })
})
