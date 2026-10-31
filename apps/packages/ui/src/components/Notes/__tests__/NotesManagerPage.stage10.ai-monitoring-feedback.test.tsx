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


// This page fixture starts with a verified owner; the interacting hook suite
// separately exercises pending discovery, identity changes, and permission loss.
vi.mock("@/hooks/useCanonicalConnectionConfig", () => {
  const config = { serverUrl: "https://notes.test", authMode: "multi-user",
    accessToken: `test.${btoa(JSON.stringify({ sub: "7" }))}.signature` }
  return { useCanonicalConnectionConfig: () => ({ config, loading: false }) }
})
vi.mock("../hooks/useNotesGraphAuthorityScope", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../hooks/useNotesGraphAuthorityScope")>()
  return { ...actual, useNotesGraphAuthorityScope: () => actual.createNotesGraphAuthorityScope("https://notes.test", 7) }
})
vi.mock("@/services/tldw/TldwAuth", () => ({
  tldwAuth: { getCurrentUser: vi.fn(async () => ({ id: 7, is_active: true })) }
}))
vi.mock("@/hooks/useCallerCapabilities", () => {
  const capabilities = { monitoringAlerts: "allowed", userId: 7,
    refreshAfterForbidden: vi.fn(async () => undefined) }
  return { useCallerCapabilities: () => capabilities }
})

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

describe("NotesManagerPage stage 10 monitoring feedback", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockConfirmDanger.mockResolvedValue(true)
    mockGetSetting.mockResolvedValue(null)
    mockSetSetting.mockResolvedValue(undefined)
    mockClearSetting.mockResolvedValue(undefined)
    mockBgRequest.mockImplementation(async (request: { path?: string; method?: string }) => {
      const path = String(request.path || "")
      const method = String(request.method || "GET").toUpperCase()
      if (path.startsWith("/api/v1/notes/?")) {
        return { items: [], pagination: { total_items: 0, total_pages: 1 } }
      }
      if (path === "/api/v1/admin/notes/title-settings" && method === "GET") {
        return {
          llm_enabled: false,
          default_strategy: "heuristic",
          effective_strategy: "heuristic",
          strategies: ["heuristic", "llm", "llm_fallback"]
        }
      }
      if (path === "/api/v1/notes/" && method === "POST") {
        return {
          id: "note-monitor-1",
          title: "Monitoring note",
          content: "monitored text",
          version: 1,
          last_modified: new Date().toISOString()
        }
      }
      if (path === "/api/v1/notes/note-monitor-1" && method === "GET") {
        return {
          id: "note-monitor-1",
          title: "Monitoring note",
          content: "monitored text",
          version: 1,
          last_modified: new Date().toISOString()
        }
      }
      if (path.startsWith("/api/v1/monitoring/alerts?") && method === "GET") {
        return {
          items: [
            {
              id: 101,
              user_id: "7",
              source: "notes.create",
              source_id: "note-monitor-1",
              rule_severity: "warning",
              created_at: new Date().toISOString()
            }
          ]
        }
      }
      return {}
    })
  })

  it("shows a non-blocking monitoring warning banner after save when an alert is detected", async () => {
    renderPage()
    fireEvent.change(screen.getByPlaceholderText("Write your note here... (Markdown supported)"), {
      target: { value: "Potentially sensitive monitored content." }
    })

    fireEvent.click(screen.getByTestId("notes-save-button"))

    await waitFor(() => {
      expect(mockMessageSuccess).toHaveBeenCalledWith("Note created")
    })
    await waitFor(() => {
      expect(screen.getByTestId("notes-monitoring-alert")).toBeInTheDocument()
    })
    expect(screen.getByTestId("notes-monitoring-alert")).toHaveTextContent(
      "Monitoring warning detected"
    )
    expect(screen.getByTestId("notes-monitoring-alert")).toHaveTextContent(
      "Review this note for sensitive material before sharing."
    )
    expect(screen.getByTestId("notes-monitoring-alert")).toHaveAttribute("aria-live", "polite")
  })

  it("keeps save flow successful when monitoring alerts endpoint is unavailable", async () => {
    mockBgRequest.mockImplementation(async (request: { path?: string; method?: string }) => {
      const path = String(request.path || "")
      const method = String(request.method || "GET").toUpperCase()
      if (path.startsWith("/api/v1/notes/?")) {
        return { items: [], pagination: { total_items: 0, total_pages: 1 } }
      }
      if (path === "/api/v1/admin/notes/title-settings" && method === "GET") {
        return {
          llm_enabled: false,
          default_strategy: "heuristic",
          effective_strategy: "heuristic",
          strategies: ["heuristic", "llm", "llm_fallback"]
        }
      }
      if (path === "/api/v1/notes/" && method === "POST") {
        return {
          id: "note-monitor-2",
          title: "Monitoring note",
          content: "monitored text",
          version: 1,
          last_modified: new Date().toISOString()
        }
      }
      if (path === "/api/v1/notes/note-monitor-2" && method === "GET") {
        return {
          id: "note-monitor-2",
          title: "Monitoring note",
          content: "monitored text",
          version: 1,
          last_modified: new Date().toISOString()
        }
      }
      if (path.startsWith("/api/v1/monitoring/alerts?") && method === "GET") {
        throw new Error("forbidden")
      }
      return {}
    })

    renderPage()
    fireEvent.change(screen.getByPlaceholderText("Write your note here... (Markdown supported)"), {
      target: { value: "Save should still work if alerts endpoint is blocked." }
    })
    fireEvent.click(screen.getByTestId("notes-save-button"))

    await waitFor(() => {
      expect(mockMessageSuccess).toHaveBeenCalledWith("Note created")
    })
    expect(screen.queryByTestId("notes-monitoring-alert")).toBeNull()
  })
})
