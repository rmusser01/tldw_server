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
} = vi.hoisted(() => {
  return {
    mockBgRequest: vi.fn(),
    mockMessageSuccess: vi.fn(),
    mockMessageError: vi.fn(),
    mockMessageWarning: vi.fn(),
    mockNavigate: vi.fn(),
    mockConfirmDanger: vi.fn(),
    mockGetSetting: vi.fn(),
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

const setupVersionDriftMock = () => {
  let detailCalls = 0
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
      return { id: "note-a", version: 1, last_modified: "2026-02-18T11:00:00.000Z" }
    }
    if (path.startsWith("/api/v1/notes/note-a/neighbors")) {
      return {
        nodes: [{ id: "note-a", type: "note", label: "Drift note" }],
        edges: []
      }
    }
    if (path === "/api/v1/notes/note-a" && method === "GET") {
      detailCalls += 1
      if (detailCalls === 1) {
        return {
          id: "note-a",
          title: "Drift note",
          content: "local content",
          metadata: { keywords: [] },
          version: 1,
          last_modified: "2026-02-18T11:00:00.000Z"
        }
      }
      return {
        id: "note-a",
        title: "Drift note",
        content: "remote content",
        metadata: { keywords: [] },
        version: 2,
        last_modified: "2026-02-18T11:10:00.000Z"
      }
    }
    if (path.startsWith("/api/v1/notes/note-a?expected_version=") && method === "PUT") {
      return {
        id: "note-a",
        title: "Drift note",
        content: "saved content",
        metadata: { keywords: [] },
        version: 3,
        last_modified: "2026-02-18T11:20:00.000Z"
      }
    }
    return {}
  })
}

describe("NotesManagerPage stage 9 stale-version warning", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockConfirmDanger.mockResolvedValue(true)
    mockGetSetting.mockResolvedValue(null)
    mockClearSetting.mockResolvedValue(undefined)
  })

  it("warns before save when server version is newer and allows cancel", async () => {
    setupVersionDriftMock()
    renderPage()

    fireEvent.change(screen.getByPlaceholderText("Title"), {
      target: { value: "Drift note" }
    })
    fireEvent.change(screen.getByPlaceholderText("Write your note here... (Markdown supported)"), {
      target: { value: "local content" }
    })
    fireEvent.click(screen.getByTestId("notes-save-button"))

    expect(await screen.findByTestId("notes-editor-revision-meta")).toHaveTextContent("Version 1")
    expect(await screen.findByTestId("notes-stale-version-warning")).toHaveTextContent(
      "This note was updated elsewhere. Reload to see the latest version."
    )

    fireEvent.change(screen.getByPlaceholderText("Write your note here... (Markdown supported)"), {
      target: { value: "local edits pending" }
    })
    mockConfirmDanger.mockResolvedValueOnce(false)
    fireEvent.click(screen.getByTestId("notes-save-button"))

    await waitFor(() => {
      expect(mockConfirmDanger).toHaveBeenCalled()
    })

    const putCalled = mockBgRequest.mock.calls.some(([request]) => {
      const path = String(request?.path || "")
      const method = String(request?.method || "GET").toUpperCase()
      return path.startsWith("/api/v1/notes/note-a?expected_version=") && method === "PUT"
    })
    expect(putCalled).toBe(false)
  })

  it("can reload latest note from stale-version warning", async () => {
    setupVersionDriftMock()
    renderPage()

    fireEvent.change(screen.getByPlaceholderText("Title"), {
      target: { value: "Drift note" }
    })
    fireEvent.change(screen.getByPlaceholderText("Write your note here... (Markdown supported)"), {
      target: { value: "local content" }
    })
    fireEvent.click(screen.getByTestId("notes-save-button"))

    expect(await screen.findByTestId("notes-stale-version-warning")).toBeInTheDocument()
    fireEvent.click(screen.getByTestId("notes-stale-version-reload"))

    await waitFor(() => {
      expect(screen.queryByTestId("notes-stale-version-warning")).not.toBeInTheDocument()
    })
    expect(screen.getByPlaceholderText("Write your note here... (Markdown supported)")).toHaveValue(
      "remote content"
    )
    expect(screen.getByTestId("notes-editor-revision-meta")).toHaveTextContent("Version 2")
  })
})
