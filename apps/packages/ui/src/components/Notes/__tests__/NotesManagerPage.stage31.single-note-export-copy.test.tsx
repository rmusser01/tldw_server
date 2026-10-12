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
}));

vi.mock("@/hooks/useCanonicalConnectionConfig", () => ({
  useCanonicalConnectionConfig: () => ({
    config: {
      serverUrl: "https://notes.test",
      authMode: "single-user",
      apiKey: "synthetic",
    },
    loading: false,
    authorityLoading: false,
  }),
}));
vi.mock("@/components/Notes/hooks/useNotesGraphAuthorityScope", () => ({
  useNotesGraphAuthorityScope: () => "synthetic-authority",
  createNotesGraphAuthorityScope: () => "synthetic-authority",
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
  default: () => <div data-testid="notes-list-panel" />
}))

vi.mock("@/components/Notes/NotesEditorHeader", () => ({
  default: ({ onCopy, onExport }: any) => (
    <div>
      <button data-testid="copy-content-action" onClick={() => onCopy("content")}>
        Copy content
      </button>
      <button data-testid="copy-markdown-action" onClick={() => onCopy("markdown")}>
        Copy markdown
      </button>
      <button data-testid="export-md-action" onClick={() => onExport("md")}>
        Export md
      </button>
      <button data-testid="export-json-action" onClick={() => onExport("json")}>
        Export json
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

describe("NotesManagerPage stage 31 single-note export/copy parity", () => {
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
          strategies: ["heuristic"]
        }
      }
      return {}
    })
  })

  it("routes copy modes to clipboard payloads and routes single-note export formats", async () => {
    const clipboardWriteText = vi.fn(async () => undefined)
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: {
        writeText: clipboardWriteText
      }
    })

    const exportedBlobs: Blob[] = []
    Object.defineProperty(URL, "createObjectURL", {
      configurable: true,
      writable: true,
      value: vi.fn((blob: Blob) => {
        exportedBlobs.push(blob)
        return `blob:notes-${exportedBlobs.length}`
      })
    })
    Object.defineProperty(URL, "revokeObjectURL", {
      configurable: true,
      writable: true,
      value: vi.fn()
    })
    vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => undefined)

    renderPage()

    fireEvent.change(screen.getByPlaceholderText("Title"), {
      target: { value: "Export parity note" }
    })
    fireEvent.change(screen.getByPlaceholderText("Write your note here... (Markdown supported)"), {
      target: { value: "Single note body" }
    })

    fireEvent.click(screen.getByTestId("copy-content-action"))
    fireEvent.click(screen.getByTestId("copy-markdown-action"))

    await waitFor(() => {
      expect(clipboardWriteText).toHaveBeenCalledTimes(2)
    })
    const clipboardCalls = clipboardWriteText.mock.calls as unknown as Array<[string]>
    expect(clipboardCalls[0]?.[0]).toBe("Single note body")
    expect(String(clipboardCalls[1]?.[0])).toContain("# Export parity note")
    expect(String(clipboardCalls[1]?.[0])).toContain("Single note body")

    fireEvent.click(screen.getByTestId("export-md-action"))
    fireEvent.click(screen.getByTestId("export-json-action"))

    await waitFor(() => {
      expect(exportedBlobs.length).toBe(2)
    })
    expect(exportedBlobs[0].type).toContain("text/markdown")
    expect(exportedBlobs[1].type).toContain("application/json")
    expect(await exportedBlobs[0].text()).toContain("# Export parity note")
    expect(await exportedBlobs[1].text()).toContain('"title": "Export parity note"')
  })
})
