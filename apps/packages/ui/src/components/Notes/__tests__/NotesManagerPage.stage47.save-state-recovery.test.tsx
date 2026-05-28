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
    return path === "/api/v1/notes/11" && method === "PUT"
  })

const seedBaseNotesMock = () => {
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
        last_modified: "2026-02-18T11:00:00.000Z"
      }
    }

    if (path === "/api/v1/notes/11" && method === "GET") {
      return {
        id: 11,
        title: "Saved note",
        content: "Saved body",
        metadata: { keywords: [] },
        version: 1,
        last_modified: "2026-02-18T11:00:00.000Z"
      }
    }

    if (path === "/api/v1/notes/11" && method === "PUT") {
      return {
        id: 11,
        version: 2,
        last_modified: "2026-02-18T11:05:00.000Z"
      }
    }

    return {}
  })
}

describe("NotesManagerPage stage 47 save state and recovery", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockConfirmDanger.mockResolvedValue(true)
    mockGetSetting.mockResolvedValue(null)
    mockClearSetting.mockResolvedValue(undefined)
    seedBaseNotesMock()
  })

  it("announces dirty and saving states and blocks duplicate saves while one is pending", async () => {
    let resolveCreate: (value: unknown) => void = () => undefined
    const pendingCreate = new Promise((resolve) => {
      resolveCreate = resolve
    })

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
        return pendingCreate
      }

      if (path === "/api/v1/notes/11" && method === "GET") {
        return {
          id: 11,
          title: "Pending save",
          content: "Body while save is pending",
          metadata: { keywords: [] },
          version: 1,
          last_modified: "2026-02-18T11:00:00.000Z"
        }
      }

      return {}
    })

    renderPage()

    fireEvent.change(screen.getByPlaceholderText("Title"), {
      target: { value: "Pending save" }
    })
    fireEvent.change(screen.getByPlaceholderText("Write your note here... (Markdown supported)"), {
      target: { value: "Body while save is pending" }
    })

    expect(screen.getByTestId("notes-save-feedback")).toHaveTextContent("Unsaved changes")

    fireEvent.click(screen.getByTestId("notes-save-button"))

    await waitFor(() => {
      expect(screen.getByTestId("notes-save-feedback")).toHaveTextContent("Saving...")
    })
    expect(screen.getByTestId("notes-save-button")).toBeDisabled()

    fireEvent.click(screen.getByTestId("notes-save-button"))
    expect(createCalls()).toHaveLength(1)

    resolveCreate({
      id: 11,
      version: 1,
      last_modified: "2026-02-18T11:00:00.000Z"
    })

    await waitFor(() => {
      expect(screen.getByTestId("notes-save-feedback")).toHaveTextContent("All changes saved")
    })
  })

  it("keeps failed-save drafts editable and exposes a retry action", async () => {
    let createAttempt = 0
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
        createAttempt += 1
        if (createAttempt === 1) {
          throw new Error("Network unavailable")
        }
        return {
          id: 11,
          version: 1,
          last_modified: "2026-02-18T11:00:00.000Z"
        }
      }

      if (path === "/api/v1/notes/11" && method === "GET") {
        return {
          id: 11,
          title: "Recoverable draft",
          content: "Draft should stay visible",
          metadata: { keywords: [] },
          version: 1,
          last_modified: "2026-02-18T11:00:00.000Z"
        }
      }

      return {}
    })

    renderPage()

    fireEvent.change(screen.getByPlaceholderText("Title"), {
      target: { value: "Recoverable draft" }
    })
    fireEvent.change(screen.getByPlaceholderText("Write your note here... (Markdown supported)"), {
      target: { value: "Draft should stay visible" }
    })
    fireEvent.click(screen.getByTestId("notes-save-button"))

    expect(await screen.findByTestId("notes-save-retry")).toBeInTheDocument()
    expect(screen.getByPlaceholderText("Write your note here... (Markdown supported)")).toHaveValue(
      "Draft should stay visible"
    )
    expect(screen.getByTestId("notes-save-feedback")).toHaveTextContent(
      "Could not save — check your connection and try again."
    )

    fireEvent.click(screen.getByTestId("notes-save-retry"))

    await waitFor(() => {
      expect(createCalls()).toHaveLength(2)
    })
    await waitFor(() => {
      expect(screen.getByTestId("notes-save-feedback")).toHaveTextContent("All changes saved")
    })
  })

  it("shows a persistent conflict recovery action without replacing the local draft", async () => {
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
          last_modified: "2026-02-18T11:00:00.000Z"
        }
      }

      if (path === "/api/v1/notes/11" && method === "GET") {
        return {
          id: 11,
          title: "Conflict note",
          content: "Server body",
          metadata: { keywords: [] },
          version: 1,
          last_modified: "2026-02-18T11:00:00.000Z"
        }
      }

      if (path === "/api/v1/notes/11" && method === "PUT") {
        throw { status: 409, message: "Version conflict" }
      }

      return {}
    })

    renderPage()

    fireEvent.change(screen.getByPlaceholderText("Title"), {
      target: { value: "Conflict note" }
    })
    fireEvent.change(screen.getByPlaceholderText("Write your note here... (Markdown supported)"), {
      target: { value: "Server body" }
    })
    fireEvent.click(screen.getByTestId("notes-save-button"))

    await waitFor(() => {
      expect(createCalls()).toHaveLength(1)
    })

    fireEvent.change(screen.getByPlaceholderText("Write your note here... (Markdown supported)"), {
      target: { value: "Local conflicting edit" }
    })
    fireEvent.click(screen.getByTestId("notes-save-button"))

    expect(await screen.findByTestId("notes-save-conflict-reload")).toBeInTheDocument()
    expect(screen.getByPlaceholderText("Write your note here... (Markdown supported)")).toHaveValue(
      "Local conflicting edit"
    )
    expect(updateCalls()).toHaveLength(1)
  })
})
