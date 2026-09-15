import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import NotesManagerPage from "../NotesManagerPage"
import { applyRefreshRotation } from "@/services/tldw/single-user-credential"
import type { TldwConfig } from "@/services/tldw/TldwApiClient"

const {
  mockBgRequest,
  mockMessageSuccess,
  mockMessageError,
  mockMessageWarning,
  mockMessageInfo,
  mockNavigate,
  mockConfirmDanger,
  mockGetSetting,
  mockClearSetting,
  canonicalBoundary
} = vi.hoisted(() => ({
  mockBgRequest: vi.fn(),
  mockMessageSuccess: vi.fn(),
  mockMessageError: vi.fn(),
  mockMessageWarning: vi.fn(),
  mockMessageInfo: vi.fn(),
  mockNavigate: vi.fn(),
  mockConfirmDanger: vi.fn(),
  mockGetSetting: vi.fn(),
  mockClearSetting: vi.fn(),
  canonicalBoundary: {
    storage: {} as Record<string, unknown>,
    getConfig: vi.fn(),
    userId: 7
  }
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
      if (typeof defaultValueOrOptions === "string")
        return defaultValueOrOptions
      if (defaultValueOrOptions?.defaultValue)
        return defaultValueOrOptions.defaultValue
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
  useStoreMessageOption: (
    selector: (state: Record<string, unknown>) => unknown
  ) =>
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
  const actual =
    await importOriginal<typeof import("@/services/settings/registry")>()
  return {
    ...actual,
    getSetting: mockGetSetting,
    clearSetting: mockClearSetting
  }
})

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: vi.fn(async () => undefined),
    getConfig: (...args: unknown[]) => canonicalBoundary.getConfig(...args),
    getChat: vi.fn(async () => null),
    listChatMessages: vi.fn(async () => []),
    getCharacter: vi.fn(async () => null)
  }
}))

vi.mock("@/components/Notes/NotesListPanel", () => ({
  default: ({ onSelectNote }: { onSelectNote: (id: string) => void }) => (
    <button data-testid="notes-list-panel" onClick={() => onSelectNote("11")}>
      Open saved note
    </button>
  )
}))

// The canonical config, Notes owner, and editor hooks are real. Only I/O is held.
vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (key: string, fallback: unknown) => [
    canonicalBoundary.storage[key] ?? fallback,
    vi.fn()
  ]
}))
vi.mock("@/services/tldw/TldwAuth", () => ({
  tldwAuth: {
    getCurrentUser: vi.fn(async () => ({
      id: canonicalBoundary.userId,
      is_active: true
    }))
  }
}))

const renderPage = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false }
    }
  })
  const page = () => (
    <QueryClientProvider client={queryClient}>
      <NotesManagerPage />
    </QueryClientProvider>
  )
  const view = render(page())
  return { ...view, rerenderPage: () => view.rerender(page()) }
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
  mockBgRequest.mockImplementation(
    async (request: { path?: string; method?: string }) => {
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
    }
  )
}

const token = (id: number, revision: number) =>
  `test.${btoa(JSON.stringify({ sub: String(id), revision }))}.signature`
const sourceConfig = (): TldwConfig => ({
  serverUrl: "https://notes.test",
  authMode: "multi-user",
  accessToken: token(7, 1),
  refreshToken: "refresh-one"
})
const rotation = (revision: number, overrides = {}) => ({
  version: 1,
  serverUrl: "https://notes.test",
  authMode: "multi-user",
  sourceAccessToken: token(7, 1),
  sourceRefreshToken: "refresh-one",
  accessToken: token(7, revision),
  refreshToken: `refresh-${revision}`,
  ...overrides
})
const currentConfig = () =>
  applyRefreshRotation(
    canonicalBoundary.storage.tldwConfig as TldwConfig,
    canonicalBoundary.storage.tldwRefreshRotation
  )
const deferred = <T,>() => {
  let resolve!: (value: T) => void
  let reject!: (error: Error) => void
  const promise = new Promise<T>((yes, no) => {
    resolve = yes
    reject = no
  })
  return { promise, resolve, reject }
}
const beginCreate = async () => {
  const original = mockBgRequest.getMockImplementation()!
  const create = deferred<unknown>()
  mockBgRequest.mockImplementation((request) =>
    request.method === "POST" ? create.promise : original(request)
  )
  const view = renderPage()
  await act(async () => {})
  fireEvent.change(
    screen.getByPlaceholderText("Write your note here... (Markdown supported)"),
    { target: { value: "Own draft" } }
  )
  fireEvent.click(screen.getByTestId("notes-save-button"))
  await waitFor(() => expect(createCalls()).toHaveLength(1))
  return { view, create }
}
const startHydration = (
  view: ReturnType<typeof renderPage>,
  record: unknown
) => {
  const configRead = deferred<TldwConfig>()
  canonicalBoundary.getConfig.mockReturnValueOnce(configRead.promise)
  canonicalBoundary.storage = {
    ...canonicalBoundary.storage,
    tldwRefreshRotation: record
  }
  act(() =>
    window.dispatchEvent(
      new CustomEvent("tldw:config-updated", {
        detail: { authorityChanged: false }
      })
    )
  )
  view.rerenderPage()
  return configRead
}
const acknowledgeCreate = async (
  create: ReturnType<typeof deferred<unknown>>
) => {
  await act(async () =>
    create.resolve({
      id: 11,
      version: 1,
      last_modified: "2026-09-15T12:00:00Z"
    })
  )
}

describe("Notes pending saves through actual canonical storage hydration", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    canonicalBoundary.userId = 7
    canonicalBoundary.storage = {
      serverUrl: "https://notes.test",
      authMode: "multi-user",
      accessToken: token(7, 1),
      tldwConfig: sourceConfig()
    }
    canonicalBoundary.getConfig
      .mockReset()
      .mockImplementation(async () => currentConfig())
    mockConfirmDanger.mockResolvedValue(true)
    mockGetSetting.mockResolvedValue(null)
    mockClearSetting.mockResolvedValue(undefined)
    seedBaseNotesMock()
  })

  it("retains a pending create across repeated lineage-valid rotations and follows with a versioned PUT", async () => {
    const { view, create } = await beginCreate()
    for (const revision of [2, 3]) {
      const read = startHydration(view, rotation(revision))
      await act(async () => {})
      await act(async () => read.resolve(currentConfig()))
    }
    await acknowledgeCreate(create)
    expect(screen.getByTestId("notes-save-status")).toHaveAttribute(
      "data-state",
      "saved"
    )
    fireEvent.change(
      screen.getByPlaceholderText(
        "Write your note here... (Markdown supported)"
      ),
      { target: { value: "Updated own draft" } }
    )
    fireEvent.click(screen.getByTestId("notes-save-button"))
    await waitFor(() => expect(updateCalls()).toHaveLength(1))
    expect(createCalls()).toHaveLength(1)
    expect(updateCalls()[0][0].headers["expected-version"]).toBe("1")
  })

  it("retains an acknowledgement arriving while a proved canonical refresh is still loading", async () => {
    const { view, create } = await beginCreate()
    const read = startHydration(view, rotation(2))
    await act(async () => {})
    await acknowledgeCreate(create)
    expect(screen.getByTestId("notes-save-status")).toHaveAttribute(
      "data-state",
      "saved"
    )
    await act(async () => read.resolve(currentConfig()))
    expect(screen.getByTestId("notes-editor-revision-meta")).toHaveTextContent(
      "Version 1"
    )
    expect(createCalls()).toHaveLength(1)
  })

  it("preserves a dirty versioned draft throughout proved canonical loading", async () => {
    const view = renderPage()
    await act(async () => {})
    fireEvent.click(screen.getByTestId("notes-list-panel"))
    await waitFor(() =>
      expect(
        screen.getByTestId("notes-editor-revision-meta")
      ).toHaveTextContent("Version 1")
    )
    const editor = screen.getByPlaceholderText(
      "Write your note here... (Markdown supported)"
    )
    fireEvent.change(editor, { target: { value: "Unsaved local changes" } })
    const read = startHydration(view, rotation(2))
    await act(async () => {})
    expect(editor).toHaveValue("Unsaved local changes")
    expect(screen.getByTestId("notes-save-status")).not.toHaveAttribute(
      "data-state",
      "saved"
    )
    await act(async () => read.resolve(currentConfig()))
    expect(editor).toHaveValue("Unsaved local changes")
    expect(screen.getByTestId("notes-editor-revision-meta")).toHaveTextContent(
      "Version 1"
    )
    expect(createCalls()).toHaveLength(0)
    expect(updateCalls()).toHaveLength(0)
  })

  it.each([
    "malformed",
    "source-access",
    "source-refresh",
    "unknown-token",
    "record-target",
    "account",
    "server",
    "api-key",
    "raw-token-only",
    "org",
    "auth-source"
  ])(
    "masks an unproved or changed %s boundary during the actual deferred read",
    async (boundary) => {
      if (boundary === "api-key") {
        canonicalBoundary.storage = {
          serverUrl: "https://notes.test",
          authMode: "single-user",
          apiKey: "old-key",
          tldwConfig: {
            serverUrl: "https://notes.test",
            authMode: "single-user",
            apiKey: "old-key"
          }
        }
      }
      const { view, create } = await beginCreate()
      let record: unknown = rotation(2)
      if (boundary === "malformed") record = token(7, 2)
      if (boundary === "source-access")
        record = rotation(2, { sourceAccessToken: token(7, 99) })
      if (boundary === "source-refresh")
        record = rotation(2, { sourceRefreshToken: "other-session" })
      if (boundary === "unknown-token")
        record = rotation(2, { accessToken: "unverifiable-token" })
      if (boundary === "record-target")
        record = rotation(2, { serverUrl: "https://other.test" })
      if (boundary === "account")
        record = rotation(2, { accessToken: token(8, 2) })
      if (boundary === "server")
        canonicalBoundary.storage = {
          ...canonicalBoundary.storage,
          serverUrl: "https://other.test",
          tldwConfig: { ...sourceConfig(), serverUrl: "https://other.test" }
        }
      if (boundary === "api-key")
        canonicalBoundary.storage = {
          ...canonicalBoundary.storage,
          apiKey: "new-key",
          tldwConfig: {
            serverUrl: "https://notes.test",
            authMode: "single-user",
            apiKey: "new-key"
          }
        }
      if (boundary === "raw-token-only") {
        canonicalBoundary.storage = {
          ...canonicalBoundary.storage,
          tldwConfig: { ...sourceConfig(), accessToken: token(7, 99) }
        }
        record = undefined
      }
      if (boundary === "org")
        canonicalBoundary.storage = {
          ...canonicalBoundary.storage,
          tldwConfig: { ...sourceConfig(), orgId: 9 }
        }
      if (boundary === "auth-source")
        canonicalBoundary.storage = {
          ...canonicalBoundary.storage,
          tldwConfig: { ...sourceConfig(), authSource: "cookie-session" }
        }
      const read = startHydration(view, record)
      await act(async () => {})
      await acknowledgeCreate(create)
      expect(screen.getByTestId("notes-save-status")).not.toHaveAttribute(
        "data-state",
        "saved"
      )
      expect(mockMessageSuccess).not.toHaveBeenCalledWith("Note created")
      await act(async () => read.resolve(currentConfig()))
      expect(
        screen.getByTestId("notes-editor-revision-meta")
      ).not.toHaveTextContent("Version 1")
    }
  )

  it("withdraws continuity when the canonical read fails, retaining existing fallback behavior", async () => {
    const { view, create } = await beginCreate()
    const read = startHydration(view, rotation(2))
    await act(async () => {})
    await act(async () =>
      read.reject(new Error("Canonical configuration unavailable"))
    )
    await acknowledgeCreate(create)
    expect(mockMessageSuccess).not.toHaveBeenCalledWith("Note created")
    expect(
      screen.getByTestId("notes-editor-revision-meta")
    ).not.toHaveTextContent("Version 1")
  })

  it.each(["session", "logout"])(
    "retires a pending save on explicit %s invalidation during proved hydration",
    async (kind) => {
      const { view, create } = await beginCreate()
      const read = startHydration(view, rotation(2))
      await act(async () => {})
      act(() =>
        window.dispatchEvent(
          kind === "session"
            ? new CustomEvent("tldw:config-updated", {
                detail: {
                  authorityChanged: false,
                  refreshSessionInvalidated: true
                }
              })
            : new CustomEvent("tldw:auth-principal-changed", {
                detail: { kind: "logout" }
              })
        )
      )
      await act(async () => read.resolve(currentConfig()))
      await acknowledgeCreate(create)
      expect(mockMessageSuccess).not.toHaveBeenCalledWith("Note created")
      expect(
        screen.getByTestId("notes-editor-revision-meta")
      ).not.toHaveTextContent("Version 1")
    }
  )

  it("does not revive a save or stale canonical response after A-to-B-to-A storage changes", async () => {
    const { view, create } = await beginCreate()
    const configB = {
      ...sourceConfig(),
      accessToken: token(8, 1),
      refreshToken: "refresh-B"
    }
    canonicalBoundary.storage = {
      ...canonicalBoundary.storage,
      tldwConfig: configB
    }
    const oldRead = startHydration(view, undefined)
    await act(async () => {})
    canonicalBoundary.storage = {
      ...canonicalBoundary.storage,
      tldwConfig: sourceConfig()
    }
    const currentRead = startHydration(view, rotation(2))
    await act(async () => currentRead.resolve(currentConfig()))
    await act(async () => oldRead.resolve(configB))
    await acknowledgeCreate(create)
    expect(mockMessageSuccess).not.toHaveBeenCalledWith("Note created")
    expect(
      screen.getByTestId("notes-editor-revision-meta")
    ).not.toHaveTextContent("Version 1")
  })
})
