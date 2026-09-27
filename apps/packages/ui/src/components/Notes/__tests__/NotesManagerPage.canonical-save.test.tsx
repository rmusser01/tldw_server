import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import NotesManagerPage from "../NotesManagerPage"
import { applyRefreshRotation } from "@/services/tldw/single-user-credential"
import type { TldwConfig } from "@/services/tldw/TldwApiClient"
import { Link, MemoryRouter, Route, Routes } from "react-router-dom"
import OptionNotes from "@/routes/option-notes"
import { getFlashcardSourceMeta } from "@/components/Flashcards/utils/source-reference"
import { Modal } from "antd"

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

vi.mock("react-router-dom", async (importOriginal) => ({
  ...await importOriginal<typeof import("react-router-dom")>(),
  useNavigate: () => mockNavigate
}))

vi.mock("~/components/Layouts/Layout", () => ({
  default: ({ children }: { children: React.ReactNode }) => <>{children}</>
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

const renderSourceRoute = (initialEntry = "/flashcards") => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  const source = (id: string) => getFlashcardSourceMeta({ source_ref_type: "note", source_ref_id: id })!.href!
  const page = () => <QueryClientProvider client={queryClient}>
    <MemoryRouter initialEntries={[initialEntry]}>
      <Link to={source("11")}>Open source eleven</Link>
      <Link to={source("12")}>Open source twelve</Link>
      <Link to="/notes">Close source</Link>
      <Routes><Route path="/flashcards" element={<p>Saved cards</p>} /><Route path="/notes" element={<OptionNotes />} /></Routes>
    </MemoryRouter>
  </QueryClientProvider>
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
  afterEach(() => vi.restoreAllMocks())
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

  it("opens an owned saved Flashcard Note source through the actual route even outside the list page", async () => {
    renderSourceRoute()
    fireEvent.click(screen.getByRole("link", { name: "Open source eleven" }))
    await waitFor(() => expect(mockBgRequest.mock.calls.map(([request]) => request.path)).toContain("/api/v1/notes/11"))
    await waitFor(() => expect(screen.getByPlaceholderText("Write your note here... (Markdown supported)")).toHaveValue("Saved body"))
    const detail = mockBgRequest.mock.calls.find(([request]) => request.path === "/api/v1/notes/11" && request.method === "GET")![0]
    expect(detail.servicePromptConfig).toMatchObject({ serverUrl: "https://notes.test", expectedUserId: 7 })
    expect(detail.headers).toMatchObject({ "X-TLDW-Expected-User-ID": "7" })
    expect(createCalls()).toHaveLength(0)
  })

  it("reopens a direct Note source after reload and ignores the unrelated legacy last-note hint", async () => {
    mockGetSetting.mockResolvedValue("91")
    const first = renderSourceRoute("/notes?source_ref_id=11")
    await waitFor(() => expect(screen.getByPlaceholderText("Write your note here... (Markdown supported)")).toHaveValue("Saved body"))
    first.unmount()
    renderSourceRoute("/notes?source_ref_id=11")
    await waitFor(() => expect(screen.getByPlaceholderText("Write your note here... (Markdown supported)")).toHaveValue("Saved body"))
    expect(mockBgRequest.mock.calls.map(([request]) => request.path)).not.toContain("/api/v1/notes/91")
    expect(createCalls()).toHaveLength(0)
  })

  it("preserves the current dirty Note when opening another source is cancelled", async () => {
    renderSourceRoute("/notes?source_ref_id=11")
    const input = await screen.findByPlaceholderText("Write your note here... (Markdown supported)")
    await waitFor(() => expect(input).toHaveValue("Saved body"))
    fireEvent.change(input, { target: { value: "Keep my unsaved edit" } })
    const original = mockBgRequest.getMockImplementation()!
    mockBgRequest.mockImplementation(request => request.method === "PUT"
      ? Promise.reject(new Error("Save unavailable")) : original(request))
    const confirm = vi.spyOn(Modal, "confirm").mockImplementation(options => {
      options.onCancel?.()
      return { destroy: vi.fn(), update: vi.fn() }
    })
    fireEvent.click(screen.getByRole("link", { name: "Open source twelve" }))
    await waitFor(() => expect(confirm).toHaveBeenCalled())
    confirm.mockRestore()
    expect(input).toHaveValue("Keep my unsaved edit")
    expect(mockBgRequest.mock.calls.map(([request]) => request.path)).not.toContain("/api/v1/notes/12")
  })

  it("preserves newer typing while the source navigation waits for its confirmation save", async () => {
    renderSourceRoute("/notes?source_ref_id=11")
    const input = await screen.findByPlaceholderText("Write your note here... (Markdown supported)")
    await waitFor(() => expect(input).toHaveValue("Saved body"))
    fireEvent.change(input, { target: { value: "Saved at click time" } })
    const original = mockBgRequest.getMockImplementation()!
    const pendingSave = deferred<unknown>()
    mockBgRequest.mockImplementation(request => request.path === "/api/v1/notes/11" && request.method === "PUT" ? pendingSave.promise
      : request.path === "/api/v1/notes/12" ? Promise.resolve({ id: 12, content: "Requested source body", version: 1 }) : original(request))
    fireEvent.click(screen.getByRole("link", { name: "Open source twelve" }))
    await waitFor(() => expect(updateCalls()).toHaveLength(1))
    fireEvent.change(input, { target: { value: "New typing not included in the save" } })
    expect(updateCalls()[0][0].body.content).toBe("Saved at click time")
    await act(async () => pendingSave.resolve({ id: 11, version: 2, last_modified: "2026-09-15T12:00:00Z" }))
    expect(input).toHaveValue("New typing not included in the save")
    expect(mockBgRequest.mock.calls.map(([request]) => request.path)).not.toContain("/api/v1/notes/12")
  })

  it("opens the requested source after its confirmation successfully saves a new draft", async () => {
    const original = mockBgRequest.getMockImplementation()!
    mockBgRequest.mockImplementation(request => request.path === "/api/v1/notes/12"
      ? Promise.resolve({ id: 12, content: "Requested source body", version: 1 }) : original(request))
    renderSourceRoute("/notes")
    await act(async () => {})
    fireEvent.change(screen.getByPlaceholderText("Write your note here... (Markdown supported)"), { target: { value: "Draft before source navigation" } })
    fireEvent.click(screen.getByRole("link", { name: "Open source twelve" }))
    await waitFor(() => expect(screen.getByPlaceholderText("Write your note here... (Markdown supported)")).toHaveValue("Requested source body"))
    expect(createCalls()).toHaveLength(1)
  })

  it.each([403, 404])("shows an unavailable source after %s without leaving the previous Note or creating a draft", async (status) => {
    const original = mockBgRequest.getMockImplementation()!
    mockBgRequest.mockImplementation(request => request.path === "/api/v1/notes/12"
      ? Promise.reject(Object.assign(new Error("Source unavailable"), { status })) : original(request))
    renderSourceRoute("/notes?source_ref_id=11")
    await waitFor(() => expect(screen.getByPlaceholderText("Write your note here... (Markdown supported)")).toHaveValue("Saved body"))
    fireEvent.click(screen.getByRole("link", { name: "Open source twelve" }))
    expect(await screen.findByRole("region", { name: "Linked note unavailable" })).toHaveTextContent("may have been deleted")
    expect(screen.queryByPlaceholderText("Write your note here... (Markdown supported)")).not.toBeInTheDocument()
    expect(createCalls()).toHaveLength(0)
  })

  it("allows an explicit retry of an unavailable linked Note", async () => {
    const original = mockBgRequest.getMockImplementation()!
    let available = false
    mockBgRequest.mockImplementation(request => request.path === "/api/v1/notes/11" && !available
      ? Promise.reject(Object.assign(new Error("Unavailable"), { status: 404 })) : original(request))
    renderSourceRoute("/notes?source_ref_id=11")
    await screen.findByRole("region", { name: "Linked note unavailable" })
    available = true
    fireEvent.click(screen.getByRole("button", { name: "Retry source" }))
    await waitFor(() => expect(screen.getByPlaceholderText("Write your note here... (Markdown supported)")).toHaveValue("Saved body"))
  })

  it("ignores a late source response after the route selects a different Note", async () => {
    const original = mockBgRequest.getMockImplementation()!
    const pending = deferred<unknown>()
    mockBgRequest.mockImplementation(request => request.path === "/api/v1/notes/11" ? pending.promise
      : request.path === "/api/v1/notes/12" ? Promise.resolve({ id: 12, title: "New source", content: "New source body", version: 1 }) : original(request))
    renderSourceRoute("/notes?source_ref_id=11")
    await waitFor(() => expect(mockBgRequest.mock.calls.some(([request]) => request.path === "/api/v1/notes/11")).toBe(true))
    const staleRequest = mockBgRequest.mock.calls.find(([request]) => request.path === "/api/v1/notes/11")![0]
    fireEvent.click(screen.getByRole("link", { name: "Open source twelve" }))
    await waitFor(() => expect(screen.getByPlaceholderText("Write your note here... (Markdown supported)")).toHaveValue("New source body"))
    await act(async () => pending.resolve({ id: 11, title: "Stale source", content: "Stale source body", version: 1 }))
    expect(staleRequest.abortSignal.aborted).toBe(true)
    expect(screen.getByPlaceholderText("Write your note here... (Markdown supported)")).toHaveValue("New source body")
  })

  it("keeps a pending linked Note read across a proven same-owner credential rotation", async () => {
    const original = mockBgRequest.getMockImplementation()!
    const pending = deferred<unknown>()
    mockBgRequest.mockImplementation(request => request.path === "/api/v1/notes/11" ? pending.promise : original(request))
    const view = renderSourceRoute("/notes?source_ref_id=11")
    await waitFor(() => expect(mockBgRequest.mock.calls.some(([request]) => request.path === "/api/v1/notes/11")).toBe(true))
    const read = startHydration(view, rotation(2))
    await act(async () => read.resolve(currentConfig()))
    await act(async () => pending.resolve({ id: 11, title: "Same owner", content: "Retained source", version: 1 }))
    await waitFor(() => expect(screen.getByPlaceholderText("Write your note here... (Markdown supported)")).toHaveValue("Retained source"))
  })

  it("cancels a pending source without leaving the loading state stuck when the query is removed", async () => {
    const original = mockBgRequest.getMockImplementation()!
    const pending = deferred<unknown>()
    mockBgRequest.mockImplementation(request => request.path === "/api/v1/notes/11" ? pending.promise : original(request))
    renderSourceRoute("/notes?source_ref_id=11")
    await waitFor(() => expect(mockBgRequest.mock.calls.some(([request]) => request.path === "/api/v1/notes/11")).toBe(true))
    fireEvent.click(screen.getByRole("link", { name: "Close source" }))
    await act(async () => pending.resolve({ id: 11, content: "Cancelled source", version: 1 }))
    expect(screen.queryByText("Loading note details...")).not.toBeInTheDocument()
    expect(screen.getByPlaceholderText("Write your note here... (Markdown supported)")).not.toHaveValue("Cancelled source")
  })

  it("does not revive an old linked Note response through an A-to-B-to-A account transition", async () => {
    const original = mockBgRequest.getMockImplementation()!
    const pending = deferred<unknown>()
    let first = true
    mockBgRequest.mockImplementation(request => {
      if (request.path !== "/api/v1/notes/11") return original(request)
      if (first) { first = false; return pending.promise }
      return Promise.resolve({ id: 11, title: "Current owned source", content: `Current owner ${request.servicePromptConfig.expectedUserId}`, version: 1 })
    })
    const view = renderSourceRoute("/notes?source_ref_id=11")
    await waitFor(() => expect(mockBgRequest.mock.calls.some(([request]) => request.path === "/api/v1/notes/11")).toBe(true))
    const oldRequest = mockBgRequest.mock.calls.find(([request]) => request.path === "/api/v1/notes/11")![0]
    for (const userId of [8, 7]) {
      canonicalBoundary.userId = userId
      const next = { ...sourceConfig(), accessToken: token(userId, 4), refreshToken: `owner-${userId}` }
      canonicalBoundary.storage = { ...canonicalBoundary.storage, accessToken: next.accessToken, tldwConfig: next }
      act(() => window.dispatchEvent(new CustomEvent("tldw:config-updated", { detail: { authorityChanged: true } })))
      view.rerenderPage()
      await waitFor(() => expect(screen.getByPlaceholderText("Write your note here... (Markdown supported)")).toHaveValue(`Current owner ${userId}`))
    }
    await act(async () => pending.resolve({ id: 11, title: "Stale owner", content: "Old private source", version: 1 }))
    expect(oldRequest.abortSignal.aborted).toBe(true)
    expect(screen.getByPlaceholderText("Write your note here... (Markdown supported)")).toHaveValue("Current owner 7")
    expect(createCalls()).toHaveLength(0)
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
