/**
 * XP-05 (#3101, G18): the open note has an address, /notes?note=<id>.
 * Opening notes records them in browser history so Back and Forward move
 * between them, links open the note they name, /notes without a note
 * reopens the last note opened, and the page title names the note.
 */
import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { MemoryRouter, Route, Routes, useLocation, useNavigate } from "react-router-dom"
import OptionNotes from "@/routes/option-notes"

// The web build's Next shim keys each location by router.asPath (pathname +
// search), so two visits to the same address share a key. Tests opt in.
const routerKeys = vi.hoisted(() => ({ shim: false }))
vi.mock("react-router-dom", async () => {
  const actual = await vi.importActual<typeof import("react-router-dom")>("react-router-dom")
  return {
    ...actual,
    useLocation: () => {
      const location = actual.useLocation()
      const shim = routerKeys.shim
      return React.useMemo(
        () => (shim ? { ...location, key: `${location.pathname}${location.search}` } : location),
        [location, shim]
      )
    }
  }
})

const {
  mockBgRequest,
  mockConfirmDanger,
  mockGetSetting,
  mockClearSetting,
  canonicalBoundary
} = vi.hoisted(() => ({
  mockBgRequest: vi.fn(),
  mockConfirmDanger: vi.fn(),
  mockGetSetting: vi.fn(),
  mockClearSetting: vi.fn(),
  canonicalBoundary: {
    storage: {} as Record<string, unknown>,
    getConfig: vi.fn()
  }
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      defaultValueOrOptions?: string | { defaultValue?: string; [key: string]: unknown }
    ) => {
      if (typeof defaultValueOrOptions === "string") return defaultValueOrOptions
      if (defaultValueOrOptions?.defaultValue) return defaultValueOrOptions.defaultValue
      return key
    }
  })
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
  useServerCapabilities: () => ({ capabilities: { hasNotes: true }, loading: false })
}))

vi.mock("@/components/Common/confirm-danger", () => ({
  useConfirmDanger: () => mockConfirmDanger
}))

vi.mock("@/hooks/useAntdMessage", () => ({
  useAntdMessage: () => ({ success: vi.fn(), error: vi.fn(), warning: vi.fn(), info: vi.fn() })
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
  return { ...actual, getSetting: mockGetSetting, clearSetting: mockClearSetting }
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
    <>
      <button type="button" data-testid="notes-list-panel" onClick={() => onSelectNote("11")}>
        Open Alpha
      </button>
      <button type="button" onClick={() => onSelectNote("12")}>Open Beta</button>
    </>
  )
}))

// The canonical config, Notes owner and editor hooks are real; only I/O is held.
vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (key: string, fallback: unknown) => [canonicalBoundary.storage[key] ?? fallback, vi.fn()]
}))
vi.mock("@/services/tldw/TldwAuth", () => ({
  tldwAuth: { getCurrentUser: vi.fn(async () => ({ id: 7, is_active: true })) }
}))

const token = (id: number, revision: number) =>
  `test.${btoa(JSON.stringify({ sub: String(id), revision }))}.signature`

const NOTES: Record<string, { title: string; content: string }> = {
  "11": { title: "Alpha note", content: "Alpha body" },
  "12": { title: "Beta note", content: "Beta body" }
}

/** Shows the router's address and offers the browser's Back and Forward. */
const BrowserChrome = () => {
  const location = useLocation()
  const navigate = useNavigate()
  return (
    <>
      <output data-testid="harness-address">{`${location.pathname}${location.search}`}</output>
      <button type="button" onClick={() => navigate(-1)}>Browser Back</button>
      <button type="button" onClick={() => navigate(1)}>Browser Forward</button>
      <button type="button" onClick={() => navigate("/notes?new=1")}>Palette New note</button>
      <button
        type="button"
        onClick={() => {
          const params = new URLSearchParams(location.search)
          params.set("note", "12")
          navigate(`/notes?${params.toString()}`)
        }}>
        Open Beta via address
      </button>
    </>
  )
}

const renderNotesRoute = (initialEntries: string[]) => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } }
  })
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={initialEntries} initialIndex={initialEntries.length - 1}>
        <BrowserChrome />
        <Routes>
          <Route path="/chat" element={<p>Chat page</p>} />
          <Route path="/notes" element={<OptionNotes />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>
  )
}

const address = () => screen.getByTestId("harness-address").textContent
/** Wait until the Notes owner is known and the list has been requested. */
const waitForNotesReady = async () => {
  await waitFor(() =>
    expect(
      mockBgRequest.mock.calls.some(([request]) => String(request?.path).startsWith("/api/v1/notes/?"))
    ).toBe(true)
  )
  await act(async () => {})
}
const editor = () =>
  screen.getByPlaceholderText("Write your note here... (Markdown supported)") as HTMLTextAreaElement

describe("Notes addresses (XP-05)", { timeout: 60_000 }, () => {
  beforeEach(() => {
    vi.clearAllMocks()
    localStorage.clear()
    document.title = "Options | tldw"
    canonicalBoundary.storage = {
      serverUrl: "https://notes.test",
      authMode: "multi-user",
      accessToken: token(7, 1),
      tldwConfig: {
        serverUrl: "https://notes.test",
        authMode: "multi-user",
        accessToken: token(7, 1),
        refreshToken: "refresh-one"
      }
    }
    canonicalBoundary.getConfig
      .mockReset()
      .mockImplementation(async () => canonicalBoundary.storage.tldwConfig)
    mockConfirmDanger.mockResolvedValue(true)
    mockGetSetting.mockResolvedValue(null)
    mockClearSetting.mockResolvedValue(undefined)
    mockBgRequest.mockImplementation(async (request: { path?: string; method?: string }) => {
      const path = String(request.path || "")
      const method = String(request.method || "GET").toUpperCase()
      if (path.startsWith("/api/v1/notes/?")) {
        return { items: [], pagination: { total_items: 0, total_pages: 1 } }
      }
      const detail = path.match(/^\/api\/v1\/notes\/(\d+)$/)
      if (detail && method === "GET" && NOTES[detail[1]]) {
        return {
          id: Number(detail[1]),
          ...NOTES[detail[1]],
          metadata: { keywords: [] },
          version: 1,
          last_modified: "2026-10-19T10:00:00.000Z"
        }
      }
      return {}
    })
  })

  it("records each opened note in the address so Back and Forward move between them", async () => {
    renderNotesRoute(["/chat", "/notes"])
    await waitForNotesReady()

    fireEvent.click(screen.getByRole("button", { name: "Open Alpha" }))
    await waitFor(() => expect(editor()).toHaveValue("Alpha body"))
    await waitFor(() => expect(address()).toBe("/notes?note=11"))

    fireEvent.click(screen.getByRole("button", { name: "Open Beta" }))
    await waitFor(() => expect(editor()).toHaveValue("Beta body"))
    await waitFor(() => expect(address()).toBe("/notes?note=12"))

    fireEvent.click(screen.getByRole("button", { name: "Browser Back" }))
    await waitFor(() => {
      expect(address()).toBe("/notes?note=11")
      expect(editor()).toHaveValue("Alpha body")
    })

    fireEvent.click(screen.getByRole("button", { name: "Browser Forward" }))
    await waitFor(() => {
      expect(address()).toBe("/notes?note=12")
      expect(editor()).toHaveValue("Beta body")
    })

    // The first note replaced the bare /notes entry, so Back from it leaves Notes.
    fireEvent.click(screen.getByRole("button", { name: "Browser Back" }))
    await waitFor(() => expect(address()).toBe("/notes?note=11"))
    fireEvent.click(screen.getByRole("button", { name: "Browser Back" }))
    await waitFor(() => expect(address()).toBe("/chat"))
  })

  it("opens the note a /notes?note=<id> link names", async () => {
    renderNotesRoute(["/notes?note=12"])

    await waitFor(() => expect(editor()).toHaveValue("Beta body"))
    expect(address()).toBe("/notes?note=12")
  })

  it("still opens source_ref_id links, and names the note with ?note= afterwards", async () => {
    renderNotesRoute(["/notes?source_ref_id=11"])

    await waitFor(() => expect(editor()).toHaveValue("Alpha body"))
    await waitFor(() => expect(address()).toBe("/notes?note=11"))
  })

  it("reopens the last note opened when /notes names no note", async () => {
    mockGetSetting.mockImplementation(async (setting: { key?: string }) =>
      String(setting?.key || "").startsWith("tldw:notesRecentOpened:")
        ? [{ id: "12", title: "Beta note" }, { id: "11", title: "Alpha note" }]
        : null
    )

    renderNotesRoute(["/notes"])

    await waitFor(() => expect(editor()).toHaveValue("Beta body"))
    await waitFor(() => expect(address()).toBe("/notes?note=12"))
  })

  it("drops the note from the address when a new note is started", async () => {
    renderNotesRoute(["/notes?note=11"])
    await waitFor(() => expect(editor()).toHaveValue("Alpha body"))

    fireEvent.keyDown(document.body, { key: "n", altKey: true })

    await waitFor(() => expect(editor()).toHaveValue(""))
    await waitFor(() => expect(address()).toBe("/notes"))
  })

  it("starts a new note at /notes?new=1 instead of reopening the last note (XP-06)", async () => {
    mockGetSetting.mockImplementation(async (setting: { key?: string }) => {
      const key = String(setting?.key || "")
      if (key.startsWith("tldw:notesRecentOpened:")) return [{ id: "12", title: "Beta note" }]
      if (key === "tldw:lastNoteId") return "11"
      return null
    })

    renderNotesRoute(["/chat", "/notes?new=1"])
    await waitForNotesReady()

    await waitFor(() => expect(address()).toBe("/notes"))
    await waitFor(() => expect(screen.getByRole("textbox", { name: "Note title" })).toHaveFocus())
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 50))
    })
    expect(editor()).toHaveValue("")
    const openedNotes = mockBgRequest.mock.calls
      .map(([request]) => String(request?.path || ""))
      .filter((path) => /^\/api\/v1\/notes\/\d+$/.test(path))
    expect(openedNotes).toEqual([])
    // The request is consumed: Back returns to the chat page.
    fireEvent.click(screen.getByRole("button", { name: "Browser Back" }))
    await waitFor(() => expect(address()).toBe("/chat"))
  })

  it("starts a new note when the palette asks while a note is open (XP-06)", async () => {
    renderNotesRoute(["/notes?note=11"])
    await waitFor(() => expect(editor()).toHaveValue("Alpha body"))

    fireEvent.click(screen.getByRole("button", { name: "Palette New note" }))

    await waitFor(() => expect(editor()).toHaveValue(""))
    await waitFor(() => expect(address()).toBe("/notes"))
    await waitFor(() => expect(screen.getByRole("textbox", { name: "Note title" })).toHaveFocus())
  })

  it("names the open note in the page title", async () => {
    renderNotesRoute(["/notes?note=11"])

    await waitFor(() => expect(document.title).toBe("Alpha note · Notes | tldw"))
  })
})

describe.each([
  ["router keys", false],
  ["web shim keys (pathname + search)", true]
])("New note requests (XP-06) with %s", { timeout: 60_000 }, (_label, shim) => {
  beforeEach(() => {
    routerKeys.shim = shim
    vi.clearAllMocks()
    localStorage.clear()
    canonicalBoundary.storage = {
      serverUrl: "https://notes.test",
      authMode: "multi-user",
      accessToken: token(7, 1),
      tldwConfig: {
        serverUrl: "https://notes.test",
        authMode: "multi-user",
        accessToken: token(7, 1),
        refreshToken: "refresh-one"
      }
    }
    canonicalBoundary.getConfig
      .mockReset()
      .mockImplementation(async () => canonicalBoundary.storage.tldwConfig)
    mockConfirmDanger.mockResolvedValue(true)
    mockGetSetting.mockResolvedValue(null)
    mockClearSetting.mockResolvedValue(undefined)
    mockBgRequest.mockImplementation(async (request: { path?: string; method?: string }) => {
      const path = String(request.path || "")
      const method = String(request.method || "GET").toUpperCase()
      if (path.startsWith("/api/v1/notes/?")) {
        return { items: [], pagination: { total_items: 0, total_pages: 1 } }
      }
      const detail = path.match(/^\/api\/v1\/notes\/(\d+)$/)
      if (detail && method === "GET" && NOTES[detail[1]]) {
        return {
          id: Number(detail[1]),
          ...NOTES[detail[1]],
          metadata: { keywords: [] },
          version: 1,
          last_modified: "2026-10-19T10:00:00.000Z"
        }
      }
      return {}
    })
  })
  afterEach(() => {
    routerKeys.shim = false
  })

  const settle = () =>
    act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 300))
    })
  const writes = () =>
    mockBgRequest.mock.calls
      .map(([request]) => request as { method?: string; path?: string })
      .filter((request) => String(request?.method || "GET").toUpperCase() !== "GET")

  it("starts a second new note during the same visit", async () => {
    renderNotesRoute(["/notes?note=11"])
    await waitFor(() => expect(editor()).toHaveValue("Alpha body"))

    fireEvent.click(screen.getByRole("button", { name: "Palette New note" }))
    await waitFor(() => expect(address()).toBe("/notes"))
    await waitFor(() => expect(editor()).toHaveValue(""))

    const title = screen.getByRole("textbox", { name: "Note title" })
    await waitFor(() => expect(title).toHaveFocus())
    act(() => editor().focus())
    fireEvent.click(screen.getByRole("button", { name: "Palette New note" }))

    // The second request starts another draft (its title takes focus) and leaves the address.
    await waitFor(() => expect(title).toHaveFocus())
    await waitFor(() => expect(address()).toBe("/notes"))
    expect(editor()).toHaveValue("")
  })

  it("opens a note after New note instead of starting another draft", async () => {
    renderNotesRoute(["/notes?note=11"])
    await waitFor(() => expect(editor()).toHaveValue("Alpha body"))
    fireEvent.click(screen.getByRole("button", { name: "Palette New note" }))
    await waitFor(() => expect(address()).toBe("/notes"))
    fireEvent.click(screen.getByRole("button", { name: "Palette New note" }))
    await settle()
    expect(address()).toBe("/notes")

    fireEvent.click(screen.getByRole("button", { name: "Open Beta via address" }))

    await waitFor(() => expect(editor()).toHaveValue("Beta body"))
    await settle()
    expect(editor()).toHaveValue("Beta body")
    expect(address()).toBe("/notes?note=12")
  })

  it("never carries a new-note request into the address of an opened note", async () => {
    renderNotesRoute(["/notes?new=1&note=12"])

    await waitFor(() => expect(editor()).toHaveValue("Beta body"))
    await waitFor(() => expect(address()).toBe("/notes?note=12"))
    await settle()
    expect(editor()).toHaveValue("Beta body")
  })

  it("keeps the open note in the address when the user stays on unsaved changes", async () => {
    const answerGet = mockBgRequest.getMockImplementation()!
    mockBgRequest.mockImplementation(async (request: { path?: string; method?: string }) => {
      if (String(request.method || "GET").toUpperCase() !== "GET" && String(request.path).startsWith("/api/v1/notes/11")) {
        throw Object.assign(new Error("Service unavailable"), { status: 503 })
      }
      return answerGet(request)
    })
    renderNotesRoute(["/notes?note=11"])
    await waitFor(() => expect(editor()).toHaveValue("Alpha body"))
    await waitFor(() => expect(address()).toBe("/notes?note=11"))
    fireEvent.change(editor(), { target: { value: "Alpha body, edited" } })

    fireEvent.click(screen.getByRole("button", { name: "Palette New note" }))
    // The save fails, so the page asks; the user stays on the note.
    fireEvent.click(await screen.findByRole("button", { name: "Stay on this note" }))

    await waitFor(() => expect(address()).toBe("/notes?note=11"))
    await settle()
    expect(editor()).toHaveValue("Alpha body, edited")
    expect(address()).toBe("/notes?note=11")
    expect(writes().every((request) => String(request.path).startsWith("/api/v1/notes/11"))).toBe(true)
  })
})
