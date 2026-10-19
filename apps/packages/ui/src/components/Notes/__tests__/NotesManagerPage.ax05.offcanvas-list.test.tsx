import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import NotesManagerPage from "../NotesManagerPage"

// AX-05: a closed off-canvas notes list must leave the Tab order, and an open
// mobile list must behave like a modal dialog (focus in, Esc out, focus back).

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
  mockNavigate,
  mockConfirmDanger,
  mockGetSetting,
  mockClearSetting,
  responsiveState
} = vi.hoisted(() => ({
  mockBgRequest: vi.fn(),
  mockNavigate: vi.fn(),
  mockConfirmDanger: vi.fn(),
  mockGetSetting: vi.fn(),
  mockClearSetting: vi.fn(),
  responsiveState: { isMobile: false }
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

vi.mock("@/hooks/useMediaQuery", () => ({
  useMobile: () => responsiveState.isMobile
}))

vi.mock("@/components/Common/confirm-danger", () => ({
  useConfirmDanger: () => mockConfirmDanger
}))

vi.mock("@/hooks/useAntdMessage", () => ({
  useAntdMessage: () => ({
    success: vi.fn(),
    error: vi.fn(),
    warning: vi.fn(),
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

const queryClients: QueryClient[] = []

const renderPage = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false }
    }
  })
  queryClients.push(queryClient)
  return render(
    <QueryClientProvider client={queryClient}>
      <NotesManagerPage />
    </QueryClientProvider>
  )
}

describe("NotesManagerPage AX-05 off-canvas notes list", { timeout: 60_000 }, () => {
  beforeEach(() => {
    vi.clearAllMocks()
    responsiveState.isMobile = false
    mockConfirmDanger.mockResolvedValue(true)
    mockGetSetting.mockResolvedValue(null)
    mockClearSetting.mockResolvedValue(undefined)
    mockBgRequest.mockImplementation(async (request: { path?: string }) => {
      const path = String(request.path || "")
      if (path.startsWith("/api/v1/notes/?")) {
        return { items: [], pagination: { total_items: 0, total_pages: 1 } }
      }
      return {}
    })
  })

  afterEach(async () => {
    cleanup()
    while (queryClients.length > 0) {
      const queryClient = queryClients.pop()
      if (!queryClient) continue
      await queryClient.cancelQueries()
      queryClient.clear()
    }
  })

  it("takes the closed mobile list out of the Tab order and the accessibility tree", async () => {
    responsiveState.isMobile = true
    renderPage()

    const region = await screen.findByTestId("notes-list-region")
    expect(region).toHaveAttribute("inert")
    expect(region).toHaveAttribute("aria-hidden", "true")

    const trigger = screen.getByTestId("notes-mobile-open-list-button")
    expect(trigger).toHaveAttribute("aria-expanded", "false")
    expect(trigger).toHaveAttribute("aria-controls", "notes-list-region")
  })

  it("opens the mobile list as a modal dialog, focuses search, and returns focus on Escape", async () => {
    responsiveState.isMobile = true
    renderPage()

    const trigger = await screen.findByTestId("notes-mobile-open-list-button")
    trigger.focus()
    fireEvent.click(trigger)

    const region = screen.getByTestId("notes-list-region")
    await waitFor(() => {
      expect(region).not.toHaveAttribute("inert")
    })
    expect(region).not.toHaveAttribute("aria-hidden")
    expect(region).toHaveAttribute("role", "dialog")
    expect(region).toHaveAttribute("aria-modal", "true")
    expect(trigger).toHaveAttribute("aria-expanded", "true")
    expect(screen.getByTestId("notes-mobile-sidebar-close")).toBeInTheDocument()
    await waitFor(() => {
      expect(screen.getByLabelText("Search notes")).toHaveFocus()
    })

    fireEvent.keyDown(screen.getByLabelText("Search notes"), { key: "Escape" })

    await waitFor(() => {
      expect(region).toHaveAttribute("inert")
    })
    expect(region.className).toContain("-translate-x-full")
    await waitFor(() => {
      expect(trigger).toHaveFocus()
    })
  })

  it("keeps Tab inside the open mobile list", async () => {
    responsiveState.isMobile = true
    renderPage()

    fireEvent.click(await screen.findByTestId("notes-mobile-open-list-button"))
    const closeButton = await screen.findByTestId("notes-mobile-sidebar-close")
    const region = screen.getByTestId("notes-list-region")
    await waitFor(() => {
      expect(region).not.toHaveAttribute("inert")
    })

    const tabStops = Array.from(
      region.querySelectorAll<HTMLElement>("button:not([disabled]), input:not([disabled])")
    ).filter((element) => element.getAttribute("tabindex") !== "-1")
    const first = tabStops[0]
    const last = tabStops[tabStops.length - 1]
    expect(tabStops.length).toBeGreaterThan(1)

    first.focus()
    fireEvent.keyDown(first, { key: "Tab", shiftKey: true })
    expect(last).toHaveFocus()

    fireEvent.keyDown(last, { key: "Tab" })
    expect(first).toHaveFocus()

    fireEvent.click(closeButton)
    await waitFor(() => {
      expect(region).toHaveAttribute("inert")
    })
  })

  it("marks the collapsed desktop list inert and reports the toggle state", async () => {
    renderPage()

    const region = await screen.findByTestId("notes-list-region")
    const toggle = screen.getByTestId("notes-desktop-sidebar-toggle")
    expect(region).not.toHaveAttribute("inert")
    expect(region).toHaveAttribute("role", "region")
    expect(toggle).toHaveAttribute("aria-expanded", "true")
    expect(toggle).toHaveAttribute("aria-controls", "notes-list-region")

    fireEvent.click(toggle)

    await waitFor(() => {
      expect(region).toHaveAttribute("inert")
    })
    expect(region).toHaveAttribute("aria-hidden", "true")
    expect(toggle).toHaveAttribute("aria-expanded", "false")
  })
})
