import React from "react"
import { act, fireEvent, render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { MemoryRouter } from "react-router-dom"

import { KnowledgeQA } from "../index"
import {
  createKnowledgeQaStateFixture,
  type KnowledgeQaStateFixtureName,
} from "./knowledgeQaStateFixtures"

const state = {
  settingsPanelOpen: false,
  setSettingsPanelOpen: vi.fn(),
  currentThreadId: null as string | null,
  selectThread: vi.fn(),
  selectSharedThread: vi.fn(),
}

const stateBaseKeys = new Set(Object.keys(state))

const connectivity = {
  online: true,
  isChecking: false,
  lastCheckedAt: Date.now(),
  serverUrl: "http://127.0.0.1:8000",
  configStep: "health" as "none" | "url" | "auth" | "health",
  errorKind: "none" as "none" | "auth" | "unreachable" | "partial",
  lastError: null as string | null,
  lastStatusCode: null as number | null,
  knowledgeStatus: "ready" as "unknown" | "ready" | "indexing" | "offline" | "empty",
  knowledgeLastCheckedAt: Date.now() as number | null,
  knowledgeError: null as string | null,
  hasCompletedFirstRun: true,
  uxState: "connected_ok" as
    | "connected_ok"
    | "testing"
    | "configuring_url"
    | "configuring_auth"
    | "error_auth"
    | "error_unreachable"
    | "unconfigured",
  checkOnce: vi.fn(),
  navigate: vi.fn()
}

const capabilitiesState = {
  loading: false,
  capabilities: { hasRag: true },
  refresh: vi.fn(),
}

const layoutModeState = {
  mode: "simple" as "simple" | "research" | "expert",
  isSimple: true,
  isResearch: false,
  showPromotionToast: false,
}

vi.mock("../KnowledgeQAProvider", () => ({
  KnowledgeQAProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  useKnowledgeQA: () => state
}))

vi.mock("react-router-dom", async () => {
  const actual = await vi.importActual<typeof import("react-router-dom")>(
    "react-router-dom"
  )
  return {
    ...actual,
    useNavigate: () => connectivity.navigate
  }
})

vi.mock("@/hooks/useServerOnline", () => ({
  useServerOnline: () => connectivity.online
}))

vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => ({
    loading: capabilitiesState.loading,
    capabilities: capabilitiesState.capabilities,
    refresh: capabilitiesState.refresh,
  })
}))

vi.mock("@/hooks/useConnectionState", () => ({
  useConnectionActions: () => ({
    checkOnce: connectivity.checkOnce,
  }),
  useConnectionState: () => ({
    isChecking: connectivity.isChecking,
    lastCheckedAt: connectivity.lastCheckedAt,
    serverUrl: connectivity.serverUrl,
    configStep: connectivity.configStep,
    errorKind: connectivity.errorKind,
    lastError: connectivity.lastError,
    lastStatusCode: connectivity.lastStatusCode,
  }),
  useKnowledgeStatus: () => ({
    knowledgeStatus: connectivity.knowledgeStatus,
    knowledgeLastCheckedAt: connectivity.knowledgeLastCheckedAt,
    knowledgeError: connectivity.knowledgeError,
  }),
  useConnectionUxState: () => ({
    uxState: connectivity.uxState,
    hasCompletedFirstRun: connectivity.hasCompletedFirstRun,
  }),
}))

vi.mock("@/hooks/useMediaQuery", () => ({
  useMobile: () => false,
  useDesktop: () => true,
}))

vi.mock("../hooks/useLayoutMode", () => ({
  useLayoutMode: () => ({
    mode: layoutModeState.mode,
    setLayoutMode: vi.fn(),
    isSimple: layoutModeState.isSimple,
    isResearch: layoutModeState.isResearch,
    showPromotionToast: layoutModeState.showPromotionToast,
    dismissPromotion: vi.fn(),
    acceptPromotion: vi.fn(),
  }),
}))

vi.mock("../SearchBar", () => ({
  SearchBar: () => <input aria-label="Search your knowledge base" />
}))

vi.mock("../HistorySidebar", () => ({
  HistorySidebar: () => <div />
}))

vi.mock("../AnswerPanel", () => ({
  AnswerPanel: () => <div />
}))

vi.mock("../SearchDetailsPanel", () => ({
  SearchDetailsPanel: () => <div />
}))

vi.mock("../SourceList", () => ({
  SourceList: () => <div />
}))

vi.mock("../FollowUpInput", () => ({
  FollowUpInput: () => <div />
}))

vi.mock("../ConversationThread", () => ({
  ConversationThread: () => <div />
}))

vi.mock("../SettingsPanel", () => ({
  SettingsPanel: () => <div data-testid="knowledge-settings-panel" />
}))

vi.mock("../ExportDialog", () => ({
  ExportDialog: () => <div data-testid="knowledge-export-dialog" />
}))

describe("KnowledgeQA connection states", () => {
  const renderKnowledgeQa = () =>
    render(
      <MemoryRouter initialEntries={["/knowledge"]}>
        <KnowledgeQA />
      </MemoryRouter>
    )
  const resetKnowledgeQaState = () => {
    const mutableState = state as unknown as Record<string, unknown>
    for (const key of Object.keys(mutableState)) {
      if (!stateBaseKeys.has(key)) {
        delete mutableState[key]
      }
    }
    state.settingsPanelOpen = false
    state.setSettingsPanelOpen = vi.fn()
    state.currentThreadId = null
    state.selectThread = vi.fn()
    state.selectSharedThread = vi.fn()
  }
  const applyStateFixture = (name: KnowledgeQaStateFixtureName) => {
    const fixture = createKnowledgeQaStateFixture(name)
    resetKnowledgeQaState()
    Object.assign(state as unknown as Record<string, unknown>, fixture.knowledgeQa)
    connectivity.online = fixture.connection.online
    connectivity.isChecking = fixture.connection.isChecking
    connectivity.lastCheckedAt = fixture.connection.lastCheckedAt
    connectivity.uxState = fixture.connection.uxState
    connectivity.hasCompletedFirstRun = fixture.connection.hasCompletedFirstRun
    connectivity.serverUrl = fixture.connection.serverUrl
    connectivity.configStep = fixture.connection.configStep
    connectivity.errorKind = fixture.connection.errorKind
    connectivity.lastError = fixture.connection.lastError
    connectivity.lastStatusCode = fixture.connection.lastStatusCode
    connectivity.knowledgeStatus =
      fixture.connection.uxState === "unconfigured" ||
      fixture.connection.uxState === "configuring_url" ||
      fixture.connection.uxState === "configuring_auth"
        ? "unknown"
        : !fixture.connection.online
          ? "offline"
          : fixture.sourceInventory.media.length === 0 &&
              fixture.sourceInventory.notes.length === 0
            ? "empty"
            : "ready"
    connectivity.knowledgeLastCheckedAt = fixture.connection.lastCheckedAt
    connectivity.knowledgeError =
      connectivity.knowledgeStatus === "offline"
        ? fixture.connection.lastError
        : null
    capabilitiesState.loading = fixture.capabilities.loading
    capabilitiesState.capabilities = fixture.capabilities.capabilities
  }

  beforeEach(() => {
    vi.clearAllMocks()
    vi.useRealTimers()
    resetKnowledgeQaState()
    connectivity.online = true
    connectivity.isChecking = false
    connectivity.lastCheckedAt = Date.now()
    connectivity.serverUrl = "http://127.0.0.1:8000"
    connectivity.configStep = "health"
    connectivity.errorKind = "none"
    connectivity.lastError = null
    connectivity.lastStatusCode = null
    connectivity.knowledgeStatus = "ready"
    connectivity.knowledgeLastCheckedAt = Date.now()
    connectivity.knowledgeError = null
    connectivity.hasCompletedFirstRun = true
    connectivity.uxState = "connected_ok"
    capabilitiesState.loading = false
    capabilitiesState.capabilities = { hasRag: true }
  })

  it("keeps fixture connection timestamps deterministic", () => {
    const dateNow = vi.spyOn(Date, "now").mockReturnValue(42)

    try {
      expect(createKnowledgeQaStateFixture("readySearch").connection.lastCheckedAt).toBe(
        Date.parse("2026-06-07T12:00:00.000Z")
      )
    } finally {
      dateNow.mockRestore()
    }
  })

  it("renders the backend offline audited state from a deterministic fixture", () => {
    applyStateFixture("backendOffline")

    renderKnowledgeQa()

    expect(
      screen.getByText("Can't reach your tldw server right now")
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Health & diagnostics" })
    ).toBeInTheDocument()
  })

  it("renders the setup required audited state from a deterministic fixture", () => {
    applyStateFixture("setupRequired")

    renderKnowledgeQa()

    expect(screen.getByText("Setup Required")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Finish Setup" })).toBeInTheDocument()
    expect(screen.getByTestId("knowledge-setup-diagnostics")).toBeInTheDocument()
    expect(screen.getByText("Server URL")).toBeInTheDocument()
    expect(screen.getByText("Add a tldw server URL before Knowledge QA can search your library.")).toBeInTheDocument()
  })

  it("renders settings and export audited states from deterministic fixtures", async () => {
    applyStateFixture("settingsDrawer")
    const { rerender } = renderKnowledgeQa()

    expect(await screen.findByTestId("knowledge-settings-panel")).toBeInTheDocument()

    applyStateFixture("exportDialog")
    rerender(
      <MemoryRouter initialEntries={["/knowledge"]}>
        <KnowledgeQA />
      </MemoryRouter>
    )

    fireEvent.click(await screen.findByRole("button", { name: "Export" }))
    expect(await screen.findByTestId("knowledge-export-dialog")).toBeInTheDocument()
  })

  it("shows credential guidance instead of the generic offline screen", () => {
    connectivity.online = false
    connectivity.uxState = "error_auth"

    renderKnowledgeQa()

    expect(
      screen.getByText("Add your credentials to use Knowledge QA")
    ).toBeInTheDocument()
    expect(screen.queryByText("Server Offline")).not.toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Open Settings" }))
    expect(connectivity.navigate).toHaveBeenCalledWith("/settings/tldw")
  })

  it("shows setup guidance and routes users to setup", () => {
    connectivity.online = false
    connectivity.uxState = "unconfigured"
    connectivity.serverUrl = null
    connectivity.configStep = "url"
    connectivity.hasCompletedFirstRun = false

    renderKnowledgeQa()

    expect(screen.getByText("Setup Required")).toBeInTheDocument()
    expect(screen.getByText("Credentials")).toBeInTheDocument()
    expect(screen.getByText("Waiting for a server URL before checking credentials.")).toBeInTheDocument()
    expect(screen.queryByLabelText("Search your knowledge base")).not.toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Finish Setup" }))
    expect(connectivity.navigate).toHaveBeenCalledWith("/")
  })

  it("shows credential diagnostics when a saved server URL is missing auth", () => {
    connectivity.online = false
    connectivity.uxState = "configuring_auth"
    connectivity.serverUrl = "http://127.0.0.1:8000"
    connectivity.configStep = "auth"

    renderKnowledgeQa()

    expect(screen.getByText("Add your credentials to use Knowledge QA")).toBeInTheDocument()
    expect(screen.getByText("Configured server: http://127.0.0.1:8000")).toBeInTheDocument()
    expect(screen.getByText("Add the API key or login token for this tldw server.")).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Open Settings" }))
    expect(connectivity.navigate).toHaveBeenCalledWith("/settings/tldw")
  })

  it("calls out browser access and allowlist blockers for extension recovery", () => {
    connectivity.online = false
    connectivity.uxState = "error_unreachable"
    connectivity.serverUrl = "http://127.0.0.1:8000/private?api_key=hidden"
    connectivity.configStep = "health"
    connectivity.errorKind = "unreachable"
    connectivity.lastError =
      "Absolute URL requests are blocked unless the request origin is explicitly allowlisted."
    connectivity.lastStatusCode = 400

    renderKnowledgeQa()

    expect(screen.getByText("Configured server: http://127.0.0.1:8000")).toBeInTheDocument()
    expect(screen.getByText("Browser access")).toBeInTheDocument()
    expect(
      screen.getByText(/Allowlist this server origin or grant extension host access/i)
    ).toBeInTheDocument()
    expect(
      screen.getByText(/Absolute URL requests are blocked/i)
    ).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Health & diagnostics" })).toBeInTheDocument()
  })

  it("keeps retry behavior for unreachable servers", () => {
    vi.useFakeTimers()
    connectivity.online = false
    connectivity.uxState = "error_unreachable"
    connectivity.lastCheckedAt = Date.now() - 1_000

    renderKnowledgeQa()

    expect(
      screen.getByText("Can't reach your tldw server right now")
    ).toBeInTheDocument()
    expect(screen.getByText(/Retrying automatically in/)).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Retry connection" }))
    expect(connectivity.checkOnce).toHaveBeenCalled()

    act(() => {
      vi.runOnlyPendingTimers()
    })
  })

  it("stretches the knowledge workspace root to the full route width", () => {
    renderKnowledgeQa()

    expect(screen.getByTestId("knowledge-page-root")).toHaveClass("w-full")
    expect(screen.getByTestId("knowledge-page-root")).toHaveClass("flex-1")
    expect(screen.getByTestId("knowledge-page-root")).toHaveClass("min-w-0")
  })
})
