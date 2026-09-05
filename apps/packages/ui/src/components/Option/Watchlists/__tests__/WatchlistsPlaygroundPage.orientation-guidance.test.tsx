// @vitest-environment jsdom

import React from "react"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { MemoryRouter } from "react-router-dom"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { getTutorialById } from "@/tutorials/registry"
import { WatchlistsPlaygroundPage } from "../WatchlistsPlaygroundPage"

const mocks = vi.hoisted(() => {
  const state = {
    activeTab: "overview" as
      | "overview"
      | "sources"
      | "jobs"
      | "runs"
      | "items"
      | "alerts"
      | "outputs"
      | "templates"
      | "settings",
    setActiveTab: vi.fn((next: string) => {
      state.activeTab = next as typeof state.activeTab
    })
  }
  return {
    watchlistContainer: {
      id: 42,
      name: "Healthcare ransomware",
      description: "Track hospital impact",
      objective: "Find new ransomware affecting hospitals",
      domain: "cti_osint",
      status: "active",
      priority: "high",
      tags: ["ransomware", "hospitals"],
      created_at: "2026-05-15T00:00:00Z",
      updated_at: "2026-05-15T00:00:00Z"
    },
    createWatchlistMock: vi.fn(),
    fetchWatchlistRunsMock: vi.fn(),
    fetchWatchlistsMock: vi.fn(),
    updateWatchlistMock: vi.fn(),
    recordWatchlistsIaExperimentTelemetryMock: vi.fn(),
    trackWatchlistsOnboardingTelemetryMock: vi.fn(),
    notificationDestroyMock: vi.fn(),
    state
  }
})

const connectionMocks = vi.hoisted(() => ({
  useConnectionUxState: vi.fn()
}))

const navigationMocks = vi.hoisted(() => ({
  navigate: vi.fn()
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, defaultValue?: unknown, options?: Record<string, unknown>) => {
      if (typeof defaultValue !== "string") return _key
      if (!options) return defaultValue
      return defaultValue.replace(/\{\{(\w+)\}\}/g, (_, token) => String(options[token] ?? ""))
    }
  })
}))

vi.mock("antd", async () => {
  const actual = await vi.importActual<typeof import("antd")>("antd")
  const Alert = ({ title, description, action, closable, onClose }: any) => (
    <div>
      <div>{title}</div>
      <div>{description}</div>
      <div>{action}</div>
      {closable ? (
        <button type="button" onClick={() => onClose?.()}>
          Dismiss
        </button>
      ) : null}
    </div>
  )

  const Tabs = ({ items = [] }: any) => (
    <div>
      {items.map((item: any) => (
        <button key={item.key} type="button" data-testid={`watchlists-tab-${item.key}`}>
          {item.label}
        </button>
      ))}
      <div>{items[0]?.children}</div>
    </div>
  )

  const Modal = ({ open, title, children, footer, onCancel, afterOpenChange }: any) => {
    React.useEffect(() => {
      afterOpenChange?.(open)
    }, [afterOpenChange, open])

    return open ? (
      <div
        role="dialog"
        aria-label={typeof title === "string" ? title : "dialog"}
        tabIndex={-1}
        onKeyDown={(event) => {
          if (event.key === "Escape") onCancel?.()
        }}
      >
        <h3>{title}</h3>
        {children}
        <div>{footer}</div>
      </div>
    ) : null
  }
  const Drawer = ({ open, title, children }: any) =>
    open ? (
      <div>
        <h3>{title}</h3>
        {children}
      </div>
    ) : null

  const Empty = ({ description }: any) => <div>{description}</div>
  const Tooltip = ({ children }: any) => <>{children}</>
  const Button = React.forwardRef<HTMLButtonElement, any>(
    ({ children, onClick, disabled, ...rest }, ref) => (
      <button ref={ref} type="button" onClick={() => onClick?.()} disabled={Boolean(disabled)} {...rest}>
        {children}
      </button>
    )
  )
  const Switch = ({ checked, onChange, ...rest }: any) => (
    <button
      type="button"
      aria-label={rest["aria-label"] || "Toggle"}
      aria-pressed={Boolean(checked)}
      onClick={() => onChange?.(!checked)}
      {...rest}
    />
  )
  const Select = ({ value, options = [], onChange, "aria-label": ariaLabel, ...rest }: any) => (
    <select
      aria-label={ariaLabel}
      value={value == null ? "" : String(value)}
      onChange={(event) => {
        const rawValue = event.currentTarget.value
        const numeric = Number(rawValue)
        onChange?.(Number.isFinite(numeric) && rawValue.trim() !== "" ? numeric : rawValue)
      }}
      {...rest}
    >
      {options.map((option: any) => (
        <option key={String(option.value)} value={String(option.value)}>
          {option.label}
        </option>
      ))}
    </select>
  )
  const Tag = ({ children }: any) => <span>{children}</span>
  return { ...actual, Alert, Tabs, Empty, Button, Modal, Drawer, Tooltip, Switch, Select, Tag }
})

vi.mock("@/components/Common/WorkspaceConnectionGate", () => ({
  default: ({ children }: { children: React.ReactNode }) => <>{children}</>
}))

vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => ({
    destroy: mocks.notificationDestroyMock,
    success: vi.fn(),
    error: vi.fn(),
    warning: vi.fn()
  })
}))

vi.mock("@/hooks/useServerOnline", () => ({
  useServerOnline: () => true
}))

vi.mock("@/hooks/useConnectionState", () => ({
  useConnectionUxState: () => connectionMocks.useConnectionUxState()
}))

vi.mock("react-router-dom", async () => {
  const actual = await vi.importActual<typeof import("react-router-dom")>(
    "react-router-dom"
  )
  return {
    ...actual,
    useNavigate: () => navigationMocks.navigate
  }
})

vi.mock("@/services/watchlists", () => ({
  createWatchlist: (...args: any[]) => mocks.createWatchlistMock(...args),
  fetchWatchlistRuns: (...args: any[]) => mocks.fetchWatchlistRunsMock(...args),
  fetchWatchlists: (...args: any[]) => mocks.fetchWatchlistsMock(...args),
  updateWatchlist: (...args: any[]) => mocks.updateWatchlistMock(...args),
  recordWatchlistsIaExperimentTelemetry: (...args: any[]) =>
    mocks.recordWatchlistsIaExperimentTelemetryMock(...args)
}))

vi.mock("@/utils/watchlists-onboarding-telemetry", () => ({
  trackWatchlistsOnboardingTelemetry: (...args: any[]) =>
    mocks.trackWatchlistsOnboardingTelemetryMock(...args)
}))

vi.mock("@/store/watchlists", () => ({
  useWatchlistsStore: (selector: (state: Record<string, unknown>) => unknown) =>
    selector({
      activeTab: mocks.state.activeTab,
      watchlists: [mocks.watchlistContainer],
      watchlistsLoading: false,
      watchlistsError: null,
      selectedWatchlistId: 42,
      setActiveTab: mocks.state.setActiveTab,
      setWatchlists: vi.fn(),
      setWatchlistsLoading: vi.fn(),
      setWatchlistsError: vi.fn(),
      setSelectedWatchlistId: vi.fn(),
      addWatchlist: vi.fn(),
      updateWatchlistInList: vi.fn(),
      openRunDetail: vi.fn(),
      resetStore: vi.fn()
    })
}))

vi.mock("../OverviewTab/OverviewTab", () => ({
  OverviewTab: () => <div>Overview tab</div>
}))
vi.mock("../SourcesTab/SourcesTab", () => ({
  SourcesTab: () => <div>Sources tab</div>
}))
vi.mock("../JobsTab/JobsTab", () => ({
  JobsTab: () => <div>Jobs tab</div>
}))
vi.mock("../RunsTab/RunsTab", () => ({
  RunsTab: () => <div>Runs tab</div>
}))
vi.mock("../OutputsTab/OutputsTab", () => ({
  OutputsTab: () => <div>Outputs tab</div>
}))
vi.mock("../TemplatesTab/TemplatesTab", () => ({
  TemplatesTab: () => <div>Templates tab</div>
}))
vi.mock("../SettingsTab/SettingsTab", () => ({
  SettingsTab: () => <div>Settings tab</div>
}))
vi.mock("../ItemsTab/ItemsTab", () => ({
  ItemsTab: () => <div>Items tab</div>
}))
vi.mock("../shared/WatchlistsHealthBar", () => ({
  WatchlistsHealthBar: () => <div data-testid="watchlists-health-bar" />
}))

const renderPage = () =>
  render(
    <MemoryRouter initialEntries={["/watchlists"]}>
      <WatchlistsPlaygroundPage />
    </MemoryRouter>
  )

const showTabGuidance = () => {
  if (!screen.queryByTestId("watchlists-help-panel")) {
    fireEvent.click(screen.getByTestId("watchlists-help-icon"))
  }
}

describe("WatchlistsPlaygroundPage orientation guidance", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    connectionMocks.useConnectionUxState.mockReturnValue({
      uxState: "connected_ok",
      hasCompletedFirstRun: true
    })
    mocks.fetchWatchlistsMock.mockResolvedValue({
      items: [mocks.watchlistContainer],
      total: 1,
      has_more: false
    })
    mocks.fetchWatchlistRunsMock.mockResolvedValue({ items: [], total: 0, has_more: false })
    mocks.recordWatchlistsIaExperimentTelemetryMock.mockResolvedValue({ accepted: true })
    mocks.trackWatchlistsOnboardingTelemetryMock.mockResolvedValue(undefined)
    mocks.state.activeTab = "overview"
    ;(window as { __TLDW_WATCHLISTS_IA_EXPERIMENT__?: unknown }).__TLDW_WATCHLISTS_IA_EXPERIMENT__ = false
    localStorage.removeItem("watchlists:guided-tour:v1")
    localStorage.removeItem("watchlists:ia-experiment:v1")
    localStorage.removeItem("watchlists:show-all-views:v1")
    localStorage.removeItem("watchlists:secondary-expanded:v1")
    localStorage.removeItem("watchlists:teach-points:v1")
  })

  afterEach(() => {
    cleanup()
    delete (window as { __TLDW_WATCHLISTS_IA_EXPERIMENT__?: unknown }).__TLDW_WATCHLISTS_IA_EXPERIMENT__
    localStorage.removeItem("watchlists:guided-tour:v1")
    localStorage.removeItem("watchlists:ia-experiment:v1")
    localStorage.removeItem("watchlists:show-all-views:v1")
    localStorage.removeItem("watchlists:secondary-expanded:v1")
    localStorage.removeItem("watchlists:teach-points:v1")
  })

  it("renders every Watchlists tutorial target on the default route", async () => {
    renderPage()
    await screen.findByTestId("watchlists-outcome-first-region")

    for (const step of getTutorialById("watchlists-basics")?.steps ?? []) {
      expect(document.querySelector(step.target), step.target).not.toBeNull()
    }
  })

  it("closes Help before navigating from orientation guidance and restores trigger focus", async () => {
    mocks.state.activeTab = "runs"
    localStorage.setItem("watchlists:show-all-views:v1", "true")
    renderPage()
    const helpTrigger = screen.getByRole("button", { name: "Open Watchlists help" })
    helpTrigger.focus()
    showTabGuidance()

    expect(screen.getByTestId("watchlists-orientation-title")).toHaveTextContent("Activity")
    expect(screen.getByTestId("watchlists-help-panel")).toContainElement(
      screen.getByTestId("watchlists-orientation-title")
    )
    expect(screen.getByTestId("watchlists-orientation-description")).toHaveTextContent("Reports")

    fireEvent.click(screen.getByTestId("watchlists-orientation-action-open-reports"))
    expect(mocks.state.setActiveTab).toHaveBeenCalledWith("outputs")
    await waitFor(() => {
      expect(screen.queryByRole("dialog", { name: "Watchlists help" })).not.toBeInTheDocument()
      expect(screen.queryAllByRole("dialog")).toHaveLength(0)
      expect(helpTrigger).toHaveFocus()
    })
  })

  it("uses the canonical alert for tab teach points", () => {
    mocks.state.activeTab = "jobs"
    localStorage.setItem("watchlists:show-all-views:v1", "true")
    render(<WatchlistsPlaygroundPage />)
    fireEvent.click(screen.getByTestId("watchlists-help-icon"))

    expect(screen.getByText("Monitor setup tip")).toBeInTheDocument()
    expect(screen.getByTestId("watchlists-help-panel")).toContainElement(
      screen.getByText("Monitor setup tip")
    )
  })

  it("supports the primary workflow journey from overview through reports", () => {
    const { rerender } = renderPage()
    showTabGuidance()

    fireEvent.click(screen.getByTestId("watchlists-orientation-action-open-feeds"))
    expect(mocks.state.setActiveTab).toHaveBeenCalledWith("sources")

    mocks.state.activeTab = "sources"
    rerender(
      <MemoryRouter initialEntries={["/watchlists"]}>
        <WatchlistsPlaygroundPage />
      </MemoryRouter>
    )
    showTabGuidance()
    fireEvent.click(screen.getByTestId("watchlists-orientation-action-open-monitors"))
    expect(mocks.state.setActiveTab).toHaveBeenCalledWith("sources")
    expect(localStorage.getItem("watchlists:secondary-expanded:v1")).toContain("\"monitors\":true")

    mocks.state.activeTab = "jobs"
    rerender(
      <MemoryRouter initialEntries={["/watchlists"]}>
        <WatchlistsPlaygroundPage />
      </MemoryRouter>
    )
    showTabGuidance()
    fireEvent.click(screen.getByTestId("watchlists-orientation-action-open-activity"))
    expect(mocks.state.setActiveTab).toHaveBeenCalledWith("items")
    expect(localStorage.getItem("watchlists:secondary-expanded:v1")).toContain("\"activity\":true")

    mocks.state.activeTab = "runs"
    rerender(
      <MemoryRouter initialEntries={["/watchlists"]}>
        <WatchlistsPlaygroundPage />
      </MemoryRouter>
    )
    showTabGuidance()
    fireEvent.click(screen.getByTestId("watchlists-orientation-action-open-reports"))
    expect(mocks.state.setActiveTab).toHaveBeenCalledWith("outputs")
  })

  it("normalizes a direct Activity store transition into the expanded Updates surface", async () => {
    const { rerender } = renderPage()

    mocks.state.setActiveTab("runs")
    rerender(
      <MemoryRouter initialEntries={["/watchlists"]}>
        <WatchlistsPlaygroundPage />
      </MemoryRouter>
    )

    await waitFor(() => {
      expect(mocks.state.setActiveTab).toHaveBeenLastCalledWith("items")
      expect(localStorage.getItem("watchlists:secondary-expanded:v1")).toContain("\"activity\":true")
    })
  })

  it("keeps command palette reachable through Help without duplicate jump strips", () => {
    renderPage()

    expect(screen.getByTestId("watchlists-health-bar")).toBeInTheDocument()
    expect(screen.queryByTestId("watchlists-repeat-actions")).not.toBeInTheDocument()
    fireEvent.click(screen.getByTestId("watchlists-help-icon"))
    fireEvent.click(screen.getByTestId("watchlists-open-command-palette"))
    expect(screen.getByTestId("watchlists-command-palette-input")).toBeInTheDocument()
  })

  it("keeps orientation guidance in Help instead of the viewport", () => {
    mocks.state.activeTab = "runs"
    localStorage.setItem("watchlists:show-all-views:v1", "true")
    renderPage()

    expect(screen.queryByTestId("watchlists-orientation-alert")).not.toBeInTheDocument()
    expect(screen.queryByTestId("watchlists-orientation-title")).not.toBeInTheDocument()

    showTabGuidance()
    expect(screen.getByTestId("watchlists-orientation-title")).toHaveTextContent("Activity")
    expect(screen.getByTestId("watchlists-help-panel")).toContainElement(
      screen.getByTestId("watchlists-orientation-title")
    )
  })

  it("exposes an accessible label on the Watchlists help button", () => {
    renderPage()

    expect(
      screen.getByRole("button", { name: "Open Watchlists help" })
    ).toHaveAttribute("data-testid", "watchlists-help-icon")
  })
})
