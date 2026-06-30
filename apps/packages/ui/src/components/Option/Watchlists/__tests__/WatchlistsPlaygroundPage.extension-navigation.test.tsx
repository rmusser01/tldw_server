// @vitest-environment jsdom

import React from "react"
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { WatchlistsPlaygroundPage } from "../WatchlistsPlaygroundPage"

const container = {
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
}

const mocks = vi.hoisted(() => {
  const state = {
    activeTab: "sources" as
      | "overview"
      | "sources"
      | "jobs"
      | "runs"
      | "items"
      | "alerts"
      | "outputs"
      | "templates"
      | "settings",
    watchlists: [] as Array<Record<string, unknown>>,
    watchlistsLoading: false,
    watchlistsError: null as string | null,
    selectedWatchlistId: 42 as number | null,
    setActiveTab: vi.fn((next: string) => {
      state.activeTab = next as typeof state.activeTab
    }),
    setWatchlists: vi.fn((nextWatchlists: Array<Record<string, unknown>>, selectedId?: number | null) => {
      state.watchlists = nextWatchlists
      state.selectedWatchlistId = selectedId ?? (Number(nextWatchlists[0]?.id ?? null) || null)
    }),
    setWatchlistsLoading: vi.fn((loading: boolean) => {
      state.watchlistsLoading = loading
    }),
    setWatchlistsError: vi.fn((error: string | null) => {
      state.watchlistsError = error
    }),
    setSelectedWatchlistId: vi.fn((nextId: number | null) => {
      state.selectedWatchlistId = nextId
    })
  }
  return {
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

const setViewport = (width: number) => {
  Object.defineProperty(window, "innerWidth", {
    configurable: true,
    value: width
  })
  Object.defineProperty(window, "matchMedia", {
    configurable: true,
    value: vi.fn((query: string) => ({
      matches: query.includes("max-width") ? width <= 767 : width >= 768,
      media: query,
      onchange: null,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      addListener: vi.fn(),
      removeListener: vi.fn(),
      dispatchEvent: vi.fn()
    }))
  })
}

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
  const Alert = ({ title, description, action, children }: any) => (
    <div>
      <div>{title}</div>
      <div>{description}</div>
      <div>{action}</div>
      {children}
    </div>
  )
  const Button = ({ children, icon, onClick, disabled, block: _block, ...rest }: any) => (
    <button type="button" onClick={() => onClick?.()} disabled={Boolean(disabled)} {...rest}>
      {icon}
      {children}
    </button>
  )
  const Drawer = ({ open, title, children, "data-testid": testId }: any) =>
    open ? (
      <div data-testid={testId}>
        <h3>{title}</h3>
        {children}
      </div>
    ) : null
  const Input = ({ value, onChange, ...rest }: any) => (
    <input value={value ?? ""} onChange={(event) => onChange?.(event)} {...rest} />
  )
  Input.TextArea = ({ value, onChange, ...rest }: any) => (
    <textarea value={value ?? ""} onChange={(event) => onChange?.(event)} {...rest} />
  )
  const Modal = ({ open, title, children, footer }: any) =>
    open ? (
      <div role="dialog" aria-label={typeof title === "string" ? title : "dialog"}>
        <h3>{title}</h3>
        {children}
        <div>{footer}</div>
      </div>
    ) : null
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
          {typeof option.label === "string" ? option.label : String(option.value)}
        </option>
      ))}
    </select>
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
  const Tabs = ({ activeKey, items = [] }: any) => {
    const activeItem = items.find((item: any) => item.key === activeKey) || items[0]
    return (
      <div data-testid="watchlists-desktop-tabs">
        {items.map((item: any) => (
          <button key={item.key} type="button" data-testid={`watchlists-tab-${item.key}`}>
            {item.label}
          </button>
        ))}
        <div>{activeItem?.children}</div>
      </div>
    )
  }
  const Tag = ({ children }: any) => <span>{children}</span>
  const Tooltip = ({ children }: any) => <>{children}</>
  return { ...actual, Alert, Button, Drawer, Input, Modal, Select, Switch, Tabs, Tag, Tooltip }
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

vi.mock("@/services/watchlists", () => ({
  createWatchlist: (...args: unknown[]) => mocks.createWatchlistMock(...args),
  fetchWatchlistRuns: (...args: unknown[]) => mocks.fetchWatchlistRunsMock(...args),
  fetchWatchlists: (...args: unknown[]) => mocks.fetchWatchlistsMock(...args),
  updateWatchlist: (...args: unknown[]) => mocks.updateWatchlistMock(...args),
  recordWatchlistsIaExperimentTelemetry: (...args: unknown[]) =>
    mocks.recordWatchlistsIaExperimentTelemetryMock(...args)
}))

vi.mock("@/utils/watchlists-onboarding-telemetry", () => ({
  trackWatchlistsOnboardingTelemetry: (...args: unknown[]) =>
    mocks.trackWatchlistsOnboardingTelemetryMock(...args)
}))

vi.mock("@/store/watchlists", () => ({
  useWatchlistsStore: (selector: (state: Record<string, unknown>) => unknown) =>
    selector({
      ...mocks.state,
      addWatchlist: vi.fn(),
      updateWatchlistInList: vi.fn(),
      openJobForm: vi.fn(),
      openRunDetail: vi.fn(),
      openSourceForm: vi.fn(),
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
vi.mock("../ItemsTab/ItemsTab", () => ({
  ItemsTab: () => <div>Items tab</div>
}))
vi.mock("../AlertsTab/AlertsTab", () => ({
  AlertsTab: () => <div>Alerts tab</div>
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
vi.mock("../shared/WatchlistsHealthBar", () => ({
  WatchlistsHealthBar: () => <div data-testid="watchlists-health-bar" />
}))

describe("WatchlistsPlaygroundPage extension navigation", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    connectionMocks.useConnectionUxState.mockReturnValue({
      uxState: "connected_ok",
      hasCompletedFirstRun: true
    })
    mocks.fetchWatchlistsMock.mockResolvedValue({
      items: [container],
      total: 1,
      has_more: false
    })
    mocks.fetchWatchlistRunsMock.mockResolvedValue({ items: [], total: 0, has_more: false })
    mocks.recordWatchlistsIaExperimentTelemetryMock.mockResolvedValue({ accepted: true })
    mocks.trackWatchlistsOnboardingTelemetryMock.mockResolvedValue(undefined)
    mocks.state.activeTab = "sources"
    mocks.state.watchlists = [container]
    mocks.state.selectedWatchlistId = 42
    localStorage.clear()
    ;(window as { __TLDW_WATCHLISTS_IA_EXPERIMENT__?: unknown }).__TLDW_WATCHLISTS_IA_EXPERIMENT__ = false
  })

  afterEach(() => {
    cleanup()
  })

  it("exposes all management destinations through constrained navigation at extension width", async () => {
    setViewport(420)
    render(<WatchlistsPlaygroundPage />)

    expect(await screen.findByRole("heading", { name: "Healthcare ransomware" })).toBeInTheDocument()
    expect(screen.getByLabelText("Selected Watchlist")).toBeInTheDocument()
    expect(screen.queryByTestId("watchlists-mobile-tab-select")).not.toBeInTheDocument()

    const trigger = screen.getByTestId("watchlists-constrained-nav-trigger")
    expect(trigger).toHaveTextContent("Feeds")
    fireEvent.click(trigger)

    const drawer = screen.getByTestId("watchlists-constrained-nav-drawer")
    for (const label of [
      "Overview",
      "Feeds",
      "Monitors",
      "Alerts",
      "Updates",
      "Activity",
      "Reports",
      "Templates",
      "Settings"
    ]) {
      expect(within(drawer).getByRole("button", { name: label })).toBeInTheDocument()
    }

    mocks.state.setActiveTab.mockClear()
    fireEvent.click(within(drawer).getByRole("button", { name: "Monitors" }))
    expect(mocks.state.setActiveTab).toHaveBeenLastCalledWith("jobs")
  })

  it("keeps the desktop tab layout above the constrained breakpoint", async () => {
    setViewport(1024)
    render(<WatchlistsPlaygroundPage />)

    await waitFor(() => expect(mocks.fetchWatchlistsMock).toHaveBeenCalled())
    expect(screen.queryByTestId("watchlists-constrained-nav-trigger")).not.toBeInTheDocument()
    expect(screen.getByTestId("watchlists-desktop-tabs")).toBeInTheDocument()
    expect(screen.getByTestId("watchlists-tab-sources")).toBeInTheDocument()
  })
})
