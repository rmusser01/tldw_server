// @vitest-environment jsdom

import React from "react"
import { cleanup, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { WatchlistsPlaygroundPage } from "../WatchlistsPlaygroundPage"

const IA_STORAGE_KEY = "watchlists:ia-experiment:v1"
const IA_ROLLOUT_STORAGE_KEY = "watchlists:ia-rollout:v1"

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
    openSourceFormMock: vi.fn(),
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
  const Alert = ({ title, description, closable, onClose }: any) => (
    <div>
      <div>{title}</div>
      <div>{description}</div>
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

  const Modal = ({ open, title, children, footer }: any) =>
    open ? (
      <div>
        <h3>{title}</h3>
        {children}
        <div>{footer}</div>
      </div>
    ) : null
  const Drawer = ({ open, title, children }: any) =>
    open ? (
      <div>
        <h3>{title}</h3>
        {children}
      </div>
    ) : null

  const Empty = ({ description }: any) => <div>{description}</div>
  const Tooltip = ({ children }: any) => <>{children}</>
  const Button = ({ children, onClick, disabled, ...rest }: any) => (
    <button type="button" onClick={() => onClick?.()} disabled={Boolean(disabled)} {...rest}>
      {children}
    </button>
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
      openSourceForm: mocks.openSourceFormMock,
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

describe("WatchlistsPlaygroundPage experimental IA", () => {
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
    mocks.state.activeTab = "sources"
    localStorage.removeItem(IA_STORAGE_KEY)
    localStorage.removeItem(IA_ROLLOUT_STORAGE_KEY)
    ;(window as { __TLDW_WATCHLISTS_IA_EXPERIMENT__?: unknown }).__TLDW_WATCHLISTS_IA_EXPERIMENT__ = true
  })

  afterEach(() => {
    cleanup()
    localStorage.removeItem(IA_STORAGE_KEY)
    localStorage.removeItem(IA_ROLLOUT_STORAGE_KEY)
    delete (window as { __TLDW_WATCHLISTS_IA_EXPERIMENT__?: unknown }).__TLDW_WATCHLISTS_IA_EXPERIMENT__
    delete (
      window as { __TLDW_WATCHLISTS_IA_EXPERIMENT_ROLLOUT_PERCENT__?: unknown }
    ).__TLDW_WATCHLISTS_IA_EXPERIMENT_ROLLOUT_PERCENT__
  })

  it("shows task-centered primary tabs and exposes implementation tabs via More views", () => {
    render(<WatchlistsPlaygroundPage />)

    expect(screen.getByTestId("watchlists-tab-overview")).toBeInTheDocument()
    expect(screen.getByTestId("watchlists-tab-sources")).toBeInTheDocument()
    expect(screen.getByTestId("watchlists-tab-items")).toBeInTheDocument()
    expect(screen.getByTestId("watchlists-tab-outputs")).toBeInTheDocument()
    expect(screen.getByTestId("watchlists-tab-settings")).toBeInTheDocument()

    expect(screen.queryByTestId("watchlists-tab-jobs")).not.toBeInTheDocument()
    expect(screen.queryByTestId("watchlists-tab-runs")).not.toBeInTheDocument()
    expect(screen.queryByTestId("watchlists-tab-templates")).not.toBeInTheDocument()

    fireEvent.click(screen.getByTestId("watchlists-experimental-tab-jobs"))
    expect(mocks.state.setActiveTab).toHaveBeenCalledWith("jobs")
    fireEvent.click(screen.getByTestId("watchlists-experimental-tab-runs"))
    expect(mocks.state.setActiveTab).toHaveBeenCalledWith("runs")
  })

  it("keeps hidden tabs reachable when currently selected", () => {
    mocks.state.activeTab = "templates"

    render(<WatchlistsPlaygroundPage />)

    expect(screen.getByTestId("watchlists-tab-templates")).toBeInTheDocument()
  })

  it("routes task views to user outcomes and keeps legacy tabs mapped to the active task", () => {
    render(<WatchlistsPlaygroundPage />)

    fireEvent.click(screen.getByTestId("watchlists-task-view-collect"))
    fireEvent.click(screen.getByTestId("watchlists-task-view-review"))
    fireEvent.click(screen.getByTestId("watchlists-task-view-briefings"))

    expect(mocks.state.setActiveTab).toHaveBeenCalledWith("sources")
    expect(mocks.state.setActiveTab).toHaveBeenCalledWith("items")
    expect(mocks.state.setActiveTab).toHaveBeenCalledWith("outputs")

    cleanup()
    mocks.state.activeTab = "runs"
    render(<WatchlistsPlaygroundPage />)
    expect(screen.getByTestId("watchlists-task-view-review")).toHaveAttribute("aria-pressed", "true")

    cleanup()
    mocks.state.activeTab = "templates"
    render(<WatchlistsPlaygroundPage />)
    expect(screen.getByTestId("watchlists-task-view-briefings")).toHaveAttribute("aria-pressed", "true")
  })

  it("routes the new-entity keyboard shortcut to feeds before opening the source form", () => {
    mocks.state.activeTab = "items"

    render(<WatchlistsPlaygroundPage />)

    fireEvent.keyDown(document, { key: "n" })

    expect(mocks.state.setActiveTab).toHaveBeenCalledWith("sources")
    expect(mocks.openSourceFormMock).toHaveBeenCalledTimes(1)
  })

  it("records tab transition telemetry when experiment mode is active", () => {
    const { rerender } = render(<WatchlistsPlaygroundPage />)

    let payload = JSON.parse(localStorage.getItem(IA_STORAGE_KEY) || "{}")
    expect(payload.transitions).toBe(0)
    expect(payload.variant).toBe("experimental")
    expect(payload.visited_tabs).toContain("sources")

    mocks.state.activeTab = "runs"
    rerender(<WatchlistsPlaygroundPage />)
    payload = JSON.parse(localStorage.getItem(IA_STORAGE_KEY) || "{}")
    expect(payload.transitions).toBe(1)
    expect(payload.visited_tabs).toContain("sources")
    expect(payload.visited_tabs).toContain("runs")
    expect(mocks.recordWatchlistsIaExperimentTelemetryMock).toHaveBeenCalled()
  })

  it("can opt into the full tab map and records baseline telemetry when experiment is disabled", () => {
    ;(window as { __TLDW_WATCHLISTS_IA_EXPERIMENT__?: unknown }).__TLDW_WATCHLISTS_IA_EXPERIMENT__ = false

    render(<WatchlistsPlaygroundPage />)

    fireEvent.click(screen.getByTestId("watchlists-show-all-views-toggle"))

    expect(screen.getByTestId("watchlists-tab-jobs")).toBeInTheDocument()
    expect(screen.getByTestId("watchlists-tab-items")).toBeInTheDocument()
    expect(screen.getByTestId("watchlists-tab-templates")).toBeInTheDocument()
    expect(screen.queryByTestId("watchlists-experimental-tab-jobs")).not.toBeInTheDocument()
    const payload = JSON.parse(localStorage.getItem(IA_STORAGE_KEY) || "{}")
    expect(payload.variant).toBe("baseline")
    expect(payload.visited_tabs).toContain("sources")
    expect(mocks.recordWatchlistsIaExperimentTelemetryMock).toHaveBeenCalledWith(
      expect.objectContaining({ variant: "baseline" })
    )
  })

  it("honors rollout percentage when runtime override is absent", () => {
    delete (window as { __TLDW_WATCHLISTS_IA_EXPERIMENT__?: unknown }).__TLDW_WATCHLISTS_IA_EXPERIMENT__
    ;(
      window as { __TLDW_WATCHLISTS_IA_EXPERIMENT_ROLLOUT_PERCENT__?: unknown }
    ).__TLDW_WATCHLISTS_IA_EXPERIMENT_ROLLOUT_PERCENT__ = 100

    render(<WatchlistsPlaygroundPage />)

    expect(screen.getByTestId("watchlists-experimental-tab-jobs")).toBeInTheDocument()
    expect(screen.queryByTestId("watchlists-tab-jobs")).not.toBeInTheDocument()

    const payload = JSON.parse(localStorage.getItem(IA_STORAGE_KEY) || "{}")
    expect(payload.variant).toBe("experimental")
    const rolloutSnapshot = JSON.parse(localStorage.getItem(IA_ROLLOUT_STORAGE_KEY) || "{}")
    expect(rolloutSnapshot.variant).toBe("experimental")
  })
})
