// @vitest-environment jsdom

import React from "react"
import { render, waitFor } from "@testing-library/react"
import { MemoryRouter } from "react-router-dom"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { WatchlistsPlaygroundPage } from "../WatchlistsPlaygroundPage"

const mocks = vi.hoisted(() => {
  const state = {
    activeTab: "sources" as
      | "overview"
      | "sources"
      | "jobs"
      | "runs"
      | "items"
      | "outputs"
      | "templates"
      | "settings",
    pollingActive: false,
    setActiveTab: vi.fn((next: string) => {
      state.activeTab = next as typeof state.activeTab
    })
  }
  return {
    fetchWatchlistRunsMock: vi.fn(),
    trackWatchlistsOnboardingTelemetryMock: vi.fn(),
    notificationDestroyMock: vi.fn(),
    notificationErrorMock: vi.fn(),
    notificationSuccessMock: vi.fn(),
    notificationWarningMock: vi.fn(),
    openRunDetailMock: vi.fn(),
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
  const Alert = ({ title, description }: any) => (
    <div>
      <div>{title}</div>
      <div>{description}</div>
    </div>
  )
  const Tabs = ({ items = [] }: any) => (
    <div>
      {items.map((item: any) => (
        <button key={item.key} type="button">
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
  const Button = ({ children, onClick, ...rest }: any) => (
    <button type="button" onClick={() => onClick?.()} {...rest}>
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
  return { ...actual, Alert, Tabs, Modal, Drawer, Empty, Button, Tooltip, Switch }
})

vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => ({
    destroy: (...args: unknown[]) => mocks.notificationDestroyMock(...args),
    success: (...args: unknown[]) => mocks.notificationSuccessMock(...args),
    error: (...args: unknown[]) => mocks.notificationErrorMock(...args),
    warning: (...args: unknown[]) => mocks.notificationWarningMock(...args)
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
  fetchWatchlistRuns: (...args: unknown[]) => mocks.fetchWatchlistRunsMock(...args)
}))

vi.mock("@/store/watchlists", () => ({
  useWatchlistsStore: (selector: (state: Record<string, unknown>) => unknown) =>
    selector({
      activeTab: mocks.state.activeTab,
      pollingActive: mocks.state.pollingActive,
      setActiveTab: mocks.state.setActiveTab,
      openRunDetail: mocks.openRunDetailMock,
      resetStore: vi.fn()
    })
}))

vi.mock("@/utils/watchlists-ia-experiment-telemetry", () => ({
  trackWatchlistsIaExperimentTransition: vi.fn()
}))

vi.mock("@/utils/watchlists-onboarding-telemetry", () => ({
  trackWatchlistsOnboardingTelemetry: (...args: unknown[]) =>
    mocks.trackWatchlistsOnboardingTelemetryMock(...args)
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

const renderWatchlistsPlaygroundPage = () =>
  render(
    <MemoryRouter initialEntries={["/watchlists"]}>
      <WatchlistsPlaygroundPage />
    </MemoryRouter>
  )

describe("WatchlistsPlaygroundPage run notifications", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    connectionMocks.useConnectionUxState.mockReturnValue({
      uxState: "connected_ok",
      hasCompletedFirstRun: true
    })
    mocks.state.activeTab = "sources"
    mocks.state.pollingActive = false
    ;(window as { __TLDW_WATCHLISTS_RUN_NOTIFICATIONS_POLL_MS?: unknown })
      .__TLDW_WATCHLISTS_RUN_NOTIFICATIONS_POLL_MS = 100
    ;(window as { __TLDW_WATCHLISTS_IA_EXPERIMENT__?: unknown }).__TLDW_WATCHLISTS_IA_EXPERIMENT__ = false
    mocks.fetchWatchlistRunsMock
      .mockResolvedValueOnce({
        items: [
          {
            id: 10,
            job_id: 1,
            status: "running",
            started_at: "2026-02-18T09:55:00Z",
            finished_at: null,
            error_msg: null,
            stats: {}
          },
          {
            id: 12,
            job_id: 2,
            status: "running",
            started_at: "2026-02-18T09:54:00Z",
            finished_at: null,
            error_msg: null,
            stats: {}
          }
        ],
        total: 2,
        has_more: false
      })
      .mockResolvedValue({
        items: [
          {
            id: 10,
            job_id: 1,
            status: "failed",
            started_at: "2026-02-18T09:55:00Z",
            finished_at: "2026-02-18T10:00:00Z",
            error_msg: "timeout while fetching",
            stats: {}
          },
          {
            id: 12,
            job_id: 2,
            status: "failed",
            started_at: "2026-02-18T09:54:00Z",
            finished_at: "2026-02-18T10:00:00Z",
            error_msg: "dns lookup failed",
            stats: {}
          }
        ],
        total: 2,
        has_more: false
      })
  })

  afterEach(() => {
    vi.useRealTimers()
    delete (window as { __TLDW_WATCHLISTS_RUN_NOTIFICATIONS_POLL_MS?: unknown })
      .__TLDW_WATCHLISTS_RUN_NOTIFICATIONS_POLL_MS
    delete (window as { __TLDW_WATCHLISTS_IA_EXPERIMENT__?: unknown }).__TLDW_WATCHLISTS_IA_EXPERIMENT__
  })

  it("groups repeat failures into one notification and deep-links to the newest run", async () => {
    renderWatchlistsPlaygroundPage()

    await waitFor(() => {
      expect(mocks.notificationErrorMock).toHaveBeenCalledTimes(1)
    }, { timeout: 3000 })

    const config = mocks.notificationErrorMock.mock.calls[0][0] as {
      description?: string
      onClick?: () => void
      key?: string
    }
    expect(String(config.description || "")).toContain("2 runs failed")

    config.onClick?.()

    expect(mocks.state.setActiveTab).toHaveBeenCalledWith("items")
    expect(mocks.openRunDetailMock).toHaveBeenCalledWith(12)
    expect(mocks.notificationDestroyMock).toHaveBeenCalledWith(config.key)
  })

  it("suppresses run-notification polling while Activity tab already auto-refreshes", async () => {
    mocks.state.activeTab = "runs"
    mocks.state.pollingActive = true
    mocks.fetchWatchlistRunsMock.mockReset()

    renderWatchlistsPlaygroundPage()

    await new Promise((resolve) => {
      setTimeout(resolve, 250)
    })

    expect(mocks.fetchWatchlistRunsMock).not.toHaveBeenCalled()
  })

  it("dedupes overlapping polling requests when previous fetch is still in flight", async () => {
    let resolveFirstPoll: ((value: unknown) => void) | null = null
    mocks.fetchWatchlistRunsMock.mockReset()
    mocks.fetchWatchlistRunsMock
      .mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            resolveFirstPoll = resolve
          })
      )
      .mockResolvedValue({
        items: [],
        total: 0,
        has_more: false
      })

    renderWatchlistsPlaygroundPage()

    await new Promise((resolve) => {
      setTimeout(resolve, 250)
    })

    expect(mocks.fetchWatchlistRunsMock).toHaveBeenCalledTimes(1)

    resolveFirstPoll?.({
      items: [],
      total: 0,
      has_more: false
    })

    await waitFor(() => {
      expect(mocks.fetchWatchlistRunsMock).toHaveBeenCalledTimes(2)
    }, { timeout: 3000 })
  })

  it("tracks first successful run once polling sees a completed run", async () => {
    mocks.fetchWatchlistRunsMock.mockReset()
    mocks.fetchWatchlistRunsMock.mockResolvedValue({
      items: [
        {
          id: 77,
          job_id: 9,
          status: "completed",
          started_at: "2026-02-18T09:58:00Z",
          finished_at: "2026-02-18T10:01:00Z",
          error_msg: null,
          stats: {}
        }
      ],
      total: 1,
      has_more: false
    })

    renderWatchlistsPlaygroundPage()

    await waitFor(() => {
      expect(mocks.trackWatchlistsOnboardingTelemetryMock).toHaveBeenCalledWith({
        type: "first_run_succeeded",
        runId: 77
      })
    }, { timeout: 3000 })
  })
})
