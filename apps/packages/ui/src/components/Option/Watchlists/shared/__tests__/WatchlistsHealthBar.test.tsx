// @vitest-environment jsdom

import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { WatchlistsHealthBar } from "../WatchlistsHealthBar"
import type { WatchlistsOverviewData } from "@/services/watchlists-overview"

const healthBarMocks = vi.hoisted(() => ({
  fetchOverviewMock: vi.fn(),
  setOverviewHealth: vi.fn(),
  setActiveTab: vi.fn(),
  overviewHealth: null as WatchlistsOverviewData["health"] | null
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

vi.mock("antd", () => ({
  Button: ({ children, onClick, danger: _danger, ...rest }: any) => (
    <button type="button" onClick={(event) => onClick?.(event)} {...rest}>
      {children}
    </button>
  ),
  Spin: () => <span data-testid="health-bar-spinner" />,
  Tag: ({ children, onClick, ...rest }: any) => (
    <button type="button" onClick={onClick} {...rest}>
      {children}
    </button>
  ),
  Tooltip: ({ children }: any) => <>{children}</>
}))

vi.mock("@/services/watchlists-overview", () => ({
  fetchWatchlistsOverviewData: () => healthBarMocks.fetchOverviewMock()
}))

vi.mock("@/store/watchlists", () => ({
  useWatchlistsStore: (selector: (state: Record<string, unknown>) => unknown) =>
    selector({
      setOverviewHealth: healthBarMocks.setOverviewHealth,
      setActiveTab: healthBarMocks.setActiveTab,
      overviewHealth: healthBarMocks.overviewHealth
    })
}))

const buildHealth = (overrides: Partial<WatchlistsOverviewData["health"]> = {}): WatchlistsOverviewData["health"] => ({
  statuses: {
    sources: "unknown",
    jobs: "unknown",
    runs: "unknown",
    outputs: "unknown"
  },
  attention: {
    sources: 0,
    jobs: 0,
    runs: 0,
    outputs: 0,
    total: 0
  },
  tabBadges: {
    sources: 0,
    runs: 0,
    outputs: 0
  },
  ...overrides
})

const buildOverviewData = (
  overrides: Partial<WatchlistsOverviewData> = {}
): WatchlistsOverviewData => ({
  fetchedAt: "2026-05-19T12:00:00Z",
  sources: {
    total: 0,
    healthy: 0,
    degraded: 0,
    inactive: 0,
    unknown: 0
  },
  jobs: {
    total: 0,
    active: 0,
    nextRunAt: null,
    attention: 0
  },
  items: {
    unread: 0
  },
  alerts: {
    unread: 0
  },
  runs: {
    running: 0,
    pending: 0,
    failed: 0,
    recentFailed: []
  },
  outputs: {
    total: 0,
    expired: 0,
    deliveryIssues: 0,
    attention: 0
  },
  health: buildHealth(),
  systemHealth: "healthy",
  ...overrides
})

describe("WatchlistsHealthBar", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    healthBarMocks.overviewHealth = null
    healthBarMocks.fetchOverviewMock.mockResolvedValue(buildOverviewData())
  })

  it("turns the empty health summary into concrete setup actions", async () => {
    const onNavigate = vi.fn()
    render(<WatchlistsHealthBar onNavigate={onNavigate} />)

    await waitFor(() => {
      expect(screen.getByTestId("watchlists-health-bar-summary")).toHaveTextContent(
        "No watchlist data yet"
      )
    })

    fireEvent.click(screen.getByTestId("watchlists-health-setup-feeds"))
    fireEvent.click(screen.getByTestId("watchlists-health-setup-monitors"))

    expect(onNavigate).toHaveBeenCalledWith("sources")
    expect(onNavigate).toHaveBeenCalledWith("jobs")
  })

  it("places failed run recovery next to the Activity health state", async () => {
    healthBarMocks.overviewHealth = buildHealth({
      statuses: {
        sources: "healthy",
        jobs: "healthy",
        runs: "attention",
        outputs: "healthy"
      },
      attention: {
        sources: 0,
        jobs: 0,
        runs: 2,
        outputs: 0,
        total: 2
      },
      tabBadges: {
        sources: 0,
        runs: 2,
        outputs: 0
      }
    })
    healthBarMocks.fetchOverviewMock.mockResolvedValue(
      buildOverviewData({
        sources: {
          total: 3,
          healthy: 3,
          degraded: 0,
          inactive: 0,
          unknown: 0
        },
        jobs: {
          total: 2,
          active: 2,
          nextRunAt: null,
          attention: 0
        },
        runs: {
          running: 0,
          pending: 0,
          failed: 2,
          recentFailed: []
        }
      })
    )
    const onNavigate = vi.fn()
    render(<WatchlistsHealthBar onNavigate={onNavigate} />)

    await waitFor(() => {
      expect(screen.getByTestId("watchlists-health-bar-summary")).toHaveTextContent("2 failed")
    })

    fireEvent.click(screen.getByLabelText("Toggle health bar"))
    fireEvent.click(screen.getByTestId("watchlists-health-open-activity"))

    expect(onNavigate).toHaveBeenCalledWith("runs")
  })
})
