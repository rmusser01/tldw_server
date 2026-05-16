// @vitest-environment jsdom

import React from "react"
import { render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { WatchlistsHealthBar } from "../WatchlistsHealthBar"

const fetchWatchlistsOverviewDataMock = vi.hoisted(() => vi.fn())

const watchlistsStoreState = vi.hoisted(() => ({
  overviewHealth: {
    statuses: {
      sources: "attention",
      jobs: "attention",
      runs: "attention",
      outputs: "attention"
    },
    attention: {
      total: 10,
      sources: 2,
      jobs: 4,
      runs: 1,
      outputs: 3
    },
    tabBadges: {
      sources: 2,
      runs: 1,
      outputs: 3
    }
  },
  setOverviewHealth: vi.fn(),
  setActiveTab: vi.fn()
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: unknown, options?: Record<string, unknown>) => {
      if (typeof fallback !== "string") return _key
      const values =
        options && typeof options === "object"
          ? options
          : fallback && typeof fallback === "object"
            ? fallback as Record<string, unknown>
            : {}
      return fallback.replace(/\{\{(\w+)\}\}/g, (_, token) => String(values[token] ?? ""))
    }
  })
}))

vi.mock("@/services/watchlists-overview", () => ({
  fetchWatchlistsOverviewData: fetchWatchlistsOverviewDataMock
}))

vi.mock("@/store/watchlists", () => ({
  useWatchlistsStore: (selector: (state: typeof watchlistsStoreState) => unknown) =>
    selector(watchlistsStoreState)
}))

vi.mock("antd", () => ({
  Button: ({ children, icon, onClick, ...props }: any) => (
    <button type="button" onClick={onClick} {...props}>
      {icon}
      {children}
    </button>
  ),
  Spin: () => <span role="status">Loading</span>,
  Tag: ({ children, onClick, ...props }: any) => (
    <span onClick={onClick} {...props}>
      {children}
    </span>
  ),
  Tooltip: ({ children }: any) => <>{children}</>
}))

const makeOverviewData = () => ({
  fetchedAt: "2026-05-16T12:00:00.000Z",
  sources: {
    total: 5,
    healthy: 3,
    degraded: 2,
    inactive: 0,
    unknown: 0
  },
  jobs: {
    total: 6,
    active: 4,
    nextRunAt: null,
    attention: 4
  },
  items: {
    unread: 7
  },
  runs: {
    running: 0,
    pending: 0,
    failed: 1,
    recentFailed: []
  },
  outputs: {
    total: 3,
    expired: 0,
    deliveryIssues: 3,
    attention: 3
  },
  health: watchlistsStoreState.overviewHealth,
  systemHealth: "degraded" as const
})

describe("WatchlistsHealthBar", () => {
  beforeEach(() => {
    localStorage.clear()
    localStorage.setItem("watchlists:health-bar-expanded:v1", "true")
    fetchWatchlistsOverviewDataMock.mockReset()
    fetchWatchlistsOverviewDataMock.mockResolvedValue(makeOverviewData())
    watchlistsStoreState.setOverviewHealth.mockClear()
    watchlistsStoreState.setActiveTab.mockClear()
  })

  it("renders attention items as design-system badges and preserves tab navigation", async () => {
    const onNavigate = vi.fn()
    const user = userEvent.setup()

    render(<WatchlistsHealthBar onNavigate={onNavigate} />)

    const attention = await screen.findByTestId("watchlists-health-bar-attention")

    const expectedBadges = [
      { label: "Feeds need review (2)", variant: "warning", tab: "sources" },
      { label: "Failed activity runs (1)", variant: "danger", tab: "runs" },
      { label: "Reports with delivery issues (3)", variant: "warning", tab: "outputs" },
      { label: "Monitors need schedule fixes (4)", variant: "warning", tab: "jobs" }
    ]

    for (const { label, variant } of expectedBadges) {
      const labelNode = within(attention).getByText(label)
      const badge = labelNode.closest('[data-ds-component="Badge"]')
      expect(badge).toHaveAttribute("data-ds-variant", variant)
    }

    for (const { label, tab } of expectedBadges) {
      const control = within(attention).getByRole("button", { name: label })
      await user.click(control)
      expect(onNavigate).toHaveBeenLastCalledWith(tab)
    }

    await waitFor(() => {
      expect(fetchWatchlistsOverviewDataMock).toHaveBeenCalledTimes(1)
    })
  })
})
