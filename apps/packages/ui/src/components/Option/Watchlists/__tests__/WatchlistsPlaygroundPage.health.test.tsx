// @vitest-environment jsdom

import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { OverviewTab } from "../OverviewTab/OverviewTab"
import type { WatchlistsOverviewData } from "@/services/watchlists-overview"

const healthMocks = vi.hoisted(() => ({
  fetchOverview: vi.fn(),
  setActiveTab: vi.fn(),
  setRunsStatusFilter: vi.fn(),
  setOverviewHealth: vi.fn(),
  openRunDetail: vi.fn(),
  openOutputPreview: vi.fn(),
  openSourceForm: vi.fn(),
  openJobForm: vi.fn()
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
  Alert: ({ title, description, action }: any) => (
    <section>
      <h2>{title}</h2>
      {description && <p>{description}</p>}
      {action}
    </section>
  ),
  Button: ({ children, onClick, disabled, danger: _danger, loading: _loading, ...rest }: any) => (
    <button type="button" disabled={disabled} onClick={(event) => onClick?.(event)} {...rest}>
      {children}
    </button>
  ),
  Card: ({ children, title, extra }: any) => (
    <section>
      {title && <h3>{title}</h3>}
      {extra}
      {children}
    </section>
  ),
  Checkbox: Object.assign(({ children }: any) => <label>{children}</label>, {
    Group: ({ children }: any) => <div>{children}</div>
  }),
  Empty: () => <div />,
  Form: Object.assign(
    ({ children }: any) => <form>{children}</form>,
    {
      Item: ({ children }: any) => <div>{children}</div>,
      useForm: () => [
        {
          getFieldsValue: () => ({}),
          getFieldValue: () => undefined,
          setFieldsValue: vi.fn(),
          setFields: vi.fn(),
          validateFields: vi.fn(),
          resetFields: vi.fn()
        }
      ],
      useWatch: () => undefined
    }
  ),
  Input: Object.assign(({ children }: any) => <input>{children}</input>, {
    TextArea: () => <textarea />
  }),
  List: () => <div />,
  Modal: ({ children, open }: any) => (open ? <div>{children}</div> : null),
  Select: Object.assign(() => <select />, {
    Option: ({ children }: any) => <option>{children}</option>
  }),
  Space: ({ children }: any) => <div>{children}</div>,
  Spin: () => <span data-testid="overview-spinner" />,
  Statistic: ({ title, value, suffix }: any) => (
    <div>
      <span>{title}</span>
      <strong>
        {value}
        {suffix}
      </strong>
    </div>
  ),
  Steps: () => <div />,
  Switch: () => <button type="button" />,
  Tag: ({ children }: any) => <span>{children}</span>,
  Tooltip: ({ children }: any) => <>{children}</>
}))

vi.mock("@/services/watchlists-overview", () => ({
  fetchWatchlistsOverviewData: () => healthMocks.fetchOverview(),
  getOverviewTabBadges: (model: WatchlistsOverviewData["health"] | null | undefined) =>
    model?.tabBadges || { sources: 0, runs: 0, outputs: 0 }
}))

vi.mock("@/services/watchlists", () => ({
  bulkCreateSources: vi.fn(),
  createWatchlistOutput: vi.fn(),
  createWatchlistJob: vi.fn(),
  createWatchlistSource: vi.fn(),
  deleteWatchlistJob: vi.fn(),
  fetchWatchlistRuns: vi.fn(),
  fetchWatchlistSources: vi.fn(),
  getWatchlistTemplate: vi.fn(),
  previewWatchlistTemplate: vi.fn(),
  testWatchlistSourceDraft: vi.fn(),
  triggerWatchlistRun: vi.fn()
}))

vi.mock("@/store/watchlists", () => ({
  useWatchlistsStore: (selector: (state: Record<string, unknown>) => unknown) =>
    selector({
      setActiveTab: healthMocks.setActiveTab,
      setRunsStatusFilter: healthMocks.setRunsStatusFilter,
      setOverviewHealth: healthMocks.setOverviewHealth,
      openRunDetail: healthMocks.openRunDetail,
      openOutputPreview: healthMocks.openOutputPreview,
      openSourceForm: healthMocks.openSourceForm,
      openJobForm: healthMocks.openJobForm
    })
}))

vi.mock("@/utils/watchlists-onboarding-telemetry", () => ({
  trackWatchlistsOnboardingTelemetry: vi.fn()
}))

const buildOverviewData = (): WatchlistsOverviewData => ({
  fetchedAt: "2026-05-20T12:00:00Z",
  sources: {
    total: 1,
    healthy: 0,
    degraded: 1,
    inactive: 0,
    unknown: 0
  },
  jobs: {
    total: 1,
    active: 1,
    nextRunAt: "2026-05-21T12:00:00Z",
    attention: 0
  },
  items: {
    unread: 0
  },
  runs: {
    running: 0,
    pending: 0,
    failed: 0,
    sourceErrors: 1,
    zeroItemSourceErrors: 1,
    recentFailed: []
  },
  outputs: {
    total: 1,
    expired: 0,
    deliveryIssues: 0,
    audioIssues: 1,
    attention: 1
  },
  health: {
    statuses: {
      sources: "attention",
      jobs: "healthy",
      runs: "attention",
      outputs: "attention"
    },
    attention: {
      sources: 1,
      jobs: 0,
      runs: 1,
      outputs: 1,
      total: 3
    },
    tabBadges: {
      sources: 1,
      runs: 1,
      outputs: 1
    }
  },
  systemHealth: "degraded"
})

describe("Watchlists health truthfulness", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    healthMocks.fetchOverview.mockResolvedValue(buildOverviewData())
  })

  it("surfaces source, activity, and report warnings instead of System healthy", async () => {
    render(<OverviewTab />)

    await waitFor(() => {
      expect(screen.getByText("System requires attention")).toBeInTheDocument()
    })

    expect(screen.queryByText("System healthy")).not.toBeInTheDocument()
    expect(screen.getByTestId("watchlists-overview-attention-sources")).toHaveTextContent(
      "Feeds need review (1)"
    )
    expect(screen.getByTestId("watchlists-overview-attention-runs")).toHaveTextContent(
      "Activity needs review (1)"
    )
    expect(screen.getByTestId("watchlists-overview-attention-outputs")).toHaveTextContent(
      "Reports need review (1)"
    )

    fireEvent.click(screen.getByTestId("watchlists-overview-attention-runs"))
    expect(healthMocks.setActiveTab).toHaveBeenCalledWith("runs")
  })
})
