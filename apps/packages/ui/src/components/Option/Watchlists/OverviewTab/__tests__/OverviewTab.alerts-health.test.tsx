// @vitest-environment jsdom

import React from "react"
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { OverviewTab } from "../OverviewTab"

const mocks = vi.hoisted(() => ({
  fetchOverviewMock: vi.fn(),
  fetchWatchlistSourcesMock: vi.fn(),
  fetchWatchlistRunsMock: vi.fn(),
  bulkCreateSourcesMock: vi.fn(),
  createWatchlistSourceMock: vi.fn(),
  createWatchlistJobMock: vi.fn(),
  deleteWatchlistJobMock: vi.fn(),
  deleteWatchlistSourceMock: vi.fn(),
  triggerWatchlistRunMock: vi.fn(),
  retryWatchlistBriefingStageMock: vi.fn(),
  createWatchlistOutputMock: vi.fn(),
  getWatchlistTemplateMock: vi.fn(),
  previewWatchlistTemplateMock: vi.fn(),
  testWatchlistSourceDraftMock: vi.fn(),
  trackWatchlistsOnboardingTelemetryMock: vi.fn(),
  setActiveTabMock: vi.fn(),
  setOutputsRunFilterMock: vi.fn(),
  setRunsStatusFilterMock: vi.fn(),
  setOverviewHealthMock: vi.fn(),
  openSourceFormMock: vi.fn(),
  openJobFormMock: vi.fn(),
  openRunDetailMock: vi.fn(),
  openOutputPreviewMock: vi.fn(),
  selectedWatchlistId: 42 as number | null
}))

vi.mock("react-i18next", () => {
  const t = (
      key: string,
      fallbackOrOptions?: string | { defaultValue?: string },
      maybeOptions?: Record<string, unknown>
    ) => {
      if (typeof fallbackOrOptions === "string") {
        return fallbackOrOptions.replace(/\{\{(\w+)\}\}/g, (_match, token) => {
          const value = maybeOptions?.[token]
          return value == null ? "" : String(value)
        })
      }
      if (fallbackOrOptions && typeof fallbackOrOptions === "object") {
        const fallback = fallbackOrOptions.defaultValue
        if (typeof fallback === "string") return fallback
      }
      return key
    }
  return {
    useTranslation: () => ({
      i18n: { resolvedLanguage: "en-US", language: "en-US", dir: () => "ltr" },
      t
    })
  }
})

vi.mock("@/services/watchlists-overview", () => ({
  fetchWatchlistsOverviewData: (...args: unknown[]) => mocks.fetchOverviewMock(...args),
  getOverviewTabBadges: (model?: { tabBadges?: { sources?: number; runs?: number; outputs?: number } } | null) => ({
    sources: Number(model?.tabBadges?.sources || 0),
    runs: Number(model?.tabBadges?.runs || 0),
    outputs: Number(model?.tabBadges?.outputs || 0)
  })
}))

vi.mock("@/services/watchlists", () => ({
  fetchWatchlistSources: (...args: unknown[]) => mocks.fetchWatchlistSourcesMock(...args),
  fetchWatchlistRuns: (...args: unknown[]) => mocks.fetchWatchlistRunsMock(...args),
  bulkCreateSources: (...args: unknown[]) => mocks.bulkCreateSourcesMock(...args),
  createWatchlistSource: (...args: unknown[]) => mocks.createWatchlistSourceMock(...args),
  createWatchlistJob: (...args: unknown[]) => mocks.createWatchlistJobMock(...args),
  deleteWatchlistJob: (...args: unknown[]) => mocks.deleteWatchlistJobMock(...args),
  deleteWatchlistSource: (...args: unknown[]) => mocks.deleteWatchlistSourceMock(...args),
  triggerWatchlistRun: (...args: unknown[]) => mocks.triggerWatchlistRunMock(...args),
  retryWatchlistBriefingStage: (...args: unknown[]) =>
    mocks.retryWatchlistBriefingStageMock(...args),
  createWatchlistOutput: (...args: unknown[]) => mocks.createWatchlistOutputMock(...args),
  getWatchlistTemplate: (...args: unknown[]) => mocks.getWatchlistTemplateMock(...args),
  previewWatchlistTemplate: (...args: unknown[]) => mocks.previewWatchlistTemplateMock(...args),
  testWatchlistSourceDraft: (...args: unknown[]) => mocks.testWatchlistSourceDraftMock(...args)
}))

vi.mock("@/utils/watchlists-onboarding-telemetry", () => ({
  trackWatchlistsOnboardingTelemetry: (...args: unknown[]) =>
    mocks.trackWatchlistsOnboardingTelemetryMock(...args)
}))

vi.mock("@/store/watchlists", () => ({
  useWatchlistsStore: (selector: (state: Record<string, unknown>) => unknown) =>
    selector({
      setActiveTab: mocks.setActiveTabMock,
      setOutputsRunFilter: mocks.setOutputsRunFilterMock,
      setRunsStatusFilter: mocks.setRunsStatusFilterMock,
      setOverviewHealth: mocks.setOverviewHealthMock,
      openSourceForm: mocks.openSourceFormMock,
      openJobForm: mocks.openJobFormMock,
      openRunDetail: mocks.openRunDetailMock,
      openOutputPreview: mocks.openOutputPreviewMock,
      selectedWatchlistId: mocks.selectedWatchlistId
    })
}))

vi.mock("@/utils/dateFormatters", () => ({
  formatRelativeTime: () => "just now"
}))

const createOverviewPayload = (overrides?: Partial<Record<string, unknown>>) => ({
  fetchedAt: "2026-05-15T12:00:00Z",
  latestBriefing: null,
  sources: {
    total: 1,
    healthy: 1,
    degraded: 0,
    inactive: 0,
    unknown: 0,
    ...(overrides?.sources as Record<string, unknown> | undefined)
  },
  jobs: {
    total: 1,
    active: 1,
    nextRunAt: null,
    attention: 0,
    ...(overrides?.jobs as Record<string, unknown> | undefined)
  },
  items: {
    unread: 0,
    ...(overrides?.items as Record<string, unknown> | undefined)
  },
  alerts: {
    unread: 0,
    ...(overrides?.alerts as Record<string, unknown> | undefined)
  },
  runs: {
    running: 0,
    pending: 0,
    failed: 0,
    recentFailed: [],
    ...(overrides?.runs as Record<string, unknown> | undefined)
  },
  outputs: {
    total: 0,
    expired: 0,
    deliveryIssues: 0,
    attention: 0,
    ...(overrides?.outputs as Record<string, unknown> | undefined)
  },
  health: {
    statuses: {
      sources: "healthy",
      jobs: "healthy",
      runs: "unknown",
      outputs: "unknown"
    },
    attention: {
      total: 0,
      sources: 0,
      jobs: 0,
      runs: 0,
      outputs: 0
    },
    tabBadges: {
      sources: 0,
      runs: 0,
      outputs: 0
    },
    ...(overrides?.health as Record<string, unknown> | undefined)
  },
  systemHealth: "healthy" as const,
  ...overrides
})

describe("OverviewTab alert and health summary", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.fetchOverviewMock.mockReset()
    mocks.selectedWatchlistId = 42
    mocks.trackWatchlistsOnboardingTelemetryMock.mockResolvedValue(undefined)
    mocks.fetchWatchlistSourcesMock.mockResolvedValue({ items: [], total: 0 })
    mocks.fetchWatchlistRunsMock.mockResolvedValue({ items: [], total: 0 })
  })

  it("shows unread content alerts separately from pipeline health issues", async () => {
    mocks.fetchOverviewMock.mockResolvedValue(createOverviewPayload({
      alerts: { unread: 4 },
      runs: { failed: 2, recentFailed: [] },
      health: {
        attention: {
          total: 2,
          sources: 0,
          jobs: 0,
          runs: 2,
          outputs: 0
        },
        tabBadges: {
          sources: 0,
          runs: 2,
          outputs: 0
        }
      },
      systemHealth: "degraded"
    }))

    render(<OverviewTab />)

    const summary = await screen.findByTestId("watchlists-overview-alert-health-summary")
    const contentAlerts = within(summary).getByTestId("watchlists-overview-content-alerts")
    const healthIssues = within(summary).getByTestId("watchlists-overview-health-issues")

    expect(contentAlerts).toHaveTextContent("Unread content alerts")
    expect(contentAlerts).toHaveTextContent("4")
    expect(contentAlerts).toHaveTextContent("New updates matching your Watchlist alert rules.")
    expect(healthIssues).toHaveTextContent("Health issues")
    expect(healthIssues).toHaveTextContent("2")
    expect(healthIssues).toHaveTextContent("Run failures and source problems are health issues, not content alerts.")

    fireEvent.click(within(contentAlerts).getByRole("button", { name: "Review alerts" }))
    expect(mocks.setActiveTabMock).toHaveBeenCalledWith("alerts")

    fireEvent.click(within(healthIssues).getByRole("button", { name: "Open Activity" }))
    expect(mocks.setActiveTabMock).toHaveBeenCalledWith("runs")
  })

  it("offers alert rule creation from the selected Watchlist empty state", async () => {
    mocks.fetchOverviewMock.mockResolvedValue(createOverviewPayload())

    render(<OverviewTab />)

    const contentAlerts = await screen.findByTestId("watchlists-overview-content-alerts")
    expect(contentAlerts).toHaveTextContent("No unread content alerts")

    fireEvent.click(within(contentAlerts).getByRole("button", { name: "Create content alert rule" }))
    expect(mocks.setActiveTabMock).toHaveBeenCalledWith("alerts")
  })

  it("renders Overview health and setup callouts with the design-system Alert", async () => {
    mocks.fetchOverviewMock.mockResolvedValue(createOverviewPayload())

    render(<OverviewTab />)

    const healthTitle = await screen.findByText("System healthy")
    expect(healthTitle.closest('[data-ds-component="Alert"]')).toBeInTheDocument()

    const setupTitle = screen.getByText("Setup complete")
    expect(setupTitle.closest('[data-ds-component="Alert"]')).toBeInTheDocument()
  })

  it("renders Overview load failures with the design-system Alert", async () => {
    const consoleErrorSpy = vi.spyOn(console, "error").mockImplementation(() => undefined)

    try {
      mocks.fetchOverviewMock.mockRejectedValue(new Error("overview unavailable"))

      render(<OverviewTab />)

      const failureTitle = await screen.findByText("Failed to load overview")
      expect(failureTitle.closest('[data-ds-component="Alert"]')).toBeInTheDocument()
    } finally {
      consoleErrorSpy.mockRestore()
    }
  })

  it("mounts one polite and one assertive briefing region and deduplicates poll refreshes", async () => {
    const running = {
      occurrence_id: 31,
      run_id: 123,
      job_id: 7,
      artifact_status: "running",
      delivery_status: "waiting_for_artifacts",
      stages: {
        persist_text: { status: "ready" },
        generate_audio: { status: "running" }
      },
      output: { id: 71, title: "Signal Check", created_at: "2026-07-10T19:20:00Z" },
      audio: null,
      editorial: { outcome_noun: "briefing", show_name: "Signal Check" },
      selection: { candidate_count: 5, included_count: 3, omitted_count: 2 },
      next_run_at: "2026-07-12T18:00:00-07:00",
      timezone: "America/Los_Angeles",
      recovery: { can_open_report: true }
    }
    const ready = {
      ...running,
      artifact_status: "ready",
      delivery_status: "delivered",
      stages: {
        persist_text: { status: "ready" },
        generate_audio: { status: "ready" },
        persist_audio: { status: "ready" }
      },
      audio: { run_id: 123, status: "completed", download_url: "/audio/123" }
    }
    mocks.fetchOverviewMock
      .mockResolvedValueOnce(createOverviewPayload({ latestBriefing: running }))
      .mockResolvedValueOnce(createOverviewPayload({ latestBriefing: ready }))
      .mockResolvedValueOnce(createOverviewPayload({ latestBriefing: ready }))

    render(<OverviewTab />)
    await screen.findByRole("heading", { name: "Latest briefing" })
    expect(screen.getAllByTestId("watchlists-overview-live-polite")).toHaveLength(1)
    expect(screen.getAllByTestId("watchlists-overview-live-assertive")).toHaveLength(1)

    fireEvent.click(screen.getByRole("button", { name: "Refresh" }))
    await waitFor(() => {
      expect(screen.getByTestId("watchlists-overview-live-polite")).toHaveTextContent(
        "Signal Check is ready. Audio and show notes are available."
      )
    })

    fireEvent.click(screen.getByRole("button", { name: "Refresh" }))
    await waitFor(() => expect(mocks.fetchOverviewMock).toHaveBeenCalledTimes(3))
    expect(screen.getByTestId("watchlists-overview-live-assertive")).toHaveTextContent("")
  })

  it("opens the exact briefing report and suppresses AbortError refresh noise", async () => {
    const projection = {
      occurrence_id: 31,
      run_id: 123,
      job_id: 7,
      artifact_status: "ready",
      delivery_status: "delivered",
      stages: { persist_text: { status: "ready" } },
      output: { id: 71, title: "Signal Check", created_at: "2026-07-10T19:20:00Z" },
      audio: null,
      editorial: { outcome_noun: "briefing", show_name: "Signal Check" },
      selection: { candidate_count: 5, included_count: 3, omitted_count: 2 },
      next_run_at: null,
      timezone: "UTC",
      recovery: { can_open_report: true }
    }
    mocks.fetchOverviewMock
      .mockResolvedValueOnce(createOverviewPayload({ latestBriefing: projection }))
      .mockRejectedValueOnce(Object.assign(new Error("superseded"), { name: "AbortError" }))

    render(<OverviewTab />)
    fireEvent.click(await screen.findByRole("button", { name: "Open report for Signal Check" }))
    expect(mocks.setActiveTabMock).toHaveBeenCalledWith("outputs")
    expect(mocks.openOutputPreviewMock).toHaveBeenCalledWith(71)

    fireEvent.click(screen.getByRole("button", { name: "Refresh" }))
    await waitFor(() => expect(mocks.fetchOverviewMock).toHaveBeenCalledTimes(2))
    expect(screen.queryByText("Failed to load overview")).not.toBeInTheDocument()
    expect(screen.getByTestId("watchlists-overview-live-polite")).toHaveTextContent("")
    expect(screen.getByTestId("watchlists-overview-live-assertive")).toHaveTextContent("")
  })
})
