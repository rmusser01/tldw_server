// @vitest-environment jsdom

import React from "react"
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { OverviewTab } from "../OverviewTab"
import { QUICK_SETUP_DEFAULT_VALUES } from "../quick-setup"

const ONBOARDING_PATH_STORAGE_KEY = "watchlists:onboarding-path:v1"

const mockState = vi.hoisted(() => ({
  fetchOverviewMock: vi.fn(),
  fetchWatchlistSourcesMock: vi.fn(),
  fetchWatchlistRunsMock: vi.fn(),
  bulkCreateSourcesMock: vi.fn(),
  createWatchlistSourceMock: vi.fn(),
  createWatchlistJobMock: vi.fn(),
  deleteWatchlistJobMock: vi.fn(),
  triggerWatchlistRunMock: vi.fn(),
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

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
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
        if (typeof fallback === "string") {
          return fallback
        }
      }
      return key
    }
  })
}))

vi.mock("@/services/watchlists-overview", () => ({
  fetchWatchlistsOverviewData: (...args: unknown[]) =>
    mockState.fetchOverviewMock(...args),
  getOverviewTabBadges: (model?: { tabBadges?: { sources?: number; runs?: number; outputs?: number } } | null) => ({
    sources: Number(model?.tabBadges?.sources || 0),
    runs: Number(model?.tabBadges?.runs || 0),
    outputs: Number(model?.tabBadges?.outputs || 0)
  })
}))

vi.mock("@/services/watchlists", () => ({
  fetchWatchlistSources: (...args: unknown[]) => mockState.fetchWatchlistSourcesMock(...args),
  fetchWatchlistRuns: (...args: unknown[]) => mockState.fetchWatchlistRunsMock(...args),
  bulkCreateSources: (...args: unknown[]) => mockState.bulkCreateSourcesMock(...args),
  createWatchlistSource: (...args: unknown[]) => mockState.createWatchlistSourceMock(...args),
  createWatchlistJob: (...args: unknown[]) => mockState.createWatchlistJobMock(...args),
  deleteWatchlistJob: (...args: unknown[]) => mockState.deleteWatchlistJobMock(...args),
  triggerWatchlistRun: (...args: unknown[]) => mockState.triggerWatchlistRunMock(...args),
  createWatchlistOutput: (...args: unknown[]) => mockState.createWatchlistOutputMock(...args),
  getWatchlistTemplate: (...args: unknown[]) => mockState.getWatchlistTemplateMock(...args),
  previewWatchlistTemplate: (...args: unknown[]) => mockState.previewWatchlistTemplateMock(...args),
  testWatchlistSourceDraft: (...args: unknown[]) => mockState.testWatchlistSourceDraftMock(...args)
}))

vi.mock("@/utils/watchlists-onboarding-telemetry", () => ({
  trackWatchlistsOnboardingTelemetry: (...args: unknown[]) =>
    mockState.trackWatchlistsOnboardingTelemetryMock(...args)
}))

vi.mock("@/store/watchlists", () => ({
  useWatchlistsStore: (selector: (state: Record<string, unknown>) => unknown) =>
    selector({
      setActiveTab: mockState.setActiveTabMock,
      setOutputsRunFilter: mockState.setOutputsRunFilterMock,
      setRunsStatusFilter: mockState.setRunsStatusFilterMock,
      setOverviewHealth: mockState.setOverviewHealthMock,
      openSourceForm: mockState.openSourceFormMock,
      openJobForm: mockState.openJobFormMock,
      openRunDetail: mockState.openRunDetailMock,
      openOutputPreview: mockState.openOutputPreviewMock,
      selectedWatchlistId: mockState.selectedWatchlistId
    })
}))

vi.mock("@/utils/dateFormatters", () => ({
  formatRelativeTime: () => "just now"
}))

const createOverviewPayload = (overrides?: Partial<Record<string, unknown>>) => ({
  fetchedAt: "2026-02-18T12:00:00Z",
  sources: {
    total: 0,
    healthy: 0,
    degraded: 0,
    inactive: 0,
    unknown: 0,
    ...(overrides?.sources as Record<string, unknown> | undefined)
  },
  jobs: {
    total: 0,
    active: 0,
    nextRunAt: null,
    attention: 0,
    ...(overrides?.jobs as Record<string, unknown> | undefined)
  },
  items: {
    unread: 0,
    ...(overrides?.items as Record<string, unknown> | undefined)
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
      sources: "unknown",
      jobs: "unknown",
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

const getPipelineDialog = () => screen.getByRole("dialog", { name: "Briefing pipeline builder" })

const pipelineQueries = () => within(getPipelineDialog())

const getQuickSetupDialog = () => screen.getByRole("dialog", { name: "Add initial collection" })

const clickPipelineNext = () => {
  fireEvent.click(pipelineQueries().getByRole("button", { name: "Next" }))
}

const advancePipelineFromMonitorToReview = async () => {
  clickPipelineNext()
  await waitFor(() => {
    expect(pipelineQueries().getByLabelText("Template")).toBeInTheDocument()
  })
  clickPipelineNext()
  await waitFor(() => {
    expect(pipelineQueries().getByLabelText("Audio briefing")).toBeInTheDocument()
  })
  clickPipelineNext()
  await waitFor(() => {
    expect(screen.getByTestId("watchlists-pipeline-review-summary")).toBeInTheDocument()
  })
}

const selectPipelineFeed = async (label: string) => {
  await waitFor(() => {
    const checkbox = pipelineQueries().getByRole("checkbox", { name: label }) as HTMLInputElement
    if (!checkbox.checked) {
      fireEvent.click(checkbox)
    }
    expect(checkbox).toBeChecked()
  })
}

describe("OverviewTab quick setup flow", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockState.selectedWatchlistId = 42
    mockState.trackWatchlistsOnboardingTelemetryMock.mockResolvedValue(undefined)
    mockState.bulkCreateSourcesMock.mockResolvedValue({
      items: [],
      total: 0,
      created: 0,
      errors: 0
    })
    mockState.fetchWatchlistSourcesMock.mockResolvedValue({
      items: [
        { id: 11, name: "AI Feed" },
        { id: 12, name: "Security Feed" }
      ],
      total: 2
    })
    mockState.fetchWatchlistRunsMock.mockResolvedValue({
      items: [],
      total: 0,
      has_more: false
    })
    mockState.getWatchlistTemplateMock.mockResolvedValue({
      name: "briefing_md",
      format: "md",
      content: "## Briefing Template"
    })
    mockState.previewWatchlistTemplateMock.mockResolvedValue({
      rendered: "Preview output",
      context_keys: [],
      warnings: []
    })
    mockState.testWatchlistSourceDraftMock.mockResolvedValue({
      items: [
        {
          source_id: 11,
          title: "AI lead story",
          url: "https://example.com/post-1",
          decision: "ingest"
        }
      ],
      total: 1,
      ingestable: 1,
      filtered: 0
    })
    QUICK_SETUP_DEFAULT_VALUES.setupGoal = "briefing"
    QUICK_SETUP_DEFAULT_VALUES.runNow = true
    QUICK_SETUP_DEFAULT_VALUES.includeAudioBriefing = true
    localStorage.removeItem(ONBOARDING_PATH_STORAGE_KEY)
  })

  it("prompts for Watchlist creation instead of auto-opening source setup when no Watchlist is selected", async () => {
    mockState.selectedWatchlistId = null
    mockState.fetchOverviewMock.mockResolvedValue(createOverviewPayload())

    render(<OverviewTab />)

    expect(await screen.findByTestId("watchlists-overview-no-watchlist")).toBeInTheDocument()
    expect(screen.getByText("Create a Watchlist first")).toBeInTheDocument()
    expect(screen.queryByText("Add initial collection")).not.toBeInTheDocument()
    expect(mockState.fetchOverviewMock).not.toHaveBeenCalled()
  })

  it("frames selected-Watchlist onboarding as initial collection setup", async () => {
    mockState.fetchOverviewMock.mockResolvedValue(createOverviewPayload())

    render(<OverviewTab />)

    await waitFor(() => {
      expect(screen.getAllByText("Add initial collection").length).toBeGreaterThan(0)
    })
    expect(screen.getByText("Add feeds -> Configure monitor -> Check Activity -> Review Updates -> Generate Reports")).toBeInTheDocument()
    expect(screen.getByTestId("watchlists-overview-cta-guided-setup")).toHaveTextContent("Add initial collection")

    fireEvent.click(screen.getByTestId("watchlists-overview-cta-guided-setup"))
    expect(screen.getAllByText("Add initial collection").length).toBeGreaterThan(1)
  }, 20_000)

  it("opens Feed creation directly from quick setup", async () => {
    mockState.fetchOverviewMock.mockResolvedValue(createOverviewPayload())

    render(<OverviewTab />)

    await waitFor(() => {
      expect(screen.getByTestId("watchlists-overview-cta-guided-setup")).toHaveTextContent("Add initial collection")
    })

    fireEvent.click(screen.getByTestId("watchlists-overview-cta-add-feed"))

    expect(mockState.setActiveTabMock).toHaveBeenCalledWith("sources")
    expect(mockState.openSourceFormMock).toHaveBeenCalledTimes(1)
  })

  it("records onboarding success milestones when overview includes generated reports", async () => {
    mockState.fetchOverviewMock.mockResolvedValue(
      createOverviewPayload({
        sources: { total: 1, healthy: 1 },
        jobs: { total: 1, active: 1 },
        outputs: { total: 2, expired: 0, deliveryIssues: 0, attention: 0 }
      })
    )

    render(<OverviewTab />)

    await waitFor(() => {
      expect(screen.getByText("Setup complete")).toBeInTheDocument()
    })

    expect(mockState.trackWatchlistsOnboardingTelemetryMock).toHaveBeenCalledWith({
      type: "quick_setup_first_run_succeeded",
      source: "overview"
    })
    expect(mockState.trackWatchlistsOnboardingTelemetryMock).toHaveBeenCalledWith({
      type: "quick_setup_first_output_succeeded",
      source: "overview"
    })
  })

  it("opens Monitor creation directly when feeds already exist", async () => {
    mockState.fetchOverviewMock.mockResolvedValue(
      createOverviewPayload({
        sources: { total: 2, healthy: 2 },
        jobs: { total: 0, active: 0 }
      })
    )

    render(<OverviewTab />)

    await waitFor(() => {
      expect(screen.getByTestId("watchlists-overview-cta-create-monitor")).toBeInTheDocument()
    })

    fireEvent.click(screen.getByTestId("watchlists-overview-cta-create-monitor"))

    expect(mockState.setActiveTabMock).toHaveBeenCalledWith("jobs")
    expect(mockState.openJobFormMock).toHaveBeenCalledTimes(1)
  })

  it("opens guided setup modal and advances to monitor configuration", async () => {
    mockState.fetchOverviewMock.mockResolvedValue(createOverviewPayload())

    render(<OverviewTab />)

    await waitFor(() => {
      expect(screen.getByTestId("watchlists-overview-cta-guided-setup")).toBeInTheDocument()
    })

    fireEvent.click(screen.getByTestId("watchlists-overview-cta-guided-setup"))
    expect(mockState.trackWatchlistsOnboardingTelemetryMock).toHaveBeenCalledWith({
      type: "quick_setup_opened"
    })
    expect(
      screen.getByText("Add one or more feed URLs to this Watchlist. You can adjust feed settings later.")
    ).toBeInTheDocument()

    fireEvent.change(
      screen.getByPlaceholderText("e.g., Daily Tech Feed"),
      { target: { value: "AI Feed" } }
    )
    fireEvent.change(
      screen.getByPlaceholderText("https://example.com/feed.xml"),
      { target: { value: "https://example.com/rss.xml" } }
    )
    fireEvent.click(screen.getByRole("button", { name: "Next" }))

    await waitFor(() => {
      expect(screen.getByPlaceholderText("e.g., Morning Brief")).toBeInTheDocument()
    })
    expect(
      screen.getByText("Choose how this Watchlist monitor runs; presets avoid cron setup first.")
    ).toBeInTheDocument()
  }, 20_000)

  it("persists onboarding path selection and restores it on next render", async () => {
    mockState.fetchOverviewMock.mockResolvedValue(createOverviewPayload())

    const { unmount } = render(<OverviewTab />)

    await waitFor(() => {
      expect(
        screen.getByTestId("watchlists-overview-onboarding-path-beginner")
      ).toBeInTheDocument()
    })

    expect(
      screen.getByTestId("watchlists-overview-onboarding-path-beginner")
    ).toHaveAttribute("aria-pressed", "true")

    fireEvent.click(screen.getByTestId("watchlists-overview-onboarding-path-advanced"))

    expect(localStorage.getItem(ONBOARDING_PATH_STORAGE_KEY)).toBe("advanced")
    expect(
      screen.getByTestId("watchlists-overview-onboarding-path-advanced")
    ).toHaveAttribute("aria-pressed", "true")

    unmount()
    render(<OverviewTab />)

    await waitFor(() => {
      expect(
        screen.getByTestId("watchlists-overview-onboarding-path-advanced")
      ).toHaveAttribute("aria-pressed", "true")
    })
    expect(screen.getByTestId("watchlists-overview-cta-advanced-direct")).toBeInTheDocument()
  })

  it("routes to Reports after setup when briefing goal is selected and run-now is disabled", async () => {
    mockState.fetchOverviewMock.mockResolvedValue(createOverviewPayload())
    mockState.createWatchlistSourceMock.mockResolvedValue({ id: 101 })
    mockState.createWatchlistJobMock.mockResolvedValue({ id: 202 })

    render(<OverviewTab />)

    await waitFor(() => {
      expect(screen.getByTestId("watchlists-overview-cta-guided-setup")).toBeInTheDocument()
    })

    fireEvent.click(screen.getByTestId("watchlists-overview-cta-guided-setup"))
    fireEvent.change(screen.getByPlaceholderText("e.g., Daily Tech Feed"), {
      target: { value: "AI Feed" }
    })
    fireEvent.change(screen.getByPlaceholderText("https://example.com/feed.xml"), {
      target: { value: "https://example.com/rss.xml" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Next" }))

    await waitFor(() => {
      expect(screen.getByPlaceholderText("e.g., Morning Brief")).toBeInTheDocument()
    })

    fireEvent.change(screen.getByPlaceholderText("e.g., Morning Brief"), {
      target: { value: "AM Brief" }
    })
    fireEvent.click(screen.getByLabelText("Run immediately"))
    fireEvent.click(screen.getByRole("button", { name: "Next" }))
    await waitFor(() => {
      expect(screen.getByRole("button", { name: /Create collection/i })).toBeInTheDocument()
    })
    fireEvent.click(screen.getByRole("button", { name: /Create collection/i }))

    await waitFor(() => {
      expect(mockState.setActiveTabMock).toHaveBeenLastCalledWith("outputs")
    })
    expect(mockState.triggerWatchlistRunMock).not.toHaveBeenCalled()
    expect(mockState.createWatchlistSourceMock).toHaveBeenCalledWith(
      expect.objectContaining({
        watchlist_id: 42
      })
    )
    expect(mockState.createWatchlistJobMock).toHaveBeenCalledWith(
      expect.objectContaining({
        watchlist_id: 42,
        output_prefs: expect.objectContaining({
          template_name: "briefing_md"
        })
      })
    )
    expect(mockState.trackWatchlistsOnboardingTelemetryMock).toHaveBeenCalledWith({
      type: "quick_setup_completed",
      goal: "briefing",
      runNow: false,
      destination: "outputs"
    })
  }, 20_000)

  it("creates quick setup monitor scope with multiple feeds when extra URLs are provided", async () => {
    mockState.fetchOverviewMock.mockResolvedValue(createOverviewPayload())
    mockState.createWatchlistSourceMock.mockResolvedValue({ id: 101 })
    mockState.bulkCreateSourcesMock.mockResolvedValue({
      items: [
        { id: 201, url: "https://example.com/feed-b.xml", status: "created" },
        { id: 202, url: "https://example.com/feed-c.xml", status: "created" }
      ],
      total: 2,
      created: 2,
      errors: 0
    })
    mockState.createWatchlistJobMock.mockResolvedValue({ id: 303 })

    render(<OverviewTab />)

    await waitFor(() => {
      expect(screen.getByTestId("watchlists-overview-cta-guided-setup")).toBeInTheDocument()
    })

    fireEvent.click(screen.getByTestId("watchlists-overview-cta-guided-setup"))
    fireEvent.change(screen.getByPlaceholderText("e.g., Daily Tech Feed"), {
      target: { value: "AI Feed" }
    })
    fireEvent.change(screen.getByPlaceholderText("https://example.com/feed.xml"), {
      target: { value: "https://example.com/feed-a.xml" }
    })
    fireEvent.change(screen.getByLabelText("Additional feed URLs (optional)"), {
      target: {
        value: "https://example.com/feed-b.xml\nhttps://example.com/feed-c.xml"
      }
    })
    fireEvent.click(screen.getByRole("button", { name: "Next" }))

    await waitFor(() => {
      expect(screen.getByLabelText("Run immediately")).toBeInTheDocument()
    })
    fireEvent.click(screen.getByLabelText("Run immediately"))
    fireEvent.click(screen.getByRole("button", { name: "Next" }))
    await waitFor(() => {
      expect(screen.getByRole("button", { name: /Create collection/i })).toBeInTheDocument()
    })
    fireEvent.click(screen.getByRole("button", { name: /Create collection/i }))

    await waitFor(() => {
      expect(mockState.createWatchlistJobMock).toHaveBeenCalledWith(
        expect.objectContaining({
          scope: { sources: [101, 201, 202] }
        })
      )
    })
    expect(mockState.bulkCreateSourcesMock).toHaveBeenCalledTimes(1)
    expect(mockState.bulkCreateSourcesMock).toHaveBeenCalledWith(
      expect.arrayContaining([
        expect.objectContaining({ watchlist_id: 42 })
      ])
    )
    expect(mockState.triggerWatchlistRunMock).not.toHaveBeenCalled()
  }, 20_000)

  it("fails quick setup when additional feed creation is partial and does not create a monitor", async () => {
    const consoleErrorSpy = vi.spyOn(console, "error").mockImplementation(() => {})
    mockState.fetchOverviewMock.mockResolvedValue(createOverviewPayload())
    mockState.createWatchlistSourceMock.mockResolvedValue({ id: 101 })
    mockState.bulkCreateSourcesMock.mockResolvedValue({
      items: [
        { id: 201, url: "https://example.com/feed-b.xml", status: "created" },
        { url: "https://example.com/feed-c.xml", status: "error", error: "timeout" }
      ],
      total: 2,
      created: 1,
      errors: 1
    })

    render(<OverviewTab />)

    await waitFor(() => {
      expect(screen.getByTestId("watchlists-overview-cta-guided-setup")).toBeInTheDocument()
    })

    fireEvent.click(screen.getByTestId("watchlists-overview-cta-guided-setup"))
    fireEvent.change(screen.getByPlaceholderText("e.g., Daily Tech Feed"), {
      target: { value: "AI Feed" }
    })
    fireEvent.change(screen.getByPlaceholderText("https://example.com/feed.xml"), {
      target: { value: "https://example.com/feed-a.xml" }
    })
    fireEvent.change(screen.getByLabelText("Additional feed URLs (optional)"), {
      target: {
        value: "https://example.com/feed-b.xml\nhttps://example.com/feed-c.xml"
      }
    })
    fireEvent.click(screen.getByRole("button", { name: "Next" }))
    await waitFor(() => {
      expect(screen.getByLabelText("Run immediately")).toBeInTheDocument()
    })
    fireEvent.click(screen.getByRole("button", { name: "Next" }))
    await waitFor(() => {
      expect(screen.getByRole("button", { name: /Create collection/i })).toBeInTheDocument()
    })
    fireEvent.click(screen.getByRole("button", { name: /Create collection/i }))

    await waitFor(() => {
      expect(mockState.createWatchlistJobMock).not.toHaveBeenCalled()
    })
    consoleErrorSpy.mockRestore()
  }, 20_000)

  it("routes to Activity after setup when briefing goal runs immediately", async () => {
    mockState.fetchOverviewMock.mockResolvedValue(createOverviewPayload())
    mockState.createWatchlistSourceMock.mockResolvedValue({ id: 101 })
    mockState.createWatchlistJobMock.mockResolvedValue({ id: 202 })
    mockState.triggerWatchlistRunMock.mockResolvedValue({ id: 303 })

    render(<OverviewTab />)

    await waitFor(() => {
      expect(screen.getByTestId("watchlists-overview-cta-guided-setup")).toBeInTheDocument()
    })

    fireEvent.click(screen.getByTestId("watchlists-overview-cta-guided-setup"))
    fireEvent.change(screen.getByPlaceholderText("e.g., Daily Tech Feed"), {
      target: { value: "AI Feed" }
    })
    fireEvent.change(screen.getByPlaceholderText("https://example.com/feed.xml"), {
      target: { value: "https://example.com/rss.xml" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Next" }))
    await waitFor(() => {
      expect(screen.getByLabelText("Run immediately")).toBeInTheDocument()
    })
    fireEvent.click(screen.getByRole("button", { name: "Next" }))
    await waitFor(() => {
      expect(screen.getByRole("button", { name: /Create collection/i })).toBeInTheDocument()
    })
    fireEvent.click(screen.getByRole("button", { name: /Create collection/i }))

    await waitFor(() => {
      expect(mockState.setActiveTabMock).toHaveBeenLastCalledWith("runs")
    })
    expect(mockState.openRunDetailMock).toHaveBeenCalledWith(303)
    expect(mockState.trackWatchlistsOnboardingTelemetryMock).toHaveBeenCalledWith({
      type: "quick_setup_completed",
      goal: "briefing",
      runNow: true,
      destination: "runs"
    })
  }, 20_000)

  it("routes to Monitors when triage goal is selected without run-now", async () => {
    QUICK_SETUP_DEFAULT_VALUES.setupGoal = "triage"
    QUICK_SETUP_DEFAULT_VALUES.runNow = false
    QUICK_SETUP_DEFAULT_VALUES.includeAudioBriefing = false

    mockState.fetchOverviewMock.mockResolvedValue(createOverviewPayload())
    mockState.createWatchlistSourceMock.mockResolvedValue({ id: 101 })
    mockState.createWatchlistJobMock.mockResolvedValue({ id: 202 })

    render(<OverviewTab />)

    await waitFor(() => {
      expect(screen.getByTestId("watchlists-overview-cta-guided-setup")).toBeInTheDocument()
    })

    fireEvent.click(screen.getByTestId("watchlists-overview-cta-guided-setup"))
    fireEvent.change(screen.getByPlaceholderText("e.g., Daily Tech Feed"), {
      target: { value: "AI Feed" }
    })
    fireEvent.change(screen.getByPlaceholderText("https://example.com/feed.xml"), {
      target: { value: "https://example.com/rss.xml" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Next" }))
    await waitFor(() => {
      expect(screen.getByLabelText("Run immediately")).toBeInTheDocument()
    })
    fireEvent.click(screen.getByRole("button", { name: "Next" }))
    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Create collection" })).toBeInTheDocument()
    })
    fireEvent.click(screen.getByRole("button", { name: "Create collection" }))

    await waitFor(() => {
      expect(mockState.setActiveTabMock).toHaveBeenLastCalledWith("jobs")
    })
    expect(mockState.triggerWatchlistRunMock).not.toHaveBeenCalled()
    expect(mockState.createWatchlistJobMock).toHaveBeenCalledWith(
      expect.not.objectContaining({
        output_prefs: expect.anything()
      })
    )
    expect(mockState.trackWatchlistsOnboardingTelemetryMock).toHaveBeenCalledWith({
      type: "quick_setup_completed",
      goal: "triage",
      runNow: false,
      destination: "jobs"
    })
  }, 20_000)

  it("creates run-now pipeline and routes to output preview when output is generated", async () => {
    mockState.fetchOverviewMock.mockResolvedValue(
      createOverviewPayload({
        sources: { total: 2, healthy: 2 },
        jobs: { total: 1, active: 1 }
      })
    )
    mockState.createWatchlistJobMock.mockResolvedValue({ id: 303 })
    mockState.triggerWatchlistRunMock.mockResolvedValue({ id: 404 })
    mockState.createWatchlistOutputMock.mockResolvedValue({ id: 505 })

    render(<OverviewTab />)

    await waitFor(() => {
      expect(screen.getByTestId("watchlists-overview-cta-pipeline-builder")).toBeInTheDocument()
    })

    fireEvent.click(screen.getByTestId("watchlists-overview-cta-pipeline-builder"))
    await waitFor(() => {
      expect(pipelineQueries().getByLabelText("AI Feed")).toBeInTheDocument()
    })
    await selectPipelineFeed("AI Feed")
    await selectPipelineFeed("Security Feed")
    clickPipelineNext()
    await waitFor(() => {
      expect(pipelineQueries().getByLabelText("Monitor name")).toBeInTheDocument()
    })
    fireEvent.change(pipelineQueries().getByLabelText("Monitor name"), {
      target: { value: "Morning Brief" }
    })
    await advancePipelineFromMonitorToReview()
    await waitFor(() => {
      expect(pipelineQueries().getByRole("button", { name: "Create pipeline" })).toBeInTheDocument()
    })
    fireEvent.click(pipelineQueries().getByRole("button", { name: "Create pipeline" }))

    await waitFor(() => {
      expect(mockState.createWatchlistJobMock).toHaveBeenCalledWith(
        expect.objectContaining({
          scope: { sources: [11, 12] }
        })
      )
    })
    expect(mockState.triggerWatchlistRunMock).toHaveBeenCalledWith(303)
    expect(mockState.createWatchlistOutputMock).toHaveBeenCalledWith(
      expect.objectContaining({
        run_id: 404
      })
    )
    expect(mockState.setOutputsRunFilterMock).toHaveBeenCalledWith(404)
    expect(mockState.setActiveTabMock).toHaveBeenLastCalledWith("outputs")
    expect(mockState.openOutputPreviewMock).toHaveBeenCalledWith(505)
    expect(mockState.trackWatchlistsOnboardingTelemetryMock).toHaveBeenCalledWith({
      type: "pipeline_setup_opened"
    })
    expect(mockState.trackWatchlistsOnboardingTelemetryMock).toHaveBeenCalledWith({
      type: "pipeline_setup_step_completed",
      step: "review"
    })
    expect(mockState.trackWatchlistsOnboardingTelemetryMock).toHaveBeenCalledWith({
      type: "pipeline_setup_submitted",
      mode: "create",
      runNow: true
    })
    expect(mockState.trackWatchlistsOnboardingTelemetryMock).toHaveBeenCalledWith({
      type: "pipeline_setup_completed",
      mode: "create",
      runNow: true,
      destination: "outputs"
    })
  }, 20_000)

  it("rolls back monitor creation when pipeline creation fails before completion", async () => {
    const consoleErrorSpy = vi.spyOn(console, "error").mockImplementation(() => {})
    mockState.fetchOverviewMock.mockResolvedValue(
      createOverviewPayload({
        sources: { total: 2, healthy: 2 },
        jobs: { total: 1, active: 1 }
      })
    )
    mockState.createWatchlistJobMock.mockResolvedValue({ id: 303 })
    mockState.triggerWatchlistRunMock.mockRejectedValue(new Error("run failed"))
    mockState.deleteWatchlistJobMock.mockResolvedValue({ success: true, job_id: 303 })

    render(<OverviewTab />)

    await waitFor(() => {
      expect(screen.getByTestId("watchlists-overview-cta-pipeline-builder")).toBeInTheDocument()
    })

    fireEvent.click(screen.getByTestId("watchlists-overview-cta-pipeline-builder"))
    await waitFor(() => {
      expect(pipelineQueries().getByLabelText("AI Feed")).toBeInTheDocument()
    })
    await selectPipelineFeed("AI Feed")
    clickPipelineNext()
    await waitFor(() => {
      expect(pipelineQueries().getByLabelText("Monitor name")).toBeInTheDocument()
    })
    fireEvent.change(pipelineQueries().getByLabelText("Monitor name"), {
      target: { value: "Morning Brief" }
    })
    await advancePipelineFromMonitorToReview()
    await waitFor(() => {
      expect(pipelineQueries().getByRole("button", { name: "Create pipeline" })).toBeInTheDocument()
    })
    fireEvent.click(pipelineQueries().getByRole("button", { name: "Create pipeline" }))

    await waitFor(() => {
      expect(mockState.deleteWatchlistJobMock).toHaveBeenCalledWith(303)
    })
    expect(mockState.openOutputPreviewMock).not.toHaveBeenCalled()
    expect(mockState.openRunDetailMock).not.toHaveBeenCalled()
    expect(mockState.trackWatchlistsOnboardingTelemetryMock).toHaveBeenCalledWith({
      type: "pipeline_setup_failed",
      stage: "run_trigger",
      mode: "create",
      runNow: true
    })
    consoleErrorSpy.mockRestore()
  }, 20_000)

  it("generates pipeline template preview on review step when run context exists", async () => {
    mockState.fetchOverviewMock.mockResolvedValue(
      createOverviewPayload({
        sources: { total: 2, healthy: 2 },
        jobs: { total: 1, active: 1 }
      })
    )
    mockState.fetchWatchlistRunsMock.mockResolvedValue({
      items: [
        {
          id: 777,
          job_id: 99,
          status: "completed"
        }
      ],
      total: 1,
      has_more: false
    })
    mockState.getWatchlistTemplateMock.mockResolvedValue({
      name: "briefing_md",
      format: "md",
      content: "## {{ title }}"
    })
    mockState.previewWatchlistTemplateMock.mockResolvedValue({
      rendered: "Preview briefing output",
      context_keys: ["items"],
      warnings: []
    })

    render(<OverviewTab />)

    await waitFor(() => {
      expect(screen.getByTestId("watchlists-overview-cta-pipeline-builder")).toBeInTheDocument()
    })

    fireEvent.click(screen.getByTestId("watchlists-overview-cta-pipeline-builder"))
    await waitFor(() => {
      expect(pipelineQueries().getByLabelText("AI Feed")).toBeInTheDocument()
    })
    await selectPipelineFeed("AI Feed")
    clickPipelineNext()
    await waitFor(() => {
      expect(pipelineQueries().getByLabelText("Monitor name")).toBeInTheDocument()
    })
    fireEvent.change(pipelineQueries().getByLabelText("Monitor name"), {
      target: { value: "Morning Brief" }
    })
    await advancePipelineFromMonitorToReview()

    await waitFor(() => {
      expect(screen.getByTestId("watchlists-pipeline-preview-generate")).toBeInTheDocument()
    })

    fireEvent.click(screen.getByTestId("watchlists-pipeline-preview-generate"))

    await waitFor(() => {
      expect(mockState.previewWatchlistTemplateMock).toHaveBeenCalledWith(
        "## {{ title }}",
        777,
        "md"
      )
    })
    expect(screen.getByTestId("watchlists-pipeline-preview-rendered")).toHaveTextContent(
      "Preview briefing output"
    )
    expect(mockState.trackWatchlistsOnboardingTelemetryMock).toHaveBeenCalledWith({
      type: "pipeline_setup_preview_generated",
      status: "success",
      run_id: 777,
      warning_count: 0
    })
  }, 20_000)

  it("shows fallback guidance when template preview has no completed run context", async () => {
    mockState.fetchOverviewMock.mockResolvedValue(
      createOverviewPayload({
        sources: { total: 2, healthy: 2 },
        jobs: { total: 1, active: 1 }
      })
    )
    mockState.fetchWatchlistRunsMock.mockResolvedValue({
      items: [],
      total: 0,
      has_more: false
    })

    render(<OverviewTab />)

    await waitFor(() => {
      expect(screen.getByTestId("watchlists-overview-cta-pipeline-builder")).toBeInTheDocument()
    })

    fireEvent.click(screen.getByTestId("watchlists-overview-cta-pipeline-builder"))
    await waitFor(() => {
      expect(pipelineQueries().getByLabelText("AI Feed")).toBeInTheDocument()
    })
    await selectPipelineFeed("AI Feed")
    clickPipelineNext()
    await waitFor(() => {
      expect(pipelineQueries().getByLabelText("Monitor name")).toBeInTheDocument()
    })
    fireEvent.change(pipelineQueries().getByLabelText("Monitor name"), {
      target: { value: "Morning Brief" }
    })
    await advancePipelineFromMonitorToReview()

    await waitFor(() => {
      expect(screen.getByTestId("watchlists-pipeline-preview-generate")).toBeInTheDocument()
    })
    fireEvent.click(screen.getByTestId("watchlists-pipeline-preview-generate"))

    await waitFor(() => {
      expect(screen.getByTestId("watchlists-pipeline-preview-error")).toHaveTextContent(
        "Run any monitor once, then generate template preview."
      )
    })
    expect(mockState.trackWatchlistsOnboardingTelemetryMock).toHaveBeenCalledWith({
      type: "pipeline_setup_preview_generated",
      status: "no_run_context"
    })
  }, 20_000)

  it("runs test generation from review step and routes to report preview", async () => {
    mockState.fetchOverviewMock.mockResolvedValue(
      createOverviewPayload({
        sources: { total: 2, healthy: 2 },
        jobs: { total: 1, active: 1 }
      })
    )
    mockState.createWatchlistJobMock.mockResolvedValue({ id: 303 })
    mockState.triggerWatchlistRunMock.mockResolvedValue({ id: 404 })
    mockState.createWatchlistOutputMock.mockResolvedValue({ id: 505 })

    render(<OverviewTab />)

    await waitFor(() => {
      expect(screen.getByTestId("watchlists-overview-cta-pipeline-builder")).toBeInTheDocument()
    })

    fireEvent.click(screen.getByTestId("watchlists-overview-cta-pipeline-builder"))
    await waitFor(() => {
      expect(pipelineQueries().getByLabelText("AI Feed")).toBeInTheDocument()
    })
    await selectPipelineFeed("AI Feed")
    clickPipelineNext()
    await waitFor(() => {
      expect(pipelineQueries().getByLabelText("Run immediately")).toBeInTheDocument()
    })
    fireEvent.click(pipelineQueries().getByLabelText("Run immediately"))
    fireEvent.change(pipelineQueries().getByLabelText("Monitor name"), {
      target: { value: "Morning Brief" }
    })
    await advancePipelineFromMonitorToReview()
    await waitFor(() => {
      expect(screen.getByTestId("watchlists-pipeline-test-generation")).toBeInTheDocument()
    })

    fireEvent.click(screen.getByTestId("watchlists-pipeline-test-generation"))

    await waitFor(() => {
      expect(mockState.triggerWatchlistRunMock).toHaveBeenCalledWith(303)
    })
    expect(mockState.createWatchlistOutputMock).toHaveBeenCalledWith(
      expect.objectContaining({
        run_id: 404
      })
    )
    expect(mockState.setOutputsRunFilterMock).toHaveBeenCalledWith(404)
    expect(mockState.setActiveTabMock).toHaveBeenLastCalledWith("outputs")
    expect(mockState.openOutputPreviewMock).toHaveBeenCalledWith(505)
    expect(mockState.trackWatchlistsOnboardingTelemetryMock).toHaveBeenCalledWith({
      type: "pipeline_setup_submitted",
      mode: "test",
      runNow: true
    })
    expect(mockState.trackWatchlistsOnboardingTelemetryMock).toHaveBeenCalledWith({
      type: "pipeline_setup_completed",
      mode: "test",
      runNow: true,
      destination: "outputs"
    })
  })

  it("restores focus to guided setup trigger after quick setup modal closes", async () => {
    mockState.fetchOverviewMock.mockResolvedValue(createOverviewPayload())

    render(<OverviewTab />)

    const trigger = await screen.findByTestId("watchlists-overview-cta-guided-setup")
    trigger.focus()
    expect(trigger).toHaveFocus()

    fireEvent.click(trigger)
    await waitFor(() => {
      expect(getQuickSetupDialog()).toBeInTheDocument()
    })
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }))

    await waitFor(() => {
      expect(trigger).toHaveFocus()
    })
  })

  it("restores focus to pipeline builder trigger after modal closes", async () => {
    mockState.fetchOverviewMock.mockResolvedValue(
      createOverviewPayload({
        sources: { total: 2, healthy: 2 },
        jobs: { total: 1, active: 1 }
      })
    )

    render(<OverviewTab />)

    const trigger = await screen.findByTestId("watchlists-overview-cta-pipeline-builder")
    trigger.focus()
    expect(trigger).toHaveFocus()

    fireEvent.click(trigger)
    await waitFor(() => {
      expect(getPipelineDialog()).toBeInTheDocument()
    })
    fireEvent.click(pipelineQueries().getByRole("button", { name: "Cancel" }))

    await waitFor(() => {
      expect(trigger).toHaveFocus()
    })
  })
})
