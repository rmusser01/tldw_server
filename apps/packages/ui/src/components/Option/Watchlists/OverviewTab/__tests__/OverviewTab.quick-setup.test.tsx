// @vitest-environment jsdom

import React from "react"
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { extractPipelineErrorMessage, OverviewTab } from "../OverviewTab"

const mocks = vi.hoisted(() => ({
  fetchOverview: vi.fn(),
  fetchSources: vi.fn(),
  fetchRuns: vi.fn(),
  createSource: vi.fn(),
  createJob: vi.fn(),
  updateJob: vi.fn(),
  triggerRun: vi.fn(),
  createOutput: vi.fn(),
  getTemplate: vi.fn(),
  previewTemplate: vi.fn(),
  testSource: vi.fn(),
  setActiveTab: vi.fn(),
  setOutputsRunFilter: vi.fn(),
  setOverviewHealth: vi.fn(),
  openOutputPreview: vi.fn(),
  openSourceForm: vi.fn(),
  openJobForm: vi.fn(),
  openRunDetail: vi.fn(),
  selectedWatchlistId: 42 as number | null
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string, options?: Record<string, unknown>) =>
      (typeof fallback === "string" ? fallback : _key).replace(
        /\{\{(\w+)\}\}/g,
        (_match, token) => String(options?.[token] ?? "")
      )
  })
}))

vi.mock("@/services/watchlists-overview", () => ({
  fetchWatchlistsOverviewData: (...args: unknown[]) => mocks.fetchOverview(...args),
  getOverviewTabBadges: () => ({ sources: 0, runs: 0, outputs: 0 })
}))

vi.mock("@/services/watchlists", () => ({
  fetchWatchlistSources: (...args: unknown[]) => mocks.fetchSources(...args),
  fetchWatchlistRuns: (...args: unknown[]) => mocks.fetchRuns(...args),
  bulkCreateSources: vi.fn().mockResolvedValue({ items: [], total: 0 }),
  createWatchlistSource: (...args: unknown[]) => mocks.createSource(...args),
  deleteWatchlistSource: vi.fn(),
  createWatchlistJob: (...args: unknown[]) => mocks.createJob(...args),
  deleteWatchlistJob: vi.fn(),
  updateWatchlistJob: (...args: unknown[]) => mocks.updateJob(...args),
  triggerWatchlistRun: (...args: unknown[]) => mocks.triggerRun(...args),
  createWatchlistOutput: (...args: unknown[]) => mocks.createOutput(...args),
  getWatchlistTemplate: (...args: unknown[]) => mocks.getTemplate(...args),
  previewWatchlistTemplate: (...args: unknown[]) => mocks.previewTemplate(...args),
  testWatchlistSourceDraft: (...args: unknown[]) => mocks.testSource(...args)
}))

vi.mock("@/utils/watchlists-onboarding-telemetry", () => ({
  trackWatchlistsOnboardingTelemetry: vi.fn().mockResolvedValue(undefined)
}))

vi.mock("@/store/watchlists", () => ({
  useWatchlistsStore: (selector: (state: Record<string, unknown>) => unknown) =>
    selector({
      setActiveTab: mocks.setActiveTab,
      setOutputsRunFilter: mocks.setOutputsRunFilter,
      setRunsStatusFilter: vi.fn(),
      setOverviewHealth: mocks.setOverviewHealth,
      openSourceForm: mocks.openSourceForm,
      openJobForm: mocks.openJobForm,
      openRunDetail: mocks.openRunDetail,
      openOutputPreview: mocks.openOutputPreview,
      selectedWatchlistId: mocks.selectedWatchlistId
    })
}))

vi.mock("@/utils/dateFormatters", () => ({ formatRelativeTime: () => "just now" }))

const overview = (sourceCount = 2, jobCount = 1) => ({
  fetchedAt: "2026-07-10T12:00:00Z",
  sources: { total: sourceCount, healthy: sourceCount, degraded: 0, inactive: 0, unknown: 0 },
  jobs: { total: jobCount, active: jobCount, nextRunAt: null, attention: 0 },
  items: { unread: 0 },
  runs: { running: 0, pending: 0, failed: 0, recentFailed: [] },
  outputs: { total: 0, expired: 0, deliveryIssues: 0, attention: 0 },
  health: {
    statuses: { sources: "healthy", jobs: "healthy", runs: "healthy", outputs: "healthy" },
    attention: { total: 0, sources: 0, jobs: 0, runs: 0, outputs: 0 },
    tabBadges: { sources: 0, runs: 0, outputs: 0 }
  },
  systemHealth: "healthy"
})

const pipeline = () => within(screen.getByRole("dialog", { name: "Set up briefing" }))

const reachTestStep = async () => {
  fireEvent.click(pipeline().getByLabelText("AI Feed"))
  fireEvent.click(pipeline().getByRole("button", { name: "Next: Cadence" }))
  fireEvent.change(pipeline().getByLabelText("Monitor name"), { target: { value: "Morning Brief" } })
  fireEvent.click(pipeline().getByRole("button", { name: "Next: Briefing" }))
  fireEvent.click(pipeline().getByRole("button", { name: "Next: Delivery" }))
  fireEvent.click(pipeline().getByRole("button", { name: "Next: Test" }))
  await waitFor(() => {
    expect(pipeline().getByRole("button", { name: "Generate 60-second sample" })).toBeInTheDocument()
  })
}

beforeEach(() => {
  vi.clearAllMocks()
  mocks.selectedWatchlistId = 42
  mocks.fetchOverview.mockResolvedValue(overview())
  mocks.fetchSources.mockResolvedValue({
    items: [
      { id: 11, name: "AI Feed", url: "https://example.com/ai.xml", source_type: "rss", active: true },
      { id: 12, name: "Security Feed", url: "https://example.com/security.xml", source_type: "rss", active: true }
    ],
    total: 2
  })
  mocks.fetchRuns.mockResolvedValue({ items: [], total: 0, has_more: false })
  mocks.createJob.mockResolvedValue({ id: 303, active: false })
  mocks.updateJob.mockResolvedValue({ id: 303, active: true })
  mocks.triggerRun.mockResolvedValue({ id: 404 })
  mocks.createOutput.mockResolvedValue({ id: 505 })
  mocks.testSource.mockResolvedValue({ items: [], total: 0, ingestable: 0, filtered: 0 })
})

describe("extractPipelineErrorMessage", () => {
  it("preserves string and structured error details", () => {
    expect(extractPipelineErrorMessage("template_not_found")).toBe("template_not_found")
    expect(extractPipelineErrorMessage({ response: { data: { detail: { message: "missing template" } } } })).toBe("missing template")
  })
})

describe("OverviewTab canonical setup", () => {
  it("keeps setup scoped to a selected Watchlist", async () => {
    mocks.selectedWatchlistId = null
    render(<OverviewTab />)

    expect(await screen.findByTestId("watchlists-overview-no-watchlist")).toBeInTheDocument()
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
  })

  it("routes Quick Setup and Briefing setup into the same canonical wizard", async () => {
    mocks.fetchOverview.mockResolvedValue(overview(0, 0))
    render(<OverviewTab />)

    const quickTrigger = await screen.findByTestId("watchlists-overview-cta-guided-setup")
    fireEvent.click(quickTrigger)
    expect(await screen.findByRole("dialog", { name: "Set up briefing" })).toBeInTheDocument()
    expect(pipeline().getByLabelText("Add a new source")).toBeChecked()
    expect(screen.queryByRole("dialog", { name: "Add initial collection" })).not.toBeInTheDocument()

    fireEvent.click(pipeline().getByRole("button", { name: "Cancel" }))
    await waitFor(() => expect(quickTrigger).toHaveFocus())
  })

  it("tests an inactive monitor and activates the same id without creating a duplicate", async () => {
    render(<OverviewTab />)

    const trigger = await screen.findByTestId("watchlists-overview-cta-pipeline-builder")
    fireEvent.click(trigger)
    await waitFor(() => expect(pipeline().getByLabelText("AI Feed")).toBeInTheDocument())
    fireEvent.click(pipeline().getByLabelText("AI Feed"))
    fireEvent.click(pipeline().getByRole("button", { name: "Test source" }))
    await waitFor(() => expect(mocks.testSource).toHaveBeenCalledWith(
      { url: "https://example.com/ai.xml", source_type: "rss" },
      { limit: 6 }
    ))
    fireEvent.click(pipeline().getByLabelText("AI Feed"))
    await reachTestStep()

    fireEvent.click(pipeline().getByRole("button", { name: "Generate 60-second sample" }))
    await waitFor(() => expect(mocks.triggerRun).toHaveBeenCalledWith(303))
    expect(mocks.createJob).toHaveBeenCalledWith(expect.objectContaining({
      active: false,
      watchlist_id: 42,
      output_prefs: expect.objectContaining({ briefing_pipeline: expect.any(Object) })
    }))
    expect(mocks.createOutput).toHaveBeenCalledWith(expect.objectContaining({
      run_id: 404,
      target_audio_minutes: 1,
      metadata: expect.objectContaining({
        test: expect.objectContaining({ externalDelivery: false, audioSampleSeconds: 60 })
      })
    }))
    expect(mocks.setActiveTab).not.toHaveBeenCalled()
    expect(pipeline().getByText("Test started. This draft stays inactive until you activate its schedule.")).toBeInTheDocument()

    fireEvent.click(pipeline().getByRole("button", { name: "Activate schedule" }))
    await waitFor(() => expect(mocks.updateJob).toHaveBeenCalledWith(303, { active: true }))
    expect(mocks.createJob).toHaveBeenCalledTimes(1)
  }, 20_000)
})
