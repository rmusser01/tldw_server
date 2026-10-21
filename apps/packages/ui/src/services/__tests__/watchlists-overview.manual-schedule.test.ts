import { beforeEach, describe, expect, it, vi } from "vitest"
import type { WatchlistJob } from "@/types/watchlists"

const api = vi.hoisted(() => ({
  fetchScrapedItems: vi.fn(),
  fetchWatchlistContentAlerts: vi.fn(),
  fetchWatchlistJobs: vi.fn(),
  fetchWatchlistOutputs: vi.fn(),
  fetchWatchlistRuns: vi.fn(),
  fetchWatchlistSources: vi.fn(),
  getLatestWatchlistBriefing: vi.fn()
}))

vi.mock("@/services/watchlists", () => api)

import { fetchWatchlistsOverviewData } from "../watchlists-overview"

const emptyPage = { items: [], total: 0, has_more: false }
const monitor = (overrides: Partial<WatchlistJob> = {}): WatchlistJob => ({
  id: 1,
  name: "Manual research monitor",
  scope: { sources: [1] },
  active: true,
  schedule_expr: null,
  wf_schedule_id: null,
  next_run_at: null,
  created_at: "2026-10-09T00:00:00Z",
  ...overrides
})

describe("Watchlist overview schedule attention", () => {
  beforeEach(() => {
    vi.resetAllMocks()
    api.fetchWatchlistSources.mockResolvedValue({
      items: [{
        id: 1, name: "Research feed", url: "https://rss.arxiv.org/rss/cs.AI",
        source_type: "rss", active: true, tags: [], status: "healthy",
        created_at: "2026-10-09T00:00:00Z"
      }],
      total: 1,
      has_more: false
    })
    api.fetchWatchlistJobs.mockResolvedValue(emptyPage)
    api.fetchScrapedItems.mockResolvedValue({ ...emptyPage, total: 50 })
    api.fetchWatchlistContentAlerts.mockResolvedValue(emptyPage)
    api.fetchWatchlistRuns.mockResolvedValue(emptyPage)
    api.fetchWatchlistOutputs.mockResolvedValue(emptyPage)
    api.getLatestWatchlistBriefing.mockResolvedValue(null)
  })

  const overviewFor = async (job: WatchlistJob) => {
    api.fetchWatchlistJobs.mockResolvedValue({ items: [job], total: 1, has_more: false })
    return fetchWatchlistsOverviewData({ watchlist_id: 4 })
  }

  it.each([null, undefined, "", "  "])(
    "does not flag a deliberately manual active monitor with expression %j", async (schedule_expr) => {
      const result = await overviewFor(monitor({ schedule_expr }))
      expect(result.jobs).toMatchObject({
        total: 1, active: 1, attention: 0, nextRunAt: null,
        nextActiveJob: { id: 1, nextRunAt: null, timezone: "UTC" }
      })
      expect(result.health.attention).toEqual({ total: 0, sources: 0, jobs: 0, runs: 0, outputs: 0 })
      expect(result.health.statuses.jobs).toBe("healthy")
      expect(result.systemHealth).toBe("healthy")
      expect(result.items.unread).toBe(50)
      expect(api.fetchWatchlistJobs).toHaveBeenCalledWith({ watchlist_id: 4, page: 1, size: 200 })
    }
  )

  it.each([
    { schedule_expr: "* * * * *" },
    { wf_schedule_id: "scheduler-link-1" }
  ])("does not flag an inactive monitor %j", async (schedule) => {
    const result = await overviewFor(monitor({ ...schedule, active: false }))
    expect(result.jobs.attention).toBe(0)
    expect(result.health.statuses.jobs).toBe("inactive")
  })

  it.each([
    { schedule_expr: "* * * * *", next_run_at: null },
    { schedule_expr: "* * * * *", next_run_at: "" },
    { schedule_expr: "* * * * *", next_run_at: "invalid timestamp" },
    { wf_schedule_id: "scheduler-link-1", next_run_at: null },
    { wf_schedule_id: "scheduler-link-1", next_run_at: "" },
    { wf_schedule_id: "scheduler-link-1", next_run_at: "invalid timestamp" }
  ])("preserves attention for a scheduled active monitor without a valid next run %j", async (schedule) => {
    const result = await overviewFor(monitor(schedule))
    expect(result.jobs.attention).toBe(1)
    expect(result.health.statuses.jobs).toBe("attention")
    expect(result.health.attention.jobs).toBe(1)
    expect(result.systemHealth).toBe("degraded")
  })

  it.each([
    { schedule_expr: "* * * * *" },
    { wf_schedule_id: "scheduler-link-1" }
  ])("preserves a scheduled active monitor's valid next run %j", async (schedule) => {
    const next_run_at = "2026-10-09T12:00:00Z"
    const result = await overviewFor(monitor({ ...schedule, next_run_at }))
    expect(result.jobs).toMatchObject({
      attention: 0,
      nextRunAt: next_run_at,
      nextActiveJob: { id: 1, nextRunAt: next_run_at, timezone: "UTC" }
    })
    expect(result.systemHealth).toBe("healthy")
  })

  it("removes only manual schedule attention while retaining actual source, run and output failures", async () => {
    api.fetchWatchlistSources.mockResolvedValue({
      items: [{
        id: 1, name: "Research feed", url: "https://rss.arxiv.org/rss/cs.AI",
        source_type: "rss", active: true, tags: [], status: "error:403",
        created_at: "2026-10-09T00:00:00Z"
      }],
      total: 1,
      has_more: false
    })
    const failedRun = { id: 2, job_id: 1, status: "failed", started_at: "2026-10-09T01:00:00Z" }
    api.fetchWatchlistRuns.mockImplementation(async (params: { q?: string }) =>
      params.q === "failed"
        ? { items: [failedRun], total: 1, has_more: false }
        : emptyPage
    )
    api.fetchWatchlistOutputs.mockResolvedValue({
      items: [{
        id: 1, run_id: 2, job_id: 1, type: "briefing_markdown", format: "md",
        metadata: { audio_briefing_requested: true, audio_briefing_status: "enqueue_failed" },
        version: 1, expired: false, created_at: "2026-10-09T01:00:00Z"
      }],
      total: 1,
      has_more: false
    })

    const result = await overviewFor(monitor())
    expect(result.health.attention).toEqual({ total: 3, sources: 1, jobs: 0, runs: 1, outputs: 1 })
    expect(result.sources.degraded).toBe(1)
    expect(result.runs.failed).toBe(1)
    expect(result.runs.recentFailed[0]).toMatchObject({ id: 2, job_name: "Manual research monitor" })
    expect(result.outputs.audioIssues).toBe(1)
    expect(result.health.tabBadges).toEqual({ sources: 1, runs: 1, outputs: 1 })
    expect(result.systemHealth).toBe("degraded")
  })
})
