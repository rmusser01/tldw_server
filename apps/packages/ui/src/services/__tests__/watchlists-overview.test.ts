import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  fetchScrapedItems: vi.fn(),
  fetchWatchlistContentAlerts: vi.fn(),
  fetchWatchlistJobs: vi.fn(),
  fetchWatchlistOutputs: vi.fn(),
  fetchWatchlistRuns: vi.fn(),
  fetchWatchlistSources: vi.fn()
}))

vi.mock("@/services/watchlists", () => ({
  fetchScrapedItems: (...args: unknown[]) => mocks.fetchScrapedItems(...args),
  fetchWatchlistContentAlerts: (...args: unknown[]) => mocks.fetchWatchlistContentAlerts(...args),
  fetchWatchlistJobs: (...args: unknown[]) => mocks.fetchWatchlistJobs(...args),
  fetchWatchlistOutputs: (...args: unknown[]) => mocks.fetchWatchlistOutputs(...args),
  fetchWatchlistRuns: (...args: unknown[]) => mocks.fetchWatchlistRuns(...args),
  fetchWatchlistSources: (...args: unknown[]) => mocks.fetchWatchlistSources(...args)
}))

import {
  buildOverviewHealthModel,
  classifySourceHealth,
  fetchWatchlistsOverviewData,
  getEarliestNextRunAt,
  getOverviewTabBadges
} from "../watchlists-overview"

describe("watchlists overview service", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("classifies source health from active flag and status", () => {
    expect(
      classifySourceHealth({
        id: 1,
        name: "A",
        url: "https://a.example",
        source_type: "rss",
        active: true,
        tags: [],
        status: "healthy",
        created_at: "2026-02-18T00:00:00Z"
      })
    ).toBe("healthy")
    expect(
      classifySourceHealth({
        id: 2,
        name: "B",
        url: "https://b.example",
        source_type: "rss",
        active: true,
        tags: [],
        status: "error",
        created_at: "2026-02-18T00:00:00Z"
      })
    ).toBe("degraded")
    expect(
      classifySourceHealth({
        id: 3,
        name: "C",
        url: "https://c.example",
        source_type: "rss",
        active: false,
        tags: [],
        status: "ok",
        created_at: "2026-02-18T00:00:00Z"
      })
    ).toBe("inactive")
    expect(
      classifySourceHealth({
        id: 4,
        name: "D",
        url: "https://d.example",
        source_type: "rss",
        active: true,
        tags: [],
        status: "",
        created_at: "2026-02-18T00:00:00Z"
      })
    ).toBe("unknown")
    expect(
      classifySourceHealth({
        id: 5,
        name: "E",
        url: "https://e.example",
        source_type: "rss",
        active: true,
        tags: [],
        status: "error:403",
        created_at: "2026-02-18T00:00:00Z"
      })
    ).toBe("degraded")
  })

  it("picks earliest next run from active jobs only", () => {
    const earliest = getEarliestNextRunAt([
      { active: false, next_run_at: "2026-02-21T12:00:00Z" },
      { active: true, next_run_at: "2026-02-21T08:00:00Z" },
      { active: true, next_run_at: "2026-02-20T08:00:00Z" },
      { active: true, next_run_at: null }
    ])
    expect(earliest).toBe("2026-02-20T08:00:00Z")
  })

  it("aggregates counts and returns degraded health when failures exist", async () => {
    mocks.fetchWatchlistSources.mockResolvedValueOnce({
      items: [
        {
          id: 1,
          name: "Healthy Source",
          url: "https://healthy.example/rss",
          source_type: "rss",
          active: true,
          tags: [],
          status: "ok",
          created_at: "2026-02-18T00:00:00Z"
        },
        {
          id: 2,
          name: "Failing Source",
          url: "https://failing.example/rss",
          source_type: "rss",
          active: true,
          tags: [],
          status: "error",
          created_at: "2026-02-18T00:00:00Z"
        }
      ],
      total: 2,
      has_more: false
    })
    mocks.fetchWatchlistJobs.mockResolvedValueOnce({
      items: [
        {
          id: 10,
          name: "Morning Digest",
          scope: {},
          active: true,
          created_at: "2026-02-18T00:00:00Z",
          next_run_at: "2026-02-20T08:00:00Z"
        },
        {
          id: 11,
          name: "Paused Digest",
          scope: {},
          active: false,
          created_at: "2026-02-18T00:00:00Z",
          next_run_at: "2026-02-21T08:00:00Z"
        }
      ],
      total: 2,
      has_more: false
    })
    mocks.fetchScrapedItems.mockResolvedValueOnce({
      items: [],
      total: 42
    })
    mocks.fetchWatchlistContentAlerts.mockResolvedValueOnce({
      items: [],
      total: 3
    })
    mocks.fetchWatchlistRuns
      .mockResolvedValueOnce({ items: [], total: 1 })
      .mockResolvedValueOnce({ items: [], total: 2 })
      .mockResolvedValueOnce({
        items: [
          {
            id: 91,
            job_id: 10,
            status: "failed",
            error_msg: "403 forbidden",
            started_at: "2026-02-18T10:00:00Z",
            finished_at: "2026-02-18T10:01:00Z"
          }
        ],
        total: 1
      })
      .mockResolvedValueOnce({ items: [], total: 0, has_more: false })
    mocks.fetchWatchlistOutputs.mockResolvedValueOnce({
      items: [
        {
          id: 701,
          run_id: 91,
          job_id: 10,
          type: "briefing_markdown",
          format: "md",
          metadata: {
            deliveries: {
              email: "failed"
            }
          },
          version: 1,
          expired: false,
          created_at: "2026-02-18T10:05:00Z"
        },
        {
          id: 702,
          run_id: 92,
          job_id: 10,
          type: "briefing_markdown",
          format: "md",
          metadata: null,
          version: 1,
          expired: true,
          created_at: "2026-02-18T11:05:00Z"
        }
      ],
      total: 2,
      has_more: false
    })

    const result = await fetchWatchlistsOverviewData({ watchlist_id: 42 })

    expect(result.sources).toEqual({
      total: 2,
      healthy: 1,
      degraded: 1,
      inactive: 0,
      unknown: 0
    })
    expect(result.jobs.total).toBe(2)
    expect(result.jobs.active).toBe(1)
    expect(result.jobs.nextRunAt).toBe("2026-02-20T08:00:00Z")
    expect(result.jobs.attention).toBe(0)
    expect(result.items.unread).toBe(42)
    expect(result.alerts.unread).toBe(3)
    expect(result.runs.running).toBe(1)
    expect(result.runs.pending).toBe(2)
    expect(result.runs.failed).toBe(1)
    expect(result.outputs).toEqual({
      total: 2,
      expired: 1,
      deliveryIssues: 1,
      audioIssues: 0,
      attention: 2
    })
    expect(result.health.attention).toEqual({
      total: 4,
      sources: 1,
      jobs: 0,
      runs: 1,
      outputs: 2
    })
    expect(result.health.statuses.outputs).toBe("attention")
    expect(result.health.tabBadges).toEqual({
      sources: 1,
      runs: 1,
      outputs: 2
    })
    expect(result.runs.recentFailed).toEqual([
      expect.objectContaining({
        id: 91,
        job_id: 10,
        job_name: "Morning Digest",
        status: "failed"
      })
    ])
    expect(result.systemHealth).toBe("degraded")
  })

  it("forwards selected Watchlist scope into aggregate fetches", async () => {
    mocks.fetchWatchlistSources.mockResolvedValue({
      items: [],
      total: 0,
      has_more: false
    })
    mocks.fetchWatchlistJobs.mockResolvedValue({
      items: [],
      total: 0,
      has_more: false
    })
    mocks.fetchScrapedItems.mockResolvedValue({
      items: [],
      total: 0
    })
    mocks.fetchWatchlistContentAlerts.mockResolvedValue({
      items: [],
      total: 0
    })
    mocks.fetchWatchlistRuns.mockResolvedValue({
      items: [],
      total: 0
    })
    mocks.fetchWatchlistOutputs.mockResolvedValue({
      items: [],
      total: 0,
      has_more: false
    })

    await fetchWatchlistsOverviewData({ watchlist_id: 42 })

    expect(mocks.fetchWatchlistSources).toHaveBeenCalledWith({
      watchlist_id: 42,
      page: 1,
      size: 200
    })
    expect(mocks.fetchWatchlistJobs).toHaveBeenCalledWith({
      watchlist_id: 42,
      page: 1,
      size: 200
    })
    expect(mocks.fetchScrapedItems).toHaveBeenCalledWith({
      watchlist_id: 42,
      reviewed: false,
      page: 1,
      size: 1
    })
    expect(mocks.fetchWatchlistContentAlerts).toHaveBeenCalledWith(42, {
      status: "unread",
      page: 1,
      size: 1
    })
    expect(mocks.fetchWatchlistRuns).toHaveBeenNthCalledWith(1, {
      watchlist_id: 42,
      q: "running",
      page: 1,
      size: 1
    })
    expect(mocks.fetchWatchlistRuns).toHaveBeenNthCalledWith(2, {
      watchlist_id: 42,
      q: "pending",
      page: 1,
      size: 1
    })
    expect(mocks.fetchWatchlistRuns).toHaveBeenNthCalledWith(3, {
      watchlist_id: 42,
      q: "failed",
      page: 1,
      size: 5
    })
    expect(mocks.fetchWatchlistRuns).toHaveBeenNthCalledWith(4, {
      watchlist_id: 42,
      page: 1,
      size: 10
    })
    expect(mocks.fetchWatchlistOutputs).toHaveBeenCalledWith({
      watchlist_id: 42,
      page: 1,
      size: 100
    })
  })

  it("marks source-error zero-item runs and failed audio outputs as attention", async () => {
    mocks.fetchWatchlistSources.mockResolvedValueOnce({
      items: [
        {
          id: 1,
          name: "Blocked Source",
          url: "https://blocked.example/rss",
          source_type: "rss",
          active: true,
          tags: [],
          status: "error:403",
          created_at: "2026-02-18T00:00:00Z"
        }
      ],
      total: 1,
      has_more: false
    })
    mocks.fetchWatchlistJobs.mockResolvedValueOnce({
      items: [
        {
          id: 10,
          name: "Digest",
          scope: {},
          active: true,
          created_at: "2026-02-18T00:00:00Z",
          next_run_at: "2026-02-20T08:00:00Z"
        }
      ],
      total: 1,
      has_more: false
    })
    mocks.fetchScrapedItems.mockResolvedValueOnce({ items: [], total: 0 })
    mocks.fetchWatchlistContentAlerts.mockResolvedValueOnce({ items: [], total: 0 })
    mocks.fetchWatchlistRuns
      .mockResolvedValueOnce({ items: [], total: 0 })
      .mockResolvedValueOnce({ items: [], total: 0 })
      .mockResolvedValueOnce({ items: [], total: 0 })
      .mockResolvedValueOnce({
        items: [
          {
            id: 101,
            job_id: 10,
            status: "succeeded",
            started_at: "2026-02-18T10:00:00Z",
            finished_at: "2026-02-18T10:01:00Z",
            stats: {
              items_found: 0,
              items_ingested: 0,
              source_errors: 1,
              source_statuses: [
                {
                  source_id: 1,
                  name: "Blocked Source",
                  status: "error:403",
                  error: "HTTP 403",
                  items_found: 0,
                  items_ingested: 0
                }
              ]
            } as any
          }
        ],
        total: 1,
        has_more: false
      })
    mocks.fetchWatchlistOutputs.mockResolvedValueOnce({
      items: [
        {
          id: 701,
          run_id: 101,
          job_id: 10,
          type: "briefing_markdown",
          format: "md",
          metadata: {
            audio_briefing_requested: true,
            audio_briefing_status: "enqueue_failed"
          },
          version: 1,
          expired: false,
          created_at: "2026-02-18T10:05:00Z"
        },
        {
          id: 702,
          run_id: 101,
          job_id: 10,
          type: "briefing_markdown",
          format: "md",
          metadata: {
            audio: {
              requested: true,
              status: "skipped"
            }
          },
          version: 1,
          expired: false,
          created_at: "2026-02-18T10:06:00Z"
        }
      ],
      total: 2,
      has_more: false
    })

    const result = await fetchWatchlistsOverviewData()

    expect(result.sources.degraded).toBe(1)
    expect(result.runs.failed).toBe(0)
    expect(result.runs.sourceErrors).toBe(1)
    expect(result.runs.zeroItemSourceErrors).toBe(1)
    expect(result.outputs.audioIssues).toBe(2)
    expect(result.health.statuses.sources).toBe("attention")
    expect(result.health.statuses.runs).toBe("attention")
    expect(result.health.statuses.outputs).toBe("attention")
    expect(result.health.attention).toEqual({
      total: 4,
      sources: 1,
      jobs: 0,
      runs: 1,
      outputs: 2
    })
    expect(result.systemHealth).toBe("degraded")
  })

  it("derives health model and tab badges from aggregate counters", () => {
    const model = buildOverviewHealthModel({
      sources: { total: 3, degraded: 0, inactive: 3 },
      jobs: { total: 2, active: 0, attention: 1 },
      runs: { running: 0, pending: 0, failed: 0 },
      outputs: { total: 5, attention: 0 }
    })

    expect(model.statuses).toEqual({
      sources: "inactive",
      jobs: "attention",
      runs: "unknown",
      outputs: "healthy"
    })
    expect(model.attention.total).toBe(1)
    expect(getOverviewTabBadges(model)).toEqual({
      sources: 0,
      runs: 0,
      outputs: 0
    })
    expect(getOverviewTabBadges(null)).toEqual({
      sources: 0,
      runs: 0,
      outputs: 0
    })
  })
})
