import type {
  ScrapedItem,
  WatchlistJob,
  WatchlistRun,
  WatchlistSource
} from "@/types/watchlists"
import {
  filterSourcesForReader,
  getInitialSourceRenderCount,
  getNextSourceRenderCount,
  orderSourcesForReader,
  resolveSelectedItemId,
  sortItemsForReader
} from "../ItemsTab/items-utils"
import { summarizeFilters, summarizeScopeCounts } from "../JobsTab/job-summaries"
import {
  WATCHLISTS_SCALE_PROFILES,
  WATCHLISTS_SURFACE_PERFORMANCE_BUDGETS,
  type WatchlistsScaleProfile,
  type WatchlistsScaleProfileKey
} from "./scale-profiles"

const now = (): number =>
  typeof performance !== "undefined" && typeof performance.now === "function"
    ? performance.now()
    : Date.now()

const t = (_key: string, defaultValue: string): string => defaultValue

const measure = (operation: () => void): number => {
  const start = now()
  operation()
  return Number((now() - start).toFixed(2))
}

const buildSources = (count: number): WatchlistSource[] =>
  Array.from({ length: count }, (_unused, index) => {
    const id = index + 1
    return {
      id,
      name: `Feed ${id}`,
      url: `https://example.com/feed-${id}.xml`,
      source_type: "rss",
      active: index % 7 !== 0,
      tags: index % 3 === 0 ? ["ai", "research"] : ["news"],
      group_ids: [((index % 5) + 1)],
      status: index % 11 === 0 ? "degraded" : "healthy",
      last_scraped_at: "2026-02-24T08:00:00Z",
      created_at: "2026-02-20T00:00:00Z",
      updated_at: "2026-02-24T08:00:00Z"
    }
  })

const buildJobs = (count: number, sourceCount: number): WatchlistJob[] =>
  Array.from({ length: count }, (_unused, index) => {
    const id = index + 1
    const sourceA = (index % sourceCount) + 1
    const sourceB = ((index + 3) % sourceCount) + 1

    return {
      id,
      name: `Monitor ${id}`,
      description: "Automated briefing monitor",
      active: index % 6 !== 0,
      scope: {
        sources: [sourceA, sourceB],
        groups: [((index % 8) + 1)],
        tags: index % 2 === 0 ? ["ai"] : ["markets"]
      },
      job_filters: {
        filters: [
          {
            type: "keyword",
            action: "include",
            value: {
              keywords: index % 2 === 0 ? ["model", "release"] : ["funding"]
            }
          }
        ]
      },
      output_prefs: {
        template_name: "briefing_markdown",
        generate_audio: index % 4 === 0
      },
      schedule_expr: "0 8 * * *",
      timezone: "UTC",
      created_at: "2026-02-20T00:00:00Z",
      updated_at: "2026-02-24T08:00:00Z",
      last_run_at: "2026-02-24T08:00:00Z",
      next_run_at: "2026-02-24T20:00:00Z"
    }
  })

const buildRuns = (count: number, jobCount: number): WatchlistRun[] =>
  Array.from({ length: count }, (_unused, index) => {
    const id = index + 1
    const statusBucket = index % 5
    const status =
      statusBucket === 0
        ? "failed"
        : statusBucket === 1
          ? "running"
          : statusBucket === 2
            ? "pending"
            : "completed"
    return {
      id,
      job_id: ((index % jobCount) + 1),
      status,
      started_at: "2026-02-24T08:00:00Z",
      finished_at: status === "completed" ? "2026-02-24T08:05:00Z" : null,
      stats: {
        items_found: 12,
        items_ingested: 8,
        items_filtered: 4
      },
      error_msg: status === "failed" ? "Timeout" : null,
      log_path: null
    }
  })

const buildItems = (count: number, sourceCount: number): ScrapedItem[] =>
  Array.from({ length: count }, (_unused, index) => {
    const id = index + 1
    return {
      id,
      run_id: Math.floor(index / 10) + 1,
      job_id: (index % 40) + 1,
      source_id: (index % sourceCount) + 1,
      media_id: null,
      media_uuid: null,
      url: `https://example.com/article-${id}`,
      title: `Article ${id}`,
      summary: `Summary for article ${id}`,
      content: `Body for article ${id}`,
      published_at: "2026-02-24T08:00:00Z",
      tags: index % 3 === 0 ? ["ai"] : ["news"],
      status: "ingested",
      reviewed: index % 4 === 0,
      created_at: "2026-02-24T08:00:00Z"
    }
  })

export interface WatchlistsScaleBenchmarkTimings {
  feedsRenderPrepMs: number
  feedsSearchMutationMs: number
  monitorsRenderPrepMs: number
  monitorsToggleMutationMs: number
  activityRenderPrepMs: number
  articlesRenderPrepMs: number
  articlesBatchMutationMs: number
}

export interface WatchlistsScaleBenchmarkResult {
  profile: WatchlistsScaleProfile
  timings: WatchlistsScaleBenchmarkTimings
  withinBudget: Record<keyof WatchlistsScaleBenchmarkTimings, boolean>
}

export const runWatchlistsScaleBenchmark = (
  profileKey: WatchlistsScaleProfileKey
): WatchlistsScaleBenchmarkResult => {
  const profile = WATCHLISTS_SCALE_PROFILES[profileKey]
  const sources = buildSources(profile.feeds)
  const jobs = buildJobs(profile.monitors, profile.feeds)
  const runs = buildRuns(profile.runs, profile.monitors)
  const items = buildItems(profile.items, profile.feeds)

  const timings: WatchlistsScaleBenchmarkTimings = {
    feedsRenderPrepMs: measure(() => {
      const ordered = orderSourcesForReader(sources, null)
      const initialCount = getInitialSourceRenderCount(ordered.length, "")
      void getNextSourceRenderCount(initialCount, ordered.length)
    }),
    feedsSearchMutationMs: measure(() => {
      void filterSourcesForReader(sources, "feed 1")
    }),
    monitorsRenderPrepMs: measure(() => {
      void jobs.map((job) => ({
        scope: summarizeScopeCounts(job.scope, t),
        filters: summarizeFilters(job.job_filters?.filters, t).preview
      }))
    }),
    monitorsToggleMutationMs: measure(() => {
      const toggledIds = new Set(
        jobs.slice(0, Math.min(200, jobs.length)).map((job) => job.id)
      )
      void jobs.map((job) =>
        toggledIds.has(job.id) ? { ...job, active: !job.active } : job
      )
    }),
    activityRenderPrepMs: measure(() => {
      const sorted = [...runs].sort((left, right) => right.id - left.id)
      void sorted.reduce<Record<string, number>>((acc, run) => {
        acc[run.status] = (acc[run.status] || 0) + 1
        return acc
      }, {})
    }),
    articlesRenderPrepMs: measure(() => {
      const sorted = sortItemsForReader(items, "unreadFirst")
      void resolveSelectedItemId(null, sorted)
    }),
    articlesBatchMutationMs: measure(() => {
      const targetIds = new Set(
        items.slice(0, Math.min(500, items.length)).map((item) => item.id)
      )
      void items.map((item) =>
        targetIds.has(item.id) ? { ...item, reviewed: true } : item
      )
    })
  }

  const withinBudget: Record<keyof WatchlistsScaleBenchmarkTimings, boolean> = {
    feedsRenderPrepMs:
      timings.feedsRenderPrepMs <= WATCHLISTS_SURFACE_PERFORMANCE_BUDGETS.feeds.renderLatencyMs,
    feedsSearchMutationMs:
      timings.feedsSearchMutationMs <=
      WATCHLISTS_SURFACE_PERFORMANCE_BUDGETS.feeds.interactionLatencyMs,
    monitorsRenderPrepMs:
      timings.monitorsRenderPrepMs <=
      WATCHLISTS_SURFACE_PERFORMANCE_BUDGETS.monitors.renderLatencyMs,
    monitorsToggleMutationMs:
      timings.monitorsToggleMutationMs <=
      WATCHLISTS_SURFACE_PERFORMANCE_BUDGETS.monitors.interactionLatencyMs,
    activityRenderPrepMs:
      timings.activityRenderPrepMs <=
      WATCHLISTS_SURFACE_PERFORMANCE_BUDGETS.activity.renderLatencyMs,
    articlesRenderPrepMs:
      timings.articlesRenderPrepMs <=
      WATCHLISTS_SURFACE_PERFORMANCE_BUDGETS.articles.renderLatencyMs,
    articlesBatchMutationMs:
      timings.articlesBatchMutationMs <=
      WATCHLISTS_SURFACE_PERFORMANCE_BUDGETS.articles.interactionLatencyMs
  }

  return {
    profile,
    timings,
    withinBudget
  }
}
