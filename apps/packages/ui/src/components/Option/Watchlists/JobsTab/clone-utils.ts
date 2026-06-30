import type { WatchlistJob, WatchlistJobCreate } from "@/types/watchlists"

const cloneValue = <T>(value: T): T => {
  if (value == null) return value
  if (typeof structuredClone === "function") {
    return structuredClone(value)
  }
  return JSON.parse(JSON.stringify(value)) as T
}

export const buildClonedWatchlistJobPayload = (
  job: WatchlistJob
): WatchlistJobCreate => ({
  name: `${job.name} copy`,
  description: job.description || undefined,
  watchlist_id: job.watchlist_id ?? undefined,
  scope: cloneValue(job.scope),
  schedule_expr: job.schedule_expr || undefined,
  timezone: job.timezone || undefined,
  active: false,
  max_concurrency: job.max_concurrency ?? undefined,
  per_host_delay_ms: job.per_host_delay_ms ?? undefined,
  retry_policy: job.retry_policy ? cloneValue(job.retry_policy) : undefined,
  output_prefs: job.output_prefs ? cloneValue(job.output_prefs) : undefined,
  job_filters: job.job_filters ? cloneValue(job.job_filters) : undefined
})
