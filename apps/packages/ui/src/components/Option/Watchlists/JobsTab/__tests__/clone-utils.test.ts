import { describe, expect, it } from "vitest"
import { buildClonedWatchlistJobPayload } from "../clone-utils"
import type { WatchlistJob } from "@/types/watchlists"

const buildJob = (): WatchlistJob => ({
  id: 42,
  name: "Morning digest",
  description: "Collects national security news",
  watchlist_id: 7,
  scope: {
    sources: [10, 11],
    groups: [5],
    tags: ["news", "osint"]
  },
  schedule_expr: "0 */5 * * *",
  timezone: "America/New_York",
  active: true,
  max_concurrency: 3,
  per_host_delay_ms: 2500,
  retry_policy: { max_attempts: 2, backoff_seconds: 60 },
  job_filters: {
    filters: [
      { type: "keyword", action: "include", value: { query: "sanctions", case_sensitive: false } },
      { type: "regex", action: "exclude", value: { pattern: "sports", case_sensitive: false } }
    ]
  },
  output_prefs: {
    auto_output: { enabled: true, format: "md" },
    generate_audio: true,
    audio_cast: {
      speaker_count: 2,
      speakers: [
        { id: "host", label: "Host", role: "anchor", voice: "af_bella" },
        { id: "analyst", label: "Analyst", role: "analysis", voice: "am_adam" }
      ]
    },
    deliveries: {
      email: { enabled: true, recipients: ["briefing@example.com"] },
      chatbook: { enabled: true, title: "Daily brief" }
    }
  },
  created_at: "2026-05-01T12:00:00Z",
  updated_at: "2026-05-01T12:30:00Z",
  last_run_at: "2026-05-01T13:00:00Z",
  next_run_at: "2026-05-01T18:00:00Z",
  wf_schedule_id: "wf-schedule-42"
})

describe("buildClonedWatchlistJobPayload", () => {
  it("preserves monitor scope, cadence, filters, output, delivery, and audio config while resetting runtime state", () => {
    const original = buildJob()

    const clone = buildClonedWatchlistJobPayload(original)

    expect(clone).toEqual({
      name: "Morning digest copy",
      description: "Collects national security news",
      watchlist_id: 7,
      scope: original.scope,
      schedule_expr: "0 */5 * * *",
      timezone: "America/New_York",
      active: false,
      max_concurrency: 3,
      per_host_delay_ms: 2500,
      retry_policy: original.retry_policy,
      job_filters: original.job_filters,
      output_prefs: original.output_prefs
    })
    expect(JSON.stringify(clone)).not.toContain("last_run_at")
    expect(JSON.stringify(clone)).not.toContain("next_run_at")
    expect(JSON.stringify(clone)).not.toContain("wf_schedule_id")
  })

  it("deep clones nested config so edits to the clone cannot mutate the original monitor", () => {
    const original = buildJob()
    const clone = buildClonedWatchlistJobPayload(original)

    clone.scope.sources?.push(999)
    clone.output_prefs!.deliveries!.email!.recipients!.push("copy@example.com")

    expect(original.scope.sources).toEqual([10, 11])
    expect(original.output_prefs?.deliveries?.email?.recipients).toEqual(["briefing@example.com"])
  })
})
