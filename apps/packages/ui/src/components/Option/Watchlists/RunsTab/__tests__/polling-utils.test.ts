import { describe, expect, it } from "vitest"
import {
  hasActiveWatchlistRuns,
  resolveAdaptiveRunNotificationsPollMs,
  resolveRunNotificationsPageSize
} from "../polling-utils"

describe("watchlists polling utilities", () => {
  it("detects active runs for polling start/stop decisions", () => {
    expect(hasActiveWatchlistRuns([{ status: "running" }])).toBe(true)
    expect(hasActiveWatchlistRuns([{ status: "queued" }])).toBe(true)
    expect(hasActiveWatchlistRuns([{ status: "pending" }])).toBe(true)
    expect(
      hasActiveWatchlistRuns([{ status: "completed" }, { status: "succeeded" }, { status: "failed" }])
    ).toBe(false)
  })

  it("resolves adaptive run-notification poll intervals by visibility and workload", () => {
    expect(
      resolveAdaptiveRunNotificationsPollMs(15_000, {
        documentHidden: false,
        hasActiveRuns: true
      })
    ).toBe(15_000)

    expect(
      resolveAdaptiveRunNotificationsPollMs(15_000, {
        documentHidden: false,
        hasActiveRuns: false
      })
    ).toBe(30_000)

    expect(
      resolveAdaptiveRunNotificationsPollMs(15_000, {
        documentHidden: true,
        hasActiveRuns: true
      })
    ).toBe(60_000)
  })

  it("reduces run-notification payload size when hidden or idle", () => {
    expect(
      resolveRunNotificationsPageSize({
        documentHidden: false,
        hasActiveRuns: true
      })
    ).toBe(25)

    expect(
      resolveRunNotificationsPageSize({
        documentHidden: false,
        hasActiveRuns: false
      })
    ).toBe(10)

    expect(
      resolveRunNotificationsPageSize({
        documentHidden: true,
        hasActiveRuns: true
      })
    ).toBe(10)
  })
})
