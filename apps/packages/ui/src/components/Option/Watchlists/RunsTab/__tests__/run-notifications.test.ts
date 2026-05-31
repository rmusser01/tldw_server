import { describe, expect, it } from "vitest"
import {
  type RunNotificationEvent,
  buildRunStateNotificationKey,
  dedupeRunNotificationEvents,
  getRunFailureHint,
  groupRunNotificationEvents,
  resolveRunNotificationsPollPlan,
  resolveStalledRunNotification,
  resolveRunTransitionNotification,
  shouldNotifyNewTerminalRun
} from "../run-notifications"

describe("run notification helpers", () => {
  const translatedHints: Record<string, string> = {
    "watchlists:notifications.failureHints.timeout": "Localized timeout hint",
    "watchlists:notifications.failureHints.auth": "Localized auth hint"
  }
  const t = (key: string, defaultValue?: string) =>
    translatedHints[key] ?? defaultValue ?? key

  it("returns completed notification when run transitions from running", () => {
    const result = resolveRunTransitionNotification("running", {
      status: "completed",
      error_msg: null
    })
    expect(result).toEqual({ kind: "completed" })
  })

  it("returns completed notification when backend reports succeeded", () => {
    const result = resolveRunTransitionNotification("running", {
      status: "succeeded",
      error_msg: null
    })
    expect(result).toEqual({ kind: "completed" })
    expect(buildRunStateNotificationKey(9, "succeeded")).toBe("9:completed")
  })

  it("returns failed notification and remediation hint for failed transitions", () => {
    const result = resolveRunTransitionNotification(
      "pending",
      {
        status: "failed",
        error_msg: "timeout while fetching"
      },
      t
    )
    expect(result).toEqual({
      kind: "failed",
      hint: "Localized timeout hint"
    })
  })

  it("does not notify without a previous status transition", () => {
    const result = resolveRunTransitionNotification(null, {
      status: "failed",
      error_msg: "403 forbidden"
    })
    expect(result).toBeNull()
  })

  it("notifies for newly observed terminal runs that finished after session start", () => {
    const sessionStart = Date.parse("2026-02-18T09:00:00Z")
    expect(
      shouldNotifyNewTerminalRun(
        {
          status: "succeeded",
          finished_at: "2026-02-18T09:01:00Z"
        },
        sessionStart
      )
    ).toBe(true)
    expect(
      shouldNotifyNewTerminalRun(
        {
          status: "completed",
          finished_at: "2026-02-18T09:01:00Z"
        },
        sessionStart
      )
    ).toBe(true)
  })

  it("skips notifications for old runs or cancelled runs", () => {
    const sessionStart = Date.parse("2026-02-18T09:00:00Z")
    expect(
      shouldNotifyNewTerminalRun(
        {
          status: "failed",
          finished_at: "2026-02-18T08:59:00Z"
        },
        sessionStart
      )
    ).toBe(false)
    expect(
      shouldNotifyNewTerminalRun(
        {
          status: "cancelled",
          finished_at: "2026-02-18T09:05:00Z"
        },
        sessionStart
      )
    ).toBe(false)
  })

  it("maps common failure modes to actionable hints", () => {
    expect(getRunFailureHint("403 Forbidden")).toContain("authentication")
    expect(getRunFailureHint("rate limit exceeded")).toContain("rate-limiting")
    expect(getRunFailureHint("dns lookup failed")).toContain("resolved")
    expect(getRunFailureHint("")).toContain("inspect logs")
  })

  it("resolves localized hint keys when translator is provided", () => {
    expect(getRunFailureHint("403 Forbidden", t)).toBe("Localized auth hint")
    expect(getRunFailureHint("timeout while fetching", t)).toBe("Localized timeout hint")
  })

  it("falls back to default copy when translator is missing key", () => {
    const missingTranslator = (key: string) => key
    expect(getRunFailureHint("timeout while fetching", missingTranslator)).toBe(
      "The source request timed out. Retry, or lower concurrency for this source."
    )
  })

  it("detects stalled active runs and returns localized remediation hint", () => {
    const sessionNow = Date.parse("2026-02-18T10:00:00Z")
    const stalled = resolveStalledRunNotification(
      {
        id: 55,
        status: "running",
        started_at: "2026-02-18T09:00:00Z",
        finished_at: null
      },
      sessionNow,
      20 * 60_000
    )
    expect(stalled).not.toBeNull()
    expect(stalled?.kind).toBe("stalled")
    expect(stalled?.eventKey).toBe("55:stalled")
    expect(stalled?.hint).toContain("stalled")
  })

  it("dedupes repeat notification keys and groups events with deep-link payloads", () => {
    const seen = new Set<string>()
    const deduped = dedupeRunNotificationEvents(
      [
        {
          eventKey: buildRunStateNotificationKey(11, "failed"),
          kind: "failed",
          runId: 11,
          hint: "hint A"
        },
        {
          eventKey: buildRunStateNotificationKey(11, "failed"),
          kind: "failed",
          runId: 11,
          hint: "hint A"
        },
        {
          eventKey: buildRunStateNotificationKey(13, "failed"),
          kind: "failed",
          runId: 13,
          hint: "hint B"
        },
        {
          eventKey: buildRunStateNotificationKey(9, "completed"),
          kind: "completed",
          runId: 9
        }
      ],
      seen
    )

    expect(deduped).toHaveLength(3)
    expect(seen.has("11:failed")).toBe(true)
    expect(seen.has("13:failed")).toBe(true)
    expect(seen.has("9:completed")).toBe(true)

    const grouped = groupRunNotificationEvents(deduped)
    expect(grouped).toHaveLength(2)
    expect(grouped[0]).toEqual(
      expect.objectContaining({
        kind: "failed",
        count: 2,
        runIds: [11, 13],
        deepLinkRunId: 13,
        hint: "hint A"
      })
    )
    expect(grouped[1]).toEqual(
      expect.objectContaining({
        kind: "completed",
        count: 1,
        runIds: [9],
        deepLinkRunId: 9
      })
    )
  })

  it("groups high-volume notification bursts without losing newest deep-link target", () => {
    const seen = new Set<string>()
    const events: RunNotificationEvent[] = [
      ...Array.from({ length: 120 }, (_, index) => ({
        eventKey: buildRunStateNotificationKey(index + 1, "failed"),
        kind: "failed" as const,
        runId: index + 1,
        hint: index === 0 ? "first hint" : null
      })),
      {
        eventKey: buildRunStateNotificationKey(8, "completed"),
        kind: "completed" as const,
        runId: 8
      },
      {
        eventKey: buildRunStateNotificationKey(8, "completed"),
        kind: "completed" as const,
        runId: 8
      }
    ]

    const deduped = dedupeRunNotificationEvents(events, seen)
    const grouped = groupRunNotificationEvents(deduped)

    expect(deduped).toHaveLength(121)
    expect(grouped[0]).toEqual(
      expect.objectContaining({
        kind: "failed",
        count: 120,
        deepLinkRunId: 120
      })
    )
    expect(grouped[1]).toEqual(
      expect.objectContaining({
        kind: "completed",
        count: 1,
        deepLinkRunId: 8
      })
    )
  })

  it("resolves adaptive poll plans for active, hidden, and runs-focused states", () => {
    expect(
      resolveRunNotificationsPollPlan({
        isOnline: false,
        activeTab: "sources",
        runsPollingActive: false,
        documentVisible: true,
        baseIntervalMs: 15000
      })
    ).toEqual(
      expect.objectContaining({
        enabled: false,
        pageSize: 25,
        suppressCompleted: false
      })
    )

    expect(
      resolveRunNotificationsPollPlan({
        isOnline: true,
        activeTab: "runs",
        runsPollingActive: true,
        documentVisible: true,
        baseIntervalMs: 15000
      })
    ).toEqual(
      expect.objectContaining({
        enabled: false,
        pageSize: 10,
        suppressCompleted: true
      })
    )

    expect(
      resolveRunNotificationsPollPlan({
        isOnline: true,
        activeTab: "sources",
        runsPollingActive: false,
        documentVisible: false,
        baseIntervalMs: 15000
      })
    ).toEqual(
      expect.objectContaining({
        enabled: true,
        intervalMs: 60000,
        pageSize: 10,
        suppressCompleted: true
      })
    )

    expect(
      resolveRunNotificationsPollPlan({
        isOnline: true,
        activeTab: "sources",
        runsPollingActive: false,
        documentVisible: true,
        baseIntervalMs: 15000
      })
    ).toEqual(
      expect.objectContaining({
        enabled: true,
        intervalMs: 15000,
        pageSize: 25,
        suppressCompleted: false
      })
    )
  })
})
