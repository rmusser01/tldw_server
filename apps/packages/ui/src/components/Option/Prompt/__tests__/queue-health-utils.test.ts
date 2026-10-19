import { describe, expect, it } from "vitest"
import { getQueueSuccessPercent, summarizeQueueHealth } from "../Studio/queue-health-utils"

describe("summarizeQueueHealth", () => {
  it("returns healthy_idle when queue is idle and successful", () => {
    const summary = summarizeQueueHealth({
      queue_depth: 0,
      processing: 0,
      leases: {},
      success_rate: 100
    })

    expect(summary.level).toBe("healthy")
    expect(summary.code).toBe("healthy_idle")
  })

  it("returns healthy_processing when jobs are running", () => {
    const summary = summarizeQueueHealth({
      queue_depth: 2,
      processing: 3,
      leases: {},
      success_rate: 98
    })

    expect(summary.level).toBe("healthy")
    expect(summary.code).toBe("healthy_processing")
    expect(summary.values.processing).toBe(3)
  })

  it("returns degraded_failures when success rate drops with failures", () => {
    const summary = summarizeQueueHealth({
      queue_depth: 2,
      processing: 1,
      leases: {},
      success_rate: 85,
      by_status: { failed: 4 }
    })

    expect(summary.level).toBe("degraded")
    expect(summary.code).toBe("degraded_failures")
    expect(summary.values.failedCount).toBe(4)
  })

  it("returns degraded_backlog when queue depth is too high", () => {
    const summary = summarizeQueueHealth({
      queue_depth: 11,
      processing: 2,
      leases: {},
      success_rate: 99
    })

    expect(summary.level).toBe("degraded")
    expect(summary.code).toBe("degraded_backlog")
    expect(summary.values.queueDepth).toBe(11)
  })

  it.each([
    ["100", 100, 100, "healthy_idle"],
    ["85", 85, 85, "degraded_success_rate"],
    ["0", 0, 0, "degraded_success_rate"],
    ["1", 1, 1, "degraded_success_rate"],
    ["absent", undefined, 0, "degraded_success_rate"]
  ] as const)("uses canonical rate %s for displayed percent and health", (_label, rate, percent, code) => {
    const status = { queue_depth: 0, processing: 0, leases: {}, success_rate: rate }
    expect(getQueueSuccessPercent(status)).toBe(percent)
    expect(summarizeQueueHealth(status).code).toBe(code)
  })
})
