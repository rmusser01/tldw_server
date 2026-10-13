import { describe, expect, it, vi } from "vitest"
import { adminMethods } from "../domains/admin"
import { bgRequest } from "@/services/background-proxy"

vi.mock("@/services/background-proxy", () => ({ bgRequest: vi.fn() }))

/**
 * B-S5: the four usage methods pass optional `start`/`end` straight through
 * to the documented backend params — YYYY-MM-DD on /usage/top, ISO
 * timestamps on the llm-usage* endpoints (admin_usage.py).
 */
const queryOf = (params: Record<string, string | number>): string =>
  `?${new URLSearchParams(
    Object.entries(params).map(([key, value]) => [key, String(value)])
  ).toString()}`

describe("admin usage analytics date-range passthrough", () => {
  it("sends start/end on getTopUsage alongside the existing filters", async () => {
    vi.mocked(bgRequest).mockResolvedValue({})
    await adminMethods.getTopUsage({
      metric: "requests",
      limit: 20,
      start: "2026-09-06",
      end: "2026-10-06"
    })
    expect(bgRequest).toHaveBeenCalledWith({
      path: `/api/v1/admin/usage/top${queryOf({
        metric: "requests",
        limit: 20,
        start: "2026-09-06",
        end: "2026-10-06"
      })}`,
      method: "GET"
    })
  })

  it("sends ISO start/end on getLlmUsage, summary, and top-spenders", async () => {
    vi.mocked(bgRequest).mockResolvedValue({})
    const start = "2026-09-06T00:00:00.000Z"
    const end = "2026-10-06T23:59:59.999Z"

    await adminMethods.getLlmUsage({ limit: 50, start, end })
    expect(bgRequest).toHaveBeenLastCalledWith({
      path: `/api/v1/admin/llm-usage${queryOf({ limit: 50, start, end })}`,
      method: "GET"
    })

    await adminMethods.getLlmUsageSummary({ start, end })
    expect(bgRequest).toHaveBeenLastCalledWith({
      path: `/api/v1/admin/llm-usage/summary${queryOf({ start, end })}`,
      method: "GET"
    })

    await adminMethods.getLlmTopSpenders({ limit: 10, start, end })
    expect(bgRequest).toHaveBeenLastCalledWith({
      path: `/api/v1/admin/llm-usage/top-spenders${queryOf({
        limit: 10,
        start,
        end
      })}`,
      method: "GET"
    })
  })

  it("stays backward compatible when start/end are omitted", async () => {
    vi.mocked(bgRequest).mockResolvedValue({})
    await adminMethods.getTopUsage({ limit: 20 })
    expect(bgRequest).toHaveBeenLastCalledWith({
      path: `/api/v1/admin/usage/top${queryOf({ limit: 20 })}`,
      method: "GET"
    })

    await adminMethods.getLlmUsageSummary()
    expect(bgRequest).toHaveBeenLastCalledWith({
      path: "/api/v1/admin/llm-usage/summary",
      method: "GET"
    })

    await adminMethods.getLlmTopSpenders({ limit: 10 })
    expect(bgRequest).toHaveBeenLastCalledWith({
      path: `/api/v1/admin/llm-usage/top-spenders${queryOf({ limit: 10 })}`,
      method: "GET"
    })

    await adminMethods.getLlmUsage({ limit: 50 })
    expect(bgRequest).toHaveBeenLastCalledWith({
      path: `/api/v1/admin/llm-usage${queryOf({ limit: 50 })}`,
      method: "GET"
    })
  })
})
