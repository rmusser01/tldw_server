import { beforeEach, describe, expect, it, vi } from "vitest"

import {
  clearCapabilityProbeCacheForTests,
  serverSupportsPath
} from "@/services/tldw/capability-probe"

const fetchMock = vi.fn()
vi.stubGlobal("fetch", fetchMock)

const SERVER_URL = "http://127.0.0.1:8000"

const okSpec = (paths: Record<string, unknown>) => ({
  ok: true,
  json: async () => ({ paths })
})

describe("serverSupportsPath", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    clearCapabilityProbeCacheForTests()
  })

  it("test_fetches_spec_once_for_many_paths", async () => {
    fetchMock.mockResolvedValue(
      okSpec({
        "/api/v1/admin/billing/overview": {},
        "/api/v1/admin/rate-limits": {}
      })
    )

    await expect(
      serverSupportsPath(SERVER_URL, "/api/v1/admin/billing/overview")
    ).resolves.toBe(true)
    await expect(
      serverSupportsPath(SERVER_URL, "/api/v1/admin/rate-limits")
    ).resolves.toBe(true)

    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(fetchMock).toHaveBeenCalledWith(`${SERVER_URL}/openapi.json`)
  })

  it("test_failure_is_cached_short_term", async () => {
    fetchMock.mockRejectedValue(new Error("network down"))

    // A failed probe is UNKNOWN (null), not "absent": the caller decides how
    // to proceed; only a fetched spec may answer false.
    await expect(
      serverSupportsPath(SERVER_URL, "/api/v1/admin/billing/overview")
    ).resolves.toBeNull()
    // Second call inside the failure TTL answers null without refetching.
    await expect(
      serverSupportsPath(SERVER_URL, "/api/v1/admin/billing/overview")
    ).resolves.toBeNull()

    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it("test_concurrent_calls_share_inflight_request", async () => {
    let resolveFetch!: (value: unknown) => void
    fetchMock.mockImplementation(
      () => new Promise((resolve) => { resolveFetch = resolve })
    )

    const first = serverSupportsPath(SERVER_URL, "/api/v1/a")
    const second = serverSupportsPath(SERVER_URL, "/api/v1/b")

    // Both calls must consume the same in-flight spec: one fetch, one answer
    // set shared by both paths.
    resolveFetch(okSpec({ "/api/v1/a": {} }))

    await expect(first).resolves.toBe(true)
    await expect(second).resolves.toBe(false)

    expect(fetchMock).toHaveBeenCalledTimes(1)
  })
})
