import { afterEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({ bgRequest: vi.fn() }))
vi.mock("@/services/background-proxy", () => ({
  bgRequest: (...args: unknown[]) => mocks.bgRequest(...args)
}))

import { checkWatchlistSourcesNow, triggerWatchlistRun } from "@/services/watchlists"

afterEach(() => {
  vi.useRealTimers()
  vi.clearAllMocks()
})

describe("Watchlist synchronous worker requests", () => {
  it.each([
    ["Check Now", () => checkWatchlistSourcesNow([7])],
    ["Run Now", () => triggerWatchlistRun(9)]
  ] as const)("%s can finish after the generic ten-second request budget", async (_name, action) => {
    vi.useFakeTimers()
    const response = { id: 12, status: "succeeded" }
    mocks.bgRequest.mockImplementation(({ timeoutMs = 10000 }: { timeoutMs?: number }) =>
      new Promise((resolve, reject) => {
        const abort = setTimeout(() => reject(new Error("request timeout")), timeoutMs)
        setTimeout(() => { clearTimeout(abort); resolve(response) }, 18000)
      })
    )
    let settled = false
    const pending = action().then(
      value => { settled = true; return { value } },
      error => { settled = true; return { error } }
    )
    await vi.advanceTimersByTimeAsync(10001)
    expect(settled).toBe(false)
    await vi.advanceTimersByTimeAsync(8000)
    expect(await pending).toEqual({ value: response })
    expect(mocks.bgRequest).toHaveBeenCalledTimes(1)
    expect(mocks.bgRequest).toHaveBeenCalledWith(expect.objectContaining({ timeoutMs: 600000 }))
  })
})
