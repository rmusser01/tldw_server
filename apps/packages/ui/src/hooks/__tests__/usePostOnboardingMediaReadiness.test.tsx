import { act, renderHook, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { usePostOnboardingMediaReadiness } from "../usePostOnboardingMediaReadiness"

const mocks = vi.hoisted(() => ({
  getConfig: vi.fn(),
  listMedia: vi.fn(),
  updateConfig: vi.fn()
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    getConfig: mocks.getConfig,
    listMedia: mocks.listMedia,
    updateConfig: mocks.updateConfig
  }
}))

const configuredSingleUser = {
  serverUrl: "http://127.0.0.1:8000",
  authMode: "single-user" as const,
  apiKey: "test-key"
}

const deferred = <T,>() => {
  let resolve!: (value: T) => void
  let reject!: (error: unknown) => void
  const promise = new Promise<T>((promiseResolve, promiseReject) => {
    resolve = promiseResolve
    reject = promiseReject
  })
  return { promise, resolve, reject }
}

describe("usePostOnboardingMediaReadiness", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("ignores stale readiness results when a newer check has completed", async () => {
    const staleMediaRequest = deferred<unknown[]>()
    mocks.getConfig
      .mockResolvedValueOnce(configuredSingleUser)
      .mockResolvedValueOnce(null)
    mocks.listMedia.mockReturnValueOnce(staleMediaRequest.promise)

    const { result } = renderHook(() => usePostOnboardingMediaReadiness(false))

    const firstCheck = result.current.retry()
    await waitFor(() => {
      expect(mocks.listMedia).toHaveBeenCalledTimes(1)
    })

    await act(async () => {
      await result.current.retry()
    })
    expect(result.current.status).toBe("needs_config")

    await act(async () => {
      staleMediaRequest.resolve([])
      await firstCheck
    })

    expect(result.current.status).toBe("needs_config")
    expect(result.current.config).toBeNull()
  })
})
