import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  bgRequest: vi.fn()
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: (...args: unknown[]) =>
    (mocks.bgRequest as (...args: unknown[]) => unknown)(...args),
  bgStream: vi.fn(),
  bgUpload: vi.fn()
}))

const importClient = async () => {
  // Fresh module graph per test so the module-level cache starts empty.
  vi.resetModules()
  const mod = await import("../TldwApiClient")
  return mod
}

describe("getProvidersStatus TTL cache", () => {
  beforeEach(() => {
    mocks.bgRequest.mockReset()
    mocks.bgRequest.mockResolvedValue({
      providers: [{ name: "openai", configured: true, requires_api_key: true }],
      any_configured: true
    })
  })

  it("serves concurrent callers from one in-flight request", async () => {
    let resolveFirst!: (value: unknown) => void
    mocks.bgRequest.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          resolveFirst = resolve
        })
    )

    const { TldwApiClient } = await importClient()
    const client = new TldwApiClient()

    const first = client.getProvidersStatus()
    const second = client.getProvidersStatus()
    resolveFirst({
      providers: [{ name: "openai", configured: true, requires_api_key: true }],
      any_configured: true
    })

    const [a, b] = await Promise.all([first, second])
    expect(mocks.bgRequest).toHaveBeenCalledTimes(1)
    expect(a).toEqual(b)
    expect(a.any_configured).toBe(true)
  })

  it("reuses a cached response within the TTL window", async () => {
    const { TldwApiClient } = await importClient()
    const client = new TldwApiClient()

    await client.getProvidersStatus()
    await client.getProvidersStatus()
    await client.getProvidersStatus()

    expect(mocks.bgRequest).toHaveBeenCalledTimes(1)
  })

  it("does not serve a cached response to a fresh client after TTL expiry", async () => {
    vi.useFakeTimers()
    try {
      const { TldwApiClient } = await importClient()
      const client = new TldwApiClient()

      await client.getProvidersStatus()
      expect(mocks.bgRequest).toHaveBeenCalledTimes(1)

      vi.advanceTimersByTime(61_000)

      await client.getProvidersStatus()
      expect(mocks.bgRequest).toHaveBeenCalledTimes(2)
    } finally {
      vi.useRealTimers()
    }
  })

  it("does not cache failed responses", async () => {
    mocks.bgRequest.mockRejectedValueOnce(new Error("boom"))
    const { TldwApiClient } = await importClient()
    const client = new TldwApiClient()

    await expect(client.getProvidersStatus()).rejects.toThrow("boom")
    await client.getProvidersStatus()

    expect(mocks.bgRequest).toHaveBeenCalledTimes(2)
  })
})
