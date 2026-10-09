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

  it("does not reuse another server's cached provider status after an authority change", async () => {
    const { TldwApiClient } = await importClient()
    const client = new TldwApiClient()
    await client.getProvidersStatus()
    const current = { providers: [], any_configured: false }
    mocks.bgRequest.mockResolvedValueOnce(current)

    window.dispatchEvent(new CustomEvent("tldw:config-updated", {
      detail: { authorityChanged: true }
    }))

    expect(await new TldwApiClient().getProvidersStatus()).toEqual(current)
    expect(mocks.bgRequest).toHaveBeenCalledTimes(2)
  })

  it("does not join or publish a held response from the previous authority", async () => {
    let resolveOld!: (value: unknown) => void
    let resolveCurrent!: (value: unknown) => void
    mocks.bgRequest.mockImplementationOnce(() => new Promise(resolve => { resolveOld = resolve }))
    const { TldwApiClient } = await importClient()
    const client = new TldwApiClient()
    const old = client.getProvidersStatus().catch(error => error)
    window.dispatchEvent(new Event("tldw:auth-principal-changed"))
    mocks.bgRequest.mockImplementationOnce(() => new Promise(resolve => { resolveCurrent = resolve }))
    const current = client.getProvidersStatus()

    expect(mocks.bgRequest).toHaveBeenCalledTimes(2)
    resolveOld({ providers: [{ name: "previous" }], any_configured: true })
    expect(await old).toMatchObject({ status: 412 })
    const joined = client.getProvidersStatus()
    expect(mocks.bgRequest).toHaveBeenCalledTimes(2)
    const value = { providers: [{ name: "current" }], any_configured: true }
    resolveCurrent(value)
    expect(await Promise.all([current, joined])).toEqual([value, value])
    expect(await client.getProvidersStatus()).toEqual(value)
  })

  it("does not overwrite the current cache when an old authority settles last", async () => {
    let resolveOld!: (value: unknown) => void
    mocks.bgRequest.mockImplementationOnce(() => new Promise(resolve => { resolveOld = resolve }))
    const { TldwApiClient } = await importClient()
    const client = new TldwApiClient()
    const old = client.getProvidersStatus().catch(error => error)
    window.dispatchEvent(new Event("tldw:auth-credentials-changed"))
    const value = { providers: [{ name: "current" }], any_configured: true }
    mocks.bgRequest.mockResolvedValueOnce(value)
    const current = client.getProvidersStatus()
    expect(mocks.bgRequest).toHaveBeenCalledTimes(2)
    expect(await current).toEqual(value)
    resolveOld({ providers: [{ name: "previous" }], any_configured: true })
    expect(await old).toMatchObject({ status: 412 })
    expect(await client.getProvidersStatus()).toEqual(value)
    expect(mocks.bgRequest).toHaveBeenCalledTimes(2)
  })

  it("rejects an authority change between a cache read and delivery to its caller", async () => {
    const { TldwApiClient } = await importClient()
    const client = new TldwApiClient()
    await client.getProvidersStatus()
    const cached = client.getProvidersStatus()
    window.dispatchEvent(new Event("tldw:auth-principal-changed"))
    await expect(cached).rejects.toMatchObject({ status: 412 })
    expect(mocks.bgRequest).toHaveBeenCalledTimes(1)
  })
})
