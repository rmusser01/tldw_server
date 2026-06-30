import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { TldwAuthService } from "../tldw/TldwAuth"

const mocks = vi.hoisted(() => ({
  getConfig: vi.fn(),
  updateConfig: vi.fn(),
  bgRequest: vi.fn(),
  fetch: vi.fn(),
  emitSplashAfterLoginSuccess: vi.fn()
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    getConfig: mocks.getConfig,
    updateConfig: mocks.updateConfig
  }
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: mocks.bgRequest
}))

vi.mock("@/services/splash-events", () => ({
  emitSplashAfterLoginSuccess: mocks.emitSplashAfterLoginSuccess
}))

describe("TldwAuthService.testApiKey", () => {
  beforeEach(() => {
    mocks.getConfig.mockReset()
    mocks.updateConfig.mockReset()
    mocks.bgRequest.mockReset()
    mocks.fetch.mockReset()
    mocks.emitSplashAfterLoginSuccess.mockReset()
    vi.stubGlobal("fetch", mocks.fetch)
    vi.spyOn(console, "error").mockImplementation(() => undefined)
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("uses the candidate setup server URL for API key validation", async () => {
    const auth = new TldwAuthService()

    mocks.fetch.mockImplementation(async (url, init) => {
      expect(url).toBe("https://example.com/api/v1/users/me/profile")
      expect(init).toMatchObject({
        method: "GET",
        headers: { "X-API-KEY": "real-api-key" }
      })
      return new Response(JSON.stringify({ id: 1 }), {
        status: 200,
        headers: { "Content-Type": "application/json" }
      })
    })

    const ok = await auth.testApiKey("https://example.com", "real-api-key")

    expect(ok).toBe(true)
    expect(mocks.fetch).toHaveBeenCalledTimes(1)
  })

  it("returns false for invalid API key responses", async () => {
    const auth = new TldwAuthService()

    mocks.fetch.mockResolvedValueOnce(
      new Response("Unauthorized", {
        status: 401,
        statusText: "Unauthorized"
      })
    )

    const ok = await auth.testApiKey("https://example.com", "bad-api-key")

    expect(ok).toBe(false)
  })

  it("throws a connection-style error when the request is aborted", async () => {
    const auth = new TldwAuthService()
    const aborted = Object.assign(new Error("The operation was aborted."), {
      status: 0,
      name: "AbortError"
    })

    mocks.fetch.mockRejectedValueOnce(aborted)

    await expect(
      auth.testApiKey("https://example.com", "real-api-key")
    ).rejects.toThrow(/timed out|aborted/i)
  })
})
