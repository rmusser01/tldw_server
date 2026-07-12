import { beforeEach, describe, expect, it, vi } from "vitest"

import { TldwAuthService } from "../tldw/TldwAuth"

const mocks = vi.hoisted(() => ({
  getConfig: vi.fn(),
  clearCookieSingleUserSession: vi.fn(),
  clearManualSingleUserCredentials: vi.fn(),
  bgRequest: vi.fn()
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    getConfig: mocks.getConfig,
    clearCookieSingleUserSession: mocks.clearCookieSingleUserSession,
    clearManualSingleUserCredentials: mocks.clearManualSingleUserCredentials
  }
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: mocks.bgRequest
}))

describe("TldwAuthService single-user logout", () => {
  beforeEach(() => {
    mocks.getConfig.mockReset()
    mocks.clearCookieSingleUserSession.mockReset()
    mocks.clearManualSingleUserCredentials.mockReset()
    mocks.bgRequest.mockReset()
    mocks.clearCookieSingleUserSession.mockResolvedValue(undefined)
    mocks.clearManualSingleUserCredentials.mockResolvedValue(undefined)
    mocks.bgRequest.mockResolvedValue({ authenticated: false })
  })

  it("logs out the cookie session before clearing its local marker", async () => {
    mocks.getConfig.mockResolvedValue({
      serverUrl: window.location.origin,
      authMode: "single-user",
      authSource: "cookie-session"
    })
    const auth = new TldwAuthService()

    await auth.logout()

    expect(mocks.bgRequest).toHaveBeenCalledWith({
      path: "/api/v1/auth/single-user/session",
      method: "DELETE"
    })
    expect(mocks.clearCookieSingleUserSession).toHaveBeenCalledOnce()
    expect(mocks.bgRequest.mock.invocationCallOrder[0]).toBeLessThan(
      mocks.clearCookieSingleUserSession.mock.invocationCallOrder[0]
    )
    expect(mocks.clearManualSingleUserCredentials).not.toHaveBeenCalled()
  })

  it("preserves cookie state and reports a server logout failure", async () => {
    mocks.getConfig.mockResolvedValue({
      serverUrl: window.location.origin,
      authMode: "single-user",
      authSource: "cookie-session"
    })
    mocks.bgRequest.mockRejectedValue(new Error("logout unavailable"))
    const auth = new TldwAuthService()

    await expect(auth.logout()).rejects.toThrow("logout unavailable")

    expect(mocks.clearCookieSingleUserSession).not.toHaveBeenCalled()
    expect(mocks.clearManualSingleUserCredentials).not.toHaveBeenCalled()
  })

  it("keeps manual single-user logout local", async () => {
    mocks.getConfig.mockResolvedValue({
      serverUrl: "https://api.example.test",
      authMode: "single-user",
      authSource: "manual",
      apiKey: "secret"
    })
    const auth = new TldwAuthService()

    await auth.logout()

    expect(mocks.clearManualSingleUserCredentials).toHaveBeenCalledOnce()
    expect(mocks.bgRequest).not.toHaveBeenCalled()
    expect(mocks.clearCookieSingleUserSession).not.toHaveBeenCalled()
  })
})
