import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { TldwAuthService } from "../tldw/TldwAuth"

const mocks = vi.hoisted(() => ({
  bgRequest: vi.fn(),
  clearCookieSingleUserSession: vi.fn(),
  clearManualSingleUserCredentials: vi.fn(),
  getConfig: vi.fn(),
  updateConfig: vi.fn()
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    getConfig: mocks.getConfig,
    clearCookieSingleUserSession: mocks.clearCookieSingleUserSession,
    clearManualSingleUserCredentials: mocks.clearManualSingleUserCredentials,
    updateConfig: mocks.updateConfig
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

  it("treats an active cookie session as authenticated without a readable key", async () => {
    mocks.getConfig.mockResolvedValue({
      serverUrl: window.location.origin,
      authMode: "single-user",
      authSource: "cookie-session"
    })
    const auth = new TldwAuthService()

    await expect(auth.isAuthenticated()).resolves.toBe(true)
  })
})

vi.mock("@/services/splash-events", () => ({
  emitSplashAfterLoginSuccess: vi.fn()
}))

vi.mock("@/services/tldw/deployment-mode", () => ({
  isHostedTldwDeployment: () => false
}))

describe("TldwAuthService logout", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    sessionStorage.clear()
    mocks.getConfig.mockResolvedValue({
      serverUrl: "https://tldw.example",
      authMode: "multi-user",
      accessToken: "access-token",
      refreshToken: "refresh-token"
    })
    mocks.bgRequest.mockResolvedValue(undefined)
    mocks.updateConfig.mockResolvedValue(undefined)
  })

  afterEach(() => {
    sessionStorage.clear()
    vi.restoreAllMocks()
  })

  it("clears Task 14 records after tokens and before the logout boundary without reading values", async () => {
    const draftKey = "tldw:presentation-studio:html:draft:v1:https%3A%2F%2Ftldw.example:42"
    const resumeKey = "tldw:presentation-studio:html:resume:v1:https%3A%2F%2Ftldw.example:42"
    sessionStorage.setItem(draftKey, "PRIVATE DIRECT MATERIAL")
    sessionStorage.setItem(resumeKey, '{"idempotencyKey":"PRIVATE-KEY"}')
    sessionStorage.setItem("unrelated:session:key", "keep")
    const getSpy = vi.spyOn(Object.getPrototypeOf(window.sessionStorage) as Storage, "getItem")
    let tokensCleared = false
    mocks.updateConfig.mockImplementation(async () => {
      tokensCleared = true
    })
    const boundarySnapshots: Array<{ tokensCleared: boolean; keys: Array<string | null> }> = []
    const onLogout = () => {
      boundarySnapshots.push({
        tokensCleared,
        keys: Array.from(
          { length: sessionStorage.length },
          (_, index) => sessionStorage.key(index)
        )
      })
    }
    window.addEventListener("tldw:auth-principal-changed", onLogout)

    const auth = new TldwAuthService()
    await auth.logout()

    window.removeEventListener("tldw:auth-principal-changed", onLogout)
    expect(mocks.updateConfig).toHaveBeenCalledWith({
      accessToken: undefined,
      refreshToken: undefined
    })
    expect(boundarySnapshots).toEqual([{
      tokensCleared: true,
      keys: ["unrelated:session:key"]
    }])
    expect(getSpy).not.toHaveBeenCalled()
    getSpy.mockRestore()
    expect(sessionStorage.getItem("unrelated:session:key")).toBe("keep")
  })
})
