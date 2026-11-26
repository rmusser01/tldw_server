import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  initialize: vi.fn(),
  bgRequest: vi.fn(),
  emitSplashAfterLoginSuccess: vi.fn(),
  getConfig: vi.fn(),
  updateConfig: vi.fn(),
  commitTokenRefresh: vi.fn(),
  invalidateRefreshSession: vi.fn(),
  getCurrentUserProfile: vi.fn()
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: (...args: unknown[]) => mocks.bgRequest(...args)
}))

vi.mock("@/services/splash-events", () => ({
  emitSplashAfterLoginSuccess: (...args: unknown[]) =>
    mocks.emitSplashAfterLoginSuccess(...args)
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: (...args: unknown[]) => mocks.initialize(...args),
    getConfig: (...args: unknown[]) => mocks.getConfig(...args),
    updateConfig: (...args: unknown[]) => mocks.updateConfig(...args),
    commitTokenRefresh: (...args: unknown[]) =>
      mocks.commitTokenRefresh(...args),
    invalidateRefreshSession: (...args: unknown[]) =>
      mocks.invalidateRefreshSession(...args),
    getCurrentUserProfile: (...args: unknown[]) =>
      mocks.getCurrentUserProfile(...args)
  }
}))

describe("TldwAuthService session identity", () => {
  beforeEach(() => {
    vi.resetModules()
    localStorage.clear()
    delete process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE
    for (const fn of Object.values(mocks)) fn.mockReset()
    mocks.updateConfig.mockResolvedValue(undefined)
    mocks.commitTokenRefresh.mockResolvedValue(true)
    mocks.invalidateRefreshSession.mockResolvedValue(true)
    mocks.initialize.mockResolvedValue(undefined)
    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://localhost:8000",
      authMode: "multi-user",
      refreshToken: "refresh-token"
    })
    mocks.bgRequest.mockResolvedValue({
      access_token: "a",
      refresh_token: "r",
      token_type: "bearer"
    })
  })

  const makeAuth = async () => {
    const { TldwAuthService } = await import("@/services/tldw/TldwAuth")
    return new TldwAuthService()
  }

  it("sends the install id on password login", async () => {
    const auth = await makeAuth()
    await auth.login({ username: "u", password: "p" })
    expect(mocks.bgRequest.mock.calls[0][0].headers["X-TLDW-Device-Id"]).toBe(
      localStorage.getItem("tldw:install-id")
    )
    expect(localStorage.getItem("tldw:install-id")).toBeTruthy()
  })

  it("sends the install id on magic-link verify", async () => {
    const auth = await makeAuth()
    await auth.verifyMagicLink("t")
    expect(mocks.bgRequest.mock.calls[0][0].headers["X-TLDW-Device-Id"]).toBe(
      localStorage.getItem("tldw:install-id")
    )
    expect(localStorage.getItem("tldw:install-id")).toBeTruthy()
  })

  it("clears the replaced-session notice after a successful login", async () => {
    localStorage.setItem("tldw:session-replaced", "1")
    const auth = await makeAuth()
    await auth.login({ username: "u", password: "p" })
    const { hasSessionReplacedNotice } = await import(
      "@/services/tldw/session-identity"
    )
    expect(hasSessionReplacedNotice()).toBe(false)
  })

  it("records session_replaced when refresh is refused with it", async () => {
    mocks.bgRequest.mockRejectedValue({
      status: 401,
      details: { detail: { code: "session_replaced" } }
    })
    const auth = await makeAuth()
    await expect(auth.refreshToken()).rejects.toBeTruthy()
    const { hasSessionReplacedNotice } = await import(
      "@/services/tldw/session-identity"
    )
    expect(hasSessionReplacedNotice()).toBe(true)
  })

  it("does not record session_replaced when the invalidation did not apply", async () => {
    mocks.invalidateRefreshSession.mockResolvedValue(false)
    mocks.bgRequest.mockRejectedValue({
      status: 401,
      details: { detail: { code: "session_replaced" } }
    })
    const auth = await makeAuth()
    await expect(auth.refreshToken()).rejects.toBeTruthy()
    const { hasSessionReplacedNotice } = await import(
      "@/services/tldw/session-identity"
    )
    expect(hasSessionReplacedNotice()).toBe(false)
  })

  it("clears the replaced-session notice on logout", async () => {
    localStorage.setItem("tldw:session-replaced", "1")
    const auth = await makeAuth()
    await auth.logout()
    const { hasSessionReplacedNotice } = await import(
      "@/services/tldw/session-identity"
    )
    expect(hasSessionReplacedNotice()).toBe(false)
  })

  it("clears a notice set while the remote logout was in flight", async () => {
    const { hasSessionReplacedNotice } = await import(
      "@/services/tldw/session-identity"
    )
    mocks.bgRequest.mockImplementation(async () => {
      localStorage.setItem("tldw:session-replaced", "1")
      throw { status: 401 }
    })
    const auth = await makeAuth()
    await auth.logout()
    expect(hasSessionReplacedNotice()).toBe(false)
  })
})
