import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  bgRequest: vi.fn(),
  emitSplashAfterLoginSuccess: vi.fn(),
  getConfig: vi.fn(),
  updateConfig: vi.fn(),
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
    getConfig: (...args: unknown[]) => mocks.getConfig(...args),
    updateConfig: (...args: unknown[]) => mocks.updateConfig(...args),
    getCurrentUserProfile: (...args: unknown[]) =>
      mocks.getCurrentUserProfile(...args)
  }
}))

const originalDeploymentMode = process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE

describe("TldwAuthService token refresh single-flight", () => {
  beforeEach(() => {
    vi.resetModules()
    delete process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE
    mocks.bgRequest.mockReset()
    mocks.getConfig.mockReset()
    mocks.updateConfig.mockReset()
    mocks.updateConfig.mockResolvedValue(undefined)
    mocks.getConfig.mockResolvedValue({
      authMode: "multi-user",
      refreshToken: "refresh-token"
    })
  })

  afterEach(() => {
    vi.useRealTimers()
    if (originalDeploymentMode === undefined) {
      delete process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE
    } else {
      process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = originalDeploymentMode
    }
  })

  it("coalesces concurrent refreshToken() calls into one refresh request", async () => {
    mocks.bgRequest.mockResolvedValue({
      access_token: "fresh-access",
      refresh_token: "rotated-refresh",
      token_type: "bearer"
    })

    const { TldwAuthService } = await import("@/services/tldw/TldwAuth")
    const auth = new TldwAuthService()

    const [a, b] = await Promise.all([auth.refreshToken(), auth.refreshToken()])

    expect(a).toBe(b)
    const refreshCalls = mocks.bgRequest.mock.calls.filter(
      (call) => (call[0] as { path?: string })?.path === "/api/v1/auth/refresh"
    )
    expect(refreshCalls).toHaveLength(1)
  })

  it("allows a fresh refresh once the previous one settles", async () => {
    mocks.bgRequest.mockResolvedValue({
      access_token: "fresh-access",
      token_type: "bearer"
    })

    const { TldwAuthService } = await import("@/services/tldw/TldwAuth")
    const auth = new TldwAuthService()

    await auth.refreshToken()
    await auth.refreshToken()

    const refreshCalls = mocks.bgRequest.mock.calls.filter(
      (call) => (call[0] as { path?: string })?.path === "/api/v1/auth/refresh"
    )
    expect(refreshCalls).toHaveLength(2)
  })

  it("initTokenRefresh arms the refresh timer when a valid refresh token is present", async () => {
    vi.useFakeTimers()
    mocks.bgRequest.mockResolvedValue({
      access_token: "fresh-access",
      refresh_token: "rotated-refresh",
      token_type: "bearer",
      expires_in: 1800
    })

    const { TldwAuthService } = await import("@/services/tldw/TldwAuth")
    const auth = new TldwAuthService()

    await auth.initTokenRefresh()
    // A second init is a no-op because a timer is already armed.
    await auth.initTokenRefresh()

    const refreshCalls = mocks.bgRequest.mock.calls.filter(
      (call) => (call[0] as { path?: string })?.path === "/api/v1/auth/refresh"
    )
    expect(refreshCalls).toHaveLength(1)
  })

  it("initTokenRefresh is a no-op without a refresh token", async () => {
    mocks.getConfig.mockResolvedValue({ authMode: "multi-user" })

    const { TldwAuthService } = await import("@/services/tldw/TldwAuth")
    const auth = new TldwAuthService()

    await auth.initTokenRefresh()

    expect(mocks.bgRequest).not.toHaveBeenCalled()
  })
})
