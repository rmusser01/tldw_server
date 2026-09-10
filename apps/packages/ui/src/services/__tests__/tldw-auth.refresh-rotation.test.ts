import { beforeEach, describe, expect, it, vi } from "vitest"

import { TldwAuthService } from "../tldw/TldwAuth"

const jwtForUser = (userId: string | number): string =>
  `header.${btoa(JSON.stringify({ sub: String(userId) }))}.signature`

const mocks = vi.hoisted(() => ({
  initialize: vi.fn(),
  getConfig: vi.fn(),
  updateConfig: vi.fn(),
  commitTokenRefresh: vi.fn(),
  clearManualSingleUserCredentials: vi.fn(),
  bgRequest: vi.fn()
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: mocks.initialize,
    getConfig: mocks.getConfig,
    updateConfig: mocks.updateConfig,
    commitTokenRefresh: mocks.commitTokenRefresh,
    clearManualSingleUserCredentials: mocks.clearManualSingleUserCredentials
  }
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: mocks.bgRequest
}))

vi.mock("@/services/splash-events", () => ({
  emitSplashAfterLoginSuccess: vi.fn()
}))

vi.mock("@/services/tldw/deployment-mode", () => ({
  isHostedTldwDeployment: () => false
}))

describe("TldwAuthService refresh token rotation", () => {
  beforeEach(() => {
    mocks.initialize.mockReset()
    mocks.getConfig.mockReset()
    mocks.updateConfig.mockReset()
    mocks.commitTokenRefresh.mockReset()
    mocks.clearManualSingleUserCredentials.mockReset()
    mocks.bgRequest.mockReset()
    mocks.initialize.mockResolvedValue(undefined)

    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "multi-user",
      accessToken: "expired-access",
      refreshToken: "old-refresh"
    })
    mocks.updateConfig.mockResolvedValue(undefined)
    mocks.commitTokenRefresh.mockResolvedValue(true)
    mocks.clearManualSingleUserCredentials.mockResolvedValue(undefined)
    mocks.bgRequest.mockResolvedValue({
      access_token: "new-access",
      refresh_token: "new-refresh",
      token_type: "bearer",
      expires_in: 1800
    })
  })

  it("persists rotated refresh token during token refresh", async () => {
    const auth = new TldwAuthService()

    await auth.refreshToken()

    expect(mocks.commitTokenRefresh).toHaveBeenCalledWith(
      expect.objectContaining({ refreshToken: "old-refresh" }),
      "old-refresh",
      { accessToken: "new-access", refreshToken: "new-refresh" }
    )
  })

  it("binds refresh dispatch to the captured target, principal, and refresh lineage", async () => {
    mocks.getConfig.mockResolvedValue({
      serverUrl: "https://api.example.test",
      authMode: "multi-user",
      authSource: "manual",
      orgId: 7,
      accessToken: jwtForUser(42),
      refreshToken: "old-refresh"
    })
    const auth = new TldwAuthService()

    await auth.refreshToken()

    expect(mocks.bgRequest).toHaveBeenCalledWith(expect.objectContaining({
      path: "/api/v1/auth/refresh",
      method: "POST",
      servicePromptConfig: {
        serverUrl: "https://api.example.test",
        authMode: "multi-user",
        authSource: "manual",
        orgId: 7,
        expectedUserId: "42",
        expectedRefreshToken: "old-refresh"
      }
    }))
  })

  it("fails closed when the account changes before refreshed tokens commit", async () => {
    mocks.commitTokenRefresh.mockResolvedValue(false)
    mocks.bgRequest.mockResolvedValue({
      access_token: "new-access",
      token_type: "bearer",
      expires_in: 1800
    })

    const auth = new TldwAuthService()

    await expect(auth.refreshToken()).rejects.toMatchObject({
      status: 412,
      details: { detail: { code: "request_config_scope_changed" } }
    })
    expect(mocks.updateConfig).not.toHaveBeenCalled()
  })

  it("clears manual single-user credentials on logout", async () => {
    mocks.getConfig.mockResolvedValue({
      serverUrl: "https://api.example.test",
      authMode: "single-user",
      apiKey: "secret",
      credentialSource: "manual",
      apiKeyPersistence: "device",
      apiKeyServerOrigin: "https://api.example.test"
    })
    const auth = new TldwAuthService()

    await auth.logout()

    expect(mocks.clearManualSingleUserCredentials).toHaveBeenCalledOnce()
    expect(mocks.bgRequest).not.toHaveBeenCalled()
  })
})
