import { beforeEach, describe, expect, it, vi } from "vitest"

import { TldwAuthService } from "../tldw/TldwAuth"

const mocks = vi.hoisted(() => ({
  getConfig: vi.fn(),
  updateConfig: vi.fn(),
  bgRequest: vi.fn()
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
  emitSplashAfterLoginSuccess: vi.fn()
}))

vi.mock("@/services/tldw/deployment-mode", () => ({
  isHostedTldwDeployment: () => false
}))

describe("TldwAuthService refresh token rotation", () => {
  beforeEach(() => {
    mocks.getConfig.mockReset()
    mocks.updateConfig.mockReset()
    mocks.bgRequest.mockReset()

    mocks.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:8000",
      authMode: "multi-user",
      accessToken: "expired-access",
      refreshToken: "old-refresh"
    })
    mocks.updateConfig.mockResolvedValue(undefined)
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

    expect(mocks.updateConfig).toHaveBeenCalledWith({
      accessToken: "new-access",
      refreshToken: "new-refresh"
    })
  })

  it("preserves a newer stored refresh token when the backend does not rotate it", async () => {
    mocks.getConfig.mockReset()
    mocks.getConfig
      .mockResolvedValueOnce({
        serverUrl: "http://127.0.0.1:8000",
        authMode: "multi-user",
        accessToken: "expired-access",
        refreshToken: "old-refresh"
      })
      .mockResolvedValueOnce({
        serverUrl: "http://127.0.0.1:8000",
        authMode: "multi-user",
        accessToken: "expired-access",
        refreshToken: "newer-refresh"
      })
    mocks.bgRequest.mockResolvedValue({
      access_token: "new-access",
      token_type: "bearer",
      expires_in: 1800
    })

    const auth = new TldwAuthService()

    await auth.refreshToken()

    expect(mocks.updateConfig).toHaveBeenCalledWith({
      accessToken: "new-access",
      refreshToken: "newer-refresh"
    })
  })
})
