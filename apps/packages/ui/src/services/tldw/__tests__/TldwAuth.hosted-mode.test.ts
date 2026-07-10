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

describe("TldwAuthService hosted mode", () => {
  const originalDeploymentMode = process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE

  beforeEach(() => {
    vi.resetModules()
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "hosted"
    mocks.bgRequest.mockReset()
    mocks.emitSplashAfterLoginSuccess.mockReset()
    mocks.getConfig.mockReset()
    mocks.updateConfig.mockReset()
    mocks.getCurrentUserProfile.mockReset()
    mocks.getConfig.mockResolvedValue(null)
    mocks.updateConfig.mockResolvedValue(undefined)
    mocks.getCurrentUserProfile.mockResolvedValue({ active_org_id: 23 })
    sessionStorage.clear()
  })

  afterEach(() => {
    if (originalDeploymentMode === undefined) {
      delete process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE
    } else {
      process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = originalDeploymentMode
    }
  })

  it("does not create a personal org from the browser in hosted mode", async () => {
    mocks.bgRequest.mockImplementation(async (payload: Record<string, unknown>) => {
      if (payload.path === "/api/auth/login") {
        return {
          token_type: "bearer",
          expires_in: 1800
        }
      }
      if (payload.path === "/api/v1/orgs" && payload.method === "GET") {
        return {
          items: [{ id: 23 }]
        }
      }
      throw new Error(`Unexpected request: ${String(payload.method)} ${String(payload.path)}`)
    })

    const { TldwAuthService } = await import("@/services/tldw/TldwAuth")
    const auth = new TldwAuthService()
    const result = await auth.login({ username: "user", password: "pass" })

    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/auth/login",
        method: "POST",
        noAuth: true
      })
    )
    expect(mocks.bgRequest).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/orgs",
        method: "GET"
      })
    )
    expect(mocks.bgRequest).not.toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/orgs",
        method: "POST"
      })
    )
    expect(mocks.updateConfig).toHaveBeenCalledWith(
      expect.objectContaining({
        authMode: "multi-user",
        accessToken: undefined,
        refreshToken: undefined
      })
    )
    expect(mocks.updateConfig).toHaveBeenCalledWith(
      expect.objectContaining({
        orgId: 23
      })
    )
    expect(result.access_token).toBeUndefined()
    expect(mocks.emitSplashAfterLoginSuccess).toHaveBeenCalledTimes(1)
  })

  it("clears source-review handoffs when logging out", async () => {
    mocks.getConfig.mockResolvedValue({ authMode: "multi-user" })
    mocks.bgRequest.mockResolvedValue(undefined)
    const { TldwAuthService } = await import("@/services/tldw/TldwAuth")
    const {
      loadSourceReviewHandoff,
      saveSourceReviewHandoff
    } = await import("@/services/tldw/source-review-handoff")
    const handoffToken = saveSourceReviewHandoff({
      occurrence_id: 31,
      plan_id: 7,
      activity_type: "quiz",
      source_bundle: {
        items: [
          {
            source_type: "note",
            source_id: "note-42",
            excerpt_text: "private source excerpt"
          }
        ]
      }
    })
    sessionStorage.setItem("unrelated", "keep")

    await new TldwAuthService().logout()

    expect(loadSourceReviewHandoff(handoffToken, false)).toBeNull()
    expect(sessionStorage.getItem("unrelated")).toBe("keep")
  })
})
