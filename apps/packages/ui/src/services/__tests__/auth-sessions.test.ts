import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  bgRequest: vi.fn(),
  runtimeId: null as string | null,
  storageGet: vi.fn(async (_key?: string) => null as unknown)
}))

vi.mock("wxt/browser", () => ({
  browser: { runtime: { get id() { return mocks.runtimeId } } }
}))
vi.mock("@/utils/safe-storage", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/utils/safe-storage")>()),
  createSafeStorage: () => ({
    get: (...a: unknown[]) => (mocks.storageGet as (...a: unknown[]) => unknown)(...a),
    set: async () => undefined,
    remove: async () => undefined
  })
}))

describe("auth-sessions client", () => {
  beforeEach(() => {
    vi.resetModules()
    vi.unstubAllGlobals()
    mocks.bgRequest.mockReset()
  })

  const withMockedBg = async () => {
    vi.doMock("@/services/background-proxy", () => ({ bgRequest: mocks.bgRequest }))
    return import("@/services/auth-sessions")
  }

  it("lists sessions", async () => {
    mocks.bgRequest.mockResolvedValue([{ id: 1 }])
    const { listAuthSessions } = await withMockedBg()
    await expect(listAuthSessions()).resolves.toEqual([{ id: 1 }])
    expect(mocks.bgRequest).toHaveBeenCalledWith({ path: "/api/v1/auth/sessions", method: "GET" })
  })

  it("revokes one session", async () => {
    mocks.bgRequest.mockResolvedValue({ message: "ok" })
    const { revokeAuthSession } = await withMockedBg()
    await expect(revokeAuthSession(12)).resolves.toBeUndefined()
    expect(mocks.bgRequest).toHaveBeenCalledWith({ path: "/api/v1/auth/sessions/12", method: "DELETE" })
  })

  it("revokes the other sessions and reads the count", async () => {
    mocks.bgRequest.mockResolvedValue({ message: "ok", details: { sessions_revoked: 3 } })
    const { revokeOtherAuthSessions } = await withMockedBg()
    await expect(revokeOtherAuthSessions()).resolves.toEqual({ sessions_revoked: 3 })
    expect(mocks.bgRequest).toHaveBeenCalledWith({
      path: "/api/v1/auth/sessions/revoke-all?keep_current=true",
      method: "POST"
    })
  })

  it("reads the error code from all three error shapes", async () => {
    const { errorCode, isInteractiveRequiredError } = await withMockedBg()
    expect(errorCode({ errorCode: "a" })).toBe("a")
    expect(errorCode({ details: { detail: { code: "b" } } })).toBe("b")
    expect(errorCode({ detail: { code: "c" } })).toBe("c")
    expect(errorCode(new Error("x"))).toBeUndefined()
    expect(isInteractiveRequiredError({ detail: { code: "interactive_session_required" } })).toBe(true)
  })

  it("recognises a refusal thrown by the real bgRequest", async () => {
    mocks.runtimeId = null
    mocks.storageGet.mockImplementation(async (key?: string) =>
      key === "tldwConfig"
        ? {
            serverUrl: "https://api.example.com",
            authMode: "single-user",
            authSource: "manual",
            credentialSource: "manual",
            apiKeyPersistence: "device",
            apiKeyServerOrigin: "https://api.example.com",
            apiKey: "synthetic-test-key"
          }
        : null
    )
    vi.stubGlobal(
      "fetch",
      vi.fn(
        async () =>
          new Response(
            JSON.stringify({ detail: { code: "interactive_session_required", message: "x" } }),
            { status: 403, headers: { "content-type": "application/json" } }
          )
      )
    )
    vi.spyOn(console, "warn").mockImplementation(() => undefined)
    vi.doUnmock("@/services/background-proxy")
    const { revokeAuthSession, isInteractiveRequiredError } = await import("@/services/auth-sessions")
    const caught = await revokeAuthSession(1).catch((e) => e)
    expect(caught).toMatchObject({ status: 403, details: { detail: { code: "interactive_session_required" } } })
    expect(isInteractiveRequiredError(caught)).toBe(true)
  })
})
