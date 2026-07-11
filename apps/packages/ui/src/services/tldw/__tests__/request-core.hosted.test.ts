import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

describe("tldwRequest hosted mode", () => {
  const originalDeploymentMode = process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE

  beforeEach(() => {
    vi.resetModules()
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "hosted"
  })

  afterEach(() => {
    if (originalDeploymentMode === undefined) {
      delete process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE
    } else {
      process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = originalDeploymentMode
    }
  })

  it("uses /api/proxy paths in hosted mode and omits browser Authorization headers", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ id: 1 }), {
        status: 200,
        headers: {
          "Content-Type": "application/json"
        }
      })
    )

    const { tldwRequest } = await import("@/services/tldw/request-core")
    const result = await tldwRequest(
      {
        path: "/api/v1/users/me/profile?sections=storage",
        method: "GET"
      },
      {
        getConfig: async () => ({
          authMode: "multi-user",
          orgId: 17
        }),
        fetchFn: fetchMock
      }
    )

    expect(result.ok).toBe(true)
    expect(fetchMock).toHaveBeenCalledTimes(1)
    const [url, init] = fetchMock.mock.calls[0]
    expect(url).toBe("/api/proxy/users/me/profile?sections=storage")
    expect((init.headers as Record<string, string>).Authorization).toBeUndefined()
    expect((init.headers as Record<string, string>)["X-TLDW-Org-Id"]).toBe("17")
  })

  it("ignores a stale cookie marker in hosted mode", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ id: 1 }), {
        status: 200,
        headers: { "Content-Type": "application/json" }
      })
    )

    const { tldwRequest } = await import("@/services/tldw/request-core")
    await tldwRequest(
      {
        path: "/api/v1/users/me/profile",
        method: "GET",
        headers: { "X-CSRF-Token": "hosted-csrf" }
      },
      {
        getConfig: async () => ({
          authMode: "single-user",
          authSource: "cookie-session",
          apiKey: "stale-key",
          orgId: 17
        }),
        fetchFn: fetchMock
      }
    )

    const [url, init] = fetchMock.mock.calls[0]
    const headers = new Headers(init.headers)
    expect(url).toBe("/api/proxy/users/me/profile")
    expect(init.credentials).toBeUndefined()
    expect(headers.get("X-CSRF-Token")).toBe("hosted-csrf")
    expect(headers.get("X-TLDW-Org-Id")).toBe("17")
  })
})
