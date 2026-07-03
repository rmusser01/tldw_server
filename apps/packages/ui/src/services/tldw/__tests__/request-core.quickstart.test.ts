import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

describe("tldwRequest quickstart and advanced transport", () => {
  const originalDeploymentMode = process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE
  const originalWindow = globalThis.window

  beforeEach(() => {
    vi.resetModules()
    Object.defineProperty(globalThis, "window", {
      value: {
        location: {
          origin: "https://webui.example.test",
          protocol: "https:"
        }
      },
      configurable: true
    })
  })

  afterEach(() => {
    if (originalDeploymentMode === undefined) {
      delete process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE
    } else {
      process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = originalDeploymentMode
    }

    Object.defineProperty(globalThis, "window", {
      value: originalWindow,
      configurable: true
    })
  })

  it("uses same-origin quickstart requests with self-host auth headers", async () => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"

    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ items: [] }), {
        status: 200,
        headers: {
          "Content-Type": "application/json"
        }
      })
    )

    const { tldwRequest } = await import("@/services/tldw/request-core")
    const result = await tldwRequest(
      {
        path: "/api/v1/notifications?limit=50",
        method: "GET"
      },
      {
        getConfig: async () => ({
          serverUrl: "http://127.0.0.1:8080",
          authMode: "single-user",
          apiKey: "test-key"
        }),
        fetchFn: fetchMock
      }
    )

    expect(result.ok).toBe(true)
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/v1/notifications?limit=50",
      expect.objectContaining({
        headers: expect.objectContaining({ "X-API-KEY": "test-key" })
      })
    )
  })

  it("uses the configured absolute origin in advanced mode and keeps self-host auth headers", async () => {
    delete process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE

    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: {
          "Content-Type": "application/json"
        }
      })
    )

    const { tldwRequest } = await import("@/services/tldw/request-core")
    const result = await tldwRequest(
      {
        path: "/api/v1/notifications?limit=10",
        method: "GET"
      },
      {
        getConfig: async () => ({
          serverUrl: "https://api.example.test:9443",
          authMode: "single-user",
          apiKey: "test-key"
        }),
        fetchFn: fetchMock
      }
    )

    expect(result.ok).toBe(true)
    expect(fetchMock).toHaveBeenCalledWith(
      "https://api.example.test:9443/api/v1/notifications?limit=10",
      expect.objectContaining({
        headers: expect.objectContaining({ "X-API-KEY": "test-key" })
      })
    )
  })

  it("rejects placeholder runtime single-user API keys before sending a request", async () => {
    delete process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE

    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: {
          "Content-Type": "application/json"
        }
      })
    )

    const runtimeAuth = await import("@/services/tldw/runtime-auth-override")
    runtimeAuth.setRuntimeSingleUserApiKeyOverride("CHANGE_ME_TO_SECURE_API_KEY")
    expect(runtimeAuth.getRuntimeSingleUserApiKeyOverride()).toBeNull()
    try {
      const { tldwRequest } = await import("@/services/tldw/request-core")
      const result = await tldwRequest(
        {
          path: "/api/v1/health",
          method: "GET"
        },
        {
          getConfig: async () => ({
            serverUrl: "https://api.example.test:9443",
            authMode: "single-user"
          }),
          fetchFn: fetchMock
        }
      )

      expect(result.ok).toBe(false)
      expect(result.status).toBe(401)
      expect(fetchMock).not.toHaveBeenCalled()
    } finally {
      runtimeAuth.clearRuntimeAuthOverride()
    }
  })
})
