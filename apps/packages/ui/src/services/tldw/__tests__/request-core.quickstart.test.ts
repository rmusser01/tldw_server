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

  it("uses cookie auth and csrf for same-origin mutations without stale headers", async () => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
    document.cookie = "csrf_token=csrf-123; Path=/"
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: { "Content-Type": "application/json" }
      })
    )

    const { tldwRequest } = await import("@/services/tldw/request-core")
    const result = await tldwRequest(
      {
        path: "/api/v1/notes",
        method: "POST",
        headers: {
          Authorization: "Bearer stale-token",
          "X-API-KEY": "stale-key",
          "x-csrf-token": "stale-csrf"
        },
        body: { title: "Cookie note" }
      },
      {
        getConfig: async () => ({
          serverUrl: "https://remote.example.test",
          authMode: "single-user",
          authSource: "cookie-session",
          apiKey: "stale-key"
        }),
        fetchFn: fetchMock
      }
    )

    expect(result.ok).toBe(true)
    const [url, init] = fetchMock.mock.calls[0]
    const requestHeaders = new Headers(init.headers)
    expect(url).toBe("/api/v1/notes")
    expect(init.credentials).toBe("same-origin")
    expect(requestHeaders.get("X-CSRF-Token")).toBe("csrf-123")
    expect(requestHeaders.get("X-API-KEY")).toBeNull()
    expect(requestHeaders.get("Authorization")).toBeNull()
  })

  it("uses cookie auth on safe methods without attaching csrf", async () => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
    document.cookie = "csrf_token=csrf-123; Path=/"
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: { "Content-Type": "application/json" }
      })
    )

    const { tldwRequest } = await import("@/services/tldw/request-core")
    await tldwRequest(
      {
        path: "/api/v1/users/me/profile",
        method: "GET",
        headers: { "X-CSRF-Token": "stale-csrf" }
      },
      {
        getConfig: async () => ({
          serverUrl: window.location.origin,
          authMode: "single-user",
          authSource: "cookie-session",
          apiKey: "stale-key"
        }),
        fetchFn: fetchMock
      }
    )

    const [, init] = fetchMock.mock.calls[0]
    const requestHeaders = new Headers(init.headers)
    expect(init.credentials).toBe("same-origin")
    expect(requestHeaders.get("X-CSRF-Token")).toBeNull()
    expect(requestHeaders.get("X-API-KEY")).toBeNull()
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

  it("keeps explicit remote auth when cookie source is not on a same-origin transport", async () => {
    delete process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: { "Content-Type": "application/json" }
      })
    )

    const { tldwRequest } = await import("@/services/tldw/request-core")
    await tldwRequest(
      { path: "/api/v1/health", method: "GET" },
      {
        getConfig: async () => ({
          serverUrl: "https://api.example.test",
          authMode: "single-user",
          authSource: "cookie-session",
          apiKey: "manual-remote-key"
        }),
        fetchFn: fetchMock
      }
    )

    const [url, init] = fetchMock.mock.calls[0]
    expect(url).toBe("https://api.example.test/api/v1/health")
    expect(init.credentials).toBeUndefined()
    expect(new Headers(init.headers).get("X-API-KEY")).toBe(
      "manual-remote-key"
    )
  })
})
