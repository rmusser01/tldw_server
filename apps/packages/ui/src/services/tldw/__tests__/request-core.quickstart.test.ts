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

  it("uses the managed WebUI origin for cookie-session requests", async () => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "managed"
    document.cookie = "csrf_token=csrf-123; Path=/"
    const fetchMock = vi.fn().mockResolvedValue(new Response("{}", { status: 200 }))

    const { tldwRequest } = await import("@/services/tldw/request-core")
    await tldwRequest(
      { path: "/api/v1/notes", method: "POST", body: { title: "Managed note" } },
      {
        getConfig: async () => ({
          serverUrl: window.location.origin,
          authMode: "single-user",
          authSource: "cookie-session"
        }),
        fetchFn: fetchMock
      }
    )

    const call = fetchMock.mock.calls[0]
    expect(call?.[0]).toBe("/api/v1/notes")
    const init = call[1]
    expect(init.credentials).toBe("same-origin")
    expect(new Headers(init.headers).get("X-CSRF-Token")).toBe("csrf-123")
  })

  it("reads the runtime-configured CSRF cookie for managed mutations", async () => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "managed"
    document.cookie = "tldw_csrf_a1=instance-token; Path=/"
    const fetchMock = vi.fn().mockResolvedValue(new Response("{}", { status: 200 }))
    const { setRuntimeCsrfCookieName } = await import("@/services/tldw/runtime-auth-override")
    setRuntimeCsrfCookieName("tldw_csrf_a1")

    const { tldwRequest } = await import("@/services/tldw/request-core")
    await tldwRequest(
      { path: "/api/v1/notes", method: "POST", body: { title: "Managed note" } },
      {
        getConfig: async () => ({
          serverUrl: window.location.origin,
          authMode: "single-user",
          authSource: "cookie-session"
        }),
        fetchFn: fetchMock
      }
    )

    expect(new Headers(fetchMock.mock.calls[0][1].headers).get("X-CSRF-Token")).toBe("instance-token")
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

  describe("managed noAuth setup mutations", () => {
    const setupPath = "/api/v1/setup/first-run/state"
    const pageOrigin = "https://webui.example.test"
    const instanceCookies =
      "tldw_csrf_a1=instance-a; tldw_csrf_b2=instance-b; csrf_token=legacy-token"

    it.each([
      {
        label: "relative POST",
        method: "POST",
        path: setupPath,
        cookieName: "tldw_csrf_a1",
        token: "instance-a"
      },
      {
        label: "relative PATCH",
        method: "PATCH",
        path: setupPath,
        cookieName: "tldw_csrf_a1",
        token: "instance-a"
      },
      {
        label: "relative PUT",
        method: "PUT",
        path: setupPath,
        cookieName: "tldw_csrf_a1",
        token: "instance-a"
      },
      {
        label: "relative DELETE",
        method: "DELETE",
        path: setupPath,
        cookieName: "tldw_csrf_a1",
        token: "instance-a"
      },
      {
        label: "configured page-origin absolute POST",
        method: "POST",
        path: `${pageOrigin}${setupPath}`,
        cookieName: "tldw_csrf_b2",
        token: "instance-b"
      },
      {
        label: "safe GET",
        method: "GET",
        path: setupPath,
        cookieName: "tldw_csrf_a1",
        token: null
      },
      {
        label: "safe HEAD",
        method: "HEAD",
        path: setupPath,
        cookieName: "tldw_csrf_a1",
        token: null
      },
      {
        label: "safe OPTIONS",
        method: "OPTIONS",
        path: setupPath,
        cookieName: "tldw_csrf_a1",
        token: null
      },
      {
        label: "safe TRACE",
        method: "TRACE",
        path: setupPath,
        cookieName: "tldw_csrf_a1",
        token: null
      },
      {
        label: "missing instance cookie",
        method: "POST",
        path: setupPath,
        cookieName: "tldw_csrf_missing",
        cookies: "",
        token: null
      },
      {
        label: "foreign and legacy cookies only",
        method: "POST",
        path: setupPath,
        cookieName: "tldw_csrf_missing",
        token: null
      }
    ])(
      "uses only the expected instance CSRF for $label",
      async ({ method, path, cookieName, cookies, token }) => {
        process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "managed"
        vi.spyOn(document, "cookie", "get").mockReturnValue(
          cookies ?? instanceCookies
        )
        const { setRuntimeCsrfCookieName } =
          await import("@/services/tldw/runtime-auth-override")
        setRuntimeCsrfCookieName(cookieName)
        const fetchMock = vi
          .fn<typeof fetch>()
          .mockResolvedValue(new Response("{}", { status: 200 }))
        const { tldwRequest } = await import("@/services/tldw/request-core")

        const result = await tldwRequest(
          {
            path,
            method,
            noAuth: true,
            headers: {
              aUtHoRiZaTiOn: "Bearer stale-token",
              "x-aPi-KeY": "stale-key",
              "x-CsRf-ToKeN": "stale-csrf"
            }
          },
          {
            getConfig: async () => ({
              serverUrl: pageOrigin,
              authMode: "single-user",
              authSource: "cookie-session",
              apiKey: "configured-key",
              accessToken: "configured-bearer"
            }),
            fetchFn: fetchMock
          }
        )

        expect(result.ok).toBe(true)
        const [url, init] = fetchMock.mock.calls[0]
        const requestHeaders = new Headers(init?.headers)
        expect(url).toBe(path)
        expect(init?.credentials).toBe("same-origin")
        expect(requestHeaders.get("X-CSRF-Token")).toBe(token)
        expect(requestHeaders.get("X-API-KEY")).toBeNull()
        expect(requestHeaders.get("Authorization")).toBeNull()
      }
    )

    it.each([
      {
        label: "allowlisted external origin",
        serverUrl: pageOrigin,
        path: `https://external.example.test${setupPath}`
      },
      {
        label: "configured origin different from the page",
        serverUrl: "https://api.example.test",
        path: `https://api.example.test${setupPath}`
      },
      {
        label: "page origin different from configured origin",
        serverUrl: "https://api.example.test",
        path: `${pageOrigin}${setupPath}`
      }
    ])("does not send page CSRF to $label", async ({ serverUrl, path }) => {
      process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "managed"
      vi.spyOn(document, "cookie", "get").mockReturnValue(instanceCookies)
      const { setRuntimeCsrfCookieName } =
        await import("@/services/tldw/runtime-auth-override")
      setRuntimeCsrfCookieName("tldw_csrf_a1")
      const fetchMock = vi
        .fn<typeof fetch>()
        .mockResolvedValue(new Response("{}", { status: 200 }))
      const { tldwRequest } = await import("@/services/tldw/request-core")

      const result = await tldwRequest(
        { path, method: "POST", noAuth: true },
        {
          getConfig: async () => ({
            serverUrl,
            authMode: "single-user",
            authSource: "cookie-session",
            apiKey: "configured-key",
            accessToken: "configured-bearer",
            absoluteUrlAllowlist: [pageOrigin, "https://external.example.test"]
          }),
          fetchFn: fetchMock
        }
      )

      expect(result.ok).toBe(true)
      const [url, init] = fetchMock.mock.calls[0]
      const requestHeaders = new Headers(init?.headers)
      expect(url).toBe(path)
      expect(init?.credentials).toBeUndefined()
      expect(requestHeaders.get("X-CSRF-Token")).toBeNull()
      expect(requestHeaders.get("X-API-KEY")).toBeNull()
      expect(requestHeaders.get("Authorization")).toBeNull()
    })
    it.each([
      {
        label: "manual WebUI",
        mode: "managed",
        protocol: "https:",
        authSource: "manual",
        url: setupPath
      },
      {
        label: "hosted WebUI",
        mode: "hosted",
        protocol: "https:",
        authSource: "cookie-session",
        url: "/api/proxy/setup/first-run/state"
      },
      {
        label: "extension",
        mode: "managed",
        protocol: "chrome-extension:",
        authSource: "cookie-session",
        url: `${pageOrigin}${setupPath}`
      }
    ])(
      "preserves credential-free noAuth for $label",
      async ({ mode, protocol, authSource, url }) => {
        process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = mode
        window.location.protocol = protocol
        vi.spyOn(document, "cookie", "get").mockReturnValue(instanceCookies)
        const { setRuntimeCsrfCookieName } =
          await import("@/services/tldw/runtime-auth-override")
        setRuntimeCsrfCookieName("tldw_csrf_a1")
        const fetchMock = vi
          .fn<typeof fetch>()
          .mockResolvedValue(new Response("{}", { status: 200 }))
        const { tldwRequest } = await import("@/services/tldw/request-core")

        const result = await tldwRequest(
          { path: setupPath, method: "POST", noAuth: true },
          {
            getConfig: async () => ({
              serverUrl: pageOrigin,
              authMode: "single-user",
              authSource,
              apiKey: "configured-key"
            }),
            fetchFn: fetchMock
          }
        )

        expect(result.ok).toBe(true)
        const [requestUrl, init] = fetchMock.mock.calls[0]
        const requestHeaders = new Headers(init?.headers)
        expect(requestUrl).toBe(url)
        expect(init?.credentials).toBeUndefined()
        expect(requestHeaders.get("X-CSRF-Token")).toBeNull()
        expect(requestHeaders.get("X-API-KEY")).toBeNull()
        expect(requestHeaders.get("Authorization")).toBeNull()
      }
    )
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
