import { resolveRecipeRequestSnapshot } from "@/services/tldw/recipe-request-snapshot"
import { afterEach, beforeEach, describe, expect, it } from "vitest"

describe("recipe request snapshot", () => {
  const originalDeploymentMode = process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE
  const originalWindow = globalThis.window

  const setWindowLocation = (origin: string, protocol = "https:") => {
    Object.defineProperty(globalThis, "window", {
      configurable: true,
      value: { location: { origin, protocol } }
    })
  }

  beforeEach(() => {
    delete process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE
    setWindowLocation("https://webui.example.test")
  })

  afterEach(() => {
    if (originalDeploymentMode === undefined) {
      delete process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE
    } else {
      process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = originalDeploymentMode
    }
    Object.defineProperty(globalThis, "window", {
      configurable: true,
      value: originalWindow
    })
  })

  it("binds a manual API key to the normalized advanced request snapshot", () => {
    const result = resolveRecipeRequestSnapshot({
      config: {
        serverUrl: "https://api.example.test/",
        authMode: "single-user",
        authSource: "manual",
        apiKey: "manual-key",
        orgId: 7
      },
      path: "/api/v1/prompt-studio/prompts/create",
      method: "POST"
    })

    expect(result.view).toMatchObject({
      ownerId: expect.stringMatching(/^recipe-owner:sha256:[0-9a-f]{64}$/),
      authorizationRevision: expect.stringMatching(
        /^recipe-authorization:sha256:[0-9a-f]{64}$/
      )
    })
    expect(result.snapshot).toMatchObject({
      url: "https://api.example.test/api/v1/prompt-studio/prompts/create",
      effectiveBase: "https://api.example.test/",
      headers: {
        "X-API-KEY": "manual-key",
        "X-TLDW-Org-Id": "7"
      }
    })
  })

  it("gives an eligible runtime key precedence over cookie and configured credentials", () => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
    const runtime = resolveRecipeRequestSnapshot({
      config: {
        serverUrl: window.location.origin,
        authMode: "single-user",
        authSource: "cookie-session",
        apiKey: "configured-key"
      },
      path: "/api/v1/prompts/",
      method: "POST",
      runtimeApiKey: "runtime-key",
      authenticatedPrincipalId: "cookie-user",
      cookieSessionRevision: "cookie-revision"
    })
    const manual = resolveRecipeRequestSnapshot({
      config: {
        serverUrl: window.location.origin,
        authMode: "single-user",
        authSource: "manual",
        apiKey: "runtime-key"
      },
      path: "/api/v1/prompts/",
      method: "POST"
    })

    expect(runtime.snapshot.headers).toMatchObject({
      "X-API-KEY": "runtime-key"
    })
    expect(runtime.snapshot.credentials).toBeUndefined()
    expect(runtime.view?.ownerId).not.toBe(manual.view?.ownerId)
  })

  it("uses an authoritative principal for manual bearer ownership", () => {
    const result = resolveRecipeRequestSnapshot({
      config: {
        serverUrl: "https://api.example.test",
        authMode: "multi-user",
        authSource: "manual",
        accessToken: "bearer-token",
        userId: "untrusted-hint"
      },
      path: "/api/v1/prompts/",
      method: "POST",
      authenticatedPrincipalId: "authoritative-user"
    })

    expect(result.view?.ownerId).toMatch(/^recipe-owner:sha256:[0-9a-f]{64}$/)
    expect(result.snapshot.headers.Authorization).toBe("Bearer bearer-token")
  })

  it("does not infer manual bearer ownership without an authoritative principal", () => {
    const result = resolveRecipeRequestSnapshot({
      config: {
        serverUrl: "https://api.example.test",
        authMode: "multi-user",
        authSource: "manual",
        accessToken: "h.eyJzdWIiOiJ1bnRydXN0ZWQifQ.s",
        userId: "also-untrusted"
      },
      path: "/api/v1/prompts/",
      method: "POST"
    })

    expect(result.view).toBeNull()
    expect(result.snapshot.headers.Authorization).toBe(
      "Bearer h.eyJzdWIiOiJ1bnRydXN0ZWQifQ.s"
    )
  })

  it("uses same-origin cookie transport only with an authoritative principal", () => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "quickstart"
    const owned = resolveRecipeRequestSnapshot({
      config: {
        serverUrl: window.location.origin,
        authMode: "single-user",
        authSource: "cookie-session"
      },
      path: "/api/v1/prompts/",
      method: "POST",
      authenticatedPrincipalId: "cookie-user",
      cookieSessionRevision: "session-a",
      csrfToken: "csrf-a"
    })
    const unowned = resolveRecipeRequestSnapshot({
      config: {
        serverUrl: window.location.origin,
        authMode: "single-user",
        authSource: "cookie-session"
      },
      path: "/api/v1/prompts/",
      method: "POST",
      cookieSessionRevision: "session-b",
      csrfToken: "csrf-b"
    })

    expect(owned.view?.ownerId).toMatch(/^recipe-owner:sha256:[0-9a-f]{64}$/)
    expect(owned.snapshot).toEqual({
      url: "/api/v1/prompts/",
      effectiveBase: "https://webui.example.test/",
      headers: { "X-CSRF-Token": "csrf-a" },
      credentials: "same-origin"
    })
    expect(unowned.view).toBeNull()
    expect(unowned.snapshot.headers).toEqual({ "X-CSRF-Token": "csrf-b" })
  })

  it("normalizes an advanced configured URL and preserves its active base path", () => {
    setWindowLocation("chrome-extension://extension-id", "chrome-extension:")
    const result = resolveRecipeRequestSnapshot({
      config: {
        serverUrl: "HTTPS://API.EXAMPLE.TEST:443/deployment-a///",
        authMode: "single-user",
        authSource: "manual",
        apiKey: "manual-key"
      },
      path: "/api/v1/prompts/",
      method: "POST"
    })

    expect(result.snapshot.url).toBe(
      "https://api.example.test/deployment-a/api/v1/prompts/"
    )
    expect(result.snapshot.effectiveBase).toBe(
      "https://api.example.test/deployment-a/"
    )
  })

  it("keeps active deployment base paths in separate owner domains", () => {
    setWindowLocation("chrome-extension://extension-id", "chrome-extension:")
    const owner = (serverUrl: string) =>
      resolveRecipeRequestSnapshot({
        config: {
          serverUrl,
          authMode: "single-user",
          authSource: "manual",
          apiKey: "same-key"
        },
        path: "/api/v1/prompts/",
        method: "POST"
      }).view?.ownerId

    expect(owner("https://api.example.test/deployment-a")).not.toBe(
      owner("https://api.example.test/deployment-b")
    )
  })

  it.each([
    ["absolute URL", { path: "https://other.example.test/api/v1/prompts/" }],
    ["no-auth", { path: "/api/v1/prompts/", noAuth: true }]
  ])("rejects %s ownership", (_label, request) => {
    const result = resolveRecipeRequestSnapshot({
      config: {
        serverUrl: "https://api.example.test",
        authMode: "single-user",
        authSource: "manual",
        apiKey: "manual-key"
      },
      method: "POST",
      ...request
    })

    expect(result.view).toBeNull()
  })

  it.each([
    { authMode: "unknown", authSource: "manual", apiKey: "key" },
    { authMode: "single-user", authSource: "unknown", apiKey: "key" },
    { authMode: "multi-user", authSource: "manual", accessToken: "token" },
    {
      authMode: "single-user",
      authSource: "cookie-session",
      apiKey: "CHANGE_ME_TO_SECURE_API_KEY"
    }
  ])(
    "does not invent an owner for an unknown auth combination %#",
    (config) => {
      const result = resolveRecipeRequestSnapshot({
        config: { serverUrl: "https://api.example.test", ...config },
        path: "/api/v1/prompts/",
        method: "POST"
      })

      expect(result.view).toBeNull()
    }
  )

  it("keeps URL, headers, and credentials out of the sanitized owner view", () => {
    const result = resolveRecipeRequestSnapshot({
      config: {
        serverUrl: "https://api.example.test",
        authMode: "single-user",
        authSource: "manual",
        apiKey: "secret-key"
      },
      path: "/api/v1/prompts/",
      method: "POST"
    })

    expect(result.view).toEqual({
      ownerId: expect.any(String),
      authorizationRevision: expect.any(String)
    })
    expect(JSON.stringify(result.view)).not.toContain("api.example.test")
    expect(JSON.stringify(result.view)).not.toContain("secret-key")
  })
})
