import { afterEach, describe, expect, it, vi } from "vitest"

describe("managed direct browser configuration", () => {
  afterEach(() => {
    vi.unstubAllEnvs()
  })

  it("selects the cookie session bound to the page origin", async () => {
    vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "managed")
    const { COOKIE_SESSION_CONFIG_KEY } = await import("@/services/tldw/browser-networking")
    const { resolveDirectBrowserConfig } = await import("@/services/tldw/direct-browser-config")
    const cookieSession = {
      authMode: "single-user" as const,
      authSource: "cookie-session" as const,
      serverUrl: window.location.origin
    }
    const storage = {
      get: vi.fn(async (key: string) => key === COOKIE_SESSION_CONFIG_KEY ? cookieSession : null),
      set: vi.fn(async () => undefined),
      remove: vi.fn(async () => undefined)
    }

    expect(await resolveDirectBrowserConfig(storage)).toEqual(cookieSession)
  })
})
