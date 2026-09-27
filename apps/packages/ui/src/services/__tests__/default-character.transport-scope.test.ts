import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const boundary = vi.hoisted(() => ({ get: vi.fn(), fetch: vi.fn() }))
vi.mock("wxt/browser", () => ({ browser: { runtime: { id: null } } }))
vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: () => ({ get: boundary.get, set: vi.fn(), remove: vi.fn() }),
  safeStorageSerde: { serialize: (value: unknown) => value, deserialize: (value: unknown) => value }
}))
vi.mock("@/services/tldw/runtime-auth-override", () => ({
  getRuntimeSingleUserApiKeyOverride: () => null,
  isCookieSessionConfigInvalidated: () => false
}))

import { TldwApiClient } from "../tldw/TldwApiClient"
import { bgRequest } from "../background-proxy"

const config = (user = 7) => ({
  serverUrl: "https://characters.test", authMode: "multi-user" as const,
  accessToken: `test.${btoa(JSON.stringify({ sub: String(user) }))}.signature`
})
const requestScope = { config: { serverUrl: "https://characters.test", authMode: "multi-user" as const }, userId: 7 }
const preferenceKey = "preferences.chat.default_character_id"

describe("default Character through the real account-scoped transport", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    boundary.get.mockImplementation(async (key: string) => key === "tldwConfig" ? config() : null)
    boundary.fetch.mockImplementation(async (_url: string, init: RequestInit) => new Response(JSON.stringify(
      init.method === "PATCH" ? { applied: [preferenceKey], skipped: [] } : { preferences: { [preferenceKey]: { value: 4 } } }
    ), { status: 200, headers: { "Content-Type": "application/json" } }))
    vi.stubGlobal("fetch", boundary.fetch)
  })
  afterEach(() => {
    vi.unstubAllGlobals()
    vi.unstubAllEnvs()
  })

  it("reads the server default with the captured account through the real proxy", async () => {
    expect(await new TldwApiClient().getDefaultCharacterPreference({ requestScope })).toBe("4")
    const [url, init] = boundary.fetch.mock.calls[0]
    expect(String(url)).toBe("https://characters.test/api/v1/users/me/profile?sections=preferences")
    expect(new Headers(init.headers).get("X-TLDW-Expected-User-ID")).toBe("7")
  })

  it.each(["4", null])("saves or clears the default through the real proxy: %s", async id => {
    await new TldwApiClient().setDefaultCharacterPreference(id, { requestScope })
    const [url, init] = boundary.fetch.mock.calls[0]
    expect(String(url)).toBe("https://characters.test/api/v1/users/me/profile")
    expect(init.method).toBe("PATCH")
    expect(JSON.parse(init.body)).toEqual({ updates: [{ key: preferenceKey, value: id }] })
    expect(new Headers(init.headers).get("X-TLDW-Expected-User-ID")).toBe("7")
    expect(new Headers(init.headers).get("Authorization")).toBe(`Bearer ${config().accessToken}`)
  })

  it("rejects a different current account before dispatch", async () => {
    boundary.get.mockImplementation(async (key: string) => key === "tldwConfig" ? config(8) : null)
    await expect(new TldwApiClient().setDefaultCharacterPreference("4", { requestScope })).rejects.toMatchObject({ status: 412 })
    expect(boundary.fetch).not.toHaveBeenCalled()
  })

  it.each(["GET", "PATCH"])("carries the owner assertion through cookie-session %s", async method => {
    vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "hosted")
    const cookieConfig = { ...requestScope.config, authSource: "cookie-session" as const }
    boundary.get.mockImplementation(async (key: string) => key === "tldwConfig" ? cookieConfig : null)
    const options = { requestScope: { config: cookieConfig, userId: 7 } }
    const client = new TldwApiClient()
    if (method === "GET") await client.getDefaultCharacterPreference(options)
    else await client.setDefaultCharacterPreference("4", options)
    const [, init] = boundary.fetch.mock.calls[0]
    expect(init.method).toBe(method)
    expect(new Headers(init.headers).get("X-TLDW-Expected-User-ID")).toBe("7")
    expect(new Headers(init.headers).has("Authorization")).toBe(false)
  })

  it.each([
    ["GET", "/api/v1/users/other/profile"], ["PATCH", "/api/v1/users/other/profile"],
    ["DELETE", "/api/v1/users/me/profile"], ["POST", "/api/v1/users/me/profile"],
    ["PUT", "/api/v1/users/me/profile"], ["GET", "/api/v1/users/me/profile/nested"],
    ["GET", "/api/v1/users/me/profile/"], ["GET", "/api/v1/users//me/profile"],
    ["PATCH", "/api/v1/users/me/../me/profile"], ["GET", "/api/v1/users/me%2fprofile"],
    ["GET", "https://foreign.test/api/v1/users/me/profile"]
  ])("rejects unrelated profile operations: %s %s", async (method, path) => {
    await expect(bgRequest({
      path, method,
      servicePromptConfig: { ...requestScope.config, expectedUserId: 7 }
    } as Parameters<typeof bgRequest>[0])).rejects.toThrow(/Service Prompt/)
    expect(boundary.fetch).not.toHaveBeenCalled()
  })
})
