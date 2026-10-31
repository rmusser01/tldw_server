import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { TldwApiClient } from "../tldw/TldwApiClient"

const boundary = vi.hoisted(() => ({ get: vi.fn(), fetch: vi.fn() }))
vi.mock("wxt/browser", () => ({ browser: { runtime: { id: null } } }))
vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: () => ({
    get: boundary.get,
    set: vi.fn(),
    remove: vi.fn()
  }),
  safeStorageSerde: {
    serialize: (value: unknown) => value,
    deserialize: (value: unknown) => value
  }
}))
vi.mock("@/services/tldw/runtime-auth-override", () => ({
  getRuntimeSingleUserApiKeyOverride: () => null,
  isCookieSessionConfigInvalidated: () => false
}))

const config = (user = 7) => ({
  serverUrl: "https://characters.test",
  authMode: "multi-user" as const,
  accessToken: `test.${btoa(JSON.stringify({ sub: String(user) }))}.signature`
})
const requestScope = {
  config: {
    serverUrl: "https://characters.test",
    authMode: "multi-user" as const
  },
  userId: 7
}

describe("sidepanel history owner through the real transport", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    boundary.get.mockImplementation(async (key: string) =>
      key === "tldwConfig" ? config() : null
    )
    boundary.fetch.mockResolvedValue(
      new Response(JSON.stringify({ chats: [], total: 0 }), {
        status: 200,
        headers: { "Content-Type": "application/json" }
      })
    )
    vi.stubGlobal("fetch", boundary.fetch)
  })
  afterEach(() => {
    vi.unstubAllGlobals()
    vi.unstubAllEnvs()
  })
  it.each(["search", "list"])(
    "carries the captured account through %s",
    async (method) => {
      const client = new TldwApiClient()
      if (method === "search")
        await client.searchConversationsWithMeta(
          { query: "private" },
          { requestScope }
        )
      else await client.listChatsWithMeta({}, { requestScope })
      const [, init] = boundary.fetch.mock.calls[0]
      expect(new Headers(init.headers).get("X-TLDW-Expected-User-ID")).toBe("7")
    }
  )
  it.each(["search", "list"])(
    "rejects an account replacement before %s dispatch",
    async (method) => {
      boundary.get.mockImplementation(async (key: string) =>
        key === "tldwConfig" ? config(8) : null
      )
      const client = new TldwApiClient()
      const read =
        method === "search"
          ? client.searchConversationsWithMeta(
              { query: "private" },
              { requestScope }
            )
          : client.listChatsWithMeta({}, { requestScope })
      await expect(read).rejects.toMatchObject({ status: 412 })
      expect(boundary.fetch).not.toHaveBeenCalled()
    }
  )
})
