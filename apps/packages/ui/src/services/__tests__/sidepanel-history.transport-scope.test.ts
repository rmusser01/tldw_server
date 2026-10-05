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
  it.each(["search", "list", "get"])(
    "carries the captured account through %s",
    async (method) => {
      const client = new TldwApiClient()
      if (method === "search")
        await client.searchConversationsWithMeta(
          { query: "private" },
          { requestScope }
        )
      else if (method === "list") await client.listChatsWithMeta({}, { requestScope })
      else await client.getChat("folder-chat", { requestScope, scope: { type: "global" }, signal: new AbortController().signal })
      const [url, init] = boundary.fetch.mock.calls[0]
      if (method === "get") {
        expect(new URL(url).pathname).toBe("/api/v1/chats/folder-chat")
        expect(new URL(url).searchParams.get("scope_type")).toBe("global")
        expect(init.signal).toBeInstanceOf(AbortSignal)
      }
      expect(new Headers(init.headers).get("X-TLDW-Expected-User-ID")).toBe("7")
    }
  )
  it.each(["search", "list", "get"])(
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
          : method === "list"
            ? client.listChatsWithMeta({}, { requestScope })
            : client.getChat("folder-chat", { requestScope, scope: { type: "global" } })
      await expect(read).rejects.toMatchObject({ status: 412 })
      expect(boundary.fetch).not.toHaveBeenCalled()
    }
  )
})
