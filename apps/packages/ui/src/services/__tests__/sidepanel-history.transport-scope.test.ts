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
  it.each(["search", "list", "get", "create"])(
    "carries the captured account through %s",
    async (method) => {
      const client = new TldwApiClient()
      if (method === "search")
        await client.searchConversationsWithMeta(
          { query: "private" },
          { requestScope }
        )
      else if (method === "list") await client.listChatsWithMeta({}, { requestScope })
      else if (method === "get") await client.getChat("folder-chat", { requestScope, scope: { type: "global" }, signal: new AbortController().signal })
      else await client.createChat({character_id: 42, provider: "openai", model: "configured-model"}, {
        requestScope, signal: new AbortController().signal, seedFirstMessage: true, greetingStrategy: "alternate_index", alternateIndex: 0
      })
      const [url, init] = boundary.fetch.mock.calls[0]
      if (method === "get") {
        expect(new URL(url).pathname).toBe("/api/v1/chats/folder-chat")
        expect(new URL(url).searchParams.get("scope_type")).toBe("global")
        expect(init.signal).toBeInstanceOf(AbortSignal)
      }
      if (method === "create") {
        expect(new URL(url).pathname).toBe("/api/v1/chats/")
        expect(new URL(url).searchParams.get("seed_first_message")).toBe("true")
        expect(new URL(url).searchParams.get("alternate_index")).toBe("0")
        expect(JSON.parse(init.body)).toEqual({character_id: 42, provider: "openai", model: "configured-model", scope_type: "global"})
        expect(init.signal).toBeInstanceOf(AbortSignal)
      }
      expect(new Headers(init.headers).get("X-TLDW-Expected-User-ID")).toBe("7")
    }
  )
  it.each(["search", "list", "get", "create"])(
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
            : method === "get"
              ? client.getChat("folder-chat", { requestScope, scope: { type: "global" } })
              : client.createChat({character_id: 42}, {requestScope, seedFirstMessage: true})
      await expect(read).rejects.toMatchObject({ status: 412 })
      expect(boundary.fetch).not.toHaveBeenCalled()
    }
  )
})
