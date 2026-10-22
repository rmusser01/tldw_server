import { QueryObserver } from "@tanstack/react-query"
import { waitFor } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { getQueryClient } from "../query-client"

vi.mock("wxt/browser", () => ({ browser: { runtime: { id: null } } }))

const models = vi.hoisted(() => ({
  getConfig: vi.fn(),
  getChatModels: vi.fn(),
  getCachedChatModels: vi.fn(),
  clearCache: vi.fn(),
  subscribeInvalidation: vi.fn()
}))
vi.mock("@/services/tldw", () => ({
  tldwClient: { getConfig: (...args: unknown[]) => models.getConfig(...args) },
  tldwModels: {
    getChatModels: (...args: unknown[]) => models.getChatModels(...args),
    getCachedChatModels: (...args: unknown[]) => models.getCachedChatModels(...args),
    clearCache: (...args: unknown[]) => models.clearCache(...args),
    subscribeInvalidation: (...args: unknown[]) => models.subscribeInvalidation(...args)
  }
}))
vi.mock("@/services/app", () => ({ setNoOfRetrievedDocs: vi.fn(), setTotalFilePerKB: vi.fn() }))

const client = getQueryClient()
const readiness = [
  {
    key: "playground:providersStatus",
    old: { any_configured: true, providers: [{ name: "openai", configured: true }] },
    current: { any_configured: false, providers: [{ name: "openai", configured: false }] }
  },
  {
    key: "playground:chatModels",
    old: [{ model: "old-server-model", provider: "openai", is_configured: true }],
    current: []
  }
]
const boundaries = [
  ["server", "tldw:config-updated", { authorityChanged: true }],
  ["account", "tldw:auth-principal-changed", { kind: "logout" }],
  ["credentials", "tldw:auth-credentials-changed", {}]
] as const

afterEach(() => client.clear())

describe("shared Chat readiness query cache", () => {
  for (const { key, old, current } of readiness) {
    it.each(boundaries)("retires cached " + key + " on a %s boundary", async (_kind, event, detail) => {
      const queryKey = [key]
      const queryFn = vi.fn().mockResolvedValueOnce(old).mockResolvedValueOnce(current)
      await client.fetchQuery({ queryKey, queryFn, staleTime: 60_000 })

      window.dispatchEvent(new CustomEvent(event, { detail }))

      expect(client.getQueryData(queryKey)).toBeUndefined()
      expect(await client.fetchQuery({ queryKey, queryFn, staleTime: 60_000 })).toEqual(current)
    })

    it("keeps cached " + key + " after a same-authority update", async () => {
      const queryKey = [key]
      const queryFn = vi.fn().mockResolvedValue(old)
      await client.fetchQuery({ queryKey, queryFn, staleTime: 60_000 })

      window.dispatchEvent(new CustomEvent("tldw:config-updated", {
        detail: { authorityChanged: false }
      }))

      expect(await client.fetchQuery({ queryKey, queryFn, staleTime: 60_000 })).toEqual(old)
      expect(queryFn).toHaveBeenCalledTimes(1)
    })
  }

  it("refreshes mounted readiness and rejects publication from a retired pending query", async () => {
    const { key, old, current } = readiness[0]
    let release!: (value: unknown) => void
    const held = new Promise(resolve => { release = resolve })
    const queryFn = vi.fn().mockReturnValueOnce(held).mockResolvedValue(current)
    const observer = new QueryObserver(client, {
      queryKey: [key], queryFn, staleTime: 60_000, retry: false
    })
    const unsubscribe = observer.subscribe(() => undefined)
    try {
      window.dispatchEvent(new CustomEvent("tldw:config-updated", {
        detail: { authorityChanged: true }
      }))
      await waitFor(() => expect(client.getQueryData([key])).toEqual(current))
      release(old)
      await held
      await Promise.resolve()
      expect(observer.getCurrentResult().data).toEqual(current)
      expect(queryFn).toHaveBeenCalledTimes(2)
    } finally {
      release(old)
      unsubscribe()
    }
  })

  it("preserves unrelated query data during readiness retirement", () => {
    client.setQueryData(["serverChatHistory", "existing-owner"], [{ id: "owned-chat" }])
    window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed", {
      detail: { kind: "logout" }
    }))
    expect(client.getQueryData(["serverChatHistory", "existing-owner"])).toEqual([{ id: "owned-chat" }])
  })

  it("refreshes the real model service after later connection listeners retire its cache", async () => {
    models.getConfig.mockResolvedValue({
      serverUrl: "http://127.0.0.1:19999", authMode: "single-user", apiKey: "test-key"
    })
    models.getChatModels.mockResolvedValueOnce([
      { id: "old", name: "Old", provider: "openai", type: "chat" }
    ]).mockResolvedValue([
      { id: "current", name: "Current", provider: "openai", type: "chat" }
    ])
    models.getCachedChatModels.mockResolvedValue([])
    let notify: ((token: string) => void) | undefined
    let revision = 0
    models.subscribeInvalidation.mockImplementation((listener: (token: string) => void) => {
      notify = listener
      return () => undefined
    })
    models.clearCache.mockImplementation(async () => { notify?.("models-" + ++revision) })
    // The shared client listener is already installed; load the actual model service afterward.
    const { fetchChatModels } = await import("@/services/tldw-server")
    const observer = new QueryObserver(client, {
      queryKey: ["playground:chatModels"],
      queryFn: () => fetchChatModels({ returnEmpty: true }),
      staleTime: 60_000,
      retry: false
    })
    const unsubscribe = observer.subscribe(() => undefined)
    try {
      await waitFor(() => expect(observer.getCurrentResult().data).toEqual([
        expect.objectContaining({ model: "tldw:old" })
      ]))
      window.dispatchEvent(new CustomEvent("tldw:config-updated", {
        detail: { authorityChanged: true }
      }))
      await waitFor(() => expect(observer.getCurrentResult().data).toEqual([
        expect.objectContaining({ model: "tldw:current" })
      ]))
    } finally {
      unsubscribe()
    }
  })
})
