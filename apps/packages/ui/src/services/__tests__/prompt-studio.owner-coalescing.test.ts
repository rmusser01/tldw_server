import { afterEach, beforeEach, expect, it, vi } from "vitest"
import { QueryClient } from "@tanstack/react-query"
import { listProjects } from "@/services/prompt-studio"

const fixture = vi.hoisted(() => ({ user: "bob" }))
vi.mock("wxt/browser", () => ({ browser: { runtime: { id: null } } }))
vi.mock("@/utils/safe-storage", () => ({
  safeStorageSerde: { serializer: JSON.stringify, deserializer: (value: unknown) => value },
  createSafeStorage: () => ({
    get: async (key: string) => key === "tldwConfig" ? {
      serverUrl: "http://127.0.0.1:19999", authMode: "multi-user",
      accessToken: `header.${btoa(JSON.stringify({ sub: fixture.user }))}.signature`
    } : null,
    set: vi.fn(), remove: vi.fn()
  })
}))
const response = (name: string) => new Response(JSON.stringify({ success: true, data: [{ id: 1, name }] }), {
  status: 200, headers: { "content-type": "application/json" }
})
const deferred = <T,>() => { let resolve!: (value: T) => void; const promise = new Promise<T>(done => { resolve = done }); return { promise, resolve } }
beforeEach(() => { fixture.user = "bob" })
afterEach(() => vi.unstubAllGlobals())

it("does not admit a held A transport response into B's separate Studio cache", async () => {
  const a = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const b = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const held = deferred<Response>()
  const started = deferred<void>()
  const fetcher = vi.fn<typeof fetch>(async (_url, init) => {
    const authorization = new Headers(init?.headers).get("Authorization")
    if (authorization?.includes(btoa(JSON.stringify({ sub: "bob" })))) { started.resolve(); return held.promise }
    return response("Alice private")
  })
  vi.stubGlobal("fetch", fetcher)
  const key = ["prompt-studio", "projects"]
  const old = a.fetchQuery({ queryKey: key, queryFn: () => listProjects() }).catch(() => null)
  try {
    await started.promise
    fixture.user = "alice"
    a.clear()
    const current = b.fetchQuery({ queryKey: key, queryFn: () => listProjects() })
    held.resolve(response("Bob private"))
    const result = await current
    expect(result.data?.data).toEqual([{ id: 1, name: "Alice private" }])
    expect(b.getQueryData(key)).toEqual(result)
    expect(fetcher).toHaveBeenCalledTimes(2)
  } finally { held.resolve(response("Bob private")); await old; a.clear(); b.clear() }
})

it("keeps ordinary same-owner Studio reads successful through real transport", async () => {
  const fetcher = vi.fn<typeof fetch>(async () => response("Bob private"))
  vi.stubGlobal("fetch", fetcher)
  const result = await listProjects({ page: 1, per_page: 100 })
  expect(result.ok).toBe(true)
  expect(result.data?.data).toEqual([{ id: 1, name: "Bob private" }])
  expect(fetcher).toHaveBeenCalledTimes(1)
})
