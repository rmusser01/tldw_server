import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const boundary = vi.hoisted(() => ({ get: vi.fn(), fetch: vi.fn() }))
vi.mock("wxt/browser", () => ({ browser: { runtime: { id: null } } }))
vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: () => ({ get: boundary.get, set: vi.fn(), remove: vi.fn(), watch: vi.fn(), unwatch: vi.fn() }),
  safeStorageSerde: { serialize: (value: unknown) => value, deserialize: (value: unknown) => value },
}))
vi.mock("@/services/tldw/runtime-auth-override", () => ({ getRuntimeSingleUserApiKeyOverride: () => null, isCookieSessionConfigInvalidated: () => false }))
import { TldwApiClient, tldwClient } from "../tldw/TldwApiClient"
import { bgRequest } from "../background-proxy"
import { requestScopeFields } from "../tldw/domains/service-prompts"
import { mediaMethods } from "../tldw/domains/media"


const config = (user = 1, serverUrl = "https://ingest.test") => ({ serverUrl, authMode: "multi-user" as const, accessToken: `test.${btoa(JSON.stringify({ sub: String(user) }))}.signature` })
const options = { requestScope: { config: { serverUrl: "https://ingest.test", authMode: "multi-user" as const }, userId: 1 } }
let client: TldwApiClient
const operations = [
  ['Inspector quota', () => bgRequest({ path: '/api/v1/users/storage', method: 'GET', ...requestScopeFields(options.requestScope) })],
  ['Inspector note trash', () => bgRequest({ path: '/api/v1/notes/7', method: 'DELETE', headers: { 'expected-version': '7' }, ...requestScopeFields(options.requestScope) })],
  ['Inspector note restore', () => bgRequest({ path: '/api/v1/notes/7/restore?expected_version=8', method: 'POST', ...requestScopeFields(options.requestScope) })],
  ['Inspector media restore', () => bgRequest({ path: '/api/v1/media/7/restore', method: 'POST', ...requestScopeFields(options.requestScope) })],
  ['bulk tags', () => client.bulkUpdateMediaKeywords({media_ids:[7], keywords:['owned']}, options)],
  ['trash', () => client.deleteMedia(7, options)],
  ['reprocess', () => client.reprocessMedia(7, {perform_chunking:true}, options)],
  ['domain bulk tags', () => mediaMethods.bulkUpdateMediaKeywords({media_ids:[7], keywords:['owned']}, options)],
  ['domain trash', () => mediaMethods.deleteMedia(7, options)],
  ['domain reprocess', () => mediaMethods.reprocessMedia(7, {perform_chunking:true}, options)],
] as const

beforeEach(() => {
  vi.clearAllMocks()
  client = new TldwApiClient()
  vi.spyOn(tldwClient, "initialize").mockResolvedValue(undefined)
  vi.spyOn(client, "getConfig").mockResolvedValue(config())
  boundary.get.mockImplementation(async (key: string) => key === "tldwConfig" ? config() : null)
  boundary.fetch.mockImplementation(async () => new Response('{"status":"completed","result":{"media_id":7}}', { status: 200, headers: { "Content-Type": "application/json" } }))
  vi.stubGlobal("fetch", boundary.fetch)
})
afterEach(() => vi.unstubAllGlobals())

describe("Review batch real outbound scope", () => {
  it.each(operations)("sends %s only to the captured target and principal", async (_label, run) => {
    await run()
    expect(boundary.fetch).toHaveBeenCalledTimes(1)
    const [url, init] = boundary.fetch.mock.calls[0]
    expect(new URL(url).origin).toBe("https://ingest.test")
    expect(new Headers(init.headers).get("X-TLDW-Expected-User-ID")).toBe("1")
    expect(new Headers(init.headers).get("Authorization")).toBe(`Bearer ${config().accessToken}`)
  })
  it.each(operations)("blocks %s before sending to a replacement server", async (_label, run) => {
    boundary.get.mockImplementation(async (key: string) => key === "tldwConfig" ? config(1, "https://foreign.test") : null)
    await expect(run()).rejects.toMatchObject({ status: 412 })
    expect(boundary.fetch).not.toHaveBeenCalled()
  })
  it.each(operations)("blocks %s before using replacement credentials with colliding IDs", async (_label, run) => {
    boundary.get.mockImplementation(async (key: string) => key === "tldwConfig" ? config(2) : null)
    await expect(run()).rejects.toMatchObject({ status: 412 })
    expect(boundary.fetch).not.toHaveBeenCalled()
  })
  it.each(operations)('blocks %s when owner changes while request config is loading', async (_label, run) => {
    let release!: () => void
    let entered!: () => void
    const started = new Promise<void>(resolve => {entered = resolve})
    const wait = new Promise<void>(resolve => {release = resolve})
    boundary.get.mockImplementation(async (key: string) => {
      if (key !== 'tldwConfig') return null
      entered()
      await wait
      return config(2, 'https://foreign.test')
    })
    const pending = run()
    await started
    release()
    await expect(pending).rejects.toMatchObject({status:412})
    expect(boundary.fetch).not.toHaveBeenCalled()
  })
  it.each(['client','domain'])('keeps the %s bulk fallback scoped after a server transition', async (kind) => {
    boundary.fetch.mockImplementation(async () => {
      boundary.get.mockImplementation(async (key: string) => key === 'tldwConfig' ? config(2, 'https://foreign.test') : null)
      return new Response('{}', {status:404, headers:{'Content-Type':'application/json'}})
    })
    const api = kind === 'client' ? client : mediaMethods
    const result = await api.bulkUpdateMediaKeywords({media_ids:[7],keywords:['owned']}, options)
    expect(result.failed).toBe(1)
    expect(boundary.fetch).toHaveBeenCalledTimes(1)
  })
})
