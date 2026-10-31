import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const boundary = vi.hoisted(() => ({ get: vi.fn(), fetch: vi.fn() }))
vi.mock("wxt/browser", () => ({ browser: { runtime: { id: null } } }))
vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: () => ({ get: boundary.get, set: vi.fn(), remove: vi.fn(), watch: vi.fn(), unwatch: vi.fn() }),
  safeStorageSerde: { serialize: (value: unknown) => value, deserialize: (value: unknown) => value },
}))
vi.mock("@/services/tldw/runtime-auth-override", () => ({ getRuntimeSingleUserApiKeyOverride: () => null, isCookieSessionConfigInvalidated: () => false }))
import { TldwApiClient, tldwClient } from "../tldw/TldwApiClient"
import { submitQuickIngestBatch, cancelQuickIngestSession } from "../tldw/quick-ingest-batch"
import { reattachQuickIngestSession } from "../tldw/quick-ingest-session-reattach"

const config = (user = 1, serverUrl = "https://ingest.test") => ({ serverUrl, authMode: "multi-user" as const, accessToken: `test.${btoa(JSON.stringify({ sub: String(user) }))}.signature` })
const options = { requestScope: { config: { serverUrl: "https://ingest.test", authMode: "multi-user" as const }, userId: 1 } }
let client: TldwApiClient
const file = () => ({ name: "owned.pdf", type: "application/pdf", arrayBuffer: async () => new ArrayBuffer(3) }) as File
const operations = [
  ["direct batch upload", () => submitQuickIngestBatch({ ...options, entries: [], files: [{ name: "owned.pdf", type: "application/pdf", data: new Uint8Array([1]) }], storeRemote: false, processOnly: true })],
  ["reattach colliding ID", () => reattachQuickIngestSession({ mode: "webui-direct", jobIds: [7] }, options)],
  ["owned cancellation", () => cancelQuickIngestSession({ ...options, sessionId: "qi-direct-owned", batchIds: ["owned-batch"] })],
  ["picker library", () => client.listMedia({ page: 1 }, options)],
  ["picker search", () => client.searchMedia({ query: "Owned title" }, {}, options)],
  ["picker upload", () => client.uploadMedia(file(), { media_type: "pdf", keep_original_file: true }, options)],
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

describe("Quick Ingest real outbound scope", () => {
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
  it("does not upload a file after its capture generation changes during file reading", async () => {
    let complete!: (value: ArrayBuffer) => void
    const data = new Promise<ArrayBuffer>(resolve => { complete = resolve })
    const controller = new AbortController()
    const pending = client.uploadMedia({ ...file(), arrayBuffer: () => data } as File, {}, { ...options, signal: controller.signal })
    controller.abort(); complete(new ArrayBuffer(1))
    await expect(pending).rejects.toMatchObject({ name: "AbortError" })
    expect(boundary.fetch).not.toHaveBeenCalled()
  })
})
