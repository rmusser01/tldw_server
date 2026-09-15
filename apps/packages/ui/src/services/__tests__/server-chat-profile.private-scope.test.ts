import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
const boundary = vi.hoisted(() => ({ get: vi.fn(), fetch: vi.fn() }))
vi.mock("wxt/browser", () => ({ browser: { runtime: { id: null } } }))
vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: () => ({ get: boundary.get, set: vi.fn(), remove: vi.fn(), watch: vi.fn(), unwatch: vi.fn() }),
  safeStorageSerde: { serialize: (value: unknown) => value, deserialize: (value: unknown) => value }
}))
vi.mock("@/services/tldw/runtime-auth-override", () => ({ getRuntimeSingleUserApiKeyOverride: () => null, isCookieSessionConfigInvalidated: () => false }))
import { TldwApiClient, tldwClient } from "../tldw/TldwApiClient"
const config = (user = 1, serverUrl = "https://chat.test") => ({ serverUrl, authMode: "multi-user" as const, accessToken: `test.${btoa(JSON.stringify({ sub: String(user) }))}.signature` })
const options = (userId = 1) => ({ requestScope: { config: { serverUrl: "https://chat.test", authMode: "multi-user" as const }, userId } })
let client: TldwApiClient
let current = config()
describe("saved Chat profile request authority", () => {
  beforeEach(() => {
    vi.clearAllMocks(); current = config(); client = new TldwApiClient()
    vi.spyOn(tldwClient, "initialize").mockResolvedValue(undefined)
    vi.spyOn(client, "ensureConfigForRequest").mockImplementation(async () => current)
    vi.spyOn(client, "resolveApiPath").mockImplementation(async (_key, paths) => paths[0])
    boundary.get.mockImplementation(async key => key === "tldwConfig" ? current : null)
    boundary.fetch.mockImplementation(async () => new Response(JSON.stringify({ id: 4, name: current.accessToken === config().accessToken ? "Alice Cedar" : "Bob Cedar" }), { status: 200, headers: { "Content-Type": "application/json" } }))
    vi.stubGlobal("fetch", boundary.fetch)
  })
  afterEach(() => vi.unstubAllGlobals())
  it("bypasses the global ID-only cache for an owned profile, preserving the unscoped default", async () => {
    expect((await client.getCharacter(4)).name).toBe("Alice Cedar")
    current = config(2)
    expect((await client.getCharacter(4, options(2))).name).toBe("Bob Cedar")
    expect((await client.getCharacter(4)).name).toBe("Alice Cedar")
    expect(boundary.fetch).toHaveBeenCalledTimes(2)
    expect(new Headers(boundary.fetch.mock.calls[1][1].headers).get("X-TLDW-Expected-User-ID")).toBe("2")
  })
  it.each(["owner", "target"])("never dispatches an old profile request after its %s changes", async kind => {
    current = kind === "owner" ? config(2) : config(1, "https://other.test")
    await expect(client.getCharacter(4, options())).rejects.toMatchObject({ status: 412 })
    expect(boundary.fetch).not.toHaveBeenCalled()
  })
  it.each(["current", "owner", "target"])("binds the common loader's persona profile to its %s scope", async kind => {
    current = kind === "owner" ? config(2) : kind === "target" ? config(1, "https://other.test") : config()
    if (kind !== "current") {
      await expect(client.getPersonaProfile(4, options())).rejects.toMatchObject({ status: 412 })
      expect(boundary.fetch).not.toHaveBeenCalled()
    } else {
      await client.getPersonaProfile(4, options())
      expect(new Headers(boundary.fetch.mock.calls[0][1].headers).get("X-TLDW-Expected-User-ID")).toBe("1")
    }
  })
  it("does not join an unscoped in-flight profile from another owner", async () => {
    let resolve!: (value: Response) => void
    boundary.fetch.mockImplementationOnce(() => new Promise<Response>(done => { resolve = done }))
    const alice = client.getCharacter(4)
    await vi.waitFor(() => expect(boundary.fetch).toHaveBeenCalledTimes(1))
    current = config(2)
    const bob = client.getCharacter(4, options(2))
    resolve(new Response(JSON.stringify({ id: 4, name: "Alice Cedar" }), { status: 200, headers: { "Content-Type": "application/json" } }))
    expect((await bob).name).toBe("Bob Cedar")
    expect((await alice).name).toBe("Alice Cedar")
  })
})
