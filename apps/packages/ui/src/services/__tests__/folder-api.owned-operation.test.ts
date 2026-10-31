import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
const boundary = vi.hoisted(() => ({ get: vi.fn(), fetch: vi.fn() }))
vi.mock("wxt/browser", () => ({ browser: { runtime: { id: null } } }))
vi.mock("@/utils/safe-storage", () => ({ createSafeStorage: () => ({ get: boundary.get, set: vi.fn(), remove: vi.fn() }) }))
vi.mock("@/services/tldw/runtime-auth-override", () => ({ getRuntimeSingleUserApiKeyOverride: () => null, isCookieSessionConfigInvalidated: () => false }))
import { createFolder, createKeyword, fetchFolders, fetchFolderKeywordLinks, fetchConversationKeywordLinks, linkKeywordToFolder, linkKeywordToConversation } from "../folder-api"
import { isServicePromptRequestPath } from "../tldw/service-prompt-scope-error"
const config = (user = 7) => ({ serverUrl: "https://folders.test", authMode: "multi-user" as const, accessToken: `test.${btoa(JSON.stringify({ sub: String(user) }))}.signature` })
const scope = (user = 7) => ({ config: { serverUrl: "https://folders.test", authMode: "multi-user" as const }, userId: user })

describe("captured history folder/tag API through real resource and transport paths", () => {
  beforeEach(() => { vi.clearAllMocks(); boundary.get.mockImplementation(async (key: string) => key === "tldwConfig" ? config() : null)
    boundary.fetch.mockImplementation(async () => new Response(JSON.stringify([]), { status: 200, headers: { "Content-Type": "application/json" } }))
    vi.stubGlobal("fetch", boundary.fetch) })
  afterEach(() => { vi.unstubAllGlobals(); vi.unstubAllEnvs() })
  it("forwards account checks through resource-client and direct link requests", async () => {
    const requestScope = scope()
    await createFolder("Adventure", null, { requestScope })
    await createKeyword("Story", { requestScope })
    await fetchFolderKeywordLinks({ requestScope })
    await fetchConversationKeywordLinks(["owned-chat"], { requestScope })
    await linkKeywordToFolder(1, 9, { requestScope })
    await linkKeywordToConversation("owned-chat", 9, { requestScope })
    expect(boundary.fetch).toHaveBeenCalledTimes(6)
    expect(boundary.fetch.mock.calls.every(([, init]) => new Headers(init.headers).get("X-TLDW-Expected-User-ID") === "7")).toBe(true)
  })
  it("refuses replacement credentials before any captured folder/tag request", async () => {
    boundary.get.mockImplementation(async (key: string) => key === "tldwConfig" ? config(8) : null)
    expect(await createFolder("Old account folder", null, { requestScope: scope() })).toMatchObject({ ok: false, status: 412 })
    expect(await linkKeywordToConversation("owned-chat", 9, { requestScope: scope() })).toMatchObject({ ok: false, status: 412 })
    expect(boundary.fetch).not.toHaveBeenCalled()
  })
  it("does not join an in-flight identical path from another account", async () => {
    let release!: (response: Response) => void
    boundary.fetch.mockImplementationOnce(() => new Promise<Response>(resolve => { release = resolve }))
    const old = fetchFolders({ requestScope: scope() })
    await vi.waitFor(() => expect(boundary.fetch).toHaveBeenCalledTimes(1))
    boundary.get.mockImplementation(async (key: string) => key === "tldwConfig" ? config(8) : null)
    const next = fetchFolders({ requestScope: scope(8) })
    await vi.waitFor(() => expect(boundary.fetch).toHaveBeenCalledTimes(2)).finally(() => release(new Response(JSON.stringify([{ id: 7, name: "Old" }]), { status: 200, headers: { "Content-Type": "application/json" } })))
    await old; expect(await next).toMatchObject({ ok: true, data: [] })
  })
  it("a canceled page does not dispatch the next captured-account GET", async () => {
    let release!: (response: Response) => void
    boundary.fetch.mockImplementationOnce(() => new Promise<Response>(resolve => { release = resolve }))
    const controller = new AbortController()
    const pending = fetchFolders({ requestScope: scope(), abortSignal: controller.signal })
    await vi.waitFor(() => expect(boundary.fetch).toHaveBeenCalledTimes(1))
    controller.abort()
    release(new Response(JSON.stringify({ collections: [{ id: 7, name: "First" }], total: 2,
      pagination: { has_more: true, next_offset: 1 } }), { status: 200, headers: { "Content-Type": "application/json" } }))
    expect(await pending).toMatchObject({ ok: false })
    expect(boundary.fetch).toHaveBeenCalledTimes(1)
  })
  it("a fresh picker on the same account does not join a canceled read lifetime", async () => {
    let release!: (response: Response) => void
    boundary.fetch.mockImplementationOnce(() => new Promise<Response>(resolve => { release = resolve }))
    const requestScope = scope()
    const oldController = new AbortController()
    const pending = fetchFolders({ requestScope, abortSignal: oldController.signal })
    await vi.waitFor(() => expect(boundary.fetch).toHaveBeenCalledTimes(1))
    oldController.abort()
    const next = fetchFolders({ requestScope, abortSignal: new AbortController().signal })
    await vi.waitFor(() => expect(boundary.fetch).toHaveBeenCalledTimes(2)).finally(() =>
      release(new Response(JSON.stringify([]), { status: 200, headers: { "Content-Type": "application/json" } })))
    await pending
    expect(await next).toMatchObject({ ok: true, data: [] })
  })
  it("identical reads for a new credential lineage on the same user remain separate", async () => {
    let release!: (response: Response) => void
    boundary.fetch.mockImplementationOnce(() => new Promise<Response>(resolve => { release = resolve }))
    const pending = fetchFolders({ requestScope: scope() })
    await vi.waitFor(() => expect(boundary.fetch).toHaveBeenCalledTimes(1))
    boundary.get.mockImplementation(async (key: string) => key === "tldwConfig" ? { ...config(), refreshToken: "new-lineage" } : null)
    const next = fetchFolders({ requestScope: { ...scope(), config: { ...scope().config, expectedRefreshToken: "new-lineage" } } })
    await vi.waitFor(() => expect(boundary.fetch).toHaveBeenCalledTimes(2)).finally(() =>
      release(new Response(JSON.stringify([]), { status: 200, headers: { "Content-Type": "application/json" } })))
    await pending
    expect(await next).toMatchObject({ ok: true, data: [] })
  })
  it.each([
    ["/api/v1/notes/collections", "POST", true],
    ["/api/v1/notes/keywords", "POST", true],
    ["/api/v1/notes/collections/1/keywords/9", "POST", true],
    ["/api/v1/notes/conversations/owned-chat/keywords/9", "POST", true],
    ["/api/v1/notes/collections/keyword-links", "GET", true],
    ["/api/v1/notes/conversations/keyword-links?ids=owned-chat", "GET", true],
    ["/api/v1/notes/collections/1/keywords/9/extra", "POST", false],
    ["/api/v1/notes/collections/1/keywords/no-id", "POST", false],
    ["/api/v1/notes/conversations/owned-chat/keywords/9", "PUT", false],
    ["/api/v1/notes/collections/../keywords", "POST", false],
    ["/api/v1/notes/collections/%2e%2e/keywords/9", "POST", false]
  ])("allows only exact existing folder/tag operations %s %s", (path, method, allowed) => {
    expect(isServicePromptRequestPath(path, method)).toBe(allowed)
  })
  it("cookie transport keeps the expected-account header and returns the server's stale-account refusal", async () => {
    vi.stubEnv("NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE", "hosted")
    const cookieConfig = { serverUrl: window.location.origin, authMode: "multi-user", authSource: "cookie-session" }
    boundary.get.mockImplementation(async (key: string) => ["tldwConfig", "tldwCookieSessionConfig"].includes(key) ? cookieConfig : null)
    boundary.fetch.mockResolvedValue(new Response(JSON.stringify({ detail: { code: "request_config_scope_changed" } }), { status: 412, headers: { "Content-Type": "application/json" } }))
    const result = await createFolder("Captured account", null, { requestScope: {
      config: { serverUrl: window.location.origin, authMode: "multi-user", authSource: "cookie-session" }, userId: 7 } })
    expect(result).toMatchObject({ ok: false, status: 412 })
    const [url, init] = boundary.fetch.mock.calls[0]
    const target = new URL(String(url), window.location.origin)
    expect(target.origin).toBe(window.location.origin)
    expect(new Headers(init.headers).get("X-TLDW-Expected-User-ID")).toBe("7")
    expect(new Headers(init.headers).has("Authorization")).toBe(false)
    expect(new Request(target, init).credentials).toBe("same-origin")
  })
  it("normalizes actual collection_id links from every reported backend page", async () => {
    boundary.fetch.mockImplementation(async (url: string) => {
      const offset = Number(new URL(url).searchParams.get("offset") || 0)
      const response = { links: [{ collection_id: offset === 0 ? 1 : 2, keyword_id: offset === 0 ? 9 : 10 }],
        offset, count: 1, total: 2, pagination: { has_more: offset === 0, next_offset: offset === 0 ? 1 : null } }
      return new Response(JSON.stringify(response), { status: 200, headers: { "Content-Type": "application/json" } })
    })
    const result = await fetchFolderKeywordLinks({ requestScope: scope() })
    expect(result).toMatchObject({ ok: true, data: [{ folder_id: 1, keyword_id: 9 }, { folder_id: 2, keyword_id: 10 }] })
    expect(boundary.fetch).toHaveBeenCalledTimes(2)
  })
  it("fails an incomplete page sequence rather than publishing a partial successful snapshot", async () => {
    boundary.fetch.mockResolvedValueOnce(new Response(JSON.stringify({ collections: [{ id: 1, name: "First" }], total: 2, pagination: { has_more: true, next_offset: 1 } }), { status: 200, headers: { "Content-Type": "application/json" } }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ detail: "Unavailable" }), { status: 503, headers: { "Content-Type": "application/json" } }))
    expect(await fetchFolders({ requestScope: scope() })).toMatchObject({ ok: false, status: 503 })
    expect(boundary.fetch).toHaveBeenCalledTimes(2)
  })

})
