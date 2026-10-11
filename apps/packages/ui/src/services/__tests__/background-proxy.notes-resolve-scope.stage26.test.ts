import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const boundary = vi.hoisted(() => ({
  get: vi.fn(),
  fetch: vi.fn(),
  sendMessage: vi.fn(),
  runtimeId: null as string | null
}))
vi.mock("wxt/browser", () => ({
  browser: {
    runtime: {
      get id() {
        return boundary.runtimeId
      },
      sendMessage: (...args: unknown[]) => boundary.sendMessage(...args)
    }
  }
}))
vi.mock("@/utils/safe-storage", () => ({
  safeStorageSerde: {
    serializer: JSON.stringify,
    deserializer: (value: unknown) => value
  },
  createSafeStorage: () => ({
    get: boundary.get,
    set: vi.fn(async () => {}),
    remove: vi.fn(async () => {})
  })
}))
vi.mock("@/services/tldw/runtime-auth-override", () => ({
  getRuntimeSingleUserApiKeyOverride: () => null,
  isCookieSessionConfigInvalidated: () => false
}))

import { bgRequest } from "../background-proxy"
import { requestScopeFields } from "../tldw/domains/service-prompts"

const config = (user = 7, serverUrl = "https://notes.example") => ({
  serverUrl,
  authMode: "multi-user" as const,
  accessToken: `fixture.${btoa(JSON.stringify({ sub: String(user) }))}.signature`
})
const capturedConfig = config()
const requestScope = {
  config: { serverUrl: capturedConfig.serverUrl, authMode: capturedConfig.authMode },
  userId: 7
}
const path = "/api/v1/notes/wikilinks/resolve"
const noteId = "925871f6-30fa-470b-a2c4-2272051f2373"
const payload = { titles: ["Private note"], ids: [], source_note_id: noteId }
const response = {
  titles: [{ title: "Private note", note_id: noteId, note_title: "Private note", candidate_count: 1 }],
  ids: []
}
const resolve = (
  requestPath = path,
  method = "POST",
  abortSignal = new AbortController().signal
) => bgRequest({
  ...requestScopeFields(requestScope),
  path: requestPath as never,
  method: method as never,
  body: payload,
  abortSignal
})

beforeEach(() => {
  vi.clearAllMocks()
  boundary.runtimeId = null
  boundary.get.mockImplementation(async (key: string) =>
    key === "tldwConfig" ? config() : null
  )
  boundary.fetch.mockImplementation(async (_input: RequestInfo | URL, init?: RequestInit) => {
    init?.signal?.throwIfAborted()
    return new Response(JSON.stringify(response), {
      status: 200,
      headers: { "Content-Type": "application/json" }
    })
  })
  boundary.sendMessage.mockImplementation(async (message: { type: string }) =>
    message.type === "tldw:connection-authority"
      ? { ok: true, epoch: "notes-fixture-epoch" }
      : { ok: true, status: 200, data: response }
  )
  vi.stubGlobal("fetch", boundary.fetch)
})
afterEach(() => vi.unstubAllGlobals())

describe("captured Notes wikilink resolve through guarded transport", () => {
  it("dispatches the actual resolve POST with captured credentials and expected user", async () => {
    const signal = new AbortController().signal
    await expect(resolve(path, "POST", signal)).resolves.toEqual(response)
    expect(boundary.fetch).toHaveBeenCalledTimes(1)
    const [url, init] = boundary.fetch.mock.calls[0]
    expect(String(url)).toBe(`https://notes.example${path}`)
    expect(init.method).toBe("POST")
    expect(new Headers(init.headers).get("Authorization")).toBe(`Bearer ${capturedConfig.accessToken}`)
    expect(new Headers(init.headers).get("X-TLDW-Expected-User-ID")).toBe("7")
    expect(JSON.parse(init.body)).toEqual(payload)
    expect(init.signal).toBeInstanceOf(AbortSignal)
    expect(boundary.sendMessage).not.toHaveBeenCalled()
  })

  it.each(["principal", "server", "auth source", "organization"])(
    "rejects a changed %s while loading credentials before dispatch",
    async (change) => {
      let release!: (value: unknown) => void
      const pendingConfig = new Promise((resolveConfig) => { release = resolveConfig })
      boundary.get.mockImplementation(async (key: string) =>
        key === "tldwConfig" ? pendingConfig : null
      )
      const pending = resolve()
      const rejected = expect(pending).rejects.toMatchObject({ status: 412 })
      release(change === "principal" ? config(8)
        : change === "server" ? config(7, "https://other.example")
        : change === "auth source" ? { ...config(), authSource: "cookie-session" }
        : { ...config(), orgId: "other-org" })
      await rejected
      expect(boundary.fetch).not.toHaveBeenCalled()
      expect(boundary.sendMessage).not.toHaveBeenCalled()
    }
  )

  it("binds extension dispatch to the captured target and expected user without mixed guards", async () => {
    boundary.runtimeId = "extension-fixture"
    await expect(resolve()).resolves.toEqual(response)
    expect(boundary.sendMessage).toHaveBeenCalledTimes(1)
    expect(boundary.sendMessage).toHaveBeenCalledWith(expect.objectContaining({
      type: "tldw:request",
      payload: expect.objectContaining({
        path,
        method: "POST",
        body: payload,
        servicePromptConfig: { ...requestScope.config, expectedUserId: 7 },
        headers: { "X-TLDW-Expected-User-ID": "7" }
      })
    }))
    expect(boundary.sendMessage.mock.calls[0][0].payload).not.toHaveProperty("expectedConnectionAuthority")
    expect(boundary.sendMessage.mock.calls[0][0].payload).not.toHaveProperty("expectedConnectionEpoch")
    expect(boundary.fetch).not.toHaveBeenCalled()
  })

  it("does not fall back to direct credentials when the extension rejects the captured scope", async () => {
    boundary.runtimeId = "extension-fixture"
    boundary.sendMessage.mockResolvedValue({ ok: false, status: 412, error: "Captured scope changed" })
    await expect(resolve()).rejects.toMatchObject({ status: 412 })
    expect(boundary.sendMessage).toHaveBeenCalledTimes(1)
    expect(boundary.fetch).not.toHaveBeenCalled()
  })

  it("propagates an already aborted captured request to direct fetch", async () => {
    const controller = new AbortController()
    controller.abort()
    await expect(resolve(path, "POST", controller.signal)).rejects.toMatchObject({ name: "AbortError" })
    expect(boundary.fetch.mock.calls[0][1].signal.aborted).toBe(true)
  })

  it.each(["GET", "PUT", "PATCH", "DELETE"])("rejects resolve method %s", async (method) => {
    await expect(resolve(path, method)).rejects.toThrow(/Service Prompt config/)
    expect(boundary.fetch).not.toHaveBeenCalled()
    expect(boundary.sendMessage).not.toHaveBeenCalled()
  })

  it.each([
    `${path}/`,
    `${path}/extra`,
    "/api/v1/notes/wikilinks/referrers",
    "/api/v1/notes/wikilinks/rewrite",
    "/api/v1/notes/wikilinks//resolve",
    "/api/v1/notes/wikilinks/../wikilinks/resolve",
    "/api/v1/notes/wikilinks/%2e%2e/resolve",
    "/api/v1/notes/wikilinks%2fresolve",
    "/api/v1/notes/wikilinks\\resolve",
    `${path}%zz`,
    `https://other.example${path}`
  ])("rejects malformed or expanded resolve path %s", async (requestPath) => {
    await expect(resolve(requestPath)).rejects.toThrow(/Service Prompt config/)
    expect(boundary.fetch).not.toHaveBeenCalled()
    expect(boundary.sendMessage).not.toHaveBeenCalled()
  })
})
