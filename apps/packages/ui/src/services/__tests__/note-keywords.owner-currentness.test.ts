import { waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import type { TldwConfig } from "../tldw/TldwApiClient"

const boundary = vi.hoisted(() => ({
  request: vi.fn(), ensureConfig: vi.fn(), user: vi.fn(), config: null as TldwConfig | null
}))
vi.mock("@/services/background-proxy", () => ({ bgRequest: boundary.request }))
vi.mock("@/services/tldw/TldwApiClient", () => ({ tldwClient: {
  initialize: async () => {}, ensureConfigForRequest: boundary.ensureConfig
} }))
vi.mock("@/services/tldw/TldwAuth", () => ({ tldwAuth: { getCurrentUser: boundary.user } }))
vi.mock("@/services/tldw/deployment-mode", () => ({ isHostedTldwDeployment: () => false }))
vi.mock("@/services/tldw-server", () => ({
  LEGACY_SERVICE_PROMPT_DEFAULTS: {}, promptForRag: vi.fn(), getWebSearchPrompt: vi.fn()
}))
vi.mock("wxt/browser", () => ({ browser: { storage: { onChanged: { addListener: vi.fn(), removeListener: vi.fn() } } } }))
vi.mock("@/utils/safe-storage", () => ({ createSafeStorage: () => ({
  get: async () => null, set: async () => {}, remove: async () => {}, watch: () => {}, unwatch: () => {}
}), safeStorageSerde: { deserializer: (value: unknown) => value } }))

const token = (user: string, revision = "first") => `test.${btoa(JSON.stringify({ sub: user }))}.${revision}`
const config = (user = "alice", serverUrl = "https://keywords.test", revision = "first"): TldwConfig => ({
  serverUrl, authMode: "multi-user", authSource: "manual", accessToken: token(user, revision), refreshToken: `refresh-${user}-${revision}`
})
const deferred = <T,>() => {
  let resolve!: (value: T) => void
  const promise = new Promise<T>(yes => { resolve = yes })
  return { promise, resolve }
}
const changed = (kind: string) => {
  if (kind === "owner") boundary.config = config("bob")
  if (kind === "server") boundary.config = config("alice", "https://other.test")
  if (kind === "credentials") boundary.config = config("alice", "https://keywords.test", "second")
  if (kind === "ABA") {
    boundary.config = config("bob")
    window.dispatchEvent(new Event("tldw:auth-principal-changed"))
    boundary.config = config()
  }
  window.dispatchEvent(new Event(kind === "credentials" ? "tldw:auth-credentials-changed" : "tldw:auth-principal-changed"))
}
let service: typeof import("../note-keywords")
const operations = [
  { name: "list", run: () => service.getNoteKeywords(200), result: (keyword: string) => [keyword] },
  { name: "all keywords", run: () => service.getAllNoteKeywords(2), result: (keyword: string) => [keyword] },
  { name: "all stats", run: () => service.getAllNoteKeywordStats(2), result: (keyword: string) => [{ keyword, noteCount: 4 }] },
  { name: "search", run: () => service.searchNoteKeywords(" private ", 10), result: (keyword: string) => [keyword] }
]
const transitions = ["owner", "server", "credentials", "ABA"]

describe("real shared note keyword reads retain captured ownership", () => {
  beforeEach(async () => {
    vi.resetModules()
    vi.clearAllMocks()
    boundary.config = config()
    boundary.ensureConfig.mockImplementation(async () => boundary.config ? { ...boundary.config } : null)
    boundary.user.mockImplementation(async () => ({ id: JSON.parse(atob(boundary.config!.accessToken!.split(".")[1])).sub }))
    boundary.request.mockResolvedValue([{ keyword: "private-alice", note_count: 4 }])
    vi.stubGlobal("fetch", vi.fn(() => { throw new Error("Unexpected network operation") }))
    service = await import("../note-keywords")
  })
  afterEach(() => vi.unstubAllGlobals())

  for (const operation of operations) {
    it.each(transitions)(`${operation.name} does not reuse old results across %s`, async kind => {
      expect(await operation.run()).toEqual(operation.result("private-alice"))
      changed(kind)
      boundary.request.mockResolvedValue([{ keyword: "current-owner", note_count: 4 }])
      expect(await operation.run()).toEqual(operation.result("current-owner"))
      expect(boundary.request).toHaveBeenCalledTimes(2)
    })

    it.each(transitions)(`${operation.name} retires an in-flight old read across %s without sharing it`, async kind => {
      const gate = deferred<unknown>()
      boundary.request.mockImplementationOnce(() => gate.promise)
      const old = operation.run().then(value => ({ value }), error => ({ error }))
      await waitFor(() => expect(boundary.request).toHaveBeenCalledTimes(1))
      changed(kind)
      boundary.request.mockResolvedValue([{ keyword: "current-owner", note_count: 4 }])
      const fresh = operation.run()
      try {
        await waitFor(() => expect(boundary.request).toHaveBeenCalledTimes(2))
        expect(await fresh).toEqual(operation.result("current-owner"))
        expect(boundary.request.mock.calls[0][0].abortSignal).toBeInstanceOf(AbortSignal)
        expect(boundary.request.mock.calls[0][0].abortSignal.aborted).toBe(true)
      } finally {
        gate.resolve([{ keyword: "private-alice", note_count: 4 }])
      }
      expect(await old).toMatchObject({ error: { status: 412 } })
      expect(await operation.run()).toEqual(operation.result("current-owner"))
    })

    it(`${operation.name} detects a quiet server change before accepting the reply`, async () => {
      const gate = deferred<unknown>()
      boundary.request.mockImplementationOnce(() => gate.promise)
      const old = operation.run().then(value => ({ value }), error => ({ error }))
      await waitFor(() => expect(boundary.request).toHaveBeenCalledTimes(1))
      boundary.config = config("alice", "https://other.test")
      gate.resolve([{ keyword: "private-alice", note_count: 4 }])
      expect(await old).toMatchObject({ error: { status: 412 } })
    })

    it(`${operation.name} pins the verified owner, config and cancellation to the actual request`, async () => {
      await operation.run()
      const request = boundary.request.mock.calls[0][0]
      expect(request.method).toBe("GET")
      expect(request.configSnapshot).toEqual(config())
      expect(request.configSnapshot).not.toBe(boundary.config)
      expect(request.headers).toEqual({ "X-TLDW-Expected-User-ID": "alice" })
      expect(request.abortSignal).toBeInstanceOf(AbortSignal)
      expect(request.abortSignal.aborted).toBe(false)
      // Keyword search is not on the Service Prompt path allowlist.
      expect(request.servicePromptConfig).toBeUndefined()
      expect(fetch).not.toHaveBeenCalled()
    })

    it(`${operation.name} refuses an unresolved principal before dispatch`, async () => {
      boundary.user.mockResolvedValue(null)
      await expect(operation.run()).rejects.toMatchObject({ code: "service_prompt_scope_unresolved" })
      expect(boundary.request).not.toHaveBeenCalled()
    })

    it(`${operation.name} preserves request failures instead of treating them as keywords`, async () => {
      const failure = new Error("Keywords unavailable")
      boundary.request.mockRejectedValue(failure)
      await expect(operation.run()).rejects.toBe(failure)
    })
  }

  it.each(["owner", "server", "credentials", "ABA"])("stops pagination after %s retirement without dispatching a new page", async kind => {
    const gate = deferred<unknown>()
    boundary.request.mockImplementationOnce(() => gate.promise)
    const old = service.getAllNoteKeywordStats(2).then(value => ({ value }), error => ({ error }))
    await waitFor(() => expect(boundary.request).toHaveBeenCalledTimes(1))
    changed(kind)
    gate.resolve([{ keyword: "private-one" }, { keyword: "private-two" }])
    expect(await old).toMatchObject({ error: { status: 412 } })
    expect(boundary.request).toHaveBeenCalledTimes(1)
  })

  it("detects a quiet credential change before starting the next page", async () => {
    boundary.request.mockImplementationOnce(async () => {
      boundary.config = config("alice", "https://keywords.test", "second")
      return [{ keyword: "private-one" }, { keyword: "private-two" }]
    })
    await expect(service.getAllNoteKeywordStats(2)).rejects.toMatchObject({ status: 412 })
    expect(boundary.request).toHaveBeenCalledTimes(1)
  })

  it.each(["getNoteKeywords", "getAllNoteKeywords", "getAllNoteKeywordStats"] as const)("does not retain a %s cache within the same owner", async name => {
    await service[name](200)
    boundary.request.mockResolvedValue([{ keyword: "new-keyword", note_count: 4 }])
    expect(await service[name](200)).toEqual(name === "getAllNoteKeywordStats" ? [{ keyword: "new-keyword", noteCount: 4 }] : ["new-keyword"])
    expect(boundary.request).toHaveBeenCalledTimes(2)
  })

  it("preserves list/search normalization and their distinct query contracts", async () => {
    boundary.request.mockResolvedValue([" alpha ", { keyword_text: "beta" }, { text: "alpha" }, "", null, { keyword: "Beta" }])
    expect(await service.getNoteKeywords(7)).toEqual(["alpha", "beta", "Beta"])
    expect(boundary.request.mock.calls[0][0].path).toBe("/api/v1/notes/keywords/?limit=7")
    expect(await service.searchNoteKeywords("  query  ", 3)).toEqual(["alpha", "beta", "Beta"])
    expect(boundary.request.mock.calls[1][0].path).toBe("/api/v1/notes/keywords/search/?query=query&limit=3")
    expect(await service.searchNoteKeywords("  ")).toEqual([])
    expect(boundary.request).toHaveBeenCalledTimes(2)
  })

  it("preserves paginated counts, deduplication and the captured config on every page", async () => {
    boundary.request.mockResolvedValueOnce([{ keyword: " Alpha ", note_count: 1.9 }, { keyword_text: "beta", count: -2 }])
      .mockResolvedValueOnce([{ text: "alpha", count: 7 }, { keyword: "gamma", count: "bad" }]).mockResolvedValueOnce([])
    expect(await service.getAllNoteKeywordStats(2)).toEqual([
      { keyword: "Alpha", noteCount: 7 }, { keyword: "beta", noteCount: 0 }, { keyword: "gamma", noteCount: 0 }
    ])
    expect(boundary.request.mock.calls.map(([request]) => request.path)).toEqual([
      "/api/v1/notes/keywords/?limit=2&offset=0&include_note_counts=true",
      "/api/v1/notes/keywords/?limit=2&offset=2&include_note_counts=true",
      "/api/v1/notes/keywords/?limit=2&offset=4&include_note_counts=true"
    ])
    for (const [request] of boundary.request.mock.calls) {
      expect(request.configSnapshot).toEqual(config())
      expect(request.headers["X-TLDW-Expected-User-ID"]).toBe("alice")
    }
  })

  it("retains the 100-page ceiling", async () => {
    boundary.request.mockResolvedValue([{ keyword: "alpha", count: 2 }])
    expect(await service.getAllNoteKeywordStats(1)).toEqual([{ keyword: "alpha", noteCount: 2 }])
    expect(boundary.request).toHaveBeenCalledTimes(100)
    expect(boundary.request.mock.calls[99][0].path).toContain("offset=99")
  })

  it("pins and retires a single-user API-key read without retaining its old result", async () => {
    boundary.config = { serverUrl: "https://keywords.test", authMode: "single-user", apiKey: "key-one" }
    const gate = deferred<unknown>()
    boundary.request.mockImplementationOnce(() => gate.promise)
    const old = service.getNoteKeywords().then(value => ({ value }), error => ({ error }))
    await waitFor(() => expect(boundary.request).toHaveBeenCalledTimes(1))
    expect(boundary.request.mock.calls[0][0].configSnapshot).toEqual(boundary.config)
    boundary.config = { ...boundary.config, apiKey: "key-two" }
    window.dispatchEvent(new Event("tldw:auth-credentials-changed"))
    gate.resolve(["private-old-key"])
    expect(await old).toMatchObject({ error: { status: 412 } })
    boundary.request.mockResolvedValue(["current-key"])
    expect(await service.getNoteKeywords()).toEqual(["current-key"])
    expect(boundary.user).not.toHaveBeenCalled()
  })
})
