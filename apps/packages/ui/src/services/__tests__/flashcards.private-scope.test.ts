import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
const boundary = vi.hoisted(() => ({ get: vi.fn(), fetch: vi.fn() }))
vi.mock("wxt/browser", () => ({ browser: { runtime: { id: null } } }))
vi.mock("@/utils/safe-storage", () => ({ createSafeStorage: () => ({ get: boundary.get, set: vi.fn(), remove: vi.fn() }) }))
vi.mock("@/services/tldw/runtime-auth-override", () => ({ getRuntimeSingleUserApiKeyOverride: () => null, isCookieSessionConfigInvalidated: () => false }))
import { createDeck, createFlashcard, generateFlashcards } from "../flashcards"
import { mediaMethods } from "../tldw/domains/media"
import { isServicePromptRequestPath } from "../tldw/service-prompt-scope-error"

const config = (user = 1, serverUrl = "https://source.test") => ({ serverUrl, authMode: "multi-user", accessToken: `test.${btoa(JSON.stringify({ sub: String(user) }))}.signature` })
const requestScope = { config: { serverUrl: "https://source.test", authMode: "multi-user" as const }, userId: 1 }
const operations = [
  ["generate", () => generateFlashcards({ text: "Private source" }, { requestScope })],
  ["deck", () => createDeck({ name: "Private deck" }, { requestScope })],
  ["card", () => createFlashcard({ front: "Private front", back: "Private back" }, { requestScope })],
  ["source", () => mediaMethods.getMediaDetails(42, { requestScope })]
] as const

describe("private Flashcards real outbound scope", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    boundary.get.mockImplementation(async (key: string) => key === "tldwConfig" ? config() : null)
    boundary.fetch.mockImplementation(async () => new Response("{}", { status: 200, headers: { "Content-Type": "application/json" } }))
    vi.stubGlobal("fetch", boundary.fetch)
  })
  afterEach(() => vi.unstubAllGlobals())
  it.each(operations)("sends %s with the captured account and target", async (_name, run) => {
    await run()
    expect(boundary.fetch).toHaveBeenCalledTimes(1)
    const [url, init] = boundary.fetch.mock.calls[0]
    expect(new URL(url).origin).toBe("https://source.test")
    expect(new Headers(init.headers).get("X-TLDW-Expected-User-ID")).toBe("1")
  })
  it.each(operations)("blocks %s before dispatch after an account change", async (_name, run) => {
    boundary.get.mockImplementation(async (key: string) => key === "tldwConfig" ? config(2) : null)
    await expect(run()).rejects.toMatchObject({ status: 412 })
    expect(boundary.fetch).not.toHaveBeenCalled()
  })
  it.each(operations)("blocks %s before dispatch after a server change", async (_name, run) => {
    boundary.get.mockImplementation(async (key: string) => key === "tldwConfig" ? config(1, "https://other.test") : null)
    await expect(run()).rejects.toMatchObject({ status: 412 })
    expect(boundary.fetch).not.toHaveBeenCalled()
  })
  it.each(["/api/v1/flashcards/bulk", "/api/v1/flashcards/decks/2", "/api/v1/flashcards//generate", "/api/v1/flashcards/../generate"])("does not expand the scoped POST contract to %s", path => {
    expect(isServicePromptRequestPath(path, "POST")).toBe(false)
  })
  it.each(["/api/v1/media/search", "/api/v1/media/42/versions", "/api/v1/media/%34%32", "/api/v1/media/42/"])("does not expand source GET to %s", path => {
    expect(isServicePromptRequestPath(path, "GET")).toBe(false)
  })
})
