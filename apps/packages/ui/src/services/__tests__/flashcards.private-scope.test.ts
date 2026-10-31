import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
const boundary = vi.hoisted(() => ({ get: vi.fn(), fetch: vi.fn() }))
vi.mock("wxt/browser", () => ({ browser: { runtime: { id: null } } }))
vi.mock("@/utils/safe-storage", () => ({ createSafeStorage: () => ({ get: boundary.get, set: vi.fn(), remove: vi.fn() }) }))
vi.mock("@/services/tldw/runtime-auth-override", () => ({ getRuntimeSingleUserApiKeyOverride: () => null, isCookieSessionConfigInvalidated: () => false }))
import { createDeck, createFlashcard, generateFlashcards, listDecks, createFlashcardsBulk, getFlashcard, deleteFlashcard } from "../flashcards"
import { uploadFlashcardAsset } from "../flashcard-assets"
import { mediaMethods } from "../tldw/domains/media"
import { isServicePromptRequestPath } from "../tldw/service-prompt-scope-error"

const config = (user = 1, serverUrl = "https://source.test") => ({ serverUrl, authMode: "multi-user", accessToken: `test.${btoa(JSON.stringify({ sub: String(user) }))}.signature` })
const requestScope = { config: { serverUrl: "https://source.test", authMode: "multi-user" as const }, userId: 1 }
const cardUuid = "bb4da9d8-1d8c-4ac0-960b-7ef644b37865"
const asset = { name: "diagram.png", type: "image/png", arrayBuffer: async () => new Uint8Array([1, 2]).buffer } as File
const operations = [
  ["decks list", () => listDecks({ workspace_id: "workspace", include_workspace_items: true }, { requestScope })],
  ["bulk", () => createFlashcardsBulk([{ front: "Occluded image", back: "Label", deck_id: 7 }], { requestScope })],
  ["asset", () => uploadFlashcardAsset(asset, { requestScope })],
  ["undo read", () => getFlashcard(cardUuid, { requestScope })],
  ["undo delete", () => deleteFlashcard(cardUuid, 3, { requestScope })],
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
  it("preserves scoped bulk payload and versioned Undo query", async () => {
    const cards = [{ front: "![Question](flashcard-asset://q)", back: "![Answer](flashcard-asset://a)", deck_id: 7 }]
    await createFlashcardsBulk(cards, { requestScope })
    expect(JSON.parse(boundary.fetch.mock.calls[0][1].body)).toEqual(cards)
    await deleteFlashcard(cardUuid, 3, { requestScope })
    const [url, init] = boundary.fetch.mock.calls[1]
    expect(new URL(url).searchParams.get("expected_version")).toBe("3")
    expect(init.method).toBe("DELETE")
  })
  it("keeps scoped deck visibility in the query and scope out of its body", async () => {
    await listDecks({ workspace_id: "workspace", include_workspace_items: true }, { requestScope })
    const [url, init] = boundary.fetch.mock.calls[0]
    expect(Object.fromEntries(new URL(url).searchParams)).toEqual({ workspace_id: "workspace", include_workspace_items: "true" })
    expect(init.body).toBeUndefined()
  })
  it("leaves legacy bulk and Undo callers unscoped", async () => {
    await createFlashcardsBulk([{ front: "Q", back: "A" }])
    await getFlashcard(cardUuid)
    await deleteFlashcard(cardUuid, 3)
    expect(boundary.fetch).toHaveBeenCalledTimes(3)
    for (const [, init] of boundary.fetch.mock.calls) expect(new Headers(init.headers).has("X-TLDW-Expected-User-ID")).toBe(false)
  })
  it("forwards cancellation to Fetch for scoped reads and mutations", async () => {
    boundary.fetch.mockImplementation(async (_url, init) => {
      init.signal.throwIfAborted()
      return new Response("{}", { status: 200 })
    })
    const controller = new AbortController()
    controller.abort()
    const options = { requestScope, signal: controller.signal }
    for (const run of [() => listDecks({}, options), () => createFlashcardsBulk([{ front: "Q", back: "A" }], options), () => getFlashcard(cardUuid, options), () => deleteFlashcard(cardUuid, 3, options)]) {
      await expect(run()).rejects.toMatchObject({ name: "AbortError" })
    }
    expect(boundary.fetch).toHaveBeenCalledTimes(4)
    for (const [, init] of boundary.fetch.mock.calls) expect(init.signal.aborted).toBe(true)
  })
  it.each(["/api/v1/flashcards/bulk/extra", "/api/v1/flashcards/decks/2", "/api/v1/flashcards//generate", "/api/v1/flashcards/../generate"])("does not expand the scoped POST contract to %s", path => {
    expect(isServicePromptRequestPath(path, "POST")).toBe(false)
  })
  it.each(["GET", "DELETE"])("allows only canonical card-resource %s for Undo", method => {
    expect(isServicePromptRequestPath('/api/v1/flashcards/' + cardUuid, method)).toBe(true)
    for (const path of ['/api/v1/flashcards/not-a-uuid', '/api/v1/flashcards/' + cardUuid + '/assistant', '/api/v1/flashcards/%62' + cardUuid.slice(1), '/api/v1/flashcards/' + cardUuid + '/']) {
      expect(isServicePromptRequestPath(path, method)).toBe(false)
    }
  })
  it("does not permit arbitrary mutations on the new scoped routes", () => {
    for (const path of ["/api/v1/flashcards/decks", "/api/v1/flashcards/assets", "/api/v1/flashcards/bulk", '/api/v1/flashcards/' + cardUuid]) {
      for (const method of ["PUT", "PATCH"]) expect(isServicePromptRequestPath(path, method)).toBe(false)
    }
  })
  it.each(["/api/v1/media/search", "/api/v1/media/42/versions", "/api/v1/media/%34%32", "/api/v1/media/42/"])("does not expand source GET to %s", path => {
    expect(isServicePromptRequestPath(path, "GET")).toBe(false)
  })
})
