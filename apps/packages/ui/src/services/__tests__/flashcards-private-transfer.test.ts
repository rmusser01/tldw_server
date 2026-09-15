import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import {
  buildFlashcardsGenerateRoute,
  clearFlashcardsGenerateHandoffs,
  consumeFlashcardsGenerateHandoff,
  createFlashcardsGenerateHandoff,
  FLASHCARDS_GENERATE_HANDOFF_PREFIX,
  FLASHCARDS_GENERATE_HANDOFF_TTL_MS,
  readFlashcardsGenerateRoute,
  parseFlashcardsGenerateIntentFromSearch
} from "@/services/tldw/flashcards-generate-handoff"

// Exercise the actual WebUI storage implementation, including its memory fallback.
vi.mock("@plasmohq/storage", async () =>
  import("../../../../../tldw-frontend/extension/shims/plasmo-storage")
)

const authority = "verified-server-a-user-1"
const intent = {
  text: " \nPrivate unsaved note\n\t ",
  sourceType: "note" as const,
  sourceId: "private-note-id",
  sourceTitle: "Private note title",
  conversationId: "private-conversation",
  messageId: "private-message"
}

const deferred = <T,>() => {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((done) => { resolve = done })
  return { promise, resolve }
}

describe("private Flashcards handoff", () => {
  beforeEach(() => {
    window.localStorage.clear()
    let tail = Promise.resolve()
    vi.stubGlobal("navigator", Object.create(window.navigator, {
      locks: { value: {
        request: vi.fn((_name: string, work: () => unknown) => {
          const next = tail.then(work)
          tail = next.then(() => undefined, () => undefined)
          return next
        })
      } }
    }))
  })
  afterEach(() => {
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
  })

  it("rejects old plaintext URLs instead of assigning them to the current account", () => {
    expect(parseFlashcardsGenerateIntentFromSearch(
      "?generate=1&generate_text=Alice%20private%20note&generate_source_id=alice-note"
    )).toBeNull()
  })

  it("shares exact unsaved content once through storage and an opaque route", async () => {
    const token = await createFlashcardsGenerateHandoff(intent, authority)
    const route = buildFlashcardsGenerateRoute(token)
    expect(route).toMatch(/^\/flashcards\?tab=importExport&generate_handoff=[0-9a-f-]+$/)
    for (const value of Object.values(intent)) expect(route).not.toContain(encodeURIComponent(value))
    const stored = window.localStorage.getItem(`${FLASHCARDS_GENERATE_HANDOFF_PREFIX}${token}`)
    expect(stored).toContain("Private unsaved note")
    expect(await consumeFlashcardsGenerateHandoff(token, authority)).toEqual(intent)
    await expect(consumeFlashcardsGenerateHandoff(token, authority)).rejects.toThrow(/expired|consumed|missing/i)
    expect(window.localStorage.getItem(`${FLASHCARDS_GENERATE_HANDOFF_PREFIX}${token}`)).toBeNull()
  })

  it("permits only one of two independent consumers to claim a token", async () => {
    const token = await createFlashcardsGenerateHandoff(intent, authority)
    const results = await Promise.allSettled([
      consumeFlashcardsGenerateHandoff(token, authority),
      consumeFlashcardsGenerateHandoff(token, authority)
    ])
    expect(results.filter(result => result.status === "fulfilled")).toHaveLength(1)
    expect(results.filter(result => result.status === "rejected")).toHaveLength(1)
    expect(navigator.locks.request).toHaveBeenCalled()
  })

  it("rejects another verified authority and removes its stale payload", async () => {
    const token = await createFlashcardsGenerateHandoff(intent, authority)
    await expect(consumeFlashcardsGenerateHandoff(token, "verified-server-a-user-2")).rejects.toThrow(/account|server/i)
    expect(window.localStorage.getItem(`${FLASHCARDS_GENERATE_HANDOFF_PREFIX}${token}`)).toBeNull()
  })

  it("leaves the token unconsumed when target authentication is unresolved", async () => {
    const token = await createFlashcardsGenerateHandoff(intent, authority)
    await expect(consumeFlashcardsGenerateHandoff(token, "")).rejects.toThrow(/sign in|authority/i)
    expect(await consumeFlashcardsGenerateHandoff(token, authority)).toEqual(intent)
  })

  it("expires a transfer without returning its content", async () => {
    const token = await createFlashcardsGenerateHandoff(intent, authority)
    vi.spyOn(Date, "now").mockReturnValue(Date.now() + FLASHCARDS_GENERATE_HANDOFF_TTL_MS + 1)
    await expect(consumeFlashcardsGenerateHandoff(token, authority)).rejects.toThrow(/expired|consumed|missing/i)
    expect(window.localStorage.getItem(`${FLASHCARDS_GENERATE_HANDOFF_PREFIX}${token}`)).toBeNull()
  })

  it("preserves whitespace and reports the bounded prefix of a long source", async () => {
    const text = " \n" + "a".repeat(12_000)
    const token = await createFlashcardsGenerateHandoff({ ...intent, text }, authority)
    expect(await consumeFlashcardsGenerateHandoff(token, authority)).toEqual({
      ...intent, text: text.slice(0, 12_000), truncated: true
    })
  })

  it("rejects inaccessible shared storage even when the shim can use memory", async () => {
    vi.spyOn(window, "localStorage", "get").mockImplementation(() => { throw new Error("Blocked") })
    await expect(createFlashcardsGenerateHandoff(intent, authority)).rejects.toThrow(/storage/i)
  })

  it("rejects unsupported atomic claims rather than using a read/remove race", async () => {
    vi.stubGlobal("navigator", Object.create(window.navigator, { locks: { value: undefined } }))
    await expect(createFlashcardsGenerateHandoff(intent, authority)).rejects.toThrow(/browser|secure|transfer/i)
    expect(window.localStorage.length).toBe(0)
  })

  it("does not write after authority changes while waiting for a lock", async () => {
    const blocked = deferred<void>()
    const hold = navigator.locks.request("test", () => blocked.promise)
    const controller = new AbortController()
    const saving = createFlashcardsGenerateHandoff(intent, authority, controller.signal)
    controller.abort()
    blocked.resolve()
    await hold
    await expect(saving).rejects.toMatchObject({ name: "AbortError" })
    expect(window.localStorage.length).toBe(0)
  })

  it("clears only private generation handoffs on a principal boundary", async () => {
    await createFlashcardsGenerateHandoff(intent, authority)
    window.localStorage.setItem("unrelated-handoff", "keep")
    await clearFlashcardsGenerateHandoffs()
    expect(window.localStorage.length).toBe(1)
    expect(window.localStorage.getItem("unrelated-handoff")).toBe("keep")
  })

  it("cleans old private query and hash fields while preserving benign navigation", () => {
    expect(readFlashcardsGenerateRoute({
      pathname: "/flashcards",
      search: "?tab=importExport&generate_text=Secret&generate_source_id=id&keep=yes",
      hash: "#section?generate_source_title=Private"
    })).toMatchObject({
      legacy: true,
      token: null,
      cleanRoute: "/flashcards?tab=importExport&keep=yes#section"
    })
  })
})
