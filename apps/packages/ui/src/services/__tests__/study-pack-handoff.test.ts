import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { buildStudyPackRoute, parseStudyPackIntentFromLocation, parseStudyPackIntentFromSearch, readStudyPackRoute } from "@/services/tldw/study-pack-handoff"
import { clearFlashcardsGenerateHandoffs, consumeFlashcardsGenerateHandoff, consumeStudyPackHandoff, createStudyPackHandoff, FLASHCARDS_GENERATE_HANDOFF_TTL_MS } from "@/services/tldw/flashcards-generate-handoff"

vi.mock("@plasmohq/storage", async () => import("../../../../../tldw-frontend/extension/shims/plasmo-storage"))
const owner = "verified-server-a-alice"
const intent = { title: "Alice private Biology", sourceItems: [{ sourceType: "note" as const, sourceId: "alice-private-note", sourceTitle: "Private source" }] }

describe("private study-pack handoff", () => {
  beforeEach(() => {
    window.localStorage.clear()
    let tail = Promise.resolve()
    vi.stubGlobal("navigator", Object.create(window.navigator, { locks: { value: {
      request: (_name: string, work: () => unknown) => { const next = tail.then(work); tail = next.then(() => undefined, () => undefined); return next }
    } } }))
  })
  afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals() })

  it.each(["note", "media", "message"] as const)("transfers current-owner %s sources once without private data in history", async sourceType => {
    const payload = { ...intent, sourceItems: [{ ...intent.sourceItems[0], sourceType }] }
    const token = await createStudyPackHandoff(payload, owner)
    const route = buildStudyPackRoute(token)
    expect(route).toMatch(/^\/flashcards\?tab=importExport&study_pack_handoff=[0-9a-f-]+$/)
    expect(decodeURIComponent(route)).not.toContain("Alice")
    expect(decodeURIComponent(route)).not.toContain("alice-private-note")
    expect(await consumeStudyPackHandoff(token, owner)).toEqual(payload)
    await expect(consumeStudyPackHandoff(token, owner)).rejects.toThrow(/missing|expired|consumed/i)
  })

  it.each(["verified-server-a-bob", "verified-server-b-alice"])("rejects a different authority (%s)", async other => {
    const token = await createStudyPackHandoff(intent, owner)
    await expect(consumeStudyPackHandoff(token, other)).rejects.toThrow(/account|server/i)
    await expect(consumeStudyPackHandoff(token, owner)).rejects.toThrow(/missing|expired|consumed/i)
  })

  it("does not reinterpret a Study Pack token as generation source text", async () => {
    const token = await createStudyPackHandoff(intent, owner)
    await expect(consumeFlashcardsGenerateHandoff(token, owner)).rejects.toThrow(/different.*action/i)
  })

  it("rejects expired Study Pack metadata", async () => {
    const token = await createStudyPackHandoff(intent, owner)
    vi.spyOn(Date, "now").mockReturnValue(Date.now() + FLASHCARDS_GENERATE_HANDOFF_TTL_MS + 1)
    await expect(consumeStudyPackHandoff(token, owner)).rejects.toThrow(/missing|expired|consumed/i)
  })

  it("clears Study Pack records through the existing logout cleanup", async () => {
    const token = await createStudyPackHandoff(intent, owner)
    window.localStorage.setItem("unrelated", "keep")
    await clearFlashcardsGenerateHandoffs()
    await expect(consumeStudyPackHandoff(token, owner)).rejects.toThrow(/missing|expired|consumed/i)
    expect(window.localStorage.getItem("unrelated")).toBe("keep")
  })

  it("rejects old unowned query and hash URLs instead of assigning their data to the current account", () => {
    const search = "?study_pack=1&study_pack_payload=" + encodeURIComponent(JSON.stringify(intent))
    expect(parseStudyPackIntentFromSearch(search)).toBeNull()
    expect(parseStudyPackIntentFromLocation({ hash: "#/flashcards" + search })).toBeNull()
    expect(readStudyPackRoute({ pathname: "/flashcards", search: search + "&tab=importExport", hash: "#section?study_pack_title=Private" })).toEqual({ token: null, legacy: true, cleanRoute: "/flashcards?tab=importExport#section" })
  })

  it("reads an opaque hash transfer while preserving unrelated navigation", async () => {
    const token = await createStudyPackHandoff(intent, owner)
    expect(readStudyPackRoute({ pathname: "/options.html", hash: "#" + buildStudyPackRoute(token) })).toEqual({ token, legacy: false, cleanRoute: "/options.html#/flashcards?tab=importExport" })
  })
})
