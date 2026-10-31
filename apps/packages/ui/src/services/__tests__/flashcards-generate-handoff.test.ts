import { describe, expect, it } from "vitest"
import { buildFlashcardsGenerateRoute, parseFlashcardsGenerateIntentFromLocation, parseFlashcardsGenerateIntentFromSearch, readFlashcardsGenerateRoute } from "@/services/tldw/flashcards-generate-handoff"

describe("Flashcards route privacy", () => {
  it("preserves benign direct entry", () => expect(buildFlashcardsGenerateRoute()).toBe("/flashcards?tab=importExport"))
  it("accepts an opaque identifier only", () => {
    const token = "33e35eb7-12ba-40d4-b3ed-8b90ac593401"
    expect(buildFlashcardsGenerateRoute(token)).toBe(`/flashcards?tab=importExport&generate_handoff=${token}`)
    expect(() => buildFlashcardsGenerateRoute("private text")).toThrow()
  })
  it.each(["media", "note", "message", "manual"])("rejects old %s plaintext links", sourceType => {
    const search = `?generate=1&generate_text=Private&generate_source_type=${sourceType}&generate_source_id=secret-id`
    expect(parseFlashcardsGenerateIntentFromSearch(search)).toBeNull()
    expect(parseFlashcardsGenerateIntentFromLocation({ hash: `#/flashcards${search}` })).toBeNull()
    expect(readFlashcardsGenerateRoute({ pathname: "/flashcards", search }).cleanRoute).not.toContain("Private")
  })
})
