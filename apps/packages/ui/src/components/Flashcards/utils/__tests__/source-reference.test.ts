import { describe, expect, it } from "vitest"
import { getFlashcardSourceMeta } from "../source-reference"

describe("Flashcard saved source availability", () => {
  it.each([null, undefined, "", "  "])("does not invent a Chat destination without conversation identity %s", (conversation_id) => {
    expect(getFlashcardSourceMeta({ source_ref_type: "message", source_ref_id: "message-1", conversation_id })).toMatchObject({ href: null, unavailable: true })
  })
  it("does not present a nonnumeric Media reference as a supported destination", () => {
    expect(getFlashcardSourceMeta({ source_ref_type: "media", source_ref_id: "legacy-invalid" })).toMatchObject({ href: null, unavailable: true })
  })
  it("encodes the canonical conversation without adding an unsupported message focus parameter", () => {
    const source = getFlashcardSourceMeta({ source_ref_type: "message", source_ref_id: "message-1", conversation_id: "conversation/one" })!
    const url = new URL(source.href!, "https://client.test")
    expect([...url.searchParams.values()]).toEqual(["conversation/one"])
    expect(source.label).toBe("Conversation for message #message-1")
  })
})
