import { describe, expect, it } from "vitest"
import { appendPinnedContext, sendablePins, withoutNotePins } from "../note-pins"

// Decision [4] review: an attached note is an explicit attachment of one message.
const note = { id: "note:7", type: "note", title: "Rebase notes", snippet: "Use rebase." }
const search = { id: "rag-1", title: "Doc", snippet: "A result." }

describe("note pins", () => {
  it("always sends attached notes; other pins only without file retrieval", () => {
    expect(sendablePins([note, search], {})).toEqual([note, search])
    expect(sendablePins([note, search], { fileRetrievalEnabled: true })).toEqual([note])
    expect(sendablePins([note, search], { ignoreOtherPins: true })).toEqual([note])
  })

  it("appends what is sent and leaves a plain message alone", () => {
    expect(appendPinnedContext("Hi", [search], { fileRetrievalEnabled: true })).toBe("Hi")
    expect(appendPinnedContext("Hi", [note], { fileRetrievalEnabled: true })).toContain("Use rebase.")
  })

  it("drops only the notes for a new chat", () => {
    expect(withoutNotePins([note, search])).toEqual([search])
  })
})
