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

// Review fix: the model sees an attached note as a labelled, fenced block.
describe("attached note block", () => {
  it("labels the note and fences its content", () => {
    expect(appendPinnedContext("Hi", [note], {})).toBe("Hi\n\nAttached note: Rebase notes\n```\nUse rebase.\n```")
  })

  it("fences content that has its own code fences", () => {
    const fenced = { ...note, snippet: "Run:\n```sh\ngit rebase -i\n```" }
    const text = appendPinnedContext("", [fenced], {})
    expect(text.startsWith("Attached note: Rebase notes\n````\n")).toBe(true)
    expect(text.endsWith("\n````")).toBe(true)
  })

  it("says a shortened note was shortened, in the user's language", () => {
    const t = (key: string, fallback: string) => (key === "playground:noteMentions.shortened" ? "[Gekürzt]" : fallback)
    const text = appendPinnedContext("", [{ ...note, truncated: true }], { t })
    expect(text.endsWith("```\n[Gekürzt]")).toBe(true)
  })

  it("keeps other pins' format", () => {
    expect(appendPinnedContext("", [search], {})).toBe("**Doc**\n\nA result.")
  })
})
