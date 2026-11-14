import { describe, expect, it } from "vitest"
import { suggestNoteKeywords } from "../notes-keyword-suggestions"

// NL-17: "Suggest tags" offers fewer, better tags and pre-checks only the ones
// that already exist as tags.

const READING_GROUP_NOTE = [
  "First paragraph: the key idea is that reading groups need a shared goal.",
  "Second paragraph: other ideas include rotating the facilitator.",
  "The key point of the first session was the reading list."
].join(" ")

describe("suggestNoteKeywords (NL-17)", () => {
  it("skips discourse words and words that appear only once", () => {
    const { suggestions } = suggestNoteKeywords(READING_GROUP_NOTE, [], [])

    expect(suggestions).toContain("reading")
    for (const noise of ["first", "key", "paragraph", "second", "point", "goal", "facilitator"]) {
      expect(suggestions).not.toContain(noise)
    }
  })

  it("merges a plural into its singular", () => {
    const { suggestions } = suggestNoteKeywords(
      "One idea about groups. Another group idea. Ideas for the group.",
      [],
      []
    )

    expect(suggestions).toContain("group")
    expect(suggestions).not.toContain("groups")
    expect(suggestions).not.toContain("ideas")
  })

  it("offers existing tags first, in their own spelling, even when they appear once", () => {
    const { suggestions } = suggestNoteKeywords(
      "Quantum notes on decoherence. Quantum experiments repeat entanglement checks.",
      [],
      ["Entanglement", "physics"]
    )

    expect(suggestions[0]).toBe("Entanglement")
    expect(suggestions).toContain("quantum")
    expect(suggestions).not.toContain("decoherence")
    expect(suggestions).not.toContain("physics")
  })

  it("pre-checks only suggestions that are already tags", () => {
    const { suggestions, preselected } = suggestNoteKeywords(
      "Quantum notes on decoherence. Quantum experiments repeat entanglement checks.",
      [],
      ["entanglement"]
    )

    expect(suggestions).toEqual(["entanglement", "quantum"])
    expect(preselected).toEqual(["entanglement"])
  })

  it("pre-checks nothing when no suggestion is an existing tag", () => {
    const { preselected } = suggestNoteKeywords("Quantum quantum entanglement entanglement", [], [])

    expect(preselected).toEqual([])
  })

  it("leaves out tags the note already has", () => {
    const { suggestions } = suggestNoteKeywords(
      "Quantum quantum entanglement entanglement",
      ["Quantum"],
      []
    )

    expect(suggestions).toEqual(["entanglement"])
  })
})
