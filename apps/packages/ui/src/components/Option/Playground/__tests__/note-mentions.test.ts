import { describe, expect, it } from "vitest"
import {
  NOTE_CONTENT_MAX_CHARS,
  detectNoteTrigger,
  noteIdFromPinnedResult,
  noteToPinnedResult,
  removeNoteTrigger
} from "../note-mentions"

// Decision [4]: web "@" and /note attach a note's content to the message.
describe("note mention triggers", () => {
  it("finds an @ query at the caret on the web", () => {
    expect(detectNoteTrigger("Compare @rebase", 15, { atEnabled: true })).toEqual({
      kind: "at", start: 8, end: 15, query: "rebase"
    })
    expect(detectNoteTrigger("@", 1, { atEnabled: true })).toEqual({ kind: "at", start: 0, end: 1, query: "" })
  })

  it("leaves @ alone where tab mentions own it, and in addresses", () => {
    expect(detectNoteTrigger("Compare @rebase", 15, { atEnabled: false })).toBeNull()
    expect(detectNoteTrigger("mail bob@example.com", 20, { atEnabled: true })).toBeNull()
    expect(detectNoteTrigger("@two words", 10, { atEnabled: true })).toBeNull()
  })

  it("treats a draft of /note and its text as a note search, everywhere", () => {
    expect(detectNoteTrigger("/note ", 6, { atEnabled: false })).toEqual({
      kind: "command", start: 0, end: 6, query: ""
    })
    expect(detectNoteTrigger("/note weekly review", 19, { atEnabled: false })).toMatchObject({
      kind: "command", query: "weekly review"
    })
    expect(detectNoteTrigger("/note", 5, { atEnabled: true })).toBeNull()
    expect(detectNoteTrigger("/notes x", 8, { atEnabled: true })).toBeNull()
  })

  it("removes the trigger text and keeps the rest of the draft", () => {
    const at = detectNoteTrigger("Compare @reb with this", 12, { atEnabled: true })!
    expect(removeNoteTrigger("Compare @reb with this", at)).toEqual({ text: "Compare with this", caret: 8 })
    const command = detectNoteTrigger("/note weekly", 12, { atEnabled: true })!
    expect(removeNoteTrigger("/note weekly", command)).toEqual({ text: "", caret: 0 })
  })
})

describe("a note as pinned context", () => {
  it("carries the note's content, title and id", () => {
    const result = noteToPinnedResult({ id: 7, title: "Rebase notes", content: "  Use git rebase -i.  " })
    expect(result).toEqual({ id: "note:7", type: "note", title: "Rebase notes", source: "Notes", snippet: "Use git rebase -i." })
    expect(noteIdFromPinnedResult(result)).toBe("7")
    expect(noteIdFromPinnedResult({ id: "rag-1", snippet: "x" })).toBeNull()
  })

  it("shortens a very long note and says so", () => {
    const result = noteToPinnedResult({ id: "a", title: "Long", content: "x".repeat(NOTE_CONTENT_MAX_CHARS + 50) })
    expect(result.snippet.length).toBeLessThan(NOTE_CONTENT_MAX_CHARS + 40)
    expect(result.snippet.endsWith("[Note shortened]")).toBe(true)
  })
})
