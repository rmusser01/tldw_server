import { describe, expect, it, vi } from "vitest"

vi.mock("@/services/tldw/TldwApiClient", () => ({ tldwClient: {} }))

import {
  buildAnswerNoteContent,
  buildAnswerNoteTitle,
  canSaveAnswerAsNote
} from "../chat-answer-note"

// XP-01: the plain-note fallback for answers with no saved server message.
const t = (_key: string, fallback: string, options?: Record<string, unknown>) =>
  fallback.replace(/\{\{(\w+)\}\}/g, (_match, name) => String(options?.[name] ?? ""))
const savedAt = new Date("2026-11-19T10:00:00Z")

describe("chat answer note", () => {
  it("titles the note from the question's first line, shortened", () => {
    expect(buildAnswerNoteTitle("How do I rebase?\nMore detail", savedAt, t)).toBe("How do I rebase?")
    const long = "x".repeat(80)
    const title = buildAnswerNoteTitle(long, savedAt, t)
    expect(title.length).toBeLessThanOrEqual(60)
    expect(title.endsWith("…")).toBe(true)
  })

  it("never cuts the title through a surrogate pair", () => {
    const title = buildAnswerNoteTitle("x".repeat(58) + "\u{1F600}" + "y".repeat(10), savedAt, t)
    expect(/[\uD800-\uDBFF]…$/.test(title)).toBe(false)
  })

  it("falls back to a dated title without a question", () => {
    expect(buildAnswerNoteTitle("  ", savedAt, t)).toMatch(/^Chat answer, /)
  })

  it("adds a one-line provenance footer", () => {
    const content = buildAnswerNoteContent("  The answer.  ", { temporaryChat: true, modelName: "gpt-4o", savedAt }, t)
    expect(content.startsWith("The answer.\n\n---\nSaved from a temporary chat (gpt-4o) on ")).toBe(true)
  })

  it("is offered for an assistant answer outside a workspace only", () => {
    const base = { hasNotes: true, isBot: true, text: "Answer", errored: false }
    expect(canSaveAnswerAsNote(base)).toBe(true)
    expect(canSaveAnswerAsNote({ ...base, scope: { type: "global" } })).toBe(true)
    expect(canSaveAnswerAsNote({ ...base, scope: { type: "workspace", workspaceId: "w" } })).toBe(false)
    expect(canSaveAnswerAsNote({ ...base, isBot: false })).toBe(false)
    expect(canSaveAnswerAsNote({ ...base, text: "  " })).toBe(false)
    expect(canSaveAnswerAsNote({ ...base, errored: true })).toBe(false)
    expect(canSaveAnswerAsNote({ ...base, hasNotes: false })).toBe(false)
  })
})
