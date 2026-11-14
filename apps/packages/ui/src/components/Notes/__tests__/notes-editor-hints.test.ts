import { describe, expect, it } from "vitest"
import { notesEditorSupportHint } from "../notes-editor-hints"

const t = (_key: string, options?: Record<string, unknown>) =>
  String(options?.defaultValue ?? _key)

describe("notesEditorSupportHint (NO-04)", () => {
  it("tells a writable blank draft how to start", () => {
    expect(notesEditorSupportHint(t, "markdown", true)).toMatch(/^Start typing to create a note/)
  })

  it("keeps a read-only editor to the neutral format hint", () => {
    // The pane is read-only both when the server has no notes support and
    // while a note is still loading, so the hint must not name a cause.
    const hint = notesEditorSupportHint(t, "markdown", true, true)
    expect(hint).toBe("Markdown + LaTeX supported")
  })

  it("shows the format hint once a note is open", () => {
    expect(notesEditorSupportHint(t, "wysiwyg", false)).toBe(
      "WYSIWYG mode keeps markdown structure while you edit."
    )
  })
})
