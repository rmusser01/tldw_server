import { describe, expect, it } from "vitest"
import { CAPTURED_NOTE_KEYWORD } from "@/services/note-capture"
import {
  appendSourceUrlToCapturedNoteContent,
  buildSidepanelCapturedNotePayload
} from "../sidepanel-note-capture"

describe("sidepanel note capture payload helpers", () => {
  it("persists source URL through note content and adds the Inbox marker tag", () => {
    const payload = buildSidepanelCapturedNotePayload({
      title: "Selected quote",
      content: "Important selected text",
      sourceUrl: "https://example.com/story"
    })

    expect(payload.content).toBe(
      "Important selected text\n\nSource: https://example.com/story"
    )
    expect(payload.noteFields).toEqual({
      title: "Selected quote",
      keywords: [CAPTURED_NOTE_KEYWORD]
    })
    expect(payload.noteFields).not.toHaveProperty("source_url")
    expect(payload.noteFields).not.toHaveProperty("metadata")
  })

  it("does not duplicate the source URL when selected text already includes it", () => {
    expect(
      appendSourceUrlToCapturedNoteContent(
        "Important selected text\n\nSource: https://example.com/story",
        "https://example.com/story"
      )
    ).toBe("Important selected text\n\nSource: https://example.com/story")
  })
})
