import { describe, expect, it } from "vitest"
import {
  formatTranscriptJson,
  formatTranscriptMarkdown,
  formatTranscriptMessagesMarkdown,
  transcriptFileName
} from "../chat-transcript"

const transcript = {
  id: "chat-1",
  title: "Trip plan",
  createdAt: "2026-11-01T09:00:00.000Z",
  model: "gpt-test",
  messages: [
    { role: "user" as const, content: "Where first?", timestamp: "2026-11-01T09:00:00.000Z" },
    { role: "assistant" as const, content: "Lisbon.", speaker: "Navigator" },
    { role: "system" as const, content: "Be brief." }
  ]
}

describe("chat transcript formatting (CS-09)", () => {
  it("writes a Markdown transcript with a title, date, model and one section per message", () => {
    const markdown = formatTranscriptMarkdown(transcript)
    expect(markdown.startsWith("# Trip plan\n")).toBe(true)
    expect(markdown).toContain("**Model:** gpt-test")
    expect(markdown).toContain("## You\n")
    expect(markdown).toContain("Where first?")
    expect(markdown).toContain("## Navigator\n\nLisbon.")
    expect(markdown).toContain("## System\n\nBe brief.")
    expect(markdown.indexOf("Where first?")).toBeLessThan(markdown.indexOf("Lisbon."))
  })

  it("writes the messages alone for a copy, with images as Markdown images", () => {
    const markdown = formatTranscriptMessagesMarkdown([
      { role: "user", content: "Look", images: ["data:image/png;base64,AAA", ""] },
      { role: "assistant", content: "Seen" }
    ])
    expect(markdown).toBe("## You\n\nLook\n\n![Image](data:image/png;base64,AAA)\n\n## Assistant\n\nSeen")
  })

  it("writes JSON with the conversation fields and its messages", () => {
    expect(JSON.parse(formatTranscriptJson(transcript))).toEqual(transcript)
  })

  it("names the file after the chat, without characters a file name can't hold", () => {
    expect(transcriptFileName("Plan: a/b?", "md", 42)).toBe("Plan-_a-b-_42.md")
    expect(transcriptFileName("", "json", 7)).toBe("chat_7.json")
  })
})
