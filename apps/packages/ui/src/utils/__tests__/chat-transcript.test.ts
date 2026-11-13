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

  it("writes JSON with the export format, the conversation fields and its messages", () => {
    expect(JSON.parse(formatTranscriptJson(transcript))).toEqual({ format: "tldw-chat-export", version: 1, ...transcript })
  })

  it("says in the JSON when images were left out", () => {
    const json = JSON.parse(formatTranscriptJson({ ...transcript, imagesOmitted: true }))
    expect(json.images_omitted).toBe(true)
    expect(json).not.toHaveProperty("imagesOmitted")
    expect(JSON.parse(formatTranscriptJson(transcript))).not.toHaveProperty("images_omitted")
  })

  it("names a message of another role after its role instead of 'undefined'", () => {
    const markdown = formatTranscriptMessagesMarkdown([{ role: "tool", content: "42" }])
    expect(markdown).toBe("## Tool\n\n42")
  })

  it("names the file after the chat, without characters a file name can't hold", () => {
    expect(transcriptFileName("Plan: a/b?", "md", 42)).toBe("Plan-_a-b-_42.md")
    expect(transcriptFileName("", "json", 7)).toBe("chat_7.json")
  })
})

describe("lossless JSON and newest-path Markdown (CS-09 follow-up)", () => {
  const branched = {
    id: "chat-1",
    title: "Trip plan",
    messages: [
      { id: "u1", parentId: null, role: "user" as const, content: "Where first?", timestamp: "2026-11-01T09:00:00Z" },
      { id: "a1", parentId: "u1", role: "assistant" as const, content: "Porto.", timestamp: "2026-11-01T09:00:01Z", model: "model-a" },
      { id: "a1b", parentId: "u1", role: "assistant" as const, content: "Lisbon.", timestamp: "2026-11-01T09:00:02Z", model: "model-b" }
    ]
  }

  it("keeps every message, alternatives included, in the JSON export", () => {
    expect(JSON.parse(formatTranscriptJson(branched)).messages).toEqual(branched.messages)
  })

  it("writes the newest path in Markdown and says the alternatives are in the JSON export", () => {
    const markdown = formatTranscriptMarkdown(branched)
    expect(markdown).toContain("Lisbon.")
    expect(markdown).not.toContain("Porto.")
    const lines = markdown.split("\n").filter((line) => line.trim())
    expect(lines[0]).toBe("# Trip plan")
    expect(lines[1]).toMatch(/alternative replies.*omitted.*JSON export includes them/i)
  })

  it("adds no note when the chat has no alternative replies", () => {
    const linear = { ...branched, messages: branched.messages.slice(0, 2) }
    const markdown = formatTranscriptMarkdown(linear)
    expect(markdown).toContain("Porto.")
    expect(markdown).not.toMatch(/alternative replies/i)
  })
})

describe("Markdown order when a reply is listed before its question (review follow-up)", () => {
  it("keeps every turn of the newest path", () => {
    const markdown = formatTranscriptMarkdown({
      id: "chat-1",
      title: "Order",
      messages: [
        { id: "u1", parentId: null, role: "user", content: "Q1" },
        { id: "a1", parentId: "u1", role: "assistant", content: "A1" },
        { id: "a2", parentId: "u2", role: "assistant", content: "A2" },
        { id: "u2", parentId: "a1", role: "user", content: "Q2" }
      ]
    })
    const order = ["Q1", "A1", "Q2", "A2"].map((text) => markdown.indexOf(text))
    expect(order.every((at) => at >= 0)).toBe(true)
    expect([...order].sort((a, b) => a - b)).toEqual(order)
    expect(markdown).not.toMatch(/alternative replies/i)
  })
})
