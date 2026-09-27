import { describe, expect, it } from "vitest"
import { buddyConversationLabels } from "../buddy-conversation-labels"

describe("Buddy conversation labels", () => {
  it("treats nullable legacy timestamps as missing instead of January 1970", () => {
    const labels = buddyConversationLabels([
      { id: "one", title: "Legacy", created_at: null },
      { id: "two", title: "Legacy" }
    ])
    expect(labels.get("one")).toBe("one — Legacy")
    expect(labels.get("two")).toBe("two — Legacy")
  })
  it("leaves unique titles unchanged and localizes duplicate creation times", () => {
    const conversations = [
      { id: "one", title: "Research", created_at: "2026-09-09T05:10:00Z" },
      { id: "two", title: "Research", created_at: "2026-09-09T05:12:00Z" },
      { id: "three", title: "Other", created_at: "" }
    ]
    const english = buddyConversationLabels(conversations, "en-US")
    const french = buddyConversationLabels(conversations, "fr-FR")
    expect(english.get("three")).toBe("Other")
    expect(english.get("one")).not.toBe(english.get("two"))
    expect(english.get("one")).not.toBe(french.get("one"))
    expect(english.get("one")).toMatch(/ — Research$/)
  })

  it("uses stable IDs for missing, invalid or identical timestamps, including prefix collisions", () => {
    const conversations = [
      { id: "abcdefgh-one", title: "Research", created_at: "" },
      { id: "abcdefgh-two", title: "Research", created_at: "invalid" },
      { id: "third", title: "Research", created_at: "2026-09-09T05:10:00Z" },
      { id: "fourth", title: "Research", created_at: "2026-09-09T05:10:00Z" }
    ]
    const labels = buddyConversationLabels(conversations, "en-US")
    const reordered = buddyConversationLabels(
      [...conversations].reverse(),
      "en-US"
    )
    expect(new Set(labels.values()).size).toBe(4)
    expect(labels.get("abcdefgh-one")).toBe("abcdefgh-one — Research")
    expect(labels.get("abcdefgh-two")).toBe("abcdefgh-two — Research")
    expect(labels.get("third")).toContain("third — Research")
    for (const [id, label] of labels) expect(reordered.get(id)).toBe(label)
  })
})
