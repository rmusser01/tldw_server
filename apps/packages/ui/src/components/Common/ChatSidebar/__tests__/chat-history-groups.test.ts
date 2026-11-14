import { describe, expect, it } from "vitest"
import { getChatRecencyGroup, groupChatsByRecency } from "../chat-history-groups"

const NOW = new Date("2026-10-19T15:00:00")
const at = (iso: string) => new Date(iso).getTime()

describe("chat history recency groups (CS-08)", () => {
  it("puts chats from earlier today under Today", () => {
    expect(getChatRecencyGroup(at("2026-10-19T00:05:00"), NOW)).toBe("today")
  })

  it("puts the previous six days under This week", () => {
    expect(getChatRecencyGroup(at("2026-10-18T23:00:00"), NOW)).toBe("thisWeek")
    expect(getChatRecencyGroup(at("2026-10-13T00:00:00"), NOW)).toBe("thisWeek")
  })

  it("puts everything before that, or without a date, under Older", () => {
    expect(getChatRecencyGroup(at("2026-10-12T23:59:00"), NOW)).toBe("older")
    expect(getChatRecencyGroup(Number.NaN, NOW)).toBe("older")
    expect(getChatRecencyGroup(null, NOW)).toBe("older")
  })

  it("keeps the list's order inside each group and skips empty groups", () => {
    const groups = groupChatsByRecency(
      [
        { id: "a", updatedAtMs: at("2026-10-19T14:00:00") },
        { id: "b", updatedAtMs: at("2026-01-01T00:00:00") },
        { id: "c", updatedAtMs: at("2026-10-19T09:00:00") },
        { id: "d", updatedAtMs: null, createdAtMs: at("2025-12-01T00:00:00") }
      ],
      NOW
    )
    expect(groups.map(({ group, chats }) => [group, chats.map((chat) => chat.id)])).toEqual([
      ["today", ["a", "c"]],
      ["older", ["b", "d"]]
    ])
  })
})
