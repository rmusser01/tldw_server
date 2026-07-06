import { describe, expect, it, vi } from "vitest"

vi.mock("@/db/dexie/chat", () => ({
  PageAssistDatabase: class {}
}))

import { formatToMessage } from "../helpers"

describe("formatToMessage character emotes", () => {
  it("promotes explicit mood metadata from local history", () => {
    const [message] = formatToMessage([
      {
        id: "assistant-1",
        history_id: "history-1",
        name: "Ashley",
        role: "assistant",
        content: "What now?",
        createdAt: 1000,
        metadataExtra: {
          mood_label: "smug",
          emote_events: [{ state: "smug", at_char: 0 }]
        }
      }
    ])

    expect(message.moodLabel).toBe("smug")
    expect(message.metadataExtra?.emote_events).toEqual([
      { state: "smug", at_char: 0 }
    ])
  })
})
