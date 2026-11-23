import { beforeEach, describe, expect, it, vi } from "vitest"

const store = vi.hoisted(() => ({ history: null as unknown, rows: [] as unknown[] }))
const download = vi.hoisted(() => ({ transcript: vi.fn() }))
vi.mock("@/db/dexie/schema", () => ({
  db: {
    chatHistories: { get: async () => store.history },
    messages: { where: () => ({ equals: () => ({ toArray: async () => store.rows }) }) }
  }
}))
vi.mock("@/utils/chat-transcript", async (original) => ({
  ...(await original<typeof import("@/utils/chat-transcript")>()),
  downloadTranscript: download.transcript
}))

import { exportToJSON } from "../conversation-export"

describe("device chat export (CS-09 follow-up)", () => {
  beforeEach(() => {
    download.transcript.mockReset()
    store.history = { id: "h1", title: "Local plan", createdAt: Date.parse("2026-11-01T09:00:00Z") }
    // Stored rows come back in key order, not in the order they were written.
    store.rows = [
      { id: "a1b", history_id: "h1", name: "model-b", role: "assistant", content: "Lisbon.",
        createdAt: Date.parse("2026-11-01T09:00:02Z"), parent_message_id: "u1", modelId: "model-b" },
      { id: "u1", history_id: "h1", name: "You", role: "user", content: "Where first?",
        createdAt: Date.parse("2026-11-01T09:00:00Z"), parent_message_id: null },
      { id: "a1", history_id: "h1", name: "model-a", role: "assistant", content: "Porto.",
        createdAt: Date.parse("2026-11-01T09:00:01Z"), parent_message_id: "u1" }
    ]
  })

  it("exports every saved message with its id and parent, in creation order", async () => {
    expect(await exportToJSON("h1")).toBe(true)
    const [transcript, format] = download.transcript.mock.calls[0]
    expect(format).toBe("json")
    expect(transcript.messages).toEqual([
      { id: "u1", parentId: null, role: "user", content: "Where first?", timestamp: "2026-11-01T09:00:00.000Z" },
      { id: "a1", parentId: "u1", role: "assistant", content: "Porto.", timestamp: "2026-11-01T09:00:01.000Z", model: "model-a" },
      { id: "a1b", parentId: "u1", role: "assistant", content: "Lisbon.", timestamp: "2026-11-01T09:00:02.000Z", model: "model-b" }
    ])
  })
})
