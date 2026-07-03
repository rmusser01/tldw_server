import { beforeEach, describe, expect, it, vi } from "vitest"

// ── In-memory Dexie message store shared with the mocked PageAssistDatabase ──
// TASK-12104: in a greeting-led character chat the greeting seed sits at UI
// index 0 but is never written to Dexie, so the UI list [greeting, user,
// assistant] and the Dexie list [user, assistant] are off by one. These tests
// prove the id-addressed helpers target the correct stored row regardless of
// that offset, and contrast them with the buggy index-addressed helpers.

const { store } = vi.hoisted(() => {
  const store = { messages: [] as any[] }
  return { store }
})

vi.mock("@/db/dexie/chat", () => {
  class PageAssistDatabase {
    async getChatHistory(history_id: string) {
      return store.messages
        .filter((m) => m.history_id === history_id)
        .map((m) => ({ ...m }))
    }
    async removeMessage(_history_id: string, message_id: string) {
      const idx = store.messages.findIndex((m) => m.id === message_id)
      if (idx >= 0) store.messages.splice(idx, 1)
    }
    async updateMessage(_history_id: string, message_id: string, content: string) {
      const target = store.messages.find((m) => m.id === message_id)
      if (target) target.content = content
    }
  }
  return { PageAssistDatabase }
})

import {
  removeMessageById,
  updateMessageById,
  deleteChatAfterMessageId,
  removeMessageByIndex,
  updateMessageByIndex
} from "../helpers"

const HISTORY_ID = "history-1"

// Dexie holds only the persisted user + assistant rows (no greeting).
const seedDexie = () => {
  store.messages = [
    {
      id: "user-1",
      history_id: HISTORY_ID,
      role: "user",
      content: "hello",
      createdAt: 1000
    },
    {
      id: "assistant-1",
      history_id: HISTORY_ID,
      role: "assistant",
      content: "hi there",
      createdAt: 2000
    }
  ]
}

// The UI list carries the non-persisted greeting seed at index 0, so UI indexes
// are offset by one relative to Dexie.
const uiMessages = [
  { id: "greeting-1", role: "assistant", message: "Greetings, traveler." },
  { id: "user-1", role: "user", message: "hello" },
  { id: "assistant-1", role: "assistant", message: "hi there" }
]

describe("TASK-12104 message targeting by stable id", () => {
  beforeEach(() => {
    seedDexie()
  })

  it("deleting the user bubble (UI index 1) removes the user row and preserves the assistant row", async () => {
    // deleteMessage resolves the target's stable id from the UI list.
    const targetId = uiMessages[1].id

    await removeMessageById(HISTORY_ID, targetId)

    const remaining = store.messages.map((m) => m.id)
    expect(remaining).toContain("assistant-1")
    expect(remaining).not.toContain("user-1")
  })

  it("deleting the greeting bubble (UI index 0) does not touch any Dexie row", async () => {
    const targetId = uiMessages[0].id // greeting is not persisted

    await removeMessageById(HISTORY_ID, targetId)

    expect(store.messages.map((m) => m.id)).toEqual(["user-1", "assistant-1"])
  })

  it("editing the user bubble updates the user row, not the assistant row", async () => {
    const targetId = uiMessages[1].id

    await updateMessageById(HISTORY_ID, targetId, "hello (edited)")

    const user = store.messages.find((m) => m.id === "user-1")
    const assistant = store.messages.find((m) => m.id === "assistant-1")
    expect(user?.content).toBe("hello (edited)")
    expect(assistant?.content).toBe("hi there")
  })

  it("deleteChatAfterMessageId removes rows after the edited user row (regenerate flow)", async () => {
    const targetId = uiMessages[1].id

    await deleteChatAfterMessageId(HISTORY_ID, targetId)

    // Assistant row (which followed the user row) is cleared for regeneration.
    expect(store.messages.map((m) => m.id)).toEqual(["user-1"])
  })

  it("regression: the old index-addressed helpers corrupt the wrong row under a greeting offset", async () => {
    // Passing the UI index (1) into the index helper hits Dexie[1] = assistant.
    await removeMessageByIndex(HISTORY_ID, 1)
    expect(store.messages.map((m) => m.id)).toEqual(["user-1"]) // assistant wrongly removed

    seedDexie()
    await updateMessageByIndex(HISTORY_ID, 1, "hello (edited)")
    const assistant = store.messages.find((m) => m.id === "assistant-1")
    expect(assistant?.content).toBe("hello (edited)") // assistant wrongly overwritten
  })
})
