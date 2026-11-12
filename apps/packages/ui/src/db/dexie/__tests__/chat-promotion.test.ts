import { beforeEach, describe, expect, it, vi } from "vitest"
import type { MemoryDexie } from "@/components/Option/Playground/__tests__/harness/memory-dexie"

const memory = vi.hoisted(() => ({ db: null as unknown }))

vi.mock("@/db/dexie/schema", async () => {
  const { createMemoryDexie } = await import(
    "@/components/Option/Playground/__tests__/harness/memory-dexie"
  )
  memory.db = memory.db ?? createMemoryDexie()
  return { db: memory.db }
})

import { getChatPromotionBlocker } from "../chat-promotion"

const db = () => memory.db as MemoryDexie

const saveHistory = (history: Record<string, unknown>) =>
  db().chatHistories.put({
    title: "Chat",
    is_rag: false,
    createdAt: 1,
    message_source: "web-ui",
    ...history
  })

// CS-N3 (#3104): the Playground's legacy promotion never copies a chat that
// the history selection owns, or one already on the server.
describe("getChatPromotionBlocker", () => {
  beforeEach(() => {
    db().resetAll()
  })

  it("blocks a chat the history selection owns", async () => {
    await saveHistory({ id: "owned", local_owner_key: "local-history-v1:profile" })
    expect(await getChatPromotionBlocker("owned")).toBe("history_selection_owned")
  })

  it("blocks a chat that already mirrors a server chat", async () => {
    await saveHistory({ id: "mirror", server_chat_id: "server-1" })
    expect(await getChatPromotionBlocker("mirror")).toBe("server_linked")
    await saveHistory({ id: "server-copy", message_source: "server" })
    expect(await getChatPromotionBlocker("server-copy")).toBe("server_linked")
  })

  it("allows an unowned local chat and a draft with no local history yet", async () => {
    await saveHistory({ id: "draft" })
    expect(await getChatPromotionBlocker("draft")).toBeNull()
    expect(await getChatPromotionBlocker(null)).toBeNull()
    expect(await getChatPromotionBlocker("not-saved-yet")).toBeNull()
  })
})
