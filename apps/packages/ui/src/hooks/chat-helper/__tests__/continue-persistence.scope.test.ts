import { beforeEach, describe, expect, it, vi } from "vitest"
import type { MemoryDexie } from "@/components/Option/Playground/__tests__/harness/memory-dexie"

/**
 * TASK-15626: continuing a saved local chat rewrites its last assistant row
 * inside runChatPersistenceTransaction. updateMessage opens
 * withLocalMessageMutation's nested transaction, which also names
 * userSettings, so every table it names must be in the enclosing list.
 * Real helpers, real transactions, in-memory Dexie that enforces Dexie's
 * parent-scope rule.
 */
const memory = vi.hoisted(() => ({ db: null as unknown }))

vi.mock("@/db/dexie/schema", async () => {
  const { createMemoryDexie } = await import(
    "@/components/Option/Playground/__tests__/harness/memory-dexie"
  )
  memory.db = createMemoryDexie()
  return { db: memory.db }
})
vi.mock("@/services/recipe-persistence-uncertainty", () => ({
  resolveRecipePersistenceOwnerView: async () => ({ ownerId: "local-owner" }),
  clearRecipePersistenceScoped: vi.fn()
}))
vi.mock("@/services/title", () => ({ generateTitle: vi.fn() }))
vi.mock("@/utils/update-page-title", () => ({ updatePageTitle: vi.fn() }))

import { saveMessageOnError, saveMessageOnSuccess } from "../index"
import { pageAssistDatabase } from "@/db/dexie/chat"

const db = () => memory.db as MemoryDexie

const seedLocalChat = async () => {
  await db().userSettings.put({ id: "main", user_id: "p1", history_profile_id: "p1" })
  await db().chatHistories.put({
    id: "h1",
    title: "Saved chat",
    is_rag: false,
    createdAt: 1,
    message_source: "web-ui",
    local_owner_key: "local-history-v1:p1"
  })
  await db().messages.bulkPut([
    { id: "m1", history_id: "h1", name: "You", role: "user", content: "Tell me", createdAt: 2 },
    { id: "m2", history_id: "h1", name: "model", role: "assistant", content: "Once", createdAt: 3 }
  ])
}

const continuedReply = async () => (await db().messages.get("m2"))?.content

describe("continuing a saved local chat", () => {
  beforeEach(async () => {
    db().resetAll()
    await seedLocalChat()
  })

  it("persists the continued reply after it finishes", async () => {
    await saveMessageOnSuccess({
      historyId: "h1",
      setHistoryId: vi.fn(),
      isRegenerate: false,
      selectedModel: "model",
      message: "",
      image: "",
      fullText: "Once upon a marmalade",
      source: [],
      isContinue: true
    })

    expect(await continuedReply()).toBe("Once upon a marmalade")
    const found = await pageAssistDatabase.fullTextSearchChatHistories("marmalade")
    expect(found.map((history) => history.id)).toEqual(["h1"])
  })

  it("persists the partial reply when a scoped continue is stopped", async () => {
    await saveMessageOnError({
      e: Object.assign(new Error("AbortError"), { name: "AbortError" }),
      history: [],
      setHistory: vi.fn(),
      image: "",
      userMessage: "",
      botMessage: "Once upon",
      historyId: "h1",
      selectedModel: "model",
      setHistoryId: vi.fn(),
      isRegenerating: false,
      isContinue: true,
      scopeInvalidatedSignal: new AbortController().signal
    })

    expect(await continuedReply()).toBe("Once upon")
  })
})
